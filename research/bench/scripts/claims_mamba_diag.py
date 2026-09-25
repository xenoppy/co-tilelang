"""Diagnostic (not part of the verdict): how much of the TileLang-vs-Triton gap on sm_120 comes from
TileLang's default Hopper-style lowering (TMA loads + warp specialisation with mma.sync consumers)?

    source research/env.sh
    python research/bench/scripts/run_guarded.py -- flock $GPU_LOCK \
        python research/bench/scripts/claims_mamba_diag.py --shape CC1 [--topk 6]

For the autotuner's top-k configs (from tune/<shape>.json) the kernel is compiled three ways -
as shipped, with tl.disable_warp_specialized, and with tl.disable_tma_lower (pass configs of the
upstream compiler; kernel source unchanged) - checked against the as-shipped output, and all of them
are timed together with Triton in one cobench.bench_variants call (clean flush, clock). Also records
the dynamic shared memory of each compiled kernel (from the generated source) and the resulting
CTAs/SM bound on sm_120 (102400 B smem per SM, 1 KB reserved per CTA) vs H100 (233472 B per SM).
Writes C_mamba/diag/<shape>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402

VARIANTS = {"shipped": {}, "no_ws": {"tl.disable_warp_specialized": True}, "no_tma": {"tl.disable_tma_lower": True}}


def compile_variant(shape, cfg, extra_pc):
    mod = C.module(shape["kind"])
    fn = mod.chunk_scan_fwd if shape["kind"] in ("scan_ex", "scan_bm") else mod.chunk_state_fwd
    ji = fn.jit_impl
    orig = ji.pass_configs
    ji.pass_configs = {**(orig or {}), **extra_pc}
    try:
        sa = C.shape_args(shape)
        if shape["kind"] == "scan_bm":
            return ji.compile(sa["batch"], sa["seqlen"], sa["chunk_size"], sa["ngroups"], sa["nheads"], sa["headdim"],
                              sa["dstate"], **cfg, threads=128)
        return ji.compile(**sa, **cfg, threads=128)
    finally:
        ji.pass_configs = orig


def launch_info(kernel) -> dict:
    """grid, block and dynamic smem of the (single) device kernel, read from the generated host code: the
    last 7 int64 launch arguments before the kernel call are gridDim.xyz, blockDim.xyz, dynamic smem."""
    host = kernel.get_host_source()
    call = [m.start() for m in re.finditer(r"TVMFFIFunctionCall\(\w+_kernel_packed", host)]
    ints = [int(x) for x in re.findall(r"\.v_int64\) = \(\(int64_t\)(\d+)\);", host[:call[-1]])][-7:] if call else []
    src = kernel.get_kernel_source()
    stat = sum(int(n) * {"uint64_t": 8, "float": 4, "int": 4, "half_t": 2}.get(ty, 8)
               for ty, n in re.findall(r"__shared__ __align__\(\d+\) (\w+) \w+\[(\d+)\]", src))
    if len(ints) != 7:
        return dict(launch_parse_error=True, static_smem=stat)
    per_cta = ints[6] + stat + 1024  # + 1 KB reserved per CTA
    return dict(grid=ints[0:3], block=ints[3:6], dyn_smem=ints[6], static_smem=stat,
                ctas_per_sm_smem_sm120=102400 // per_cta, ctas_per_sm_smem_h100=233472 // per_cta)


def cubin_resources(kernel) -> dict:
    """Registers / spills / static smem of the device kernel: the generated source is recompiled with the
    options tilelang/cuda/backend.py uses (-std=c++20, template + CUTLASS includes, --use_fast_math when the
    kernel's pass config enables it, arch sm_120a) and inspected with `cuobjdump -res-usage` (the cached
    executable.so does not expose its cubin to cuobjdump)."""
    import subprocess
    import tempfile
    from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    pc = getattr(kernel.adapter, "pass_configs", None) or {}
    fast = bool(pc.get("tl.enable_fast_math", False))
    with tempfile.TemporaryDirectory() as td:
        cu_path, cubin = os.path.join(td, "k.cu"), os.path.join(td, "k.cubin")
        with open(cu_path, "w") as f:
            f.write(kernel.get_kernel_source())
        cmd = [os.path.join(cuda, "bin", "nvcc"), "-cubin", "-arch=sm_120a", "-std=c++20", "-I" + TILELANG_TEMPLATE_PATH,
               "-I" + CUTLASS_INCLUDE_DIR, "-o", cubin, cu_path] + (["--use_fast_math"] if fast else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return dict(nvcc_error=r.stderr[-300:])
        txt = subprocess.run([os.path.join(cuda, "bin", "cuobjdump"), "-res-usage", cubin], capture_output=True, text=True).stdout
    m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", txt)
    return dict(regs=int(m.group(1)), stack=int(m.group(2)), static_smem_cubin=int(m.group(3)), local=int(m.group(4)),
                fast_math=fast) if m else dict(cuobjdump_parse_error=txt[-300:])


def occupancy_bound(regs, threads, smem_total) -> int:
    """Resident CTAs/SM on sm_120 from registers (64K/SM, allocation granularity 256 regs per warp) and
    shared memory (102400 B/SM incl. 1 KB reserved per CTA); threads <= 1536/SM."""
    warps = (threads + 31) // 32
    regs_per_warp = ((regs * 32 + 255) // 256) * 256
    by_regs = 65536 // (regs_per_warp * warps) if regs else 99
    by_smem = 102400 // (smem_total + 1024)
    return min(by_regs, by_smem, 1536 // threads, 32)


def triton_info(shape) -> list:
    """Compiled variants of the Triton kernel after its autotuning (Triton 3.4 JITFunction caches)."""
    import torch
    from cuda.bindings import driver as cu
    scan, state = C.triton_modules()
    at = scan._chunk_scan_fwd_kernel if shape["kind"] in ("scan_ex", "scan_bm") else state._chunk_state_fwd_kernel
    best = at.best_config
    names = at.fn.arg_names
    out = []
    for ck in at.fn.device_caches[torch.cuda.current_device()][0].values():
        md = ck.metadata
        consts = {}
        for k2, v2 in (getattr(ck.src, "constexprs", None) or {}).items():
            idx = k2[0] if isinstance(k2, tuple) else k2
            nm = names[idx] if isinstance(idx, int) else str(idx)
            if str(nm).startswith("BLOCK_SIZE"):
                consts[nm] = v2
        is_best = (md.num_warps == best.num_warps and md.num_stages == best.num_stages
                   and all(consts.get(k3) == v3 for k3, v3 in best.kwargs.items()))
        occ = None
        if getattr(ck, "function", None):
            err, occ = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(cu.CUfunction(ck.function), 32 * md.num_warps, md.shared)
            occ = int(occ) if int(err) == 0 else f"err {err}"
        out.append(dict(num_warps=md.num_warps, num_stages=md.num_stages, shared=md.shared, n_regs=getattr(ck, "n_regs", None),
                        n_spills=getattr(ck, "n_spills", None), consts=consts, ctas_per_sm=occ, is_best=is_best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--compile-only", action="store_true", help="fill the kernel cache only (no launch, no GPU lock needed)")
    a = ap.parse_args()
    shape = C.SHAPES[a.shape]
    tune = json.load(open(os.path.join(C.RESULTS, "tune", f"{a.shape}.json")))
    os.makedirs(os.path.join(C.RESULTS, "diag", "logs"), exist_ok=True)
    os.chdir(os.path.join(C.RESULTS, "diag", "logs"))
    if a.compile_only:
        for pc in tune["per_config"][:a.topk]:
            for extra in VARIANTS.values():
                try:
                    compile_variant(shape, pc["config"], extra)
                except Exception as e:  # reported; the timed run records it per variant
                    print("compile error", pc["config"], extra, repr(e)[-200:])
        return
    import torch
    import cobench as cb
    inp = C.make_inputs(shape, seed=0)
    args = C.call_args(shape, inp)
    tri = C.triton_fn(shape)
    ref = tri(*args)
    torch.cuda.synchronize()
    variants, info = {"triton": lambda: tri(*args)}, {}
    for j, pc in enumerate(tune["per_config"][:a.topk]):
        cfg = pc["config"]
        for vname, extra in VARIANTS.items():
            name = f"top{j}_{vname}"
            try:
                k = compile_variant(shape, cfg, extra)
                o = k(*args)
                torch.cuda.synchronize()
            except Exception as e:  # recorded, not silenced
                info[name] = dict(config=cfg, error=repr(e)[-300:])
                continue
            pe = C.pair_error(o, ref)
            del o
            src = k.get_kernel_source()
            li, rr = launch_info(k), cubin_resources(k)
            if "dyn_smem" in li and "regs" in rr:
                stat = max(li["static_smem"], rr.get("static_smem_cubin", 0))
                li["ctas_per_sm_sm120"] = occupancy_bound(rr["regs"], li["block"][0], li["dyn_smem"] + stat)
                li["ctas_per_sm_smem_sm120"] = 102400 // (li["dyn_smem"] + stat + 1024)
                li["ctas_per_sm_smem_h100"] = 233472 // (li["dyn_smem"] + stat + 1024)
            info[name] = dict(config=cfg, variant=vname, vs_triton=pe, ok=pe["rel_l2"] <= 1e-2, **li, **rr,
                              tma=("tma_load" in src),
                              # threads=128 is requested; a 256-thread launch means a producer warpgroup was added
                              ws=(li.get("block", [0])[0] == 256),
                              autotuner_ms=pc["latency_ms"])
            if info[name]["ok"]:
                variants[name] = (lambda k=k: k(*args))
    v = cb.bench_variants(variants, reference="triton", reps=a.reps, clock=True, label=f"diag {a.shape}")
    print(v, flush=True)
    for n, d in info.items():
        if n in v.variants:
            d["median_us"] = v.variants[n]["total"]["median"]
            d["ratio_triton_over"] = v.derived["speedup"][n]
    res = dict(shape_name=a.shape, shape=shape, triton_us=v.variants["triton"]["total"]["median"], variants=info,
               triton_compiled=triton_info(shape), triton_best_config=str(C.triton_modules()[0]._chunk_scan_fwd_kernel.best_config
                                                                          if shape["kind"] != "state_ex" else C.triton_modules()[1]._chunk_state_fwd_kernel.best_config),
               primary=v.to_dict(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.dump(res, os.path.join(C.RESULTS, "diag", f"{a.shape}.json"))
    for t in res["triton_compiled"]:
        if t["is_best"]:
            print("  triton best compiled:", t)
    for n, d in sorted(info.items(), key=lambda kv: kv[1].get("median_us", 1e9)):
        print(f"  {n:18s} {str(d['config']):80s} {d.get('median_us', float('nan')):9.2f} us  x{d.get('ratio_triton_over', float('nan')):.3f}"
              f"  smem {d.get('dyn_smem')} regs {d.get('regs')} ctas/SM {d.get('ctas_per_sm_sm120')} (smem-only H100 {d.get('ctas_per_sm_smem_h100')})"
              f"  block {d.get('block')} tma={d.get('tma')} ws={d.get('ws')} {d.get('error', '')}")


if __name__ == "__main__":
    main()
