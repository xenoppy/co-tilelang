"""M4 (methodology v1): root cause of the "static persistent penalty" of P1-S.

    source research/env.sh
    python research/bench/scripts/mv1_static.py [--parts 1234] [--shapes gemm4096,gemm8192,decode64]

Shapes (catalog configs):
  gemm4096  GEMM 4096x4096x4096,   solo-best 128x256x64_s2_t256_direct (512 tiles, 2.7 waves)
  gemm8192  GEMM 8192x14336x4096,  solo-best 256x128x64_s2_t256_direct (3584 tiles, 19.1 waves)
  decode64  GQA decode B64xS8192,  n64_h1_sp4_t128_s2 (8192 tiles, 43.6 waves; P1-S: 1.72x)
Builds / schedules (same tile body):
  grid        one CTA per tile (op library)       persistent  op library static grid-stride
  static      sched_variants static (== persistent + knobs)
  dynamic     sched_variants dynamic queue          co_dyn      CoKernel, partner compiled in with
                                                                0 SMs, dynamic chunk 1
  static_perm TPC-split rank permutation            chunked     contiguous tile range per CTA
  static_slot static ids via smem slot + barrier    static_atomic  + one global atomic per tile
Parts:
  1  clean flush (bench_variants, ClockProbe on, probe carveout fixed = 100%): all builds
  2  steady (bench_steady, back-to-back, rotation): grid / persistent / dynamic / co_dyn / chunked
  3  artefact reproduction: ClockProbe with carveout 0 (the P1-S behaviour), and no probe
  4  per-tile timelines (smem-buffered %globaltimer traces) with the fixed and the old probe
Results -> research/results/2026-09-23_methodology_v1/static_penalty.json
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

import mv1_common as C
import sched_variants as sv
from mv1_common import cb, log

SHAPES = {
    "gemm4096": ("gemm", "M4096_N4096_K4096", None, ("gqa_decode", "B16_S8192")),
    "gemm8192": ("gemm", "M8192_N14336_K4096", None, ("gqa_decode", "B16_S8192")),
    "decode64": ("gqa_decode", "B64_S8192", "n64_h1_sp4_t128_s2", ("gemm", "M4096_N4096_K4096")),
}
NSM = 188


def build(key):
    op, st, tag, (pop, pst) = SHAPES[key]
    mod, shape, cfg, rec = C.solo_best(op, st) if tag is None else C.cfg_of(op, st, tag)
    pmod, pshape, pcfg, _ = C.solo_best(pop, pst)
    ctas = rec.sig["persistent"]["num_ctas"]
    from cotile.cokernel import Orch, build_cokernel
    specs = {"grid": mod.build_grid(shape, cfg), "persistent": mod.build_persistent(shape, cfg, ctas),
             "grid_tr": sv.build_grid_traced(mod, shape, cfg),
             "co_dyn": build_cokernel(mod, shape, cfg, pmod, pshape, pcfg,
                                      Orch(binding="sm", schedule="dynamic", chunk=1, num_ctas=ctas, timing=False))}
    for s in sv.SCHEDULES:
        specs[s] = sv.build_sched(mod, shape, cfg, ctas, s)
    for s in ("static", "dynamic"):
        specs[s + "_tr"] = sv.build_sched(mod, shape, cfg, ctas, s, trace=True)
    C.compile_all(list(specs.values()))
    info = {"op": op, "shape": st, "cfg": rec.tag, "num_ctas": ctas, "tiles": specs["grid"].num_tiles,
            "waves": specs["grid"].num_tiles / ctas, "catalog_grid_us": rec.time(),
            "catalog_persistent_us": rec.time("persistent")}
    from cotile import resources
    info["sig"] = {k: {kk: v for kk, v in resources.signature(s).items() if kk in ("regs", "smem_dynamic", "ctas_per_sm")}
                   for k, s in specs.items()}
    return mod, shape, cfg, (pmod, pshape, pcfg), specs, info


def launchers(mod, shape, partner, specs):
    """variant name -> (callable(i), rotation) ; persistent-style variants share nothing."""
    from cotile.cokernel import CoRunner, sm_role_table
    ref = C.OpLauncher(specs["grid"])
    rot = cb.Rotation(ref.make_args)
    V = {}
    for name in ("grid", "persistent"):
        L = C.OpLauncher(specs[name])
        V[name] = (lambda L: (lambda i: L.run(*rot[i])))(L)
    perms = {"static_perm": sv.tpc_split_perm(specs["static"].grid)}
    for name in sv.SCHEDULES:
        run = sv.SvRunner(specs[name], perm=perms.get(name))
        L = C.OpLauncher(specs[name], state=run.state)
        arg_rot = [_extend(ref, L, rot[k]) for k in range(rot.n)]
        V[name] = (lambda L, a: (lambda i: L.run(*a[i % len(a)])))(L, arg_rot)
    # CoKernel with the partner compiled in but given no SMs (all SMs -> role A)
    pmod, pshape, pcfg = partner
    co = CoRunner(specs["co_dyn"])
    co.set_knobs(sm_role=sm_role_table(NSM))
    pin = pmod.make_inputs(pshape)
    arglists = []
    for k in range(rot.n):
        ins, outs = _io(ref, rot[k])
        a, _ = co.make_args([ins, pin])
        # route role A's outputs to this copy's output tensors
        a = [outs[p.name[2:]] if (p.role == "out" and p.name.startswith("a_")) else x
             for p, x in zip(co.spec.params, a)]
        arglists.append(a)
    V["co_dyn"] = (lambda k, al: (lambda i: k(*al[i % len(al)])))(co.spec.kernel, arglists)
    # correctness: every build's output == grid output (bitwise, copy 0)
    out_i = [j for j, p in enumerate(ref.spec.params) if p.role == "out"]
    V["grid"](0)
    torch.cuda.synchronize()
    want = [rot[0][j].clone() for j in out_i]
    for name, f in V.items():
        for j in out_i:
            rot[0][j].zero_()
        f(0)
        torch.cuda.synchronize()
        if not all(torch.equal(rot[0][j], w) for j, w in zip(out_i, want)):
            raise RuntimeError(f"{name}: output differs from the grid build")
    return V, rot


def _io(L, args):
    ins = {p.name: a for p, a in zip(L.spec.params, args) if p.role == "in"}
    outs = {p.name: a for p, a in zip(L.spec.params, args) if p.role == "out"}
    return ins, outs


def _extend(ref, L, args):
    """args of `ref` (op params) -> args of L (op params + extra params from L.state)."""
    by = {p.name: a for p, a in zip(ref.spec.params, args)}
    return tuple(by[p.name] if p.name in by else L.state[p.name] for p in L.spec.params)


def flush_run(V, names, label, clock=True):
    r = cb.bench_variants({n: V[n] for n in names}, reference="grid", clock=clock, nvml=True, label=label)
    log("\n" + str(r))
    return {n: {"median_us": v["total"]["median"], "p10_us": v["total"]["p10"], "p90_us": v["total"]["p90"],
                "cv": v["total"]["cv"], "clock_mhz": (v.get("clock") or {}).get("median"),
                "kcycles": ((v.get("clock") or {}).get("cycles_median") or 0) / 1e3 or None,
                "vs_grid": r.variants["grid"]["total"]["median"] and v["total"]["median"] / r.variants["grid"]["total"]["median"]}
            for n, v in r.variants.items()}


def set_probe_carveout(pct):
    from cobench.clock import _kernel
    k = _kernel(torch.cuda.current_device())
    k.set_carveout(pct)


def traces(mod, shape, specs, rot, probe_carveout):
    """One traced launch (after 10 warm launches, clean flush before each) per build, with a
    ClockProbe resident (carveout = probe_carveout; None = no probe)."""
    from cobench.clock import ClockProbe
    from cobench.timing import _flush, _flush_buffer
    buf = _flush_buffer(0, 2 * cb.l2_bytes())
    ref = C.OpLauncher(specs["grid"])
    out = {}
    for name in ("grid_tr", "static_tr", "dynamic_tr"):
        run = sv.SvRunner(specs[name])
        L = C.OpLauncher(specs[name], state=run.state)
        args = _extend(ref, L, rot[0])
        L.run(*args)
        torch.cuda.synchronize()
        pk = None
        if probe_carveout is not None:
            set_probe_carveout(probe_carveout)
            pk = ClockProbe()
            pk.start()
        for rep in range(10):
            _flush(buf, rep, "clean")
            L.run(*args)
        torch.cuda.current_stream().synchronize()
        tr = run.trace()
        if pk:
            pk.stop()
            set_probe_carveout(100)
        out[name] = summarize_trace(tr, pk.smid if pk else None)
        log(f"   trace {name} probe={probe_carveout}: {out[name]['summary']}")
    return out


def summarize_trace(tr, probe_smid):
    meta = tr["meta"]
    t0 = meta[:, 2].min()
    smid = meta[:, 0].astype(int)
    start = (meta[:, 2] - t0) / 1e3
    end = (meta[:, 3] - t0) / 1e3
    s = {"makespan_us": float(end.max()), "ctas": int(meta.shape[0]), "distinct_sms": int(len(set(smid.tolist()))),
         "probe_smid": probe_smid, "probe_sm_used": (probe_smid in set(smid.tolist())) if probe_smid is not None else None,
         "cta_start_us": {"p50": float(np.median(start)), "p99": float(np.percentile(start, 99)), "max": float(start.max())}}
    late = np.argsort(start)[-3:][::-1]
    s["latest_ctas"] = [{"bid": int(b), "start_us": float(start[b]), "smid": int(smid[b]), "tiles": int(meta[b, 1])}
                        for b in late]
    d = {"summary": s}
    if "tr" in tr:
        n = np.minimum(meta[:, 1].astype(int), tr["tr"].shape[1])
        steps = []
        for j in range(int(n.max())):
            m = n > j
            st = (tr["tr"][m, j, 0] - t0) / 1e3
            du = (tr["tr"][m, j, 1] - tr["tr"][m, j, 0]) / 1e3
            steps.append({"step": j, "n": int(m.sum()), "start_min": float(st.min()), "start_p50": float(np.median(st)),
                          "start_max": float(st.max()), "dur_p10": float(np.percentile(du, 10)),
                          "dur_p50": float(np.median(du)), "dur_p90": float(np.percentile(du, 90))})
        d["steps"] = steps if len(steps) <= 12 else steps[:6] + steps[-6:]
        busy = np.array([((tr["tr"][b, :n[b], 1] - tr["tr"][b, :n[b], 0]).sum()) / 1e3 for b in range(meta.shape[0])])
        s["per_cta_busy_us"] = {"min": float(busy.min()), "p50": float(np.median(busy)), "max": float(busy.max())}
        s["tiles_per_cta"] = {"min": int(n.min()), "max": int(n.max())}
    else:
        du = end - start
        s["tile_dur_us"] = {"p10": float(np.percentile(du, 10)), "p50": float(np.median(du)), "p90": float(np.percentile(du, 90))}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="1234")
    ap.add_argument("--shapes", default="gemm4096,gemm8192,decode64")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--slice", type=float, default=1.2)
    args = ap.parse_args()
    cb.wait_until_free()
    import json
    import os
    path = "static_penalty.json"
    full = os.path.join(C.OUT, path)
    out = json.load(open(full)) if os.path.exists(full) else {}   # a partial re-run updates in place
    out.setdefault("runs", []).append(C.env_meta() | {"parts": args.parts, "shapes": args.shapes})
    out["ncu"] = "unavailable: ERR_NVGPUCTRPERM (RmProfilingAdminOnly=1, no sudo)"
    for key in args.shapes.split(","):
        log(f"== {key}")
        mod, shape, cfg, partner, specs, info = build(key)
        V, rot = launchers(mod, shape, partner, specs)
        res = out.get(key, {})
        res["info"] = info
        allv = ["grid", "persistent", "static", "dynamic", "co_dyn", "static_perm", "chunked", "static_slot",
                "static_atomic"]
        if "1" in args.parts:
            res["flush_fixed_probe"] = flush_run(V, allv, f"{key} clean flush, probe carveout 100")
        if "2" in args.parts:
            names = ["grid", "persistent", "dynamic", "co_dyn", "chunked"]
            r = cb.bench_steady({n: V[n] for n in names}, reference="grid", slice_s=args.slice, settle_s=0.4,
                                rounds=args.rounds, warmup_s=3.0, label=f"{key} steady")
            log("\n" + str(r))
            res["steady"] = {n: {"t_iter_us": v["t_iter_us"], "cv_slices": v["cv_slices"], "clock_mhz": v["clock_mhz"],
                                 "power_w": v["power_w"], "vs_grid": v["t_iter_us"] / r.variants["grid"]["t_iter_us"]}
                             for n, v in r.variants.items()}
        if "3" in args.parts:
            names = ["grid", "persistent", "dynamic", "co_dyn", "chunked"]
            set_probe_carveout(0)
            try:
                res["flush_old_probe"] = flush_run(V, names, f"{key} clean flush, probe carveout 0 (P1-S behaviour)")
            finally:
                set_probe_carveout(100)
            res["flush_no_probe"] = flush_run(V, names, f"{key} clean flush, no probe", clock=False)
        if "4" in args.parts:
            res["traces_fixed_probe"] = traces(mod, shape, specs, rot, 100)
            res["traces_old_probe"] = traces(mod, shape, specs, rot, 0)
            res["traces_no_probe"] = traces(mod, shape, specs, rot, None)
        out[key] = res
        C.save(path, out)
        del V, rot, specs
        torch.cuda.empty_cache()
    out["guard"] = cb.get_guard().summary()
    log("saved", C.save(path, out))


if __name__ == "__main__":
    sys.exit(main())
