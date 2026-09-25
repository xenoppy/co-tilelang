"""Claim C2: FlashAttention forward, TileLang vs Triton / PyTorch / FA3 (images/mha_performance_h100.png).

Shapes FA0-FA4 (tilelang-benchmark README Table 2 = TileLang paper arXiv 2504.17577 appendix Table 3):
batch 1, 32 heads, head_dim 128, fp16, seq_len 512/512/1024/1024/4096, causal T/F/T/F/T.

One shape per process (Triton's autotune key of both vendored kernels omits the causal flag, so
FA0/FA1 -- same N_CTX -- would share one tuned config inside a process).

Usage (repo root, `source research/env.sh`):
  python research/bench/scripts/claims_attn_fa.py --shape FA2 --phase compile      # CPU only
  flock <gpu.lock> python research/bench/scripts/claims_attn_fa.py --shape FA2 --phase run
Output: research/results/2026-09-24_claims_repro/B_attention/raw/fa_<shape>.json

Implementations (all fp16 in, fp16 out, softmax scale 1/sqrt(d)):
  TileLang (examples/flash_attention, imported unmodified; configs passed explicitly):
    tl_bhsd_h100cfg_s1   (block_M, block_N, stages, threads) = (128, 128, 1, 256): the config of
                         example_mha_fwd_bhsd.py's autotuner and of tilelang-benchmark's
                         benchmark_tilelang_mha.py (H100) is (128, 128, 2, 256) = 160 KB smem,
                         unlaunchable on sm_120 (99 KB/CTA); adapted by num_stages 2 -> 1 (96 KB).
                         PRIMARY TileLang number.
    tl_bhsd_main         (64, 64, 1, 128)  example_mha_fwd_bhsd.main() default (tune=False)
    tl_bshd_main         (128, 128, 1, 128) example_mha_fwd_bshd.main() default (tune=False)
    tl_bshd_tune         (64, 64, 1, 128)  example_mha_fwd_bshd.py autotuner config
    tl_bhsd_sweep        best of a 36-config sweep (block_M {64,128} x block_N {32,64,128} x
                         stages {1,2,3} x threads {128,256}, <= 99 KB), picked with
                         tilelang.profiler.do_bench -- an sm_120 upper bound, not "as shipped".
  Triton:
    triton_tlbench_ws    tilelang-benchmark's copy of tutorial 06 as its benchmark called it
                         (TMA descriptors, warp_specialize=True), own autotuner
    triton_tlbench       same kernel, warp_specialize=False
    triton_tut06         Triton v3.4.0 tutorial 06, warp_specialize=False (the tutorial's
                         setting on non-Blackwell GPUs), own autotuner
  PyTorch:
    sdpa_flash / sdpa_efficient / sdpa_cudnn / sdpa_math   torch SDPA with one backend forced
    torch_naive          the upstream "Ref"/torch program (einsum + masked softmax + einsum, fp16):
                         examples/flash_attention/example_mha_fwd_bhsd.ref_program
  Extra references:
    flashinfer_prefill   flashinfer.single_prefill_with_kv_cache (NHD, backend auto -> fa2 on sm_120)
    fa2_pip              flash_attn.flash_attn_func (only if the flash-attn wheel is installed)
  FlashAttention-3: not testable on sm_120 (sm_90a-only kernels; see README).
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_attn_common as C  # noqa: E402

SHAPES = {
    "FA0": dict(batch=1, heads=32, seq=512, dim=128, causal=True),
    "FA1": dict(batch=1, heads=32, seq=512, dim=128, causal=False),
    "FA2": dict(batch=1, heads=32, seq=1024, dim=128, causal=True),
    "FA3": dict(batch=1, heads=32, seq=1024, dim=128, causal=False),
    "FA4": dict(batch=1, heads=32, seq=4096, dim=128, causal=True),
    # extra, larger shapes (not in the figure): the upstream scripts' own defaults
    "X0": dict(batch=8, heads=32, seq=4096, dim=128, causal=True),     # example_mha_fwd_bshd.py defaults
    "X1": dict(batch=8, heads=32, seq=4096, dim=128, causal=False),
    "X2": dict(batch=64, heads=64, seq=8192, dim=128, causal=True),    # tilelang-benchmark scripts' default
}

TL_FIXED = {
    # name: (layout, (block_M, block_N, num_stages, threads), provenance)
    "tl_bhsd_h100cfg_s1": ("bhsd", (128, 128, 1, 256),
                           "example_mha_fwd_bhsd autotune config / tilelang-benchmark H100 config (128,128,2,256), "
                           "num_stages 2->1 to fit 99 KB"),
    "tl_bhsd_main": ("bhsd", (64, 64, 1, 128), "example_mha_fwd_bhsd.main(tune=False)"),
    "tl_bshd_main": ("bshd", (128, 128, 1, 128), "example_mha_fwd_bshd.main(tune=False)"),
    "tl_bshd_tune": ("bshd", (64, 64, 1, 128), "example_mha_fwd_bshd autotune config"),
}
TL_UNLAUNCHABLE = {"tl_bhsd_h100cfg": ("bhsd", (128, 128, 2, 256), "shipped config, 160 KB smem")}
SWEEP = list(itertools.product((64, 128), (32, 64, 128), (1, 2, 3), (128, 256)))
PRIMARY = "tl_bhsd_h100cfg_s1"


def import_examples():
    ex_dir = os.path.join(C.REPO, "examples", "flash_attention")
    if ex_dir not in sys.path:
        sys.path.insert(0, ex_dir)
    import example_mha_fwd_bhsd as exb
    import example_mha_fwd_bshd as exs
    return exb, exs


def tl_build(layout, cfg, sh, exb, exs):
    bm, bn, st, th = cfg
    if layout == "bhsd":
        return exb.flashattn(sh["batch"], sh["heads"], sh["seq"], sh["seq"], sh["dim"], sh["causal"],
                             block_M=bm, block_N=bn, num_stages=st, threads=th)
    return exs.flashattn(sh["batch"], sh["heads"], sh["seq"], sh["dim"], sh["causal"],
                         block_M=bm, block_N=bn, num_stages=st, threads=th)


def compile_phase(shape_name, sh, sweep: bool) -> dict:
    """Compile every TileLang config for this shape (fills the kernel cache) and record resources."""
    exb, exs = import_examples()
    out = {"fixed": {}, "sweep": {}}
    todo = [(n, lay, cfg) for n, (lay, cfg, _) in {**TL_FIXED, **TL_UNLAUNCHABLE}.items()]
    if sweep:
        todo += [(f"sweep_{bm}_{bn}_{st}_{th}", "bhsd", (bm, bn, st, th)) for bm, bn, st, th in SWEEP]
    for name, lay, cfg in todo:
        t0 = time.time()
        rec = {"layout": lay, "cfg": list(cfg)}
        try:
            k = tl_build(lay, cfg, sh, exb, exs)
            rec["resources"] = C.tl_resources(k)
            rec["fits_99KB"] = all(r["fits_99KB"] for r in rec["resources"].values())
        except Exception as e:  # noqa: BLE001 - recorded
            rec["compile_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        rec["compile_s"] = round(time.time() - t0, 1)
        (out["sweep"] if name.startswith("sweep_") else out["fixed"])[name] = rec
        print(name, rec.get("fits_99KB"), rec.get("compile_error", ""), rec["compile_s"], flush=True)
    return out


def make_inputs(sh, device="cuda"):
    import torch
    g = torch.Generator(device=device)
    g.manual_seed(1234)
    B, H, S, D = sh["batch"], sh["heads"], sh["seq"], sh["dim"]
    q, k, v = [torch.randn(B, H, S, D, device=device, dtype=torch.float32, generator=g).half() for _ in range(3)]
    bshd = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    return {"bhsd": (q, k, v), "bshd": tuple(bshd)}


def reference(sh, inp):
    import torch
    import torch.nn.functional as F
    q, k, v = inp["bhsd"]
    D = sh["dim"]
    outs = []
    # fp32 math SDPA, chunked over batch to bound memory
    step = max(1, int(2 ** 31 // (sh["heads"] * sh["seq"] * sh["seq"] * 4)))
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        for b0 in range(0, sh["batch"], step):
            sl = slice(b0, b0 + step)
            outs.append(F.scaled_dot_product_attention(q[sl].float(), k[sl].float(), v[sl].float(),
                                                       is_causal=sh["causal"], scale=1.0 / math.sqrt(D)))
    return torch.cat(outs, 0)  # BHSD fp32


def build_variants(sh, inp, tl_kernels, have_fa2):
    """name -> (callable returning the output in BHSD, meta)."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    C.install_pytest_stub()
    if C.UPSTREAM not in sys.path:
        sys.path.insert(0, C.UPSTREAM)
    import tlbench_triton_mha as tb
    import triton340_tutorial06_fused_attention as t6
    exb, _ = import_examples()

    q, k, v = inp["bhsd"]
    qs, ks, vs = inp["bshd"]
    causal = sh["causal"]
    scale = 1.0 / math.sqrt(sh["dim"])
    V = {}
    for name, (lay, kern) in tl_kernels.items():
        if lay == "bhsd":
            V[name] = (lambda kern=kern: kern(q, k, v), "bhsd")
        else:
            V[name] = (lambda kern=kern: kern(qs, ks, vs), "bshd")
    V["triton_tlbench_ws"] = (lambda: tb.attention(q, k, v, causal, scale, True), "bhsd")
    V["triton_tlbench"] = (lambda: tb.attention(q, k, v, causal, scale, False), "bhsd")
    V["triton_tut06"] = (lambda: t6.attention(q, k, v, causal, scale, False), "bhsd")
    for tag, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                    ("cudnn", SDPBackend.CUDNN_ATTENTION), ("math", SDPBackend.MATH)):
        def f(be=be):
            with sdpa_kernel(be):
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
        V[f"sdpa_{tag}"] = (f, "bhsd")
    V["torch_naive"] = (lambda: exb.ref_program(q, k, v, causal), "bhsd")
    if sh["batch"] == 1:
        import flashinfer
        qn, kn, vn = qs[0], ks[0], vs[0]           # NHD [S, H, D]
        V["flashinfer_prefill"] = (lambda: flashinfer.single_prefill_with_kv_cache(
            qn, kn, vn, causal=causal, sm_scale=scale).unsqueeze(0), "bshd")
    if have_fa2:
        from flash_attn import flash_attn_func
        V["fa2_pip"] = (lambda: flash_attn_func(qs, ks, vs, causal=causal, softmax_scale=scale), "bshd")
    return V


def to_bhsd(o, lay):
    return o if lay == "bhsd" else o.transpose(1, 2)


def run_phase(shape_name, sh, args) -> dict:
    import torch
    info = {"wait": C.wait_gpu(min_free_gib=8), "pmon_before": C.pmon_snapshot()}
    import cobench as cb
    import tilelang  # noqa: F401
    exb, exs = import_examples()
    try:
        import flash_attn  # noqa: F401
        have_fa2 = True
        info["flash_attn_version"] = flash_attn.__version__
    except Exception as e:  # noqa: BLE001
        have_fa2 = False
        info["flash_attn"] = f"not available: {type(e).__name__}"
    res = {"shape_name": shape_name, "shape": sh, "env": C.env_info(), "info": info, "timings_s": {},
           "variants": {}, "unlaunchable": {}}
    inp = make_inputs(sh)
    with C.timed("reference", res["timings_s"]):
        ref = reference(sh, inp)

    # --- TileLang kernels (compiled in the compile phase -> kernel cache hits) ---
    tl_kernels = {}
    for name, (lay, cfg, prov) in TL_FIXED.items():
        k = tl_build(lay, cfg, sh, exb, exs)
        tl_kernels[name] = (lay, k)
        res["variants"][name] = {"cfg": list(cfg), "layout": lay, "provenance": prov,
                                 "resources": C.tl_resources(k)}
    # the shipped H100 config: verify it compiles but fails at launch -- in a child process, so a
    # failed launch cannot affect the kernels timed here (see claims_attn_mla.py)
    res["unlaunchable"] = C.run_isolated([sys.executable, os.path.abspath(__file__), "--shape", shape_name,
                                          "--phase", "shipped"])

    # --- sm_120 sweep (bhsd), picked with tilelang.profiler.do_bench like the TileLang autotuner ---
    if args.sweep:
        from tilelang.profiler import do_bench as tl_do_bench
        sweep = {}
        q, kk, v = inp["bhsd"]
        for bm, bn, st, th in SWEEP:
            nm = f"{bm}_{bn}_{st}_{th}"
            rec = {}
            try:
                kern = tl_build("bhsd", (bm, bn, st, th), sh, exb, exs)
                r = C.tl_resources(kern)
                rec["smem"] = max(x["smem_total"] for x in r.values())
                rec["regs"] = max(x["regs"] for x in r.values())
                rec["local_bytes"] = max(x["local_bytes"] for x in r.values())
                if rec["smem"] > C.SMEM_PER_CTA:
                    rec["skipped"] = "smem > 99 KB"
                else:
                    o = kern(q, kk, v)
                    rec["check"] = C.err_stats(o, ref)
                    if rec["check"]["ok"]:
                        rec["tl_do_bench_us"] = 1e3 * float(tl_do_bench(lambda kern=kern: kern(q, kk, v)))
                    tl_kernels.setdefault("_sweep", {})[nm] = kern
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            sweep[nm] = rec
        res["sweep"] = sweep
        ok = {n: r["tl_do_bench_us"] for n, r in sweep.items() if "tl_do_bench_us" in r}
        best = min(ok, key=ok.get)
        res["sweep_best"] = {"cfg": best, "tl_do_bench_us": ok[best]}
        tl_kernels["tl_bhsd_sweep"] = ("bhsd", tl_kernels["_sweep"][best])
        res["variants"]["tl_bhsd_sweep"] = {"cfg": [int(x) for x in best.split("_")], "layout": "bhsd",
                                            "provenance": "best of the sm_120 sweep (tilelang do_bench)",
                                            "resources": C.tl_resources(tl_kernels["_sweep"][best])}
    tl_kernels.pop("_sweep", None)

    # --- baselines + correctness ---
    V = build_variants(sh, inp, tl_kernels, have_fa2)
    timed_fns = {}
    for name, (fn, lay) in V.items():
        rec = res["variants"].setdefault(name, {"layout": lay})
        t0 = time.time()
        try:
            o = fn()
            torch.cuda.synchronize()
            rec["first_call_s"] = round(time.time() - t0, 1)
            rec["check"] = C.err_stats(to_bhsd(o, lay), ref)
            if rec["check"]["ok"]:
                timed_fns[name] = fn
            else:
                rec["excluded"] = "failed correctness check"
        except Exception as e:  # noqa: BLE001 - recorded, not silenced
            rec["error"] = f"{type(e).__name__}: {str(e)[:600]}"
            rec["traceback"] = traceback.format_exc()[-1500:]
            rec["cleared_last_error"] = C.clear_cuda_error()
        print(f"{name}: {rec.get('check', {}).get('ok')} {rec.get('check', {}).get('max_abs')} "
              f"{rec.get('error', '')[:200]}", flush=True)
    # Triton configs chosen by the autotuners (first call)
    import tlbench_triton_mha as tb
    import triton340_tutorial06_fused_attention as t6
    res["triton_best_configs"] = {
        f"{m.__name__}.{a}": {str(k): str(v) for k, v in getattr(getattr(m, a), "cache", {}).items()}
        for m, a in ((tb, "_attn_fwd_tma"), (tb, "_attn_fwd"), (t6, "_attn_fwd"))}

    # --- primary timer: cobench flush-mode interleaved ---
    with C.timed("bench_variants", res["timings_s"]):
        vr = cb.bench_variants(dict(timed_fns), reference=PRIMARY if PRIMARY in timed_fns else None,
                               reps=args.reps, clock=True, nvml=True, keep_samples=True, label=f"C2 {shape_name}")
    res["cobench"] = {"summary": C.variants_summary(vr), "derived": vr.derived, "config": vr.config,
                      "nvml": vr.nvml, "clock_window": vr.clock, "guard": vr.guard,
                      "samples_us": vr.samples}
    # --- secondary: the upstream scripts' timers ---
    with C.timed("upstream timers", res["timings_s"]):
        for name, fn in timed_fns.items():
            res["variants"][name]["upstream"] = C.upstream_timers(fn, tl_warmup=500 if name.startswith("tl_") else 25)
    res["pmon_after"] = C.pmon_snapshot()
    return res


def shipped_phase(shape_name, sh) -> dict:
    """Try the configs in TL_UNLAUNCHABLE once (expected: launch fails, smem > 99 KB)."""
    import torch
    exb, exs = import_examples()
    inp = make_inputs(sh)
    ref = reference(sh, inp)
    out = {}
    for name, (lay, cfg, prov) in TL_UNLAUNCHABLE.items():
        rec = {"cfg": list(cfg), "layout": lay, "provenance": prov}
        try:
            k = tl_build(lay, cfg, sh, exb, exs)
            rec["resources"] = C.tl_resources(k)
            q, kk, v = inp[lay]
            o = k(q, kk, v)
            torch.cuda.synchronize()
            rec["launch"] = "succeeded"
            rec["check"] = C.err_stats(to_bhsd(o, lay), ref)
        except Exception as e:  # noqa: BLE001 - this is the expected outcome, recorded
            rec["launch"] = f"failed: {type(e).__name__}: {str(e)[:400]}"
        out[name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=list(SHAPES))
    ap.add_argument("--phase", choices=("compile", "run", "shipped"), required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--no-sweep", dest="sweep", action="store_false")
    args = ap.parse_args()
    sh = SHAPES[args.shape]
    if args.phase == "compile":
        out = compile_phase(args.shape, sh, args.sweep)
        C.save_json(out, os.path.join(C.RESULTS, "raw", f"fa_{args.shape}_compile.json"))
        return
    if args.phase == "shipped":           # child of the run phase (already under the GPU lock)
        C.emit_isolated(shipped_phase(args.shape, sh))
        return
    res = run_phase(args.shape, sh, args)
    C.save_json(res, os.path.join(C.RESULTS, "raw", f"fa_{args.shape}.json"))
    s = res["cobench"]["summary"]
    ref = s.get(PRIMARY, {}).get("median_us")
    print(f"\n{args.shape} {sh}")
    for n, d in sorted(s.items(), key=lambda kv: kv[1]["median_us"]):
        up = res["variants"][n].get("upstream", {})
        r = f"{d['median_us'] / ref:5.2f}x" if ref else ""
        print(f"  {n:22s} {d['median_us']:9.1f} us  cv {d['cv']:.3f}  {d.get('clock_mhz_median') or 0:6.0f} MHz  "
              f"{r}  tl_do_bench {up.get('tilelang_do_bench_us', 0):8.1f}  triton_do_bench "
              f"{up.get('triton_do_bench_us', 0):8.1f}")


if __name__ == "__main__":
    main()
