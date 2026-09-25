"""Correctness check + timing of TileLang vs the mamba-ssm Triton baseline for one Mamba-2 shape.

    source research/env.sh
    python research/bench/scripts/run_guarded.py -- flock $GPU_LOCK \
        python research/bench/scripts/claims_mamba_bench.py --shape CC0

Needs tune/<shape>.json from claims_mamba_tune.py (the upstream autotuner's pick). Steps:
 1. compile the picked TileLang config (kernel-cache hit), build realistic inputs (seed 0);
 2. correctness: TileLang and Triton outputs vs the upstream torch reference (ref_program of the
    example, evaluated in fp32, sliced over batch x chunk-blocks), and TileLang vs Triton;
 3. primary timer: cobench.bench_variants({tilelang, triton}) - clean L2 flush, host gate, interleaved,
    >= 100 reps, SM clock per rep (ClockProbe), NVML power;
 4. autotuner-noise check: the autotuner's top-k configs re-timed with the same protocol (each checked
    against the verified TileLang output first);
 5. secondary timers = the upstream scripts' own: tilelang.profiler.do_bench (event timer, 256 MB
    zero_() flush, mean) for both kernels, alternated 3x; for C6 also the benchmark script's Triton call
    do_bench(..., _n_warmup=10, _n_repeat=10); plus the autotuner's own latency (from tune/<shape>.json).
Writes research/results/2026-09-24_claims_repro/C_mamba/bench/<shape>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402


def src_features(src: str) -> dict:
    return dict(
        launch_bounds=re.findall(r"__launch_bounds__\(([^)]*)\)", src)[:2],
        mma_sync=len(re.findall(r"mma_sync|mma\.sync|tl::gemm_ss|tl::gemm_rs|gemm_sr|gemm_rr", src)),
        wgmma=("wgmma" in src), tma=("tma_load" in src or "cp.async.bulk" in src),
        cp_async=len(re.findall(r"cp_async_gs|cp\.async\.cg|cp\.async\.ca", src)),
        mbarrier=("mbarrier" in src or "barrier_arrive" in src),
        warp_specialized=("warpgroup_reg_alloc" in src or "warpgroup_reg_dealloc" in src),
        n_lines=src.count("\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=sorted(C.SHAPES))
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    shape = C.SHAPES[a.shape]
    tune = json.load(open(os.path.join(C.RESULTS, "tune", f"{a.shape}.json")))
    os.makedirs(os.path.join(C.RESULTS, "bench", "logs"), exist_ok=True)
    os.chdir(os.path.join(C.RESULTS, "bench", "logs"))  # keep tilelang's autotuner.log out of the repo root

    import torch
    import cobench as cb
    from tilelang.profiler import do_bench

    res = dict(shape_name=a.shape, shape=shape, flops=C.flops(shape), env=C.env_info(),
               time_start=time.strftime("%Y-%m-%d %H:%M:%S"), tune=dict(
                   best_config=tune["best_config"], autotuner_latency_ms=tune["best_latency_ms"],
                   n_ok=tune["n_ok"], n_configs=tune["n_configs"], failure_summary=tune.get("failure_summary")))
    kern = C.tilelang_kernel(shape, tune["best_config"])
    res["tilelang_src"] = src_features(kern.get_kernel_source())
    inp = C.make_inputs(shape, seed=a.seed)
    args = C.call_args(shape, inp)
    tri = C.triton_fn(shape)

    # ---- 2. correctness --------------------------------------------------------------------------
    out_tl = kern(*args)
    out_tr = tri(*args)
    torch.cuda.synchronize()
    res["triton_best_config"] = C.triton_best_config(shape)
    res["check"] = C.reference_error(shape, inp, {"tilelang": out_tl, "triton": out_tr})
    res["check"]["tilelang_vs_triton"] = C.pair_error(out_tl, out_tr)
    ok = res["check"]["tilelang"]["ok"] and res["check"]["triton"]["ok"]
    print(f"[{a.shape}] check tilelang {res['check']['tilelang']} triton {res['check']['triton']}", flush=True)
    del out_tr

    f_tl = lambda: kern(*args)   # noqa: E731
    f_tr = lambda: tri(*args)    # noqa: E731

    # ---- 3. primary timer ------------------------------------------------------------------------
    v = cb.bench_variants({"tilelang": f_tl, "triton": f_tr}, reference="triton", reps=a.reps, clock=True,
                          nvml=True, label=f"mamba {a.shape}")
    print(v, flush=True)
    res["primary"] = v.to_dict()
    t_tl = v.variants["tilelang"]["total"]["median"]
    t_tr = v.variants["triton"]["total"]["median"]
    res["summary"] = dict(tilelang_us=t_tl, triton_us=t_tr, ratio_triton_over_tilelang=t_tr / t_tl,
                          tilelang_tflops=C.flops(shape) / t_tl * 1e-6, triton_tflops=C.flops(shape) / t_tr * 1e-6,
                          clock_mhz_tilelang=(v.variants["tilelang"].get("clock") or {}).get("median"),
                          clock_mhz_triton=(v.variants["triton"].get("clock") or {}).get("median"),
                          correctness_ok=ok)

    # ---- 4. autotuner top-k re-timed ---------------------------------------------------------------
    if a.topk > 0:
        alts, alt_cfgs = {}, {}
        for j, pc in enumerate(tune["per_config"][:a.topk]):
            cfg = pc["config"]
            k = kern if cfg == tune["best_config"] else C.tilelang_kernel(shape, cfg)
            o = k(*args)
            torch.cuda.synchronize()
            pe = C.pair_error(o, out_tl)
            del o
            name = f"tl_top{j}"
            alt_cfgs[name] = dict(config=cfg, autotuner_ms=pc["latency_ms"], vs_checked_tilelang=pe,
                                  ok=pe["rel_l2"] <= 1e-2, picked=(cfg == tune["best_config"]))
            if alt_cfgs[name]["ok"]:
                alts[name] = (lambda k=k: k(*args))
        alts["triton"] = f_tr
        vt = cb.bench_variants(alts, reference="triton", reps=max(50, a.reps // 2), clock=True, label=f"topk {a.shape}")
        print(vt, flush=True)
        for n, d in alt_cfgs.items():
            if n in vt.variants:
                d["median_us"] = vt.variants[n]["total"]["median"]
                d["clock_mhz"] = (vt.variants[n].get("clock") or {}).get("median")
        res["topk"] = dict(configs=alt_cfgs, triton_us=vt.variants["triton"]["total"]["median"])
    del out_tl

    # ---- 5. secondary timers (upstream methodology) ------------------------------------------------
    sec = {"tilelang": [], "triton": []}
    for _ in range(3):
        sec["tilelang"].append(do_bench(f_tl))
        sec["triton"].append(do_bench(f_tr))
    res["secondary"] = dict(
        method="tilelang.profiler.do_bench(fn) defaults: warmup 25 ms, rep 100 ms, event timer, 256 MB zero_() "
               "flush before each call, return mean; alternated 3x, median of the 3 means reported",
        tilelang_ms=statistics.median(sec["tilelang"]), triton_ms=statistics.median(sec["triton"]), raw=sec)
    res["secondary"]["ratio_triton_over_tilelang"] = res["secondary"]["triton_ms"] / res["secondary"]["tilelang_ms"]
    if shape["kind"] == "scan_bm":
        bm = [do_bench(f_tr, _n_warmup=10, _n_repeat=10) for _ in range(3)]
        res["secondary"]["triton_benchmark_script_ms"] = statistics.median(bm)
        res["secondary"]["triton_benchmark_script_raw"] = bm
    res["secondary"]["tilelang_autotuner_ms"] = tune["best_latency_ms"]
    res["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.dump(res, os.path.join(C.RESULTS, "bench", f"{a.shape}.json"))
    s = res["summary"]
    print(f"[{a.shape}] TL {s['tilelang_us']:.2f} us  Triton {s['triton_us']:.2f} us  ratio {s['ratio_triton_over_tilelang']:.3f}"
          f"  (secondary {res['secondary']['ratio_triton_over_tilelang']:.3f})  ok={ok}", flush=True)


if __name__ == "__main__":
    main()
