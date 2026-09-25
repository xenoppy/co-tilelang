"""Claims C1 / C6 (sub-study A): steady-state cross-check of the flush-mode GEMM results.

Flush mode (claims_gemm_tl.py run, cobench.bench_variants) gives every rep a ~140 us low-power
L2-flush phase that lends the power controller clock headroom, and how much a kernel profits
depends on its own power draw. This script re-times the same kernels back-to-back under
sustained load with cobench.bench_steady (primary cobench mode for power-capped kernels):
  * inputs AND outputs rotate through cobench.Rotation copies (>= 2x L2) so every iteration is
    DRAM-cold without a flush;
  * warmup until the GPU temperature plateaus, then interleaved slices (5 rounds x 1.5 s,
    first 0.4 s of each slice excluded);
  * per variant: time/iter, SM clock (ClockProbe), board power (NVML 20 ms samples), energy/iter,
    kcycles/iter.

Kernels: the TileLang config the flush-mode autotuner picked for that shape (read from
raw/<suite>_<shape>.json, not re-tuned), compiled with out_idx=None so it writes the rotating
output tensor like the baselines (the device code is the same PrimFunc; checked against the
saved kernels/<suite>_<shape>.cu). Baselines: cuBLAS (torch.matmul, out=), Triton tutorial 03 NT
(fp32acc) or tilelang-benchmark's fp16-accumulate Triton NT (fp16acc); Triton re-autotunes on
its first call (before timing).

    python claims_gemm_steady.py compile --suite fp32acc --shapes M0,M1      # CPU
    flock <gpu.lock> python claims_gemm_steady.py run --suite fp32acc --shape M0[,M1,...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from claims_gemm_common import ALL_SHAPES, BENCH, OUT_DIR, gpu_state_brief, now, rel_err, save_json  # noqa: E402

sys.path.insert(0, BENCH)
RAW_DIR = os.path.join(OUT_DIR, "raw")
SRC_DIR = os.path.join(OUT_DIR, "kernels")


def best_config(suite, shape):
    d = json.load(open(os.path.join(RAW_DIR, f"{suite}_{shape}.json")))
    return dict(d["tl_tune"]["best_config"])


def tl_prim(suite, shape):
    """The PrimFunc of the flush-mode winner, built exactly as in claims_gemm_tl.Suite."""
    import tilelang.language as T
    import claims_gemm_tl as G
    M, N, K = ALL_SHAPES[shape]
    cfg = best_config(suite, shape)
    if suite == "fp32acc":
        if "policy" in cfg:
            cfg["policy"] = T.GemmWarpPolicy(int(cfg["policy"]))
        mod = G.bm_fp16()
        return mod.matmul.jit_impl.get_tir(M, N, K, False, **cfg), cfg
    if suite == "fp16acc":
        s = G.Suite("fp16acc", M, N, K)
        return s.fp16acc_kernel()(**cfg), cfg
    raise ValueError(suite)


def compile_tl(suite, shape):
    import tilelang
    prim, cfg = tl_prim(suite, shape)
    k = tilelang.compile(prim, out_idx=None, target="auto")
    return k, cfg


def do_compile(suite, shapes):
    for sh in shapes:
        t = time.time()
        k, cfg = compile_tl(suite, sh)
        print(f"[compile] {suite} {sh}: {cfg} {time.time() - t:.1f} s", flush=True)


def do_run(suite, shape, rounds, slice_s):
    import torch
    import cobench as cb
    from baselines import triton_matmul as tm

    M, N, K = ALL_SHAPES[shape]
    flops = 2.0 * M * N * K
    res = {"suite": suite, "shape": shape, "MNK": [M, N, K], "start": now(), "mode": "steady",
           "gpu_state_before": gpu_state_brief()}
    cb.wait_until_free()
    tl_k, cfg = compile_tl(suite, shape)
    res["tl_config"] = cfg
    src = tl_k.get_kernel_source()
    flush_src_path = os.path.join(SRC_DIR, f"{suite}_{shape}.cu")
    res["same_device_source_as_flush_run"] = (open(flush_src_path).read() == src) if os.path.exists(flush_src_path) else None

    def make_inputs():
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(N, K, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)
        return A, B, C

    rot = cb.Rotation(make_inputs)
    res["rotation"] = {"copies": rot.n, "bytes_per_copy": rot.bytes_per_copy, "total_bytes": rot.total_bytes}

    if suite == "fp32acc":
        tri, tuner = tm.make_matmul("tut03")
        tri_name, ref_name = "triton_tut03_nt", "cublas"

        def cublas(i):
            A, B, C = rot[i]
            torch.matmul(A, B.T, out=C)
    else:
        tri, tuner = tm.make_matmul("tlbench_fp16acc")
        tri_name, ref_name = "triton_tlbench_fp16acc_nt", "cublas_fp16acc"

        def cublas(i):
            A, B, C = rot[i]
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            try:
                torch.matmul(A, B.T, out=C)
            finally:
                torch.backends.cuda.matmul.allow_fp16_accumulation = False

    def tl(i):
        A, B, C = rot[i]
        tl_k(A, B, C)

    def triton_v(i):
        A, B, C = rot[i]
        tri(A, B.T, out=C)

    V = {"tl": tl, ref_name: cublas, tri_name: triton_v}

    # correctness on copy 0 (triton autotunes on its first call here, before any timing)
    A, B, C = rot[0]
    ref = A.float() @ B.float().T
    checks = {}
    for n, f in V.items():
        C.fill_(float("nan"))
        f(0)
        torch.cuda.synchronize()
        checks[n] = {"rel_err_max": rel_err(C, ref)}
    tol = 5e-3 if suite == "fp32acc" else max(5e-3, 3 * checks[ref_name]["rel_err_max"])
    for c in checks.values():
        c["tol"] = tol
        c["pass"] = c["rel_err_max"] <= tol
    res["correctness"] = checks
    res["triton_best_config"] = tm.best_config(tuner)
    del ref
    failed = [n for n, c in checks.items() if not c["pass"]]
    if failed:
        res["correctness_failed"] = failed
        save_json(os.path.join(RAW_DIR, f"steady_{suite}_{shape}.json"), res)
        raise SystemExit(f"correctness failed: {failed}")
    print(f"[steady] {suite} {shape}: correctness " + " ".join(f"{n}={c['rel_err_max']:.2e}" for n, c in checks.items()),
          flush=True)

    r = cb.bench_steady(V, reference=ref_name, slice_s=slice_s, settle_s=0.4, rounds=rounds, warmup_s=5.0,
                        thermal=True, label=f"steady {suite} {shape}")
    print(r, flush=True)
    res["cobench_steady"] = r.to_dict()
    summ = {}
    for n, v in r.variants.items():
        summ[n] = {k: v.get(k) for k in ("t_iter_us", "cv_slices", "clock_mhz", "power_w", "energy_mj_per_iter",
                                         "kcycles_per_iter", "iter_p10_us", "iter_p90_us", "n_iter")}
        summ[n]["tflops"] = flops / (v["t_iter_us"] * 1e-6) / 1e12
        if v.get("energy_mj_per_iter"):
            summ[n]["tflop_per_joule"] = flops / (v["energy_mj_per_iter"] * 1e-3) / 1e12
        sp = r.derived.get("speedup", {}).get(n)
        if sp:
            summ[n]["speedup_vs_" + ref_name] = sp["ratio_of_medians"]
            summ[n]["speedup_paired_min_max"] = [sp.get("paired_min"), sp.get("paired_max")]
    res["summary"] = summ
    res["gpu_state_after"] = gpu_state_brief()
    res["end"] = now()
    save_json(os.path.join(RAW_DIR, f"steady_{suite}_{shape}.json"), res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["compile", "run"])
    ap.add_argument("--suite", required=True, choices=["fp32acc", "fp16acc"])
    ap.add_argument("--shapes", default=None)
    ap.add_argument("--shape", default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--slice", type=float, default=1.5)
    a = ap.parse_args()
    if a.phase == "compile":
        do_compile(a.suite, a.shapes.split(","))
    else:  # --shape may be a comma list: shapes run back-to-back in one process (one GPU-lock hold),
        # so the thermal plateau reached for the first shape carries over to the next ones
        for sh in a.shape.split(","):
            do_run(a.suite, sh, a.rounds, a.slice)


if __name__ == "__main__":
    main()
