"""Claims C1 / C6 (sub-study A): TileLang fp16 and fp8 GEMM vs cuBLAS / Triton / cuBLASLt on sm_120.

Two phases, so that the GPU lock is held only for GPU work:

  compile  (CPU only, no GPU lock needed) - compiles every config of the shipped search space
           into ~/.tilelang/cache and records the shared memory each one requests at launch
           (parsed from the generated host code). Configs above the device's opt-in limit
           (101376 B on sm_120) cannot launch; they are filtered out here and listed.
  run      (GPU; run it under `flock <gpu.lock>`) - runs the shipped autotuner on the filtered
           configs (its compile phase now hits the kernel cache), checks every timed variant
           against torch, then times TileLang and the baselines interleaved in one process with
           cobench.bench_variants (clean L2 flush, host gate, clock probe, NVML) = primary timer;
           secondary = tilelang.profiler.do_bench (event) and triton.testing.do_bench.

Suites (kernel sources are imported unmodified):
  fp32acc  TileLang benchmark/matmul/benchmark_matmul.py `matmul` (@autotune, 288-config grid,
           fp16 in / fp32 accumulate / fp16 out, C = A @ B^T) + the heuristic config of
           examples/gemm/example_gemm_autotune.py (what that example runs by default);
           baselines: cuBLAS = torch.matmul(A, B.T) (fp32 compute), Triton tutorial 03 (v3.4.0)
           with b = B.T (same NT layout).
  fp16acc  the upstream tilelang-benchmark methodology for the 4090/H100 bars (all three
           providers accumulate in fp16): TileLang = examples/gemm/example_gemm_autotune.py
           `matmul` with accum_dtype=float16, autotuned over that example's own get_configs grid
           with the autotuner settings of benchmark_matmul.py (warmup=3, rep=20);
           cuBLAS with torch.backends.cuda.matmul.allow_fp16_accumulation (CUBLAS_COMPUTE_16F,
           like cublas_benchmark.cu's CUDA_R_16F); Triton = tilelang-benchmark's fp16-accumulate
           kernel, NT (same layout) and NN (the layout that script used).
  fp8      TileLang benchmark/matmul_fp8/benchmark_matmul.py `matmul` (576-config grid, e4m3 in,
           fp32 accumulate, e4m3 out); baseline torch._scaled_mm (cuBLASLt), same out dtype.

    python claims_gemm_tl.py compile --suite fp32acc --shapes M0,M1
    flock <gpu.lock> python claims_gemm_tl.py run --suite fp32acc --shape M0
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from claims_gemm_common import (ALL_SHAPES, BENCH, OUT_DIR, REPO, gpu_state_brief, kernel_facts,  # noqa: E402
                                launch_smem_bytes, now, rel_err, save_json, smem_optin_bytes)

sys.path.insert(0, BENCH)
CFG_DIR = os.path.join(OUT_DIR, "configs")
RAW_DIR = os.path.join(OUT_DIR, "raw")
SRC_DIR = os.path.join(OUT_DIR, "kernels")


# --------------------------------------------------------------------------- TileLang sources
def _import_path(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    return mod


def bm_fp16():
    return _import_path(os.path.join(REPO, "benchmark/matmul/benchmark_matmul.py"), "benchmark_matmul_fp16")


def bm_fp8():
    return _import_path(os.path.join(REPO, "benchmark/matmul_fp8/benchmark_matmul.py"), "benchmark_matmul_fp8")


def ex_gemm():
    return _import_path(os.path.join(REPO, "examples/gemm/example_gemm_autotune.py"), "example_gemm_autotune")


_FP16ACC = {}


def _fp16acc_kernel_factory(M, N, K):
    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, thread_num=None,
               enable_rasteration=None):
        mod, T = _FP16ACC["mod"], _FP16ACC["T"]
        return mod.matmul.get_tir(M, N, K, block_M, block_N, block_K, num_stages, thread_num,
                                  enable_rasteration, dtype=T.float16, accum_dtype=T.float16)
    return kernel


class Suite:
    """What to tune and how to compile one config of it."""

    def __init__(self, name, M, N, K):
        import tilelang.language as T
        self.name, self.M, self.N, self.K = name, M, N, K
        if name == "fp32acc":
            self.mod = bm_fp16()
            self.configs = self.mod.get_configs(M, N, K, False)
        elif name == "fp8":
            self.mod = bm_fp8()
            # The shipped get_configs(args, kwargs) predates the autotuner's current calling
            # convention (configs(*kernel_args, **kernel_kwargs)) and raises TypeError when the
            # autotuner calls it; call it the old way and hand the list to the autotuner.
            self.configs = self.mod.get_configs((M, N, K, False), {})
            # The script's @jit(out_idx=[2]) returns C in float8_e4m3fn. With the default tvm_ffi
            # backend the output is allocated through torch's DLPack importer, which in torch
            # 2.8 rejects DLPack dtype code 10 (float8_e4m3fn): "MemoryError: Unsupported code
            # 10" at launch, for every config. The cython backend allocates C with torch.empty
            # instead; the generated device kernel is identical (only the host launcher differs).
            self.mod.matmul.jit_impl.execution_backend = "cython"
        elif name == "fp16acc":
            self.mod = ex_gemm()
            self.configs = self.mod.get_configs(M, N, K, False)
            self._T = T
        else:
            raise ValueError(name)

    # fp16acc: a kernel factory in the style of example_gemm_autotune.get_best_config, but fp16
    # in / fp16 accumulate (the example's own tuner path is hard-wired to bf16 + fp32 accumulate).
    # AutoTuner.run requires every closure cell of the kernel function to be a plain scalar, so
    # the example module and the dtype are reached through a module-level global.
    def fp16acc_kernel(self):
        _FP16ACC["mod"], _FP16ACC["T"] = self.mod, self._T
        return _fp16acc_kernel_factory(self.M, self.N, self.K)

    def compile_one(self, cfg):
        if self.name in ("fp32acc", "fp8"):
            return self.mod.matmul.jit_impl.compile(self.M, self.N, self.K, False, **cfg)
        from tilelang.autotuner.param import CompileArgs
        return CompileArgs(out_idx=[-1], target="auto").compile_program(self.fp16acc_kernel()(**cfg))

    def tune(self, configs):
        """Run the shipped autotuner on `configs`; returns the best JITKernel (.latency ms, .config)."""
        if self.name in ("fp32acc", "fp8"):
            self.mod.matmul.configs = configs  # AutoTuneImpl attribute; same warmup=3, rep=20
            return self.mod.matmul(self.M, self.N, self.K, False)
        import tilelang as tl
        from tilelang.autotuner import AutoTuner
        tuner = (AutoTuner.from_kernel(kernel=self.fp16acc_kernel(), configs=configs)
                 .set_compile_args(out_idx=[-1], target="auto")
                 .set_profile_args(supply_type=tl.TensorSupplyType.Auto, skip_check=True, backend="event"))
        res = tuner.run(warmup=3, rep=20)
        k = res.kernel
        k.update_tuner_result(res.latency, res.config, res.ref_latency)
        return k


def cfg_key(cfg):
    return ",".join(f"{k}={getattr(v, 'name', v)}" for k, v in cfg.items())


# --------------------------------------------------------------------------- compile phase
def do_compile(suite_name, shape, workers):
    M, N, K = ALL_SHAPES[shape]
    s = Suite(suite_name, M, N, K)
    limit = smem_optin_bytes()
    t0 = time.time()
    recs = [None] * len(s.configs)

    def one(i):
        cfg = s.configs[i]
        t = time.time()
        try:
            k = s.compile_one(cfg)
            sm = launch_smem_bytes(k)
            tot = max(v["total"] for v in sm.values())
            return i, {"config": cfg, "ok": True, "smem": tot, "launchable": tot <= limit,
                       "compile_s": time.time() - t}
        except Exception as e:
            msg = str(e)
            m = re.search(r"allowed dynamic shared memory size to (\d+)", msg)
            if m:  # launcher backends that set the smem attribute at load time (cython) fail here
                return i, {"config": cfg, "ok": True, "smem": int(m.group(1)), "launchable": False,
                           "compile_s": time.time() - t, "note": "rejected at module load"}
            return i, {"config": cfg, "ok": False, "error": f"{type(e).__name__}: {msg[:400]}",
                       "compile_s": time.time() - t}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in ex.map(one, range(len(s.configs))):
            recs[i] = r
    n_ok = sum(r["ok"] for r in recs)
    n_launch = sum(r.get("launchable", False) for r in recs)
    out = {"suite": suite_name, "shape": shape, "MNK": [M, N, K], "smem_limit": limit,
           "n_configs": len(recs), "n_compiled": n_ok, "n_launchable": n_launch,
           "wall_s": time.time() - t0, "when": now(),
           "records": [{**r, "config": cfg_key(r["config"])} for r in recs]}
    save_json(os.path.join(CFG_DIR, f"{suite_name}_{shape}.json"), out)
    errs = [r for r in recs if not r["ok"]]
    print(f"[compile] {suite_name} {shape}: {len(recs)} configs, {n_ok} compiled, {n_launchable_str(n_launch, limit)}, "
          f"{len(errs)} compile errors, {out['wall_s']:.0f} s", flush=True)
    for r in errs[:5]:
        print("   error:", cfg_key(r["config"]), r["error"][:200])


def n_launchable_str(n, limit):
    return f"{n} with smem <= {limit} B"


# --------------------------------------------------------------------------- run phase
def make_inputs(suite, M, N, K, seed=0):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    if suite == "fp8":
        # |C| stays well below e4m3's max (448) for K <= 16384: std(C) ~ 0.25 * sqrt(K)
        A = (torch.randn(M, K, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
        B = (torch.randn(N, K, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    else:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)
        B = torch.randn(N, K, device="cuda", dtype=torch.float16, generator=g)
    return A, B


def blas_libs():
    """Which cuBLAS / cuBLASLt files this process has mapped (torch's pip wheel 12.8 vs the
    CUDA 12.9 toolkit when LD_PRELOADed)."""
    libs = set()
    with open("/proc/self/maps") as f:
        for line in f:
            p = line.split()[-1]
            if "libcublas" in p:
                libs.add(os.path.realpath(p))
    return sorted(libs)


def cuda_kernel_names(fn):
    """Names of the CUDA kernels one call of `fn` launches (torch profiler / CUPTI); used to
    identify which cuBLAS / cuBLASLt kernel the library picked on sm_120."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return sorted({e.name for e in prof.events() if str(e.device_type).endswith("CUDA")})
    except Exception as e:  # informational only
        return [f"profiler error: {e!r}"[:200]]


def do_run(suite_name, shape, reps, skip_secondary, tag="", cublas_nn=False):
    import torch
    import tilelang
    import cobench as cb
    from tilelang.profiler import do_bench as tl_do_bench
    import triton.testing

    M, N, K = ALL_SHAPES[shape]
    flops = 2.0 * M * N * K
    res = {"suite": suite_name, "shape": shape, "MNK": [M, N, K], "start": now(),
           "gpu_state_before": gpu_state_brief(), "torch": torch.__version__,
           "tilelang": tilelang.__version__}
    cb.wait_until_free()  # rules.md 7: foreign SM activity -> wait 30 min (GpuBusy after 2 h)

    cfg_path = os.path.join(CFG_DIR, f"{suite_name}_{shape}.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"run the compile phase first: {cfg_path} missing")
    import json
    cinfo = json.load(open(cfg_path))
    s = Suite(suite_name, M, N, K)
    keep = {r["config"] for r in cinfo["records"] if r.get("launchable")}
    configs = [c for c in s.configs if cfg_key(c) in keep]
    res["configs"] = {"n_grid": len(s.configs), "n_launchable": len(configs),
                      "smem_limit": cinfo["smem_limit"],
                      "dropped_smem": [r["config"] for r in cinfo["records"] if r["ok"] and not r["launchable"]],
                      "dropped_compile_error": [(r["config"], r["error"]) for r in cinfo["records"] if not r["ok"]]}

    # ---- TileLang autotune (shipped autotuner; its compile phase hits the kernel cache)
    t0 = time.time()
    tl_k = s.tune(configs)
    res["tl_tune"] = {"wall_s": time.time() - t0, "best_config": tl_k.config,
                      "autotuner_latency_ms": tl_k.latency,
                      "autotuner_tflops": flops / (tl_k.latency * 1e-3) / 1e12 if tl_k.latency else None,
                      "facts": kernel_facts(tl_k), "smem": launch_smem_bytes(tl_k)}
    os.makedirs(SRC_DIR, exist_ok=True)
    with open(os.path.join(SRC_DIR, f"{suite_name}_{shape}{tag}.cu"), "w") as f:
        f.write(tl_k.get_kernel_source())
    print(f"[run] {suite_name} {shape}: tuned in {res['tl_tune']['wall_s']:.0f} s, best {tl_k.config}, "
          f"autotuner {tl_k.latency:.4f} ms", flush=True)

    A, B = make_inputs(suite_name, M, N, K)
    variants, outs = {}, {}

    if suite_name == "fp8":
        ref = (A.float() @ B.float().T)
        one = torch.ones((), device="cuda")
        out_dtype = torch.float8_e4m3fn
        try:
            torch._scaled_mm(A, B.T, scale_a=one, scale_b=one, out_dtype=out_dtype)
        except Exception as e:  # recorded, then fall back to fp16 output
            res["scaled_mm_fp8_out_error"] = repr(e)[:300]
            out_dtype = torch.float16
        res["scaled_mm_out_dtype"] = str(out_dtype)
        variants["tl"] = lambda: tl_k(A, B)
        variants["cublaslt_scaled_mm"] = lambda: torch._scaled_mm(A, B.T, scale_a=one, scale_b=one,
                                                                  out_dtype=out_dtype)
        reference = "cublaslt_scaled_mm"
        tol = 0.07  # e4m3 output: 3 mantissa bits -> <= 1/16 relative rounding
    else:
        ref = (A.float() @ B.float().T)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)
        from baselines import triton_matmul as tm
        if suite_name == "fp32acc":
            ex = ex_gemm()
            heur = ex.get_heuristic_config()
            heur_k = ex.matmul(M, N, K, **heur)
            res["tl_example_heuristic"] = {"config": heur, "facts": kernel_facts(heur_k),
                                           "smem": launch_smem_bytes(heur_k)}
            tri, tri_tuner = tm.make_matmul("tut03")
            Bt = B.T  # (K, N) view, strides (1, K): same NT operands as TileLang / cuBLAS

            def cublas():
                torch.matmul(A, B.T, out=C)
                return C
            heur_ok = max(v["total"] for v in res["tl_example_heuristic"]["smem"].values()) <= smem_optin_bytes()
            variants["tl"] = lambda: tl_k(A, B)
            if heur_ok:
                variants["tl_example_heuristic"] = lambda: heur_k(A, B)
            variants["cublas"] = cublas
            if cublas_nn:  # informational: cuBLAS on a row-major (K, N) copy of B (NN layout)
                Bkn_c = B.T.contiguous()
                Cn = torch.empty(M, N, device="cuda", dtype=torch.float16)

                def cublas_nn_fn():
                    torch.matmul(A, Bkn_c, out=Cn)
                    return Cn
                variants["cublas_nn"] = cublas_nn_fn
            variants["triton_tut03_nt"] = lambda: tri(A, Bt)
            tuners = {"triton_tut03_nt": tri_tuner}
            reference = "cublas"
            tol = 5e-3
        else:  # fp16acc
            tri_nt, tu_nt = tm.make_matmul("tlbench_fp16acc")
            tri_nn, tu_nn = tm.make_matmul("tlbench_fp16acc")
            Bt = B.T
            Bkn = B.T.contiguous()  # row-major (K, N): the upstream Triton script's layout

            def cublas16():
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
                try:
                    torch.matmul(A, B.T, out=C)
                finally:
                    torch.backends.cuda.matmul.allow_fp16_accumulation = False
                return C
            variants["tl"] = lambda: tl_k(A, B)
            variants["cublas_fp16acc"] = cublas16
            variants["triton_tlbench_fp16acc_nt"] = lambda: tri_nt(A, Bt)
            variants["triton_tlbench_fp16acc_nn"] = lambda: tri_nn(A, Bkn)
            tuners = {"triton_tlbench_fp16acc_nt": tu_nt, "triton_tlbench_fp16acc_nn": tu_nn}
            reference = "cublas_fp16acc"
            tol = None  # set from cuBLAS's own fp16-accumulate error below

    # ---- correctness (every timed variant; triton autotunes on its first call here)
    t0 = time.time()
    checks = {}
    for n, f in variants.items():
        o = f().clone()
        torch.cuda.synchronize()
        checks[n] = {"rel_err_max": rel_err(o, ref), "finite": bool(o.float().isfinite().all())}
        outs[n] = o
    if suite_name == "fp16acc":
        tol = max(5e-3, 3.0 * checks["cublas_fp16acc"]["rel_err_max"])
    for n in checks:
        checks[n]["tol"] = tol
        checks[n]["pass"] = checks[n]["finite"] and checks[n]["rel_err_max"] <= tol
    res["correctness"] = checks
    res["correctness_wall_s"] = time.time() - t0
    res["blas_libs"] = blas_libs()
    res["kernel_names"] = {n: cuda_kernel_names(f) for n, f in variants.items() if not n.startswith("tl")}
    if suite_name != "fp8":
        res["triton_best_config"] = {n: tm.best_config(t) for n, t in tuners.items()}
    del outs
    print(f"[run] {suite_name} {shape}: correctness " +
          " ".join(f"{n}={c['rel_err_max']:.2e}{'' if c['pass'] else '(FAIL)'}" for n, c in checks.items()), flush=True)
    failed = [n for n, c in checks.items() if not c["pass"]]
    if failed:
        res["correctness_failed"] = failed
    timed = {n: f for n, f in variants.items() if n not in failed}

    # ---- primary timer: cobench clean-flush interleaved
    r = cb.bench_variants(timed, reference=reference if reference in timed else None, reps=reps,
                          clock=True, nvml=True, label=f"{suite_name} {shape}")
    print(r, flush=True)
    res["cobench"] = r.to_dict()
    res["summary"] = {}
    for n, v in r.variants.items():
        t = v["total"]["median"]
        res["summary"][n] = {"median_us": t, "tflops": flops / (t * 1e-6) / 1e12,
                             "clock_mhz": (v.get("clock") or {}).get("median"),
                             "cv": v["total"]["cv"]}
    if reference in r.variants:
        rt = r.variants[reference]["total"]["median"]
        for n in res["summary"]:
            res["summary"][n]["speedup_vs_" + reference] = rt / res["summary"][n]["median_us"]

    # ---- secondary timers (upstream methodology: mean over a time budget, 256 MB L2 flush)
    if not skip_secondary:
        sec = {}
        for n, f in timed.items():
            cb.wait_until_free()
            sec[n] = {"tilelang_do_bench_event_ms": tl_do_bench(f, backend="event"),
                      "triton_do_bench_ms": triton.testing.do_bench(f)}
        res["secondary"] = sec
        print("[run] secondary (ms): " + " ".join(
            f"{n}={v['tilelang_do_bench_event_ms']:.4f}/{v['triton_do_bench_ms']:.4f}" for n, v in sec.items()), flush=True)
    res["gpu_state_after"] = gpu_state_brief()
    res["end"] = now()
    res["tag"] = tag
    save_json(os.path.join(RAW_DIR, f"{suite_name}_{shape}{tag}.json"), res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["compile", "run"])
    ap.add_argument("--suite", required=True, choices=["fp32acc", "fp16acc", "fp8"])
    ap.add_argument("--shapes", default=None, help="comma list (compile)")
    ap.add_argument("--shape", default=None, help="one shape (run)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--skip-secondary", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output files (e.g. _cublas129)")
    ap.add_argument("--cublas-nn", action="store_true", help="fp32acc: also time cuBLAS on an NN copy of B")
    a = ap.parse_args()
    if a.phase == "compile":
        for sh in a.shapes.split(","):
            try:
                do_compile(a.suite, sh, a.workers)
            except Exception:
                traceback.print_exc()
                raise
    else:
        do_run(a.suite, a.shape, a.reps, a.skip_secondary, a.tag, a.cublas_nn)


if __name__ == "__main__":
    main()
