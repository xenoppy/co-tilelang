"""Claim C5 (sub-study A): BitBLAS(-TileLang) dequantized GEMV vs cuBLAS fp16 GEMV on sm_120.

The claim (images/op_benchmark_a100_wq_gemv.png, A100) was produced with BitBLAS on its TileLang
backend (tile-ai/tilelang-benchmark@b658f7e ampere_benchmark/dequant_matmul/4.bitblas_benchmark:
bitblas.Matmul(MatmulConfig(M, N, K, A, W, out, accum, "nt", no bias/scaling/zeros),
enable_tuning=True), latency = matmul.profile_latency()) against cuBLAS fp16 GEMV
(0.cublas-benchmark/cublas_benchmark.cu: cublasGemmEx, CUDA_R_16F compute, W transposed).

This script runs exactly those BitBLAS operators (pip bitblas 0.1.0.post1 with its bundled TVM
and TileLang - NOT the co-tilelang fork; it must run in a process without the fork's tilelang,
so do not source research/env.sh's PYTHONPATH) and cuBLAS through torch in the same process:
  * correctness of every operator against a torch fp32/fp64 reference built from the unpacked
    weights;
  * primary timer: cobench.bench_variants, all operators of one shape interleaved (clean L2
    flush, host gate, clock probe);
  * secondary: BitBLAS's own profile_latency() (TVM time_evaluator, warm L2) and
    triton.testing.do_bench.

    env -u PYTHONPATH python claims_gemv_bitblas.py --shape V0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from claims_gemm_common import BENCH, OUT_DIR, V_SHAPES, gpu_state_brief, now, rel_err, save_json  # noqa: E402

sys.path.insert(0, BENCH)
RAW_DIR = os.path.join(OUT_DIR, "raw")

# (A_dtype, W_dtype, out_dtype, accum_dtype) as in the upstream benchmark_bitblas_matmul.sh
COMBOS = {
    "W_INT4A_FP16": ("float16", "int4", "float16", "float16"),
    "W_INT2A_FP16": ("float16", "int2", "float16", "float16"),
    "W_INT2A_INT8": ("int8", "int2", "int32", "int32"),
    "W_NF4A_FP16": ("float16", "nf4", "float16", "float16"),
}


def make_operand(combo, M, N, K, g):
    """Activation, logical weight (what the reference multiplies) and the raw weight handed to
    BitBLAS's transform_weight."""
    import torch
    A_dtype, W_dtype, _, _ = COMBOS[combo]
    if A_dtype == "float16":
        A = (torch.randn(M, K, device="cuda", generator=g) * 0.05).half()
    else:  # int8 activations: |sum| <= 16*2*K < 2^24 -> the fp32 reference is exact
        A = torch.randint(-16, 17, (M, K), device="cuda", generator=g, dtype=torch.int32).to(torch.int8)
    if W_dtype == "int4":
        Wraw = torch.randint(-8, 8, (N, K), device="cuda", generator=g, dtype=torch.int32).to(torch.int8)
        Wlog = Wraw.half()  # exact
    elif W_dtype == "int2":
        Wraw = torch.randint(-2, 2, (N, K), device="cuda", generator=g, dtype=torch.int32).to(torch.int8)
        Wlog = Wraw.half()
    elif W_dtype == "nf4":
        Wraw = torch.randint(0, 16, (N, K), device="cuda", generator=g, dtype=torch.int32).to(torch.int8)
        Wlog = None  # needs the operator's LUT
    else:
        raise ValueError(W_dtype)
    return A, Wraw, Wlog


def reference(A, Wlog, chunk=4096):
    import torch
    out = torch.empty(A.shape[0], Wlog.shape[0], device="cuda", dtype=torch.float64)
    Ad = A.double()
    for i in range(0, Wlog.shape[0], chunk):
        out[:, i:i + chunk] = Ad @ Wlog[i:i + chunk].double().T
    return out


def run_shape(shape, reps, combos, skip_secondary, target_override=None, tag=""):
    import torch
    import bitblas
    import cobench as cb
    import triton.testing
    from bitblas.utils.target_detector import auto_detect_nvidia_target

    M, N, K = V_SHAPES[shape]
    res = {"shape": shape, "MNK": [M, N, K], "start": now(), "bitblas": bitblas.__version__,
           "torch": torch.__version__, "gpu_state_before": gpu_state_brief()}
    target = target_override or auto_detect_nvidia_target()
    res["bitblas_target"] = str(target)
    res["bitblas_target_overridden"] = bool(target_override)
    cb.wait_until_free()
    g = torch.Generator(device="cuda").manual_seed(0)

    # cuBLAS fp16 GEMV, NT (cublasGemmEx with W transposed), fp16 compute like the upstream
    # CUDA_R_16F run, plus torch's default fp32 compute
    Af = (torch.randn(M, K, device="cuda", generator=g) * 0.05).half()
    Wf = torch.randn(N, K, device="cuda", generator=g).half()
    Cf = torch.empty(M, N, device="cuda", dtype=torch.float16)

    def cublas16():
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        try:
            torch.matmul(Af, Wf.T, out=Cf)
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
        return Cf

    def cublas32():
        torch.matmul(Af, Wf.T, out=Cf)
        return Cf

    variants = {"cublas_fp16acc": cublas16, "cublas": cublas32}
    checks, ops, info = {}, {}, {}
    ref_f = reference(Af, Wf)
    for n, f in variants.items():
        checks[n] = {"rel_err_max": rel_err(f().clone(), ref_f)}

    for combo in combos:
        A_dtype, W_dtype, out_dtype, accum_dtype = COMBOS[combo]
        t0 = time.time()
        try:
            cfg = bitblas.MatmulConfig(M, N, K, A_dtype, W_dtype, out_dtype, accum_dtype, "nt",
                                       False, None, False, False, None)
            op = bitblas.Matmul(cfg, target=target, enable_tuning=True)
        except Exception as e:
            info[combo] = {"build_error": f"{type(e).__name__}: {str(e)[-1500:]}",
                           "traceback": traceback.format_exc()[-3000:], "build_s": time.time() - t0}
            print(f"[{shape}] {combo}: BUILD FAILED {type(e).__name__}: {str(e)[-300:]}", flush=True)
            continue
        info[combo] = {"build_s": time.time() - t0}
        try:
            info[combo]["hint"] = str(getattr(op, "hint", None) or getattr(op, "config", None))[:500]
            src = op.get_source() if hasattr(op, "get_source") else None
            if src:
                os.makedirs(os.path.join(OUT_DIR, "kernels"), exist_ok=True)
                with open(os.path.join(OUT_DIR, "kernels", f"bitblas_{shape}_{combo}.cu"), "w") as f:
                    f.write(src)
                info[combo]["src_has_lop3"] = "lop3" in src
                info[combo]["src_has_mma"] = "mma" in src
                info[combo]["src_has_dp4a"] = "dp4a" in src or "__dp4a" in src
        except Exception as e:
            info[combo]["source_error"] = repr(e)[:300]
        A, Wraw, Wlog = make_operand(combo, M, N, K, g)
        if Wlog is None:  # nf4: logical value = LUT[index]
            Wlog = op.lut.half()[Wraw.long()]  # the fp16 LUT the kernel uses
        Wq = op.transform_weight(Wraw)
        out = torch.empty(M, N, device="cuda", dtype=getattr(torch, out_dtype))
        o = op(A, Wq, output=out).clone()
        torch.cuda.synchronize()
        ref = reference(A, Wlog)
        checks["bitblas_" + combo] = {"rel_err_max": rel_err(o, ref),
                                      "exact_int": bool((o.double() == ref).all()) if out_dtype == "int32" else None}
        del ref, Wlog, Wraw
        ops[combo] = (op, A, Wq, out)
        variants["bitblas_" + combo] = (lambda op=op, A=A, Wq=Wq, out=out: op(A, Wq, output=out))
        print(f"[{shape}] {combo}: built in {info[combo]['build_s']:.0f} s, rel_err "
              f"{checks['bitblas_' + combo]['rel_err_max']:.2e}", flush=True)
    # tolerances: fp16-accumulating operators vs cuBLAS's own fp16-accumulate error
    tol16 = max(1e-2, 3 * checks["cublas_fp16acc"]["rel_err_max"])
    for n, c in checks.items():
        if n.endswith("INT8"):
            c["tol"] = 0.0
            c["pass"] = bool(c["exact_int"])
        else:
            c["tol"] = tol16 if ("fp16acc" in n or "FP16" in n) else 5e-3
            c["pass"] = c["rel_err_max"] <= c["tol"]
    res["correctness"], res["bitblas_ops"] = checks, info
    failed = [n for n, c in checks.items() if not c["pass"]]
    res["correctness_failed"] = failed
    timed = {n: f for n, f in variants.items() if n not in failed}
    cb.wait_until_free()
    r = cb.bench_variants(timed, reference="cublas_fp16acc" if "cublas_fp16acc" in timed else None,
                          reps=reps, clock=True, nvml=True, label=f"gemv {shape}")
    print(r, flush=True)
    res["cobench"] = r.to_dict()
    wbytes = {"cublas_fp16acc": 2.0, "cublas": 2.0, "bitblas_W_INT4A_FP16": 0.5, "bitblas_W_INT2A_FP16": 0.25,
              "bitblas_W_INT2A_INT8": 0.25, "bitblas_W_NF4A_FP16": 0.5}
    res["summary"] = {}
    ref_t = r.variants.get("cublas_fp16acc", {}).get("total", {}).get("median")
    for n, v in r.variants.items():
        t = v["total"]["median"]
        res["summary"][n] = {"median_us": t, "weight_GBps": N * K * wbytes[n] / (t * 1e-6) / 1e9,
                             "clock_mhz": (v.get("clock") or {}).get("median"), "cv": v["total"]["cv"],
                             "speedup_vs_cublas_fp16acc": (ref_t / t) if ref_t else None}
    if not skip_secondary:
        sec = {}
        for n, f in timed.items():
            cb.wait_until_free()
            sec[n] = {"triton_do_bench_ms": triton.testing.do_bench(f)}
            if n.startswith("bitblas_"):
                try:
                    sec[n]["bitblas_profile_latency_ms"] = float(ops[n[len("bitblas_"):]][0].profile_latency())
                except Exception as e:
                    sec[n]["bitblas_profile_latency_error"] = repr(e)[:300]
        res["secondary"] = sec
        print("[secondary] " + str(sec), flush=True)
    res["gpu_state_after"] = gpu_state_brief()
    res["end"] = now()
    res["tag"] = tag
    save_json(os.path.join(RAW_DIR, f"gemv_bitblas_{shape}{tag}.json"), res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--combos", default=",".join(COMBOS))
    ap.add_argument("--skip-secondary", action="store_true")
    ap.add_argument("--target", default=None,
                    help="TVM target for BitBLAS (default: its auto-detection, 'cuda' + sm_120 here)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if "tilelang" in sys.modules or any(p.rstrip("/").endswith("co-tilelang") for p in sys.path):
        print("warning: the co-tilelang source tree is on sys.path; BitBLAS inserts its bundled "
              "tilelang/tvm first, but run with `env -u PYTHONPATH` to be safe", flush=True)
    run_shape(a.shape, a.reps, a.combos.split(","), a.skip_secondary, a.target, a.tag)


if __name__ == "__main__":
    main()
