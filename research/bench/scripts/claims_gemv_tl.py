"""Claim C5 (sub-study A), fork side: TileLang's own dequant GEMV example vs cuBLAS fp16 GEMV.

examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py (imported unmodified). The claim's
artifact (BitBLAS 0.1.0.post1) cannot run on sm_120 (see README), so this is the TileLang code
that can. Variants, all built by the example's `dequantize_gemv` generator with its shipped
schedule (n_partition=4, reduce_thread=32, lop3 fast decoding on interleaved weights, no
scaling, no autotuner):
  uint4_fp16  exactly the example's main()/run_regression_perf() configuration (W uint4, A fp16,
              fp16 accumulate and output)                          -> compare with W_INT4A_FP16
  int2_fp16   the same generator with num_bits=2; its lop3 path cannot run (bug in the example's
              interleave helper, see COMBOS), so the generic signed decode is used
                                                                   -> compare with W_INT2A_FP16
  uint2_int8  num_bits=2, A int8, int32 accumulate/output (the generator's dp4a path, lop3)
                                                                   -> compare with W_INT2A_INT8
Shapes V0-V7 (m = 1).

Correctness: the example's own check (atol=1e3, reference built from the *interleaved*
weights) cannot fail, so every output is compared with a float64 reference built from the
logical (pre-interleave) weights; the uint4 logical weights, as fp16, also feed the cuBLAS
baseline. Primary timer: cobench.bench_variants (clean flush, clock), all variants of a shape
interleaved. Secondary: tilelang.profiler.do_bench (cupti, as run_regression_perf; and event)
and triton.testing.do_bench.

    source research/env.sh; flock <gpu.lock> python claims_gemv_tl.py --shape V0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from claims_gemm_common import (BENCH, OUT_DIR, REPO, V_SHAPES, gpu_state_brief, kernel_facts,  # noqa: E402
                                now, rel_err, save_json)

sys.path.insert(0, BENCH)
RAW_DIR = os.path.join(OUT_DIR, "raw")
EX_DIR = os.path.join(REPO, "examples/dequantize_gemm")
COMBOS = {  # name: (num_bits, activation dtype, accum/out dtype, weight bytes/element, lop3 fast decoding)
    "uint4_fp16": (4, "float16", "float16", 0.5, True),
    # the example's 2-bit fp16 fast-decoding path cannot run: quantize/utils.py interleave_weight
    # calls torch.int32(0xFF0000FF) (a dtype, not callable) for nbits=2/1 with a float16 target.
    # Use the generator's generic decode instead (fast_decoding=False: sign-extending
    # conversion, i.e. signed int2, no interleave).
    "int2_fp16": (2, "float16", "float16", 0.25, False),
    "uint2_int8": (2, "int8", "int32", 0.25, True),
}


def unpack(qB, nbits, N, K, signed=False):
    """Logical weights (N, K) from the packed int8 tensor (element e of a byte at bits
    [e*nbits, (e+1)*nbits), lowest first - the example's reference convention); `signed`
    sign-extends each field like the generator's generic (non-lop3) decode."""
    import torch
    q = qB.view(torch.uint8)
    per = 8 // nbits
    parts = [((q >> (nbits * e)) & ((1 << nbits) - 1)).to(torch.int16) for e in range(per)]
    w = torch.stack(parts, dim=-1).reshape(N, K)
    if signed:
        w = torch.where(w >= (1 << (nbits - 1)), w - (1 << nbits), w)
    return w


def ref_mm(A, W, chunk=4096):
    """float64 A @ W.T, chunked over the rows of W (V3/V4 weights are 0.8 G elements)."""
    import torch
    out = torch.empty(A.shape[0], W.shape[0], device="cuda", dtype=torch.float64)
    Ad = A.double()
    for i in range(0, W.shape[0], chunk):
        out[:, i:i + chunk] = Ad @ W[i:i + chunk].double().T
    return out


def run(shape, reps, skip_secondary, combos):
    import torch
    import tilelang
    import tilelang.language as T
    import cobench as cb
    import triton.testing
    from tilelang.profiler import do_bench as tl_do_bench
    sys.path.insert(0, EX_DIR)
    import example_dequant_gemv_fp16xint4 as ex
    from quantize.utils import interleave_weight

    M, N, K = V_SHAPES[shape]
    res = {"shape": shape, "MNK": [M, N, K], "start": now(), "tilelang": tilelang.__version__,
           "gpu_state_before": gpu_state_brief(), "combos": {}}
    cb.wait_until_free()
    g = torch.Generator(device="cuda").manual_seed(0)

    # cuBLAS fp16 GEMV baseline on the uint4 logical weights (exact in fp16)
    A16 = (torch.randn(M, K, device="cuda", generator=g) * 0.01).half()
    q4 = torch.randint(-128, 128, (N, K // 2), device="cuda", generator=g, dtype=torch.int32).to(torch.int8)
    W16 = unpack(q4, 4, N, K).half()
    Cc = torch.empty(M, N, device="cuda", dtype=torch.float16)

    def cublas16():
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        try:
            torch.matmul(A16, W16.T, out=Cc)
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
        return Cc

    def cublas32():
        torch.matmul(A16, W16.T, out=Cc)
        return Cc

    variants = {"cublas_fp16acc": cublas16, "cublas": cublas32}
    wbytes = {"cublas_fp16acc": 2.0, "cublas": 2.0}
    checks = {}
    ref16 = ref_mm(A16, W16)
    for n, f in variants.items():
        checks[n] = {"rel_err_max": rel_err(f().clone(), ref16)}
    tol16 = max(1e-2, 3 * checks["cublas_fp16acc"]["rel_err_max"])
    checks["cublas_fp16acc"]["tol"] = tol16
    checks["cublas"]["tol"] = 5e-3

    for combo in combos:
        nbits, act, acc, wb, fast = COMBOS[combo]
        name = "tl_" + combo
        info = res["combos"][combo] = {}
        in_dtype, acc_dtype = getattr(T, act), getattr(T, acc)
        cfg = dict(in_dtype=in_dtype, out_dtype=acc_dtype, accum_dtype=acc_dtype, num_bits=nbits,
                   storage_dtype=T.int8, source_format="uint", n_partition=4, reduce_thread=32,
                   fast_decoding=fast, trans_A=False, trans_B=True, group_size=-1, with_scaling=False)
        info["config"] = {k: str(v) for k, v in cfg.items()}
        t0 = time.time()
        try:
            kernel = ex.dequantize_gemv(M, N, K, cfg["in_dtype"], cfg["out_dtype"], cfg["accum_dtype"],
                                        cfg["num_bits"], cfg["storage_dtype"], cfg["source_format"],
                                        cfg["n_partition"], cfg["reduce_thread"], cfg["fast_decoding"],
                                        cfg["trans_A"], cfg["trans_B"], cfg["group_size"], cfg["with_scaling"])
        except Exception as e:
            info["build_error"] = f"{type(e).__name__}: {str(e)[-1200:]}"
            info["traceback"] = traceback.format_exc()[-2500:]
            print(f"[{shape}] {combo}: BUILD FAILED {info['build_error'][:300]}", flush=True)
            continue
        info["build_s"] = time.time() - t0
        info["facts"] = kernel_facts(kernel)
        if combo == "uint4_fp16":
            A, qB, Wlog = A16, q4, W16
        else:
            if act == "int8":
                A = torch.randint(-64, 65, (M, K), device="cuda", generator=g, dtype=torch.int32).to(torch.int8)
            else:
                A = (torch.randn(M, K, device="cuda", generator=g) * 0.01).half()
            qB = torch.randint(-128, 128, (N, K * nbits // 8), device="cuda", generator=g,
                               dtype=torch.int32).to(torch.int8)
            Wlog = unpack(qB, nbits, N, K, signed=not fast).to(torch.int8 if act == "int8" else torch.float16)
        qB_il = interleave_weight(qB, nbits, act) if fast else qB
        C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, acc))

        def fn(kernel=kernel, A=A, qB_il=qB_il, C=C):
            kernel(A, qB_il, C)
            return C
        try:
            o = fn().clone()
            torch.cuda.synchronize()
        except Exception as e:
            info["run_error"] = f"{type(e).__name__}: {str(e)[-800:]}"
            print(f"[{shape}] {combo}: RUN FAILED {info['run_error'][:300]}", flush=True)
            continue
        ref = ref_mm(A, Wlog)
        c = {"rel_err_max": rel_err(o, ref)}
        if acc == "int32":
            c["exact_int"] = bool((o.double() == ref).all())
            c["tol"] = 0.0
            c["pass"] = c["exact_int"]
        else:
            c["tol"] = tol16
        if combo == "uint4_fp16":  # what the example itself compares against
            qi = qB_il.view(torch.uint8)
            B_il = torch.stack((qi & 0x0F, qi >> 4), dim=-1).reshape(N, K).half()
            c["rel_err_vs_example_reference"] = rel_err(o, ref_mm(A, B_il))
            del B_il, qi
        checks[name] = c
        variants[name] = fn
        wbytes[name] = wb
        del ref
    for n, c in checks.items():
        if "pass" not in c:
            c["pass"] = c["rel_err_max"] <= c["tol"]
    res["correctness"] = checks
    failed = [n for n, c in checks.items() if not c["pass"]]
    res["correctness_failed"] = failed
    print(f"[{shape}] correctness " + " ".join(f"{n}={c['rel_err_max']:.2e}{'' if c['pass'] else '(FAIL)'}"
                                              for n, c in checks.items()), flush=True)
    timed = {n: f for n, f in variants.items() if n not in failed}
    r = cb.bench_variants(timed, reference="cublas_fp16acc" if "cublas_fp16acc" in timed else None,
                          reps=reps, clock=True, nvml=True, label=f"tl gemv {shape}")
    print(r, flush=True)
    res["cobench"] = r.to_dict()
    ref_t = r.variants.get("cublas_fp16acc", {}).get("total", {}).get("median")
    res["summary"] = {n: {"median_us": v["total"]["median"],
                          "weight_GBps": N * K * wbytes[n] / (v["total"]["median"] * 1e-6) / 1e9,
                          "clock_mhz": (v.get("clock") or {}).get("median"), "cv": v["total"]["cv"],
                          "speedup_vs_cublas_fp16acc": (ref_t / v["total"]["median"]) if ref_t else None}
                      for n, v in r.variants.items()}
    # backwards-compatible alias for the example's own configuration
    if "tl_uint4_fp16" in res["summary"]:
        res["summary"]["tl_dequant_gemv_int4"] = res["summary"]["tl_uint4_fp16"]
    if not skip_secondary:
        sec = {}
        for n, f in timed.items():
            cb.wait_until_free()
            sec[n] = {"tilelang_do_bench_cupti_ms": tl_do_bench(f, backend="cupti"),
                      "tilelang_do_bench_event_ms": tl_do_bench(f, backend="event"),
                      "triton_do_bench_ms": triton.testing.do_bench(f)}
        res["secondary"] = sec
        print("[secondary] " + str(sec), flush=True)
    res["gpu_state_after"] = gpu_state_brief()
    res["end"] = now()
    save_json(os.path.join(RAW_DIR, f"gemv_tl_{shape}.json"), res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--combos", default=",".join(COMBOS))
    ap.add_argument("--skip-secondary", action="store_true")
    a = ap.parse_args()
    run(a.shape, a.reps, a.skip_secondary, a.combos.split(","))


if __name__ == "__main__":
    main()
