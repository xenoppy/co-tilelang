"""Shared helpers for the claims reproduction, sub-study A (GEMM + dequant GEMV).

Shapes and their sources, the sm_120 shared-memory filter, correctness metrics, JSON output.
See research/results/2026-09-24_claims_repro/A_gemm/README.md.
"""
from __future__ import annotations

import json
import os
import re
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO, "research/results/2026-09-24_claims_repro/A_gemm")
BENCH = os.path.join(REPO, "research/bench")

# C1: tile-ai/tilelang-benchmark@b658f7e README "Table 1" == TileLang paper (arXiv 2504.17577)
# Appendix A Table 2.  (m, n, k); C = A[m,k] @ W[n,k]^T  (NT, as in the upstream cuBLAS runs,
# cublas_benchmark.cu inference_server_set a_t=false b_t=true).
M_SHAPES = {
    "M0": (4096, 1024, 8192), "M1": (4096, 8192, 8192), "M2": (4096, 28672, 8192),
    "M3": (4096, 8192, 28672), "M4": (8192, 1024, 8192), "M5": (8192, 8192, 8192),
    "M6": (8192, 28672, 8192), "M7": (8192, 8192, 28672),
}
# C5: same table, V0-V7 (the figure shows V0-V6); m = 1.  Upstream data file:
# ampere_benchmark/dequant_matmul/data/data_gemv.py lists the same shapes in the same order.
V_SHAPES = {
    "V0": (1, 16384, 16384), "V1": (1, 43008, 14336), "V2": (1, 14336, 14336),
    "V3": (1, 57344, 14336), "V4": (1, 14336, 57344), "V5": (1, 9216, 9216),
    "V6": (1, 36864, 9216), "V7": (1, 9216, 36864),
}
# C6: benchmark/matmul/README.md and benchmark/matmul_fp8/README.md (M = N = 8192, K sweep)
K_SWEEP = {f"K{k}": (8192, 8192, k) for k in (256, 512, 1024, 2048, 4096, 8192, 16384)}
ALL_SHAPES = {**M_SHAPES, **V_SHAPES, **K_SWEEP}

# Claimed H800 numbers (benchmark/*/README.md; the "Latency (s)" column is really ms)
C6_CLAIMED_TFLOPS = {
    "fp16": {256: 386, 512: 520, 1024: 628, 2048: 705, 4096: 736, 8192: 758, 16384: 766},
    "fp8": {256: 569, 512: 858, 1024: 1129, 2048: 1343, 4096: 1467, 8192: 1507, 16384: 1541},
}


def smem_optin_bytes() -> int:
    import torch
    return int(torch.cuda.get_device_properties(0).shared_memory_per_block_optin)


_CALL = re.compile(r"TVMFFIFunctionCall\((\w+)_packed,\s*\(TVMFFIAny\*\)\s*stack_ffi_any,\s*(\d+),")


def launch_smem_bytes(kernel) -> dict:
    """Shared memory a TileLang (tvm_ffi backend) kernel requests at launch, read from the
    generated host code: the device-function call passes its launch parameters after the
    tensor args, and the dynamic shared-memory size is the last one when the kernel uses
    dynamic smem (``extern __shared__`` in the device source). Static ``__shared__`` arrays
    are added from the device source. Returns {kernel_name: {dyn, static, total}}."""
    host = kernel.get_host_source()
    dev = kernel.get_kernel_source()
    uses_dyn = "extern __shared__" in dev
    static = 0
    for typ, dims in re.findall(r"__shared__\s+(?:__align__\(\d+\)\s+)?(\w+)\s+\w+((?:\[\d+\])+);", dev):
        n = 1
        for d in re.findall(r"\[(\d+)\]", dims):
            n *= int(d)
        size = {"half_t": 2, "half": 2, "bfloat16_t": 2, "float": 4, "uint64_t": 8, "int": 4,
                "uint": 4, "signed char": 1, "uchar": 1, "char": 1, "int64_t": 8}.get(typ, 4)
        static += n * size
    out = {}
    for m in _CALL.finditer(host):
        name, nargs = m.group(1), int(m.group(2))
        if name.startswith("__tvm"):  # runtime helpers (set_device, tensormap creation)
            continue
        head = host[:m.start()]
        pat = re.compile(r"stack_ffi_any\)\[%d\]\.v_int64\) = \(\(int64_t\)(\d+)\)" % (nargs - 1))
        vals = pat.findall(head)
        dyn = int(vals[-1]) if (uses_dyn and vals) else 0
        out[name] = {"dyn": dyn, "static": static, "total": dyn + static}
    if not out:
        # cython / C++ launcher backends: cudaLaunchKernelEx config or <<<grid, block, smem>>>
        for i, v in enumerate(re.findall(r"dynamicSmemBytes\s*=\s*(\d+)", host) +
                              re.findall(r"<<<[^,>]+,[^,>]+,\s*(\d+)", host)):
            out[f"kernel{i}"] = {"dyn": int(v), "static": static, "total": int(v) + static}
    if not out:
        raise RuntimeError("could not find a kernel launch in the host source")
    return out


def kernel_facts(kernel) -> dict:
    """What the generated CUDA looks like (sm_120 has no wgmma/tcgen05)."""
    src = kernel.get_kernel_source()
    lb = re.findall(r"__launch_bounds__\((\d+)(?:,\s*(\d+))?\)", src)
    return {
        "launch_bounds": [list(map(lambda x: int(x) if x else None, t)) for t in lb],
        "mma_sync": bool(re.search(r"mma_sync|mma\.sync|tl::mma|ptx_mma|gemm_ss|gemm_rs", src)),
        "wgmma": "wgmma" in src,
        "tcgen05": "tcgen05" in src,
        "tma_load": "tma_load" in src,
        "cp_async": "cp_async" in src,
        "warp_specialized": ("warpgroup_reg_alloc" in src) or ("reg_alloc" in src),
        "fork_l2hint": "_l2hint" in src,  # co-tilelang fork feature; must be absent here
        "src_bytes": len(src),
    }


def rel_err(out, ref) -> float:
    d = (out.float() - ref.float()).abs()
    finite = bool(d.isfinite().all())
    if not finite:
        return float("inf")
    return float(d.max() / ref.float().abs().max().clamp_min(1e-30))


def gpu_state_brief() -> dict:
    import subprocess
    q = "clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used"
    try:
        s = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return dict(zip(q.split(","), [x.strip() for x in s.split(",")]))
    except Exception as e:  # informational only
        return {"error": repr(e)}


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")
