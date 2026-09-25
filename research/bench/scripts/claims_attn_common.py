"""Shared helpers for the attention claim-reproduction scripts (sub-study B, claims C2 / C4).

Scripts: claims_attn_fa.py (C2, FlashAttention forward) and claims_attn_mla.py (C4, MLA decode).
Results: research/results/2026-09-24_claims_repro/B_attention/.

GPU policy (research/rules.md 7 + the claims-repro task): every script that touches the GPU is
launched under ``flock <scratchpad>/gpu.lock`` by the caller (the three sub-studies share one
GPU and must not time concurrently) and calls ``wait_gpu()`` first, which applies cobench's pmon
guard (foreign SM activity -> wait 30 min, give up after 2 h). cobench's benches re-check the
guard around every measurement point on their own.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH = os.path.join(REPO, "research", "bench")
RESULTS = os.path.join(REPO, "research", "results", "2026-09-24_claims_repro", "B_attention")
UPSTREAM = os.path.join(BENCH, "baselines", "upstream")
SMEM_PER_CTA = 101376          # sm_120 opt-in limit per CTA (bytes)

for p in (BENCH, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


def install_pytest_stub() -> None:
    """The vendored Triton attention files ``import pytest`` for their test decorators only;
    ~/mpk-env has no pytest. Register a no-op stand-in (never replaces a real pytest)."""
    try:
        import pytest  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    stub = types.ModuleType("pytest")

    class _Mark:
        def __getattr__(self, name):
            def deco(*a, **k):
                if len(a) == 1 and callable(a[0]) and not k:
                    return a[0]
                return lambda f: f
            return deco

    stub.mark = _Mark()
    stub.__co_tilelang_stub__ = True
    sys.modules["pytest"] = stub


def wait_gpu(min_free_gib: float = 0.0) -> dict:
    """rules.md 7: block while a foreign process computes on the GPU (30 min polls, 2 h cap)."""
    import cobench as cb
    t0 = time.time()
    cb.wait_until_free()
    st = cb.occupied()
    free = gpu_free_gib()
    if min_free_gib and free < min_free_gib:
        raise RuntimeError(f"only {free:.1f} GiB GPU memory free, need {min_free_gib}")
    return {"waited_s": round(time.time() - t0, 1), "occupied": bool(st.get("occupied")),
            "free_gib": round(free, 1)}


def gpu_free_gib() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip().splitlines()[0]) / 1024


def pmon_snapshot() -> str:
    try:
        return subprocess.run(["nvidia-smi", "pmon", "-s", "u", "-c", "1"], capture_output=True, text=True,
                              timeout=20).stdout
    except Exception as e:  # noqa: BLE001 - informational only
        return f"pmon failed: {e!r}"


def env_info() -> dict:
    import torch
    import triton
    info = {
        "host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "triton": triton.__version__,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
        "cc": list(torch.cuda.get_device_capability(0)),
        "sms": torch.cuda.get_device_properties(0).multi_processor_count,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import tilelang
        info["tilelang"] = tilelang.__version__
    except Exception as e:  # noqa: BLE001
        info["tilelang"] = f"import failed: {e!r}"
    try:
        import flashinfer
        info["flashinfer"] = flashinfer.__version__
    except Exception as e:  # noqa: BLE001
        info["flashinfer"] = f"import failed: {e!r}"
    try:
        info["git_head"] = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    q = "clocks.max.sm,clocks.max.mem,power.limit,driver_version,temperature.gpu"
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                                            capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return info


# ---------------------------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------------------------
def err_stats(out, ref, atol: float = 1e-2, rtol: float = 1e-2) -> dict:
    """Elementwise |out-ref| <= atol + rtol*|ref| (torch.testing.assert_close semantics, the check
    the upstream scripts use with atol=rtol=1e-2), plus summary numbers."""
    import torch
    o, r = out.float(), ref.float()
    if o.shape != r.shape:
        return {"ok": False, "error": f"shape {tuple(o.shape)} vs ref {tuple(r.shape)}"}
    d = (o - r).abs()
    finite = bool(torch.isfinite(o).all().item())
    bad = (d > atol + rtol * r.abs()) | ~torch.isfinite(o)
    n_bad = int(bad.sum().item())
    return {"ok": finite and n_bad == 0, "max_abs": float(d.max().item()), "mean_abs": float(d.mean().item()),
            "ref_max": float(r.abs().max().item()), "n_bad": n_bad, "numel": o.numel(), "finite": finite,
            "atol": atol, "rtol": rtol}


# ---------------------------------------------------------------------------------------------
# TileLang kernel resources (regs / spills / smem) from the compiled cubin + host stub
# ---------------------------------------------------------------------------------------------
def tl_resources(kernel) -> dict:
    from cotile import resources as R
    cub = R.cubin_bytes(kernel)
    ci = R.cubin_info(cub)
    host = kernel.get_host_source()
    src = kernel.get_kernel_source()
    out = {}
    for fn, r in ci["ru"].items():
        nv = ci["nv"].get(fn, {})
        d = {"regs": r["REG"], "local_bytes": r.get("LOCAL", 0), "smem_static": r["SHARED"],
             "max_threads": list(nv.get("MAX_THREADS") or [])}
        try:
            launch = R.host_launch_values(host, fn, nv.get("NUM_PARAMS", 0))
            d["launch_values"] = launch
            uses_dyn = "extern __shared__" in src
            d["smem_dynamic"] = launch[-1] if uses_dyn else 0
        except Exception as e:  # noqa: BLE001
            d["launch_values_error"] = repr(e)
        d["smem_total"] = d["smem_static"] + d.get("smem_dynamic", 0)
        d["fits_99KB"] = d["smem_total"] <= SMEM_PER_CTA
        out[fn] = d
    return out


# ---------------------------------------------------------------------------------------------
# timers
# ---------------------------------------------------------------------------------------------
def upstream_timers(fn, which=("tilelang", "triton"), tl_warmup: float = 25.0) -> dict:
    """The timers the upstream scripts use: tilelang.profiler.do_bench (event backend, L2 flush,
    mean) and triton.testing.do_bench (L2 flush, mean). Both in ms -> reported in us."""
    res = {}
    if "tilelang" in which:
        from tilelang.profiler import do_bench as tl_do_bench
        res["tilelang_do_bench_us"] = 1e3 * float(tl_do_bench(fn, warmup=tl_warmup))
    if "triton" in which:
        import triton
        res["triton_do_bench_us"] = 1e3 * float(triton.testing.do_bench(fn))
    return res


def variants_summary(vr) -> dict:
    """Compact per-variant numbers from a cobench VariantsResult."""
    out = {}
    for n, v in vr.variants.items():
        t = v["total"]
        d = {"median_us": t["median"], "p10_us": t["p10"], "p90_us": t["p90"], "mean_us": t["mean"],
             "cv": t["cv"], "n": t["n"]}
        c = v.get("clock")
        if c:
            d["clock_mhz_median"] = c.get("median")
            d["clock_mhz_min"] = c.get("min")
            d["clock_mhz_max"] = c.get("max")
            d["clock_covered"] = c.get("covered")
            d["kcycles_median"] = c["cycles_median"] / 1e3 if c.get("cycles_median") else None
        out[n] = d
    return out


def to_jsonable(x):
    import numpy as np
    import torch
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, torch.dtype):
        return str(x).replace("torch.", "")
    if isinstance(x, torch.Tensor):
        return x.tolist() if x.numel() < 64 else f"tensor{tuple(x.shape)}"
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    if hasattr(x, "to_dict"):
        return to_jsonable(x.to_dict())
    if hasattr(x, "__dict__") and not isinstance(x, type):
        return to_jsonable({k: v for k, v in vars(x).items() if not k.startswith("_")})
    return x


def save_json(obj, path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def clear_cuda_error() -> str | None:
    """Synchronise, then read-and-reset the CUDA runtime's per-thread "last error" so that a failed
    (expected) launch of one library cannot be reported by the next library's launch check.
    Returns the error name that was pending, if any."""
    import torch
    try:
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - reported below via the return value
        return f"sync: {type(e).__name__}: {str(e)[:120]}"
    # every libcudart instance loaded in this process keeps its own per-thread last error
    import ctypes
    libs = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libcudart" in line and "/" in line})
    pending = []
    for lib in libs:
        try:
            code = int(ctypes.CDLL(lib).cudaGetLastError())
        except (OSError, AttributeError):
            continue
        if code:
            pending.append(f"{os.path.basename(lib)}: cudaError {code}")
    return "; ".join(pending) or None


ISOLATED_MARK = "@@CLAIMS_ATTN_JSON@@"


def emit_isolated(rec) -> None:
    """Child side of run_isolated: print one JSON record behind a marker."""
    print(ISOLATED_MARK + json.dumps(to_jsonable(rec)), flush=True)


def run_isolated(cmd, timeout: float = 1800) -> dict:
    """Run a child process (e.g. a launch that is expected to fail) and return its JSON record;
    stderr tail and exit code are kept when it produced none."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=os.environ.copy())
    for line in r.stdout.splitlines():
        if line.startswith(ISOLATED_MARK):
            rec = json.loads(line[len(ISOLATED_MARK):])
            rec["isolated_process"] = True
            return rec
    return {"isolated_process": True, "returncode": r.returncode, "stderr_tail": r.stderr[-2000:],
            "stdout_tail": r.stdout[-1000:]}


@contextlib.contextmanager
def timed(label: str, log: dict | None = None):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {label}: {dt:.1f}s", flush=True)
    if log is not None:
        log[label] = round(dt, 2)
