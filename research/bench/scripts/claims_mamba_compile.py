"""Pre-compile every autotuner config of one Mamba-2 shape into the TileLang kernel cache (CPU only).

    source research/env.sh
    python research/bench/scripts/claims_mamba_compile.py --shape CC0 [--workers 12]

The upstream autotuner compiles its configs in a thread pool; lowering is GIL-bound, so on this
machine it compiles ~7 s per config serially (90 configs = 10 min) *while* claims_mamba_tune.py
holds the GPU lock. Compiling them first, in separate processes and without the lock, makes the
autotuner's own compile step a kernel-cache hit. No kernel is launched here (compile + module load
only), so this runs without the GPU lock. The autotuner itself is unchanged.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _work(args):
    shape_name, idxs = args
    import claims_mamba_common as C
    shape = C.SHAPES[shape_name]
    mod = C.module(shape["kind"])
    fn = mod.chunk_scan_fwd if shape["kind"] in ("scan_ex", "scan_bm") else mod.chunk_state_fwd
    configs = fn.configs() if callable(fn.configs) else fn.configs
    out = []
    for i in idxs:
        t0 = time.time()
        try:
            C.tilelang_kernel(shape, configs[i])
            out.append((i, "ok", time.time() - t0, ""))
        except Exception as e:  # reported, not silenced: the tune step records the same failure
            out.append((i, "error", time.time() - t0, "".join(traceback.format_exception_only(type(e), e))[-400:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    import claims_mamba_common as C
    shape = C.SHAPES[a.shape]
    mod = C.module(shape["kind"])
    fn = mod.chunk_scan_fwd if shape["kind"] in ("scan_ex", "scan_bm") else mod.chunk_state_fwd
    n = len(fn.configs() if callable(fn.configs) else fn.configs)
    chunks = [(a.shape, list(range(w, n, a.workers))) for w in range(a.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = [r for part in pool.map(_work, chunks) for r in part]
    res.sort()
    errs = [r for r in res if r[1] != "ok"]
    print(f"[compile {a.shape}] {n} configs, {len(errs)} errors, wall {time.time() - t0:.0f}s, "
          f"per-config median {sorted(r[2] for r in res)[len(res) // 2]:.1f}s")
    for r in errs:
        print("  error", r[0], r[3].strip().splitlines()[-1] if r[3].strip() else "")


if __name__ == "__main__":
    main()
