"""Op-library tests. Runnable with plain python (pytest is not installed in ~/mpk-env):

    source research/env.sh
    python -m cotile.tests.test_ops                      # all ops, all configs
    python -m cotile.tests.test_ops --ops gemm --limit 4 # quick subset
    python -m cotile.tests.test_ops --cold               # fresh TileLang cache (compile timing)
    python -m cotile.tests.test_ops --out research/results/2026-09-22_op_library

The `test_*` functions are also pytest-compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time


def _run_op(name: str, limit: int | None = None, gpu: bool = True, workers: int = 32):
    from cotile.tests import harness as H

    op = H.OPS[name]
    shape = H.TEST_SHAPES[name]
    cfgs = op.configs(shape)
    if limit:
        cfgs = cfgs[:limit]
    run = H.build_and_compile(op, shape, cfgs, num_workers=workers)
    H.extract_signatures(run)
    if gpu:
        if not H.wait_for_gpu():
            raise RuntimeError("GPU busy with another compute process for 15 min; aborting GPU tests")
        H.gpu_tests(run)
    return run


def _assert_run(run):
    from cotile.tests import harness as H

    s = H.summarize(run)
    assert s["compile_fail"] == 0, s
    assert s["both_ok"] == s["configs"], s
    assert s["grid_eq_persistent"] == s["configs"], s
    assert s["repeat_ok"] == s["split_configs"] and s["counters_zero"] == s["split_configs"], s


def test_gemm():
    _assert_run(_run_op("gemm"))


def test_gqa_decode():
    _assert_run(_run_op("gqa_decode"))


def test_rmsnorm():
    _assert_run(_run_op("rmsnorm"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="gemm,gqa_decode,rmsnorm")
    ap.add_argument("--limit", type=int, default=None, help="only the first N configs per op")
    ap.add_argument("--no-gpu", action="store_true", help="compile + signatures only")
    ap.add_argument("--cold", action="store_true", help="use a fresh, empty TileLang kernel cache")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None, help="directory for signatures.csv / tests.csv / summary.json")
    a = ap.parse_args(argv)

    if a.cold:
        # must happen before tilelang is imported (env is read at import time)
        cache = tempfile.mkdtemp(prefix="cotile_tlcache_", dir=os.environ.get("COTILE_TMP"))
        os.environ["TILELANG_CACHE_DIR"] = cache
        print(f"[cold] TILELANG_CACHE_DIR={cache}", flush=True)
    from cotile.tests import harness as H

    summaries, sigs, results = [], [], []
    t_all = time.time()
    for name in a.ops.split(","):
        t0 = time.time()
        run = _run_op(name, a.limit, gpu=not a.no_gpu, workers=a.workers)
        s = H.summarize(run)
        s["total_s"] = round(time.time() - t0, 1)
        summaries.append(s)
        sigs += run.signatures
        results += run.results
        H.log(json.dumps(s))
        for r in run.results:
            bad = [k for k, v in r.items() if k.endswith(("_ok", "_eq_persistent", "_zero", "_group_ref")) and v is False]
            if bad:
                H.log(f"  FAIL {r['config']}: {bad} {r.get('grid_error') or ''} {r.get('persistent_error') or ''}")
        for spec in run.specs.values():
            if spec.compile_error:
                H.log(f"  COMPILE-FAIL {spec.name}: {spec.compile_error}")
    H.log(f"total wall {time.time() - t_all:.1f}s")
    if a.out:
        H.write_csv(os.path.join(a.out, "signatures.csv"), sigs)
        H.write_csv(os.path.join(a.out, "tests.csv"), results)
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(summaries, f, indent=1)
    ok = all(
        s["compile_fail"] == 0 and (a.no_gpu or (s["both_ok"] == s["configs"] and s["grid_eq_persistent"] == s["configs"]))
        for s in summaries
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
