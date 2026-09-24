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
            raise RuntimeError("GPU occupied past the guard deadline (cobench.GuardPolicy); aborting GPU tests")
        H.gpu_tests(run)
    return run


def _assert_run(run):
    from cotile.tests import harness as H

    s = H.summarize(run)
    assert s["compile_fail"] == 0, s
    assert s["both_ok"] == s["configs"], s
    assert s["grid_eq_persistent"] == s["configs"], s
    assert s["repeat_ok"] == s["split_configs"] and s["counters_zero"] == s["split_configs"], s


def _l2_hint_cases():
    """(op, base cfg, config field) for the L2 eviction-policy axes: decode K/V loads
    (cp.async; no split and split-KV) and GEMM A/B loads (cp.async, split-K, and TMA via
    auto warp specialization)."""
    from cotile.tests import harness as H

    G, D = H.OPS["gemm"], H.OPS["gqa_decode"]
    return [
        (D, D.DecodeConfig(128, 4, 1, 128, 1), "kv_l2"),
        (D, D.DecodeConfig(64, 4, 4, 128, 2), "kv_l2"),
        (D, D.DecodeConfig(32, 4, 1, 64, 2), "kv_l2"),
        (G, G.GemmConfig(128, 256, 64, 2, 256, epilogue="direct"), "ab_l2"),
        # these two (and the split-K one) hit the ptxas desc[UR1] miscompile with the
        # 32-bit shared address (see resources.invalid_memory_descriptors)
        (G, G.GemmConfig(128, 128, 64, 2, 128), "ab_l2"),
        (G, G.GemmConfig(64, 64, 64, 2, 128), "ab_l2"),
        (G, G.GemmConfig(128, 128, 64, 2, 128, split_k=2), "ab_l2"),
        (G, G.GemmConfig(128, 128, 64, 2, 256, ws="auto"), "ab_l2"),
    ]


def _hint_calls(src: str) -> dict:
    import re

    return {
        "cp_async_plain": len(re.findall(r"cp_async_gs(?:_conditional)?<", src)),
        "cp_async_hint": {p: len(re.findall(rf"cp_async_gs(?:_conditional)?_l2hint<\d+, tl::L2EvictionPolicy::{p}>", src))
                          for p in ("EVICT_FIRST", "EVICT_LAST")},
        "tma_hint": {p: src.count(f"tma_load<tl::CacheHintSm90::{p}>") for p in ("EVICT_FIRST", "EVICT_LAST")},
        "tma_plain": len(re.findall(r"tma_load\(", src)),
    }


def _strip_hints(src: str) -> str:
    """Source with the L2 hints and kernel names removed, and TileLang's AllReduce
    workspace names canonicalized (their numbering is not deterministic between
    compilations of the same kernel)."""
    import re

    src = re.sub(r"cp_async_gs(_conditional)?_l2hint<(\d+), tl::L2EvictionPolicy::\w+>", r"cp_async_gs\1<\2>", src)
    src = re.sub(r"tma_load<tl::CacheHintSm90::\w+>", "tma_load", src)
    src = re.sub(r"\b(gemm|gqa_decode)_(grid|persistent)_\w+?_[0-9a-f]{6}(_kernel)?\b", "KERNEL", src)
    # TileLang's shared-memory views (v, v_1, ...) and AllReduce workspaces: which view /
    # workspace a statement uses is not deterministic between two compilations of the
    # SAME kernel (verified: two processes, no hint), so the names are collapsed here and
    # the set of their declarations (offsets) is compared separately (_smem_decls)
    src = re.sub(r"\bworkspace(_\d+)?\b", "WORKSPACE", src)
    return re.sub(r"\bv(_\d+)?\b", "V", src)


def _smem_decls(src: str) -> list[str]:
    """Sorted shared-memory view / workspace declarations with the names collapsed."""
    return sorted(_strip_hints(line).strip() for line in src.splitlines() if "buf_dyn_shmem" in line or "__shared__" in line)


def run_l2_hints(gpu: bool = True, workers: int = 32) -> list[dict]:
    """L2 eviction-policy axes (gqa_decode.kv_l2, gemm.ab_l2):
    * the hinted kernel's source equals its unhinted twin's except for the hint (every
      K/V resp. A/B load carries it, nothing else does);
    * the SASS has no invalid memory descriptor (ptxas miscompile guard,
      resources.invalid_memory_descriptors);
    * GPU: outputs of the grid and persistent builds are bitwise identical to the unhinted
      twin's and within tolerance of the fp32 reference (a cache hint must not change data).
    """
    import dataclasses

    import torch

    from cotile import resources
    from cotile.kernel import Runner, compile_specs
    from cotile.tests import harness as H

    rows, specs = [], {}
    for op, base, field in _l2_hint_cases():
        shape = H.TEST_SHAPES[op.NAME]
        for pol in ("normal", "evict_first", "evict_last"):
            cfg = dataclasses.replace(base, **{field: pol})
            specs[(op.NAME, base, pol, "grid")] = op.build_grid(shape, cfg)
            specs[(op.NAME, base, pol, "persistent")] = op.build_persistent(shape, cfg, 188)
    st = compile_specs(list(specs.values()), num_workers=workers)
    H.log(f"[l2] compiled {st}")
    if gpu and not H.wait_for_gpu():
        raise RuntimeError("GPU occupied past the guard deadline (cobench.GuardPolicy); aborting GPU tests")
    for op, base, field in _l2_hint_cases():
        shape = H.TEST_SHAPES[op.NAME]
        inputs = op.make_inputs(shape, seed=0) if gpu else None
        ref = op.reference(shape, inputs) if gpu else None
        out_name = next(p.name for p in op.io_params(shape, base) if p.role == "out")
        for build in ("grid", "persistent"):
            twin = specs[(op.NAME, base, "normal", build)]
            twin_out = None
            if gpu and twin.kernel is not None:
                twin_out = H._launch(Runner(twin), inputs)[out_name].clone()
            for pol in ("evict_first", "evict_last"):
                s = specs[(op.NAME, base, pol, build)]
                row = {"op": op.NAME, "config": op.cfg_tag(dataclasses.replace(base, **{field: pol})), "build": build,
                       "compiled": s.kernel is not None, "error": s.compile_error}
                if s.kernel is None or twin.kernel is None:
                    row["ok"] = False
                    rows.append(row)
                    continue
                src, tsrc = s.kernel.get_kernel_source(), twin.kernel.get_kernel_source()
                h, th = _hint_calls(src), _hint_calls(tsrc)
                want = "EVICT_FIRST" if pol == "evict_first" else "EVICT_LAST"
                other = "EVICT_LAST" if want == "EVICT_FIRST" else "EVICT_FIRST"
                n_twin = th["cp_async_plain"] + th["tma_plain"]
                n_hint = h["cp_async_hint"][want] + h["tma_hint"][want]
                # the unhinted twin's loads = Q/other plain loads + the hinted ones
                row["hinted_loads"] = n_hint
                row["plain_loads_left"] = h["cp_async_plain"] + h["tma_plain"]
                row["hint_ok"] = n_hint > 0 and h["cp_async_hint"][other] == 0 and h["tma_hint"][other] == 0 \
                    and n_hint + row["plain_loads_left"] == n_twin
                row["source_eq_modulo_hint"] = (_strip_hints(src) == _strip_hints(tsrc)
                                                and _smem_decls(src) == _smem_decls(tsrc))
                bad = resources.invalid_memory_descriptors(resources.sass(resources.cubin_bytes(s.kernel)))
                row["sass_desc_ok"] = not bad
                if bad:
                    row["error"] = f"invalid memory descriptors: {bad[:2]}"
                if gpu and bad:
                    row["ok"] = False  # would fault with an illegal instruction: do not launch
                    rows.append(row)
                    H.log(f"  [l2] {row['config']:44s} {build:10s} ok=False {row['error']}")
                    continue
                if gpu:
                    o = H._launch(Runner(s), inputs)[out_name]
                    row["ref_ok"] = H.compare(o, ref[out_name], op.TOLERANCE)["ok"]
                    row["bitwise_eq_twin"] = bool(torch.equal(o, twin_out))
                row["ok"] = all(row.get(k, True) for k in ("hint_ok", "source_eq_modulo_hint", "sass_desc_ok", "ref_ok",
                                                           "bitwise_eq_twin"))
                rows.append(row)
                H.log(f"  [l2] {row['config']:44s} {build:10s} ok={row['ok']} hinted={n_hint} "
                      f"plain_left={row['plain_loads_left']} src_eq={row['source_eq_modulo_hint']} "
                      f"bitwise={row.get('bitwise_eq_twin')} ref={row.get('ref_ok')}")
        if gpu:
            del inputs, ref
            torch.cuda.empty_cache()
    return rows


def test_l2_hints():
    rows = run_l2_hints()
    assert rows and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]


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
    ap.add_argument("--no-l2", action="store_true", help="skip the L2 eviction-policy axis tests")
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
    l2_rows = []
    if not a.no_l2:
        l2_rows = run_l2_hints(gpu=not a.no_gpu, workers=a.workers)
        H.log(f"[l2] {sum(r['ok'] for r in l2_rows)}/{len(l2_rows)} hinted kernels ok")
    H.log(f"total wall {time.time() - t_all:.1f}s")
    if a.out:
        H.write_csv(os.path.join(a.out, "signatures.csv"), sigs)
        H.write_csv(os.path.join(a.out, "tests.csv"), results)
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(summaries, f, indent=1)
    ok = all(
        s["compile_fail"] == 0 and (a.no_gpu or (s["both_ok"] == s["configs"] and s["grid_eq_persistent"] == s["configs"]))
        for s in summaries
    ) and all(r["ok"] for r in l2_rows)
    if a.out and l2_rows:
        H.write_csv(os.path.join(a.out, "l2_hint_tests.csv"), l2_rows)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
