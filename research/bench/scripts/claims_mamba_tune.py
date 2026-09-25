"""Run the upstream TileLang autotuner, as shipped, for one Mamba-2 shape (claims C3 / C6-mamba).

    source research/env.sh
    python research/bench/scripts/run_guarded.py -- flock $GPU_LOCK \
        python research/bench/scripts/claims_mamba_tune.py --shape CC0

Writes research/results/2026-09-24_claims_repro/C_mamba/tune/<shape>.json with the chosen config,
the autotuner's own latency for every config that compiled and ran ("Tuned Latency ..." lines), the
configs that failed (compile error, launch error such as > 99 KB dynamic smem) and the autotuner.log
excerpt for them. The autotuner decorators in the example / benchmark files are used unchanged
(configs, warmup=10, rep=10, no reference check during tuning).
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402


_TS = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ ", re.M)


def parse_autotuner_log(path: str) -> dict:
    """config-string -> failure reason, from tilelang's autotuner.log (records start with a timestamp).
    'Compilation failed for config {...} ... with error: ...'  -> 'compile: <error>'
    'An error occurred while testing config {...}' + next 'Error: Traceback ...' record -> last line."""
    if not os.path.exists(path):
        return {}
    txt = open(path, errors="replace").read()
    starts = [m.start() for m in _TS.finditer(txt)] + [len(txt)]
    recs = [txt[a:b] for a, b in zip(starts[:-1], starts[1:])]
    out = {}
    for i, r in enumerate(recs):
        m = re.search(r"Compilation failed for config (\{.*?\}) at index \d+ with error: (.*)", r, re.S)
        if m:
            out[m.group(1)] = "compile: " + " ".join(m.group(2).split())[:200]
            continue
        m = re.search(r"An error occurred while testing config (\{.*?\})", r)
        if m:
            nxt = recs[i + 1] if i + 1 < len(recs) else ""
            lines = [ln.strip() for ln in nxt.strip().splitlines() if ln.strip()]
            last = lines[-1] if lines else "?"
            sm = re.search(r"Failed to set the allowed dynamic shared memory size to (\d+)", nxt)
            out[m.group(1)] = (f"launch: dynamic smem {sm.group(1)} B rejected (sm_120 opt-in limit 101376 B per CTA incl. static smem)" if sm else "run: " + last[:200])
    return out


def summarize_failures(failed: list) -> dict:
    s = {}
    for f in failed:
        k = f["reason"].split(":")[0] + (": smem" if "smem" in f["reason"] else "")
        s[k] = s.get(k, 0) + 1
    return s


def filter_configs(shape: dict, configs: list, ref_name: str) -> dict:
    """Adaptation for sm_120 (listed in the README): the upstream autotuner skips configs whose launch fails
    (dynamic smem above the 101376 B per-CTA limit), but each failed config leaks its freshly allocated
    input tensors, so at large shapes later configs - including launchable ones - fail with CUDA OOM
    (observed for CC4: 20 OOM failures, 56 GB held by the tuning process). Configs that failed to launch
    in the reference shape's tune are removed before tuning when their dynamic smem (read from the
    generated host code) is identical in both shapes; configs that launched there, configs whose smem
    differs, and configs that did not compile there are kept, so the tuned set equals the set the
    shipped autotuner would have benchmarked successfully without the leak."""
    import json as _json
    from claims_mamba_diag import launch_info
    ref = _json.load(open(os.path.join(C.RESULTS, "tune", f"{ref_name}.json")))
    ref_shape = C.SHAPES[ref_name]
    ok_ref = {str(c["config"]) for c in ref["per_config"]}
    fail_ref = {str(f["config"]): f["reason"] for f in ref["failed"]}
    kept, removed, mismatch = [], [], []
    for cfg in configs:
        key = str(cfg)
        if key in ok_ref or not fail_ref.get(key, "").startswith("launch"):
            kept.append(cfg)
            continue
        try:
            d_t = launch_info(C.tilelang_kernel(shape, cfg)).get("dyn_smem")
            d_r = launch_info(C.tilelang_kernel(ref_shape, cfg)).get("dyn_smem")
        except Exception as e:  # keep it; the autotuner will record the failure itself
            kept.append(cfg)
            mismatch.append(dict(config=cfg, error=repr(e)[-200:]))
            continue
        if d_t is not None and d_t == d_r:
            removed.append(dict(config=cfg, dyn_smem=d_t, ref_reason=fail_ref[key]))
        else:
            kept.append(cfg)
            mismatch.append(dict(config=cfg, dyn_smem=d_t, ref_dyn_smem=d_r))
    return dict(ref=ref_name, kept_configs=kept, removed=removed, mismatch=mismatch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=sorted(C.SHAPES))
    ap.add_argument("--reparse", action="store_true", help="only re-parse the saved autotuner.log (no GPU)")
    ap.add_argument("--filter-ref", default=None,
                    help="drop configs that failed to launch (smem) in this reference shape's tune, if their "
                         "dynamic smem is identical here (see module docstring)")
    a = ap.parse_args()
    if a.reparse:
        path = os.path.join(C.RESULTS, "tune", f"{a.shape}.json")
        import json
        res = json.load(open(path))
        reasons = parse_autotuner_log(os.path.join(C.RESULTS, "tune", "logs", a.shape, "autotuner.log"))
        for f in res["failed"]:
            f["reason"] = reasons.get(str(f["config"]), "no record in autotuner.log")
        res["failure_summary"] = summarize_failures(res["failed"])
        C.dump(res, path)
        print(a.shape, res["failure_summary"])
        return
    shape = C.SHAPES[a.shape]
    logdir = os.path.join(C.RESULTS, "tune", "logs", a.shape)
    os.makedirs(logdir, exist_ok=True)
    os.chdir(logdir)  # tilelang's autotuner writes ./autotuner.log (opened when tilelang.autotuner is imported)

    import torch
    mod = C.module(shape["kind"])
    fn = mod.chunk_scan_fwd if shape["kind"] in ("scan_ex", "scan_bm") else mod.chunk_state_fwd
    configs = fn.configs() if callable(fn.configs) else fn.configs
    cfg_filter = None
    if a.filter_ref:
        cfg_filter = filter_configs(shape, configs, a.filter_ref)
        fn.configs = cfg_filter["kept_configs"]
        print(f"[tune {a.shape}] config filter (ref {a.filter_ref}): kept {len(fn.configs)}/{len(configs)}; "
              f"removed {len(cfg_filter['removed'])}; footprint mismatches kept {len(cfg_filter['mismatch'])}", flush=True)
    t0 = time.time()
    with C.capture_stdout() as buf:
        kernel = C.tilelang_kernel(shape, None)
    wall = time.time() - t0
    lat = {}
    for m in re.finditer(r"Tuned Latency (\S+) with config (\{.*?\}) at index (\d+)", buf.getvalue()):
        lat[int(m.group(3))] = dict(latency_ms=float(m.group(1)), config=ast.literal_eval(m.group(2)))
    tuned = fn.configs
    failed = [dict(index=i, config=c) for i, c in enumerate(tuned) if i not in lat]
    reasons = parse_autotuner_log(os.path.join(logdir, "autotuner.log"))
    for f in failed:
        f["reason"] = reasons.get(str(f["config"]), "no record in autotuner.log")
    best_cfg = dict(kernel.config) if getattr(kernel, "config", None) is not None else None
    res = dict(
        shape_name=a.shape, shape=shape, env=C.env_info(), n_configs=len(configs), n_tuned=len(tuned), n_ok=len(lat),
        config_filter=({k: v for k, v in cfg_filter.items() if k != "kept_configs"} if cfg_filter else None),
        best_config=best_cfg, best_latency_ms=getattr(kernel, "latency", None),
        ref_latency_ms=getattr(kernel, "ref_latency", None), tune_wall_s=wall,
        per_config=[dict(index=i, **v) for i, v in sorted(lat.items(), key=lambda kv: kv[1]["latency_ms"])],
        failed=failed, failure_summary=summarize_failures(failed), time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    C.dump(res, os.path.join(C.RESULTS, "tune", f"{a.shape}.json"))
    print(f"[tune {a.shape}] best {best_cfg} {res['best_latency_ms']} ms; ok {len(lat)}/{len(configs)}; "
          f"wall {wall:.0f}s; failures: {summarize_failures(failed)}")
    del kernel
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
