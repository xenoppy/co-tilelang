"""Sensitivity check of the Triton baseline's own autotuning (not part of the verdict): time every config
in mamba-ssm's @triton.autotune list for one shape, next to Triton's autotuned pick and the TileLang pick.

    source research/env.sh
    python research/bench/scripts/run_guarded.py -- flock $GPU_LOCK \
        python research/bench/scripts/claims_mamba_triton_sweep.py --shape CT5

The baseline is called exactly as in the main bench (the example's chunk_*_triton wrapper). A config is
forced by temporarily giving the kernel's Autotuner a one-element config list (Triton 3.4 then skips
benchmarking and launches that config); configs that fail to compile/launch on sm_120 (OutOfResources)
are recorded and skipped. Each config's output is compared with the TileLang output (which the main
bench verified against the fp32 reference). All variants are timed in one cobench.bench_variants call
(clean flush, clock). Writes C_mamba/triton_sweep/<shape>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--reps", type=int, default=60)
    a = ap.parse_args()
    shape = C.SHAPES[a.shape]
    tune = json.load(open(os.path.join(C.RESULTS, "tune", f"{a.shape}.json")))
    os.makedirs(os.path.join(C.RESULTS, "triton_sweep"), exist_ok=True)
    os.chdir(os.path.join(C.RESULTS, "triton_sweep"))
    import torch
    import cobench as cb
    scan, state = C.triton_modules()
    at = scan._chunk_scan_fwd_kernel if shape["kind"] != "state_ex" else state._chunk_state_fwd_kernel
    all_cfgs = list(at.configs)
    kern = C.tilelang_kernel(shape, tune["best_config"])
    inp = C.make_inputs(shape, seed=0)
    args = C.call_args(shape, inp)
    tri = C.triton_fn(shape)
    ref = kern(*args)
    o = tri(*args)  # Triton's own autotuning (as in the main bench)
    torch.cuda.synchronize()
    pick = at.best_config
    res = dict(shape_name=a.shape, shape=shape, triton_autotuned_pick=str(pick), configs={},
               tilelang_config=tune["best_config"], time=time.strftime("%Y-%m-%d %H:%M:%S"))
    res["pick_vs_tilelang"] = C.pair_error(o, ref)
    del o

    def forced(cfg):
        def f():
            at.configs = [cfg]
            try:
                return tri(*args)
            finally:
                at.configs = all_cfgs
        return f

    variants = {"tilelang": lambda: kern(*args), "triton_autotuned": lambda: tri(*args)}
    for i, cfg in enumerate(all_cfgs):
        name = f"cfg{i}"
        d = dict(config=str(cfg), is_pick=(str(cfg) == str(pick)))
        try:
            o = forced(cfg)()
            torch.cuda.synchronize()
            d["vs_tilelang"] = C.pair_error(o, ref)
            d["ok"] = d["vs_tilelang"]["rel_l2"] <= 1e-2
            del o
            if d["ok"]:
                variants[name] = forced(cfg)
        except Exception as e:  # e.g. triton OutOfResources (shared memory) on sm_120; recorded
            d["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        res["configs"][name] = d
    # the forced calls reset at.configs; make sure the autotuned variant still uses its cached pick
    v = cb.bench_variants(variants, reference="triton_autotuned", reps=a.reps, clock=True, label=f"triton sweep {a.shape}")
    print(v, flush=True)
    for n, d in res["configs"].items():
        if n in v.variants:
            d["median_us"] = v.variants[n]["total"]["median"]
            d["clock_mhz"] = (v.variants[n].get("clock") or {}).get("median")
    res["tilelang_us"] = v.variants["tilelang"]["total"]["median"]
    res["triton_autotuned_us"] = v.variants["triton_autotuned"]["total"]["median"]
    timed = {n: d["median_us"] for n, d in res["configs"].items() if "median_us" in d}
    best = min(timed, key=timed.get)
    res["triton_best_config"] = res["configs"][best]["config"]
    res["triton_best_us"] = timed[best]
    res["ratio_best_triton_over_tilelang"] = timed[best] / res["tilelang_us"]
    res["ratio_autotuned_triton_over_tilelang"] = res["triton_autotuned_us"] / res["tilelang_us"]
    res["primary"] = v.to_dict()
    C.dump(res, os.path.join(C.RESULTS, "triton_sweep", f"{a.shape}.json"))
    print(f"[triton sweep {a.shape}] TileLang {res['tilelang_us']:.1f} us; Triton autotuned {res['triton_autotuned_us']:.1f} us "
          f"({pick}); best forced {timed[best]:.1f} us ({res['triton_best_config']}); ratio best/TL "
          f"{res['ratio_best_triton_over_tilelang']:.3f}", flush=True)


if __name__ == "__main__":
    main()
