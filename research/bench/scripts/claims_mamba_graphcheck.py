"""Resolution-free cross-check for the short kernels of sub-study C (claims C3 / minference / linear attn).

Flush-mode per-call event times on this machine are quantized (steps of ~2-4 us, found by sub-study B),
which matters for kernels below ~60 us. Here each sample is one CUDA-graph replay of k back-to-back
calls over rotated input copies (> 2x L2 in total, so every call reads DRAM-cold inputs), time / k:
cobench.bench(mode="graph") with make_inputs. TileLang and Triton are measured alternately, 3 rounds
each (TL, Triton, TL, Triton, ...), and the median of the three per-call medians is reported.

    source research/env.sh
    python research/bench/scripts/run_guarded.py -- flock $GPU_LOCK \
        python research/bench/scripts/claims_mamba_graphcheck.py --shapes CC0,CC1,CC2,CT0,CT1,CT2

Also records per-call samples of the flush-mode primary run's quantization (distinct values seen in a
fresh bench_variants run with keep_samples) for the first shape.
Writes C_mamba/graphcheck/<shape>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    out_dir = os.path.join(C.RESULTS, "graphcheck")
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    import torch
    import cobench as cb
    for name in a.shapes.split(","):
        shape = C.SHAPES[name]
        tune = json.load(open(os.path.join(C.RESULTS, "tune", f"{name}.json")))
        kern = C.tilelang_kernel(shape, tune["best_config"])
        tri = C.triton_fn(shape)
        seed = iter(range(1000, 100000))

        def make_inputs():
            return C.call_args(shape, C.make_inputs(shape, seed=next(seed)))

        a0 = make_inputs()
        o_tl, o_tr = kern(*a0), tri(*a0)  # Triton autotunes here (outside any capture)
        torch.cuda.synchronize()
        pe = C.pair_error(o_tl, o_tr)
        del o_tl, o_tr, a0
        res = dict(shape_name=name, shape=shape, tilelang_config=tune["best_config"], triton_pick=C.triton_best_config(shape),
                   tilelang_vs_triton=pe, method="cobench.bench(mode='graph'): CUDA graph of k back-to-back calls over "
                   "rotated input copies (> 2x L2), per-call = replay time / k; TL and Triton alternated",
                   rounds={"tilelang": [], "triton": []}, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        for r in range(a.rounds):
            for n, f in (("tilelang", kern), ("triton", tri)):
                b = cb.bench(f, make_inputs=make_inputs, mode="graph", reps=a.reps, clock=True, keep_samples=False,
                             label=f"graph {name} {n} r{r}")
                res["rounds"][n].append(dict(median=b.median, p10=b.p10, p90=b.p90, cv=b.cv, k=b.k, n_copies=b.n_copies,
                                             clock_mhz=((b.clock or {}).get("per_rep") or {}).get("median")))
                torch.cuda.empty_cache()
        tl = statistics.median(x["median"] for x in res["rounds"]["tilelang"])
        tr = statistics.median(x["median"] for x in res["rounds"]["triton"])
        res.update(tilelang_us=tl, triton_us=tr, ratio_triton_over_tilelang=tr / tl)
        C.dump(res, os.path.join(out_dir, f"{name}.json"))
        print(f"[graphcheck {name}] TL {tl:.2f} us  Triton {tr:.2f} us  ratio {tr / tl:.3f}  "
              f"(k={res['rounds']['tilelang'][0]['k']}, copies={res['rounds']['tilelang'][0]['n_copies']}) "
              f"TL-vs-Triton rel {pe['rel_l2']:.1e}", flush=True)


if __name__ == "__main__":
    main()
