"""Fork effect check: time the picked TileLang configs with whichever `tilelang` is importable.

Run once with the fork (research/env.sh) and once with the upstream v0.1.14 PyPI wheel (installed
with `pip install --no-deps --target DIR tilelang==0.1.14`, plus z3-solver 4.15.x in another --target
dir because the wheel links libz3.so.4.15), with a separate TILELANG_CACHE_DIR so no kernel compiled
by one build can be served to the other:

    # fork
    source research/env.sh
    python research/bench/scripts/claims_mamba_forkcheck.py --tag fork --shapes CC0,CT0,B8192
    # upstream wheel (PYTHONPATH must NOT contain the repo root)
    env -u PYTHONPATH PYTHONPATH=$UP:$Z3 LD_LIBRARY_PATH=$Z3/z3/lib CUDA_HOME=/usr/local/cuda-12.9 \
        TILELANG_CACHE_DIR=$UPCACHE ~/mpk-env/bin/python research/bench/scripts/claims_mamba_forkcheck.py \
        --tag upstream --shapes CC0,CT0,B8192

The example / benchmark files are identical in the fork and in v0.1.14 (git diff v0.1.14 HEAD is
empty for examples/linear_attention and benchmark/mamba2). Per shape: the autotuner's pick from
tune/<shape>.json, output compared with Triton (which the main bench verified against the fp32
reference), then cobench.bench_variants({tilelang, triton}) as in claims_mamba_bench.py.
Writes C_mamba/forkcheck/<tag>.json and the generated CUDA source per shape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--compile-only", action="store_true", help="fill this build's kernel cache only (no launch)")
    a = ap.parse_args()
    outdir = os.path.join(C.RESULTS, "forkcheck")
    os.makedirs(os.path.join(outdir, "src"), exist_ok=True)
    os.chdir(outdir)
    import torch
    import tilelang
    import cobench as cb
    from claims_mamba_bench import src_features
    res = dict(tag=a.tag, tilelang_version=tilelang.__version__, tilelang_file=tilelang.__file__,
               cache_dir=os.environ.get("TILELANG_CACHE_DIR"), time=time.strftime("%Y-%m-%d %H:%M:%S"), shapes={})
    for name in a.shapes.split(","):
        shape = C.SHAPES[name]
        cfg = json.load(open(os.path.join(C.RESULTS, "tune", f"{name}.json")))["best_config"]
        t0 = time.time()
        k = C.tilelang_kernel(shape, cfg)
        compile_s = time.time() - t0
        src = k.get_kernel_source()
        with open(os.path.join(outdir, "src", f"{name}_{a.tag}.cu"), "w") as f:
            f.write(src)
        if a.compile_only:
            print(f"[{a.tag}] compiled {name} {cfg} in {compile_s:.1f}s sha256 {hashlib.sha256(src.encode()).hexdigest()[:16]}", flush=True)
            continue
        inp = C.make_inputs(shape, seed=0)
        args = C.call_args(shape, inp)
        tri = C.triton_fn(shape)
        o_tl, o_tr = k(*args), tri(*args)
        torch.cuda.synchronize()
        pe = C.pair_error(o_tl, o_tr)
        del o_tl, o_tr
        v = cb.bench_variants({"tilelang": lambda: k(*args), "triton": lambda: tri(*args)}, reference="triton",
                              reps=a.reps, clock=True, label=f"{a.tag} {name}")
        print(v, flush=True)
        res["shapes"][name] = dict(
            config=cfg, compile_s=compile_s, src_sha256=hashlib.sha256(src.encode()).hexdigest(), src=src_features(src),
            tilelang_vs_triton=pe, ok=pe["rel_l2"] <= 1e-2,
            tilelang_us=v.variants["tilelang"]["total"]["median"], triton_us=v.variants["triton"]["total"]["median"],
            ratio_triton_over_tilelang=v.derived["speedup"]["tilelang"],
            clock_mhz=(v.variants["tilelang"].get("clock") or {}).get("median"))
        del inp, args
        torch.cuda.empty_cache()
    if not a.compile_only:
        C.dump(res, os.path.join(outdir, f"{a.tag}.json"))


if __name__ == "__main__":
    main()
