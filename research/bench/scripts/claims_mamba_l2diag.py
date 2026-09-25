"""Diagnostic (not part of the verdict): is the Triton baseline's slowdown at large shapes caused by its
CTA order? mamba-ssm launches grid = (tiles, batch*nchunks, nheads), i.e. heads vary slowest, so a tensor
shared by all heads of a group (cb and C for chunk-scan, B for chunk-state; ngroups = 1 here) is re-read
from DRAM once per head when the reuse distance exceeds the L2. TileLang's kernels use
grid = (nheads, tiles, batch*nchunks) (heads fastest), so consecutive CTAs share those tiles in L2.

    source research/env.sh
    python research/bench/scripts/run_guarded.py -- flock $GPU_LOCK \
        python research/bench/scripts/claims_mamba_l2diag.py --shape CT5

Builds two plain @triton.jit copies of the vendored kernel (source of mamba-ssm 2.2.6.post3, decorator
@triton.autotune removed): "orig" with the original program_id mapping and "heads_first" with the
axes rotated (heads -> axis 0, tiles -> axis 1, batch*nchunks -> axis 2; the kernel body is otherwise
untouched). Both are launched through mamba-ssm's own host function (the module's kernel symbol is
swapped for a launcher that rotates the grid accordingly), with the config Triton's autotuner picked
in this process. Outputs are compared with the TileLang output; the four variants (TileLang, Triton as
shipped, orig copy, heads_first copy) are timed together with cobench.bench_variants (clean flush).
Writes C_mamba/l2diag/<shape>.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402

_HDR = "import triton\nimport triton.language as tl\n\n\n"


def make_copy(autotuner, heads_first: bool, tmpdir: str, tag: str):
    src = inspect.getsource(autotuner.fn.fn)
    src = src[src.index("def "):]
    if heads_first:
        src = src.replace("tl.program_id(axis=0)", "@@A1@@").replace("tl.program_id(axis=1)", "@@A2@@")
        src = src.replace("tl.program_id(axis=2)", "tl.program_id(axis=0)")
        src = src.replace("@@A1@@", "tl.program_id(axis=1)").replace("@@A2@@", "tl.program_id(axis=2)")
    path = os.path.join(tmpdir, f"l2diag_{tag}.py")
    with open(path, "w") as f:
        f.write(_HDR + "@triton.jit\n" + src)
    spec = importlib.util.spec_from_file_location(f"l2diag_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, autotuner.fn.fn.__name__), src


class Launcher:
    """Stands in for the autotuned kernel symbol inside mamba-ssm's host function."""

    def __init__(self, jitfn, cfg, heads_first: bool):
        self.jitfn, self.cfg, self.heads_first = jitfn, cfg, heads_first

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            g = grid(dict(self.cfg.kwargs)) if callable(grid) else grid
            g = (g[2], g[0], g[1]) if self.heads_first else g
            return self.jitfn[g](*args, **kwargs, **self.cfg.all_kwargs())
        return launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    shape = C.SHAPES[a.shape]
    tune = json.load(open(os.path.join(C.RESULTS, "tune", f"{a.shape}.json")))
    out_dir = os.path.join(C.RESULTS, "l2diag")
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    import torch
    import cobench as cb
    scan, state = C.triton_modules()
    is_scan = shape["kind"] != "state_ex"
    hostmod = scan if is_scan else state
    sym = "_chunk_scan_fwd_kernel" if is_scan else "_chunk_state_fwd_kernel"
    at = getattr(hostmod, sym)
    kern = C.tilelang_kernel(shape, tune["best_config"])
    inp = C.make_inputs(shape, seed=0)
    args = C.call_args(shape, inp)
    tri = C.triton_fn(shape)
    ref = kern(*args)
    o = tri(*args)  # Triton's own autotuning, as shipped
    torch.cuda.synchronize()
    pick = at.best_config
    tmpdir = tempfile.mkdtemp(prefix="l2diag_")
    copies = {}
    res = dict(shape_name=a.shape, shape=shape, triton_pick=str(pick), variants={}, time=time.strftime("%Y-%m-%d %H:%M:%S"),
               shipped_vs_tilelang=C.pair_error(o, ref))
    del o
    for tag, hf in (("orig", False), ("heads_first", True)):
        fn, src = make_copy(at, hf, tmpdir, tag)
        copies[tag] = Launcher(fn, pick, hf)
        res["variants"][tag] = dict(program_id_lines=[ln.strip() for ln in src.splitlines() if "program_id" in ln])

    def via(tag):
        def f():
            setattr(hostmod, sym, copies[tag])
            try:
                return tri(*args)
            finally:
                setattr(hostmod, sym, at)
        return f

    variants = {"tilelang": lambda: kern(*args), "triton_shipped": lambda: tri(*args)}
    for tag in copies:
        o = via(tag)()
        torch.cuda.synchronize()
        pe = C.pair_error(o, ref)
        del o
        res["variants"][tag].update(vs_tilelang=pe, ok=pe["rel_l2"] <= 1e-2)
        if pe["rel_l2"] <= 1e-2:
            variants[f"triton_{tag}"] = via(tag)
    v = cb.bench_variants(variants, reference="triton_shipped", reps=a.reps, clock=True, label=f"l2diag {a.shape}")
    print(v, flush=True)
    for n, d in v.variants.items():
        res.setdefault("timing_us", {})[n] = d["total"]["median"]
        res.setdefault("clock_mhz", {})[n] = (d.get("clock") or {}).get("median")
    res["primary"] = v.to_dict()
    C.dump(res, os.path.join(out_dir, f"{a.shape}.json"))
    t = res["timing_us"]
    print(f"[l2diag {a.shape}] " + "  ".join(f"{n} {x:.1f} us" for n, x in t.items()), flush=True)


if __name__ == "__main__":
    main()
