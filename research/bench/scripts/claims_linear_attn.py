"""Optional, informational: examples/linear_attention/example_linear_attn_fwd.py (TileLang fused-chunk
linear attention) vs flash-linear-attention's Triton `fused_chunk_linear_attn`. The example README makes
no numeric claim; the example itself prints a speed-up. Part of sub-study C.

    source research/env.sh
    pip install --no-deps --target $FLA fla-core==0.5.2          # scratch dir, not ~/mpk-env
    python research/bench/scripts/claims_linear_attn.py --fla-path $FLA --compile-only
    python research/bench/scripts/run_guarded.py -- flock $GPU_LOCK \
        python research/bench/scripts/claims_linear_attn.py --fla-path $FLA

The example's main(B, S, H, D) runs unchanged (its TileLang-vs-torch assert) with `do_bench` replaced by a
recorder that captures the two timed callables (FLA first, TileLang second). The FLA output is checked
against the example's torch ref_program as well. Primary timer cobench.bench_variants; secondary the
example's do_bench(backend="cupti"), falling back to the event timer if CUPTI is unavailable.
The TileLang callable prints the kernel source on every call (upstream code); stdout is discarded while
timing. Writes research/results/2026-09-24_claims_repro/C_mamba/linear_attn/<B>_<S>_<H>_<D>.json.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402

SHAPES = [(1, 512, 16, 128), (8, 1024, 32, 128), (8, 4096, 32, 128)]  # main() default, CLI default, longer


def closure_vars(fn) -> dict:
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in (fn.__closure__ or ()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fla-path", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, a.fla_path)
    outdir = os.path.join(C.RESULTS, "linear_attn")
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    mod = C._load(os.path.join(C.ROOT, "examples/linear_attention/example_linear_attn_fwd.py"), "tl_ex_linear_attn")
    if a.compile_only:
        for B, S, H, D in SHAPES:
            mod.tl_fused_chunk_fwd_kernel(B, S, H, D, D)
            print("compiled", B, S, H, D, flush=True)
        return
    import torch
    import cobench as cb
    import fla
    real = mod.do_bench
    for B, S, H, D in SHAPES:
        cap = []
        mod.do_bench = lambda fn, *args, **kw: (cap.append(fn), 1.0)[1]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main(B, S, H, D)  # asserts TileLang o/h vs ref_program (atol = rtol = 1e-2)
        finally:
            mod.do_bench = real
        f_fla, f_tl = cap
        cv = closure_vars(f_fla)
        q, k, v = cv["q"], cv["k"], cv["v"]
        o_ref, h_ref = mod.ref_program(q, k, v)
        o_fla, h_fla = f_fla()
        with contextlib.redirect_stdout(io.StringIO()):
            o_tl, h_tl = f_tl()
        torch.cuda.synchronize()
        chk = {"tilelang_o": C.pair_error(o_tl, o_ref), "tilelang_h": C.pair_error(h_tl, h_ref),
               "fla_o": C.pair_error(o_fla, o_ref), "fla_h": C.pair_error(h_fla, h_ref)}
        ok = all(d["rel_l2"] <= 1e-2 for d in chk.values())
        with contextlib.redirect_stdout(io.StringIO()):
            vres = cb.bench_variants({"tilelang": f_tl, "fla": f_fla}, reference="fla", reps=a.reps, clock=True,
                                     label=f"linear_attn {B} {S} {H} {D}")
            sec = {"tilelang": [], "fla": []}
            backend = "cupti"
            for _ in range(3):
                for n, f in (("fla", f_fla), ("tilelang", f_tl)):
                    try:
                        sec[n].append(real(f, backend=backend))
                    except Exception as e:  # CUPTI may be unavailable (profiling is admin-only here)
                        backend = "event"
                        sec[n].append(real(f, backend=backend))
                        sec.setdefault("cupti_error", repr(e)[-200:])
        print(vres, flush=True)
        t_tl, t_fla = vres.variants["tilelang"]["total"]["median"], vres.variants["fla"]["total"]["median"]
        res = dict(shape=dict(B=B, S=S, H=H, D=D), check=chk, correctness_ok=ok, tilelang_us=t_tl, fla_us=t_fla,
                   ratio_fla_over_tilelang=t_fla / t_tl, fla_version=getattr(fla, "__version__", "?"),
                   secondary=dict(method=f"example's do_bench(backend={backend!r}), median of 3", raw=sec,
                                  ratio_fla_over_tilelang=statistics.median(sec["fla"]) / statistics.median(sec["tilelang"])),
                   primary=vres.to_dict(), env=C.env_info(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        C.dump(res, os.path.join(outdir, f"{B}_{S}_{H}_{D}.json"))
        print(f"[linear_attn {B} {S} {H} {D}] TL {t_tl:.1f} us FLA {t_fla:.1f} us ratio {t_fla / t_tl:.3f} ok={ok}", flush=True)


if __name__ == "__main__":
    main()
