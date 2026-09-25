"""Resolution-free cross-check of the C2 (and spot-check C4 / sink) timings with CUDA graphs.

Why: on this GPU a kernel launched eagerly on a stream is seen complete on a ~2.048 us grid
(claims_attn_timerprobe.py: a %globaltimer-spin kernel swept in 128 ns steps gives CUDA-event
durations in 2.048 us steps, also per kernel for 8 back-to-back eager launches; inside a CUDA graph
the per-call time is continuous = kernel + ~0.9 us; two back-to-back events read 0.38 us). The C2
flush-mode medians of 20-80 us kernels therefore sit on 2.048 us multiples (+-5-10 % per kernel).

Two graph-based timers, all implementations of a shape interleaved as before:
  graph_flush  cobench.bench_variants (clean L2 flush + host gate, 100 reps) of one CUDA-graph
               replay containing N calls, each on its own input copy (all DRAM-cold after the
               flush); per-call time = rep time / N. N = ceil(300 us / t), so the residual
               boundary quantisation is <= 2.048 us / (N t) < 1 %. Same protocol as the primary
               C2 timer otherwise.
  steady       cobench.bench_steady (thermal warmup, 5 rounds x 1.5 s slices, rotating order) of
               graph replays of M calls cycling through >= 2x L2 of input copies (back-to-back,
               DRAM-cold inputs, sustained power); per-call = t_iter / M.

  flock <gpu.lock> python research/bench/scripts/claims_attn_graph.py --kind fa --shape FA0
  flock <gpu.lock> python research/bench/scripts/claims_attn_graph.py --kind mla --batch 64 --seqlen 1024
  -> research/results/2026-09-24_claims_repro/B_attention/raw/graph_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_attn_common as C  # noqa: E402

TARGET_US = 300.0


def capture(fns, reps=1):
    """CUDA graph of fns[0](), ..., fns[-1]() repeated `reps` times; outputs are kept referenced
    (distinct buffers). Returns (graph, outputs)."""
    import torch
    g = torch.cuda.CUDAGraph()
    outs = []
    with torch.cuda.graph(g):
        for _ in range(reps):
            for f in fns:
                outs.append(f())
    torch.cuda.synchronize()
    return g, outs


def fa_setup(args):
    import torch
    import claims_attn_fa as FA
    sh = FA.SHAPES[args.shape]
    prim = json.load(open(os.path.join(C.RESULTS, "raw", f"fa_{args.shape}.json")))
    t_est = prim["cobench"]["summary"][FA.PRIMARY]["median_us"]
    exb, exs = FA.import_examples()
    tl_kernels = {n: (lay, FA.tl_build(lay, cfg, sh, exb, exs)) for n, (lay, cfg, _) in FA.TL_FIXED.items()}
    best = [int(x) for x in prim["sweep_best"]["cfg"].split("_")]
    tl_kernels["tl_bhsd_sweep"] = ("bhsd", FA.tl_build("bhsd", tuple(best), sh, exb, exs))
    try:
        import flash_attn  # noqa: F401
        have_fa2 = True
    except Exception:  # noqa: BLE001
        have_fa2 = False
    base = FA.make_inputs(sh)
    ref = FA.reference(sh, base)
    per_layout = sum(x.numel() * x.element_size() for x in base["bhsd"])

    def clone_inputs():
        return {k: tuple(x.clone() for x in v) for k, v in base.items()}

    def fns_for(copy):
        return {n: (lambda f=f, lay=lay: FA.to_bhsd(f(), lay)) for n, (f, lay) in
                FA.build_variants(sh, copy, tl_kernels, have_fa2).items()}
    return {"tag": args.shape, "shape": sh, "t_est": t_est, "per_copy_bytes": per_layout, "clone": clone_inputs,
            "fns_for": fns_for, "ref": ref, "reference": FA.PRIMARY, "check": lambda o: C.err_stats(o, ref),
            "exclude": set(), "prim": prim}


def mla_setup(args):
    import claims_attn_mla as MLA
    sh = MLA.shape_of(args.batch, args.seqlen)
    prim = json.load(open(os.path.join(C.RESULTS, "raw", f"mla_b{args.batch}_s{args.seqlen}.json")))
    t_est = prim["cobench"]["summary"]["tl_adapted"]["median_us"]
    base = MLA.make_inputs(sh)
    ref = MLA.reference(sh, base)
    adapted = prim["tl_adapt"]["picked"]
    per_copy = sum(base[k].numel() * base[k].element_size() for k in ("q_nope", "q_pe", "k_nope", "k_pe"))

    def clone_inputs():
        return {k: (v.clone() if k in ("q", "q_nope", "q_pe", "k_nope", "k_pe", "blocked_k") else v)
                for k, v in base.items()}

    def fns_for(copy):
        V = MLA.build_variants(sh, copy, adapted, {"variants": {}})
        return {n: (lambda f=f: f().view(sh["batch"], MLA.H_Q, MLA.DV)) for n, f in V.items()}
    return {"tag": f"mla_b{args.batch}_s{args.seqlen}", "shape": {k: v for k, v in sh.items() if k != "lens"},
            "t_est": t_est, "per_copy_bytes": per_copy, "clone": clone_inputs, "fns_for": fns_for, "ref": ref,
            "reference": "tl_adapted", "check": lambda o: C.err_stats(o, ref), "exclude": set(), "prim": prim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("fa", "mla"), required=True)
    ap.add_argument("--shape")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--seqlen", type=int)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--skip", default="", help="comma-separated variants to leave out")
    args = ap.parse_args()
    import torch
    res = {"wait": C.wait_gpu(min_free_gib=8), "env": C.env_info(), "timings_s": {}}
    import cobench as cb
    S = fa_setup(args) if args.kind == "fa" else mla_setup(args)
    skip = {x for x in args.skip.split(",") if x}
    L2 = cb.l2_bytes()
    n_flush = max(1, min(16, math.ceil(TARGET_US / S["t_est"])))
    n_rot = max(1, math.ceil(2 * L2 / S["per_copy_bytes"]))
    n_copies = max(n_flush, n_rot)
    res.update({"tag": S["tag"], "shape": S["shape"], "n_flush_calls": n_flush, "n_rotation_copies": n_rot,
                "n_copies": n_copies, "per_copy_bytes": S["per_copy_bytes"]})
    copies = [S["clone"]() for _ in range(n_copies)]
    fns = [S["fns_for"](c) for c in copies]                  # fns[copy][variant]
    names = [n for n in fns[0] if n not in skip]
    res["variants"] = {}
    g_flush, g_steady, steady_calls = {}, {}, {}
    for n in names:
        rec = res["variants"].setdefault(n, {})
        try:
            for c in range(n_copies):                        # eager warm-up (autotuners, JIT, lazy loads)
                fns[c][n]()
            torch.cuda.synchronize()
            gf, of = capture([fns[c][n] for c in range(n_flush)])
            m = max(1, math.ceil(TARGET_US / (S["t_est"] * n_rot)))
            gs, os_ = capture([fns[c][n] for c in range(n_rot)], reps=m)
            gf.replay()
            gs.replay()
            torch.cuda.synchronize()
            chk = [S["check"](o) for o in (of[0], of[-1], os_[-1])]
            rec["check"] = chk[-1]
            rec["check_all_ok"] = all(x["ok"] for x in chk)
            if not rec["check_all_ok"]:
                rec["excluded"] = "graph output failed the correctness check"
                continue
            g_flush[n], g_steady[n] = gf, gs
            steady_calls[n] = n_rot * m
            rec["graph_flush_calls"], rec["graph_steady_calls"] = n_flush, n_rot * m
        except Exception as e:  # noqa: BLE001 - recorded, not silenced
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            rec["traceback"] = traceback.format_exc()[-1200:]
            rec["cleared_last_error"] = C.clear_cuda_error()
        print(n, rec.get("check_all_ok"), rec.get("error", "")[:160], flush=True)
    ref = S["reference"] if S["reference"] in g_flush else None
    with C.timed("graph_flush", res["timings_s"]):
        vr = cb.bench_variants({n: g.replay for n, g in g_flush.items()}, reference=ref, reps=args.reps,
                               clock=True, nvml=True, keep_samples=True, label=f"graph-flush {S['tag']}")
    gfs = C.variants_summary(vr)
    for n, d in gfs.items():
        d["per_call_median_us"] = d["median_us"] / n_flush
        d["per_call_mean_us"] = d["mean_us"] / n_flush
    res["graph_flush"] = {"summary": gfs, "nvml": vr.nvml, "guard": vr.guard, "samples_us": vr.samples,
                          "calls_per_rep": n_flush}
    with C.timed("steady", res["timings_s"]):
        sr = cb.bench_steady({n: g.replay for n, g in g_steady.items()}, reference=ref, rounds=args.rounds,
                             label=f"graph-steady {S['tag']}")
    st = {}
    for n, v in sr.variants.items():
        st[n] = {"per_call_us": v["t_iter_us"] / steady_calls[n], "t_iter_us": v["t_iter_us"],
                 "cv_slices": v["cv_slices"], "clock_mhz": v.get("clock_mhz"), "power_w": v.get("power_w"),
                 "calls_per_iter": steady_calls[n]}
    res["steady"] = {"summary": st, "derived": sr.derived, "config": sr.config, "guard": sr.guard,
                     "nvml": sr.nvml}
    tag = S["tag"]
    C.save_json(res, os.path.join(C.RESULTS, "raw", f"graph_{tag}.json"))
    prim = S["prim"]["cobench"]["summary"]
    r0 = gfs.get(S["reference"], {}).get("per_call_median_us")
    s0 = st.get(S["reference"], {}).get("per_call_us")
    print(f"\n{tag}: per-call us  [eager flush median | graph-flush median | graph-steady]  ratio vs {S['reference']}")
    for n in sorted(gfs, key=lambda n: gfs[n]["per_call_median_us"]):
        e = prim.get(n, {}).get("median_us")
        gf, sp = gfs[n]["per_call_median_us"], st.get(n, {}).get("per_call_us")
        print(f"  {n:22s} {e or 0:9.2f} | {gf:9.2f} | {sp or 0:9.2f}   "
              f"{(prim[n]['median_us'] / prim[S['reference']]['median_us']) if e else 0:5.2f} | "
              f"{gf / r0 if r0 else 0:5.2f} | {sp / s0 if (sp and s0) else 0:5.2f}")


if __name__ == "__main__":
    main()
