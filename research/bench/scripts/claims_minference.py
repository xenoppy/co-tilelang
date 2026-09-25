"""Optional claim: examples/minference/README.md (vertical-slash sparse attention, TileLang vs the Triton
kernel shipped in the same example; H100 PCIe, batch 1, heads 1, dim 64). Part of sub-study C.

    source research/env.sh
    python research/bench/scripts/claims_minference.py --compile-only          # CPU: kernels + index op
    python research/bench/scripts/run_guarded.py -- flock $GPU_LOCK \
        python research/bench/scripts/claims_minference.py --seq 8192

The example's own main() is run unchanged for each (seq_len, vertical, slash) point, with the module's
`do_bench` replaced by a recorder, so that its input construction, its index conversion and its
TileLang-vs-Triton assert_close run exactly as shipped and the two timed callables (`_attn(True)` =
Triton incl. its zeros_like output init, `_attn(False)` = TileLang) are captured. Then:
  * torch reference: dense fp32 softmax attention restricted to the mask the index op produced
    (per 64-row query block: the 64-key slash blocks of block_offset with key <= query, plus the
    column_index columns), compared with both outputs;
  * primary timer cobench.bench_variants (clean flush, gate, clock), secondary the example's own
    tilelang.profiler.do_bench.
Writes research/results/2026-09-24_claims_repro/C_mamba/minference/<seq>_<v>_<s>.json.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402

POINTS = [(L, v, s) for L in (8192, 16384, 32768, 65536) for v, s in ((1000, 200), (1000, 600), (800, 600))]
CLAIMED = {  # README table: (triton ms, tilelang ms, speedup)
    (8192, 1000, 200): (0.168, 0.105, 1.60), (8192, 1000, 600): (0.207, 0.119, 1.74), (8192, 800, 600): (0.207, 0.122, 1.70),
    (16384, 1000, 200): (0.261, 0.167, 1.56), (16384, 1000, 600): (0.419, 0.258, 1.62), (16384, 800, 600): (0.422, 0.255, 1.65),
    (32768, 1000, 200): (0.374, 0.248, 1.51), (32768, 1000, 600): (0.823, 0.554, 1.49), (32768, 800, 600): (0.826, 0.558, 1.48),
    (65536, 1000, 200): (0.637, 0.524, 1.22), (65536, 1000, 600): (1.758, 1.501, 1.17), (65536, 800, 600): (1.783, 1.489, 1.20),
}


def closure_vars(fn) -> dict:
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in (fn.__closure__ or ()))))


def reference(cv) -> "torch.Tensor":
    """fp32 attention over the index op's sparsity pattern (the semantics of the Triton kernel)."""
    import torch
    q, k, v = (cv[n].float() for n in ("query", "key", "value"))
    L = int(cv["context_size"])
    bm, bn = int(cv["block_size_M"]), int(cv["block_size_N"])
    bc, bo, cc, ci = cv["block_count"][0, 0], cv["block_offset"][0, 0], cv["column_count"][0, 0], cv["column_index"][0, 0]
    out = torch.empty(L, q.shape[-1], device=q.device)
    scale = float(cv["sm_scale"])
    ar_n = torch.arange(bn, device=q.device)
    for m in range(bc.shape[0]):
        r0, r1 = m * bm, min(L, (m + 1) * bm)
        keys = []
        blocks = bo[m, : int(bc[m])]
        if blocks.numel():
            keys.append((blocks[:, None] + ar_n[None, :]).reshape(-1))
        ncol = int(cc[m])
        cols = ci[m, :ncol].long()
        rows = torch.arange(r0, r1, device=q.device)
        s_parts, v_parts = [], []
        if keys:
            kk = keys[0].long()
            valid = kk < L
            kk_c = kk.clamp(max=L - 1)
            s = (q[0, 0, r0:r1] @ k[0, 0, kk_c].T) * scale
            s = s.masked_fill(~(valid[None, :] & (kk[None, :] <= rows[:, None])), float("-inf"))
            s_parts.append(s)
            v_parts.append(v[0, 0, kk_c])
        if ncol:
            s_parts.append((q[0, 0, r0:r1] @ k[0, 0, cols].T) * scale)
            v_parts.append(v[0, 0, cols])
        s = torch.cat(s_parts, 1)
        p = torch.softmax(s, dim=-1)
        out[r0:r1] = p @ torch.cat(v_parts, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=None, help="only points with this seq_len")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()
    outdir = os.path.join(C.RESULTS, "minference")
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    mod = C._load(os.path.join(C.ROOT, "examples/minference/example_vertical_slash_sparse_attn.py"), "tl_ex_minference")
    pts = [p for p in POINTS if a.seq is None or p[0] == a.seq]
    if a.compile_only:
        mod._load_vertical_slash_index_ops()
        for L, v, s in pts:
            mod._tl_vs_sparse_flashattn(1, 1, L, 64, min(L, v), min(L, s))
            print("compiled", L, v, s, flush=True)
        return
    import torch
    import cobench as cb
    real_do_bench = mod.do_bench
    for L, v, s in pts:
        captured = []
        mod.do_bench = lambda fn, *args, **kw: (captured.append(fn), 1.0)[1]
        try:
            mod.main(batch=1, heads=1, seq_len=L, head_dim=64, vertical_size=v, slash_size=s)  # asserts TL ~ Triton
        finally:
            mod.do_bench = real_do_bench
        f_tr, f_tl = captured  # main(): do_bench(lambda: _attn(True)) then do_bench(lambda: _attn(False))
        cv = closure_vars(closure_vars(f_tr)["_attn"])
        o_tr, o_tl = f_tr(), f_tl()
        ref = reference(cv)
        torch.cuda.synchronize()
        chk = {n: C.pair_error(o[0, 0], ref) for n, o in (("tilelang", o_tl), ("triton", o_tr))}
        for d in chk.values():
            d["ok"] = d["rel_l2"] <= 1e-2
        del o_tr, o_tl, ref
        vres = cb.bench_variants({"tilelang": f_tl, "triton": f_tr}, reference="triton", reps=a.reps, clock=True,
                                 label=f"minference {L} {v} {s}")
        print(vres, flush=True)
        sec = {"tilelang": [], "triton": []}
        for _ in range(3):
            sec["triton"].append(real_do_bench(f_tr))
            sec["tilelang"].append(real_do_bench(f_tl))
        t_tl, t_tr = vres.variants["tilelang"]["total"]["median"], vres.variants["triton"]["total"]["median"]
        res = dict(point=dict(seq_len=L, vertical=v, slash=s, batch=1, heads=1, head_dim=64), check=chk,
                   tilelang_us=t_tl, triton_us=t_tr, ratio_triton_over_tilelang=t_tr / t_tl,
                   secondary=dict(method="example's tilelang.profiler.do_bench (event, 256 MB flush, mean), median of 3",
                                  tilelang_ms=statistics.median(sec["tilelang"]), triton_ms=statistics.median(sec["triton"]), raw=sec),
                   claimed=dict(zip(("triton_ms", "tilelang_ms", "speedup"), CLAIMED[(L, v, s)])),
                   clock_mhz=(vres.variants["tilelang"].get("clock") or {}).get("median"), primary=vres.to_dict(),
                   env=C.env_info(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        res["secondary"]["ratio_triton_over_tilelang"] = res["secondary"]["triton_ms"] / res["secondary"]["tilelang_ms"]
        C.dump(res, os.path.join(outdir, f"{L}_{v}_{s}.json"))
        print(f"[minference {L} {v} {s}] TL {t_tl:.1f} us Triton {t_tr:.1f} us ratio {t_tr / t_tl:.3f} "
              f"(claimed {CLAIMED[(L, v, s)][2]}) check {chk}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
