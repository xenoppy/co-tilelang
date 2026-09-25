"""Tables for the Mamba-2 claims sub-study (C3 / C6-mamba) from the JSON results. CPU only.

    python research/bench/scripts/claims_mamba_report.py > /tmp/tables.md

Verdict rule (README §2.5): ratio = Triton time / TileLang time (the figure's "normalized latency").
reproduced: measured >= 0.9 x claimed; partly: TileLang measurably faster (measured >= 1.05) but
< 0.9 x claimed; not reproduced: measured < 1.05 (TileLang not faster than Triton by more than 5%; the
interleaved medians repeat to < 1%, so 0.95-1.05 is reported as parity).
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402

SM, FLOP_PER_CLK = 188, 1024        # dense 16-bit mma.sync, fp32 accumulate (cobench mma_peak, research/bench/README.md)
H800_PEAK = 989.4                   # dense FP16/BF16 tensor TFLOPS, H800/H100 SXM


def load(kind, name):
    p = os.path.join(C.RESULTS, kind, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def verdict(meas, claim):
    if claim is None:
        return "no claim (table-only shape)"
    if meas >= 0.9 * claim:
        return "reproduced"
    if meas >= 1.05:
        return "partly"
    return "not reproduced (parity)" if meas >= 0.95 else "not reproduced (Triton faster)"


def sweep_cols(name):
    sw = load("triton_sweep", name)
    if sw is None:
        return "- | -"
    return f"{sw['triton_best_us']:.1f} | {sw['ratio_best_triton_over_tilelang']:.2f}"


def cycles_ratio(r):
    v = r["primary"]["variants"]
    return v["triton"]["clock"]["cycles_median"] / v["tilelang"]["clock"]["cycles_median"]


def power(r):
    pw = ((r["primary"].get("nvml") or {}).get("power_w") or {}).get("mean")
    return f"{pw:.0f}" if pw else "-"


def cfg_str(c, kind):
    if c is None:
        return "-"
    if kind == "state_ex":
        return f"{c['block_M']}/{c['block_N']}/{c['block_K']}/s{c['num_stages']}"
    return f"{c['block_M']}/{c['block_N']}/{c['block_K']}/s{c['num_stages']}"


_OCC = None


def ctas(x, shape=None, top=None, variant=None):
    """Resident CTAs/SM from cuOccupancyMaxActiveBlocksPerMultiprocessor (diag/occupancy.json, written by
    claims_mamba_occupancy.py) when available; otherwise '?'."""
    global _OCC
    if _OCC is None:
        p_ = os.path.join(C.RESULTS, "diag", "occupancy.json")
        _OCC = {(o["shape"], o["top"], o["variant"]): o for o in (json.load(open(p_)) if os.path.exists(p_) else [])
                if "ctas_per_sm" in o}
    o = _OCC.get((shape, top, variant))
    return o["ctas_per_sm"] if o else "?"


def extra_tables() -> list:
    out = []
    # graph-mode cross-check (resolution-free) for short kernels
    rows = [(n, load("graphcheck", n), load("bench", n)) for n in [f"CC{i}" for i in range(6)] + [f"CT{i}" for i in range(6)] + [f"B{L}" for L in C.C6_SEQ]]
    rows = [r for r in rows if r[1] is not None]
    if rows:
        out.append("\n#### Cross-check for short kernels: CUDA-graph back-to-back timer (no per-call event quantization)\n")
        out.append("| shape | flush-mode TL / Triton µs (primary) | ratio (primary) | graph-mode TL / Triton µs | ratio (graph) | graph-mode clock TL / Triton MHz | copies x k |")
        out.append("|---|---|---|---|---|---|---|")
        for n, g, b in rows:
            s_ = b["summary"]
            ck = lambda k: statistics.median(x["clock_mhz"] for x in g["rounds"][k] if x["clock_mhz"])  # noqa: E731
            out.append(f"| {n} | {s_['tilelang_us']:.1f} / {s_['triton_us']:.1f} | {s_['ratio_triton_over_tilelang']:.2f} | "
                       f"{g['tilelang_us']:.1f} / {g['triton_us']:.1f} | **{g['ratio_triton_over_tilelang']:.2f}** | "
                       f"{ck('tilelang'):.0f} / {ck('triton'):.0f} | {g['rounds']['tilelang'][0]['n_copies']} x {g['rounds']['tilelang'][0]['k']} |")
    # Triton autotuner run-to-run spread
    out.append("\n#### Triton baseline: as-autotuned time in independent processes vs best forced config\n")
    out.append("| shape | main bench | triton_sweep (auto) | diag run | l2diag run | best forced config (sweep) | TileLang (main) |")
    out.append("|---|---|---|---|---|---|---|")
    for n in ["CC3", "CC4", "CC5", "CT3", "CT4", "CT5", "B8192", "B16384", "B32768"]:
        b, sw, dg, l2 = load("bench", n), load("triton_sweep", n), load("diag", n), load("l2diag", n)
        if b is None:
            continue
        f = lambda x: f"{x:.0f}" if x else "-"  # noqa: E731
        out.append(f"| {n} | {f(b['summary']['triton_us'])} | {f(sw and sw['triton_autotuned_us'])} | {f(dg and dg['triton_us'])} | "
                   f"{f(l2 and l2['timing_us'].get('triton_shipped'))} | {f(sw and sw['triton_best_us'])} | {f(b['summary']['tilelang_us'])} |")
    # warp specialisation diag
    out.append("\n#### Diagnostic: TileLang lowering on sm_120 (same kernel source, upstream pass configs)\n")
    out.append("| shape | config (M/N/K/stages) | as shipped: threads, regs, dyn smem KB, CTAs/SM, µs | tl.disable_warp_specialized: threads, regs, dyn smem KB, CTAs/SM, µs | Triton (same run) µs | ratio Triton/TL shipped -> no-WS |")
    out.append("|---|---|---|---|---|---|")
    for n in ["CC1", "CC4", "B8192", "CT1", "CT4"]:
        d = load("diag", n)
        if d is None:
            continue
        v = d["variants"]
        for j in range(4):
            a_, b_ = v.get(f"top{j}_shipped"), v.get(f"top{j}_no_ws")
            if not a_ or not b_:
                continue
            c = a_["config"]
            cs = f"{c['block_M']}/{c['block_N']}/{c['block_K']}/s{c['num_stages']}"
            g = lambda x, vn: (f"{x['block'][0]}, {x.get('regs')}, {x.get('dyn_smem', 0) // 1024}, {ctas(x, n, j, vn)}, "  # noqa: E731
                           + (f"{x['median_us']:.1f}" if 'median_us' in x else ("fails check" if x.get("ok") is False else "-")))
            ra = d["triton_us"] / a_["median_us"] if "median_us" in a_ else float("nan")
            rb = d["triton_us"] / b_["median_us"] if "median_us" in b_ else float("nan")
            out.append(f"| {n} | {cs} | {g(a_, 'shipped')} | {g(b_, 'no_ws')} | {d['triton_us']:.1f} | {ra:.2f} -> {rb:.2f} |")
    # grid-order diag
    out.append("\n#### Diagnostic: Triton baseline with the grid axes rotated (heads fastest, as in the TileLang kernels)\n")
    out.append("| shape | TileLang µs | Triton as shipped µs | Triton heads-first µs | ratio shipped | ratio heads-first |")
    out.append("|---|---|---|---|---|---|")
    for n in ["CC4", "CC5", "CT4", "CT5", "B32768"]:
        d = load("l2diag", n)
        if d is None:
            continue
        t = d["timing_us"]
        out.append(f"| {n} | {t['tilelang']:.0f} | {t['triton_shipped']:.0f} | {t.get('triton_heads_first', float('nan')):.0f} | "
                   f"{t['triton_shipped'] / t['tilelang']:.2f} | {t.get('triton_heads_first', float('nan')) / t['tilelang']:.2f} |")
    # fork check
    fk, up = load("forkcheck", "fork"), load("forkcheck", "upstream")
    if fk and up:
        out.append("\n#### Fork vs upstream TileLang v0.1.14 (PyPI wheel), same configs, separate kernel caches\n")
        out.append("| shape | CUDA source identical | fork TL µs | upstream TL µs | fork Triton/TL | upstream Triton/TL |")
        out.append("|---|---|---|---|---|---|")
        for n, a_ in fk["shapes"].items():
            b_ = up["shapes"][n]
            out.append(f"| {n} | {'yes' if a_['src_sha256'] == b_['src_sha256'] else 'NO'} | {a_['tilelang_us']:.1f} | {b_['tilelang_us']:.1f} | "
                       f"{a_['ratio_triton_over_tilelang']:.2f} | {b_['ratio_triton_over_tilelang']:.2f} |")
    # minference
    import glob
    mf = sorted(glob.glob(os.path.join(C.RESULTS, "minference", "*.json")), key=lambda p: [int(x) for x in os.path.basename(p)[:-5].split("_")])
    if mf:
        out.append("\n#### Optional: examples/minference (vertical-slash sparse attention, batch 1, heads 1, dim 64)\n")
        out.append("| seq_len | vertical, slash | TileLang µs | Triton µs | measured Triton/TileLang | secondary (example's do_bench) | claimed (H100 PCIe) | verdict | check |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for p_ in mf:
            r = json.load(open(p_))
            pt = r["point"]
            ok = all(x["ok"] for x in r["check"].values())
            out.append(f"| {pt['seq_len']} | {pt['vertical']}, {pt['slash']} | {r['tilelang_us']:.1f} | {r['triton_us']:.1f} | **{r['ratio_triton_over_tilelang']:.2f}** | "
                       f"{r['secondary']['ratio_triton_over_tilelang']:.2f} | {r['claimed']['speedup']:.2f} | {verdict(r['ratio_triton_over_tilelang'], r['claimed']['speedup'])} | {'ok' if ok else 'FAIL'} |")
    la = sorted(glob.glob(os.path.join(C.RESULTS, "linear_attn", "*.json")))
    if la:
        out.append("\n#### Optional: examples/linear_attention/example_linear_attn_fwd.py vs flash-linear-attention (fla-core 0.5.2) `fused_chunk_linear_attn` (no numeric claim upstream)\n")
        out.append("| B, S, H, D | TileLang µs | FLA µs | FLA/TileLang (primary) | secondary (event do_bench) | check |")
        out.append("|---|---|---|---|---|---|")
        for p_ in la:
            r = json.load(open(p_))
            sh = r["shape"]
            out.append(f"| {sh['B']}, {sh['S']}, {sh['H']}, {sh['D']} | {r['tilelang_us']:.1f} | {r['fla_us']:.1f} | **{r['ratio_fla_over_tilelang']:.2f}** | "
                       f"{r['secondary']['ratio_fla_over_tilelang']:.2f} | {'ok' if r['correctness_ok'] else 'FAIL'} |")
    return out


def main():
    claimed = json.load(open(os.path.join(C.RESULTS, "claimed.json")))
    out = []
    summary = {}
    for pre, key, title in (("CC", "chunk_scan_triton_over_tilelang", "C3 chunk-scan (example_mamba_chunk_scan.py)"),
                            ("CT", "chunk_state_triton_over_tilelang", "C3 chunk-state (example_mamba_chunk_state.py)")):
        claims = claimed["c3"][key]
        out.append(f"\n#### {title}\n")
        out.append("| shape | batch x seq | TileLang cfg (M/N/K/stages) | TileLang µs | Triton µs | Triton cfg (M/N/K/warps/stages) | **measured Triton/TileLang** | cycles ratio | clock TL / Triton MHz | board W | secondary (do_bench) | TL best of top-6 | best forced Triton µs | Triton-best / TileLang | claimed | verdict | check |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        ratios = []
        for i in range(6):
            name = f"{pre}{i}"
            r = load("bench", name)
            if r is None:
                out.append(f"| {name} | | | not run |" + " |" * 13)
                continue
            s, sh = r["summary"], r["shape"]
            claim = claims[i] if i < len(claims) else None
            topk = r.get("topk", {}).get("configs", {})
            best_alt = min((d["median_us"] for d in topk.values() if "median_us" in d), default=None)
            alt = f"{r['topk']['triton_us'] / best_alt:.3f}" if best_alt else "-"
            tc = r.get("triton_best_config") or ""
            tcs = "/".join(x.split(": ")[1] for x in tc.split(", ")[:3]) + (f"/w{tc.split('num_warps: ')[1].split(',')[0]}/s{tc.split('num_stages: ')[1].split(',')[0]}" if tc else "")
            chk = "ok" if s["correctness_ok"] else "FAIL"
            v = verdict(s["ratio_triton_over_tilelang"], claim)
            if claim is not None:
                ratios.append((s["ratio_triton_over_tilelang"], claim))
            cyc = cycles_ratio(r)
            out.append(f"| {name} | {sh['batch']} x {sh['seqlen']} | {cfg_str(r['tune']['best_config'], sh['kind'])} | {s['tilelang_us']:.1f} | "
                       f"{s['triton_us']:.1f} | {tcs} | **{s['ratio_triton_over_tilelang']:.2f}** | {cyc:.2f} | "
                       f"{s['clock_mhz_tilelang']:.0f} / {s['clock_mhz_triton']:.0f} | {power(r)} | {r['secondary']['ratio_triton_over_tilelang']:.2f} | {alt} | "
                       f"{sweep_cols(name)} | {claim if claim is not None else '-'} | {v} | {chk} |")
        if ratios:
            gm = statistics.geometric_mean([m for m, _ in ratios])
            am = statistics.mean([m for m, _ in ratios])
            out.append(f"\nFigure shapes ({pre}0-{pre}4): measured mean {am:.2f} (geomean {gm:.2f}); claimed mean "
                       f"{statistics.mean([c for _, c in ratios]):.2f} (paper text: 1.77 / 2.10 average speed-up).")
        summary[pre] = ratios

    out.append("\n#### C6 chunk-scan benchmark (benchmark/mamba2, batch 8, heads 80, groups 1, chunk 256, dim 64, dstate 128)\n")
    c6 = claimed.get("c6") or {}
    out.append("| seq_len | TileLang cfg | TileLang µs | TileLang TFLOPS | Triton µs | Triton TFLOPS | **measured TL/Triton** | cycles ratio | clock TL / Triton MHz | secondary | Triton-best / TileLang | claimed TL/Triton (H800 fig.) | verdict | TL % of mma.sync peak @clk | H800 TL TFLOPS (README) | H800 % of 989 TF | check |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for j, L in enumerate(C.C6_SEQ):
        name = f"B{L}"
        r = load("bench", name)
        if r is None:
            out.append(f"| {L} | not run |" + " |" * 15)
            continue
        s = r["summary"]
        claim = (c6.get("ratio_tilelang_over_triton") or [None] * 6)[j]
        peak = SM * FLOP_PER_CLK * s["clock_mhz_tilelang"] * 1e6 / 1e12
        h800 = (c6.get("tilelang_tflops_readme") or [None] * 6)[j]
        out.append(f"| {L} | {cfg_str(r['tune']['best_config'], 'scan_bm')} | {s['tilelang_us']:.1f} | {s['tilelang_tflops']:.1f} | {s['triton_us']:.1f} | "
                   f"{s['triton_tflops']:.1f} | **{s['ratio_triton_over_tilelang']:.2f}** | {cycles_ratio(r):.2f} | "
                   f"{s['clock_mhz_tilelang']:.0f} / {s['clock_mhz_triton']:.0f} | {r['secondary']['ratio_triton_over_tilelang']:.2f} | "
                   f"{sweep_cols(name).split(' | ')[1]} | {claim} | {verdict(s['ratio_triton_over_tilelang'], claim)} | {100 * s['tilelang_tflops'] / peak:.1f}% | {h800} | "
                   f"{100 * h800 / H800_PEAK:.1f}% | {'ok' if s['correctness_ok'] else 'FAIL'} |")
    # roofline position of both kernels (compulsory traffic, upstream FLOP formula)
    out.append("\n#### Roofline position (compulsory DRAM traffic = each input read once, output written once)\n")
    out.append("| shape | compulsory MB | GFLOP | TileLang TB/s | Triton TB/s | TileLang TFLOPS | Triton TFLOPS | TL % of DRAM peak 1.79 TB/s | TL % of mma.sync peak @clk |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for name in [f"CC{i}" for i in range(6)] + [f"CT{i}" for i in range(6)] + [f"B{L}" for L in C.C6_SEQ]:
        r = load("bench", name)
        if r is None:
            continue
        sh, s_ = C.SHAPES[name], r["summary"]
        mb, gf = C.min_bytes(sh), C.flops(sh)
        peak = SM * FLOP_PER_CLK * s_["clock_mhz_tilelang"] * 1e6
        out.append(f"| {name} | {mb / 1e6:.0f} | {gf / 1e9:.1f} | {mb / s_['tilelang_us'] / 1e6:.2f} | {mb / s_['triton_us'] / 1e6:.2f} | "
                   f"{gf / s_['tilelang_us'] / 1e6:.1f} | {gf / s_['triton_us'] / 1e6:.1f} | {100 * mb / s_['tilelang_us'] / 1e6 / 1.792:.0f}% | "
                   f"{100 * gf / (s_['tilelang_us'] * 1e-6) / peak:.0f}% |")
    out += extra_tables()
    print("\n".join(out))


if __name__ == "__main__":
    main()
