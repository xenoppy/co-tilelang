"""Tables + verdicts for sub-study B (C2 FlashAttention forward, C4 MLA decode, optional attention
sink) from the raw JSON in research/results/2026-09-24_claims_repro/B_attention/raw/.

  python research/bench/scripts/claims_attn_report.py [median_us|mean_us]   # markdown to stdout
  -> also writes B_attention/summary_<stat>.json

Primary statistic: cobench clean-flush median per variant (same process, interleaved reps, clock
probe). Secondary: the upstream scripts' own timers (tilelang.profiler.do_bench for TileLang,
triton.testing.do_bench for the baselines, as benchmark_mla.py / the tilelang-benchmark scripts do).
Ratios are baseline_time / TileLang_time (> 1: TileLang faster), the quantity the figures plot.
Verdict rule (claims README section 2.5; "partly" made explicit here):
  reproduced     measured >= 0.9 x claimed
  partly         TileLang still faster (measured > 1.0) but measured < 0.9 x claimed
  not reproduced measured <= 1.0 (TileLang not faster than the baseline)
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_attn_common as C  # noqa: E402

RAW = os.path.join(C.RESULTS, "raw")
FA_SHAPES = ["FA0", "FA1", "FA2", "FA3", "FA4"]
TL_PRIMARY = "tl_bhsd_h100cfg_s1"
TL_EXAMPLES = ["tl_bhsd_h100cfg_s1", "tl_bhsd_main", "tl_bshd_main", "tl_bshd_tune"]
TRITON = ["triton_tlbench_ws", "triton_tlbench", "triton_tut06"]


def verdict(measured, claimed):
    if measured is None or claimed is None:
        return "n/a"
    if measured >= 0.9 * claimed:
        return "reproduced"
    if measured > 1.0:
        return "partly"
    return "not reproduced"


def gmean(v):
    v = [x for x in v if x]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def ratio(a, b):
    return a / b if (a and b) else None


def up_time(d, name):
    u = (d["variants"].get(name) or {}).get("upstream") or {}
    return u.get("tilelang_do_bench_us") if name.startswith("tl_") else u.get("triton_do_bench_us")


# ------------------------------------------------------------------------------------------------ C2
def fa_section(claims, stat, out):
    L = []
    cl = claims["fa"]["ratios"]
    rows, per = {}, {}
    for s in FA_SHAPES:
        p = os.path.join(RAW, f"fa_{s}.json")
        if not os.path.exists(p):
            continue
        d = load(p)
        sm = d["cobench"]["summary"]
        v = {n: sm[n][stat] for n in sm}
        tri_best_name = min((n for n in TRITON if n in v), key=lambda n: v[n])
        r = {"shape": d["shape"], "t": v, "triton_best_name": tri_best_name,
             "tl_examples_best": min(v[n] for n in TL_EXAMPLES if n in v),
             "clock": {n: sm[n].get("clock_mhz_median") for n in sm},
             "kcycles": {n: sm[n].get("kcycles_median") for n in sm},
             "cv": {n: sm[n].get("cv") for n in sm},
             "power_w": ((d["cobench"].get("nvml") or {}).get("power_w") or {}).get("mean"),
             "power_max_w": ((d["cobench"].get("nvml") or {}).get("power_w") or {}).get("max"),
             "temp_c": (d["cobench"].get("nvml") or {}).get("temp_c"),
             "guard": d["cobench"].get("guard"),
             "up": {n: up_time(d, n) for n in v},
             "resources": {n: d["variants"][n].get("resources") for n in d["variants"] if d["variants"][n].get("resources")},
             "unlaunchable": d.get("unlaunchable"), "sweep_best": d.get("sweep_best"),
             "checks": {n: (x.get("check") or {}).get("max_abs") for n, x in d["variants"].items()},
             "errors": {n: x.get("error") for n, x in d["variants"].items() if x.get("error")},
             "triton_configs": d.get("triton_best_configs")}
        rows[s] = r
        t = v
        tl = t[TL_PRIMARY]
        m = {"triton_best": ratio(t[tri_best_name], tl), "triton_upstream_ws": ratio(t.get("triton_tlbench_ws"), tl),
             "pytorch_sdpa_flash": ratio(t.get("sdpa_flash"), tl), "sdpa_cudnn": ratio(t.get("sdpa_cudnn"), tl),
             "sdpa_efficient": ratio(t.get("sdpa_efficient"), tl), "sdpa_math": ratio(t.get("sdpa_math"), tl),
             "torch_naive": ratio(t.get("torch_naive"), tl), "flashinfer": ratio(t.get("flashinfer_prefill"), tl),
             "fa2_pip": ratio(t.get("fa2_pip"), tl),
             "tl_best_example_vs_primary": ratio(r["tl_examples_best"], tl), "tl_sweep_vs_primary": ratio(t.get("tl_bhsd_sweep"), tl)}
        up = r["up"]
        m_up = {"triton_best": ratio(min(up[n] for n in TRITON if up.get(n)), up[TL_PRIMARY]),
                "pytorch_sdpa_flash": ratio(up.get("sdpa_flash"), up[TL_PRIMARY])}
        c = {"triton": cl[s]["Triton"], "pytorch": cl[s]["PyTorch"], "fa3": cl[s]["FA3"]}
        per[s] = {"measured": m, "measured_upstream_timers": m_up, "claimed": c,
                  "verdict_triton": verdict(m["triton_best"], c["triton"]),
                  "verdict_pytorch": verdict(m["pytorch_sdpa_flash"], c["pytorch"]),
                  "verdict_triton_upstream_timers": verdict(m_up["triton_best"], c["triton"]),
                  "verdict_pytorch_upstream_timers": verdict(m_up["pytorch_sdpa_flash"], c["pytorch"])}
    out["fa"] = {"rows": rows, "per_shape": per}

    L.append(f"#### C2 times (cobench clean-flush {stat.replace('_us', '')}, us)\n")
    cols = [("TileLang primary", TL_PRIMARY), ("TL bhsd main", "tl_bhsd_main"), ("TL bshd main", "tl_bshd_main"),
            ("TL bshd tune", "tl_bshd_tune"), ("TL sweep best", "tl_bhsd_sweep"),
            ("Triton tlbench ws", "triton_tlbench_ws"), ("Triton tlbench", "triton_tlbench"), ("Triton tut06", "triton_tut06"),
            ("SDPA flash", "sdpa_flash"), ("SDPA cuDNN", "sdpa_cudnn"), ("SDPA efficient", "sdpa_efficient"),
            ("SDPA math", "sdpa_math"), ("torch naive", "torch_naive"), ("FlashInfer", "flashinfer_prefill"),
            ("flash-attn 2.8.3", "fa2_pip")]
    L.append("| shape | " + " | ".join(c for c, _ in cols) + " |")
    L.append("|---" * (len(cols) + 1) + "|")
    for s, r in rows.items():
        sh = r["shape"]
        L.append(f"| {s} (S={sh['seq']}, {'causal' if sh['causal'] else 'full'}) | "
                 + " | ".join(fmt(r["t"].get(n)) for _, n in cols) + " |")
    L.append("")
    L.append("#### C2 ratios (baseline / TileLang primary) and verdicts\n")
    L.append("| shape | Triton (best of 3): measured / claimed / verdict | Triton as upstream ran it (ws=True) | PyTorch = SDPA flash (FA2): measured / claimed / verdict | SDPA cuDNN | SDPA efficient | SDPA math | torch naive | FlashInfer | flash-attn 2.8.3 | FA3 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    agg = {k: [] for k in ("tri", "tri_ws", "pt", "cud", "eff", "math", "nv", "fi", "fa2", "c_tri", "c_pt", "c_fa3",
                           "u_tri", "u_pt")}
    for s, p in per.items():
        m, c = p["measured"], p["claimed"]
        L.append(f"| {s} | {fmt(m['triton_best'], 2)} / {c['triton']:.2f} / {p['verdict_triton']} | "
                 f"{fmt(m['triton_upstream_ws'], 2) if m['triton_upstream_ws'] else 'unlaunchable'} | "
                 f"{fmt(m['pytorch_sdpa_flash'], 2)} / {c['pytorch']:.2f} / {p['verdict_pytorch']} | "
                 f"{fmt(m['sdpa_cudnn'], 2)} | {fmt(m['sdpa_efficient'], 2)} | {fmt(m['sdpa_math'], 1)} | "
                 f"{fmt(m['torch_naive'], 2)} | {fmt(m['flashinfer'], 2)} | {fmt(m['fa2_pip'], 2)} | "
                 f"not testable (claimed {c['fa3']:.2f}) |")
        for k, x in (("tri", m["triton_best"]), ("tri_ws", m["triton_upstream_ws"]), ("pt", m["pytorch_sdpa_flash"]),
                     ("cud", m["sdpa_cudnn"]), ("eff", m["sdpa_efficient"]), ("math", m["sdpa_math"]),
                     ("nv", m["torch_naive"]), ("fi", m["flashinfer"]), ("fa2", m["fa2_pip"]),
                     ("c_tri", c["triton"]), ("c_pt", c["pytorch"]), ("c_fa3", c["fa3"]),
                     ("u_tri", p["measured_upstream_timers"]["triton_best"]),
                     ("u_pt", p["measured_upstream_timers"]["pytorch_sdpa_flash"])):
            agg[k].append(x)
    g = {k: gmean(v) for k, v in agg.items()}
    L.append(f"| **geomean** | **{fmt(g['tri'], 2)} / {fmt(g['c_tri'], 2)} / {verdict(g['tri'], g['c_tri'])}** | "
             f"{fmt(g['tri_ws'], 2)} (3 causal shapes) | **{fmt(g['pt'], 2)} / {fmt(g['c_pt'], 2)} / {verdict(g['pt'], g['c_pt'])}** | "
             f"{fmt(g['cud'], 2)} | {fmt(g['eff'], 2)} | {fmt(g['math'], 1)} | {fmt(g['nv'], 2)} | {fmt(g['fi'], 2)} | "
             f"{fmt(g['fa2'], 2)} | not testable (claimed {fmt(g['c_fa3'], 2)}) |")
    out["fa"]["geomean"] = g
    # sensitivity: TileLang = best of every TileLang variant measured (example configs + sm_120 sweep)
    sens = {"tri": [], "pt": [], "cud": [], "fa2": []}
    for s, r in rows.items():
        t = r["t"]
        tlb = min(t[n] for n in t if n.startswith("tl_"))
        sens["tri"].append(ratio(t[r["triton_best_name"]], tlb))
        sens["pt"].append(ratio(t.get("sdpa_flash"), tlb))
        sens["cud"].append(ratio(t.get("sdpa_cudnn"), tlb))
        sens["fa2"].append(ratio(t.get("fa2_pip"), tlb))
        per[s]["measured_vs_tl_best_variant"] = {"triton_best": sens["tri"][-1], "pytorch_sdpa_flash": sens["pt"][-1],
                                                 "sdpa_cudnn": sens["cud"][-1], "fa2_pip": sens["fa2"][-1]}
    gs = {k: gmean(v) for k, v in sens.items()}
    out["fa"]["geomean_vs_tl_best_variant"] = gs
    L.append(f"| sensitivity: TileLang = fastest TileLang variant per shape (geomean) | {fmt(gs['tri'], 2)} / "
             f"{fmt(g['c_tri'], 2)} / {verdict(gs['tri'], g['c_tri'])} | | {fmt(gs['pt'], 2)} / {fmt(g['c_pt'], 2)} / "
             f"{verdict(gs['pt'], g['c_pt'])} | {fmt(gs['cud'], 2)} | | | | | {fmt(gs['fa2'], 2)} | |")
    L.append("")
    L.append("#### C2 with the upstream scripts' timers (tilelang do_bench for TileLang, triton do_bench for baselines)\n")
    L.append("| shape | TileLang us | Triton best us | SDPA flash us | Triton ratio / verdict | SDPA flash ratio / verdict |")
    L.append("|---|---|---|---|---|---|")
    for s, p in per.items():
        up = rows[s]["up"]
        tb = min(up[n] for n in TRITON if up.get(n))
        mu = p["measured_upstream_timers"]
        L.append(f"| {s} | {fmt(up[TL_PRIMARY])} | {fmt(tb)} | {fmt(up.get('sdpa_flash'))} | "
                 f"{fmt(mu['triton_best'], 2)} / {p['verdict_triton_upstream_timers']} | "
                 f"{fmt(mu['pytorch_sdpa_flash'], 2)} / {p['verdict_pytorch_upstream_timers']} |")
    L.append(f"| **geomean** | | | | {fmt(g['u_tri'], 2)} / {verdict(g['u_tri'], g['c_tri'])} | "
             f"{fmt(g['u_pt'], 2)} / {verdict(g['u_pt'], g['c_pt'])} |")
    L.append("")
    L.append("#### C2 SM clock (ClockProbe median per variant, MHz), board power (NVML, mean / max W), CV of the primary\n")
    L.append("| shape | TileLang | Triton best | SDPA flash | SDPA cuDNN | power mean / max | TileLang CV | kcycles TL / Triton best / SDPA flash |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s, r in rows.items():
        ck, kc = r["clock"], r["kcycles"]
        L.append(f"| {s} | {fmt(ck.get(TL_PRIMARY), 0)} | {fmt(ck.get(r['triton_best_name']), 0)} | "
                 f"{fmt(ck.get('sdpa_flash'), 0)} | {fmt(ck.get('sdpa_cudnn'), 0)} | {fmt(r['power_w'], 0)} / "
                 f"{fmt(r['power_max_w'], 0)} | {fmt(r['cv'].get(TL_PRIMARY), 3)} | {fmt(kc.get(TL_PRIMARY), 1)} / "
                 f"{fmt(kc.get(r['triton_best_name']), 1)} / {fmt(kc.get('sdpa_flash'), 1)} |")
    return L


# ------------------------------------------------------------------------------------------------ C2, graph timers
def fa_graph_section(claims, out):
    """C2 with the resolution-free CUDA-graph timers (claims_attn_graph.py): graph_flush (clean flush,
    host gate, interleaved; N calls on N cold input copies per rep, per-call = rep / N) and steady
    (bench_steady of graph replays). The eager flush timer of fa_section is quantised to ~2.048 us."""
    L = []
    cl = claims["fa"]["ratios"]
    per, rows = {}, {}
    for s in FA_SHAPES:
        p = os.path.join(RAW, f"graph_{s}.json")
        if not os.path.exists(p):
            continue
        d = load(p)
        gf = {n: v["per_call_median_us"] for n, v in d["graph_flush"]["summary"].items()}
        st = {n: v["per_call_us"] for n, v in d["steady"]["summary"].items()}
        eager = load(os.path.join(RAW, f"fa_{s}.json"))["cobench"]["summary"]
        rows[s] = {"graph_flush": gf, "steady": st, "eager_flush": {n: v["median_us"] for n, v in eager.items()},
                   "calls_per_rep": d["graph_flush"]["calls_per_rep"],
                   "steady_clock": {n: v.get("clock_mhz") for n, v in d["steady"]["summary"].items()},
                   "steady_power": {n: v.get("power_w") for n, v in d["steady"]["summary"].items()},
                   "graph_flush_cv": {n: v["cv"] for n, v in d["graph_flush"]["summary"].items()},
                   "steady_cv": {n: v["cv_slices"] for n, v in d["steady"]["summary"].items()},
                   "excluded": {n: v.get("excluded") or v.get("error") for n, v in d["variants"].items()
                                if v.get("excluded") or v.get("error")},
                   "guard": [d["graph_flush"]["guard"], d["steady"]["guard"]]}
        c = {"triton": cl[s]["Triton"], "pytorch": cl[s]["PyTorch"], "fa3": cl[s]["FA3"]}
        rr = {}
        for timer, t in (("graph_flush", gf), ("steady", st)):
            tl = t[TL_PRIMARY]
            tb = min(t[n] for n in TRITON if n in t)
            rr[timer] = {"triton_best": tb / tl, "triton_upstream_ws": ratio(t.get("triton_tlbench_ws"), tl),
                         "pytorch_sdpa_flash": ratio(t.get("sdpa_flash"), tl), "sdpa_cudnn": ratio(t.get("sdpa_cudnn"), tl),
                         "sdpa_efficient": ratio(t.get("sdpa_efficient"), tl), "torch_naive": ratio(t.get("torch_naive"), tl),
                         "flashinfer": ratio(t.get("flashinfer_prefill"), tl), "fa2_pip": ratio(t.get("fa2_pip"), tl),
                         "triton_best_vs_tl_best": tb / min(t[n] for n in t if n.startswith("tl_")),
                         "sdpa_flash_vs_tl_best": ratio(t.get("sdpa_flash"), min(t[n] for n in t if n.startswith("tl_")))}
            rr[timer]["verdict_triton"] = verdict(rr[timer]["triton_best"], c["triton"])
            rr[timer]["verdict_pytorch"] = verdict(rr[timer]["pytorch_sdpa_flash"], c["pytorch"])
        per[s] = {"claimed": c, **rr}
    if not rows:
        return L
    out["fa_graph"] = {"rows": rows, "per_shape": per}
    L.append("#### C2 per-call times with the CUDA-graph timers (us): graph-flush (**primary for C2**) / steady\n")
    cols = [("TileLang primary", TL_PRIMARY), ("TL bhsd main", "tl_bhsd_main"), ("TL bshd main", "tl_bshd_main"),
            ("TL bshd tune", "tl_bshd_tune"), ("TL sweep best", "tl_bhsd_sweep"),
            ("Triton tlbench ws", "triton_tlbench_ws"), ("Triton tlbench", "triton_tlbench"), ("Triton tut06", "triton_tut06"),
            ("SDPA flash", "sdpa_flash"), ("SDPA cuDNN", "sdpa_cudnn"), ("SDPA efficient", "sdpa_efficient"),
            ("SDPA math", "sdpa_math"), ("torch naive", "torch_naive"), ("FlashInfer", "flashinfer_prefill"),
            ("flash-attn 2.8.3", "fa2_pip")]
    L.append("| shape (calls per graph-flush rep) | " + " | ".join(c for c, _ in cols) + " |")
    L.append("|---" * (len(cols) + 1) + "|")
    for s, r in rows.items():
        cells = []
        for _, n in cols:
            a, b = r["graph_flush"].get(n), r["steady"].get(n)
            cells.append("–" if a is None else f"{a:.2f} / {b:.2f}" if b else f"{a:.2f}")
        L.append(f"| {s} ({r['calls_per_rep']}) | " + " | ".join(cells) + " |")
    L.append("")
    L.append("#### C2 ratios (baseline / TileLang primary) and verdicts with the graph timers (graph-flush; steady in brackets)\n")
    L.append("| shape | Triton best: measured / claimed / verdict | Triton as upstream ran it (ws=True) | PyTorch = SDPA flash: measured / claimed / verdict | SDPA cuDNN | SDPA efficient | torch naive | FlashInfer | flash-attn 2.8.3 | old eager-flush Triton / SDPA flash ratios |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    agg = {k: [] for k in ("tri", "tri_s", "pt", "pt_s", "cud", "cud_s", "eff", "nv", "fi", "fa2", "ws", "c_tri", "c_pt",
                           "tb_tri", "tb_pt", "tb_tri_s", "tb_pt_s")}
    old = (out.get("fa") or {}).get("per_shape", {})
    for s, p in per.items():
        g, st_, c = p["graph_flush"], p["steady"], p["claimed"]
        om = (old.get(s) or {}).get("measured", {})
        L.append(f"| {s} | {g['triton_best']:.2f} [{st_['triton_best']:.2f}] / {c['triton']:.2f} / {g['verdict_triton']} "
                 f"[{st_['verdict_triton']}] | {fmt(g['triton_upstream_ws'], 2) if g['triton_upstream_ws'] else 'unlaunchable'} | "
                 f"{g['pytorch_sdpa_flash']:.2f} [{st_['pytorch_sdpa_flash']:.2f}] / {c['pytorch']:.2f} / {g['verdict_pytorch']} "
                 f"[{st_['verdict_pytorch']}] | {fmt(g['sdpa_cudnn'], 2)} [{fmt(st_['sdpa_cudnn'], 2)}] | {fmt(g['sdpa_efficient'], 2)} | "
                 f"{fmt(g['torch_naive'], 2)} | {fmt(g['flashinfer'], 2)} | {fmt(g['fa2_pip'], 2)} [{fmt(st_['fa2_pip'], 2)}] | "
                 f"{fmt(om.get('triton_best'), 2)} / {fmt(om.get('pytorch_sdpa_flash'), 2)} |")
        for k, x in (("tri", g["triton_best"]), ("tri_s", st_["triton_best"]), ("pt", g["pytorch_sdpa_flash"]),
                     ("pt_s", st_["pytorch_sdpa_flash"]), ("cud", g["sdpa_cudnn"]), ("cud_s", st_["sdpa_cudnn"]),
                     ("eff", g["sdpa_efficient"]), ("nv", g["torch_naive"]), ("fi", g["flashinfer"]), ("fa2", g["fa2_pip"]),
                     ("ws", g["triton_upstream_ws"]), ("c_tri", c["triton"]), ("c_pt", c["pytorch"]),
                     ("tb_tri", g["triton_best_vs_tl_best"]), ("tb_pt", g["sdpa_flash_vs_tl_best"]),
                     ("tb_tri_s", st_["triton_best_vs_tl_best"]), ("tb_pt_s", st_["sdpa_flash_vs_tl_best"])):
            agg[k].append(x)
    gm = {k: gmean(v) for k, v in agg.items()}
    L.append(f"| **geomean** | **{fmt(gm['tri'], 2)} [{fmt(gm['tri_s'], 2)}] / {fmt(gm['c_tri'], 2)} / {verdict(gm['tri'], gm['c_tri'])} "
             f"[{verdict(gm['tri_s'], gm['c_tri'])}]** | {fmt(gm["ws"], 2)} ({sum(1 for x in agg["ws"] if x)} causal shapes) | **{fmt(gm['pt'], 2)} [{fmt(gm['pt_s'], 2)}] / "
             f"{fmt(gm['c_pt'], 2)} / {verdict(gm['pt'], gm['c_pt'])} [{verdict(gm['pt_s'], gm['c_pt'])}]** | {fmt(gm['cud'], 2)} "
             f"[{fmt(gm['cud_s'], 2)}] | {fmt(gm['eff'], 2)} | {fmt(gm['nv'], 2)} | {fmt(gm['fi'], 2)} | {fmt(gm['fa2'], 2)} | |")
    L.append(f"| sensitivity: TileLang = fastest TileLang variant per shape | {fmt(gm['tb_tri'], 2)} [{fmt(gm['tb_tri_s'], 2)}] / "
             f"{fmt(gm['c_tri'], 2)} / {verdict(gm['tb_tri'], gm['c_tri'])} | | {fmt(gm['tb_pt'], 2)} [{fmt(gm['tb_pt_s'], 2)}] / "
             f"{fmt(gm['c_pt'], 2)} / {verdict(gm['tb_pt'], gm['c_pt'])} | | | | | | |")
    out["fa_graph"]["geomean"] = gm
    return L


# ------------------------------------------------------------------------------------------------ C4 graph spot check
def mla_graph_section(out):
    L = []
    files = sorted(glob.glob(os.path.join(RAW, "graph_mla_*.json")))
    if not files:
        return L
    L.append("#### C4 spot check with the CUDA-graph timers (per-call us; ratio = baseline / TileLang)\n")
    L.append("| shape | timer | TileLang | FlashInfer fa2 | Triton | FlashInfer XQA bf16 | FI / TL | Triton / TL |")
    L.append("|---|---|---|---|---|---|---|---|")
    res = {}
    for p in files:
        d = load(p)
        tag = d["tag"]
        eager = load(os.path.join(RAW, tag.replace("mla_", "mla_") + ".json"))["cobench"]["summary"]
        for timer, t in (("eager flush", {n: v["median_us"] for n, v in eager.items()}),
                         ("graph-flush", {n: v["per_call_median_us"] for n, v in d["graph_flush"]["summary"].items()}),
                         ("steady", {n: v["per_call_us"] for n, v in d["steady"]["summary"].items()})):
            tl = t.get("tl_adapted")
            res.setdefault(tag, {})[timer] = {"t": t, "fi_over_tl": ratio(t.get("flashinfer_mla_fa2"), tl),
                                              "triton_over_tl": ratio(t.get("triton_mla"), tl)}
            L.append(f"| {tag} | {timer} | {fmt(tl, 2)} | {fmt(t.get('flashinfer_mla_fa2'), 2)} | {fmt(t.get('triton_mla'), 2)} | "
                     f"{fmt(t.get('flashinfer_xqa_bf16'), 2)} | {fmt(ratio(t.get('flashinfer_mla_fa2'), tl), 3)} | "
                     f"{fmt(ratio(t.get('triton_mla'), tl), 2)} |")
    out["mla_graph"] = res
    return L


# ------------------------------------------------------------------------------------------------ C4
def mla_section(claims, stat, out):
    L = []
    cr = claims["mla_ratios"]
    rows, per = {}, {}
    for p in sorted(glob.glob(os.path.join(RAW, "mla_b*_s*.json"))):
        if p.endswith("_compile.json"):
            continue
        d = load(p)
        b, Lc = d["shape"]["batch"], d["shape"]["seqlen"]
        sm = d["cobench"]["summary"]
        rows[(b, Lc)] = {
            "t": {n: sm[n][stat] for n in sm}, "flops": d["shape"]["flops"], "bytes": d["shape"]["bytes"],
            "torch_up": ((d["variants"].get("torch_mla") or {}).get("upstream") or {}).get("triton_do_bench_us"),
            "torch_check": ((d["variants"].get("torch_mla") or {}).get("check") or {}).get("ok"),
            "up": {n: up_time(d, n) for n in sm},
            "clock": {n: sm[n].get("clock_mhz_median") for n in sm}, "cv": {n: sm[n].get("cv") for n in sm},
            "power_w": ((d["cobench"].get("nvml") or {}).get("power_w") or {}).get("mean"),
            "power_max_w": ((d["cobench"].get("nvml") or {}).get("power_w") or {}).get("max"),
            "guard": d["cobench"].get("guard"), "fi": d.get("flashinfer"), "triton": d.get("triton_mla"),
            "tl_adapt": d.get("tl_adapt"), "tl_shipped": d.get("tl_shipped"), "faithful": d.get("faithful"),
            "checks": {n: (x.get("check") or {}).get("max_abs") for n, x in d["variants"].items()},
            "errors": {n: x.get("error") for n, x in d["variants"].items() if x.get("error")}}
    for (b, Lc), r in sorted(rows.items()):
        t = r["t"]
        tl = t.get("tl_adapted")
        c = cr[str(b)][str(Lc)]
        m = {"flashinfer_fa2": ratio(t.get("flashinfer_mla_fa2"), tl),
             "flashinfer_xqa_bf16": ratio(t.get("flashinfer_xqa_bf16"), tl),
             "flashinfer_cutile": ratio(t.get("flashinfer_mla_cutile"), tl), "triton": ratio(t.get("triton_mla"), tl),
             "torch_upstream_timers": ratio(r["torch_up"], r["up"].get("tl_adapted")),
             "tl_split2_vs_split1": ratio(t.get("tl_adapted_split2"), tl), "tl_split4_vs_split1": ratio(t.get("tl_adapted_split4"), tl)}
        m_up = {"flashinfer_fa2": ratio(r["up"].get("flashinfer_mla_fa2"), r["up"].get("tl_adapted")),
                "triton": ratio(r["up"].get("triton_mla"), r["up"].get("tl_adapted"))}
        cc = {"flashinfer": c["TileLang_over_FlashInfer"], "triton": c["TileLang_over_Triton"],
              "flashmla": c["TileLang_over_FlashMLA"], "torch_paper": 1075.9}
        tfl = {n: r["flops"] / (x * 1e-6) / 1e12 for n, x in t.items()}
        gbs = {n: r["bytes"] / (x * 1e-6) / 1e9 for n, x in t.items()}
        per[f"b{b}_L{Lc}"] = {"measured": m, "measured_upstream_timers": m_up, "claimed": cc, "tflops": tfl, "gbps": gbs,
                              "verdict_flashinfer": verdict(m["flashinfer_fa2"], cc["flashinfer"]),
                              "verdict_triton": verdict(m["triton"], cc["triton"]),
                              "verdict_torch": verdict(m["torch_upstream_timers"], cc["torch_paper"]),
                              "verdict_flashinfer_upstream_timers": verdict(m_up["flashinfer_fa2"], cc["flashinfer"]),
                              "verdict_triton_upstream_timers": verdict(m_up["triton"], cc["triton"])}
    out["mla"] = {"rows": {f"b{b}_L{Lc}": r for (b, Lc), r in rows.items()}, "per_shape": per}

    L.append(f"#### C4 times (cobench clean-flush {stat.replace('_us', '')}, us; TFLOPS with benchmark_mla.py's FLOP count)\n")
    L.append("| batch | KV ctx | TileLang adapted | TL split 2 | TL split 4 | Triton | FlashInfer fa2 | FlashInfer XQA (bf16) | Torch (upstream timer) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for (b, Lc), r in sorted(rows.items()):
        t, tf = r["t"], per[f"b{b}_L{Lc}"]["tflops"]
        cell = lambda n: f"{fmt(t.get(n))} ({fmt(tf.get(n), 0)})" if t.get(n) else "–"  # noqa: E731
        L.append(f"| {b} | {Lc} | {cell('tl_adapted')} | {cell('tl_adapted_split2')} | {cell('tl_adapted_split4')} | "
                 f"{cell('triton_mla')} | {cell('flashinfer_mla_fa2')} | {cell('flashinfer_xqa_bf16')} | "
                 f"{fmt(r['torch_up'], 0)} |")
    L.append("")
    L.append("#### C4 ratios (baseline / TileLang adapted) and verdicts\n")
    L.append("| batch | KV ctx | FlashInfer (fa2): measured / claimed / verdict | FlashInfer XQA bf16 | Triton: measured / claimed / verdict | Torch (upstream timers) | FlashMLA |")
    L.append("|---|---|---|---|---|---|---|")
    agg = {k: [] for k in ("fi", "c_fi", "x", "tr", "c_tr", "to", "c_fm", "u_fi", "u_tr")}
    for key, p in per.items():
        m, c = p["measured"], p["claimed"]
        b, Lc = key[1:].split("_L")
        tri_cell = (f"{fmt(m['triton'], 2)} / {c['triton']:.1f} / {p['verdict_triton']}" if m["triton"] else
                    f"faults (int32 offset overflow in the kernel) / {c['triton']:.1f} / not testable")
        L.append(f"| {b} | {Lc} | {fmt(m['flashinfer_fa2'], 2)} / {c['flashinfer']:.2f} / {p['verdict_flashinfer']} | "
                 f"{fmt(m['flashinfer_xqa_bf16'], 2)} | {tri_cell} | "
                 f"{fmt(m['torch_upstream_timers'], 0)} | not testable (claimed TL/FlashMLA {c['flashmla']:.2f}) |")
        for k, x in (("fi", m["flashinfer_fa2"]), ("c_fi", c["flashinfer"] if m["flashinfer_fa2"] else None),
                     ("x", m["flashinfer_xqa_bf16"]), ("tr", m["triton"]),
                     ("c_tr", c["triton"] if m["triton"] else None),        # claims over the measured shapes only
                     ("to", m["torch_upstream_timers"]), ("c_fm", c["flashmla"]),
                     ("u_fi", p["measured_upstream_timers"]["flashinfer_fa2"]),
                     ("u_tr", p["measured_upstream_timers"]["triton"])):
            agg[k].append(x)
    g = {k: gmean(v) for k, v in agg.items()}
    L.append(f"| **geomean** | | **{fmt(g['fi'], 2)} / {fmt(g['c_fi'], 2)} / {verdict(g['fi'], g['c_fi'])}** | {fmt(g['x'], 2)} | "
             f"**{fmt(g['tr'], 2)} / {fmt(g['c_tr'], 2)} / {verdict(g['tr'], g['c_tr'])}** | {fmt(g['to'], 0)} (paper 1075.9) | "
             f"not testable |")
    out["mla"]["geomean"] = g
    L.append("")
    L.append("#### C4 with the upstream timers (benchmark_mla.py: tilelang do_bench for TileLang, triton do_bench otherwise)\n")
    L.append("| batch | KV ctx | TileLang us | FlashInfer fa2 us | Triton us | FlashInfer ratio / verdict | Triton ratio / verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for (b, Lc), r in sorted(rows.items()):
        p = per[f"b{b}_L{Lc}"]
        mu = p["measured_upstream_timers"]
        L.append(f"| {b} | {Lc} | {fmt(r['up'].get('tl_adapted'))} | {fmt(r['up'].get('flashinfer_mla_fa2'))} | "
                 f"{fmt(r['up'].get('triton_mla'))} | {fmt(mu['flashinfer_fa2'], 2)} / {p['verdict_flashinfer_upstream_timers']} | "
                 f"{fmt(mu['triton'], 2)} / {p['verdict_triton_upstream_timers']} |")
    L.append(f"| **geomean** | | | | | {fmt(g['u_fi'], 2)} / {verdict(g['u_fi'], g['c_fi'])} | "
             f"{fmt(g['u_tr'], 2)} / {verdict(g['u_tr'], g['c_tr'])} |")
    L.append("")
    L.append("#### C4 SM clock (ClockProbe median, MHz), power (NVML mean / max W), DRAM throughput of TileLang\n")
    L.append("| batch | KV ctx | TileLang MHz | FlashInfer MHz | Triton MHz | power mean / max | TileLang GB/s | FlashInfer GB/s | TileLang CV |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for (b, Lc), r in sorted(rows.items()):
        p = per[f"b{b}_L{Lc}"]
        ck = r["clock"]
        L.append(f"| {b} | {Lc} | {fmt(ck.get('tl_adapted'), 0)} | {fmt(ck.get('flashinfer_mla_fa2'), 0)} | "
                 f"{fmt(ck.get('triton_mla'), 0)} | {fmt(r['power_w'], 0)} / {fmt(r['power_max_w'], 0)} | "
                 f"{fmt(p['gbps'].get('tl_adapted'), 0)} | {fmt(p['gbps'].get('flashinfer_mla_fa2'), 0)} | "
                 f"{fmt(r['cv'].get('tl_adapted'), 3)} |")
    return L


# ------------------------------------------------------------------------------------------------ sink
def sink_section(stat, out):
    L = []
    files = sorted(glob.glob(os.path.join(RAW, "sink_s*_d*.json")))
    if not files:
        return L
    L.append(f"#### Attention sink (GQA fwd, bf16; cobench {stat.replace('_us', '')}, us)\n")
    L.append("| seq | dim | TileLang fixed cfg us | TileLang autotuned us (TFLOPS) | Triton us (TFLOPS) | speedup measured (tuned / fixed) | claimed (H800) | verdict (tuned) | upstream-timer speedup (tuned) | tuned config (block_M, block_N, stages, threads) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    per = {}
    files.sort(key=lambda p: (load(p)["shape"]["dim"], load(p)["shape"]["seq"]))
    for p in files:
        d = load(p)
        sm = d["cobench"]["summary"]
        t = {n: sm[n][stat] for n in sm}
        sh = d["shape"]
        cl = d["readme_claim"]["speedup"]
        m_t = ratio(t.get("triton"), t.get("tl_tuned"))
        m_f = ratio(t.get("triton"), t.get("tl_fixed"))
        up = {n: d["variants"][n].get("upstream_tl_do_bench_us") for n in t}
        m_up = ratio(up.get("triton"), up.get("tl_tuned"))
        cfg = (d["variants"].get("tl_tuned") or {}).get("config")
        per[f"s{sh['seq']}_d{sh['dim']}"] = {"t": t, "measured_tuned": m_t, "measured_fixed": m_f, "claimed": cl,
                                             "upstream": m_up, "verdict": verdict(m_t, cl), "tuned_config": cfg,
                                             "fixed": d["variants"].get("tl_fixed", {}).get("skipped")}
        fl = 2 * (2.0 * sh["batch"] * sh["heads"] * sh["seq"] * sh["seq"] * sh["dim"] * 0.5)   # benchmark's count
        tf = {n: fl / (x * 1e-6) / 1e12 for n, x in t.items()}
        per[f"s{sh['seq']}_d{sh['dim']}"]["tflops"] = tf
        cfgs = f"{cfg['block_M']}, {cfg['block_N']}, {cfg['num_stages']}, {cfg['threads']}" if cfg else "–"
        L.append(f"| {sh['seq']} | {sh['dim']} | {fmt(t.get('tl_fixed')) if t.get('tl_fixed') else 'unlaunchable (162 KB)'} | "
                 f"{fmt(t.get('tl_tuned'))} ({fmt(tf.get('tl_tuned'), 0)}) | {fmt(t.get('triton'))} ({fmt(tf.get('triton'), 0)}) | "
                 f"{fmt(m_t, 2)} / {fmt(m_f, 2)} | {cl:.2f} | {verdict(m_t, cl)} | {fmt(m_up, 2)} | {cfgs} |")
    g = {"tuned": gmean([p_["measured_tuned"] for p_ in per.values()]), "claimed": gmean([p_["claimed"] for p_ in per.values()])}
    L.append(f"| **geomean** | | | | | {fmt(g['tuned'], 2)} | {fmt(g['claimed'], 2)} | {verdict(g['tuned'], g['claimed'])} | | |")
    out["sink"] = per
    return L


def main():
    stat = sys.argv[1] if len(sys.argv) > 1 else "median_us"
    claims = load(os.path.join(C.RESULTS, "claims_digitized.json"))
    out = {"statistic": stat}
    L = (fa_section(claims, stat, out) + [""] + fa_graph_section(claims, out) + [""] + mla_section(claims, stat, out)
         + [""] + mla_graph_section(out) + [""] + sink_section(stat, out))
    print("\n".join(L))
    C.save_json(out, os.path.join(C.RESULTS, f"summary_{stat}.json"))


if __name__ == "__main__":
    main()
