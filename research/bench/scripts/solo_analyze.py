#!/usr/bin/env python
"""Analysis of the P1-S solo profile (research/bench/scripts/solo_profile.py).

Reads <results>/points/*.json through cotile.catalog, writes <results>/summary.json
(derived data: solo best vs references, persistent penalty, budget effect, C_lib,
pairing tables, measurement-quality stats) and prints the README tables (markdown).

    python research/bench/scripts/solo_analyze.py [--out DIR] [--md tables.md]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

from cotile import catalog  # noqa: E402

FULL = catalog.FULL_SMS
FLAG = 1.10  # best TileLang slower than the best reference by > 10%


def r(x, n=4):
    if x is None:
        return None
    return float(f"{x:.{n}g}")


def pct(x):
    return "–" if x is None else f"{100 * x:+.1f}%"


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else None


def rate_str(e, t_us):
    w = catalog.work_min(e.op, e.shape)
    if e.op == "gemm":
        return f"{w['flops'] / (t_us * 1e-6) / 1e12:.0f} TF/s"
    return f"{w['bytes_min'] / (t_us * 1e-6) / 1e9:.0f} GB/s"


def s2_best_vs_ref(cat) -> tuple[list, list[str]]:
    rows, md = [], []
    md.append("| op | shape | configs | best TileLang (grid) | µs | rate | MHz | best persistent µs | reference µs (MHz) | TL / ref | flag |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in cat:
        sb = e.solo_best
        if sb is None:
            continue
        m = sb.meas[("grid", FULL)]
        sp = e.solo_best_persistent
        refs = {n: v[FULL] for n, v in e.refs.items() if FULL in v}
        rb = e.ref_best(FULL)  # fastest reference that passed the fp32-reference check
        ratio = m.median / rb[1].median if rb else None
        row = {
            "op": e.op, "shape": e.tag, "n_configs": len(e.configs), "best": sb.tag, "best_us": m.median,
            "best_clk": m.clock_mhz, "best_cycles": m.cycles, "best_persistent": sp.tag if sp else None,
            "best_persistent_us": sp.time("persistent") if sp else None,
            "refs": {n: {"us": v.median, "clk": v.clock_mhz, "cycles": v.cycles} for n, v in refs.items()},
            "best_ref": rb[0] if rb else None, "tl_over_ref": ratio, "flag_gt10pct": bool(ratio and ratio > FLAG),
            "tl_over_ref_cycles": (m.cycles / rb[1].cycles) if rb and m.cycles and rb[1].cycles else None,
            "ref_failed_check": [n for n in refs if not e.ref_check.get(n, {}).get("ok", True)],
        }
        rows.append(row)
        ref_s = ", ".join(f"{n}{'†' if n in row['ref_failed_check'] else ''} {v.median:.1f} ({v.clock_mhz:.0f})"
                          for n, v in sorted(refs.items(), key=lambda x: x[1].median))
        md.append(
            f"| {e.op} | {e.tag} | {len(e.configs)} | `{sb.tag}` | {m.median:.1f} | {rate_str(e, m.median)} | {m.clock_mhz:.0f} | "
            f"{row['best_persistent_us']:.1f} | {ref_s} | {'–' if ratio is None else f'{ratio:.3f}'} | "
            f"{'**>10%**' if row['flag_gt10pct'] else ''} |"
        )
    return rows, md


def s3_persistent(cat) -> tuple[dict, list[str]]:
    out, md = {}, []
    md.append("| op | configs×shapes | median t_pers/t_grid | p10 | p90 | max | share > 1.05 | share < 0.95 | solo-best config: pers/grid (median, max) | best pers / best grid (median, max) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for op in cat.ops():
        ratios, sb_r, best_r, occ_drop = [], [], [], 0
        worst = None
        n_cfg = 0
        for e in (x for x in cat if x.op == op):
            pp = e.persistent_penalty()
            ratios += list(pp.values())
            for tag, v in pp.items():
                if worst is None or v > worst[0]:
                    worst = (v, e.tag, tag)
            for c in e.configs.values():
                n_cfg += 1
                g, p = c.sig.get("grid", {}), c.sig.get("persistent", {})
                if g.get("ctas_sm") and p.get("ctas_sm") and p["ctas_sm"] < g["ctas_sm"]:
                    occ_drop += 1
            sb = e.solo_best
            if sb and sb.tag in pp:
                sb_r.append(pp[sb.tag])
            if e.solo_best and e.solo_best_persistent:
                best_r.append(e.solo_best_persistent.time("persistent") / e.solo_best.time("grid"))
        a = np.array(ratios)
        out[op] = {
            "n": len(a), "median": r(q(a, 50)), "p10": r(q(a, 10)), "p90": r(q(a, 90)), "max": r(a.max()) if len(a) else None,
            "worst": worst, "share_gt_1.05": r(float((a > 1.05).mean())), "share_lt_0.95": r(float((a < 0.95).mean())),
            "solo_best_cfg_ratio": {"median": r(q(sb_r, 50)), "max": r(max(sb_r)) if sb_r else None},
            "best_pers_over_best_grid": {"median": r(q(best_r, 50)), "max": r(max(best_r)) if best_r else None, "min": r(min(best_r)) if best_r else None},
            "configs_with_lower_occupancy_persistent": occ_drop, "n_configs": n_cfg,
        }
        o = out[op]
        md.append(
            f"| {op} | {o['n']} | {o['median']:.3f} | {o['p10']:.3f} | {o['p90']:.3f} | {o['max']:.3f} | {100 * o['share_gt_1.05']:.0f}% | "
            f"{100 * o['share_lt_0.95']:.0f}% | {o['solo_best_cfg_ratio']['median']:.3f}, {o['solo_best_cfg_ratio']['max']:.3f} | "
            f"{o['best_pers_over_best_grid']['median']:.3f}, {o['best_pers_over_best_grid']['max']:.3f} |"
        )
    return out, md


def s4_budgets(cat) -> tuple[dict, list[str]]:
    out = {"per_shape": [], "per_op": {}}
    md = ["| op | shape | best @188 | best @94 (loss of @188-best) | best @48 (loss of @188-best) | best-ref @94 / @48 vs TL budget-best |",
          "|---|---|---|---|---|---|"]
    for e in cat:
        sb = e.solo_best
        if sb is None:
            continue
        row = {"op": e.op, "shape": e.tag, "best188": sb.tag}
        cells = []
        for sms in e.budgets:
            if sms == FULL:
                continue
            b = e.budget_best(sms)
            loss = e.budget_loss(sms)
            same = b.tag == sb.tag
            rb = e.ref_best(sms)
            row[f"best{sms}"] = b.tag
            row[f"loss{sms}"] = loss
            row[f"same{sms}"] = same
            row[f"t_best{sms}"] = b.time("grid", sms)
            row[f"ref_over_tl{sms}"] = rb[1].median / b.time("grid", sms) if rb else None
            cells.append(("same" if same else f"`{b.tag}`") + f" ({pct(loss)})")
        out["per_shape"].append(row)
        refc = ", ".join(f"{row.get(f'ref_over_tl{s}', 0):.2f}" for s in e.budgets if s != FULL)
        md.append(f"| {e.op} | {e.tag} | `{sb.tag}` | {cells[0]} | {cells[1] if len(cells) > 1 else ''} | {refc} |")
    for op in cat.ops():
        rows = [x for x in out["per_shape"] if x["op"] == op]
        d = {}
        for sms in sorted({int(k[4:]) for x in rows for k in x if k.startswith("loss")}, reverse=True):
            losses = [x[f"loss{sms}"] for x in rows if x.get(f"loss{sms}") is not None]
            diff = [not x[f"same{sms}"] for x in rows if f"same{sms}" in x]
            d[str(sms)] = {"n_shapes": len(losses), "best_differs": int(sum(diff)), "loss_median": r(q(losses, 50)),
                           "loss_max": r(max(losses)) if losses else None, "n_loss_gt_2pct": int(sum(1 for x in losses if x > 0.02)),
                           "n_loss_gt_5pct": int(sum(1 for x in losses if x > 0.05))}
        out["per_op"][op] = d
    return out, md


def s5_clib(cat) -> tuple[dict, list[str]]:
    out = {}
    md = ["| op | shape | configs | Pareto (grid) | C_lib = Pareto ∪ budget-best | C_lib (persistent) | solo-best E/call (µJ) | solo-best P_kernel (W) |",
          "|---|---|---|---|---|---|---|---|"]
    for e in cat:
        sb = e.solo_best
        m = sb.meas[("grid", FULL)] if sb else None
        d = {"n_configs": len(e.configs), "pareto": len(e.pareto("grid")), "c_lib": len(e.c_lib),
             "c_lib_persistent": len(e.c_lib_persistent), "c_lib_tags": e.c_lib_reasons,
             "solo_best_e_kernel_uj": m.e_kernel_uj if m else None, "solo_best_p_kernel_w": m.raw.get("P_kernel_w") if m else None,
             "solo_best_work": sb.work if sb else None}
        out[f"{e.op}:{e.tag}"] = d
        md.append(f"| {e.op} | {e.tag} | {d['n_configs']} | {d['pareto']} | {d['c_lib']} | {d['c_lib_persistent']} | "
                  f"{d['solo_best_e_kernel_uj']:.0f} | {d['solo_best_p_kernel_w']:.0f} |")
    return out, md


def s7_pairs(cat) -> tuple[dict, list[str]]:
    out, md = {}, []
    for a, b in (("gemm", "gqa_decode"), ("gemm", "rmsnorm")):
        rows = cat.pairs(a, b)
        out[f"{a}x{b}"] = [{k: (r(v, 5) if isinstance(v, float) else v) for k, v in x.items()} for x in rows]
        md.append(f"\n**{a} × {b}**: {len(rows)} shape pairs with t_a/t_b in [0.25, 4] "
                  f"(of {len(cat.shapes(a)) * len(cat.shapes(b))}); sorted by the speedup bound.\n")
        md.append("| A | B | t_A µs | t_B µs | t_A/t_B | T_serial | LB_tc | LB_dram | LB_power (flush) | LB_power (steady) | binding | T_serial / max(LB, max solo) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in rows:
            md.append(f"| {x['a']} | {x['b']} | {x['t_a']:.0f} | {x['t_b']:.0f} | {x['ratio']:.2f} | {x['t_serial']:.0f} | "
                      f"{x['lb_tc']:.0f} | {x['lb_dram']:.0f} | {x['lb_power']:.0f} | {x['lb_power_ss']:.0f} | {x['lb_binding']} | "
                      f"{x['speedup_bound']:.2f} |")
    return out, md


def quality(cat) -> tuple[dict, list[str]]:
    cvs, spreads, halves, clk, cont, n = [], [], [], [], 0, 0
    cv_bad = []
    for e in cat:
        for c in e.configs.values():
            for (b, s), m in c.meas.items():
                n += 1
                cvs.append(m.cv)
                spreads.append(m.raw.get("spread") or 0)
                if m.raw.get("half_ratio"):
                    halves.append(m.raw["half_ratio"])
                if m.raw.get("contaminated"):
                    cont += 1
                if m.cv > 0.02:
                    cv_bad.append((e.op, e.tag, c.tag, b, s, m.median, m.cv))
    drift = []  # reference re-measured at the end of its batch / at the start
    for e in cat:
        for rn, by in e.refs_end.items():
            for sms, m in by.items():
                if sms in e.refs.get(rn, {}):
                    drift.append((e.op, e.tag, rn, sms, m.median / e.refs[rn][sms].median))
    dr = np.array([x[4] for x in drift]) if drift else np.array([1.0])
    cvs, spreads, halves = np.array(cvs), np.array(spreads), np.array(halves)
    d = {"n_points": n, "cv_median": r(q(cvs, 50)), "cv_p90": r(q(cvs, 90)), "n_cv_gt_2pct": int((cvs > 0.02).sum()),
         "spread_median": r(q(spreads, 50)), "spread_p90": r(q(spreads, 90)),
         "half_ratio_p1": r(q(halves, 1)), "half_ratio_p50": r(q(halves, 50)), "half_ratio_p99": r(q(halves, 99)),
         "n_half_ratio_off_1pct": int((np.abs(halves - 1) > 0.01).sum()), "n_contaminated": cont,
         "cv_gt_2pct_by_op": {op: sum(1 for x in cv_bad if x[0] == op) for op in cat.ops()},
         "cv_gt_2pct_short": sum(1 for x in cv_bad if x[5] < 150),
         "ref_end_over_start": {"n": len(drift), "p1": r(q(dr, 1)), "p50": r(q(dr, 50)), "p99": r(q(dr, 99)),
                                "min": r(dr.min()), "max": r(dr.max()), "n_off_1pct": int((np.abs(dr - 1) > 0.01).sum()),
                                "worst": sorted(drift, key=lambda x: -abs(x[4] - 1))[:5]}}
    md = [f"- points: {n}; CV median {100 * d['cv_median']:.2f}%, p90 {100 * d['cv_p90']:.2f}%; CV > 2%: {d['n_cv_gt_2pct']} "
          f"({d['cv_gt_2pct_short']} of them < 150 µs) by op {d['cv_gt_2pct_by_op']}",
          f"- (p90−p10)/median: median {100 * d['spread_median']:.2f}%, p90 {100 * d['spread_p90']:.2f}%",
          f"- drift inside the timed window (median of 2nd half / 1st half): p1 {d['half_ratio_p1']}, p50 {d['half_ratio_p50']}, "
          f"p99 {d['half_ratio_p99']}; |ratio−1| > 1%: {d['n_half_ratio_off_1pct']} points",
          f"- reference re-measured at the end of its batch / start: {d['ref_end_over_start']}",
          f"- contaminated (foreign GPU process during the point, after re-measurement): {cont}"]
    return d, md


def studies(out_dir) -> tuple[dict, list[str]]:
    """Summaries of <results>/studies/*.json (those that exist)."""
    sd = os.path.join(out_dir, "studies")
    res, md = {}, []

    def load(name):
        try:
            with open(os.path.join(sd, name)) as f:
                return json.load(f)
        except OSError:
            return None

    w = load("warmup.json")
    if w:
        md += ["\n**Per-point warmup study** (median vs the steady-state reference = pred 'self' with W >= 1 s; "
               "|half_ratio - 1| = drift inside the timed window; mean of 2 reps):\n",
               "| point | predecessor | " + " | ".join(f"W={x}" for x in w["W"]) + " |",
               "|---|---|" + "---|" * len(w["W"])]
        res["warmup"] = []
        for t in w["targets"]:
            rows = t["rows"]
            ref = float(np.median([x["median"] for x in rows if x["pred"] == "self" and x["W"] >= 1.0]))
            for pred in w["preds"]:
                cells = []
                for W in w["W"]:
                    rr = [x for x in rows if x["pred"] == pred and x["W"] == W]
                    dm = float(np.mean([x["median"] / ref - 1 for x in rr]))
                    dh = float(np.mean([abs(x["half_ratio"] - 1) for x in rr if x["half_ratio"]]))
                    cells.append(f"{100 * dm:+.2f}% / {100 * dh:.2f}%")
                    res["warmup"].append({"point": f"{t['op']}:{t['shape']}@{t['sms']}", "pred": pred, "W": W, "d_median": dm, "abs_half": dh})
                md.append(f"| {t['op']} {t['shape']} @{t['sms']} | {pred} | " + " | ".join(cells) + " |")
    pm = load("persistent_modes.json")
    if pm:
        md += ["\n**Grid vs persistent under four regimes** (t_persistent / t_grid; cycles ratio in parentheses):\n",
               "| op | shape | config | flush (write) | flush (read) | hot | graph | grid µs flush / hot / graph |",
               "|---|---|---|---|---|---|---|---|"]
        res["persistent_modes"] = []
        for r_ in pm["rows"]:
            m = r_["modes"]
            cells = [f"{m[k]['ratio_time']:.3f} ({m[k].get('ratio_cycles', float('nan')):.3f})" for k in ("flush", "flush_read", "hot", "graph")]
            md.append(f"| {r_['op']} | {r_['shape']} | `{r_['cfg']}` | " + " | ".join(cells) +
                      f" | {m['flush']['grid']['median']:.1f} / {m['hot']['grid']['median']:.1f} / {m['graph']['grid']['median']:.1f} |")
            res["persistent_modes"].append({"op": r_["op"], "shape": r_["shape"], "cfg": r_["cfg"],
                                            **{k: m[k]["ratio_time"] for k in ("flush", "flush_read", "hot", "graph")}})
    fb = load("flush_bias.json")
    if fb:
        md += ["\n**cobench write-flush vs write+read flush** (same 2xL2 buffer written, then read: evicted *and* clean), "
               "solo-best TileLang config and best reference, 188 SMs (ratio = t_clean / t_write; delta = t_write − t_clean):\n",
               "| op | shape | kernel | write µs | write+read µs | ratio | delta µs |", "|---|---|---|---|---|---|---|"]
        res["flush_bias"] = []
        for x in fb["rows"]:
            md.append(f"| {x['op']} | {x['shape']} | `{x['name']}` | {x['write']['median']:.1f} | {x['write+read']['median']:.1f} | "
                      f"{x['clean_over_write']:.3f} | {x['delta_us']:.1f} |")
            res["flush_bias"].append({k: x[k] for k in ("op", "shape", "name", "clean_over_write", "delta_us")})
        for op in sorted({x["op"] for x in fb["rows"]}):
            rr = [x["clean_over_write"] for x in fb["rows"] if x["op"] == op]
            dd = [x["delta_us"] for x in fb["rows"] if x["op"] == op]
            md.append(f"\n- {op}: clean/write median {np.median(rr):.3f} (min {min(rr):.3f}, max {max(rr):.3f}); "
                      f"delta median {np.median(dd):.1f} µs (max {max(dd):.1f})")
    fk = load("flush_kinds.json")
    if fk:
        md += ["\n**Flush kinds** (mean of 2 reps, µs):\n", "| op | shape | config | write (cobench) | read | write+read |",
               "|---|---|---|---|---|---|"]
        for x in fk["rows"]:
            md.append(f"| {x['op']} | {x['shape']} | `{x['cfg']}` | {x['write']:.1f} | {x['read']:.1f} | {x['write+read']:.1f} |")
    rc = load("recheck.json")
    if rc:
        md += ["\n**Re-measurement at the end of the sweep** (new / original median):\n",
               "| point | original µs (MHz) | re-measured µs (MHz) | ratio |", "|---|---|---|---|"]
        for x in rc["rows"]:
            md.append(f"| {x['op']} {x['shape']} `{x['cfg']}` @{x['sms']} (rep {x['rep']}) | {x['old']:.1f} ({x['old_clk']:.0f}) | "
                      f"{x['new']:.1f} ({x['new_clk']:.0f}) | {x['ratio']:.4f} |")
        res["recheck"] = [x["ratio"] for x in rc["rows"]]
    en = load("energy.json")
    if en:
        md += ["\n**Energy per call: flush-mode subtraction vs back-to-back (cobench graph mode)**:\n",
               "| point | flush: t µs (MHz) | P_w | E_naive µJ | E_kernel µJ (P_kernel W) | graph: t µs (MHz) | P W | E = P·t µJ | E_kernel / E_graph |",
               "|---|---|---|---|---|---|---|---|---|"]
        res["energy"] = []
        for x in en["rows"]:
            f_, g = x["flush"], x["graph"]
            ratio = f_["E_kernel_uj"] / g["E_uj"] if f_.get("E_kernel_uj") and g.get("E_uj") else None
            res["energy"].append({"point": f"{x['op']}:{x['shape']}:{x['cfg']}", "e_kernel": f_.get("E_kernel_uj"), "e_graph": g.get("E_uj"), "ratio": ratio})
            md.append(f"| {x['op']} {x['shape']} `{x['cfg']}` | {f_['median']:.1f} ({f_['clk']:.0f}) | {f_['P_w']:.0f} | {f_['E_naive_uj']:.0f} | "
                      f"{f_['E_kernel_uj']:.0f} ({f_['P_kernel_w']:.0f}) | {g['median']:.1f} ({g['clk']:.0f}) | {g['P_w']:.0f} | {g['E_uj']:.0f} | "
                      f"{ratio:.3f} |")
    return res, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=catalog.DEFAULT_DIR)
    ap.add_argument("--md", default=None, help="also write the markdown tables to this file")
    args = ap.parse_args()
    cat = catalog.load(args.out)
    summary = catalog.summary_dict(cat)
    md = [f"measured rates (best over all points @188): {cat.rates()}", ""]
    s2, m2 = s2_best_vs_ref(cat)
    s3, m3 = s3_persistent(cat)
    s4, m4 = s4_budgets(cat)
    s5, m5 = s5_clib(cat)
    s7, m7 = s7_pairs(cat)
    qq, mq = quality(cat)
    st, mst = studies(args.out)
    summary.update({"best_vs_ref": s2, "persistent_penalty": s3, "budget_effect": s4, "c_lib_sizes": s5, "pairs": s7, "quality": qq,
                    "studies": st})
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, separators=(",", ":"))
    for title, m in (("S2 best TileLang vs reference", m2), ("S3 persistent penalty", m3), ("S4 SM budgets", m4),
                     ("S5 C_lib", m5), ("S7 pairing", m7), ("quality", mq), ("studies", mst)):
        md += [f"\n### {title}\n"] + m
    text = "\n".join(md)
    print(text)
    if args.md:
        with open(args.md, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
