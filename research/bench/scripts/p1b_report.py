"""P1 3x2 study, part B (B2-B4): complete 3x2 tables, derived-axis ablations, no-oracle
robustness, D1 interim readout, GPU time.

    python research/bench/scripts/p1b_report.py     # -> <OUT_B>/tables.json, <OUT_B>/tables.md

Every T comes from the pair's final interleaved steady run (stage F of part B), in which serial,
the solo runs, part A's four winners (re-measured), the derived and lib (B search) winners and
the ablations ran together. Cells:
  T[solo,c]    part A's solo-row winner of column c, re-measured in F
  T[lib,c]     min(part A's lib winner re-measured, the lib-row winner of the B search)
  T[derived,c] min over every variant of column c measured in F (C_derived includes C_lib)
  T_inter*     min(T[solo,inter], T[lib,inter], T[derived,inter])
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import p1_common as C
import p1b_study as B

COLS = ("inter", "intra")
AX_LABEL = {"gemm_nonlib": "GEMM cfg not in C_lib", "gemm_splitk": "split-K GEMM", "gemm_hint": "GEMM A/B evict_last",
            "decode_nonlib": "decode cfg not in C_lib", "decode_splitkv": "split-KV decode",
            "decode_hint": "decode K/V evict_first", "regcap": "register cap (min_blocks_per_sm=2)"}


def load(pair):
    return C.load(os.path.join(B.OUT_B, pair, "study.json"))


def solo_info(summ, shape, tag):
    """A1 clean-flush solo rank / slowdown of a config (hints: its base config)."""
    import re
    base = re.sub(r"_l2e[fl]$", "", tag)
    s = summ["shapes"][shape]
    rank = s["flush_rank"].get(base)
    t, tb = s["t188"].get(base), s["t188"].get(s["solo_best"])
    return {"tag": tag, "rank": rank, "n": len(s["flush_rank"]), "slowdown": (t / tb - 1) if t and tb else None,
            "in_c_lib": base in s["c_lib"]}


def knobs(d):
    if d["kind"] == "green":
        return f"green {d['n_a']}/{188 - d['n_a']}"
    if d["kind"] == "co":
        if d["binding"] == "cta_dyn":
            s = f"CTA-bind {d['num_ctas']} CTAs"
        else:
            s = f"SM-bind {'dyn' if d['binding'] == 'sm_dyn' else 'static'} {d['n_a']}/{188 - d['n_a']}"
        s += " +takeover" if d["takeover"] else ""
        s += f" chunk {d['chunk'][0]}/{d['chunk'][1]}" if d["binding"] != "sm_static" else ""
        s += f" mbps {d['mbps']}" if d.get("mbps", 1) != 1 else ""
        return s
    return d["kind"]


def axes_str(d):
    ax = d.get("axes") or {}
    on = [AX_LABEL[k] for k, v in ax.items() if v]
    return ", ".join(on) if on else "none (lib space)"


def vkey(d):
    """Identity of a variant: configs and knobs (not its row / anchor labels)."""
    return (d.get("kind"), d.get("a"), d.get("b"), d.get("binding"), d.get("n_a"), d.get("num_ctas"),
            d.get("takeover"), tuple(d.get("chunk") or ()), d.get("mbps", 1), bool(d.get("timing")))


def anchors_in_F(pair, D) -> dict:
    """{part-A cell: F variant name}: part A's final winners matched by configs and knobs (a
    variant can be the winner of two cells, e.g. solo and lib when C_lib selection kept the
    solo-best configs)."""
    import p1_study as PS
    A = C.load(os.path.join(C.OUT, pair, "study.json"))
    best = PS.winners(A)
    descs = {}
    for s in A["stages"].values():
        if isinstance(s, dict):
            descs.update(s.get("desc", {}))
    byk = {vkey(d): n for n, d in D.items()}
    out = {}
    for cell, (_, name, _) in best.items():
        d = descs.get(name)
        if d and vkey(d) in byk:
            out[cell] = byk[vkey(d)]
    return out


def pair_table(pair, summ):
    st = load(pair)
    if not st or "F" not in st["stages"]:
        return None
    F = st["stages"]["F"]
    S, D = F["steady"], F["desc"]
    ser = S["serial"]["t_iter_us"]
    cell = {}

    def entry(name, how):
        v, d = S[name], D[name]
        e = {"variant": name, "how": how, "T": v["t_iter_us"], "speedup": ser / v["t_iter_us"], "paired": v.get("paired"),
             "clock_mhz": v.get("clock_mhz"), "power_w": v.get("power_w"), "energy_mj": v.get("energy_mj_per_iter"),
             "kind": d["kind"], "knobs": knobs(d), "a": d.get("a"), "b": d.get("b"), "axes": axes_str(d),
             "row": d.get("row"), "ops": v.get("ops")}
        if d.get("a"):
            e["cfg_a"] = solo_info(summ, st["A"], d["a"])
        if d.get("b"):
            e["cfg_b"] = solo_info(summ, st["B"], d["b"])
        return e

    kinds = {"inter": "green", "intra": "co"}
    anc = anchors_in_F(pair, D)
    for col in COLS:
        if f"solo_{col}" in anc:
            cell[f"solo_{col}"] = entry(anc[f"solo_{col}"], "part A solo winner, re-measured")
        lib_c = [(S[anc[f"lib_{col}"]]["t_iter_us"], anc[f"lib_{col}"], "part A lib winner, re-measured")] if f"lib_{col}" in anc else []
        bl = F["best"].get(f"lib_B_{col}")
        if bl and bl[1] in S:
            lib_c.append((S[bl[1]]["t_iter_us"], bl[1], "lib (B search) winner"))
        if lib_c:
            t, n, how = min(lib_c)
            cell[f"lib_{col}"] = entry(n, how)
            cell[f"lib_{col}"]["alternatives"] = {h: {"variant": nn, "T": tt} for tt, nn, h in lib_c}
        cands = [(v["t_iter_us"], n) for n, v in S.items() if D[n]["kind"] == kinds[col] and not D[n].get("timing")]
        if cands:
            t, n = min(cands)
            cell[f"derived_{col}"] = entry(n, "best of every measured " + ("green" if col == "inter" else "CoKernel") + " variant")
            bd = F["best"].get(f"derived_{col}")
            if bd and bd[1] in S:
                cell[f"derived_{col}"]["search_winner"] = {"variant": bd[1], "T": S[bd[1]]["t_iter_us"],
                                                           "axes": axes_str(D[bd[1]])}
    T = {k: v["T"] for k, v in cell.items()}
    out = {"pair": pair, "A": st["A"], "B": st["B"], "T_serial": ser, "cells": cell, "T": T,
           "serial": {k: S["serial"].get(k) for k in ("clock_mhz", "power_w", "energy_mj_per_iter")},
           "solo_a": {k: S["solo_a"].get(k) for k in ("t_iter_us", "clock_mhz", "power_w")},
           "solo_b": {k: S["solo_b"].get(k) for k in ("t_iter_us", "clock_mhz", "power_w")}}
    inter = [T[k] for k in ("solo_inter", "lib_inter", "derived_inter") if k in T]
    if inter:
        out["T_inter_star"] = min(inter)
    r = {}
    for col in COLS:
        if f"lib_{col}" in T and f"derived_{col}" in T:
            r[f"lib_over_derived_{col}"] = T[f"lib_{col}"] / T[f"derived_{col}"]
        if f"solo_{col}" in T and f"lib_{col}" in T:
            r[f"solo_over_lib_{col}"] = T[f"solo_{col}"] / T[f"lib_{col}"]
    if "T_inter_star" in out:
        for k in ("derived_intra", "lib_intra"):
            if k in T:
                r[f"inter_star_over_{k}"] = out["T_inter_star"] / T[k]
    out["ratios"] = r
    # ablations of the derived search winners
    abl = {}
    for c, alts in F.get("ablations", {}).items():
        w = F["best"][c][1]
        if w not in S:
            continue
        tw = S[w]["t_iter_us"]
        abl[c] = {"winner": w, "T": tw, "axes": axes_str(D[w]), "knobs": knobs(D[w]),
                  "reverted": {ax: {"variant": n, "T": S[n]["t_iter_us"], "gain": S[n]["t_iter_us"] / tw - 1,
                                    "clock_mhz": S[n].get("clock_mhz"), "energy_mj": S[n].get("energy_mj_per_iter")}
                               for ax, n in alts.items() if n in S}}
    out["ablations"] = abl
    # decode K/V hint at identical knobs: every hinted variant of F whose unhinted twin (same
    # configs otherwise, same split / binding / chunk / takeover) ran in the same F run
    import re
    byk = {vkey(d): n for n, d in D.items() if n in S}
    hp = []
    for n, d in D.items():
        if n not in S or d.get("timing") or not (d.get("axes") or {}).get("decode_hint"):
            continue
        tw = dict(d, b=re.sub(r"_l2ef$", "", d["b"]))
        m = byk.get(vkey(tw))
        if m and m != n:
            hp.append({"hinted": n, "unhinted": m, "kind": d["kind"], "knobs": knobs(d), "gemm_hint": (d.get("axes") or {}).get("gemm_hint"),
                       "T_hinted": S[n]["t_iter_us"], "T_unhinted": S[m]["t_iter_us"], "gain": S[m]["t_iter_us"] / S[n]["t_iter_us"] - 1,
                       "E_hinted": S[n].get("energy_mj_per_iter"), "E_unhinted": S[m].get("energy_mj_per_iter")})
    out["hint_pairs"] = sorted(hp, key=lambda x: (x["kind"], x["knobs"]))
    # solo runs of the derived winners' configs (steady, full GPU)
    so = {}
    for k, n in F.get("solo", {}).items():
        if n in S:
            so[k] = {"variant": n, "T": S[n]["t_iter_us"], "clock_mhz": S[n].get("clock_mhz"), "power_w": S[n].get("power_w"),
                     "cfg": D[n].get("a") or D[n].get("b")}
    for k, v in so.items():
        side = "a" if k.endswith("_a") or "_a_" in k else "b"
        ref = out[f"solo_{side}"]["t_iter_us"]
        v["vs_solo_best"] = v["T"] / ref - 1
    out["solo_runs"] = so
    out["role_times"] = F.get("role_times")
    out["timing"] = F.get("timing")
    for c, tn in (F.get("timing") or {}).items():
        if tn in S:
            out["role_times"][c]["T_timing_build"] = S[tn]["t_iter_us"]
    # screening fidelity (B confirm stages)
    out["fidelity"] = {s: st["stages"][s].get("fidelity") for s in ("DI3", "DC3") if s in st["stages"]}
    out["dc1_cta_info"] = st["stages"].get("DC1", {}).get("cta_info")
    out["dc1_n_cta"] = len(st["stages"].get("DC1", {}).get("names_cta", []))
    out["failed"] = st.get("failed")
    out["stage_wall_s"] = st.get("stage_wall_s")
    out["compile_s"] = st.get("compile_s")
    if "R" in st["stages"]:
        R = st["stages"]["R"]
        out["robustness"] = {"summary": R["summary"], "rules": R["rules"], "splits": R["splits"],
                             "transfer_from_main": R.get("transfer_from_main")}
    return out


def gpu_time(pairs):
    out = {}
    for p in pairs:
        st = load(p)
        if not st:
            continue
        # completed stages (their last, complete run) + runs cut short by a GPU-sharing yield
        # + complete runs superseded by a later --redo of the same stage (stage_runs keeps
        # every run; stage_wall_s only the last one)
        runs = st.get("stage_runs", [])
        wall = sum((st.get("stage_wall_s") or {}).values())
        wall += sum(r["wall_s"] for r in runs if not r.get("done"))
        done = {}
        for r in runs:
            if r.get("done"):
                done.setdefault(r["stage"], []).append(r["wall_s"])
        superseded = sum(sum(w[:-1]) for w in done.values())
        wall += superseded
        waits = 0.0
        for s in st["stages"].values():
            if not isinstance(s, dict):
                continue
            for a in (s.get("meta", {}).get("guard", {}) or {}).get("attempts", []):
                waits += a.get("waited_s") or 0.0
            for g in (s.get("flush") or {}).get("_guard", []) or []:
                for a in (g or {}).get("attempts", []):
                    waits += a.get("waited_s") or 0.0
        out[p] = {"stage_wall_s": wall, "superseded_s": superseded, "guard_wait_s": waits, "compile_s": st.get("compile_s", 0.0),
                  "gpu_s": wall - waits - st.get("compile_s", 0.0), "yields": st.get("yields", [])}
    return out


def _f(x, fmt="{:.1f}"):
    return "–" if x is None else fmt.format(x)


def cfg_s(e, side):
    c = e.get(f"cfg_{side}")
    if not c:
        return "–"
    tag = c["tag"].replace("_k1", "").replace("_g8", "")
    sd = f"{c['slowdown'] * 100:+.1f}%" if c["slowdown"] is not None else "?"
    return f"`{tag}` (#{c['rank']}/{c['n']}, {sd}{', C_lib' if c['in_c_lib'] else ''})"


def markdown(tabs, gpu):
    L = ["## 3×2 tables (steady, stage F of part B)\n",
         "| pair | T_serial | T[solo,inter] | T[lib,inter] | T[derived,inter] | T[solo,intra] | T[lib,intra] | T[derived,intra] | "
         "lib/derived inter | lib/derived intra | T_inter*/T[derived,intra] | T_inter*/T[lib,intra] |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, t in tabs.items():
        if not t:
            continue
        T, s, r = t["T"], t["T_serial"], t["ratios"]
        c = lambda k: f"{T[k]:.1f} (×{s / T[k]:.3f})" if k in T else "–"  # noqa: E731
        L.append(f"| {p} | {s:.1f} | {c('solo_inter')} | {c('lib_inter')} | {c('derived_inter')} | {c('solo_intra')} | "
                 f"{c('lib_intra')} | {c('derived_intra')} | {_f(r.get('lib_over_derived_inter'), '{:.3f}')} | "
                 f"{_f(r.get('lib_over_derived_intra'), '{:.3f}')} | {_f(r.get('inter_star_over_derived_intra'), '{:.3f}')} | "
                 f"{_f(r.get('inter_star_over_lib_intra'), '{:.3f}')} |")
    for p, t in tabs.items():
        if not t:
            continue
        L.append(f"\n### {p}: GEMM {t['A']} × decode {t['B']}\n")
        L.append("| cell | variant | knobs | GEMM config (solo rank, slowdown) | decode config (solo rank, slowdown) | derived axes | "
                 "steady µs | × serial | MHz | W | mJ/iter |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        L.append(f"| serial | – | – | – | – | – | {t['T_serial']:.1f} | 1.000 | {_f(t['serial']['clock_mhz'], '{:.0f}')} | "
                 f"{_f(t['serial']['power_w'], '{:.0f}')} | {_f(t['serial']['energy_mj_per_iter'], '{:.0f}')} |")
        for k in ("solo_inter", "lib_inter", "derived_inter", "solo_intra", "lib_intra", "derived_intra"):
            e = t["cells"].get(k)
            if not e:
                continue
            L.append(f"| **T[{k.replace('_', ',')}]** | `{e['variant']}` | {e['knobs']} | {cfg_s(e, 'a')} | {cfg_s(e, 'b')} | "
                     f"{e['axes']} | {e['T']:.1f} | ×{e['speedup']:.3f} | {_f(e['clock_mhz'], '{:.0f}')} | "
                     f"{_f(e['power_w'], '{:.0f}')} | {_f(e['energy_mj'], '{:.0f}')} |")
        for k in ("lib_inter", "lib_intra"):
            e = t["cells"].get(k)
            if e and len(e.get("alternatives", {})) > 1:
                L.append(f"\n{k}: " + "; ".join(f"{h}: `{v['variant']}` {v['T']:.1f} µs" for h, v in e["alternatives"].items()))
        for k in ("derived_inter", "derived_intra"):
            e = t["cells"].get(k)
            if e and e.get("search_winner") and e["search_winner"]["variant"] != e["variant"]:
                w = e["search_winner"]
                L.append(f"\n{k}: the search winner `{w['variant']}` ({w['axes']}) re-measured at {w['T']:.1f} µs in F.")
        if t["ablations"]:
            L.append("\n**Ablations of the derived search winners** (same knobs, one derived axis reverted; gain = T_reverted / T_winner − 1)\n")
            L.append("| winner | axes | reverted axis | variant | µs | gain of the axis |")
            L.append("|---|---|---|---|---|---|")
            for c, a in t["ablations"].items():
                L.append(f"| {c} `{a['winner']}` ({a['knobs']}) | {a['axes']} | – | – | {a['T']:.1f} | – |")
                for ax, v in a["reverted"].items():
                    L.append(f"| | | {ax} | `{v['variant']}` | {v['T']:.1f} | {v['gain'] * 100:+.2f}% |")
        if t.get("hint_pairs"):
            L.append("\n**Decode K/V evict_first at identical knobs** (every hinted F variant whose unhinted twin ran in the same run; "
                     "gain = T_unhinted / T_hinted − 1, energy per iteration)\n")
            L.append("| mechanism | knobs | hinted | unhinted | µs hinted / unhinted | gain | mJ hinted / unhinted |")
            L.append("|---|---|---|---|---|---|---|")
            for h in t["hint_pairs"]:
                L.append(f"| {h['kind']} | {h['knobs']} | `{h['hinted']}` | `{h['unhinted']}` | {h['T_hinted']:.1f} / {h['T_unhinted']:.1f} | "
                         f"{h['gain'] * 100:+.2f}% | {_f(h['E_hinted'], '{:.0f}')} / {_f(h['E_unhinted'], '{:.0f}')} |")
        if t["solo_runs"]:
            L.append("\n**Solo (full GPU, steady) of the derived winners' configs** vs the solo-best of the op: " +
                     "; ".join(f"{k}: `{v['cfg']}` {v['T']:.1f} µs ({v['vs_solo_best'] * 100:+.1f}%)" for k, v in t["solo_runs"].items()))
        rt = t.get("role_times") or {}
        if rt:
            L.append("\n**Role completion times (timing builds, µs)**: " + "; ".join(
                f"{c}: T_A {v['T_A_us']:.0f}, T_B {v['T_B_us']:.0f}, CTAs {v['ctas_A']:.0f}/{v['ctas_B']:.0f}, steals {v['steal_A']:.0f}/{v['steal_B']:.0f}"
                for c, v in rt.items() if v.get("T_A_us") is not None))
    # robustness
    L.append("\n## B3 — no-oracle robustness (steady; speed-up vs serial of the same run)\n")
    L.append("| pair | mechanism | oracle (split) | R1 split: speed-up | R2 split: speed-up | worst of the 8-split sweep | "
             "regret R1 / R2 / worst | main's split: speed-up (regret) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p, t in tabs.items():
        if not t or "robustness" not in t:
            continue
        R = t["robustness"]
        for mech in ("green", "co"):
            m = R["summary"][mech]
            tr = (f"{m['transfer_split']}: ×{m['transfer']:.3f} ({m['regret_transfer']:.3f})"
                  if m.get("transfer") else ("–" if p != "main" else "(source)"))
            L.append(f"| {p} | {'green ctx' if mech == 'green' else 'CoKernel dyn+TO'} | ×{m['oracle']:.3f} ({m['oracle_split']}) | "
                     f"{R['rules']['R1']}: ×{_f(m['R1'], '{:.3f}')} | {R['rules']['R2']}: ×{_f(m['R2'], '{:.3f}')} | "
                     f"×{m['worst_sweep']:.3f} | {_f(m['regret_R1'], '{:.3f}')} / {_f(m['regret_R2'], '{:.3f}')} / "
                     f"{m['regret_worst']:.3f} | {tr} |")
    L.append("\n**Speed-up per GEMM share** (best candidate configs per split; rule / transfer splits included)\n")
    for p, t in tabs.items():
        if not t or "robustness" not in t:
            continue
        R = t["robustness"]
        sp = [int(x) for x in R["splits"]]
        L.append(f"\n{p}:\n")
        L.append("| mechanism | " + " | ".join(str(x) for x in sp) + " |")
        L.append("|---|" + "---|" * len(sp))
        for mech in ("green", "co"):
            cur = R["summary"][mech]["curve"]
            L.append(f"| {mech} | " + " | ".join(_f(cur.get(str(x)), "{:.3f}") for x in sp) + " |")
            bo = R["summary"][mech].get("budget_only", {})
            if bo:
                L.append(f"| {mech} (budget-best configs only) | " + " | ".join(_f(bo.get(str(x)), "{:.3f}") for x in sp) + " |")
    L.append("\n## GPU time\n\n| pair | GPU s (stage wall − guard waits − compile) |\n|---|---|")
    for p, v in gpu.items():
        sup = f" incl. {v['superseded_s']:.0f} s of superseded runs" if v.get("superseded_s") else ""
        L.append(f"| {p} | {v['gpu_s']:.0f} (wall {v['stage_wall_s']:.0f}{sup}, waits {v['guard_wait_s']:.0f}, compile {v['compile_s']:.0f}) |")
    L.append(f"| total | {sum(v['gpu_s'] for v in gpu.values()):.0f} s ({sum(v['gpu_s'] for v in gpu.values()) / 3600:.2f} h) |")
    return "\n".join(L)


def main():
    summ = C.load(os.path.join(C.SOLO_DIR, "summary.json"))
    tabs = {p: pair_table(p, summ) for p in C.PAIRS}
    gpu = gpu_time(C.PAIRS)
    out = {"pairs": tabs, "gpu_time": gpu}
    # D1 interim readout (plan §2.8, thresholds unchanged)
    d1 = {}
    for p, t in tabs.items():
        if not t:
            continue
        r = t["ratios"]
        d1[p] = {"inter_star_over_derived_intra": r.get("inter_star_over_derived_intra"),
                 "lib_over_derived_intra": r.get("lib_over_derived_intra"),
                 "inter_star_over_lib_intra": r.get("inter_star_over_lib_intra"),
                 "full": bool(r.get("inter_star_over_derived_intra", 0) >= 1.10 and r.get("lib_over_derived_intra", 0) >= 1.05),
                 "weak": bool(r.get("inter_star_over_lib_intra", 0) >= 1.10)}
    out["D1"] = d1
    C.save(os.path.join(B.OUT_B, "tables.json"), out)
    md = markdown(tabs, gpu)
    md += "\n\n## D1 interim readout (plan §2.8, thresholds unchanged)\n\n| pair | T_inter*/T[derived,intra] (≥1.10) | T[lib,intra]/T[derived,intra] (≥1.05) | full claim | T_inter*/T[lib,intra] (≥1.10) | weak claim |\n|---|---|---|---|---|---|\n"
    for p, d in d1.items():
        md += (f"| {p} | {_f(d['inter_star_over_derived_intra'], '{:.3f}')} | {_f(d['lib_over_derived_intra'], '{:.3f}')} | "
               f"{'yes' if d['full'] else 'no'} | {_f(d['inter_star_over_lib_intra'], '{:.3f}')} | {'yes' if d['weak'] else 'no'} |\n")
    with open(os.path.join(B.OUT_B, "tables.md"), "w") as f:
        f.write(md + "\n")
    print(json.dumps({p: {"T": t["T"], "ratios": t["ratios"]} if t else None for p, t in tabs.items()}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
