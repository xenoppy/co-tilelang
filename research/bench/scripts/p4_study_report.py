"""P4-b tables (research/results/2026-09-24_p4_study): per-pair final tables, the 3x2 cells,
POD, attribution, D1 readout and B3 robustness, from <OUT>/<pair>/study.json (p4_study.py).

    python research/bench/scripts/p4_study_report.py        # -> tables.json, tables.md

Every T is from the pair's final interleaved steady run (stage F): TileLang variants are
normalized by the TileLang serial of the same run, POD and FlashInfer variants by serial_fi
(FlashInfer prefill then FlashInfer decode) of the same run. Absolute times are also given.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p4_study import OUT, PRIMARY, SECONDARY, R_SPLITS  # noqa: E402

CELLS = {
    "solo_inter": lambda d: d["kind"] in ("green", "streams") and d["row"] == "solo",
    "lib_inter": lambda d: d["kind"] in ("green", "streams") and d["row"] in ("solo", "lib"),
    "derived_inter": lambda d: d["kind"] in ("green", "streams"),
    "solo_intra": lambda d: d["kind"] == "co" and d["row"] == "solo",
    "lib_intra": lambda d: d["kind"] == "co" and d["row"] in ("solo", "lib"),
    "derived_intra": lambda d: d["kind"] == "co",
}
MECH = {
    "streams": lambda d: d["kind"] == "streams",
    "green": lambda d: d["kind"] == "green",
    "green (no hint)": lambda d: d["kind"] == "green" and not d["axes"]["decode_hint"],
    "co SM": lambda d: d["kind"] == "co" and d["binding"] == "sm_dyn",
    "co SM (no hint)": lambda d: d["kind"] == "co" and d["binding"] == "sm_dyn" and not d["axes"]["decode_hint"],
    "co CTA": lambda d: d["kind"] == "co" and d["binding"] == "cta_dyn",
    "co tile (POD policy)": lambda d: d["kind"] == "co" and d["binding"] == "tile_dyn",
    "co static": lambda d: d["kind"] == "co" and d["binding"] == "sm_static",
}


def load(pair):
    try:
        with open(os.path.join(OUT, pair, "study.json")) as f:
            return json.load(f)
    except OSError:
        return None


def sid_of(d):
    import p4_study as S
    return d.get("a", "-"), d.get("b", "-")


def knobs(d) -> str:
    k = d.get("kind")
    if k == "green":
        return f"green {d['n_a']}/{188 - d['n_a']}"
    if k == "streams":
        return f"streams {d['prio']} {d['order']}"
    if k == "co":
        b = d["binding"]
        s = {"sm_dyn": "SM dyn", "sm_static": "SM static", "cta_dyn": "CTA", "tile_dyn": "tile"}[b]
        if b.startswith("sm"):
            s += f" {d['n_a']}/{188 - d['n_a']}"
        else:
            s += f" {d['num_ctas']} CTAs"
            if b == "tile_dyn":
                s += f" P:D {d['ratio'][0]}:{d['ratio'][1]}"
        if b != "tile_dyn":
            s += " +TO" if d["takeover"] else " noTO"
        if tuple(d["chunk"]) != (1, 1):
            s += f" chunk {d['chunk'][0]}/{d['chunk'][1]}"
        if d.get("mbps", 1) != 1:
            s += " regcap"
        if d.get("timing"):
            s += " (timing build)"
        return s
    return k


def final(res):
    F = res["stages"].get("F")
    if not F:
        return None
    S, D = F["steady"], F["desc"]
    ser = S["serial"]["t_iter_us"]
    ser_fi = S["serial_fi"]["t_iter_us"] if "serial_fi" in S else None
    fl = F.get("flush", {})
    rows = {}
    for n, v in S.items():
        d = D.get(n, {})
        own = ser_fi if d.get("kind") in ("pod", "fi_solo", "serial_fi") else ser
        rows[n] = {"name": n, "kind": d.get("kind"), "row": d.get("row"), "knobs": knobs(d), "a": d.get("a"), "b": d.get("b"),
                   "t_us": v["t_iter_us"], "x_own": own / v["t_iter_us"] if own else None, "x_tl": ser / v["t_iter_us"],
                   "paired": v.get("paired"), "mhz": v.get("clock_mhz"), "w": v.get("power_w"),
                   "mj": v.get("energy_mj_per_iter"), "cv": v.get("cv_slices"), "ops": v.get("ops"),
                   "flush_us": (fl.get(n) or {}).get("t_us"), "flush_x": (fl.get(n) or {}).get("speedup"),
                   "timing": d.get("timing", False), "axes": d.get("axes")}
        if rows[n]["flush_us"] and d.get("kind") in ("pod", "fi_solo", "serial_fi") and fl.get("serial_fi"):
            rows[n]["flush_x"] = fl["serial_fi"]["t_us"] / rows[n]["flush_us"]
    return rows, D


def best(rows, D, pred):
    c = [r for n, r in rows.items() if not r["timing"] and D.get(n) and D[n].get("kind") in ("green", "streams", "co")
         and pred(D[n])]
    return min(c, key=lambda r: r["t_us"]) if c else None


def summarize(pair):
    res = load(pair)
    if res is None or "F" not in res["stages"]:
        return None
    rows, D = final(res)
    F = res["stages"]["F"]
    ser = rows["serial"]["t_us"]
    out = {"pair": pair, "protocol": F.get("protocol", "full"), "T_serial": ser,
           "T_serial_fi": rows.get("serial_fi", {}).get("t_us"), "pod": rows.get("pod"),
           "solo_a": rows.get("solo_a"), "solo_b": rows.get("solo_b"), "fi_a": rows.get("fi_a"), "fi_b": rows.get("fi_b"),
           "serial_row": rows["serial"], "serial_fi_row": rows.get("serial_fi"), "cells": {}, "mech": {}}
    for c, p in CELLS.items():
        out["cells"][c] = best(rows, D, p)
    for m, p in MECH.items():
        out["mech"][m] = best(rows, D, p)
    inter = [out["cells"][c] for c in ("solo_inter", "lib_inter", "derived_inter") if out["cells"][c]]
    out["T_inter*"] = min(inter, key=lambda r: r["t_us"]) if inter else None
    ce = out["cells"]
    if ce["derived_intra"] and out["T_inter*"]:
        out["d1"] = {"inter*/derived_intra": out["T_inter*"]["t_us"] / ce["derived_intra"]["t_us"],
                     "lib_intra/derived_intra": (ce["lib_intra"]["t_us"] / ce["derived_intra"]["t_us"]) if ce["lib_intra"] else None,
                     "inter*/lib_intra": (out["T_inter*"]["t_us"] / ce["lib_intra"]["t_us"]) if ce["lib_intra"] else None,
                     "lib_inter/derived_inter": (ce["lib_inter"]["t_us"] / ce["derived_inter"]["t_us"]) if ce["lib_inter"] else None}
    att = F.get("attribution", {})
    out["attribution"] = {k: rows.get(v) for k, v in att.items()}
    out["role_times"] = F.get("role_times", {})
    out["timing_names"] = F.get("timing", {})
    out["ablations"] = {c: {k: rows.get(v) for k, v in a.items()} for c, a in F.get("ablations", {}).items()}
    out["rows"] = rows
    out["pod_ok"] = F.get("pod_side_stream_bitwise_eq_default")
    out["guard_clean"] = F["meta"]["guard"]["clean"]
    R = res["stages"].get("R")
    if R:
        out["R"] = {"summary": R["summary"], "rules": R["rules"], "splits": R["splits"],
                    "transfer_from": R.get("transfer_from"), "subst": R.get("cta_substitutes"),
                    "guard_clean": R["meta"]["guard"]["clean"]}
    AT = res["stages"].get("AT")
    if AT:
        S, D = AT["steady"], AT["desc"]
        ser, ser_fi = S["serial"]["t_iter_us"], S["serial_fi"]["t_iter_us"]
        att = AT["attribution"]
        own = {"pod_emu": "ser_filike", "pod_emu_natural": "ser_filike_natural", "pod_emu_r11": "ser_filike",
               "pod_emu_cta": "ser_filike", "pod_emu_hint": "ser_filike", "sm_filike": "ser_filike",
               "streams_filike": "ser_filike_lpt", "streams_filike_pB": "ser_filike_lpt",
               "pod_emu_lpt": "ser_filike_lpt", "sm_filike_lpt": "ser_filike_lpt"}
        at = {}
        for n, v in S.items():
            d = D.get(n, {})
            tag = next((k for k, x in att.items() if x == n), None)
            if d.get("kind") in ("pod", "fi_solo", "serial_fi"):
                o = ser_fi
            elif tag in own and att.get(own[tag]) in S:
                o = S[att[own[tag]]]["t_iter_us"]
            else:
                o = ser
            at[n] = {"name": n, "tag": tag, "knobs": knobs(d), "a": d.get("a"), "b": d.get("b"), "t_us": v["t_iter_us"],
                     "x_own": o / v["t_iter_us"], "own": "serial_fi" if o == ser_fi else (own.get(tag) or "serial"),
                     "x_tl": ser / v["t_iter_us"], "mhz": v.get("clock_mhz"), "w": v.get("power_w"),
                     "mj": v.get("energy_mj_per_iter"), "ops": v.get("ops"), "timing": d.get("timing", False)}
        out["AT"] = {"rows": at, "picks": AT["picks"], "role_times": AT.get("role_times", {}), "timing": AT.get("timing", {}),
                     "T_serial": ser, "T_serial_fi": ser_fi, "guard_clean": AT["meta"]["guard"]["clean"],
                     "pod_ok": AT.get("pod_side_stream_bitwise_eq_default")}
        out["wall_AT"] = res.get("stage_wall_s", {}).get("AT")
    SC = res["stages"].get("SC")
    if SC:
        out["fidelity"] = SC.get("fidelity")
    walls = res.get("stage_wall_s", {})
    out["gpu_s"] = sum(walls.values()) - res.get("compile_s", 0.0)
    out["wall_s"] = walls
    out["compile_s"] = res.get("compile_s", 0.0)
    out["guards"] = {s: (v.get("meta") or {}).get("guard") for s, v in res["stages"].items() if isinstance(v, dict) and "meta" in v}
    out["flush_guards_clean"] = all(g.get("clean", True) for s, v in res["stages"].items() if isinstance(v, dict)
                                    for g in (v.get("flush", {}) or {}).get("_guard", []))
    return out


def f1(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def cellstr(r, own_key="x_own"):
    if not r:
        return "-"
    return f"{r['t_us']:.1f} (×{r[own_key]:.3f})"


def cfgs(r):
    if not r:
        return "-"
    return f"`{r['a']}` / `{r['b']}`"


def md_pair(s) -> list[str]:
    L = []
    p = s["pair"]
    L.append(f"### {p} ({s['protocol']})")
    L.append("")
    L.append(f"T_serial (TileLang) = {s['T_serial']:.1f} µs; T_serial_fi (FlashInfer) = {f1(s['T_serial_fi'])} µs; "
             f"POD = {cellstr(s['pod'])} vs serial_fi, ×{s['pod']['x_tl']:.3f} vs TileLang serial. "
             f"Guard clean: {s['guard_clean']}; POD side-stream output == default-stream output: {s['pod_ok']}.")
    L.append("")
    L.append("| cell | variant | knobs | configs (prefill / decode) | µs | × own serial | paired (med, min, max) | MHz | W | mJ/iter | flush µs (×) | per-op / per-role end µs |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    def line(label, r, extra=""):
        if not r:
            L.append(f"| {label} | – | | | | | | | | | | |")
            return
        ops = r.get("ops") or {}
        opstr = " ".join(f"{k} {v:.0f}" for k, v in ops.items()) + extra
        pr = r.get("paired")
        L.append(f"| {label} | `{r['name']}` | {r['knobs']} | {cfgs(r) if r.get('a') else '-'} | {r['t_us']:.1f} | "
                 f"×{r['x_own']:.3f} | {'-' if not pr else f'{pr[0]:.3f} ({pr[1]:.3f}–{pr[2]:.3f})'} | {f1(r['mhz'], 0)} | "
                 f"{f1(r['w'], 0)} | {f1(r['mj'])} | {f1(r['flush_us'])} (×{f1(r['flush_x'], 3)}) | {opstr} |")

    line("serial (TL)", s["serial_row"])
    line("solo prefill (TL)", s["solo_a"])
    line("solo decode (TL)", s["solo_b"])
    line("serial_fi (FlashInfer)", s["serial_fi_row"])
    line("solo prefill (FI)", s["fi_a"])
    line("solo decode (FI)", s["fi_b"])
    line("**POD** (vs serial_fi)", s["pod"])
    rt = s.get("role_times", {})
    tn = s.get("timing_names", {})
    rows = s["rows"]
    role_of = {}
    for tag, n in tn.items():
        if tag in rt and rt[tag]:
            base = n[:-3] if n.endswith("_tm") else n
            role_of[base] = f" | T_P {rt[tag]['T_A_us']:.0f}, T_D {rt[tag]['T_B_us']:.0f}"
    for c in CELLS:
        r = s["cells"][c]
        line(f"**T[{c.replace('_', ',')}]**", r, role_of.get(r["name"], "") if r else "")
    line("T_inter*", s["T_inter*"])
    for m, r in s["mech"].items():
        line(f"best {m}", r, role_of.get(r["name"], "") if r else "")
    for k, r in s["attribution"].items():
        line(f"attr: {k}", r, role_of.get(r["name"], "") if r else "")
    for c, a in s["ablations"].items():
        for k, r in a.items():
            line(f"abl {c}: {k}", r)
    L.append("")
    if rt:
        L.append("Per-role completion times (timing builds, device %globaltimer, median of 24 launches after warm-up):")
        L.append("")
        L.append("| build | variant | T_P µs | T_D µs | makespan µs | steals P / D | CTAs |")
        L.append("|---|---|---|---|---|---|---|")
        for tag, v in rt.items():
            if v:
                L.append(f"| {tag} | `{tn.get(tag)}` | {f1(v.get('T_A_us'))} | {f1(v.get('T_B_us'))} | {f1(v.get('makespan_us'))} | "
                         f"{f1(v.get('steal_A'), 0)} / {f1(v.get('steal_B'), 0)} | {f1(v.get('ctas_A'), 0)} / {f1(v.get('ctas_B'), 0)} |")
        L.append("")
    return L


def md_all_rows(s) -> list[str]:
    L = [f"#### {s['pair']}: every variant of the final run", "",
         "| variant | kind | row | knobs | configs | µs | × own serial | MHz | W | mJ/iter | flush × |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(s["rows"].items(), key=lambda kv: kv[1]["t_us"]):
        L.append(f"| `{n}` | {r['kind']} | {r['row']} | {r['knobs']} | {cfgs(r) if r.get('a') else '-'} | {r['t_us']:.1f} | "
                 f"×{f1(r['x_own'], 3)} | {f1(r['mhz'], 0)} | {f1(r['w'], 0)} | {f1(r['mj'])} | ×{f1(r['flush_x'], 3)} |")
    L.append("")
    return L


def decomposition(A) -> dict | None:
    """POD vs our best intra variant within the AT run: abs ratio = (serial_fi / serial) x
    (x_ours / x_POD), i.e. the solo-kernel effect times the co-location effect; and the POD
    policy emulated in TileLang (same run) against POD itself."""
    rows = A["rows"]
    by_tag = {r["tag"]: r for r in rows.values() if r["tag"]}
    pod = next((r for r in rows.values() if r["name"] == "pod"), None)
    ours = rows.get(A["picks"].get("intra")) if A["picks"].get("intra") else None
    inter = rows.get(A["picks"].get("inter")) if A["picks"].get("inter") else None
    if not pod or not ours:
        return None
    ser, ser_fi = A["T_serial"], A["T_serial_fi"]
    out = {"pod_us": pod["t_us"], "x_pod": pod["x_own"], "ours_us": ours["t_us"], "x_ours": ours["x_own"],
           "inter_us": inter["t_us"] if inter else None, "x_inter": inter["x_own"] if inter else None,
           "abs_pod_over_ours": pod["t_us"] / ours["t_us"], "solo_effect": ser_fi / ser,
           "coloc_effect": ours["x_own"] / pod["x_own"]}
    for t in ("pod_emu", "pod_emu_lpt", "pod_emu_natural", "tile_ourtiles", "ser_filike", "sm_filike", "win_pod_decode",
              "streams_filike"):
        if t in by_tag:
            out[f"{t}_us"] = by_tag[t]["t_us"]
            out[f"x_{t}"] = by_tag[t]["x_own"]
    return out


def md_attr(pairs) -> list[str]:
    L = ["## Attribution run (stage AT, steady, one interleaved run per primary pair)", ""]
    L += ["Decomposition (all numbers from the AT run): POD / ours (absolute) = solo-kernel effect "
          "(serial_fi / serial_TL) × co-location effect (x_ours / x_POD, each vs its own serial). "
          "POD emulation = tile binding (POD's per-SM ticket policy, 2 CTAs/SM, POD's ratio) with "
          "FlashInfer-like tiles (prefill 128×32 / 128 thr, FlashInfer's CTA order; decode 16×64 / "
          "128 thr, POD's KV split), vs its own FI-like TileLang serial.", "",
          "| pair | POD µs (×) | ours (intra) µs (×) | best inter µs (×) | POD/ours abs | solo effect | co-location effect | "
          "POD emu µs (× own) | POD emu, LPT order µs (× own) | POD policy + our tiles, 1 CTA/SM µs (×) | FI-like serial µs |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in pairs:
        A = s.get("AT")
        if not A:
            continue
        d = decomposition(A)
        s["decomposition"] = d
        if not d:
            continue
        L.append(f"| {s['pair']} | {d['pod_us']:.1f} (×{d['x_pod']:.3f}) | {d['ours_us']:.1f} (×{d['x_ours']:.3f}) | "
                 f"{f1(d['inter_us'])} (×{f1(d['x_inter'], 3)}) | {d['abs_pod_over_ours']:.3f} | {d['solo_effect']:.3f} | "
                 f"{d['coloc_effect']:.3f} | {f1(d.get('pod_emu_us'))} (×{f1(d.get('x_pod_emu'), 3)}) | "
                 f"{f1(d.get('pod_emu_lpt_us'))} (×{f1(d.get('x_pod_emu_lpt'), 3)}) | "
                 f"{f1(d.get('tile_ourtiles_us'))} (×{f1(d.get('x_tile_ourtiles'), 3)}) | {f1(d.get('ser_filike_us'))} |")
    L.append("")
    L += ["Own serial: POD -> serial_fi; POD emulation with FI-like tiles -> the FI-like TileLang serial of the same "
          "tiles (FI order: `ser_filike`; LPT order: `ser_filike_lpt`; natural: `ser_filike_natural`); everything else -> the TileLang serial.", ""]
    for s in pairs:
        A = s.get("AT")
        if not A:
            continue
        L.append(f"### {s['pair']}: T_serial {A['T_serial']:.1f} µs, T_serial_fi {A['T_serial_fi']:.1f} µs (guard clean: {A['guard_clean']})")
        L.append("")
        L.append("| tag | variant | knobs | configs | µs | × own serial (own) | × TL serial | MHz | W | mJ/iter |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for n, r in sorted(A["rows"].items(), key=lambda kv: kv[1]["t_us"]):
            if r["timing"]:
                continue
            tag = r["tag"] or {v: k for k, v in A["picks"].items()}.get(n, "")
            if tag in A["picks"]:
                tag = f"F winner ({tag})"
            L.append(f"| {tag} | `{n}` | {r['knobs']} | {cfgs(r) if r.get('a') else '-'} | {r['t_us']:.1f} | ×{r['x_own']:.3f} ({r['own']}) | "
                     f"×{r['x_tl']:.3f} | {f1(r['mhz'], 0)} | {f1(r['w'], 0)} | {f1(r['mj'])} |")
        rt = A.get("role_times", {})
        if rt:
            L.append("")
            L.append("Role ends (timing builds): " + "; ".join(
                f"{k}: T_P {f1(v.get('T_A_us'))}, T_D {f1(v.get('T_B_us'))} µs" for k, v in rt.items() if v))
        L.append("")
    return L


def md_robust(pairs) -> list[str]:
    L = ["## B3-style robustness (stage R, steady)", "",
         "| pair | mechanism | oracle (split) | R1 split: × | R2 split: × | worst of sweep (split) | regret R1 / R2 / worst | transfer split: × (regret) |",
         "|---|---|---|---|---|---|---|---|"]
    for s in pairs:
        R = s.get("R")
        if not R:
            continue
        ru = R["rules"]
        for mech in ("green", "co"):
            m = R["summary"][mech]
            tr = (f"{m['transfer_split']}: ×{f1(m.get('transfer'), 3)} ({f1(m.get('regret_transfer'), 3)})"
                  if m.get("transfer_split") else "(source)" if s["pair"] == PRIMARY[0] else "-")
            L.append(f"| {s['pair']} | {'green ctx' if mech == 'green' else 'CoKernel SM dyn+TO'} | ×{m['oracle']:.3f} ({m['oracle_split']}) | "
                     f"{ru['R1']}: ×{f1(m['R1'], 3)} | {ru['R2']}: ×{f1(m['R2'], 3)} | ×{m['worst_sweep']:.3f} ({m['worst_split']}) | "
                     f"{f1(m['regret_R1'], 3)} / {f1(m['regret_R2'], 3)} / {m['regret_worst']:.3f} | {tr} |")
    L.append("")
    for s in pairs:
        R = s.get("R")
        if not R:
            continue
        sp = [int(x) for x in R["splits"]]
        L.append(f"{s['pair']} (speed-up vs TL serial per prefill share; best of the candidates):")
        L.append("")
        L.append("| mechanism | " + " | ".join(str(x) for x in sp) + " |")
        L.append("|---|" + "---|" * len(sp))
        for mech in ("green", "co"):
            cv = R["summary"][mech]["curve"]
            L.append(f"| {mech} | " + " | ".join(f1(cv.get(str(x)), 3) for x in sp) + " |")
            for src, c in sorted(R["summary"][mech].get("per_src", {}).items()):
                L.append(f"| {mech} [{src}] | " + " | ".join(f1(c.get(str(x)), 3) for x in sp) + " |")
        L.append("")
    return L


def main():
    summ = {}
    for p in PRIMARY + SECONDARY:
        s = summarize(p)
        if s:
            summ[p] = s
    L = ["# P4-b tables (generated by p4_study_report.py)", ""]
    L.append("## Summary (steady, final interleaved run per pair; × = own serial / T)")
    L.append("")
    L.append("| pair | T_serial TL | T_serial FI | POD µs (× FI serial; × TL serial) | T[solo,inter] | T[lib,inter] | T[derived,inter] | "
             "T[solo,intra] | T[lib,intra] | T[derived,intra] | T_inter* |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p, s in summ.items():
        c = s["cells"]
        pod = s["pod"]
        L.append(f"| {p} | {s['T_serial']:.1f} | {f1(s['T_serial_fi'])} | {pod['t_us']:.1f} (×{pod['x_own']:.3f}; ×{pod['x_tl']:.3f}) | "
                 + " | ".join(cellstr(c[k]) for k in CELLS) + f" | {cellstr(s['T_inter*'])} |")
    L.append("")
    L.append("## D1 readout (plan §2.8, thresholds unchanged: full claim needs ≥1.10 and ≥1.05 on ≥2 pairs; weak claim ≥1.10)")
    L.append("")
    L.append("| pair | T_inter*/T[derived,intra] | T[lib,intra]/T[derived,intra] | T_inter*/T[lib,intra] | T[lib,inter]/T[derived,inter] |")
    L.append("|---|---|---|---|---|")
    for p, s in summ.items():
        d = s.get("d1")
        if d:
            L.append(f"| {p} | {d['inter*/derived_intra']:.3f} | {f1(d['lib_intra/derived_intra'], 3)} | {f1(d['inter*/lib_intra'], 3)} | "
                     f"{f1(d['lib_inter/derived_inter'], 3)} |")
    L.append("")
    L.append("## POD vs our mechanisms (absolute µs and × own serial)")
    L.append("")
    L.append("| pair | POD | best green | best CoKernel SM | best CoKernel CTA | best tile (POD policy) | POD emulation (tile, FI-like tiles) | FI-like TL serial | POD / best-TL (abs) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p, s in summ.items():
        m, a = s["mech"], s["attribution"]
        bt = min([r for r in (m["green"], m["co SM"], m["co CTA"], m["co tile (POD policy)"], m["streams"]) if r],
                 key=lambda r: r["t_us"])
        L.append(f"| {p} | {cellstr(s['pod'])} | {cellstr(m['green'])} | {cellstr(m['co SM'])} | {cellstr(m['co CTA'])} | "
                 f"{cellstr(m['co tile (POD policy)'])} | {cellstr(a.get('pod_emu'))} | {cellstr(a.get('ser_filike'))} | "
                 f"{s['pod']['t_us'] / bt['t_us']:.3f} |")
    L.append("")
    L.append("## Per-pair final tables")
    L.append("")
    for p, s in summ.items():
        L += md_pair(s)
    L += md_attr(list(summ.values()))
    L += md_robust(list(summ.values()))
    L.append("## Screening fidelity (flush screen vs steady confirmation, stage SC)")
    L.append("")
    L.append("| pair | n | Spearman | Kendall | steady winner's screen rank |")
    L.append("|---|---|---|---|---|")
    for p, s in summ.items():
        fd = s.get("fidelity")
        if fd:
            L.append(f"| {p} | {fd['n']} | {fd['spearman']:.2f} | {fd['kendall']:.2f} | {fd['winner_screen_rank']} |")
    L.append("")
    L.append("## GPU time (stage wall time minus compile time)")
    L.append("")
    L.append("| pair | GPU s | stages (s) | compile s |")
    L.append("|---|---|---|---|")
    tot = 0.0
    for p, s in summ.items():
        tot += s["gpu_s"]
        L.append(f"| {p} | {s['gpu_s']:.0f} | " + ", ".join(f"{k} {v:.0f}" for k, v in s["wall_s"].items())
                 + f" | {s['compile_s']:.0f} |")
    L.append(f"| total | {tot:.0f} ({tot / 3600:.2f} h) | | |")
    L.append("")
    L.append("## Every variant of the final runs")
    L.append("")
    for p, s in summ.items():
        L += md_all_rows(s)
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    slim = {p: {k: v for k, v in s.items() if k != "rows"} for p, s in summ.items()}
    with open(os.path.join(OUT, "tables.json"), "w") as f:
        json.dump(slim, f, indent=1, default=str)
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
