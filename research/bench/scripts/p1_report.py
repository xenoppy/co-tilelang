"""P1 3x2 study (A4-A5): per-pair tables, D1 interim readout, screening fidelity, GPU time.

    python research/bench/scripts/p1_report.py        # -> <OUT>/tables.json, <OUT>/tables.md

Sources: <OUT>/solo_v1/summary.json (A1) and <OUT>/<pair>/study.json (A2/A3). Every T in the
tables comes from the final steady run (stage F) of the pair, where serial, the solo runs and
the winner of every cell were interleaved; the flush column is the clean-flush run of the same
variants. Winners were selected in the steady search stages (I2, I3, C2, C3), with times
normalized by the serial of their own stage.

Lower bound: LB = max(LB_tc, LB_dram, LB_power).
  LB_tc    = (flops_A + flops_B) / best measured tensor rate (P1-S catalog, 357 TFLOP/s)
  LB_dram  = (bytes_min_A + bytes_min_B) / best measured DRAM rate (P1-S catalog, 1642 GB/s)
  LB_power = (E_A + E_B) / 600 W, E = NVML board power x time per iteration of the op running
             back-to-back alone in the same steady run (solo_a, solo_b). Energy per op depends on
             the voltage/clock point, so this is an estimate, not a strict bound.
"""
from __future__ import annotations

import json
import os
import sys

import p1_common as C
import p1_study as P
from cotile import catalog

CELLS = ("solo_inter", "lib_inter", "solo_intra", "lib_intra")
P_STATIC = None   # idle-but-clocked board power (W), set in main() from the main pair's stage P


def static_power() -> float | None:
    st = C.load(os.path.join("main", "study.json"))
    p = (st or {}).get("stages", {}).get("P")
    return p["steady"]["idle_spin"]["power_w"] if p else None


def knobs(d: dict) -> str:
    k = d.get("kind")
    if k == "green":
        return f"green {d['n_a']}/{188 - d['n_a']}"
    if k == "streams":
        return f"streams prio={d['prio']} order={d['order']}"
    if k == "co":
        b = d["binding"]
        if b == "cta_dyn":
            s = f"CTA-bind {d['num_ctas']} CTAs (GEMM CTA on 188 SMs, decode CTA beside it on {d['num_ctas'] - 188})"
        else:
            s = f"SM-bind {'dyn' if b == 'sm_dyn' else 'static'} {d['n_a']}/{188 - d['n_a']}"
        return s + (" +takeover" if d["takeover"] else "") + (f" chunk {d['chunk'][0]}/{d['chunk'][1]}" if b != "sm_static" else "")
    return k or ""


def cfg_info(summ, shape, tag):
    s = summ["shapes"][shape]
    sb = s["solo_best"]
    t = s["t188"].get(tag)
    return {"tag": tag, "flush_rank": s["flush_rank"].get(tag), "n_configs": len(s["flush_rank"]),
            "solo_slowdown": (t / s["t188"][sb] - 1) if t and s["t188"].get(sb) else None,
            "steady_us": (s["steady"].get("variants", {}).get(tag) or {}).get("t_iter_us")}


def pair_table(pair: str, summ, rates) -> dict | None:
    st = C.load(os.path.join(pair, "study.json"))
    if not st or "F" not in st["stages"]:
        return None
    F = st["stages"]["F"]
    S, FL, D = F["steady"], F["flush"], F["desc"]
    best = P.winners(st)        # re-derived with the current rule (config identity defines the row)
    missing = [n for _, n, _ in best.values() if n not in S]
    if missing:
        raise RuntimeError(f"{pair}: winners {missing} were not measured in stage F; re-run F")
    ser = S["serial"]["t_iter_us"]
    wa = catalog.work_min("gemm", C._shape_fields("gemm", st["A"]))
    wb = catalog.work_min("gqa_decode", C._shape_fields("gqa_decode", st["B"]))
    ea = S["solo_a"]["power_w"] * S["solo_a"]["t_iter_us"]     # uJ
    eb = S["solo_b"]["power_w"] * S["solo_b"]["t_iter_us"]
    lb = {"tc": (wa["flops"] + wb["flops"]) / (rates["tflops"] * 1e12) * 1e6,
          "dram": (wa["bytes_min"] + wb["bytes_min"]) / (rates["gbps"] * 1e9) * 1e6,
          "power": (ea + eb) / catalog.POWER_CAP_W}
    lb["LB"] = max(lb["tc"], lb["dram"], lb["power"])
    lb["binding"] = max(("tc", "dram", "power"), key=lambda k: lb[k])
    lb["E_A_uJ"], lb["E_B_uJ"] = ea, eb
    # same estimate with the idle-but-clocked board power P_s removed from both solo energies
    # (P_s shared by the two ops in a co-run; measured in stage P of the main pair)
    if P_STATIC:
        ta, tb = S["solo_a"]["t_iter_us"], S["solo_b"]["t_iter_us"]
        lb["power_dyn"] = (ea - P_STATIC * ta + eb - P_STATIC * tb) / (catalog.POWER_CAP_W - P_STATIC)
    energy = {n: S[n]["power_w"] * S[n]["t_iter_us"] / 1e3 for n in S if S[n].get("power_w")}
    out = {"pair": pair, "A": st["A"], "B": st["B"], "T_serial": ser,
           "serial": {k: S["serial"].get(k) for k in ("clock_mhz", "power_w")},
           "solo_a": {k: S["solo_a"].get(k) for k in ("t_iter_us", "clock_mhz", "power_w")},
           "solo_b": {k: S["solo_b"].get(k) for k in ("t_iter_us", "clock_mhz", "power_w")},
           "LB": lb, "speedup_bound": ser / max(lb["LB"], S["solo_a"]["t_iter_us"], S["solo_b"]["t_iter_us"]),
           "cells": {}, "flush_serial": FL.get("_serial", [None])[0], "energy_mJ_per_iter": energy}
    if "solo_a_wsoff" in S:
        out["solo_a_wsoff"] = {"t_iter_us": S["solo_a_wsoff"]["t_iter_us"],
                               "vs_solo_a": S["solo_a_wsoff"]["t_iter_us"] / S["solo_a"]["t_iter_us"] - 1}
    for cell, (x, name, stage) in best.items():
        v, d = S[name], D[name]
        c = {"variant": name, "selected_in": stage, "T": v["t_iter_us"], "speedup": v["speedup"],
             "paired": v.get("paired"), "clock_mhz": v.get("clock_mhz"), "power_w": v.get("power_w"),
             "kcycles": v.get("kcycles_per_iter"), "ops": v.get("ops"), "knobs": knobs(d),
             "flush_T": FL.get(name, {}).get("t_us"), "flush_speedup": FL.get(name, {}).get("speedup"),
             "flush_clock_mhz": FL.get(name, {}).get("clock_mhz"),
             "cfg_a": cfg_info(summ, st["A"], d["a"]), "cfg_b": cfg_info(summ, st["B"], d["b"])}
        tcell = next((k for k, tn in F.get("timing", {}).items() if tn == name + "_tm"), None)
        if tcell is not None:
            tn = name + "_tm"
            c["timing_build"] = {"T": S[tn]["t_iter_us"], "overhead": S[tn]["t_iter_us"] / v["t_iter_us"] - 1,
                                 "roles": F["role_times"][tcell]}
        out["cells"][cell] = c
    T = {k: out["cells"][k]["T"] for k in CELLS if k in out["cells"]}
    out["T"] = T
    if "solo_inter" in T and "lib_inter" in T:
        out["T_inter_star"] = min(T["solo_inter"], T["lib_inter"])
        if "lib_intra" in T:
            out["D1_inter_over_lib_intra"] = out["T_inter_star"] / T["lib_intra"]
        if "solo_intra" in T:
            out["inter_star_over_solo_intra"] = out["T_inter_star"] / T["solo_intra"]
    if "solo_inter" in T and "lib_inter" in T:
        out["lib_gain_inter"] = T["solo_inter"] / T["lib_inter"]
    if "solo_intra" in T and "lib_intra" in T:
        out["lib_gain_intra"] = T["solo_intra"] / T["lib_intra"]
    c2 = st["stages"].get("C2", {})
    out["fidelity"] = c2.get("fidelity")
    out["cta_feasible"] = st["stages"].get("C1", {}).get("cta_feasible")
    out["cta_solo_feasible"] = st["stages"].get("C1", {}).get("cta_solo_feasible")
    out["stage_wall_s"] = st.get("stage_wall_s")
    return out


ROWS = (("solo_inter", "T[solo,inter]"), ("lib_inter", "T[lib,inter]"), ("solo_streams", "two streams (solo)"),
        ("solo_intra", "T[solo,intra]"), ("lib_intra", "T[lib,intra]"), ("static", "static schedule (best)"),
        ("cta", "CTA binding (best)"))


def _f(x, fmt="{:.1f}"):
    return "–" if x is None else fmt.format(x)


def cfgs(c):
    a, b = c["cfg_a"], c["cfg_b"]
    sa = f"`{a['tag'].replace('_k1', '').replace('_g8', '')}` (#{a['flush_rank']}, {a['solo_slowdown'] * 100:+.1f}%)"
    sb = f"`{b['tag']}` (#{b['flush_rank']}, {b['solo_slowdown'] * 100:+.1f}%)"
    return f"{sa} / {sb}"


def markdown(tabs: dict, gpu: dict) -> str:
    L = []
    L.append("| pair | A / B | T_serial | LB tc / dram / power | T[solo,inter] | T[lib,inter] | T[solo,intra] | T[lib,intra] | "
             "T_inter* / T[lib,intra] | lib/solo inter | lib/solo intra |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p, t in tabs.items():
        if not t:
            continue
        T, s, lb = t["T"], t["T_serial"], t["LB"]
        cell = lambda k: f"{T[k]:.1f} (×{s / T[k]:.3f})" if k in T else "–"  # noqa: E731
        L.append(f"| {p} | {t['A']} / {t['B']} | {s:.1f} | {lb['tc']:.0f} / {lb['dram']:.0f} / {lb['power']:.0f} | "
                 f"{cell('solo_inter')} | {cell('lib_inter')} | {cell('solo_intra')} | {cell('lib_intra')} | "
                 f"{_f(t.get('D1_inter_over_lib_intra'), '{:.3f}')} | {_f(t.get('lib_gain_inter'), '{:.3f}')} | "
                 f"{_f(t.get('lib_gain_intra'), '{:.3f}')} |")
    for p, t in tabs.items():
        if not t:
            continue
        L.append(f"\n### {p}: GEMM {t['A']} × decode {t['B']}\n")
        L.append("| variant | knobs | configs A / B (solo rank, solo slowdown) | steady µs | × serial (paired min–max) | MHz | W | "
                 "flush µs | flush × | per-op or per-role end (µs) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        L.append(f"| serial | – | solo-best | {t['T_serial']:.1f} | 1.000 | {t['serial']['clock_mhz']:.0f} | "
                 f"{t['serial']['power_w']:.0f} | {_f(t.get('flush_serial'))} | 1.000 | – |")
        for k, lab in (("solo_a", "solo A"), ("solo_b", "solo B")):
            v = t[k]
            L.append(f"| {lab} | full GPU, back-to-back | solo-best | {v['t_iter_us']:.1f} | – | {v['clock_mhz']:.0f} | "
                     f"{v['power_w']:.0f} | – | – | – |")
        if "solo_a_wsoff" in t:
            L.append(f"| solo A, ws=off twin | full GPU | – | {t['solo_a_wsoff']['t_iter_us']:.1f} "
                     f"({t['solo_a_wsoff']['vs_solo_a'] * 100:+.1f}% vs solo A) | – | – | – | – | – | – |")
        for k, lab in ROWS:
            c = t["cells"].get(k)
            if not c:
                continue
            p_ = c.get("paired") or [None, None, None]
            if c.get("timing_build"):
                r = c["timing_build"]["roles"]
                ends = f"T_A {r['T_A_us']:.0f}, T_B {r['T_B_us']:.0f} (CTAs {r['ctas_A']:.0f}/{r['ctas_B']:.0f})"
            elif c.get("ops"):
                ends = ", ".join(f"{o} {v:.0f}" for o, v in c["ops"].items())
            else:
                ends = "–"
            L.append(f"| **{lab}** `{c['variant']}` | {c['knobs']} | {cfgs(c)} | {c['T']:.1f} | ×{c['speedup']:.3f} "
                     f"({_f(p_[1], '{:.3f}')}–{_f(p_[2], '{:.3f}')}) | {_f(c['clock_mhz'], '{:.0f}')} | {_f(c['power_w'], '{:.0f}')} | "
                     f"{_f(c['flush_T'])} | {_f(c['flush_speedup'], '×{:.3f}')} | {ends} |")
        lb = t["LB"]
        L.append(f"\nLB: tensor {lb['tc']:.0f}, DRAM {lb['dram']:.0f}, power (solo energies {lb['E_A_uJ'] / 1e3:.0f} + "
                 f"{lb['E_B_uJ'] / 1e3:.0f} mJ at 600 W) {lb['power']:.0f} µs. T_inter* = {_f(t.get('T_inter_star'))} µs.")
    L.append("\n### Screening fidelity (lib-row CoKernels, C1 clean-flush screen vs C2 steady)\n")
    L.append("| pair | confirmed | Spearman | Kendall | Spearman within top-8 | steady winner | its screen rank | in top-8 | regret of screen #1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p, t in tabs.items():
        fd = (t or {}).get("fidelity")
        if not fd:
            continue
        L.append(f"| {p} | {fd['n']} | {fd['spearman']:.2f} | {fd['kendall']:.2f} | {_f(fd.get('spearman_topk'), '{:.2f}')} | "
                 f"`{fd['steady_winner']}` | {fd['winner_screen_rank']} | {fd['winner_in_topk']} | "
                 f"{fd['regret_of_screen_top1'] * 100:.1f}% |")
    return "\n".join(L)


def carveout_rows(pair: str) -> list[str]:
    st = C.load(os.path.join(pair, "study.json"))
    K = (st or {}).get("stages", {}).get("K")
    if not K:
        return []
    S, info = K["steady"], K["info"]
    L = [f"\n**{pair}** (pairs: " + "; ".join(f"{k}: `{v['a']}` + `{v['b']}` (smem {v['sig_a']['smem'] // 1024}+{v['sig_b']['smem'] // 1024} KB, "
                                         f"co-residable {v['co_residable']})" for k, v in info["pairs"].items()) + ")\n",
         "| pair / priority / order | default carveout µs (×) | carveout 100 µs (×) | per-op end, default (µs) |", "|---|---|---|---|"]
    for pk in info["pairs"]:
        for prio in ("eq", "pA", "pB"):
            for order in ("ab", "ba"):
                d, m = S.get(f"K_{pk}_def_{prio}_{order}"), S.get(f"K_{pk}_max_{prio}_{order}")
                if d and m:
                    L.append(f"| {pk} {prio} {order} | {d['t_iter_us']:.1f} (×{d['speedup']:.3f}) | {m['t_iter_us']:.1f} (×{m['speedup']:.3f}) | "
                             + ", ".join(f"{o} {v:.0f}" for o, v in (d.get("ops") or {}).items()) + " |")
    return L


def gpu_time(tabs) -> dict:
    """Wall time of the GPU stages minus guard waits and kernel compilation (s)."""
    out = {}
    for p in C.PAIRS:
        st = C.load(os.path.join(p, "study.json"))
        if not st:
            continue
        wall = sum(st.get("stage_wall_s", {}).values())
        waits = 0.0
        for s in st["stages"].values():
            for a in (s.get("meta", {}).get("guard", {}) or {}).get("attempts", []) if isinstance(s, dict) else []:
                waits += a.get("waited_s") or 0.0
            fl = s.get("flush") if isinstance(s, dict) else None
            for g in (fl or {}).get("_guard", []) or []:
                for a in (g or {}).get("attempts", []):
                    waits += a.get("waited_s") or 0.0
        out[p] = {"stage_wall_s": wall, "guard_wait_s": waits, "compile_s": st.get("compile_s", 0.0),
                  "gpu_s": wall - waits - st.get("compile_s", 0.0)}
    return out


def main():
    global P_STATIC
    P_STATIC = static_power()
    summ = C.load(os.path.join(C.SOLO_DIR, "summary.json"))
    rates = catalog.load().rates()
    tabs = {p: pair_table(p, summ, rates) for p in C.PAIRS}
    runlog = C.load(os.path.join(C.SOLO_DIR, "runlog.json")) or {"runs": []}
    gpu = {"A1_s": sum(r.get("gpu_s", 0) for r in runlog["runs"]), "pairs": gpu_time(tabs), "P_static_w": P_STATIC}
    gpu["total_s"] = gpu["A1_s"] + sum(v["gpu_s"] for v in gpu["pairs"].values())
    out = {"rates": rates, "pairs": tabs, "gpu_time": gpu}
    C.save("tables.json", out)
    md = markdown(tabs, gpu)
    md += "\n\n### Energy per iteration (steady F runs, NVML board power x time, mJ)\n\n"
    md += "| pair | E_A + E_B (solo) | serial | T[solo,inter] | T[lib,inter] | T[solo,intra] | T[lib,intra] | LB_power (solo E) | LB_power,dyn (P_s removed) | best T |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for p, t in tabs.items():
        if not t:
            continue
        E, lb = t["energy_mJ_per_iter"], t["LB"]
        e = lambda k: f"{E[t['cells'][k]['variant']]:.0f}" if k in t["cells"] else "–"  # noqa: E731
        md += (f"| {p} | {(lb['E_A_uJ'] + lb['E_B_uJ']) / 1e3:.0f} | {E['serial']:.0f} | {e('solo_inter')} | {e('lib_inter')} | "
               f"{e('solo_intra')} | {e('lib_intra')} | {lb['power']:.0f} µs | {_f(lb.get('power_dyn'), '{:.0f} µs')} | "
               f"{min(t['T'].values()):.0f} µs |\n")
    md += f"\nP_s (idle-but-clocked board power, main pair stage P) = {_f(P_STATIC)} W.\n"
    md += "\n### Carveout (stage K, steady, NVRTC-backend builds of the same kernels)\n" + "\n".join(sum((carveout_rows(p) for p in C.PAIRS), []))
    md += "\n\n### GPU time\n\n| part | GPU s (stage wall - guard waits - compile) |\n|---|---|\n"
    md += f"| A1 (solo re-validation) | {gpu['A1_s']:.0f} |\n" + "".join(f"| {p} | {v['gpu_s']:.0f} (wall {v['stage_wall_s']:.0f}, waits {v['guard_wait_s']:.0f}, compile {v['compile_s']:.0f}) |\n" for p, v in gpu["pairs"].items())
    md += f"| total | {gpu['total_s']:.0f} ({gpu['total_s'] / 3600:.2f} h) |\n"
    with open(os.path.join(C.OUT, "tables.md"), "w") as f:
        f.write(md + "\n")
    print(json.dumps({p: {k: t.get(k) for k in ("T_serial", "T", "T_inter_star", "D1_inter_over_lib_intra")} if t else None
                      for p, t in tabs.items()}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
