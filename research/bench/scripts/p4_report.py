"""P4-a / P3' tables from the p4_solo.py outputs (research/results/2026-09-24_p4_prep).

    source research/env.sh
    python research/bench/scripts/p4_report.py        # -> tables.json, tables.md

Tables
  T1  TileLang prefill vs FlashInfer prefill: steady mode at 188 SMs (primary) and clean flush
      at every budget (TileLang budget-best vs FlashInfer). TileLang slower by > 10% is flagged.
  T2  decode: TileLang solo-best vs FlashInfer (CUDA-core / tensor-core path), steady and flush.
  T3  per shape: solo-best (steady), flush ranking, C_lib members with reasons, budget-best
      per budget and the loss of the full-GPU best at reduced budgets (GOLDYLOC effect).
  T4  spot check of the reused A1 decode points (new / A1 median).
  T5  pairing table for the 8 P4 pairs (steady mode, one run per pair): TileLang solos and
      measured serial, duration ratio, serial / sum of solos, lower bounds LB_tc, LB_dram,
      LB_power (see below), max solo; FlashInfer serial and POD (patched), with clocks and power.
Lower bounds (P1 conventions, research/results/2026-09-23_p1_3x2_A):
  LB_tc    = (issued MMA FLOPs of the two solo-best configs) / R_tc, R_tc = best steady tensor
             rate measured on this GPU (max of: prefill solo-best issued-MMA rate here, GEMM 4096^3
             TileLang/cuBLAS steady rate from A1);
  LB_dram  = (compulsory bytes of both ops) / R_dram, R_dram = best steady compulsory-byte rate
             of any decode solo measured here (TileLang or FlashInfer);
  LB_power = (E_P + E_D) / 600 W with the steady solo energies per iteration. P1 showed this is
             NOT a valid bound (co-located runs pay the ~143 W idle-but-clocked power once, and a
             memory-bound partner runs at a lower clock), so it is reported, not used in the bound.
  bound    = max(LB_tc, LB_dram, max solo); speedup_bound = T_serial / bound.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p4_common as C  # noqa: E402
from p4_common import FULL, catalog  # noqa: E402

POWER_CAP_W = 600.0


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return None


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def work_of(op, tag, cfg_tag):
    cat = catalog.load(C.SOLO_DIR)
    e = cat.get(op, tag)
    return e.configs[cfg_tag].work, e


def main():
    summ = load(os.path.join(C.SOLO_DIR, "summary.json"))["shapes"]
    steady = load(os.path.join(C.SOLO_DIR, "steady_solo.json")) or {}
    pairs = load(os.path.join(C.OUT, "pairs_steady.json")) or {}
    a1_steady = load(os.path.join(C.A1_SOLO_DIR, "steady_solo.json")) or {}
    cat = catalog.load(C.SOLO_DIR)
    out, md = {}, []

    # ---- T1 prefill vs FlashInfer -------------------------------------------------------
    t1 = {}
    md.append("## T1 TileLang prefill vs FlashInfer prefill\n")
    md.append("Steady mode, 188 SMs (back-to-back, rotation > 2x L2); ratio = TileLang / FlashInfer "
              "(> 1: TileLang slower; flagged if > 1.10).\n")
    md.append("| S | TileLang solo-best | TL us | TL MHz | FI us | FI MHz | TL/FI | TL TFLOP/s (useful) | FI TFLOP/s | flag |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag in C.PREFILL_SHAPES:
        s = summ.get(tag)
        st = steady.get(tag)
        if not s or not st:
            continue
        v = st["variants"]
        best = s["solo_best"]
        tl, fi = v[best], v["ref:fi_prefill"]
        e = cat.get("prefill_attn", tag)
        fl = e.configs[best].work["flops"]
        r = tl["t_iter_us"] / fi["t_iter_us"]
        t1[tag] = {"solo_best": best, "tl_us": tl["t_iter_us"], "fi_us": fi["t_iter_us"], "ratio": r,
                   "tl_mhz": tl["clock_mhz"], "fi_mhz": fi["clock_mhz"], "flag_gt_10pct": r > 1.10,
                   "tl_tflops": fl / tl["t_iter_us"] / 1e6, "fi_tflops": fl / fi["t_iter_us"] / 1e6,
                   "steady_all": {k: vv["t_iter_us"] for k, vv in v.items()}}
        md.append(f"| {tag[1:]} | `{best}` | {fmt(tl['t_iter_us'])} | {fmt(tl['clock_mhz'], 0)} | {fmt(fi['t_iter_us'])} | "
                  f"{fmt(fi['clock_mhz'], 0)} | {r:.3f} | {fmt(t1[tag]['tl_tflops'])} | {fmt(t1[tag]['fi_tflops'])} | "
                  f"{'**TL > 10% slower**' if r > 1.10 else 'ok'} |")
    md.append("\nClean flush, per budget (TileLang budget-best config vs FlashInfer single_prefill on the same "
              "green partition):\n")
    md.append("| S | SMs | TL budget-best | TL us | FI us | TL/FI | full-GPU best at this budget (loss) |")
    md.append("|---|---|---|---|---|---|---|")
    for tag in C.PREFILL_SHAPES:
        s = summ.get(tag)
        if not s:
            continue
        e = cat.get("prefill_attn", tag)
        fi = e.refs.get("fi_prefill", {})
        rows = []
        for n in sorted(e.budgets, reverse=True):
            b = e.budget_best(n)
            f = fi.get(n)
            full_best = e.configs[s["solo_best"]].time("grid", n)
            rows.append({"sms": n, "tl_best": b.tag, "tl_us": b.time("grid", n), "fi_us": f.median if f else None,
                         "ratio": b.time("grid", n) / f.median if f else None, "solo_best_us": full_best})
            md.append(f"| {tag[1:]} | {n} | `{b.tag}` | {fmt(b.time('grid', n))} | {fmt(f.median if f else None)} | "
                      f"{fmt(rows[-1]['ratio'], 3)} | {fmt(full_best)} ({fmt((full_best / b.time('grid', n) - 1) * 100 if full_best else None)}%) |")
        t1[tag + "_budgets"] = rows
    out["T1_prefill_vs_flashinfer"] = t1

    # ---- T2 decode vs FlashInfer ---------------------------------------------------------
    t2 = {}
    md.append("\n## T2 TileLang decode vs FlashInfer decode\n")
    md.append("| shape | TileLang solo-best | TL steady us | FI cc us | FI tc us | TL/best FI | TL GB/s | TL flush @188 | FI best flush @188 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for tag in C.DECODE_SHAPES:
        s, st = summ.get(tag), steady.get(tag)
        if not s or not st:
            continue
        v = st["variants"]
        best = s["solo_best"]
        e = cat.get("gqa_decode", tag)
        by = e.configs[best].work["bytes_min"]
        fic, fit = v["ref:fi_decode_cc"]["t_iter_us"], v["ref:fi_decode_tc"]["t_iter_us"]
        tl = v[best]["t_iter_us"]
        rb = e.ref_best(FULL)
        t2[tag] = {"solo_best": best, "tl_us": tl, "fi_cc_us": fic, "fi_tc_us": fit, "ratio": tl / min(fic, fit),
                   "tl_gbps": by / tl / 1e3, "tl_flush188": e.configs[best].time("grid", FULL),
                   "fi_flush188": rb[1].median if rb else None, "fi_flush188_path": rb[0] if rb else None}
        md.append(f"| {tag} | `{best}` | {fmt(tl)} | {fmt(fic)} | {fmt(fit)} | {tl / min(fic, fit):.3f} | "
                  f"{fmt(t2[tag]['tl_gbps'], 0)} | {fmt(t2[tag]['tl_flush188'])} | {fmt(t2[tag]['fi_flush188'])} |")
    out["T2_decode_vs_flashinfer"] = t2

    # ---- T3 solo-best / C_lib / budget-best ------------------------------------------------
    t3 = {}
    md.append("\n## T3 solo-best, C_lib and budget-best per shape\n")
    for op, tags in C.SHAPES.items():
        for tag in tags:
            s = summ.get(tag)
            if not s:
                continue
            e = cat.get(op, tag)
            t3[tag] = {"op": op, "solo_best": s["solo_best"], "solo_best_flush": s["solo_best_flush"],
                       "c_lib": s["c_lib"], "c_lib_reasons": s["c_lib_reasons"], "budget_best": s["budget_best"],
                       "loss_of_full_best": s["budget_loss_of_full_best"], "n_configs": len(e.configs),
                       "reused_from": s.get("reused_from")}
            md.append(f"**{op} {tag}** ({len(e.configs)} configs{'; A1 points reused' if s.get('reused_from') else ''}): "
                      f"solo-best (steady) `{s['solo_best']}`, flush-best `{s['solo_best_flush']}`; C_lib "
                      f"({len(s['c_lib'])}): " + ", ".join(f"`{t}` ({'/'.join(s['c_lib_reasons'][t])})" for t in s["c_lib"]))
            md.append("")
            md.append("| SMs | " + " | ".join(str(n) for n in sorted(e.budgets)) + " |")
            md.append("|---|" + "---|" * len(e.budgets))
            md.append("| budget-best us | " + " | ".join(fmt(s["budget_best"][str(n)][1]) for n in sorted(e.budgets)) + " |")
            md.append("| solo-best us | " + " | ".join(fmt(e.configs[s["solo_best"]].time("grid", n)) for n in sorted(e.budgets)) + " |")
            md.append("| budget-best config | " + " | ".join(f"`{s['budget_best'][str(n)][0]}`" for n in sorted(e.budgets)) + " |")
            md.append("")
    out["T3_solo_clib_budgets"] = t3

    # ---- T4 spot check of reused A1 points -------------------------------------------------
    t4 = {}
    for tag in C.DECODE_REUSED:
        rec = load(os.path.join(C.SOLO_DIR, "points", f"gqa_decode__{tag}.json"))
        a1 = load(os.path.join(C.A1_SOLO_DIR, "points", f"gqa_decode__{tag}.json"))
        if not rec or "spot_check" not in rec:
            continue
        rows = {}
        for k, p in rec["spot_check"].items():
            old = a1["points"].get(k)
            if old:
                rows[k] = {"new_us": p["median"], "a1_us": old["median"], "ratio": p["median"] / old["median"]}
        if tag in a1_steady and tag in steady:
            for k, v in steady[tag]["variants"].items():
                o = a1_steady[tag]["variants"].get(k)
                if o:
                    rows[f"{k}|steady|188"] = {"new_us": v["t_iter_us"], "a1_us": o["t_iter_us"],
                                               "ratio": v["t_iter_us"] / o["t_iter_us"]}
        t4[tag] = rows
    if t4:
        md.append("\n## T4 spot check of the reused A1 decode points (new / A1)\n")
        md.append("| shape | point | new us | A1 us | new/A1 |")
        md.append("|---|---|---|---|---|")
        for tag, rows in t4.items():
            for k, r in rows.items():
                md.append(f"| {tag} | `{k}` | {fmt(r['new_us'])} | {fmt(r['a1_us'])} | {r['ratio']:.3f} |")
    out["T4_a1_spot_check"] = t4

    # ---- T5 pairing table ------------------------------------------------------------------
    rates = {"tc": [], "dram": []}
    for tag in C.PREFILL_SHAPES:
        s, st = summ.get(tag), steady.get(tag)
        if s and st:
            w = cat.get("prefill_attn", tag).configs[s["solo_best"]].work
            rates["tc"].append((w["flops_mma"] / (st["variants"][s["solo_best"]]["t_iter_us"] * 1e-6),
                                f"prefill {tag} {s['solo_best']} (issued MMA FLOPs)"))
    for tag, M, N, K in (("M4096_N4096_K4096", 4096, 4096, 4096),):
        for k, v in a1_steady.get(tag, {}).get("variants", {}).items():
            rates["tc"].append((2.0 * M * N * K / (v["t_iter_us"] * 1e-6), f"A1 GEMM {tag} {k}"))
    for tag in C.DECODE_SHAPES:
        st = steady.get(tag)
        if not st:
            continue
        by = catalog.work_min("gqa_decode", C.shape_of("gqa_decode", tag))["bytes_min"]
        for k, v in st["variants"].items():
            rates["dram"].append((by / (v["t_iter_us"] * 1e-6), f"decode {tag} {k}"))
    r_tc = max(rates["tc"]) if rates["tc"] else (None, None)
    r_dram = max(rates["dram"]) if rates["dram"] else (None, None)
    out["rates"] = {"tc_flops": r_tc[0], "tc_at": r_tc[1], "dram_bps": r_dram[0], "dram_at": r_dram[1]}
    t5 = {}
    md.append("\n## T5 pairing table (steady mode, one interleaved run per pair)\n")
    md.append(f"R_tc = {fmt(r_tc[0] / 1e12 if r_tc[0] else None)} TFLOP/s ({r_tc[1]}); "
              f"R_dram = {fmt(r_dram[0] / 1e9 if r_dram[0] else None, 0)} GB/s ({r_dram[1]}).\n")
    md.append("| pair | t_P us | t_D us | t_P/t_D | T_serial us | serial/(t_P+t_D) | LB_tc | LB_dram | LB_power (not a bound) | max solo | speedup bound | FI serial us | POD us | POD vs FI serial | POD vs TL serial | MHz serial / POD | W serial / POD |")
    md.append("|---|" + "---|" * 16)
    for p, r in pairs.items():
        v = r["variants"]
        tp, td, ts = v["prefill"]["t_iter_us"], v["decode"]["t_iter_us"], v["serial"]["t_iter_us"]
        wp, _ = work_of("prefill_attn", r["prefill"], r["cfg_prefill"])
        wd, _ = work_of("gqa_decode", r["decode"], r["cfg_decode"])
        lb_tc = (wp["flops_mma"] + wd["flops_mma"]) / r_tc[0] * 1e6 if r_tc[0] else None
        bmin = (catalog.work_min("prefill_attn", C.shape_of("prefill_attn", r["prefill"]))["bytes_min"]
                + catalog.work_min("gqa_decode", C.shape_of("gqa_decode", r["decode"]))["bytes_min"])
        lb_dram = bmin / r_dram[0] * 1e6 if r_dram[0] else None
        ep, ed = v["prefill"].get("energy_mj_per_iter"), v["decode"].get("energy_mj_per_iter")
        lb_pow = (ep + ed) * 1e-3 / POWER_CAP_W * 1e6 if ep is not None and ed is not None else None
        bound = max(x for x in (lb_tc, lb_dram, tp, td) if x is not None)
        fis, pod = v["serial_fi"]["t_iter_us"], v["pod"]["t_iter_us"]
        row = {"t_P": tp, "t_D": td, "ratio": tp / td, "T_serial": ts, "serial_over_sum": ts / (tp + td),
               "LB_tc": lb_tc, "LB_dram": lb_dram, "LB_power_ss": lb_pow, "max_solo": max(tp, td),
               "bound": bound, "speedup_bound": ts / bound, "T_serial_fi": fis, "T_pod": pod,
               "pod_vs_fi_serial": fis / pod, "pod_vs_tl_serial": ts / pod,
               "t_P_fi": v["prefill_fi"]["t_iter_us"], "t_D_fi": v["decode_fi"]["t_iter_us"],
               "mhz": {k: vv["clock_mhz"] for k, vv in v.items()}, "power_w": {k: vv["power_w"] for k, vv in v.items()},
               "energy_mj": {k: vv.get("energy_mj_per_iter") for k, vv in v.items()},
               "cfg_prefill": r["cfg_prefill"], "cfg_decode": r["cfg_decode"], "fi_decode_path": r["fi_decode_path"],
               "pod_side_stream_bitwise_eq_default": r["pod_side_stream_bitwise_eq_default"],
               "guard_clean": r["guard"]["clean"]}
        t5[p] = row
        md.append(f"| {p} | {fmt(tp)} | {fmt(td)} | {tp / td:.2f} | {fmt(ts)} | {ts / (tp + td):.3f} | {fmt(lb_tc)} | "
                  f"{fmt(lb_dram)} | {fmt(lb_pow)} | {fmt(max(tp, td))} | {ts / bound:.2f} | {fmt(fis)} | {fmt(pod)} | "
                  f"{fis / pod:.3f} | {ts / pod:.3f} | {fmt(v['serial']['clock_mhz'], 0)} / {fmt(v['pod']['clock_mhz'], 0)} | "
                  f"{fmt(v['serial']['power_w'], 0)} / {fmt(v['pod']['power_w'], 0)} |")
    out["T5_pairs"] = t5
    if t5:
        md.append("\nFlashInfer solos in the same runs (t_P_fi / t_D_fi, us) and TileLang/FlashInfer solo ratios:\n")
        md.append("| pair | FI prefill | FI decode (path) | TL/FI prefill | TL/FI decode | POD bitwise side==default |")
        md.append("|---|---|---|---|---|---|")
        for p, r in t5.items():
            md.append(f"| {p} | {fmt(r['t_P_fi'])} | {fmt(r['t_D_fi'])} ({r['fi_decode_path']}) | {r['t_P'] / r['t_P_fi']:.3f} | "
                      f"{r['t_D'] / r['t_D_fi']:.3f} | {r['pod_side_stream_bitwise_eq_default']} |")
    with open(os.path.join(C.OUT, "tables.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    with open(os.path.join(C.OUT, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
