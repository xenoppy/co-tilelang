"""Tables and verdicts for sub-study A (claims C1, C5, C6) from the raw JSON.

    python research/bench/scripts/claims_gemm_report.py [--md out.md] [--json summary.json]

Verdict rules (README of the claims study, section 2.5; the thresholds for "partly" are ours):
  speed-up claim (C1 vs Triton, C5): reproduced if measured >= 0.9 x claimed; partly if below
      that but still a real speed-up (> 1.0); not reproduced otherwise.
  parity claim (C1 vs cuBLAS: "TileLang ~ cuBLAS", claimed band 0.9-1.15x): reproduced if
      measured >= 0.9 x the claimed per-shape ratio or measured >= 0.9 (inside or above the
      claimed band); partly if >= 0.8; not reproduced below.
The claimed per-shape ratios are the RTX 4090 bars of images/op_benchmark_consistent_gemm_fp16.png
(the closest card to sm_120: consumer-class, mma.sync, no wgmma) and the A100 bars of
images/op_benchmark_a100_wq_gemv.png, digitised by claims_gemm_figures.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from claims_gemm_common import ALL_SHAPES, C6_CLAIMED_TFLOPS, M_SHAPES, OUT_DIR, V_SHAPES  # noqa: E402

RAW = os.path.join(OUT_DIR, "raw")


def load(name):
    p = os.path.join(RAW, name)
    return json.load(open(p)) if os.path.exists(p) else None


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def verdict_speedup(meas, claimed):
    if meas is None:
        return "not measured"
    if meas >= 0.9 * claimed:
        return "reproduced"
    return "partly" if meas > 1.0 else "not reproduced"


def verdict_parity(meas, claimed, band_lo=0.9):
    if meas is None:
        return "not measured"
    if meas >= 0.9 * claimed or meas >= band_lo:
        return "reproduced"
    return "partly" if meas >= 0.8 else "not reproduced"


def f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def c1_tables(claimed, out, md):
    cl = claimed["C1_gemm_fp16"]["speedup_vs_cublas"]
    for suite, ref, tri_names in (("fp32acc", "cublas", ["triton_tut03_nt"]),
                                  ("fp16acc", "cublas_fp16acc", ["triton_tlbench_fp16acc_nt", "triton_tlbench_fp16acc_nn"])):
        rows, meas_c, meas_t = [], [], []
        md.append(f"\n#### C1, suite `{suite}` (primary timer: cobench clean flush, median of interleaved reps)\n")
        hdr = ["shape", "M,N,K", "TL µs", f"{ref} µs"] + [f"{t} µs" for t in tri_names] + [
            "TL/cuBLAS", "claimed (4090; A100, H100)", "verdict", "TL/Triton", "claimed TL/Triton (4090)", "verdict",
            "TL TFLOPS", "clock TL/cuBLAS MHz", "cycles cuBLAS/TL"]
        md.append("| " + " | ".join(hdr) + " |")
        md.append("|" + "---|" * len(hdr))
        for sh in M_SHAPES:
            d = load(f"{suite}_{sh}.json")
            c = cl[sh]
            claim_c = c["TileLang-RTX4090"]
            claim_t = c["TileLang-RTX4090"] / c["Triton-RTX4090"]
            if d is None or "summary" not in d:
                md.append(f"| {sh} | {ALL_SHAPES[sh]} | not measured |" + " |" * (len(hdr) - 3))
                continue
            s = d["summary"]
            tl, cu = s.get("tl"), s.get(ref)
            tris = [s.get(t) for t in tri_names]
            r_c = cu["median_us"] / tl["median_us"] if (tl and cu) else None
            tri_best = min((t for t in tris if t), key=lambda t: t["median_us"], default=None)
            r_t = tris[0]["median_us"] / tl["median_us"] if (tl and tris[0]) else None  # same layout
            cyc = (cu["median_us"] * cu["clock_mhz"]) / (tl["median_us"] * tl["clock_mhz"]) if (
                tl and cu and tl["clock_mhz"] and cu["clock_mhz"]) else None
            meas_c.append(r_c)
            meas_t.append(r_t)
            row = {"shape": sh, "MNK": ALL_SHAPES[sh], "tl_us": tl["median_us"], "ref_us": cu["median_us"],
                   "triton_us": {t: (v["median_us"] if v else None) for t, v in zip(tri_names, tris)},
                   "tl_vs_cublas": r_c, "claimed_vs_cublas_4090": claim_c,
                   "claimed_vs_cublas_A100": c["TileLang-A100"], "claimed_vs_cublas_H100": c["TileLang-H100"],
                   "verdict_vs_cublas": verdict_parity(r_c, claim_c),
                   "tl_vs_triton_same_layout": r_t, "claimed_vs_triton_4090": claim_t,
                   "verdict_vs_triton": verdict_speedup(r_t, claim_t),
                   "tl_tflops": tl["tflops"], "clock_tl": tl["clock_mhz"], "clock_ref": cu["clock_mhz"],
                   "cycle_ratio_ref_over_tl": cyc,
                   "tl_best_config": d["tl_tune"]["best_config"],
                   "correct": all(v["pass"] for v in d["correctness"].values()),
                   "secondary": d.get("secondary")}
            if tri_best is not None and len(tri_names) > 1:
                row["tl_vs_best_triton"] = tri_best["median_us"] / tl["median_us"]
            rows.append(row)
            md.append("| " + " | ".join([
                sh, ",".join(map(str, ALL_SHAPES[sh])), f(tl["median_us"], 1), f(cu["median_us"], 1)] +
                [f(t["median_us"], 1) if t else "-" for t in tris] + [
                f(r_c), f"{claim_c:.2f} ({c['TileLang-A100']:.2f}, {c['TileLang-H100']:.2f})", row["verdict_vs_cublas"],
                f(r_t), f"{claim_t:.2f}", row["verdict_vs_triton"], f(tl["tflops"], 0),
                f"{f(tl['clock_mhz'], 0)}/{f(cu['clock_mhz'], 0)}", f(cyc)]) + " |")
        gc, gt = gmean(meas_c), gmean(meas_t)
        md.append(f"\nGeometric mean TL/cuBLAS = {f(gc)}, TL/Triton (same layout) = {f(gt)} over "
                  f"{len([x for x in meas_c if x])} shapes. Paper (Sec. 5, RTX 4090): 1.10x over the vendor "
                  f"library, 1.08x over Triton; digitised 4090 bars: gmean "
                  f"{f(gmean([cl[s]['TileLang-RTX4090'] for s in M_SHAPES]))} vs cuBLAS, "
                  f"{f(gmean([cl[s]['TileLang-RTX4090'] / cl[s]['Triton-RTX4090'] for s in M_SHAPES]))} vs Triton.")
        # secondary timers
        md.append(f"\nSecondary timers ({suite}): TL/cuBLAS and TL/Triton ratios with tilelang.profiler.do_bench(event) "
                  "and triton.testing.do_bench (both: mean over a time budget, 256 MB L2 flush between calls)\n")
        md.append("| shape | TL/cuBLAS tl.do_bench | TL/cuBLAS triton.do_bench | TL/Triton tl.do_bench | TL/Triton triton.do_bench | autotuner TL latency ms |")
        md.append("|---|---|---|---|---|---|")
        for r in rows:
            sec = r["secondary"] or {}
            d = load(f"{suite}_{r['shape']}.json")
            try:
                a = sec[ref]["tilelang_do_bench_event_ms"] / sec["tl"]["tilelang_do_bench_event_ms"]
                b = sec[ref]["triton_do_bench_ms"] / sec["tl"]["triton_do_bench_ms"]
                c_ = sec[tri_names[0]]["tilelang_do_bench_event_ms"] / sec["tl"]["tilelang_do_bench_event_ms"]
                d_ = sec[tri_names[0]]["triton_do_bench_ms"] / sec["tl"]["triton_do_bench_ms"]
            except (KeyError, TypeError):
                a = b = c_ = d_ = None
            r["secondary_ratios"] = {"tl_vs_cublas_tl_do_bench": a, "tl_vs_cublas_triton_do_bench": b,
                                     "tl_vs_triton_tl_do_bench": c_, "tl_vs_triton_triton_do_bench": d_}
            md.append(f"| {r['shape']} | {f(a)} | {f(b)} | {f(c_)} | {f(d_)} | {f(d['tl_tune']['autotuner_latency_ms'], 4)} |")
        out[f"C1_{suite}"] = {"rows": rows, "gmean_tl_vs_cublas": gc, "gmean_tl_vs_triton": gt}
        if suite == "fp32acc":
            md.append("\nexamples/gemm/example_gemm_autotune.py default (heuristic) config on sm_120, same run:\n")
            md.append("| shape | config | µs | vs cuBLAS | vs autotuned TL |")
            md.append("|---|---|---|---|---|")
            for sh in M_SHAPES:
                d = load(f"{suite}_{sh}.json")
                if not d or "tl_example_heuristic" not in d.get("summary", {}):
                    continue
                h = d["summary"]["tl_example_heuristic"]
                md.append(f"| {sh} | {d['tl_example_heuristic']['config']} | {f(h['median_us'], 1)} | "
                          f"{f(h['speedup_vs_cublas'])} | {f(d['summary']['tl']['median_us'] / h['median_us'])} |")


def c6_tables(out, md):
    for suite, ref, key in (("fp32acc", "cublas", "fp16"), ("fp8", "cublaslt_scaled_mm", "fp8")):
        md.append(f"\n#### C6 {key} GEMM, M = N = 8192 (suite `{suite}`; K = 8192 is shape M5 for fp16)\n")
        md.append(f"| K | TL µs | {ref} µs | TL TFLOPS | {ref} TFLOPS | TL/{ref} | TL clock MHz | TL % of mma peak at that clock | claimed H800 TL TFLOPS | TL here / H800 claim |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for k in (256, 512, 1024, 2048, 4096, 8192, 16384):
            sh = f"K{k}" if not (suite == "fp32acc" and k == 8192) else "M5"
            d = load(f"{suite}_{sh}.json")
            claim = C6_CLAIMED_TFLOPS[key][k]
            if not d or "summary" not in d:
                md.append(f"| {k} | not measured | | | | | | {claim} | |")
                continue
            s = d["summary"]
            tl, cu = s.get("tl"), s.get(ref)
            r = cu["median_us"] / tl["median_us"] if (tl and cu) else None
            rows.append({"K": k, "shape": sh, "tl_us": tl["median_us"] if tl else None, "ref_us": cu["median_us"] if cu else None,
                         "tl_tflops": tl["tflops"] if tl else None, "ref_tflops": cu["tflops"] if cu else None,
                         "tl_vs_ref": r, "claimed_h800_tflops": claim,
                         "tl_best_config": d["tl_tune"]["best_config"],
                         "correct": {n: v["pass"] for n, v in d["correctness"].items()},
                         "scaled_mm_out_dtype": d.get("scaled_mm_out_dtype")})
            # dense mma.sync peak per SM per clock: 1024 FLOP (fp16/bf16, fp32 acc; cobench mma_peak),
            # 2048 for e4m3 (twice the fp16 rate on the same tensor cores)
            per_clk = 1024 if key == "fp16" else 2048
            frac = (tl["tflops"] * 1e12 / (188 * per_clk * tl["clock_mhz"] * 1e6)) if (tl and tl["clock_mhz"]) else None
            rows[-1]["tl_frac_of_peak_at_clock"] = frac
            md.append(f"| {k} | {f(tl['median_us'], 1) if tl else '-'} | {f(cu['median_us'], 1) if cu else '-'} | "
                      f"{f(tl['tflops'], 0) if tl else '-'} | {f(cu['tflops'], 0) if cu else '-'} | {f(r)} | "
                      f"{f(tl['clock_mhz'], 0) if tl else '-'} | {f(frac * 100, 0) if frac else '-'} | {claim} | "
                      f"{f(tl['tflops'] / claim) if tl else '-'} |")
        out[f"C6_{key}"] = rows


C5_COMBOS = ["W_INT4A_FP16", "W_INT2A_FP16", "W_INT2A_INT8", "W_NF4A_FP16"]
# claim format -> TileLang example-generator variant that can run on sm_120 (None: no such path)
C5_TL = {"W_INT4A_FP16": "tl_uint4_fp16", "W_INT2A_FP16": "tl_int2_fp16", "W_INT2A_INT8": "tl_uint2_int8",
         "W_NF4A_FP16": None}


def c5_tables(claimed, out, md):
    cl = claimed["C5_gemv_a100"]["speedup_vs_cublas_fp16"]
    # BitBLAS status (the claim's artifact)
    md.append("\n#### C5 BitBLAS 0.1.0.post1 (the artifact behind the claim) on sm_120\n")
    for tag in ("", "_ada_target"):
        b = load(f"gemv_bitblas_V0{tag}.json")
        if not b:
            md.append(f"- V0{tag}: not run")
            continue
        ops = b.get("bitblas_ops", {})
        st = "; ".join(f"{c}: " + (("BUILD FAILED - " + v["build_error"].splitlines()[-1][:160]) if "build_error" in v
                                   else "built") for c, v in ops.items())
        md.append(f"- target `{b.get('bitblas_target')}`: {st}")
    md.append("\n#### C5 dequant GEMV vs cuBLAS fp16 GEMV, TileLang's own example generator "
              "(m = 1; cobench clean flush, all variants of a shape interleaved)\n")
    hdr = ["shape", "N,K", "cuBLAS fp16acc µs (weight GB/s)"]
    for c in C5_COMBOS[:3]:
        hdr += [f"{C5_TL[c]} µs (GB/s)", "speed-up", f"claimed {c} (BitBLAS, A100)", "verdict"]
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "---|" * len(hdr))
    rows = []
    for sh in V_SHAPES:
        t = load(f"gemv_tl_{sh}.json")
        claim = cl.get(sh)
        row = {"shape": sh, "NK": V_SHAPES[sh][1:]}
        cells = [sh, f"{V_SHAPES[sh][1]},{V_SHAPES[sh][2]}"]
        if not (t and "summary" in t):
            md.append("| " + " | ".join(cells + ["not measured"]) + " |")
            continue
        cu = t["summary"]["cublas_fp16acc"]
        row["cublas_fp16acc_us"], row["cublas_fp16acc_GBps"] = cu["median_us"], cu["weight_GBps"]
        row["cublas_fp32acc_us"] = t["summary"].get("cublas", {}).get("median_us")
        cells.append(f"{f(cu['median_us'], 1)} ({f(cu['weight_GBps'], 0)})")
        for c in C5_COMBOS[:3]:
            v = t["summary"].get(C5_TL[c])
            cc = claim.get("BitBLAS-TileLang-" + c) if claim else None
            m = v["speedup_vs_cublas_fp16acc"] if v else None
            if v is None:
                info = t.get("combos", {}).get(C5_TL[c][3:], {})
                vd = "build failed" if "build_error" in info else (
                    "correctness failed" if C5_TL[c] in str(t.get("correctness_failed")) else "not measured")
            else:
                vd = verdict_speedup(m, cc) if cc else "no claim (V7 not in figure)"
            row[c] = {"variant": C5_TL[c], "us": v["median_us"] if v else None, "speedup": m, "claimed": cc,
                      "verdict": vd, "GBps": v["weight_GBps"] if v else None,
                      "rel_err": (t["correctness"].get(C5_TL[c]) or {}).get("rel_err_max")}
            cells += [f"{f(v['median_us'], 1)} ({f(v['weight_GBps'], 0)})" if v else "-", f(m, 2), f(cc, 2), vd]
        row["W_NF4A_FP16"] = {"verdict": "not testable on sm_120 (BitBLAS cannot build; no NF4 path in the TileLang example)",
                              "claimed": claim.get("BitBLAS-TileLang-W_NF4A_FP16") if claim else None}
        rows.append(row)
        md.append("| " + " | ".join(cells) + " |")
    ms = {c: [r[c]["speedup"] for r in rows if r.get(c, {}).get("speedup") and r[c].get("claimed")] for c in C5_COMBOS[:3]}
    cs = {c: [r[c]["claimed"] for r in rows if r.get(c, {}).get("speedup") and r[c].get("claimed")] for c in C5_COMBOS[:3]}
    md.append("\nGeometric means over V0-V6: " + "; ".join(
        f"{c}: measured {f(gmean(ms[c]), 2)} vs claimed {f(gmean(cs[c]), 2)}" for c in C5_COMBOS[:3]))
    out["C5"] = {"rows": rows, "gmean_measured": {c: gmean(ms[c]) for c in ms},
                 "gmean_claimed": {c: gmean(cs[c]) for c in cs}}


def steady_tables(claimed, out, md):
    """Steady-state (cobench.bench_steady) cross-check next to the flush-mode ratios."""
    cl = claimed["C1_gemm_fp16"]["speedup_vs_cublas"]
    for suite, ref, tri in (("fp32acc", "cublas", "triton_tut03_nt"),
                            ("fp16acc", "cublas_fp16acc", "triton_tlbench_fp16acc_nt")):
        shapes = [s for s in list(M_SHAPES) + [k for k in ALL_SHAPES if k.startswith("K")]
                  if load(f"steady_{suite}_{s}.json")]
        if not shapes:
            continue
        md.append(f"\n#### Steady state (bench_steady, back-to-back, rotation, thermal plateau), suite `{suite}`\n")
        md.append("| shape | TL µs | cuBLAS µs | Triton µs | TL/cuBLAS steady | TL/cuBLAS flush | TL/Triton steady | TL/Triton flush | "
                  "MHz TL/cuBLAS/Triton | W TL/cuBLAS/Triton | mJ/iter TL/cuBLAS/Triton | Mcycles/iter TL/cuBLAS/Triton | "
                  "cycles cuBLAS/TL | energy cuBLAS/TL | nJ/cycle TL/cuBLAS | verdict vs cuBLAS (steady) | verdict vs Triton (steady) |")
        md.append("|" + "---|" * 17)
        rows = []
        for sh in shapes:
            d = load(f"steady_{suite}_{sh}.json")
            f_ = load(f"{suite}_{sh}.json" if not (suite == "fp32acc" and sh == "K8192") else f"{suite}_M5.json")
            s = d["summary"]
            t, c, r = s["tl"], s[ref], s[tri]
            rc, rt = c["t_iter_us"] / t["t_iter_us"], r["t_iter_us"] / t["t_iter_us"]
            fs = f_["summary"] if f_ else {}
            frc = fs[ref]["median_us"] / fs["tl"]["median_us"] if fs else None
            frt = fs[tri]["median_us"] / fs["tl"]["median_us"] if (fs and tri in fs) else None
            cyc = c["kcycles_per_iter"] / t["kcycles_per_iter"]
            en = c["energy_mj_per_iter"] / t["energy_mj_per_iter"]
            nj = (t["power_w"] / t["clock_mhz"] * 1e3, c["power_w"] / c["clock_mhz"] * 1e3)
            if sh in cl:
                claim_c = cl[sh]["TileLang-RTX4090"]
                claim_t = cl[sh]["TileLang-RTX4090"] / cl[sh]["Triton-RTX4090"]
                vc, vt = verdict_parity(rc, claim_c), verdict_speedup(rt, claim_t)
            else:
                vc = vt = "n/a (C6)"
            row = {"shape": sh, "tl_us": t["t_iter_us"], "ref_us": c["t_iter_us"], "triton_us": r["t_iter_us"],
                   "tl_vs_cublas_steady": rc, "tl_vs_cublas_flush": frc, "tl_vs_triton_steady": rt,
                   "tl_vs_triton_flush": frt, "clock": [t["clock_mhz"], c["clock_mhz"], r["clock_mhz"]],
                   "power": [t["power_w"], c["power_w"], r["power_w"]],
                   "energy_mj": [t["energy_mj_per_iter"], c["energy_mj_per_iter"], r["energy_mj_per_iter"]],
                   "kcycles": [t["kcycles_per_iter"], c["kcycles_per_iter"], r["kcycles_per_iter"]],
                   "cycle_ratio_ref_over_tl": cyc, "energy_ratio_ref_over_tl": en, "nj_per_cycle_tl_ref": nj,
                   "tl_tflops": t["tflops"], "verdict_vs_cublas": vc, "verdict_vs_triton": vt,
                   "guard_clean": d["cobench_steady"]["guard"]["clean"],
                   "same_device_source_as_flush_run": d.get("same_device_source_as_flush_run")}
            rows.append(row)
            md.append(f"| {sh} | {t['t_iter_us']:.1f} | {c['t_iter_us']:.1f} | {r['t_iter_us']:.1f} | {rc:.3f} | {f(frc)} | "
                      f"{rt:.3f} | {f(frt)} | {t['clock_mhz']:.0f}/{c['clock_mhz']:.0f}/{r['clock_mhz']:.0f} | "
                      f"{t['power_w']:.0f}/{c['power_w']:.0f}/{r['power_w']:.0f} | "
                      f"{t['energy_mj_per_iter']:.1f}/{c['energy_mj_per_iter']:.1f}/{r['energy_mj_per_iter']:.1f} | "
                      f"{t['kcycles_per_iter'] / 1e3:.3f}/{c['kcycles_per_iter'] / 1e3:.3f}/{r['kcycles_per_iter'] / 1e3:.3f} | "
                      f"{cyc:.3f} | {en:.3f} | {nj[0]:.3f}/{nj[1]:.3f} | {vc} | {vt} |")
        mrows = [x for x in rows if x["shape"] in M_SHAPES]
        if mrows:
            md.append(f"\nC1 shapes, geometric means (steady): TL/cuBLAS {f(gmean([x['tl_vs_cublas_steady'] for x in mrows]))} "
                      f"(flush {f(gmean([x['tl_vs_cublas_flush'] for x in mrows]))}), TL/Triton "
                      f"{f(gmean([x['tl_vs_triton_steady'] for x in mrows]))} (flush {f(gmean([x['tl_vs_triton_flush'] for x in mrows]))}).")
        out[f"steady_{suite}"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=os.path.join(OUT_DIR, "summary.json"))
    a = ap.parse_args()
    claimed = json.load(open(os.path.join(OUT_DIR, "claimed_ratios_digitized.json")))
    out, md = {}, []
    c1_tables(claimed, out, md)
    c6_tables(out, md)
    c5_tables(claimed, out, md)
    steady_tables(claimed, out, md)
    text = "\n".join(md)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
