"""Reproducibility check (P0-2 criterion 6).

Runs bf16 matmul 4096^3 with cobench.bench(mode="graph") in N *separate processes*
(sequentially) and reports the CV of the per-run medians, the SM clock of each run
(ClockProbe), and the clock-normalised time (cycles per call = us * MHz), which
separates clock drift from everything else.

    python research/bench/scripts/repro_matmul.py --runs 5 [--inproc 1] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def child(inproc: int, size: int) -> list[dict]:
    import torch
    import cobench as cb
    M = N = K = size

    def mk():
        return (torch.randn(M, K, device="cuda", dtype=torch.bfloat16),
                torch.randn(K, N, device="cuda", dtype=torch.bfloat16),
                torch.empty(M, N, device="cuda", dtype=torch.bfloat16))

    out = []
    for _ in range(inproc):
        r = cb.bench(lambda a, b, c: torch.matmul(a, b, out=c), make_inputs=mk, mode="graph",
                     flops=2 * M * N * K, nvml=True, clock=True, label=f"mm{size}")
        per = r.clock["per_rep"]
        cyc = [t * m for t, m in zip(r.samples, per["values"]) if m is not None]
        out.append({
            "median_us": r.median, "p10_us": r.p10, "p90_us": r.p90, "cv_within": r.cv,
            "tflops": r.tflops, "n": r.n, "k": r.k, "n_copies": r.n_copies,
            "clock_rep_mhz": {k: per[k] for k in ("min", "median", "max") if k in per},
            "clock_window_mhz": r.clock["window"]["mhz"],
            "corr_time_vs_clock": per.get("corr_time_vs_clock"),
            "cycles_per_call_median": float(np.median(cyc)) if cyc else None,
            "power_w": r.nvml["power_w"], "temp_c": r.nvml["temp_c"],
            "throttle": r.nvml["throttle"]["reasons_seen"],
            "wall": time.time(),
        })
    return out


def cv(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.std(ddof=1) / a.mean()) if a.size > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--inproc", type=int, default=1, help="bench() calls per process")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--gap-s", type=float, default=0.0, help="idle gap between processes")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.child:
        print("RESULT " + json.dumps(child(args.inproc, args.size)), flush=True)
        return
    runs = []
    for i in range(args.runs):
        p = subprocess.run([sys.executable, __file__, "--child", "--inproc", str(args.inproc),
                            "--size", str(args.size)], capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if p.returncode != 0 or not line:
            raise RuntimeError(f"child {i} failed:\n{p.stdout}\n{p.stderr}")
        rs = json.loads(line[0][7:])
        for r in rs:
            r["process"] = i
            runs.append(r)
            print(f"run {i}: median {r['median_us']:.2f}us ({r['tflops']:.1f} TFLOP/s) cv_within "
                  f"{r['cv_within']*100:.2f}% clock {r['clock_rep_mhz'].get('median', 0):.0f} MHz "
                  f"cycles/call {r['cycles_per_call_median']/1e3:.1f}k T {r['temp_c']['max']:.0f}C "
                  f"P {r['power_w']['mean']:.0f}W {r['throttle']}", flush=True)
        if args.gap_s:
            time.sleep(args.gap_s)
    meds = [r["median_us"] for r in runs]
    summary = {
        "runs": len(runs), "processes": args.runs, "inproc": args.inproc, "size": args.size,
        "cv_across_runs": cv(meds),
        "median_us": {"min": min(meds), "max": max(meds), "mean": float(np.mean(meds))},
        "clock_median_mhz": [r["clock_rep_mhz"].get("median") for r in runs],
        "cv_clock_across_runs": cv([r["clock_rep_mhz"]["median"] for r in runs]),
        "cv_cycles_per_call_across_runs": cv([r["cycles_per_call_median"] for r in runs]),
        "temps_max_c": [r["temp_c"]["max"] for r in runs],
        "power_mean_w": [r["power_w"]["mean"] for r in runs],
        "per_run": runs,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_run"}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    return summary


if __name__ == "__main__":
    main()
