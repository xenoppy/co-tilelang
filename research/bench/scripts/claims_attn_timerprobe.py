"""What is the resolution of cobench's flush-mode times on this GPU?

cobench.bench / bench_variants time a rep with two CUDA events (torch.cuda.Event.elapsed_time).
C2 medians clustered on multiples of ~2.048 us. This probe separates event-timestamp
quantisation from real kernel-duration quantisation: a kernel whose duration is set by spinning
on %globaltimer (resolution 32 ns on this card, research/results/2026-09-22_smid_probe) is swept
in 128 ns steps and timed (a) with cobench.bench(mode="flush") exactly like the C2 runs, and
(b) in-kernel (%globaltimer at the first CTA start / last CTA end). With N back-to-back launches
per rep the tick is amortised over N kernels.

  flock <gpu.lock> python research/bench/scripts/claims_attn_timerprobe.py
  -> research/results/2026-09-24_claims_repro/B_attention/raw/timerprobe.json
"""
from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_attn_common as C  # noqa: E402

SRC = r"""
extern "C" __global__ void spinc(unsigned long long cyc, unsigned long long* out) {
    unsigned long long g0, g1, c0, c;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    c0 = clock64();
    do { c = clock64(); } while (c - c0 < cyc);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    if (threadIdx.x == 0 && blockIdx.x == 0) { out[0] = g0; out[1] = g1; out[2] = c - c0; }
}
extern "C" __global__ void spin(unsigned long long dur_ns, unsigned long long* out) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < dur_ns);
    if (threadIdx.x == 0) { out[2 * blockIdx.x] = t0; out[2 * blockIdx.x + 1] = t; }
}
"""


def main():
    import torch
    res = {"wait": C.wait_gpu(), "env": C.env_info()}
    import cobench as cb
    k = cb.CudaKernel(SRC, "spin", "Qp")
    G = torch.cuda.get_device_properties(0).multi_processor_count
    out = torch.zeros(2 * G, dtype=torch.int64, device="cuda")
    rows = []
    for dur in range(16000, 26001, 128):
        def one(dur=dur):
            k(G, 32, dur, out)
        r = cb.bench(one, mode="flush", reps=50, warmup_s=0.3, keep_samples=True, label=f"spin{dur}")
        one()
        torch.cuda.synchronize()
        o = out.view(G, 2).cpu()
        span = (int(o[:, 1].max()) - int(o[:, 0].min())) / 1e3
        s = sorted(r.samples)
        rows.append({"dur_ns": dur, "event_median_us": r.median, "event_min_us": s[0], "event_max_us": s[-1],
                     "event_distinct": len({round(x, 3) for x in s}), "kernel_span_us": span})
        print(f"{dur / 1e3:7.3f} us spin: event median {r.median:7.3f} [{s[0]:7.3f}, {s[-1]:7.3f}]  "
              f"in-kernel span {span:7.3f}", flush=True)
    res["single"] = rows
    # N back-to-back launches in one rep
    rows_n = []
    for n in (8,):
        for dur in range(16000, 26001, 512):
            def many(dur=dur, n=n):
                for _ in range(n):
                    k(G, 32, dur, out)
            r = cb.bench(many, mode="flush", reps=50, warmup_s=0.3, label=f"spin{dur}x{n}")
            rows_n.append({"n": n, "dur_ns": dur, "event_median_per_call_us": r.median / n})
            print(f"{dur / 1e3:7.3f} us spin x{n}: per call {r.median / n:7.3f}", flush=True)
    res["back_to_back"] = rows_n
    # (c) clock64-timed spin: is %globaltimer linear in real time? (cycles / MHz vs globaltimer)
    kc = cb.CudaKernel(SRC, "spinc", "Qp")
    rows_c = []
    for cyc in range(40000, 70001, 5000):
        r = cb.bench(lambda cyc=cyc: kc(G, 32, cyc, out), mode="flush", reps=50, warmup_s=0.3, clock=True)
        kc(G, 32, cyc, out)
        torch.cuda.synchronize()
        o = out.cpu()
        mhz = r.clock["per_rep"]["median"]
        rows_c.append({"cycles": cyc, "event_median_us": r.median, "globaltimer_span_us": (int(o[1]) - int(o[0])) / 1e3,
                       "cycles_over_mhz_us": cyc / mhz, "mhz": mhz})
        print(rows_c[-1], flush=True)
    res["clock64_spin"] = rows_c
    # (d) the same %globaltimer spin, 8 calls inside one CUDA graph (mode="hot", k=8)
    rows_g = []
    for dur in range(16000, 26001, 512):
        r = cb.bench(lambda dur=dur: k(G, 32, dur, out), make_inputs=lambda: (), mode="hot", k=8, reps=50,
                     warmup_s=0.3)
        rows_g.append({"dur_ns": dur, "graph_per_call_us": r.median})
        print(f"{dur / 1e3:7.3f} us spin in graph: per call {r.median:7.3f}", flush=True)
    res["graph_back_to_back"] = rows_g
    # (e) event resolution: two events around nothing
    r = cb.bench(lambda: None, mode="flush", reps=100, warmup_s=0.3)
    res["empty_rep"] = {"median_us": r.median, "distinct": sorted({round(x, 3) for x in r.samples})}
    print("empty rep", res["empty_rep"], flush=True)
    # staircase summary: fit event median vs in-kernel span
    d = [x["event_median_us"] - x["kernel_span_us"] for x in rows]
    res["summary"] = {"event_minus_span_us": {"min": min(d), "median": statistics.median(d), "max": max(d)},
                      "distinct_event_medians": sorted({round(x["event_median_us"], 2) for x in rows})}
    C.save_json(res, os.path.join(C.RESULTS, "raw", "timerprobe.json"))


if __name__ == "__main__":
    main()
