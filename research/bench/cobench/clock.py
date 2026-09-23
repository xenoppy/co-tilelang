"""GPU-side SM clock probe.

NVML refreshes clocks only every ~500 ms on this driver (see nvml.py), which is far too
coarse to attribute per-rep variance to clock changes. ``ClockProbe`` runs one warp on a
side stream that logs (%globaltimer, clock64) every ``period_us`` until the host stops
it; d(clock64)/d(globaltimer) is the SM clock. All SMs share the graphics clock domain,
so one SM is representative.

Per-rep clocks: CUDA event timestamps are mapped onto %globaltimer through an anchor
event recorded right before the probe launch (error: the probe's launch latency, a few
microseconds), and cycles are interpolated at the rep's start/end.

Cost: one resident warp (32 threads, no smem) that mostly sleeps. Must be started after
the workload has been warmed up: while it spins, a lazy module load (first launch of a
kernel in a context) would need a context-wide sync and stall until the probe's timeout.

Shared-memory carveout (methodology v1, 2026-09-23): the probe kernel requests the maximum
shared-memory carveout. With the driver's default choice the probe's SM was usually
configured for a small carveout, and since an SM can only change its carveout while idle,
that SM could not host any CTA needing ~>50 KB of smem for as long as the probe ran: every
large-smem kernel measured with clock=True ran on 187 SMs. For a static persistent grid
(one CTA per SM, fixed tiles) that made one CTA start only after another exited: the
P1-S "static persistent penalty" of large-smem kernels (GEMM 1.2-1.85x, looping decode up to
1.8x) was this artefact (research/results/2026-09-23_methodology_v1, M4).
"""
from __future__ import annotations

import functools
import time

import numpy as np
import torch

from .cudrv import CudaKernel, device_index

_SRC = r"""
__device__ __forceinline__ unsigned long long gt_() {
  unsigned long long r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r; }
extern "C" __global__ void clock_probe(const volatile unsigned long long* stop,
                                       unsigned long long* buf, int cap,
                                       unsigned long long period_ns,
                                       unsigned long long timeout_ns,
                                       unsigned long long* info) {
  if (threadIdx.x != 0) return;
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const unsigned long long t0 = gt_();
  unsigned long long t = t0;
  int n = 0;
  for (;;) {
    t = gt_();
    long long c = clock64();
    if (n < cap) { buf[2 * n] = t; buf[2 * n + 1] = (unsigned long long)c; ++n; }
    if (*stop != 0ull || n >= cap || t - t0 > timeout_ns) break;
    const unsigned long long tw = t;
    do { __nanosleep(1000); } while (gt_() - tw < period_ns);
  }
  info[0] = n; info[1] = smid; info[2] = (n >= cap); info[3] = (t - t0 > timeout_ns);
}
"""


@functools.lru_cache(maxsize=None)
def _kernel(device: int) -> CudaKernel:
    k = CudaKernel(_SRC, "clock_probe", "ppiQQp", device=device)
    k.set_carveout(100)   # never keep an SM from hosting a large-smem CTA (see module doc)
    return k


@functools.lru_cache(maxsize=None)
def _stream(device: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device)


class ClockProbe:
    def __init__(self, device=None, period_us: float = 50.0, capacity: int = 400_000,
                 timeout_s: float = 300.0):
        self.device = device_index(device)
        self.k = _kernel(self.device)
        self.period_ns = int(period_us * 1e3)
        self.cap = int(capacity)
        self.timeout_ns = int(timeout_s * 1e9)
        dev = f"cuda:{self.device}"
        self.buf = torch.zeros(2 * self.cap, dtype=torch.int64, device=dev)
        self.info = torch.zeros(4, dtype=torch.int64, device=dev)
        self.stop_flag = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self._stop_np = self.stop_flag.numpy()
        self.t = self.c = None

    def start(self) -> "ClockProbe":
        P = _stream(self.device)
        P.wait_stream(torch.cuda.current_stream(self.device))
        self._stop_np[0] = 0
        self.anchor = torch.cuda.Event(enable_timing=True)
        self.anchor.record(P)
        self.k(1, 32, self.stop_flag.data_ptr(), self.buf, self.cap, self.period_ns,
               self.timeout_ns, self.info, stream=P)
        return self

    def stop(self) -> "ClockProbe":
        # keep sampling a few periods past the last event so its (anchor-shifted) timestamp
        # is inside the trace
        time.sleep(max(4 * self.period_ns * 1e-9, 2e-4))
        self._stop_np[0] = 1
        _stream(self.device).synchronize()
        info = self.info.cpu().numpy()
        n = int(info[0])
        a = self.buf[: 2 * n].view(n, 2).cpu().numpy().astype(np.int64)
        self.t, self.c = a[:, 0], a[:, 1]
        self.smid, self.truncated, self.timed_out = int(info[1]), bool(info[2]), bool(info[3])
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- queries --------------------------------------------------------------
    def event_to_gt(self, e: torch.cuda.Event) -> float:
        """Map a completed CUDA event to %globaltimer ns (via the anchor)."""
        return float(self.t[0]) + self.anchor.elapsed_time(e) * 1e6

    def mhz_between_ns(self, t0: float, t1: float) -> float | None:
        if self.t is None or len(self.t) < 2 or t1 <= t0 or t0 < self.t[0] or t1 > self.t[-1]:
            return None
        c0, c1 = np.interp([t0, t1], self.t, self.c)
        return float((c1 - c0) / (t1 - t0) * 1e3)

    def mhz_between(self, e0: torch.cuda.Event, e1: torch.cuda.Event) -> float | None:
        return self.mhz_between_ns(self.event_to_gt(e0), self.event_to_gt(e1))

    def summary(self, bin_ms: float = 1.0) -> dict:
        if self.t is None or len(self.t) < 3:
            return {"n": 0 if self.t is None else int(len(self.t))}
        dt = np.diff(self.t).astype(np.float64)
        dc = np.diff(self.c).astype(np.float64)
        # aggregate into bins of ~bin_ms
        per = max(1, int(round(bin_ms * 1e6 / max(1.0, float(np.median(dt))))))
        m = (len(dt) // per) * per
        if m == 0:
            mhz = np.array([dc.sum() / dt.sum() * 1e3])
        else:
            mhz = dc[:m].reshape(-1, per).sum(1) / dt[:m].reshape(-1, per).sum(1) * 1e3
        return {
            "n_samples": int(len(self.t)), "smid": self.smid, "span_ms": float((self.t[-1] - self.t[0]) / 1e6),
            "sample_period_us_median": float(np.median(dt) / 1e3),
            "bin_ms": bin_ms, "n_bins": int(mhz.size),
            "mhz": {"mean": float(dc.sum() / dt.sum() * 1e3), "min": float(mhz.min()),
                    "p10": float(np.percentile(mhz, 10)), "median": float(np.median(mhz)),
                    "p90": float(np.percentile(mhz, 90)), "max": float(mhz.max())},
            "truncated": self.truncated, "timed_out": self.timed_out,
        }


def per_rep_clock(probe: ClockProbe, starts, ends, times_us=None) -> dict:
    """Clock (MHz) over each [start_i, end_i] and its correlation with the rep times."""
    mhz = [probe.mhz_between(a, b) for a, b in zip(starts, ends)]
    ok = [m for m in mhz if m is not None]
    out = {"n": len(ok), "covered": len(ok) == len(mhz)}
    if ok:
        arr = np.asarray(ok)
        out.update({"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max()),
                    "cv": float(arr.std(ddof=1) / arr.mean()) if arr.size > 1 else 0.0})
        if times_us is not None and len(ok) == len(times_us) and len(ok) > 2 and arr.std() > 0:
            out["corr_time_vs_clock"] = float(np.corrcoef(np.asarray(times_us), arr)[0, 1])
        if times_us is not None:
            # clock-normalised time: cycles = us * MHz (insensitive to clock drift)
            cyc = [t * m for t, m in zip(times_us, mhz) if m is not None]
            if cyc:
                out["cycles_median"] = float(np.median(cyc))
                out["cycles_cv"] = float(np.std(cyc, ddof=1) / np.mean(cyc)) if len(cyc) > 1 else 0.0
    out["values"] = [None if m is None else round(m, 1) for m in mhz]
    return out
