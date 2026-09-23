"""Steady-state multi-variant measurement: ``bench_steady``.

Protocol (research/plan.md §1.2, v0.3: the primary mode for co-run comparisons):

* Every *variant* (see ``cobench.variants``) enqueues one iteration of work on the launcher
  stream, e.g. ``serial`` = A then B, ``streams`` = Par(A on s1, B on s2), ``green`` = Par(A on
  an n-SM green stream, B on the complement), ``cokernel`` = one CoKernel launch.
* A variant runs **back-to-back** (no flush, no host gaps: the host stays ``lead_ms`` ahead of
  the GPU and every iteration is checked for a drained queue) for one *slice* of wall time
  (``slice_s``); inputs rotate over enough copies (``cobench.Rotation``, > 2x L2) that every
  iteration reads DRAM-cold inputs without a flush.
* Slices of all variants are interleaved in rotating order for ``rounds`` rounds (slice
  boundaries are also gap-free), after a round-robin warmup. The first ``settle_s`` of every
  slice is excluded so the power controller (0.1-0.3 s) and the clock have settled.
* Per slice: mean time per iteration over the measured part (= throughput), per-iteration
  median/p10/p90, SM clock (``ClockProbe``, GPU-side), board power (NVML 20 ms samples) and
  temperature. Per variant: median over slices. Derived: speedup of every variant vs the
  ``reference`` variant measured in the same run (ratio of medians, and median of per-round
  paired ratios). Par variants also report per-op completion times (end of each op relative
  to the iteration start).
* GPU guard: waits until no foreign process shows SM% > 0 before starting; after the run,
  keeps the GPU loaded until pmon has covered the whole window, and re-runs everything if a
  foreign process was active during any slice.
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch

from .clock import ClockProbe
from .cudrv import device_index
from .guard import SLACK_S, get_guard
from .nvml import NvmlDevice, NvmlSampler, gpu_state
from .timing import CV_TARGET, _side_stream
from .variants import Par, _nargs


class HostGapError(RuntimeError):
    pass


class _EventPool:
    """Recycled timing events (torch creates the CUDA handle lazily on first record, which
    costs tens of µs of host time: pre-create and reuse)."""

    def __init__(self, stream, n: int = 0):
        self.free: list = []
        self.stream = stream
        self.grow(n)

    def grow(self, n: int):
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for e in evs:
            e.record(self.stream)
        self.free.extend(evs)

    def get(self):
        if not self.free:
            self.grow(256)
        return self.free.pop()

    def put(self, evs):
        self.free.extend(evs)


@dataclass
class _Slice:
    variant: str
    round: int
    kind: str                  # "warm" | "timed" | "tail"
    begin: object = None       # event recorded before the first iteration
    ends: list = field(default_factory=list)   # (event, n_iters_so_far)
    marks: list = field(default_factory=list)  # (iter index, e0, {op: end})
    gaps: int = 0
    host_t0: float = 0.0
    host_t1: float = 0.0
    n_iter: int = 0


@dataclass
class SteadyResult:
    label: str | None
    config: dict
    variants: dict            # name -> summary
    derived: dict
    slices: list              # per timed slice summaries
    clock: dict | None = None
    nvml: dict | None = None
    guard: dict | None = None
    gpu_state_before: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self):
        lines = [f"[{self.label or 'steady'}] rounds={self.config['rounds']} slice={self.config['slice_s']}s "
                 f"settle={self.config['settle_s']}s"]
        ref = self.config.get("reference")
        sp = self.derived.get("speedup", {})
        for n, v in self.variants.items():
            s = (f"  {n:24s} {v['t_iter_us']:9.1f} us/iter (cv {v['cv_slices'] * 100:.2f}%, "
                 f"{v['n_iter']} it) clk {v.get('clock_mhz') or 0:.0f} MHz P {v.get('power_w') or 0:.0f} W")
            if ref and n in sp:
                s += f" | x{sp[n]['ratio_of_medians']:.3f} vs {ref}"
            if v.get("ops"):
                s += " | " + " ".join(f"{k}:{o['median_us']:.1f}" for k, o in v["ops"].items())
            lines.append(s)
        return "\n".join(lines)


def _stats(a) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "mean": float(a.mean()),
            "cv": float(a.std(ddof=1) / a.mean()) if a.size > 1 and a.mean() > 0 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def bench_steady(variants: dict, *, reference: str | None = "serial", slice_s: float = 1.5,
                 settle_s: float = 0.4, rounds: int = 5, warmup_s: float = 3.0, thermal: bool = True,
                 thermal_window_s: float = 15.0, thermal_max_s: float = 240.0, lead_ms: float = 8.0,
                 min_group_us: float = 20.0, clock: bool = True, nvml: bool = True, guard: bool = True,
                 guard_attempts: int = 3, max_gaps: int = 0, label: str | None = None,
                 strict: bool = True, log=None) -> SteadyResult:
    """Steady-state interleaved measurement of ``variants`` (name -> callable(i) or callable()).

    reference     variant the speedups are computed against (None: no speedups).
    slice_s       wall time per slice; settle_s of it is excluded (power controller settling).
    rounds        slices per variant (rotating order); per-variant result = median over slices.
    warmup_s      round-robin warmup before the timed rounds (slices of <= 0.5 s), at least.
    thermal       continue the warmup until the GPU temperature has plateaued: the max over
                  the last thermal_window_s is not above the max of the window before (1 C
                  NVML resolution), at most thermal_max_s. Power-capped kernels lose ~5% over
                  the first minutes of load as the GPU heats up (P1 smoke test: GEMM 403 ->
                  424 us over 2 min after a 5 s warmup).
    lead_ms       how far (GPU time) the host enqueues ahead of the GPU.
    min_group_us  iterations shorter than this are timed in groups (one event per group).
    max_gaps      tolerated drained-queue events inside measured windows (strict -> raise).
    guard         wait for a free GPU first; re-run everything if a foreign process showed
                  SM% > 0 during the run (up to guard_attempts runs).
    """
    if not variants:
        raise ValueError("no variants")
    if reference is not None and reference not in variants:
        raise ValueError(f"reference {reference!r} is not a variant")
    if settle_s >= slice_s:
        raise ValueError("settle_s must be < slice_s")
    say = log or (lambda *a: print(*a, flush=True))
    g = get_guard() if guard else None
    attempts = []
    for attempt in range(max(1, guard_attempts if guard else 1)):
        waited = g.wait_until_free(log=say) if g else 0.0
        res, window, slice_windows = _run(variants, reference, slice_s, settle_s, rounds, warmup_s,
                                          (thermal_window_s, thermal_max_s) if thermal else None,
                                          lead_ms, min_group_us, clock, nvml, g, max_gaps, label,
                                          strict, say)
        if g is None:
            return res
        chk = g.check(*window, wait=False)
        bad = [sw for sw in slice_windows if g.activity_between(sw[2], sw[3])]
        attempts.append({"waited_s": waited, "clean": chk["clean"], "complete": chk["complete"],
                         "active": chk["active"], "contaminated_slices": [(v, r) for v, r, *_ in bad]})
        res.guard = {"attempts": attempts, "clean": chk["clean"] and not bad,
                     "complete": chk["complete"], "summary": g.summary(window[0] - 2, window[1] + 2)}
        if chk["clean"] and not bad:
            if not chk["complete"]:
                say("[steady] warning: pmon did not cover the end of the run")
            return res
        say(f"[steady] foreign SM activity during the run ({chk['active']}); re-measuring")
    raise RuntimeError(f"bench_steady: every attempt was contaminated by foreign GPU work: {attempts}")


def _run(variants, reference, slice_s, settle_s, rounds, warmup_s, thermal, lead_ms, min_group_us,
         use_clock, use_nvml, g, max_gaps, label, strict, say):
    dev = device_index(None)
    caller = torch.cuda.current_stream(dev)
    L = _side_stream(dev, "steady")
    L.wait_stream(caller)
    names = list(variants)
    takes_i = {n: _nargs(f) >= 1 for n, f in variants.items()}
    state0 = gpu_state(dev) if use_nvml else None

    # ---- pre-warm (ungated, synchronizing: lazy module loads) + rough timings -------------
    it_counter = [0]

    def call(n):
        f = variants[n]
        if takes_i[n]:
            f(it_counter[0])
        else:
            f()
        it_counter[0] += 1

    est = {}
    host_cost = {}
    with torch.cuda.stream(L):
        for n in names:
            for _ in range(2):
                call(n)
        L.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for n in names:
            k = 3
            e0.record(L)
            h0 = time.perf_counter()
            for _ in range(k):
                call(n)
            h1 = time.perf_counter()
            e1.record(L)
            e1.synchronize()
            est[n] = e0.elapsed_time(e1) * 1e3 / k           # us per iteration (rough)
            host_cost[n] = (h1 - h0) * 1e6 / k
    for n in names:
        if host_cost[n] > 0.8 * est[n]:
            msg = (f"variant {n}: host enqueue {host_cost[n]:.1f} us/iter vs GPU {est[n]:.1f} us/iter: "
                   "the host cannot stay ahead (no back-to-back execution)")
            if strict:
                raise HostGapError(msg)
            say("[steady] warning: " + msg)
    group = {n: max(1, math.ceil(min_group_us / max(est[n], 1e-3))) for n in names}
    lead = {n: max(2, math.ceil(lead_ms * 1e3 / max(est[n] * group[n], 1e-3))) for n in names}

    # pre-create every event the timed slices will need (a pool that grows mid-slice costs
    # ~10 us of host time per event and could let the GPU drain)
    need = 0
    for n, f in variants.items():
        per_group = 1 + (len(f.names) + 1) * group[n] if isinstance(f, Par) else 1
        need += int(1.3 * rounds * slice_s * 1e6 / (est[n] * group[n]) * per_group) + 64
    pool = _EventPool(L, need + 4096)
    for n, f in variants.items():
        if isinstance(f, Par):
            f.set_mark_pool(pool.get)

    total_s = warmup_s + (thermal[1] if thermal else 0.0) + rounds * len(names) * slice_s
    probe = None
    if use_clock:
        period_us = 100.0
        probe = ClockProbe(dev, period_us=period_us,
                           capacity=int((total_s + 30) * 1e6 / period_us),
                           timeout_s=3 * total_s + 120)
    sampler = NvmlSampler(interval_ms=50.0, device=dev) if use_nvml else None

    state = {"last": None, "gaps": 0}

    def run_slice(n, rnd, kind, dur_s) -> _Slice:
        s = _Slice(variant=n, round=rnd, kind=kind)
        s.begin = pool.get()
        s.begin.record(L)
        gsz, D = group[n], lead[n]
        k = 0
        s.host_t0 = time.time()
        t_end = time.perf_counter() + dur_s
        pending = []                 # events of in-flight groups (lead control)
        with torch.cuda.stream(L):
            while True:
                if len(pending) >= D:
                    pending.pop(0).synchronize()
                last = state["last"]
                if last is not None and last.query():
                    s.gaps += 1
                for _ in range(gsz):
                    call(n)
                    f = variants[n]
                    if isinstance(f, Par) and f.last_marks is not None:
                        s.marks.append((k, *f.last_marks))
                    k += 1
                e = pool.get()
                e.record(L)
                s.ends.append((e, k))
                pending.append(e)
                state["last"] = e
                if time.perf_counter() >= t_end:
                    break
        s.n_iter = k
        return s

    timed: list[_Slice] = []
    done: list[_Slice] = []
    probe_host_t0 = None
    gc_was = gc.isenabled()
    gc.disable()
    try:
        if probe:
            probe_host_t0 = time.time()
            probe.start()
        if sampler:
            sampler.start()
        # warmup: round-robin slices of <= 0.5 s, for >= warmup_s and (thermal) until the
        # temperature plateaus
        tw = time.perf_counter()
        wsl = min(0.5, slice_s)
        i = 0
        temps = []
        nvdev = NvmlDevice(dev) if thermal else None
        thermal_info = None
        while True:
            done.append(run_slice(names[i % len(names)], -1, "warm", wsl))
            i += 1
            # recycle completed warmup events
            while done and done[0].ends and done[0].ends[-1][0].query():
                d = done.pop(0)
                pool.put([d.begin] + [e for e, _ in d.ends] + [m for _, e0, ends in d.marks
                                                               for m in (e0, *ends.values())])
            el = time.perf_counter() - tw
            if nvdev is not None:
                temps.append((el, nvdev.temperature_c()))
            if el < warmup_s:
                continue
            if thermal is None:
                break
            win, tmax = thermal
            if el >= tmax:
                thermal_info = {"plateau": False, "seconds": el, "temp_c": temps[-1][1]}
                say(f"[steady] temperature still rising after {el:.0f}s ({temps[-1][1]} C); starting anyway")
                break
            if el >= 2 * win:
                last = max(t for x, t in temps if x > el - win)
                prev = max(t for x, t in temps if el - 2 * win < x <= el - win)
                if last <= prev:
                    thermal_info = {"plateau": True, "seconds": el, "temp_c": last}
                    break
        t_run0 = time.time()
        for r in range(rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            for n in order:
                sl = run_slice(n, r, "timed", slice_s)
                timed.append(sl)
                # recycle warmup events once they are certainly complete
                while done and done[0].ends and done[0].ends[-1][0].query():
                    d = done.pop(0)
                    pool.put([d.begin] + [e for e, _ in d.ends] + [m for _, e0, ends in d.marks
                                                                   for m in (e0, *ends.values())])
        # keep the GPU loaded (tail iterations) while the timed window is closed, the
        # instruments stop and pmon catches up with the end of the window
        last_timed = state["last"]
        tail_ref = reference or names[0]
        run_slice(tail_ref, -2, "tail", 0.05)
        last_timed.synchronize()
        t_run1 = time.time()
        if sampler:
            sampler.stop()
        if probe:
            probe.stop()
        if g is not None:
            deadline = time.time() + 8.0
            while g.covered_until < t_run1 + SLACK_S and time.time() < deadline:
                run_slice(tail_ref, -2, "tail", 0.25)
        state["last"].synchronize()
    finally:
        if gc_was:
            gc.enable()
    L.synchronize()
    caller.wait_stream(L)

    # ---- process ---------------------------------------------------------------------
    def gt_host(e) -> float | None:
        if probe is None or probe.t is None or len(probe.t) < 2:
            return None
        return probe_host_t0 + (probe.event_to_gt(e) - float(probe.t[0])) * 1e-9

    if sampler and sampler.t:
        tt = np.asarray(sampler.t) - sampler._t0 + sampler._w0      # wall-clock seconds
        tc = np.asarray(sampler.temp, float)
    else:
        tt = tc = np.zeros(0)
    pw = [(ts * 1e-6, w) for ts, w in sampler.power_buf] if sampler else []
    pw_t = np.array([t for t, _ in pw]) if pw else np.zeros(0)
    pw_w = np.array([w for _, w in pw]) if pw else np.zeros(0)
    slices_out, slice_windows = [], []
    per_var: dict = {n: {"slice_t": [], "slice_med": [], "iters": [], "clk": [], "pw": [], "gaps": 0,
                         "n_iter": 0, "ops": {}, "rounds": {}} for n in names}
    for s in timed:
        t_ends = np.array([s.begin.elapsed_time(e) * 1e3 for e, _ in s.ends])   # us since begin
        cnt = np.array([c for _, c in s.ends])
        starts = np.concatenate([[0.0], t_ends[:-1]])
        m = starts >= settle_s * 1e6
        if not m.any():
            raise RuntimeError(f"slice {s.variant}/{s.round} has no iterations after settle_s")
        j0 = int(np.argmax(m))                 # first measured group
        prev_cnt = cnt[j0 - 1] if j0 > 0 else 0
        n_meas = int(cnt[-1] - prev_cnt)
        span = t_ends[-1] - starts[j0]
        per_iter = span / n_meas
        gdur = np.diff(np.concatenate([[starts[j0]], t_ends[j0:]]))
        gn = np.diff(np.concatenate([[prev_cnt], cnt[j0:]]))
        it_times = gdur / gn
        e_first = s.begin if j0 == 0 else s.ends[j0 - 1][0]
        e_last = s.ends[-1][0]
        mhz = probe.mhz_between(e_first, e_last) if probe else None
        h0, h1 = gt_host(e_first), gt_host(e_last)
        if h0 is None:
            h0, h1 = s.host_t0 + settle_s, s.host_t0 + slice_s
        sel = (pw_t >= h0 + 0.02) & (pw_t <= h1) if pw_t.size else np.zeros(0, bool)
        pmean = float(pw_w[sel].mean()) if sel.any() else None
        tsel = (tt >= h0) & (tt <= h1) if tt.size else np.zeros(0, bool)
        tmax = float(tc[tsel].max()) if tsel.any() else None
        # gaps inside the measured window are what matters; gaps are counted per group
        rec = {"variant": s.variant, "round": s.round, "t_iter_us": float(per_iter),
               "iter_median_us": float(np.median(it_times)), "iter_p10_us": float(np.percentile(it_times, 10)),
               "iter_p90_us": float(np.percentile(it_times, 90)), "n_iter": n_meas, "group": group[s.variant],
               "span_ms": float(span / 1e3), "clock_mhz": mhz, "power_w": pmean, "power_n": int(sel.sum()),
               "gaps": s.gaps, "temp_c": tmax, "host_window": [h0, h1]}
        slices_out.append(rec)
        slice_windows.append((s.variant, s.round, h0, h1))
        pv = per_var[s.variant]
        pv["slice_t"].append(per_iter)
        pv["slice_med"].append(float(np.median(it_times)))
        pv["iters"].append(it_times)
        pv["rounds"][s.round] = per_iter
        pv["gaps"] += s.gaps
        pv["n_iter"] += n_meas
        if mhz is not None:
            pv["clk"].append(mhz)
        if pmean is not None:
            pv["pw"].append(pmean)
        if s.marks:
            first_iter = prev_cnt
            for k, e0, ends in s.marks:
                if k < first_iter:
                    continue
                for op, e in ends.items():
                    pv["ops"].setdefault(op, []).append(e0.elapsed_time(e) * 1e3)
    total_gaps = sum(pv["gaps"] for pv in per_var.values())
    if total_gaps > max_gaps:
        msg = f"{total_gaps} drained-queue events (host fell behind the GPU): {[(n, pv['gaps']) for n, pv in per_var.items()]}"
        if strict:
            raise HostGapError(msg)
        say("[steady] warning: " + msg)

    out_vars = {}
    for n in names:
        pv = per_var[n]
        st = np.asarray(pv["slice_t"])
        allit = np.concatenate(pv["iters"]) if pv["iters"] else np.zeros(0)
        v = {"t_iter_us": float(np.median(st)), "t_iter_mean_us": float(st.mean()),
             "cv_slices": float(st.std(ddof=1) / st.mean()) if st.size > 1 else 0.0,
             "slices_us": [float(x) for x in st], "iter_median_us": float(np.median(pv["slice_med"])),
             "iter_p10_us": float(np.percentile(allit, 10)), "iter_p90_us": float(np.percentile(allit, 90)),
             "n_iter": pv["n_iter"], "group": group[n], "lead": lead[n], "gaps": pv["gaps"],
             "est_us": est[n], "host_us": host_cost[n],
             "clock_mhz": float(np.median(pv["clk"])) if pv["clk"] else None,
             "power_w": float(np.median(pv["pw"])) if pv["pw"] else None}
        v["cv_ok"] = v["cv_slices"] <= CV_TARGET
        if v["clock_mhz"]:
            v["kcycles_per_iter"] = v["t_iter_us"] * v["clock_mhz"] / 1e3
        if v["power_w"]:
            v["energy_mj_per_iter"] = v["power_w"] * v["t_iter_us"] * 1e-3
        if pv["ops"]:
            v["ops"] = {op: {"median_us": float(np.median(x)), "p10_us": float(np.percentile(x, 10)),
                             "p90_us": float(np.percentile(x, 90)), "n": len(x)} for op, x in pv["ops"].items()}
        out_vars[n] = v
    derived = {}
    if reference is not None:
        ref = per_var[reference]
        sp = {}
        for n in names:
            paired = [ref["rounds"][r] / per_var[n]["rounds"][r] for r in ref["rounds"] if r in per_var[n]["rounds"]]
            sp[n] = {"ratio_of_medians": out_vars[reference]["t_iter_us"] / out_vars[n]["t_iter_us"],
                     "paired_median": float(np.median(paired)),
                     "paired_min": float(np.min(paired)), "paired_max": float(np.max(paired))}
        derived["speedup"] = sp
    temps = sampler.summary()["temp_c"] if sampler else None
    config = {"reference": reference, "slice_s": slice_s, "settle_s": settle_s, "rounds": rounds,
              "warmup_s": warmup_s, "thermal_warmup": thermal_info, "lead_ms": lead_ms, "variants": names,
              "run_s": t_run1 - t_run0, "iterations_total": it_counter[0]}
    res = SteadyResult(
        label=label, config=config, variants=out_vars, derived=derived, slices=slices_out,
        clock=probe.summary(bin_ms=10.0) if probe else None,
        nvml=({"temp_c": temps, "throttle": sampler.summary().get("throttle")} if sampler else None),
        gpu_state_before=state0)
    window = (t_run0, t_run1)
    return res, window, slice_windows
