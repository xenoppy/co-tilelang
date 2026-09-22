"""Timing primitives.

``bench``        single callable; modes
                   * "flush": per-call CUDA events, L2 flushed (>= 2x L2 written) before
                              every call on a separate full-GPU stream; flush excluded.
                   * "graph": CUDA graph of k back-to-back calls rotating over enough
                              input copies (from ``make_inputs``) to exceed 2x L2;
                              reports per-call time = replay time / k.
                   * "hot":   CUDA graph of k back-to-back calls on ONE input copy, no
                              flush -> L2-hot numbers (labelled l2="hot").
``bench_corun``  two callables on two streams from a common start event, interleaved
                 with solo (and serial) reference measurements.

All times are microseconds. Every timed rep is preceded by a ``HostGate`` so the
host has enqueued the whole rep before the start event fires (no host-launch
latency inside the timed region).
"""
from __future__ import annotations

import functools
import itertools
import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch

from .clock import ClockProbe, per_rep_clock
from .cudrv import device_index
from .kernels import HostGate
from .nvml import NvmlSampler, gpu_state

MIN_WARMUP, MIN_REPS = 20, 50
WARMUP_S = 1.0     # minimum warmup wall time: reach the power/thermal steady state
CV_TARGET = 0.02


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def l2_bytes(device=None) -> int:
    return int(torch.cuda.get_device_properties(device_index(device)).L2_cache_size)


@functools.lru_cache(maxsize=None)
def _flush_buffer(device: int, nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes // 4, dtype=torch.int32, device=f"cuda:{device}")


@functools.lru_cache(maxsize=None)
def _side_stream(device: int, name: str) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device)


@functools.lru_cache(maxsize=None)
def _gate(device: int) -> HostGate:
    return HostGate(device)


def _flush(buf: torch.Tensor, i: int) -> None:
    buf.fill_(i & 0xFF)


def summarize(samples) -> dict:
    a = np.asarray(samples, dtype=np.float64)
    n = int(a.size)
    if n == 0:
        return {"n": 0}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n, "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)), "mean": mean, "std": std,
        "cv": std / mean if mean > 0 else float("nan"), "min": float(a.min()), "max": float(a.max()),
    }


def _tensors(obj, out: dict) -> dict:
    if isinstance(obj, torch.Tensor):
        out[(obj.untyped_storage().data_ptr())] = obj.untyped_storage().nbytes()
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _tensors(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _tensors(x, out)
    return out


def tensor_bytes(obj) -> int:
    """Total bytes of distinct tensor storages reachable from obj (tuple/list/dict)."""
    return int(sum(_tensors(obj, {}).values()))


def _as_args(x) -> tuple:
    if x is None:
        return ()
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _check_counts(warmup, reps, strict):
    if warmup < MIN_WARMUP or reps < MIN_REPS:
        msg = f"protocol requires warmup>={MIN_WARMUP}, reps>={MIN_REPS} (got {warmup}, {reps})"
        if strict:
            raise ValueError(msg + "; pass strict=False for exploratory runs")
        warnings.warn(msg)


def _us(e0: torch.cuda.Event, e1: torch.cuda.Event) -> float:
    return e0.elapsed_time(e1) * 1e3


def _timing_events(n: int, stream: torch.cuda.Stream) -> list:
    """Timing events whose CUDA handles already exist (torch creates them lazily on the
    first record, ~tens of us of host time each)."""
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for e in evs:
        e.record(stream)
    return evs


class _Instruments:
    """NVML sampler + clock probe, fully constructed (kernel compiled, buffers allocated)
    BEFORE warmup so that starting them adds no idle gap between warmup and timed reps
    (a ~0.2 s gap lets the power-capped clock recover and biases the first timed reps)."""

    def __init__(self, dev: int, nvml: bool, clock: bool):
        self.sampler = NvmlSampler(device=dev) if nvml else None
        self.probe = ClockProbe(dev) if clock else None

    def start(self):
        if self.sampler:
            self.sampler.start()
        if self.probe:
            self.probe.start()

    def stop(self):
        if self.probe:
            self.probe.stop()
        if self.sampler:
            self.sampler.stop()


# ---------------------------------------------------------------------------
# single-callable benchmark
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    label: str | None
    mode: str                 # flush | graph | hot
    l2: str                   # cold-flush | cold-rotate | hot
    unit: str
    n: int
    median: float
    p10: float
    p90: float
    mean: float
    std: float
    cv: float
    min: float
    max: float
    k: int = 1                # calls per sample (graph modes)
    n_copies: int = 1         # rotated input copies (graph mode)
    bytes_per_copy: int | None = None
    flops: float | None = None
    nbytes: float | None = None
    tflops: float | None = None   # from median
    gbps: float | None = None     # from median
    cv_ok: bool = True
    samples: list = field(default_factory=list)
    nvml: dict | None = None
    clock: dict | None = None     # ClockProbe: window summary + per-rep clocks
    gpu_state_before: dict | None = None

    def to_dict(self, samples: bool = False) -> dict:
        d = asdict(self)
        if not samples:
            d.pop("samples")
            if d.get("clock") and "per_rep" in d["clock"]:
                d["clock"]["per_rep"].pop("values", None)
        return d

    def __str__(self):
        s = (f"[{self.label or 'bench'}] mode={self.mode} l2={self.l2} n={self.n} k={self.k} "
             f"copies={self.n_copies}: median={self.median:.2f}us p10={self.p10:.2f} "
             f"p90={self.p90:.2f} cv={self.cv*100:.2f}%")
        if self.tflops is not None:
            s += f" {self.tflops:.1f} TFLOP/s"
        if self.gbps is not None:
            s += f" {self.gbps:.1f} GB/s"
        if self.clock and self.clock.get("per_rep", {}).get("n"):
            c = self.clock["per_rep"]
            s += f" | clk(rep) {c['min']:.0f}/{c['median']:.0f}/{c['max']:.0f} MHz"
        if self.nvml and "power_w" in self.nvml:
            s += (f" | P~{self.nvml['power_w']['mean']:.0f}W T{self.nvml['temp_c']['max']:.0f}C "
                  f"lim={self.nvml['throttle']['reasons_seen']}")
        return s


def bench(fn: Callable, *, make_inputs: Callable | None = None, mode: str = "flush",
          warmup: int = MIN_WARMUP, reps: int = MIN_REPS, warmup_s: float = WARMUP_S,
          k: int | None = None, min_k: int = 20,
          stream: torch.cuda.Stream | None = None, flops: float | None = None,
          nbytes: float | None = None, nvml: bool = False, clock: bool = False,
          rotate_bytes: int | None = None, max_copies: int = 4096, flush_bytes: int | None = None,
          gate: bool = True, label: str | None = None, strict: bool = True,
          keep_samples: bool = True) -> BenchResult:
    """Time ``fn(*make_inputs())``.

    fn           callable; launches its GPU work on torch's current stream (which bench
                 sets to ``stream``). May be torch ops or CudaKernel launches.
    make_inputs  factory returning the argument tuple (include output buffers so they
                 rotate too). Required for mode="graph"; optional otherwise.
    mode         "flush" | "graph" | "hot" (see module doc).
    warmup_s     keep warming up (in chunks of ``warmup`` reps) until this much wall time has
                 passed: under the 600 W cap the SM clock settles only after ~0.3 s of load,
                 and a short warmup leaves a bimodal (fast-then-slow) timed window.
    k            calls per graph (graph/hot); default: smallest multiple of n_copies >= min_k.
    max_copies   refuse to rotate more copies than this (tiny inputs cannot be rotated past
                 2x L2 sensibly -- use mode="flush" for them).
    flops/nbytes per-call work, used to report TFLOP/s and GB/s at the median.
    nvml         attach an NvmlSampler over the timed reps (power/temp/throttle; ~500 ms
                 refresh on this driver).
    clock        attach a ClockProbe (GPU-side SM clock, per-rep MHz + window summary).
    """
    if mode not in ("flush", "graph", "hot"):
        raise ValueError(f"unknown mode {mode!r}")
    _check_counts(warmup, reps, strict)
    dev = device_index(stream.device if stream is not None else None)
    S = stream if stream is not None else torch.cuda.current_stream(dev)
    caller = torch.cuda.current_stream(dev)
    L2 = l2_bytes(dev)
    g8 = _gate(dev) if gate else None
    state0 = gpu_state(dev) if nvml else None

    if mode == "flush":
        F = _side_stream(dev, "flush")
        buf = _flush_buffer(dev, int(flush_bytes or 2 * L2))
        inputs = _as_args(make_inputs()) if make_inputs else ()
        bpc = tensor_bytes(inputs) if inputs else None
        F.wait_stream(caller)
        S.wait_stream(caller)
        ready = torch.cuda.Event()
        k_calls, n_copies, l2_label = 1, 1, "cold-flush"

        def one(i, e_start, e_end):
            with torch.cuda.stream(F):
                _flush(buf, i)
                if g8:
                    g8.wait(F)
                ready.record(F)
            S.wait_event(ready)
            e_start.record(S)
            with torch.cuda.stream(S):
                fn(*inputs)
            e_end.record(S)
            F.wait_event(e_end)
            if g8:
                g8.release()

        # ungated pre-warm on S: lazy module loading / allocator growth may need a
        # context-wide sync, which would deadlock against a spinning gate
        with torch.cuda.stream(S):
            fn(*inputs)
        S.synchronize()
        scratch = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        starts, ends = _timing_events(reps, F), _timing_events(reps, F)
        inst = _Instruments(dev, nvml, clock)
        F.synchronize()
        t0, i = time.perf_counter(), 0
        while True:
            for _ in range(max(1, warmup)):
                one(i, *scratch)
                i += 1
            scratch[1].synchronize()
            F.synchronize()
            if i >= warmup and time.perf_counter() - t0 >= warmup_s:
                break
        if g8:
            g8.check("warmup")
        inst.start()
        for i in range(reps):
            one(i, starts[i], ends[i])
        for e in ends:
            e.synchronize()
        F.synchronize()
        inst.stop()
        samples = [_us(a, b) for a, b in zip(starts, ends)]
    else:
        if make_inputs is None and mode == "graph":
            raise ValueError('mode="graph" needs make_inputs to rotate input copies; '
                             'use mode="hot" for fixed inputs')
        first = _as_args(make_inputs()) if make_inputs else ()
        bpc = tensor_bytes(first) if first else 0
        if mode == "graph":
            if bpc <= 0:
                raise ValueError("make_inputs returned no tensors; cannot rotate")
            n_copies = max(1, math.ceil((rotate_bytes or 2 * L2) / bpc))
            if n_copies > max_copies:
                raise ValueError(
                    f"inputs are {bpc} B per call: exceeding {(rotate_bytes or 2 * L2) >> 20} MB "
                    f"needs {n_copies} copies (> max_copies={max_copies}). Use mode='flush' "
                    "(per-call L2 flush) or raise max_copies / lower rotate_bytes")
            copies = [first] + [_as_args(make_inputs()) for _ in range(n_copies - 1)]
            l2_label = "cold-rotate"
        else:
            n_copies, copies, l2_label = 1, [first], "hot"
        k_calls = k if k is not None else n_copies * max(1, math.ceil(min_k / n_copies))
        if mode == "graph" and k_calls < n_copies:
            raise ValueError(f"k={k_calls} < n_copies={n_copies}: rotation would not cycle")
        S.wait_stream(caller)
        # eager warmup (lazy init, cuBLAS heuristics, module loads) before capture
        with torch.cuda.stream(S):
            for _ in range(2):
                for c in copies:
                    fn(*c)
        S.synchronize()
        graph = torch.cuda.CUDAGraph()
        # capture on S itself (so kernels bind to S's context, e.g. a green context);
        # the legacy default stream cannot be captured -> torch's side stream.
        cap = S if S.cuda_stream != 0 else None
        with torch.cuda.graph(graph, stream=cap):
            for i in range(k_calls):
                fn(*copies[i % n_copies])
        torch.cuda.synchronize(dev)
        starts, ends = _timing_events(reps, S), _timing_events(reps, S)
        inst = _Instruments(dev, nvml, clock)
        S.synchronize()
        t0, i = time.perf_counter(), 0
        while True:
            with torch.cuda.stream(S):
                for _ in range(max(1, warmup)):
                    if g8:  # also loads the gate kernel into S's context before timing
                        g8.wait(S)
                    graph.replay()
                    if g8:
                        g8.release()
                    i += 1
            S.synchronize()
            if i >= warmup and time.perf_counter() - t0 >= warmup_s:
                break
        if g8:
            g8.check("warmup")
        inst.start()
        with torch.cuda.stream(S):
            for i in range(reps):
                if g8:
                    g8.wait(S)
                starts[i].record(S)
                graph.replay()
                ends[i].record(S)
                if g8:
                    g8.release()
        for e in ends:
            e.synchronize()
        inst.stop()
        samples = [_us(a, b) / k_calls for a, b in zip(starts, ends)]
        del graph
    if g8:
        g8.check("timed reps")
    caller.wait_stream(S)

    st = summarize(samples)
    med_s = st["median"] * 1e-6
    return BenchResult(
        label=label, mode=mode, l2=l2_label, unit="us", **st,
        k=k_calls, n_copies=n_copies, bytes_per_copy=bpc,
        flops=flops, nbytes=nbytes,
        tflops=(flops / med_s / 1e12) if flops else None,
        gbps=(nbytes / med_s / 1e9) if nbytes else None,
        cv_ok=st["cv"] <= CV_TARGET,
        samples=samples if keep_samples else [],
        nvml=inst.sampler.summary() if inst.sampler else None,
        clock=({"window": inst.probe.summary(),
                "per_rep": per_rep_clock(inst.probe, starts, ends, samples)} if inst.probe else None),
        gpu_state_before=state0,
    )


# ---------------------------------------------------------------------------
# two-stream co-run
# ---------------------------------------------------------------------------
@dataclass
class CorunResult:
    label: str | None
    config: dict
    corun: dict               # makespan / a_end / b_end summaries (+ by_order)
    solo_a: dict | None
    solo_b: dict | None
    serial: dict | None       # {"total": ..., "a": ...}
    derived: dict
    nvml: dict | None = None
    clock: dict | None = None     # ClockProbe window summary + per-variant per-rep clocks
    gpu_state_before: dict | None = None
    samples: dict | None = None

    def to_dict(self, samples: bool = False) -> dict:
        d = asdict(self)
        if not samples:
            d.pop("samples")
            for v in (d.get("clock") or {}).get("per_variant", {}).values():
                v.pop("values", None)
        return d

    def __str__(self):
        c = self.corun
        s = (f"[{self.label or 'corun'}] makespan med={c['makespan']['median']:.1f}us "
             f"(p10 {c['makespan']['p10']:.1f}, p90 {c['makespan']['p90']:.1f}, "
             f"cv {c['makespan']['cv']*100:.2f}%) | A end {c['a_end']['median']:.1f} | "
             f"B end {c['b_end']['median']:.1f}")
        if self.solo_a:
            s += f" | solo A {self.solo_a['median']:.1f} B {self.solo_b['median']:.1f}"
        if self.serial:
            s += f" | serial {self.serial['total']['median']:.1f}"
        d = self.derived
        if "speedup_vs_serial" in d:
            s += f" | speedup vs serial {d['speedup_vs_serial']:.3f}"
        return s


def bench_corun(fn_a: Callable, fn_b: Callable, *, stream_a: torch.cuda.Stream | None = None,
                stream_b: torch.cuda.Stream | None = None, warmup: int = MIN_WARMUP,
                reps: int = MIN_REPS, warmup_s: float = WARMUP_S, flush: bool = True,
                solo: bool = True, serial: bool = True,
                order: str = "alternate", nvml: bool = False, clock: bool = False, gate: bool = True,
                flush_bytes: int | None = None, label: str | None = None, strict: bool = True,
                keep_samples: bool = True) -> CorunResult:
    """Co-run ``fn_a()`` on stream_a and ``fn_b()`` on stream_b from a common start.

    Per rep (on a full-GPU launcher stream L): [L2 flush] -> host gate -> E0 on L;
    each op stream waits E0, runs its fn, records its end event. Reported times are
    relative to E0: per-op completion (a_end, b_end) and makespan = max of the two.
    Variants corun / solo_a (on stream_a) / solo_b (on stream_b) / serial (a then b on
    L) are interleaved within every rep, with rotating order, to cancel clock drift.
    order: "ab" | "ba" | "alternate" -- host launch order of the two ops in corun.
    """
    if order not in ("ab", "ba", "alternate"):
        raise ValueError(order)
    _check_counts(warmup, reps, strict)
    dev = device_index(None)
    caller = torch.cuda.current_stream(dev)
    L = _side_stream(dev, "launcher")
    A = stream_a if stream_a is not None else _side_stream(dev, "corun_a")
    B = stream_b if stream_b is not None else _side_stream(dev, "corun_b")
    if A.cuda_stream == B.cuda_stream:
        raise ValueError("stream_a and stream_b must differ")
    buf = _flush_buffer(dev, int(flush_bytes or 2 * l2_bytes(dev))) if flush else None
    g8 = _gate(dev) if gate else None
    state0 = gpu_state(dev) if nvml else None
    for s in (L, A, B):
        s.wait_stream(caller)

    variants = ["corun"] + (["solo_a", "solo_b"] if solo else []) + (["serial"] if serial else [])

    def run_variant(v, i, rep_order, ev):
        e0, ea, eb = next(ev), next(ev), next(ev)
        with torch.cuda.stream(L):
            if buf is not None:
                _flush(buf, i)
            if g8:
                g8.wait(L)
            e0.record(L)
        if v == "corun":
            ops = [("a", A, fn_a, ea), ("b", B, fn_b, eb)]
            if rep_order == "ba":
                ops.reverse()
            for _, st, f, e in ops:
                st.wait_event(e0)
            for _, st, f, e in ops:
                with torch.cuda.stream(st):
                    f()
                e.record(st)
            L.wait_event(ea)
            L.wait_event(eb)
        elif v in ("solo_a", "solo_b"):
            st, f, e = (A, fn_a, ea) if v == "solo_a" else (B, fn_b, eb)
            st.wait_event(e0)
            with torch.cuda.stream(st):
                f()
            e.record(st)
            L.wait_event(e)
        else:  # serial on the launcher stream (full GPU)
            with torch.cuda.stream(L):
                fn_a()
                ea.record(L)
                fn_b()
                eb.record(L)
        if g8:
            g8.release()
        return (v, rep_order, e0, ea, eb)

    def schedule(nrep, offset, ev):
        recs = []
        for r in range(nrep):
            rot = r % len(variants)
            vs = variants[rot:] + variants[:rot]
            rep_order = order if order != "alternate" else ("ab" if r % 2 == 0 else "ba")
            for j, v in enumerate(vs):
                recs.append(run_variant(v, offset + r * len(vs) + j, rep_order, ev))
        return recs

    def sync(recs):
        for (v, _, e0, ea, eb) in recs:
            if v in ("corun", "serial", "solo_a"):
                ea.synchronize()
            if v in ("corun", "serial", "solo_b"):
                eb.synchronize()
        L.synchronize()

    # ungated pre-warm on every stream a fn will run on (lazy module loading is per
    # context and may force a sync that would deadlock against a spinning gate)
    for st, f in ((A, fn_a), (B, fn_b), (L, fn_a), (L, fn_b)):
        with torch.cuda.stream(st):
            f()
        st.synchronize()
    # everything the timed phase needs is created before warmup (no idle gap after it)
    warm_ring = itertools.cycle(_timing_events(3 * len(variants), L))
    timed_pool = iter(_timing_events(3 * len(variants) * reps, L))
    inst = _Instruments(dev, nvml, clock)
    L.synchronize()
    t0, done = time.perf_counter(), 0
    while True:
        sync(schedule(max(1, warmup), done * len(variants), warm_ring))
        done += max(1, warmup)
        if done >= warmup and time.perf_counter() - t0 >= warmup_s:
            break
    if g8:
        g8.check("warmup")
    inst.start()
    recs = schedule(reps, done * len(variants), timed_pool)
    sync(recs)
    inst.stop()
    sampler, probe = inst.sampler, inst.probe
    if g8:
        g8.check("timed reps")
    caller.wait_stream(L)

    raw = {v: [] for v in variants}
    for (v, o, e0, ea, eb) in recs:
        if v == "corun":
            a, b = _us(e0, ea), _us(e0, eb)
            raw[v].append((o, a, b, max(a, b)))
        elif v == "solo_a":
            raw[v].append(_us(e0, ea))
        elif v == "solo_b":
            raw[v].append(_us(e0, eb))
        else:
            raw[v].append((_us(e0, ea), _us(e0, eb)))

    cr = raw["corun"]
    corun = {
        "makespan": summarize([x[3] for x in cr]),
        "a_end": summarize([x[1] for x in cr]),
        "b_end": summarize([x[2] for x in cr]),
        "by_order": {o: {"n": sum(1 for x in cr if x[0] == o),
                         "makespan_median": float(np.median([x[3] for x in cr if x[0] == o])),
                         "a_end_median": float(np.median([x[1] for x in cr if x[0] == o])),
                         "b_end_median": float(np.median([x[2] for x in cr if x[0] == o]))}
                     for o in sorted({x[0] for x in cr})},
    }
    solo_a = summarize(raw["solo_a"]) if solo else None
    solo_b = summarize(raw["solo_b"]) if solo else None
    ser = ({"total": summarize([x[1] for x in raw["serial"]]),
            "a": summarize([x[0] for x in raw["serial"]])} if serial else None)
    m = corun["makespan"]["median"]
    derived = {"cv_ok": corun["makespan"]["cv"] <= CV_TARGET}
    if solo:
        derived.update({
            "slowdown_a": corun["a_end"]["median"] / solo_a["median"],
            "slowdown_b": corun["b_end"]["median"] / solo_b["median"],
            "makespan_over_max_solo": m / max(solo_a["median"], solo_b["median"]),
            "makespan_over_sum_solo": m / (solo_a["median"] + solo_b["median"]),
        })
    if serial:
        derived["speedup_vs_serial"] = ser["total"]["median"] / m
    clk = None
    if probe:
        spans = {v: [] for v in variants}
        for (v, o, e0, ea, eb) in recs:
            if v in ("corun", "serial"):
                end = ea if _us(e0, ea) >= _us(e0, eb) else eb
            else:
                end = ea if v == "solo_a" else eb
            spans[v].append((e0, end))
        clk = {"window": probe.summary(),
               "per_variant": {v: per_rep_clock(probe, [a for a, _ in sp], [b for _, b in sp])
                               for v, sp in spans.items()}}
    return CorunResult(
        label=label,
        config={"flush": flush, "order": order, "warmup": warmup, "warmup_s": warmup_s,
                "reps": reps, "gate": gate,
                "stream_a": int(A.cuda_stream), "stream_b": int(B.cuda_stream)},
        corun=corun, solo_a=solo_a, solo_b=solo_b, serial=ser, derived=derived,
        nvml=sampler.summary() if sampler else None, clock=clk, gpu_state_before=state0,
        samples={k: v for k, v in raw.items()} if keep_samples else None,
    )
