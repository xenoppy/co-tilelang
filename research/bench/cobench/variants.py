"""Building blocks for multi-variant measurements (bench_corun extras, bench_steady).

A *variant* is a callable that enqueues one iteration of work on torch's **current stream**
and, if it forks work to other streams, joins them back into the current stream before it
returns (the usual CUDA fork/join idiom). ``bench_steady`` calls ``variant(i)`` with a global
iteration counter ``i`` (use it to rotate input copies); ``bench_corun`` extras are called
with no argument.

* ``Par(("a", stream_a, fn_a), ("b", stream_b, fn_b))``: fork-join of ops on streams. It can
  record per-op completion events (``marks``), from which the benches report per-op
  completion times relative to the iteration start.
* ``Rotation(make_inputs, min_bytes)``: enough copies of an op's inputs (and outputs) to
  exceed ``min_bytes`` (default 2x L2) so that consecutive uses of one copy are separated by
  more than the L2 capacity of other traffic.
"""
from __future__ import annotations

import inspect
import math
from typing import Callable

import torch

from .timing import l2_bytes, tensor_bytes, _as_args


def _nargs(fn) -> int:
    """Number of required positional parameters (a variant that wants the iteration index
    must declare one explicitly; ``*args`` callables such as functools.partial(kernel, ...)
    are called without it)."""
    if isinstance(fn, Par):
        return 1
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 0
    return len([p for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty])


def call_variant(fn, i):
    """Call fn(i) or fn() depending on its signature."""
    return fn(i) if _nargs(fn) >= 1 else fn()


class Par:
    """Fork-join of ops on streams, as one variant.

    ``Par((name, stream, fn), ...)``: records a start event on the current stream, makes each
    op stream wait on it, runs ``fn(i)`` (or ``fn()``) on its stream, records the op's end
    event and makes the current stream wait on all end events. ``order`` ("given" |
    "reverse" | "alternate") is the host launch order of the ops.

    Per-op marks: when ``set_mark_pool`` gave it events, every call records
    ``(start, {name: end})`` into ``self.last_marks`` for the bench to collect.
    """

    def __init__(self, *ops, order: str = "given"):
        if len(ops) < 1:
            raise ValueError("Par needs at least one (name, stream, fn)")
        names = [o[0] for o in ops]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate op names {names}")
        if order not in ("given", "reverse", "alternate"):
            raise ValueError(order)
        streams = [o[1] for o in ops]
        if len({int(s.cuda_stream) for s in streams}) != len(streams):
            raise ValueError("every op of a Par needs its own stream")
        self.ops = [(n, s, fn, _nargs(fn) >= 1) for n, s, fn in ops]
        self.names = names
        self.order = order
        self._start = torch.cuda.Event()
        self._ends = {n: torch.cuda.Event() for n in names}
        self._pool = None
        self.last_marks = None
        self._k = 0

    def set_mark_pool(self, pool):
        """pool: callable() -> a fresh timing event (or None to stop recording marks)."""
        self._pool = pool

    def __call__(self, i=None):
        L = torch.cuda.current_stream()
        timed = self._pool is not None
        e0 = self._pool() if timed else self._start
        e0.record(L)
        ops = self.ops
        if self.order == "reverse" or (self.order == "alternate" and self._k % 2):
            ops = ops[::-1]
        self._k += 1
        ends = {}
        for name, s, fn, _ in ops:
            s.wait_event(e0)
        for name, s, fn, takes_i in ops:
            with torch.cuda.stream(s):
                if takes_i:
                    fn(0 if i is None else i)
                else:
                    fn()
            e = self._pool() if timed else self._ends[name]
            e.record(s)
            ends[name] = e
        for e in ends.values():
            L.wait_event(e)
        self.last_marks = (e0, ends) if timed else None


class Rotation:
    """Input copies for cold (DRAM) inputs without a flush.

    ``rot = Rotation(make_inputs, min_bytes=None)``; ``rot[i]`` is copy ``i % rot.n`` (an
    argument tuple). ``make_inputs()`` must return everything the op touches, outputs
    included, so that outputs rotate too. ``min_bytes`` defaults to 2x L2."""

    def __init__(self, make_inputs: Callable, min_bytes: int | None = None, max_copies: int = 4096,
                 n: int | None = None):
        first = _as_args(make_inputs())
        self.bytes_per_copy = tensor_bytes(first)
        if self.bytes_per_copy <= 0:
            raise ValueError("make_inputs returned no tensors")
        self.min_bytes = int(min_bytes if min_bytes is not None else 2 * l2_bytes())
        need = max(1, math.ceil(self.min_bytes / self.bytes_per_copy))
        self.n = int(n) if n is not None else need
        if self.n < need:
            raise ValueError(f"n={n} copies ({self.n * self.bytes_per_copy >> 20} MB) < min_bytes "
                             f"({self.min_bytes >> 20} MB)")
        if self.n > max_copies:
            raise ValueError(f"{self.n} copies of {self.bytes_per_copy} B needed (> max_copies)")
        self.copies = [first] + [_as_args(make_inputs()) for _ in range(self.n - 1)]

    @property
    def total_bytes(self) -> int:
        return self.n * self.bytes_per_copy

    def __getitem__(self, i: int):
        return self.copies[i % self.n]

    def __len__(self):
        return self.n
