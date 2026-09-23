"""Tile-schedule variants of the op library's persistent build, with optional per-tile traces.

Used by the static-persistent-penalty study (static_penalty.py, methodology v1, M4). The
variants are written once for every op of `cotile.ops` (op protocol, see cotile/kernel.py)
and differ from `cotile.kernel.build_persistent` only in where a CTA's tile ids come from:

  static         tile = it * n + bid                      (== build_persistent)
  static_perm    tile = it * n + perm[bid]                 (perm: runtime int32[n] knob)
  chunked        tile = perm[bid] * waves + it             (contiguous range per CTA)
  static_slot    static ids, but thread 0 publishes each id through a shared slot +
                 CTA barrier (the dispatch structure of the dynamic queue, no atomic)
  static_atomic  static_slot + one global atomic per tile by thread 0 whose (unused)
                 result gates the slot write (the dynamic queue's latency, static order)
  dynamic        thread 0 grabs the next id with atomic_add on a global queue head
                 (chunk 1), publishes it through the slot (== CoKernel dynamic dispatch)

Knobs (runtime, int32 tensors): `sv_perm[n]` (rank of each CTA for static_perm/chunked)
and `sv_delay[n]` (ns thread 0 of each CTA spins at kernel start; 0 = none).
Counters (`sv_state`, int32[4]) self-reset: the last CTA to exit zeroes them.

Tracing (trace=True): thread 0 of every CTA records %globaltimer before and after each
tile it runs, plus the tile id, into SHARED memory (no global traffic while tiles run);
at exit the CTA copies its trace, %smid, first/last timestamps to global buffers. The
grid build gets the same (one tile per CTA).
"""
from __future__ import annotations

import hashlib
import math

import tilelang.language as T

from cotile.kernel import KernelSpec, IOParam, alloc_scratch, ceildiv, make_prim_func

SCHEDULES = ("static", "static_perm", "chunked", "static_slot", "static_atomic", "dynamic")

PRELUDE = r"""
#ifndef SV_PRELUDE
#define SV_PRELUDE
__device__ __forceinline__ int sv_smid() {
  unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return (int)r;
}
__device__ __forceinline__ long long sv_globaltimer() {
  unsigned long long r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return (long long)r;
}
__device__ __forceinline__ void sv_spin_until(long long t) {
  unsigned long long r;
  do { __nanosleep(100); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); } while ((long long)r < t);
}
#endif
"""


def _smid():
    return T.call_extern("int32", "sv_smid")


def _gt():
    return T.call_extern("int64", "sv_globaltimer")


def _bounded(t, n):
    return T.min(T.max(t, 0), n - 1)


def _name(op, cfg, tag):
    h = hashlib.sha1(repr((op.NAME, cfg, tag)).encode()).hexdigest()[:8]
    return f"sv_{op.NAME}_{tag}_{h}".replace("-", "_")


def build_sched(op, shape, cfg, num_ctas: int, schedule: str, trace: bool = False,
                trace_cap: int | None = None) -> KernelSpec:
    if schedule not in SCHEDULES:
        raise ValueError(schedule)
    op.validate(shape, cfg)
    ts = op.tile_space(shape, cfg)
    body = op.make_tile_body(shape, cfg)
    params = list(op.io_params(shape, cfg))
    sspec = op.scratch_spec(shape, cfg)
    threads = op.threads(cfg)
    N = ts.num_tiles
    n = num_ctas
    waves = ceildiv(N, n)
    cap = trace_cap or (waves if schedule in ("static", "static_perm", "chunked", "static_slot", "static_atomic")
                        else max(8, 3 * waves))
    uses_slot = schedule in ("static_slot", "static_atomic", "dynamic")
    uses_exit = schedule in ("static_atomic", "dynamic")
    opaque = schedule != "static"
    MAX_IT = N + 1 if schedule == "dynamic" else waves
    params += [IOParam("sv_perm", (n,), "int32", "knob"), IOParam("sv_delay", (n,), "int32", "knob"),
               IOParam("sv_state", (4,), "int32", "ctr")]
    if trace:
        params += [IOParam("sv_tr", (n, cap, 2), "int64", "dbg"), IOParam("sv_tid", (n, cap), "int32", "dbg"),
                   IOParam("sv_meta", (n, 4), "int64", "dbg")]

    @T.macro
    def kernel_body(args):
        io = op.make_io(args)
        perm = args["sv_perm"]
        delay = args["sv_delay"]
        st = args["sv_state"]
        with T.Kernel(n, threads=threads, prelude=PRELUDE) as bid:
            scratch = alloc_scratch(sspec)
            tx = T.get_thread_binding()
            if uses_slot:
                slot = T.alloc_shared((2,), "int32")
            if trace:
                tr = T.alloc_shared((2 * cap,), "int64")
                tid = T.alloc_shared((cap,), "int32")
                kk = T.alloc_shared((1,), "int32")
            t0v = T.alloc_var("int64")
            k = T.alloc_var("int32")
            cur = T.alloc_var("int32")
            dm = T.alloc_var("int32")
            prev = T.alloc_var("int32")
            if tx == 0:
                k = 0
                t0v = _gt()
                if delay[bid] > 0:
                    T.call_extern("handle", "sv_spin_until", t0v + T.cast(delay[bid], "int64"))
            if schedule in ("static_perm", "chunked"):
                rank = perm[bid]
            else:
                rank = bid
            for it in T.serial(MAX_IT):
                if uses_slot:
                    if tx == 0:
                        if schedule == "dynamic":
                            cur = T.atomic_add(st[0], 1, return_prev=True)
                        elif schedule == "static_atomic":
                            dm = T.atomic_add(st[2], 1, return_prev=True)
                            cur = T.if_then_else(dm >= 0, it * n + rank, -1)
                        else:
                            cur = it * n + rank
                        slot[it % 2] = cur
                    T.sync_threads()
                    t = slot[it % 2]
                    if t >= N:
                        T.loop_break()
                else:
                    t = (rank * waves + it) if schedule == "chunked" else (it * n + rank)
                if t < N:
                    if trace:
                        if tx == 0:
                            if k < cap:
                                tr[2 * k] = _gt()
                    if opaque:
                        # id not affine in (it, bid): guard AND clamp so TileLang can prove
                        # the body's loads in range (see cotile.cokernel.bounded)
                        if t >= 0:
                            body(_bounded(t, N), io, scratch)
                    else:
                        body(t, io, scratch)
                    if trace:
                        if tx == 0:
                            if k < cap:
                                tr[2 * k + 1] = _gt()
                                tid[k] = t
                            k = k + 1
            if uses_exit:
                T.sync_threads()
                if tx == 0:
                    prev = T.atomic_add(st[1], 1, memory_order="acq_rel", return_prev=True)
                    if prev == n - 1:
                        st[0] = 0
                        st[1] = 0
                        st[2] = 0
            if trace:
                meta = args["sv_meta"]
                gtr = args["sv_tr"]
                gtid = args["sv_tid"]
                if tx == 0:
                    kk[0] = T.min(k, cap)
                    meta[bid, 0] = T.cast(_smid(), "int64")
                    meta[bid, 1] = T.cast(k, "int64")
                    meta[bid, 2] = t0v
                    meta[bid, 3] = _gt()
                T.sync_threads()
                for j in T.serial(ceildiv(cap, threads)):
                    if j * threads + tx < kk[0]:
                        gtr[bid, j * threads + tx, 0] = tr[2 * (j * threads + tx)]
                        gtr[bid, j * threads + tx, 1] = tr[2 * (j * threads + tx) + 1]
                        gtid[bid, j * threads + tx] = tid[j * threads + tx]

    tag = f"{schedule}{'_tr' if trace else ''}_n{n}"
    name = _name(op, cfg, tag)
    pf = make_prim_func(name, params, kernel_body)
    spec = KernelSpec(op=op, shape=shape, cfg=cfg, build="persistent", name=name, prim_func=pf,
                      pass_configs=dict(op.pass_configs(cfg, "persistent")), params=params, num_tiles=N,
                      grid=n, threads=threads, smem_estimate=op.smem_bytes(shape, cfg, "persistent"))
    spec.extra["sv"] = {"schedule": schedule, "trace": trace, "cap": cap, "waves": waves, "num_ctas": n}
    return spec


def build_grid_traced(op, shape, cfg) -> KernelSpec:
    """Grid build (one CTA per tile) that records (%smid, start, end) per CTA."""
    op.validate(shape, cfg)
    ts = op.tile_space(shape, cfg)
    body = op.make_tile_body(shape, cfg)
    params = list(op.io_params(shape, cfg)) + [IOParam("sv_meta", (ts.num_tiles, 4), "int64", "dbg")]
    sspec = op.scratch_spec(shape, cfg)
    threads = op.threads(cfg)
    N = ts.num_tiles

    @T.macro
    def kernel_body(args):
        io = op.make_io(args)
        meta = args["sv_meta"]
        with T.Kernel(N, threads=threads, prelude=PRELUDE) as bid:
            scratch = alloc_scratch(sspec)
            tx = T.get_thread_binding()
            t0v = T.alloc_var("int64")
            if tx == 0:
                t0v = _gt()
            body(bid, io, scratch)
            if tx == 0:
                meta[bid, 0] = T.cast(_smid(), "int64")
                meta[bid, 1] = T.int64(1)
                meta[bid, 2] = t0v
                meta[bid, 3] = _gt()

    name = _name(op, cfg, "grid_tr")
    pf = make_prim_func(name, params, kernel_body)
    spec = KernelSpec(op=op, shape=shape, cfg=cfg, build="grid", name=name, prim_func=pf,
                      pass_configs=dict(op.pass_configs(cfg, "grid")), params=params, num_tiles=N, grid=N,
                      threads=threads, smem_estimate=op.smem_bytes(shape, cfg, "grid"))
    spec.extra["sv"] = {"schedule": "grid", "trace": True, "cap": 1, "waves": 1, "num_ctas": N}
    return spec


class SvRunner:
    """Launch closures for a (traced) schedule variant: owns knobs, counters and traces."""

    def __init__(self, spec: KernelSpec, perm=None, delay=None, device="cuda"):
        import torch

        if spec.kernel is None:
            raise RuntimeError(f"{spec.name}: {spec.compile_error}")
        self.spec = spec
        self.sv = spec.extra.get("sv", {})
        self.state = {}
        n = self.sv.get("num_ctas", spec.grid)
        for p in spec.params:
            if p.name == "sv_perm":
                self.state[p.name] = torch.tensor(list(perm) if perm is not None else list(range(n)),
                                                  dtype=torch.int32, device=device)
            elif p.name == "sv_delay":
                self.state[p.name] = torch.tensor(list(delay) if delay is not None else [0] * n,
                                                  dtype=torch.int32, device=device)
            elif p.role in ("ctr", "dbg", "ws"):
                tdt = getattr(torch, p.dtype)
                self.state[p.name] = (torch.zeros(p.shape, dtype=tdt, device=device) if p.role != "ws"
                                      else torch.empty(p.shape, dtype=tdt, device=device))

    def set_perm(self, perm):
        self.state["sv_perm"].copy_(torch_tensor(perm, self.state["sv_perm"]))

    def set_delay(self, delay):
        self.state["sv_delay"].copy_(torch_tensor(delay, self.state["sv_delay"]))

    def args(self, inputs: dict, outputs: dict) -> list:
        out = []
        for p in self.spec.params:
            if p.role == "in":
                out.append(inputs[p.name])
            elif p.role == "out":
                out.append(outputs[p.name])
            else:
                out.append(self.state[p.name])
        return out

    def fn(self, inputs: dict, outputs: dict):
        import functools

        return functools.partial(self.spec.kernel, *self.args(inputs, outputs))

    def trace(self) -> dict:
        """Host copy of the last launch's trace: meta [n,4] (smid, ntiles, t_first, t_exit),
        tr [n,cap,2] (tile start/end ns), tid [n,cap]."""
        d = {"meta": self.state["sv_meta"].cpu().numpy()}
        if "sv_tr" in self.state:
            d["tr"] = self.state["sv_tr"].cpu().numpy()
            d["tid"] = self.state["sv_tid"].cpu().numpy()
        return d


def torch_tensor(values, like):
    import torch

    return torch.tensor(list(values), dtype=like.dtype, device=like.device)


def tpc_split_perm(n: int, canonical: list[int] | None = None) -> list[int]:
    """Rank permutation that gives the two CTAs of a TPC (blocks 2k, 2k+1: smid pair) ranks
    n/2 apart, so TPC neighbours never run adjacent tile ids."""
    half = n // 2
    return [(b // 2) + (b % 2) * half for b in range(n)]


def reverse_perm(n: int) -> list[int]:
    return list(range(n))[::-1]


def random_perm(n: int, seed: int = 0) -> list[int]:
    import random

    r = list(range(n))
    random.Random(seed).shuffle(r)
    return r
