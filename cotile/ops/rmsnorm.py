"""RMSNorm tile program: Y[t, :] = X[t, :] * rsqrt(mean(X[t, :]^2) + eps) * W.

Layouts: X [T, D] bf16, W [D] bf16, Y [T, D] bf16; fp32 accumulation.

Tile = `rows_per_cta` consecutive rows (R). Written SIMT-style so the thread mapping,
the vector width and the reduction tree are explicit parameters rather than TileLang
layout-inference choices:
  * thread tx owns, in each row, columns (k*threads + tx)*vec + [0, vec) for
    k = 0 .. D/(threads*vec)-1; loads/stores are `vec`-wide vector accesses;
  * the whole tile stays in registers between the reduction and the scaling pass
    (X is read once);
  * sum of squares: per-thread sequential (k, v order) -> warp butterfly
    (T.warp_reduce_sum) -> per-warp partials in the shared scratch `red[R, warps]` ->
    every thread sums the warp partials in fixed order w = 0..warps-1.
The reduction uses only __syncthreads (no TileLang AllReduce, whose cross-warp path
hard-codes named barriers 1/2 and adds a hidden smem workspace).
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import tilelang.language as T

from ..kernel import (
    PASS_CONFIGS_NO_WS,
    IOParam,
    ScratchBuf,
    TileSpace,
    build_grid as _build_grid,
    build_persistent as _build_persistent,
)

NAME = "rmsnorm"
DTYPE = "bfloat16"
ACCUM = "float32"


@dataclass(frozen=True)
class RMSNormShape:
    tokens: int
    hidden: int
    eps: float = 1e-6


@dataclass(frozen=True)
class RMSNormConfig:
    rows_per_cta: int = 1
    threads: int = 256
    vec: int = 8  # elements per vector access (8 x bf16 = 16 B)


Coords = namedtuple("RMSNormTile", "row0")
IO = namedtuple("RMSNormIO", "X W Y")


def cfg_tag(cfg: RMSNormConfig) -> str:
    return f"r{cfg.rows_per_cta}_t{cfg.threads}_v{cfg.vec}"


MAX_ELEMS_PER_THREAD = 128  # register budget for the tile held in registers


def validate(shape: RMSNormShape, cfg: RMSNormConfig) -> None:
    if cfg.threads % 32 or not 32 <= cfg.threads <= 1024:
        raise ValueError("threads must be a multiple of 32 in [32, 1024]")
    if cfg.vec not in (1, 2, 4, 8):
        raise ValueError("vec must be 1, 2, 4 or 8")
    if shape.hidden % (cfg.threads * cfg.vec):
        raise ValueError(f"hidden={shape.hidden} must be divisible by threads*vec={cfg.threads * cfg.vec}")
    if cfg.rows_per_cta < 1 or shape.tokens % cfg.rows_per_cta:
        raise ValueError(f"tokens={shape.tokens} must be divisible by rows_per_cta={cfg.rows_per_cta}")
    if cfg.rows_per_cta * shape.hidden // cfg.threads > MAX_ELEMS_PER_THREAD:
        raise ValueError("tile does not fit the per-thread register budget")


def tile_space(shape: RMSNormShape, cfg: RMSNormConfig) -> TileSpace:
    R = cfg.rows_per_cta
    return TileSpace(num_tiles=shape.tokens // R, decode=lambda t: Coords(t * R))


def io_params(shape: RMSNormShape, cfg: RMSNormConfig) -> list[IOParam]:
    return [
        IOParam("X", (shape.tokens, shape.hidden), DTYPE, "in"),
        IOParam("W", (shape.hidden,), DTYPE, "in"),
        IOParam("Y", (shape.tokens, shape.hidden), DTYPE, "out"),
    ]


def make_io(args: dict) -> IO:
    return IO(args["X"], args["W"], args["Y"])


def scratch_spec(shape: RMSNormShape, cfg: RMSNormConfig) -> list[ScratchBuf]:
    return [ScratchBuf("red", (cfg.rows_per_cta, cfg.threads // 32), ACCUM)]


def threads(cfg: RMSNormConfig) -> int:
    return cfg.threads


def pass_configs(cfg: RMSNormConfig, build: str) -> dict:
    return dict(PASS_CONFIGS_NO_WS)


def make_tile_body(shape: RMSNormShape, cfg: RMSNormConfig):
    validate(shape, cfg)
    ts = tile_space(shape, cfg)
    D, eps = shape.hidden, shape.eps
    R, NT, VEC = cfg.rows_per_cta, cfg.threads, cfg.vec
    NW = NT // 32
    L = D // NT  # elements per thread per row
    KV = L // VEC  # vector accesses per thread per row
    inv_d = 1.0 / D

    @T.macro
    def tile_body(tile_id, io, scr):
        c = ts.decode(tile_id)
        T.sync_threads()  # scratch reuse guard (see gemm.py)
        tx = T.get_thread_binding()
        lane = tx % 32
        warp = tx // 32
        x = T.alloc_local((R * L,), DTYPE)
        acc = T.alloc_local((R,), ACCUM)
        for r in T.unroll(R):
            for k in T.unroll(KV):
                for v in T.vectorized(VEC):
                    x[r * L + k * VEC + v] = io.X[c.row0 + r, (k * NT + tx) * VEC + v]
        for r in T.unroll(R):
            acc[r] = T.float32(0)
            for e in T.unroll(L):
                acc[r] += T.cast(x[r * L + e], ACCUM) * T.cast(x[r * L + e], ACCUM)
            acc[r] = T.warp_reduce_sum(acc[r])
            if lane == 0:
                scr.red[r, warp] = acc[r]
        T.sync_threads()
        for r in T.unroll(R):
            acc[r] = T.float32(0)
            for w in T.unroll(NW):
                acc[r] += scr.red[r, w]
            acc[r] = T.rsqrt(acc[r] * inv_d + eps)
            for k in T.unroll(KV):
                for v in T.vectorized(VEC):
                    io.Y[c.row0 + r, (k * NT + tx) * VEC + v] = T.cast(
                        T.cast(x[r * L + k * VEC + v], ACCUM) * acc[r] * T.cast(io.W[(k * NT + tx) * VEC + v], ACCUM),
                        DTYPE,
                    )

    return tile_body


def smem_bytes(shape: RMSNormShape, cfg: RMSNormConfig, build: str) -> int:
    return cfg.rows_per_cta * (cfg.threads // 32) * 4


def numerics(cfg: RMSNormConfig) -> tuple[str, tuple]:
    """threads and vec change which elements each thread sums and the reduction tree
    (E1); rows_per_cta does not (each row is reduced identically)."""
    return ("E1", (cfg.threads, cfg.vec))


def configs(shape: RMSNormShape, smem_limit: int = 101376, max_configs: int = 50) -> list[RMSNormConfig]:
    out = []
    for thr in (64, 128, 256, 512):
        for rows in (1, 2, 4, 8):
            for vec in (2, 4, 8):
                cfg = RMSNormConfig(rows_per_cta=rows, threads=thr, vec=vec)
                try:
                    validate(shape, cfg)
                except ValueError:
                    continue
                if smem_bytes(shape, cfg, "persistent") <= smem_limit:
                    out.append(cfg)
    return out[:max_configs]


def build_grid(shape: RMSNormShape, cfg: RMSNormConfig):
    import sys

    return _build_grid(sys.modules[__name__], shape, cfg)


def build_persistent(shape: RMSNormShape, cfg: RMSNormConfig, num_ctas: int):
    import sys

    return _build_persistent(sys.modules[__name__], shape, cfg, num_ctas)


def make_inputs(shape: RMSNormShape, seed: int = 0, device: str = "cuda") -> dict:
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(shape.tokens, shape.hidden, generator=g, device=device).to(torch.bfloat16)
    W = (1.0 + 0.1 * torch.randn(shape.hidden, generator=g, device=device)).to(torch.bfloat16)
    return {"X": X, "W": W}


def reference(shape: RMSNormShape, inputs: dict) -> dict:
    import torch

    X, W = inputs["X"].float(), inputs["W"].float()
    return {"Y": X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + shape.eps) * W}


TOLERANCE = (1e-3, 2**-7)
