"""GEMM tile program: C[M, N] = A[M, K] @ B[N, K]^T  ("NT" layout).

Layouts (row-major):
    A [M, K] bf16    activations
    B [N, K] bf16    weights, stored like torch.nn.Linear.weight (so C = A @ B.T)
    C [M, N] bf16    output; fp32 accumulation in registers

Tile = (tm, tn, ks): output block C[tm*bM:(tm+1)*bM, tn*bN:(tn+1)*bN] restricted to the
K-slice ks of `split_k` equal slices. With split_k == 1 a tile is a whole output block.
With split_k > 1 the S tiles of one output block are independent members of the flat
tile bag; the combine runs inside the kernel ("last-arriving tile combines"):

    every split tile   : write fp32 partial -> Wk[ks, block]; __syncthreads
                         thread 0: prev = atomic_add(Ctr[block], 1, acq_rel)
    the tile seeing prev == S-1 (the last to arrive):
                         C_block = sum_{s=0..S-1} Wk[s, block]  (fixed order s = 0..S-1,
                         so the result does not depend on arrival order / scheduling),
                         write C_block, reset Ctr[block] = 0 (counters self-clean).

The release half of the acq_rel RMW (issued after a CTA barrier, i.e. the CUTLASS
`arrive_inc` pattern) publishes the CTA's partial; the acquire half on the last
arriver plus the following CTA barrier makes all S partials visible to all its threads.

Tile id order: ks is the fastest-varying index (the S tiles of a block are adjacent),
then (tm, tn) in grouped ("swizzled") order with groups of `group_m` M-blocks, as in
Triton's grouped matmul ordering. All of these are E0 choices.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import tilelang.language as T

from ..kernel import (
    PASS_CONFIGS_AUTO_WS,
    PASS_CONFIGS_NO_WS,
    IOParam,
    ScratchBuf,
    TileSpace,
    build_grid as _build_grid,
    build_persistent as _build_persistent,
    ceildiv,
    pmin,
)

NAME = "gemm"
DTYPE = "bfloat16"
ACCUM = "float32"

# numerics class of the MMA instruction used on sm_120 (T.gemm -> mma.sync m16n8k16).
MMA_SHAPE = "mma.sync.m16n8k16.bf16.f32"


@dataclass(frozen=True)
class GemmShape:
    M: int
    N: int
    K: int


@dataclass(frozen=True)
class GemmConfig:
    block_M: int = 128
    block_N: int = 128
    block_K: int = 64
    num_stages: int = 2
    threads: int = 128
    split_k: int = 1
    # "off": tl.disable_warp_specialized (cp.async + mma.sync pipeline).
    # "auto": TileLang's default on sm_120 (TMA + mbarrier producer/consumer warp
    #         specialization, adds 128 producer threads). Grid build only: persistent
    #         and CoKernel builds always use "off".
    ws: str = "off"
    # "smem": fragment -> smem -> global (TileLang lowers the smem->global copy to a TMA
    #         store); "direct": fragment -> global stores, no C staging buffer.
    epilogue: str = "smem"
    group_m: int = 8


Coords = namedtuple("GemmTile", "tm tn ks mn")
IO = namedtuple("GemmIO", "A B C Wk Ctr")


def cfg_tag(cfg: GemmConfig) -> str:
    return (
        f"{cfg.block_M}x{cfg.block_N}x{cfg.block_K}_s{cfg.num_stages}_t{cfg.threads}"
        f"_k{cfg.split_k}_ws{cfg.ws}_{cfg.epilogue}_g{cfg.group_m}"
    )


def validate(shape: GemmShape, cfg: GemmConfig) -> None:
    bM, bN, bK = cfg.block_M, cfg.block_N, cfg.block_K
    if bM % 16 or bN % 16 or bK % 16:
        raise ValueError("block_M/N/K must be multiples of 16 (mma m16n8k16)")
    if cfg.threads % 32 or not 32 <= cfg.threads <= 1024:
        raise ValueError("threads must be a multiple of 32 in [32, 1024]")
    if cfg.split_k < 1:
        raise ValueError("split_k >= 1")
    if shape.K % (bK * cfg.split_k):
        raise ValueError(f"K={shape.K} must be divisible by block_K*split_k={bK * cfg.split_k}")
    if shape.M % bM or shape.N % bN:
        # Ragged M/N would need predicated loads/stores; v0 does not implement them.
        raise ValueError(f"M, N must be multiples of block_M, block_N (got {shape.M}x{shape.N} vs {bM}x{bN})")
    if cfg.ws not in ("off", "auto"):
        raise ValueError("ws must be 'off' or 'auto'")
    if cfg.ws == "auto" and cfg.split_k != 1:
        # TileLang's producer/consumer WS pass does not support the conditional combine
        # code after the pipelined loop (producer_consumer_ws.cc: "No conditionally
        # guarded loop bodies"), so auto-WS is only offered for split_k == 1.
        raise ValueError("ws='auto' requires split_k == 1")
    if cfg.epilogue not in ("smem", "direct"):
        raise ValueError("epilogue must be 'smem' or 'direct'")
    if cfg.num_stages < 1 or cfg.group_m < 1:
        raise ValueError("num_stages >= 1 and group_m >= 1")


def _mn_tiles(shape: GemmShape, cfg: GemmConfig):
    return ceildiv(shape.M, cfg.block_M), ceildiv(shape.N, cfg.block_N)


def tile_space(shape: GemmShape, cfg: GemmConfig) -> TileSpace:
    m_tiles, n_tiles = _mn_tiles(shape, cfg)
    S = cfg.split_k
    gm = cfg.group_m
    per_group = gm * n_tiles

    def decode(tile_id):
        ks = tile_id % S
        mn = tile_id // S
        group = mn // per_group
        first_m = group * gm
        group_rows = pmin(m_tiles - first_m, gm)
        in_group = mn % per_group
        tm = first_m + in_group % group_rows
        tn = in_group // group_rows
        return Coords(tm, tn, ks, tm * n_tiles + tn)

    return TileSpace(num_tiles=m_tiles * n_tiles * S, decode=decode)


def io_params(shape: GemmShape, cfg: GemmConfig) -> list[IOParam]:
    M, N, K = shape.M, shape.N, shape.K
    ps = [
        IOParam("A", (M, K), DTYPE, "in"),
        IOParam("B", (N, K), DTYPE, "in"),
        IOParam("C", (M, N), DTYPE, "out"),
    ]
    if cfg.split_k > 1:
        m_tiles, n_tiles = _mn_tiles(shape, cfg)
        ps.append(IOParam("Wk", (cfg.split_k, M, N), ACCUM, "ws"))
        ps.append(IOParam("Ctr", (m_tiles * n_tiles,), "int32", "ctr"))
    return ps


def make_io(args: dict) -> IO:
    return IO(args["A"], args["B"], args["C"], args.get("Wk"), args.get("Ctr"))


def scratch_spec(shape: GemmShape, cfg: GemmConfig) -> list[ScratchBuf]:
    bufs = [
        ScratchBuf("A_s", (cfg.block_M, cfg.block_K), DTYPE),
        ScratchBuf("B_s", (cfg.block_N, cfg.block_K), DTYPE),
    ]
    if cfg.epilogue == "smem":
        bufs.append(ScratchBuf("C_s", (cfg.block_M, cfg.block_N), DTYPE))
    if cfg.split_k > 1:
        bufs.append(ScratchBuf("flag", (1,), "int32"))
    return bufs


def threads(cfg: GemmConfig) -> int:
    return cfg.threads


def pass_configs(cfg: GemmConfig, build: str) -> dict:
    if build == "grid" and cfg.ws == "auto":
        return dict(PASS_CONFIGS_AUTO_WS)
    return dict(PASS_CONFIGS_NO_WS)


def make_tile_body(shape: GemmShape, cfg: GemmConfig):
    validate(shape, cfg)
    ts = tile_space(shape, cfg)
    bM, bN, bK = cfg.block_M, cfg.block_N, cfg.block_K
    S = cfg.split_k
    kt = shape.K // (bK * S)  # k-blocks per split
    stages = cfg.num_stages
    smem_epilogue = cfg.epilogue == "smem"

    @T.macro
    def store_c(C_local, io, scr, m0, n0):
        if smem_epilogue:
            T.copy(C_local, scr.C_s)
            T.copy(scr.C_s, io.C[m0 : m0 + bM, n0 : n0 + bN])
        else:
            T.copy(C_local, io.C[m0 : m0 + bM, n0 : n0 + bN])

    @T.macro
    def tile_body(tile_id, io, scr):
        c = ts.decode(tile_id)
        m0 = c.tm * bM
        n0 = c.tn * bN
        k0 = c.ks * kt
        # The scratch may still be in use by the previous tile this CTA ran (possibly
        # of another role in a CoKernel): make every thread finish with it first.
        T.sync_threads()
        C_local = T.alloc_fragment((bM, bN), ACCUM)
        T.clear(C_local)
        for k in T.Pipelined(kt, num_stages=stages):
            T.copy(io.A[m0 : m0 + bM, (k0 + k) * bK : (k0 + k + 1) * bK], scr.A_s)
            T.copy(io.B[n0 : n0 + bN, (k0 + k) * bK : (k0 + k + 1) * bK], scr.B_s)
            T.gemm(scr.A_s, scr.B_s, C_local, transpose_B=True)
        if S == 1:
            store_c(C_local, io, scr, m0, n0)
        else:
            tx = T.get_thread_binding()
            T.copy(C_local, io.Wk[c.ks, m0 : m0 + bM, n0 : n0 + bN])
            T.sync_threads()
            if tx == 0:
                scr.flag[0] = T.atomic_add(io.Ctr[c.mn], 1, memory_order="acq_rel", return_prev=True)
            T.sync_threads()
            if scr.flag[0] == S - 1:
                T.clear(C_local)
                for s in T.serial(S):
                    for i, j in T.Parallel(bM, bN):
                        C_local[i, j] += io.Wk[s, m0 + i, n0 + j]
                store_c(C_local, io, scr, m0, n0)
                if tx == 0:
                    io.Ctr[c.mn] = 0

    return tile_body


def smem_bytes(shape: GemmShape, cfg: GemmConfig, build: str) -> int:
    """Predicted dynamic shared memory of the compiled kernel (bytes).

    Pipelined buffers are multi-versioned num_stages times. Inside a persistent loop
    TileLang's allocation merger keeps every scratch buffer live for the whole loop, so
    the persistent footprint is the sum of all buffers. In the grid build the merger
    overlaps the C staging buffer with the (dead) A/B pipeline buffers."""
    v = max(cfg.num_stages, 1)
    a = cfg.block_M * cfg.block_K * 2 * v
    b = cfg.block_N * cfg.block_K * 2 * v
    c = cfg.block_M * cfg.block_N * 2 if cfg.epilogue == "smem" else 0
    flag = 16 if cfg.split_k > 1 else 0
    if build == "persistent":
        return _align(a + b + c + flag)
    if cfg.ws == "auto":
        # TMA pipeline: 2*stages 8-byte mbarriers live in *static* smem, and the static
        # region is padded to 1 KB because the dynamic arena is 1024-byte aligned.
        return _align(max(a + b, c) + flag) + 1024
    return _align(max(a + b, c + flag))


def _align(n: int, a: int = 16) -> int:
    return -(-n // a) * a


def numerics(cfg: GemmConfig) -> tuple[str, tuple]:
    """(class, key). Configs with equal keys are expected to be bitwise identical:
    block_M/N/K, stages, threads, ws, epilogue and rasterization do not change the
    per-element K accumulation order as long as the MMA instruction is the same
    (proposal §5.5, E0); split_k changes it (E1)."""
    return ("E0" if cfg.split_k == 1 else "E1", (MMA_SHAPE, cfg.split_k))


# ----------------------------------------------------------------------------------
# config space
# ----------------------------------------------------------------------------------

_TILES = [(64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]


def _acc_regs_ok(bM: int, bN: int, threads: int) -> bool:
    # fp32 accumulators per thread; > 128 spills on 255-register threads
    return 16 <= bM * bN // threads <= 128


def configs(shape: GemmShape, smem_limit: int = 101376, max_configs: int = 50) -> list[GemmConfig]:
    """Enumerate valid configs covering tile shape, block_K, stages, threads, split-K,
    epilogue and warp specialization. Every returned config passes `validate` and fits
    `smem_limit` in *both* builds (persistent footprint is the binding one)."""
    out: list[GemmConfig] = []

    def add(**kw):
        cfg = GemmConfig(**kw)
        try:
            validate(shape, cfg)
        except ValueError:
            return
        if not _acc_regs_ok(cfg.block_M, cfg.block_N, cfg.threads):
            return
        if max(smem_bytes(shape, cfg, b) for b in ("grid", "persistent")) > smem_limit:
            return
        if cfg not in out:
            out.append(cfg)

    # F1: E0 baseline (split 1, no WS, smem epilogue): tile x threads x (bK, stages)
    for bM, bN in _TILES:
        for thr in (128, 256):
            for bK, st in ((32, 3), (64, 2), (64, 3)):
                add(block_M=bM, block_N=bN, block_K=bK, num_stages=st, threads=thr)
    # F2: direct epilogue -> no C staging buffer; frees smem for deeper pipelines and
    # is the only way the big 128x256 / 256x128 tiles fit a persistent CTA
    for bM, bN, thr in ((128, 128, 128), (128, 128, 256), (64, 128, 128)):
        for bK, st in ((64, 3), (64, 4)):
            add(block_M=bM, block_N=bN, block_K=bK, num_stages=st, threads=thr, epilogue="direct")
    for bM, bN in ((128, 256), (256, 128)):
        for bK, st in ((32, 2), (32, 3), (64, 2)):
            add(block_M=bM, block_N=bN, block_K=bK, num_stages=st, threads=256, epilogue="direct")
    # F3: split-K (E1), in-kernel last-arriver combine
    for bM, bN, thr in ((64, 64, 128), (64, 128, 128), (128, 128, 128), (128, 128, 256)):
        for sk in (2, 4):
            add(block_M=bM, block_N=bN, block_K=64, num_stages=2, threads=thr, split_k=sk)
    # F4: TileLang default auto warp specialization (grid build only)
    for bM, bN, thr in ((64, 128, 128), (128, 128, 128), (128, 128, 256), (128, 256, 256)):
        add(block_M=bM, block_N=bN, block_K=64, num_stages=2, threads=thr, ws="auto")
    return out[:max_configs]


# ----------------------------------------------------------------------------------
# builders, inputs, reference
# ----------------------------------------------------------------------------------


def build_grid(shape: GemmShape, cfg: GemmConfig):
    import sys

    return _build_grid(sys.modules[__name__], shape, cfg)


def build_persistent(shape: GemmShape, cfg: GemmConfig, num_ctas: int):
    import sys

    return _build_persistent(sys.modules[__name__], shape, cfg, num_ctas)


def make_inputs(shape: GemmShape, seed: int = 0, device: str = "cuda") -> dict:
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(shape.M, shape.K, generator=g, device=device).to(torch.bfloat16)
    B = (torch.randn(shape.N, shape.K, generator=g, device=device) / shape.K**0.5).to(torch.bfloat16)
    return {"A": A, "B": B}


def reference(shape: GemmShape, inputs: dict) -> dict:
    A, B = inputs["A"].float(), inputs["B"].float()
    return {"C": A @ B.T}


# (atol, rtol) for |out - ref| <= atol + rtol*|ref|: bf16 output rounding is <= 2^-9
# relative; fp32 accumulation-order differences are ~1e-6 relative.
TOLERANCE = (1e-3, 2**-7)
