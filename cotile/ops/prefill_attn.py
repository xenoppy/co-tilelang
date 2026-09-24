"""Causal prefill attention tile program (one request, GQA), FlashAttention-2 style.

Layouts (row-major, FlashInfer's "NHD" layout of a single request, batch 1):
    Q [S, Hq, D]     bf16
    K [S, Hkv, D]    bf16
    V [S, Hkv, D]    bf16
    O [S, Hq, D]     bf16
Query head h attends with KV head h // G, G = Hq / Hkv (FlashInfer's GQA convention).
Causal (default): query position p attends to key positions 0..p. Softmax state (running
max, running sum, output accumulator) is fp32; P is rounded to bf16 before the PV GEMM
(as in FlashAttention-2 and FlashInfer).

Head configuration. The P4 study pairs this op with GQA decode in POD's scenario, and
FlashInfer's POD requires prefill and decode to share (Hq, Hkv, D) (pod.cuh asserts it; the
wrapper `research/bench/baselines/flashinfer_ops.POD` checks it). The defaults therefore
match the decode op: Hq = 32, Hkv = 8, D = 128 (Llama-3-8B), the configuration of the
2026-09-22 FlashInfer POD baseline.

Tile = (qb, h): query block qb (positions qb*block_M .. +block_M) of query head h, i.e. a
block_M x D output block. The tile streams the causal K/V range of KV head h // G in blocks
of block_N (`nkv(qb) = ceil((qb+1)*block_M / block_N)` blocks; non-causal: S / block_N). All
warps are laid out along M (T.GemmWarpPolicy.FullRow), so every row of S = Q K^T and O is
owned by one warp: the row max / sum reductions are warp-local (no cross-warp smem
reduction, no named barriers) and P stays in registers as the A operand of the PV GEMM.

Per-tile work differs by up to nq x under causal masking. `tile_space(...).work(t)` returns
the MMA FLOPs issued by tile t (Python int), so a scheduler can order tiles longest-first.
Tile order (config `order`):
    "lpt"      (default) longest first: tile t -> qb = nq-1 - t // Hq, h = t % Hq. The Hq
               heads of one query block are adjacent (equal work; the G query heads of a KV
               group run back to back and share their K/V stream through L2). This is the
               order of the grid build's blockIdx, of the persistent build's grid-stride and
               of a CoKernel's dynamic queue.
    "natural"  shortest first: qb = t // Hq (control; the classic tail-heavy order).
Both orders are bitwise-neutral (a tile's arithmetic does not depend on its id).

Causal masking. For KV block k the score tile is initialised with 0 (visible) or -inf
(masked) and the QK^T GEMM accumulates into it (same cost as a clear). Fully masked rows
of a block contribute exactly nothing: alpha = exp2(0) = 1, P = 0, so extra blocks beyond a
row's diagonal are bitwise-neutral. Every row sees key 0, so the running max is finite after
the first block.

Numerics (E1). Scores are identical for every config (same m16n8k16 MMA, same k16 order
over D). Configs with the same block_N (online-softmax rescale points, PV accumulation
order) give bitwise-identical outputs: block_M, threads, num_stages, epilogue and order are
neutral. As in the decode op, the exponent is written exp2((s - m) * scale) (a subtraction
feeding a multiply cannot be contracted into FFMA by ptxas; see cotile/README.md).
The exponential is the accurate exp2f (no --use_fast_math: a pass config would also apply
to a CoKernel partner and change its bits).
"""

from __future__ import annotations

import math
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
    ceildiv,
)

NAME = "prefill_attn"
DTYPE = "bfloat16"
ACCUM = "float32"
LOG2E = 1.44269504088896340736
ORDERS = ("lpt", "natural")
EPILOGUES = ("smem", "direct")


@dataclass(frozen=True)
class PrefillShape:
    seqlen: int  # qo_len == kv_len (one request, no KV prefix)
    heads: int = 32
    kv_heads: int = 8
    dim: int = 128
    causal: bool = True

    @property
    def group(self) -> int:
        return self.heads // self.kv_heads


@dataclass(frozen=True)
class PrefillConfig:
    block_M: int = 128  # query positions per tile
    block_N: int = 64  # KV positions per pipelined step
    num_stages: int = 2  # K/V smem buffers (software pipeline depth)
    threads: int = 128  # warps along M (FullRow): block_M % (16 * warps) == 0
    # "smem": O staged through the (dead) Q_s buffer -> 16-byte coalesced stores;
    # "direct": fragment -> global stores (no staging, no extra barrier).
    epilogue: str = "smem"
    order: str = "lpt"


Coords = namedtuple("PrefillTile", "qb h kvh")
IO = namedtuple("PrefillIO", "Q K V O")


def cfg_tag(cfg: PrefillConfig) -> str:
    return (f"m{cfg.block_M}_n{cfg.block_N}_s{cfg.num_stages}_t{cfg.threads}"
            + ("" if cfg.epilogue == "smem" else f"_{cfg.epilogue}")
            + ("" if cfg.order == "lpt" else f"_{cfg.order}"))


def validate(shape: PrefillShape, cfg: PrefillConfig) -> None:
    if shape.heads % shape.kv_heads:
        raise ValueError("heads must be a multiple of kv_heads")
    if cfg.threads % 32 or not 32 <= cfg.threads <= 1024:
        raise ValueError("threads must be a multiple of 32 in [32, 1024]")
    warps = cfg.threads // 32
    if cfg.block_M % (16 * warps):
        raise ValueError(f"block_M={cfg.block_M} must be a multiple of 16*warps={16 * warps} (FullRow)")
    if cfg.block_N % 16 or shape.dim % 16:
        raise ValueError("block_N and dim must be multiples of 16")
    if shape.seqlen % cfg.block_M or shape.seqlen % cfg.block_N:
        # ragged sequence ends would need predicated loads / row masks; not implemented
        raise ValueError(f"seqlen={shape.seqlen} must be a multiple of block_M and block_N")
    if cfg.num_stages < 1:
        raise ValueError("num_stages >= 1")
    if cfg.epilogue not in EPILOGUES:
        raise ValueError(f"epilogue must be one of {EPILOGUES}")
    if cfg.order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")


def num_kv_blocks(shape: PrefillShape, cfg: PrefillConfig, qb):
    """K/V blocks streamed by a tile of query block qb (int or PrimExpr)."""
    if not shape.causal:
        return shape.seqlen // cfg.block_N
    return ((qb + 1) * cfg.block_M + cfg.block_N - 1) // cfg.block_N


def tile_space(shape: PrefillShape, cfg: PrefillConfig) -> TileSpace:
    nq = shape.seqlen // cfg.block_M
    H, G = shape.heads, shape.group
    lpt = cfg.order == "lpt"

    def decode(tile_id):
        r = tile_id // H
        h = tile_id % H
        qb = (nq - 1 - r) if lpt else r
        return Coords(qb, h, h // G)

    def work(tile_id: int) -> int:
        """MMA FLOPs issued by tile `tile_id` (QK^T and PV, masked columns included)."""
        c = decode(int(tile_id))
        return 4 * cfg.block_M * cfg.block_N * shape.dim * num_kv_blocks(shape, cfg, c.qb)

    return TileSpace(num_tiles=nq * H, decode=decode, work=work)


def io_params(shape: PrefillShape, cfg: PrefillConfig) -> list[IOParam]:
    S, Hq, Hkv, D = shape.seqlen, shape.heads, shape.kv_heads, shape.dim
    return [
        IOParam("Q", (S, Hq, D), DTYPE, "in"),
        IOParam("K", (S, Hkv, D), DTYPE, "in"),
        IOParam("V", (S, Hkv, D), DTYPE, "in"),
        IOParam("O", (S, Hq, D), DTYPE, "out"),
    ]


def make_io(args: dict) -> IO:
    return IO(args["Q"], args["K"], args["V"], args["O"])


def scratch_spec(shape: PrefillShape, cfg: PrefillConfig) -> list[ScratchBuf]:
    return [
        ScratchBuf("Q_s", (cfg.block_M, shape.dim), DTYPE),  # also the O staging buffer
        ScratchBuf("K_s", (cfg.block_N, shape.dim), DTYPE),
        ScratchBuf("V_s", (cfg.block_N, shape.dim), DTYPE),
    ]


def threads(cfg: PrefillConfig) -> int:
    return cfg.threads


def pass_configs(cfg: PrefillConfig, build: str) -> dict:
    return dict(PASS_CONFIGS_NO_WS)


def make_tile_body(shape: PrefillShape, cfg: PrefillConfig):
    validate(shape, cfg)
    ts = tile_space(shape, cfg)
    D = shape.dim
    bM, bN, stages = cfg.block_M, cfg.block_N, cfg.num_stages
    causal = shape.causal
    sc = LOG2E / math.sqrt(D)  # softmax scale folded into exp2
    smem_epilogue = cfg.epilogue == "smem"
    NEG_INF = -T.infinity(ACCUM)

    @T.macro
    def tile_body(tile_id, io, scr):
        c = ts.decode(tile_id)
        q0 = c.qb * bM
        T.sync_threads()  # scratch reuse guard (see gemm.py)
        acc_s = T.alloc_fragment((bM, bN), ACCUM)
        acc_p = T.alloc_fragment((bM, bN), DTYPE)
        acc_o = T.alloc_fragment((bM, D), ACCUM)
        m = T.alloc_fragment((bM,), ACCUM)
        m_prev = T.alloc_fragment((bM,), ACCUM)
        alpha = T.alloc_fragment((bM,), ACCUM)
        rsum = T.alloc_fragment((bM,), ACCUM)
        l = T.alloc_fragment((bM,), ACCUM)

        T.copy(io.Q[q0 : q0 + bM, c.h, :], scr.Q_s)
        T.fill(acc_o, 0)
        T.fill(l, 0)
        T.fill(m, NEG_INF)
        nkv = num_kv_blocks(shape, cfg, c.qb)
        for k in T.Pipelined(nkv, num_stages=stages):
            T.copy(io.K[k * bN : (k + 1) * bN, c.kvh, :], scr.K_s)
            if causal:
                for i, j in T.Parallel(bM, bN):
                    acc_s[i, j] = T.if_then_else(q0 + i >= k * bN + j, 0, NEG_INF)
            else:
                T.clear(acc_s)
            T.gemm(scr.Q_s, scr.K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(m, m_prev)
            T.reduce_max(acc_s, m, dim=1, clear=True)
            for i in T.Parallel(bM):
                m[i] = T.max(m[i], m_prev[i])
            # (x - m) * sc, not x * sc - m * sc: see the module docstring
            for i in T.Parallel(bM):
                alpha[i] = T.exp2((m_prev[i] - m[i]) * sc)
            for i, j in T.Parallel(bM, bN):
                acc_s[i, j] = T.exp2((acc_s[i, j] - m[i]) * sc)
            T.reduce_sum(acc_s, rsum, dim=1)
            for i in T.Parallel(bM):
                l[i] = l[i] * alpha[i] + rsum[i]
            T.copy(acc_s, acc_p)
            for i, j in T.Parallel(bM, D):
                acc_o[i, j] *= alpha[i]
            T.copy(io.V[k * bN : (k + 1) * bN, c.kvh, :], scr.V_s)
            T.gemm(acc_p, scr.V_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
        for i, j in T.Parallel(bM, D):
            acc_o[i, j] = acc_o[i, j] / l[i]
        if smem_epilogue:
            # Q_s is dead after the last QK^T GEMM; every warp must be done reading it
            T.sync_threads()
            T.copy(acc_o, scr.Q_s)
            T.copy(scr.Q_s, io.O[q0 : q0 + bM, c.h, :])
        else:
            T.copy(acc_o, io.O[q0 : q0 + bM, c.h, :])

    return tile_body


def _align(n: int, a: int = 16) -> int:
    return -(-n // a) * a


def smem_bytes(shape: PrefillShape, cfg: PrefillConfig, build: str) -> int:
    """Predicted per-CTA user shared memory: Q_s + num_stages copies of K_s and V_s (the
    software pipeline multi-versions both). FullRow keeps the row reductions warp-local, so
    there is no hidden AllReduce workspace. Same for both builds."""
    q = cfg.block_M * shape.dim * 2
    kv = 2 * max(cfg.num_stages, 1) * cfg.block_N * shape.dim * 2
    return _align(q + kv)


def numerics(cfg: PrefillConfig) -> tuple[str, tuple]:
    """block_N (online-softmax rescale points and the PV accumulation order) changes the
    bits; block_M, threads, num_stages, epilogue and tile order do not (see module doc)."""
    return ("E1", (cfg.block_N,))


def configs(shape: PrefillShape, smem_limit: int = 101376, max_configs: int = 50) -> list[PrefillConfig]:
    """Tile shape x KV block x pipeline depth x threads, plus the direct epilogue and the
    natural (shortest-first) order as controls. Every returned config passes `validate` and
    fits `smem_limit` (same footprint in both builds)."""
    out: list[PrefillConfig] = []

    def add(**kw):
        cfg = PrefillConfig(**kw)
        try:
            validate(shape, cfg)
        except ValueError:
            return
        if smem_bytes(shape, cfg, "persistent") > smem_limit:
            return
        # fp32 accumulators per thread (O + S); > 192 leaves no room below 255 registers
        if (cfg.block_M * shape.dim + cfg.block_M * cfg.block_N) // cfg.threads > 192:
            return
        if cfg not in out:
            out.append(cfg)

    # F1: tile x KV block x stages x threads (smem epilogue, longest-first order)
    for bM, thr_list in ((64, (128,)), (128, (128, 256))):
        for bN in (32, 64, 128):
            for st in (1, 2, 3):
                for thr in thr_list:
                    add(block_M=bM, block_N=bN, num_stages=st, threads=thr)
    # F2: 256-row tiles (8 warps x 32 rows, or 16 warps x 16 rows)
    for bN, st in ((32, 1), (32, 2), (64, 1)):
        for thr in (256, 512):
            add(block_M=256, block_N=bN, num_stages=st, threads=thr)
    # F3: direct epilogue (no staging barrier) for the main tile shapes
    for bM, bN, st, thr in ((64, 64, 2, 128), (128, 64, 2, 256), (128, 32, 2, 128), (128, 64, 2, 128)):
        add(block_M=bM, block_N=bN, num_stages=st, threads=thr, epilogue="direct")
    # F4: natural (shortest-first) order, control for the tile-order axis
    for bM, bN, st, thr in ((64, 64, 2, 128), (128, 64, 2, 256)):
        add(block_M=bM, block_N=bN, num_stages=st, threads=thr, order="natural")
    return out[:max_configs]


def build_grid(shape: PrefillShape, cfg: PrefillConfig):
    import sys

    return _build_grid(sys.modules[__name__], shape, cfg)


def build_persistent(shape: PrefillShape, cfg: PrefillConfig, num_ctas: int):
    import sys

    return _build_persistent(sys.modules[__name__], shape, cfg, num_ctas)


def make_inputs(shape: PrefillShape, seed: int = 0, device: str = "cuda") -> dict:
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    S, Hq, Hkv, D = shape.seqlen, shape.heads, shape.kv_heads, shape.dim
    # Q scaled up so rows are peaked rather than near-uniform (exercises the rescaling)
    Q = (2.0 * torch.randn(S, Hq, D, generator=g, device=device)).to(torch.bfloat16)
    K = torch.randn(S, Hkv, D, generator=g, device=device).to(torch.bfloat16)
    V = torch.randn(S, Hkv, D, generator=g, device=device).to(torch.bfloat16)
    return {"Q": Q, "K": K, "V": V}


def reference(shape: PrefillShape, inputs: dict, dtype=None) -> dict:
    """fp32 torch SDPA (causal as the shape says); K/V heads expanded with repeat_interleave
    (query head h -> KV head h // G)."""
    import torch

    dtype = dtype or torch.float32
    G = shape.group
    q = inputs["Q"].to(dtype).transpose(0, 1).unsqueeze(0)  # [1, Hq, S, D]
    k = inputs["K"].to(dtype).repeat_interleave(G, dim=1).transpose(0, 1).unsqueeze(0)
    v = inputs["V"].to(dtype).repeat_interleave(G, dim=1).transpose(0, 1).unsqueeze(0)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=shape.causal)
    return {"O": o.squeeze(0).transpose(0, 1).contiguous().float()}


def flops(shape: PrefillShape) -> float:
    """Useful FLOPs (QK^T + PV, 2 per MAC), causal = S(S+1)/2 visible pairs per head."""
    S, Hq, D = shape.seqlen, shape.heads, shape.dim
    pairs = S * (S + 1) / 2 if shape.causal else S * S
    return 4.0 * pairs * D * Hq


# |out - ref| <= atol*rms(ref) + rtol*|ref| (cotile.tests.harness.compare): P is rounded to
# bf16 before the PV GEMM and O to bf16 at the end (FlashAttention-2 / FlashInfer do the same).
TOLERANCE = (2**-5, 2**-6)
