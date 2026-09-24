"""GQA decode attention tile program (one query token per sequence).

Layouts (row-major, same as examples/flash_decoding/example_gqa_decode.py):
    Q [B, Hq, D]        bf16
    K [B, S, Hkv, D]    bf16
    V [B, S, Hkv, D]    bf16
    O [B, Hq, D]        bf16
Query head h attends with KV head h // G, G = Hq / Hkv. No mask: all S positions are
attended. Softmax state (running max, running sum, output accumulator) is fp32.

Tile = (b, kvh, hb, s): the `heads_per_cta` (HPC) query heads kvh*G + hb*HPC + [0, HPC)
of one KV group, over KV chunk s of `num_split` equal chunks. The HPC heads share every
K/V load. They are zero-padded to one MMA M-tile of 16 rows (block_H = 16) and the
warps of the CTA are laid out along N (T.GemmWarpPolicy.FullCol) in both GEMMs, so
more threads split the KV block (QK^T) and the head dim (PV) instead of computing
padding rows. P = softmax numerators is staged through shared memory (bf16) between
the two GEMMs because the two GEMMs partition warps along different axes.

split-KV > 1: every split tile writes its normalized partial output and its base-2
log-sum-exp to fp32 workspaces (Op, Lse); the last-arriving split of a (b, kvh, hb)
group (per-group int32 counter, acq_rel atomic, same protocol as gemm.py) merges the
S partials in fixed order s = 0..S-1 and resets the counter.
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
)

NAME = "gqa_decode"
DTYPE = "bfloat16"
ACCUM = "float32"
BLOCK_H = 16  # padded MMA M-tile holding the HPC query heads
LOG2E = 1.44269504088896340736


@dataclass(frozen=True)
class DecodeShape:
    batch: int
    seqlen: int
    heads: int = 32
    kv_heads: int = 8
    dim: int = 128

    @property
    def group(self) -> int:
        return self.heads // self.kv_heads


@dataclass(frozen=True)
class DecodeConfig:
    block_N: int = 64  # KV positions per pipelined step
    heads_per_cta: int = 4  # query heads of one KV group processed together (<= G)
    num_split: int = 1  # split-KV chunks, combined in-kernel
    threads: int = 128
    num_stages: int = 2
    # L2 eviction priority of the streaming K/V loads (cp.async .L2::cache_hint,
    # T.copy(..., eviction_policy=...)): "normal" | "evict_first" | "evict_last".
    # K/V are read exactly once per call, so the hint has no solo value; in a
    # co-run, evict_first keeps the stream from evicting the partner's L2-resident
    # data (e.g. a GEMM's operand panels). Data and instructions are unchanged
    # (bitwise-neutral).
    kv_l2: str = "normal"


L2_POLICIES = ("normal", "evict_first", "evict_last")
_L2_TAG = {"normal": "", "evict_first": "_l2ef", "evict_last": "_l2el"}


def l2_policy_arg(policy: str):
    """T.copy eviction_policy argument for a config's L2 policy (None = no hint)."""
    if policy not in L2_POLICIES:
        raise ValueError(f"L2 policy must be one of {L2_POLICIES}, got {policy!r}")
    return None if policy == "normal" else policy


Coords = namedtuple("DecodeTile", "b kvh hb s g")
IO = namedtuple("DecodeIO", "Q K V O Op Lse Ctr")


def cfg_tag(cfg: DecodeConfig) -> str:
    return (f"n{cfg.block_N}_h{cfg.heads_per_cta}_sp{cfg.num_split}_t{cfg.threads}_s{cfg.num_stages}"
            + _L2_TAG[cfg.kv_l2])


def validate(shape: DecodeShape, cfg: DecodeConfig) -> None:
    if shape.heads % shape.kv_heads:
        raise ValueError("heads must be a multiple of kv_heads")
    G = shape.group
    if not 1 <= cfg.heads_per_cta <= min(G, BLOCK_H) or G % cfg.heads_per_cta:
        raise ValueError(f"heads_per_cta must divide the group size {G} and be <= {BLOCK_H}")
    if cfg.threads % 32 or not 32 <= cfg.threads <= 1024:
        raise ValueError("threads must be a multiple of 32 in [32, 1024]")
    warps = cfg.threads // 32
    # FullCol: every warp owns >= one 8-wide MMA column block in both GEMMs.
    if cfg.block_N % (8 * warps) or shape.dim % (8 * warps):
        raise ValueError(f"block_N={cfg.block_N} and dim={shape.dim} must be multiples of 8*warps={8 * warps}")
    if cfg.block_N % 16 or shape.dim % 16:
        raise ValueError("block_N and dim must be multiples of 16")
    if warps >= 8 and cfg.block_N // warps >= 16:
        # TileLang limitation (not a semantic one): with >= 8 warps along the KV block
        # and >= 2 MMA n8-atoms per warp, T.reduce_max/T.reduce_sum lowering fails with
        # "ReduceOp cannot lower a layout where a source index depends on a
        # thread-owned reduce segment" (observed for block_N/warps = 128/8, 256/8,
        # 256/16; 64/8, 128/16 and every <= 4-warp case compile).
        raise ValueError("block_N/warps >= 16 with >= 8 warps is not lowerable by TileLang's ReduceOp")
    if cfg.num_split < 1 or shape.seqlen % (cfg.num_split * cfg.block_N):
        raise ValueError(f"seqlen={shape.seqlen} must be divisible by num_split*block_N={cfg.num_split * cfg.block_N}")
    if cfg.num_stages < 1:
        raise ValueError("num_stages >= 1")
    l2_policy_arg(cfg.kv_l2)


def tile_space(shape: DecodeShape, cfg: DecodeConfig) -> TileSpace:
    S = cfg.num_split
    HB = shape.group // cfg.heads_per_cta
    Hkv = shape.kv_heads

    def decode(tile_id):
        s = tile_id % S
        g = tile_id // S
        hb = g % HB
        r = g // HB
        return Coords(r // Hkv, r % Hkv, hb, s, g)

    return TileSpace(num_tiles=shape.batch * Hkv * HB * S, decode=decode)


def io_params(shape: DecodeShape, cfg: DecodeConfig) -> list[IOParam]:
    B, S, Hq, Hkv, D = shape.batch, shape.seqlen, shape.heads, shape.kv_heads, shape.dim
    ps = [
        IOParam("Q", (B, Hq, D), DTYPE, "in"),
        IOParam("K", (B, S, Hkv, D), DTYPE, "in"),
        IOParam("V", (B, S, Hkv, D), DTYPE, "in"),
        IOParam("O", (B, Hq, D), DTYPE, "out"),
    ]
    if cfg.num_split > 1:
        HB = shape.group // cfg.heads_per_cta
        ps.append(IOParam("Op", (B, Hq, cfg.num_split, D), ACCUM, "ws"))
        ps.append(IOParam("Lse", (B, Hq, cfg.num_split), ACCUM, "ws"))
        ps.append(IOParam("Ctr", (B * Hkv * HB,), "int32", "ctr"))
    return ps


def make_io(args: dict) -> IO:
    return IO(args["Q"], args["K"], args["V"], args["O"], args.get("Op"), args.get("Lse"), args.get("Ctr"))


def scratch_spec(shape: DecodeShape, cfg: DecodeConfig) -> list[ScratchBuf]:
    bufs = [
        ScratchBuf("Q_s", (BLOCK_H, shape.dim), DTYPE),
        ScratchBuf("K_s", (cfg.block_N, shape.dim), DTYPE),
        ScratchBuf("V_s", (cfg.block_N, shape.dim), DTYPE),
        ScratchBuf("P_s", (BLOCK_H, cfg.block_N), DTYPE),
    ]
    if cfg.num_split > 1:
        bufs.append(ScratchBuf("flag", (1,), "int32"))
    return bufs


def threads(cfg: DecodeConfig) -> int:
    return cfg.threads


def pass_configs(cfg: DecodeConfig, build: str) -> dict:
    # The example also disables WS ("TODO(lei): fix warp specialized pass").
    return dict(PASS_CONFIGS_NO_WS)


def make_tile_body(shape: DecodeShape, cfg: DecodeConfig):
    validate(shape, cfg)
    ts = tile_space(shape, cfg)
    D, G = shape.dim, shape.group
    BH, BN, HPC, S = BLOCK_H, cfg.block_N, cfg.heads_per_cta, cfg.num_split
    chunk = shape.seqlen // S
    nblk = chunk // BN
    stages = cfg.num_stages
    sc = LOG2E / math.sqrt(D)  # softmax scale folded into exp2
    kv_hint = l2_policy_arg(cfg.kv_l2)

    @T.macro
    def tile_body(tile_id, io, scr):
        c = ts.decode(tile_id)
        h0 = c.kvh * G + c.hb * HPC
        s0 = c.s * chunk
        T.sync_threads()  # scratch reuse guard (see gemm.py)
        acc_s = T.alloc_fragment((BH, BN), ACCUM)
        acc_o = T.alloc_fragment((BH, D), ACCUM)
        m = T.alloc_fragment((BH,), ACCUM)
        m_prev = T.alloc_fragment((BH,), ACCUM)
        alpha = T.alloc_fragment((BH,), ACCUM)
        rsum = T.alloc_fragment((BH,), ACCUM)
        l = T.alloc_fragment((BH,), ACCUM)

        for i, d in T.Parallel(BH, D):
            scr.Q_s[i, d] = T.if_then_else(i < HPC, io.Q[c.b, h0 + i, d], T.cast(0, DTYPE))
        T.fill(acc_o, 0)
        T.fill(l, 0)
        T.fill(m, -T.infinity(ACCUM))
        for k in T.Pipelined(nblk, num_stages=stages):
            T.copy(io.K[c.b, s0 + k * BN : s0 + (k + 1) * BN, c.kvh, :], scr.K_s, eviction_policy=kv_hint)
            T.clear(acc_s)
            T.gemm(scr.Q_s, scr.K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            T.copy(m, m_prev)
            T.reduce_max(acc_s, m, dim=1, clear=True)
            for i in T.Parallel(BH):
                m[i] = T.max(m[i], m_prev[i])
            # (x - m) * sc rather than x * sc - m * sc: TileLang vectorizes the latter
            # into mul.rn.f32x2 + sub.rn.f32x2, which ptxas (CUDA 12.9, sm_120a)
            # contracts into FFMA despite the .rn modifiers, fusing a different operand
            # depending on the surrounding code; results then differ by 1-3 ulp between
            # e.g. pipeline depths. A subtraction feeding a multiply cannot be contracted.
            for i in T.Parallel(BH):
                alpha[i] = T.exp2((m_prev[i] - m[i]) * sc)
            for i, j in T.Parallel(BH, BN):
                acc_s[i, j] = T.exp2((acc_s[i, j] - m[i]) * sc)
            T.reduce_sum(acc_s, rsum, dim=1)
            for i in T.Parallel(BH):
                l[i] = l[i] * alpha[i] + rsum[i]
            T.copy(acc_s, scr.P_s)
            for i, j in T.Parallel(BH, D):
                acc_o[i, j] *= alpha[i]
            T.copy(io.V[c.b, s0 + k * BN : s0 + (k + 1) * BN, c.kvh, :], scr.V_s, eviction_policy=kv_hint)
            T.gemm(scr.P_s, scr.V_s, acc_o, policy=T.GemmWarpPolicy.FullCol)
        # Outputs are written straight from the accumulator fragment, rows < HPC only
        # (a region copy of a 1-row fragment slice fails TileLang layout inference).
        if S == 1:
            for i, j in T.Parallel(BH, D):
                if i < HPC:
                    io.O[c.b, h0 + i, j] = T.cast(acc_o[i, j] / l[i], DTYPE)
        else:
            tx = T.get_thread_binding()
            for i, j in T.Parallel(BH, D):
                if i < HPC:
                    io.Op[c.b, h0 + i, c.s, j] = acc_o[i, j] / l[i]
            for i in T.Parallel(BH):
                if i < HPC:
                    # base-2 log-sum-exp of the scaled scores
                    io.Lse[c.b, h0 + i, c.s] = T.log2(l[i]) + m[i] * sc
            T.sync_threads()
            if tx == 0:
                scr.flag[0] = T.atomic_add(io.Ctr[c.g], 1, memory_order="acq_rel", return_prev=True)
            T.sync_threads()
            if scr.flag[0] == S - 1:
                lmax = T.alloc_fragment((HPC, D), ACCUM)
                num = T.alloc_fragment((HPC, D), ACCUM)
                den = T.alloc_fragment((HPC, D), ACCUM)
                T.fill(lmax, -T.infinity(ACCUM))
                T.clear(num)
                T.clear(den)
                for s in T.serial(S):
                    for i, d in T.Parallel(HPC, D):
                        lmax[i, d] = T.max(lmax[i, d], io.Lse[c.b, h0 + i, s])
                for s in T.serial(S):
                    for i, d in T.Parallel(HPC, D):
                        num[i, d] += T.exp2(io.Lse[c.b, h0 + i, s] - lmax[i, d]) * io.Op[c.b, h0 + i, s, d]
                        den[i, d] += T.exp2(io.Lse[c.b, h0 + i, s] - lmax[i, d])
                for i, d in T.Parallel(HPC, D):
                    io.O[c.b, h0 + i, d] = num[i, d] / den[i, d]
                if tx == 0:
                    io.Ctr[c.g] = 0

    return tile_body


def _align(n: int, a: int = 16) -> int:
    return -(-n // a) * a


def smem_bytes(shape: DecodeShape, cfg: DecodeConfig, build: str) -> int:
    """Predicted per-CTA user shared memory (static + dynamic) of the compiled kernel.

    Scratch: Q_s, P_s, and K_s/V_s multi-versioned num_stages times. On top of that,
    TileLang's cross-warp T.reduce_max/T.reduce_sum (AllReduce) allocate a hidden
    workspace of threads*4 bytes per reduce call site. Software pipelining replicates
    the loop body (min(num_stages + 1, iterations) copies incl. peeled iterations); the
    persistent build keeps all of them live. The grid build's liveness-based planner
    reuses workspaces and sometimes dead scratch, so the grid figure is an upper bound.
    These workspaces are not part of the macro's scratch, so a CoKernel cannot alias
    them."""
    v = max(cfg.num_stages, 1)
    q = BLOCK_H * shape.dim * 2
    kv = 2 * v * cfg.block_N * shape.dim * 2
    p = BLOCK_H * cfg.block_N * 2
    # In the grid build the planner overlaps the split flag (and sometimes one
    # workspace) with dead buffers, so the grid figure is an upper bound.
    flag = 16 if cfg.num_split > 1 and build == "persistent" else 0
    nblk = shape.seqlen // (cfg.num_split * cfg.block_N)  # pipelined iterations per tile
    red_sites = 2 * min(v + 1, nblk) if build == "persistent" else 2
    red = red_sites * cfg.threads * 4 if cfg.threads > 32 else 0
    return _align(q + kv + p + red + flag)


def numerics(cfg: DecodeConfig) -> tuple[str, tuple]:
    """KV block size (online-softmax rescale points), split-KV (combine) and the thread
    count (cross-warp reduction tree of the row max/sum) change the bits: E1.
    heads_per_cta, num_stages and the K/V L2 policy do not (rows are independent in
    both GEMMs; a cache hint does not change the data)."""
    return ("E1", (cfg.block_N, cfg.num_split, cfg.threads))


def configs(shape: DecodeShape, smem_limit: int = 101376, max_configs: int = 50) -> list[DecodeConfig]:
    out: list[DecodeConfig] = []
    G = shape.group

    def add(**kw):
        cfg = DecodeConfig(**kw)
        try:
            validate(shape, cfg)
        except ValueError:
            return
        if max(smem_bytes(shape, cfg, b) for b in ("grid", "persistent")) > smem_limit:
            return
        if cfg not in out:
            out.append(cfg)

    hpc_all = [h for h in (1, 2, 4, 8) if h <= G and G % h == 0]
    hpc_max = hpc_all[-1]
    # F1: no split, all heads of a group per CTA: KV block x stages x threads
    for bn, st in ((32, 2), (32, 3), (64, 1), (64, 2), (128, 1)):
        for thr in (64, 128, 256):
            add(block_N=bn, heads_per_cta=hpc_max, num_split=1, threads=thr, num_stages=st)
    # F2: split-KV (E1), in-kernel combine
    for bn, st, thr in ((32, 2, 64), (64, 2, 128), (128, 1, 128), (64, 1, 256)):
        for sp in (2, 4, 8, 16):
            add(block_N=bn, heads_per_cta=hpc_max, num_split=sp, threads=thr, num_stages=st)
    # F3: fewer heads per CTA (more CTAs, K/V re-read per head block)
    for hpc in hpc_all[:-1]:
        for bn, st, thr, sp in ((64, 2, 128, 1), (64, 2, 128, 4), (32, 2, 64, 1), (128, 1, 128, 8)):
            add(block_N=bn, heads_per_cta=hpc, num_split=sp, threads=thr, num_stages=st)
    return out[:max_configs]


def build_grid(shape: DecodeShape, cfg: DecodeConfig):
    import sys

    return _build_grid(sys.modules[__name__], shape, cfg)


def build_persistent(shape: DecodeShape, cfg: DecodeConfig, num_ctas: int):
    import sys

    return _build_persistent(sys.modules[__name__], shape, cfg, num_ctas)


def make_inputs(shape: DecodeShape, seed: int = 0, device: str = "cuda") -> dict:
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    B, S, Hq, Hkv, D = shape.batch, shape.seqlen, shape.heads, shape.kv_heads, shape.dim
    # Q scaled up so attention is peaked rather than uniform (a stricter test of the
    # online-softmax rescaling than near-uniform weights).
    Q = (3.0 * torch.randn(B, Hq, D, generator=g, device=device)).to(torch.bfloat16)
    K = torch.randn(B, S, Hkv, D, generator=g, device=device).to(torch.bfloat16)
    V = torch.randn(B, S, Hkv, D, generator=g, device=device).to(torch.bfloat16)
    return {"Q": Q, "K": K, "V": V}


def reference(shape: DecodeShape, inputs: dict) -> dict:
    import torch

    Q, K, V = inputs["Q"].float(), inputs["K"].float(), inputs["V"].float()
    B, Hq, D = Q.shape
    Hkv = K.shape[2]
    q = Q.view(B, Hkv, Hq // Hkv, D)
    k = K.permute(0, 2, 1, 3)  # B Hkv S D
    v = V.permute(0, 2, 1, 3)
    p = torch.softmax(torch.einsum("bkgd,bksd->bkgs", q, k) / math.sqrt(D), dim=-1)
    o = torch.einsum("bkgs,bksd->bkgd", p, v).reshape(B, Hq, D)
    return {"O": o}


# |out - ref| <= atol*rms(ref) + rtol*|ref|. P is rounded to bf16 before the PV GEMM
# and O to bf16 at the end; with the peaked test inputs the max error is ~2.3% of
# rms(ref) -- identical to torch's own bf16 SDPA on the same inputs.
TOLERANCE = (2**-5, 2**-6)
