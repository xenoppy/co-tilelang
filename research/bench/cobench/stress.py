"""Synthetic stress kernels (plan §1.4 v0) and the building blocks of the co-location limit study
(research/results/2026-09-24_limit_study).

One configurable raw-CUDA kernel (NVRTC, ``CudaKernel``) covers every role layout.

A work (MMA), two families:
  * ``reg``  register-only bf16 ``mma.sync.m16n8k16`` loops (pure tensor pipe). A unit is one
    warp's ``a_iters`` x ``NACC`` mma instructions. The operands (``NSETS`` distinct register sets
    derived from the unit index) alternate between consecutive mma so that the tensor datapath
    toggles like in a real GEMM (constant operands draw less power).
  * ``gemm`` GEMM-like: a group of ``WM*WN`` warps computes one BMxBN tile over K (BK = 32
    k-steps, ``STAGES``-deep cp.async pipeline, 64 B rows XOR-swizzled in smem, ldmatrix.x4,
    warp tile (MT*16)x(NT*8)). The operands come from an L2-resident A[TM*128, K] / B[TN*128, K]
    (K-major, "TN") pair of buffers: the LSU / smem / L2 traffic of a real GEMM mainloop.
B work (stream): a unit is one warp reading a contiguous ``chunk`` of a DRAM buffer (> 2x L2)
  with 16 B vector loads, ``SU`` loads per lane per batch, double-buffered (2*SU in flight),
  optionally with an L2 evict-first policy and a write of the data to a second buffer
  (read+write). Solo stream kernels can instead use cp.async (``SKIND=1``: SU slots of 512 B per
  warp) or bulk copies (``SKIND=2``: cp.async.bulk + mbarrier, SU slots of SBULK bytes per warp).

Role layouts (``MODE``); every kernel is persistent with dynamic queues (atomic tickets):
  0 solo_a  every warp / group does A units
  1 solo_b  every warp does B units
  2 sm      SM-level roles: CTAs whose dense SM index is < split do A first, the others B
            first; a CTA whose queue is empty takes over the other queue (takeover)
  3 cta     CTA-level co-residence: blocks [0, split) do A first, the others B first (blocks b
            and b + nsm land on the same SM in the first wave: 2026-09-22_smid_probe); takeover
  4 ws      warp-specialised: warps [0, WA) do A first, [WA, NWARPS) B first; per-warp takeover
            (gemm: the B warps cannot take over tiles, they own no smem pipeline). Optional
            setmaxnreg rebalancing (SMAXNREG, sm_120a): A warps inc to RA, B warps dec to RB.
  5 sw      same-warp interleaving: combined unit u = A unit u + its share of the stream, the
            stream loads software-pipelined with the MMA work (reg: a register ring of SD
            packets, one packet consumed and re-issued between MMA bursts; gemm: each warp's
            packets are cp.async'd into a per-stage slot together with the operand tiles of the
            same k-step -- one joint pipeline -- and consumed after that k-step's MMA).

Work-completion check: every unit adds 1 to ``hits_a[u]`` / ``hits_b[c]`` (c = 32 KB stream
chunk) and adds a hash of its result to ``sums`` (A: hash of every accumulator element with its
coordinates, independent of the tiling; B: 64-bit sum of the 32-bit words read). After k
launches every hit must be k and the sums k x the reference (B: computed by torch from the
buffer). The tickets self-reset: the last CTA to finish zeroes them, so launches of one instance
must be stream-ordered (fork/join variants guarantee it).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field, replace

import torch

from . import cudrv
from .cudrv import CudaKernel, device_index

MMA_FLOP = 2 * 16 * 8 * 16            # one m16n8k16
PACKET = 512                          # bytes one warp moves with one 16 B load per lane
MODES = {"solo_a": 0, "solo_b": 1, "sm": 2, "cta": 3, "ws": 4, "sw": 5}
AKINDS = {"none": 0, "reg": 1, "gemm": 2}

SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

#define NWARPS (NTHREADS / 32)
#define G_WARPS (WM * WN)
#define BK 32
#define BM (WM * MT * 16)
#define BN (WN * NT * 8)
#define A_STAGE_BYTES (BM * BK * 2)
#define B_STAGE_BYTES (BN * BK * 2)
#if MODE == 5 && AK == 2
#define S_STAGE_BYTES (G_WARPS * SPK * 512)
#else
#define S_STAGE_BYTES 0
#endif
#define STAGE_BYTES (A_STAGE_BYTES + B_STAGE_BYTES + S_STAGE_BYTES)

__device__ __forceinline__ u32 hash32(u32 x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
__device__ __forceinline__ u32 lane_id() { return threadIdx.x & 31; }
__device__ __forceinline__ u32 smem_addr(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u64 warp_sum64(u64 v) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ int warp_grab(u32* q) {
  u32 t = 0;
  if (lane_id() == 0) t = atomicAdd(q, 1u);
  return (int)__shfl_sync(0xffffffffu, t, 0);
}
__device__ __forceinline__ void bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}

// ------------------------------------------------------------------ stream loads
__device__ __forceinline__ u64 make_policy() {
  u64 p = 0;
#if SHINT
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
#endif
  return p;
}
__device__ __forceinline__ uint4 ldg_s(const uint4* ptr, u64 pol) {
  uint4 v;
#if SHINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr), "l"(pol));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
#endif
  return v;
}
__device__ __forceinline__ void stg_s(uint4* ptr, uint4 v, u64 pol) {
#if SHINT
  asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.u32 [%0], {%1,%2,%3,%4}, %5;"
               :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol) : "memory");
#else
  asm volatile("st.global.L1::no_allocate.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
#endif
}
__device__ __forceinline__ u64 sum4(uint4 v) {
  return (u64)v.x + (u64)v.y + (u64)v.z + (u64)v.w;
}
__device__ __forceinline__ void cp_async16(u32 saddr, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(saddr), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 saddr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(saddr) : "memory");
  return v;
}

// B unit: one warp streams chunk_vec uint4 (16 B) of unit u; batches of SU loads per lane,
// double-buffered (batch i+1 is issued before batch i is consumed).
__device__ __forceinline__ u64 stream_unit_ldg(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                               int u, int chunk_vec, u64 pol) {
  const u32 lane = lane_id();
  const uint4* p = src + (size_t)u * chunk_vec + lane;
  uint4* q = dst + (size_t)u * chunk_vec + lane;
  u64 s = 0;
  uint4 v[SU], w[SU];
  #pragma unroll
  for (int j = 0; j < SU; ++j) v[j] = ldg_s(p + j * 32, pol);
  for (int i = 0; i < chunk_vec; i += 64 * SU) {
    const int i1 = i + 32 * SU;
    #pragma unroll
    for (int j = 0; j < SU; ++j) w[j] = ldg_s(p + i1 + j * 32, pol);
    #pragma unroll
    for (int j = 0; j < SU; ++j) {
      s += sum4(v[j]);
#if SRW
      stg_s(q + i + j * 32, v[j], pol);
#endif
    }
    const int i2 = i + 64 * SU;
    if (i2 < chunk_vec) {
      #pragma unroll
      for (int j = 0; j < SU; ++j) v[j] = ldg_s(p + i2 + j * 32, pol);
    }
    #pragma unroll
    for (int j = 0; j < SU; ++j) {
      s += sum4(w[j]);
#if SRW
      stg_s(q + i1 + j * 32, w[j], pol);
#endif
    }
  }
  return s;
}

#if MODE == 1 && SKIND == 1
// cp.async stream: per-warp ring of SU slots of 512 B (each lane moves and reads its own 16 B).
__device__ __forceinline__ u64 stream_unit_cpasync(const uint4* __restrict__ src, int u, int chunk_vec,
                                                   u32 ring) {
  const u32 lane = lane_id();
  const uint4* p = src + (size_t)u * chunk_vec + lane;
  const int np = chunk_vec / 32;
  u64 s = 0;
  #pragma unroll
  for (int j = 0; j < SU; ++j) {
    if (j < np) cp_async16(ring + j * 512 + lane * 16, p + j * 32);
    cp_commit();
  }
  for (int i = 0; i < np; ++i) {
    cp_wait<SU - 1>();
    const u32 slot = ring + (i % SU) * 512 + lane * 16;
    s += sum4(lds128(slot));       // consumed (scoreboard) before the slot is refilled
    const int nx = i + SU;
    if (nx < np) cp_async16(slot, p + nx * 32);
    cp_commit();
  }
  cp_wait<0>();
  return s;
}
#endif

#if MODE == 1 && SKIND == 2
// bulk-copy stream: per-warp ring of SU slots of SBULK bytes, one mbarrier per slot.
__device__ __forceinline__ void mbar_init(u32 bar, u32 cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 phase) {
  u32 done;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bar), "r"(phase) : "memory");
  } while (!done);
}
__device__ __forceinline__ void bulk_load(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ u64 stream_unit_bulk(const uint4* __restrict__ src, int u, int chunk_vec,
                                                u32 ring, u32 bars, u32& phases) {
  const u32 lane = lane_id();
  const char* base = (const char*)(src + (size_t)u * chunk_vec);
  const int nreq = chunk_vec * 16 / SBULK;
  u64 s = 0;
  if (lane == 0) {
    #pragma unroll
    for (int j = 0; j < SU; ++j)
      if (j < nreq) bulk_load(ring + j * SBULK, base + (size_t)j * SBULK, SBULK, bars + j * 8);
  }
  for (int r = 0; r < nreq; ++r) {
    const int sl = r % SU;
    mbar_wait(bars + sl * 8, (phases >> sl) & 1u);
    phases ^= 1u << sl;
    #pragma unroll
    for (int j = 0; j < SBULK / 512; ++j) s += sum4(lds128(ring + sl * SBULK + j * 512 + lane * 16));
    __syncwarp();                   // every lane has read the slot before it is refilled
    const int nx = r + SU;
    if (lane == 0 && nx < nreq) bulk_load(ring + sl * SBULK, base + (size_t)nx * SBULK, SBULK, bars + sl * 8);
  }
  return s;
}
#endif

// ------------------------------------------------------------------ MMA
__device__ __forceinline__ void mma16816(float* d, const u32* a, const u32* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// two bf16 values in [0.5, 1) with random signs and mantissas
__device__ __forceinline__ u32 bf16pair(u32 h) { return (h & 0x807F807Fu) | 0x3F003F00u; }

#if AK == 1
struct RegOps { u32 a[NSETS][4]; u32 b[NSETS][2]; };
__device__ __forceinline__ void reg_ops(RegOps& o, int u) {
  const u32 lane = lane_id();
  #pragma unroll
  for (int s = 0; s < NSETS; ++s) {
    #pragma unroll
    for (int k = 0; k < 4; ++k)
      o.a[s][k] = bf16pair(hash32((u32)u * 0x9E3779B1u + lane * 131u + s * 17u + k * 7919u + 1u));
    #pragma unroll
    for (int k = 0; k < 2; ++k)
      o.b[s][k] = bf16pair(hash32((u32)u * 0x85EBCA77u + lane * 257u + s * 29u + k * 104729u + 7u));
  }
}
__device__ __forceinline__ void reg_burst(float (&acc)[NACC][4], const RegOps& o, int iters) {
  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int j = 0; j < NACC; ++j) mma16816(acc[j], o.a[j % NSETS], o.b[j % NSETS]);
  }
}
__device__ __forceinline__ u64 reg_hash(const float (&acc)[NACC][4]) {
  const u32 lane = lane_id();
  u64 h = 0;
  #pragma unroll
  for (int j = 0; j < NACC; ++j)
    #pragma unroll
    for (int k = 0; k < 4; ++k) h += hash32(__float_as_uint(acc[j][k]) ^ hash32(lane * 64u + j * 4u + k));
  return h;
}
__device__ __forceinline__ u64 reg_unit(int u, int iters) {
  RegOps o; reg_ops(o, u);
  float acc[NACC][4];
  #pragma unroll
  for (int j = 0; j < NACC; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  reg_burst(acc, o, iters);
  return reg_hash(acc);
}
#endif

#if AK == 2
static_assert((BM * 4) % (G_WARPS * 32) == 0 && (BN * 4) % (G_WARPS * 32) == 0, "tile fill");
// 64 B rows (BK = 32 bf16): 16 B chunk c of row r is stored at chunk c ^ ((r >> 1) & 3), which
// is conflict-free for ldmatrix (8 rows x 16 B) and for the cp.async fills.
__device__ __forceinline__ u32 swz(int r, int c) { return (u32)(r * 64 + ((c ^ ((r >> 1) & 3)) << 4)); }

struct GemmArgs {
  const unsigned short* A; const unsigned short* B; int K, TM, TNx;
  const uint4* sbuf; int sw_p;      // sw: stream packets per warp per tile
  void* cdbg;                        // CDBG (tests): C is stored here as fp32 [TM*BM, TNx*BN]
};

__device__ __forceinline__ void gemm_load_stage(u32 st, const unsigned short* Ag, const unsigned short* Bg,
                                                int K, int kt, int gt) {
  #pragma unroll
  for (int it = 0; it < BM * 4 / (G_WARPS * 32); ++it) {
    const int idx = gt + it * G_WARPS * 32, r = idx >> 2, c = idx & 3;
    cp_async16(st + swz(r, c), Ag + (size_t)r * K + kt * BK + c * 8);
  }
  #pragma unroll
  for (int it = 0; it < BN * 4 / (G_WARPS * 32); ++it) {
    const int idx = gt + it * G_WARPS * 32, r = idx >> 2, c = idx & 3;
    cp_async16(st + A_STAGE_BYTES + swz(r, c), Bg + (size_t)r * K + kt * BK + c * 8);
  }
}

// One BMxBN tile over K by the G_WARPS warps of a group (gt = thread index in the group).
__device__ __forceinline__ void gemm_unit(int u, const GemmArgs& g, unsigned char* smem, int gt, int bar,
                                          u64& sa, u64& sb, u64 pol) {
  const int lane = gt & 31, gw = gt >> 5;
  const int wm = gw / WN, wn = gw % WN;
  const int tm = u % g.TM, tn = (u / g.TM) % g.TNx;
  const unsigned short* Ag = g.A + (size_t)tm * BM * g.K;
  const unsigned short* Bg = g.B + (size_t)tn * BN * g.K;
  const int nk = g.K / BK;
  const u32 s0 = smem_addr(smem);
#if MODE == 5
  // this warp's stream bytes of the unit: [u*G*P + gw*P, +P) packets; packet j is issued with
  // (and consumed after) k-step floor-ish(j * nk / P)
  const int P = g.sw_p;
  const uint4* sp = g.sbuf + ((size_t)u * G_WARPS * P + (size_t)gw * P) * 32 + lane;
#define SW_ISSUE(stage_addr, kt_) {                                                     \
      const int j0 = ((kt_) * P + nk - 1) / nk, j1 = (((kt_) + 1) * P + nk - 1) / nk;    \
      for (int j = j0; j < j1; ++j)                                                       \
        cp_async16((stage_addr) + A_STAGE_BYTES + B_STAGE_BYTES + (gw * SPK + (j - j0)) * 512 + lane * 16, \
                   sp + (size_t)j * 32); }
#endif
  float acc[MT][NT][4];
  #pragma unroll
  for (int i = 0; i < MT; ++i)
    #pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  #pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) {
      gemm_load_stage(s0 + s * STAGE_BYTES, Ag, Bg, g.K, s, gt);
#if MODE == 5
      SW_ISSUE(s0 + s * STAGE_BYTES, s)
#endif
    }
    cp_commit();
  }
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<STAGES - 2>();
    bar_sync(bar, G_WARPS * 32);
    const int nx = kt + STAGES - 1;
    if (nx < nk) {
      gemm_load_stage(s0 + (nx % STAGES) * STAGE_BYTES, Ag, Bg, g.K, nx, gt);
#if MODE == 5
      SW_ISSUE(s0 + (nx % STAGES) * STAGE_BYTES, nx)
#endif
    }
    cp_commit();
    const u32 st = s0 + (kt % STAGES) * STAGE_BYTES;
    #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      u32 af[MT][4], bfr[NT][2];
      #pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int r = wm * MT * 16 + i * 16 + (lane & 15);
        const u32 addr = st + swz(r, ks * 2 + (lane >> 4));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3]) : "r"(addr));
      }
      #pragma unroll
      for (int j = 0; j < NT / 2; ++j) {
        const int r = wn * NT * 8 + j * 16 + (lane & 7) + ((lane >> 4) << 3);
        const u32 addr = st + A_STAGE_BYTES + swz(r, ks * 2 + ((lane >> 3) & 1));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(bfr[2 * j][0]), "=r"(bfr[2 * j][1]), "=r"(bfr[2 * j + 1][0]), "=r"(bfr[2 * j + 1][1])
                     : "r"(addr));
      }
      #pragma unroll
      for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j) mma16816(acc[i][j], af[i], bfr[j]);
    }
#if MODE == 5
    {
      const int j0 = (kt * P + nk - 1) / nk, j1 = ((kt + 1) * P + nk - 1) / nk;
      for (int j = j0; j < j1; ++j)
        sb += sum4(lds128(st + A_STAGE_BYTES + B_STAGE_BYTES + (gw * SPK + (j - j0)) * 512 + lane * 16));
    }
#endif
  }
  cp_wait<0>();
  bar_sync(bar, G_WARPS * 32);        // every warp is done with the smem before it is refilled
  // hash of every C element with its global coordinates (independent of the tiling)
  const int g8 = lane >> 2, t4 = lane & 3;
  const u32 ncols = (u32)g.TNx * BN;
  u64 h = 0;
  #pragma unroll
  for (int i = 0; i < MT; ++i)
    #pragma unroll
    for (int j = 0; j < NT; ++j)
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        const u32 row = tm * BM + wm * MT * 16 + i * 16 + g8 + ((k >> 1) << 3);
        const u32 col = tn * BN + wn * NT * 8 + j * 8 + 2 * t4 + (k & 1);
        h += hash32(__float_as_uint(acc[i][j][k]) ^ hash32(row * ncols + col));
#if CDBG
        ((float*)g.cdbg)[(size_t)row * ncols + col] = acc[i][j][k];   // test only: C to memory
#endif
      }
  sa += h;
}
#endif

// ------------------------------------------------------------------ the kernel
#if MAXREG > 0
#define KERNEL_BOUNDS __maxnreg__(MAXREG)
#else
#define KERNEL_BOUNDS __launch_bounds__(NTHREADS, MINB)
#endif
extern "C" __global__ void KERNEL_BOUNDS
stress(u32* ctl, u64* sums, u32* hits_a, u32* hits_b, int n_a, int n_b, int a_iters,
       const unsigned short* __restrict__ aop, const unsigned short* __restrict__ bop, int K, int TM, int TNx,
       const uint4* __restrict__ sbuf, uint4* __restrict__ dbuf, int chunk_vec,
       int split, const int* __restrict__ remap, int sw_p) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ u64 s_sum[2];
  __shared__ int s_slot[4];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) { s_sum[0] = 0ull; s_sum[1] = 0ull; }
  __syncthreads();
  const u64 pol = make_policy();
  u64 sa = 0, sb = 0;

  // ---- role of this thread's warp
#if MODE == 0 || MODE == 5
  const bool a_first = true;
#elif MODE == 1
  const bool a_first = false;
#elif MODE == 2
  u32 smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const bool a_first = remap[smid] < split;
#elif MODE == 3
  const bool a_first = (int)blockIdx.x < split;
#elif MODE == 4
  const bool a_first = warp < WA;
#endif
  (void)a_first;

#if MODE == 1 && SKIND == 1
  const u32 ring = smem_addr(smem) + warp * SU * 512;
#elif MODE == 1 && SKIND == 2
  const u32 ring = smem_addr(smem) + warp * SU * SBULK;
  const u32 bars = smem_addr(smem) + NWARPS * SU * SBULK + warp * SU * 8;
  u32 phases = 0;
  if (lane == 0) {
    for (int j = 0; j < SU; ++j) mbar_init(bars + j * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncwarp();
#endif

// per-warp reduction of the checksums into the CTA's smem accumulators
#define FINISH_WARP()                                                                \
  { sa = warp_sum64(sa); sb = warp_sum64(sb);                                        \
    if (lane == 0) { atomicAdd(&s_sum[0], sa); atomicAdd(&s_sum[1], sb); }           \
    sa = 0; sb = 0; }
#if SMAXNREG
#define SETMAXNREG_A() asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(RA))
#define SETMAXNREG_B() asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(RB))
#else
#define SETMAXNREG_A()
#define SETMAXNREG_B()
#endif

#define B_LOOP_WARP()                                                                \
  for (;;) {                                                                         \
    const int u = warp_grab(&ctl[1]);                                                \
    if (u >= n_b) break;                                                             \
    sb += stream_unit_ldg(sbuf, dbuf, u, chunk_vec, pol);                            \
    if (lane == 0) atomicAdd(&hits_b[u], 1u);                                        \
  }

#if MODE == 1
  // ---- solo stream
  for (;;) {
    const int u = warp_grab(&ctl[1]);
    if (u >= n_b) break;
#if SKIND == 0
    sb += stream_unit_ldg(sbuf, dbuf, u, chunk_vec, pol);
#elif SKIND == 1
    sb += stream_unit_cpasync(sbuf, u, chunk_vec, ring);
#else
    sb += stream_unit_bulk(sbuf, u, chunk_vec, ring, bars, phases);
#endif
    if (lane == 0) atomicAdd(&hits_b[u], 1u);
  }

#elif AK == 1 && MODE != 5
  // ---- register-only MMA: per-warp units for both roles, takeover in both directions
#define A_LOOP_WARP()                                                                \
  for (;;) {                                                                         \
    const int u = warp_grab(&ctl[0]);                                                \
    if (u >= n_a) break;                                                             \
    sa += reg_unit(u, a_iters);                                                      \
    if (lane == 0) atomicAdd(&hits_a[u], 1u);                                        \
  }
  if (a_first) {
    SETMAXNREG_A();
    A_LOOP_WARP()
#if MODE != 0
    B_LOOP_WARP()
#endif
    FINISH_WARP()
  } else {
    SETMAXNREG_B();
    B_LOOP_WARP()
    A_LOOP_WARP()
    FINISH_WARP()
  }

#elif AK == 1 && MODE == 5
  // ---- same-warp interleaving (reg): unit u = reg A unit u + stream chunk u (P = chunk_vec/32
  //      packets); burst j runs ((j+1)*a_iters)/P - (j*a_iters)/P iterations of NACC mma; a
  //      register ring keeps SD packets in flight (packet j + SD is issued once j is consumed).
  const int P = chunk_vec / 32;
  for (;;) {
    const int u = warp_grab(&ctl[0]);
    if (u >= n_a) break;
    RegOps o; reg_ops(o, u);
    float acc[NACC][4];
    #pragma unroll
    for (int j = 0; j < NACC; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const uint4* p = sbuf + (size_t)u * chunk_vec + lane;
    uint4 ring[SD];
    #pragma unroll
    for (int d = 0; d < SD; ++d) ring[d] = ldg_s(p + d * 32, pol);
    u64 s = 0;
    for (int j0 = 0; j0 < P; j0 += SD) {
      #pragma unroll
      for (int d = 0; d < SD; ++d) {
        const int j = j0 + d;
        reg_burst(acc, o, ((j + 1) * a_iters) / P - (j * a_iters) / P);
        s += sum4(ring[d]);
        if (j + SD < P) ring[d] = ldg_s(p + (j + SD) * 32, pol);
      }
    }
    sa += reg_hash(acc);
    sb += s;
    if (lane == 0) { atomicAdd(&hits_a[u], 1u); atomicAdd(&hits_b[u], 1u); }
  }

#elif AK == 2
  // ---- GEMM-like: A units are tiles computed by a group of G_WARPS warps
  GemmArgs g; g.A = aop; g.B = bop; g.K = K; g.TM = TM; g.TNx = TNx; g.sbuf = sbuf; g.sw_p = sw_p; g.cdbg = dbuf;
#if MODE == 5
  // combined unit: tile u + stream bytes [u*G*P*512, +G*P*512) = cpt whole 32 KB chunks
#define GEMM_SW_HITS(u_, gt_)                                                        \
  { const int cpt = G_WARPS * sw_p * 32 / chunk_vec;                                 \
    if ((gt_) < cpt) atomicAdd(&hits_b[(u_) * cpt + (gt_)], 1u); }
#else
#define GEMM_SW_HITS(u_, gt_)
#endif
#define A_LOOP_GROUP(gt_, bar_, sm_, slot_)                                          \
  for (;;) {                                                                         \
    if ((gt_) == 0) (slot_) = (int)atomicAdd(&ctl[0], 1u);                           \
    bar_sync((bar_), G_WARPS * 32);                                                  \
    const int u = (slot_);                                                           \
    if (u >= n_a) break;                                                             \
    gemm_unit(u, g, (sm_), (gt_), (bar_), sa, sb, pol);                              \
    if ((gt_) == 0) atomicAdd(&hits_a[u], 1u);                                       \
    GEMM_SW_HITS(u, (gt_))                                                           \
  }
#if MODE == 0 || MODE == 5
  A_LOOP_GROUP((int)threadIdx.x, 1, smem, s_slot[0])
#elif MODE == 2 || MODE == 3
  if (a_first) {
    A_LOOP_GROUP((int)threadIdx.x, 1, smem, s_slot[0])
    B_LOOP_WARP()
  } else {
    B_LOOP_WARP()
    __syncthreads();
    A_LOOP_GROUP((int)threadIdx.x, 1, smem, s_slot[0])
  }
#elif MODE == 4
  // each role's whole code (incl. its checksum reduction) is dominated by its setmaxnreg, so
  // ptxas allocates the A path with RA and the B path with RB registers
  if (a_first) {
    SETMAXNREG_A();
    // NGA groups of G_WARPS warps, each with its own smem pipeline, named barrier and slot
    const int grp = warp / G_WARPS;
    A_LOOP_GROUP((int)threadIdx.x - grp * G_WARPS * 32, 1 + grp, smem + grp * (STAGES * STAGE_BYTES),
                 s_slot[grp])
    B_LOOP_WARP()
    FINISH_WARP()
  } else {
    SETMAXNREG_B();
    B_LOOP_WARP()
    FINISH_WARP()
  }
#endif
#if MODE != 4
  FINISH_WARP()
#endif
#endif
#if MODE == 1 || (AK == 1 && MODE == 5)
  FINISH_WARP()
#endif

  // ---- checksums (one global atomic per CTA) and self-reset of the tickets
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&sums[0], s_sum[0]);
    atomicAdd(&sums[1], s_sum[1]);
    __threadfence();
    const u32 prev = atomicAdd(&ctl[2], 1u);
    if (prev == gridDim.x - 1) {
      atomicExch(&ctl[0], 0u); atomicExch(&ctl[1], 0u); atomicExch(&ctl[2], 0u);
      __threadfence();
    }
  }
}
"""

SIG = "ppppiiippiiippiipi"   # ctl sums hits_a hits_b | n_a n_b a_iters | aop bop | K TM TNx | sbuf dbuf | chunk_vec split | remap | sw_p


@dataclass(frozen=True)
class KCfg:
    """Compile-time configuration of the stress kernel (one NVRTC compile per distinct KCfg)."""
    ak: str = "reg"               # "none" | "reg" | "gemm"
    mode: str = "solo_a"          # see MODES
    warps: int = 8                # warps per CTA
    minb: int = 1                 # __launch_bounds__ min blocks per SM (caps the registers)
    nacc: int = 8                 # reg: independent accumulator chains per warp
    nsets: int = 2                # reg: operand sets alternated between consecutive mma
    wm: int = 2                   # gemm: warps along M in the group
    wn: int = 4                   # gemm: warps along N
    mt: int = 4                   # gemm: m16 tiles per warp
    nt: int = 4                   # gemm: n8 tiles per warp (even)
    stages: int = 4               # gemm: cp.async pipeline depth
    su: int = 4                   # stream: loads per lane per batch (ldg) / ring slots (cp.async, bulk)
    skind: int = 0                # stream (solo_b only): 0 ldg, 1 cp.async, 2 bulk
    sbulk: int = 4096             # bulk request bytes
    shint: int = 1                # stream: 1 = L2 evict-first policy
    srw: int = 0                  # stream: 1 = also write the data (read+write)
    wa: int = 0                   # ws: A warps (gemm: the group, = wm*wn)
    sd: int = 4                   # sw reg: register ring depth (packets in flight per warp)
    spk: int = 1                  # sw gemm: max stream packets per warp per k-step
    nga: int = 1                  # ws gemm: A groups per CTA (wa = nga * wm * wn)
    smaxnreg: int = 0             # ws: setmaxnreg rebalancing (sm_120a)
    ra: int = 0                   # ws + smaxnreg: registers of the A warps
    rb: int = 0                   # ws + smaxnreg: registers of the B warps
    maxreg: int = 0               # __maxnreg__ cap (0: __launch_bounds__(threads, minb) instead)
    cdbg: int = 0                 # tests: gemm tiles also store C (fp32) into the dbuf argument

    @property
    def nthreads(self) -> int:
        return 32 * self.warps

    @property
    def g_warps(self) -> int:
        return self.wm * self.wn

    @property
    def bm(self) -> int:
        return self.wm * self.mt * 16

    @property
    def bn(self) -> int:
        return self.wn * self.nt * 8

    def smem_bytes(self) -> int:
        """Dynamic shared memory per CTA."""
        if self.mode == "solo_b":
            if self.skind == 1:
                return self.warps * self.su * PACKET
            if self.skind == 2:
                return self.warps * self.su * (self.sbulk + 8)
            return 0
        if self.ak != "gemm":
            return 0
        stage = (self.bm + self.bn) * 32 * 2
        if self.mode == "sw":
            stage += self.g_warps * self.spk * PACKET
        return self.stages * stage * (self.nga if self.mode == "ws" else 1)

    def validate(self) -> None:
        if self.cdbg and (self.ak != "gemm" or self.mode != "solo_a"):
            raise ValueError("cdbg: gemm solo_a only")
        if self.smem_bytes() > 99 * 1024:
            raise ValueError(f"{self.smem_bytes()} B of shared memory per CTA (> 99 KB)")
        if self.maxreg and self.minb != 1:
            raise ValueError("maxreg (__maxnreg__) and minb (__launch_bounds__) are exclusive")
        if self.mode not in MODES or self.ak not in AKINDS:
            raise ValueError(f"bad mode/ak {self}")
        if self.mode == "solo_b" and self.ak != "none":
            raise ValueError("solo_b has no A work (ak='none')")
        if self.mode != "solo_b" and self.ak == "none":
            raise ValueError("A work needs ak='reg'|'gemm'")
        if self.skind and self.mode != "solo_b":
            raise ValueError("cp.async / bulk streams only in solo_b")
        if self.ak == "gemm" and self.mode in ("solo_a", "sm", "cta", "sw") and self.g_warps != self.warps:
            raise ValueError("gemm: the group must be the whole CTA in this mode")
        if self.mode == "ws":
            if not 0 < self.wa < self.warps:
                raise ValueError("ws: need 0 < wa < warps")
            if self.ak == "gemm" and self.g_warps * self.nga != self.wa:
                raise ValueError("ws gemm: wa must equal nga*wm*wn")
            if not 1 <= self.nga <= 4:
                raise ValueError("ws gemm: 1..4 A groups")
        elif self.nga != 1:
            raise ValueError("nga > 1 only in ws mode")
        if self.smaxnreg:
            if self.mode != "ws":
                raise ValueError("setmaxnreg only in ws mode")
            if self.wa % 4 or self.warps % 4:
                raise ValueError("setmaxnreg works on whole warpgroups: wa and warps must be multiples of 4")
            if self.ra % 8 or self.rb % 8 or not (24 <= self.rb <= self.ra <= 256):
                raise ValueError("setmaxnreg counts must be multiples of 8 in [24, 256]")
        if self.nt % 2:
            raise ValueError("nt must be even (ldmatrix.x4 loads two n8 tiles)")

    def defines(self) -> str:
        d = {"AK": AKINDS[self.ak], "MODE": MODES[self.mode], "NTHREADS": self.nthreads, "MINB": self.minb,
             "NACC": self.nacc, "NSETS": self.nsets, "WM": self.wm, "WN": self.wn, "MT": self.mt,
             "NT": self.nt, "STAGES": self.stages, "SU": self.su, "SKIND": self.skind, "SBULK": self.sbulk,
             "SHINT": self.shint, "SRW": self.srw, "WA": self.wa, "SD": self.sd, "SPK": self.spk,
             "SMAXNREG": self.smaxnreg, "RA": self.ra, "RB": self.rb, "MAXREG": self.maxreg,
             "CDBG": self.cdbg}
        return "".join(f"#define {k} {v}\n" for k, v in d.items())

    def short(self) -> dict:
        """The fields that differ from the defaults."""
        base = KCfg()
        return {k: v for k, v in asdict(self).items() if getattr(base, k) != v}


# ---------------------------------------------------------------------- compile (disk-cached)
_KERNELS: dict = {}
_DISK = os.environ.get("COBENCH_CUBIN_CACHE", os.path.expanduser("~/.cache/cobench_stress"))


def _options(cfg: KCfg) -> list[str]:
    return ["-default-device"]


def kernel(cfg: KCfg, device=None) -> CudaKernel:
    """The compiled stress kernel for ``cfg`` (in-process cache + cubin cache on disk)."""
    cfg.validate()
    dev = device_index(device)
    key = (cfg, dev)
    if key in _KERNELS:
        return _KERNELS[key]
    src = cfg.defines() + SRC
    arch = cudrv.arch_string(dev) + ("a" if cfg.smaxnreg else "")
    opts = [f"--gpu-architecture={arch}", "-std=c++17", *_options(cfg)]
    h = hashlib.sha1((src + "\0" + "\0".join(opts)).encode()).hexdigest()   # = compile_cubin's key
    path = os.path.join(_DISK, h + ".cubin")
    if h not in cudrv._CUBIN_CACHE and os.path.exists(path):
        with open(path, "rb") as f:
            cudrv._CUBIN_CACHE[h] = f.read()
    k = CudaKernel(src, "stress", SIG, options=_options(cfg), device=dev, arch=arch)
    if not os.path.exists(path):
        try:
            os.makedirs(_DISK, exist_ok=True)
            with open(path + ".tmp", "wb") as f:
                f.write(k.cubin)
            os.replace(path + ".tmp", path)
        except OSError:
            pass
    smem = cfg.smem_bytes()
    if smem > 0:
        k.set_max_dynamic_smem(smem)     # opt-in (static + dynamic > 48 KB needs it too)
    k.set_carveout(100)
    _KERNELS[key] = k
    return k


def ptxas_info(cfg: KCfg, device=None) -> dict:
    """ptxas -v of the kernel for ``cfg`` (compiled again, not cached): launch registers, spill
    stores/loads (bytes), stack frame; ``error`` holds the NVRTC log if compilation fails (e.g.
    setmaxnreg for a non-``a`` target)."""
    import re
    from cuda.bindings import nvrtc
    from .cudrv import check
    cfg.validate()
    dev = device_index(device)
    src = cfg.defines() + SRC
    arch = cudrv.arch_string(dev) + ("a" if cfg.smaxnreg else "")
    return _ptxas_log(src, arch)


def _ptxas_log(src: str, arch: str) -> dict:
    import re
    from cuda.bindings import nvrtc
    from .cudrv import check
    opts = [f"--gpu-architecture={arch}", "-std=c++17", "-default-device", "--ptxas-options=-v"]
    prog = check(nvrtc.nvrtcCreateProgram(src.encode(), b"stress.cu", 0, [], []))
    try:
        ret = nvrtc.nvrtcCompileProgram(prog, len(opts), [o.encode() for o in opts])
        n = check(nvrtc.nvrtcGetProgramLogSize(prog))
        buf = b" " * n
        check(nvrtc.nvrtcGetProgramLog(prog, buf))
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    log = buf.decode(errors="replace")
    if ret[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return {"arch": arch, "error": log.strip()[-600:]}
    out = {"arch": arch}
    m = re.search(r"Used (\d+) registers", log)
    out["regs"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
    if m:
        out["stack"], out["spill_stores"], out["spill_loads"] = (int(x) for x in m.groups())
    return out


def kernel_info(cfg: KCfg, device=None) -> dict:
    k = kernel(cfg, device)
    smem = cfg.smem_bytes()
    return {"regs": k.num_regs, "smem": smem, "static_smem": k.static_smem,
            "ctas_per_sm": k.occupancy(cfg.nthreads, smem), "threads": cfg.nthreads}


# ---------------------------------------------------------------------- workload
@dataclass
class Workload:
    """A fixed amount of A work (MMA) and B work (stream bytes) plus the buffers.

    reg family : n_a units of a_iters x nacc mma per warp (default: one per stream chunk).
    gemm family: n_tiles 128x128xK tiles (the unit count scales with the kernel's tile size).
    stream     : stream_bytes read (and also written if rw) in chunk_bytes per warp unit.
    """
    family: str                       # "reg" | "gemm"
    stream_bytes: int = 512 << 20
    chunk_bytes: int = 32 << 10
    rw: bool = False
    a_iters: int = 256                # reg: iterations of nacc mma per unit
    nacc: int = 8
    n_a_reg: int | None = None        # reg: number of A units (default: n_b)
    n_tiles: int = 2048               # gemm: 128x128 tiles
    K: int = 2048
    TM: int = 8
    TN: int = 8                       # B operand rows = TN * 128
    seed: int = 0
    device: int | None = None
    sbuf: torch.Tensor | None = field(default=None, repr=False)
    dbuf: torch.Tensor | None = field(default=None, repr=False)
    aop: torch.Tensor | None = field(default=None, repr=False)
    bop: torch.Tensor | None = field(default=None, repr=False)
    sum_b_ref: int | None = None
    sum_a_ref: int | None = None

    def __post_init__(self):
        if self.family not in ("reg", "gemm"):
            raise ValueError(self.family)
        if self.stream_bytes % self.chunk_bytes or self.chunk_bytes % 16384:
            raise ValueError("stream_bytes must be a multiple of chunk_bytes, chunk a multiple of 16 KB")
        self.device = device_index(self.device)
        dev = f"cuda:{self.device}"
        g = torch.Generator(device=dev)
        g.manual_seed(self.seed)
        self.sbuf = torch.randint(-2 ** 31, 2 ** 31 - 1, (self.stream_bytes // 4,), dtype=torch.int32,
                                  device=dev, generator=g)
        self.dbuf = torch.empty_like(self.sbuf) if self.rw else torch.empty(16, dtype=torch.int32, device=dev)
        tot = 0                                  # sum of the unsigned 32-bit words (mod 2^64)
        for part in self.sbuf.split(1 << 26):
            tot += int((part.to(torch.int64) & 0xFFFFFFFF).sum().item())
        self.sum_b_ref = tot % (1 << 64)
        if self.family == "gemm":
            self.aop = torch.randn(self.TM * 128, self.K, device=dev, dtype=torch.bfloat16, generator=g)
            self.bop = torch.randn(self.TN * 128, self.K, device=dev, dtype=torch.bfloat16, generator=g)
        else:
            self.aop = self.bop = torch.empty(8, device=dev, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)

    @property
    def n_b(self) -> int:
        return self.stream_bytes // self.chunk_bytes

    @property
    def chunk_vec(self) -> int:
        return self.chunk_bytes // 16

    def n_a_units(self, cfg: KCfg | None = None) -> int:
        if self.family == "reg":
            return self.n_a_reg if self.n_a_reg is not None else self.n_b
        bm, bn = (cfg.bm, cfg.bn) if cfg is not None else (128, 128)
        if bm != 128 or 128 % bn:
            raise ValueError("gemm tiles must be 128 x (128 / 2^k)")
        return self.n_tiles * (128 // bn)

    def tnx(self, cfg: KCfg) -> int:
        return self.TN * 128 // cfg.bn

    @property
    def flops_a(self) -> float:
        if self.family == "reg":
            return float(self.n_a_units()) * self.a_iters * self.nacc * MMA_FLOP
        return 2.0 * self.n_tiles * 128 * 128 * self.K

    @property
    def bytes_b(self) -> int:
        return self.stream_bytes * (2 if self.rw else 1)

    def sw_packets(self, cfg: KCfg) -> int:
        """sw gemm: stream packets per warp per tile."""
        n = self.n_a_units(cfg)
        per_tile = self.stream_bytes // n
        if self.stream_bytes % n or per_tile % (cfg.g_warps * PACKET) or per_tile % self.chunk_bytes:
            raise ValueError(f"sw gemm: {self.stream_bytes} B over {n} tiles is not a whole number of "
                             "packets per warp / chunks per tile")
        return per_tile // (cfg.g_warps * PACKET)

    def describe(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not isinstance(v, torch.Tensor)}
        d.update({"n_b": self.n_b, "n_a_units_default": self.n_a_units(), "flops_a": self.flops_a,
                  "bytes_b": self.bytes_b})
        return d

    def free(self) -> None:
        self.sbuf = self.dbuf = self.aop = self.bop = None


# ---------------------------------------------------------------------- one launch instance
class Launch:
    """One kernel instance (own tickets, hit counters and checksums) with a fixed grid and args.
    ``launch(stream=None)`` enqueues it on ``stream`` (default: torch's current stream)."""

    def __init__(self, wl: Workload, cfg: KCfg, grid: int, *, split: int = 0,
                 remap: torch.Tensor | None = None, a_iters: int | None = None, name: str = ""):
        self.wl, self.cfg, self.grid, self.name = wl, cfg, int(grid), name
        self.k = kernel(cfg, wl.device)
        dev = f"cuda:{wl.device}"
        self.smem = cfg.smem_bytes()
        self.ctl = torch.zeros(4, dtype=torch.int32, device=dev)
        self.sums = torch.zeros(2, dtype=torch.int64, device=dev)
        self.does_a = cfg.mode != "solo_b"
        self.does_b = cfg.mode != "solo_a"
        self.n_a = wl.n_a_units(cfg) if self.does_a else 0
        self.n_b = wl.n_b if self.does_b else 0
        self.hits_a = torch.zeros(max(1, self.n_a), dtype=torch.int32, device=dev)
        self.hits_b = torch.zeros(max(1, self.n_b), dtype=torch.int32, device=dev)
        self.remap = remap if remap is not None else torch.zeros(1024, dtype=torch.int32, device=dev)
        self.split = int(split)
        self.a_iters = int(a_iters if a_iters is not None else wl.a_iters)
        self.sw_p = 0
        if self.does_b and cfg.skind == 0 and wl.chunk_vec % (64 * cfg.su):
            raise ValueError(f"chunk ({wl.chunk_bytes} B) must be a multiple of 2*su packets")
        if cfg.skind == 2 and wl.chunk_bytes % cfg.sbulk:
            raise ValueError("chunk must be a multiple of sbulk")
        if cfg.mode == "sw":
            if wl.family == "gemm":
                self.sw_p = wl.sw_packets(cfg)
                nk = wl.K // 32
                if -(-self.sw_p // nk) > cfg.spk:
                    raise ValueError(f"sw gemm: {self.sw_p} packets per warp over {nk} k-steps needs "
                                     f"spk >= {-(-self.sw_p // nk)}")
            else:
                if self.n_a != self.n_b:
                    raise ValueError("sw reg: needs one A unit per stream chunk")
                if (wl.chunk_vec // 32) % cfg.sd:
                    raise ValueError("sw reg: packets per chunk must be a multiple of sd")
        if wl.family == "gemm" and self.does_a and wl.K % 32:
            raise ValueError("K must be a multiple of 32")
        self.occupancy = self.k.occupancy(cfg.nthreads, self.smem)
        if self.occupancy < 1:
            raise ValueError(f"{name or cfg.short()}: does not fit on an SM (regs {self.k.num_regs}, "
                             f"smem {self.smem})")
        tnx = wl.tnx(cfg) if wl.family == "gemm" else 1
        self.args = (self.ctl, self.sums, self.hits_a, self.hits_b, self.n_a, self.n_b, self.a_iters,
                     wl.aop, wl.bop, wl.K, wl.TM, tnx, wl.sbuf, wl.dbuf, wl.chunk_vec, self.split,
                     self.remap, self.sw_p)

    def launch(self, stream=None) -> None:
        self.k(self.grid, self.cfg.nthreads, *self.args, smem=self.smem, stream=stream)

    def __call__(self, *_):
        self.launch()

    def reset(self) -> None:
        self.ctl.zero_()
        self.sums.zero_()
        self.hits_a.zero_()
        self.hits_b.zero_()

    def info(self) -> dict:
        return {"name": self.name, "cfg": self.cfg.short(), "grid": self.grid, "block": self.cfg.nthreads,
                "smem": self.smem, "regs": self.k.num_regs, "occupancy": self.occupancy, "split": self.split,
                "n_a": self.n_a, "n_b": self.n_b, "a_iters": self.a_iters, "sw_p": self.sw_p}


def _u64(x: int) -> int:
    return int(x) % (1 << 64)


class Variant:
    """A co-location variant: ``fn(i)`` enqueues one iteration on the current stream (forking /
    joining internally), plus the work-completion check. ``fn`` is what bench_steady runs (a
    ``Par`` for green variants, so per-op completion times are reported)."""

    def __init__(self, name: str, wl: Workload, launches: list, fn, desc: dict):
        self.name, self.wl, self.launches, self.fn, self.desc = name, wl, launches, fn, desc

    def __call__(self, i=0):
        return self.fn(i)

    def verify(self, k: int = 2, *, sum_a_ref: int | None = None, check_a: bool = True) -> dict:
        """Run k iterations from zeroed counters. Every A and B unit must be done exactly k times
        (summed over the variant's kernel instances) and the checksums must be k x the reference
        (A: ``sum_a_ref`` or ``wl.sum_a_ref``; B: computed by torch from the buffer)."""
        for L in self.launches:
            L.reset()
        torch.cuda.synchronize(self.wl.device)
        for i in range(k):
            self.fn(i)
        torch.cuda.synchronize(self.wl.device)
        out = {"k": k, "ok": True, "errors": []}
        for kind in ("a", "b"):
            arrs = [getattr(L, f"hits_{kind}")[: getattr(L, f"n_{kind}")] for L in self.launches
                    if getattr(L, f"n_{kind}")]
            if not arrs:
                continue
            if len({a.numel() for a in arrs}) != 1:
                raise ValueError("instances disagree on the unit decomposition")
            tot = torch.stack(arrs).sum(0)
            bad = int((tot != k).sum().item())
            out[f"units_{kind}"] = int(tot.numel())
            out[f"bad_units_{kind}"] = bad
            if bad:
                out["ok"] = False
                out["errors"].append(f"{bad} {kind} units not done exactly {k}x "
                                     f"(min {int(tot.min())}, max {int(tot.max())})")
        sa = _u64(sum(int(L.sums[0].item()) for L in self.launches))
        sb = _u64(sum(int(L.sums[1].item()) for L in self.launches))
        out["sum_a"], out["sum_b"] = sa, sb
        if any(L.does_b for L in self.launches) and sb != _u64(k * self.wl.sum_b_ref):
            out["ok"] = False
            out["errors"].append(f"stream checksum {sb} != {k} x reference")
        ref_a = sum_a_ref if sum_a_ref is not None else self.wl.sum_a_ref
        if check_a and ref_a is not None and any(L.does_a for L in self.launches) and sa != _u64(k * ref_a):
            out["ok"] = False
            out["errors"].append(f"MMA checksum {sa} != {k} x reference {ref_a}")
        for L in self.launches:
            L.reset()
        torch.cuda.synchronize(self.wl.device)
        return out


def nsm(device=None) -> int:
    return torch.cuda.get_device_properties(device_index(device)).multi_processor_count


def reference_sum_a(wl: Workload, cfg: KCfg) -> int:
    """MMA checksum of one full A pass with the solo kernel (stored in wl.sum_a_ref)."""
    L = solo_a(wl, cfg, name="ref")
    L.reset()
    L.launch()
    torch.cuda.synchronize(wl.device)
    if int((L.hits_a[: L.n_a] != 1).sum().item()):
        raise RuntimeError("reference A pass did not do every unit exactly once")
    wl.sum_a_ref = _u64(int(L.sums[0].item()))
    return wl.sum_a_ref


# ---------------------------------------------------------------------- variant builders
def solo_a(wl: Workload, cfg: KCfg, *, n_sms: int | None = None, ctas_per_sm: int = 1, name="solo_a") -> Launch:
    cfg = replace(cfg, mode="solo_a")
    return Launch(wl, cfg, (n_sms or nsm(wl.device)) * ctas_per_sm, name=name)


def solo_b(wl: Workload, cfg: KCfg, *, n_sms: int | None = None, ctas_per_sm: int = 2, name="solo_b") -> Launch:
    cfg = replace(cfg, mode="solo_b", ak="none")
    return Launch(wl, cfg, (n_sms or nsm(wl.device)) * ctas_per_sm, name=name)


def v_single(name: str, L: Launch, desc: dict) -> Variant:
    return Variant(name, L.wl, [L], lambda i=0: L.launch(), desc)


def v_serial(wl: Workload, a: Launch, b: Launch, name="serial") -> Variant:
    def fn(i=0):
        a.launch()
        b.launch()
    return Variant(name, wl, [a, b], fn, {"kind": "serial", "a": a.info(), "b": b.info()})


def v_green(wl: Workload, part, cfg_a: KCfg, cfg_b: KCfg, *, ctas_a: int = 1, ctas_b: int = 2,
            name=None) -> Variant:
    """A on the partition's group (part.n_sms SMs), B on its complement (part.n_rest SMs)."""
    from .variants import Par
    a = solo_a(wl, cfg_a, n_sms=part.n_sms, ctas_per_sm=ctas_a, name="green_a")
    b = solo_b(wl, cfg_b, n_sms=part.n_rest, ctas_per_sm=ctas_b, name="green_b")
    par = Par(("a", part.stream, lambda: a.launch()), ("b", part.rest_stream, lambda: b.launch()))
    return Variant(name or f"green_{part.n_sms}", wl, [a, b], par,
                   {"kind": "green", "sms_a": part.n_sms, "sms_b": part.n_rest, "a": a.info(), "b": b.info()})


def v_sm(wl: Workload, cfg: KCfg, n_a_sms: int, remap: torch.Tensor, *, ctas_per_sm: int = 1,
         n_sms: int | None = None, name=None) -> Variant:
    cfg = replace(cfg, mode="sm")
    L = Launch(wl, cfg, (n_sms or nsm(wl.device)) * ctas_per_sm, split=n_a_sms, remap=remap)
    return v_single(name or f"sm_{n_a_sms}", L, {"kind": "sm", "sms_a": n_a_sms, "launch": L.info()})


def v_cta(wl: Workload, cfg: KCfg, k_a: int, k_b: int, *, n_sms: int | None = None, name=None) -> Variant:
    cfg = replace(cfg, mode="cta")
    n = n_sms or nsm(wl.device)
    L = Launch(wl, cfg, n * (k_a + k_b), split=n * k_a)
    if L.occupancy < k_a + k_b:
        raise ValueError(f"cta {k_a}+{k_b}: only {L.occupancy} CTAs fit per SM "
                         f"(regs {L.k.num_regs}, smem {L.smem})")
    return v_single(name or f"cta_{k_a}_{k_b}", L, {"kind": "cta", "k_a": k_a, "k_b": k_b, "launch": L.info()})


def v_ws(wl: Workload, cfg: KCfg, w_a: int, w_b: int, *, n_sms: int | None = None, name=None) -> Variant:
    """One CTA per SM: w_a MMA warps (gemm: cfg.nga groups of wm*wn warps) + w_b stream warps."""
    cfg = replace(cfg, mode="ws", wa=w_a, warps=w_a + w_b)
    L = Launch(wl, cfg, n_sms or nsm(wl.device))
    tag = f"_r{cfg.ra}_{cfg.rb}" if cfg.smaxnreg else ""
    return v_single(name or f"ws_{w_a}_{w_b}{tag}", L, {"kind": "ws", "w_a": w_a, "w_b": w_b,
                                                        "setmaxnreg": bool(cfg.smaxnreg), "launch": L.info()})


def v_sw(wl: Workload, cfg: KCfg, *, ctas_per_sm: int = 1, n_sms: int | None = None, name=None) -> Variant:
    cfg = replace(cfg, mode="sw")
    L = Launch(wl, cfg, (n_sms or nsm(wl.device)) * ctas_per_sm)
    return v_single(name or f"sw_{cfg.warps}w", L, {"kind": "sw", "ctas_per_sm": ctas_per_sm, "launch": L.info()})


# ---------------------------------------------------------------------- nested green partitions
def split_ranges(sizes, *, device=None) -> list:
    """Green contexts over consecutive SM ranges [0, s0), [s0, s0 + s1), ... (even sizes).

    The driver refuses to split a resource that came from a previous split
    (CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION, driver 580), so the device is split once into
    2-SM groups (IGNORE_SM_COSCHEDULING: contiguous, one TPC each) and consecutive groups are
    recombined into one descriptor per range."""
    from cuda.bindings import driver as cu
    from .cudrv import check
    from .green import _FLAG_IGNORE_COSCHED, GreenContext, device_sm_resource
    res = device_sm_resource(device)
    groups, nb, _ = check(cu.cuDevSmResourceSplitByCount(int(res.sm.smCount) // 2, res, _FLAG_IGNORE_COSCHED, 2))
    out, i = [], 0
    for s in sizes:
        if s <= 0 or s % 2:
            raise ValueError(f"range sizes must be even and positive: {sizes}")
        k = s // 2
        if i + k > nb:
            raise ValueError(f"{sum(sizes)} SMs requested, the device has {2 * nb}")
        out.append(GreenContext.create(list(groups[i:i + k]), device))
        i += k
    return out


def split_nested(n_outer: int, n_a: int, *, device=None):
    """A split of the SM range [0, n_outer) (the sub-device of ``split_sms(n_outer,
    ignore_coscheduling=True)``) into [0, n_a) and [n_a, n_outer): an SmPartition whose
    ``.stream`` has the n_a SMs and ``.rest_stream`` the other n_outer - n_a. Close it."""
    from .green import SmPartition
    ga, gb = split_ranges([n_a, n_outer - n_a], device=device)
    return SmPartition(requested=n_a, ignore_coscheduling=True, part=ga, rest=gb)
