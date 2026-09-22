# P1-M3a CoKernel builder v0: correctness, smem aliasing, overhead (2026-09-22)

Code: `cotile/cokernel.py` (design and API in `cotile/README.md`, section "CoKernel builder").
Machine: RTX PRO 6000 Blackwell (sm_120, 188 SMs), CUDA 12.9, clocks not locked.

Files:
- `cokernel_tests.csv`, `cokernel_summary.json`: correctness matrix (`python -m cotile.tests.test_cokernel --out ...`).
- `cokernel_signatures.csv`: resource signature of every CoKernel from the compiled cubin, plus generated-code facts.
- `overhead_probe.csv/json`: K5 probe (`python -m cotile.tests.probe_cokernel_overhead --out ...`).

Shapes: GEMM 2048x4096x4096 (bf16, NT), GQA decode batch 16 x S 8192 (Hq 32, Hkv 8, D 128), RMSNorm 16384x4096.

## K4 correctness matrix

Seven config pairs:

| pair | role A (GEMM cfg) | role B | what it exercises |
|---|---|---|---|
| gd_e0 | 128x128x64 s2 t128 | decode n64 h4 sp1 t128 | E0 GEMM, equal threads |
| gd_split | 128x128x64 s2 t128 split-K 2 | decode n64 h4 sp4 t128 | both roles do in-kernel last-arriver combines |
| gd_thr | 128x128x64 s2 t256 | decode n64 h4 sp2 t128 | decode on 128 of 256 threads (bar.sync 3,128 + AllReduce ids 1/2) |
| gd_small | 64x64x64 s2 t128 | decode n32 h4 sp2 t64 | 2 CTAs/SM -> CTA binding; decode on 64 of 128 threads |
| gr_e0 | 128x128x64 s2 t128 | RMSNorm r1 t256 v8 | GEMM on 128 of 256 threads, 16384 tiny tiles |
| gr_split | 128x128x64 s2 t256 split-K 2 | RMSNorm r4 t128 v8 | RMSNorm on 128 of 256 threads |
| gr_small | 64x64x32 s2 t128 | RMSNorm r2 t128 v8 | 4 CTAs/SM -> CTA binding ratios 1:1, 1:3, 3:1 |

Each pair is built with {SM binding} (and {CTA binding} when 2 or more CTAs/SM fit) x {static, dynamic} x
{takeover off, on}, both as a debug build (per-tile counters) and as a timing build, plus
one `smem="sum"` baseline. Dynamic uses chunk 1 for GEMM and decode and chunk 4 for RMSNorm (a per-role
chunk). Runtime knobs, with no recompilation: SM splits 94 (contiguous), 140 (interleaved), 40, and with
takeover also 188 (all A) and 0 (all B); CTA ratios as in the table, and with takeover also 1:0 and 0:1.
Every (kernel, knob) setting is launched 3 times without host re-zeroing. Outputs and split workspaces
are poisoned with NaN before every launch.

Checks: both outputs are within tolerance of the fp32 reference **and bitwise equal to the same op/cfg's
solo persistent build**; the debug per-tile exec counter is 1 everywhere; with SM binding and no
takeover every tile ran on an SM of its own role; done_A/done_B equal the tile counts; the actual
per-role CTA count equals the static schedule's expectation; all counters are zero after every launch
(only the epoch advances); the epoch increments by 1 per launch.

**Result: 184 / 184 launchable (kernel, knob) settings pass every check, 3 launches each (552 launches).**
There are 79 CoKernels. 72 are orchestration variants: 7 pairs x {SM binding; CTA binding also for gd_small/gr_small} x
{static, dynamic} x {takeover off, on} x {debug, timing build}. The other 7 are `smem="sum"` baselines: 4 run correctly and
the 3 GEMM x decode ones cannot launch (173–175 KB smem/CTA). With the 13 solo persistent builds, all 92 kernels compile
(32 workers, ~58 s cold).

- Bitwise identity with the solo persistent builds holds for every output. That includes E1 configs (split-K,
  split-KV, RMSNorm) and roles running on a subset of the CTA's threads. The ptxas f32x2->FFMA contraction hazard
  from the op-library notes did not show up: the decode body already avoids the `x*a - y*b` pattern, and the
  compiled role bodies are the same arithmetic as the solo kernels. Note that commit 7e3bbf37's
  `tl.enable_fp32x2_reduction` only switches the packed accumulator of `T.reduce_sum/abssum`. It does **not**
  disable `tl::mul2/sub2/add2` emission for element-wise code (`CanEmitPackedX2Math` depends only on arch >= sm_100),
  so it is not a switch for that hazard.
- Takeover is exercised. For example, gd_e0 static+takeover with 94 SMs per role: `steal_A` = 230 steal tickets on GEMM
  tiles by the decode CTAs. With all SMs assigned to A (or to B), the other role completes entirely through takeover.
- The static schedule's placement assumption (num_ctas/188 co-resident CTAs per SM, breadth-first) held in every
  run: tickets == expected.


## K2 shared memory: per-CTA user smem (bytes, from the cubin)

| pair | solo A (persistent) | solo B (persistent) | CoKernel `smem="sum"` (TileLang default plan) | CoKernel `smem="alias"` |
|---|---|---|---|---|
| gd_e0 | 98304 | 74752 | 173072 (**not launchable**, > 101376) | **73216** |
| gd_split | 98320 | 74768 | 175120 (not launchable) | **73216** |
| gd_thr | 98304 | 74768 | 174096 (not launchable) | **73216** |
| gd_small | 40960 | 39440 | 80912 (1 CTA/SM) | **38912** (2 CTAs/SM) |
| gr_e0 | 98304 | 32 | 98352 | **66560** |
| gr_split | 98320 | 64 | 99344 | **66560** |
| gr_small | 24576 | 32 | 24624 (3 CTAs/SM) | **17408** (4 CTAs/SM) |

`alias` is at most max(solo A, solo B) everywhere. It is lower because inside a lifetime scope each role gets
the grid build's intra-tile reuse (GEMM's C staging buffer overlays its A/B pipeline buffers; the decode's
AllReduce workspaces overlay each other), which the solo persistent build does not get. The views in the
generated code confirm the overlay. For gd_e0 the layout is: slot at 0 (16 B), GEMM A_s/C_s and decode Q_s at 1024, GEMM B_s at
33792, and the decode K_s/V_s/P_s following. The 1 KB head is alignment for the swizzled buffers.

## K5 overhead (cobench flush mode, 50 reps, L2 cold; µs = median event time; kcyc = clock-normalised)

(i) Role B compiled in but assigned no SM (SM binding, all SMs -> A, takeover off), 188 CTAs:

| A (partner) | variant | µs | MHz | kcyc | regs | smem | CTAs/SM |
|---|---|---|---|---|---|---|---|
| GEMM (solo persistent) | static grid-stride | 262.5 | 2702 | 710 | 192 | 98304 | 1 |
| GEMM (decode) | static, no timestamps | 268.2 | 2568 | 688 | 232 | 73216 | 1 |
| GEMM (decode) | static | 268.3 | 2570 | 690 | 236 | 73216 | 1 |
| GEMM (decode) | dynamic c1 | 210.9 | 2499 | 528 | 236 | 73216 | 1 |
| GEMM (GEMM) | static, no timestamps | 267.9 | 2566 | 688 | 211 | 66560 | 1 |
| GEMM (RMSNorm, 256 thr) | static, no timestamps | 268.3 | 2553 | 684 | 214 | 66560 | 1 |
| GEMM (RMSNorm, 256 thr) | dynamic c1 | 210.9 | 2478 | 522 | 216 | 66560 | 1 |
| decode (solo persistent) | 1 tile per CTA | 343.3 | 2860 | 980 | 118 | 74752 | 1 |
| decode (GEMM) | static, no timestamps | 355.6 | 2856 | 1015 | 229 | 73216 | 1 |
| decode (GEMM) | dynamic c1 | 356.1 | 2855 | 1016 | 232 | 73216 | 1 |
| decode (decode) | static, no timestamps | 368.3 (cv 2%) | 2854 | 1051 | 140 | 72720 | 1 |
| RMSNorm (solo persistent) | static grid-stride | 200.7 | 2846 | 571 | 39 | 32 | 6 |
| RMSNorm (GEMM) | static, no timestamps | 266.2 | 2859 | 761 | 214 | 66560 | 1 |
| RMSNorm (GEMM) | dynamic c1 | 194.5 | 2855 | 555 | 218 | 66560 | 1 |
| RMSNorm (RMSNorm) | static, no timestamps | 204.4 | 2859 | 583 | 56 | 48 | 4 |
| RMSNorm (RMSNorm) | dynamic c1 | 194.2 | 2860 | 556 | 63 | 48 | 4 |

(ii) Dispatch cost for tiny tiles: RMSNorm x RMSNorm, one row (16 KB of traffic) per tile, 752 CTAs (4/SM):

| variant | µs | kcyc |
|---|---|---|
| solo persistent (752 CTAs) | 204.7 | 585 |
| static, no timestamps | 196.2 | 559 |
| static + per-tile timestamps | 204.4 | 582 |
| dynamic chunk 1, no timestamps / with | 197.2 / 198.7 | 564 / 566 |
| dynamic chunk 4, no timestamps / with | 198.7 / 200.7 | 566 / 572 |
| dynamic chunk 16, no timestamps / with | 198.2 / 197.3 | 564 / 562 |

Observations:
- **Coupling.** Compiling in a partner raises registers from 192 to 211–236 for GEMM, from 118 to 140–229 for decode and from 39 to
  56–220 for RMSNorm. Relative to the matching A x A CoKernel, the cost of the partner itself is within noise
  (GEMM|decode 268.2 vs GEMM|GEMM 267.9 µs; decode|GEMM 355.6 vs decode|decode 362–368 µs). The dispatch structure
  costs +2% for static GEMM against the solo persistent build (-3% in cycles, because the clock was lower) and +3.6% for decode (+12 µs,
  one tile per CTA). Register coupling costs real occupancy only where the lighter role could otherwise
  run more CTAs per SM (RMSNorm: 6 -> 1 CTAs/SM next to a GEMM). These probes launch 1 CTA/SM, so they do not exercise that.
- **Static vs dynamic.** In the timing (non-debug) builds under flush mode, the static round-robin schedule is
  20–26% slower than the dynamic queue for GEMM (268 vs 211 µs). The op library's own static persistent build shows
  the same (262.5 µs), so this is not caused by the CoKernel dispatcher. Static RMSNorm is also slow (266 µs) when a
  large partner is compiled in, but normal (204 µs) next to a small one. The effect vanishes in the debug builds and
  with a warm L2; see "Follow-up diagnostics" below. The mechanism is **not identified**.
  In a back-to-back (warm-L2) debug run the RMSNorm timeline showed static per-SM finish times spread over 150–186 µs
  (dynamic 175–176 µs); under flush mode the debug static run had uniform per-SM busy time.
- **Dispatch cost.** For 2 µs tiles at 4 CTAs/SM, dynamic chunk 1 costs no more than static (+0.5% ± noise). The per-tile
  atomic latency is hidden by the other CTAs on the SM. Per-tile timestamps (one extra CTA barrier + one
  %globaltimer read per tile) cost 0–4%. Chunk size 1/4/16 makes no measurable difference here.
- **K3 cross-check.** The in-kernel makespan (last CTA exit - first CTA start) is 3–6 µs below the event
  time in every row, which matches the ~3 µs launch floor of flush mode.

## Generated-code facts (K6)

- `__launch_bounds__(threads, min_blocks_per_sm)`: (128,1), (256,1), (128,2) for gd_small, (128,4) for gr_small.
  At (128,4) GEMM 64x64 compiles to 126–128 registers with no local memory.
- Barriers: `__syncthreads` (id 0) everywhere. The decode's cross-warp `T.reduce_*` uses `NamedBarrier<n>` with the
  hard-coded ids 1/2, where n is the role's thread count (128, or 64 for gd_small). A partial-thread role's `__syncthreads` is
  rewritten to `tl::__sync_thread_partial(3, n)` by ThreadPartialSyncRewriter. `EIATTR_NUM_BARRIERS` = 3 for
  128-thread GEMM x decode, 4 when a partial role exists, 1 for gr_small. The barrier limit (24 / n per SM) is not binding.
- One `__syncthreads` per dispatch iteration, plus one when timestamps are on. The tile bodies keep their own
  leading barrier.
- Shared memory: a single `extern __shared__ __align__(1024) buf_dyn_shmem[]`; the broadcast slot sits in the
  dynamic arena (a static `__shared__` array costs 1 KB because of the arena's 1024-byte alignment).
- The GEMM TMA-store epilogue keeps `tma_store_arrive(); tma_store_wait<0,true>()` followed by a barrier,
  so the next role's writes to the aliased bytes cannot race the bulk store.

## Follow-up diagnostics on static vs dynamic (ad-hoc scripts, numbers only)

1. Partner footprint (flush mode; RMSNorm as the only active role, 188 CTAs, no timestamps):

| partner compiled in | regs | smem/CTA | static µs | dynamic µs |
|---|---|---|---|---|
| GEMM 128x128 s2 | 214 | 66560 | 266.2 | 196.6 |
| GEMM 128x128 direct s3 | 220 | 98320 | 266.2 | 196.6 |
| GEMM 64x64x32 s2 | 104 | 17408 | 203.7 | 196.6 |
| RMSNorm | 56 | 48 | 204.4 | 196.8 |

2. Debug builds, which add per-tile thread-0 global atomics and stores plus the timeline, measured in flush mode
   (30 reps): GEMM|decode static 214.6 vs dynamic 213.5 µs; RMSNorm|GEMM static 206.8 vs dynamic 199.8 µs. The
   **non-debug** builds of the same pairs give static 268 vs dynamic 211 µs and 266 vs 197 µs. In a back-to-back
   (L2-warm) run of the debug builds: GEMM static 189 vs dynamic 186.5 µs.
   Timeline (debug, flush): GEMM first tiles take 72–77 µs and later tiles 64–68 µs; the per-SM busy time is uniform.

**Open:** the static round-robin schedule (and the op library's own static persistent build) loses 20–30% against the
dynamic queue under flush mode when the kernel is big (>= 66 KB smem or > 200 regs at 1 CTA/SM). The loss
disappears with a small-footprint partner, in debug builds, and with a warm L2. The mechanism is not identified.
Candidates: the L1/smem carve-out the driver selects for large-smem kernels; the fixed SM<->address mapping of
round-robin; the write-back of the flush buffer's dirty L2 lines. It must be resolved (ncu, or
non-perturbing per-CTA timestamps kept in smem) before static and dynamic orchestration are compared in the
performance study.
