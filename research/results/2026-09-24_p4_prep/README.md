# P4-a: prefill attention op, POD stream fix and solo profiling for P4 (prefill × decode)

**Setup**
- Date: 2026-09-24, 04:35–06:05. RTX PRO 6000 Blackwell (sm_120, 188 SMs, 600 W cap, clocks not lockable), driver 580.173.02, CUDA 12.9, torch 2.8.0+cu128, FlashInfer 0.7.0 (JIT) with the local POD patch.
- TileLang: source tree at `ed3bc558` plus this task's changes. `libtilelang.so` was rebuilt once, for the pipeline fix in §1.3.
- GPU sharing: no foreign process was on the GPU during any of this work. All pmon guard records are clean: 0 contaminated flush points, and every steady run has `guard.clean = true`.
- GPU time: about 65 min of wall time with the GPU in use, of which about 45 min was measurement. The two `p4_solo.py` runs took 17.8 and 21.6 min, including two global warm-ups of about 3 min each. The first run stopped at a reference check (§5) and the second resumed it.

Head configuration: Hq = 32, Hkv = 8, D = 128 (Llama-3-8B), bf16, for both ops.
- FlashInfer's POD requires prefill and decode to share `num_qo_heads`, `num_kv_heads`, `head_dim` and the dtypes. `pod.cuh` asserts the heads and static-asserts the dtypes, and the wrapper `flashinfer_ops.POD` checks all of them.
- The 2026-09-22 POD baseline used 32/8/128. That baseline's prefill-only runs also measured 32/32; the MHA variant is not used here.

| criterion | result |
|---|---|
| P1' prefill op | **done**. `cotile/ops/prefill_attn.py` has 29 configs. All are correct in both builds; grid == persistent bitwise 29/29; 3 numerics groups, bitwise 29/29; smem prediction exact for 58/58 kernels; no spills. CoKernel prefill × decode (4 pairs, SM + CTA binding, static/dynamic, takeover): 95/95 launchable settings pass. Needed a TileLang fix (§1.3). |
| P2' POD fix | **done**. `research/patches/flashinfer_pod_stream_memset.patch`. On the default stream, outputs are bitwise equal to unpatched (8 configs × 3 input sets). Side stream, green context and CUDA graph (24 replays per config): 576/576 calls bitwise equal to the default-stream result; unpatched gets 138/144 wrong. POD now runs in steady mode. |
| P3' solo profiling | **done**. TileLang prefill is **8–17% faster** than FlashInfer prefill in steady mode, so no "> 10% slower" flag. There is a fairness caveat in the other direction (§4). Solo-best and C_lib are listed per shape, and the pairing table covers the 8 pairs (§4). |
| P4' outputs | this directory: JSON, `tables.md`, and scripts `research/bench/scripts/p4_*.py` |

## 1. Prefill attention op (`cotile/ops/prefill_attn.py`)

### 1.1 Design

- **Math.** Causal self-attention of one request of length S, FlashAttention-2 style:
  - online softmax with fp32 state (m, l, O accumulator);
  - P rounded to bf16 before the PV GEMM;
  - bf16 in/out, using FlashInfer's NHD layout: Q/O `[S, Hq, D]`, K/V `[S, Hkv, D]`. The same tensors feed both libraries.
- **Tile** = (query block qb of `block_M` positions, query head h). Other details:
  - The tile streams K/V of head h // G in `block_N` blocks, from 0 to the diagonal: `nkv(qb) = ceil((qb+1)·block_M / block_N)` blocks.
  - Warps are laid out along M (`GemmWarpPolicy.FullRow`), so the row max and row sum are warp-local. There is no cross-warp smem workspace and no named barrier, and P stays in registers as the A operand of the PV GEMM.
  - The causal mask initialises the score tile with 0 or −inf before the QK^T GEMM accumulates into it, which costs the same as a clear.
- **Per-tile work.** Work differs by up to S/block_M × between tiles. `TileSpace` gained an optional `work(t)` field (issued MMA FLOPs of tile t), and `TileSpace.works()` returns the full list. The test checks that the per-tile work sums to the catalog work model and is monotone in the configured order.
- **Tile order.** Default `order="lpt"` (longest first): t → qb = nq−1 − t // Hq, h = t % Hq.
  - The Hq heads of a query block are adjacent, so the 4 query heads of a KV group read the same K/V stream back to back.
  - This order is what the grid build dispatches, what the persistent grid-stride walks, and what a CoKernel dynamic queue hands out.
  - `order="natural"` (shortest first) is a control. At 188 SMs (clean flush) it is 8.8% / 14.8% slower at S=2048 and 1.5% / 4.4% slower at S=8192 (configs m64_n64_s2_t128 / m128_n64_s2_t256).
- **Configs (29).**
  - F1: block_M ∈ {64, 128} × block_N ∈ {32, 64, 128} × stages 1–3 × threads 128/256.
  - F2: 256-row tiles with 256 or 512 threads.
  - F3: `epilogue="direct"` controls. The alternative epilogue stages O through the dead Q buffer. The two are within ±1.7%.
  - F4: natural-order controls.
  - Filters: smem ≤ 101376 B (`smem_bytes` is exact for all 58 compiled kernels), and ≤ 192 fp32 accumulators per thread. No config spills; registers range from 128 to 255.
- **Numerics.** Class E1 with key `(block_N,)`, which sets the online-softmax rescale points and the PV accumulation order.
  - block_M, threads, stages, epilogue and order are bitwise-neutral: in a block past a row's diagonal, alpha = 1 and P = 0.
  - The exponent is written as `exp2((s − m)·scale)`: a subtraction feeding a multiply cannot be contracted into an FFMA (the ptxas issue in `cotile/README.md`).
  - The accurate `exp2f` is used, not `--use_fast_math`. Fast math is a pass config, and CoKernels merge pass configs, so it would change the partner's bits.

### 1.2 Tests (`python -m cotile.tests.test_ops`, `python -m cotile.tests.test_cokernel`)

- **Op tests, S = 1536.** 1536 = 6 × 256, so every block_M divides it and the number of query blocks is not a power of two.
  - 29/29 configs match the fp32 torch SDPA reference (causal) in both builds. The worst ratio is 0.47, the same as FlashInfer's and torch bf16's.
  - grid == persistent bitwise 29/29; persistent builds use ⌈tiles/3⌉ ≤ 188 CTAs.
  - 3 numerics groups, bitwise 29/29.
  - Tile-space checks 29/29.
  - Details: `tests/prefill_tests.csv`, `tests/prefill_signatures.csv`; register and CTAs/SM figures are cross-checked with the driver.
- **CoKernel pairs** (prefill S=2048 × decode B16×8192), with SM binding, CTA binding, static/dynamic scheduling, takeover on/off and 3–5 runtime splits or ratios each:

  | pair | setup |
  |---|---|
  | `pd_e0` | 256-thread prefill; decode on 128 of 256 threads |
  | `pd_split` | split-KV decode; 2-stage prefill |
  | `pd_small` | 2 CTAs/SM, CTA binding; decode on 64 of 128 threads |
  | `dp_e0` | roles swapped; FlashInfer's 128×32 prefill tile |

  - 95/95 launchable settings (285 launches) pass: outputs within tolerance **and bitwise equal to the solo persistent builds**, every tile executed exactly once, and counters clean after every launch.
  - The 3 `smem="sum"` controls of the 1-CTA/SM pairs cannot launch (124–158 KB), as expected.
  - Details: `tests/cokernel_pd_*`.
- **Regression after the library rebuild.**
  - `test_ops`: all ops pass, now also asserting tile-space and numerics-group checks. GEMM 43, decode 36, RMSNorm 39, prefill 29, and the L2-hint axis 32/32.
  - `test_cokernel` full matrix: 299/306 settings pass. The other 7 are the expected non-launchable `sum` controls. The old pairs reproduce P1-3x2-B's 204/208.
  - Details: `tests/regression_*`.

### 1.3 TileLang bug fixed: pipelining a loop with a run-time trip count

- **Symptom.** Every prefill config with num_stages ≥ 2 failed in CUDA codegen with `Downcast from tirx.Sub to ir.IntImm failed`. The prefill K/V loop's trip count depends on the tile.
- **Root cause.** In `InjectSoftwarePipeline::EmitImpl` (`src/transform/inject_pipeline.cc`):
  - The epilogue range is [n, n + max_stage). Its extent, (n + max_stage) − n, was not simplified before `as_const_int`.
  - The epilogue was therefore emitted as a loop instead of being expanded, and its `cp.async.wait_group` counts depended on the loop variable (`ptx_wait_group(3 - k*2)`).
  - PTX needs an immediate count there.
- **Fix.** Simplify the extent. Static-extent kernels are unchanged: the regression tests above pass, and the rebuilt library is byte-identical across rebuilds.
- **Regression test.** `testing/python/transform/test_tilelang_transform_pipeline_dynamic_extent.py`, with num_stages 1–4, including trip counts below the pipeline depth. It fails before the fix and passes after.
- **Other pipeline tests.** The existing InjectSoftwarePipeline (14/14) and pipeline-planning (18/18) tests pass. One pipeline-barrier-ownership test, `test_nonws_im2col_tma_num_stages_3_uses_pipeline_barrier`, fails **with and without** this change. It is pre-existing, involves im2col TMA, and was not investigated.

## 2. FlashInfer POD fix (`research/patches/flashinfer_pod_stream_memset.patch`)

- **Change.** In `pod.cuh`:
  - `cudaMemset(tbAssign, …)` on the legacy default stream becomes `cudaMemsetAsync(tbAssign, 0, …, stream)` on the kernel's own stream. The reset is therefore ordered with the kernel and captured as a graph memset node.
  - The counter buffer is kept per device; before, it was one process-wide pointer.
  - The kernel code is unchanged.
- **Apply / revert.** Instructions are in `research/env_versions.md` §5.5.
  - The FlashInfer JIT rebuilds POD automatically: ninja tracks header dependencies, and the rebuild takes about 130 s.
  - The rebuild is verified: `pod_patch_rebuilt` in `pod_patch_patched.json`.
- **Remaining limitations.** POD launches that overlap in time on two streams still share the counters. The first call must be made outside stream capture; `flashinfer_ops.POD.run` enforces this.
- **PDL.** POD launches with programmatic stream serialization. After the memset, which is a non-kernel operation, the kernel cannot start before the reset completes. The 576 checked calls, plus the steady runs, show no corruption.

Verification (`research/bench/scripts/p4_pod_patch.py`; `pod_patch_unpatched.json`, `pod_patch_patched.json`). The configs are the 8 P4 pairs; each uses 3 input sets.

| check | unpatched | patched |
|---|---|---|
| default stream: repeat bitwise; fp32 reference | yes; ok | yes; ok |
| default stream: SHA-256 of both outputs vs unpatched | — | **equal (24/24)** |
| torch side stream, 24 back-to-back calls per config, no host sync | 44/48 wrong (2 configs) | **0/192 wrong** |
| green-context stream (96 SMs), 24 calls | 48/48 wrong | **0/192 wrong** |
| CUDA graph: 24 replays per config, inputs rotated, outputs NaN-poisoned before each replay | 46/48 wrong (only the first replay works) | **0/192 wrong**; replay times are real (e.g. 2019 µs for P8192_B64_S8192) |
| research wrapper `POD.run` on a side stream / in a graph | raises (guard) | 0/64 and 0/160 wrong |

- `research/bench/baselines/flashinfer_ops.py` now has `pod_patch_status()` and a `POD(require_patch=)` argument.
  - With the patch, `POD.run` works on any stream and under capture, after one eager call.
  - Without it, the old guard stays in place.
- In the steady pair runs (§4), POD runs on cobench's side stream. Afterwards its side-stream output is bitwise equal to a default-stream call (8/8).

## 3. Solo profiling protocol (`research/bench/scripts/p4_solo.py`)

**Flush sweeps** (clean flush, A1 protocol, `solo/points/*.json` in catalog format).
- Prefill: every config at every budget of the green-split grid, n_P ∈ {32, 48, 64, 80, 94, 108, 124, 140, 148, 156, 164, 172, 180} plus 188. That is 406 points per shape.
- Decode, new shapes (B16_S2048, B64_S2048): every config at 188 / 94 / 48, then C_lib ∪ top-6 at the complement budgets 188 − n_P.
- Decode, B16_S8192 and B64_S8192: the A1 points are reused (same 36 configs). Only budget 156 was added.
  - Spot check against A1: 8 flush points at ratios 0.999–1.003, and 12 steady configs at 0.999–1.001.
- FlashInfer references (single prefill; batch decode on the CUDA-core and tensor-core paths) at every budget.

**Steady solos** (`solo/steady_solo.json`): the top-6 flush configs plus the FlashInfer references at 188 SMs, back-to-back, with input rotation above 2× L2.
- solo-best is the fastest in steady mode (P1 definition).
- C_lib = Pareto ∪ best@94 ∪ best@48 ∪ solo-best.

**Steady pair runs** (`pairs_steady.json`): one interleaved run per pair with 7 variants: TileLang serial (the reference), the two TileLang solos, FlashInfer serial (prefill then the faster decode path), the two FlashInfer solos, and POD.

**Reference checks.** FlashInfer outputs are checked against fp32 with the op library's tolerance rule, or pass if they are no worse than torch's own bf16 computation (see §5).

## 4. Results (full tables: `tables.md`, `tables.json`; `p4_report.py`)

**TileLang prefill vs FlashInfer prefill** (steady, 188 SMs). The TL/FI flush ratio is the range over all 14 budgets.

| S | TileLang solo-best | TL µs @ MHz | FlashInfer µs @ MHz | TL/FI | TFLOP/s TL / FI | TL/FI flush, all budgets |
|---|---|---|---|---|---|---|
| 2048 | `m128_n128_s1_t256` | 135.7 @ 2247 (600 W) | 162.9 @ 2431 (533 W) | **0.833** | 253 / 211 | 0.74–0.93 |
| 8192 | `m256_n64_s1_t256` | 1699.9 @ 2219 (600 W) | 1846.9 @ 2221 (600 W) | **0.920** | 323 / 298 | 0.92–0.98 |

- TileLang is never slower than FlashInfer, at any budget.
- At S=2048 much of the gap is tile order. FlashInfer runs 512 CTAs (1.36 waves) in natural order. TileLang's natural-order control (`m128_n64_s2_t256_natural`, flush) is 150.5 µs, against 131.1 µs for LPT and 159.3 µs for FlashInfer.
- TileLang prefill S=2048 draws 600 W, where FlashInfer draws 533 W.

**Decode** (steady): TileLang is 0.96–0.99× FlashInfer's best path. TileLang runs at 1549–1644 GB/s; the table below lists the per-shape solo-best.

**Solo-best / C_lib** (details and budget-best rows: `tables.md` T3).

| shape | solo-best (steady) | C_lib | budget effect |
|---|---|---|---|
| prefill S2048 | `m128_n128_s1_t256` | 3: + `m64_n64_s1_t128` (flush-best, best@94), `m64_n32_s1_t128` (best@48) | full-GPU best loses 9–12% at ≤ 94 SMs |
| prefill S8192 | `m256_n64_s1_t256` | 6: + `m256_n32_s1_t256`, `m128_n128_s1_t256`, `m128_n32_s1_t128` (best@94/48), `m64_n64_s1_t128`, `m64_n32_s1_t128` | solo-best is budget-best from 156 to 188 SMs; at ≤ 140 SMs `m128_n32_s1_t128` wins, by up to 11.7% (32 SMs) |
| decode B16_S2048 | `n128_h4_sp1_t128_s1` (86.8 µs) | 7 | budget-best is within 2% of the full-GPU time from 32 SMs up (DRAM-bound) |
| decode B16_S8192 | `n128_h4_sp1_t128_s1` (331.1) | 7 (A1) | same |
| decode B64_S2048 | `n64_h4_sp1_t128_s1` (331.8) | 5 | same |
| decode B64_S8192 | `n64_h4_sp1_t128_s1` (1306.8) | 5 (A1) | same |

- The flush and steady rankings disagree for prefill S2048. The flush-best `m64_n64_s1_t128` is 4% slower in steady mode: 141.1 µs at 1853 MHz, against 135.7 µs at 2247 MHz. This is the power-cap effect seen in P1, so solo-best must come from steady mode.

**Pairing table** (steady; one run per pair).
- R_tc = 333.5 TFLOP/s: issued MMA FLOPs of prefill S8192 solo-best.
- R_dram = 1644 GB/s: decode B64_S8192.
- LB_power = (E_P + E_D) / 600 W. It is **not** a valid bound (P1): POD beats it on P8192_B64_S8192. It is shown only for reference.

| pair | t_P µs | t_D µs | t_P/t_D | T_serial (TL) µs | serial/Σsolo | LB_tc | LB_dram | LB_power | bound (tc, dram, max solo) | speed-up bound | FI serial µs | POD µs | POD vs FI serial | POD vs TL serial | MHz serial / POD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2048_B16_S2048 | 135.5 | 86.9 | 1.56 | 216.1 | 0.972 | 115.9 | 107.3 | 197.3 | 135.5 | 1.59 | 245.7 | 206.9 | 1.188 | 1.045 | 2547 / 2253 |
| P2048_B16_S8192 | 135.3 | 331.1 | 0.41 | 466.5 | 1.000 | 135.2 | 352.2 | 373.6 | 352.2 | 1.32 | 500.8 | 377.9 | 1.325 | 1.234 | 2712 / 2611 |
| P2048_B64_S2048 | 135.4 | 330.8 | 0.41 | 467.1 | 1.002 | 135.2 | 352.7 | 378.4 | 352.7 | 1.32 | 501.8 | 365.1 | 1.374 | 1.279 | 2709 / 2742 |
| P2048_B64_S8192 | 135.4 | 1306.7 | 0.10 | 1442.8 | 1.001 | 212.5 | 1332.3 | 1100.7 | 1332.3 | 1.08 | 1481.1 | 1348.3 | 1.098 | 1.070 | 2783 / 2799 |
| P8192_B16_S2048 | 1690.2 | 86.8 | 19.5 | 1759.4 | 0.990 | 1706.4 | 183.8 | 1752.4 | 1706.4 | 1.03 | 1908.6 | 1884.6 | 1.013 | **0.934** | 2279 / 2200 |
| P8192_B16_S8192 | 1690.9 | 331.3 | 5.10 | 1960.4 | 0.969 | 1725.7 | 428.7 | 1933.3 | 1725.7 | 1.14 | 2115.3 | 1979.1 | 1.069 | 0.991 | 2407 / 2119 |
| P8192_B64_S2048 | 1692.1 | 331.7 | 5.10 | 1965.6 | 0.971 | 1725.7 | 429.2 | 1939.3 | 1725.7 | 1.14 | 2117.6 | 1999.9 | 1.059 | 0.983 | 2400 / 2110 |
| P8192_B64_S8192 | 1694.0 | 1306.6 | 1.30 | 2862.8 | 0.954 | 1802.9 | 1408.8 | 2676.5 | 1802.9 | 1.59 | 3159.6 | 2357.2 | 1.340 | 1.215 | 2643 / 1909 |

**Readings for the P4 study**
1. **Duration ratios.** Only P2048_B16_S2048 (1.56) and P8192_B64_S8192 (1.30) fall in the plan's 0.5–2 window.
   - P2048 × {B16_S8192, B64_S2048} are at 0.41, which is borderline.
   - P2048_B64_S8192 (0.10) and the three prefill-heavy P8192 pairs (5.1–19.5) leave little room: speed-up bounds are 1.03–1.14.
   - The most headroom is in the two balanced pairs (bound 1.59) and the 0.41 pairs (bound 1.32).
2. **POD in steady mode vs its own serial:** 1.01–1.37×.
   - The 2026-09-22 flush-mode numbers were 1.02–1.38×. The main change is P2048_B16_S2048, which drops from 1.35× in flush mode to 1.19× in steady mode.
   - At P8192_B64_S8192 POD runs at 1909 MHz and 600 W. The power cap is binding, as in P1.
3. **Fairness caveat for CoKernel vs POD.** TileLang's solo ops are faster than FlashInfer's: prefill by 8–18% and decode by 1–4%. TileLang serial is therefore 3–12% faster than FlashInfer serial (2.6% for P2048_B64_S8192 up to 12.0% for P2048_B16_S2048).
   - Against TileLang serial, POD is only 0.93–1.28×. It is *slower* than TileLang serial on the three prefill-heavy P8192 pairs.
   - An absolute CoKernel-vs-POD comparison would therefore credit the CoKernel with TileLang's solo advantage.
   - The P4 study should report each system against its own serial as well as absolute times. It should add an attribution control, for example a TileLang prefill with FlashInfer-like scheduling (the natural-order configs) or FlashInfer-matched tiles (128×32, 128 threads, `m128_n32_s1_t128`: steady 138.4 µs at S2048 and 1776 µs at S8192).
4. **Serial vs sum of solos:** 0.954–1.002. This is power averaging, as in P1. Speed-ups must use the measured serial.

## 5. Problems and caveats

- **TileLang bug** with dynamic-trip-count pipelines: fixed (§1.3). One pre-existing, unrelated failure remains in `test_tilelang_transform_pipeline_barrier_ownership.py`.
- **Tolerance rule vs unscaled inputs.** On FlashInfer's own inputs (unscaled randn), the op library's tolerance rule fails at S=8192: |err| ≤ 2⁻⁵·rms(ref) + 2⁻⁶·|ref|.
  - Long causal rows average to tiny outputs, so rms(ref) is small, while short rows keep bf16-sized errors.
  - torch's own bf16 SDPA, FlashInfer and TileLang all get the identical worst ratio, 1.47.
  - The first `p4_solo.py` run stopped at this check. The FlashInfer reference check now also passes when the output is no worse than torch bf16's (≤ 1.05× its worst ratio); `fi_check` records both.
  - The op's own test inputs scale Q by 2 (peaked rows) and pass the rule at S=8192 (worst ratio 0.58). A test of the prefill op on unscaled long inputs would need a row-aware tolerance.
- **CV above 2% after 2 attempts.** FlashInfer prefill S2048 flush at 8 of 14 budgets (2.2–4.6%; tail effects of its 1.36 waves; it showed CV 2.9% on 09-22 too). FlashInfer decode_cc at 4 points (2.0–2.4%). One TileLang decode config (`n32_h1_sp1_t64_s2`, 2.0–2.2%). All steady runs have slice CV ≤ 0.53%.
- **Solo-best depends on the mode.** It differs between flush and steady for prefill S2048 (§4); steady is used, as in P1.
- **Kernel cache size.** `~/.tilelang/cache` holds 2.7 GB of old git-keyed kernel caches, and the root disk has about 50 GB free. Not touched.

## 6. Files

| file | content |
|---|---|
| `pod_patch_unpatched.json`, `pod_patch_patched.json` | P2' verification |
| `solo/points/*.json` | flush points (catalog format). The B16_S8192 and B64_S8192 files start from the A1 points (`meta.reused_from`) and include `spot_check`. |
| `solo/steady_solo.json`, `solo/summary.json` | steady solos per shape (with FlashInfer reference checks); solo-best, C_lib, budget-best |
| `pairs_steady.json` | the 8 steady pair runs (7 variants each) |
| `tables.md`, `tables.json` | T1–T5 (`p4_report.py`) |
| `tests/` | prefill op tests and signatures, prefill × decode CoKernel tests, regression summaries |
| `run_solo.log`, `run_guarded_solo.jsonl` | run logs (`*.log` is gitignored) |

**Reproduce** (after `source research/env.sh`):

```bash
python -m cotile.tests.test_ops --ops prefill_attn --no-l2 --out <dir>
python -m cotile.tests.test_cokernel --pairs pd_e0,pd_split,pd_small,dp_e0 --out <dir>
python testing/python/transform/test_tilelang_transform_pipeline_dynamic_extent.py
python research/bench/scripts/p4_pod_patch.py --phase unpatched   # before patching
FI=$(python -c 'import flashinfer,os;print(os.path.dirname(flashinfer.__file__))')
patch -p1 -d "$FI" < research/patches/flashinfer_pod_stream_memset.patch
python research/bench/scripts/p4_pod_patch.py --phase patched
python research/bench/scripts/run_guarded.py -- python research/bench/scripts/p4_solo.py --phases 1234
python research/bench/scripts/p4_report.py
```
