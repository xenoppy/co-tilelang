# P1 3×2 study, part A: solo and lib rows for GEMM × GQA decode (P1-3x2-A)

**Setup**
- Date: 2026-09-23, 02:14–04:44.
- Machine: RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs, 600 W cap, clocks not lockable), driver 580.173.02, torch 2.8.0+cu128.
- TileLang source tree at `01c40f00` plus this work's uncommitted changes. The kernel cache is keyed by the native-library hash (`research/env.sh`).
- GPU sharing:
  - qzr's RL job (ray worker) trained at 91–95 GB and ~600 W from ~02:25 to ~03:00. The guard held A1 for ~30 min.
  - Our first A1 attempt hit a CUDA OOM (below).
  - From 03:00 qzr was idle and later left the GPU. No measurement window after 03:03 saw foreign SM activity. Every JSON carries its guard record.
- Protocol:
  - Steady mode is primary (`cobench.bench_steady`: back-to-back, input rotation > 2×L2, thermal-plateau warmup, interleaved slices, speed-ups against the `serial` variant of the same run).
  - Clean-flush mode is secondary (`cobench.bench_variants`, write + `discard.global.L2` flush).
  - The fixed ClockProbe (carveout 100) is used throughout.
  - Green splits use `IGNORE_SM_COSCHEDULING`: GEMM on SMs [0, n_A), exact counts. The CoKernel SM split uses the same contiguous `%smid` ranges.

**Code**
- `research/bench/scripts/`:
  - `p1_common.py`: study pairs, budget grid, `OpData` (rotation copies shared by every config of a shape), `CoVariant`, `wait_gpu`/`retry_oom`.
  - `p1_solo_v1.py`: A1.
  - `p1_study.py`: A2/A3 stages and the winner rule.
  - `p1_carveout.py`: stage K.
  - `p1_report.py`: A4/A5 tables.
- Changes to shared code:
  - `cotile/catalog.py`: `C_LIB_BUDGETS = (94, 48)`. Only the budget-best at the definition's budgets joins C_lib; the extra budgets measured here do not.
  - `research/bench/scripts/solo_profile.py`: `FlushMeter` passes the flush kind to cobench explicitly. Since methodology v1 changed cobench's default flush, its "write" flush had silently become "clean".
  - **TileLang bug fix** (`tilelang/jit/adapter/nvrtc/adapter.py`, `tilelang/jit/adapter/cutedsl/adapter.py`): the NVRTC and CuTeDSL backends picked the launch stream with `str(self.target).startswith("cuda")`. `str(Target)` is now a JSON dict (`{"kind":"cuda",…}`), so every launch went to stream 0, the legacy default stream, whatever torch's current stream was. Events on a side stream measured 11 µs for a 363 µs GEMM. The check now uses `is_cuda_target(self.target)`.

## Files

| file | content |
|---|---|
| `solo_v1/points/*.json` | A1 points in the catalog format (`catalog.load("…/solo_v1")`): every config @188/94/48 and C_lib ∪ top-6 at the green-split budgets, clean flush |
| `solo_v1/summary.json` | per shape: solo-best (steady and flush), flush ranking, C_lib with reasons, budget-best per budget, budget curves, changes vs P1-S |
| `solo_v1/steady_solo.json` | steady-mode solo runs of the top-6 configs per shape (+ cuBLAS for GEMM) |
| `<pair>/study.json` | every stage: I1 (flush screen), I2/I3 (inter, steady), C1 (intra flush screen), C2 (steady confirmation + fidelity), C3 (refinement), C4 (CTA binding, steady), F (final table: steady + flush + role times). The main pair also has P (idle power) and K (carveout); second also has K |
| `tables.json`, `tables.md` | A4/A5 derived tables (`p1_report.py`) |
| `a1_run.log`, `run_<pair>.log` | run logs (gitignored `*.log`, local only) |

Pairs:

| pair | GEMM (A) | decode (B) | P1-S duration ratio |
|---|---|---|---|
| main | 4096×4096×4096 | B16×S8192 | 1.14 |
| r029 | 4096³ | B64×S8192 | 0.29 |
| r058 | 4096³ | B32×S8192 | 0.58 |
| r223 | 4096³ | B32×S2048 | 2.23 |
| second | 2048×4096×4096 | B32×S2048 | 1.15 |

Decode: Hq 32, Hkv 8, D 128.

## Summary

Steady mode. Every T is from the pair's final run (stage F), where serial, both solo runs and every cell's winner were interleaved. The ratio in parentheses is T_serial / T.

| pair | T_serial µs | LB tc / dram / power µs | T[solo,inter] | T[lib,inter] | T[solo,intra] | T[lib,intra] | T_inter* / T[lib,intra] | lib/solo: inter, intra |
|---|---|---|---|---|---|---|---|---|
| main | 725.9 | 386 / 388 / 673 | 593.2 (×1.224) | 586.4 (×1.238) | 592.5 (×1.225) | 590.0 (×1.230) | **0.994** | 1.012, 1.004 |
| r029 | 1710.9 | 404 / 1370 / 1408 | 1461.5 (×1.171) | 1441.4 (×1.187) | 1435.8 (×1.192) | 1435.8 (×1.192) | **1.004** | 1.014, 1.000 |
| r058 | 1061.6 | 392 / 715 / 927 | 797.4 (×1.331) | 797.4 (×1.331) | 786.9 (×1.349) | 786.9 (×1.349) | **1.013** | 1.000, 1.000 |
| r223 | 574.2 | 383 / 225 / 558 | 506.9 (×1.133) | 504.9 (×1.137) | 507.6 (×1.131) | 507.6 (×1.131) | **0.995** | 1.004, 1.000 |
| second | 375.0 | 193 / 205 / 354 | 315.8 (×1.188) | 313.3 (×1.197) | 320.2 (×1.171) | 317.3 (×1.182) | **0.987** | 1.008, 1.009 |

**Headline findings.**

1. **Kernel-internal orchestration does not beat the best inter-kernel co-location at the lib level.**
   - T_inter* / T[lib,intra] = 0.987–1.013 across the 5 pairs.
   - The CoKernel wins by 1.3% (r058) and 0.4% (r029). It loses by 0.5–1.3% on main, r223 and second.
   - Every speed-up vs serial, inter or intra, is ×1.13–1.35 and set mainly by the pair, not by the mechanism.
2. **Choosing from C_lib instead of the solo-best configs is worth ≤ 1.4%** in either column: inter 1.000–1.014, intra 1.000–1.009.
   - In 3 of 5 pairs, the best CoKernel uses exactly the solo-best configs.
3. **The 600 W power cap is the binding resource in 4 of 5 pairs.**
   - Every co-located variant of main, r058, r223 and second runs at 600 W, at 1.84–2.47 GHz.
   - Under the cap, T ≈ E_iter / 600 W. The best inter and intra variants of a pair differ in energy per iteration by < 2% (main: 352–356 mJ).
   - r029 is DRAM-bound instead: its best variant is 4.8% above the DRAM bound, at 552 W.
4. **The "power lower bound" built from solo energies is not a bound.**
   - Co-located iterations use 7–16% less energy than E_A + E_B. For example, main uses 354 vs 404 mJ.
   - Two effects combine. The idle-but-clocked board power (P_s ≈ 143 W, measured) is paid once instead of twice. And the memory-bound decode, which runs at 2.82 GHz solo, runs at 1.8–2.4 GHz in the co-run and needs less energy there.
   - Even with P_s removed, the estimate exceeds the achieved T by 8–11% on the power-capped pairs.
5. **Screening fidelity is good at the coarse level but weak among near-ties.**
   - Clean-flush screen vs steady on 14 confirmed lib CoKernels per pair: Spearman 0.77–0.99.
   - The steady winner was always in the screen's top-8.
   - Regret of trusting the screen's #1: 0–1.3%.
   - Within the top-8, the correlation is ≈ 0 on the power-capped main and r058 pairs. There the flush screen systematically prefers more GEMM SMs.

## A1 — C_lib re-validation (fixed probe, clean flush)

**Protocol** (`p1_solo_v1.py`)
- The P1-S rep loop (`solo_profile.FlushMeter`) with the clean flush and the fixed ClockProbe.
- Global warmup until the temperature is flat, then a batch warmup per shape.
- Config-outer order.
- Each point: ≥ 0.3 s + 20 reps of warmup, then ≥ 0.25 s + 50 reps timed.
- pmon guard; contaminated points are re-measured (none were).
- Every config is checked against the fp32 reference first.

**Phases**
1. All configs (GEMM 43, decode 36 per shape) at 188 / 94 / 48 SMs: 690 points.
2. C_lib ∪ top-6 at the green-split budgets: 520 points.
   - GEMM at {64, 80, 108, 124, 140, 148, 156, 164, 172, 180} SMs.
   - Decode at {8, 16, 24, 32, 40, 64, 80, 108, 124, 140} SMs.
3. Steady mode at 188 SMs: the top-6 flush configs per shape, plus cuBLAS (fp32 reduction) for GEMM.

**Definitions (recorded in `solo_v1/summary.json`).**
- **solo-best** = the fastest in steady mode among the top-6 clean-flush configs. The study's primary mode is steady, and back-to-back power-capped execution can reorder configs.
- **C_lib** = Pareto(time@188 × (smem, regs·threads, threads, −CTAs/SM)) ∪ best@94 ∪ best@48 ∪ {solo-best}.
- **Intra column:** GEMM `ws=auto` members become their `ws=off` twins, because auto warp specialization cannot host a second role.

| shape | solo-best (steady) | flush-best / P1-S best | C_lib (size; members beyond the Pareto front) | change vs P1-S |
|---|---|---|---|---|
| GEMM 4096³ | `128x256x64_s2_t256_direct` (425.7 µs; cuBLAS 422.0) | same / same | 10; best@94 `128x128x32_s3_t128`, best@48 `128x128x64_s2_t256_wsauto` | −`128x256x32_s2_t256_direct`. best@94 changed (was `128x128x64_s2_t256_wsauto`) |
| GEMM 2048×4096² | `128x128x64_s2_t128_wsauto` (227.5 µs; the t256 variant 228.5, a 0.4% tie; cuBLAS 244.0) | `…_t256_wsauto` / `…_t256_wsauto` | 9 | −`64x64x64_s3_t128` |
| decode B16×8192 | `n128_h4_sp1_t128_s1` (330.8) | same / same | 7; best@94 `n64_h4_sp1_t128_s2` | −`n32_h2_sp1_t64_s2`, −`n32_h4_sp1_t64_s3` |
| decode B64×8192 | `n64_h4_sp1_t128_s1` (1306.9) | same / same | 5; best@48 `n128_h4_sp1_t128_s1` | +`n128_h4_sp1_t128_s1`, −`n32_h4_sp1_t128_s2`, −`n64_h4_sp1_t128_s2` |
| decode B32×8192 | `n64_h4_sp1_t128_s1` (656.1) | same / same | 4 | −`n32_h4_sp1_t128_s2`, −`n64_h4_sp1_t128_s2` |
| decode B32×2048 | `n64_h4_sp1_t128_s1` (168.3; all top-6 within 0.7%) | `n32_h4_sp1_t64_s2` / `n64_h4_sp1_t64_s1` | 4 | +`n64_h4_sp1_t128_s1`, −`n32_h4_sp1_t128_s2`, −`n64_h4_sp1_t128_s2` |

**Changes vs P1-S** (median new/old time of the same config; the clean flush removes the write-flush bias).

| op | @188 | @94 | @48 |
|---|---|---|---|
| GEMM | 0.975–0.985 | 0.970–0.977 | 0.978–0.996 |
| decode | 0.91–0.96 | 0.92–0.96 | 0.93–0.97 |

- Individual GEMM points at @94 moved by up to −8.7%, which is consistent with the old probe costing large-smem kernels one SM. The GEMM 4096³ @94 budget-best changed accordingly.
- **The GOLDYLOC effect for GEMM 4096³ remains.** The full-GPU best loses 14.0% at 94 SMs and 11.1% at 48.
- **Decode changes are ties.** The decode C_lib changes are among near-tied configs: at 188 SMs the top-6 of every decode shape lie within 0.7% of each other.
- **Decode on few SMs shows wave quantization.** Examples:
  - B16×8192 `n128` (1 CTA/SM, 128 tiles): 40 SMs → 372 µs; 32 or 48 SMs → 332–334 µs; 124 SMs → 382 µs.
  - Energy per call is flat (~140 mJ) from 32 to 188 SMs.

**Steady vs flush solo.**
- The GEMM steady ranking matches the flush ranking for the top-2 (4096³). For 2048 the top two swap, a 0.4% tie.
- The TileLang solo-best is within 0.9% of cuBLAS for GEMM 4096³ (425.7 vs 422.0 µs) and 7% faster for 2048 (227.5 vs 244.0 µs).

## A2 — inter-kernel variants

**Variants**
- serial (reference); solo_a, solo_b.
- Two streams: equal priority, A-high or B-high priority × host order AB / BA.
- Green splits, where n_A ∈ {48, 64, 80, 94, 108, 124, 140, 148, 156, 164, 172, 180}, then refined by ±2/4/6 SMs around the best.
  - **Solo row:** solo-best configs.
  - **Lib row:** per split, each side takes the C_lib config with the lowest solo time at its budget (A1 phase 2). At the best split, the 2nd-best config per side is also measured.
- Lib-row streams: every C_lib × C_lib pair in the flush screen; the top-3 in steady; the best with all order × priority variants.

**Two streams are deterministic in steady mode here, and priorities matter only one way.**
- Equal priority and A-high always ran GEMM-first: the decode waits until the GEMM's CTAs drain, because 96 + 73 KB cannot share an SM. Examples: main ×1.13 (a 455, b 640 µs); second ×1.15.
- B-high priority is serial-like: ×0.99–1.05.
- For a co-residable lib pair (GEMM `128x128x32_s3_t128` 48 KB + decode 38–39 KB), equal priority still runs GEMM-first: ×1.05 main, ×1.10 second.
- **B-high priority makes that pair co-reside** (main ×1.147, the best stream variant there). In second it hurts (×1.05). The streams mode is thus both config- and priority-sensitive, and never the best inter variant.

**Green splits.**
- The curves are jagged, because both sides quantize.
  - GEMM: 512 128×256 tiles at 1 CTA/SM → 5 waves at 124 SMs, 4 at 128–170, 3 at ≥ 171.
  - Decode: 128–512 tiles at 1–2 CTAs/SM on the complement.
  - Example (r058): 140 → ×1.328, 148 → ×1.237, 156 → ×1.302.
- The best splits:

  | pair | solo row | lib row |
  |---|---|---|
  | main | 148/40 | 156/32 |
  | r058 | 142/46 | same configs |
  | r223 | 172/16 | 172/16 |
  | second | 164/24 | 164/24 |
  | r029 | 80/108 | 148/40 |

- **Lib selection helps green by 0–1.4%.**
  - Main: GEMM `128x128x64_s2_t256_wsauto` (solo #3, +2.1%) on 156 SMs, at 1836 MHz vs 2175 MHz for the solo pair.
  - r029: the ws=auto GEMM + `n128` decode on 148/40, which lowers power (546 W) and lets the decode own more DRAM time.

**Carveout (stage K).**
- The attribute cannot be set on TileLang's default tvm_ffi launch path (no function handle). It can on NVRTC-backend builds of the same kernels (`cuKernelSetAttribute`). This required the stream fix above.
- Default vs 100% carveout gives the same times (±0.2%) and the same per-op completion times for all 12 order × priority variants of the solo pair and of a co-residable lib pair, in main and second.
- **The carveout does not change co-residence for these kernels.** The co-residable pair co-resides only when the decode has the higher stream priority.
- NVRTC builds time within 0.5% of the tvm_ffi builds.

## A3 — intra-kernel variants (CoKernel)

**Search (per pair).**
- **C1 flush screen.**
  - Solo row (40 variants): SM binding, dynamic, takeover on/off at all 12 splits; decode chunk 2 and 4 and the static schedule (takeover on/off) at 4 splits.
  - Lib row: every C_lib(A)′ × C_lib(B) pair (32–63 pairs) × SM binding, dynamic, takeover, chunk 1 × splits {64, 94, 124, 148, 172}. Plus CTA binding (376 CTAs, one GEMM + one decode CTA per SM) for every pair whose CoKernel fits 2 CTAs/SM: 24 per pair.
  - Scored as t / t_serial of the same `bench_variants` group of ≤ 24 variants, 50 reps each.
- **C2 steady confirmation** (1.0 s slices, 3 rounds): lib top-8 + 6 random others, and the solo top-3.
- **C3 steady refinement** around the lib and solo winners: split ±4/8/12; takeover flipped; chunk (1,1)/(1,2)/(1,4)/(2,1); static with takeover on/off; CTA binding if feasible.
- **C4 steady CTA binding:** the screen's best 3 CTA variants, plus the decode-CTA count sweep for the best one. With 188 + k CTAs, every SM hosts a GEMM CTA and k SMs also host a decode CTA, k ∈ {47, 94, 141, 188}.
  - The rank-based 1:1 rule is the only co-resident ratio at 2 CTAs/SM. k is the effective ratio knob.
- **Correctness.** Every CoKernel variant was checked bitwise against its configs' grid kernels on first use. All matched. Every role smem buffer offset ≡ 0 mod 128.

**Screening fidelity** (the confirmed set is the top-8 + 6 random lib variants).

| pair | Spearman | Kendall | Spearman within top-8 | steady winner's screen rank | in top-8 | regret of screen #1 |
|---|---|---|---|---|---|---|
| main | 0.77 | 0.63 | −0.24 | 7 | yes | 1.3% |
| r029 | 0.93 | 0.87 | 0.62 | 1 | yes | 0.0% |
| r058 | 0.82 | 0.69 | 0.00 | 8 | yes | 0.2% |
| r223 | 0.99 | 0.96 | 0.98 | 1 | yes | 0.0% |
| second | 0.96 | 0.89 | 0.83 | 1 | yes | 0.0% |

- In every pair, every randomly sampled variant was slower than every top-8 variant. Example (main): random ×0.83–1.15 vs top-8 ×1.19–1.23.
- **The screen separates good from bad reliably, but not near-ties (within ~1%).**
- **On the power-capped pairs the flush screen is biased toward more GEMM SMs.** Flush-mode reps have power headroom, so the GEMM runs faster. Example: main's screen favored n_A = 148, steady favored 124.
- The C3 refinement changed the final winner in r029 (chunk 4 on `a0b0`) and r058 (split 128). C4 changed it in second (partial CTA binding).
- Total cost: 184–379 flush variants plus ~45 steady variants per pair.

**Orchestration results.**
- **SM binding + dynamic queue is the best CoKernel orchestration in 4 of 5 pairs.**
  - The split optimum is flat within ~1% over ±8–12 SMs.
  - **Exception, r058:** n_A = 128 gives exactly 4 GEMM waves, ×1.349 vs ×1.318 at 124 and ×1.329 at 132. The role times show the GEMM finishing at 596 vs 710 µs.
- **Takeover.**
  - About neutral when the split is balanced: main solo, off at 140 ×1.225 vs on ×1.219; second similar.
  - Essential when the decode role is small: main lib winner (64-thread decode on 64 SMs), on ×1.229 vs off ×1.124.
  - In r223 the best CoKernel gives the decode only 4 SMs (184/4) and relies on takeover: the GEMM CTAs finish the decode tiles after the GEMM ends (T_A 463, T_B 501 µs).
- **Chunk.** A decode chunk of 4 helped only r029 (512 decode tiles; ×1.192 vs ×1.161 at chunk 1). Elsewhere it cost 1–9%.
  - Chunk 2 in r029 was anomalously bad (×1.106). This is unexplained.
- **Static schedule:** 0.3–2.9% slower than dynamic at the same split in every pair.
- **CTA binding (POD-style co-residence)**:
  - Feasible only with small tiles: GEMM ≤ 48 KB smem at 128 threads (`128x128x32_s3_t128` +7.3%, `128x64x64_s2_t128` +19% solo), decode ≤ 39 KB. **The solo-best pair is infeasible in every pair**, because of the 96 KB GEMM and the 256-thread × 217–224-register CTAs.
  - Best CTA variant per pair: main ×1.177, r029 ×1.185, r058 ×1.321, r223 ×1.018, second ×1.182.
  - **It is the lib-row intra winner only in second** (282 CTAs: 94 SMs host a decode CTA; ×1.182 vs ×1.171 for SM binding).
  - Partial co-residence (k = 94 or 141) often beats full co-residence (k = 188).
  - On the power-capped pairs, co-resident variants run at a lower clock (1.9–2.1 GHz): two busy roles per SM draw more power per SM.
- **WS-off cost of the minimal adjustment.** In the second pair, the solo-best GEMM (`128x128x64_s2_t128_wsauto`, TMA warp-specialized) becomes its ws=off twin inside a CoKernel.
  - The twin costs +2.6% in steady mode (234.3 vs 228.4 µs) and +4.0% in flush mode.
  - The inter-kernel variants keep ws=auto: the green lib winner of second uses the ws=auto 128×128 t256 GEMM. This is one reason intra trails inter there by 1.3%.
  - The GEMM 4096³ solo-best is ws=off already.
- **Timing builds.**
  - The per-tile barrier + `%globaltimer` build of each intra winner runs within ±0.3% of the untimed build (F table).
  - Its device-side role completion times (T_A / T_B) are in the tables.

## A4 — per-pair tables

Generated by `p1_report.py` (also in `tables.md`). How to read them:
- **Configs** show each config's clean-flush solo rank at 188 SMs and its solo slowdown vs the solo-best.
- **Per-op ends** are Par completion times relative to the iteration start. For CoKernels they are device-side role completion times from the timing build (median of 24 launches after a warm-up).
- **LB_tc and LB_dram** use the P1-S catalog's best measured machine rates (357 TFLOP/s, 1642 GB/s).
- **LB_power** = (E_A + E_B) / 600 W, with E = NVML board power × time of the op running back-to-back alone in the same F run. It is **not a valid bound** (see above).

### main: GEMM 4096³ × decode B16×S8192

| variant | knobs | configs A / B (solo rank, solo slowdown) | steady µs | × serial (paired min–max) | MHz | W | flush µs | flush × | per-op or per-role end (µs) |
|---|---|---|---|---|---|---|---|---|---|
| serial | – | solo-best | 725.9 | 1.000 | 2653 | 580 | 731.5 | 1.000 | – |
| solo A / solo B | full GPU, back-to-back | solo-best | 430.0 / 330.8 | – | 2259 / 2821 | 600 / 442 | – | – | – |
| **T[solo,inter]** `gr_s_148` | green 148/40 | `128x256x64_s2_t256_direct` (#1) / `n128_h4_sp1_t128_s1` (#1) | 593.2 | ×1.224 (1.219–1.227) | 2175 | 600 | 549.3 | ×1.332 | a 591, b 506 |
| **T[lib,inter]** `gr_l_156` | green 156/32 | `128x128x64_s2_t256_wsauto_smem` (#3, +2.1%) / `n32_h4_sp1_t64_s2` (#7, +0.6%) | 586.4 | ×1.238 (1.235–1.242) | 1836 | 600 | 537.2 | ×1.362 | a 583, b 483 |
| two streams (solo) `st_eq_ab` | eq priority, AB | solo-best | 643.4 | ×1.128 | 2465 | 600 | 681.4 | ×1.073 (decode-first mode) | a 455, b 640 |
| **T[solo,intra]** `co_sm_dyn…a0b0_n140` | SM-bind dyn 140/48, no takeover | solo-best | 592.5 | ×1.225 (1.223–1.228) | 2183 | 600 | 546.8 | ×1.338 | T_A 584, T_B 461 |
| **T[lib,intra]** `co_sm_dynT…a0b3_n124` | SM-bind dyn 124/64 + takeover | `128x256x64_s2_t256_direct` (#1) / `n64_h4_sp1_t64_s1` (#4, +0.5%) | 590.0 | ×1.230 (1.227–1.233) | 2183 | 600 | 548.9 | ×1.333 | T_A 586, T_B 464 |
| static (best) | SM static 124/64 + takeover | as lib winner | 596.3 | ×1.217 | 2209 | 600 | 538.6 | ×1.358 | – |
| CTA binding (best) | 376 CTAs | `128x128x32_s3_t128_smem` (#5, +7.3%) / `n64_h4_sp1_t128_s1` (#3, +0.5%) | 616.8 | ×1.177 | 1895 | 600 | 570.9 | ×1.281 | – |

The complete tables for all five pairs (with the same columns) are in [`tables.md`](tables.md). The summary rows are above.

## A4 — attribution (qualitative)

**Power cap: main, r058, r223, second.**
- Every co-located variant runs at 600 W, versus 543–597 W for serial.
- The clock falls from 2.53–2.71 GHz (serial) to 1.84–2.38 GHz.
- The speed-up equals the drop in energy per iteration: main serial 421 → 352–356 mJ.
- What lowers the energy is overlap, not orchestration: the static power is shared, and the decode runs at a lower clock. The mechanisms differ in energy per iteration by < 2%, hence in T by < 2%.
- **Clock is not the objective.** The lib-inter winner of main has the *lowest* clock (1836 MHz) and the best time, because a more power-dense GEMM finishes its tiles on more SMs. Cycles and time disagree; time wins.

**DRAM: r029, and r058's tail.**
- r029's decode moves 2.1 GB per iteration. Its best variants sit 4.8–6.7% above the DRAM bound, below the power cap (546–552 W).
- In r058 the decode (1.07 GB) is the critical path of every good variant: T_B 782 µs vs T_A 596 µs in the intra winner.

**Wave quantization.**
- Visible in the green curves and in r058's CoKernel optimum at n_A = 128 (512 tiles / 128 SMs = 4 waves).
- Visible in the decode budgets (A1).
- The dynamic queue with takeover absorbs part of it. The green splits cannot.

**SM capacity / resource coupling.**
- CTA-level co-residence needs small tiles, which lose 6–19% solo.
- It also runs at a lower clock (both roles busy per SM, more power per SM).
- It won only in the short second pair, and only with partial co-residence.

**Tail / takeover.**
- Takeover's value is concentrated where one role would otherwise leave SMs idle: decode on 4–64 SMs.
- With balanced splits the per-role ends differ by 40–120 µs, and takeover gives ≤ 0.6%.

## A5 — D1 interim readout (thresholds unchanged)

| pair | T_inter* / T[lib,intra] | lib vs solo, inter | lib vs solo, intra |
|---|---|---|---|
| main | 0.994 | 1.012 | 1.004 |
| r029 | 1.004 | 1.014 | 1.000 |
| r058 | 1.013 | 1.000 | 1.000 |
| r223 | 0.995 | 1.004 | 1.000 |
| second | 0.987 | 1.008 | 1.009 |

**Against plan §2.8.**
- **Continue (weakened claim)** needs T_inter* / T[lib,intra] ≥ 1.10 on ≥ 2 pairs. It holds on **0 of 5** GEMM × decode pairs; the ratio is within ±1.3% of 1.
- **Continue (full claim)** additionally needs lib/derived ≥ 1.05, which is not measured yet. Its first condition, T_inter* / T[derived,intra] ≥ 1.10, would need the derived row to be ≥ 8.7–11.4% faster than T[lib,intra].
- **Lib vs solo selection matters equally little in both columns** (≤ 1.4% inter, ≤ 0.9% intra). There is no sign that C_lib selection interacts with the kernel-internal mechanism.
- This is one of the plan's four operator pairs (P1). D1 is decided on P1–P4, so this is interim.

## Observations for the derived row

1. **The contract that matters here is the energy per iteration under the 600 W cap, not SM slots.**
   - In 4 of 5 pairs, T ≈ E_iter / 600 W for every good variant.
   - A derived implementation helps only if it moves fewer bytes or burns less SM power per unit of work.
   - The decode at 2.82 GHz, stalled on DRAM, is the obvious waste. Candidates: fewer resident warps or CTAs per unit of bandwidth, larger KV blocks per load, and L2 streaming hints (evict-first) so the KV stream does not evict the GEMM's panels.
   - Nsight Compute counters are unavailable (admin-only), so the L2-interference part is a hypothesis.
2. **Decode tile granularity is missing from C_lib.**
   - Split-KV configs never enter C_lib: they are dominated in solo time and have identical resource vectors.
   - Yet the co-location optima depend on tile counts. Chunk 4 over 512 decode tiles helped r029. Decode wave quantization on 16–64 SMs is visible in A1. The r223 winner runs the decode on 4 SMs + takeover.
   - A derived decode whose split (tile count) follows the partner's SM share, and a GEMM tile count matched to the SM split (e.g. split-K / Stream-K to avoid 4.13-wave splits), are concrete candidates.
3. **Register/thread coupling blocks CTA-level co-residence for all fast configs.**
   - CoKernel registers ≈ max(roles) + 10–40, i.e. 217–224 × 256 threads, so 1 CTA/SM.
   - Co-residence requires a 128-thread ≤ 48 KB GEMM, 6–19% slower solo.
   - The contract to derive against: GEMM (128 threads, ≤ ~55 KB smem, ≤ ~200 regs) + decode (≤ 39 KB, 64–128 threads), with the GEMM re-tiled to keep its solo speed at that contract (e.g. a register-capped 128×128 with more stages).
   - Partial co-residence (k = 94–141 decode CTAs) beat full co-residence in 3 of 5 pairs, so the contract should allow asymmetric per-SM mixes.
4. **WS is incompatible with a two-role CoKernel.** The GEMM 2048 CoKernel pays +2.6% for losing TMA warp specialization, and inter-kernel keeps it. A role-local producer/consumer split (WS inside the GEMM role) would remove a systematic intra-column handicap.
5. **Measurement for the derived row.**
   - Screen in flush mode only to prune (it is reliable at the coarse level).
   - Confirm the top-8 and refine the split in steady mode. On power-capped pairs, never trust the flush optimum's split.

## Caveats and negative results

- **LB_power is an estimate that the data violate** (co-runs use 7–16% less energy than E_A + E_B). A valid power bound needs energy as a function of the clock/voltage point, which cannot be measured without clock locking.
- **Numerical-quality notes.**
  - Decode configs in C_lib are near-ties (≤ 0.7%). Which one the lib rows pick is partly noise, though the steady confirmations bound that at < 1%.
  - Flush-mode speed-ups are 8–13% higher than steady ones on power-capped variants (e.g. green lib main ×1.362 vs ×1.238), as in methodology v1.
- **Search scope.**
  - The lib-row screen covered SM binding with takeover at 5 splits, plus CTA binding at 1:1. No-takeover, chunk and static variants were tried only around the winners (C3). The solo row, where they were swept fully, never had them win by more than 0.6%.
  - M1+ (persistent / occupancy-limited kernels on streams) and the Rammer / virtual-CTA alignment baselines were not measured.
- **GPU sharing.**
  - qzr's worker grew to 91–95 GB while training. Our first A1 attempt died of CUDA OOM.
  - The scripts now wait for both SM-idle and free memory (`p1_common.wait_gpu`, `retry_oom`).
  - Conversely, our processes (1–5 GB) could OOM their job if it ramps up while we hold memory. This is worth coordinating.
- **Bookkeeping.**
  - Main's `stage_wall_s["F"]` is from the redo after C4 was added. The first F run (159 s) and one aborted K attempt (~30 s) are not counted in the GPU time.
  - One winner-rule fix was applied after the r058 run: a lib-row CoKernel that uses exactly the solo-best configs now also counts for T[solo,·]. It changed only r058's T[solo,intra], whose new winner was already in F.

## GPU time

| part | GPU time |
|---|---|
| A1 | 24 min (incl. 2.5 min global warmup) |
| main (incl. P, K) | 18 min |
| r029 | 12 min |
| r058 | 11 min |
| r223 | 12 min |
| second (incl. K) | 16 min |
| **total** | **≈ 1.55 h** (+ ~3 min uncounted redo), excluding ~30 min of guard waits for qzr and ~4 min of kernel compilation |

The CPU-only CoKernel precompile (472 kernels, 6 min) overlapped the guard wait.

## Reproduce

```bash
source research/env.sh
python research/bench/scripts/p1_solo_v1.py                     # A1 -> solo_v1/ (~25 min GPU)
for p in main r029 r058 r223 second; do
  python research/bench/scripts/p1_study.py $p --stages I1,I2,I3,C1,C2,C3,C4,F   # ~12-16 min each
done
python research/bench/scripts/p1_study.py main --stages P,K
python research/bench/scripts/p1_study.py second --stages K
python research/bench/scripts/p1_report.py                      # tables.json, tables.md
```

`p1_study.py <pair> --compile-only` precompiles the screen's CoKernels (CPU). Every script waits for a free GPU: pmon rule plus free memory.
