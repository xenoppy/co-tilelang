# Co-location limit study (Task L, 2026-09-24)

**Question.** On the RTX PRO 6000 (188 SMs, 600 W cap), is fine-grained intra-SM co-location (CTA-level co-residence, warp specialisation, same-warp interleaving) fundamentally better than SM partitioning? The test uses idealised synthetic kernels, so the ceiling is set by the hardware and not by a particular CoKernel design.

**Answer.**
- **Under the 600 W cap: no.** Across 3 work ratios × 2 MMA families, the best intra-SM variant beats the best partition by at most **3.5%** (register-only MMA, ratio 1). For the GEMM-like family the gap is **−0.7% to +0.7%**.
- **Where power does not bind for the partition:** intra-SM co-location wins by **11.8% to 27.9%**, measured in two regimes: a reduced-intensity MMA on the full GPU, and the full kernels on a 94-SM sub-device. The GEMM on 94 SMs still hits the cap; there the gap is 8.1%.
- **The deciding resource is power.** Under the cap, the MMA side is power-bound, not SM-bound. A partition gives the MMA fewer SMs, but they clock higher (2427 vs 2197 MHz). That recovers most of the SM loss.
- **In-SM contention is second order.** For the register-only MMA it costs ≤ 4%, for the GEMM-like kernel 7–9%. It uses up most of the remaining V/f advantage of spreading the MMA over all SMs.

Files
- `limit_study.json`: every stage, including screening and final steady runs, work-completion checks, variant descriptions (launch shapes, registers, smem) and the calibration.
- `test_output.json` (+ `repro_matmul_graph.json`, written by test_6): `research/bench/tests/test_cobench.py` run on the final code, 18/18 PASS (test_14–16 are new).
- Code:
  - `research/bench/cobench/stress.py`: the kernels, workloads, variant builders and `split_nested`.
  - `research/bench/scripts/limit_study.py`: stages and report.

Reproduce: `source research/env.sh; python research/bench/scripts/run_guarded.py -- python research/bench/scripts/limit_study.py run`, then `... limit_study.py report`. Measurement GPU time was about 42 min: 36 min of study stages, including a superseded first reg 1 cell, plus kernel development runs. The tests took about 8 min. The guard records are clean for every run: no foreign SM activity, no re-measurement.

## Kernels (plan §1.4 stress kernels v0)

All kernels come from one NVRTC kernel family, `cobench.stress`. Every kernel is persistent and takes work from dynamic atomic queues.

**A, register-only MMA (`reg`).**
- One unit = one warp's `a_iters × 8` bf16 `mma.sync.m16n8k16` instructions, in 8 independent accumulator chains.
- Two operand register sets alternate between consecutive mma, so the datapath toggles. Constant operands turned out to give the same clock and power (V table).
- ≤ 64 registers.

**A, GEMM-like (`gemm`).**
- One unit = a 128×128 tile over K = 2048, i.e. 64 k-steps of BK = 32.
- Mainloop: cp.async pipeline, 64-byte rows XOR-swizzled in smem, ldmatrix.x4, mma.sync.
- A and B come from an 8 MB K-major buffer that stays in L2, so the kernel has the LSU / smem / L2 traffic of a real GEMM mainloop.
- The C tiles equal torch to a relative error of 2.6e-6 (test_14).
- The solo configuration is 4 warps with 64×64 warp tiles, 3 stages, 2 CTAs/SM and 232 registers. It reaches 324 TFLOP/s = 80% of 1024 FLOP/clk/SM at 2092 MHz and 600 W. For comparison, cuBLAS 4096³ reaches about 86%.

**B, stream.**
- Each warp streams 32 KB chunks of a DRAM buffer of at least 320 MB (> 2× L2).
- 16-byte `ld.global.nc.L1::no_allocate`, 4 per lane per batch, double-buffered, with an L2 evict-first policy.
- Variants: cp.async ring, bulk copy (`cp.async.bulk` + mbarrier), read+write.

**Work-completion check (L2).**
- Every unit increments a per-unit hit counter and adds a hash of its result to a checksum. The GEMM hash covers every C element with its coordinates, so it does not depend on the tiling. The stream checksum is the 64-bit sum of the words read, compared with a torch reference.
- Every variant of every cell was checked before measurement: 2 launches, every A and B unit done exactly 2×, both checksums equal to 2× the reference. 25–35 variants per cell passed, 0 failed.
- A negative control is in test_14: tickets that start at n/2 are detected as n/2 units that were never done.

## Validation (L1, stage V, steady state after the thermal plateau)

| kernel | time | clock | power | result |
|---|---|---|---|---|
| register-only MMA, 8 warps/SM | 541.0 µs | 2721 MHz | 601 W | **993 FLOP/clk/SM = 97.0% of 1024** (508 TFLOP/s) — PASS |
| same, constant operands | 540.8 µs | 2721 MHz | 600 W | 97.0%, same power: operand toggling does not change power here |
| stream ldg (the study's B) | 662.9 µs (1 GB) | 2837 MHz | 415 W | 1620 GB/s = **99.0% of the best measured read (1637 GB/s, bulk copy)** — PASS; 91% of the 1792 GB/s theoretical peak |
| cp.async / bulk / `cobench.read_u4` / no evict-first hint | | | 409–441 W | 1633 / 1637 / 1633 / 1617 GB/s |
| read+write (512 MB → 512 MB) | 748.2 µs | | 386 W | 1435 GB/s moved |
| GEMM-like 4w×2 CTAs / 8w×2 / 8w×1 | 424.5 / 442.7 / 465.0 µs | 2092 / 2057 / 2336 MHz | 600 W | 80.4% / 78.4% / 65.7% of peak per clock |

Stream GB/s as a function of SM count (green partitions): 8 SMs 544, 16 → 1044, 24 → 1404, 32 → 1556, 40 → 1590, 48 → 1609. The DRAM is saturated by about 40 SMs, 21% of the GPU. Even 8 KB in flight per SM saturates it across 188 SMs.

## Co-location results (L3)

**Setup.**
- Every variant in a cell does the same work, W_A MMA FLOPs + W_B stream bytes.
- Ratio = T_A / T_B measured solo:
  - reg family: the stream is fixed at 512 MB and a_iters is scaled;
  - gemm family: 2048 tiles, with the stream scaled to 5, 10 or 21 × 32 KB per tile.
- Screening: every variant and split in one steady round.
- Final: `bench_steady` with 4 interleaved rounds of 1.2 s slices after the thermal plateau, run on serial, both solos and the best two of every kind. Slice CV ≤ 0.54% in every final run.
- The speed-up is against serial from the same run (A then B, each on the full GPU).
- The variant kinds:
  - **green**: green-context partition, A on [0, n);
  - **sm**: one persistent kernel with SM-level roles, dynamic queues and takeover;
  - **cta**: k_A MMA CTAs + k_B stream CTAs on every SM, one kernel;
  - **ws**: one CTA per SM with w_A MMA warps + w_B stream warps;
  - **sw**: same-warp software-pipelined interleaving.

### Speed-up vs serial, best configuration of each kind (final runs)

| cell | T_A/T_B | serial µs | ideal max(T_A,T_B) | green | sm | cta | ws | sw | **best intra / best partition** |
|---|---|---|---|---|---|---|---|---|---|
| reg 0.5 | 0.51 | 499.3 | ×1.495 | ×1.471 | ×1.511 | ×1.511 | ×1.512 | ×1.477 | **1.000** |
| reg 1 | 1.02 | 662.2 | ×1.947 | ×1.479 | ×1.488 | ×1.540 | ×1.527 | ×1.419 | **1.035** |
| reg 2 | 1.98 | 976.5 | ×1.476 | ×1.272 | ×1.283 | ×1.313 | ×1.306 | ×1.243 | **1.023** |
| gemm 0.5 | 0.51 | 1263.1 | ×1.448 | ×1.435 | ×1.469 | ×1.470 | ×1.470 | ×1.366 | **1.001** |
| gemm 1 | 1.07 | 808.7 | ×1.815 | ×1.315 | ×1.340 | ×1.348 | ×1.350 | ×1.263 | **1.007** |
| gemm 2 | 2.12 | 624.0 | ×1.394 | ×1.153 | ×1.178 | ×1.167 | ×1.170 | ×1.094 | **0.993** |
| *power not binding for the partition:* | | | | | | | | | |
| reg 1, 94-SM sub-device | 1.04 | 690.8 | ×1.946 | ×1.427 | ×1.449 | ×1.853 | ×1.836 | ×1.682 | **1.279** |
| reg 1, 2 MMA warps/SM, 188 SMs | 0.93 | 643.8 | ×1.929 | ×1.736 | ×1.236¹ | ×1.941 | ×1.940 | ×0.894² | **1.118** |
| gemm 1, 94-SM sub-device³ | 0.99 | 1251.6 | ×1.981 | ×1.381 | ×1.445 | ×1.510 | ×1.562 | ×1.484 | **1.081** |

¹ The SM-level kernel keeps the A intensity at one 2-warp CTA per SM, so its B-role SMs stream with only 2 warps.
² Two warps per SM do both jobs: the stream is starved.
³ The GEMM alone on 94 SMs still draws 600 W at 2755 MHz, so the cap binds here too, less tightly.

### Best partition vs best intra-SM: clock, power, energy per iteration

The contention-free prediction is t/pred = t / max(T_A,solo · f_A,solo / f, T_B,solo). It uses the variant's measured clock and assumes A has all the SMs. A value near 1 means the variant only lost clock.

| cell | best partition | µs | MHz | W | mJ/it | best intra-SM | µs | MHz | W | mJ/it | intra t/pred | serial mJ/it |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reg 0.5 | sm_124 | 330.4 | 2636 | 601 | 198.7 | ws_4_4 | 330.2 | 2525 | 600 | 198.1 | 0.99 | 250.0 |
| reg 1 | sm_156 | 445.1 | 2427 | 600 | 267.0 | cta 2+2 (4-warp CTAs) | 429.9 | 2197 | 600 | 257.9 | 1.04 | 358.9 |
| reg 2 | sm_156 | 761.1 | 2543 | 600 | 456.7 | cta 2+1 | 744.0 | 2415 | 600 | 446.4 | 1.01 | 571.7 |
| gemm 0.5 | sm_124 | 859.7 | 2057 | 600 | 515.8 | ws 4+4 | 859.2 | 2007 | 600 | 515.5 | 0.99 | 672.4 |
| gemm 1 | sm_140 | 603.7 | 1780 | 600 | 362.2 | ws 2×4 + 8, setmaxnreg | 599.2 | 1609 | 600 | 359.4 | 1.09 | 470.9 |
| gemm 2 | sm_140 | 529.6 | 1882 | 600 | 317.7 | ws 2×4 + 8, setmaxnreg | 533.6 | 1782 | 600 | 320.0 | 1.07 | 374.5 |
| reg 1 / 94 SMs | sm_62 | 476.8 | 2807 | 522 | 249.0 | cta 2+1 | 372.8 | 2735 | 601 | 223.9 | 1.02 | 287.1 |
| reg 1 / low int. | green_160 | 370.8 | 2803 | 597 | 221.2 | cta 1+1 (2-warp CTAs) | 331.8 | 2707 | 600 | 199.2 | 0.99 | 265.2 |
| gemm 1 / 94 SMs | sm_62 | 865.9 | 2624 | 600 | 519.8 | ws 2×4 + 8, setmaxnreg | 801.1 | 2301 | 600 | 480.6 | 1.07 | 657.0 |

**Solo references (full GPU):**

| kernel | clock | power |
|---|---|---|
| register-only MMA | 2679 MHz | 600 W |
| GEMM-like | 1992 MHz | 600 W |
| stream | 2820 MHz | 435–441 W (the SMs mostly idle) |

**Repeatability.** The reg 1 cell was measured twice, with a coarser split grid the first time (`superseded` in the JSON): best intra / best partition was 1.041, and 1.035 in the rerun.

## Verdict (L4)

**Under the power cap, fine-grained intra-SM co-location never beats SM partitioning by ≥ 10%. The largest gap is 3.5%.**
- Register-only MMA: 1.000 / 1.035 / 1.023 at ratios 0.5 / 1 / 2.
- GEMM-like: 1.001 / 1.007 / 0.993.
- All six cells are power-capped: every co-located variant runs at 599–603 W. The NVML SwPowerCap reason is set in 81–91% of the samples of the final runs; reg 0.5 is the exception at 12%. These fractions cover the whole run, including the unthrottled stream-alone slices.
- Both approaches keep most of the gain over serial: ×1.15–1.51.

**Without the cap binding for the partition, it does: 1.118 to 1.279.**
- Register family, 94-SM sub-device: 1.279. The partition draws only 514–522 W and clocks at 2800 MHz; its A loses the SMs the stream needs.
- 2-warp MMA on the full GPU: 1.118. Intra-SM reaches the ideal overlap (t/pred 0.99).
- GEMM on 94 SMs: 1.081, with the cap still partly binding.

**Which resource explains it: power first, then in-SM contention.**

1. **Power (dominant).** Every co-located variant runs at the cap, so time ≈ energy per iteration / 600 W. Energy per iteration differs by only 0–3.5% between the best partition and the best intra-SM variant (e.g. reg 1: 267.0 vs 257.9 mJ; gemm 1: 362.2 vs 359.4 mJ). Both overlap the stream's largely idle-SM power (serial: 358.9 mJ) the same way.
   - A partition hands the MMA fewer SMs, but the power budget then drives those SMs to a higher clock. In reg 1 the partition's MMA runs on 156 SMs at 2427 MHz; the intra-SM MMA runs on 188 SMs at 2197 MHz.
   - In the green partitions, the MMA finishes within 0–3% of "solo time × 188/n_A × f_solo/f" (per-op completion times, final runs). The exception is gemm 0.5 at +8–13%: there the long stream-only tail raises the iteration-average clock used in the prediction. So there is no cross-partition interference, and the partition loses only SM count, which the higher clock partly buys back.
   - What intra-SM co-location has left is a better V/f point: all SMs at a lower clock, about 9% more SM·MHz in reg 1. In-SM contention eats most of it.
2. **In-SM contention: issue slots / tensor pipe (negligible), LSU / smem / L2 path (moderate).**
   - For the register-only MMA, intra-SM variants land at t/pred 0.99–1.04. Stream warps do not slow the tensor pipe: an mma issues every 16 cycles per warp, so issue slots are almost idle.
   - For the GEMM-like kernel, the best intra-SM variants land at 1.07–1.09 (up to 1.17 for other layouts). Its operand path (cp.async from L2, ldmatrix) competes with the stream warps' loads. Evidence: a warp-specialised CTA with 8 GEMM warps runs 8% slower with 8 stream warps than with 4, even though its clock is higher (2171 vs 1807 MHz, gemm 1).
3. **Same-warp interleaving is the worst intra-SM design in every cell** (t/pred 1.01–1.12 for reg, 1.06–1.16 for gemm; ×0.894 at low intensity).
   - Each warp issues in order. When the oldest streamed packet has not arrived, the warp stops issuing MMA too. Under DRAM saturation the loaded latency is 1–2 µs.
   - The prefetch depth is limited by registers (a 16-packet ring needs 168 registers) or by the smem slot. Deeper rings did not help.
   - Warp specialisation and CTA co-residence decouple these stalls from the MMA issue.
4. **Register file (small).** It matters only for GEMM warp specialisation; see setmaxnreg below. It is worth 2–3% under the cap and does not change the ceiling.

**Implication for the proposal's new direction (tile-step orchestration inside one kernel).**
- On this GPU, with MMA-heavy + DRAM-streaming pairs under the 600 W cap, even an ideal intra-SM design gains ≤ 3.5% over a well-chosen SM partition.
- The earlier CoKernel designs are therefore not limited by their design here. They are limited by the power cap.
- Intra-SM orchestration pays off (≥ 10%) only when the MMA side is SM-bound rather than power-bound: an under-occupying op, or a partial-GPU / low-power setting.
- Same-warp step interleaving is the least attractive variant. It needs deep asynchronous prefetch (TMA/cp.async into smem, consumed late) to avoid stalling the MMA.

## setmaxnreg on sm_120

**Compilation.**
- `setmaxnreg.inc/dec.sync.aligned` is rejected by ptxas for `sm_120` ("not supported on .target 'sm_120'").
- It is accepted for `sm_120a` (NVRTC 12.8). The cubin runs on the card, and the SASS contains `USETMAXREG.TRY_ALLOC.CTAPOOL` / `USETMAXREG.DEALLOC.CTAPOOL`: registers are pooled per CTA.
- ptxas allocates each branch with its own limit only if the whole role code is dominated by its `setmaxnreg`. A shared tail after the role branches forced the kernel-wide cap and spilled 1.3 KB.

**Measured configuration** (test_16): 2 GEMM groups × 4 warps at 232 registers + 4 stream warps at 40.
- The kernel launches at 168 registers/thread, and the MMA path uses registers up to R215 with 0 spills.
- The same kernel with a uniform 168-register cap spills 112 B.
- Without rebalancing, this layout (2 × 4 warps of 64×64-tile GEMM + stream warps) cannot fit the register file at all.

**Performance.**
- Rebalanced warp specialisation (2×4 + 8 stream warps, 200/40 registers) is the best warp-specialised design for GEMM at ratios 1 and 2 and on 94 SMs: +2.3%, +2.8% and +7.3% over the best uniform-register variant (screening).
- It ties with CTA co-residence of 224-register CTAs on the full GPU (599.2 vs 600.1 µs, gemm 1) and beats it by 3.5% on 94 SMs.
- At ratio 0.5 it is 0.6% slower than the best uniform variant.
- It does not change the verdict.

## Caveats

- **Synthetic kernels.**
  - The GEMM-like kernel reaches 80% of peak per clock, about 90% of cuBLAS. Its operands stay in L2; a real GEMM also streams them from DRAM.
  - The stream only sums the data.
  - Real op pairs have more coupling (smem, registers per tile) than these idealised units. That makes the intra-SM ceiling measured here an upper bound.
- **Screening vs final.** Screening slices (0.5 s) showed clocks about 3% higher than the final slices (1.2 s) because of the power controller's transient. Only final numbers are reported. Within a kind, the finalists differ by ≤ 1%.
- **Clock probe.** The ClockProbe's resident warp shares one SM. Configurations that use the whole register file (e.g. the 16-warp setmaxnreg variant) then run one CTA late on that SM. Its dynamic queue absorbs this: ≤ 0.5% effect.
- **Reference stream.** The stream-alone configuration (8 warps × 2 CTAs, 1608 GB/s) is about 1% slower than the stream inside some co-located variants. That is why t/pred can be 0.985–0.99.
- **Nested green contexts.** Driver 580 refuses to re-split a split result (CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION). `stress.split_ranges` therefore recombines 2-SM groups into one descriptor. The resulting SM ranges were verified with the %smid probe: [0, 48) and [48, 94), disjoint.
