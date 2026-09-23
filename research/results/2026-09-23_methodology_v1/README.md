# Measurement methodology v1 (M1–M5)

**Setup**
- Date: 2026-09-23.
- Machine: RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs, 600 W cap, clocks not lockable), driver 580.173.02, torch 2.8.0+cu128.
- TileLang source tree at HEAD `a3b74ff8` with uncommitted changes (this work).
- Foreign processes on the GPU:
  - user qzr's idle retriever (PID 354072) throughout;
  - a ray training worker (PID 682501), idle during the measurements below.
  - Neither showed SM% > 0 during any reported window (pmon guard record in every JSON).
  - The ray worker became active later (≈90% SM at 600 W, 94 GB). The guard held the post-fix re-runs for 46 min and re-measured one contaminated steady run.

**Code** (all under `research/bench/`):
- `cobench/`: `timing.py`, `steady.py`, `variants.py`, `guard.py`, `clock.py`, `kernels.py`, `cudrv.py`.
- `scripts/`: `mv1_flush.py` (M1), `mv1_steady.py` (M3), `mv1_static.py` + `sched_variants.py` (M4), `mv1_common.py`.
- `tests/test_cobench.py`: tests 8–12 are new.
- Changes outside `research/bench`:
  - `cotile/tests/harness.py`: `wait_for_gpu` now uses the guard.
  - `cotile/cokernel.py`: the dispatch slot is padded to 128 B (§M4.5).

**Files in this directory**

| file | content |
|---|---|
| `flush.json` | M1: eviction and cleanliness microbenchmarks; 4 ops × 5 modes × 2 passes; flush cost and floor |
| `steady_p1.json` | M3: P1 smoke test, steady mode and clean-flush `bench_corun` (after the §M4.5 fix) |
| `steady_stream_priority.json` | M3.5: do stream priorities make the two-stream order deterministic? |
| `steady_p1_before_slot_fix.json` | the same, before the fix (misaligned CoKernel) |
| `steady_validate.json`, `steady_repro.json` | M3 validation (a) (non-power-capped pair) and (b) (3 processes) |
| `static_penalty.json` | M4: 3 shapes × {clean flush with fixed / old / no probe, steady} × 9 builds, plus per-tile traces. `flush_fixed_probe` and `steady` are the post-fix re-run (01:34). `flush_old_probe`, `flush_no_probe` and the traces are from the first run (00:32), so their `co_dyn` rows predate the §M4.5 fix |
| `test_output.json`, `repro_matmul_graph.json` | `test_cobench.py` report (tests 1–12, all PASS) and its 5-process graph-mode matmul repeatability run (test_6) |

## Summary

| criterion | result | evidence |
|---|---|---|
| **M1** clean flush | **pass**, with a correction: write-then-read is *not* clean. The default flush is now write + `discard.global.L2` | §M1 |
| **M2** GPU guard | **pass** | test_9: own process and children ignored; a detached foreign GEMM process detected (SM up to 99%); `wait_until_free` returned 8 s after it started, once it had stopped; cotile `wait_for_gpu` uses it. In production it held this study's re-runs for 46 min while the ray worker trained, and re-measured one contaminated steady run |
| **M3** steady mode | **pass** | slice CV ≤ 0.6% after the thermal plateau; 3-process CV ≤ 0.28% (times), ≤ 0.21% (speed-ups). Serial vs sum of solos: 0.988 (non-capped pairs), 0.958 (P1 pair, power averaging) |
| **M4** static penalty | **root cause found: a measurement artefact** | cobench's ClockProbe left its SM with a small smem carveout, so every large-smem kernel ran on 187 SMs. A static grid-stride persistent kernel then has one CTA that starts only after another exits. Fixed: persistent/grid = 1.003–1.017, was 1.26–1.76. Nsight Compute was not usable (`ERR_NVGPUCTRPERM`) |
| **M4** extra finding | CoKernel role scratch misaligned by 16 B → 1.5–1.7× slower role bodies; **fixed** (co_dyn / grid 0.999–1.006) | §M4.5 |
| **M5** docs | done | `research/bench/README.md` (modes, flush kinds, steady mode, guard, probe carveout, P1 recommendation); this README |

## M1 — clean flush

**Mechanism.** Each rep: previous rep → flush on a side stream → gate → timed op. Buffers are read with 16 B loads (`read_u4`) or written with `fill_`.

| flush kind | 32 MB buffer read 3× just before the flush, then timed read | 64 MB write-only kernel after the flush | flush cost (µs) |
|---|---|---|---|
| none (hot) | 8.2 µs | 14.5 µs (hot mode) | – |
| **clean** = write 256 MB + `discard.global.L2` | **24.16 µs** | **16.4 µs** | **136** |
| write (v0) | 26.21 µs | 38.9 µs | 164 |
| write + read (plan v0.3 wording) | 26.21 µs | 26.2 µs | 279 |
| read | 24.16 µs | 16.4 µs | 165 |
| never-touched buffer (rotation > 2×L2) | 24.16 µs | – | – |

For a 96 MB buffer the reads take 65.12 µs after clean, 65.12 µs for a never-touched buffer, and 69.2 µs after write or write+read.

- `clean` evicts completely: even a buffer read 3× just before the flush reads exactly like a never-touched one.
- `clean` leaves no dirty lines: a 64 MB write is absorbed by the L2 (16.4 µs vs 38.9 µs after the dirty write flush).
- `write+read` is not enough: reads that hit still-resident dirty lines do not clean them, so about half the L2 stays dirty.
- `write`/`write+read` make even a pure 32 MB read pay write-backs (+8%).

**Ops.**
- Solo-best configs: RMSNorm `r2_t128_v2`; decode `n128_h4_sp1_t128_s1`; matmul = cuBLAS via torch.
- Median µs of 2 passes in opposite orders; clock in MHz in parentheses.

| op | flush-write | flush-write+read | **flush-clean** | graph (rotation, back-to-back) | hot |
|---|---|---|---|---|---|
| matmul 4096³ (torch) | 354.5 / 362.2 (2388/2346) | 354.0 / 358.1 | **351.3 / 355.9 (2453/2437)** | 386.2 / 393.6 (2153/2112) | 366.0 / 371.6 (2263/2231) |
| copy 512 MB → 512 MB | 754.5 | 737.3 | **691.1** | 736.6 | 736.6 (not L2-resident) |
| RMSNorm 4096×4096 | 43.3 | 34.6 | **28.2** | 44.3 | 11.0 |
| GQA decode B16×S8192 | 342.0 | 341.8 | **330.0** | 328.1 | 328.1 (not resident) |

(Pass 2 was hotter: 58 → 75 °C; the power-capped matmul drifted 1–2%.)

**Reading.**
- **The dirty-line bias is gone.**
  - A read-dominated op now matches back-to-back execution: decode 330.0 µs vs 328.1 µs (+0.6%). With the write flush it was 342.0 µs (+4.2%).
  - The ops no longer pay for the flush's dirty lines: copy 754 → 691 µs, RMSNorm 43.3 → 28.2 µs, matmul −0.9…−1.8%.
- **What remains is by design.**
  - The op's *own* writes, up to about the L2 size, are absorbed and written back after the end event. Clean-flush times of write-heavy ops with outputs ≤ L2 are therefore *below* their steady-state cost: RMSNorm 4096² 28.2 µs vs 44.3 µs back-to-back; copy 691 µs vs 737 µs.
  - Power-capped ops run at a higher clock in any flush mode: matmul 2440–2450 MHz vs 2110–2150 MHz back-to-back, 9–10% faster in time, but 3.8% *more* cycles, because the cold start is DRAM-bound.
  - Flush-mode numbers are one-shot latencies, not throughputs.
- **Floor and duty cycle.**
  - An empty kernel reads 3.9 µs in flush mode (bimodal 2.0 / 4.1 µs) vs 0.63 µs back-to-back.
  - The clean flush adds about 140 µs of ~350 W phase per rep. The matmul duty cycle in flush mode is 69% (rep period 508 µs).
  - The clean flush is the cheapest kind: the discard avoids writing back the buffer's own lines.

## M2 — GPU guard (`cobench.guard`)

**Design.**
- `GpuGuard` streams `nvidia-smi pmon -s u -d 1`.
- Foreign = not this process, its ancestors or its descendants. A test script and its measurement subprocesses (e.g. `mv1_steady.py repro`) therefore do not block each other.
- It keeps samples with SM% > 0 and the host time up to which pmon has reported.
- API:
  - `wait_until_free(quiet_s=5, poll_s=150, max_wait_s)`;
  - `foreign_activity(t0, t1)` (waits for pmon coverage);
  - `busy()`.

**In the benches.** `bench`, `bench_corun`, `bench_variants` and `bench_steady`:
1. wait for a free GPU;
2. after the timed window, keep the GPU loaded with untimed tail reps until pmon has covered the window;
3. re-measure (up to 3×) if a foreign process was active.

The record is in `result.guard`. `cotile/tests/harness.py: wait_for_gpu(max_wait_s=900, poll_s=150) -> bool` keeps its signature and now delegates to the guard.

**Tested** (test_9).
- 3 s of in-process GEMMs and a child process's GEMMs are not flagged.
- A double-forked (non-descendant) GEMM process is flagged (SM up to 99%).
- `wait_until_free` returns after that process ends (waited 8 s with `poll_s=2`).
- `harness.wait_for_gpu` returns True on a free GPU.
- `bench` and `bench_corun` results carry a clean, complete guard record.

**In production.** The ray worker started while the post-fix re-runs were queued. The guard logged the activity and blocked the runs instead of measuring on a shared GPU.

## M3 — steady-state co-run mode (`cobench.bench_steady`)

### M3.1 Design

- Variants are callables that enqueue one iteration on the launcher stream (`Par` fork/join for multi-stream variants; `Rotation` of input and output copies > 2×L2).
- They run back-to-back in slices of 1.5 s; the first 0.5 s of each slice is excluded.
- Slices are interleaved in rotating order for 5 rounds, after a warmup that lasts until the temperature plateaus.
- Measured per slice: SM clock from the ClockProbe, board power from the NVML 20 ms samples, temperature.
- Host-gap detection runs on every enqueue; there were 0 gaps in every run.
- Details and API: `research/bench/README.md`.

**Why the thermal plateau warmup.**
- A first run with a 5 s warmup drifted over its 2 min: solo GEMM 403 → 424 µs, clock 2405 → 2290 MHz at a constant 600 W; slice CV 2.1%.
- After the plateau warmup (96 s, 82 °C) the slice CV is ≤ 0.6%, except for the bimodal streams variant (§M3.4).

### M3.2 Validation

**(a) Serial vs sum of back-to-back solos.**

| pair | serial | solo_a + solo_b | ratio | notes |
|---|---|---|---|---|
| decode B16×8192 + RMSNorm 16384×4096 (both ~425 W, max clock) | 505.5 | 511.6 | **0.988** | slice CV 0.00–0.09% |
| copy 128 MB + read 256 MB (test_10) | 349.5 | 353.8 | 0.988 | |
| GEMM 4096³ + decode B16×8192 (P1; post-fix run: 725.2 / 759.8 = 0.954) | 723.9 | 755.9 | **0.958** | GEMM clock 2277 MHz solo vs a serial average of 2659 MHz: the power controller averages over the 574 W serial mix |

The 1.2% offset of the non-capped pairs is systematic (slice CV ≤ 0.1%) but unexplained. The likely cause is L2 and DRAM state carried from the previous kernel, which differs between "after itself" and "after the partner". Conclusion: co-run speed-ups must be taken against a `serial` variant of the same run (as `bench_steady` and `bench_corun` do), never against summed solo times.

**(b) Repeatability across 3 processes** (serial, solo_a, solo_b, streams, green_100, co_to_128; 5 rounds each; run before the §M4.5 fix, which does not matter for repeatability).

| | range across processes |
|---|---|
| CV of t_iter | 0.007% (solo_b) … 0.28% (solo_a) |
| CV of speed-ups | ≤ 0.21% |
| CV of clocks | ≤ 0.5% |

Pass (< 2%).

### M3.3 P1 smoke test: GEMM 4096³ × GQA decode B16×S8192 (solo-best configs)

**Configs.**
- GEMM `128x256x64_s2_t256_direct` (grid, 512 CTAs, 96 KB smem).
- Decode `n128_h4_sp1_t128_s1` (grid, 128 CTAs, 74 KB smem).
- Green splits use `IGNORE_SM_COSCHEDULING` (A on SMs [0, n_A)).
- CoKernel: SM binding (n_A SMs → GEMM), dynamic chunk 1, `to` = takeover, `timing=False`, 222–226 regs, 98 432 B smem, 1 CTA/SM. Measured **after the §M4.5 slot fix**; the pre-fix numbers (1.5–1.7× slower role bodies) are in `steady_p1_before_slot_fix.json`.

**Protocol.**
- Steady: thermal plateau reached after 64 s at 86 °C, then 5 rounds × 1.5 s slices; the guard was clean.
- Flush: clean flush, `bench_corun` with every other variant as `extra`, 60 reps, all interleaved.
- Both runs were in the same process, one after the other.

| variant | steady µs/iter | × vs serial | MHz | W | clean-flush µs | × vs serial | MHz |
|---|---|---|---|---|---|---|---|
| serial | 725.2 | 1.000 | 2655 | 581 | 730.1 | 1.000 | 2673 |
| solo_a (GEMM) | 429.3 | – | 2260 | 600 | 389.0 | – | 2550 |
| solo_b (decode) | 330.5 | – | 2821 | 441 | 331.7 | – | 2795 |
| streams (GEMM-first mode) | 642.7 | 1.128 | 2487 | 601 | 642.7 | 1.136 | 2596 |
| green 60/128 | 1188.4 | 0.610 | 2810 | 458 | 1182.3 | 0.618 | 2797 |
| green 80/108 | 933.1 | 0.777 | 2783 | 533 | 924.7 | 0.790 | 2790 |
| green 100/88 | 797.1 | 0.910 | 2720 | 574 | 789.0 | 0.925 | 2747 |
| green 120/68 | 691.1 | 1.049 | 2639 | 600 | 682.0 | 1.071 | 2677 |
| **green 140/48** | **594.3** | **1.220** | 2214 | 600 | **552.5** | **1.322** | 2469 |
| CoKernel+to 60 | 718.3 | 1.010 | 2671 | 599 | 704.1 | 1.037 | 2694 |
| CoKernel+to 94 | 642.0 | 1.130 | 2457 | 600 | 598.8 | 1.219 | 2645 |
| CoKernel+to 128 | 603.5 | 1.202 | 2275 | 600 | 561.7 | 1.300 | 2553 |
| **CoKernel+to 160** | **602.2** | **1.204** | 2116 | 600 | **553.0** | **1.320** | 2352 |
| CoKernel 94 / 128 (no takeover) | 797.5 / 602.8 | 0.909 / 1.203 | 2751 / 2289 | 587 / 600 | 787.5 / 565.7 | 0.927 / 1.291 | 2760 / 2588 |

**Per-op completion (steady).**
- streams: A 456.7 µs, B 640.2 µs. The decode is delayed until the GEMM's CTAs drain, because 96 + 74 KB of smem cannot share an SM.
- green 140/48: A 591.6 µs, B 468.3 µs.

**Before the slot fix** (same run structure, `steady_p1_before_slot_fix.json`): the CoKernel variants ran at 0.75–0.86× serial in steady mode (e.g. co_to_128 846.2 µs). Everything else agreed with the table above within ±1% (serial 723.9, green 140 588.5).

**Result.**
- At the solo-best configs, the best inter-kernel variant (green 140/48, ×1.22) and the best CoKernel (SM split 128–160, ×1.20) are within 1.5% of each other in steady mode, and equal in flush mode (×1.32).
- Both run at the 600 W cap.
- This is a methodology smoke test, not the 3×2 study.

### M3.4 Which mode shows the larger benefit, and why

**Flush mode shows the larger co-location benefit whenever the variant is power-capped in steady state.**

| variant | speed-up, steady → flush | clock, steady → flush |
|---|---|---|
| green 140/48 | ×1.220 → ×1.322 (+8.4%) | 2214 → 2469 MHz (+11.5%) |
| CoKernel+to 160 | ×1.204 → ×1.320 (+9.6%) | 2116 → 2352 MHz (+11%) |
| CoKernel+to 128 | ×1.202 → ×1.300 (+8.2%) | – |

- The serial reference barely changes (2655 vs 2673 MHz): its GEMM/decode mix averages 581 W, below the cap, even in steady state.
- The mechanism: in flush mode every rep carries a ~140 µs low-power (~350 W) flush phase. The power controller averages over it, so a variant that draws the full 600 W in steady state gets about 11% more clock during the timed op. The serial reference was never capped and gains nothing, so the ratio inflates.
- Solo GEMM shows the same: 389 µs in flush mode vs 429 µs in steady mode.
- Variants below the cap agree between the modes within 1–2% (green 60–100/…, co_94).

**Consequence for the study.** The steady mode reflects sustained serving, where the power wall binds (proposal §9). The flush mode overstates co-location gains by 8–10% at this point, and it can reorder variants: in flush mode CoKernel+to 160 ties green 140; in steady mode it trails by 1.3%.

**Two-stream co-runs are not deterministic.**
- A rep runs in one of several modes:
  - GEMM-first: A ends at ~460 µs, B at ~633–640 µs, makespan ~640 µs;
  - decode-first: B ends at ~368 µs, A at ~678 µs, makespan ~680 µs;
  - in one run an intermediate mode: B 501 µs, A 572 µs.
- The hardware scheduler decides which. The host launch order does not: the `ab` and `ba` variants swapped modes between runs, and single steady slices of `streams_ba` switched mode (707 vs 643 µs, slice CV 4.3%).
- M1 baselines need per-op completion times, and either the order forced (stream priorities, see §M3.5) or both modes reported.

### M3.5 Stream priorities do not pin the two-stream order

Same pair. A high-priority stream means `torch.cuda.Stream(priority=-1)`. Per-op completion times in µs are in parentheses (A = GEMM, B = decode). Source: `steady_stream_priority.json`.

| variant | steady µs (× serial) | clean flush µs (× serial) |
|---|---|---|
| serial | 726.0 | 731.3 |
| equal priority, host order A,B | 644.8 ×1.126 (A 458, B 642) | 639.9 ×1.143 (A 457, B 637; p90 681) |
| equal priority, host order B,A | 644.5 ×1.127 (same mode) | 639.6 ×1.143 |
| A high priority | 644.6 ×1.126 (A 458, B 642) | 681.8 ×1.073 (**A 679, B 366**) |
| B high priority | 726.1 ×1.000 (A 724, B 535) | 682.1 ×1.072 (A 680, B 366) |

- Priorities change which mode occurs, but not in a way that holds across the two modes: "A high priority" gives GEMM-first in steady mode and decode-first in flush mode.
- "B high priority" in steady mode is as slow as serial.
- **Recommendation for M1.** Take the best of {equal priority ×2 host orders, priority A, priority B}, measured in the same mode as the study. Report which mode it ran in, using per-op completion times.

## M4 — the "static persistent penalty"

### M4.1 Root cause: the ClockProbe blocked one SM for large-smem kernels

**What happens.**
- cobench's ClockProbe is one resident warp that logs `clock64` for the whole timed window.
- With the driver's default carveout preference, the probe's SM was configured with a small shared-memory carveout in 7 of 8 trials. An SM can change its carveout only while it is idle.
- Therefore, for as long as the probe ran, that SM could not host any CTA needing more smem (tested: 96 KB).
- Every large-smem kernel measured with `clock=True`, i.e. all of P1-S, ran on 187 SMs:
  - grid kernels were mostly unaffected (the hardware scheduler redistributes blocks);
  - a static grid-stride persistent grid of 188 CTAs has exactly one CTA that cannot start until another CTA exits, and its fixed tiles then run as a serial tail;
  - a dynamic queue absorbs the missing SM (the late CTA finds the queue empty).

**Evidence** (per-tile traces, GEMM 4096³, `static_penalty.json`).
- With the old probe, the static kernel uses 187 distinct SMs.
- CTA 187 starts at 259 µs, on SM 92, after that SM's CTA finished its 2 tiles. It then runs its own 2 tiles: makespan 488 µs vs 388 µs.
- With the dynamic queue, CTA 187 starts at 265 µs and gets 0 tiles: makespan 384 µs.

**Fix.** `CudaKernel.set_carveout(100)` on the probe and on the host gate. With it, the traces show 188 SMs and every CTA starting within 0.1 µs. Regression test: test_11 (a 96 KB-smem grid uses all 188 SMs while the probe runs).

**Before/after** (t_persistent / t_grid, clean flush, 50 reps, all builds interleaved; `static_penalty.json`). The old-probe and no-probe columns come from the first run, the fixed-probe and steady columns from the post-fix run:

| shape (config, tiles / CTAs) | P1-S catalog persistent / grid | old probe (carveout 0) | **fixed probe** | no probe | steady (fixed probe) |
|---|---|---|---|---|---|
| GEMM 4096³ (`128x256_t256_direct`, 512 / 188) | 1.259 | 1.264 | **1.010** | 1.011 | 1.010 |
| GEMM 8192×14336×4096 (`256x128_t256_direct`, 3584 / 188) | 1.586 | 1.706 | **1.003** | 1.004 | 1.000 |
| decode B64×S8192 (`n64_h1_sp4_t128_s2`, 8192 / 188) | 1.724 | 1.763 | **1.016** | 1.017 | 1.008 |

The penalty grows with the number of tiles per CTA, because the late CTA runs a whole CTA's share after the earliest exit. That matches P1-S (GEMM 1.2–1.85, looping decode up to 1.82, non-looping decode 1.00).

### M4.2 What remains: static vs dynamic with a correct instrument

Post-fix run, fixed probe (`static_penalty.json`, `flush_fixed_probe` / `steady`).

| shape | static / grid: flush, steady | dynamic (minimal `sched_variants`) / grid: flush, steady | CoKernel dynamic, partner 0 tiles / grid: flush, steady |
|---|---|---|---|
| GEMM 4096³ | 1.008, 1.010 | 0.996, 0.991 | 1.005, 0.999 |
| GEMM 8192 | 1.003, 1.000 | 1.002, 1.002 | 1.003, 1.003 |
| decode64 | 1.017, 1.008 | 1.003, 0.999 | 1.006, 1.003 |

Static rows are the op library's `build_persistent`; `sched_variants` "static" agrees within 0.2%.
- Static fixed assignment is 0–1.7% slower than grid or dynamic. The traces show why: per-CTA busy time spreads by about 2%, and a fixed assignment cannot rebalance it.
- The dynamic queue, the minimal build as well as the CoKernel after §M4.5, matches the hardware grid scheduler.

### M4.3 Hypotheses tested and ruled out (fixed probe, clean flush)

Values are t / t_grid for GEMM 4096 / GEMM 8192 / decode64.

| hypothesis | variant | result | verdict |
|---|---|---|---|
| TPC pairing: consecutive ranks share a TPC (blocks 2k and 2k+1 go to SMs 2t and 2t+1) | `static_perm`: ranks of TPC neighbours 94 apart | 1.007 / 1.007 / 1.001 | ruled out (same as static) |
| per-tile timing jitter (the dynamic queue's atomic desynchronizes CTAs) | `static_atomic`: static order + 1 global atomic per tile | 1.013 / 1.004 / 1.020 | ruled out: the atomic does not help |
| dispatch structure (smem slot + barrier) | `static_slot` | 1.007 / 1.005 / 1.017 | ruled out |
| tile order / rasterization interacting with the fixed mapping | `chunked`: a contiguous tile range per CTA | 0.981 / 1.064 / 3.95 | real, but a different effect: tiles in flight are spread over the tile space. That helps the small GEMM slightly (flush mode), hurts the big GEMM (6–7%, L2 reuse of A/B panels) and destroys decode64 (×3.95: the 4 head blocks sharing each K/V chunk no longer run concurrently, so K/V is re-read from DRAM) |
| the flush or the L2 state | all modes (clean flush, steady, and P1-S hot/graph) | the penalty followed the probe, not the L2 state | ruled out |
| codegen | SASS: same main loop (P1-S); sched variants differ only in dispatch | – | ruled out |

**Nsight Compute.**
- ncu fails with `ERR_NVGPUCTRPERM`: the driver has `RmProfilingAdminOnly=1` and there is no sudo. DRAM/L2/stall counters are unavailable.
- Software instrumentation replaced it: shared-memory-buffered `%globaltimer` per tile, written out at CTA exit. Trace builds time within 0.5% of untraced builds.
- The root cause needed none of the requested counters.

### M4.4 Consequences for existing data

- **P1-S catalog (`2026-09-22_solo_profile`).**
  - Every *persistent* time is inflated wherever the kernel's smem exceeded the probe SM's carveout, which covers all GEMM configs and most decode configs. S3 ("persistent 1.52× median") and `c_lib_persistent` should not be used; persistent ≈ grid (+0–1.5%).
  - Grid times at 188 SMs are essentially unaffected: 17 of 1026 grid points lie on a 187-vs-188-SM wave boundary.
  - Green-budget points (@94, @48) may have lost one SM wherever the probe sat inside the partition. The probe usually ran on SM 0, and the IGNORE_SM_COSCHEDULING partitions are [0, n). The affected points are large-smem kernels at @94/@48, i.e. the GEMM budget curves, where one SM is 1–2% of the partition. Worth re-measuring before relying on the GOLDYLOC numbers.
- **CoKernel README (`2026-09-22_cokernel`).** The "static 20–30% slower than dynamic in flush mode, vanishes in debug builds / with a warm L2" observation is this artefact: those comparisons ran with the probe. The debug and warm runs did not.
- **Co-location in general.** The same carveout rule applies between *real* kernels. On streams, an SM whose carveout was configured by kernel X cannot host kernel Y's CTAs if Y needs more smem, until the SM idles. This is a real effect for M1 baselines, not an artefact. Carveout preferences are a knob (or a confound) of inter-kernel co-location. The CoKernel (one kernel) is immune.

### M4.5 Extra finding: misaligned CoKernel role scratch (fixed)

**Symptom.** In M4, the CoKernel partner-0 variant (all SMs → role A, dynamic chunk 1, `timing=False`) ran at 0.60–0.67× grid for all three shapes and in all modes (GEMM 4096: 635 vs 381 µs). The minimal dynamic persistent build (`sched_variants`) matched grid.
- Its tiles were uniformly slower (debug timeline: 190–220 µs vs 115–130 µs per tile).
- The partner (decode, GEMM or RMSNorm) and the schedule (static or dynamic) made no difference.
- There were no spills, and the GEMM inner loop was identical in SASS (334 vs 331 instructions, same HMMA/LDGSTS/LDSM/BAR counts).

**Cause.**
- TileLang's shared-memory merge planner forces 1 KB alignment only for TMA/wgmma operands. Every other buffer gets 16 B alignment.
- The CoKernel's 16 B dispatch slot landed at offset 0 and the role scratch right after it, so GEMM A_s sat at byte 16 and B_s at 32784. The code comments assumed the scratch started at 1024, which holds only when a TMA-store epilogue forces it.
- Swizzled cp.async/ldmatrix buffers whose 128 B rows start 16 B off a 128 B boundary run about 1.7× slower. The exact micro-architectural mechanism is not identified: bank arithmetic on the logical swizzle predicts no conflict. Causality is shown directly:

| role scratch offset | 16 B | 128 B | 256 B | 512 B | 1024 B |
|---|---|---|---|---|---|
| CoKernel GEMM 4096 / grid | 0.592 | 0.993 | 0.993 | 0.993 | 0.993 |

**Fix.**
- `cotile/cokernel.py` now pads the slot to 128 B (`SLOT_INTS = 32`, +112 B smem only where the scratch was misaligned).
- `python -m cotile.tests.test_cokernel`: 184/184 launchable settings pass; the 3 `smem="sum"` GEMM×decode builds are not launchable, as before.
- Recommended root fix in TileLang: align every merged shared buffer used by `ptx_ldmatrix`/`cp.async` (or every buffer ≥ 128 B) to ≥ 128 B in `MergeSharedMemoryAllocations`. It was not done here: it touches core C++, and the kernel cache is keyed by git commit, not pass code, so stale cached kernels would mix with new ones.

**Re-measurement after the fix** (`static_penalty.json`, `co_dyn` = CoKernel with the partner compiled in and given no SMs, dynamic chunk 1):

| shape | co_dyn / grid before: flush, steady | co_dyn / grid after: flush, steady |
|---|---|---|
| GEMM 4096³ | 1.667, 1.533 | **1.005, 0.999** |
| GEMM 8192×14336×4096 | 1.519, 1.484 | **1.003, 1.003** |
| decode B64×S8192 (partner GEMM) | 1.540, 1.577 | **1.006, 1.003** |

The CoKernel's dynamic dispatch is now as fast as the hardware grid scheduler. The P1 CoKernel variants went from 0.75–0.86× serial to 1.20× serial (§M3.3).

The guard re-measured part of this re-run on its own: the ray worker restarted during the GEMM 4096 steady slices; the guard detected it, waited and repeated the measurement.

## Recommendations for the 3×2 study (P1)

1. **Primary metric: steady-mode time per iteration** (`bench_steady`).
   - Speed-ups against the `serial` variant of the same run: report the ratio of medians and the paired per-round ratios.
   - Use the thermal plateau warmup, 1.5 s slices with 0.5 s settle, and ≥ 5 rounds.
   - Report clock and power for every variant.
   - Keep clean-flush `bench_corun`/`bench_variants` as the auxiliary one-shot-latency view. Flag its power-headroom inflation (+8–10% on the speed-ups of power-capped variants here).
2. **Never compute speed-ups from summed solo times**, in either mode (1–4% bias).
3. **Clock instrumentation.**
   - Keep the fixed ClockProbe (carveout 100).
   - Any other resident helper kernel must set its carveout too.
   - Record the probe's SM; it may sit inside a green partition.
4. **Re-measure what P1-S's probe touched** before using it:
   - persistent solo times (for the intra column, use grid ≈ persistent until re-measured);
   - @94/@48 budget points of large-smem configs.
5. **Two-stream baselines are multimodal, and stream priorities do not pin them (§M3.5).**
   - Report per-op completion times (Par marks).
   - Take M1 as the best of {both host orders, priority A, priority B} in the same mode as the study.
6. **Carveout as a knob.** For inter-kernel co-location of kernels with different smem footprints, the SM carveout decides whether their CTAs can share an SM. Evaluate M1 both with the kernels' default carveout preferences and with carveout = 100 for both.
7. **CoKernel role scratch alignment.** After the slot fix, check `smem_view_offsets` (every role buffer offset ≡ 0 mod 128) for every CoKernel in the study. Better, land the planner fix.
8. **Tile order matters for sharing ops** (chunked decode64 ×3.95). Keep the grouped/consecutive tile order for dynamic queues; do not assign contiguous tile ranges per CTA for split-head decode.
9. **The solo-best decode config is a poor co-location citizen.** B16×8192 with `sp1` has 128 tiles of 330 µs, so SM-level CoKernel splits leave long tails. The lib/derived columns need split-KV variants.
10. **Nsight Compute counters are unavailable** without admin rights. Plan on software instrumentation (tile traces, the clock probe) or ask the machine owner to set `NVreg_RestrictProfilingToAdminUsers=0`.

## Reproduce

```bash
source research/env.sh
python research/bench/scripts/mv1_flush.py                      # M1   -> flush.json          (~4 min)
python research/bench/scripts/mv1_steady.py p1                  # M3c  -> steady_p1.json      (~6 min)
python research/bench/scripts/mv1_steady.py validate            # M3a  -> steady_validate.json
python research/bench/scripts/mv1_steady.py repro --procs 3     # M3b  -> steady_repro.json   (~6 min)
python research/bench/scripts/mv1_steady.py prio                # M3.5 -> steady_stream_priority.json
python research/bench/scripts/mv1_static.py [--parts 1234]      # M4   -> static_penalty.json (~6 min)
python research/bench/tests/test_cobench.py                     # tests 1-12 -> test_output.json (~5 min)
```

Every script waits for a free GPU (pmon guard) before it starts.
