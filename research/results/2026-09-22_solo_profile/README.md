# Solo profiling and the C_lib catalog (P1-S)

- Date: 2026-09-22. Machine: RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs, 600 W cap, clocks not lockable), driver 580.173.02, torch 2.8.0+cu128, TileLang `0.1.14+cuda.git35be5243` (source tree; `cotile/kernel.py` sha256 `282d87b6…`, `libtilelang.so` `d0dbba0e…`, recorded per batch), FlashInfer 0.7.0.
- Code: `research/bench/scripts/solo_profile.py` (measurement), `research/bench/scripts/solo_analyze.py` (tables, `summary.json`), `research/bench/scripts/solo_flush_kinds.py` (flush-kind experiment), `cotile/catalog.py` (loader API, work model, Pareto/C_lib, budget curves, pairing table).
- Shapes (plan §2.1): GEMM bf16 NT M ∈ {2048, 4096, 8192} × (N,K) ∈ {(4096,4096), (14336,4096), (4096,14336)}; GQA decode (Hq 32, Hkv 8, D 128) B ∈ {16,32,64,128} × KV ∈ {2048, 8192, 32768}; RMSNorm T ∈ {4096, 16384, 65536} × H ∈ {4096, 8192}. **Every** config of `op.configs(shape)` (GEMM 43, decode 36, RMSNorm 39 or 30 per shape; 1026 configs) is measured at four points (grid @188 / @94 / @48, persistent @188): 4104 config points, plus 270 reference points (start and end of each batch), 27 cuBLAS fp32-reduction points and 27 flush-only baselines. Budgets were done for **all** shapes, not a subset.

| file | content |
|---|---|
| `points/<op>__<shape>.json` (27) | configs (fields, compiled resource signatures of the grid and persistent kernels, work model, fp32-reference check), every measured point, keyed `<cfg>/<build>/<sms>` with `/` written as a pipe character (`grid` @188/94/48, `persistent` @188; `ref:<lib>` with build `ref` or `ref_end`; `flush_only` with build `none`), batch meta (env hashes, batch warmup, foreign-GPU record) |
| `summary.json` | derived: best vs references, persistent penalty, budget effect, C_lib per shape, pairing tables, quality, study summaries |
| `compile_<op>.json`, `runlog.json` | compile statistics / kernel-cache size; sweep log with a 5-s power–temperature–clock timeline |
| `studies/warmup.json` | per-point warmup study |
| `studies/persistent_modes.json` | grid vs persistent under write-flush / read-flush / hot / graph |
| `studies/flush_kinds.json`, `studies/flush_bias.json` | cobench write-flush vs read-flush vs write+read flush |
| `studies/recheck.json` | 5 points re-measured twice at the end |
| `studies/energy.json` | energy per call: flush subtraction vs back-to-back (graph) |

Reproduce (`source research/env.sh`): `solo_profile.py compile --ops <op>` (one process per op, in parallel), `solo_profile.py run` (resumable at point granularity), `solo_profile.py warmup-study | recheck | persistent-modes | energy-check | ref-extra | flush-bias`, `solo_flush_kinds.py`, then `solo_analyze.py --md tables.md`. Load: `from cotile import catalog; cat = catalog.load()`.

## Summary

1. **S2 — the TileLang solo baseline is honest: no shape is flagged (> 10% slower than the reference).** GEMM: best TileLang / fastest correct cuBLAS = 0.85–1.065 (worst M4096_N14336_K4096, +6.5%; TileLang 15% faster at 2048×4096×4096). Decode: TileLang ≤ FlashInfer at all 12 shapes (0.934–0.997). RMSNorm: 0.978–1.036 vs FlashInfer; torch 2.8's `F.rms_norm` runs 8 unfused kernels and is 5–10× slower (not a meaningful reference). Two caveats: (a) cuBLAS runs at a *lower* clock than the TileLang GEMMs (e.g. 1928 vs 2125 MHz): in cycles cuBLAS is 10–15% ahead at 8 of 9 shapes, in time 0–6.5% — the TileLang GEMMs are less power-dense, which the power cap rewards solo but a co-located partner will take away; (b) torch's default `allow_bf16_reduced_precision_reduction=True` lets cuBLAS reduce split-K partials in bf16: at M2048_N4096_K14336 its output misses the fp32-reference tolerance (max |err| 0.029, 9.9× the bound); with the flag off cuBLAS takes 806.5 µs instead of 722.5 µs (reference `cublas_fp32red`, measured for all GEMM shapes and budgets after the sweep). S2 compares against the fastest *correct* reference.
2. **S3 — static persistent builds are much slower, and it is not the flush, occupancy or codegen.** Persistent/grid (same config, flush mode): GEMM median 1.52× (solo-best config 1.39×, max 1.85×); decode 1.00× when the persistent kernel does not loop (tiles ≤ num_ctas) but median 1.10× / p90 1.42× / max 1.82× when it loops, at the same occupancy; RMSNorm median 1.04×. For GEMM the penalty survives without any flush (hot and graph modes: 1.20–1.75× vs 1.26–1.84× in flush mode), the grid and persistent kernels have the same SASS main loop (904 instructions, identical HMMA/LDGSTS/LDSM/BAR counts), the same registers ±3 and occupancy, and the persistent kernel runs at a *higher* clock. For RMSNorm the flush does inflate it (1.05 → 1.004 in hot/graph mode). This is consistent with the CoKernel agent's static-vs-dynamic observation for GEMM in flush mode, but here the gap does **not** vanish in hot mode.
3. **S4 — the GOLDYLOC effect is large for GEMM, small for the memory-bound ops.** GEMM: the budget-best config differs from the full-GPU best in 8/9 shapes at both 94 and 48 SMs; the full-GPU best loses 6.2–13.8% at 94 SMs (median 9.1%) and 9.1–14.3% at 48 (median 12.3%). The budget winner is always `128x128x64_s2_t256_ws=auto` (TMA warp-specialized); at full GPU the large `256x128`/`128x256` direct-epilogue tiles win because they are more power-efficient under the cap, which binds much less on 94 SMs and not at all on 48 (median clock of all GEMM configs: 2015 MHz @188, 2736 @94, 2822 @48; solo-best configs @188: 2125–2488 MHz). Decode: median loss 0.5% (94) / 0.07% (48); only B16_S2048 loses 9–13%. RMSNorm: median 0.2% (94) / 3.6% (48), max 11.2% (T4096_H8192 @48). Best-config changes in decode/RMSNorm are mostly within noise (loss < 2%).
4. **S5 — C_lib sizes** (Pareto ∪ budget-best): GEMM 9–12, decode 5–10, RMSNorm 4–11 per shape (persistent variant 5–9 / 3–8 / 5–9) — in the plan's 10–15 range.
5. **S7 — the power cap is the binding resource for most balanced pairs.** Solo-best GEMMs draw 608–689 W *while running* in flush mode (the ~360 W flush phase lends the controller headroom), decode/RMSNorm 387–432 W. T_serial / max(LB, max solo) is 1.11–1.38 for GEMM × decode (63 pairs with duration ratio in [0.25, 4]) and 1.09–1.35 for GEMM × RMSNorm (24 pairs); power binds in 49/63 and 23/24 pairs, DRAM in the rest (decode-heavy pairs with ratio ≈ 0.3); the tensor bound never binds. Energy per call is almost regime-independent (flush-subtraction vs back-to-back graph mode: −2.4…+3.9%), which supports the power bound; but it is an estimate, not a strict bound (see §S7).
6. **Measurement — the cobench write-flush biases memory-bound ops.** It leaves L2 full of dirty lines, so every line the timed op allocates (including its own output writes, which would otherwise be absorbed by L2 and written back after the timed window) first evicts a dirty line. A write-then-read flush (evicted *and* clean) is faster by: GEMM 1–5.8% (median 38 µs), decode 0–2%, RMSNorm 1–30% (T4096×4096: 43.0 → 30.7 µs). A read-only flush must not be used: it does not fully evict on this GPU. Rankings are hardly affected (TileLang and references shift alike), but the bias is paid once per rep: a sum of two solo times pays it twice while one `bench_corun` rep pays it once, so co-run speed-ups must be computed against `bench_corun`'s own `serial` variant, not against summed solos (see Measurement). **Recommendation:** switch cobench's flush to write+read (or document the choice) before P1 co-run measurements.
7. **Reproducibility.** 5 points re-measured twice at the end: −0.07…+0.50% (GEMM @188 +0.50%/−0.07%, others ≤ 0.35%). References re-measured at the end of each batch: −0.6…+0.4% (135 pairs; one outlier, torch's composite rms_norm @94, −5%). Thermal state is the dominant slow variable for power-capped GEMMs (2.2–2.6% after a 2-s idle), handled by keeping the GPU loaded.

## Protocol

Every timed point is a cobench **flush-mode** point (`research/bench/README.md`), the same L2 treatment as `bench_corun`: per rep, a 256 MB (2×L2) write on a full-GPU side stream → host gate → start event on the op stream → op → end event. The script reuses cobench's flush buffer, side stream, host gate and clock probe (`cobench.timing` internals) and adds:

| item | choice | why |
|---|---|---|
| op stream | full GPU: a torch side stream; 94 and 48 SMs: green contexts with `IGNORE_SM_COSCHEDULING` (contiguous SMs [0, n)) | exact SM counts; a 47-SM request yields **48** (2-SM granularity), so the "1/4 budget" is 48 SMs. TileLang (tvm-ffi) kernels launch correctly on green streams |
| global warmup | flush-mode loop of cuBLAS 8192×14336×4096 ≥ 60 s, then until the max temperature over 20-s windows is flat (< 1 °C), ≤ 240 s | 52 → 83 °C in 162 s |
| batch warmup | before each op × shape batch: replay the batch's first config at its four points in 2-s slices, ≥ 12 s, then until the temperature is flat (10-s windows), ≤ 60 s (took 12–22.5 s) | restores the batch's own thermal state after the CPU-side prep (correctness checks) |
| point order | config-outer: grid @188, persistent @188, grid @94, grid @48 of one config adjacent; references at the start **and** end of the batch | every config sees the same thermal history; start/end references measure residual drift |
| per-point warmup | ≥ 0.3 s and ≥ 20 reps of the point itself | warmup study below |
| timed window | ≥ 50 reps and ≥ 0.25 s (≥ ~12 NVML 20-ms power samples), ≤ 4096 reps | power needs samples; short kernels get ~1000 reps |
| no idle gaps | ~0.1 s of untimed identical reps are enqueued after the timed reps (the host post-processes meanwhile); all 2432 kernels are loaded before the global warmup | idle cools the GPU |
| CV rule | CV > 2% → one more attempt; lowest-CV attempt kept | flush-mode points < ~100 µs have a bimodal +2 µs launch-jitter mode that retries do not remove; `spread` = (p90−p10)/median is stored as the robust alternative |
| correctness | every config and build checked against an fp32 reference (cotile tolerances) before timing; references too | all 2052 TileLang kernels pass, incl. the 17 GB (> 2^31-element) KV of B128_S32768; references: all pass except cuBLAS default at M2048_N4096_K14336 |
| GPU sharing | occupied = a foreign process with SM% > 0 (`nvidia-smi pmon -s u -d 1` streamed in a thread; rule clarified by the coordinator); wait (poll 150 s) before a batch while busy; points whose [warmup start − 2 s, end + 2 s] saw foreign SM% > 0 are re-measured after the batch | user `qzr`'s idle retriever (`local_retriever.py --gpu`, PID 354072) held a context during the whole sweep with no SM activity; **0 batches / 0 points had to be re-measured** |

Recorded per point: median / p10 / p90 / CV / min / max / n / `spread`; per-rep SM clock from the probe (median, min, max, `cycles` = µs × MHz, correlation with time); `half_ratio` = median of the 2nd half of the reps / 1st half; mean board power `P_w` from the NVML 20-ms samples inside the timed window (first 20 ms skipped); max temperature; fraction of NVML polls with `SwPowerCap`; rep period.

**Energy per call (µJ).** `E_naive` = P_w × t (the task's definition; P_w averages over whole reps, so for short ops it mostly measures the flush). `E_kernel` = P_w × period − P₀ × period₀, with (P₀, period₀) a flush-only rep measured at the start of the same batch (347–374 W, 167 µs): the energy one call adds on top of the flush; `P_kernel` = E_kernel / t. This is what `cotile.catalog` uses. Caveats: NVML board power (DRAM, VRM, fans included; 20-ms samples of unknown averaging window); the flush phase is assumed to cost the same with and without the op; energy per op depends on the clock/voltage point; `E_kernel` includes static power accrued during the op, so it grows when the op is slowed down (e.g. on 48 SMs). The check against back-to-back execution is in the energy table below.

**Work model** (`catalog.work`): `flops` (useful), `flops_mma` (issued; decode pads heads to 16 MMA rows), `bytes_min` (compulsory DRAM bytes, a lower bound for any implementation), `bytes_ws` (split-reduction workspace traffic, upper bound), `bytes_tiles` (global→SM traffic incl. redundant tile loads). Stored per config.

## S2 — best TileLang config vs reference (full GPU, flush mode)

`†` = output fails the fp32-reference tolerance (not used for the ratio). `cublas_fp32red` = cuBLAS with `allow_bf16_reduced_precision_reduction=False`, measured after the sweep in a separate process. Rate: GEMM TFLOP/s, others GB/s of compulsory bytes.

| op | shape | configs | best TileLang (grid) | µs | rate | MHz | best persistent µs | reference µs (MHz) | TL / ref | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| gemm | M2048_N14336_K4096 | 43 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | 718.8 | 335 TF/s | 2263 | 994.9 | cublas 693.8 (2053), cublas_fp32red 695.9 (2046) | 1.036 |  |
| gemm | M2048_N4096_K14336 | 43 | `128x128x64_s2_t128_k1_wsauto_smem_g8` | 759.4 | 317 TF/s | 2176 | 864.1 | cublas† 722.5 (2079), cublas_fp32red 806.5 (2366) | 0.942 |  |
| gemm | M2048_N4096_K4096 | 43 | `128x128x64_s2_t256_k1_wsauto_smem_g8` | 202.4 | 339 TF/s | 2296 | 253.8 | cublas 237.1 (2429), cublas_fp32red 237.2 (2417) | 0.854 |  |
| gemm | M4096_N14336_K4096 | 43 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | 1476.6 | 326 TF/s | 2172 | 2249.3 | cublas_fp32red 1386.0 (2015), cublas 1399.0 (1998) | 1.065 |  |
| gemm | M4096_N4096_K14336 | 43 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 1496.3 | 321 TF/s | 2208 | 1749.3 | cublas_fp32red 1452.8 (1984), cublas 1473.1 (1963) | 1.030 |  |
| gemm | M4096_N4096_K4096 | 43 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 392.4 | 350 TF/s | 2488 | 494.2 | cublas_fp32red 380.5 (2256), cublas 384.7 (2233) | 1.031 |  |
| gemm | M8192_N14336_K4096 | 43 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | 2978.8 | 323 TF/s | 2125 | 4692.9 | cublas_fp32red 2821.6 (1956), cublas 2858.6 (1928) | 1.056 |  |
| gemm | M8192_N4096_K14336 | 43 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 3014.6 | 319 TF/s | 2164 | 4292.5 | cublas_fp32red 2906.8 (1966), cublas 2944.7 (1941) | 1.037 |  |
| gemm | M8192_N4096_K4096 | 43 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 813.1 | 338 TF/s | 2349 | 1210.0 | cublas_fp32red 802.4 (2100), cublas 813.7 (2069) | 1.013 |  |
| gqa_decode | B128_S2048 | 36 | `n64_h4_sp1_t128_s2` | 682.0 | 1578 GB/s | 2834 | 695.2 | flashinfer_cc 684.0 (2803), flashinfer_tc 686.7 (2834) | 0.997 |  |
| gqa_decode | B128_S32768 | 36 | `n64_h4_sp1_t128_s2` | 10462.0 | 1642 GB/s | 2834 | 10664.0 | flashinfer_tc 10498.0 (2819), flashinfer_cc 10534.0 (2811) | 0.997 |  |
| gqa_decode | B128_S8192 | 36 | `n128_h4_sp1_t128_s1` | 2669.5 | 1610 GB/s | 2836 | 2715.6 | flashinfer_cc 2677.8 (2811), flashinfer_tc 2681.5 (2834) | 0.997 |  |
| gqa_decode | B16_S2048 | 36 | `n32_h4_sp1_t64_s3` | 91.7 | 1466 GB/s | 2823 | 91.7 | flashinfer_tc 98.2 (2820), flashinfer_cc 99.9 (2791) | 0.934 |  |
| gqa_decode | B16_S32768 | 36 | `n64_h4_sp1_t128_s2` | 1348.5 | 1593 GB/s | 2836 | 1348.5 | flashinfer_cc 1357.5 (2815), flashinfer_tc 1360.8 (2834) | 0.993 |  |
| gqa_decode | B16_S8192 | 36 | `n128_h4_sp1_t128_s1` | 342.9 | 1566 GB/s | 2837 | 342.6 | flashinfer_tc 350.6 (2835), flashinfer_cc 350.7 (2813) | 0.978 |  |
| gqa_decode | B32_S2048 | 36 | `n64_h4_sp1_t64_s1` | 176.0 | 1528 GB/s | 2836 | 176.0 | flashinfer_tc 177.9 (2835), flashinfer_cc 183.9 (2808) | 0.990 |  |
| gqa_decode | B32_S32768 | 36 | `n64_h4_sp1_t128_s1` | 2667.5 | 1610 GB/s | 2835 | 2669.1 | flashinfer_cc 2677.3 (2811), flashinfer_tc 2680.5 (2834) | 0.996 |  |
| gqa_decode | B32_S8192 | 36 | `n64_h4_sp1_t128_s1` | 677.8 | 1585 GB/s | 2835 | 677.9 | flashinfer_tc 682.7 (2834), flashinfer_cc 685.8 (2814) | 0.993 |  |
| gqa_decode | B64_S2048 | 36 | `n64_h4_sp1_t128_s1` | 344.1 | 1563 GB/s | 2835 | 346.0 | flashinfer_tc 349.8 (2835), flashinfer_cc 353.8 (2800) | 0.984 |  |
| gqa_decode | B64_S32768 | 36 | `n128_h4_sp2_t128_s1` | 5268.4 | 1631 GB/s | 2834 | 5268.5 | flashinfer_cc 5290.8 (2811), flashinfer_tc 5292.7 (2834) | 0.996 |  |
| gqa_decode | B64_S8192 | 36 | `n64_h4_sp1_t128_s1` | 1348.5 | 1593 GB/s | 2836 | 1348.6 | flashinfer_cc 1356.5 (2810), flashinfer_tc 1363.6 (2835) | 0.994 |  |
| rmsnorm | T16384_H4096 | 39 | `r2_t128_v8` | 179.8 | 1493 GB/s | 2836 | 186.4 | flashinfer 180.3 (2832), torch 1772.2 (2830) | 0.998 |  |
| rmsnorm | T16384_H8192 | 30 | `r8_t512_v2` | 362.4 | 1481 GB/s | 2834 | 368.6 | flashinfer 362.5 (2833), torch 3594.2 (2831) | 1.000 |  |
| rmsnorm | T4096_H4096 | 39 | `r2_t128_v2` | 44.6 | 1505 GB/s | 2835 | 42.9 | flashinfer 43.0 (2834), torch 222.8 (2775) | 1.036 |  |
| rmsnorm | T4096_H8192 | 30 | `r8_t512_v2` | 88.0 | 1525 GB/s | 2832 | 92.1 | flashinfer 90.0 (2831), torch 829.1 (2825) | 0.978 |  |
| rmsnorm | T65536_H4096 | 39 | `r4_t512_v4` | 723.9 | 1483 GB/s | 2834 | 735.2 | flashinfer 729.1 (2834), torch 7246.5 (2831) | 0.993 |  |
| rmsnorm | T65536_H8192 | 30 | `r4_t512_v8` | 1449.9 | 1481 GB/s | 2834 | 1464.2 | flashinfer 1459.1 (2832), torch 14504.0 (2831) | 0.994 |  |

## S3 — persistent vs grid build

Persistent build: `num_ctas = min(CTAs/SM of the compiled persistent kernel × 188, tiles)`, static grid-stride. Two compile passes (pass 1 with min(188, tiles) CTAs to read the occupancy); 3 RMSNorm configs needed a fixed-point rebuild because the trip count changed their registers (e.g. `r2_t64_v2` @T4096_H4096: 6 CTAs/SM with 188 or 752 CTAs, 4 with 1128). GEMM `ws=auto` grid configs have no persistent twin with the same pipeline (persistent builds always disable warp specialization), so their penalty mixes two effects; the modes study uses `ws=off` configs only.

| op | configs×shapes | median t_pers/t_grid | p10 | p90 | max | share > 1.05 | share < 0.95 | solo-best config: pers/grid (median, max) | best pers / best grid (median, max) |
|---|---|---|---|---|---|---|---|---|---|
| gemm | 387 | 1.515 | 1.158 | 1.642 | 1.851 | 97% | 0% | 1.390, 1.586 | 1.384, 1.575 |
| gqa_decode | 432 | 1.082 | 1.000 | 1.392 | 1.820 | 59% | 0% | 1.000, 1.126 | 1.000, 1.019 |
| rmsnorm | 207 | 1.040 | 1.011 | 1.084 | 1.142 | 31% | 0% | 1.064, 1.100 | 1.016, 1.046 |

By structure (flush mode):

| op | subset | n | median | p90 | max |
|---|---|---|---|---|---|
| gemm | ws=off, same occupancy as grid | 153 | 1.493 | 1.560 | – |
| gemm | ws=off, lower occupancy (smem = sum of buffers) | 135 | 1.547 | 1.658 | – |
| gemm | ws=auto (persistent uses cp.async) | 27 | 1.565 | 1.620 | – |
| gemm | split-K | 72 | 1.42–1.56 | 1.74–1.82 | – |
| gqa_decode | persistent does not loop (tiles ≤ num_ctas) | 60 | 1.000 | 1.000 | 1.001 |
| gqa_decode | persistent loops, same occupancy | 372 | 1.101 | 1.424 | 1.820 |
| rmsnorm | loops | 207 | 1.03–1.04 | 1.06–1.09 | 1.142 |

Example with identical resources: GEMM 4096×14336×4096 `256x128x64_s2_t256_direct`: grid 202 regs / 96 KB / 1 CTA/SM, 1476.6 µs at 2172 MHz; persistent 205 regs / 96 KB / 1 CTA/SM (188 CTAs), 2262.0 µs at 2517 MHz. Their SASS differs only in 2 branches and 5 compares.

**Grid vs persistent under four regimes** (t_persistent / t_grid; cycles ratio in parentheses):

| op | shape | config | flush (write) | flush (read) | hot | graph | grid µs flush / hot / graph |
|---|---|---|---|---|---|---|---|
| gemm | M4096_N4096_K4096 | `128x128x64_s2_t128_k1_wsoff_smem_g8` | 1.531 (1.568) | 1.538 (1.630) | 1.475 (1.641) | 1.437 (1.639) | 422.4 / 431.8 / 458.8 |
| gemm | M4096_N4096_K4096 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 1.260 (1.294) | 1.257 (1.288) | 1.229 (1.299) | 1.202 (1.295) | 393.1 / 400.4 / 425.0 |
| gemm | M4096_N4096_K4096 | `64x128x64_s2_t128_k4_wsoff_smem_g8` | 1.844 (2.358) | 1.914 (2.378) | 1.755 (2.405) | 1.751 (2.399) | 591.5 / 622.4 / 623.0 |
| gqa_decode | B64_S8192 | `n64_h4_sp1_t128_s1` | 0.999 (0.999) | 0.999 (0.999) | 0.999 (0.999) | 1.000 (1.000) | 1348.6 / 1305.2 / 1304.9 |
| gqa_decode | B64_S8192 | `n32_h4_sp8_t64_s2` | 1.090 (1.090) | 1.094 (1.095) | 1.094 (1.098) | 1.094 (1.093) | 1369.1 / 1315.9 / 1315.7 |
| gqa_decode | B64_S8192 | `n64_h1_sp4_t128_s2` | 1.726 (1.863) | 1.754 (1.893) | 1.712 (2.028) | 1.708 (2.032) | 1385.4 / 1362.7 / 1366.0 |
| rmsnorm | T16384_H4096 | `r2_t128_v2` | 1.057 (1.055) | 1.044 (1.046) | 1.004 (1.004) | 1.002 (1.003) | 180.2 / 181.7 / 182.1 |
| rmsnorm | T16384_H4096 | `r2_t128_v8` | 1.046 (1.049) | 1.038 (1.041) | 1.005 (1.005) | 1.003 (1.004) | 180.1 / 181.4 / 181.4 |
| rmsnorm | T16384_H4096 | `r1_t512_v8` | 1.145 (1.144) | 1.124 (1.124) | 1.068 (1.068) | 1.069 (1.069) | 180.2 / 181.2 / 181.4 |

Reading: the GEMM and decode penalties are the same (±5 points) in all four regimes, in time and in cycles — they are not an artefact of flush mode or L2 state. Only the RMSNorm penalty is mostly a flush-mode effect. The graph-mode grid GEMMs are slower in time than flush-mode ones because back-to-back execution runs at a lower power-capped clock (cycles agree). **Do not use the static persistent build as the solo reference for the intra column**; a dynamic-queue or hardware-scheduled variant is needed (the CoKernel agent reports that a dynamic queue closes the gap for GEMM in flush mode).

## S4 — SM budgets (grid build; green contexts, 94 and 48 SMs)

Loss = t(full-GPU best at the budget) / t(budget best) − 1. Last column: fastest correct reference / TileLang budget-best at 94, 48 (> 1: TileLang faster).

| op | shape | best @188 | best @94 (loss of @188-best) | best @48 (loss of @188-best) | best-ref @94 / @48 vs TL budget-best |
|---|---|---|---|---|---|
| gemm | M2048_N14336_K4096 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+8.8%) | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+14.3%) | 0.95, 1.00 |
| gemm | M2048_N4096_K14336 | `128x128x64_s2_t128_k1_wsauto_smem_g8` | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+7.2%) | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+9.1%) | 0.96, 1.06 |
| gemm | M2048_N4096_K4096 | `128x128x64_s2_t256_k1_wsauto_smem_g8` | same (+0.0%) | same (+0.0%) | 0.97, 1.06 |
| gemm | M4096_N14336_K4096 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+9.1%) | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+13.0%) | 0.96, 0.99 |
| gemm | M4096_N4096_K14336 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+9.4%) | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+12.6%) | 0.97, 0.98 |
| gemm | M4096_N4096_K4096 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+10.5%) | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+11.3%) | 0.98, 0.98 |
| gemm | M8192_N14336_K4096 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+6.2%) | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+12.3%) | 0.95, 0.98 |
| gemm | M8192_N4096_K14336 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+12.2%) | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+12.6%) | 1.00, 0.98 |
| gemm | M8192_N4096_K4096 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+13.8%) | `128x128x64_s2_t256_k1_wsauto_smem_g8` (+11.1%) | 1.01, 0.98 |
| gqa_decode | B128_S2048 | `n64_h4_sp1_t128_s2` | `n64_h4_sp1_t128_s1` (+2.3%) | `n64_h4_sp1_t128_s1` (+0.4%) | 1.01, 1.03 |
| gqa_decode | B128_S32768 | `n64_h4_sp1_t128_s2` | `n64_h4_sp1_t128_s1` (+2.8%) | `n64_h4_sp1_t64_s1` (+0.1%) | 1.00, 1.00 |
| gqa_decode | B128_S8192 | `n128_h4_sp1_t128_s1` | `n64_h4_sp1_t64_s1` (+1.7%) | same (+0.0%) | 1.00, 1.00 |
| gqa_decode | B16_S2048 | `n32_h4_sp1_t64_s3` | `n64_h4_sp1_t128_s2` (+9.0%) | `n128_h4_sp1_t128_s1` (+13.4%) | 1.11, 1.14 |
| gqa_decode | B16_S32768 | `n64_h4_sp1_t128_s2` | same (+0.0%) | same (+0.0%) | 1.02, 1.03 |
| gqa_decode | B16_S8192 | `n128_h4_sp1_t128_s1` | `n64_h4_sp1_t128_s2` (+0.5%) | `n64_h4_sp1_t128_s2` (+0.3%) | 1.04, 1.10 |
| gqa_decode | B32_S2048 | `n64_h4_sp1_t64_s1` | `n64_h4_sp1_t128_s2` (+0.1%) | `n32_h4_sp1_t64_s2` (+0.0%) | 1.06, 1.10 |
| gqa_decode | B32_S32768 | `n64_h4_sp1_t128_s1` | `n128_h4_sp1_t128_s1` (+0.1%) | `n64_h4_sp1_t64_s1` (+0.0%) | 1.01, 1.01 |
| gqa_decode | B32_S8192 | `n64_h4_sp1_t128_s1` | `n64_h4_sp1_t128_s2` (+0.6%) | `n32_h4_sp1_t64_s2` (+0.0%) | 1.02, 1.07 |
| gqa_decode | B64_S2048 | `n64_h4_sp1_t128_s1` | same (+0.0%) | `n64_h4_sp1_t128_s2` (+2.1%) | 1.01, 1.07 |
| gqa_decode | B64_S32768 | `n128_h4_sp2_t128_s1` | `n128_h4_sp1_t128_s1` (+2.1%) | `n128_h4_sp1_t128_s1` (+0.0%) | 1.00, 1.03 |
| gqa_decode | B64_S8192 | `n64_h4_sp1_t128_s1` | `n64_h4_sp1_t128_s2` (+0.2%) | `n64_h4_sp1_t128_s2` (+1.4%) | 1.01, 1.04 |
| rmsnorm | T16384_H4096 | `r2_t128_v8` | `r8_t256_v8` (+0.2%) | `r2_t128_v4` (+0.0%) | 1.01, 1.07 |
| rmsnorm | T16384_H8192 | `r8_t512_v2` | `r4_t512_v4` (+2.0%) | `r1_t128_v4` (+2.8%) | 1.01, 1.06 |
| rmsnorm | T4096_H4096 | `r2_t128_v2` | same (+0.0%) | `r1_t256_v2` (+4.3%) | 0.99, 1.04 |
| rmsnorm | T4096_H8192 | `r8_t512_v2` | `r4_t256_v4` (+2.3%) | `r4_t256_v4` (+11.2%) | 1.04, 1.12 |
| rmsnorm | T65536_H4096 | `r4_t512_v4` | `r8_t256_v8` (+0.2%) | `r8_t256_v8` (+6.4%) | 1.01, 1.02 |
| rmsnorm | T65536_H8192 | `r4_t512_v8` | same (+0.0%) | `r4_t256_v8` (+0.4%) | 1.01, 1.03 |

Per op: GEMM best differs in 8/9 shapes at 94 and at 48 (loss > 5% in all 8; median 9.1% / 12.3%, max 13.8% / 14.3%); decode 10/12 differ but loss > 2% in 4/12 (94) and 2/12 (48), > 5% only at B16_S2048; RMSNorm loss > 2% in 2/6 (94) and 4/6 (48), > 5% in 2/6 (48). References are not re-planned for the partition (FlashInfer's split-KV plan and cuBLAS heuristics assume 188 SMs); TileLang budget-best beats them by up to 14% at 48 SMs.

## S5 — C_lib

C_lib(op, shape) = Pareto front over (grid time @188; grid-kernel resources: smem/CTA ↓, regs/CTA = regs × threads ↓, threads/CTA ↓, CTAs/SM ↑) ∪ grid-best @94 ∪ grid-best @48. Direction choice for CTAs/SM: more CTAs/SM = finer granularity / smaller per-CTA footprint (the other three axes already capture size; with CTAs/SM minimized instead, the fronts would favour whole-SM configs). `c_lib_persistent`: the same with persistent time and persistent-kernel resources (for the intra column). `c_lib_reasons` tells why each member is included. Energy: `E_kernel` and `P_kernel` of the solo-best config.

| op | shape | configs | Pareto (grid) | C_lib = Pareto ∪ budget-best | C_lib (persistent) | solo-best E/call (µJ) | solo-best P_kernel (W) |
|---|---|---|---|---|---|---|---|
| gemm | M2048_N14336_K4096 | 43 | 11 | 12 | 7 | 459410 | 639 |
| gemm | M2048_N4096_K14336 | 43 | 8 | 9 | 8 | 476430 | 627 |
| gemm | M2048_N4096_K4096 | 43 | 10 | 10 | 9 | 139100 | 687 |
| gemm | M4096_N14336_K4096 | 43 | 12 | 12 | 5 | 915510 | 620 |
| gemm | M4096_N4096_K14336 | 43 | 10 | 11 | 7 | 919260 | 614 |
| gemm | M4096_N4096_K4096 | 43 | 11 | 11 | 6 | 263050 | 670 |
| gemm | M8192_N14336_K4096 | 43 | 12 | 12 | 5 | 1818700 | 611 |
| gemm | M8192_N4096_K14336 | 43 | 9 | 10 | 8 | 1832700 | 608 |
| gemm | M8192_N4096_K4096 | 43 | 8 | 9 | 6 | 516590 | 635 |
| gqa_decode | B128_S2048 | 36 | 8 | 8 | 3 | 286800 | 421 |
| gqa_decode | B128_S32768 | 36 | 9 | 10 | 4 | 4524300 | 432 |
| gqa_decode | B128_S8192 | 36 | 5 | 5 | 4 | 1119500 | 419 |
| gqa_decode | B16_S2048 | 36 | 6 | 8 | 7 | 35486 | 387 |
| gqa_decode | B16_S32768 | 36 | 8 | 8 | 6 | 568090 | 421 |
| gqa_decode | B16_S8192 | 36 | 9 | 9 | 8 | 139090 | 406 |
| gqa_decode | B32_S2048 | 36 | 4 | 5 | 3 | 69515 | 395 |
| gqa_decode | B32_S32768 | 36 | 4 | 5 | 4 | 1134800 | 425 |
| gqa_decode | B32_S8192 | 36 | 5 | 6 | 5 | 282900 | 417 |
| gqa_decode | B64_S2048 | 36 | 5 | 6 | 4 | 141690 | 412 |
| gqa_decode | B64_S32768 | 36 | 5 | 6 | 3 | 2244300 | 426 |
| gqa_decode | B64_S8192 | 36 | 5 | 6 | 4 | 564970 | 419 |
| rmsnorm | T16384_H4096 | 39 | 4 | 6 | 6 | 76466 | 425 |
| rmsnorm | T16384_H8192 | 30 | 9 | 10 | 5 | 153080 | 422 |
| rmsnorm | T4096_H4096 | 39 | 3 | 4 | 9 | 17944 | 402 |
| rmsnorm | T4096_H8192 | 30 | 10 | 11 | 6 | 37524 | 426 |
| rmsnorm | T65536_H4096 | 39 | 8 | 8 | 7 | 307410 | 425 |
| rmsnorm | T65536_H8192 | 30 | 9 | 9 | 6 | 612680 | 423 |

## S7 — pairing table (GEMM × decode, GEMM × RMSNorm)

All shape pairs with best-solo duration ratio t_A / t_B ∈ [0.25, 4] (TileLang solo-best grid times, full GPU, flush mode). T_serial = t_A + t_B. Naive lower bounds from measured machine rates (best over all full-GPU points): tensor 357 TFLOP/s (cuBLAS 4096³), DRAM 1642 GB/s (compulsory bytes, decode B128_S32768).
- LB_tc = (flops_mma_A + flops_mma_B) / 357 TFLOP/s; LB_dram = (bytes_min_A + bytes_min_B) / 1642 GB/s;
- LB_power (steady) = (E_A + E_B) / 600 W — the bound for back-to-back execution;
- LB_power (flush) = (E_A + E_B + E_f) / 600 W − t_f, with (E_f, t_f) the flush phase of a rep (≈ 60 mJ, 167 µs): the bound for a flush-mode co-run rep, where the ~360 W flush phase lends headroom (this is why solo GEMMs show P_kernel > 600 W);
- LB = max(LB_tc, LB_dram, LB_power(flush)); bound column = T_serial / max(LB, max(t_A, t_B)).

Caveats: energy per op falls with voltage when the cap lowers the clock in a co-run, so LB_power is an estimate, not a strict bound (the energy check shows E per call nearly unchanged between flush-mode and back-to-back execution, −2.4…+3.9%, i.e. across a 2080–2315 MHz range for GEMM; larger clock drops are untested); all times include the write-flush bias (§Measurement), which inflates T_serial of memory-bound partners by 0–30%; with the fastest correct reference as the solo time where it beats TileLang (`pairs(..., use_ref=True)`), the bounds shrink by ≤ 4.2%.

**Candidate P1/P2 study points** (ratio near 1, highest bound, and a ratio scan):

| use | A (GEMM) | B | t_A / t_B µs | ratio | T_serial | binding | bound |
|---|---|---|---|---|---|---|---|
| P1 balanced, short | M2048_N4096_K4096 | decode B32_S2048 | 202 / 176 | 1.15 | 378 | power | 1.35 |
| P1 balanced | M4096_N4096_K4096 | decode B16_S8192 (≡ B64_S2048) | 392 / 343 | 1.14 | 735 | power | 1.22 |
| P1 balanced, long | M2048_N4096_K14336 | decode B32_S8192 | 759 / 678 | 1.12 | 1437 | power | 1.20 |
| P1 balanced, long | M4096_N14336_K4096 | decode B64_S8192 | 1477 / 1348 | 1.09 | 2825 | power | 1.18 |
| P1 ratio scan (A = 4096³) | M4096_N4096_K4096 | decode B64_S8192 / B32_S8192 / B16_S8192 / B32_S2048 | 392 / 1348, 678, 343, 176 | 0.29 / 0.58 / 1.14 / 2.23 | – | dram / power / power / power | 1.27 / 1.27 / 1.22 / 1.16 |
| P2 balanced | M4096_N4096_K4096 | RMSNorm T16384_H8192 | 392 / 362 | 1.08 | 755 | power | 1.20 |
| P2 balanced, long | M2048_N14336_K4096 | RMSNorm T65536_H4096 | 719 / 724 | 0.99 | 1443 | power | 1.19 |
| P2 balanced, long | M4096_N14336_K4096 | RMSNorm T65536_H8192 | 1477 / 1450 | 1.02 | 2926 | power | 1.18 |
| P2 short | M2048_N4096_K4096 | RMSNorm T16384_H4096 | 202 / 180 | 1.13 | 382 | power | 1.31 |

(B16_S8192 and B64_S2048 have the same KV bytes and near-identical solo times; either can stand in.) Ratio 4 with A = 4096³ is just outside the table (B16_S2048: 392 / 92 = 4.3).

<details><summary>Full pairing tables (63 + 24 rows)</summary>

**gemm × gqa_decode**: 63 shape pairs with t_a/t_b in [0.25, 4] (of 108); sorted by the speedup bound.

| A | B | t_A µs | t_B µs | t_A/t_B | T_serial | LB_tc | LB_dram | LB_power (flush) | LB_power (steady) | binding | T_serial / max(LB, max solo) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M2048_N4096_K4096 | B16_S8192 | 202 | 343 | 0.59 | 545 | 214 | 368 | 397 | 464 | power | 1.38 |
| M2048_N4096_K4096 | B64_S2048 | 202 | 344 | 0.59 | 546 | 214 | 368 | 401 | 468 | power | 1.36 |
| M2048_N4096_K4096 | B32_S2048 | 202 | 176 | 1.15 | 378 | 202 | 205 | 281 | 348 | power | 1.35 |
| M2048_N4096_K4096 | B16_S2048 | 202 | 92 | 2.21 | 294 | 196 | 123 | 225 | 291 | power | 1.31 |
| M8192_N4096_K4096 | B128_S8192 | 813 | 2670 | 0.30 | 3483 | 951 | 2719 | 2661 | 2727 | dram | 1.28 |
| M8192_N4096_K4096 | B32_S32768 | 813 | 2668 | 0.30 | 3481 | 951 | 2718 | 2687 | 2752 | dram | 1.28 |
| M4096_N4096_K4096 | B16_S32768 | 392 | 1348 | 0.29 | 1741 | 476 | 1369 | 1319 | 1385 | dram | 1.27 |
| M4096_N4096_K4096 | B64_S8192 | 392 | 1348 | 0.29 | 1741 | 476 | 1370 | 1314 | 1380 | dram | 1.27 |
| M2048_N4096_K4096 | B128_S2048 | 202 | 682 | 0.30 | 884 | 238 | 696 | 643 | 710 | dram | 1.27 |
| M4096_N4096_K4096 | B32_S8192 | 392 | 678 | 0.58 | 1070 | 428 | 715 | 844 | 910 | power | 1.27 |
| M2048_N4096_K4096 | B32_S8192 | 202 | 678 | 0.30 | 880 | 238 | 695 | 636 | 703 | dram | 1.27 |
| M4096_N4096_K4096 | B128_S2048 | 392 | 682 | 0.58 | 1074 | 428 | 716 | 851 | 916 | power | 1.26 |
| M2048_N4096_K14336 | B64_S8192 | 759 | 1348 | 0.56 | 2108 | 761 | 1426 | 1670 | 1736 | power | 1.26 |
| M2048_N14336_K4096 | B64_S8192 | 719 | 1348 | 0.53 | 2067 | 761 | 1426 | 1641 | 1707 | power | 1.26 |
| M2048_N4096_K14336 | B16_S32768 | 759 | 1348 | 0.56 | 2108 | 761 | 1425 | 1675 | 1741 | power | 1.26 |
| M8192_N4096_K14336 | B128_S32768 | 3015 | 10462 | 0.29 | 13477 | 3425 | 10717 | 10530 | 10595 | dram | 1.26 |
| M2048_N14336_K4096 | B16_S32768 | 719 | 1348 | 0.53 | 2067 | 761 | 1425 | 1646 | 1712 | power | 1.26 |
| M2048_N4096_K14336 | B128_S8192 | 759 | 2670 | 0.28 | 3429 | 856 | 2734 | 2594 | 2660 | dram | 1.25 |
| M8192_N14336_K4096 | B128_S32768 | 2979 | 10462 | 0.28 | 13441 | 3425 | 10717 | 10506 | 10572 | dram | 1.25 |
| M4096_N4096_K14336 | B64_S32768 | 1496 | 5268 | 0.28 | 6765 | 1712 | 5394 | 5207 | 5273 | dram | 1.25 |
| M2048_N4096_K14336 | B32_S32768 | 759 | 2668 | 0.28 | 3427 | 856 | 2733 | 2619 | 2685 | dram | 1.25 |
| M4096_N14336_K4096 | B64_S32768 | 1477 | 5268 | 0.28 | 6745 | 1712 | 5394 | 5201 | 5266 | dram | 1.25 |
| M4096_N4096_K14336 | B128_S8192 | 1496 | 2670 | 0.56 | 4166 | 1522 | 2780 | 3332 | 3398 | power | 1.25 |
| M4096_N14336_K4096 | B128_S8192 | 1477 | 2670 | 0.55 | 4146 | 1522 | 2780 | 3326 | 3392 | power | 1.25 |
| M8192_N4096_K4096 | B64_S8192 | 813 | 1348 | 0.60 | 2162 | 856 | 1410 | 1737 | 1803 | power | 1.24 |
| M8192_N4096_K4096 | B16_S32768 | 813 | 1348 | 0.60 | 2162 | 856 | 1410 | 1742 | 1808 | power | 1.24 |
| M4096_N4096_K14336 | B32_S32768 | 1496 | 2668 | 0.56 | 4164 | 1522 | 2779 | 3358 | 3423 | power | 1.24 |
| M2048_N14336_K4096 | B128_S8192 | 719 | 2670 | 0.27 | 3388 | 856 | 2734 | 2565 | 2632 | dram | 1.24 |
| M2048_N14336_K4096 | B32_S32768 | 719 | 2668 | 0.27 | 3386 | 856 | 2733 | 2590 | 2657 | dram | 1.24 |
| M4096_N14336_K4096 | B32_S32768 | 1477 | 2668 | 0.55 | 4144 | 1522 | 2779 | 3351 | 3417 | power | 1.24 |
| M8192_N4096_K14336 | B64_S32768 | 3015 | 5268 | 0.57 | 8283 | 3044 | 5486 | 6730 | 6795 | power | 1.23 |
| M8192_N14336_K4096 | B64_S32768 | 2979 | 5268 | 0.57 | 8247 | 3044 | 5486 | 6706 | 6772 | power | 1.23 |
| M4096_N4096_K4096 | B16_S8192 | 392 | 343 | 1.14 | 735 | 404 | 388 | 604 | 670 | power | 1.22 |
| M4096_N4096_K4096 | B64_S2048 | 392 | 344 | 1.14 | 736 | 404 | 389 | 609 | 675 | power | 1.21 |
| M2048_N4096_K14336 | B32_S8192 | 759 | 678 | 1.12 | 1437 | 713 | 772 | 1199 | 1266 | power | 1.20 |
| M2048_N4096_K14336 | B128_S2048 | 759 | 682 | 1.11 | 1441 | 713 | 773 | 1206 | 1272 | power | 1.19 |
| M2048_N14336_K4096 | B32_S8192 | 719 | 678 | 1.06 | 1397 | 713 | 772 | 1170 | 1237 | power | 1.19 |
| M2048_N14336_K4096 | B128_S2048 | 719 | 682 | 1.05 | 1401 | 713 | 773 | 1177 | 1244 | power | 1.19 |
| M4096_N4096_K14336 | B64_S8192 | 1496 | 1348 | 1.11 | 2845 | 1427 | 1472 | 2408 | 2474 | power | 1.18 |
| M4096_N4096_K14336 | B16_S32768 | 1496 | 1348 | 1.11 | 2845 | 1427 | 1471 | 2413 | 2479 | power | 1.18 |
| M8192_N4096_K4096 | B32_S8192 | 813 | 678 | 1.20 | 1491 | 809 | 756 | 1267 | 1332 | power | 1.18 |
| M4096_N14336_K4096 | B64_S8192 | 1477 | 1348 | 1.09 | 2825 | 1427 | 1472 | 2401 | 2467 | power | 1.18 |
| M4096_N14336_K4096 | B16_S32768 | 1477 | 1348 | 1.09 | 2825 | 1427 | 1471 | 2406 | 2473 | power | 1.17 |
| M8192_N4096_K4096 | B128_S2048 | 813 | 682 | 1.19 | 1495 | 809 | 757 | 1274 | 1339 | power | 1.17 |
| M8192_N4096_K14336 | B128_S8192 | 3015 | 2670 | 1.13 | 5684 | 2854 | 2872 | 4855 | 4920 | power | 1.17 |
| M8192_N14336_K4096 | B128_S8192 | 2979 | 2670 | 1.12 | 5648 | 2854 | 2872 | 4832 | 4897 | power | 1.17 |
| M8192_N4096_K14336 | B32_S32768 | 3015 | 2668 | 1.13 | 5682 | 2854 | 2871 | 4880 | 4946 | power | 1.16 |
| M4096_N4096_K4096 | B32_S2048 | 392 | 176 | 2.23 | 568 | 392 | 225 | 489 | 554 | power | 1.16 |
| M8192_N14336_K4096 | B32_S32768 | 2979 | 2668 | 1.12 | 5646 | 2854 | 2871 | 4857 | 4922 | power | 1.16 |
| M2048_N4096_K14336 | B16_S8192 | 759 | 343 | 2.21 | 1102 | 690 | 445 | 959 | 1026 | power | 1.15 |
| M2048_N4096_K14336 | B64_S2048 | 759 | 344 | 2.21 | 1103 | 690 | 445 | 964 | 1030 | power | 1.14 |
| M2048_N14336_K4096 | B16_S8192 | 719 | 343 | 2.10 | 1062 | 690 | 445 | 931 | 998 | power | 1.14 |
| M2048_N14336_K4096 | B64_S2048 | 719 | 344 | 2.09 | 1063 | 690 | 445 | 935 | 1002 | power | 1.14 |
| M8192_N4096_K4096 | B16_S8192 | 813 | 343 | 2.37 | 1156 | 785 | 429 | 1027 | 1093 | power | 1.13 |
| M4096_N4096_K14336 | B32_S8192 | 1496 | 678 | 2.21 | 2174 | 1379 | 818 | 1938 | 2004 | power | 1.12 |
| M8192_N4096_K4096 | B64_S2048 | 813 | 344 | 2.36 | 1157 | 785 | 430 | 1032 | 1097 | power | 1.12 |
| M4096_N4096_K14336 | B128_S2048 | 1496 | 682 | 2.19 | 2178 | 1379 | 819 | 1945 | 2010 | power | 1.12 |
| M4096_N14336_K4096 | B32_S8192 | 1477 | 678 | 2.18 | 2154 | 1379 | 818 | 1931 | 1997 | power | 1.12 |
| M4096_N14336_K4096 | B128_S2048 | 1477 | 682 | 2.17 | 2159 | 1379 | 819 | 1938 | 2004 | power | 1.11 |
| M8192_N4096_K14336 | B64_S8192 | 3015 | 1348 | 2.24 | 4363 | 2759 | 1564 | 3931 | 3996 | power | 1.11 |
| M8192_N4096_K14336 | B16_S32768 | 3015 | 1348 | 2.24 | 4363 | 2759 | 1563 | 3936 | 4001 | power | 1.11 |
| M8192_N14336_K4096 | B64_S8192 | 2979 | 1348 | 2.21 | 4327 | 2759 | 1564 | 3907 | 3973 | power | 1.11 |
| M8192_N14336_K4096 | B16_S32768 | 2979 | 1348 | 2.21 | 4327 | 2759 | 1563 | 3912 | 3978 | power | 1.11 |

**gemm × rmsnorm**: 24 shape pairs with t_a/t_b in [0.25, 4] (of 54); sorted by the speedup bound.

| A | B | t_A µs | t_B µs | t_A/t_B | T_serial | LB_tc | LB_dram | LB_power (flush) | LB_power (steady) | binding | T_serial / max(LB, max solo) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M2048_N4096_K4096 | T16384_H8192 | 202 | 362 | 0.56 | 565 | 190 | 368 | 419 | 487 | power | 1.35 |
| M2048_N4096_K4096 | T16384_H4096 | 202 | 180 | 1.13 | 382 | 190 | 204 | 291 | 359 | power | 1.31 |
| M2048_N4096_K4096 | T4096_H8192 | 202 | 88 | 2.30 | 290 | 190 | 123 | 226 | 294 | power | 1.28 |
| M2048_N4096_K4096 | T65536_H4096 | 202 | 724 | 0.28 | 926 | 190 | 695 | 677 | 744 | dram | 1.28 |
| M4096_N4096_K4096 | T65536_H8192 | 392 | 1450 | 0.27 | 1842 | 381 | 1369 | 1393 | 1460 | power | 1.27 |
| M2048_N4096_K14336 | T65536_H8192 | 759 | 1450 | 0.52 | 2209 | 666 | 1425 | 1749 | 1815 | power | 1.26 |
| M4096_N4096_K4096 | T65536_H4096 | 392 | 724 | 0.54 | 1116 | 381 | 715 | 884 | 951 | power | 1.26 |
| M2048_N14336_K4096 | T65536_H8192 | 719 | 1450 | 0.50 | 2169 | 666 | 1425 | 1720 | 1787 | power | 1.26 |
| M8192_N4096_K4096 | T65536_H8192 | 813 | 1450 | 0.56 | 2263 | 761 | 1410 | 1816 | 1882 | power | 1.25 |
| M4096_N4096_K4096 | T16384_H8192 | 392 | 362 | 1.08 | 755 | 381 | 388 | 627 | 694 | power | 1.20 |
| M2048_N4096_K14336 | T65536_H4096 | 759 | 724 | 1.05 | 1483 | 666 | 771 | 1240 | 1306 | power | 1.20 |
| M2048_N14336_K4096 | T65536_H4096 | 719 | 724 | 0.99 | 1443 | 666 | 771 | 1211 | 1278 | power | 1.19 |
| M4096_N4096_K14336 | T65536_H8192 | 1496 | 1450 | 1.03 | 2946 | 1332 | 1471 | 2487 | 2553 | power | 1.18 |
| M4096_N14336_K4096 | T65536_H8192 | 1477 | 1450 | 1.02 | 2926 | 1332 | 1471 | 2480 | 2547 | power | 1.18 |
| M8192_N4096_K4096 | T65536_H4096 | 813 | 724 | 1.12 | 1537 | 761 | 756 | 1307 | 1373 | power | 1.18 |
| M4096_N4096_K4096 | T16384_H4096 | 392 | 180 | 2.18 | 572 | 381 | 225 | 499 | 566 | power | 1.15 |
| M2048_N4096_K14336 | T16384_H8192 | 759 | 362 | 2.10 | 1122 | 666 | 444 | 982 | 1049 | power | 1.14 |
| M2048_N14336_K4096 | T16384_H8192 | 719 | 362 | 1.98 | 1081 | 666 | 444 | 953 | 1021 | power | 1.13 |
| M4096_N4096_K14336 | T65536_H4096 | 1496 | 724 | 2.07 | 2220 | 1332 | 817 | 1978 | 2044 | power | 1.12 |
| M8192_N4096_K4096 | T16384_H8192 | 813 | 362 | 2.24 | 1175 | 761 | 429 | 1050 | 1116 | power | 1.12 |
| M4096_N14336_K4096 | T65536_H4096 | 1477 | 724 | 2.04 | 2200 | 1332 | 817 | 1971 | 2038 | power | 1.12 |
| M8192_N4096_K14336 | T65536_H8192 | 3015 | 1450 | 2.08 | 4464 | 2664 | 1563 | 4010 | 4076 | power | 1.11 |
| M8192_N14336_K4096 | T65536_H8192 | 2979 | 1450 | 2.05 | 4429 | 2664 | 1563 | 3986 | 4052 | power | 1.11 |
| M2048_N14336_K4096 | T16384_H4096 | 719 | 180 | 4.00 | 899 | 666 | 281 | 825 | 893 | power | 1.09 |

</details>

## Measurement quality, flush bias, warmup, reproducibility, energy

- points: 4104; CV median 0.29%, p90 1.31%; CV > 2%: 144 (105 of them < 150 µs) by op {'gemm': 39, 'gqa_decode': 0, 'rmsnorm': 105}
- (p90−p10)/median: median 0.65%, p90 2.86%
- drift inside the timed window (median of 2nd half / 1st half): p1 0.9829, p50 1.0, p99 1.007; |ratio−1| > 1%: 63 points
- reference re-measured at the end of its batch / start: {'n': 135, 'p1': 0.994, 'p50': 1.0, 'p99': 1.003, 'min': 0.9503, 'max': 1.004, 'n_off_1pct': 1, 'worst': [('rmsnorm', 'T4096_H4096', 'torch', 94, 0.9502702702702702), ('rmsnorm', 'T4096_H4096', 'flashinfer', 48, 0.9938692098092644), ('gemm', 'M8192_N4096_K4096', 'cublas', 188, 0.9941991937862551), ('gemm', 'M4096_N4096_K4096', 'cublas', 188, 0.9944372238107617), ('gemm', 'M2048_N14336_K4096', 'cublas', 188, 1.004482430601597)]}
- contaminated (foreign GPU process during the point, after re-measurement): 0

Drift diagnostic by point type (median of 2nd half / 1st half): grid @94 and @48, decode and RMSNorm: p5–p95 within ±0.3%. GEMM grid @188 (preceded by a light @48 point): median +0.44%, p95 +0.73%. GEMM persistent @188 (preceded by the heavier grid point; the clock ramps *up* more slowly than it drops): median −0.2%, p5 −2.2% — a median bias of ≤ ~1% for ~15% of the GEMM persistent points, which are also the 39 GEMM points with CV > 2%. The other 105 points with CV > 2% are RMSNorm points < 150 µs (launch-jitter floor, `spread` median 0.65%).

**Per-point warmup study** (median vs the steady-state reference = pred 'self' with W >= 1 s; |half_ratio - 1| = drift inside the timed window; mean of 2 reps):

| point | predecessor | W=0.05 | W=0.1 | W=0.2 | W=0.3 | W=0.5 | W=1.0 | W=2.0 |
|---|---|---|---|---|---|---|---|---|
| gemm M8192_N14336_K4096 @188 | idle | -2.15% / 1.08% | -2.51% / 1.00% | -2.36% / 0.91% | -2.33% / 0.52% | -2.38% / 0.43% | -2.37% / 0.25% | -2.25% / 0.21% |
| gemm M8192_N14336_K4096 @188 | hot | -2.00% / 0.37% | -1.74% / 0.35% | -1.45% / 0.24% | -1.27% / 0.14% | -1.11% / 0.17% | -0.84% / 0.13% | -0.64% / 0.11% |
| gemm M8192_N14336_K4096 @188 | self | -0.57% / 0.10% | -0.41% / 0.13% | -0.37% / 0.07% | -0.20% / 0.09% | -0.08% / 0.07% | -0.09% / 0.13% | +0.07% / 0.14% |
| gemm M8192_N14336_K4096 @48 | idle | +0.28% / 0.00% | +0.28% / 0.01% | +0.01% / 0.26% | +0.01% / 0.00% | +0.01% / 0.01% | +0.01% / 0.01% | +0.00% / 0.01% |
| gemm M8192_N14336_K4096 @48 | hot | -0.00% / 0.26% | +0.26% / 0.01% | +0.26% / 0.00% | +0.25% / 0.00% | +0.26% / 0.00% | +0.52% / 0.00% | +0.53% / 0.00% |
| gemm M8192_N14336_K4096 @48 | self | +0.26% / 0.00% | +0.26% / 0.00% | +0.26% / 0.01% | +0.25% / 0.00% | +0.26% / 0.00% | +0.00% / 0.27% | -0.00% / 0.01% |
| gqa_decode B64_S8192 @188 | idle | +0.04% / 0.02% | +0.04% / 0.03% | +0.02% / 0.03% | +0.02% / 0.03% | +0.04% / 0.05% | -0.00% / 0.04% | +0.01% / 0.02% |
| gqa_decode B64_S8192 @188 | hot | +0.02% / 0.04% | +0.02% / 0.05% | +0.03% / 0.04% | +0.01% / 0.02% | -0.01% / 0.01% | +0.02% / 0.07% | +0.01% / 0.05% |
| gqa_decode B64_S8192 @188 | self | +0.00% / 0.01% | +0.01% / 0.04% | -0.01% / 0.01% | +0.00% / 0.07% | +0.03% / 0.05% | -0.01% / 0.01% | +0.03% / 0.04% |
| rmsnorm T16384_H4096 @188 | idle | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% |
| rmsnorm T16384_H4096 @188 | hot | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% |
| rmsnorm T16384_H4096 @188 | self | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% | +0.00% / 0.00% |



**cobench write-flush vs write+read flush** (same 2xL2 buffer written, then read: evicted *and* clean), solo-best TileLang config and best reference, 188 SMs (ratio = t_clean / t_write; delta = t_write − t_clean):

| op | shape | kernel | write µs | write+read µs | ratio | delta µs |
|---|---|---|---|---|---|---|
| gemm | M2048_N4096_K4096 | `128x128x64_s2_t256_k1_wsauto_smem_g8` | 204.3 | 202.4 | 0.990 | 1.9 |
| gemm | M2048_N4096_K4096 | `ref:cublas` | 237.2 | 233.1 | 0.982 | 4.2 |
| gemm | M2048_N14336_K4096 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | 723.9 | 688.2 | 0.951 | 35.7 |
| gemm | M2048_N14336_K4096 | `ref:cublas` | 702.0 | 661.1 | 0.942 | 40.9 |
| gemm | M2048_N4096_K14336 | `128x128x64_s2_t128_k1_wsauto_smem_g8` | 762.6 | 723.6 | 0.949 | 38.9 |
| gemm | M2048_N4096_K14336 | `ref:cublas_fp32red` | 812.6 | 790.1 | 0.972 | 22.5 |
| gemm | M4096_N4096_K4096 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 393.2 | 387.1 | 0.984 | 6.2 |
| gemm | M4096_N4096_K4096 | `ref:cublas_fp32red` | 384.7 | 370.3 | 0.963 | 14.4 |
| gemm | M4096_N14336_K4096 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | 1479.4 | 1442.8 | 0.975 | 36.6 |
| gemm | M4096_N14336_K4096 | `ref:cublas_fp32red` | 1398.3 | 1358.5 | 0.972 | 39.8 |
| gemm | M4096_N4096_K14336 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 1498.1 | 1452.2 | 0.969 | 45.9 |
| gemm | M4096_N4096_K14336 | `ref:cublas_fp32red` | 1473.2 | 1428.0 | 0.969 | 45.2 |
| gemm | M8192_N4096_K4096 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 818.2 | 784.4 | 0.959 | 33.8 |
| gemm | M8192_N4096_K4096 | `ref:cublas_fp32red` | 813.6 | 776.8 | 0.955 | 36.8 |
| gemm | M8192_N14336_K4096 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | 2994.8 | 2951.6 | 0.986 | 43.2 |
| gemm | M8192_N14336_K4096 | `ref:cublas_fp32red` | 2856.2 | 2810.8 | 0.984 | 45.4 |
| gemm | M8192_N4096_K14336 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 3024.9 | 2979.8 | 0.985 | 45.1 |
| gemm | M8192_N4096_K14336 | `ref:cublas_fp32red` | 2940.5 | 2896.6 | 0.985 | 43.9 |
| gqa_decode | B16_S2048 | `n32_h4_sp1_t64_s3` | 92.0 | 90.1 | 0.979 | 1.9 |
| gqa_decode | B16_S2048 | `ref:flashinfer_tc` | 98.2 | 96.2 | 0.979 | 2.0 |
| gqa_decode | B16_S8192 | `n128_h4_sp1_t128_s1` | 342.0 | 341.9 | 1.000 | 0.1 |
| gqa_decode | B16_S8192 | `ref:flashinfer_tc` | 352.1 | 348.0 | 0.988 | 4.1 |
| gqa_decode | B16_S32768 | `n64_h4_sp1_t128_s2` | 1345.9 | 1344.5 | 0.999 | 1.4 |
| gqa_decode | B16_S32768 | `ref:flashinfer_cc` | 1356.5 | 1352.2 | 0.997 | 4.3 |
| gqa_decode | B32_S2048 | `n64_h4_sp1_t64_s1` | 176.1 | 174.0 | 0.988 | 2.1 |
| gqa_decode | B32_S2048 | `ref:flashinfer_tc` | 177.9 | 175.7 | 0.988 | 2.1 |
| gqa_decode | B32_S8192 | `n64_h4_sp1_t128_s1` | 678.9 | 675.8 | 0.995 | 3.1 |
| gqa_decode | B32_S8192 | `ref:flashinfer_tc` | 683.6 | 679.5 | 0.994 | 4.1 |
| gqa_decode | B32_S32768 | `n64_h4_sp1_t128_s1` | 2667.5 | 2651.1 | 0.994 | 16.4 |
| gqa_decode | B32_S32768 | `ref:flashinfer_cc` | 2675.4 | 2658.8 | 0.994 | 16.6 |
| gqa_decode | B64_S2048 | `n64_h4_sp1_t128_s1` | 346.0 | 342.0 | 0.988 | 4.0 |
| gqa_decode | B64_S2048 | `ref:flashinfer_tc` | 349.9 | 347.7 | 0.994 | 2.2 |
| gqa_decode | B64_S8192 | `n64_h4_sp1_t128_s1` | 1349.7 | 1346.4 | 0.998 | 3.3 |
| gqa_decode | B64_S8192 | `ref:flashinfer_cc` | 1363.6 | 1359.4 | 0.997 | 4.2 |
| gqa_decode | B64_S32768 | `n128_h4_sp2_t128_s1` | 5269.2 | 5253.2 | 0.997 | 16.0 |
| gqa_decode | B64_S32768 | `ref:flashinfer_cc` | 5273.3 | 5256.8 | 0.997 | 16.5 |
| gqa_decode | B128_S2048 | `n64_h4_sp1_t128_s2` | 683.0 | 677.9 | 0.993 | 5.1 |
| gqa_decode | B128_S2048 | `ref:flashinfer_cc` | 684.0 | 679.9 | 0.994 | 4.1 |
| gqa_decode | B128_S8192 | `n128_h4_sp1_t128_s1` | 2669.6 | 2652.8 | 0.994 | 16.8 |
| gqa_decode | B128_S8192 | `ref:flashinfer_cc` | 2673.4 | 2655.7 | 0.993 | 17.7 |
| gqa_decode | B128_S32768 | `n64_h4_sp1_t128_s2` | 10466.0 | 10452.0 | 0.999 | 14.0 |
| gqa_decode | B128_S32768 | `ref:flashinfer_tc` | 10499.0 | 10487.0 | 0.999 | 12.0 |
| rmsnorm | T4096_H4096 | `r2_t128_v2` | 43.0 | 30.7 | 0.714 | 12.3 |
| rmsnorm | T4096_H4096 | `ref:flashinfer` | 47.0 | 32.7 | 0.695 | 14.3 |
| rmsnorm | T4096_H8192 | `r8_t512_v2` | 89.7 | 69.7 | 0.777 | 20.0 |
| rmsnorm | T4096_H8192 | `ref:flashinfer` | 90.1 | 71.7 | 0.795 | 18.4 |
| rmsnorm | T16384_H4096 | `r2_t128_v8` | 180.1 | 163.4 | 0.907 | 16.7 |
| rmsnorm | T16384_H4096 | `ref:flashinfer` | 182.2 | 163.8 | 0.899 | 18.3 |
| rmsnorm | T16384_H8192 | `r8_t512_v2` | 361.4 | 345.7 | 0.957 | 15.7 |
| rmsnorm | T16384_H8192 | `ref:flashinfer` | 362.5 | 348.1 | 0.960 | 14.4 |
| rmsnorm | T65536_H4096 | `r4_t512_v4` | 724.6 | 710.5 | 0.981 | 14.0 |
| rmsnorm | T65536_H4096 | `ref:flashinfer` | 729.0 | 711.7 | 0.976 | 17.3 |
| rmsnorm | T65536_H8192 | `r4_t512_v8` | 1451.6 | 1433.5 | 0.988 | 18.1 |
| rmsnorm | T65536_H8192 | `ref:flashinfer` | 1458.2 | 1442.8 | 0.989 | 15.4 |

- gemm: clean/write median 0.972 (min 0.942, max 0.990); delta median 37.9 µs (max 45.9)

- gqa_decode: clean/write median 0.994 (min 0.979, max 1.000); delta median 4.1 µs (max 17.7)

- rmsnorm: clean/write median 0.932 (min 0.695, max 0.989); delta median 16.2 µs (max 20.0)

**Flush kinds** (mean of 2 reps, µs):

| op | shape | config | write (cobench) | read | write+read |
|---|---|---|---|---|---|
| rmsnorm | T4096_H4096 | `r2_t128_v2` | 45.0 | 28.7 | 32.7 |
| rmsnorm | T16384_H4096 | `r2_t128_v8` | 180.1 | 163.7 | 161.8 |
| rmsnorm | T65536_H8192 | `r4_t512_v8` | 1451.5 | 1435.6 | 1432.5 |
| gqa_decode | B16_S2048 | `n32_h4_sp1_t64_s3` | 92.1 | 87.6 | 90.0 |
| gqa_decode | B64_S8192 | `n64_h4_sp1_t128_s1` | 1348.3 | 1306.4 | 1345.5 |
| gemm | M4096_N4096_K4096 | `128x256x64_s2_t256_k1_wsoff_direct_g8` | 387.9 | 393.1 | 385.6 |
| gemm | M2048_N14336_K4096 | `256x128x64_s2_t256_k1_wsoff_direct_g8` | 714.2 | 693.2 | 678.1 |

**Re-measurement at the end of the sweep** (new / original median):

| point | original µs (MHz) | re-measured µs (MHz) | ratio |
|---|---|---|---|
| gemm M8192_N14336_K4096 `256x128x64_s2_t256_k1_wsoff_direct_g8` @188 (rep 0) | 2978.8 (2125) | 2993.8 (2132) | 1.0050 |
| gemm M4096_N4096_K4096 `128x128x64_s2_t256_k1_wsauto_smem_g8` @48 (rep 0) | 1100.4 (2823) | 1100.7 (2823) | 1.0003 |
| gqa_decode B64_S8192 `n64_h4_sp1_t128_s1` @188 (rep 0) | 1348.5 (2836) | 1349.2 (2821) | 1.0005 |
| gqa_decode B128_S2048 `n64_h4_sp1_t128_s1` @94 (rep 0) | 681.7 (2836) | 684.0 (2821) | 1.0034 |
| rmsnorm T65536_H8192 `r4_t512_v8` @188 (rep 0) | 1449.9 (2834) | 1450.6 (2819) | 1.0005 |
| gemm M8192_N14336_K4096 `256x128x64_s2_t256_k1_wsoff_direct_g8` @188 (rep 1) | 2978.8 (2125) | 2976.8 (2127) | 0.9993 |
| gemm M4096_N4096_K4096 `128x128x64_s2_t256_k1_wsauto_smem_g8` @48 (rep 1) | 1100.4 (2823) | 1100.5 (2823) | 1.0001 |
| gqa_decode B64_S8192 `n64_h4_sp1_t128_s1` @188 (rep 1) | 1348.5 (2836) | 1349.2 (2820) | 1.0005 |
| gqa_decode B128_S2048 `n64_h4_sp1_t128_s1` @94 (rep 1) | 681.7 (2836) | 684.0 (2821) | 1.0035 |
| rmsnorm T65536_H8192 `r4_t512_v8` @188 (rep 1) | 1449.9 (2834) | 1450.9 (2819) | 1.0007 |

**Energy per call: flush-mode subtraction vs back-to-back (cobench graph mode)**:

| point | flush: t µs (MHz) | P_w | E_naive µJ | E_kernel µJ (P_kernel W) | graph: t µs (MHz) | P W | E = P·t µJ | E_kernel / E_graph |
|---|---|---|---|---|---|---|---|---|
| gemm M8192_N14336_K4096 `256x128x64_s2_t256_k1_wsoff_direct_g8` | 2975.7 (2126) | 601 | 1789600 | 1813800 (610) | 3031.5 (2080) | 600 | 1819300 | 0.997 |
| gemm M2048_N4096_K4096 `128x128x64_s2_t256_k1_wsauto_smem_g8` | 203.5 (2315) | 570 | 116070 | 140220 (689) | 224.9 (1986) | 600 | 134980 | 1.039 |
| gqa_decode B64_S8192 `n64_h4_sp1_t128_s1` | 1348.6 (2821) | 432 | 583220 | 578840 (429) | 1303.3 (2820) | 441 | 574130 | 1.008 |
| gqa_decode B16_S2048 `n32_h4_sp1_t64_s3` | 92.0 (2824) | 389 | 35788 | 35866 (390) | 84.8 (2822) | 433 | 36745 | 0.976 |
| rmsnorm T65536_H8192 `r4_t512_v8` | 1451.9 (2819) | 422 | 612690 | 622400 (429) | 1451.4 (2818) | 425 | 616290 | 1.010 |
| rmsnorm T4096_H4096 `r2_t128_v2` | 44.9 (2835) | 375 | 16865 | 19271 (429) | 43.8 (2830) | 436 | 19117 | 1.008 |

Reading the flush studies: with cobench's write-flush the L2 is full of dirty lines at the start of the timed op; with a write+read flush the op's data is still evicted but the L2 is clean, so up to ~128 MB of the op's output writes are absorbed by L2 and written back after the end event. RMSNorm 4096×4096 moves 64 MB; 30.7 µs is only possible because its 32 MB output stays in L2. The read-only flush does not evict completely (decode B64_S8192 gains 3% with read-only but 0.2% with write+read; RMSNorm 4096² reaches 2.2 TB/s effective), so it must not be used. Which state is "right" depends on the scenario (a producer's dirty outputs are realistic), but the choice must be the same for solo and co-run measurements, and sums of solo times double-count the write-back that one co-run rep pays once.

Warmup study reading: memory-bound points are unaffected by W and predecessor (±0.05%); GEMM @48 is not power-capped and flips between two clock bins (2824 / 2840 MHz, ±0.26%). For the power-capped GEMM @188 the within-window drift after a 2-s idle falls from 1.1% (W = 0.05 s) to 0.5% (0.3 s) and 0.2% (2 s), i.e. the clock controller needs ~0.3–0.5 s; the medians depend on W by < 0.5% but on the **thermal history** by 2.2–2.6% (after 2 s idle, at every W up to 2 s), and after the "hot" block they drift from −2.0% to −0.6% over ~25 s while the GPU re-heats. Per-point warmup cannot fix that; hence no idle gaps, the batch warmup and the interleaved order. W = 0.3 s was chosen.

## Costs

- **Compile**: 2432 kernels (1026 grid, 1026 persistent pass 1, 377 pass 2, 3 fixed-point rebuilds), 0 failures. One process per op in parallel (32 threads each): GEMM 201 s, decode 669 s, RMSNorm 98 s → **11.2 min wall**. `tilelang.par_compile` in one process is GIL-bound (~1.4 kernels/s with 48 workers, ~3 cores busy); a first single-process attempt was aborted after ~4 min. Loading all 2432 kernels from the cache: 95 s (71 s of it building the PrimFuncs in Python).
- **GPU time**: sweep 56.8 min (incl. 2.7-min global warmup and 6.0 min of batch warmups; 3196 s of batch time), warmup study 12.7 min, recheck 3 min, persistent modes 4 min, energy check 2.5 min, cuBLAS fp32-reduction 2.5 min, flush-kind / flush-bias studies ~11 min, smoke tests ~5 min: **≈ 1 h 45 min** in total.
- **Kernel cache** (`~/.tilelang/cache`): 140 MB → 1112 MB (+0.97 GB: 959 MB in namespace `git35be5243` for the 2432 kernels ≈ 0.4 MB each, 26 MB in `git87d090b4`). TileLang namespaces the cache by git HEAD, so every commit invalidates it for new processes (the studies after commit 87d090b4 recompiled their kernels; the full set takes ~11 min). `~/.cache/cotile/signatures.json`: 1.4 MB.
- **Results**: 3.1 MB (JSON + this README; nothing gitignored).

## Catalog API (`cotile/catalog.py`)

```python
from cotile import catalog
cat = catalog.load()                              # all points/*.json (default dir = this directory)
e = cat.get("gemm", "M4096_N4096_K4096")          # or a GemmShape; cat.ops(), cat.shapes(op)
e.solo_best.tag, e.solo_best.time()               # fastest grid build @188 (us)
e.solo_best.cfg()                                 # GemmConfig dataclass
e.solo_best.sig["grid"], e.solo_best.sig["persistent"]   # compiled resource signatures
e.solo_best.work                                  # flops, flops_mma, bytes_min, bytes_ws, bytes_tiles
m = e.solo_best.meas[("grid", 188)]               # Meas: median p10 p90 cv n clock_mhz cycles power_w e_kernel_uj e_naive_uj raw
e.curve(48)                                       # [(tag, us)] fastest first; budgets: e.budgets == [188, 94, 48]
e.budget_best(94), e.budget_loss(94)              # GOLDYLOC effect
e.c_lib, e.c_lib_reasons, e.c_lib_persistent, e.pareto("grid")
e.refs["cublas"][188], e.ref_best(188), e.ref_check, e.refs_end
e.persistent_penalty(), e.flush_phase()
cat.rates()                                       # measured tensor / DRAM rates
cat.pairs("gemm", "gqa_decode", ratio=(0.25, 4), use_ref=False)   # S7 rows
catalog.work(op, shape, cfg), catalog.work_min(op, shape), catalog.summary_dict(cat)
```

## Caveats and open items

- The static-persistent penalty (S3) needs a root cause before any intra-column number relies on persistent solo times; it is not flush-, occupancy- or SASS-related. The next experiment is a dynamic-queue persistent build of the same tiles in hot/graph mode.
- The write-flush bias affects every existing flush-mode number (this catalog, the FlashInfer baselines, `bench_corun`); relative comparisons within one protocol are fine.
- `LB_power` uses solo energies at solo clocks; the co-run clock will be lower.
- cuBLAS numbers at reduced SM budgets use 188-SM heuristics; FlashInfer's plan is computed for 188 SMs.
- Solo times are for the flush-mode duty cycle (~75% for GEMM): back-to-back GEMMs run at a lower power-capped clock (graph mode: 1.9% slower at 8192×14336×4096, 10.5% slower at 2048×4096×4096).
- GEMM `ws=auto` configs are grid-only; in a CoKernel they become their `ws=off` twin.
