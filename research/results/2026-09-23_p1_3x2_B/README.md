# P1 3×2 study, part B: the derived row and no-oracle robustness for GEMM × GQA decode (P1-3x2-B)

**Setup**
- Dates: 2026-09-23, 05:39–07:53 (main, r029, r058), and 2026-09-24, 03:37–04:16 (r223, second, and the r029 redo).
- Machine: RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs, 600 W cap, clocks not lockable), driver 580.173.02, torch 2.8.0+cu128.
- TileLang source tree at `abbe00ba` plus this work's uncommitted changes (below). The later commits up to `4edaeb8c` changed only research documents.
- The pairs are part A's five: main, r029, r058, r223, second (`results/2026-09-23_p1_3x2_A/README.md`).
- **Protocol:** as in part A.
  - Steady mode is primary (`cobench.bench_steady`: back-to-back, input rotation > 2×L2, thermal-plateau warmup, interleaved slices, speed-ups against the `serial` variant of the same run).
  - The clean-flush mode is used only to screen.
  - Every T in the tables comes from one final interleaved steady run per pair (stage F): serial, the solo runs, part A's four winners re-measured, the derived winners and their ablations.
- **GPU sharing.**
  - 2026-09-23 05:50: the run was paused at the user's request (other people needed the machine).
  - 06:19: resumed under a strict policy (any foreign compute process blocks us).
  - 07:10: qzr's RL job and retriever service took GPU memory (~45 GB, no SM activity). Under the strict policy, r223 yielded before its first point.
  - 07:41: resumed under the rule the user then clarified: only foreign SM activity blocks us, plus a free-memory floor (8 GiB).
  - 07:48: qzr's job started computing during r223's final run (stage FQ, one interleaved steady run). The run was discarded as contaminated, and the study yielded. The GPU stayed occupied (SM activity, down to 0.1–5 GB free) for 2 h, so `run_guarded.py` stopped at 09:48 as the policy requires. `second` had not started.
  - 2026-09-24 03:34: only qzr's ray worker held memory (32.5 GB, 0% SM), which does not count as occupied. By 03:36 the GPU was empty. r223, second and the r029 redo ran from 03:37 to 04:16 with no foreign process present.
  - No point in the tables was measured while a foreign process showed SM activity (`guard_log.jsonl`; every stage JSON carries its guard record, and all are clean).
  - Our footprint was 1.6–5.2 GB (nvidia-smi at the start of each run).

## Code

**TileLang: L2 eviction hints for cp.async** (`T.copy(..., eviction_policy=...)`).
- Before, `eviction_policy` was honoured only for TMA copies and silently ignored for cp.async and normal copies.
- `Copy::LowerCPAsync` (`src/cuda/op/copy.cc`) wraps the injected cp.async in a new AttrStmt `tl.cp_async_l2_eviction_policy` (`src/op/builtin.h`).
- The CUDA codegen emits `tl::cp_async_gs[_conditional]_l2hint<N, tl::L2EvictionPolicy::EVICT_FIRST|EVICT_LAST>` (`src/cuda/codegen/codegen_cuda.{h,cc}`).
- That is `cp.async.cg.shared.global.L2::cache_hint` with a `createpolicy.fractional` policy (fraction 1.0) (`src/tl_templates/cuda/copy.h`).
- In SASS the LDGSTS memory descriptor carries the policy: `desc[URx]` is built from 0x12F0… / 0x14F0… instead of the default descriptor `c[0x0][0x358]`.
- A requested hint that cannot be honoured logs a warning. The CuTeDSL backend fails loudly.
- Generated sources of hinted kernels equal their unhinted twins except for the call names.

**ptxas 12.9 miscompile, found and worked around.**
- With the 32-bit shared address that `cp_async_gs` uses, ptxas (sm_120 and sm_120a) sometimes folds a uniform shared-memory base into the LDGSTS address (`[Rx+UR0]`).
- It then reads the policy descriptor from an odd uniform register that no instruction writes (`desc[UR1]`), and the kernel dies with "illegal instruction".
- Affected: 15 of 43 GEMM configs of 4096³, grid and persistent builds. Unhinted kernels are never affected.
- Reproduced standalone with nvcc. Independent of createpolicy vs a constant policy, `.ca`/`.cg`, `-O1`…`-O3`, and sm_120 vs sm_120a.
- The hinted template passes a 64-bit shared-window address (`cvta.to.shared.u64`) instead. All 332 GEMM/decode kernels (every config, grid and persistent, both policies) then have valid descriptors.
- `cotile.resources.invalid_memory_descriptors(sass)` lints for the pattern. The study lints every kernel it launches, and the tests check it.
- Cost: about +5% static instructions in a hinted GEMM main loop (per-copy address registers instead of immediate offsets). The solo runs below show no measurable time cost.

**TileLang kernel cache: the lib stamp now covers the template headers** (`tilelang/cache/build_stamp.py`, used by `kernel_cache.py` and `cuda_binary_cache.py`).
- Generated sources only `#include` the `tl_templates` headers. An edit to a header (as above) used to leave stale cubins reachable under both cache levels. Their keys depend only on the generated source and, with `TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1`, on the native libraries' hash.
- The stamp now also hashes every header below `tl_templates`: relative path plus content, machine-independent. Verified: a copied tree gives the same stamp, and a one-line edit changes it.
- **Stale kernels:** after the template fix, every cache entry created since the rebuild (all kernels compiled against the buggy header) was deleted. The new stamp then made every older entry unreachable, so all kernels measured after 06:19 on 09-23 were compiled against the final headers. `research/env.sh`'s comment is updated.

**cotile.**
- New op config axes:
  - `DecodeConfig.kv_l2` for the K/V loads;
  - `GemmConfig.ab_l2` for the A/B operand loads.
  - Both take `normal | evict_first | evict_last`, with tag suffix `_l2ef` / `_l2el`. They are bitwise-neutral: `numerics()` keys are unchanged.
- `test_ops.run_l2_hints` passes 32/32. It covers 8 base configs, including split-K/KV and TMA-WS, × 2 policies × {grid, persistent}. It checks:
  - the source equals the twin's modulo the hint;
  - every K/V (A/B) load is hinted and nothing else is;
  - there are no invalid SASS descriptors;
  - GPU outputs are bitwise equal to the twin's and within tolerance of the fp32 reference.
- `test_cokernel` has a hinted pair `gd_l2`: 20/20 launchable settings bitwise equal, exactly once, clean counters. It also records `invalid_mem_desc` per CoKernel.
- `cotile/README.md` documents all of this.

**cobench GPU-sharing policy** (research/rules.md 7 as clarified on 2026-09-23; `cobench/guard.py`).
- `GuardPolicy()`:
  - **Occupied** = a process that is not ours (not this process tree, not this user) shows SM activity in pmon. Processes that only hold memory do not count.
  - **Waiting:** wait 30 min, then re-check; `GpuBusy` after 2 h blocked.
  - `min_free_mib` also waits while too little memory is free.
  - `yield_to_caller=True` raises `GpuYield` between measurement points and after a contaminated point, instead of waiting in-process.
- `GuardPolicy.strict()` counts any foreign compute process, idle or not. It was the default from 06:19 to 07:37 on 09-23. `GuardPolicy.legacy()` is the 2026-09-22 rule.
- `run_guarded.py` holds no CUDA context. It waits (with `--min-free-gib 8`), relaunches the resumable study after a yield (exit code 75), and exits with code 3 after 2 h blocked.
- `p1b_study.py` saves every flush group (one measurement point) as soon as it is measured.
- `p1_common.wait_gpu` (plus the study's own free-memory check) and `cotile/tests/harness.wait_for_gpu` use the policy.
- `test_cobench.py::test_13_guard_policy` checks the decision and wait logic with mocked process lists, pmon activity, free memory and clock. test_9 was adapted.

**Scripts** (`research/bench/scripts/`).
- `p1b_study.py`: stages DI1–DI4, DC1–DC4, F, FQ, R.
- `p1b_report.py`: `tables.json`, `tables.md`. Besides the 3×2 tables it lists every same-run pair of a hinted variant and its unhinted twin at identical knobs, and it counts runs superseded by a `--redo` in the GPU time.
- `run_guarded.py`.
- `p1_common.StatePool`: split workspaces and counters are shared by every launcher and CoKernel, so the footprint stays at one buffer per shape.

## B1 — the derived space v0

C_derived(X | partner Y) = C_lib(X) ∪ contract-conditioned variants of X. A contract is what the partner and the orchestration leave X: an SM share, the per-CTA shared memory and register budget for k co-resident CTAs, and a common thread count.

- **(a) Every op-library config that fits a contract.**
  - `cotile/ops/*.configs(shape)`: 43 GEMM and 36 decode configs per shape.
  - These include split-K GEMM (k = 2, 4), split-KV decode (2–16) and fewer heads per CTA, i.e. every tile granularity the library has. C_lib contains none of the split configs: they are dominated in solo time and have the same resource vectors.
  - SM-share contract (SM binding, green partition): every config fits (1 CTA/SM). Which one is best depends on the share, so all of them are candidates.
  - CTA co-residence contract (one GEMM and one decode CTA on one SM): both roles ≤ 48 KB of shared memory.
- **(b) A register-capped axis for the co-residence contract.**
  - `Orch.min_blocks_per_sm = 2`, i.e. `__launch_bounds__(threads, 2)`: ≤ 128 registers at 256 threads, ≤ 255 at 128.
  - A capped variant enters C_derived iff its compiled CoKernel fits 2 CTAs/SM with the cap and not without it. Configs that fit without the cap were already part A's CTA-binding space.
  - The thread-alignment re-tiling of the decode from proposal §5.3 ("256 threads → twice the heads per CTA") is not expressible with the current tile program: heads per CTA ≤ the GQA group of 4. The library's 256-thread decode configs split the KV block across 8 warps instead.
- **(c) Co-location-specific L2 eviction priorities.**
  - Decode K/V loads `evict_first`: the decode reads K/V exactly once, so the hint has no solo value (measured below). In a co-run it keeps the stream from evicting the partner's L2-resident data.
  - GEMM A/B loads `evict_last`: protect the operand panels that the GEMM re-reads many times.
  - Solo tuning is indifferent to both, so it never selects them.
- **(d) Not done:** role-local warp specialization for the GEMM role (optional in the task).

**Rows.** A variant is "lib" iff both configs are C_lib members (in the intra column, their ws=off twins), with no L2 hint and no register cap. Otherwise it is "derived". T[derived,c] is the best over every variant of column c measured in F, since C_derived ⊇ C_lib. T[lib,c] is the better of part A's lib winner and the best lib-row variant of this search, both re-measured in F.

## Search protocol (B2)

**main: full search per column.**
1. Flush screen, factorized:
   - GEMM axis: every GEMM config, with the decode fixed to part A's lib winner.
   - Decode axis: every decode config × {normal, evict_first}, with the GEMM fixed.
   - Both × 5 splits around part A's optimum.
   - Inter = green split: 575 variants. Intra = SM binding, dynamic, takeover: 555 variants, plus 64 register-capped CTA-binding variants.
2. Flush: top-4 GEMM × {normal, evict_last} × top-4 decode × {normal, evict_first} × the 3 best splits, 192 per column.
3. Steady confirmation: top-8, 4 random, the hint-flipped twins of the top-2, part A's anchors, and the 2 best register-capped CTA variants.
4. Steady refinement: split ±2…6 (green) or ±4/8/12 (CoKernel), takeover flipped, chunk (1,2)/(1,4)/(2,1), hint ablations.
5. F: re-measures the top-3 derived and top-2 lib candidates per column together.
   - The same variant measured about 1.5% differently between two steady stages while serial moved 0.2%: the power-capped operating point shifts with the thermal state. Only within-run comparisons are used.

**Other pairs: reduced protocol (FQ), as the task allows.** Main's full search found no derived axis beyond the decode K/V hint, and a derived gain < 3% (2.2% inter, 1.6% intra).
- FQ confirms that axis on part A's winners, both columns, in one interleaved steady run (full protocol). Per column it includes:
  - part A's solo and lib winners, each with and without the hint;
  - the hinted lib winner at the split refinement part A gave the lib winner (±4/8/12 CoKernel, ±2…8 green, within 8…180 GEMM SMs), with the unhinted twins at ±8;
  - for a CTA-binding lib winner (second), the hinted and unhinted winner at ±47 decode CTAs instead;
  - the hinted lib winner with chunk variants (SM-binding CoKernel) and with GEMM `evict_last` added;
  - for second, whose lib-intra winner is CTA binding, part A's best SM-binding variant ± hint as well.
- Timing builds of the intra lib winner ± hint give role completion times.
- **r029 redo.** r029's first FQ run (2026-09-23 06:48) used an earlier FQ without the split refinement and the chunk variants. It was redone with the final FQ on 2026-09-24 (`--redo`, FQ then R). The tables use the redo; the first run is kept in `r029/study_FQv0.json`.
  - The redo reproduced the first run 21 h later: serial 1709.4 vs 1709.6 µs, and every variant measured in both within 0.2% (e.g. the hinted CoKernel at 120 SMs 1374.4 vs 1374.5 µs, the hinted green split at 140 1381.1 vs 1381.0 µs). R agreed within 0.1%.
  - The added refinement changed nothing material: the hinted CoKernel at 116 SMs (1374.0 µs) ties 120 SMs.

## Summary

Measured: all five pairs. Steady mode. The ratio in parentheses is T_serial / T. Per-pair detail tables (every cell's configs, solo rank/slowdown, MHz, W, mJ; ablations; hinted/unhinted pairs; solo runs; role times; robustness curves) are in [`tables.md`](tables.md).

| pair | T_serial | T[solo,inter] | T[lib,inter] | T[derived,inter] | T[solo,intra] | T[lib,intra] | T[derived,intra] | lib/derived inter | lib/derived intra | T_inter*/T[derived,intra] | T_inter*/T[lib,intra] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 726.1 | 592.8 (×1.225) | 586.5 (×1.238) | 573.7 (×1.266) | 591.4 (×1.228) | 589.5 (×1.232) | 580.1 (×1.252) | 1.022 | 1.016 | 0.989 | 0.973 |
| r029 | 1709.4 | 1461.6 (×1.170) | 1441.3 (×1.186) | 1381.1 (×1.238) | 1435.6 (×1.191) | 1435.6 (×1.191) | 1374.0 (×1.244) | 1.044 | 1.045 | 1.005 | 0.962 |
| r058 | 1060.4 | 797.3 (×1.330) | 797.3 (×1.330) | 733.6 (×1.446) | 786.2 (×1.349) | 786.2 (×1.349) | 744.7 (×1.424) | 1.087 | 1.056 | 0.985 | 0.933 |
| r223 | 571.4 | 503.9 (×1.134) | 503.4 (×1.135) | 493.8 (×1.157) | 504.7 (×1.132) | 504.7 (×1.132) | 501.3 (×1.140) | 1.019 | 1.007 | 0.985 | 0.979 |
| second | 374.0 | 315.2 (×1.187) | 313.0 (×1.195) | 309.6 (×1.208) | 319.4 (×1.171) | 317.1 (×1.179) | 310.5 (×1.205) | 1.011 | 1.021 | 0.997 | 0.976 |

Winners (knobs; configs in `tables.md`). Every derived winner is a C_lib config pair plus decode K/V `evict_first`, at the knobs below.

| pair | T[lib,inter] | T[derived,inter] | T[lib,intra] | T[derived,intra] |
|---|---|---|---|---|
| main | green 156 | green 156 (other decode config) | SM 124/64 + TO | SM 116/72 + TO (124/64: 580.2 µs, tie) |
| r029 | green 148 | green 140 | SM 128/60 + TO, chunk 1/4 | SM 116/72 + TO, chunk 1/4 (120/68: 1374.4 µs, tie) |
| r058 | green 142 | green 142 | SM 128/60 + TO | SM 116/72 + TO |
| r223 | green 172 | green 172 | SM 184/4 + TO | SM 184/4 + TO |
| second | green 164 | green 156 | CTA binding, 282 CTAs (94 SMs host a decode CTA) + TO | the same, hinted |

(GEMM SMs / decode SMs; TO = takeover.)

**Headline findings.**

1. **One derived axis helps, and it is the co-location-specific one: decode K/V `evict_first`.**
   - The hinted decode is part A's lib decode config plus `kv_l2="evict_first"`, and the GEMM config is the lib winner's. The one exception is main's green winner, which uses another C_lib decode config (`n64_h4_sp1_t64_s1` instead of `n32_h4_sp1_t64_s2`).
   - Its gain over the best lib variant, with everything re-measured in the same run, inter / intra: main 2.2% / 1.6%, r029 4.4% / 4.5%, r058 8.7% / 5.6%, r223 1.9% / 0.7%, second 1.1% / 2.1%.
   - **At identical knobs** (every hinted variant in F whose unhinted twin ran in the same run: 31 pairs over the five pairs, green and CoKernel, SM and CTA binding), the hint is faster in all 31:
     - main +1.6 to +2.4%;
     - r029 +1.1 to +6.0%;
     - r058 +2.6 to +8.7%;
     - r223 +0.5 to +2.0%;
     - second +0.4 to +3.2%.
     Energy per iteration is lower in all 31 as well (by 0.4–8.1%).
   - It has no solo value. The hinted decode runs 0.1–0.25% faster alone, i.e. equal: 330.1 vs 330.8 µs (B16×8192), 1305.4 vs 1306.9 µs (B64×8192), 654.8 vs 656.0 µs (B32×8192), 167.9 vs 168.3 µs and 167.7 vs 168.1 µs (B32×2048). Solo tuning is indifferent to it and would never select it.
2. **It helps both columns about equally, so it is not an effect of kernel-internal orchestration.**
   - lib/derived inter vs intra: main 1.022 vs 1.016, r029 1.044 vs 1.045, r058 1.087 vs 1.056, r223 1.019 vs 1.007, second 1.011 vs 1.021 (means 1.037 vs 1.029).
   - The green split gains more in three pairs, the CoKernel in one (second), and they tie on r029. In r058 the hinted green split overtakes the CoKernel that won part A.
   - The interaction term the proposal asks for (§4: T[lib,inter]/T[derived,inter] vs T[lib,intra]/T[derived,intra]) is ≈ 0.
3. **Every other derived axis loses** (main's full search):
   - split-K GEMM −10%;
   - split-KV decode −0.8 to −1.4%;
   - fewer heads per CTA −7%;
   - other non-C_lib tile configs −1 to −2% (flush screens);
   - the register-capped CTA co-residence contract −8% (steady);
   - GEMM A/B `evict_last` 0 to −2% (steady; in the FQ pairs, added on top of the decode hint: 0 to −1.7%).
4. **Mechanism of the hint: fewer bytes and less energy, i.e. proposal effect 4.**
   - Energy per iteration drops in every pair with the hint, in the table cells: main −2.3% (inter) / −1.7% (intra), r029 −2.6% / −1.8%, r058 −8.1% / −5.3%, r223 −2.0% / −0.7%, second −1.1% / −2.1%.
   - On the power-capped pairs the time drops by about the same fraction (r058: E 479 → 440 mJ, T 797.3 → 733.6 µs).
   - The long pole finishes earlier:
     - the decode in r029 and r058 (r058 green b-end 794 → 731 µs; CoKernel T_B r029 1430 → 1385 µs, r058 781 → 761 µs);
     - the GEMM in main, r223 and second (main CoKernel T_A 583 → 569 µs; r223 green a-end 500 → 491 µs; second CTA-binding T_A 307 → 302 µs, T_B 268 → 243 µs).
   - Consistent reading: without the hint the K/V stream (0.27–2.1 GB per iteration) evicts the GEMM's operand panels from the 128 MB L2. The GEMM then re-reads them from DRAM, which costs DRAM bandwidth (DRAM-bound r029) and DRAM energy under the 600 W cap (the other four).
   - The gain is largest where the stream is large (r058 1.1 GB, r029 2.1 GB per iteration) and smallest on the 0.27 GB streams (r223, second). It is not monotone in the byte count (r058 > r029).
   - Nsight counters are admin-only here, so L2 hit rates and DRAM bytes were not measured directly.
5. **D1 (thresholds unchanged): the full claim holds on 0 of 5 pairs, the weakened claim on 0 of 5.**
   - Only r058 passes the re-derivation threshold (T[lib,intra]/T[derived,intra] = 1.056 ≥ 1.05); r029 comes close (1.045).
   - Even on r058, T_inter*/T[derived,intra] = 0.985, because the same hint makes the green split faster still.
6. **B3: the CoKernel with dynamic tiles + takeover is much less sensitive to the SM split than a green-context partition. The hypothesis holds on all five pairs.** But with an oracle split, green is as good or better.
   - Over the 8-split sweep, the CoKernel's worst case is ×0.999–1.077. Green's is ×0.590–0.854, much slower than serial.
   - The simple proportional rule R1 (split ∝ solo times) costs the CoKernel 0–9% (regret 1.000–1.092) and green 10–49% (1.097–1.490).
   - main's tuned split transferred to the other pairs costs the CoKernel 0–5% (≤ 0.3% except r223: 1.047). Green loses 0–8%.
   - With the oracle split, green is within −0.1% to +3.9% of the CoKernel, i.e. equal or better (main 1.257 vs 1.247, r029 1.239 vs 1.240, r058 1.440 vs 1.402, r223 1.158 vs 1.115, second 1.212 vs 1.191).
   - The robustness does not depend on the hint. With part A's unhinted lib configs at every split, the CoKernel's worst sweep split is ×0.991–1.071 vs green's ×0.575–0.829.
   - The CoKernel still needs co-run-aware configs. With configs picked from solo budget curves (GOLDYLOC-style), it loses up to 11–16% at the same splits on four pairs (r029: −2% to +4%).

## Per-axis findings (B1: why each axis should help, and whether it does)

| axis (C_derived part) | why it should help | measured (main, full search unless noted) | verdict |
|---|---|---|---|
| (a) split-K GEMM (k = 2, 4) | Tile count matched to the GEMM's SM share (e.g. 4.13-wave splits); finer tail | Flush screen best ×1.254 (CoKernel) / ×1.278 (green) vs ×1.400 / ×1.418 for the best non-split config: **−10%**. The fp32 partials (128–268 MB written and read per iteration) cost DRAM traffic and energy under the power cap, and the dynamic queue with takeover already absorbs the wave quantization | no |
| (a) split-KV decode (2–16) | Decode tile granularity matched to its SM share; finer tail for takeover | Best split-KV ×1.381 vs ×1.400 unsplit (CoKernel, −1.4%); green ×1.448 vs ×1.459 (−0.8%). The combine traffic outweighs the finer tail | no (neutral) |
| (a) fewer heads per CTA (h1, h2) | More, smaller decode tiles | ×1.307 vs ×1.400 (−7%): each head block re-reads the KV group's K/V | no |
| (a) other non-C_lib tiles (256×128, 128×128×64 s3 direct, …) | Resource shape or energy per FLOP under a share | Best non-C_lib GEMM ×1.387 vs ×1.400 (CoKernel), ×1.386 vs ×1.418 (green) | no |
| (b) register cap, CTA co-residence (`min_blocks_per_sm=2`) | Lets 256-thread GEMMs (≤ 48 KB) co-reside with a decode CTA (part A: fast configs could not co-reside) | 16 GEMM×decode pairs (10 GEMM configs ≤ 48 KB × the 3 fastest small decodes) fit 2 CTAs/SM only with the cap (126–128 registers at 256 threads, no local memory). Best steady ×1.171 (617.6 µs) vs ×1.270 for SM binding in the same run: **−8%**. Small-tile GEMMs lose 7–19% solo, and co-residence runs at lower clocks (1.88–1.95 GHz vs 2.26 GHz) | no |
| (c) decode K/V `evict_first` | The stream (read once) stops evicting the partner's L2 data | **+0.4% to +8.7%** at identical knobs in all 31 same-run pairs over all five pairs, both columns, SM and CTA binding (above). No solo value | **yes** |
| (c) GEMM A/B `evict_last` | Protect the panels the GEMM re-reads | Neutral to harmful: main 581.6 vs 569.8 µs (−2.1%, CoKernel). On top of the decode hint in the FQ pairs: r029 1389.8 vs 1388.9 µs (−0.1%, CoKernel), 1396.6 vs 1396.1 µs (0%, green); r058 769.5 vs 765.6 µs (−0.5%, CoKernel), 734.5 vs 733.6 µs (−0.1%, green); r223 505.3 vs 501.3 µs (−0.8%, CoKernel), 494.2 vs 493.8 µs (−0.1%, green); second 315.8 vs 310.5 µs (−1.7%, CTA binding), 313.1 vs 311.7 µs (−0.4%, green). Likely cause: rotating inputs leave last iteration's evict_last lines in L2 without reuse, and marking the stream evict-first is what protects the panels | no |
| (d) role-local warp specialization (optional) | Would remove the intra column's ws=off handicap (part A: second pair +2.6%) | not implemented | – |

**Screening fidelity** (flush screen vs steady confirmation of top-8 + 4 random, main).
- Green: Spearman 0.55; the steady winner was the screen's #6.
- CoKernel: Spearman 0.85; the winner was the screen's #4.
- Hint effects were visible in the flush screens (CoKernel decode axis: evict_first/normal median +4.4%, p10 +0.9%, p90 +9.2% over 180 config × split pairs; green +1.3%).
- The magnitudes shrink in steady mode (+1.6% / +2.4% on main), as for every power-capped effect in part A.

## B3 — no-oracle robustness (steady)

**Setup** (stage R, one steady run per pair; LIGHT protocol, 1.0 s slices × 3 rounds).
- GEMM shares: the 8-split sweep {64, 80, 94, 108, 124, 140, 156, 172}. A1 measured both sides' solo budgets there.
- Added to the sweep: the two rule splits, and for the other pairs main's oracle splits (the transfer test).
- **Mechanisms:**
  - green: green-context partition, GEMM on SMs [0, n);
  - CoKernel: SM binding, dynamic queue, takeover, with the same SM ranges.
- **Configs per split and mechanism: the best of 3 candidates.**
  1. The budget-best C_lib pair at that split: GEMM best at n SMs, decode best at 188 − n (A1 solo curves). The CoKernel uses the GEMM's ws=off twin.
  2. The column's derived winner (hinted decode).
  3. Part A's lib winner.
  - The CoKernel mechanism here is SM binding. For a column winner with CTA binding (second's intra winners), the best SM-binding variant of the same row supplies the configs: the best hinted SM-binding CoKernel of F (derived), and part A's best SM-binding CoKernel (lib), `co_sm_dynT_c1-1_a2b1_n136`, ×1.180 in part A. This is recorded as `cta_substitutes` in R.
- **Rules**, both using only solo information and applied identically to both mechanisms:
  - R1 (proportional): n_A = 188 · t_A / (t_A + t_B), full-GPU steady solo times.
  - R2 (equal finish): argmin over n of max(t_A(n), t_B(188 − n)), with t_X the best C_lib config's solo time at n SMs (A1 budget curves, linear interpolation).
- **Transfer:** main's oracle split, per mechanism (green 156, CoKernel 124), applied to the other pairs.
- **Regret** = oracle speed-up / speed-up at the chosen split (1 = no loss). "Oracle" is the best split measured in R.

| pair | mechanism | oracle (split) | R1 split: speed-up | R2 split: speed-up | worst of the 8-split sweep | regret R1 / R2 / worst | main's split: speed-up (regret) |
|---|---|---|---|---|---|---|---|
| main | green ctx | ×1.257 (156) | 106: ×1.002 | 148: ×1.241 | ×0.753 | 1.255 / 1.013 / 1.670 | (source) |
| main | CoKernel dyn+TO | ×1.247 (124) | 106: ×1.241 | 148: ×1.205 | ×1.069 | 1.005 / 1.035 / 1.166 | (source) |
| r029 | green ctx | ×1.239 (108) | 46: ×0.832 | 94: ×1.237 | ×0.838 | 1.490 / 1.001 / 1.478 | 156: ×1.181 (1.049) |
| r029 | CoKernel dyn+TO | ×1.240 (46) | 46: ×1.240 | 94: ×1.234 | ×1.061 | 1.000 / 1.005 / 1.168 | 124: ×1.237 (1.002) |
| r058 | green ctx | ×1.440 (140) | 74: ×0.988 | 94: ×1.197 | ×0.854 | 1.458 / 1.204 / 1.686 | 156: ×1.339 (1.076) |
| r058 | CoKernel dyn+TO | ×1.402 (108) | 74: ×1.285 | 94: ×1.342 | ×1.077 | 1.092 / 1.045 / 1.302 | 124: ×1.398 (1.003) |
| r223 | green ctx | ×1.158 (172) | 136: ×1.056 | 172: ×1.158 | ×0.590 | 1.097 / 1.000 / 1.962 | 156: ×1.081 (1.071) |
| r223 | CoKernel dyn+TO | ×1.115 (172)* | 136: ×1.085 | 172: ×1.115 | ×1.001 | 1.028 / 1.000 / 1.115 | 124: ×1.065 (1.047) |
| second | green ctx | ×1.212 (156) | 108: ×0.996 | 168: ×1.149 | ×0.661 | 1.217 / 1.055 / 1.835 | 156: ×1.212 (1.000) |
| second | CoKernel dyn+TO | ×1.191 (124) | 108: ×1.135 | 168: ×1.122 | ×0.999 | 1.049 / 1.061 / 1.192 | 124: ×1.191 (1.000) |

\* r223's CoKernel optimum lies outside R's split set: part A's and F's best split is 184 GEMM SMs (F: ×1.140 at 184/4 vs ×1.126 at 180, ×1.113 at 172). Against that optimum, the CoKernel's regrets on r223 are about 2% higher than listed (factor ≈ 1.02: ×1.140 / ×1.113 within F; R measured ×1.115 at 172). Green's optimum, 172, is inside the set (F: 164–180 refined, 172 best).

Speed-up per GEMM share (best of the candidate configs; "budget" = the budget-best C_lib pair only):

main:

| mechanism | 64 | 80 | 94 | 106 | 108 | 124 | 140 | 148 | 156 | 172 |
|---|---|---|---|---|---|---|---|---|---|---|
| green | 0.753 | 0.842 | 0.941 | 1.002 | 1.002 | 1.062 | 1.169 | 1.241 | 1.257 | 1.086 |
| green (budget-best configs only) | 0.753 | 0.821 | 0.925 | 0.965 | 0.965 | 1.008 | 1.127 | 1.208 | 1.231 | 1.085 |
| co | 1.069 | 1.144 | 1.196 | 1.241 | 1.242 | 1.247 | 1.227 | 1.205 | 1.183 | 1.112 |
| co (budget-best configs only) | 0.963 | 1.031 | 1.003 | 1.061 | 1.062 | 1.107 | 1.132 | 1.122 | 1.098 | 1.111 |

r029:

| mechanism | 46 | 64 | 80 | 94 | 108 | 124 | 140 | 156 | 172 |
|---|---|---|---|---|---|---|---|---|---|
| green | 0.832 | 1.017 | 1.158 | 1.237 | 1.239 | 1.237 | 1.239 | 1.181 | 0.838 |
| green (budget-best configs only) | 0.789 | 1.017 | 1.044 | 1.172 | 1.163 | 1.150 | 1.173 | 1.180 | 0.838 |
| co | 1.240 | 1.061 | 1.168 | 1.234 | 1.230 | 1.237 | 1.197 | 1.135 | 1.071 |
| co (budget-best configs only) | 1.049 | 1.045 | 1.095 | 1.158 | 1.150 | 1.184 | 1.170 | 1.099 | 1.063 |

r058:

| mechanism | 64 | 74 | 80 | 94 | 108 | 124 | 140 | 156 | 172 |
|---|---|---|---|---|---|---|---|---|---|
| green | 0.854 | 0.988 | 1.027 | 1.197 | 1.339 | 1.397 | 1.440 | 1.339 | 0.942 |
| green (budget-best configs only) | 0.801 | 0.866 | 0.946 | 1.095 | 1.096 | 1.147 | 1.314 | 1.308 | 0.942 |
| co | 1.150 | 1.285 | 1.308 | 1.342 | 1.402 | 1.398 | 1.332 | 1.223 | 1.077 |
| co (budget-best configs only) | 1.043 | 1.103 | 1.135 | 1.141 | 1.229 | 1.312 | 1.272 | 1.191 | 1.062 |

r223:

| mechanism | 64 | 80 | 94 | 108 | 124 | 136 | 140 | 156 | 172 |
|---|---|---|---|---|---|---|---|---|---|
| green | 0.590 | 0.731 | 0.813 | 0.900 | 0.931 | 1.056 | 1.061 | 1.081 | 1.158 |
| green (budget-best configs only) | 0.590 | 0.731 | 0.813 | 0.878 | 0.931 | 0.994 | 0.995 | 1.081 | 1.135 |
| co | 1.001 | 1.035 | 1.055 | 1.062 | 1.065 | 1.085 | 1.073 | 1.057 | 1.115 |
| co (budget-best configs only) | 0.913 | 0.981 | 0.938 | 0.993 | 1.011 | 1.023 | 1.023 | 1.014 | 1.102 |

second:

| mechanism | 64 | 80 | 94 | 108 | 124 | 140 | 156 | 168 | 172 |
|---|---|---|---|---|---|---|---|---|---|
| green | 0.661 | 0.801 | 0.897 | 0.996 | 1.006 | 1.172 | 1.212 | 1.149 | 1.079 |
| green (budget-best configs only) | 0.661 | 0.792 | 0.885 | 0.982 | 0.990 | 1.133 | 1.195 | 1.139 | 1.073 |
| co | 0.999 | 1.084 | 1.107 | 1.135 | 1.191 | 1.188 | 1.176 | 1.122 | 1.099 |
| co (budget-best configs only) | 0.838 | 1.054 | 1.073 | 1.093 | 1.126 | 1.146 | 1.139 | 1.066 | 1.078 |

**Reading.**
- **Tile-level dynamic scheduling with takeover makes the split much less important.**
  - The CoKernel's curve is within 3.4% over 106–148 (main), 0.7% over 94–124 (r029), 5% over 94–140 (r058), 4.8% over 80–156 (r223) and 5% over 108–156 (second).
  - Its worst sweep split is ×0.999–1.077. A wrong split never made the co-run measurably slower than serial; at worst (r223 and second, 64 GEMM SMs) it matched serial (×1.001, ×0.999).
  - Green falls to ×0.590–0.854 at the ends of the sweep and has sharp, jagged optima. Both sides' wave quantization shows:
    - r058: 140 → ×1.440, 156 → ×1.339;
    - r223, F refinement: 172 → ×1.157, 176 → ×1.075, 180 → ×0.887;
    - second: 156 → ×1.212, 172 → ×1.079.
  - **The conclusion does not depend on the hint.** With part A's unhinted lib configs at every split, the CoKernel's worst sweep split is ×0.991–1.071 vs green's ×0.575–0.829.
- **R1 (proportional to solo times) is a poor rule for green** (regret 1.10–1.49). The decode's time barely depends on its SM count (DRAM-bound), while the GEMM's does. For the CoKernel it is fine (1.000–1.092).
- **R2 (equal finish from the solo budget curves)** is a good rule for green on main, r029 and r223 (1.000–1.013) but not on r058 (1.204) or second (1.055). There the clean-flush solo curves mispredict the DRAM- and power-bound co-run. The CoKernel's R2 regret is 1.000–1.061.
- **Transfer of main's split:**
  - CoKernel regret ≤ 1.003 on r029, r058 and second, but 1.047 on r223, whose decode is short and wants almost all SMs for the GEMM;
  - green 1.000–1.076.
- **Oracle:** with the best split, green is equal to the CoKernel or better by up to 3.9%. For r223 the CoKernel optimum lies outside R's sweep (*), and F puts green ahead by 1.5% there (×1.157 vs ×1.140). second's CTA-binding CoKernel, not part of R, ties green in F (×1.205 vs ×1.208). The CoKernel's value here is robustness, not a higher optimum.
- **The CoKernel still needs co-run-aware configs.**
  - The budget-best C_lib configs (GOLDYLOC-style choice from solo budget curves, no hint) at the same splits, relative to part A's lib configs:
    - main −14% to 0%;
    - r058 −12% to −0.4%;
    - r223 −11% to −1%;
    - second −16% to −1%;
    - r029 −2% to +4% (not systematically different).
  - Main's budget-best GEMMs at most shares are 128×128 tiles (ws=auto, as ws=off twins in the CoKernel), which are poor CoKernel roles.
  - For green, the budget-best configs are sometimes better and sometimes worse than part A's winners (−13% to +12%, no consistent sign).
- **Chunk × split interaction.**
  - r029's CoKernel dips at 64 GEMM SMs (×1.061 vs ×1.240 at 46). The candidates there use part A's decode chunk of 4: 124 decode CTAs share 128 chunks, which leaves 4 straggler chunks while the GEMM CTAs (64 SMs) are still busy.
  - FQ measured chunk directly at part A's split:
    - r058, hinted, 128/60: chunk 1/1 765.6 µs, 1/2 811.1 µs, 1/4 971.7 µs (+27%);
    - r029: 1/1 1390.8, 1/2 1448.4, 1/4 1388.9 µs;
    - r223, 184/4: 1/1 501.3, 1/2 510.4, 1/4 535.5 µs.
  - Chunk has to be chosen with the split and the tile count; it is not monotone.
- **The hint lifts both mechanisms' curves:**
  - green +0.2% to +8.8%;
  - CoKernel +0.1% to +7.4%.
  - This compares the derived candidate with part A's candidate at the same split. The two differ only in the hint, except main's green, which also uses another C_lib decode config.

## B4 — D1 interim readout (plan §2.8, thresholds unchanged)

| pair | T_inter*/T[derived,intra] (≥1.10) | T[lib,intra]/T[derived,intra] (≥1.05) | full claim | T_inter*/T[lib,intra] (≥1.10) | weak claim |
|---|---|---|---|---|---|
| main | 0.989 | 1.016 | no | 0.973 | no |
| r029 | 1.005 | 1.045 | no | 0.962 | no |
| r058 | 0.985 | 1.056 | no | 0.933 | no |
| r223 | 0.985 | 1.007 | no | 0.979 | no |
| second | 0.997 | 1.021 | no | 0.976 | no |

**Against plan §2.8.**
- **Continue (full claim)** needs T_inter*/T[derived,intra] ≥ 1.10 **and** T[lib,intra]/T[derived,intra] ≥ 1.05 on ≥ 2 pairs. It holds on **0 of 5** pairs.
  - The re-derivation condition alone holds on r058 (1.056). There the same derived axis speeds up the inter-kernel baseline even more, so T_inter* moves with it: 0.985.
  - T_inter*/T[derived,intra] is 0.985–1.005 on all five pairs, i.e. the best kernel-internal variant and the best inter-kernel variant are within 1.5%.
- **Continue (weakened claim)** needs T_inter*/T[lib,intra] ≥ 1.10 on ≥ 2 pairs. It holds on **0 of 5** pairs (0.933–0.979). The derived row makes T_inter* smaller, so this ratio fell below part A's (0.987–1.013).
- **The derived row found a real, co-location-specific implementation effect**: an L2 policy with no solo value, worth 0.4–8.7% at identical knobs. It supports proposal §1.2's thesis that resource needs belong to the implementation (effect 4: bytes and energy under a shared cap).
  - It does **not** support the core claim that this value needs, or is amplified by, kernel-internal orchestration. It applies equally to a green-context split.
- **Kernel-internal orchestration** (dynamic tiles + takeover) has no advantage at the oracle split here. Its measured advantage is robustness to the split choice (B3), the direction the main agent's 04:55 note anticipated.
- This covers the P1 operator pair only. D1 is decided on P1–P4.

## Conclusions

On P1 (GEMM × GQA decode, five shapes, 600 W power-capped RTX PRO 6000):

1. **A co-location-only implementation choice exists and is worth 0.4–8.7%, but it is not specific to kernel-internal orchestration.** Decode K/V `evict_first` is invisible to solo tuning (±0.25% solo). It speeds up every co-run in which it was measured against its unhinted twin (31 of 31 same-run pairs, with lower energy in all 31). It lifts the green-context partition as much as the CoKernel (lib/derived 1.011–1.087 inter vs 1.007–1.056 intra).
   - No other derived axis helped in main's full search: split-K/KV, fewer heads, register-capped co-residence and GEMM `evict_last`.
   - The other four pairs were checked only for the hint axis (reduced protocol).
2. **With the derived row, the kernel-internal and inter-kernel optima are within 1.5%** (T_inter*/T[derived,intra] 0.985–1.005). The lib-level comparison moved against the CoKernel (T_inter*/T[lib,intra] 0.933–0.979, was 0.987–1.013 in part A).
3. **D1 on P1: full claim 0/5, weak claim 0/5** (thresholds unchanged). The re-derivation threshold alone is met on r058 (1.056), and r029 falls just short (1.045).
4. **What tile-level dynamic orchestration with takeover does buy is robustness to the partition.**
   - Worst sweep split ×0.999–1.077 vs ×0.590–0.854 for green.
   - Proportional-rule regret 1.000–1.092 vs 1.097–1.490.
   - It holds with and without the hint.
   - At the oracle split, green matches or beats it by up to 3.9%.
   - Whether that robustness can carry a paper claim depends on whether a no-oracle setting is in scope: an unknown or changing partner, or online co-location without per-pair split tuning. That is a decision for D1, not something these data establish.

## Observations and surprises

- **ptxas 12.9 miscompiles cp.async with a cache policy on sm_120 in a third of GEMM configs** (above). It is silent at compile time and faults at run time. Anyone adding cache hints on this toolchain needs the 64-bit-address form or a lint.
- **Reproducibility across days.** The r029 redo, 21 h after the first run and after a machine idle period, reproduced every common variant within 0.2% and the robustness curve within 0.1%.
- **Cross-stage drift within a session.** On power-capped pairs the same variant measured 1.5% slower in one steady stage than in the preceding one, while serial moved 0.2% (main DI3 → DI4). Speed-ups normalized by each stage's serial do not remove this. Decisions were therefore taken within runs, and F re-measures the finalists together.
- **The hint shifts the best split toward the decode in some pairs:**
  - CoKernel: r029 128 → 116/120 GEMM SMs, r058 128 → 116, main 124 → 116 (a tie with 124);
  - green: r029 148 → 140, second 164 → 156;
  - unchanged: r058 green, r223, second's CTA binding.
  With the stream cheaper, the balance moves, so split and implementation have to be chosen together.
- **The hinted decode can make the GEMM role slower where the decode is the long pole** (r058 green GEMM end 598 → 624 µs, clock 2405 → 2181 MHz; r223 CoKernel T_A 456 → 463 µs). The decode's faster stream draws more DRAM power under the cap. The total is still faster.
- **r223's best CoKernel runs the decode on 4 SMs and relies on takeover**, and the hint gains least there (+0.5–0.7% intra, +0.6–2.0% green).
- **second: CTA-level co-residence plus the hint ties the best green split** (×1.205 vs ×1.208). It is the only pair where a CTA-binding CoKernel is the intra winner.
- **Kernel-cache hazard found and fixed:** template edits were invisible to the cache keys (above).

## Caveats

- **No L2/DRAM counters.** Nsight Compute needs admin rights here. The mechanism of the hint is inferred from energy, per-op completion times and clocks, not from measured hit rates.
- **Reduced protocol on four pairs.** The full factorized search ran on main only. It found nothing beyond the hint, which is why the others ran FQ.
  - A DRAM-bound pair (r029) or a short-decode pair (r223, second) could in principle profit from an axis that main's regime hides, e.g. split-KV or register-capped co-residence. That is untested beyond main.
- **Part A anchors' configs are fixed.** T[derived,·] on the FQ pairs is "part A's lib winner + hint + refined split" (+ chunk variants for SM binding). It is a lower bound on what a full derived search would give.
- **Same-knob hint ablation of the intra derived winner.** For r029 and r058 the derived-intra winner (116 GEMM SMs) has no unhinted twin at that split in F. The same-run references are the twins at 120, 128 and 136 SMs (r029 +6.0/+3.4/+2.9%, r058 +6.0/+2.7/+2.6%).
- **r223 in B3:** the CoKernel's optimum split (184) lies outside R's split set (see *), so R understates its oracle by ≈ 2%. This does not change any B3 conclusion: green's oracle is still higher, and the CoKernel's worst case stays ≈ serial.
- **second in B3:** the CoKernel mechanism of R is SM binding. The pair's intra winners use CTA binding, so R's CoKernel candidates come from the best SM-binding variants (B3 setup). The CTA-binding winner (F: ×1.205) is not part of the split sweep.
- **The GPU was shared.**
  - On 2026-09-23 from 07:10, qzr's RL job held about 45 GB. Under the strict rule the study yielded (GpuYield before a point, process exit, memory released).
  - At 07:48 the job started computing. r223's final run was discarded as contaminated, and the study stopped after 2 h blocked (09:48).
  - On 2026-09-24 the machine had no other process during our runs.
  - See `guard_log.jsonl` and each stage's `guard` record.

## GPU time

| pair | GPU s (stage wall − guard waits − compile) |
|---|---|
| main | 1701 (wall 2137, waits 1, compile 436) |
| r029 | 968 (wall 1079 incl. 568 s of the first, superseded FQ/R run, waits 0, compile 111) |
| r058 | 535 (wall 617, waits 0, compile 82) |
| r223 | 1129 (wall 1221 incl. 521 s of the two runs cut short on 09-23, waits 0, compile 92) |
| second | 549 (wall 658, waits 0, compile 110) |
| total | 4881 s (1.36 h) |

- Test and lint runs: about 10 min on 2026-09-23 and 5 min on 2026-09-24 (the full suites, below).
- Total ≈ 1.6 h of the 4 h budget. Of that, 2026-09-24 used ≈ 37 min: r223 12 min, second 11 min, the r029 redo 9 min, tests 5 min.

## Tests (2026-09-24, on the final tree; outputs in `tests/`)

| suite | result |
|---|---|
| `python -m cotile.tests.test_ops` | pass. gemm 43/43, gqa_decode 36/36, rmsnorm 39/39 configs (grid = persistent, bitwise groups, split repeatability, counters); `run_l2_hints` 32/32 |
| `python -m cotile.tests.test_cokernel` | pass. 103/103 kernels compiled, 88 CoKernels, 204/208 knob runs ok. The 4 others are the expected non-launchable `smem="sum"` baselines of the GEMM×decode pairs (138–175 KB smem/CTA). 0 invalid memory descriptors |
| `python research/bench/tests/test_cobench.py` | 15/15 pass (test_1a…test_13) |
| `python -m pytest -q testing/python/transform/test_tilelang_transform_shared_lifetime_scope.py` | 1 passed |

## Files

| file | content |
|---|---|
| `<pair>/study.json` | every stage (flush screens with per-group guard records, steady runs, descriptors of every variant, fidelity, F, R) |
| `r029/study_FQv0.json` | r029's first FQ/R run (2026-09-23, FQ before the split/chunk refinement), superseded by the redo |
| `tables.json`, `tables.md` | B2–B4 tables (`p1b_report.py`) |
| `guard_log.jsonl` | launches, exits, yields and waits of `run_guarded.py` |
| `tests/` | summary JSONs of test_ops, test_cokernel and test_cobench (2026-09-24); the logs and CSVs of all four suites are local (gitignored) |
| `RESUME.md` | the pause note of 2026-09-23 05:50 (historical) |
| `run_<pair>.log`, `run_rest.log`, `tests/*.log`, `tests/*/*.csv` | run and test logs, test CSVs (local, gitignored) |

## Reproduce

```bash
source research/env.sh
cd research/bench/scripts
G="python run_guarded.py --log ../../results/2026-09-23_p1_3x2_B/guard_log.jsonl --"
$G python p1b_study.py main --stages DI1,DI2,DI3,DI4,DC1,DC2,DC3,DC4,F,R      # full protocol
for p in r058 r223 second; do
  $G python p1b_study.py $p --stages FQ,R                                     # reduced protocol
done
$G python p1b_study.py r029 --stages FQ --redo                                # r029 redo (one stage per wrapper
$G python p1b_study.py r029 --stages R --redo                                 #   call: a relaunch must not redo a done stage)
python p1b_report.py
```

Tests (GPU phases wait under the guard policy):
- `python -m cotile.tests.test_ops` (includes `run_l2_hints`)
- `python -m cotile.tests.test_cokernel` (includes the `gd_l2` pair)
- `python research/bench/tests/test_cobench.py --out <dir>` (without `--out` it overwrites `research/results/2026-09-23_methodology_v1/test_output.json`)
- `python -m pytest -q testing/python/transform/test_tilelang_transform_shared_lifetime_scope.py`
