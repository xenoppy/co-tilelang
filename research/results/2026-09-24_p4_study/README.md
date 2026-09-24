# P4-b: prefill attention × GQA decode — CoKernel vs FlashInfer POD vs inter-kernel co-location, same conditions

**Setup**
- Date: 2026-09-24, 06:23–09:45. RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs, 600 W cap, clocks not lockable), driver 580.173.02, CUDA 12.9, torch 2.8.0+cu128, FlashInfer 0.7.0 with the local POD stream-memset patch (P4-a).
- TileLang tree: `1fd20312` plus this task's changes (below). The TVM submodule carries the pre-existing Z3-prover change; untouched.
- Ops: causal prefill of one request (S = 2048 / 8192) and batch GQA decode (B = 16 / 64, KV = 2048 / 8192), both Hq 32 / Hkv 8 / D 128, bf16 (P4-a).
- **GPU sharing:** no foreign process appeared during any run (`guard_log.jsonl` has launches and exits only; every steady stage and every flush group has `guard.clean = true`). Our footprint was 5–12 GB.
- **Protocol** (as P1-3x2-A/B):
  - Steady mode (`cobench.bench_steady`: back-to-back, input rotation > 2×L2, thermal-plateau warm-up, interleaved slices) is primary. Clean-flush mode is used for screens and as a second view of the final runs.
  - Every T below comes from the pair's **final interleaved steady run** (stage F, 1.5 s slices × 5 rounds) that contains the TileLang serial and FlashInfer serial, the solos of both libraries, POD and every candidate.
  - Speed-ups are **against each system's own serial in the same run**: TileLang variants against the TileLang serial (TL prefill solo-best, then TL decode solo-best); POD against `serial_fi` (FlashInfer prefill, then the faster FlashInfer decode path).
- **Pairs:** the four primary pairs ran the full protocol (SA flush screen → SB derived factorized flush screen → SC steady confirmation → SD steady refinement → F → R robustness → AT attribution). The four secondary (prefill-heavy or decode-heavy) pairs ran R and a quick final run (FQ).

| criterion | status |
|---|---|
| Q1 final tables | **done** for all 8 pairs: `tables.md` (per pair: every cell, µs, × own serial, paired ratios, MHz, W, mJ/iteration, flush view, per-op ends of green/streams, per-role ends of the CoKernel timing builds) |
| Q2 (a)(b)(c) | answered below with the F and AT runs |
| Q3 D1 | below; thresholds unchanged |
| Q4 robustness | below (stage R, 8 pairs) |
| Q5 outputs, tests | this directory; `test_cokernel` and `test_ops` (prefill) on the final tree, see "Tests" |

## Headline results

Steady mode, final interleaved run. × = own serial / T. "Best CTA-level" = the better of CTA binding and tile binding (POD's policy) in our framework.

| pair (t_P/t_D) | T_serial TL / FI µs | **POD** µs (× FI serial) | T_inter* µs (×) | **best CoKernel** µs (×) | best CTA-level µs (×) | streams µs (×) | POD / best TL |
|---|---|---|---|---|---|---|---|
| P2048_B16_S2048 (1.56) | 217.5 / 246.4 | 207.4 (×1.188) | 169.4 (×1.284) green 160 | **168.6 (×1.290)** SM 140 | 179.8 (×1.209) tile | 186.9 (×1.164) | 1.231 |
| P8192_B64_S8192 (1.30) | 2870.7 / 3175.5 | 2370.5 (**×1.340**) | 2234.7 (×1.285) green 162 | **2183.8 (×1.315)** SM 148 | 2410.7 (×1.191) tile | 2701.9 (×1.062) | 1.086 |
| P2048_B16_S8192 (0.41) | 469.2 / 501.6 | 378.1 (**×1.326**) | 359.8 (×1.304) green 114 | **358.7 (×1.308)** SM 116 | 361.1 (×1.299) CTA | 403.1 (×1.164) | 1.054 |
| P2048_B64_S2048 (0.41) | 470.0 / 502.5 | 365.1 (**×1.376**) | 359.2 (×1.308) green 94 | **359.1 (×1.309)** SM 94 | 359.2 (×1.309) CTA | 423.1 (×1.111) | 1.017 |
| P8192_B16_S2048 (19.5, quick) | 1764.8 / 1917.2 | 1889.6 (×1.015) | 1743.9 (×1.012) streams | **1702.4 (×1.037)** SM 160 | 1816.5 (×0.972) CTA | 1743.9 (×1.012) | 1.110 |
| P8192_B16_S8192 (5.1, quick) | 1972.7 / 2129.6 | 1986.2 (×1.072) | 1874.8 (×1.052) green | **1802.0 (×1.095)** SM | 1952.6 (×1.010) CTA | 1907.8 (×1.034) | 1.102 |
| P8192_B64_S2048 (5.1, quick) | 1976.6 / 2127.4 | 2006.8 (×1.060) | 1874.1 (×1.055) green | **1814.3 (×1.089)** SM | 1948.5 (×1.014) tile | 1950.9 (×1.013) | 1.106 |
| P2048_B64_S8192 (0.10, quick) | 1442.1 / 1482.0 | 1347.3 (**×1.100**) | 1338.5 (×1.077) green | **1336.5 (×1.079)** SM | 1339.5 (×1.077) CTA | 1384.9 (×1.041) | 1.008 |

- POD and both serials reproduce P4-a within 0.0–0.6% (e.g. POD 206.9 → 207.4, 2357 → 2371, 377.9 → 378.1, 365.1 → 365.1 µs).
- Clock / power / energy per iteration (primary pairs; serial, POD, best CoKernel, best green):
  - P2048_B16_S2048: 126.2 mJ @ 580 W; POD 113.8 mJ @ 549 W, 2248 MHz (not power-capped, but 466 vs 350 kcycles); CoKernel 101.1 mJ @ 600 W, 2076 MHz; green 101.6 mJ @ 600 W.
  - P8192_B64_S8192: 1668 mJ; POD 1422 mJ @ 600 W, 1900 MHz; CoKernel 1310 mJ @ 600 W, 1998 MHz; green 1341 mJ. Power-capped: T ≈ E / 600 W for every co-located variant.
  - P2048_B16_S8192 / P2048_B64_S2048 (DRAM-bound, LB_dram 352 µs): CoKernel 208.9 / 215.5 mJ @ 583 / 600 W is 1.8% above LB_dram; POD 216.6 / 219.4 mJ is 7.4% / 3.5% above it.
- Per-role ends (CoKernel timing builds, device `%globaltimer`): the prefill is the long pole on the P2048_B16_S2048 winner (T_P 159, T_D 125 µs) and on P8192_B64_S8192 (2154 / 1437 µs); the decode on the two 0.41 pairs (189 / 353 µs, 281 / 354 µs).

## Q2 — answers

### (a) Does POD beat the best TileLang inter-kernel variant and/or the best CoKernel?

- **In absolute time: no, on all 8 pairs.** POD is 23.1%, 8.6%, 5.4%, 1.7% slower than the best CoKernel on the primary pairs and 0.8–11.0% slower on the secondary pairs. The best CoKernel is also never slower than T_inter* (0.0–4.0% faster).
- **Against each system's own serial: POD wins on 3 of 4 primary pairs.**
  - P8192_B64_S8192: ×1.340 vs CoKernel ×1.315 (+1.9%) and T_inter* ×1.285 (+4.3%).
  - P2048_B16_S8192: ×1.326 vs ×1.308 (+1.4%) and ×1.304 (+1.7%).
  - P2048_B64_S2048: ×1.376 vs ×1.309 (+5.1%) and ×1.308 (+5.2%).
  - It loses on P2048_B16_S2048: ×1.188 vs ×1.290 (−7.9%) and ×1.284 (−7.5%).
  - Secondary pairs: POD loses on the three prefill-heavy pairs (×1.015–1.072 vs ×1.037–1.095) and wins on P2048_B64_S8192 (×1.100 vs ×1.079).
  - On the prefill-heavy pairs POD is even *slower than the TileLang serial* (×0.934–0.993 vs TL serial).
- The two readings disagree because the own-serial ratio rewards slower solo kernels (below, (c)). The absolute comparison is the one a serving system sees.

### (b) Does CTA-level co-residence (same-SM mixing) beat SM-level partitioning on P4?

**No. It ties on the DRAM-bound pairs and loses by 6–16% where the prefill matters** (steady, F):

| pair | SM binding (CoKernel) | CTA binding (1 prefill + 1 decode CTA per SM) | tile binding (POD policy, 2 CTAs/SM) | two streams (hardware mixing) | POD (FlashInfer, CTA-level) |
|---|---|---|---|---|---|
| P2048_B16_S2048 | ×1.290 | ×1.174 (−9.0%) | ×1.209 (−6.3%) | ×1.164 | 207.4 µs (+23% abs) |
| P8192_B64_S8192 | ×1.315 | ×1.108 (−15.7%) | ×1.191 (−9.4%) | ×1.062 | 2370.5 µs (+8.6% abs) |
| P2048_B16_S8192 | ×1.308 | ×1.299 (−0.7%) | ×1.238 | ×1.164 | 378.1 µs (+5.4% abs) |
| P2048_B64_S2048 | ×1.309 | ×1.309 (0.0%) | ×1.231 | ×1.111 | 365.1 µs (+1.7% abs) |
| P8192_B16_* / B64_S2048 (quick) | ×1.037–1.095 | ×0.972–1.010 | ×0.967–1.014 | ×1.012–1.034 | +10–11% abs |

- **Why.** Co-residence needs both CTAs to fit one SM: ≤ 48 KB smem each and 128 threads, or a register cap for 256 threads.
  - The prefill's efficient tiles do not fit: 96 KB / 256 threads (`m128_n128_s1_t256` at S2048, `m256_n64_s1_t256` at S8192).
  - The small tiles cost 1.7% (S2048, LPT order) to 4.8% (S8192) solo.
  - Mixed SMs burn more energy per iteration on the power-capped pairs: P2048_B16_S2048 CTA 111.1 / tile 107.9 vs SM 101.1 mJ; P8192_B64_S8192 1554 / 1446 vs 1310 mJ. Under the 600 W cap, time follows energy.
  - On the DRAM-bound pairs every mechanism reaches the DRAM bound (358.7–361.1 µs vs LB_dram 352 µs), so the binding level does not matter.
- Register-capped co-residence (256-thread `m128_n32_s1_t256` at ≤ 128 registers, 2 CTAs/SM) was always worse than the uncapped 128-thread CTA variants in the screens (−1% to −13%) and never reached a steady stage.
- The same holds with P1's GEMM × decode (CTA binding lost to SM binding there too). P4 does not give CTA-level co-residence the advantage it was expected to.

### (c) Attribution of the POD-vs-CoKernel gap: operator characteristics vs implementation

Stage AT: one interleaved steady run per primary pair, with POD, both serials, our F winners re-measured, and POD's policy re-implemented in our framework. **POD emulation** = tile binding (POD's per-SM ticket per grab, POD's prefill:decode ratio rule, 2 CTAs/SM) with FlashInfer-like tiles:
- prefill 128×32, 128 threads, 48 KB, in FlashInfer's CTA order (`kvhead`), or in natural / LPT order;
- decode with a 16-row Q tile, 64-row KV steps, 128 threads and POD's KV split (2 at B16);
- its own serial is the TileLang serial of the same tiles.

| pair | POD µs (× own) | ours µs (×) | POD / ours | = solo effect (serial_fi / serial_TL) | × co-location effect (x_ours / x_POD) | POD emulation µs (× own), FI order | best order of the emulation µs | FI-like TL serial µs |
|---|---|---|---|---|---|---|---|---|
| P2048_B16_S2048 | 207.3 (×1.183) | 169.0 (×1.293) | 1.227 | 1.123 | 1.092 | 214.8 (×1.120) | 180.3 (LPT) | 240.5 |
| P8192_B64_S8192 | 2373.2 (×1.345) | 2182.2 (×1.315) | 1.088 | 1.112 | 0.978 | 2440.3 (×1.371) | 2412.2 (natural) | 3344.6 |
| P2048_B16_S8192 | 378.1 (×1.326) | 358.7 (×1.305) | 1.054 | 1.071 | 0.985 | 379.6 (×1.296) | 379.3 (natural) | 491.8 |
| P2048_B64_S2048 | 364.8 (×1.377) | 359.0 (×1.308) | 1.016 | 1.070 | 0.950 | 383.1 (×1.282) | 361.3 (natural) | 491.2 |

1. **Implementation parity at equal policy and tile shapes: TileLang's re-implementation of POD runs within 0–5% of FlashInfer's POD.**
   - In FlashInfer's order, POD is 3.6%, 2.8%, 0.4% and 5.0% faster than the emulation.
   - With the best of the three orders, the emulation is 13% faster (P2048_B16_S2048, LPT), 1.6% slower, 0.3% slower and 1.0% faster.
   - The remaining differences are details FlashInfer does differently: its prefill CTA packs 32 positions × the 4 heads of a KV group where ours is 128 positions × 1 head, its CTAs are non-persistent, and its decode runs in its prefill kernel.
   - So the CoKernel's absolute lead over POD is **not** TileLang tile code being faster at equal tile shapes.
2. **The absolute gap = TileLang's faster solo kernels × orchestration.**
   - **Solo effect.** The TileLang serial is 7–12% faster than FlashInfer's: prefill 16% at S2048 and 8% at S8192; decode 1–4%.
   - **P2048_B16_S2048.** The orchestration adds 9.2% in our favour: SM partition with the 96 KB prefill tile, ×1.293 vs POD's ×1.183. The best POD-policy variant in our framework reaches only ×1.215 (LPT order).
   - **The other three primary pairs.** POD's orchestration extracts 1.5–5% more *relative* overlap, and the faster solo kernels more than compensate. Net: POD is 1.6–8.8% slower in absolute time.
3. **POD's higher own-serial ratios come from the tiles, not from something only POD does.**
   - The emulation, with the same slow co-residable tiles, reaches the same range as POD (×1.120 / ×1.371 / ×1.296 / ×1.282 vs ×1.183 / ×1.345 / ×1.326 / ×1.377).
   - Its FI-like serial is 10% / 17% / 5% / 5% slower than the TileLang serial. Slower, smaller tiles leave more room to overlap.
   - On P8192_B64_S8192 the emulation has the highest own-serial ratio measured on that pair (×1.371), yet is 12% slower than our CoKernel in absolute time.
   - Own-serial ratios are therefore not comparable across implementations. This is an operator-and-tile characteristic, not an orchestration capability.
4. **Porting POD's pieces into our winner does not help.**
   - **POD's decode tile** in our SM-binding winner: 173.1 vs 169.0 µs (−2.4%), 2226.0 vs 2182.2 (−2.0%), 362.0 vs 358.7 (−0.9%). On P2048_B64_S2048 the winner already uses that tile (`n64_h4_sp1_t128_s1`).
   - **CTAs per SM.** POD's policy with our efficient tiles at 1 CTA/SM ("tile_ourtiles") runs at 202.0 / 2772.6 / 454.8 / 440.2 µs, 12–22% slower than the best 2-CTAs/SM emulation with FlashInfer-like tiles. The ticket policy needs co-residence, and the efficient prefill tiles cannot co-reside.
   - **POD's ratio rule matters inside its policy.** A 1:1 ratio costs 9%, 12% and 10% on the pairs where POD's ratio is not 1:1.
   - **Virtual decode CTAs** are not implemented in FlashInfer's POD (`TODO_AK` in `pod.cuh`), so there is nothing to port. Our CoKernels align thread counts with idle threads (Rammer-style).
5. **Tile order is the largest single knob inside POD's policy.**
   - FlashInfer's per-KV-head sawtooth vs LPT: 214.8 vs 180.3 µs on P2048_B16_S2048.
   - The best order is pair-dependent (natural wins on the three other pairs).
   - Our CoKernels' dynamic queue with LPT order and takeover is the variant that does not need this choice.

### Summary of (c)

- POD's policy and tile shapes, re-implemented in TileLang, reproduce POD within 0–5%.
- The CoKernel beats POD in absolute time because (i) TileLang's solo kernels are 7–12% faster and (ii) SM partitioning with large, efficient prefill tiles beats CTA-level mixing of small tiles wherever the prefill (tensor cores / power) matters.
- POD's larger own-serial ratios on three pairs are a property of its slower co-residable tiles.

## Q3 — D1 readout (plan §2.8, thresholds unchanged)

| pair | T_inter*/T[derived,intra] (≥1.10) | T[lib,intra]/T[derived,intra] (≥1.05) | full claim | T_inter*/T[lib,intra] (≥1.10) | weak claim |
|---|---|---|---|---|---|
| P2048_B16_S2048 | 1.005 | 1.000 | no | 1.005 | no |
| P8192_B64_S8192 | 1.023 | 1.003 | no | 1.020 | no |
| P2048_B16_S8192 | 1.003 | 1.000 | no | 1.003 | no |
| P2048_B64_S2048 | 1.000 | 1.004 | no | 0.996 | no |
| P8192_B16_S2048 (quick) | 1.024 | 1.000 | no | 1.024 | no |
| P8192_B16_S8192 (quick) | 1.040 | 1.000 | no | 1.040 | no |
| P8192_B64_S2048 (quick) | 1.033 | 1.000 | no | 1.033 | no |
| P2048_B64_S8192 (quick) | 1.001 | 1.000 | no | 1.001 | no |

- **Full claim 0/4 primary pairs (0/8 overall); weak claim 0/4 (0/8).**
- The largest kernel-internal advantage over the best inter-kernel variant is on the prefill-heavy secondary pairs: +2.4% to +4.0%. There the decode is short, the green optimum puts it on 8–16 SMs, and the CoKernel's takeover lets the decode run on more SMs and then help the prefill (e.g. P8192_B16_S2048: green ×0.999, streams ×1.012, CoKernel ×1.037).
- The secondary pairs ran the reduced protocol (R + FQ, no factorized derived search), so their T[derived,·] are lower bounds on what a full search finds.
- 3×2 cells of the primary pairs (µs, × TL serial; full table in `tables.md`):

| pair | T[solo,inter] | T[lib,inter] | T[derived,inter] | T[solo,intra] | T[lib,intra] | T[derived,intra] |
|---|---|---|---|---|---|---|
| P2048_B16_S2048 | 169.4 (×1.284) | 169.4 (×1.284) | 169.4 (×1.284) | 168.6 (×1.290) | 168.6 (×1.290) | 168.6 (×1.290) |
| P8192_B64_S8192 | 2251.9 (×1.275) | 2240.1 (×1.281) | 2234.7 (×1.285) | 2218.5 (×1.294) | 2190.4 (×1.311) | 2183.8 (×1.315) |
| P2048_B16_S8192 | 359.8 (×1.304) | 359.8 (×1.304) | 359.8 (×1.304) | 358.7 (×1.308) | 358.7 (×1.308) | 358.7 (×1.308) |
| P2048_B64_S2048 | 361.3 (×1.301) | 361.2 (×1.301) | 359.2 (×1.308) | 360.5 (×1.304) | 360.5 (×1.304) | 359.1 (×1.309) |

**Derived axes (what C_derived added over C_lib).**
- **Decode K/V `evict_first`.** Same-knob hint flips in F: +0.0/+0.1% (P2048_B16_S2048), +0.1/+0.3% (P8192_B64_S8192), +0.3/+0.3% (P2048_B16_S8192), **+1.5/+2.6%** (P2048_B64_S2048, inter/intra).
  - It helps only where the DRAM stream is the bottleneck and the partner's K/V (8–32 MB) is re-read.
  - It is much smaller than on GEMM × decode (P1: up to 8.7%), because the prefill's L2 working set is small.
- **Every other axis lost or tied** (factorized flush screens over all 29 prefill × 72 decode configs and ± hint, plus steady confirmation):
  - split-KV decode;
  - fewer heads per CTA;
  - the natural and FlashInfer tile orders: natural-order twins of SM-binding candidates −0.2 to −0.6% (P2048_B16_S8192, P2048_B64_S2048, SC). Under tile binding the order matters much more: natural vs LPT −17% on P2048_B16_S2048 (F);
  - FlashInfer-like tiles;
  - register-capped co-residence (above).
  - Non-C_lib configs, together with the hint, won two cells on P2048_B64_S2048: T[derived,intra] uses `m256_n32_s1_t256` and T[derived,inter] uses `m128_n32_s1_t128`. They beat the lib cells by 0.4% and 0.6%, i.e. less than the hint alone gives on that pair.
- T[lib,·]/T[derived,·] ≤ 1.005 on all four primary pairs.

## Q4 — no-oracle robustness (stage R, steady)

Protocol as P1-3x2-B B3.
- Prefill shares: {32, 48, 64, 80, 94, 108, 124, 140, 156, 172}, plus the rule splits and the first pair's oracle split (the transfer test).
- **green:** green-context partition. **CoKernel:** SM binding, dynamic queue, takeover.
- Configs per split and mechanism: the best of the budget-best C_lib pair (± decode hint), the solo pair (CoKernel), and the F winners' configs (primary pairs).
- **R1** = 188 · t_P / (t_P + t_D). **R2** = equal finish over the C_lib solo budget curves.
- Regret = oracle / chosen.

| pair | green: oracle (split) | green worst | green regret R1 / R2 / transfer | CoKernel: oracle (split) | CoKernel worst | CoKernel regret R1 / R2 / transfer |
|---|---|---|---|---|---|---|
| P2048_B16_S2048 | ×1.276 (156) | ×0.401 | 1.135 / 1.023 / – | ×1.283 (140) | ×1.116 | 1.019 / 1.025 / – |
| P8192_B64_S8192 | ×1.289 (168) | ×0.360 | 1.237 / 1.000 / 1.002 | ×1.316 (156) | ×1.072 | 1.069 / 1.040 / 1.002 |
| P2048_B16_S8192 | ×1.305 (108) | ×0.651 | 1.363 / 1.482 / 1.044 | ×1.309 (124) | ×1.128 | 1.118 / 1.123 / 1.003 |
| P2048_B64_S2048 | ×1.309 (94) | ×0.650 | 1.443 / 1.034 / 1.032 | ×1.309 (94) | ×1.128 | 1.023 / 1.004 / 1.006 |
| P8192_B16_S2048 | ×0.999 (180) | ×0.245 | 1.002 / 1.000 / 1.049 | ×1.036 (172) | ×1.021 | 1.002 / 1.006 / 1.006 |
| P8192_B16_S8192 | ×1.052 (172) | ×0.268 | 1.026 / 1.003 / 1.031 | ×1.093 (156) | ×1.032 | 1.001 / 1.027 / 1.005 |
| P8192_B64_S2048 | ×1.052 (172) | ×0.268 | 1.026 / 1.003 / 1.032 | ×1.084 (158) | ×1.030 | 1.000 / 1.028 / 1.000 |
| P2048_B64_S8192 | ×1.077 (80) | ×0.766 | 1.213 / 1.001 / 1.020 | ×1.079 (108) | ×1.031 | 1.060 / 1.000 / 1.004 |

- **The P1 finding replicates on all 8 pairs.**
  - The CoKernel's worst split is ×1.021–1.128 (never slower than serial). Green's is ×0.245–0.766.
  - Rule regrets: CoKernel R1 1.000–1.118 and R2 1.000–1.123; green R1 1.002–1.443 and R2 1.000–1.482.
  - The first pair's oracle split transfers with CoKernel regret ≤ 1.006; green 1.002–1.049.
- **Unlike P1, the CoKernel's oracle is ≥ green's on all 8 pairs:** equal on the balanced and DRAM-bound pairs (+0.0 to +0.5%, except P8192_B64_S8192 +2.1%) and +3.0–3.9% on the prefill-heavy pairs.
- **POD needs no split.** Its ratio comes from its tile-count rule (per-SM tickets). But its orchestration is one fixed point, and inside our framework its ratio rule and tile order each moved it by 9–16% (Q2c).
- With the budget-best C_lib pair (GOLDYLOC-style choice from solo budget curves) the CoKernel curve drops by 5–22% at its worst split per pair (e.g. P2048_B16_S2048: ×1.126 vs ×1.283 at 140). The 64-row prefill tiles that win the solo budget curves are poor co-location tiles under the power cap. The CoKernel still needs co-run-aware configs (as in P1).

## Surprises and observations

1. **Flush screening is unreliable on these pairs.**
   - Flush screen vs steady confirmation: Spearman 0.42–0.71. The steady winner's screen rank was 7, 28, 213 and 4.
   - On P2048_B16_S2048 the flush favourite for green (64-row prefill tiles, ×1.44) lost to the solo-best prefill tile in steady mode (×1.23 vs ×1.28).
   - The per-row bests added to the confirmation set (not only the screen's top-8) are what found the steady winners.
2. **POD is not power-capped on the short-prefill pair** (549 W, 2248 MHz), yet it needs 466 kcycles per iteration against 350 for the CoKernel.
3. **POD's relative gains exceed ours on three primary pairs while its absolute times are worse on all eight.** "× own serial" is not an implementation-neutral metric (Q2c).
4. **Tile order matters a lot for CTA-level policies** (FlashInfer order vs LPT: 16% on P2048_B16_S2048; natural vs LPT: 17%) and hardly at all for the SM-binding CoKernel (natural-order twins within 0.6%).
5. **CTA binding equals SM binding on the DRAM-bound pairs**, the only place where same-SM mixing is not worse (358.7 / 359.1 vs 361.1 / 359.2 µs).
6. **Takeover gives the CoKernel its largest P4 margin on the prefill-heavy pairs** (+2.4–4.0% over T_inter*): the green optimum (172–180 prefill SMs) leaves a 16–8-SM decode that finishes late, and green at 180 is no faster than serial (×0.999).
7. **The first launch of pair P8192_B64_S8192 was rejected before any measurement:** the flush-screen shortcut of 30 reps for long pairs violated cobench's ≥ 50 protocol minimum (strict check). It was fixed (always 50) and relaunched; no data was affected.

## Caveats

- **Protocol scope.**
  - Secondary pairs ran the quick protocol (R + FQ): no factorized derived screen and no SD refinement. Their cells are lower bounds.
  - The F runs of the first two pairs contain a POD emulation in TileLang's natural order (the FlashInfer-order option was added after their F stage). The AT run supersedes it for all four primary pairs.
- **POD's per-role completion times are not available** (FlashInfer kernel). Ours come from timing builds (a CTA barrier + `%globaltimer` per tile).
  - SM-binding and CTA-binding timing builds run within −0.3% to +1.0% of their untimed builds in the same F runs.
  - Tile-binding timing builds run up to +3.0% slower (P8192_B64_S8192), so their role ends are slightly pessimistic.
- **No Nsight counters** (admin-only): the power and DRAM explanations rest on NVML power, clocks, energy per iteration and LB_dram.
- **Timing-build candidates.** The CoKernel timing builds are of the stage-SC/SD best candidate. Where F re-ranked the split (e.g. P2048_B16_S2048: n140 in F vs n156 timed), the role times are for the neighbouring split.
- **GPU time accounting.** Stage time includes CPU-side variant creation and SASS linting of hinted kernels (about 1–2 min per SB stage), so real GPU-busy time is lower.
- **Register-capped CTA variants** were evaluated only in the flush screens (never top-4 of their class).

## Tests (on the final tree, 2026-09-24 09:36–09:38; outputs in `tests/`)

| suite | result |
|---|---|
| `python -m cotile.tests.test_cokernel` (full matrix) | 175/175 kernels compiled, 156 CoKernels, **371/378 knob runs ok**. The 7 others are the expected non-launchable `smem="sum"` baselines. All 72 new tile-binding runs are ok: bitwise equal to the solo persistent builds, every tile exactly once, counters clean, ratios 1:1 / 2:1 / 1:3 / 1:0 / 0:1. The P4-a matrix was 299/306, and the 7 failures are the same. |
| `python -m cotile.tests.test_ops --ops prefill_attn --no-l2` | 31/31 configs (29 library + 2 kvhead twins): correct in both builds, grid == persistent bitwise, 3 numerics groups bitwise, tile-space checks 31/31 |

Only the prefill branch of the tests changed; the other ops were not re-run.

## GPU time

| part | time |
|---|---|
| primary pairs (SA–R + AT) | 1631 + 1802 + 1614 + 1767 s |
| secondary pairs (R + FQ) | 681 + 669 + 678 + 596 s |
| total stage time (compile excluded) | 9439 s = **2.62 h** |
| tests | ≈ 2.5 min |
| aborted first launch of P8192_B64_S8192 (no measurement) | 1.3 min |

Compilation (CPU, excluded above): 1775 s ≈ 0.49 h of stage wall time across the pairs, most of it the SB factorized screens (~100 CoKernels per primary pair). The CPU-only precompile of the SA sets ran before the GPU runs.


## Code

**CoKernel: tile binding = POD's scheduling policy in a persistent kernel** (`cotile/cokernel.py`, `Orch(binding="tile")`).
- Before every grab, thread 0 takes the SM's next ticket `j = atomic_add(co_state[SMCTR + %smid], 1)` and picks the prefill role iff `j mod (kP+kD) < kP` (runtime ratio knob). FlashInfer's `pod.cuh` draws one ticket per launched CTA and each CTA runs one tile; here a persistent CTA draws one per grab.
- If the drawn role's queue is exhausted it falls back to the other role, as POD does (so `schedule="dynamic"`, `takeover=True` are required). The CTA exits when both queues are empty.
- The finished chunk is published (`done` count, `%globaltimer` end) before the next draw, so per-role counts and completion times stay exact while a CTA alternates roles. `co_out.ctas_A` counts every CTA (there is no per-CTA role).
- The per-SM ticket counters are reset by the last CTA, like the CTA binding's arrival counters.
- Existing bindings are unchanged (the new code is a separate trace-time branch).

**Prefill op: FlashInfer's CTA order as a tile-order option** (`cotile/ops/prefill_attn.py`, `order="kvhead"`).
- KV head outermost, query blocks ascending, the 4 query heads of a KV group adjacent: a per-KV-head sawtooth of tile sizes. This is how FlashInfer's single-prefill / POD kernels map blockIdx (their CTA packs 32 positions × 4 heads of one KV head; ours is 128 positions × 1 head: same MMA shape and smem).
- Bitwise-neutral like the other orders. Not added to `configs()` (the library universe of the study stays the P4-a one); used only by the POD emulation.

**Tests**
- `cotile/tests/test_cokernel.py`: tile binding for every pair at its co-resident CTA count (1 or 2–4 CTAs/SM), debug and timing builds, ratios 1:1, 2:1, 1:3, 1:0, 0:1.
- `cotile/tests/test_ops.py` / `harness.py`: kvhead twins of two prefill configs join their numerics groups (bitwise equal to their LPT twins, grid == persistent); the tile-space check accepts the per-KV-head sawtooth.
- `cotile/README.md` documents both.

**Scripts** (`research/bench/scripts/`)
- `p4_study.py`: stages SA, SB, SC, SD, F, R, AT (primary pairs) and R, FQ (secondary pairs); resumable; GPU-sharing yields through `run_guarded.py`.
- `p4_study_report.py`: `tables.md`, `tables.json`.

## Reproduce

```bash
source research/env.sh
cd research/bench/scripts
G="python run_guarded.py --log ../../results/2026-09-24_p4_study/guard_log.jsonl --"
for p in P2048_B16_S2048 P8192_B64_S8192 P2048_B16_S8192 P2048_B64_S2048; do
  $G python p4_study.py $p --stages SA,SB,SC,SD,F,R          # ~25-29 min each (incl. ~4 min compile)
done
for p in P8192_B16_S2048 P8192_B16_S8192 P8192_B64_S2048 P2048_B64_S8192; do
  $G python p4_study.py $p --stages R,FQ                     # secondary pairs
done
for p in P2048_B16_S2048 P8192_B64_S8192 P2048_B16_S8192 P2048_B64_S2048; do
  $G python p4_study.py $p --stages AT                       # attribution run
done
python p4_study_report.py
# tests
cd ../../.. && python -m cotile.tests.test_cokernel && python -m cotile.tests.test_ops --ops prefill_attn --no-l2
```

`python p4_study.py <pair> --compile-only` precompiles the grid kernels and SA's CoKernels on the CPU.

## Files

| file | content |
|---|---|
| `<pair>/study.json` | every stage: flush screens (per-group guard records), steady runs, descriptors of every variant (configs, knobs, rows, axes, CoKernel resource signatures), F / FQ with the clean-flush view and role times, R, AT |
| `tables.md`, `tables.json` | all tables (`p4_study_report.py`): summary, D1, POD vs mechanisms, per-pair final tables, the AT decomposition and runs, robustness curves, screening fidelity, GPU time, every variant of every final run |
| `guard_log.jsonl` | launches and exits of `run_guarded.py` (no waits or yields occurred) |
| `tests/cokernel/cokernel_summary.json`, `tests/ops_prefill/summary.json` | test summaries (CSVs and logs are local, gitignored) |
| `run_*.log` | run logs (gitignored) |
