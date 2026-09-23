| pair | A / B | T_serial | LB tc / dram / power | T[solo,inter] | T[lib,inter] | T[solo,intra] | T[lib,intra] | T_inter* / T[lib,intra] | lib/solo inter | lib/solo intra |
|---|---|---|---|---|---|---|---|---|---|---|
| main | M4096_N4096_K4096 / B16_S8192 | 725.9 | 386 / 388 / 673 | 593.2 (×1.224) | 586.4 (×1.238) | 592.5 (×1.225) | 590.0 (×1.230) | 0.994 | 1.012 | 1.004 |
| r029 | M4096_N4096_K4096 / B64_S8192 | 1710.9 | 404 / 1370 / 1408 | 1461.5 (×1.171) | 1441.4 (×1.187) | 1435.8 (×1.192) | 1435.8 (×1.192) | 1.004 | 1.014 | 1.000 |
| r058 | M4096_N4096_K4096 / B32_S8192 | 1061.6 | 392 / 715 / 927 | 797.4 (×1.331) | 797.4 (×1.331) | 786.9 (×1.349) | 786.9 (×1.349) | 1.013 | 1.000 | 1.000 |
| r223 | M4096_N4096_K4096 / B32_S2048 | 574.2 | 383 / 225 / 558 | 506.9 (×1.133) | 504.9 (×1.137) | 507.6 (×1.131) | 507.6 (×1.131) | 0.995 | 1.004 | 1.000 |
| second | M2048_N4096_K4096 / B32_S2048 | 375.0 | 193 / 205 / 354 | 315.8 (×1.188) | 313.3 (×1.197) | 320.2 (×1.171) | 317.3 (×1.182) | 0.987 | 1.008 | 1.009 |

### main: GEMM M4096_N4096_K4096 × decode B16_S8192

| variant | knobs | configs A / B (solo rank, solo slowdown) | steady µs | × serial (paired min–max) | MHz | W | flush µs | flush × | per-op or per-role end (µs) |
|---|---|---|---|---|---|---|---|---|---|
| serial | – | solo-best | 725.9 | 1.000 | 2653 | 580 | 731.5 | 1.000 | – |
| solo A | full GPU, back-to-back | solo-best | 430.0 | – | 2259 | 600 | – | – | – |
| solo B | full GPU, back-to-back | solo-best | 330.8 | – | 2821 | 442 | – | – | – |
| **T[solo,inter]** `gr_s_148` | green 148/40 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n128_h4_sp1_t128_s1` (#1, +0.0%) | 593.2 | ×1.224 (1.219–1.227) | 2175 | 600 | 549.3 | ×1.332 | a 591, b 506 |
| **T[lib,inter]** `gr_l_156` | green 156/32 | `128x128x64_s2_t256_wsauto_smem` (#3, +2.1%) / `n32_h4_sp1_t64_s2` (#7, +0.6%) | 586.4 | ×1.238 (1.235–1.242) | 1836 | 600 | 537.2 | ×1.362 | a 583, b 483 |
| **two streams (solo)** `st_eq_ab` | streams prio=eq order=ab | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n128_h4_sp1_t128_s1` (#1, +0.0%) | 643.4 | ×1.128 (1.127–1.129) | 2465 | 600 | 681.4 | ×1.073 | a 455, b 640 |
| **T[solo,intra]** `co_sm_dyn_c1-1_a0b0_n140` | SM-bind dyn 140/48 chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n128_h4_sp1_t128_s1` (#1, +0.0%) | 592.5 | ×1.225 (1.223–1.228) | 2183 | 600 | 546.8 | ×1.338 | T_A 584, T_B 461 (CTAs 140/48) |
| **T[lib,intra]** `co_sm_dynT_c1-1_a0b3_n124` | SM-bind dyn 124/64 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t64_s1` (#4, +0.5%) | 590.0 | ×1.230 (1.227–1.233) | 2183 | 600 | 548.9 | ×1.333 | T_A 586, T_B 464 (CTAs 124/64) |
| **static schedule (best)** `co_sm_staticT_c1-1_a0b3_n124` | SM-bind static 124/64 +takeover | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t64_s1` (#4, +0.5%) | 596.3 | ×1.217 (1.216–1.221) | 2209 | 600 | 538.6 | ×1.358 | – |
| **CTA binding (best)** `co_cta_dynT_c1-1_a3b2_g376` | CTA-bind 376 CTAs (GEMM CTA on 188 SMs, decode CTA beside it on 188) +takeover chunk 1/1 | `128x128x32_s3_t128_wsoff_smem` (#5, +7.3%) / `n64_h4_sp1_t128_s1` (#3, +0.5%) | 616.8 | ×1.177 (1.175–1.179) | 1895 | 600 | 570.9 | ×1.281 | – |

LB: tensor 386, DRAM 388, power (solo energies 258 + 146 mJ at 600 W) 673 µs. T_inter* = 586.4 µs.

### r029: GEMM M4096_N4096_K4096 × decode B64_S8192

| variant | knobs | configs A / B (solo rank, solo slowdown) | steady µs | × serial (paired min–max) | MHz | W | flush µs | flush × | per-op or per-role end (µs) |
|---|---|---|---|---|---|---|---|---|---|
| serial | – | solo-best | 1710.9 | 1.000 | 2754 | 506 | 1717.0 | 1.000 | – |
| solo A | full GPU, back-to-back | solo-best | 427.6 | – | 2269 | 600 | – | – | – |
| solo B | full GPU, back-to-back | solo-best | 1307.0 | – | 2819 | 450 | – | – | – |
| **T[solo,inter]** `gr_s_80` | green 80/108 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 1461.5 | ×1.171 (1.170–1.171) | 2804 | 600 | 1463.6 | ×1.173 | a 1364, b 1459 |
| **T[lib,inter]** `gr_l_148` | green 148/40 | `128x128x64_s2_t256_wsauto_smem` (#3, +2.1%) / `n128_h4_sp1_t128_s1` (#8, +0.2%) | 1441.4 | ×1.187 (1.187–1.187) | 2629 | 546 | 1443.5 | ×1.190 | a 572, b 1438 |
| **two streams (solo)** `st_eq_ab` | streams prio=eq order=ab | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 1627.4 | ×1.051 (1.051–1.051) | 2719 | 515 | 1628.6 | ×1.054 | a 481, b 1624 |
| **T[solo,intra]** `co_sm_dynT_c1-4_a0b0_n128` | SM-bind dyn 128/60 +takeover chunk 1/4 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 1435.8 | ×1.192 (1.191–1.192) | 2668 | 552 | 1435.6 | ×1.196 | T_A 534, T_B 1431 (CTAs 128/60) |
| **T[lib,intra]** `co_sm_dynT_c1-4_a0b0_n128` | SM-bind dyn 128/60 +takeover chunk 1/4 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 1435.8 | ×1.192 (1.191–1.192) | 2668 | 552 | 1435.6 | ×1.196 | T_A 534, T_B 1431 (CTAs 128/60) |
| **static schedule (best)** `co_sm_staticT_c1-1_a5b3_n148` | SM-bind static 148/40 +takeover | `128x64x64_s2_t128_wsoff_smem` (#16, +19.0%) / `n128_h4_sp1_t128_s1` (#8, +0.2%) | 1442.6 | ×1.186 (1.185–1.186) | 2703 | 593 | 1441.9 | ×1.191 | – |
| **CTA binding (best)** `co_cta_dynT_c1-1_a5b0_g282` | CTA-bind 282 CTAs (GEMM CTA on 188 SMs, decode CTA beside it on 94) +takeover chunk 1/1 | `128x64x64_s2_t128_wsoff_smem` (#16, +19.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 1444.0 | ×1.185 (1.184–1.185) | 2680 | 593 | 1442.5 | ×1.190 | – |

LB: tensor 404, DRAM 1370, power (solo energies 257 + 588 mJ at 600 W) 1408 µs. T_inter* = 1441.4 µs.

### r058: GEMM M4096_N4096_K4096 × decode B32_S8192

| variant | knobs | configs A / B (solo rank, solo slowdown) | steady µs | × serial (paired min–max) | MHz | W | flush µs | flush × | per-op or per-role end (µs) |
|---|---|---|---|---|---|---|---|---|---|
| serial | – | solo-best | 1061.6 | 1.000 | 2705 | 543 | 1068.0 | 1.000 | – |
| solo A | full GPU, back-to-back | solo-best | 431.0 | – | 2252 | 600 | – | – | – |
| solo B | full GPU, back-to-back | solo-best | 656.0 | – | 2818 | 454 | – | – | – |
| **T[solo,inter]** `gr_s_142` | green 142/46 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 797.4 | ×1.331 (1.330–1.332) | 2375 | 601 | 795.7 | ×1.342 | a 603, b 794 |
| **T[lib,inter]** `gr_s_142` | green 142/46 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 797.4 | ×1.331 (1.330–1.332) | 2375 | 601 | 795.7 | ×1.342 | a 603, b 794 |
| **two streams (solo)** `st_eq_ab` | streams prio=eq order=ab | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 966.8 | ×1.098 (1.097–1.098) | 2652 | 565 | 974.1 | ×1.096 | a 454, b 964 |
| **T[solo,intra]** `co_sm_dynT_c1-1_a0b0_n128` | SM-bind dyn 128/60 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 786.9 | ×1.349 (1.348–1.349) | 2301 | 600 | 785.9 | ×1.359 | T_A 596, T_B 782 (CTAs 128/60) |
| **T[lib,intra]** `co_sm_dynT_c1-1_a0b0_n128` | SM-bind dyn 128/60 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 786.9 | ×1.349 (1.348–1.349) | 2301 | 600 | 785.9 | ×1.359 | T_A 596, T_B 782 (CTAs 128/60) |
| **static schedule (best)** `co_sm_staticT_c1-1_a0b0_n124` | SM-bind static 124/64 +takeover | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 807.7 | ×1.314 (1.313–1.314) | 2344 | 600 | 805.9 | ×1.325 | – |
| **CTA binding (best)** `co_cta_dynT_c1-1_a3b0_g329` | CTA-bind 329 CTAs (GEMM CTA on 188 SMs, decode CTA beside it on 141) +takeover chunk 1/1 | `128x128x32_s3_t128_wsoff_smem` (#5, +7.3%) / `n64_h4_sp1_t128_s1` (#1, +0.0%) | 803.9 | ×1.321 (1.320–1.321) | 2079 | 600 | 804.7 | ×1.327 | – |

LB: tensor 392, DRAM 715, power (solo energies 259 + 298 mJ at 600 W) 927 µs. T_inter* = 797.4 µs.

### r223: GEMM M4096_N4096_K4096 × decode B32_S2048

| variant | knobs | configs A / B (solo rank, solo slowdown) | steady µs | × serial (paired min–max) | MHz | W | flush µs | flush × | per-op or per-role end (µs) |
|---|---|---|---|---|---|---|---|---|---|
| serial | – | solo-best | 574.2 | 1.000 | 2534 | 600 | 564.1 | 1.000 | – |
| solo A | full GPU, back-to-back | solo-best | 431.7 | – | 2248 | 600 | – | – | – |
| solo B | full GPU, back-to-back | solo-best | 168.4 | – | 2818 | 450 | – | – | – |
| **T[solo,inter]** `gr_s_172` | green 172/16 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 506.9 | ×1.133 (1.131–1.134) | 1952 | 600 | 445.4 | ×1.267 | a 505, b 470 |
| **T[lib,inter]** `gr_l_172` | green 172/16 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t64_s1` (#2, +0.0%) | 504.9 | ×1.137 (1.134–1.137) | 1957 | 600 | 441.1 | ×1.279 | a 503, b 474 |
| **two streams (solo)** `st_eq_ab` | streams prio=eq order=ab | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 509.8 | ×1.126 (1.125–1.127) | 2149 | 600 | 475.6 | ×1.186 | a 464, b 507 |
| **T[solo,intra]** `co_sm_dynT_c1-1_a0b2_n184` | SM-bind dyn 184/4 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 507.6 | ×1.131 (1.130–1.132) | 2092 | 600 | 464.9 | ×1.213 | T_A 463, T_B 501 (CTAs 184/4) |
| **T[lib,intra]** `co_sm_dynT_c1-1_a0b2_n184` | SM-bind dyn 184/4 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 507.6 | ×1.131 (1.130–1.132) | 2092 | 600 | 464.9 | ×1.213 | T_A 463, T_B 501 (CTAs 184/4) |
| **static schedule (best)** `co_sm_staticT_c1-1_a0b2_n172` | SM-bind static 172/16 +takeover | `128x256x64_s2_t256_wsoff_direct` (#1, +0.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 520.5 | ×1.103 (1.103–1.104) | 2112 | 600 | 471.0 | ×1.198 | – |
| **CTA binding (best)** `co_cta_dynT_c1-1_a3b0_g376` | CTA-bind 376 CTAs (GEMM CTA on 188 SMs, decode CTA beside it on 188) +takeover chunk 1/1 | `128x128x32_s3_t128_wsoff_smem` (#5, +7.3%) / `n32_h4_sp1_t64_s2` (#1, +0.0%) | 564.2 | ×1.018 (1.017–1.018) | 2097 | 600 | 511.3 | ×1.103 | – |

LB: tensor 383, DRAM 225, power (solo energies 259 + 76 mJ at 600 W) 558 µs. T_inter* = 504.9 µs.

### second: GEMM M2048_N4096_K4096 × decode B32_S2048

| variant | knobs | configs A / B (solo rank, solo slowdown) | steady µs | × serial (paired min–max) | MHz | W | flush µs | flush × | per-op or per-role end (µs) |
|---|---|---|---|---|---|---|---|---|---|
| serial | – | solo-best | 375.0 | 1.000 | 2576 | 597 | 379.5 | 1.000 | – |
| solo A | full GPU, back-to-back | solo-best | 228.4 | – | 2127 | 600 | – | – | – |
| solo B | full GPU, back-to-back | solo-best | 168.2 | – | 2819 | 448 | – | – | – |
| solo A, ws=off twin | full GPU | – | 234.3 (+2.6% vs solo A) | – | – | – | – | – | – |
| **T[solo,inter]** `gr_s_164` | green 164/24 | `128x128x64_s2_t128_wsauto_smem` (#2, +0.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 315.8 | ×1.188 (1.184–1.191) | 2044 | 600 | 288.1 | ×1.317 | a 313, b 296 |
| **T[lib,inter]** `gr_l_164` | green 164/24 | `128x128x64_s2_t256_wsauto_smem` (#1, -1.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 313.3 | ×1.197 (1.194–1.200) | 1917 | 600 | 281.5 | ×1.348 | a 310, b 306 |
| **two streams (solo)** `st_eq_ab` | streams prio=eq order=ab | `128x128x64_s2_t128_wsauto_smem` (#2, +0.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 329.9 | ×1.137 (1.135–1.138) | 2274 | 600 | 366.6 | ×1.035 | a 239, b 327 |
| **T[solo,intra]** `co_sm_dyn_c1-1_a2b2_n140` | SM-bind dyn 140/48 chunk 1/1 | `128x128x64_s2_t128_wsoff_smem` (#3, +4.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 320.2 | ×1.171 (1.169–1.175) | 2183 | 600 | 290.8 | ×1.305 | T_A 308, T_B 248 (CTAs 140/48) |
| **T[lib,intra]** `co_cta_dynT_c1-1_a3b2_g282` | CTA-bind 282 CTAs (GEMM CTA on 188 SMs, decode CTA beside it on 94) +takeover chunk 1/1 | `128x128x32_s3_t128_wsoff_smem` (#6, +6.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 317.3 | ×1.182 (1.180–1.186) | 1954 | 600 | 288.3 | ×1.316 | T_A 311, T_B 250 (CTAs 188/94) |
| **static schedule (best)** `co_sm_static_c1-1_a2b2_n140` | SM-bind static 140/48 | `128x128x64_s2_t128_wsoff_smem` (#3, +4.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 321.8 | ×1.165 (1.163–1.169) | 2187 | 600 | 292.4 | ×1.298 | – |
| **CTA binding (best)** `co_cta_dynT_c1-1_a3b2_g282` | CTA-bind 282 CTAs (GEMM CTA on 188 SMs, decode CTA beside it on 94) +takeover chunk 1/1 | `128x128x32_s3_t128_wsoff_smem` (#6, +6.0%) / `n64_h4_sp1_t128_s1` (#3, +0.0%) | 317.3 | ×1.182 (1.180–1.186) | 1954 | 600 | 288.3 | ×1.316 | T_A 311, T_B 250 (CTAs 188/94) |

LB: tensor 193, DRAM 205, power (solo energies 137 + 75 mJ at 600 W) 354 µs. T_inter* = 313.3 µs.

### Screening fidelity (lib-row CoKernels, C1 clean-flush screen vs C2 steady)

| pair | confirmed | Spearman | Kendall | Spearman within top-8 | steady winner | its screen rank | in top-8 | regret of screen #1 |
|---|---|---|---|---|---|---|---|---|
| main | 14 | 0.77 | 0.63 | -0.24 | `co_sm_dynT_c1-1_a0b3_n124` | 7 | True | 1.3% |
| r029 | 14 | 0.93 | 0.87 | 0.62 | `co_sm_dynT_c1-1_a5b3_n148` | 1 | True | 0.0% |
| r058 | 14 | 0.82 | 0.69 | 0.00 | `co_sm_dynT_c1-1_a0b0_n124` | 8 | True | 0.2% |
| r223 | 14 | 0.99 | 0.96 | 0.98 | `co_sm_dynT_c1-1_a0b2_n172` | 1 | True | 0.0% |
| second | 14 | 0.96 | 0.89 | 0.83 | `co_sm_dynT_c1-1_a2b1_n148` | 1 | True | 0.0% |

### Energy per iteration (steady F runs, NVML board power x time, mJ)

| pair | E_A + E_B (solo) | serial | T[solo,inter] | T[lib,inter] | T[solo,intra] | T[lib,intra] | LB_power (solo E) | LB_power,dyn (P_s removed) | best T |
|---|---|---|---|---|---|---|---|---|---|
| main | 404 | 421 | 356 | 352 | 355 | 354 | 673 µs | 646 µs | 586 µs |
| r029 | 845 | 865 | 877 | 786 | 793 | 793 | 1408 µs | 1306 µs | 1436 µs |
| r058 | 556 | 576 | 479 | 479 | 472 | 472 | 927 µs | 877 µs | 787 µs |
| r223 | 335 | 344 | 304 | 303 | 305 | 305 | 558 µs | 545 µs | 505 µs |
| second | 212 | 224 | 189 | 188 | 192 | 190 | 354 µs | 341 µs | 313 µs |

P_s (idle-but-clocked board power, main pair stage P) = 143.3 W.

### Carveout (stage K, steady, NVRTC-backend builds of the same kernels)

**main** (pairs: solo: `128x256x64_s2_t256_k1_wsoff_direct_g8` + `n128_h4_sp1_t128_s1` (smem 96+73 KB, co-residable False); coresident: `128x128x32_s3_t128_k1_wsoff_smem_g8` + `n64_h4_sp1_t64_s1` (smem 48+38 KB, co-residable True))

| pair / priority / order | default carveout µs (×) | carveout 100 µs (×) | per-op end, default (µs) |
|---|---|---|---|
| solo eq ab | 641.9 (×1.130) | 642.5 (×1.129) | a 455, b 639 |
| solo eq ba | 642.4 (×1.130) | 643.3 (×1.128) | b 639, a 455 |
| solo pA ab | 642.4 (×1.129) | 643.2 (×1.128) | a 455, b 640 |
| solo pA ba | 642.6 (×1.129) | 643.3 (×1.128) | b 640, a 455 |
| solo pB ab | 721.4 (×1.006) | 722.2 (×1.005) | a 718, b 533 |
| solo pB ba | 720.6 (×1.007) | 721.9 (×1.005) | b 532, a 718 |
| coresident eq ab | 691.5 (×1.049) | 691.6 (×1.049) | a 451, b 688 |
| coresident eq ba | 691.8 (×1.049) | 692.4 (×1.048) | b 689, a 452 |
| coresident pA ab | 692.0 (×1.049) | 692.8 (×1.047) | a 452, b 689 |
| coresident pA ba | 692.2 (×1.048) | 692.6 (×1.048) | b 689, a 452 |
| coresident pB ab | 632.5 (×1.147) | 632.9 (×1.147) | a 630, b 589 |
| coresident pB ba | 632.9 (×1.146) | 632.8 (×1.147) | b 589, a 630 |

**second** (pairs: solo: `128x128x64_s2_t128_k1_wsauto_smem_g8` + `n64_h4_sp1_t128_s1` (smem 65+39 KB, co-residable False); coresident: `128x128x32_s3_t128_k1_wsoff_smem_g8` + `n64_h4_sp1_t128_s1` (smem 48+39 KB, co-residable True))

| pair / priority / order | default carveout µs (×) | carveout 100 µs (×) | per-op end, default (µs) |
|---|---|---|---|
| solo eq ab | 328.4 (×1.143) | 328.9 (×1.141) | a 237, b 325 |
| solo eq ba | 328.4 (×1.143) | 329.0 (×1.141) | b 325, a 237 |
| solo pA ab | 328.6 (×1.143) | 329.0 (×1.141) | a 237, b 325 |
| solo pA ba | 328.6 (×1.142) | 329.1 (×1.141) | b 325, a 237 |
| solo pB ab | 378.1 (×0.993) | 378.6 (×0.992) | a 375, b 262 |
| solo pB ba | 378.1 (×0.993) | 378.4 (×0.992) | b 262, a 376 |
| coresident eq ab | 341.1 (×1.101) | 341.2 (×1.100) | a 337, b 338 |
| coresident eq ba | 340.9 (×1.101) | 341.1 (×1.101) | b 337, a 336 |
| coresident pA ab | 341.0 (×1.101) | 341.2 (×1.100) | a 336, b 338 |
| coresident pA ba | 341.0 (×1.101) | 341.4 (×1.100) | b 338, a 336 |
| coresident pB ab | 357.4 (×1.050) | 357.6 (×1.050) | a 354, b 326 |
| coresident pB ba | 357.2 (×1.051) | 357.7 (×1.050) | b 326, a 354 |

### GPU time

| part | GPU s (stage wall - guard waits - compile) |
|---|---|
| A1 (solo re-validation) | 1429 |
| main | 1097 (wall 1134, waits 0, compile 37) |
| r029 | 738 (wall 774, waits 0, compile 36) |
| r058 | 658 (wall 678, waits 0, compile 20) |
| r223 | 709 (wall 755, waits 0, compile 46) |
| second | 941 (wall 994, waits 0, compile 53) |
| total | 5571 (1.55 h) |

