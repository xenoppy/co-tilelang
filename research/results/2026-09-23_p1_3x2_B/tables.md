## 3×2 tables (steady, stage F of part B)

| pair | T_serial | T[solo,inter] | T[lib,inter] | T[derived,inter] | T[solo,intra] | T[lib,intra] | T[derived,intra] | lib/derived inter | lib/derived intra | T_inter*/T[derived,intra] | T_inter*/T[lib,intra] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 726.1 | 592.8 (×1.225) | 586.5 (×1.238) | 573.7 (×1.266) | 591.4 (×1.228) | 589.5 (×1.232) | 580.1 (×1.252) | 1.022 | 1.016 | 0.989 | 0.973 |
| r029 | 1709.4 | 1461.6 (×1.170) | 1441.3 (×1.186) | 1381.1 (×1.238) | 1435.6 (×1.191) | 1435.6 (×1.191) | 1374.0 (×1.244) | 1.044 | 1.045 | 1.005 | 0.962 |
| r058 | 1060.4 | 797.3 (×1.330) | 797.3 (×1.330) | 733.6 (×1.446) | 786.2 (×1.349) | 786.2 (×1.349) | 744.7 (×1.424) | 1.087 | 1.056 | 0.985 | 0.933 |
| r223 | 571.4 | 503.9 (×1.134) | 503.4 (×1.135) | 493.8 (×1.157) | 504.7 (×1.132) | 504.7 (×1.132) | 501.3 (×1.140) | 1.019 | 1.007 | 0.985 | 0.979 |
| second | 374.0 | 315.2 (×1.187) | 313.0 (×1.195) | 309.6 (×1.208) | 319.4 (×1.171) | 317.1 (×1.179) | 310.5 (×1.205) | 1.011 | 1.021 | 0.997 | 0.976 |

### main: GEMM M4096_N4096_K4096 × decode B16_S8192

| cell | variant | knobs | GEMM config (solo rank, slowdown) | decode config (solo rank, slowdown) | derived axes | steady µs | × serial | MHz | W | mJ/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| serial | – | – | – | – | – | 726.1 | 1.000 | 2653 | 580 | 421 |
| **T[solo,inter]** | `gr_g28d11_148` | green 148/40 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n128_h4_sp1_t128_s1` (#1/36, +0.0%, C_lib) | none (lib space) | 592.8 | ×1.225 | 2176 | 600 | 356 |
| **T[lib,inter]** | `gr_g42d0_156` | green 156/32 | `128x128x64_s2_t256_wsauto_smem` (#3/43, +2.1%, C_lib) | `n32_h4_sp1_t64_s2` (#7/36, +0.6%, C_lib) | none (lib space) | 586.5 | ×1.238 | 1835 | 600 | 352 |
| **T[derived,inter]** | `gr_g42d4F_156` | green 156/32 | `128x128x64_s2_t256_wsauto_smem` (#3/43, +2.1%, C_lib) | `n64_h4_sp1_t64_s1_l2ef` (#4/36, +0.5%, C_lib) | decode K/V evict_first | 573.7 | ×1.266 | 1869 | 600 | 344 |
| **T[solo,intra]** | `co_sm_dyn_c1-1_g28d11_n140` | SM-bind dyn 140/48 chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n128_h4_sp1_t128_s1` (#1/36, +0.0%, C_lib) | none (lib space) | 591.4 | ×1.228 | 2187 | 600 | 355 |
| **T[lib,intra]** | `co_sm_dynT_c1-1_g28d4_n124` | SM-bind dyn 124/64 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t64_s1` (#4/36, +0.5%, C_lib) | none (lib space) | 589.5 | ×1.232 | 2186 | 600 | 354 |
| **T[derived,intra]** | `co_sm_dynT_c1-1_g28d4F_n116` | SM-bind dyn 116/72 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t64_s1_l2ef` (#4/36, +0.5%, C_lib) | decode K/V evict_first | 580.1 | ×1.252 | 2219 | 600 | 348 |

lib_inter: part A lib winner, re-measured: `gr_g42d0_156` 586.5 µs; lib (B search) winner: `gr_g42d8_156` 589.1 µs

lib_intra: part A lib winner, re-measured: `co_sm_dynT_c1-1_g28d4_n124` 589.5 µs; lib (B search) winner: `co_sm_dynT_c1-1_g28d4_n124` 589.5 µs

derived_intra: the search winner `co_sm_dynT_c1-1_g28d4F_n124` (decode K/V evict_first) re-measured at 580.2 µs in F.

**Ablations of the derived search winners** (same knobs, one derived axis reverted; gain = T_reverted / T_winner − 1)

| winner | axes | reverted axis | variant | µs | gain of the axis |
|---|---|---|---|---|---|
| derived_inter `gr_g42d4F_156` (green 156/32) | decode K/V evict_first | – | – | 573.7 | – |
| | | decode_hint | `gr_g42d4_156` | 587.2 | +2.36% |
| derived_intra `co_sm_dynT_c1-1_g28d4F_n124` (SM-bind dyn 124/64 +takeover chunk 1/1) | decode K/V evict_first | – | – | 580.2 | – |
| | | decode_hint | `co_sm_dynT_c1-1_g28d4_n124` | 589.5 | +1.59% |

**Decode K/V evict_first at identical knobs** (every hinted F variant whose unhinted twin ran in the same run; gain = T_unhinted / T_hinted − 1, energy per iteration)

| mechanism | knobs | hinted | unhinted | µs hinted / unhinted | gain | mJ hinted / unhinted |
|---|---|---|---|---|---|---|
| co | SM-bind dyn 116/72 +takeover chunk 1/1 | `co_sm_dynT_c1-1_g28d4F_n116` | `co_sm_dynT_c1-1_g28d4_n116` | 580.1 / 590.4 | +1.78% | 348 / 354 |
| co | SM-bind dyn 124/64 +takeover chunk 1/1 | `co_sm_dynT_c1-1_g28d4F_n124` | `co_sm_dynT_c1-1_g28d4_n124` | 580.2 / 589.5 | +1.59% | 348 / 354 |
| green | green 156/32 | `gr_g42d4F_156` | `gr_g42d4_156` | 573.7 / 587.2 | +2.36% | 344 / 352 |

**Solo (full GPU, steady) of the derived winners' configs** vs the solo-best of the op: derived_inter_a: `128x128x64_s2_t256_k1_wsauto_smem_g8` 446.0 µs (+3.7%); derived_inter_b: `n64_h4_sp1_t64_s1_l2ef` 330.1 µs (-0.2%); derived_inter_b_unhinted: `n64_h4_sp1_t64_s1` 330.8 µs (-0.0%); derived_intra_a: `128x256x64_s2_t256_k1_wsoff_direct_g8` 428.9 µs (-0.2%); derived_intra_b: `n64_h4_sp1_t64_s1_l2ef` 330.1 µs (-0.2%); derived_intra_b_unhinted: `n64_h4_sp1_t64_s1` 330.8 µs (-0.0%)

**Role completion times (timing builds, µs)**: derived_intra: T_A 569, T_B 430, CTAs 124/64, steals 0/0; lib_B_intra: T_A 583, T_B 459, CTAs 124/64, steals 0/0; abl_decode_hint: T_A 583, T_B 460, CTAs 124/64, steals 0/0

### r029: GEMM M4096_N4096_K4096 × decode B64_S8192

| cell | variant | knobs | GEMM config (solo rank, slowdown) | decode config (solo rank, slowdown) | derived axes | steady µs | × serial | MHz | W | mJ/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| serial | – | – | – | – | – | 1709.4 | 1.000 | 2757 | 499 | 853 |
| **T[solo,inter]** | `gr_g28d5_80` | green 80/108 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#1/36, +0.0%, C_lib) | none (lib space) | 1461.6 | ×1.170 | 2807 | 590 | 862 |
| **T[lib,inter]** | `gr_g42d11_148` | green 148/40 | `128x128x64_s2_t256_wsauto_smem` (#3/43, +2.1%, C_lib) | `n128_h4_sp1_t128_s1` (#8/36, +0.2%, C_lib) | none (lib space) | 1441.3 | ×1.186 | 2651 | 537 | 774 |
| **T[derived,inter]** | `gr_g42d11F_140` | green 140/48 | `128x128x64_s2_t256_wsauto_smem` (#3/43, +2.1%, C_lib) | `n128_h4_sp1_t128_s1_l2ef` (#8/36, +0.2%, C_lib) | decode K/V evict_first | 1381.1 | ×1.238 | 2611 | 546 | 754 |
| **T[solo,intra]** | `co_sm_dynT_c1-4_g28d5_n128` | SM-bind dyn 128/60 +takeover chunk 1/4 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#1/36, +0.0%, C_lib) | none (lib space) | 1435.6 | ×1.191 | 2683 | 544 | 780 |
| **T[lib,intra]** | `co_sm_dynT_c1-4_g28d5_n128` | SM-bind dyn 128/60 +takeover chunk 1/4 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#1/36, +0.0%, C_lib) | none (lib space) | 1435.6 | ×1.191 | 2683 | 544 | 780 |
| **T[derived,intra]** | `co_sm_dynT_c1-4_g28d5F_n116` | SM-bind dyn 116/72 +takeover chunk 1/4 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1_l2ef` (#1/36, +0.0%, C_lib) | decode K/V evict_first | 1374.0 | ×1.244 | 2689 | 557 | 766 |

lib_inter: part A lib winner, re-measured: `gr_g42d11_148` 1441.3 µs; lib (B search) winner: `gr_g42d11_148` 1441.3 µs

lib_intra: part A lib winner, re-measured: `co_sm_dynT_c1-4_g28d5_n128` 1435.6 µs; lib (B search) winner: `co_sm_dynT_c1-4_g28d5_n128` 1435.6 µs

**Ablations of the derived search winners** (same knobs, one derived axis reverted; gain = T_reverted / T_winner − 1)

| winner | axes | reverted axis | variant | µs | gain of the axis |
|---|---|---|---|---|---|
| derived_inter `gr_g42d11F_140` (green 140/48) | decode K/V evict_first | – | – | 1381.1 | – |
| | | decode_hint | `gr_g42d11_140` | 1457.7 | +5.54% |

**Decode K/V evict_first at identical knobs** (every hinted F variant whose unhinted twin ran in the same run; gain = T_unhinted / T_hinted − 1, energy per iteration)

| mechanism | knobs | hinted | unhinted | µs hinted / unhinted | gain | mJ hinted / unhinted |
|---|---|---|---|---|---|---|
| co | SM-bind dyn 120/68 +takeover chunk 1/4 | `co_sm_dynT_c1-4_g28d5F_n120` | `co_sm_dynT_c1-4_g28d5_n120` | 1374.4 / 1456.5 | +5.97% | 766 / 799 |
| co | SM-bind dyn 128/60 +takeover chunk 1/4 | `co_sm_dynT_c1-4_g28d5F_n128` | `co_sm_dynT_c1-4_g28d5_n128` | 1388.9 / 1435.6 | +3.37% | 763 / 780 |
| co | SM-bind dyn 136/52 +takeover chunk 1/4 | `co_sm_dynT_c1-4_g28d5F_n136` | `co_sm_dynT_c1-4_g28d5_n136` | 1408.3 / 1448.6 | +2.86% | 772 / 789 |
| green | green 140/48 | `gr_g42d11F_140` | `gr_g42d11_140` | 1381.1 / 1457.7 | +5.54% | 754 / 806 |
| green | green 148/40 | `gr_g42d11F_148` | `gr_g42d11_148` | 1396.1 / 1441.3 | +3.24% | 751 / 774 |
| green | green 156/32 | `gr_g42d11F_156` | `gr_g42d11_156` | 1445.3 / 1461.7 | +1.13% | 761 / 767 |
| green | green 80/108 | `gr_g28d5F_80` | `gr_g28d5_80` | 1381.4 / 1461.6 | +5.81% | 830 / 862 |

**Solo (full GPU, steady) of the derived winners' configs** vs the solo-best of the op: solo_b_hinted: `n64_h4_sp1_t128_s1_l2ef` 1305.4 µs (-0.1%)

**Role completion times (timing builds, µs)**: lib_intra: T_A 528, T_B 1430, CTAs 128/60, steals 0/0; lib_hint_intra: T_A 534, T_B 1385, CTAs 128/60, steals 0/0

### r058: GEMM M4096_N4096_K4096 × decode B32_S8192

| cell | variant | knobs | GEMM config (solo rank, slowdown) | decode config (solo rank, slowdown) | derived axes | steady µs | × serial | MHz | W | mJ/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| serial | – | – | – | – | – | 1060.4 | 1.000 | 2708 | 540 | 573 |
| **T[solo,inter]** | `gr_g28d5_142` | green 142/46 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#1/36, +0.0%, C_lib) | none (lib space) | 797.3 | ×1.330 | 2405 | 601 | 479 |
| **T[lib,inter]** | `gr_g28d5_142` | green 142/46 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#1/36, +0.0%, C_lib) | none (lib space) | 797.3 | ×1.330 | 2405 | 601 | 479 |
| **T[derived,inter]** | `gr_g28d5F_142` | green 142/46 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1_l2ef` (#1/36, +0.0%, C_lib) | decode K/V evict_first | 733.6 | ×1.446 | 2181 | 600 | 440 |
| **T[solo,intra]** | `co_sm_dynT_c1-1_g28d5_n128` | SM-bind dyn 128/60 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#1/36, +0.0%, C_lib) | none (lib space) | 786.2 | ×1.349 | 2321 | 600 | 472 |
| **T[lib,intra]** | `co_sm_dynT_c1-1_g28d5_n128` | SM-bind dyn 128/60 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#1/36, +0.0%, C_lib) | none (lib space) | 786.2 | ×1.349 | 2321 | 600 | 472 |
| **T[derived,intra]** | `co_sm_dynT_c1-1_g28d5F_n116` | SM-bind dyn 116/72 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1_l2ef` (#1/36, +0.0%, C_lib) | decode K/V evict_first | 744.7 | ×1.424 | 2222 | 600 | 447 |

lib_inter: part A lib winner, re-measured: `gr_g28d5_142` 797.3 µs; lib (B search) winner: `gr_g28d5_142` 797.3 µs

lib_intra: part A lib winner, re-measured: `co_sm_dynT_c1-1_g28d5_n128` 786.2 µs; lib (B search) winner: `co_sm_dynT_c1-1_g28d5_n128` 786.2 µs

**Ablations of the derived search winners** (same knobs, one derived axis reverted; gain = T_reverted / T_winner − 1)

| winner | axes | reverted axis | variant | µs | gain of the axis |
|---|---|---|---|---|---|
| derived_inter `gr_g28d5F_142` (green 142/46) | decode K/V evict_first | – | – | 733.6 | – |
| | | decode_hint | `gr_g28d5_142` | 797.3 | +8.69% |

**Decode K/V evict_first at identical knobs** (every hinted F variant whose unhinted twin ran in the same run; gain = T_unhinted / T_hinted − 1, energy per iteration)

| mechanism | knobs | hinted | unhinted | µs hinted / unhinted | gain | mJ hinted / unhinted |
|---|---|---|---|---|---|---|
| co | SM-bind dyn 120/68 +takeover chunk 1/1 | `co_sm_dynT_c1-1_g28d5F_n120` | `co_sm_dynT_c1-1_g28d5_n120` | 745.6 / 790.2 | +5.98% | 447 / 474 |
| co | SM-bind dyn 128/60 +takeover chunk 1/1 | `co_sm_dynT_c1-1_g28d5F_n128` | `co_sm_dynT_c1-1_g28d5_n128` | 765.6 / 786.2 | +2.70% | 459 / 472 |
| co | SM-bind dyn 136/52 +takeover chunk 1/1 | `co_sm_dynT_c1-1_g28d5F_n136` | `co_sm_dynT_c1-1_g28d5_n136` | 777.8 / 797.8 | +2.57% | 467 / 479 |
| green | green 134/54 | `gr_g28d5F_134` | `gr_g28d5_134` | 749.5 / 814.9 | +8.72% | 450 / 489 |
| green | green 142/46 | `gr_g28d5F_142` | `gr_g28d5_142` | 733.6 / 797.3 | +8.69% | 440 / 479 |
| green | green 150/38 | `gr_g28d5F_150` | `gr_g28d5_150` | 775.4 / 830.1 | +7.05% | 465 / 497 |

**Solo (full GPU, steady) of the derived winners' configs** vs the solo-best of the op: solo_b_hinted: `n64_h4_sp1_t128_s1_l2ef` 654.8 µs (-0.2%)

**Role completion times (timing builds, µs)**: lib_intra: T_A 583, T_B 781, CTAs 128/60, steals 0/0; lib_hint_intra: T_A 593, T_B 761, CTAs 128/60, steals 0/0

### r223: GEMM M4096_N4096_K4096 × decode B32_S2048

| cell | variant | knobs | GEMM config (solo rank, slowdown) | decode config (solo rank, slowdown) | derived axes | steady µs | × serial | MHz | W | mJ/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| serial | – | – | – | – | – | 571.4 | 1.000 | 2548 | 600 | 343 |
| **T[solo,inter]** | `gr_g28d5_172` | green 172/16 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#3/36, +0.0%, C_lib) | none (lib space) | 503.9 | ×1.134 | 1962 | 600 | 302 |
| **T[lib,inter]** | `gr_g28d4_172` | green 172/16 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t64_s1` (#2/36, +0.0%, C_lib) | none (lib space) | 503.4 | ×1.135 | 1964 | 600 | 302 |
| **T[derived,inter]** | `gr_g28d4F_172` | green 172/16 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t64_s1_l2ef` (#2/36, +0.0%, C_lib) | decode K/V evict_first | 493.8 | ×1.157 | 2004 | 600 | 296 |
| **T[solo,intra]** | `co_sm_dynT_c1-1_g28d5_n184` | SM-bind dyn 184/4 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#3/36, +0.0%, C_lib) | none (lib space) | 504.7 | ×1.132 | 2106 | 600 | 303 |
| **T[lib,intra]** | `co_sm_dynT_c1-1_g28d5_n184` | SM-bind dyn 184/4 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#3/36, +0.0%, C_lib) | none (lib space) | 504.7 | ×1.132 | 2106 | 600 | 303 |
| **T[derived,intra]** | `co_sm_dynT_c1-1_g28d5F_n184` | SM-bind dyn 184/4 +takeover chunk 1/1 | `128x256x64_s2_t256_wsoff_direct` (#1/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1_l2ef` (#3/36, +0.0%, C_lib) | decode K/V evict_first | 501.3 | ×1.140 | 2082 | 600 | 301 |

lib_inter: part A lib winner, re-measured: `gr_g28d4_172` 503.4 µs; lib (B search) winner: `gr_g28d4_172` 503.4 µs

lib_intra: part A lib winner, re-measured: `co_sm_dynT_c1-1_g28d5_n184` 504.7 µs; lib (B search) winner: `co_sm_dynT_c1-1_g28d5_n184` 504.7 µs

**Ablations of the derived search winners** (same knobs, one derived axis reverted; gain = T_reverted / T_winner − 1)

| winner | axes | reverted axis | variant | µs | gain of the axis |
|---|---|---|---|---|---|
| derived_inter `gr_g28d4F_172` (green 172/16) | decode K/V evict_first | – | – | 493.8 | – |
| | | decode_hint | `gr_g28d4_172` | 503.4 | +1.95% |
| derived_intra `co_sm_dynT_c1-1_g28d5F_n184` (SM-bind dyn 184/4 +takeover chunk 1/1) | decode K/V evict_first | – | – | 501.3 | – |
| | | decode_hint | `co_sm_dynT_c1-1_g28d5_n184` | 504.7 | +0.67% |

**Decode K/V evict_first at identical knobs** (every hinted F variant whose unhinted twin ran in the same run; gain = T_unhinted / T_hinted − 1, energy per iteration)

| mechanism | knobs | hinted | unhinted | µs hinted / unhinted | gain | mJ hinted / unhinted |
|---|---|---|---|---|---|---|
| co | SM-bind dyn 176/12 +takeover chunk 1/1 | `co_sm_dynT_c1-1_g28d5F_n176` | `co_sm_dynT_c1-1_g28d5_n176` | 511.3 / 513.9 | +0.49% | 307 / 308 |
| co | SM-bind dyn 184/4 +takeover chunk 1/1 | `co_sm_dynT_c1-1_g28d5F_n184` | `co_sm_dynT_c1-1_g28d5_n184` | 501.3 / 504.7 | +0.67% | 301 / 303 |
| green | green 164/24 | `gr_g28d4F_164` | `gr_g28d4_164` | 537.0 / 543.2 | +1.15% | 322 / 326 |
| green | green 172/16 | `gr_g28d5F_172` | `gr_g28d5_172` | 494.9 / 503.9 | +1.82% | 297 / 302 |
| green | green 172/16 | `gr_g28d4F_172` | `gr_g28d4_172` | 493.8 / 503.4 | +1.95% | 296 / 302 |
| green | green 180/8 | `gr_g28d4F_180` | `gr_g28d4_180` | 644.3 / 648.1 | +0.59% | 359 / 361 |

**Solo (full GPU, steady) of the derived winners' configs** vs the solo-best of the op: solo_b_hinted: `n64_h4_sp1_t128_s1_l2ef` 167.9 µs (-0.2%)

**Role completion times (timing builds, µs)**: lib_intra: T_A 456, T_B 494, CTAs 184/4, steals 0/0; lib_hint_intra: T_A 463, T_B 491, CTAs 184/4, steals 0/0

### second: GEMM M2048_N4096_K4096 × decode B32_S2048

| cell | variant | knobs | GEMM config (solo rank, slowdown) | decode config (solo rank, slowdown) | derived axes | steady µs | × serial | MHz | W | mJ/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| serial | – | – | – | – | – | 374.0 | 1.000 | 2583 | 597 | 223 |
| **T[solo,inter]** | `gr_g41d5_164` | green 164/24 | `128x128x64_s2_t128_wsauto_smem` (#2/43, +0.0%, C_lib) | `n64_h4_sp1_t128_s1` (#3/36, +0.0%, C_lib) | none (lib space) | 315.2 | ×1.187 | 2048 | 600 | 189 |
| **T[lib,inter]** | `gr_g42d5_164` | green 164/24 | `128x128x64_s2_t256_wsauto_smem` (#1/43, -1.0%, C_lib) | `n64_h4_sp1_t128_s1` (#3/36, +0.0%, C_lib) | none (lib space) | 313.0 | ×1.195 | 1918 | 600 | 188 |
| **T[derived,inter]** | `gr_g42d5F_156` | green 156/32 | `128x128x64_s2_t256_wsauto_smem` (#1/43, -1.0%, C_lib) | `n64_h4_sp1_t128_s1_l2ef` (#3/36, +0.0%, C_lib) | decode K/V evict_first | 309.6 | ×1.208 | 2035 | 600 | 186 |
| **T[solo,intra]** | `co_sm_dyn_c1-1_g19d5_n140` | SM-bind dyn 140/48 chunk 1/1 | `128x128x64_s2_t128_wsoff_smem` (#3/43, +4.0%, C_lib) | `n64_h4_sp1_t128_s1` (#3/36, +0.0%, C_lib) | none (lib space) | 319.4 | ×1.171 | 2189 | 600 | 192 |
| **T[lib,intra]** | `co_cta_dynT_c1-1_g18d5_g282` | CTA-bind 282 CTAs +takeover chunk 1/1 | `128x128x32_s3_t128_wsoff_smem` (#6/43, +6.0%, C_lib) | `n64_h4_sp1_t128_s1` (#3/36, +0.0%, C_lib) | none (lib space) | 317.1 | ×1.179 | 1960 | 600 | 190 |
| **T[derived,intra]** | `co_cta_dynT_c1-1_g18d5F_g282` | CTA-bind 282 CTAs +takeover chunk 1/1 | `128x128x32_s3_t128_wsoff_smem` (#6/43, +6.0%, C_lib) | `n64_h4_sp1_t128_s1_l2ef` (#3/36, +0.0%, C_lib) | decode K/V evict_first | 310.5 | ×1.205 | 2000 | 600 | 186 |

lib_inter: part A lib winner, re-measured: `gr_g42d5_164` 313.0 µs; lib (B search) winner: `gr_g42d5_164` 313.0 µs

lib_intra: part A lib winner, re-measured: `co_cta_dynT_c1-1_g18d5_g282` 317.1 µs; lib (B search) winner: `co_cta_dynT_c1-1_g18d5_g282` 317.1 µs

**Ablations of the derived search winners** (same knobs, one derived axis reverted; gain = T_reverted / T_winner − 1)

| winner | axes | reverted axis | variant | µs | gain of the axis |
|---|---|---|---|---|---|
| derived_inter `gr_g42d5F_156` (green 156/32) | decode K/V evict_first | – | – | 309.6 | – |
| | | decode_hint | `gr_g42d5_156` | 314.0 | +1.43% |
| derived_intra `co_cta_dynT_c1-1_g18d5F_g282` (CTA-bind 282 CTAs +takeover chunk 1/1) | decode K/V evict_first | – | – | 310.5 | – |
| | | decode_hint | `co_cta_dynT_c1-1_g18d5_g282` | 317.1 | +2.15% |

**Decode K/V evict_first at identical knobs** (every hinted F variant whose unhinted twin ran in the same run; gain = T_unhinted / T_hinted − 1, energy per iteration)

| mechanism | knobs | hinted | unhinted | µs hinted / unhinted | gain | mJ hinted / unhinted |
|---|---|---|---|---|---|---|
| co | CTA-bind 235 CTAs +takeover chunk 1/1 | `co_cta_dynT_c1-1_g18d5F_g235` | `co_cta_dynT_c1-1_g18d5_g235` | 313.1 / 321.7 | +2.76% | 188 / 193 |
| co | CTA-bind 282 CTAs +takeover chunk 1/1 | `co_cta_dynT_c1-1_g18d5F_g282` | `co_cta_dynT_c1-1_g18d5_g282` | 310.5 / 317.1 | +2.15% | 186 / 190 |
| co | CTA-bind 329 CTAs +takeover chunk 1/1 | `co_cta_dynT_c1-1_g18d5F_g329` | `co_cta_dynT_c1-1_g18d5_g329` | 321.7 / 332.1 | +3.23% | 193 / 199 |
| co | SM-bind dyn 136/52 +takeover chunk 1/1 | `co_sm_dynT_c1-1_g19d4F_n136` | `co_sm_dynT_c1-1_g19d4_n136` | 312.8 / 317.9 | +1.63% | 188 / 191 |
| co | SM-bind dyn 140/48 chunk 1/1 | `co_sm_dyn_c1-1_g19d5F_n140` | `co_sm_dyn_c1-1_g19d5_n140` | 315.2 / 319.4 | +1.35% | 189 / 192 |
| green | green 156/32 | `gr_g42d5F_156` | `gr_g42d5_156` | 309.6 / 314.0 | +1.43% | 186 / 188 |
| green | green 164/24 | `gr_g41d5F_164` | `gr_g41d5_164` | 313.6 / 315.2 | +0.51% | 188 / 189 |
| green | green 164/24 | `gr_g42d5F_164` | `gr_g42d5_164` | 311.7 / 313.0 | +0.40% | 187 / 188 |
| green | green 172/16 | `gr_g42d5F_172` | `gr_g42d5_172` | 347.0 / 349.8 | +0.80% | 208 / 210 |

**Solo (full GPU, steady) of the derived winners' configs** vs the solo-best of the op: solo_b_hinted: `n64_h4_sp1_t128_s1_l2ef` 167.7 µs (-0.2%)

**Role completion times (timing builds, µs)**: lib_intra: T_A 307, T_B 268, CTAs 188/94, steals 0/0; lib_hint_intra: T_A 302, T_B 243, CTAs 188/94, steals 0/0

## B3 — no-oracle robustness (steady; speed-up vs serial of the same run)

| pair | mechanism | oracle (split) | R1 split: speed-up | R2 split: speed-up | worst of the 8-split sweep | regret R1 / R2 / worst | main's split: speed-up (regret) |
|---|---|---|---|---|---|---|---|
| main | green ctx | ×1.257 (156) | 106: ×1.002 | 148: ×1.241 | ×0.753 | 1.255 / 1.013 / 1.670 | (source) |
| main | CoKernel dyn+TO | ×1.247 (124) | 106: ×1.241 | 148: ×1.205 | ×1.069 | 1.005 / 1.035 / 1.166 | (source) |
| r029 | green ctx | ×1.239 (108) | 46: ×0.832 | 94: ×1.237 | ×0.838 | 1.490 / 1.001 / 1.478 | 156: ×1.181 (1.049) |
| r029 | CoKernel dyn+TO | ×1.240 (46) | 46: ×1.240 | 94: ×1.234 | ×1.061 | 1.000 / 1.005 / 1.168 | 124: ×1.237 (1.002) |
| r058 | green ctx | ×1.440 (140) | 74: ×0.988 | 94: ×1.197 | ×0.854 | 1.458 / 1.204 / 1.686 | 156: ×1.339 (1.076) |
| r058 | CoKernel dyn+TO | ×1.402 (108) | 74: ×1.285 | 94: ×1.342 | ×1.077 | 1.092 / 1.045 / 1.302 | 124: ×1.398 (1.003) |
| r223 | green ctx | ×1.158 (172) | 136: ×1.056 | 172: ×1.158 | ×0.590 | 1.097 / 1.000 / 1.962 | 156: ×1.081 (1.071) |
| r223 | CoKernel dyn+TO | ×1.115 (172) | 136: ×1.085 | 172: ×1.115 | ×1.001 | 1.028 / 1.000 / 1.115 | 124: ×1.065 (1.047) |
| second | green ctx | ×1.212 (156) | 108: ×0.996 | 168: ×1.149 | ×0.661 | 1.217 / 1.055 / 1.835 | 156: ×1.212 (1.000) |
| second | CoKernel dyn+TO | ×1.191 (124) | 108: ×1.135 | 168: ×1.122 | ×0.999 | 1.049 / 1.061 / 1.192 | 124: ×1.191 (1.000) |

**Speed-up per GEMM share** (best candidate configs per split; rule / transfer splits included)


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
| green (budget-best configs only) | 0.790 | 1.017 | 1.047 | 1.173 | 1.164 | 1.151 | 1.173 | 1.180 | 0.838 |
| co | 1.240 | 1.061 | 1.168 | 1.234 | 1.230 | 1.237 | 1.197 | 1.135 | 1.071 |
| co (budget-best configs only) | 1.048 | 1.045 | 1.095 | 1.158 | 1.151 | 1.184 | 1.171 | 1.098 | 1.063 |

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

## GPU time

| pair | GPU s (stage wall − guard waits − compile) |
|---|---|
| main | 1701 (wall 2137, waits 1, compile 436) |
| r029 | 968 (wall 1079 incl. 568 s of superseded runs, waits 0, compile 111) |
| r058 | 535 (wall 617, waits 0, compile 82) |
| r223 | 1129 (wall 1221, waits 0, compile 92) |
| second | 549 (wall 658, waits 0, compile 110) |
| total | 4881 s (1.36 h) |

## D1 interim readout (plan §2.8, thresholds unchanged)

| pair | T_inter*/T[derived,intra] (≥1.10) | T[lib,intra]/T[derived,intra] (≥1.05) | full claim | T_inter*/T[lib,intra] (≥1.10) | weak claim |
|---|---|---|---|---|---|
| main | 0.989 | 1.016 | no | 0.973 | no |
| r029 | 1.005 | 1.045 | no | 0.962 | no |
| r058 | 0.985 | 1.056 | no | 0.933 | no |
| r223 | 0.985 | 1.007 | no | 0.979 | no |
| second | 0.997 | 1.021 | no | 0.976 | no |

