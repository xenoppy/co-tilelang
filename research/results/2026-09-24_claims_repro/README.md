# Reproducing TileLang's claimed speed-ups on the RTX PRO 6000 (sm_120)

> 2026-09-24 → 2026-09-25. Task set by the user: run TileLang's examples and baselines on this
> machine and check whether they reach the claimed speed-ups.
> Status: **done**. Details, raw data and reproduce commands are in the three sub-study READMEs
> (`A_gemm/`, `B_attention/`, `C_mamba/`).

## 1. The claims (upstream README "Benchmark Summary" + benchmark/*/README.md)

All upstream numbers were measured on other GPUs (H100 / H800 / A100 / RTX 4090 / MI300X).
Absolute throughput is not comparable to this card. What can be checked is the **relative
claim**: TileLang vs. the same baseline, both measured on this card.

| id | figure / source | claim (read from the figure) | baselines | platform of the claim |
|---|---|---|---|---|
| C1 | `images/op_benchmark_consistent_gemm_fp16.png`, shapes M0–M7 | TileLang fp16 GEMM ≈ cuBLAS (0.9–1.15×); ≥ Triton on most shapes. On RTX 4090 (closest consumer card) TileLang 0.97–1.15× cuBLAS | cuBLAS, Triton | 4090 / A100 / H100 / MI300X |
| C2 | `images/mha_performance_h100.png` top, FA0–FA4 | FlashAttention fwd: Triton 1.3–1.6× slower than TileLang, PyTorch 1.45–2.2× slower; FA3 0.9–2.2× (Hopper-only) | FA3, Triton, PyTorch | H100 |
| C3 | same figure, middle + bottom, CC0–CC4 / CT0–CT4 | Mamba-2 chunk-scan: Triton 1.45–2.0× slower; chunk-state: Triton 1.65–2.6× slower | Triton (mamba-ssm) | H100 |
| C4 | `examples/deepseek_mla/figures/bs{64,128}_float16.png` | MLA decode: TileLang ≈ FlashMLA (0.92–1.06×), ~1.2–1.35× FlashInfer, 3–10× Triton | FlashMLA (Hopper-only), FlashInfer, Triton, Torch | H100 |
| C5 | `images/op_benchmark_a100_wq_gemv.png`, V0–V6 | Dequant GEMV (BitBLAS-TileLang) vs cuBLAS fp16: W_INT4A_FP16 ≈ 3.7–4.4×, W_INT2A_FP16 ≈ 5×, W_NF4A_FP16 ≈ 2.6–3.1×, W_INT2A_INT8 ≈ 7.4–8.5× | cuBLAS fp16, Marlin, CUTLASS, bitsandbytes | A100 |
| C6 | `benchmark/matmul/README.md`, `benchmark/matmul_fp8/README.md`, `benchmark/mamba2/README.md` | absolute TFLOPS on H800 (fp16 GEMM ≤ 766, fp8 ≤ 1541, mamba chunk-scan ≤ 136) | none (absolute) | H800 |

C6 is hardware-bound; for it we only report the fraction of this card's measured peak (cuBLAS on
the same shape), not a pass/fail.

## 2. Completion criteria (written before any code, rules.md 3)

A sub-study is done when all of the following hold:

1. **Shapes.** The shape sets behind each figure (M0–M7, FA0–FA4, CC0–CC4, CT0–CT4, V0–V6, MLA
   batch/context grid) are recovered from an upstream source (tilelang-benchmark repo, TileLang
   paper, or the example scripts) and cited. Where a shape cannot be recovered, the substitute is
   stated.
2. **Correctness first.** Every TileLang kernel and every baseline that is timed passes a numerical
   check against a reference (torch) on that shape. Failures are reported, not silenced (rules.md 4).
3. **Same timer for everyone.** TileLang and baselines are timed with the same method in the same
   process, interleaved: primary = `cobench.bench_variants` (clean L2 flush, host gate, ≥50 reps,
   `clock=True`), so the reported ratio is not biased by thermal drift or the 600 W power cap.
   Secondary = the timer the upstream script itself uses (`tilelang.profiler.do_bench` /
   `triton.testing.do_bench`), reported for comparison with the upstream methodology.
4. **Fair tuning.** TileLang uses the example's autotuner / heuristic as shipped (adapted only where
   sm_120's 99 KB smem limit makes the shipped config unlaunchable — every such change is listed).
   Triton baselines use their own autotune configs. Library baselines (cuBLAS, SDPA, FlashInfer)
   run as shipped.
5. **Verdict per claim.** A table per claim: shape, TileLang µs, each baseline µs, measured ratio,
   claimed ratio, verdict ∈ {reproduced (measured ≥ 0.9 × claimed, or parity claim within the claimed
   band), partly, not reproduced, not testable on sm_120 (with reason)}.
6. **Explanations.** For every "not reproduced", one checked explanation (e.g. baseline is
   Hopper-only, TileLang lowers to `mma.sync` without wgmma on sm_120, config limited by smem,
   power cap) — not a guess.
7. **GPU policy.** Every GPU run follows rules.md 7 (pmon guard; foreign SM activity → wait 30 min),
   and our own sub-studies never time kernels concurrently (shared `flock`).
8. **Artifacts.** Scripts under `research/bench/scripts/claims_*.py`, raw JSON + README under this
   directory; environment changes (pip installs) recorded in `research/env_versions.md`.

## 3. Sub-studies

| sub-study | claims | status |
|---|---|---|
| A: GEMM + dequant GEMV | C1, C5, C6 (fp16/fp8 GEMM) | done — [A_gemm/README.md](A_gemm/README.md) |
| B: attention (FlashAttention fwd, MLA decode, + attention sink) | C2, C4 | done — [B_attention/README.md](B_attention/README.md) |
| C: Mamba-2 chunk scan / chunk state (+ minference, linear attention) | C3, C6 (mamba) | done — [C_mamba/README.md](C_mamba/README.md) |

All three met the completion criteria of §2 on acceptance review. Two review rounds changed the method but no
verdict: (a) GEMMs were re-timed in steady state (sustained 600 W load) because the flush-mode clock headroom could
have favoured one kernel; (b) kernels < ~60 µs were re-timed with CUDA graphs because eagerly launched kernels on
this GPU complete on a **2.048 µs grid** (sub-study B, `B_attention/raw/timerprobe.json`), which quantizes
per-kernel event times.

## 4. Results

Ratio = baseline time / TileLang time, both measured on this card (> 1: TileLang faster).
Verdict rule: reproduced if measured ≥ 0.9 × claimed (or within the band of a parity claim); partly if
TileLang is faster but below that; not reproduced if measured ≤ 1 (or outside the parity band).

| claim | baseline on sm_120 | claimed (platform) | measured here | verdict |
|---|---|---|---|---|
| **C1** fp16 GEMM ≈ cuBLAS, M0–M7 | cuBLAS 12.8 (torch), fp32 acc / fp16 acc | 0.96–1.15× (RTX 4090), paper mean 1.10× | steady: 1.10–1.32× (gmean 1.22) / 0.99–1.13× (gmean 1.08) | **reproduced** 8/8 (TileLang faster) |
| C1 ≥ Triton | Triton 3.4 tutorial 03, same layout | 0.86–1.24× (4090), mean 1.08× | steady: gmean 1.04 (fp32 acc) / 1.07 (fp16 acc) | **reproduced** 7/8, partly on M7 |
| **C2** FlashAttention fwd, FA0–FA4 | Triton (fused-attention, best of 3 setups) | 1.29–1.60×, gmean 1.41 (H100) | 1.05–1.33×, gmean 1.18 | **partly** (FA2 reproduced) |
| C2 | PyTorch = FA2 via SDPA flash | 1.45–2.17×, gmean 1.70 | 1.00–1.79×, gmean 1.27 | **partly** (FA1 reproduced) |
| C2 | FlashAttention-3 | gmean 1.36 | — | **not testable** (sm_90a-only) |
| **C3** Mamba-2 chunk scan, CC0–CC4 | mamba-ssm 2.2.6 Triton | 1.44–1.96× (H100) | 0.76–0.93 (CC0–CC2), 1.01 (CC3), 1.41 (CC4) | **not reproduced** (CC4 partly) |
| C3 chunk state, CT0–CT4 | same | 1.65–2.59× | 0.90–1.07 (CT0–CT3), 1.13 (CT4) | **not reproduced** (CT4 partly) |
| **C4** MLA decode, bs 64/128 × 1K–32K | FlashMLA | 0.92–1.06× | — | **not testable** (SM90-only) |
| C4 | FlashInfer 0.7 (fa2 backend; H100 used fa3) | 1.11–1.35×, gmean 1.23 | 0.97–1.00×, gmean 0.98 | **not reproduced** (parity) |
| C4 | Triton (`benchmark_mla.py`) | 3.0–12.7×, gmean 5.5 | 2.97–6.23×, gmean 4.22 | **partly** (2/11 reproduced; 12th shape: Triton kernel int32 overflow) |
| **C5** dequant GEMV, V0–V6 | BitBLAS 0.1.0.post1 (the claim's artifact) | 2.6–8.5× vs cuBLAS fp16 (A100) | cannot build any kernel on sm_120 | **not testable** |
| C5 via TileLang's own `example_dequant_gemv`, vs cuBLAS fp16 | W_INT4A_FP16 / W_INT2A_FP16 / W_INT2A_INT8 | gmean 4.16 / 5.22 / 7.91 | gmean 3.85 / 6.86 / 7.57 | INT4 **partly** (4/7 reproduced), INT2 **reproduced** 7/7, INT2A8 **reproduced** 6/7; NF4 not testable |
| **C6** absolute (H800) | — | fp16 386–766, fp8 569–1541, mamba 126–136 TFLOPS | fp16 236–334 (steady), fp8 563–757, mamba 98–102 TFLOPS | n/a (hardware-bound); on this card TileLang fp16/fp8 GEMM = 1.03–1.32× cuBLAS / 1.08–1.51× cuBLASLt; mamba2 benchmark vs Triton: 0.86–0.91 (seq ≤ 8K), 1.12 (16K), 1.73 (32K) |

Optional claims checked on the way: attention sink vs Triton — reproduced (head_dim 128 margin inflated by a
register-spilling Triton kernel on sm_120); minference README — reproduced at 5/12 points, partly at 7/12.

Caveat: the C5 V5 rows (18–33 µs kernels) were timed eagerly and sit within the 2.048 µs completion grid
(±6–11 %); their "partly" verdicts are not decisive. All other rows are ≥ 60 µs or were re-timed with graphs.

### Checked explanations (why the H100/A100 margins do or do not carry over)

1. **Hopper-specific advantages have nothing to act on.** sm_120 has TMA but no wgmma/tcgen05; TileLang, Triton
   and FA2 all issue `mma.sync` here (checked in generated code; the same TileLang programs lowered for sm_90a use
   wgmma). The H100 margins over Triton / FA2 / mamba-ssm came largely from wgmma + TMA warp specialisation.
2. **No shipped TileLang config for C2/C4 fits 99 KB of smem** (FA 160 KB, MLA 226 KB — sized for H100's 227 KB).
   They compile and fail at launch; the smallest fitting change (fewer stages / smaller block_H) was used.
   72/90 Mamba chunk-scan autotune configs cannot launch either.
3. **TileLang's automatic warp specialisation hurts on sm_120 when it costs occupancy.** Mamba chunk scan:
   256 threads, 1 CTA/SM; with the upstream pass config `tl.disable_warp_specialized` 2 CTAs/SM and TileLang moves
   from 0.77× to 1.01× (CC1), 1.09× to 1.27× (CC4).
4. **Some baselines are weak on this card, not TileLang strong.** cuBLAS 12.8/12.9 runs Ampere CUTLASS
   kernels on sm_120 (cuBLASLt fp8: Ada kernels); TileLang's TMA + warp-specialised GEMM also draws less energy per
   cycle and so clocks 3–36 % higher at the 600 W cap. The large-shape Mamba wins (CC5, CT5, seq 32K) come mostly
   from mamba-ssm's CTA order (heads slowest → shared tensors re-read per head); rotating the Triton grid cuts
   them from 1.66–2.20× to 1.07–1.49×.
5. **Memory-bound kernels are at the roofline for both sides.** Mamba chunk state (CT2–CT4) and the GEMVs stream at
   83–89 % of the 1.79 TB/s DRAM peak, so claimed 1.6–2.6× (chunk state) or > 4× (INT4 GEMV) cannot happen.
6. **Not a fork effect.** Mamba: the upstream v0.1.14 wheel generates byte-identical CUDA source and runs within
   0.4 %. GEMM / attention: none of the fork's compiler changes is active in these kernels (checked in generated
   sources).

### Defects found in upstream examples / baselines (candidates for upstream reports)

- `benchmark_mla.py` Triton MLA kernel: int32 offset overflow at batch 128 / 32K (illegal address); `--all` does not
  check outputs, yet the H100 figure plots that point.
- `examples/attention_sink/benchmark_gqa_sink_fwd.py` imports a module deleted upstream (ded6a992).
- `benchmark/matmul_fp8`: `get_configs(args, kwargs)` no longer matches the autotuner's calling convention
  (TypeError); fp8 with the default tvm_ffi backend fails (`Unsupported code 10`, torch 2.8 DLPack float8).
- `example_dequant_gemv_fp16xint4.py`: its correctness check (`atol=1e3` against interleaved weights) cannot fail;
  the 2-bit fp16 fast path crashes in `quantize/utils.py`.
- TileLang autotuner leaks the input tensors of every config that fails to launch (reached 56 GB at CC4).
- TileLang layout inference rejects MLA block_N < 64 on the sm_120 mma path (compiles for sm_90a); one Mamba
  chunk-scan config with `tl.disable_warp_specialized` outputs NaN.
- BitBLAS 0.1.0.post1 cannot target sm_120 (arch dispatch; bundled old TileLang includes the wgmma header for all
  arch ≥ 900).

## 5. Verdict

On this RTX PRO 6000 (sm_120), TileLang's examples run correctly, but only after sm_120 adaptations for most
attention/Mamba configs, and its claimed speed-ups reproduce only where the claim does not depend on Hopper:

- **Reproduced:** fp16 GEMM parity with cuBLAS (TileLang is actually 1.08–1.22× faster here, partly because cuBLAS
  ships no native sm_120 kernels) and ≥ Triton; 2-bit dequant GEMV speed-ups (via TileLang's own example; BitBLAS
  itself does not run).
- **Partly:** FlashAttention vs Triton / FA2 (TileLang still fastest among them, parity with cuDNN, but 1.18× /
  1.27× instead of 1.41× / 1.70×); MLA decode vs Triton; INT4 GEMV (capped by the 4× byte ratio because cuBLAS
  GEMV is already at 83–89 % of DRAM peak).
- **Not reproduced:** Mamba-2 chunk scan / chunk state vs Triton (parity or slower except at the largest shapes);
  MLA decode vs FlashInfer (parity, not 1.23×).
- **Not testable on sm_120:** FA3, FlashMLA (Hopper-only), BitBLAS, NF4 GEMV.

The dividing line is architectural: the H100 claims rest on wgmma + TMA warp-specialised pipelines and 227 KB of
shared memory. On sm_120, which has neither wgmma nor more than 99 KB of shared memory, those margins shrink to
0.8–1.3×. Where the baselines are themselves poorly suited to sm_120 (cuBLAS with Ampere kernels, a spilling
Triton sink kernel, mamba-ssm's grid order), TileLang's margins can exceed the claims.
