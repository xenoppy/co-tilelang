# Sub-study A: GEMM (C1, C6) and dequantized GEMV (C5) on the RTX PRO 6000 (sm_120)

> 2026-09-24 (steady-state follow-up run the same evening, analysed 2026-09-25), sub-study A of
> `../README.md`. All GPU work ran under the shared GPU lock, one job per lock hold, with the
> rules.md 7 guard (`cobench` pmon guard: foreign SM activity → wait 30 min). Every run's guard
> record is clean (one re-measurement after foreign activity, M0 cuBLAS-NN cross-check). Raw data:
> `raw/*.json` (`raw/steady_*.json` for the steady-state runs); tables and verdicts: `summary.json`,
> produced by `research/bench/scripts/claims_gemm_report.py`.
>
> **Two timing modes.** GEMMs were timed in clean-flush mode (`cobench.bench_variants`, cold L2, one
> call per rep) and in steady state (`cobench.bench_steady`: back-to-back under sustained load,
> rotating DRAM-cold inputs/outputs, thermal-plateau warmup, interleaved slices; section 4.5).
> **For the GEMM verdicts (C1, C6 fp16), steady state is primary**, because a sustained GEMM is
> power-capped and the flush phase between flush-mode reps lends clock headroom. The two modes give
> the same verdicts. C5 (GEMV, not power-capped) and C6 fp8 are flush-mode only.

## 0. Verdicts

| claim | what was compared on this card | measured | claimed | verdict |
|---|---|---|---|---|
| **C1** TileLang fp16 GEMM ≈ cuBLAS (M0–M7) | `benchmark/matmul` autotuned kernel vs cuBLAS (torch 2.8, cuBLAS 12.8), fp32 accumulate | steady: 1.10–1.32× (gmean 1.22); flush: 1.08–1.31× (gmean 1.20) | 0.96–1.15× on RTX 4090, paper: 1.10× mean | **reproduced**, 8/8 shapes in both modes; TileLang is faster than cuBLAS |
| C1, upstream methodology (all providers fp16 accumulate, as in tilelang-benchmark) | example_gemm_autotune kernel with fp16 accumulate vs cuBLAS `CUBLAS_COMPUTE_16F` | steady: 0.99–1.13× (gmean 1.08); flush: 1.00–1.13× (gmean 1.07) | same | **reproduced**, 8/8 in both modes |
| **C1** TileLang ≥ Triton | same kernels vs Triton tutorial 03 (v3.4.0), same NT layout | steady: 1.02–1.06× (gmean 1.04); fp16-acc 1.05–1.09× (gmean 1.07). Flush: 1.02–1.06× (gmean 1.04); fp16-acc 1.07–1.09× (gmean 1.08) | 4090: 0.86–1.24× per shape, paper: 1.08× mean | **reproduced** on 7/8 shapes, **partly** on M7 in both modes (claimed 1.24×: that bar came from Triton being slow on the 4090, see 5.2) |
| **C5** dequant GEMV (BitBLAS-TileLang) vs cuBLAS fp16, V0–V6 | BitBLAS 0.1.0.post1, the claim's artifact | cannot build any kernel on sm_120 | 2.6–8.5× (A100) | **not testable on sm_120** with the claim's artifact (5.4) |
| C5 via TileLang's own `example_dequant_gemv` generator, W_INT4A_FP16 | uint4, fp16 accumulate (the example's configuration) | 3.50–4.12× (gmean 3.85) | 3.72–4.36× (gmean 4.16) | **partly**: reproduced on V1, V2, V3, V6; partly on V0, V4, V5 (0.82–0.89 of claimed; 5.3) |
| C5, W_INT2A_FP16 | same generator, 2-bit (generic decode) | 5.49–7.63× (gmean 6.86) | 4.92–5.52× (gmean 5.22) | **reproduced**, 7/7 (above the claim) |
| C5, W_INT2A_INT8 | same generator, 2-bit, int8 activations (dp4a) | 6.22–8.08× (gmean 7.57) | 7.41–8.49× (gmean 7.91) | **reproduced** on 6/7; partly on V5 |
| C5, W_NF4A_FP16 | — | — | 2.6–3.1× | **not testable on sm_120** (BitBLAS cannot build; the TileLang example has no NF4 path) |
| **C6** fp16 GEMM 8192×8192×K (H800 absolute TFLOPS) | `benchmark/matmul` vs cuBLAS on this card | steady: TileLang 236–334 TFLOPS = 1.03–1.32× cuBLAS, 59–94 % of the mma.sync peak at the clock it ran. Flush: 284–375 TFLOPS = 1.01–1.31× cuBLAS | 386–766 TFLOPS on H800 | n/a (hardware-bound): steady 42–61 % (flush 43–74 %) of the H800 numbers |
| **C6** fp8 GEMM 8192×8192×K | `benchmark/matmul_fp8` vs cuBLASLt (`torch._scaled_mm`), both e4m3 out | TileLang 563–757 TFLOPS = 1.08–1.51× cuBLASLt | 569–1541 TFLOPS on H800 | n/a: 44–99 % of the H800 numbers |

The TileLang kernel is never the slow side here. Its lead over cuBLAS has a checked cause: on sm_120,
cuBLAS 12.8 and 12.9 run **Ampere CUTLASS kernels** (`cutlass_80_tensorop_f16_s16816gemm_*`,
`cutlass_80_tensorop_h16816gemm_*`), and cuBLASLt fp8 runs **Ada kernels** (`sm89_xmma_gemm_e4m3…`).
TileLang generates a Hopper-style kernel: TMA loads, a 2-stage mbarrier pipeline, a warp-specialised
producer and consumers with `setmaxnreg`, and `mma.sync`.

**Steady state.** Under sustained load every GEMM sits at the 600 W cap. There, TileLang runs at a
3–36 % higher SM clock than cuBLAS at the same power. cuBLAS therefore needs 1.10–1.32× the energy
per GEMM (fp32 accumulate; at equal power the energy ratio equals the time ratio). On the large shapes
(M2, M3, M5–M7) cuBLAS also needs 12–18 % more cycles. This is a real energy-efficiency advantage of
the generated kernel on this power-capped card, not a flush-mode artefact: the steady ratios equal or
exceed the flush ratios, with one exception 0.006 below (section 5.1).

## 1. Shapes, sources and claimed ratios

- **M0–M7** (C1): tile-ai/tilelang-benchmark@b658f7e README "Table 1". This is identical to the
  TileLang paper (arXiv 2504.17577) Appendix A, Table 2.
  - M0 4096×1024×8192, M1 4096×8192×8192, M2 4096×28672×8192, M3 4096×8192×28672, M4 8192×1024×8192,
    M5 8192³, M6 8192×28672×8192, M7 8192×8192×28672 (m×n×k).
  - C = A[m,k]·W[n,k]ᵀ: NT layout, as in the upstream `cublas_benchmark.cu` (`a_t=false, b_t=true`) and TileLang's kernels.
- **V0–V7** (C5): same table, m = 1.
  - V0 16384×16384, V1 43008×14336, V2 14336×14336, V3 57344×14336, V4 14336×57344, V5 9216×9216,
    V6 36864×9216, V7 9216×36864 (n×k).
  - The same list appears in `ampere_benchmark/dequant_matmul/data/data_gemv.py`.
  - The figure shows V0–V6; V7 was measured but has no claimed value.
- **C6**: `benchmark/matmul/README.md` and `benchmark/matmul_fp8/README.md`, M = N = 8192, K ∈ {256 … 16384}.
  - Their "Latency (s)" column is in ms: 2·8192²·256 / 0.089056 ms = 386 TFLOPS.
  - K = 8192 fp16 is shape M5, measured once.
- **Claimed per-shape ratios.** The benchmark repo's `data_*.py` files only parse logs that were never
  committed, so the ratios were digitised from the PNGs by `claims_gemm_figures.py` into
  `claimed_ratios_digitized.json`.
  - Calibration uses the y-tick marks: residual 0.0 px / 0.3 px; 231 / 96 px per unit, so a
    resolution of 0.004× (GEMM) and 0.01× (GEMV).
  - The dashed cuBLAS line reads back as 1.002 / 0.999.
  - Bars are segmented by fill colour, and the order was checked against the legend colours.
  - Cross-check with the paper text (Sec. 5): the digitised TileLang-4090 bars average 1.09× vs
    cuBLAS (paper: 1.10×) and 1.05× vs Triton (paper: 1.08×). The digitised TileLang-A100 bars
    average 0.96× (paper: 0.97×).
  - The paper's Fig. 15 text says "max 7.65×" for W_INT2A_INT8, while the README figure shows up to
    8.49×. We compare against the README figure, which is the one named in the task.
- **Which claimed ratio is used.** C1 uses the RTX 4090 bars. The 4090 is the claim platform closest
  to sm_120: consumer-class, `mma.sync`, no wgmma, 99 KB smem. The A100 and H100 values are listed
  alongside. C5 uses the A100 bars, the only platform in that figure.
- **Accumulation dtype in the upstream methodology**, checked in the tilelang-benchmark sources:
  - For the 4090 and H100 GEMM bars, all three providers accumulated in **fp16**:
    - `cublas_benchmark.cu` sets `compute_type = CUDA_R_16F` for half;
    - the Triton script uses `tl.dot(..., out_dtype=tl.float16)`;
    - "TileLang" was BitBLAS-TileLang with `accum_dtype=float16`.
  - This card also runs fp16-accumulate GEMM faster: cuBLAS 8192³ is 1.22× faster than with fp32
    accumulate. So C1 was run twice:
    - once with the shipped `benchmark/matmul` kernel (fp32 accumulate: suite `fp32acc`);
    - once with the upstream methodology (fp16 accumulate everywhere: suite `fp16acc`).

## 2. Environment and changes

| item | value |
|---|---|
| GPU | RTX PRO 6000 Blackwell Workstation, sm_120 (TileLang target `sm_120a`), 188 SMs, 101376 B smem/CTA (opt-in), 128 MB L2, power limit 600 W (= max), max SM clock 3090 MHz, driver 580.173.02 |
| TileLang | this fork, `0.1.14+cuda` (HEAD `1bab2491`, upstream base `7e3bbf37`), dev build, tvm_ffi backend (cython backend for fp8, see 3) |
| torch / cuBLAS | 2.8.0+cu128 with its pip `nvidia-cublas-cu12 12.8.4.1` (default). Cross-check: CUDA 12.9 toolkit `libcublas{,Lt}.so.12.9.0.13` via `LD_PRELOAD`, verified from `/proc/self/maps` and recorded as `blas_libs` in each JSON |
| Triton | 3.4.0 |
| nvcc | 12.9 |

Changes made by this sub-study:

- **pip (`~/mpk-env`).** `bitblas==0.1.0.post1` plus 13 new dependencies:
  - attrs 26.1.0, cffi 2.1.1, cpplint 2.0.2, decorator 5.3.1, docutils 0.23, dtlib 0.0.0.dev2,
    execnet 2.1.2, pycparser 3.0, pytest-xdist 3.8.0, RapidFuzz 3.14.6, scipy 1.18.1, thefuzz 0.22.1,
    tornado 6.5.10.
  - Procedure: dry run first, `--no-cache-dir`, under the pip lock.
  - `pip list` before and after showed additions only; torch, triton, apache-tvm-ffi and flashinfer
    were unchanged. Disk: +467 MB.
  - BitBLAS cannot build any kernel on sm_120 (5.4), so **`bitblas` was uninstalled again**. Its 13
    dependencies (~170 MB) are still installed. The uninstall command is in
    `research/env_versions.md` §7, which also records all of this.
- **Caches.**
  - The compile phases built ~16 000 kernels, and `~/.tilelang/cache` grew by 6.5 GB. At the end,
    exactly these entries were deleted: kernels whose `params.pkl` shapes are this study's GEMMs. Other
    sub-studies' entries were left in place.
  - TileLang's cython backend (`tilelang/jit/adapter/libgen.py`, `delete=False`) left 4.6 GB of
    `tmp*.cu/.so` in `/tmp` from the fp8 compile. These were also deleted.
  - `~/.triton/cache` grew by a small amount (autotuned Triton binaries; not measured separately,
    under 100 MB).
- **No source changes.** No TileLang compiler source, example or benchmark script was modified.
  Everything is in wrappers: `research/bench/scripts/claims_gemm_{common,tl,steady,figures,report}.py`,
  `claims_gemv_{tl,bitblas}.py`, `claims_gemm_all.sh` and `research/bench/baselines/triton_matmul.py`.
- **Log files.** The TileLang autotuner writes `autotuner.log` into the working directory, here the
  repo root. Importing BitBLAS creates an empty `out.log` there too. Both are gitignored and both were
  removed.

## 3. Method and adaptations

**Kernels (all imported unmodified).**
- **`fp32acc`: `benchmark/matmul/benchmark_matmul.py` `matmul`.**
  - It is an `@autotune` over 288 configs (block 64/128/256², block_K 32/64, 0–3 stages, 128/256
    threads, rasterisation on/off), with the script's own `warmup=3, rep=20`.
  - The run also times the default (heuristic) config of `examples/gemm/example_gemm_autotune.py`.
- **`fp16acc`: `examples/gemm/example_gemm_autotune.py` `matmul(..., accum_dtype=float16)`.**
  - It is tuned by `AutoTuner.from_kernel` over that example's own `get_configs` grid (the same 288
    configs).
  - The autotuner settings are those of `benchmark_matmul.py`.
  - The example's own tuner path is hard-wired to bf16 with fp32 accumulate, so it could not be used
    as is.
- **`fp8`: `benchmark/matmul_fp8/benchmark_matmul.py` `matmul`.**
  - 576 configs; e4m3 in, fp32 accumulate, e4m3 out.

**Baselines.**
- **cuBLAS.** `torch.matmul(A, B.T, out=C)`, with the same NT operands.
  - `fp16acc` suite: `torch.backends.cuda.matmul.allow_fp16_accumulation=True`, which selects
    `cutlass_80_tensorop_h16816gemm` (fp16 accumulate).
- **Triton.** Kernels copied into `baselines/triton_matmul.py`, both with the 16 CUDA autotune configs
  of tutorial 03 and each (kernel, layout) pair with its own `Autotuner`, because the key is (M,N,K)
  only:
  - the tutorial-03 kernel of v3.4.0 (fp32 accumulate), with `b = B.T`, i.e. the same NT layout;
  - tilelang-benchmark's fp16-accumulate variant, in NT and in the NN layout that script used.
- **cuBLASLt fp8.** `torch._scaled_mm(A, B.T, scale 1, out_dtype=float8_e4m3fn)`: same output dtype
  as the TileLang kernel.

**Correctness first.** Every timed variant is checked on the timing inputs before timing. Only
passing variants are timed; there were no failures.
- **fp32-accumulate GEMM**
  - Reference: fp32 torch matmul.
  - Tolerance: max-norm relative error ≤ 5e-3.
  - Measured: 2.5e-4 to 8e-4.
- **fp16-accumulate GEMM**
  - Tolerance: ≤ max(5e-3, 3× cuBLAS-fp16acc's own error).
  - Measured: all providers are **bit-identical in error**, 9.6e-3 to 2.0e-2. With sequential
    k-order and the same HMMA instruction, they round identically.
- **fp8 GEMM**
  - Tolerance: ≤ 0.07 (e4m3 output rounding).
  - Measured: TileLang and cuBLASLt both at 4.35e-2 / 3.2e-2 …, identical.
- **GEMV**
  - Reference: float64 matmul of the unpacked logical weights.
  - int32 outputs must match exactly, and they do.

**Timers.**
- **Steady state: `cobench.bench_steady`** (`claims_gemm_steady.py`; primary for the GEMM verdicts,
  C1 and C6 fp16).
  - Kernels: TileLang with the config the flush-mode autotuner picked (not re-tuned), compiled with
    `out_idx=None` so that it writes the rotating output like the baselines. Its device source is
    byte-identical to the flush-run kernel (checked, `same_device_source_as_flush_run`). cuBLAS and
    Triton as in flush mode (NT; Triton re-autotuned).
  - Inputs and outputs rotate through `cobench.Rotation` copies (≥ 2× L2 in total), so every
    iteration is DRAM-cold without a flush.
  - Warmup until the GPU temperature plateaus (30–113 s, 84–89 °C), then 5 interleaved rounds of
    1.5 s slices per variant; the first 0.4 s of each slice is excluded.
  - Reported per variant: time/iter (median over slices; slice CV ≤ 0.18 %), SM clock (ClockProbe),
    board power (NVML), energy/iter, cycles/iter; speed-ups as ratio of medians (the paired per-round
    ratios lie within ±0.002 of it).
  - Covered: C1 M0–M7 in both suites, and the C6 fp16 K sweep. Not run: C6 fp8 and C5.
- **Flush mode: `cobench.bench_variants`**, the same for every provider in the same process
  (primary for C5 and C6 fp8, secondary for the GEMM verdicts).
  - All providers of a shape run interleaved, with rotating order.
  - Each rep does a clean L2 flush (256 MB), then the host gate, then CUDA events.
  - 60 timed reps, with the ClockProbe and NVML on.
  - Reported: median µs, per-variant SM clock (MHz) and power.
- **Secondary** (the upstream scripts' timers):
  - `tilelang.profiler.do_bench(backend="event")`, plus `"cupti"` for GEMV, as the example's
    `run_regression_perf` does;
  - `triton.testing.do_bench`;
  - the TileLang autotuner's own latency, recorded as `tl_tune.autotuner_latency_ms`.
  - Agreement with the flush-mode ratios (tables below; `raw/*.json` `secondary`):
    - GEMM: within 8.5 %; the TileLang lead is slightly larger with the secondary timers.
    - GEMV: within 12 %. The int2-fp16 speed-ups are 8–11 % lower with the secondary timers, but
      still above the claim. uint2·int8 on V5 is 12 % higher with cupti.

**Adaptations (all changes relative to "run the shipped script").**
1. **smem filter.** Configs whose launch requests more than 101376 B of shared memory are removed
   before autotuning.
   - The launch smem is read from the generated host code (tvm_ffi launch arguments, or
     `dynamicSmemBytes` for cython). It is not estimated.
   - Removed: 48/288 fp16 configs (123–197 KB: all 256×256 tiles and 3-stage 128×256×64-class
     tiles) and 48/576 fp8 configs. The lists are in `configs/*.json`.
   - The largest config that fits uses 98.3 KB.
   - Unfiltered, these configs fail at launch with `Failed to set the allowed dynamic shared memory
     size`.
2. **`benchmark/matmul_fp8` config generator.** Its `get_configs(args, kwargs)` signature predates
   the autotuner's current calling convention, `configs(*kernel_args, **kernel_kwargs)`, and raises
   `TypeError`. The wrapper calls it the old way and hands the list to `@autotune`.
   - This is an upstream script bug; the fork did not touch `benchmark/`.
3. **fp8 execution backend.** With the default tvm_ffi backend, every fp8 config fails at launch
   (`MemoryError: Unsupported code 10` from `at::toScalarType`).
   - The fp8 **output** tensor is allocated through torch's DLPack importer, and torch 2.8 does not
     know DLPack dtype code 10 (float8_e4m3fn).
   - The fp8 suite therefore uses TileLang's **cython** backend
     (`matmul.jit_impl.execution_backend = "cython"`). The device kernel is identical; only the host
     launcher differs, and host time is excluded by the host gate.
   - This is an environment issue (torch 2.8), not an sm_120 or fork issue.
4. **Compile outside the lock.** Compilation was done in a separate CPU phase, `claims_gemm_tl.py
   compile`, so that the autotuner's own compile step inside the GPU lock hits the kernel cache.
   - This worked for `fp32acc` and `fp8`.
   - For `fp16acc` the `AutoTuner.from_kernel` path did not hit the precompiled entries, and 239/240
     configs were recompiled inside the lock (about 3.5 min per shape). This cost lock time only, not
     accuracy. The cause was not investigated.
5. **GEMV example** (5.3):
   - the example's shipped correctness check is replaced by a meaningful one;
   - the 2-bit fp16 variant uses the generator's generic (non-lop3) decode, because the example's
     2-bit interleave helper crashes.

## 4. Results

### 4.1 C1, suite `fp32acc` (shipped `benchmark/matmul` kernel, fp32 accumulate)

| shape | M,N,K | TL µs | cuBLAS µs | Triton-tut03 (NT) µs | TL/cuBLAS | claimed (4090; A100, H100) | verdict | TL/Triton | claimed TL/Triton (4090) | verdict | TL TFLOPS | clock TL/cuBLAS MHz | cycles cuBLAS/TL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 4096,1024,8192 | 210.9 | 227.2 | 219.2 | 1.077 | 0.96 (0.92, 1.15) | reproduced | 1.039 | 0.86 | reproduced | 326 | 2680/2294 | 0.922 |
| M1 | 4096,8192,8192 | 1508.6 | 1668.1 | 1574.9 | 1.106 | 1.15 (0.98, 1.08) | reproduced | 1.044 | 1.01 | reproduced | 364 | 2222/1913 | 0.952 |
| M2 | 4096,28672,8192 | 5843.8 | 6892.5 | 5979.2 | 1.179 | 1.12 (0.98, 1.09) | reproduced | 1.023 | 1.02 | reproduced | 329 | 1878/1835 | 1.152 |
| M3 | 4096,8192,28672 | 5737.4 | 7352.7 | 5956.4 | 1.282 | 1.04 (0.92, 1.08) | reproduced | 1.038 | 1.14 | reproduced | 335 | 2011/1777 | 1.133 |
| M4 | 8192,1024,8192 | 399.6 | 433.9 | 423.2 | 1.086 | 1.15 (0.98, 1.06) | reproduced | 1.059 | 1.01 | reproduced | 344 | 2272/2093 | 1.000 |
| M5 | 8192,8192,8192 | 3164.0 | 4153.4 | 3300.4 | 1.313 | 1.15 (0.96, 1.02) | reproduced | 1.043 | 1.02 | reproduced | 348 | 1954/1760 | 1.182 |
| M6 | 8192,28672,8192 | 11625.9 | 15174.7 | 12118.0 | 1.305 | 1.15 (1.00, 0.98) | reproduced | 1.042 | 1.15 | reproduced | 331 | 1834/1647 | 1.173 |
| M7 | 8192,8192,28672 | 11447.4 | 14762.0 | 11818.9 | 1.290 | 1.01 (0.96, 0.98) | reproduced | 1.032 | 1.24 | partly | 336 | 1887/1730 | 1.182 |

Geometric means: TL/cuBLAS = **1.201**, TL/Triton = **1.040**.

Secondary timers, TL/cuBLAS and TL/Triton ratios:

| timer | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---|---|---|---|---|---|---|---|
| TL/cuBLAS, `tilelang.profiler.do_bench` | 1.085 | 1.142 | 1.244 | 1.343 | 1.090 | 1.391 | 1.344 | 1.339 |
| TL/cuBLAS, `triton.testing.do_bench` | 1.085 | 1.128 | 1.198 | 1.300 | 1.101 | 1.328 | 1.314 | 1.310 |
| TL/Triton, `tilelang.profiler.do_bench` | 1.038 | 1.078 | 1.061 | 1.085 | 1.094 | 1.098 | 1.069 | 1.063 |
| TL/Triton, `triton.testing.do_bench` | 1.038 | 1.047 | 1.023 | 1.048 | 1.071 | 1.054 | 1.046 | 1.041 |

**Tuned configs.** Every tuned config has 2 stages and 256 threads. The tiles are 128×256×64 on
M0, M3, M5, M6 and M7; 128×256×32 on M2; 256×128×64 on M1; and 128×128×64 on M4. Every tuned kernel
is `__launch_bounds__(384, 1)`:
warp-specialised, with TMA loads, `mma_sync<f16,f16,f32,16,8,16>`, stmatrix, and a TMA store.

**Example default config.** The default path of `examples/gemm/example_gemm_autotune.py` (no
`--use_autotune`) was timed on sm_120 in the same runs.
- Its `get_heuristic_config()` special-cases only sm_80 and sm_90. Every other GPU gets
  `num_stages=0`: no pipelining and synchronous global loads (no TMA, no cp.async in the generated
  code).
- On sm_120 it runs at **0.27–0.48× cuBLAS** (0.25–0.37× of the autotuned kernel).
- This is a config effect of the example, not of the compiler.

**Cross-checks** (`raw/fp32acc_*_cublas129.json`, `*_cublas128nn.json`):
- cuBLAS 12.9 via `LD_PRELOAD` gives the same picture: TL/cuBLAS = 1.085, 1.111, 1.182, 1.276, 1.104,
  1.315, 1.288, 1.273 for M0–M7. In these later runs the absolute times of *both* cuBLAS and TileLang
  were 1–8 % lower than in the main runs. That is a run-condition (thermal) difference, not a library
  one: the ratios are unchanged.
- cuBLAS on an NN copy of B (`cublas_nn`) is not faster than NT: 0.92–1.01× of NT.
- The profiler shows the kernels cuBLAS 12.8/12.9 picks on sm_120:
  - fp32 accumulate: `cutlass_80_tensorop_f16_s16816gemm_relu_f16_{64x256_32x4, 128x64_64x3, 64x128_64x3}_tn_align8`,
    sometimes with a split-K `Memset`;
  - fp16 accumulate: `cutlass_80_tensorop_h16816gemm_128x128_64x3_tn_align8`.
- **Run-to-run spread.** Absolute times of the same kernel in different processes differ by up to
  ~7 % (TL M7: 11.45 ms in the main run, 10.71–10.87 ms in the cross-check runs), because of power
  and thermal state (temperatures 43–86 °C). The within-run ratios agree to ≤ 2 % (M7: 1.290 / 1.273
  / 1.273). This is why only interleaved same-run ratios are reported.

### 4.2 C1, suite `fp16acc` (upstream methodology: every provider accumulates in fp16)

| shape | TL µs | cuBLAS-fp16acc µs | Triton fp16acc NT µs | Triton fp16acc NN µs | TL/cuBLAS | claimed (4090; A100, H100) | verdict | TL/Triton (NT) | claimed (4090) | verdict | TL TFLOPS | clock TL/cuBLAS MHz | cycles cuBLAS/TL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 206.8 | 206.4 | 223.2 | 223.2 | 0.998 | 0.96 (0.92, 1.15) | reproduced | 1.079 | 0.86 | reproduced | 332 | 2710/2425 | 0.893 |
| M1 | 1505.2 | 1541.1 | 1608.7 | 1597.1 | 1.024 | 1.15 (0.98, 1.08) | reproduced | 1.069 | 1.01 | reproduced | 365 | 2195/2049 | 0.956 |
| M2 | 5346.1 | 5630.0 | 5798.9 | 5671.9 | 1.053 | 1.12 (0.98, 1.09) | reproduced | 1.085 | 1.02 | reproduced | 360 | 2119/1915 | 0.952 |
| M3 | 5518.8 | 5983.2 | 5910.0 | 5836.3 | 1.084 | 1.04 (0.92, 1.08) | reproduced | 1.071 | 1.14 | reproduced | 349 | 2155/1782 | 0.896 |
| M4 | 387.0 | 387.1 | 413.7 | 427.0 | 1.000 | 1.15 (0.98, 1.06) | reproduced | 1.069 | 1.01 | reproduced | 355 | 2318/2277 | 0.982 |
| M5 | 2967.6 | 3354.6 | 3199.1 | 3155.4 | 1.130 | 1.15 (0.96, 1.02) | reproduced | 1.078 | 1.02 | reproduced | 371 | 2146/1855 | 0.977 |
| M6 | 10835.0 | 12232.2 | 11805.4 | 11538.6 | 1.129 | 1.15 (1.00, 0.98) | reproduced | 1.090 | 1.15 | reproduced | 355 | 2037/1744 | 0.967 |
| M7 | 10955.5 | 12273.7 | 11748.3 | 11565.6 | 1.120 | 1.01 (0.96, 0.98) | reproduced | 1.072 | 1.24 | partly | 351 | 2003/1724 | 0.964 |

Geometric means: TL/cuBLAS = **1.066**, TL/Triton-NT = **1.077**, TL/Triton-NN (the upstream
script's layout) = 1.068. The paper reports 1.10× and 1.08× on the RTX 4090.

Secondary timers:

| timer | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---|---|---|---|---|---|---|---|
| TL/cuBLAS, `tilelang.profiler.do_bench` | 0.998 | 1.082 | 1.105 | 1.138 | 1.015 | 1.185 | 1.165 | 1.149 |
| TL/cuBLAS, `triton.testing.do_bench` | 0.999 | 1.040 | 1.072 | 1.102 | 1.002 | 1.136 | 1.137 | 1.123 |
| TL/Triton, `tilelang.profiler.do_bench` | 1.077 | 1.117 | 1.127 | 1.115 | 1.099 | 1.125 | 1.120 | 1.099 |
| TL/Triton, `triton.testing.do_bench` | 1.075 | 1.071 | 1.092 | 1.083 | 1.079 | 1.083 | 1.094 | 1.076 |

Verdict rule for "≈ cuBLAS" (the parity claim, band 0.9–1.15×): reproduced if measured ≥ 0.9 ×
the claimed 4090 ratio, or ≥ 0.9. For "≥ Triton" (the speed-up claim): reproduced if measured ≥ 0.9
× claimed; partly if below that but still > 1.

### 4.3 C6, 8192×8192×K (absolute throughput on this card)

The "% of mma peak" column uses 188 SMs × 1024 FLOP/clk/SM for fp16 with fp32 accumulate (measured
by `cobench.mma_peak`). For e4m3 it assumes 2048 FLOP/clk/SM. This is not measured, but it is
consistent with cuBLASLt reaching ≤ 93 % of it. The clock is the ClockProbe median during the
kernel.

| K | fp16: TL µs | cuBLAS µs | TL TFLOPS | cuBLAS TFLOPS | TL/cuBLAS | TL MHz | TL % of mma peak | H800 claim | TL here / claim |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 120.8 | 122.5 | 284 | 281 | 1.014 | 2574 | 57 | 386 | 0.74 |
| 512 | 204.8 | 208.9 | 335 | 329 | 1.020 | 2486 | 70 | 520 | 0.65 |
| 1024 | 370.7 | 380.9 | 371 | 361 | 1.027 | 2465 | 78 | 628 | 0.59 |
| 2048 | 733.2 | 755.7 | 375 | 364 | 1.031 | 2290 | 85 | 705 | 0.53 |
| 4096 | 1500.3 | 1790.9 | 366 | 307 | 1.194 | 2161 | 88 | 736 | 0.50 |
| 8192 (M5) | 3164.0 | 4153.4 | 348 | 265 | 1.313 | 1954 | 92 | 758 | 0.46 |
| 16384 | 6628.6 | 8432.7 | 332 | 261 | 1.272 | 1848 | 93 | 766 | 0.43 |

| K | fp8: TL µs | cuBLASLt µs | TL TFLOPS | cuBLASLt TFLOPS | TL/cuBLASLt | TL MHz | TL % of mma peak | H800 claim | TL here / claim |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 61.0 | 92.2 | 563 | 373 | 1.510 | 2636 | 55 | 569 | 0.99 |
| 512 | 106.5 | 115.1 | 645 | 597 | 1.081 | 2552 | 66 | 858 | 0.75 |
| 1024 | 190.2 | 208.9 | 722 | 658 | 1.098 | 2433 | 77 | 1129 | 0.64 |
| 2048 | 362.9 | 442.4 | 757 | 621 | 1.219 | 2322 | 85 | 1343 | 0.56 |
| 4096 | 755.1 | 987.4 | 728 | 557 | 1.308 | 2129 | 89 | 1467 | 0.50 |
| 8192 | 1532.7 | 1844.5 | 717 | 596 | 1.203 | 2059 | 91 | 1507 | 0.48 |
| 16384 | 3272.3 | 3794.6 | 672 | 580 | 1.160 | 1889 | 92 | 1541 | 0.44 |

Secondary timers:
- TL/cuBLAS fp16 with `tilelang.profiler.do_bench`: 1.02, 1.02, 1.05, 1.09, 1.27, 1.39, 1.36.
- TL/cuBLASLt fp8 with `tilelang.profiler.do_bench`: 1.51, 1.08, 1.10, 1.24, 1.37, 1.26, 1.21.
- `triton.testing.do_bench` agrees within 0.07.

**Kernels chosen.**
- cuBLASLt picks `sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize{64x128x64, 128x128x64, 256x64x64, 128x64x64}…` (Ada kernels).
- TileLang (fp8) picks 128×256 tiles with block_K 64 (128 for K ≥ 8192), 2 stages, warp-specialised with TMA.
  TileLang (fp16) picks 128×256 tiles with block_K 32 for K ≤ 512 and 64 above.

**Why the H800 numbers are out of reach.** Large-K runs hit the 600 W cap (NVML `SwPowerCap`, 600 W
mean), which pulls the clock down to 1.85–2.1 GHz. At that clock the mma.sync peak is 356–404 TFLOPS
for fp16. The H800's dense fp16 peak is 989 TFLOPS with wgmma. So the H800 numbers are unreachable on
this card regardless of the compiler. TileLang reaches 92–93 % of the clock-adjusted peak at large K.

### 4.4 C5, dequantized GEMV (m = 1)

**BitBLAS (the claim's artifact).** BitBLAS 0.1.0.post1 does not build on sm_120. Evidence is in
`bitblas_sm120_build_errors.txt` and `raw/gemv_bitblas_V0*.json`, where the cuBLAS rows were still
timed.
1. **As shipped.** `dispatch_scheduler` in `bitblas/ops/general_matmul/tilelang/dequantize/matmul_dequantize.py`
   only knows Volta, Ampere, Ada and Hopper, so all four formats fail with
   `Unsupported architecture … -arch=sm_120` / `No optimized function available`.
2. **With the TVM target forced to Ada** (`cuda -arch=sm_89`), scheduling works.
   - BitBLAS's lib generator compiles for the real device (`-gencode arch=compute_120,code=sm_120`).
   - Its bundled old TileLang `tl_templates/cuda/gemm.h` includes the Hopper `gemm_sm90.h` for every
     `__CUDA_ARCH_LIST__ >= 900`.
   - nvcc then fails with `identifier "warpgroup_wait" is undefined`, because sm_120 has no wgmma.

   Making BitBLAS work would mean patching its dispatcher and its bundled TileLang. That is out of
   scope, and the result would no longer be the claim's artifact. **Marlin, CUTLASS fpA_intB and
   bitsandbytes: not tested.**

**TileLang's own generator.** `examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py`
`dequantize_gemv`, imported unmodified, with its shipped fixed schedule: n_partition=4,
reduce_thread=32, 128 threads/CTA, no autotuner. Three variants:
- `uint4_fp16`: exactly the example's `main()` / `run_regression_perf()` configuration (uint4,
  lop3 fast decoding, fp16 accumulate);
- `int2_fp16`: num_bits=2 with the generic decode;
- `uint2_int8`: num_bits=2, int8 activations, int32 accumulate (dp4a), lop3.

The cuBLAS fp16 GEMV multiplies the same logical weights (as fp16), with `CUBLAS_COMPUTE_16F` like
the upstream `CUDA_R_16F` run. In the table, GB/s counts weight bytes only.

| shape | N,K | cuBLAS fp16 µs (GB/s) | uint4 µs (GB/s) | ×cuBLAS | claimed INT4A16 | verdict | int2 µs (GB/s) | ×cuBLAS | claimed INT2A16 | verdict | uint2·int8 µs (GB/s) | ×cuBLAS | claimed INT2A8 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 16384,16384 | 335.1 (1602) | 90.2 (1488) | 3.72 | 4.26 | partly | 51.2 (1311) | 6.55 | 5.14 | reproduced | 45.1 (1489) | 7.44 | 7.64 | reproduced |
| V1 | 43008,14336 | 776.3 (1588) | 194.6 (1585) | 3.99 | 4.34 | reproduced | 104.6 (1474) | 7.42 | 5.41 | reproduced | 98.3 (1568) | 7.90 | 8.34 | reproduced |
| V2 | 14336,14336 | 270.4 (1520) | 70.0 (1469) | 3.87 | 4.29 | reproduced | 41.0 (1254) | 6.60 | 5.01 | reproduced | 35.5 (1447) | 7.61 | 7.41 | reproduced |
| V3 | 57344,14336 | 1032.3 (1593) | 256.0 (1606) | 4.03 | 3.72 | reproduced | 135.3 (1519) | 7.63 | 5.52 | reproduced | 129.0 (1593) | 8.00 | 8.49 | reproduced |
| V4 | 14336,57344 | 1053.7 (1560) | 282.1 (1457) | 3.73 | 4.36 | partly | 149.5 (1375) | 7.05 | 5.46 | reproduced | 132.4 (1553) | 7.96 | 8.49 | reproduced |
| V5 | 9216,9216 | 114.6 (1482) | 32.8 (1296) | 3.50 | 3.95 | partly | 20.9 (1016) | 5.49 | 4.92 | reproduced | 18.4 (1152) | 6.22 | 7.56 | partly |
| V6 | 36864,9216 | 447.4 (1519) | 108.6 (1565) | 4.12 | 4.27 | reproduced | 59.5 (1428) | 7.52 | 5.13 | reproduced | 55.4 (1533) | 8.08 | 7.56 | reproduced |
| V7 | 9216,36864 | 425.9 (1595) | 115.0 (1477) | 3.70 | – | (not in figure) | 73.7 (1152) | 5.78 | – | – | 59.8 (1420) | 7.12 | – | – |

Geometric means over V0–V6, measured vs claimed: INT4 3.85 vs 4.16; INT2A16 6.86 vs 5.22; INT2A8 7.57
vs 7.91.

**Secondary timers** (speed-up vs cuBLAS fp16; `tilelang.profiler.do_bench` cupti / event /
`triton.testing.do_bench`), V0 and V1:
- V0: uint4 3.84 / 3.77 / 3.78; int2 6.38 / 6.14 / 6.13; uint2·int8 7.37 / 7.17 / 7.20.
- V1: uint4 3.93 / 3.89 / 3.89; int2 7.08 / 6.93 / 6.92; uint2·int8 7.56 / 7.40 / 7.41.
- All shapes are in `raw/gemv_tl_V*.json`. The GEMV runs are not power-capped (360–465 W, ~2.8 GHz).

**The example's own check is vacuous.** Its reference unpacks the *interleaved* weights, and it uses
`atol=1e3` with an fp16 accumulator. Against the example's reference, our correct outputs have a
relative error of 0.87–1.15, and would still "pass". Against the proper reference built from the
logical weights, the kernels are correct: rel. err. 2.4e-3 to 1.1e-2, within 3× cuBLAS-fp16acc's own
error; int32 outputs are exact.

### 4.5 Steady-state cross-check (C1, C6 fp16): primary for the GEMM verdicts

`cobench.bench_steady`, back-to-back under sustained load (method in section 3). Every variant in
every run sat at the 600 W cap (NVML median 600 W). All guard records are clean. "Flush" columns
repeat the flush-mode ratios of 4.1–4.3 for comparison.

At equal power, energy/iter ∝ time/iter, so the energy ratio equals the time ratio. The informative
split is **cycles/iter** (how much work-time the kernel needs) against **energy per cycle** (P/f: how
much power it draws per clock). A kernel with lower nJ/cycle can run at a higher clock under the cap.

**Suite `fp32acc`** (TileLang `benchmark/matmul` winner, cuBLAS, Triton tutorial 03 NT):

| shape | TL µs | cuBLAS µs | Triton µs | TL/cuBLAS steady (flush) | TL/Triton steady (flush) | MHz TL / cuBLAS / Triton | mJ/iter TL / cuBLAS | Mcycles/iter TL / cuBLAS | cycles cuBLAS/TL | nJ/cycle TL / cuBLAS | verdict vs cuBLAS / Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 242.0 | 278.5 | 247.0 | 1.151 (1.077) | 1.021 (1.039) | 2325 / 1712 / 2423 | 145 / 167 | 0.563 / 0.477 | 0.847 | 258 / 350 | reproduced / reproduced |
| M1 | 1705.0 | 1926.4 | 1783.8 | 1.130 (1.106) | 1.046 (1.044) | 1933 / 1605 / 1953 | 1023 / 1156 | 3.30 / 3.09 | 0.938 | 310 / 374 | reproduced / reproduced |
| M2 | 6132.6 | 7292.5 | 6222.1 | 1.189 (1.179) | 1.015 (1.023) | 1778 / 1720 / 1848 | 3680 / 4376 | 10.90 / 12.54 | 1.150 | 338 / 349 | reproduced / reproduced |
| M3 | 6110.3 | 7916.2 | 6336.8 | 1.296 (1.282) | 1.037 (1.038) | 1884 / 1632 / 1898 | 3666 / 4750 | 11.51 / 12.92 | 1.122 | 318 / 368 | reproduced / reproduced |
| M4 | 485.8 | 536.5 | 514.2 | 1.104 (1.086) | 1.058 (1.059) | 1771 / 1592 / 1783 | 292 / 322 | 0.861 / 0.854 | 0.993 | 339 / 377 | reproduced / reproduced |
| M5 | 3386.7 | 4470.3 | 3530.2 | 1.320 (1.313) | 1.042 (1.043) | 1818 / 1613 / 1836 | 2032 / 2682 | 6.16 / 7.21 | 1.171 | 330 / 372 | reproduced / reproduced |
| M6 | 11981.0 | 15681.6 | 12479.8 | 1.309 (1.305) | 1.042 (1.042) | 1771 / 1588 / 1798 | 7189 / 9409 | 21.22 / 24.91 | 1.174 | 339 / 378 | reproduced / reproduced |
| M7 | 11934.3 | 15560.2 | 12372.7 | 1.304 (1.290) | 1.037 (1.032) | 1808 / 1633 / 1810 | 7160 / 9336 | 21.57 / 25.42 | 1.178 | 332 / 367 | reproduced / **partly** |
| K256 | 145.3 | 152.6 | 157.1 | 1.050 (1.014) | 1.081 (1.000) | 2073 / 1919 / 1703 | 87 / 92 | 0.301 / 0.293 | 0.972 | 289 / 313 | (C6) |
| K512 | 249.9 | 257.6 | 257.5 | 1.031 (1.020) | 1.030 (1.030) | 1955 / 1865 / 2023 | 150 / 155 | 0.489 / 0.481 | 0.983 | 307 / 322 | (C6) |
| K1024 | 451.0 | 469.8 | 465.9 | 1.042 (1.027) | 1.033 (1.033) | 1969 / 1817 / 1964 | 271 / 282 | 0.888 / 0.854 | 0.961 | 305 / 330 | (C6) |
| K2048 | 850.5 | 884.3 | 876.4 | 1.040 (1.031) | 1.030 (1.033) | 1935 / 1813 / 1953 | 510 / 531 | 1.646 / 1.603 | 0.974 | 310 / 331 | (C6) |
| K4096 | 1646.5 | 1960.9 | 1697.2 | 1.191 (1.194) | 1.031 (1.033) | 1908 / 1602 / 1937 | 988 / 1177 | 3.14 / 3.14 | 1.000 | 314 / 374 | (C6) |
| K16384 | 6807.7 | 8787.9 | 7037.2 | 1.291 (1.272) | 1.034 (1.034) | 1787 / 1617 / 1825 | 4085 / 5273 | 12.16 / 14.21 | 1.168 | 336 / 371 | (C6) |

Geometric means over M0–M7: TL/cuBLAS **1.222** steady (1.201 flush); TL/Triton **1.037** steady
(1.040 flush).

**Suite `fp16acc`** (every provider accumulates in fp16; cuBLAS `CUBLAS_COMPUTE_16F`, tilelang-benchmark Triton NT):

| shape | TL µs | cuBLAS µs | Triton µs | TL/cuBLAS steady (flush) | TL/Triton steady (flush) | MHz TL / cuBLAS / Triton | mJ/iter TL / cuBLAS | Mcycles/iter TL / cuBLAS | cycles cuBLAS/TL | nJ/cycle TL / cuBLAS | verdict vs cuBLAS / Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 237.0 | 244.0 | 247.6 | 1.030 (0.998) | 1.045 (1.079) | 2348 / 1871 / 2476 | 142 / 146 | 0.556 / 0.456 | 0.821 | 256 / 321 | reproduced / reproduced |
| M1 | 1665.9 | 1724.5 | 1781.1 | 1.035 (1.024) | 1.069 (1.069) | 1953 / 1805 / 2002 | 1000 / 1035 | 3.25 / 3.11 | 0.956 | 307 / 332 | reproduced / reproduced |
| M2 | 5659.7 | 5996.6 | 6142.5 | 1.060 (1.053) | 1.085 (1.085) | 1995 / 1789 / 1919 | 3396 / 3598 | 11.29 / 10.73 | 0.950 | 301 / 335 | reproduced / reproduced |
| M3 | 5869.9 | 6453.3 | 6276.9 | 1.099 (1.084) | 1.069 (1.071) | 2015 / 1638 / 1961 | 3522 / 3872 | 11.83 / 10.57 | 0.894 | 298 / 366 | reproduced / reproduced |
| M4 | 473.5 | 470.9 | 504.4 | 0.994 (1.000) | 1.065 (1.069) | 1813 / 1814 / 1792 | 284 / 283 | 0.859 / 0.854 | 0.995 | 331 / 331 | reproduced / reproduced |
| M5 | 3239.6 | 3673.4 | 3492.4 | 1.134 (1.130) | 1.078 (1.078) | 1953 / 1673 / 1895 | 1944 / 2204 | 6.33 / 6.14 | 0.971 | 307 / 359 | reproduced / reproduced |
| M6 | 11301.9 | 12822.0 | 12259.0 | 1.134 (1.129) | 1.085 (1.090) | 1948 / 1654 / 1876 | 6781 / 7693 | 22.02 / 21.21 | 0.963 | 308 / 363 | reproduced / reproduced |
| M7 | 11505.9 | 12884.6 | 12287.7 | 1.120 (1.120) | 1.068 (1.072) | 1906 / 1629 / 1866 | 6904 / 7731 | 21.93 / 20.99 | 0.957 | 315 / 368 | reproduced / **partly** |

Geometric means: TL/cuBLAS **1.075** steady (1.066 flush); TL/Triton **1.070** steady (1.077 flush).

**C6 fp16 in steady state.**
- TileLang reaches 236, 275, 305, 323, 334, 325 and 323 TFLOPS for K = 256 … 16384 (K8192 = M5).
- That is 59–94 % of the mma.sync peak at the clock it ran, 1.03–1.32× cuBLAS (225–311 TFLOPS), and
  42–61 % of the H800 numbers.
- Energy efficiency: TileLang 0.39–0.56 TFLOP/J, cuBLAS 0.38–0.52 TFLOP/J.

**Flush vs steady.**
- Absolute times are 3–30 % longer in steady state, most for the short kernels. Clocks drop to
  1.59–2.48 GHz at 600 W; in flush mode they were 1.65–2.74 GHz.
- The TileLang/cuBLAS ratio is unchanged or larger in steady state on every C1 shape: +0.00…+0.07,
  largest on M0 (1.077 → 1.151). The one exception is fp16acc M4 (1.000 → 0.994).
- TileLang/Triton moves by ≤ 0.03, except K256 (1.00 → 1.08) and fp16acc M0 (1.08 → 1.05).
- **No verdict changes.**

## 5. Explanations (checked)

### 5.1 Why TileLang beats cuBLAS here (C1 and C6 above the claimed band)

**Measured: different kernel generations.** In the profiler, cuBLAS on sm_120 (12.8 from torch and
12.9 from the toolkit) runs sm_80 CUTLASS 2.x kernels (cp.async multistage, mma.sync), sometimes with
a split-K `Memset` (M0, M3, M7). cuBLASLt fp8 runs sm_89 xmma kernels. TileLang's tuned kernel adds
TMA bulk loads, mbarrier warp specialisation (1 producer warpgroup + 2 consumer warpgroups), setmaxnreg
and a TMA store.

**Split of the steady-state ratio into cycles and clock** (section 4.5).
- **Clock.** Under sustained load every variant sits at the 600 W cap. TileLang runs at a 3–36 %
  higher clock than cuBLAS (fp32 accumulate, C1 shapes; median +12.5 %), and 0–26 % higher with fp16
  accumulate.
  At equal power this means TileLang draws less energy per cycle: 258–339 nJ/cycle vs 349–378 nJ/cycle
  for cuBLAS (fp32 acc).
- **Cycles.**
  - Large shapes with fp32 accumulate (M2, M3, M5–M7, K16384): cuBLAS additionally needs 12–18 % more
    cycles.
    Example M5: 1.320 ≈ 1.171 (cycles) × 1.127 (clock).
  - Short-K or small-N shapes (M0, M1, M4, K ≤ 4096), and every shape with fp16 accumulate: cuBLAS
    needs 0–18 % *fewer* cycles. There TileLang's whole lead comes from the clock, i.e. from lower
    energy per cycle.
- **Energy per GEMM.** Because power is pinned at the cap, energy/iter = 600 W × time. cuBLAS spends
  1.10–1.32× (fp32 acc) and 0.99–1.14× (fp16 acc) the energy of TileLang for the same GEMM.

**Is the clock advantage genuine or an artefact?** It is a genuine energy-efficiency property of the
kernels on this power-capped card:
- **It survives sustained load.** The steady ratios equal or exceed the flush ratios (one exception,
  fp16acc M4, 0.006 below). If anything, the
  flush phase helped cuBLAS, the higher-power kernel, more than TileLang: on M0 the ratio rises from
  1.077 in flush mode to 1.151 in steady state.
- **It is not explained by DVFS.** cuBLAS runs at the lower clock, hence at a lower voltage, which by
  itself would *reduce* its energy per cycle. Its energy per cycle is nonetheless higher, so its
  switching activity per cycle is higher.
- **It shows up as lower energy per unit work.** TileLang reaches 0.39–0.56 TFLOP/J vs cuBLAS
  0.38–0.52 TFLOP/J (C6 sweep), and 0.47 vs 0.41 on M0.

**What was not measured.** The *cause* of the lower power per cycle was not measured: Nsight Compute
is unavailable (performance counters are admin-only). Plausible contributors, all unverified:
- TMA loads issued by one elected thread, instead of cp.async plus address arithmetic in all
  threads;
- 128×256 tiles, with fewer shared-memory/L2 bytes per FLOP than cuBLAS's 64×256 and 128×64 tiles;
- cuBLAS's split-K workspace traffic on M0, M3 and M7.

**What this does not transfer to.** The advantage is a result for *this* power-capped card and these
cuBLAS versions (sm_80/sm_89 kernels on sm_120). It does not say that TileLang generates better code
than a cuBLAS with native sm_120 kernels would.

### 5.2 C1 vs Triton, M7 "partly"

- The claimed 1.24× is the 4090 bar pair: TileLang 1.015× and Triton 0.82× of cuBLAS. That is, Triton
  was unusually slow on the 4090 for M7.
- On sm_120, Triton-tut03 is 1.25× cuBLAS on M7, and TileLang is still 1.03× (fp32 acc) and 1.07×
  (fp16 acc) faster than Triton.
- The deficit is Triton on the 4090, not TileLang here.
- On the other shapes, TileLang/Triton is 1.02–1.09×, in line with the paper's 1.08× mean.

### 5.3 C5 INT4 "partly" on V0, V4, V5

- **The claim sits at the byte-ratio limit.** A memory-bound fp16→int4 GEMV can gain at most the
  weight-byte ratio of 4×, times the ratio of achieved bandwidths.
  - The A100 claims of up to 4.36× mean the A100 cuBLAS GEMV was below the int4 kernel's bandwidth.
  - On this card, cuBLAS fp16 GEMV already streams 1.48–1.60 TB/s, 83–89 % of the 1.79 TB/s DRAM
    peak. So even an int4 kernel running at the full peak would top out at 4.5–4.8×.
- **The example's shipped schedule reaches 1.30–1.61 TB/s.** It uses 4-byte weight loads per thread
  per iteration and 4 output rows per 128-thread CTA, with no tuning.
  - On V1, V3 and V6 it matches cuBLAS's bandwidth: 3.99–4.12×.
  - On V0, V2 and V4 it gets 93–97 % of cuBLAS's bandwidth: 3.72–3.87×.
- **V5 (9216², 42 MB of int4 weights, 33 µs).** Launch and ramp overhead is a large fraction of the
  run, so bandwidth drops to 1.30 TB/s. The same holds for INT2A8 on V5 (18 µs).
- **Not a fork effect.** The kernel is a plain SIMT GEMV (no TMA, no cp.async, no mma), and the fork's
  source changes do not apply to it (5.5).

### 5.4 C5 BitBLAS / NF4: not testable

The two independent build failures are described in 4.4 and recorded in
`bitblas_sm120_build_errors.txt`. Both are in BitBLAS 0.1.0.post1 and in its bundled 2024-era
TileLang, not in this fork. The NF4 format has no path in the TileLang example.

### 5.5 Fork effect vs sm_120/config effect

The fork's changes to `src/` and `tilelang/` relative to upstream `7e3bbf37` fall into five groups.
None of them is active in the kernels measured here:
- **cp.async L2 eviction hints.** Only used with `T.copy(..., eviction_policy=...)`.
  `kernel_facts.fork_l2hint` is false for every kernel, and the tuned kernels use TMA, not cp.async.
- **`kSharedLifetimeScope` in MergeSharedMemoryAllocations.** Only emitted by the research CoKernel
  builder.
- **A `Simplify` of the pipeline-epilogue extent.** It is a no-op for these static shapes.
- **Cache stamping.**
- **Four lines in the nvrtc and cutedsl adapters**, which are not used here.

The one TileLang configuration that is slow on sm_120, the example_gemm_autotune heuristic
(`num_stages=0`), is an example-config effect (sm_120 falls into the generic branch). Upstream
TileLang was not built for a side-by-side diff of the generated code. This conclusion rests on the
diff above plus the kernel facts in `raw/*.json` and the sources in `kernels/*.cu`.

## 6. Other problems found (upstream, not fork)

1. `benchmark/matmul_fp8/benchmark_matmul.py`: `get_configs(args, kwargs)` is incompatible with the
   current autotuner, so the script as shipped raises `TypeError`.
2. The TileLang tvm_ffi backend with an fp8 output (`out_idx`) under torch 2.8 fails with
   `MemoryError: Unsupported code 10` (DLPack float8_e4m3fn). The cython backend works.
3. `examples/dequantize_gemm/quantize/utils.py` `interleave_weight`: the 1-bit and 2-bit branches call
   `torch.int32(0x…)`, which is a dtype and not callable, so they raise `TypeError`. The 2-bit and
   1-bit lop3 fast-decoding paths of the GEMV example are therefore unusable with fp16.
4. `example_dequant_gemv_fp16xint4.main()` checks against a wrong reference with `atol=1e3`, so the
   check cannot fail (4.4).
5. `examples/gemm/example_gemm_autotune.py` `get_heuristic_config()`: on anything but sm_80/sm_90 it
   uses `num_stages=0`, which is 2.7–4× slower than the tuned kernel on sm_120.
6. Configs above 99 KB of shared memory (for example 256×256 tiles) are in both benchmark search
   spaces. They compile, and fail only at launch (tvm_ffi) or at module load (cython).
7. The autotuner writes `autotuner.log` into the working directory. The cython backend leaves
   `tmp*.cu/.so` in `/tmp`, about 1.1 MB per compiled config.

## 7. Reproduce

```bash
cd /home/ywc/co-tilelang && source research/env.sh
python research/bench/scripts/claims_gemm_figures.py --out research/results/2026-09-24_claims_repro/A_gemm/claimed_ratios_digitized.json
bash research/bench/scripts/claims_gemm_all.sh compile   # CPU only
bash research/bench/scripts/claims_gemm_all.sh run       # each job: flock <gpu.lock> ... claims_gemm_tl.py run --suite S --shape X
bash research/bench/scripts/claims_gemm_all.sh xcheck    # cuBLAS 12.9 (LD_PRELOAD) and NN-layout cross-checks
bash research/bench/scripts/claims_gemm_all.sh steady    # bench_steady cross-check (needs the run step's raw/*.json)
bash research/bench/scripts/claims_gemm_all.sh gemv      # TileLang GEMV variants; BitBLAS needs reinstalling (env_versions.md §7)
python research/bench/scripts/claims_gemm_report.py --md /tmp/tables.md   # tables + summary.json
```

A single job, for example:

```bash
flock <gpu.lock> python research/bench/scripts/claims_gemm_tl.py run --suite fp32acc --shape M5
```

Files:
- `raw/<suite>_<shape>[_tag].json`: per run, with fields
  - `summary`: median µs, TFLOPS, clock, speed-ups;
  - `cobench`: the full result, including NVML and the guard record;
  - `secondary`, `correctness`, `tl_tune`: the best config, autotuner latency, kernel facts, smem;
  - `kernel_names`: library kernels from the profiler;
  - `blas_libs`.
- `raw/steady_<suite>_<shape>.json`: steady-state runs, with `summary` (time/iter, clock, power,
  energy/iter, kcycles/iter, TFLOPS, TFLOP/J, speed-ups) and `cobench_steady` (the full result,
  including slices, the thermal warmup and the guard record).
- `raw/gemv_tl_V*.json`, `raw/gemv_bitblas_V0*.json`.
- `configs/<suite>_<shape>.json`: every config with its launch smem and whether it was kept.
- `kernels/*.cu`: the tuned TileLang kernels.
- `claimed_ratios_digitized.json`, `summary.json`, `bitblas_sm120_build_errors.txt`.

## 8. Open issues

- **Other baselines.** Marlin, CUTLASS fpA_intB and bitsandbytes (C5) were not tested. BitBLAS
  would need source patches (dispatcher and bundled TileLang templates) to run on sm_120.
- **fp16acc recompilation.** In the fp16acc suite, the autotuner recompiled configs that the compile
  phase had already built. The cause (a cache-key mismatch between `AutoTuner.from_kernel` and
  `CompileArgs.compile_program`) was not investigated.
- **fp8 peak.** The e4m3 mma.sync peak of 2048 FLOP/clk/SM used in 4.3 is assumed (2× fp16), not
  measured.
- **Steady state not run for C6 fp8 and C5.** fp8 GEMM is power-capped too, so its steady-state
  ratio may differ from the flush-mode 1.08–1.51×. The GEMVs run far below the cap (360–465 W), where
  the two modes should agree.
- **Leftover cache.** `~/.tilelang/cache/0.1.14_cuda/cuda-binaries` (2.5 GB, shared with other
  sub-studies) may still hold cubins from these compiles; they could not be attributed per study.
