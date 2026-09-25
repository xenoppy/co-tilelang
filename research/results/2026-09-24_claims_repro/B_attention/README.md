# Sub-study B: attention claims (C2 FlashAttention forward, C4 MLA decode)

> 2026-09-24, RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs, 99 KB smem per CTA, 600 W cap, no
> wgmma/tcgen05). Part of `../README.md` (claims C2 and C4, plus the optional attention-sink claim).
> Status: **done**. Scripts: `research/bench/scripts/claims_attn_*.py`. Raw data: `raw/`. Tables: produced by
> `claims_attn_report.py` (`summary_median_us.json`, `summary_mean_us.json`).

## 0. Verdicts

Ratio = baseline time / TileLang time (> 1 means TileLang is faster), the quantity the upstream figures plot.
Rule (claims README §2.5; "partly" made explicit): **reproduced** if measured >= 0.9 x claimed; **partly** if
TileLang is still faster (measured > 1) but below that; **not reproduced** if measured <= 1. All implementations
of a shape are timed interleaved in one process with a clean L2 flush before every rep (cobench). **Which timer:**
C2 uses the CUDA-graph clean-flush timer ("graph-flush", N calls on N cold input copies per rep; §4), because
eagerly launched kernels on this GPU complete on a 2.048 µs grid, which made the eager per-kernel times of the
20–80 µs C2 kernels coarse (±5–10 %); the steady-state graph timer is given in brackets as a second check. C4 and
the sink claim use the eager clean-flush timer (all rows >= 135 µs, quantisation <= 1.5 %; spot-checked with the
graph timers, §6).

| claim | baseline | claimed (H100, from the figure / paper) | measured on sm_120 | verdict |
|---|---|---|---|---|
| C2 FlashAttention fwd, FA0–FA4 | Triton (fused-attention tutorial kernel, fastest of 3 setups) | 1.29–1.60x, geomean 1.41 | 1.05–1.33x, geomean **1.18** [steady 1.17] | **partly** (FA2 reproduced, 4 shapes partly) |
| | PyTorch (= FlashAttention-2 via SDPA) | 1.45–2.17x, geomean 1.70 | 1.00–1.79x, geomean **1.27** [steady 1.28] | **partly** (FA1 reproduced; FA0/FA2/FA4 partly; FA3 at parity, 1.002) |
| | FlashAttention-3 | 0.91–2.19x, geomean 1.36 | — | **not testable on sm_120** (sm_90a-only kernels) |
| C4 MLA decode, batch 64/128 x 1K–32K | FlashMLA | 0.92–1.06x ("comparable", "up to 98 %") | — | **not testable on sm_120** (dense decode is SM90-only) |
| | FlashInfer (H100: fa3 backend; here: fa2, the only fp16 backend that runs) | 1.11–1.35x, geomean 1.23 | 0.97–1.00x, geomean **0.98** | **not reproduced** (parity) |
| | Triton (`benchmark_mla.py` kernel) | 3.0–12.7x, geomean 5.5 (11 shapes) | 2.97–6.23x, geomean **4.22** | **partly** (2 of 11 reproduced; 12th shape: the Triton kernel faults, int32 overflow) |
| | Torch (`benchmark_mla.py` reference loop) | 1075.9x (paper) | 305–506x, geomean 350 (upstream timer) | **partly** (orders of magnitude, but a CPU-bound loop) |
| optional: attention sink (README, H800) | Triton (benchmark's kernel) | 1.21–1.35x | head_dim 64: 1.11–1.17x; head_dim 128: 1.70–1.78x | head_dim 64 **reproduced** (3 of 4 at >= 0.9x claim, one at 0.89x); head_dim 128 **reproduced**, but inflated: the Triton kernel spills on sm_120 |

Main findings:
1. **None of TileLang's shipped configs for these claims runs on sm_120 as is.** The FlashAttention autotune /
   H100 config needs 160 KB of shared memory and the MLA example 226 KB (both sized for H100's 227 KB; lowering
   the same programs for sm_90a gives identical byte counts, `crosslower.json`). Configs over 99 KB compile and
   fail at launch. With the smallest change that fits (FA: 2 → 1 pipeline stage; MLA: 2 → 1 stage and
   block_H 64 → 16; block_N < 64 is rejected by TileLang's layout inference on the sm_120 mma path) the kernels run
   and are correct.
2. **FlashAttention: TileLang is a good sm_120 kernel, but the H100 margins shrink.** TileLang is at parity with
   NVIDIA's cuDNN SDPA kernel (geomean 0.97x graph-flush, 1.03x steady) and 1.23x faster than flash-attn 2.8.3. The claimed margins over
   Triton and FA2 were measured where TileLang used wgmma + TMA warp specialisation and FA2 could not; on sm_120 all
   of them issue `mma.sync` (TileLang's sm_120 build: `mma`, no `wgmma`; its sm_90a build of the same program:
   `wgmma`), so the Hopper-specific advantage has nothing to act on.
3. **MLA: parity with FlashInfer, not 1.2–1.35x.** The config TileLang has to fall back to (1 stage, block_H 16)
   runs at 0.53–0.58 TB/s and 111–128 TFLOPS (30–33 % of DRAM peak, 21–24 % of MMA peak): neither bound,
   latency-limited; FlashInfer's fa2 backend is 0–3 % faster on every shape with the eager timer (the graph
   timers put it 1–3 % *slower* at batch 64 / 1K: parity within the timers' disagreement).
4. Two baseline defects found on the way (§7.4): `benchmark_mla.py`'s Triton kernel overflows int32 offsets at
   batch 128 / 32K (illegal address here; the H100 figure plots a value for that point, and `benchmark_mla.py
   --all` does not check correctness); `examples/attention_sink/benchmark_gqa_sink_fwd.py` imports a module that
   upstream deleted.

## 1. Environment changes

| change | details | disk |
|---|---|---|
| `pip install flash_attn==2.8.3.post1` (prebuilt wheel `+cu12torch2.8cxx11abiTRUE-cp312`, GitHub release, `--no-deps --no-cache-dir`, dry-run first, under `pip.lock`) | only package added; every existing version unchanged (`pip list` diff). The `.so` carries sm_80/90/100/120 SASS (`cuobjdump --list-elf`: 72 cubins each), so it runs natively. Recorded in `research/env_versions.md` §6. Uninstall: `pip uninstall flash_attn`. | +960 MB site-packages |
| FlashInfer 0.7.0 JIT modules (already installed) | new cached ops in `~/.cache/flashinfer/0.7.0/120f/`: fp16 `single_prefill` (hd 128), `batch_mla_attention` fp16 (ckv 512, kpe 64), `xqa_mla` bf16 (page 64) | ~tens of MB |
| TileLang kernel cache `~/.tilelang/cache` | ~40 FA configs x 5 shapes, 9 MLA kernels x 12 shapes | small |
| vendored upstream sources (verbatim, MIT) | `research/bench/baselines/upstream/tlbench_triton_mha.py` (tilelang-benchmark @ b658f7e) and `triton340_tutorial06_fused_attention.py` (Triton v3.4.0 tutorial 06); provenance + sha256 in that package's `__init__.py`. Both `import pytest` (not installed): the scripts register a no-op stub | <100 KB |

Not installed: FlashAttention-3 and FlashMLA (Hopper-only, §3), `cuda-tile[tileiras]` (needed by FlashInfer's cuTile MLA backend, CUDA 13.1+ toolchain). Nothing in TileLang (compiler, examples, benchmark scripts) was modified.

## 2. Claims, shapes and sources

**C2 (FlashAttention forward, `images/mha_performance_h100.png`, top panel).**
- Shapes FA0–FA4 = tilelang-benchmark `README.md` Table 2 (tile-ai/tilelang-benchmark @ b658f7e) = TileLang paper
  (arXiv 2504.17577 v2) appendix Table 3: batch 1, 32 heads, head_dim 128, fp16; seq_len 512 / 512 / 1024 / 1024 / 4096;
  causal T / F / T / F / T.
- "PyTorch" = FlashAttention-2 inside PyTorch: paper §5.1 ("PyTorch, featuring hand-optimized kernels like GEMM and
  FlashAttention-2") and §5.2 ("PyTorch uses a hand-optimized FlashAttention-2 kernel"). In torch 2.8 that is
  `scaled_dot_product_attention` with the FLASH backend (`sdpa_flash`). The tilelang-benchmark repo's own
  "torch" script (`hopper_benchmark/flashattention/0.torch_benchmark/benchmark_torch_mha.py`) instead times an
  einsum + masked softmax + einsum program; it is reported as `torch_naive`. All four SDPA backends are reported.
- "Triton" = the fused-attention kernel of Triton's tutorial 06; the tilelang-benchmark copy
  (`hopper_benchmark/flashattention/2.triton_benchmark/benchmark_triton_mha.py`) takes a TMA-descriptor path on
  capability >= 9 and was benchmarked with `warp_specialize=True`. Reported: that copy with ws=True
  (`triton_tlbench_ws`) and ws=False (`triton_tlbench`), and Triton 3.4.0's own tutorial 06 with ws=False
  (`triton_tut06`, the tutorial's setting on non-Blackwell GPUs). The verdict uses the fastest of the three.
- Provenance caveat: the figure was committed to TileLang on 2025-01-11 (4d63633a) and to tilelang-benchmark on
  2025-01-20; the FA benchmark scripts arrived later (2025-05-14, 99bcbad) and their default shapes (batch 64,
  64 heads, seq 1024–8192) are not FA0–FA4. There is no data file for the figure; the paper text also mentions
  8k sequences, which Table 2 does not contain. The repo additionally contains `debug.py`, which perturbs four
  hard-coded numbers by a random ±5 %; nothing references it and I could not tie it to any figure.
- Claimed ratios were read off the figure (`claims_attn_digitize.py`, pure-Python PNG decode; bar height / TileLang
  bar height, ±0.014): they reproduce the paper's stated geometric means (FA3 1.363 vs "1.36x", Triton 1.408 vs
  "1.41x", PyTorch 1.703 vs "1.70x"), so the reading is exact enough. Values: `claims_digitized.json`.

| | FA0 | FA1 | FA2 | FA3 | FA4 | geomean (paper) |
|---|---|---|---|---|---|---|
| FA3 / TileLang | 2.19 | 2.00 | 1.11 | 1.06 | 0.91 | 1.36 (1.36) |
| Triton / TileLang | 1.60 | 1.49 | 1.29 | 1.34 | 1.34 | 1.41 (1.41) |
| PyTorch / TileLang | 1.76 | 1.65 | 1.57 | 1.45 | 2.17 | 1.70 (1.70) |

**C4 (MLA decode, `examples/deepseek_mla/figures/bs{64,128}_float16.png`, README "comparable to FlashMLA ...
significantly outperforming both FlashInfer and Triton").**
- Workload = `examples/deepseek_mla/benchmark_mla.py` `shape_configs`: s_q = 1, 128 query heads, 1 KV head,
  d = 576 (512 + 64 rope), dv = 512, fp16, page size 64, identity block table, request i has length L + 2i,
  L in {1024, ..., 32768}; batch 128 in the script, batch 64 and 128 in the figures (both run here). FLOPs as in
  the script: `sum(len) * 128 * (576 + 512) * 2`.
- Provenance caveat: the figures were committed on 2025-03-04 (#137), two days before `benchmark_mla.py` and the
  paged kernel (2025-03-06, #158); the script that produced them is not in the repo. `benchmark_mla.py` (adapted
  from FlashMLA's benchmark, same length pattern) is the closest available harness.
- Claimed ratios (digitized marker centres, ±0.6 TFLOPS; `claims_digitized.json`): TileLang/FlashMLA 0.92–1.06,
  TileLang/FlashInfer 1.11–1.35, TileLang/Triton 2.96–12.7 (falls with context length); Torch reads as 0
  TFLOPS in the figure; the paper (§5.2) states "a 1075.9x speedup over Torch" and "up to 98% of ... FlashMLA".

| batch | L | FlashMLA | TileLang | FlashInfer | Triton | TL/FlashInfer | TL/Triton | TL/FlashMLA |
|---|---|---|---|---|---|---|---|---|
| 64 | 1024 | 306 | 325 | 281 | 28 | 1.16 | 11.8 | 1.06 |
| 64 | 2048 | 401 | 413 | 329 | 57 | 1.26 | 7.2 | 1.03 |
| 64 | 4096 | 465 | 428 | 358 | 86 | 1.20 | 5.0 | 0.92 |
| 64 | 8192 | 492 | 457 | 377 | 114 | 1.21 | 4.0 | 0.93 |
| 64 | 16384 | 486 | 475 | 352 | 135 | 1.35 | 3.5 | 0.98 |
| 64 | 32768 | 485 | 469 | 354 | 150 | 1.32 | 3.1 | 0.97 |
| 128 | 1024 | 358 | 348 | 313 | 27 | 1.11 | 12.7 | 0.97 |
| 128 | 2048 | 438 | 412 | 348 | 56 | 1.18 | 7.3 | 0.94 |
| 128 | 4096 | 483 | 447 | 372 | 86 | 1.20 | 5.2 | 0.93 |
| 128 | 8192 | 494 | 471 | 371 | 114 | 1.27 | 4.1 | 0.95 |
| 128 | 16384 | 490 | 479 | 366 | 135 | 1.31 | 3.5 | 0.98 |
| 128 | 32768 | 476 | 447 | 368 | 151 | 1.22 | 3.0 | 0.94 |

**Optional: attention sink** (`examples/attention_sink/README.md`): GQA forward with sinks, bf16, batch 1,
64 heads / 8 KV heads, causal, seq 2048–16384, head_dim 64/128; TileLang 1.21–1.35x faster than the benchmark's
Triton kernel on H800 (table in that README).

## 3. What was run, and every adaptation

Correctness: every timed kernel is compared with an fp32 torch reference on the same inputs
(|out − ref| <= 1e-2 + 1e-2 |ref|, the upstream scripts' `assert_close` tolerance; bf16 sink: 2e-2); all timed
kernels passed on all shapes (max abs errors in `raw/*.json`, `variants.<name>.check`). A kernel that fails the
check is excluded from timing and listed (none were).

**C2** (`claims_attn_fa.py`, one process per shape; each implementation receives the same values in its native
layout, BHSD or BSHD/NHD, created before timing):

| name | what | adaptation for sm_120 |
|---|---|---|
| `tl_bhsd_h100cfg_s1` (**TileLang, primary**) | `example_mha_fwd_bhsd.flashattn`, config of the example's autotuner and of tilelang-benchmark's H100 script: block 128x128, 256 threads, **2 → 1 pipeline stage** | shipped config needs 160 KB dynamic smem; verified to compile and then fail at launch (`Failed to set the allowed dynamic shared memory size to 163840`, isolated child process). With 1 stage: 98 KB |
| `tl_bhsd_main`, `tl_bshd_main`, `tl_bshd_tune` | the two examples' other shipped configs (64x64/1/128, 128x128/1/128, 64x64/1/128) | none (they fit) |
| `tl_bhsd_sweep` | best of block_M {64,128} x block_N {32,64,128} x stages {1,2,3} x threads {128,256} that fits 99 KB (18 launchable per shape; 9 exceed 99 KB; 9 do not compile: block_M 64 with 256 threads has no valid warp layout), chosen with `tilelang.profiler.do_bench` like TileLang's autotuner | sensitivity only, not "as shipped" |
| `triton_tlbench_ws` | tilelang-benchmark's Triton kernel exactly as its benchmark called it (warp_specialize=True, own autotuner) | none; for the non-causal shapes no autotune config fits 99 KB (the one it falls back to needs 123,004 B) → `OutOfResources`, reported as unlaunchable |
| `triton_tlbench`, `triton_tut06` | same kernel with ws=False; Triton 3.4.0 tutorial 06 (ws=False) | none |
| `sdpa_flash` (= "PyTorch"), `sdpa_cudnn`, `sdpa_efficient`, `sdpa_math` | torch 2.8 SDPA with one backend forced (`sdpa_kernel`) | none |
| `torch_naive` | `example_mha_fwd_bhsd.ref_program` (= tilelang-benchmark's torch program) | none |
| `flashinfer_prefill`, `fa2_pip` | `flashinfer.single_prefill_with_kv_cache` (NHD, auto backend → FA2 template on sm_120); `flash_attn.flash_attn_func` 2.8.3 | none |
| FlashAttention-3 | **not testable on sm_120**: FA3 builds only `-gencode arch=compute_90a,code=sm_90a` for its Hopper kernels (`hopper/setup.py`, Dao-AILab/flash-attention @ eed1971, `cc_flag`), README "Requirements: H100 / H800 GPU"; no wheel exists. `sm_90a` cubins cannot load on cc 12.0 (arch-specific targets are not forward compatible; the same failure is observed live for FlashInfer's FA3 MLA backend: "no kernel image is available for execution on the device") | — |

Triton's autotune key in both vendored kernels omits the causal flag, so FA0 and FA1 (same N_CTX) would share one
tuned config inside a process; each shape therefore runs in its own process.

**C4** (`claims_attn_mla.py`, one process per (batch, L); inputs exactly as `benchmark_mla.compare_a`: seed 0,
`randn` fp16 q and paged cache, split nope/pe copies as the script makes them):

| name | what | adaptation for sm_120 |
|---|---|---|
| `tl_adapted` (**TileLang**) | `example_mla_decode_paged.mla_decode_tilelang` (what `benchmark_mla.py` times), block_N 64, num_split 1, **block_H 64 → 16, KV pipeline stages 2 → 1** | shipped config (block_H 64, 2 stages) needs 226 KB dynamic smem (sized for H100's 227 KB; the sm_90a lowering needs the same 231,424 B) → fails at launch (verified). `num_stages=2` is hard-coded in the example; the wrapper loads the unmodified file and re-parameterises exactly its two `T.Pipelined(loop_range, num_stages=2)` (with 2 it reproduces the example's TIR text exactly, checked every run). First config that fits, in the order stages (2,1) x block_H (64,32,16): smem (static + dynamic, KiB) 228/188/168/156/116/**96**. block_N < 64 is not an option: it fails TileLang layout inference on sm_120 (`Layout infer conflict between scores_max_prev and scores_scale`; it compiles for sm_90a, where the gemm lowers to wgmma) |
| `tl_adapted_split2/4` | same with num_split 2 / 4 | sensitivity |
| `triton_mla` | `benchmark_mla.py`'s `_mla_attn_kernel` + `_mla_softmax_reducev` (BLOCK_H 16, BLOCK_N 64, 32 KV splits) | default num_stages (3) needs 102,656 B, 2 stages 102,400 B (limit 101,376) → launched with num_stages=1 through a copy of the 20-line launcher; kernels imported unmodified |
| `flashinfer_mla_fa2` (**FlashInfer**) | `BatchMLAPagedAttentionWrapper(backend="fa2")`, split query/cache as in `benchmark_mla.py`, causal=True | `benchmark_mla.py` asks for `backend="fa3"`: fails on sm_120 ("no kernel image is available", verified in an isolated child). fa2 is FlashInfer's default MLA backend off sm_90 |
| `flashinfer_xqa_bf16` | `flashinfer.mla.xqa_batch_decode_with_kv_cache_mla`, FlashInfer's SM120 MLA decode kernel | bf16 (it accepts only bf16/fp8), packed cache; extra reference |
| FlashInfer cuTile backend | `backend="cutile"` (lists cc 12.0) | not runnable here: needs the `tileiras` compiler (CUDA 13.1+ / `cuda-tile[tileiras]`), and rejects causal=True |
| `torch_mla` | `benchmark_mla.run_torch_mla` (per-request fp32 loop) with its own `triton.testing.do_bench` | none; not under cobench (it synchronises the host via `cache_seqlens[i]`) |
| FlashMLA | **not testable on sm_120**: dense decoding is SM90-only (FlashMLA README support matrix, deepseek-ai/FlashMLA @ ba89a34); `setup.py` builds only `sm_90a`, `sm_100a`, `sm_103a` | — |

**Optional sink** (`claims_attn_sink.py`): `examples/attention_sink/benchmark_gqa_sink_fwd.py` does not run as
shipped: it imports `example_gqa_sink_fwd_bhsd`, which is not in the tree (upstream #1909 / ded6a992 changed the
import and deleted `example_gqa_sink_fwd_bhsd_wgmma_pipelined.py`, the kernel the README table was produced with,
#885). The wrapper registers that deleted file, read verbatim from git (`ded6a992^`), under the missing name and
imports the benchmark unmodified. `tl_fixed` = the benchmark's fixed config (128x128, 2 stages, 256 threads:
fits only at head_dim 64); `tl_tuned` = the example's autotuner (block 128x128, stages {0,1,2}, threads
{128,256}). The reference is a query-chunked fp32 implementation (the example's `ref_program` needs 69 GB at
seq 16384).

## 4. Measurement

- Eager clean-flush timer (the original primary; still the timer of record for C4 and the sink claim):
  `cobench.bench_variants` — all implementations of one shape in one process, interleaved in rotating
  order, clean 256 MB L2 flush before every rep (excluded), host gate (launch latency excluded), ≥ 1 s / 20-rep
  warmup, 200 reps (C2) / 100 reps (C4) / 50 reps (sink), `clock=True` (ClockProbe) and NVML power/temperature.
  Statistic: median (mean within 3.1 % of the median for every variant, within 1 % for all MLA and sink rows; raw
  samples in `raw/*.json`, `cobench.samples_us`). A few C2 variants show isolated +~175 µs reps (1–2 of 200, e.g.
  FA2 `sdpa_flash`, FA3 `sdpa_cudnn`) that inflate their CV to 16–24 %; the median is insensitive to them.
- **Timer quantum (found in review; `claims_attn_timerprobe.py`, `raw/timerprobe.json`).** cobench's flush-mode
  times are CUDA-event pairs (`torch.cuda.Event.elapsed_time`) around the launch. The event timestamps themselves are
  fine-grained (two events around nothing: 0.35–0.42 µs, 32 ns steps), but a kernel launched eagerly on a stream is
  *seen complete* on a ~2.048 µs grid: a kernel that spins for a set time (on `%globaltimer`, or on `clock64` — the
  two agree, so the global timer is linear) swept in 128 ns steps gives event durations in 2.048 µs steps (16.0–18.2 µs
  of spin → 18.4–20.5 µs, 18.3–20.2 → 22.5, 20.5–22.3 → 24.6, ...; event − kernel = +2.3 … +4.0 µs). Eight eager
  back-to-back launches per rep do **not** remove it (the per-call time still steps by ~2 µs), so it is a per-kernel
  completion granularity of stream execution, not only an end-point effect. Inside a CUDA graph the same kernels run
  back-to-back with continuous per-call time (kernel + ~0.9 µs). Consequence: eager clean-flush medians of the
  20–80 µs C2 kernels sat on 2.048 µs multiples (20.48, 22.53, 28.67, 32.77 µs, ...), each up to ~2 µs from the true
  kernel time; the >= 135 µs rows of C4 and the sink claim are affected by <= 1.5 %.
- **Resolution-free C2 timers (`claims_attn_graph.py`).** Each implementation is captured in CUDA graphs, all
  implementations interleaved as before: (1) *graph-flush* — `cobench.bench_variants` (clean flush, host gate,
  100 reps) of one graph replay of N calls, each on its own input copy (all DRAM-cold after the flush),
  N = ceil(300 µs / t) = 14 / 14 / 6 / 4 / 1 for FA0–FA4, per-call = rep / N (residual quantisation
  <= 2.048 µs / (N t) < 1 %; 0.4 % at FA4 with N = 1); (2) *steady* — `cobench.bench_steady` (thermal warmup,
  5 rounds x 1.5 s slices, rotating order) of graph replays cycling through >= 2x L2 of input copies (back-to-back,
  DRAM-cold inputs, sustained power), per-call = iteration / calls. All graph outputs were checked against the
  reference; guard records clean. Graph-flush is the C2 number of record: it keeps everything of the eager protocol
  except the per-kernel completion quantum.
- Secondary timer: what the upstream scripts use — `tilelang.profiler.do_bench` (TileLang) and
  `triton.testing.do_bench` (baselines); both flush L2 and return the mean, but include host launch overhead,
  which matters for the 20–80 µs C2 shapes.
- GPU policy: every run under `flock gpu.lock` (shared with sub-studies A and C), `cb.wait_until_free()` before
  starting, pmon guard around every timed window: all windows clean (`cobench.guard` in each JSON). A foreign
  user's job (qzr, ray worker) showed SM activity once; the guard held the MLA batch 64 / 8K run for the 30-min
  poll, then measured it clean. Otherwise foreign processes only held memory (up to 43 GB).
- Expected-failure launches (the shipped TileLang configs, FlashInfer's fa3 backend, the Triton MLA kernel at
  batch 128 / 32K) run in child processes. Reason: while FlashInfer's fa3 attempt ran in-process, the next TileLang
  launch failed with `main_no_split_kernel: CUDA_ERROR_NO_BINARY_FOR_GPU`; the same kernel runs in a clean process,
  a failed TileLang launch alone does not cause it, and moving the fa3 attempt into a child process removed it —
  consistent with fa3's `cudaErrorNoKernelImageForDevice` staying in the CUDA runtime's last-error slot and being
  picked up by the next library's launch check. The scripts also reset the last error of every loaded libcudart
  after a recorded failure. (An illegal address, as in the Triton MLA case, kills the context, hence the child.)

## 5. C2 results (FlashAttention forward, FA0–FA4)

"TileLang primary" is the H100 config with 1 stage (§3); "TL sweep best" is an sm_120 upper bound, not a shipped
config. Numbers of record: the CUDA-graph timers (first two tables). The eager clean-flush tables after them were
the original primary; they are kept for comparison and are quantised to 2.048 µs for the small shapes (§4).

#### C2 per-call times with the CUDA-graph timers (us): graph-flush (**primary for C2**) / steady

| shape (calls per graph-flush rep) | TileLang primary | TL bhsd main | TL bshd main | TL bshd tune | TL sweep best | Triton tlbench ws | Triton tlbench | Triton tut06 | SDPA flash | SDPA cuDNN | SDPA efficient | SDPA math | torch naive | FlashInfer | flash-attn 2.8.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FA0 (14) | 19.75 / 19.85 | 19.05 / 19.44 | 30.72 / 30.69 | 18.77 / 19.10 | 19.92 / 20.02 | 30.69 / 30.78 | 26.19 / 26.39 | 26.19 / 26.45 | 29.29 / 30.56 | 20.03 / 20.20 | 34.89 / 35.16 | 220.44 / 233.06 | 76.19 / 79.41 | 23.70 / 24.10 | 28.95 / 30.22 |
| FA1 (14) | 20.80 / 21.64 | 24.77 / 25.79 | 30.87 / 31.58 | 24.70 / 25.78 | 20.90 / 21.89 | – | 23.59 / 24.45 | 23.68 / 24.58 | 37.13 / 39.60 | 20.92 / 23.29 | 45.28 / 46.33 | 193.31 / 213.70 | 51.32 / 56.75 | 24.43 / 25.84 | 36.55 / 38.94 |
| FA2 (6) | 49.15 / 50.95 | 48.47 / 51.84 | 78.50 / 79.07 | 48.13 / 51.70 | 49.15 / 52.37 | 73.91 / 74.33 | 62.44 / 62.95 | 62.81 / 63.38 | 56.85 / 57.08 | 47.12 / 51.18 | 93.52 / 94.82 | 1256.03 / 1253.87 | 260.68 / 283.89 | 67.24 / 67.67 | 54.61 / 55.51 |
| FA3 (4) | 76.65 / 78.44 | 69.38 / 78.95 | 114.18 / 115.00 | 69.12 / 79.46 | 67.07 / 76.42 | – | 80.90 / 84.92 | 80.38 / 84.54 | 76.80 / 81.80 | 64.43 / 76.00 | 130.50 / 135.42 | 1073.65 / 1089.63 | 165.38 / 184.75 | 73.22 / 79.88 | 73.22 / 77.34 |
| FA4 (1) | 471.04 / 519.94 | 532.48 / 602.85 | 741.36 / 775.37 | 538.06 / 614.61 | 471.04 / 520.29 | 653.31 / 703.29 | 540.06 / 581.75 | 541.26 / 582.76 | 509.92 / 549.69 | 497.25 / 564.65 | 1008.64 / 1050.47 | 21994.45 / 22205.91 | 8291.10 / 8329.92 | 520.90 / 558.67 | 487.42 / 530.39 |

#### C2 ratios (baseline / TileLang primary) and verdicts with the graph timers (graph-flush; steady in brackets)

| shape | Triton best: measured / claimed / verdict | Triton as upstream ran it (ws=True) | PyTorch = SDPA flash: measured / claimed / verdict | SDPA cuDNN | SDPA efficient | torch naive | FlashInfer | flash-attn 2.8.3 | old eager-flush Triton / SDPA flash ratios |
|---|---|---|---|---|---|---|---|---|---|
| FA0 | 1.33 [1.33] / 1.60 / partly [partly] | 1.55 | 1.48 [1.54] / 1.76 / partly [partly] | 1.01 [1.02] | 1.77 | 3.86 | 1.20 | 1.47 [1.52] | 1.27 / 1.46 |
| FA1 | 1.13 [1.13] / 1.49 / partly [partly] | unlaunchable | 1.79 [1.83] / 1.65 / reproduced [reproduced] | 1.01 [1.08] | 2.18 | 2.47 | 1.17 | 1.76 [1.80] | 1.09 / 1.73 |
| FA2 | 1.27 [1.24] / 1.28 / reproduced [reproduced] | 1.50 | 1.16 [1.12] / 1.57 / partly [partly] | 0.96 [1.00] | 1.90 | 5.30 | 1.37 | 1.11 [1.09] | 1.26 / 1.20 |
| FA3 | 1.05 [1.08] / 1.34 / partly [partly] | unlaunchable | 1.00 [1.04] / 1.45 / partly [partly] | 0.84 [0.97] | 1.70 | 2.16 | 0.96 | 0.96 [0.99] | 1.05 / 1.00 |
| FA4 | 1.15 [1.12] / 1.34 / partly [partly] | 1.39 | 1.08 [1.06] / 2.17 / partly [partly] | 1.06 [1.09] | 2.14 | 17.60 | 1.11 | 1.03 [1.02] | 1.16 / 1.09 |
| **geomean** | **1.18 [1.17] / 1.41 / partly [partly]** | 1.48 (3 causal shapes) | **1.27 [1.28] / 1.70 / partly [partly]** | 0.97 [1.03] | 1.93 | 4.53 | 1.15 | 1.23 | |
| sensitivity: TileLang = fastest TileLang variant per shape | 1.23 [1.19] / 1.41 / partly | | 1.32 [1.30] / 1.70 / partly | | | | | | |

#### C2 with the eager clean-flush timer (original primary; per-kernel times quantised to 2.048 µs, superseded)

#### C2 times (eager cobench clean-flush median, us)

| shape | TileLang primary | TL bhsd main | TL bshd main | TL bshd tune | TL sweep best | Triton tlbench ws | Triton tlbench | Triton tut06 | SDPA flash | SDPA cuDNN | SDPA efficient | SDPA math | torch naive | FlashInfer | flash-attn 2.8.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FA0 (S=512, causal) | 22.5 | 20.5 | 32.8 | 20.5 | 22.5 | 32.8 | 28.7 | 28.7 | 32.8 | 22.5 | 36.9 | 239.5 | 88.1 | 26.6 | 32.7 |
| FA1 (S=512, full) | 22.5 | 26.7 | 32.8 | 26.6 | 22.5 | – | 24.7 | 24.6 | 39.0 | 22.5 | 47.1 | 202.7 | 57.3 | 27.0 | 38.9 |
| FA2 (S=1024, causal) | 51.2 | 49.2 | 79.9 | 49.2 | 49.2 | 75.8 | 64.5 | 65.5 | 61.3 | 53.2 | 94.3 | 1251.3 | 267.2 | 69.7 | 57.3 |
| FA3 (S=1024, full) | 79.8 | 71.8 | 114.7 | 71.7 | 69.6 | – | 84.0 | 84.0 | 80.1 | 69.7 | 133.1 | 1066.9 | 161.6 | 75.9 | 75.7 |
| FA4 (S=4096, causal) | 458.7 | 518.3 | 729.1 | 523.3 | 458.8 | 641.1 | 530.4 | 532.4 | 498.7 | 483.4 | 993.3 | 21730.7 | 8249.3 | 508.9 | 477.1 |

#### C2 ratios with the eager clean-flush timer (baseline / TileLang primary)

| shape | Triton (best of 3): measured / claimed / verdict | Triton as upstream ran it (ws=True) | PyTorch = SDPA flash (FA2): measured / claimed / verdict | SDPA cuDNN | SDPA efficient | SDPA math | torch naive | FlashInfer | flash-attn 2.8.3 | FA3 |
|---|---|---|---|---|---|---|---|---|---|---|
| FA0 | 1.27 / 1.60 / partly | 1.46 | 1.46 / 1.76 / partly | 1.00 | 1.64 | 10.6 | 3.91 | 1.18 | 1.45 | not testable (claimed 2.19) |
| FA1 | 1.09 / 1.49 / partly | unlaunchable | 1.73 / 1.65 / reproduced | 1.00 | 2.09 | 9.0 | 2.55 | 1.20 | 1.73 | not testable (claimed 2.00) |
| FA2 | 1.26 / 1.28 / reproduced | 1.48 | 1.20 / 1.57 / partly | 1.04 | 1.84 | 24.5 | 5.22 | 1.36 | 1.12 | not testable (claimed 1.11) |
| FA3 | 1.05 / 1.34 / partly | unlaunchable | 1.00 / 1.45 / partly | 0.87 | 1.67 | 13.4 | 2.03 | 0.95 | 0.95 | not testable (claimed 1.06) |
| FA4 | 1.16 / 1.34 / partly | 1.40 | 1.09 / 2.17 / partly | 1.05 | 2.17 | 47.4 | 17.98 | 1.11 | 1.04 | not testable (claimed 0.91) |
| **geomean** | **1.16 / 1.41 / partly** | 1.44 (3 causal shapes) | **1.27 / 1.70 / partly** | 0.99 | 1.87 | 17.2 | 4.52 | 1.15 | 1.23 | not testable (claimed 1.36) |
| sensitivity: TileLang = fastest TileLang variant per shape (geomean) | 1.23 / 1.41 / partly | | 1.34 / 1.70 / partly | 1.05 | | | | | 1.29 | |

#### C2 with the upstream scripts' timers (tilelang do_bench for TileLang, triton do_bench for baselines)

| shape | TileLang us | Triton best us | SDPA flash us | Triton ratio / verdict | SDPA flash ratio / verdict |
|---|---|---|---|---|---|
| FA0 | 22.9 | 33.4 | 62.0 | 1.46 / reproduced | 2.71 / reproduced |
| FA1 | 23.9 | 27.0 | 60.5 | 1.13 / partly | 2.53 / reproduced |
| FA2 | 53.2 | 75.7 | 70.2 | 1.42 / reproduced | 1.32 / partly |
| FA3 | 82.2 | 88.6 | 91.0 | 1.08 / partly | 1.11 / partly |
| FA4 | 475.9 | 542.0 | 514.5 | 1.14 / partly | 1.08 / partly |
| **geomean** | | | | 1.23 / partly | 1.61 / reproduced |

#### C2 SM clock (ClockProbe median per variant, MHz), board power (NVML, mean / max W), CV of the primary (eager clean-flush run)

| shape | TileLang | Triton best | SDPA flash | SDPA cuDNN | power mean / max | TileLang CV | kcycles TL / Triton best / SDPA flash |
|---|---|---|---|---|---|---|---|
| FA0 | 2848 | 2848 | 2804 | 2844 | 368 / 372 | 0.013 | 64.1 / 81.7 / 91.9 |
| FA1 | 2829 | 2834 | 2776 | 2795 | 387 / 391 | 0.011 | 63.7 / 69.8 / 108.2 |
| FA2 | 2759 | 2772 | 2777 | 2694 | 437 / 445 | 0.013 | 141.2 / 178.6 / 169.7 |
| FA3 | 2716 | 2737 | 2723 | 2551 | 457 / 462 | 0.012 | 215.8 / 229.9 / 220.0 |
| FA4 | 2476 | 2674 | 2617 | 2317 | 530 / 578 | 0.003 | 1135.5 / 1418.4 / 1305.3 |

TileLang primary at FA4: 137.4 GFLOP in 458.7 µs = 300 TFLOPS, 63 % of the MMA peak at the measured 2476 MHz
(188 SMs x 1024 FLOP/clk). Board power 368–530 W mean (578 W max at FA4, cap 600 W); NVML reported no throttle
reason in any C2 run. The interleaving means every implementation of a shape sees the same power/thermal state.

## 6. C4 results (MLA decode) and the optional sink claim

All times in µs (cobench clean-flush median, 100 interleaved reps; CV <= 1.1 %). Torch only with its own timer.
The 600 W cap (NVML `SwPowerCap`) was hit only at batch 128 / 32K (17 % of samples) and in 5 of the 8 sink runs.
TileLang = the adapted config (block_H 16, 1 stage, §3). The batch 128 / 32K Triton cell is empty: the kernel faults
(§7.4).

#### C4 times (cobench clean-flush median, us; TFLOPS with benchmark_mla.py's FLOP count)

| batch | KV ctx | TileLang adapted | TL split 2 | TL split 4 | Triton | FlashInfer fa2 | FlashInfer XQA (bf16) | Torch (upstream timer) |
|---|---|---|---|---|---|---|---|---|
| 64 | 1024 | 174.1 (111) | 194.5 (100) | 192.5 (101) | 1074.5 (18) | 174.1 (111) | 615.8 (31) | 55419 |
| 64 | 2048 | 317.5 (119) | 347.6 (108) | 339.8 (111) | 1713.2 (22) | 313.2 (120) | 1093.6 (34) | 102520 |
| 64 | 4096 | 602.1 (123) | 656.3 (113) | 629.7 (118) | 2505.7 (30) | 590.3 (126) | 2016.2 (37) | 192922 |
| 64 | 8192 | 1155.3 (127) | 1256.5 (117) | 1191.0 (124) | 4081.7 (36) | 1128.4 (130) | 3811.5 (39) | 372497 |
| 64 | 16384 | 2287.1 (128) | 2472.9 (119) | 2330.3 (126) | 7237.7 (41) | 2237.4 (131) | 7404.9 (40) | 821602 |
| 64 | 32768 | 4562.0 (128) | 4911.6 (119) | 4612.0 (127) | 13556.3 (43) | 4468.6 (131) | 14597.3 (40) | 2325576 |
| 128 | 1024 | 352.3 (116) | 376.6 (109) | 403.4 (102) | 2196.0 (19) | 342.0 (120) | 992.7 (41) | 116224 |
| 128 | 2048 | 633.8 (122) | 665.6 (117) | 696.2 (111) | 3484.8 (22) | 614.4 (126) | 1720.0 (45) | 210827 |
| 128 | 4096 | 1198.3 (126) | 1239.1 (122) | 1271.7 (118) | 5055.6 (30) | 1174.4 (128) | 3151.8 (48) | 390769 |
| 128 | 8192 | 2327.4 (127) | 2378.8 (125) | 2411.5 (123) | 8261.7 (36) | 2290.7 (129) | 5953.8 (50) | 749380 |
| 128 | 16384 | 4597.7 (128) | 4661.2 (126) | 4694.0 (125) | 14699.5 (40) | 4531.2 (130) | 11533.4 (51) | 1660633 |
| 128 | 32768 | 9257.6 (127) | 9301.1 (126) | 9346.8 (125) | – | 9160.0 (128) | 22666.2 (52) | 4664429 |

#### C4 ratios (baseline / TileLang adapted) and verdicts

| batch | KV ctx | FlashInfer (fa2): measured / claimed / verdict | FlashInfer XQA bf16 | Triton: measured / claimed / verdict | Torch (upstream timers) | FlashMLA |
|---|---|---|---|---|---|---|
| 64 | 1024 | 1.00 / 1.16 / not reproduced | 3.54 | 6.17 / 11.8 / partly | 305 | not testable (claimed TL/FlashMLA 1.06) |
| 64 | 2048 | 0.99 / 1.26 / not reproduced | 3.44 | 5.40 / 7.2 / partly | 314 | not testable (claimed TL/FlashMLA 1.03) |
| 64 | 4096 | 0.98 / 1.20 / not reproduced | 3.35 | 4.16 / 5.0 / partly | 316 | not testable (claimed TL/FlashMLA 0.92) |
| 64 | 8192 | 0.98 / 1.21 / not reproduced | 3.30 | 3.53 / 4.0 / partly | 321 | not testable (claimed TL/FlashMLA 0.93) |
| 64 | 16384 | 0.98 / 1.35 / not reproduced | 3.24 | 3.16 / 3.5 / partly | 358 | not testable (claimed TL/FlashMLA 0.98) |
| 64 | 32768 | 0.98 / 1.32 / not reproduced | 3.20 | 2.97 / 3.1 / reproduced | 506 | not testable (claimed TL/FlashMLA 0.97) |
| 128 | 1024 | 0.97 / 1.11 / not reproduced | 2.82 | 6.23 / 12.7 / partly | 317 | not testable (claimed TL/FlashMLA 0.97) |
| 128 | 2048 | 0.97 / 1.18 / not reproduced | 2.71 | 5.50 / 7.3 / partly | 327 | not testable (claimed TL/FlashMLA 0.94) |
| 128 | 4096 | 0.98 / 1.20 / not reproduced | 2.63 | 4.22 / 5.2 / partly | 323 | not testable (claimed TL/FlashMLA 0.93) |
| 128 | 8192 | 0.98 / 1.27 / not reproduced | 2.56 | 3.55 / 4.1 / partly | 320 | not testable (claimed TL/FlashMLA 0.95) |
| 128 | 16384 | 0.99 / 1.31 / not reproduced | 2.51 | 3.20 / 3.5 / reproduced | 359 | not testable (claimed TL/FlashMLA 0.98) |
| 128 | 32768 | 0.99 / 1.22 / not reproduced | 2.45 | faults (int32 offset overflow in the kernel) / 3.0 / not testable | 504 | not testable (claimed TL/FlashMLA 0.94) |
| **geomean** | | **0.98 / 1.23 / not reproduced** | 2.95 | **4.22 / 5.47 / partly** | 350 (paper 1075.9) | not testable |

#### C4 spot check with the CUDA-graph timers (per-call us; ratio = baseline / TileLang)

| shape | timer | TileLang | FlashInfer fa2 | Triton | FlashInfer XQA bf16 | FI / TL | Triton / TL |
|---|---|---|---|---|---|---|---|
| mla_b64_s1024 | eager flush | 174.14 | 174.08 | 1074.48 | 615.84 | 1.000 | 6.17 |
| mla_b64_s1024 | graph-flush | 173.56 | 176.12 | 1081.80 | 617.98 | 1.015 | 6.23 |
| mla_b64_s1024 | steady | 178.10 | 183.52 | 1090.49 | 622.59 | 1.030 | 6.12 |

At batch 64 / 1K (the smallest C4 row, 174 µs) the three timers agree to 1–3 %: FlashInfer / TileLang = 1.00
(eager), 1.01 (graph-flush), 1.03 (steady) vs the claimed 1.16; Triton / TileLang = 6.1–6.2 with all three. The
eager numbers are kept for C4; the verdict does not depend on the timer (by the letter of the rule a row at 1.01 is
"partly" and one at 1.00 "not reproduced"; both are parity within the timers' disagreement).

#### C4 with the upstream timers (benchmark_mla.py: tilelang do_bench for TileLang, triton do_bench otherwise)

| batch | KV ctx | TileLang us | FlashInfer fa2 us | Triton us | FlashInfer ratio / verdict | Triton ratio / verdict |
|---|---|---|---|---|---|---|
| 64 | 1024 | 181.7 | 186.3 | 1103.4 | 1.03 / partly | 6.07 / partly |
| 64 | 2048 | 326.5 | 331.5 | 1733.3 | 1.02 / partly | 5.31 / partly |
| 64 | 4096 | 610.2 | 605.0 | 2534.3 | 0.99 / not reproduced | 4.15 / partly |
| 64 | 8192 | 1160.4 | 1143.9 | 4114.4 | 0.99 / not reproduced | 3.55 / partly |
| 64 | 16384 | 2297.4 | 2275.1 | 7378.3 | 0.99 / not reproduced | 3.21 / reproduced |
| 64 | 32768 | 4592.8 | 4506.1 | 13650.2 | 0.98 / not reproduced | 2.97 / reproduced |
| 128 | 1024 | 366.9 | 356.8 | 2222.3 | 0.97 / not reproduced | 6.06 / partly |
| 128 | 2048 | 644.6 | 630.4 | 3507.8 | 0.98 / not reproduced | 5.44 / partly |
| 128 | 4096 | 1209.2 | 1202.8 | 5178.5 | 0.99 / not reproduced | 4.28 / partly |
| 128 | 8192 | 2344.9 | 2320.5 | 8395.3 | 0.99 / not reproduced | 3.58 / partly |
| 128 | 16384 | 4631.7 | 4559.9 | 14854.7 | 0.98 / not reproduced | 3.21 / reproduced |
| 128 | 32768 | 9253.6 | 9064.5 | – | 0.98 / not reproduced | – / n/a |
| **geomean** | | | | | 0.99 / not reproduced | 4.21 / partly |

#### C4 SM clock (ClockProbe median, MHz), power (NVML mean / max W), DRAM throughput of TileLang

| batch | KV ctx | TileLang MHz | FlashInfer MHz | Triton MHz | power mean / max | TileLang GB/s | FlashInfer GB/s | TileLang CV |
|---|---|---|---|---|---|---|---|---|
| 64 | 1024 | 2786 | 2731 | 2829 | 444 / 447 | 563 | 563 | 0.008 |
| 64 | 2048 | 2784 | 2711 | 2830 | 457 / 462 | 546 | 554 | 0.005 |
| 64 | 4096 | 2793 | 2698 | 2836 | 483 / 499 | 539 | 550 | 0.004 |
| 64 | 8192 | 2835 | 2729 | 2856 | 481 / 507 | 542 | 555 | 0.004 |
| 64 | 16384 | 2826 | 2698 | 2849 | 501 / 537 | 538 | 550 | 0.003 |
| 64 | 32768 | 2811 | 2676 | 2843 | 521 / 620 | 534 | 546 | 0.003 |
| 128 | 1024 | 2783 | 2698 | 2831 | 487 / 492 | 583 | 600 | 0.005 |
| 128 | 2048 | 2796 | 2697 | 2836 | 502 / 514 | 562 | 580 | 0.005 |
| 128 | 4096 | 2802 | 2685 | 2839 | 529 / 557 | 549 | 561 | 0.004 |
| 128 | 8192 | 2801 | 2674 | 2841 | 544 / 600 | 542 | 551 | 0.004 |
| 128 | 16384 | 2793 | 2660 | 2827 | 558 / 633 | 537 | 545 | 0.005 |
| 128 | 32768 | 2752 | 2571 | – | 600 / 658 | 528 | 533 | 0.005 |

#### Attention sink (GQA fwd, bf16; cobench median, us)

| seq | dim | TileLang fixed cfg us | TileLang autotuned us (TFLOPS) | Triton us (TFLOPS) | speedup measured (tuned / fixed) | claimed (H800) | verdict (tuned) | upstream-timer speedup (tuned) | tuned config (block_M, block_N, stages, threads) |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 64 | 138.7 | 135.2 (254) | 149.5 (230) | 1.11 / 1.08 | 1.21 | reproduced | 1.12 | 128, 128, 1, 256 |
| 4096 | 64 | 452.6 | 448.4 (307) | 509.9 (270) | 1.14 / 1.13 | 1.25 | reproduced | 1.14 | 128, 128, 1, 256 |
| 8192 | 64 | 1750.0 | 1738.8 (316) | 2008.0 (274) | 1.15 / 1.15 | 1.29 | partly | 1.17 | 128, 128, 1, 256 |
| 16384 | 64 | 6976.5 | 6937.6 (317) | 8097.6 (272) | 1.17 / 1.16 | 1.29 | reproduced | 1.18 | 128, 128, 1, 256 |
| 2048 | 128 | unlaunchable (162 KB) | 239.6 (287) | 407.6 (169) | 1.70 / – | 1.30 | reproduced | 1.68 | 128, 128, 1, 256 |
| 4096 | 128 | unlaunchable (162 KB) | 806.9 (341) | 1438.7 (191) | 1.78 / – | 1.35 | reproduced | 1.72 | 128, 128, 1, 256 |
| 8192 | 128 | unlaunchable (162 KB) | 3191.9 (344) | 5495.7 (200) | 1.72 / – | 1.27 | reproduced | 1.71 | 128, 128, 1, 256 |
| 16384 | 128 | unlaunchable (162 KB) | 12656.3 (347) | 21697.4 (203) | 1.71 / – | 1.31 | reproduced | 1.70 | 128, 128, 1, 256 |
| **geomean** | | | | | 1.40 | 1.28 | reproduced | | |

## 7. Explanations (checked) for everything not reproduced

### 7.1 C2 vs Triton: partly (geomean 1.18 vs 1.41; numbers below from the graph-flush timer unless stated)
- **Which Triton.** Against the Triton setup the upstream benchmark used (`warp_specialize=True`) TileLang is
  1.55 / 1.50 / 1.39x faster on the three causal shapes (FA0 / FA2 / FA4), i.e. the claim (1.60 / 1.29 / 1.34) holds
  for that setup (0.97 / 1.17 / 1.04 of it). But on sm_120 that setup is not Triton's best: the same kernel with
  ws=False is 15–17 % faster (FA0 26.2 vs 30.7 µs, FA2 62.4 vs 73.9, FA4 540 vs 653), and for the non-causal shapes no ws=True config fits 99 KB of shared
  memory, so it cannot launch. Triton 3.4's tutorial itself enables warp specialisation only on Blackwell
  datacenter parts (`is_blackwell()` = cc 10). The verdict uses Triton's fastest setup here.
- **Not TileLang's config.** TileLang's primary config lost one pipeline stage to the 99 KB limit. Replacing it by
  the fastest TileLang variant per shape (shipped configs + an 18-config sm_120 sweep) moves the geomean only from
  1.18 to 1.23 (steady: 1.17 to 1.19; still partly). The sweep's best is within 1 % of the primary on FA0/FA1/FA2/FA4
  and 14 % faster on FA3.
- **Mechanism.** Both compilers emit `mma.sync` on sm_120: TileLang's sm_120 kernels contain `mma`, no `wgmma`
  (`crosslower.json`: the identical program lowered for sm_90a uses `wgmma`), and Triton 3.4's PTX for these kernels
  targets `sm_120a` with `mma.sync` only (0 `wgmma`, checked in `~/.triton/cache`). On H100, TileLang's schedule (TMA producer warp group + async wgmma overlapping the softmax, "pipeline
  scheduling schemes as complex as those used in FlashAttention-3", paper §5.2) is what separated it from Triton;
  with synchronous `mma.sync` that overlap is not available, so the gap narrows. Clock is not the cause (eager run's
  ClockProbe, §5): at FA0–FA3
  TileLang and Triton run at the same SM clock (within 21 MHz); at FA4 TileLang draws more power and runs 200 MHz *lower*
  (2476 vs 2674 MHz), so per cycle its advantage is larger (1135 vs 1418 kcycles = 1.25x) than per µs (1.16x).

### 7.2 C2 vs PyTorch (FA2): partly (geomean 1.27 vs 1.70)
- The paper attributes PyTorch's deficit to FlashAttention-2 being hand-optimised for pre-Hopper GPUs
  ("PyTorch uses a hand-optimized FlashAttention-2 kernel, which results in lower performance compared to
  FlashAttention-3"). FA2's kernels are `mma.sync` / `cp.async` designs; on H100 they leave wgmma and TMA unused.
  sm_120 has no wgmma, so FA2's design is the native one here and the gap is what remains between two mma.sync
  kernels: 1.00–1.79x, largest at the tiny FA0/FA1 shapes and 1.08x at the largest shape (FA4), where the figure
  claims its biggest gap (2.17x).
- Cross-checks on the same GPU: NVIDIA's own cuDNN SDPA kernel is at parity with TileLang (0.84–1.06, geomean 0.97;
  steady 0.97–1.09, geomean 1.03), and the separately built flash-attn 2.8.3 (with sm_120 SASS) is at 0.96–1.76x
  (geomean 1.23; it beats TileLang's primary only at FA3), i.e. TileLang's sm_120 kernel is competitive with the best
  available; the missing margin is the Hopper-specific part.
- Timer sensitivity: with the upstream scripts' own timers (host launch overhead included) the SDPA geomean becomes
  1.61 ("reproduced" by the rule), entirely because SDPA's Python dispatch adds ~20–30 µs to the 20–40 µs FA0/FA1
  kernels (SDPA flash FA0: 29.3 µs per call in a graph, 32.8 µs eager clean-flush, 62.0 µs with host overhead).
  That is a timer artefact, so the verdict stays with the graph-flush timer (1.27; steady 1.28; eager clean-flush,
  quantised, 1.27).

### 7.3 C4 vs FlashInfer: not reproduced (parity, 0.97–1.00)
- **TileLang runs a degraded config, forced by the hardware.** The shipped MLA kernel (block_H 64, 2-stage KV
  pipeline) needs 226 KB of shared memory; the identical program lowered for sm_90a needs the same 231,424 B, i.e.
  it was sized for H100's 227 KB. The only config that fits 99 KB is block_H 16 with a single KV buffer (96 KB,
  1 CTA/SM): no double buffering, and each request's KV is streamed by 8 CTAs instead of 2 (every head group re-reads it).
  Measured: TileLang reaches 111–128 TFLOPS and 0.53–0.58 TB/s, i.e. ~21–24 % of this GPU's MMA peak at 2.8 GHz and
  ~30 % of its 1.79 TB/s — neither bound, latency-limited. On H100 the same example reached 325–479 TFLOPS (figure).
  Splitting the KV (num_split 2/4) does not help (0.5–15 % slower).
- **Different baseline kernel.** The H100 figure compared against FlashInfer's `fa3` (Hopper) MLA backend
  (`benchmark_mla.py` hard-codes `backend="fa3"`); on sm_120 that backend has no kernel image (verified), so the
  comparison is against FlashInfer's `fa2` backend, which lands at the same throughput as TileLang on every shape.
- **Not a fork effect.** The fork's compiler changes (13 files vs upstream 7e3bbf37) are: cp.async L2 eviction
  hints (only with an explicit `eviction_policy`, unused by these examples), a shared-memory-lifetime scope (only
  emitted by `cotile`), a pipeline-epilogue extent simplification (cp.async pipelines with dynamic trip counts),
  and cache keys. All C2/C4 TileLang kernels here are TMA + mbarrier warp-specialised with no cp.async
  (checked in the generated source: `tma_load`, `mbarrier`, producer `warpgroup_reg_dealloc<24>` /
  consumer `<240>`, no `cp.async`, no cache hints), so none of the fork changes apply. The smem numbers are the
  programs' own (identical for sm_90a), and the block_N < 64 failure is in TileLang's layout inference for the mma
  path, which the fork does not touch.

### 7.4 C4 vs Triton: partly (geomean 4.22 vs 5.47); vs Torch: partly
- The ratio falls with context length as in the figure (here 6.2 → 3.0, figure 11.8–12.7 → 3.0–3.1); the measured
  ratios are 0.49–0.95x of the claimed ones, lowest at short contexts. Both kernels lost pipelining to the 99 KB limit
  (Triton: default num_stages 3 needs 102,656 B, 2 stages 102,400 B, so it runs with 1 stage), but TileLang lost
  more: it dropped from 325–479 TFLOPS on H100 to 111–128 here (2.9–3.7x), Triton from 27–151 to 18–43 (1.5–3.5x).
  The H100 advantage came largely from wgmma and the 226 KB two-stage pipeline, which do not exist here.
- **Baseline defect:** at batch 128 / 32K the Triton kernel's element offsets `kv_loc * stride` (int32) exceed
  2^31 (the nope cache has 2,164,260,864 elements) and the kernel faults with an illegal address (verified in an
  isolated child process; batch 128 / 16K, 1.09e9 elements, is fine). The H100 figure shows a Triton value at this
  point; it was produced by a kernel whose addressing overflows at this shape, and `benchmark_mla.py --all` reports
  throughput without checking the output, so that point should not be relied on.
- Torch: `run_torch_mla` loops over requests in Python with fp32 matmuls on 128-fold repeated K/V; measured 305–506x
  slower than TileLang with the upstream timer vs 1075.9x in the paper. The ratio mostly measures that loop (CPU
  launch overhead, fp32 throughput, memory traffic of the repeats), so the difference in magnitude says little.
- FlashMLA and FA3: not testable (Hopper-only); nothing about "TileLang ≈ FlashMLA" can be checked on this GPU.

## 8. Optional: attention sink

Verdict: head_dim 64 **reproduced** (1.11–1.17x vs claimed 1.21–1.29x, i.e. 0.89–0.92 of the claim; 3 of 4 shapes
pass the 0.9 threshold); head_dim 128 **reproduced**, with 1.70–1.78x vs claimed 1.27–1.35x, but that excess comes
from the baseline: on sm_120 the benchmark's Triton kernel (BLOCK 64x64, 4 warps, 3 stages) uses 255 registers
with 14 spills at head_dim 128 (210 registers, 2 spills at 64; Triton kernel metadata), and runs at 169–203
TFLOPS vs 230–274 at head_dim 64. As shipped the benchmark does not run at all (missing module, §3), and its fixed
TileLang config (128x128, 2 stages, 256 threads) needs 162 KB at head_dim 128 (fails at launch); the example's
own autotuner picks 128x128 / 1 stage / 256 threads everywhere. Tables in §6.

## 9. Reproduce

```bash
cd /home/ywc/co-tilelang && source research/env.sh
LOCK=/tmp/claude-1008/-home-ywc-co-tilelang/d36518d2-9f4d-4232-868c-46d863e88350/scratchpad/gpu.lock
S=research/bench/scripts
python $S/claims_attn_digitize.py                      # claimed ratios from the figures (CPU)
python $S/claims_attn_crosslower.py                    # sm_90a vs sm_120a lowering (CPU)
for s in FA0 FA1 FA2 FA3 FA4; do
  python $S/claims_attn_fa.py --shape $s --phase compile                      # TileLang configs + sweep (CPU)
  flock $LOCK python $S/claims_attn_fa.py --shape $s --phase run --reps 200
done
for b in 64 128; do for L in 1024 2048 4096 8192 16384 32768; do
  python $S/claims_attn_mla.py --phase compile --batch $b --seqlen $L          # CPU
  flock $LOCK python $S/claims_attn_mla.py --phase run --batch $b --seqlen $L
done; done
for d in 64 128; do for L in 2048 4096 8192 16384; do
  flock $LOCK python $S/claims_attn_sink.py --seq $L --dim $d
done; done
flock $LOCK python $S/claims_attn_timerprobe.py         # event / stream-completion quantum (raw/timerprobe.json)
for s in FA0 FA1 FA2 FA3 FA4; do                           # CUDA-graph timers for C2 (after the eager runs above)
  flock $LOCK python $S/claims_attn_graph.py --kind fa --shape $s
done
flock $LOCK python $S/claims_attn_graph.py --kind mla --batch 64 --seqlen 1024   # C4 spot check
python $S/claims_attn_report.py median_us    # tables of §5/§6 + summary_median_us.json (mean_us: cross-check)
```
Run times: ~2–4 min per FA shape (incl. the 36-config sweep), 1–5 min per MLA shape (the Torch reference
dominates at 32K: 4.7 s per call), ~2–3 min per sink shape (the example's autotuner runs inside).

## 10. Problems found, open issues

- **Timer quantum (found in acceptance review, fixed):** eagerly launched kernels complete on a ~2.048 µs grid on this
  GPU/driver (§4), so single-kernel CUDA-event times of short kernels are quantised; the C2 numbers of record now come
  from CUDA-graph timers. Effect on C2 ratios: FA0 Triton 1.27 → 1.33, SDPA flash 1.46 → 1.48; FA1 Triton 1.09 →
  1.13, SDPA flash 1.73 → 1.79; geomeans 1.16 → 1.18 (Triton) and 1.27 → 1.27 (SDPA flash); no verdict changed.
  This applies to every cobench flush-mode number of a short kernel in this project (e.g. other sub-studies' small
  shapes): it is a property of stream execution, and N eager launches per rep do not amortise it — only graphs do.
- **TileLang on sm_120 (upstream behaviour, not the fork):** (a) shared-memory budgets of the shipped configs are
  unchecked at compile time; FA and MLA configs over 99 KB fail only at launch. (b) `example_mla_decode_paged.py`
  hard-codes `num_stages=2` and cannot be made to fit 99 KB through its parameters. (c) MLA with block_N 32/16
  fails layout inference on the mma.sync path (`Layout infer conflict between scores_max_prev and scores_scale`)
  while the same program compiles for sm_90a; block_N 32/16 with block_H 16 has no valid 8-warp `FullCol` partition.
  (d) block_M 64 with 256 threads cannot be partitioned (FA sweep) — expected.
- **Upstream benchmark defects:** `benchmark_mla.py`'s Triton kernel overflows int32 at batch 128 / 32K (§7.4);
  `benchmark_mla.py`'s FlashInfer path builds `kv_indptr` with 2b entries (a second loop appends b−1 more; FlashInfer
  only reads the first b+1, so it is harmless) — not used here, the wrapper builds CSR metadata itself;
  `benchmark_gqa_sink_fwd.py` imports a deleted module (§3). The vendored tilelang-benchmark Triton copy's autotune
  key omits the causal flag (FA0/FA1 would share configs in one process).
- **Cross-library CUDA error leakage** (§4): a failed launch in one library (FlashInfer fa3) was reported by the next
  TileLang launch in the same process. Handled by isolating expected failures in child processes and resetting
  the runtime's last error of every loaded libcudart (two are loaded: torch's 12.8 and the system 12.9).
- **FlashInfer XQA MLA (bf16)**, FlashInfer's dedicated SM120 decode kernel, is 2.5–3.5x *slower* than both
  TileLang and FlashInfer fa2 here. Not investigated (it is an extra reference in a different dtype; its fp8 path is
  presumably the tuned one). The cuTile backend could not be tried (needs CUDA 13.1+ `tileiras`).
- **Not done:** FlashMLA and FA3 (Hopper-only, no build attempted — their build scripts target only sm_90a/100a/103a,
  cited in §3). The upstream shapes with batch 64 / 64 heads / seq 8K from the tilelang-benchmark scripts were not
  run (they are not FA0–FA4). The AMD TileLang MLA kernel (`examples/deepseek_mla/amd/`, sized for 64 KB LDS) might
  fit sm_120 better than the Hopper example; not tried (different, non-paged workload).
- Figure provenance (§2): the FA figure predates the scripts; the MLA figures predate `benchmark_mla.py`; no data
  files exist; ratios were read off the images (the FA reading reproduces the paper's geomeans to 0.01).

## 11. Files

- Scripts: `research/bench/scripts/claims_attn_common.py` (helpers: GPU wait, correctness, TileLang resources, timers,
  child-process isolation, CUDA error reset), `claims_attn_fa.py` (C2), `claims_attn_mla.py` (C4),
  `claims_attn_sink.py` (optional), `claims_attn_digitize.py` (figure reading), `claims_attn_crosslower.py`
  (sm_90a vs sm_120a lowering), `claims_attn_timerprobe.py` (timer quantum), `claims_attn_graph.py` (CUDA-graph
  timers), `claims_attn_report.py` (tables, verdicts).
- Vendored baselines: `research/bench/baselines/upstream/` (`__init__.py` documents provenance and sha256).
- Data: `raw/graph_FA{0..4}.json` and `raw/graph_mla_b64_s1024.json` (CUDA-graph timers: per-variant graph-flush
  summary with raw samples, steady-state summary, checks, guard), `raw/timerprobe.json`, `raw/fa_FA{0..4}.json`, `raw/mla_b{64,128}_s{1024..32768}.json`, `raw/sink_s*_d*.json` (per variant: config,
  TileLang resources, correctness, cobench summary incl. per-rep clock, raw samples, NVML, guard record, upstream
  timers; Triton autotune choices; isolated-failure records), `raw/*_compile.json` (TileLang compile-phase resources
  incl. every sweep config), `claims_digitized.json`, `crosslower.json`, `summary_{median,mean}_us.json`.
