# FlashInfer baselines on sm_120 (P1-baselines)

- Date: 2026-09-22. Machine: RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs, 99 KB smem/CTA, 600 W cap), driver 580.173.02.
- Software: flashinfer-python 0.7.0 (JIT, `compute_120f`, nvcc 12.9), torch 2.8.0+cu128, TileLang `0.1.14+cuda.gitc5f6c782`. Install details: `research/env_versions.md` §5.
- Wrapper used by every script here: `research/bench/baselines/flashinfer_ops.py` (see §6).
- Timing: cobench (`research/bench/README.md`) with `clock=True`. Every timed batch ran under `gpu_guard.py`: no foreign GPU process before it, no foreign process seen during it (NVML polled every 0.25 s), none after it. No timed batch saw a foreign process, so none had to be re-measured for contamination (per-batch guard records are in the JSON files). A point whose CV stayed above 2% after 3 attempts is flagged ⚠.

| file | content |
|---|---|
| `check_correctness.py` → `correctness.json` | criteria 2–4: correctness at the full shapes |
| `run_timing.py` → `timing.json` | criteria 2–4: decode, prefill, and POD against serial, two streams and green contexts |
| `tilelang_gqa_decode.py` → `tilelang_decode.json` | criterion 2: TileLang `examples/flash_decoding/example_gqa_decode.py`, unmodified, over 12 configs that fit in 99 KB |
| `pod_plan_info.py` → `pod_plan_info.json` | POD work decomposition: CTA counts and ticket ratio |
| `make_tables.py` | prints every table below from the JSON files |
| `gpu_guard.py` | wait-for-free-GPU and in-run watchdog |

To reproduce, run `source research/env.sh`, then run the four scripts in the order above; `run_timing.py` resumes from `timing.json`. The first FlashInfer JIT build takes about 2.5 min, of which POD alone is about 107 s. After that, each script takes 1–12 min.

## Summary

| criterion | result | key numbers |
|---|---|---|
| 1. install | **pass** | 15 new packages; no existing package changed version (torch 2.8.0+cu128 and tvm-ffi 0.1.12 kept). About 0.93 GB in site-packages. JIT cache `~/.cache/flashinfer/0.7.0/120f/` is 51 MB for the 4 modules. |
| 2. GQA decode | **pass** | max abs err 2.8e-4–8.4e-4 against fp32, at most 2.1× the bf16 rounding floor. Flush mode: 1370–1584 GB/s (the smallest shape pays the ~3 µs flush floor). Graph mode: 1501–1636 GB/s, i.e. 84–91% of the 1792 GB/s DRAM peak. The TileLang example with a ≤99 KB config is at parity: within ±1% at 3 shapes, and within 5% at B16/KV2048 (faster in flush mode, 3% slower in graph mode). |
| 3. prefill (causal) | **pass** | max abs err 7.7e-3–8.5e-3 against fp32 SDPA; bf16 SDPA's own error is 7.8e-3–1.1e-2. S=8192: 1.66 ms, 331 TFLOP/s, 70% of the mma.sync peak at the measured clock. S=2048: 164 µs, 210 TFLOP/s, 46%. |
| 4. POD | **pass, with a FlashInfer bug** | Builds for sm_120 and matches the standalone kernels bit for bit. POD is 1.02–1.38× faster than serial. Against the best of streams and an oracle green split it ranges from 0.95× to 1.15×. It is only valid on the legacy default stream and without CUDA graphs (§4.3). |
| 5. outputs | done | this directory and `research/bench/baselines/flashinfer_ops.py` |

## 1. Install (criterion 1)

- I ran `pip install --dry-run` first, then `pip install --no-cache-dir flashinfer-python==0.7.0`. Comparing `pip list` before and after shows only additions.
- FlashInfer 0.7.0 accepts the tvm-ffi 0.1.12 that TileLang already uses. `import tilelang, flashinfer` works in a single process.
- Side effect: `nvidia-cutlass-dsl` installs a `.pth` hook, which makes `import cutlass` (CuTe DSL) succeed. TileLang uses it only when the `cutedsl` backend is selected explicitly.
- FlashInfer launches on **torch's current stream**, because tvm-ffi reads it through the DLPack exchange API. As a result, `torch.cuda.stream(green_stream)` works for FlashInfer. The green-partition timings in §4.5 confirm this: the prefill slows down in proportion to the partition size.

## 2. GQA decode (criterion 2)

**Configuration**
- API: `BatchDecodeWithPagedKVCacheWrapper`, bf16, H_q=32, H_kv=8, D=128.
- Paged NHD cache with page size 16. K and V are separate tensors of shape `[num_pages, 16, 8, 128]`.
- Request b owns pages `[b·P, (b+1)·P)` through an identity page table. The cache is therefore byte-identical to a dense `[B, S, 8, 128]` tensor, which is the TileLang example's layout.
- Two kernel paths:
  - `use_tensor_cores=False`: the CUDA-core `decode.cuh` kernel, 64 regs/thread.
  - `use_tensor_cores=True`: the FA2 batch-prefill kernel with a 16-row Q tile, 134 regs/thread. This is the same code as POD's decode half.
- The plan is computed once per shape. B=16 uses a 2-way KV split (padded batch 32); B=64 uses no split.

**Correctness** against the fp32 torch GQA reference: all pass (|err| ≤ 1e-2 + 1e-2·|ref| per element).

| B×KV | max abs err, CUDA-core | max abs err, tensor-core | bf16 rounding floor | max abs ref |
|---|---|---|---|---|
| 16×2048 | 6.8e-4 | 6.3e-4 | 4.8e-4 | 0.165 |
| 16×8192 | 4.9e-4 | 3.7e-4 | 2.4e-4 | 0.087 |
| 64×2048 | 8.4e-4 | 5.4e-4 | 4.8e-4 | 0.199 |
| 64×8192 | 4.2e-4 | 2.8e-4 | 2.4e-4 | 0.091 |

**Timing.** Bytes = K+V+Q+O. MHz is the ClockProbe per-rep median. Graph mode rotates input copies past 2×L2.

| B | KV | path | flush µs | flush GB/s | MHz | graph µs | graph GB/s | MHz | CV flush/graph |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 2048 | CUDA-core | 98.0 | 1372 | 2868 | 87.7 | 1533 | 2856 | 0.70% / 0.09% |
| 16 | 2048 | tensor-core | 98.1 | 1370 | 2870 | 89.6 | 1501 | 2861 | 1.01% / 0.10% |
| 16 | 8192 | CUDA-core | 350.2 | 1534 | 2845 | 333.0 | 1613 | 2859 | 0.59% / 0.05% |
| 16 | 8192 | tensor-core | 351.1 | 1530 | 2864 | 333.7 | 1610 | 2864 | 0.52% / 0.05% |
| 64 | 2048 | CUDA-core | 352.3 | 1527 | 2848 | 334.6 | 1608 | 2851 | 0.36% / 0.05% |
| 64 | 2048 | tensor-core | 349.8 | 1538 | 2865 | 334.8 | 1607 | 2866 | 0.44% / 0.07% |
| 64 | 8192 | CUDA-core | 1356.5 | 1584 | 2847 | 1313.6 | 1636 | 2849 | 0.18% / 0.02% |
| 64 | 8192 | tensor-core | 1360.9 | 1579 | 2867 | 1316.9 | 1631 | 2867 | 0.27% / 0.05% |

- The two paths are within 1% of each other.
- The decode reads 1.6 TB/s in graph mode. That is more than cobench's 16 MB–512 MB copy measurement (1460–1490 GB/s), which spends half its traffic on writes.
- Graph mode sits 3–11% below flush mode in time. It hides the ~3 µs flush-mode launch/event floor, and FlashInfer enables PDL on sm_120, so consecutive calls in a graph can overlap their tails.

**TileLang comparison.** The TileLang example is `flashattn()`, unmodified: fp16 (hard-coded), a `uint8` mask (all ones for timing), dense K/V. GB/s counts K+V+Q+O. The mask adds another 1/512.

The table shows the two smoke-test configs from `run_gqa_decode.py` and the best of 12 configs. The 12 configs are block_N∈{64,128}, stages∈{1,2}, num_split∈{1,2,4,8}, keeping those ≤99 KB.

| B | KV | TileLang config (block_N/stages/num_split) | flush µs | GB/s | graph µs | graph GB/s | FlashInfer best flush µs | FI µs / TL µs (flush; >1 = TileLang faster) |
|---|---|---|---|---|---|---|---|---|
| 16 | 2048 | 64/2/8 (smoke cfg) | 108.5 | 1240 | 90.4 | 1488 | 98.0 | 0.90 |
| 16 | 2048 | 128/1/1 (best of 12) | 93.8 | 1434 | 92.8 | 1448 | 98.0 | 1.05 |
| 16 | 2048 | 128/1/8 (smoke cfg) | 108.1 | 1244 | 90.8 | 1482 | 98.0 | 0.91 |
| 16 | 8192 | 64/2/8 (smoke cfg) | 360.4 | 1491 | 335.9 | 1599 | 350.2 | 0.97 |
| 16 | 8192 | 128/1/1 (best of 12) | 346.0 | 1552 | 331.9 | 1618 | 350.2 | 1.01 |
| 16 | 8192 | 128/1/8 (smoke cfg) | 356.4 | 1507 | 334.1 | 1608 | 350.2 | 0.98 |
| 64 | 2048 | 64/2/8 (smoke cfg) | 370.3 | 1453 | 340.0 | 1582 | 349.8 | 0.94 |
| 64 | 2048 | 128/1/1 (best of 12) | 348.2 | 1545 | 333.5 | 1613 | 349.8 | 1.00 |
| 64 | 2048 | 128/1/8 (smoke cfg) | 370.7 | 1451 | 340.2 | 1581 | 349.8 | 0.94 |
| 64 | 8192 | 64/2/8 (smoke cfg) | 1377.1 | 1560 | 1323.0 | 1624 | 1356.5 | 0.99 |
| 64 | 8192 | 128/1/1 (best of 12) | 1358.8 | 1581 | 1314.2 | 1635 | 1356.5 | 1.00 |
| 64 | 8192 | 128/1/8 (smoke cfg) | 1372.6 | 1565 | 1322.6 | 1625 | 1356.5 | 0.99 |

- With the best config, TileLang matches FlashInfer.
  - Flush mode: within ±1% at three shapes, and 5% faster at B16/KV2048.
  - Graph mode: within 0.3% at three shapes, and 3% slower at 16×2048 (90.4 against 87.7 µs, using the best graph config).
- The best config is num_split=1 at every shape. The smoke configs (num_split=8, tuned for B=1) lose up to 10% at small shapes.
- Even num_split=1 with B=16 (only 128 CTAs, 1 CTA/SM at 80 KB) reaches 1.55 TB/s. So 128 SMs are enough to saturate DRAM, which agrees with the green-context decode numbers in §4.5.
- Correctness: every config has max abs err ≤ 4.3e-4 against fp32. The example's own `ref_program` with a random mask gives ≤ 4.9e-4.
- The decode kernel is therefore **not** a weak point of TileLang at these shapes, and a TileLang-built decode role in a CoKernel starts from a strong solo baseline.

## 3. Prefill (criterion 3)

**Configuration**
- API: `single_prefill_with_kv_cache`, causal, NHD, backend auto (FA2 on sm_120).
- Kernel tile: CTA_TILE_Q=128, KV tile 32, 128 threads, 255 regs/thread, ≈48 KB smem, so 2 CTAs/SM.
- FLOPs = 2·2·S²·D·H_q/2.

**Correctness** against fp32 SDPA (K/V expanded with `repeat_interleave`): all pass.

| S | H_kv | max abs err | bf16 SDPA vs fp32 SDPA | bf16 rounding floor |
|---|---|---|---|---|
| 2048 | 8 | 8.0e-3 | 1.1e-2 | 7.7e-3 |
| 2048 | 32 | 8.0e-3 | 8.6e-3 | 7.8e-3 |
| 8192 | 8 | 8.5e-3 | 7.8e-3 | 7.8e-3 |
| 8192 | 32 | 7.7e-3 | 8.9e-3 | 7.7e-3 |

**Timing.** The peak is 188 × 1024 FLOP/clk × measured MHz.

| S | H_q/H_kv | flush µs | flush TFLOP/s | MHz | graph µs | graph TFLOP/s | MHz | % of mma.sync peak at MHz (graph) |
|---|---|---|---|---|---|---|---|---|
| 2048 | 32/8 | 163.8 | 209.7 | 2541 | 155.2 | 221.4 | 2477 | 46% |
| 2048 | 32/32 | 172.0 ⚠ CV 2.9% | 199.7 | 2515 | 177.5 | 193.5 | 2442 | 41% |
| 8192 | 32/8 | 1660.1 | 331.2 | 2479 | 1707.6 | 321.9 | 2401 | 70% |
| 8192 | 32/32 | 1722.1 | 319.2 | 2405 | 1762.3 | 312.0 | 2342 | 69% |

- S=2048 has 512 CTAs on 376 slots (1.36 waves, with causal imbalance), so the tail dominates.
- In graph mode S=8192 is *slower* than in flush mode. It runs at a lower clock (2401 against 2479 MHz), because back-to-back calls keep the GPU at the power cap.

## 4. POD-Attention (criterion 4)

### 4.1 Build and kernel shape on sm_120

- `PODWithPagedKVCacheWrapper` JIT-builds for `compute_120f`, with 16 mask-combination instantiations (~107 s).
- There is no arch gate. The smem budget is taken from `cudaDevAttrMaxSharedMemoryPerMultiprocessor` (100 KB), and `%nsmid` = 188 = SM count with dense `%smid` (see `2026-09-22_smid_probe`), so the per-SM `tbAssign` array is in bounds.

The instance used (causal prefill + non-causal decode):

| role | tile | threads | notes |
|---|---|---|---|
| prefill | 128×32 | 128 | identical to standalone `single_prefill` |
| decode | 16-row Q tile, 64-row KV per step | 128 | identical to `BatchDecode(use_tensor_cores=True)` |

- The fused kernel uses 255 regs/thread (`cuobjdump -res-usage`) and `smem = max(≈48 KB, ≈36 KB)` (Q + K + V tiles), giving 2 CTAs/SM.
- The decode role inherits the prefill's 255 registers; standalone it needs 134. On sm_120 this coupling does not reduce decode occupancy, because standalone TC decode is already limited to 2 CTAs/SM by smem (36 KB each; the standalone smem is my reading of the dispatch formula, not measured).
- There are no virtual decode CTAs (`TODO_AK` in `pod.cuh`), and prefill runs without KV split: `max_num_kv_chunks` = 0 at these sizes.

Work decomposition (`pod_plan_info.json`). The prefill:decode ticket ratio is chosen per SM by `%smid` ticket.

| config | prefill CTAs | decode CTAs | grid | waves @2/SM | ticket P:D |
|---|---|---|---|---|---|
| P2048 + B16 (KV 2048 or 8192) | 512 | 256 (2-way KV split) | 768 | 2.04 | 2:1 |
| P2048 + B64 | 512 | 512 | 1024 | 2.72 | 1:1 |
| P8192 + B16 | 2048 | 256 | 2304 | 6.13 | 8:1 |
| P8192 + B64 | 2048 | 512 | 2560 | 6.81 | 4:1 |

Decode CTA count does not depend on KV length. The plan splits only by batch, so at KV 8192 each decode CTA streams 2–4 MB.

### 4.2 Correctness

For all 8 configurations (`correctness.json`):
- The prefill output matches fp32 SDPA (max abs err 7.8e-3–9.5e-3) and the decode output matches fp32 (2.8e-4–9.3e-4).
- Both are **bit-identical** to the standalone FlashInfer kernels (single prefill; tensor-core decode).
- 10 back-to-back calls on the default stream are bit-identical.
- After each timed batch, the last outputs were re-checked against the pre-timing reference. POD: 0 diff in all 8 configurations. Serial, stream and green variants: 0, or exactly the CUDA-core-vs-tensor-core 1-ulp difference, identical across variants, so no race.

### 4.3 FlashInfer 0.7.0 bug: scheduler counters reset on the legacy default stream

The code is `include/flashinfer/attention/pod.cuh:413-415`:

```c++
static int* tbAssign = nullptr;
if (tbAssign == nullptr) cudaMalloc(&tbAssign, sizeof(int) * (num_sm + 2));
cudaMemset(tbAssign, 0, sizeof(int) * (num_sm + 2));      // no stream -> legacy default stream
...
cudaLaunchKernelEx(&config /* .stream = torch current stream */, kernel, ...);
```

The per-SM ticket counters and the two per-op CTA-id counters are reset by a *stream-less* `cudaMemset`, while the kernel runs on the caller's stream. The buffer is one process-wide static. I measured the consequences on prefill 8192 + decode 64×8192 (scratch scripts; the numbers are quoted here):

- **Non-blocking stream** (any torch side stream, or a green-context stream): 20 back-to-back calls were **all wrong**. Prefill max abs diff was 3.56, the size of the output itself; decode was 0.086. The memset of call i+1 runs immediately on the idle legacy stream while kernel i is still executing, and resets its counters mid-flight.
- **CUDA graph**:
  - Capture succeeds silently, because the memset runs once, eagerly, and is not captured.
  - Replay 0 is correct (2025 µs).
  - Every later replay takes 14–30 µs and writes nothing. With the outputs zeroed before each replay, they stay zero. All CTAs find the counters exhausted and exit.
  - A graph-mode benchmark of POD would therefore report a ~100× "speedup".
- **Legacy default stream, eager** (what cobench `flush` mode uses when no stream is given): correct and deterministic. The memset is ordered before the kernel and is part of the timed region (a few µs).

What I did:
- The wrapper's `POD.run` raises during stream capture, and on any stream other than the legacy default stream (unless `unsafe_stream_ok=True`).
- All POD timings are flush mode on the default stream.
- I did not patch FlashInfer. The fix would be `cudaMemsetAsync(tbAssign, 0, ..., stream)` plus a per-stream or per-call counter buffer.

A consequence for the project: POD cannot be used inside `bench_corun` or on green streams as-is. It also cannot be CUDA-graphed, which is how serving engines use it.

### 4.4 POD vs serial vs two streams vs green contexts

Variants:
- **serial**: prefill then the faster standalone decode path, one stream, cobench flush mode. This is the same protocol as POD.
- **streams**: `bench_corun(prefill, decode)` on two torch streams.
- **green**: `bench_corun` with prefill on a green context of P SMs (IGNORE_SM_COSCHEDULING, SMs [0,P)) and decode on the other 188−P.
- For streams and green, "vs serial" uses the serial measured inside the same `bench_corun` run, interleaved per rep. That serial agrees with the flush-mode serial within 0.2–1.6%.
- **best green** is an *oracle*: the best of 5 splits, chosen after the fact.
- Solo times are from `bench_corun`.

| prefill S | decode B×KV | solo prefill µs | solo decode µs | serial µs | POD µs (MHz) | POD vs serial | streams makespan µs | streams vs serial | best green (P/D SMs) µs | green vs serial | POD vs best alternative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 16×2048 | 166.0 | 101.4 | 255.6 | 188.6 ⚠ CV 3.3% (2481) | 1.35× | 246.2 | 1.05× | 201.5 (160/28) | 1.29× | 1.07× |
| 2048 | 16×8192 | 165.0 | 351.9 | 509.5 | 397.2 (2691) | 1.28× | 421.6 | 1.21× | 399.3 (128/60) | 1.28× | 1.01× |
| 2048 | 64×2048 | 164.7 | 354.7 | 512.7 | 391.2 (2823) | 1.31× | 419.7 | 1.23× | 393.3 (96/92) | 1.31× | 1.01× |
| 2048 | 64×8192 | 165.7 | 1361.2 | 1520.2 | 1402.8 (2844) | 1.08× | 1460.5 | 1.04× | 1404.1 (60/128) | 1.09× | 1.00× |
| 8192 | 16×2048 | 1673.7 | 102.0 | 1765.0 | 1732.6 (2426) | 1.02× | 1707.2 | 1.04× | 1823.7 (160/28) | 0.97× | 0.99× |
| 8192 | 16×8192 | 1690.0 | 352.0 | 2029.2 | 1837.9 (2312) | 1.10× | 1805.2 | 1.13× | 1940.9 (160/28) | 1.05× | 0.98× |
| 8192 | 64×2048 | 1690.2 | 356.0 | 2037.7 | 1860.9 (2291) | 1.10× | 1771.1 | 1.15× | 1884.8 (160/28) | 1.09× | 0.95× |
| 8192 | 64×8192 | 1694.4 | 1359.1 | 3047.0 | 2204.7 (2066) | 1.38× | 2706.2 | 1.13× | 2541.4 (128/60) | 1.20× | 1.15× |

Where POD helps:
- **POD beats serial everywhere (1.02–1.38×)**. The largest gains come in the two regimes where it fills idle resources:
  - The short, tail-heavy prefill (S=2048: 512 CTAs, 1.36 prefill waves) gets decode CTAs packed into the tail.
  - The balanced heavy pair (P8192 + 64×8192, whose solo times are 1.69 and 1.36 ms) runs at 2205 µs. That is 1.30× the longer solo, against 1.80× for serial.
- **Two streams overlap poorly when the prefill starts first with a big grid.** A prefill CTA takes half an SM (255 regs × 128 threads, 48 KB smem), so decode CTAs get slots only as prefill CTAs retire.
  - For S=2048, the stream makespan is 246 µs against POD's 189 µs.
  - Launch order does not matter: the ab and ba medians are within 1%.

Where POD is tied or loses:
- **POD against the oracle green split**:
  - P2048: ties (≤1%) in 3 of 4 cases, and POD is 7% faster at 16×2048.
  - P8192: POD is 1–6% faster in the three smaller-decode cases and 15% faster at 64×8192.
- **POD against two streams**:
  - POD is faster in all P2048 cases (1.04–1.30×) and at P8192 + 64×8192 (1.23×).
  - It is 1.5–5% *slower* in the other three P8192 cases (1733 against 1707, 1838 against 1805, 1861 against 1771 µs).
- Those losses are mostly a power/clock effect (§4.6). In cycles (µs × MHz), POD and streams are within ±1.6%:
  - P8192_B64_S2048: 4.26 M against 4.25 M;
  - P8192_B16_S8192: 4.25 M against 4.30 M;
  - P8192_B16_S2048: 4.20 M against 4.14 M.

### 4.5 Green-context split sweep (makespan µs; speedup vs serial in parentheses)

| config | 28/160 | 60/128 | 96/92 | 128/60 | 160/28 |
|---|---|---|---|---|---|
| P2048_B16_S2048 | 651 (0.40×) | 366 (0.71×) | 274 (0.95×) | 236 (1.10×) | 201 (1.29×) |
| P2048_B16_S8192 | 832 (0.62×) | 522 (0.98×) | 426 (1.20×) | 399 (1.28×) | 631 (0.81×) |
| P2048_B64_S2048 | 768 (0.67×) | 492 (1.05×) | 393 (1.31×) | 411 (1.25×) | 475 (1.09×) |
| P2048_B64_S8192 | 1519 (1.01×) | 1404 (1.09×) | 1508 (1.01×) | 1541 (0.99×) | 2371 (0.64×) |
| P8192_B16_S2048 | 8369 (0.21×) | 4000 (0.44×) | 2638 (0.67×) | 2111 (0.84×) | 1824 (0.97×) |
| P8192_B16_S8192 | 8582 (0.23×) | 4139 (0.49×) | 2789 (0.73×) | 2196 (0.92×) | 1941 (1.05×) |
| P8192_B64_S2048 | 8518 (0.24×) | 4131 (0.49×) | 2720 (0.75×) | 2143 (0.95×) | 1885 (1.09×) |
| P8192_B64_S8192 | 9253 (0.33×) | 4854 (0.63×) | 3200 (0.96×) | 2541 (1.20×) | 2842 (1.08×) |

Solo times inside each partition (prefill alone on P SMs / decode alone on 188−P SMs, µs):

| config | 28/160 | 60/128 | 96/92 | 128/60 | 160/28 |
|---|---|---|---|---|---|
| P2048_* (prefill) | 584–589 | 299–304 | 217–219 | 184 | 174–176 |
| P8192_* (prefill) | 8304–8352 | 3937–3954 | 2585–2598 | 2060–2067 | 1791–1807 |
| decode 16×2048 | 106 | 105–106 | 102–104 | 106–108 | 166–168 |
| decode 16×8192 | 397 | 363–364 | 357–358 | 362–363 | 596–601 |
| decode 64×2048 | 354 | 352 | 355–356 | 375–376 | 446 |
| decode 64×8192 | 1359–1363 | 1363–1364 | 1482–1486 | 1500–1504 | 2334–2354 |

- The best split differs per configuration (160/28 … 60/128), and a wrong split costs up to 5×. A static partition needs per-pair tuning, which POD's dynamic ticket scheduling avoids.
- Decode saturates DRAM with about 128 SMs: 64×8192 takes 1363 µs on 128 SMs, the same as on 188. It is 9–10% slower on 60–92 SMs, and +73% on 28.
- Prefill scales sub-linearly with SMs. Smaller partitions run at a higher clock and suffer less wave quantization. Solo prefill clock: ≈2840–2870 MHz on 28–60 SMs, ≈2610–2730 MHz on 160 SMs, ≈2440–2680 MHz on the full GPU.
- The FlashInfer decode plan (split-KV) was computed for 188 SMs and is not re-planned for the partition, i.e. the "solo" tier of proposal §4, not "lib".

### 4.6 Clocks: the power cap takes back part of the co-location gain

| config | serial µs @ MHz | POD µs @ MHz | POD vs serial: time | POD vs serial: cycles | streams makespan @ MHz (corun) | streams vs serial: cycles |
|---|---|---|---|---|---|---|
| P2048_B16_S2048 | 256 @ 2640 | 189 @ 2481 | 1.35× | 1.44× | 246 @ 2719 | 1.03× |
| P2048_B16_S8192 | 510 @ 2744 | 397 @ 2691 | 1.28× | 1.31× | 422 @ 2684 | 1.24× |
| P2048_B64_S2048 | 513 @ 2754 | 391 @ 2823 | 1.31× | 1.28× | 420 @ 2710 | 1.25× |
| P2048_B64_S8192 | 1520 @ 2812 | 1403 @ 2844 | 1.08× | 1.07× | 1461 @ 2807 | 1.05× |
| P8192_B16_S2048 | 1765 @ 2481 | 1733 @ 2426 | 1.02× | 1.04× | 1707 @ 2423 | 1.06× |
| P8192_B16_S8192 | 2029 @ 2501 | 1838 @ 2312 | 1.10× | 1.19× | 1805 @ 2380 | 1.19× |
| P8192_B64_S2048 | 2038 @ 2506 | 1861 @ 2291 | 1.10× | 1.20× | 1771 @ 2398 | 1.20× |
| P8192_B64_S8192 | 3047 @ 2606 | 2205 @ 2066 | 1.38× | **1.74×** | 2706 @ 2517 | 1.17× |

- When tensor cores and DRAM are busy at the same time, board power rises and the 600 W cap lowers the SM clock.
- In the heaviest case, POD runs at 2066 MHz against 2606 MHz for serial. Its 1.74× cycle advantage becomes 1.38× in time.
- Co-location results on this GPU must therefore be reported in time, with the clock alongside. A cycles-only or lock-clock evaluation would overstate the gain.

## 5. Surprises and caveats

1. **POD's stream-less `cudaMemset` (§4.3)** is the main practical finding.
   - It silently breaks POD on any non-default stream and under CUDA graphs.
   - The graph failure mode looks like a huge speedup, so any POD number from graph-mode timing, or from the POD kernel inside a multi-stream harness, should be distrusted.
2. POD's advantage over a well-chosen static green split is small (≤1%) in 3 of the 4 short-prefill cases, and POD is 1.07× faster at 16×2048. Its clear wins are:
   - against naive two-stream execution, in tail-heavy cases;
   - in the balanced heavy case (1.15× over the oracle green split).

   It also never needs the per-pair split tuning that green contexts need. Against two streams, it loses 1.5–5% in time in three prefill-dominated cases, which is a tie in cycles.
3. The power cap (§4.6) turns some cycle-level wins into ties. This is a real effect of the target GPU that the evaluation has to model.
4. The TileLang decode example is as fast as FlashInfer's decode at B ≥ 16, once its smem-heavy default config is replaced. Decode is DRAM-bound at 85–91% of peak in both, so there is little solo headroom for any implementation.
5. The FlashInfer POD port on sm_120 has no virtual decode CTAs. Its decode role inherits 255 registers from the prefill role. The register coupling that proposal §4 discusses is visible, but on this GPU it does not reduce CTAs/SM, because smem already limits both roles to 2 CTAs/SM.
6. Two points kept CV > 2% after 3 attempts: prefill S2048/H_kv32 in flush mode (2.9%) and POD P2048_B16_S2048 (3.3%; p10–p90 180–197 µs). Their medians are used as is.
7. Process note: the first correctness run (not timing) overlapped a sibling agent's GPU test. It was re-run under the guard with the same outcome; `correctness.json` is the clean re-run.

## 6. Wrapper module (`research/bench/baselines/flashinfer_ops.py`)

```python
sys.path.insert(0, "research/bench"); import cobench as cb
from baselines import flashinfer_ops as fo
dec = fo.BatchDecode(fo.DecodeShape(batch=64, kv_len=8192), use_tensor_cores=False)  # make_inputs -> (q, k, v, out)
pre = fo.SinglePrefill(fo.PrefillShape(seq_len=8192, num_kv_heads=8))                # make_inputs -> (q, k, v)
pod = fo.POD(fo.PrefillShape(seq_len=8192), fo.DecodeShape(batch=64, kv_len=8192))  # make_inputs -> (q_p,k_p,v_p,q_d,k_c,v_c)
cb.bench(dec.run, make_inputs=dec.make_inputs, mode="graph", nbytes=dec.nbytes, clock=True)
cb.bench(pre.run, make_inputs=pre.make_inputs, mode="flush", flops=pre.flops, clock=True)
x = pod.make_inputs(); cb.bench(pod.run, make_inputs=lambda: x, mode="flush")          # default stream only (§4.3)
a, b = pre.make_inputs(), dec.make_inputs(); cb.bench_corun(lambda: pre.run(*a), lambda: dec.run(*b))
fn = fo.serial(pre, dec); cb.bench(fn, make_inputs=lambda: fo.serial_inputs(pre, dec), mode="flush")
```

- Each class also has:
  - `reference(*inputs)`: the fp32 torch reference;
  - `nbytes` and `flops`;
  - `describe()`.
- Module-level helpers: `fo.err_stats`, `fo.decode_reference`, `fo.prefill_reference`.
- Each `BatchDecode` and `POD` owns its own 128 MB workspace, so two instances can co-run without sharing split-KV scratch.
