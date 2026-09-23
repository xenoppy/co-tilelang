# cobench: measurement framework (P0-2 v0, methodology v1)

cobench is a small Python package for timing kernels and kernel pairs on the RTX PRO 6000 (sm_120). It uses PyTorch plus small raw CUDA kernels compiled with NVRTC through `cuda.bindings`, and has no TileLang dependency.

```
research/bench/
  cobench/   timing.py (bench, bench_corun, bench_variants, flush kinds)  steady.py (bench_steady)
             variants.py (Par, Rotation)  guard.py (pmon GPU guard)  clock.py (ClockProbe)
             nvml.py  green.py  smid.py  kernels.py  cudrv.py
  scripts/   smid_probe.py, repro_matmul.py (P0-2); solo_*.py (P1-S);
             mv1_flush.py, mv1_steady.py, mv1_static.py, sched_variants.py, mv1_common.py (methodology v1)
  tests/test_cobench.py   acceptance tests (P0-2 + methodology v1)
```
Environment: `source research/env.sh`, then `sys.path.insert(0, "research/bench"); import cobench as cb`. Tests: `python research/bench/tests/test_cobench.py [filter...]` (or `python -m pytest research/bench/tests/test_cobench.py`).

Methodology v1 (2026-09-23) changed four things; the evidence is in `research/results/2026-09-23_methodology_v1/README.md`:
1. the default L2 flush is now **clean** (write + `discard.global.L2`), not write-only;
2. a **steady-state** co-run mode (`bench_steady`) is the primary mode for co-run comparisons;
3. every bench waits for, and records, a free GPU with the **pmon guard**;
4. the **ClockProbe** requests the maximum shared-memory carveout. Before this, its SM could not host large-smem CTAs while the probe ran. That artefact caused the P1-S "static persistent penalty" for large-smem kernels (GEMM, looping decode; verified on 3 shapes).

## Which mode for what

| question | mode | result |
|---|---|---|
| co-run comparison (serial / streams / green / CoKernel), **primary for P1** | `bench_steady` | time per iteration under sustained load, clock, power, speed-up vs `serial` from the same run |
| one-shot latency of one op or one co-run rep, cold L2 | `bench(mode="flush")`, `bench_corun`, `bench_variants` (clean flush) | per-call time; flush excluded; flush phase lends power headroom |
| solo throughput of one kernel | `bench(mode="graph")` (rotation) or `bench_steady({"x": f})` | back-to-back, cold inputs, power-capped |
| weights resident in L2 | `bench(mode="hot")` | labelled `l2="hot"` |

**Recommendation for P1 (3×2 study).**
- Report `bench_steady` numbers as the primary result.
  - Speed-ups are taken against the `serial` variant of the same run, as the ratio of medians. The paired per-round ratios are reported alongside.
  - Never compute speed-ups against sums of solo times: power averaging and cache state make the serial run 1–4% faster than the sum.
- Report clean-flush `bench_corun`/`bench_variants` numbers as the auxiliary result.
- Flush mode overstates the benefit of variants that hit the power cap in steady state. In the P1 smoke test:
  - green 140/48: ×1.322 in flush mode vs ×1.220 in steady mode;
  - the best CoKernel: ×1.320 vs ×1.204.

## Protocol

### L2 state and flush kinds (`flush_kind`, default `"clean"`)

The L2 is 128 MB, so every result is labelled with its L2 state (`l2="cold-flush-<kind>" | "cold-rotate" | "hot"`).

| kind | what the flush does (2×L2 = 256 MB buffer, on a side stream, excluded from the time) | op data evicted? | dirty lines left? | cost |
|---|---|---|---|---|
| `clean` (default) | write the buffer, then `discard.global.L2` over it: previous dirty lines are written back during the flush, the buffer's own dirty lines are dropped without write-back | yes | no | 136 µs |
| `write` (v0) | write the buffer | yes | yes, the whole L2 | 164 µs |
| `write+read` | write, then read the same buffer (plan v0.3 wording) | yes | about half (reads that hit still-resident dirty lines do not clean them) | 279 µs |
| `read` | read the buffer | yes for read-only data | no | 165 µs |

Measured (methodology v1, part A):
- A 32 MB buffer that was read 3 times right before the flush reads in 24.2 µs after `clean`, the same as a never-touched buffer (24.4 µs). After `write` or `write+read` it takes 26.2–26.5 µs, because the read pays write-backs.
- A 64 MB write-only kernel takes 16.4 µs after `clean` (hot: 14.5 µs), but 38.9 µs after `write` and 26.2 µs after `write+read`.

**What a clean-flush time means.**
- The op's inputs come from DRAM, and it pays no foreign write-backs.
- Its own output writes, up to about the L2 size, are absorbed by the L2 and written back after the end event. Clean-flush times of write-heavy ops with outputs ≤ L2 are therefore below their steady-state cost:
  - RMSNorm 4096²: 28.2 µs clean-flush vs 44 µs back-to-back;
  - copy 512 MB: 691 vs 737 µs.
- Read-dominated ops agree with back-to-back execution: decode B16×8192 takes 330 µs clean-flush vs 328 µs back-to-back.
- With the old write flush the same ops paid for the flush's dirty lines: decode 342 µs, RMSNorm 43.4 µs, copy 754 µs.

**Flush floor and duty cycle.**
- An empty kernel reads 3.9 µs (bimodal 2.0 / 4.1 µs) in flush mode vs 0.63 µs back-to-back. That is the launch plus event floor.
- Every flush-mode rep adds about 140 µs of low-power (~350 W) flush phase. Under the 600 W cap this lends the clock headroom: matmul 4096³ runs at 2450 MHz in clean-flush mode vs 2150 MHz back-to-back, 9–10% faster in time.

### Steady state (`bench_steady`) — primary co-run mode

- **Variants.** Each variant is a callable that enqueues one iteration on the current stream. If it forks to other streams, it joins them back before returning (`cobench.variants`):
  - `serial = lambda i: (A(i), B(i))`
  - `streams = Par(("a", s1, A), ("b", s2, B))`
  - `green = Par(("a", part.stream, A), ("b", part.rest_stream, B))`
  - `cokernel = lambda i: kernel(*args[i % n])`

  `i` is a global iteration counter used to rotate input copies. `Rotation(make_inputs)` builds ⌈2×L2 / bytes⌉ copies, outputs included, so every iteration is DRAM-cold without a flush.
- **Back-to-back execution.**
  - The host stays `lead_ms` (8 ms) of GPU work ahead.
  - Before every enqueue it checks whether the GPU has already drained the queue. Such a gap raises `HostGapError`.
  - Iterations shorter than 20 µs are timed in groups.
  - Variants whose host enqueue cost exceeds 0.8× their GPU time are rejected.
  - Python GC is off during the run.
- **Warmup.** Round-robin slices for at least `warmup_s`, then (`thermal=True`) until the temperature plateaus: the maximum over the last 15 s is not above the maximum of the 15 s before, capped at 240 s.
  - Power-capped kernels otherwise drift about 5% over the first minutes. With a 5 s warmup, the GEMM solo went 403 → 424 µs over 2 min.
  - With the plateau warmup the slice CV is ≤ 0.6%.
- **Slices.** The variants run in slices of `slice_s` (1.5 s), interleaved in rotating order for `rounds` rounds, with no gaps at slice boundaries. The first `settle_s` (0.5 s) of every slice is excluded, which covers power-controller settling.
- **Reported per variant.**
  - `t_iter_us`: the median over slices of the mean time per iteration.
  - The CV over slices.
  - Per-iteration median, p10 and p90.
  - SM clock from the ClockProbe; board power from the NVML 20 ms samples; temperature per slice.
  - Energy per iteration.
  - For Par variants, per-op completion times relative to the iteration start.
- **Derived.** Speed-up vs `reference` as the ratio of medians, plus the median, min and max of the per-round paired ratios.
- **Validation.**
  - Repeatability: across 3 processes the CV is ≤ 0.28% for times and ≤ 0.21% for speed-ups.
  - Serial vs sum of solos:
    - 0.988 for non-power-capped pairs (decode + RMSNorm, copy + read);
    - 0.958 for GEMM 4096³ + decode, where the power controller averages over both ops.

### Flush mode (`bench`, `bench_corun`, `bench_variants`)

**Host gate.** Before each timed rep, a 1-thread kernel spins on a pinned-memory ticket. The host publishes the ticket only after the whole rep is enqueued, so the start event never includes host latency.
- The timed code must not synchronize the device. This rules out `.item()`, `torch.cuda.synchronize()`, `cudaFree`, and the first launch of a kernel that is not yet loaded.
- The benches run the workload once, ungated, on the target streams before gating.
- A gate that times out (2 s) raises.

**Per-rep order.**
- The flush runs on a full-GPU stream, then the gate, then E0, then the variant, then the end event(s).
- `bench_corun`:
  - Interleaves `corun` (A on stream_a ∥ B on stream_b from E0, with alternating host launch order), `solo_a`, `solo_b`, `serial`, and any `extra` variants in rotating order every rep.
  - Reports `a_end`/`b_end` and makespan relative to E0.
  - Adds per-op completion for Par extras and a speed-up vs `serial` for each extra.
- `bench_variants(variants, reference=...)` is the same without the built-in pair, for single-op or multi-op variants.

**Warmup and statistics.**
- Warmup is ≥ 20 reps and ≥ 1 s (`warmup_s`).
- There are ≥ 50 timed reps.
- Results report median, p10, p90, mean, std, CV, min and max.
- `cv_ok` is CV ≤ 2%.
- The start-to-start period of the timed reps is reported as `period_us`.

### Two-stream co-runs are multimodal

- With GEMM 4096³ ∥ decode B16×8192 on two plain streams, a rep runs in one of two modes:
  - GEMM-first: A ends at about 460 µs, B at about 633 µs, makespan about 637 µs;
  - decode-first: B ends at about 368 µs, A at about 678 µs, makespan about 680 µs.
- Which mode occurs is decided by the hardware scheduler. It does not follow the host launch order: the `ab` and `ba` variants swapped modes between two runs.
- Report per-op completion times; they identify the mode.
- Stream priorities change the mode, but not consistently across steady and flush mode (results §M3.5).
- For an M1 baseline, take the best of both host orders and both priority assignments, measured in the same mode as the study.

### GPU guard (`cobench.guard`)

**Rule.** The GPU is occupied only while a foreign process shows SM% > 0 in `nvidia-smi pmon -s u`. A process that merely holds a context does not count.

**Implementation.**
- `GpuGuard` streams `nvidia-smi pmon -s u -d 1` in a background thread.
- It records foreign samples with SM% > 0 and the host time up to which pmon has reported.
- It ignores this process, its ancestors and its descendants, so a script and its measurement subprocesses do not block each other.

**Helpers.**
- `cb.wait_until_free(poll_s=150, max_wait_s=6h)` blocks while the GPU is busy.
- `cb.foreign_activity(t0, t1)` judges a window after waiting for pmon to cover it.
- `cotile/tests/harness.py: wait_for_gpu()` uses the same guard.

**In the benches.** `bench`, `bench_corun`, `bench_variants` and `bench_steady`:
1. wait for a free GPU;
2. after the timed window, keep the GPU loaded with untimed tail reps until pmon has covered the window (about 1.5–2.5 s);
3. re-measure, up to 3 attempts, if a foreign process was active.

`result.guard` records the attempts.

### Clocks, power, ClockProbe

**Clocks cannot be locked** (no sudo), so they are measured.
- `clock=True` attaches a `ClockProbe`: one sleeping warp logs (`%globaltimer`, `clock64`) every 50 µs, or every 100 µs in `bench_steady`.
- Results include MHz per rep or per slice, and `cycles` = µs × MHz.
- **ClockProbe carveout (v1).** The probe kernel requests `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 100`, and so does the host gate.
  - An SM's carveout can change only while the SM is idle. With the driver's default choice, the probe's SM kept a small carveout in 7 of 8 trials.
  - A 96 KB-smem CTA could then never run there: every large-smem kernel measured with `clock=True` ran on 187 SMs.
  - Regression test: `test_11`.

**NVML** (`nvml=True`).
- The sampler polls clock, power, temperature and throttle reasons, and drains the 20 ms power-sample buffer.
- Direct queries refresh only every ~500 ms.
- `nvmlDeviceGetTotalEnergyConsumption` is unreliable (reads ~1/6 of the sampled power) and is not used.

**Nsight Compute is not usable on this machine.**
- Performance counters are admin-only (`RmProfilingAdminOnly=1`), and ncu fails with `ERR_NVGPUCTRPERM`.
- Use software instrumentation instead: shared-memory-buffered `%globaltimer` tile traces (`scripts/sched_variants.py`).

## API (times in µs)

```python
r = cb.bench(fn, *, make_inputs=None, mode="flush"|"graph"|"hot", flush_kind="clean", warmup=20, reps=50,
             warmup_s=1.0, k=None, min_k=20, stream=None, flops=None, nbytes=None, nvml=False, clock=False,
             rotate_bytes=2*L2, max_copies=4096, gate=True, guard=True, strict=True, label=None) -> BenchResult
#   BenchResult: median p10 p90 mean std cv min max n k n_copies period_us tflops gbps cv_ok samples
#                nvml{...} clock{window, per_rep{...,cycles_median}} flush_kind guard
c = cb.bench_corun(fn_a, fn_b, *, stream_a=None, stream_b=None, flush=True, flush_kind="clean", solo=True,
                   serial=True, extra={name: variant}, order="alternate", reps=50, nvml=False, clock=False,
                   guard=True, label=None) -> CorunResult   # .corun .solo_a .solo_b .serial .extra .derived
v = cb.bench_variants({name: variant}, *, reference=None, flush_kind="clean", reps=50, clock=False,
                      nvml=False, guard=True) -> VariantsResult   # .variants[name]{total, ops, clock} .derived
s = cb.bench_steady({name: variant}, *, reference="serial", slice_s=1.5, settle_s=0.4, rounds=5,
                    warmup_s=3.0, thermal=True, lead_ms=8.0, clock=True, nvml=True, guard=True,
                    max_gaps=0, strict=True, label=None) -> SteadyResult
#   s.variants[name]: t_iter_us cv_slices slices_us iter_median/p10/p90 n_iter clock_mhz power_w
#                     kcycles_per_iter energy_mj_per_iter ops{op: median_us...} gaps
#   s.derived["speedup"][name]: ratio_of_medians paired_median paired_min paired_max ; s.slices ; s.guard
cb.Par((name, stream, fn), ..., order="given"|"reverse"|"alternate");  cb.Rotation(make_inputs, min_bytes=2*L2)
cb.wait_until_free(quiet_s=5, poll_s=150, max_wait_s=6*3600); cb.foreign_activity(t0, t1); cb.get_guard()
with cb.NvmlSampler(interval_ms=10) as s: ...;  s.summary()
p = cb.split_sms(n, ignore_coscheduling=False)   # SmPartition: .n_sms .n_rest .stream .rest_stream
cb.query_split(n, ignore_coscheduling=False); remap = cb.build_sm_remap(stream=None, expected=None)
cb.probe_ctas(grid, block, smem=0, spin_ns=50_000); cb.globaltimer_resolution(); cb.globaltimer_skew()
k = cb.CudaKernel(src, "name", "ppQi"); k(grid, block, *args, smem=0, stream=None); k.set_carveout(pct)
cb.copy_u4(src, dst); cb.read_u4(buf); cb.discard_l2(buf); cb.mma_peak(); cb.device_info(); cb.gpu_state()
```

**Green-context granularity.**
- With default flags the granularity is 8 SMs: 94 → 96 and 180 → 184. Small partitions are spread across four 24-SM groups.
- With `ignore_coscheduling=True` the granularity is 2 SMs and the partition is the contiguous range [0, n).

The ClockProbe runs in the primary context and may sit inside a green partition. Since v1 it no longer takes an SM away from large-smem CTAs there. See `research/results/2026-09-22_smid_probe/README.md`.

## Reference numbers

Values from 2026-09-22 (`results/2026-09-22_cobench_validation/`) and 2026-09-23 (`results/2026-09-23_methodology_v1/`).

- **Peaks.**
  - Dense bf16 (fp32 accumulate) = 188 SMs × 1024 FLOP/clk/SM × f_SM. This gives 504 TFLOP/s at 2617 MHz and about 450 TFLOP/s at the power-capped ~2.2–2.35 GHz.
  - DRAM peak = 1792 GB/s.
- **torch bf16 matmul 4096³.**

  | mode | time | clock | power |
  |---|---|---|---|
  | clean flush | 351–356 µs | 2440–2450 MHz | ~575 W while running |
  | back-to-back (`graph`) | 386–394 µs | 2110–2150 MHz | 600 W |
  | `hot` | 366–372 µs | — | — |

- **copy 512 MB → 512 MB.** 1457 GB/s back-to-back.
- **Reproducibility.**
  - `graph` mode across 5 processes: CV 0.3%.
  - `bench_steady` across 3 processes: CV ≤ 0.3%.
