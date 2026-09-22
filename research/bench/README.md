# cobench: measurement framework v0 (P0-2)

cobench is a small Python package for timing kernels and kernel pairs on the RTX PRO 6000 (sm_120). It uses PyTorch plus small raw CUDA kernels compiled with NVRTC through `cuda.bindings`, and has no TileLang dependency.

```
research/bench/
  cobench/            timing.py  nvml.py  clock.py  green.py  smid.py  kernels.py  cudrv.py
  scripts/            smid_probe.py (P0-2 §5)   repro_matmul.py (P0-2 §6)
  tests/test_cobench.py   acceptance tests (writes results/2026-09-22_cobench_validation/)
```
Environment: `CUDA_HOME=/usr/local/cuda-12.9 ~/mpk-env/bin/python ...`. Then `sys.path.insert(0, "research/bench"); import cobench as cb`.

## Protocol

**L2 state.** The L2 is 128 MB, so every result is labelled with its L2 state.

| mode | what is timed | L2 | use for |
|---|---|---|---|
| `flush` | one call between CUDA events; beforehand a 256 MB (2×L2) buffer is written on a separate full-GPU stream (excluded from the time) | cold | calls ≳ 50 µs; co-run reps |
| `graph` | a CUDA graph of k back-to-back calls that rotate over ⌈2×L2 / input bytes⌉ copies from `make_inputs`; the result is per call = replay time / k | cold | default for single kernels, including short ones |
| `hot` | a graph of k calls on one copy, no flush | hot | "weights resident in L2" scenarios only |

- `flush` carries a floor of about 3 µs of launch plus event latency, with jitter. A 1-element kernel reads 4.1 µs in `flush` against 0.94 µs in `hot`.
- `flush` also runs at a lower compute duty cycle: every call is followed by a 256 MB write. Power is lower, so the power-capped clock is higher. Matmul 4096³ ran at 2580 MHz in `flush` against 2214 MHz in `graph`.
- Absolute times therefore differ across modes partly because of the clock. Compare modes with `clock=True` and `cycles_median` (µs × MHz). In cycles, matmul in `flush` is slower (79% of peak) than in `graph` or `hot` (86%).
- `graph` refuses to rotate more than `max_copies` = 4096 copies. Inputs that are too small to rotate cannot be made cold that way; use `flush` for them.

**Host gate.** Before each timed rep, a 1-thread kernel spins on a pinned-memory ticket that the host publishes only after the whole rep is enqueued. The start event therefore never includes host latency.
- Consequence: the timed code must not synchronize the device (`.item()`, `torch.cuda.synchronize()`, `cudaFree`, or the first launch of a not-yet-loaded kernel, since lazy module loading forces a context-wide sync).
- `bench`/`bench_corun` run the workload once without the gate, on the target streams, before gating.
- A gate that times out (2 s) raises; the reps are never silently corrupted.

**Warmup.**
- The minimum is ≥ 20 reps AND ≥ `warmup_s` = 1.0 s of load.
- Under the 600 W cap, the SM clock drops from about 2.4–2.9 GHz to about 2.2–2.35 GHz within 0.1–0.3 s of heavy load.
- A short warmup, or any host gap of more than ~0.1 s before the timed reps, produces a fast-then-slow timed window. With 20 warmup replays, one matmul run measured a within-run CV of 3.9%.
- The sampler and probe are therefore constructed, and their timing events pre-created, before warmup.

**Reps and statistics.**
- There are ≥ 50 timed reps.
- Results report median, p10, p90, mean, std, CV, min and max, all in µs.
- `cv_ok` is CV ≤ 2%. Points that fail it are re-measured.

**Clocks.** The clocks cannot be locked (no sudo), so they are measured:
- `clock=True` attaches a `ClockProbe`: one sleeping warp logs (`%globaltimer`, `clock64`) every 50 µs.
- The result then includes the SM MHz per rep, the correlation between time and clock, and the clock-normalized time `cycles_median` = µs × MHz. Use cycles to compare runs taken at different thermal states.
- `nvml=True` attaches an `NvmlSampler`, which polls every 10 ms and reports power (true 20 ms samples from `nvmlDeviceGetSamples`), temperature and throttle reasons.

**NVML caveats** (measured on driver 580.173.02):
- Clock, power, temperature and reasons are all refreshed only every ~500 ms. Polling at 10 ms therefore repeats values, and windows shorter than about 1 s may see stale ones (`stale_warning`).
- `nvmlDeviceGetTotalEnergyConsumption` reads about 1/6 of the sampled power and is not used.

**Co-run** (`bench_corun`). Each rep runs on a full-GPU launcher stream L, in this order:
1. Flush the L2.
2. Pass the gate.
3. Record the common start event E0.
4. Both op streams wait on E0; each then runs its op and records its own end event.

- Results are reported relative to E0: `a_end`, `b_end`, and makespan = max of the two.
- `solo_a` (on stream_a), `solo_b` (on stream_b) and `serial` (a then b on L) are measured in the same way. They are interleaved with the co-run inside every rep, in rotating order, so that clock drift cancels.
- The host launch order alternates between ab and ba, and results are also reported per order (`by_order`).

**Other rules.**
- Check `nvidia-smi --query-compute-apps=...` before every GPU run. The tests record other GPU processes at start and end.
- The ClockProbe and HostGate kernels are resident while timing is in progress. Never call `torch.cuda.synchronize()` in code that runs while a probe is active, since it would wait for the probe.

## API (times in µs)

```python
r = cb.bench(fn, *, make_inputs=None, mode="flush"|"graph"|"hot", warmup=20, reps=50,
             warmup_s=1.0, k=None, min_k=20, stream=None, flops=None, nbytes=None,
             nvml=False, clock=False, rotate_bytes=2*L2, max_copies=4096, gate=True,
             strict=True, label=None) -> BenchResult
#   fn(*make_inputs()) launches on torch's current stream (= `stream`, e.g. a green stream).
#   BenchResult: median p10 p90 mean std cv min max n k n_copies tflops gbps cv_ok samples
#                nvml{...} clock{window, per_rep{min,median,max,cycles_median,corr_time_vs_clock}}
c = cb.bench_corun(fn_a, fn_b, *, stream_a=None, stream_b=None, warmup=20, reps=50,
                   warmup_s=1.0, flush=True, solo=True, serial=True, order="alternate",
                   nvml=False, clock=False, label=None) -> CorunResult
#   c.corun{makespan,a_end,b_end: stats, by_order}, c.solo_a, c.solo_b, c.serial{total,a},
#   c.derived{speedup_vs_serial, slowdown_a/b, makespan_over_max_solo, cv_ok}, c.clock{per_variant}
with cb.NvmlSampler(interval_ms=10) as s: ...;  s.summary()
p = cb.split_sms(n, ignore_coscheduling=False)   # -> SmPartition; p.n_sms, p.n_rest,
                                                 #    p.stream / p.rest_stream (torch ExternalStream)
cb.query_split(n, ignore_coscheduling=False)     # (got, rest) without creating contexts
remap = cb.build_sm_remap(stream=None, expected=None)  # SmRemap: .table (smid->dense), .to_tensor()
cb.probe_ctas(grid, block, smem=0, spin_ns=50_000, stream=None)  # per-CTA smid/ticket/globaltimer
cb.globaltimer_resolution(); cb.globaltimer_skew()
k = cb.CudaKernel(src, "name", "ppQi")            # NVRTC + cuLibrary (context-independent):
k(grid, block, *args, smem=0, stream=None)        #   launchable on normal AND green streams
cb.copy_u4(src, dst); cb.mma_peak(); cb.device_info(); cb.gpu_state()
```

Green-context granularity:
- With default flags it is 8 SMs, so 94→96 and 180→184, and small partitions are spread across four 24-SM groups.
- With `ignore_coscheduling=True` it is 2 SMs and the partition is contiguous [0, n).

See `research/results/2026-09-22_smid_probe/README.md`.

## Reference numbers (2026-09-22; see `research/results/2026-09-22_cobench_validation/`)

**Peaks.**
- Dense bf16 (fp32 accumulate) peak = 188 SMs × 1024 FLOP/clk/SM × f_SM. The 1024 FLOP/clk/SM was measured with `mma_peak` (mma.sync m16n8k16).
- That is 504 TFLOP/s at the 2617 MHz spec boost and 595 TFLOP/s at the 3090 MHz max clock. Under sustained matmul the power cap holds the clock at about 2.2–2.35 GHz, which gives about 450 TFLOP/s.
- DRAM peak = 512 bit × 14001 MHz × 2 = 1792 GB/s.

**Measured.**

| workload | time | throughput | fraction of peak | conditions |
|---|---|---|---|---|
| torch bf16 matmul 4096³, `graph` | ~370 µs | ~370 TFLOP/s | ~86% of the peak at the measured clock | 600 W cap, ~2.2 GHz |
| same, `flush` | — | ~395 TFLOP/s | — | short window, ~2.5 GHz |
| copy 512 MB→512 MB | — | 1460–1490 GB/s | ~82% of DRAM peak | — |

- 16 MB copy: 1490 GB/s in `flush` and `graph` (cold), 6550 GB/s in `hot` (L2).
- Reproducibility of matmul in `graph` mode across 5 processes: CV 0.3%, within-run CV about 0.3%. The residual is thermal drift, about 1% over five runs as the clock falls 2215→2199 MHz; cycles/call CV is 0.06%.
