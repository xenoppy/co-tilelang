# %smid / CTA placement / green-context / %globaltimer probe (P0-2)

- Date: 2026-09-22. Machine: RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs), driver 580.173.02, torch 2.8.0+cu128, NVRTC 12.8.
- Script: `research/bench/scripts/smid_probe.py` (kernels in `research/bench/cobench/smid.py`).
- Files: `summary.json` (all summaries, incl. the full canonical blockIdx→smid order and every green partition's smid ranges) and `placement_raw.json` (per-config smid/ticket/start-time per CTA, first repeat only).
- Reproduce: `CUDA_HOME=/usr/local/cuda-12.9 ~/mpk-env/bin/python research/bench/scripts/smid_probe.py` (about 1 min including NVRTC compilation).

## (i) %smid range and holes

- `%nsmid` = 188 = `multi_processor_count`. The observed `%smid` values are exactly 0..187: dense, with no holes. On the full device the remap table is therefore the identity.
- Inside a green context `%nsmid` still reports 188, and `%smid` is the physical id, not a partition-local one. SM-level binding inside a partition needs that partition's own table: `build_sm_remap(stream, expected=n)`.
- For plan item M4 (FlashInfer POD): per-SM arrays sized by the SM count are safe on the full device, because `%nsmid` equals the SM count and the ids are dense.

## (ii) CTA → SM placement (persistent grids 188 / 376 / 752)

Seven configurations were tested: smem-limited 60/40/20 KB → 1/2/4 CTAs/SM, and thread-limited 1024/512/128/32 threads → 1/3/12/24 CTAs/SM. Each CTA spins for 200 µs so that a wave stays co-resident, and each launch was repeated 3×.

**Breadth-first.**
- In every configuration, the first min(grid, 188) blocks land on 188 distinct SMs.
- An SM receives its second CTA only after every SM has one.
- With grid = c×188 ≤ occupancy×188, every SM gets exactly c CTAs (histogram {c: 188}).
- The first wave never exceeds the occupancy calculator's figure. For example, 512-thread CTAs with grid 752 put exactly 3 CTAs on each SM (564 in wave 1).

**Deterministic, identical canonical order.**
- In the first wave, block b goes to `canonical[b mod 188]`.
- This order is identical across all 7 configurations and all repeats. When capacity allows, block b+188 lands on the same SM as block b.

**Not smid order.**
- Consecutive block pairs fill both SMs of a TPC (smids 2t and 2t+1).
- TPCs are taken round-robin over 8 groups of 24 smids (0–23, 24–47, …, 144–167, 168–187). The last group has only 20 SMs (10 TPCs) and joins from the 3rd round, which balances 12-TPC groups against a 10-TPC group.
- Head of the canonical order: `0,1,24,25,48,49,72,73,96,97,120,121,144,145,2,3,26,27,…` (the full list is in `summary.json`).
- The 24-SM groups are probably GPCs. That is an inference from the ordering, not verified.

**Launch timing.**
- All first-wave CTAs, up to 752, start within 32–96 ns of each other (1–3 `%globaltimer` ticks), so order within a wave cannot be resolved.
- Atomic-ticket order bears no relation to blockIdx.

**Later waves** (grid > occupancy×188) go to whichever SM frees first. Their placement is not reproducible: the same mapping recurs in only 26–75% of repeats, and block b+188 shares block b's SM only 0–2% of the time.

**Warp ids.** `%nwarpid` = 48. `%warpid` of a CTA's warp 0 is usually not 0; for 512-thread CTAs it takes values 0–3, 16–19 and 32–35. It is an SM slot id, not a CTA-local index.

**Implication for SM-level binding.**
- A persistent grid of 188×c CTAs gives exactly c CTAs per SM, and the blockIdx→SM map is deterministic.
- That map is not the identity, so role-by-SM decisions must read `%smid` (through the remap table), not blockIdx.

## (iii) Green contexts (cuDevSmResourceSplitByCount + cuGreenCtxCreate)

| request n | default: got / rest | default: SMs used | IGNORE_SM_COSCHEDULING: got / rest | SMs used |
|---|---|---|---|---|
| 8 | 8 / 180 | 0-1,24-25,48-49,72-73 | 8 / 180 | 0-7 |
| 16 | 16 / 172 | 0-3,24-27,48-51,72-75 | 16 / 172 | 0-15 |
| 32 | 32 / 156 | 0-7,24-31,48-55,72-79 | 32 / 156 | 0-31 |
| 64 | 64 / 124 | 0-15,24-39,48-63,72-87 | 64 / 124 | 0-63 |
| 94 | **96** / 92 | 0-95 | 94 / 94 | 0-93 |
| 128 | 128 / 60 | 0-95,100-107,124-131,148-155,168-175 | 128 / 60 | 0-127 |
| 180 | **184** / 4 | all but 122-123,146-147 | 180 / 8 | 0-179 |

**Default flags.**
- Granularity is 8 SMs: requests 1..8 give 8, 94 gives 96, 180 gives 184, and 185..188 give 188 (a full sweep over 1..188 is in `summary.json`).
- An 8-SM unit is one TPC from each of 4 of the 24-SM groups, so small partitions are spread across groups and are not contiguous.

**With `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING`.**
- Granularity is 2 SMs (one TPC): odd requests round up.
- The partition is the contiguous range [0, n).
- The driver documents that this flag gives up "advanced features" such as large clusters.

**In both modes.**
- The partition and its remainder are disjoint and together cover all 188 SMs.
- Re-creating the same request selects the same SMs.
- Work is confined to the partition's SMs, checked with the probe (distinct smids = partition size). This holds for torch ops on the `ExternalStream`, for NVRTC `CUkernel` launches, and for CUDA graphs captured on the green stream.
- Torch events recorded on green streams time correctly against events on primary-context streams.

## (iv) %globaltimer

**Resolution: 32 ns.**
- Successive observed changes differ by 32 ns (79%) or 64 ns (21%), never less.
- There were 0 decreases across about 96k changes on 188 SMs.
- A read costs about 111 SM cycles.
- This is not the ~1 µs granularity seen on some older GPUs.
- CUDA event timestamps have the same 32 ns quantum.

**Cross-SM consistency.**
- Pairwise flag ping-pong between all 188 SMs (564 rounds) gives a one-way latency of 320–512 ns (median 384).
- Clock offsets relative to SM 0 are within ±64 ns (median |offset| 16 ns), i.e. ≤ 2 ticks.
- There were no causality violations: no receiver timestamp preceded its sender's.
- `%globaltimer` can therefore serve as a device-global clock for per-role completion timestamps in CoKernels, with roughly ±0.1 µs accuracy.

**Clock readout.** d(clock64)/d(`%globaltimer`) gives the SM clock (2868 MHz idle boost in this run). `cobench.clock.ClockProbe` uses this because NVML refreshes clocks only every ~500 ms (see `research/bench/README.md`).

## API

`cobench.build_sm_remap(stream=None, expected=None) -> SmRemap` returns an object with:
- `.table`: int32, `smid → dense index`, −1 for unused ids.
- `.smids`
- `.holes`
- `.to_tensor()`

It launches a 1-CTA/SM probe with 4×expected CTAs and raises if the number of distinct smids differs from `expected`. `expected` defaults to 188; pass the partition size for a green stream.
