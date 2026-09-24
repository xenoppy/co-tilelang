# P1-3x2-B — resume note (paused 2026-09-23 ~05:50 on the main agent's instruction)

> **Historical note (2026-09-24):** this pause note is superseded. All remaining work (r223, second, the r029 redo, the tables, the README and the full test suites) was finished on 2026-09-24; see `README.md`.

GPU work was stopped because other people are using the cluster. No process of ours is running.
Nothing below has been committed.

## Finished

**Code (all built; the kernel cache holds no stale hinted kernels).**
- TileLang core: `T.copy(..., eviction_policy=...)` now also works for cp.async copies. Before, the annotation was honoured only for TMA copies and silently ignored otherwise.
  - `src/op/builtin.h`: new AttrStmt key `tl.cp_async_l2_eviction_policy`.
  - `src/cuda/op/copy.cc` (`LowerCPAsync`): wraps the injected cp.async in that key. It warns when the hint cannot be honoured.
  - `src/cuda/codegen/codegen_cuda.{h,cc}`: emits `tl::cp_async_gs[_conditional]_l2hint<N, tl::L2EvictionPolicy::…>`.
  - `src/cuda/codegen/codegen_cutedsl.cc`: fails loudly for the new key.
  - `src/tl_templates/cuda/copy.h`: the hinted cp.async is `.L2::cache_hint` with a `createpolicy.fractional` policy. It uses a 64-bit `cvta.to.shared` address.
  - `cmake --build build` has been run.
- **ptxas 12.9 miscompile, found and worked around.**
  - With the 32-bit shared address, 15 of 43 hinted GEMM configs got an LDGSTS reading its descriptor from a never-written odd uniform register (`[Rx+UR0], desc[UR1]`). These kernels fault with "illegal instruction" at run time.
  - Reproduced standalone with nvcc. Independent of -O level, .ca/.cg, createpolicy vs constant policy, and sm_120 vs sm_120a.
  - With the 64-bit address, all 332 GEMM/decode kernels are lint-clean.
  - Lint: `cotile.resources.invalid_memory_descriptors(sass)`. The fix is documented in `cotile/README.md`.
- `cotile/ops`: new config axes, bitwise-neutral, with tag suffixes `_l2ef` / `_l2el`:
  - `DecodeConfig.kv_l2` for the K/V loads;
  - `GemmConfig.ab_l2` for the A/B loads.
- Tests:
  - `cotile/tests/test_ops.py::run_l2_hints` passes **32/32** on the GPU: source equals the twin modulo the hint, every K/V (A/B) load is hinted, SASS lint is clean, and outputs are bitwise equal to the twin and within tolerance.
  - `test_cokernel.py` has a new pair `gd_l2`: `--pairs gd_l2` passed **20/20** launchable settings. The one non-launchable setting is `smem="sum"`, as expected.
  - It also records `invalid_mem_desc` per CoKernel.
- `research/bench/scripts/p1_common.py`: `StatePool` shares split workspaces and counters, and `CoVariant(pool=)` / `OpData.launcher(state=)` use it. This keeps our GPU memory at about 2.4 GB.
- `research/bench/scripts/p1b_study.py`: the B study. `p1b_report.py`: the tables (see "untested").

**Data: main pair, inter column (green derived search), stages DI1–DI4.** File: `main/study.json`; log: `run_main.log`.
- Preliminary result, not final. Stage F, the interleaved final run, has not run yet.
- Decode K/V `evict_first` in the green split: 577.0 vs 585.0 µs (same configs and split, one steady run, −1.4%).
- The best derived green variant is 568.9 µs (×1.273). It uses lib configs (GEMM `128x128x64_s2_t256_wsauto` on 156 SMs, decode `n64_h4_sp1_t64_s1`) plus the decode hint.
  - Part A's lib-inter winner, re-measured in the same run, is 586.0 µs (×1.236), so the derived variant is 3.0% faster.
  - Energy is 341 vs 352 mJ/iter.
- GEMM `evict_last` is neutral.
- **Caveat:** all green variants measured about 1.5% slower in DI4 than in DI3, while serial moved only 0.2%. The power-capped operating point shifts between steady stages. Only within-run comparisons (and F) count, which is why F re-measures the top-3 derived and top-2 lib candidates per column together.

## Interrupted

- Main pair, stage **DC1** (intra flush screen). It was still in the pmon guard wait (a foreign benchmark was active), so nothing of DC1 was measured or saved.

## Remaining

1. Main: DC1, DC2, DC3, DC4 (intra derived search, incl. the CTA register-cap axis), F (final 3×2 run and ablations), R (B3 robustness).
2. r029, r058, r223, second: all stages DI1…R.
   - Stage R of these pairs needs main's R first (it provides the cross-pair transfer split).
   - If main's F shows no derived gain ≥ 3%, the task allows reducing these pairs to a confirmation of the best axis.
3. `python p1b_report.py` → `tables.json`, `tables.md`. Then write `README.md` for this directory (B1 definition, per-axis findings, 3×2 tables, robustness, D1 readout, GPU time) and add the new scripts to `research/bench/README.md`.
4. Full test suites (B5):
   - `python -m cotile.tests.test_ops`
   - `python -m cotile.tests.test_cokernel`
   - Neither has been re-run in full after these changes; only the L2 subset and the `gd_l2` pair ran.

GPU time used so far is about 15 min: DI1–DI4 took 520 s, plus the lint and test runs. Estimate for the rest: about 12 min (main) + 4 × 16 min + about 15 min of tests ≈ 1.5 h.

## Resume commands

Run only after the go-ahead. Every command below creates a CUDA context, including `--compile-only`, because `Study.__init__` makes torch streams.

```bash
source research/env.sh
cd research/bench/scripts
python p1b_study.py main --stages DC1,DC2,DC3,DC4,F,R \
    >> ../../results/2026-09-23_p1_3x2_B/run_main.log 2>&1        # resumable; done stages are skipped
for p in r029 r058 r223 second; do
  python p1b_study.py $p > ../../results/2026-09-23_p1_3x2_B/run_$p.log 2>&1
done
python p1b_report.py
cd ../../.. && python -m cotile.tests.test_ops && python -m cotile.tests.test_cokernel
```

To stop a run, kill its PID. `pkill -f "p1b_study.py …"` also matches a watcher shell whose own command line contains that string.

## Untested code (still to be exercised)

- `p1b_study.py`: stages DC1–DC4, F and R have never run. DI1–DI4 ran.
  - DC1: after the first run, the CTA-contract pre-filter was switched to the grid-build smem model. The compiled CTAs/SM remains the authoritative check.
  - F: re-measures the top-k candidates (`ranked`), includes the ablations of each derived axis, the solo runs of the winners' configs, and the timing builds.
  - R: `rule_splits`, `robustness_summary`.
- `p1b_report.py`: never run, because there are no F data yet.
- The full `test_ops` / `test_cokernel` suites (see above).
- `3rdparty/tvm` shows as modified because of the Z3 patch from part A, not because of this task.
