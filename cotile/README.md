# cotile — tile-program library for CoKernels (v0)

`cotile` sits on top of TileLang (this repo's `tilelang/`) and will host the CoKernel
compiler (research/plan.md §3, §8). v0 contains the single-op pieces a two-role
CoKernel is built from, and the CoKernel builder v0 (see "CoKernel builder" below):

| module | contents |
|---|---|
| `cotile/ops/gemm.py` | GEMM `C = A @ B^T` (NT), bf16 in/out, fp32 accumulate, split-K with in-kernel combine |
| `cotile/ops/gqa_decode.py` | GQA decode attention (Hq=32, Hkv=8, D=128 by default), split-KV with in-kernel combine |
| `cotile/ops/rmsnorm.py` | RMSNorm `Y = X * rsqrt(mean(X^2) + eps) * W`, explicit thread/vector mapping |
| `cotile/kernel.py` | op protocol types, generic `build_grid` / `build_persistent`, `compile_specs` (par_compile), `Runner` |
| `cotile/resources.py` | resource signature of a *compiled* kernel + occupancy model + driver cross-check |
| `cotile/device.py` | `DeviceSpec` for the RTX PRO 6000 Blackwell (sm_120) |
| `cotile/cokernel.py` | two-role persistent CoKernel builder (`build_cokernel`, `Orch`, `CoRunner`), `%smid`/`%globaltimer` helpers |
| `cotile/tests/` | correctness / bitwise / repeated-launch tests and signature export; `test_cokernel.py`, `probe_cokernel_overhead.py` |

## Op protocol

Every op module exposes the same functions, so a CoKernel builder can treat ops
uniformly:

```python
from cotile.ops import gemm
shape = gemm.GemmShape(M=1024, N=4096, K=4096)
cfg   = gemm.GemmConfig(block_M=128, block_N=128, block_K=64, num_stages=2, threads=128, split_k=2)

gemm.validate(shape, cfg)                 # ValueError if unsupported
ts   = gemm.tile_space(shape, cfg)        # ts.num_tiles, ts.decode(tile_id) -> namedtuple coords
io   = gemm.io_params(shape, cfg)         # [IOParam(name, shape, dtype, role)], role in in/out/ws/ctr
scr  = gemm.scratch_spec(shape, cfg)      # [ScratchBuf(name, shape, dtype)]  shared memory of one tile
body = gemm.make_tile_body(shape, cfg)    # @T.macro body(tile_id, io_tuple, scratch_tuple)
gemm.smem_bytes(shape, cfg, "persistent") # predicted per-CTA user smem (static + dynamic)
gemm.numerics(cfg)                        # ("E0"|"E1", key): equal keys => bitwise-identical outputs
gemm.configs(shape)                       # 30-50 valid configs (fit 101376 B in both builds)
gemm.build_grid(shape, cfg)               # KernelSpec: one CTA per tile
gemm.build_persistent(shape, cfg, 188)    # KernelSpec: 188 CTAs, static grid-stride over tiles
gemm.make_inputs(shape), gemm.reference(shape, inputs), gemm.TOLERANCE
```

* **Tile body.** `tile_body(tile_id, io, scratch)` performs one whole tile: loads,
  compute, epilogue, global writes, and (for split tiles) the arrival/combine
  protocol. `tile_id` may be any block-uniform expression (blockIdx, a grid-stride
  counter, an atomically fetched queue slot). The body starts with `__syncthreads()`
  so the scratch can be reused by the next tile the CTA executes (of any role).
  Register fragments are allocated inside the macro; **shared memory is not**: the
  kernel builder calls `kernel.alloc_scratch(op.scratch_spec(...))` once per CTA and
  passes the namedtuple in, so a CoKernel can allocate, alias or partition it.
  Exception: TileLang's cross-warp `T.reduce_*` (used by the decode tile) allocates a
  hidden workspace outside the scratch (see resource notes).
* **Builders.** `kernel.build_grid(op, shape, cfg)` / `kernel.build_persistent(op,
  shape, cfg, num_ctas)` are written once for all ops. Kernel signatures are assembled
  programmatically (`kernel.make_prim_func`, eager `Builder` + `builder.arg`), so a
  CoKernel can take the union of two ops' `io_params`.
* **Pass configs.** Persistent builds (and later CoKernels) always use
  `{"tl.disable_warp_specialized": True}` (cp.async + mma.sync). GEMM grid builds can
  also use TileLang's default auto warp specialization (`GemmConfig.ws="auto"`: TMA +
  mbarriers, +128 producer threads). `tl.disable_tma_lower=True` is *not* used: it
  crashes lowering of the GEMM smem epilogue ("original buffer shape must match the
  layout input shape"), so smem->global copies of the GEMM epilogue become TMA stores.

## Split reductions (flat bag of independent tiles)

Split-K GEMM and split-KV decode keep all tiles independent (no tile waits for
another, so any persistent CTA count is deadlock-free). Each split tile writes an
fp32 partial to a workspace (`ws` role), then `__syncthreads()`, then thread 0 does
`prev = T.atomic_add(Ctr[g], 1, memory_order="acq_rel", return_prev=True)`
(`atom.acq_rel.gpu`, the CUTLASS arrive pattern: CTA barrier then a gpu-scope release
by one thread). The tile that sees `prev == S-1` combines all S partials **in fixed
order s = 0..S-1** (so results do not depend on which split arrives last, or on the
build) and writes `Ctr[g] = 0`.

**Counters self-reset**: zero them once at allocation (`Runner` does this); every
completed launch leaves them zero again. Only an aborted launch requires re-zeroing.
Tested: 3 extra launches per build with outputs *and* workspaces poisoned with NaN
before each launch, counters checked after each, plus grid/persistent kernels
alternating on one shared counter/workspace state.

## Ops and config spaces

* **GEMM** (`GemmConfig`): `block_M/N/K`, `num_stages`, `threads`, `split_k` in
  {1,2,4}, `ws` in {off, auto} (grid build only), `epilogue` in {smem, direct},
  `group_m` (grouped rasterization, done in `decode`, not with `T.use_swizzle`, which
  would hijack blockIdx kernel-wide). Requires M % block_M == N % block_N ==
  K % (block_K*split_k) == 0 (ragged shapes not implemented in v0).
  Numerics: E0 for split_k = 1 (all E0 configs are bitwise identical, grid and
  persistent), E1 otherwise; key = (mma shape, split_k).
* **GQA decode** (`DecodeConfig`): `block_N` (KV block), `heads_per_cta` (query heads
  of one KV group processed together, padded to one 16-row MMA tile), `num_split`
  (split-KV), `threads`, `num_stages`. Layouts as in
  `examples/flash_decoding/example_gqa_decode.py`: Q [B,Hq,D], K/V [B,S,Hkv,D],
  O [B,Hq,D]; no mask. Warps are laid out along N in both GEMMs
  (`GemmWarpPolicy.FullCol`) and P goes through smem. Requires
  S % (num_split*block_N) == 0. Numerics: all E1; key = (block_N, num_split,
  threads); heads_per_cta, num_stages and the build are bitwise-neutral.
  TileLang limitation: with >= 8 warps and block_N/warps >= 16 the cross-warp
  `T.reduce_*` lowering fails ("ReduceOp cannot lower a layout where a source index
  depends on a thread-owned reduce segment"; seen for 128/8, 256/8, 256/16 warps);
  `validate` rejects that combination.
* **RMSNorm** (`RMSNormConfig`): `rows_per_cta`, `threads`, `vec` (vector width of
  every global access). SIMT-style so the reduction tree is explicit: per-thread
  sequential -> warp butterfly -> per-warp partials in the smem scratch -> fixed-order
  sum. Only `__syncthreads` (no TileLang AllReduce named barriers). Numerics: E1;
  key = (threads, vec).

## Resource signatures (`cotile.resources.signature(spec)`)

Read from the compiled kernel, not from source parameters:

* cubin from the kernel's CUDA module (`inspect_source("cubin")`; the FFI layer cannot
  UTF-8-decode the bytes and the UnicodeDecodeError carries them),
* `cuobjdump --dump-resource-usage` (REG, LOCAL, STACK, SHARED) and `cuobjdump -elf`
  (`EIATTR_MAX_THREADS` = compiled threads/CTA, `EIATTR_NUM_BARRIERS`, parameter count),
* launch constants (grid, block, dynamic smem) from TileLang's host stub source.

Fields: `threads` vs `threads_src` (`threads_changed`), `regs`, `smem_static`
(user bytes; cuobjdump's SHARED includes the 1 KB per-CTA reserved smem on sm_90+,
which is subtracted), `smem_dynamic`, `smem_total`, `smem_estimate` (the op's own
prediction), `num_barriers`, `ctas_per_sm` + per-resource limits
(`cuda_occupancy.h` rules for cc 12.0: 48 warps, 24 CTAs, per-sub-partition register
allocation in 256-register units, smem + 1 KB reserve in 128 B units out of 100 KB,
**and 24 / num_barriers**). `driver_check` loads the cubin with the CUDA driver and
reports `cuFuncGetAttribute` values and `cuOccupancyMaxActiveBlocksPerMultiprocessor`;
the tests store them as `drv_*` columns. Parsed cuobjdump results are cached in
`$COTILE_CACHE_DIR/signatures.json` (default `~/.cache/cotile`).

`op.smem_bytes(shape, cfg, build)` predicts per-CTA user smem before compiling
(configs are filtered with it). On the test shapes it matches the compiled kernels
exactly for all 118 persistent builds and all GEMM/RMSNorm grid builds; for 6 decode
grid builds it is a safe upper bound (TileLang's liveness planner reuses dead
buffers there). Model notes: persistent loops keep *every* scratch buffer live (sum,
no phase reuse), the grid build overlaps GEMM's C staging buffer with the dead A/B
pipeline buffers, auto-WS puts its mbarriers in static smem padded to 1 KB, and each
cross-warp `T.reduce_*` call site of the decode tile adds threads*4 bytes per
pipeline copy of the loop body.

## Status (2026-09-22, results in research/results/2026-09-22_op_library/)

| op | configs | kernels | compile (cold, 32 workers) | both builds correct | grid == persistent (bitwise) | numerics-key groups bitwise | split configs: repeat x3 / counters 0 / shared state |
|---|---|---|---|---|---|---|---|
| gemm | 43 | 86 | 31.7 s | 43/43 | 43/43 | 3 groups, 43/43 | 8/8 / 8/8 / 8/8 |
| gqa_decode | 36 | 72 | 39.2 s | 36/36 | 36/36 | 23 groups, 36/36 | 20/20 / 20/20 / 20/20 |
| rmsnorm | 39 | 78 | 10.6 s | 39/39 | 39/39 | 12 groups, 39/39 | - |

Registers and CTAs/SM from `cotile.resources` agree with the CUDA driver
(`cuFuncGetAttribute`, `cuOccupancyMaxActiveBlocksPerMultiprocessor`) for all 236
kernels.

## Running the tests

```bash
source research/env.sh
python -m cotile.tests.test_ops                                  # all ops, all configs
python -m cotile.tests.test_ops --ops gemm --limit 4             # quick subset
python -m cotile.tests.test_ops --no-gpu                         # compile + signatures only
python -m cotile.tests.test_ops --cold --out research/results/2026-09-22_op_library
```

The GPU phase first checks `nvidia-smi` for other compute processes and waits (poll
every 150 s, up to 15 min) if a sibling is measuring. pytest is not installed in
`~/mpk-env`; `test_gemm/test_gqa_decode/test_rmsnorm` are pytest-compatible anyway.
Test shapes: GEMM 1024x4096x4096, decode batch 16 / S 2048, RMSNorm 4096x4096.
Persistent test builds use min(188, ceil(tiles/3)) CTAs so every CTA loops.

## Numerics caveat found while testing (sm_120, CUDA 12.9)

TileLang vectorizes paired fp32 arithmetic into `tl::mul2/sub2/add2`, which on
`__CUDA_ARCH__ >= 1000` emit `mul.rn.f32x2` / `sub.rn.f32x2`. On sm_120a (no native
f32x2 ALU) ptxas contracts such a mul/sub pair into an FFMA **despite the `.rn`
modifiers and even with `-fmad=false`** (minimal repro: two `tl::mul2` feeding
`tl::sub2` compiles to FMUL + FFMA). Which operand gets fused depends on the
surrounding code, so the "same" expression gives 1-3 ulp different results in
kernels that differ only in, e.g., pipeline depth. The decode tile therefore computes
`exp2((s - m) * scale)` instead of `exp2(s*scale - m*scale)`; a subtraction feeding a
multiply cannot be contracted. Any new tile body should avoid `x*a - y*b` / `x*a + y`
patterns on vectorized fp32 fragments (or use explicit `fma`) if bitwise E0 claims
matter. A TileLang-side fix would be to use scalar `__fmul_rn/__fsub_rn/__fadd_rn` in
`src/tl_templates/cuda/common.h` for sm_12x.

## CoKernel builder (`cotile/cokernel.py`, P1-M3a v0)

`build_cokernel(opA, shapeA, cfgA, opB, shapeB, cfgB, orch) -> KernelSpec` fuses two
op-library tile bodies into **one persistent TileLang kernel**. Nothing is op-specific:
roles are described only by the op protocol above. Tested with GEMM x GQA decode and
GEMM x RMSNorm (and A x A for overhead probes).

```python
from cotile.cokernel import Orch, build_cokernel, CoRunner, sm_role_table
spec = build_cokernel(gemm, gs, gc, gqa_decode, ds, dc,
                      Orch(binding="sm", schedule="dynamic", chunk=1, takeover=True,
                           num_ctas=188, threads=None, smem="alias", timing=True, debug=False))
compile_specs([spec])
run = CoRunner(spec)
run.set_knobs(sm_role=sm_role_table(94))   # runtime knob: SM split (no recompile)
(out_a, out_b) = run([inputs_a, inputs_b])  # dicts keyed by each op's own io names
run.stats()   # T_A_ns, T_B_ns, makespan_ns, exit_ns, done_A/B, ctas_A/B, ... (device-side)
```

**Kernel signature** = op A's `io_params` prefixed `a_`, op B's prefixed `b_`, plus
`co_knobs` int32[8] (runtime knobs: kA, kB, static strides), `co_state` int32[16+num_sms]
(queue heads, rank tickets, steal counters, done counters, per-role CTA counts, exit
counter, epoch, per-SM arrival counters), `co_tacc` int64[4] (timestamp accumulators),
`co_out` int64[16] (results of the launch), `co_sm_role` int32[num_sms] (SM binding),
`co_claim` int32[NA+NB] (static+takeover) and, in debug builds, `co_dbg` (per-tile
execution count), `co_dbg_sm` (executing SM) and `co_dbg_time` (tile start/end ns).
Pass configs: `tl.disable_warp_specialized` (plus each op's own).

**Dispatch loop.** Each CTA: thread 0 reads `%smid`, determines its role, then every
iteration of a `for it in T.serial(MAX)` loop (TileLang does loop-carried sync analysis
for `for`, not for `while`) thread 0 produces the next (role, tile) pair, writes it to a
double-buffered 4-int shared slot (`slot[(it%2)*2 ...]`), `__syncthreads()`, everyone
reads it, `break` on DONE, and runs the role's tile body. That one barrier per
iteration also makes cross-role scratch reuse safe (every thread has left the previous
tile). Thread-0 state (phase, rank, chunk cursor, counts) lives in registers.

| `Orch` field | meaning |
|---|---|
| `binding="sm"` | role = `co_sm_role[%smid]` (runtime table; `sm_role_table(n_a, order=contiguous\|interleave)`) |
| `binding="cta"` | POD-style: r = atomic_add(co_state[SMCTR+%smid], 1); role A iff r mod (kA+kB) < kA; kA/kB runtime (`set_knobs(ratio=(kA,kB))`) |
| `schedule="dynamic"` | per-role queue head; thread 0 grabs `chunk` tiles with `atomic_add(return_prev=True)`; `chunk` = int or (chunk_A, chunk_B) |
| `schedule="static"` | rank = per-role ticket (1 atomic per CTA), tiles rank, rank+n_r, ...; n_r = host-computed CTAs of role r (`expected_role_ctas`: assumes num_ctas/num_sms co-resident CTAs per SM, breadth-first placement, as measured by the smid probe); the kernel reports the actual count and `CoRunner.stats()` raises on mismatch |
| `takeover=True` | after its own role is exhausted a CTA continues with the other role: dynamic = grab from the other queue; static = steal from the back of the other role's range with epoch-tagged per-tile claims (`atomic_max(co_claim[t], epoch)`; owners claim each of their tiles, thieves stop at the first failed claim; exactly-once regardless of how far thieves get) |
| `threads` | common CTA size, default max of the roles. A smaller role runs under `if tx < n` (Rammer-style idle threads); TileLang rewrites its `__syncthreads` to `bar.sync 3, n` |
| `smem="alias"` | each role's tile call is wrapped in a `tl.shared_lifetime_scope` AttrStmt (core addition, below): per-CTA smem = max over roles (and within a role the grid-build reuse, e.g. GEMM's C staging over its A/B pipeline) instead of the sum. `"sum"`: TileLang's default plan, for comparison |
| `timing` | CTA barrier after each tile + `%globaltimer`; per-role end = max over CTAs of their last tile's completion |
| `debug` | per-tile exec counter / SM / start-end timeline |
| `min_blocks_per_sm` | `__launch_bounds__` 2nd argument (register cap for CTA binding co-residence) |

**Tile id bounds.** The dispatched tile id comes from shared memory, opaque to TVM's
analyzer; the dispatcher guards the call with `0 <= t < N` *and* passes
`min(max(t,0),N-1)` (`bounded()`): only the combination lets LegalizeSafeMemoryAccess
prove the bodies' loads in range (0/10 predicated cp.async, vs 4/10 with the clamp
alone and 10/10 with the guard alone) and lets the simplifier turn the GEMM grouped
rasterization divisions into shifts.

**Timestamps (K3).** `%smid`, `%nsmid`, `%globaltimer` and a u64 atomicMax come from a
`prelude` (`cotile_smid()` ... via `T.call_extern`). Kernel start = min over CTAs of
their first `%globaltimer` (atomicMax of the complement, so the zero-initialised
accumulator needs no host init); role end = atomicMax of each CTA's last tile completion
of that role (flushed when the CTA's phase for that role ends, together with its done
count). `stats()` gives T_A, T_B, makespan and the last CTA's exit time in ns. They agree
with CUDA-event times minus the ~3-5 us launch floor.

**Counters / repeated launches.** All counters are zeroed once at allocation. Every CTA
ends with `atomic_add(co_state[EXIT], 1, acq_rel)`; the last one copies the results to
`co_out`, zeroes every counter (incl. per-SM arrival counters and timestamp
accumulators) and bumps the epoch. No host re-zeroing between launches; after an aborted
launch call `CoRunner.reset_state()` (also zeroes the claim array). The ops' split-K/KV
counters self-reset as before.

**Core changes this needed** (co-tilelang, not upstream):
* `tl.shared_lifetime_scope` (`src/op/builtin.h`, `src/transform/merge_shared_memory_allocations.cc`):
  an AttrStmt declaring that no shared-memory value crosses its boundary. The merge
  planner attributes touches inside it to the scope's direct children (as it does for a
  kernel's top level), so the live range of a role's scratch ends with the tile call.
  Without it every buffer touched anywhere inside the dispatch loop is live for the whole
  loop (TileLang's default plan; StorageRewrite hoists all shared allocations to kernel
  scope and nested attach scopes are not supported), i.e. smem = sum of both roles, and
  GEMM x decode does not even launch (173 KB > 99 KB). Regression test:
  `testing/python/transform/test_tilelang_transform_shared_lifetime_scope.py`.
* `3rdparty/tvm/src/target/z3/z3_prover_on.cc`: `Z3Prover::CanProve` returns "not provable"
  when Z3 throws (it throws `canceled` when its rlimit is exhausted on some queries)
  instead of aborting LayoutInference/LowerTileOp with `InternalError: canceled`. Hit by
  GEMM swizzle-layout identities in CoKernel builds (e.g. gd_thr with takeover); whether
  a query exceeds the budget depends on the solver's accumulated context, so it is not
  reproducible in isolation. Only kernels that previously failed to compile are affected.

**Status (2026-09-22, research/results/2026-09-22_cokernel/).** GEMM(2048x4096x4096) x GQA decode(16 x
8192) and GEMM x RMSNorm(16384x4096), 7 config pairs (incl. split-K, split-KV, roles on a thread subset, 2 and
4 CTAs/SM), SM and CTA binding x static/dynamic x takeover on/off x 3–5 runtime splits/ratios: 184/184 settings
(552 launches) give outputs within tolerance **and bitwise equal to the solo persistent builds**, every tile runs
exactly once, and counters are clean after every launch. Per-CTA smem with `alias` is <= max(solo A, solo B)
(e.g. GEMM x decode 73216 B vs 173072 B for the default plan, which cannot launch). Overhead of a compiled-in but
idle partner: +2% (static GEMM) to +3.6% (decode) vs the solo persistent build at equal CTA count; registers rise to
about max(roles) + 10–40. Dynamic dispatch is no slower than static even for 2 µs RMSNorm tiles (4 CTAs/SM).

**Generated code.** `__launch_bounds__(threads, min_blocks_per_sm)`; one `__syncthreads()` per dispatch iteration
(+1 with timestamps) in front of each role body's own leading barrier. The partial-thread role's barriers become
`bar.sync 3, n`; the decode's `T.reduce_*` keeps `NamedBarrier<n>` ids 1/2 (safe here: one role per CTA at a time).
Single 1024-aligned `buf_dyn_shmem` arena: slot at offset 0, role scratch overlaid from 1024. The GEMM TMA-store
epilogue waits for the bulk store (`tma_store_wait<0,true>`) before its barrier, so aliasing is safe.

**Running.**
```bash
python -m cotile.tests.test_cokernel [--pairs gd_e0,gr_small] [--no-gpu] [--out DIR]
python -m cotile.tests.probe_cokernel_overhead [--out DIR]           # K5 overhead probe (cobench flush mode)
python testing/python/transform/test_tilelang_transform_shared_lifetime_scope.py   # needs pytest
```

**Known limits / open problems (v0).**
* Static schedule assumes num_ctas/num_sms co-resident CTAs per SM (breadth-first placement); violations are
  detected (`stats()` raises), not repaired. It cannot coexist with foreign kernels taking SMs (e.g. M1 baselines).
* Static + takeover needs one claim atomic per owner tile (not atomic-free).
* In flush mode the static schedule of large kernels is 20–30% slower than dynamic for reasons not yet understood
  (see the results README); the same holds for the op library's static `build_persistent`.
* Role bodies use absolute `threadIdx` (the op protocol), so a role can only occupy threads [0, n): fine for SM/CTA
  binding, not for warp-level binding (needs a thread-offset argument in `make_tile_body`).
* No prefetch of the next dynamic grab (atomic latency is exposed at 1 CTA/SM for tiny tiles).
