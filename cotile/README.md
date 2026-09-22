# cotile — tile-program library for CoKernels (v0)

`cotile` sits on top of TileLang (this repo's `tilelang/`) and will host the CoKernel
compiler (research/plan.md §3, §8). v0 contains the single-op pieces a two-role
CoKernel is built from:

| module | contents |
|---|---|
| `cotile/ops/gemm.py` | GEMM `C = A @ B^T` (NT), bf16 in/out, fp32 accumulate, split-K with in-kernel combine |
| `cotile/ops/gqa_decode.py` | GQA decode attention (Hq=32, Hkv=8, D=128 by default), split-KV with in-kernel combine |
| `cotile/ops/rmsnorm.py` | RMSNorm `Y = X * rsqrt(mean(X^2) + eps) * W`, explicit thread/vector mapping |
| `cotile/kernel.py` | op protocol types, generic `build_grid` / `build_persistent`, `compile_specs` (par_compile), `Runner` |
| `cotile/resources.py` | resource signature of a *compiled* kernel + occupancy model + driver cross-check |
| `cotile/device.py` | `DeviceSpec` for the RTX PRO 6000 Blackwell (sm_120) |
| `cotile/tests/` | correctness / bitwise / repeated-launch tests and signature export |

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
