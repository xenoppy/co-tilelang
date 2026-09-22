# TileLang capability survey for a CoKernel prototype (sm_120)

> Read-only code survey, 2026-09-22, repo `co-tilelang` @ `03c613bd` (branch `explore`).
> Nothing was built or run. Scope: what TileLang already supports for a single CUDA kernel in which two
> independent operations (e.g. GEMM and GQA decode attention) run as two "roles" with tile-level dispatch
> (proposal §5.3, plan §3.3).
>
> Legend: **[V]** verified by reading the code (file:line cited); **[I]** inferred from code, not executed;
> **[T]** needs a compile/run test to confirm. Q6/Q7 were partly surveyed by a helper agent; I spot-checked
> its key claims (marked [V] only where I re-read the code myself).

---

## 0. TL;DR

| Capability | Status | Where | Conf. |
|---|---|---|---|
| Role dispatch `if role==0: <tile body> else: <tile body>` with runtime `role` | Works structurally: layout inference, pipelining, sync insertion all treat a block-uniform runtime `if` correctly | `layout_inference.cc:800-805`, `pipeline_planning.cc:1083`, `thread_storage_sync.cc:1036-1090` | [V]/[I] |
| `T.Pipelined` nested under `for`/`if` | Supported (persistent GEMM example) | `examples/gemm/example_gemm_persistent.py:57-81` | [V] |
| `T.Pipelined` nested under `while`, runtime trip count | Works on the non-warp-specialized path (stream-K example), **but the example is only tested on cc ≤ 8.9** | `examples/gemm_streamk/*` | [V]/[T] |
| **Auto warp specialization on sm_120** | **ON by default** (sm_120 counts as TMA-capable). It rewrites only the *first* pipelined loop, adds 128 producer threads to the whole kernel, cannot handle a `while` ancestor, and otherwise strips all pipelining. **Must be disabled for CoKernels**: `tl.disable_warp_specialized=True` | `tilelang/cuda/pipeline.py:27-36,105-107`; `producer_consumer_ws.cc` | [V]/[I] |
| `T.gemm` path on sm_120 | `cuda.mma` (ldmatrix + `mma.sync`); WGMMA only for 90≤arch<100, tcgen05 only for 100≤arch≤110 | `src/cuda/op/gemm.cc:80-99,337-389` | [V] |
| `T.atomic_add(x[i], v, return_prev=True)` | Exists for **scalar element** destinations (int32/uint32/int64/f32/f16/bf16); raises on a whole buffer/region | `tilelang/language/atomic.py:211-312` | [V] |
| `__syncthreads` / named barriers from the DSL | `T.sync_threads()`, `T.sync_threads(id, count)`, `T.named_barrier_arrive(id, count)`, `T.syncthreads_or/and/count` | `tilelang/language/builtin.py:964-1000,1128-1150` | [V] |
| `%smid`, `%nsmid`, `%globaltimer` | **Not available.** Zero-C++-change route: `T.Kernel(..., prelude=<inline-asm helpers>)` + `T.call_extern` | `tilelang/language/kernel.py:277-338` | [V] |
| Persistent kernels | `T.Persistent` (static grid-stride), `T.PersistentTileScheduler` (+`while sched.valid()`), hand-written `for w`/`while`, `T.loop_break()` | `src/ir.cc:146-222`, `tilelang/language/tile_schedule.py` | [V] |
| `T.ws(i)` warp-group regions | `if 128*i <= tid < 128*(i+1)`, plus an attribute. Tile ops inside use the thread sub-range (gemm, parallel, copy, reduce all offset-aware) | `src/ir.cc:351-389` | [V] |
| Named-barrier allocation | Fixed IDs: 0 = `__syncthreads`, **1/2 = hard-coded in every cross-warp `T.reduce`**, 3+ = one per distinct partial thread range. No check against the 16-barrier limit | `thread_sync_types.h:27-32`, `thread_storage_sync.cc:343-357`, `tl_templates/cuda/reduce.h:205-208,277-300` | [V] |
| Cross-role shared-memory reuse | **Not in the default mode.** Allocations at kernel scope that are touched inside the role `if`/`while` stay live over the whole construct, so smem = **sum** of both roles. Aggressive merge can reuse, but is unproven under `while` | `merge_shared_memory_allocations.cc:100-110,160-276` | [I] |
| Register cap | `T.annotate_min_blocks_per_sm(k)` gives `__launch_bounds__(threads, k)`. No `__maxnreg__`. `-maxrregcount` via compile flags is likely overridden by `__launch_bounds__` (check with ptxas -v). `setmaxnreg` intrinsics exist (warp-level only) | `annotations.py:85-104`, `codegen_cuda.cc:628-645` | [V]/[I] |
| Query registers/smem of a compiled kernel | No CUDA API in TileLang (`n_regs` is HIP-only). Use `TILELANG_VERBOSE` (logs ptxas -v on a cache miss) or `cuobjdump --dump-resource-usage` on the cached cubin | helper report, `tilelang/cuda/backend.py:97-104` | [V] |
| Many-config compilation | `tilelang.par_compile` (thread pool), `AutoTuner.run(enable_grouped_compile=True)` (one nvcc call per group of configs), 2-level cache | `tilelang/jit/__init__.py:174`, `tilelang/autotuner/grouped_compile.py:30` | [V] |
| Existing horizontal fusion / multi-role kernel | None. Closest: two-phase persistent MLA decode with `T.sync_grid()`; the two-warpgroup `if tx < 128 … else …` FlashMLA example (Hopper-only) | `examples/deepseek_mla/example_mla_decode_persistent.py`, `examples/warp_specialize/example_warp_specialize_flashmla.py` | [V] |

---

## 1. Role dispatch inside one `T.Kernel`

### 1.1 The lowering pipeline order (CUDA)

`tilelang/cuda/pipeline.py` [V]:

```
68  CUDAPassPipelineBodyPrologue:
105   if allow_warp_specialized(target):             # sm_90+ incl. sm_120, unless tl.disable_warp_specialized
106       MaterializeWSSchedule; ProducerConsumerWarpSpecialized
116   IfStmtBinding; UnrollLoop; Simplify
127   PipelinePlanning; InjectSoftwarePipeline        # BEFORE layout inference
132   LayoutInference; ReducerPlanAndMaterialize; LowerTileOp
...
249 MergeSharedMemoryAllocations                      # after SplitHostDevice
256 ThreadSync("shared"); ThreadSync("shared.dyn")    # includes ThreadPartialSyncRewriter
269 AnnotateWarpGroupRegAlloc (if WS allowed) ... PersistThreadblock
```

`allow_warp_specialized` returns `not tl.disable_warp_specialized` when `have_tma(target)` holds (`pipeline.py:27-36`). `have_tma` is `major >= 9` (`tilelang/contrib/nvcc.py:634-649`), so **sm_120 is TMA-capable here and auto-WS is on by default**. `TargetHasBulkCopy` is `arch >= 90` (`src/cuda/target_utils.cc:117-122`). The regression test `testing/python/issue/test_tilelang_issue_sm120_tma_smem_alignment.py` confirms that a plain pipelined GEMM on sm_120 lowers to `tl.tma_load` [V].

### 1.2 Layout inference with divergent branches [V]

- Each tile op records `thread_bounds = ComputeThreadBounds(thread_binding_, analyzer_)` at its collection point (`layout_inference.cc:800-805, 1098-1100`). That is the `const_int_bound` of `threadIdx.x` under the constraints of the enclosing `if`s (`src/transform/common/pipeline_utils.h:121-134`).
- A condition such as `role == 0` does not constrain `threadIdx.x`, so each role's ops get the full CTA range [0, threads).
- `T.ws(1)` and `if tx >= 128` give [128, 256).
- The gemm uses `block_size = thread_bounds->extent` (`src/op/gemm.cc:283-289`). Emitter math is relative to `thread_bounds.min`: `local_thread_var = thread_index - thread_bounds.min` (`tilelang/cuda/op/gemm/gemm_mma.py:75-80`).
- **Caveat [I]:** only *contiguous* thread ranges infer correctly. `T.ws(0, 2)` builds `(tx<128) || (256<=tx<384)`, whose const-int bound is [0, 384). Layouts would be computed for 384 threads while only 256 participate.
- **Caveat:** do not share one fragment buffer between the two roles. It gets exactly one layout, so a single fragment layout cannot serve two thread ranges. `producer_consumer_ws.cc:2031-2036` makes the same point for WS partitions.

### 1.3 Pipeline planning / software pipelining [V]

- `PipelinePlanner::VisitStmt_(const ForNode*)` (`src/transform/pipeline_planning.cc:1083-1180`) fires on *any* `For` carrying `num_stages`, wherever it sits (under `if`, `for` or `while`). There is no "top-level only" restriction.
- `InjectSoftwarePipeline` is likewise per loop (`inject_pipeline.cc`).
- Evidence that this works in practice:
  - `examples/gemm/example_gemm_persistent.py:68-81`: `for w in T.serial(waves): if …: for k in T.Pipelined(...)`.
  - `examples/gemm_streamk/example_tilelang_gemm_streamk.py:99-121`: `while start_iter[0] < last_iter:` containing `T.Pipelined(end_iter[0] - start_iter[0], …)`, i.e. a runtime trip count. Its test is restricted to `requires_cuda_compute_version_le(8, 9)` with the comment "not fully supported on sm90". That matches the WS problem in §1.5.
- On the non-WS path an implicit `T.copy(global → shared)` inside a pipelined loop becomes `cp.async`:
  - `SelectCopyInstForLowering` never auto-selects a TMA *load*: `SelectTmaInst(facts, /*allow_load=*/false, /*allow_store=*/true, …)` (`src/cuda/op/copy_analysis.cc:795-801`).
  - Auto TMA *stores* (shared → global) are still selected unless `tl.disable_tma_lower`.
- Each role's shared buffers are multi-versioned by `num_stages`.

### 1.4 `IfStmtBinding` rewrites else-less ifs [V]

`src/transform/if_stmt_binding.cc:345-356`: an `if c: {s1; s2; s3}` *without* an else becomes `if c: s1; if c: s2; if c: s3`. If the body writes a buffer the condition reads, the condition is snapshotted into a `Bind` first (`GuardStmts`, `:195-219`). Harmless for role dispatch. Writing `if role == 0: … else: …` (with an else) avoids it entirely.

### 1.5 Warp-specialization / producer-consumer passes vs. two roles [V + I]

`ProducerConsumerWarpSpecialized` (`src/cuda/transform/producer_consumer_ws.cc`):

- **Documented limits** (`:16-20`): pure TMA pipelines only; "No conditionally guarded loop bodies"; "**Single pipelined loop per block**".
- **Skip conditions** (`:2840-2860`):
  - pass config `tl.disable_warp_specialized`;
  - `ManualWSDetector` finds a `"warp_specialize"` attribute, which `T.ws` emits (`:2704-2725`);
  - the target has no bulk copy;
  - no pipelined loop contains a TMA-eligible copy (`TiledWSCandidate`, `:2745-2786`).
- **Which loop it rewrites:**
  - `PipelineLoopFinder` returns the **first** `num_stages` loop only (`:2185-2229`).
  - The other role's pipelined loop is left for normal pipelining.
- **Thread count** (`:1433-1434`, `:2176` in `BuildWSBlock`):
  - `producer_extent = 128`, `consumer_extent = original threads`.
  - The *kernel-wide* `threadIdx.x` extent becomes consumer + 128, e.g. 256 → 384.
  - Code after the loop is guarded `tid >= 128` and thread-index-remapped (`GuardConsumerOnly`, `:2555-2562`).
  - **The other branch of an enclosing `if/else` is not remapped** (`PipelineLoopInStmtReplacer::VisitStmt_(IfThenElse)`, `:2521-2537`). Role B would then run with 384 threads and have its layouts inferred for 384 [I].
- **`while` ancestors:**
  - The replacer handles `SeqStmt/Block/Attr/If/For` but has no `WhileNode` case; `VisitStmtDefault_` returns empty (`:2550-2552`).
  - A TMA-eligible pipelined loop under a `while` therefore most likely trips `ICHECK(replaced.defined()) << "ProducerConsumerWS: failed to replace pipeline loop"` (`:2027-2028`) [I, T].
- **Fallback when the rewriter doesn't fire** (`:2869-2896`): it **strips `num_stages` from every loop in the function**, which silently disables software pipelining for both roles.

⇒ For CoKernels on sm_120 use `pass_configs={"tl.disable_warp_specialized": True}`. The GQA decode example already does this (`examples/flash_decoding/example_gqa_decode.py:44-46`, "TODO(lei): fix warp specialized pass"), as do many others. This gives the sm_80/89-style cp.async + mma.sync pipeline.

- **Side note for baselines:** the default solo GEMM on sm_120 is TMA + auto-WS. The numerics are the same as the cp.async path (same mma sequence), but performance may differ. The P1 "solo-best" set should include both variants.

`MaterializeWSSchedule` (`src/cuda/transform/materialize_ws_schedule.cc:1-60`, `tilelang/language/ws_schedule.py`) is a separate, user-driven WS framework:
- `WSRole(name, warps_lo, warps_hi, max_nreg)` covers *arbitrary contiguous warp ranges* with per-role `setmaxnreg`.
- `WSPipeline` defines mbarrier pipelines.
- It can drive `T.ws_op`-wrapped `while` loops.
- It is producer/consumer-oriented and exercised only by SM100 examples (`examples/aws/*`, tests restricted to cc 10.x). It is interesting later for warp-level binding with register partitioning, but not a first-prototype path.

### 1.6 Shared-memory allocation merging across roles [I, medium-high confidence]

`MergeSharedMemoryAllocations` (`src/transform/merge_shared_memory_allocations.cc`) merges all dynamic shared memory into one `buf_dyn_shmem` and plans reuse with a linear liveness scan:

- Design comment (`:100-110`): "This pass tries to detect last point that we need to keep memory alive under the same scope as Allocate… The free point is only inserted at the same scope of Allocate."
- `If`, `While` and `For` each open a scope (`VisitNewScope`, `:253-276, 295, 309-319`). In the default (non-aggressive) mode an access is attributed to the scope entry at the *allocation's* level (`scope_[it->second.level]` / `scope_[access_level]`, `:170-176, 220-228, 244-250`).
- **Consequence for CoKernels** (all `T.alloc_shared` sit at kernel scope):
  - Every buffer touched anywhere inside `if role…` or `while …` is "touched" by that whole construct.
  - Role-A and role-B buffers are therefore live simultaneously, and **smem = sum of both roles**, not the max.
  - This hurts every binding level. For example, a 64 KB GEMM plus a 48 KB attention role gives 112 KB, which exceeds sm_120's 99 KB per CTA.
- **Aggressive mode** (`tl.enable_aggressive_shared_memory_merge=True`, default False; `tilelang/backend/pass_pipeline/pipeline_utils.py:39-43`):
  - It attributes accesses to the innermost statement (`:170-173`).
  - It only lifts kill points out of `For` scopes (`:1160-1240`); `scope_level_` increments only for `For` (`:309-315`), not for `While` or `If`.
  - So then/else buffers can share memory, which is correct for mutually exclusive branches.
  - But the back edge of a `while` is not modelled. A buffer read at the top of iteration i+1 and written at the bottom of iteration i (e.g. prefetching the next tile id) could be mis-planned.
  - Needs a dedicated test. Inspect the plan with `tl.debug_merge_shared_memory_allocations=True`.
- **Sibling `T.ws` regions** (warp-level binding):
  - The `"warp_specialize"` attribute from `T.ws` is ignored by the planner. Only the compiler-generated `kWarpSpecializationScope` gets special treatment (`:288-289`).
  - Two sibling `with T.ws(0): A` / `with T.ws(1): B` blocks look like two *sequential* ifs. In the default mode A-only buffers die at the end of `if_A`, so B-only buffers **may be overlaid on them even though the warps run concurrently**.
  - `ThreadSync` would then see an overlap conflict and insert a CTA-wide barrier between the two regions. That either serializes the roles or is a race.
  - Mitigations:
    - use a single `if tx < split: A else: B` (one If scope, so both live);
    - or set `tl.disable_shared_memory_reuse=True`;
    - verify with the debug flag.
- **Manual aliasing:** `T.view(src, shape, dtype)` / `T.reshape` create an alias with the same `data` var, but require identical total bits and no offset (`tilelang/language/customize.py:60-90`). There is no first-class "union" of role buffers.

### 1.7 Thread-sync insertion with a runtime role [V]

`TileLangThreadSyncPlanner` (`src/transform/thread_storage_sync.cc`):

- **If-handling** (`:1036-1090`) keeps syncs inside an `if` in place when the condition is *block-uniform*.
  - `ConditionThreadPropertyChecker` (`:467-557`) marks a condition non-uniform only if it references `threadIdx`.
  - A `BufferLoad` (e.g. `role_s[0] == 0`, or an `alloc_var` holding it) is "runtime but block-uniform".
  - A `call_extern` of `%smid` is not flagged at all.
  - So `__syncthreads()` stays inside the role branch. This is legal because the branch is CTA-uniform.
- **`While` loops get no loop-carried analysis:** `Summarize(std::move(scope_.back()), nullptr)` at `:1123`, whereas `For` passes `op` at `:933`.
  - In a dynamic-queue `while` loop, a write by thread 0 at iteration i+1 to the broadcast slot is not guarded against slower threads still reading the iteration-i value.
  - Tile bodies usually contain CTA barriers that hide this, but that is not guaranteed.
  - Fixes: an explicit `T.sync_threads()`; a double-buffered slot (`slot[it % 2]`); or a `for it in T.serial(MAX)` + `T.loop_break()` loop, which does get loop-carried analysis.

### 1.8 Other gotchas [V]

- **`T.use_swizzle(...)`** emits `const dim3 blockIdx = tl::rasterization2DRow<…>();` and shadows `blockIdx` for the whole kernel body (`src/cuda/codegen/codegen_cuda.cc:5055-5086`; `tilelang/language/annotations.py:29-41`). Do not use it in a CoKernel. Rasterize manually per role, e.g. `T.PersistentTileScheduler(..., swizzle_size=8, stateful=False).coord(tile_id)` (`tilelang/language/tile_schedule.py:253-275`).
- **`LoopUnswitching`** refuses to unswitch on `threadIdx`-dependent predicates (`src/transform/loop_unswitching.cc:358-362`), so `if tx == 0:` inside a persistent loop is safe. It may unswitch a loop-invariant `role` condition out of a `for`, which is harmless.
- **Multiple `T.Kernel` in one `prim_func`** gives several *sequential* launches, e.g. GQA decode split + combine (`examples/flash_decoding/example_gqa_decode.py:75,161`). This is not horizontal fusion.

### 1.9 Which gemm on sm_120 [V]

`cuda::Gemm::SelectInst` (`src/cuda/op/gemm.cc:337-389`):

- It returns `kCudaTCGEN05` only if `AllowTcgen5Mma`, which requires `TargetIsSm100`, i.e. 100 ≤ arch ≤ 110 (`:80-89`; `target_utils.cc:71-76`).
- It returns `kCudaWGMMA` only if `AllowWgmma`, which requires `TargetIsHopper`, i.e. 90 ≤ arch < 100, and `num_warps % 4 == 0` (`:91-99`).
- Otherwise, on sm_120, it returns **`kCudaMMA`**. That lowers via `GemmMMA` + `TensorCoreIntrinEmitter`: ldmatrix + `mma.sync`, no named barriers.
- The warp partition comes from `ComputeDefaultWarpPartition` (`:170-245`): each warp owns multiples of 16 rows × 8 cols.
- `T.wgmma_gemm` / `T.tcgen05_gemm` are FATAL on sm_120.
- sm_120-only extra: `T.mma_gemm_blockscaled` (NVFP4/MX block-scaled mma, `kCudaMMABlockScaled`, `:363-375`).
- Kernels compile with `-arch=sm_120a` (`nvcc.py:523-531`, `backend.py:62-68`), so arch-specific PTX is available.

---

## 2. Persistent kernels and dynamic tile loops [V]

- **`T.Persistent(domain, wave_size, index, group_size=8)`** (`tilelang/language/loop.py:90-109`, C++ `src/ir.cc:146-222`):
  - It emits `for w in range(ceildiv(padded, wave_size))`, decodes grouped coordinates, and guards with `if in_range`.
  - It adds `loop_break` only when `waves >= 2` is provable (`ir.cc:201`).
  - A runtime `wave_size` also works, because the `in_range` guard alone is sufficient [I].
  - Used in `examples/gemm/example_gemm_persistent.py:51-66`.
- **`T.PersistentTileScheduler(m_tiles, n_tiles, num_workers=None, swizzle_size=1, column_major=True, cluster_size=1, stateful=True)`** (`tilelang/language/tile_schedule.py:92-290`):
  - Pattern: `sched.init(block_id); while sched.valid(): …; sched.next_tile()`. State lives in `T.alloc_var` registers.
  - `stateful=False` gives a pure `coord(tile_id)` decoder, useful with atomic tile ids.
- **Runtime trip counts:**
  - `T.serial(expr)` / `T.Pipelined(expr)` accept `PrimExpr` extents (stream-K: `T.Pipelined(end_iter[0] - start_iter[0], …)`).
  - `while cond:` is supported in both frontends:
    - lazy: `tilelang/language/parser/parser.py:201-216` → `T.While`;
    - eager `@tilelang.jit`: `tilelang/language/eager/builder.py:504-521`, which **rejects a constant-true condition as an infinite loop**, so use a flag variable.
  - `break`/`continue` are supported in eager mode (`builder.py:492-502`), and so is `T.loop_break()` (`customize.py:93-100` → `break;`, `codegen_cuda.cc:2962-2964`).
- **Stream-K:** `examples/gemm_streamk/example_tilelang_gemm_streamk.py:85-143` has a runtime `while`, a runtime-extent `T.Pipelined`, and `T.atomic_add` partial-tile fix-up. Its test is cc ≤ 8.9 only (see §1.3/§1.5).
- **Two-phase persistent kernel:** `examples/deepseek_mla/example_mla_decode_persistent.py:36-133`:
  - A persistent split-KV decode (`for w: if valid: T.Pipelined …`), then `T.sync_grid()`, then the combine phase in the same kernel.
  - The cooperative launch is set automatically (`src/cuda/transform/persist_threadblock.cc`).
- **Cluster Launch Control:** `T.clc_try_cancel` etc. exist (`tilelang/cuda/language/cluster.py:52-100`). The templates are gated on `CUTLASS_ARCH_CLC_ENABLED`, which CUTLASS defines for SM120A too (`3rdparty/cutlass/include/cutlass/arch/config.h:219-225`). That suggests hardware work-stealing may be available on sm_120a [I, T]. It is only used in SM100 examples. Not needed for v0.

---

## 3. Atomics and CTA broadcast [V]

- **`T.atomic_add(dst, value, memory_order=None, return_prev=False, use_tma=False, annotations=None)`** (`tilelang/language/atomic.py:211-312`):
  - Thread-level path (`:267-280`): taken when neither argument has inferable extents, e.g. `ctr[i]` (a scalar `BufferLoad` with non-Ramp indices returns `None` from `get_extent`, `tilelang/language/utils.py:111-172`).
  - With `return_prev=True` it emits `tl.atomic_add_ret_elem_op` with result dtype `dst.dtype`. Codegen: `AtomicAddRet(ptr, val[, order])` (`codegen_cuda.cc:4896-4906`).
  - **Passing the whole buffer `ctr`, even of shape (1,), takes the tile-region path and raises `NotImplementedError("return_prev is not supported for tile-region-based atomic operations")` (`:303-304`).** Always index the element.
  - Device template (`src/tl_templates/cuda/atomic.h:516-560`):
    - int32/uint32/float: `cuda::atomic_ref<T, thread_scope_device>::fetch_add` (default `memory_order_relaxed`);
    - int64: via `normalize_atomic_type` → `unsigned long long`;
    - half/bf16: `atomicAdd` or a CAS helper.
  - Memory orders: `"relaxed"|"consume"|"acquire"|"release"|"acq_rel"|"seq_cst"` (`atomic.py:13-20`).
- **Related:**
  - `T.atomic_load(src, memory_order)` / `T.atomic_store(dst, v, memory_order)` (`:391-493`);
  - `T.atomic_max/min(..., return_prev=…)`, scalar only;
  - `T.atomic_or` (`:496-508`).
  - `ThreadSync` treats atomic destination pointers specially and does not insert barriers between atomics (`thread_storage_sync.cc:1175-1195`).
- **Syncs:**
  - `T.sync_threads()` → `tvm_storage_sync("shared")` → `__syncthreads()`. Inside a partial-thread region it is automatically rewritten to `bar.sync id, n`.
  - `T.sync_threads(id, n)` → `tl::__sync_thread_partial(id, n)` = `bar.sync id, n` (`builtin.py:964-971`; `codegen_cuda.cc:1559-1593`; `common.h:876-878`).
  - `T.named_barrier_arrive(id, n)` → `bar.arrive` (`builtin.py:974-1000`).
  - `T.syncthreads_or/and/count(pred)` (`:1128-1150`).
- **Broadcast from thread 0** (no special primitive needed; `ThreadSync` inserts the RAW barrier automatically [V]; the WAR across `while` iterations must be handled manually, §1.7):

```python
slot = T.alloc_shared((2,), T.int32, scope="shared")   # static smem, outside the merged dyn arena
tile = T.alloc_var(T.int32)
tx = T.get_thread_binding()
...
if tx == 0:
    slot[it % 2] = T.atomic_add(queue_ctr[role], 1, return_prev=True)
T.sync_threads()            # explicit, so correctness does not depend on the planner
tile = slot[it % 2]         # block-uniform from here on
```

- **Counters:** pass them as explicit `int32` tensors and zero them on the host per launch (inside the CUDA graph), or have the last CTA reset them.
  - `T.alloc_global` workspaces come from `cudaMalloc` and are not zeroed (`allocate.py:171-191`).

---

## 4. Special registers / inline PTX

**Present [V]:**
- Nothing reads `%smid`, `%nsmid` or `%globaltimer` in `tilelang/`, `src/` or `examples/`.
- The only hit is the external IKET profiler macro (`tilelang/tools/cuda/iket/codegen.py:170-185`, `mov.u32 t, %globaltimer_lo`), which is not a DSL primitive.

**Mechanisms available [V]:**
1. **`T.Kernel(*grid, threads=…, prelude=SRC)`** (`tilelang/language/kernel.py:277-338`) or `T.import_source(SRC)` (`tilelang/language/common.py:173-177`):
   - Sets `pragma_import_c`. `CodeGenC::VisitStmt_(AttrStmt)` appends it to `decl_stream` (`3rdparty/tvm/src/target/source/codegen_c.cc:1120-1124`). The CUDA codegen falls through to it (`codegen_cuda.cc:5094`).
   - Used by `examples/dequantize_gemm/example_dequant_gemm_fine_grained.py:244`.
   - Note: the prelude text lands *before* the `#include <tl_templates…>` lines, which are appended in `Finish()` (`codegen_cuda.cc:668-750`). Keep the helpers self-contained (builtin types + inline asm only) and `#ifndef`-guarded in case two roles import them [I].
2. **`T.call_extern(dtype, "fn", *args)`** → `fn(args)`. `call_extern` is `CallEffectKind::kOpaque` (`3rdparty/tvm/src/tirx/op/builtin.cc:139-140`), so it is never CSE'd or hoisted.
3. **A proper intrinsic** follows the `get_lane_idx` pattern:
   - `TIR_DEFINE_TL_BUILTIN(get_lane_idx)` (`src/cuda/op/builtin.cc:293-296`);
   - a codegen case (`codegen_cuda.cc:4562-4570`);
   - `TL_DEVICE int get_lane_idx()` in `src/tl_templates/cuda/intrin.h:44-47`;
   - a Python wrapper in `tilelang/language/builtin.py:572-596`.
   - This needs a C++ rebuild.

**Least-invasive path (zero C++ changes)** [I; syntax to be confirmed by a compile test]:

```python
COTL_SREG = r"""
#ifndef COTL_SREG_H
#define COTL_SREG_H
__device__ __forceinline__ int cotl_smid()  { unsigned r; asm volatile("mov.u32 %0, %%smid;"  : "=r"(r)); return (int)r; }
__device__ __forceinline__ int cotl_nsmid() { unsigned r; asm volatile("mov.u32 %0, %%nsmid;" : "=r"(r)); return (int)r; }
__device__ __forceinline__ unsigned long long cotl_globaltimer() {
  unsigned long long r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r; }
#endif
"""
with T.Kernel(num_ctas, threads=256, prelude=COTL_SREG) as bid:
    smid = T.call_extern("int32", "cotl_smid")
    t0   = T.call_extern("uint64", "cotl_globaltimer")
```

**Stable path:**
- Put `tl::get_smid()/get_nsmid()/globaltimer()` in `src/tl_templates/cuda/common.h`, which is included transitively on every kernel, and add Python wrappers that call `T.call_extern(…, "tl::get_smid")`.
- Header-only, so no C++ rebuild.
- Caveat: the kernel cache and `CUDABinaryCache` keys do not hash template headers, so clear `~/.tilelang/cache` after editing headers [I].
- Promote to a TIR builtin later if needed.

**PTX semantics to respect** (from the PTX ISA, not the repo):
- `%smid` can change after preemption and is documented as a hint for profiling.
- SM IDs need not be contiguous, and `%nsmid` can exceed the physical SM count. Read it once per CTA, broadcast it, and remap it through a host-built table (plan §1.5).
- `%globaltimer` resolution is platform-dependent; measure it in P0.

---

## 5. Warp-level roles, `T.ws`, named barriers

### 5.1 `T.ws` [V]

- `T.ws(*warp_group_ids)` (`tilelang/cuda/language/warpgroup.py:19-59`):
  - Linear `tid` computed from tx/ty/tz, warp-group size fixed at 128.
  - The C++ `WarpSpecialize` (`src/ir.cc:351-389`) merges consecutive groups into `(tid >= lo*128 && tid < hi*128)` terms OR'ed together, then emits `If(cond) → Then → AttrStmt(0, "warp_specialize", 1)`.
  - There is no else-branch form. For finer ranges or an if/else split, write `if tx < k: … else: …` directly, as `example_warp_specialize_flashmla.py:111-123` does.
  - The attribute's only consumer is `ManualWSDetector`, which disables auto-WS. Every other pass sees a plain `if`.

### 5.2 `ThreadPartialSyncRewriter` [V]

`src/transform/thread_storage_sync.cc:263-435`:
- **Scope:** it runs on every `tvm_storage_sync("shared"|"shared.dyn")` with exactly one argument (`:283-293`). User syncs with explicit IDs are left alone.
- **Full-range syncs:** if the const-int bounds of tx/ty/tz equal the full launch extent, it stays `__syncthreads()` (`:296-311`).
- **Partial ranges:**
  - The thread count is computed by Z3 model counting (`CalculateThreadExtent`, `:372-390`).
  - Barrier IDs are allocated per distinct bound key: `id = barrier_id_map_.size() + kFirstUsedBarrier` (`:343-357`).
  - It emits `bar.sync id, count`.
  - A count that is not a multiple of 32 → `LOG(FATAL)` (`:321-331`).
- **Reserved IDs** (`src/transform/common/thread_sync_types.h:27-32`): `kSyncThreads = 0, kReduce_0 = 1, kReduce_1 = 2, kFirstUsedBarrier = 3`.
- **16-barrier limit:** no check anywhere. More than 13 distinct partial ranges in one kernel produce `bar.sync` IDs ≥ 16, passed through a register (`"r"(barrier_id)`), which is undefined [I]. Two roles use 2 IDs, so this is fine in practice.

### 5.3 Who else uses named barriers [V]

- **`T.reduce` / `T.finalize_reducer` / `T.reduce_*` crossing warps:**
  - They lower to `tl::AllReduce<…, thread_offset, tl::NamedBarrier<all_threads>>` (`src/cuda/op/reduce.cc:62-91`, `src/cuda/op/finalize_reducer.cc:30-47`).
  - `NamedBarrier::sync<phase>()` is `bar.sync phase, all_threads` with **phase hard-coded to 1 and 2** (`src/tl_templates/cuda/reduce.h:205-208, 277-280, 295-300`).
  - `all_threads` = `thread_bounds.extent`; `thread_offset` = `thread_bounds.min` (`src/backend/common/op/reduce.h:1008-1034, 1201-1222`).
  - **⇒ If two warp-level roles both run a cross-warp reduction at the same time, both use `bar.sync 1, 128` and `bar.sync 2, 128`. Arrivals from different roles mix → broken synchronization [I from code; T].**
  - Reductions that stay within a warp (common for MMA-layout row reductions over N) emit no barrier.
  - At SM/CTA-level binding (one role per CTA at a time) there is no conflict.
- **`T.gemm` on the mma.sync path:** no barriers (`gemm_mma.py` lowering).
- **Auto-WS / manual pipelines:** use **mbarriers** (`T.alloc_barrier`, sm_90+, allowed on sm_120: `have_mbarrier` = `major >= 9`, `nvcc.py:668-678`), not named barrier IDs.
- **User `T.sync_threads(id, n)` / `T.named_barrier_arrive(id, n)`:** no allocator. Pick IDs ≥ 3 + (number of auto-allocated IDs), or better 15 downwards.

### 5.4 Tile ops on a warp-group subset [V code, T on sm_120]

- **`T.gemm`:** offset-aware (§1.2). Used inside `T.ws(0)` in `examples/warp_specialize/*.py`. Those tests run only on cc 9.0 (`test_example_warp_specialize.py`), with `T.tma_copy` + mbarriers, both available on sm_120.
- **`T.Parallel` / SIMT `T.copy`:** layouts use `BindThreadRange(thread_bounds)`, and loop partition normalizes `thread_index - ThreadRange()->min` (`src/op/parallel.cc:398-407, 574-606`; `src/transform/loop_partition.cc:125`).
- **`T.reduce`:** `thread_offset = thread_bounds->min` (above).
- **Evidence of all of these inside concurrent `if tx < 128 / else` roles:** `examples/warp_specialize/example_warp_specialize_flashmla.py:123-300` (reduce_max/sum, T.Parallel, T.copy). It uses `T.wgmma_gemm`, so it is Hopper-only, and its test is disabled as "non-deterministic on H20".
- **Contiguous ranges only** (§1.2).
- **`setmaxnreg`:**
  - Available via `T.set_max_nreg/inc_max_nreg/dec_max_nreg` (`builtin.py:390-445`).
  - The template enables it for arch-specific targets ≥ 90 (`src/tl_templates/cuda/intrin.h:7-11, 168-190`). CUTLASS enables it for `__CUDA_ARCH__ == 1200` with SM120_ALL (`3rdparty/cutlass/include/cutlass/arch/reg_reconfig.h:45-66`).
  - Warp-level register partitioning on sm_120a is plausible [I, T].

---

## 6. Register cap, launch bounds, dynamic smem, resource query

- **`__launch_bounds__`:**
  - `PrintExtraAttrs` always emits `__launch_bounds__(threads, min_blocks_per_sm)` when the thread count is static (`src/cuda/codegen/codegen_cuda.cc:628-645`) [V].
  - `min_blocks_per_sm` defaults to 1. Set it with `T.annotate_min_blocks_per_sm(k)` inside the kernel (`tilelang/language/annotations.py:85-104`, attribute `tl.min_blocks_per_sm`) [V].
  - This is the practical per-kernel register cap, e.g. 256 threads × k = 2 → ≤ 128 regs/thread.
  - There is no `__maxnreg__` support and no 3rd launch-bounds argument (helper report).
- **Compile flags [V]:**
  - `tilelang.jit(compile_flags=[…])`, `T.annotate_compile_flags` or pass config `tl.device_compile_flags` (`pass_config.py:86-98`), applied in `tilelang/cuda/backend.py:59-104` for the default tvm_ffi backend.
  - `tl.ptxas_register_usage_level` (0-10) maps to `--ptxas-options=--register-usage-level=N`.
  - `--maxrregcount=N` can be passed (used in `maint/gemm/gemm_sm120/benchmark_sm120_nvfp4_blockscaled_gemm.py:37-49`). Because TileLang always emits `__launch_bounds__`, the CUDA docs say launch bounds take precedence, so the flag may be a no-op [I; check with ptxas -v].
  - A true per-kernel `__maxnreg__` would need a small codegen change (attribute → `PrintExtraAttrs`).
- **Dynamic smem [V, helper report]:**
  - `MergeSharedMemoryAllocations` produces one `shared.dyn` buffer.
  - `LowerDeviceKernelLaunch` computes `dyn_shmem_size` (`src/transform/lower_device_kernel_launch.cc:147-162`).
  - The tvm_ffi runtime calls `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` lazily (`3rdparty/tvm/src/runtime/cuda/cuda_module.cc:265-276`).
  - No user knob beyond allocating buffers. Static `scope="shared"` buffers are separate.
- **Query registers/smem [V, helper report]:**
  - `JITKernel.n_regs/resource_usage` are HIP-only (`tilelang/jit/kernel.py:707-736`).
  - For CUDA:
    - set `TILELANG_VERBOSE=1`: `--ptxas-options=--verbose` is added and the output logged (`backend.py:97-104`, `nvcc.py:202-204`). This happens only on a binary-cache miss.
    - or run `cuobjdump --dump-resource-usage` on the cached cubin `$TILELANG_CACHE_DIR/<ver>/cuda-binaries/<sha>.cubin`;
    - or load that cubin with cuda-python (`cuModuleLoadData` → `cuFuncGetAttribute(NUM_REGS / SHARED_SIZE_BYTES / LOCAL_SIZE_BYTES)`).
  - Recommend a small `research/bench/resources.py` helper.
  - `kernel.show_sass()/export_sass()` recompile with only `compile_flags`, not pass-config flags, so the SASS can differ.

## 7. Compiling many configs cheaply [V, helper report + spot checks]

- **`tilelang.par_compile(funcs, out_idx=None, execution_backend=None, target=None, target_host=None, verbose=None, pass_configs=None, compile_flags=None, num_workers=None, ignore_error=False)`** (`tilelang/jit/__init__.py:174-264`):
  - `ThreadPoolExecutor`; lowering releases the GIL and nvcc is a subprocess.
  - `JITImpl.par_compile(configs, …)` (`:403-450`) elaborates serially, then compiles in parallel.
  - Per-kernel pass configs are possible via `T.annotate_pass_configs` / `T.annotate_compile_flags`.
- **`AutoTuner`** (`tilelang/autotuner/tuner.py`):
  - Thread pool sized by `TILELANG_AUTO_TUNING_CPU_COUNTS / _CPU_UTILITIES / _MAX_CPU_COUNT`.
  - `@tilelang.autotune` **serializes compiles** under `_pass_configs_lock` (`:1207-1232`). The `AutoTuner(...).run()` API does not.
- **Grouped compile** (`tilelang/autotuner/grouped_compile.py:30-195`):
  - Merges device IR of N configs, so one device codegen and one nvcc call per group.
  - Enabled with `AutoTuner.run(enable_grouped_compile=True, group_compile_size=k)` (`tuner.py:809-830`). CUDA + tvm_ffi only.
  - Drops `compile_flags`. Pass-config values must be hashable, e.g. string flags, not lists.
- **Caches:**
  - `KernelCache` key = SHA-256 of script + args + target + pass_configs + compile_flags + version (`tilelang/cache/kernel_cache.py:244-285`).
  - Plus `CUDABinaryCache` keyed by generated code + nvcc options.
  - Env: `TILELANG_CACHE_DIR` (default `~/.tilelang/cache`), `TILELANG_DISABLE_CACHE`.
- **Compile-only:** `tilelang.tools.compile_only` (source only, pinned to sm_80, single kernel).

---

## 8. Best starting files

| Need | File | Notes |
|---|---|---|
| (a) sm_120 GEMM (non-wgmma) | `examples/gemm/example_gemm.py` | `T.Pipelined` + `T.copy` + `T.gemm` → mma.sync. **Add `tl.disable_warp_specialized` to get the cp.async path**; the default on sm_120 is TMA + auto-WS |
| | `examples/gemm/example_gemm_persistent.py` | Persistent `T.Persistent` and manual `for w` forms: the template for a role loop |
| | `examples/gemm_streamk/example_tilelang_gemm_streamk.py` | `while` + runtime-extent pipelined loop + atomics; tested only on cc ≤ 8.9 |
| | `examples/gemm/example_gemm_intrinsics.py` | Explicit `TensorCoreIntrinEmitter` (ldmatrix/mma) if tile ops ever get in the way |
| (b) GQA decode, split-KV | `examples/flash_decoding/example_gqa_decode.py` | Split kernel + separate combine kernel; already disables WS. K/V layout `[B, S, H_kv, D]`; `valid_block_H = min(block_H, H_q/H_kv)` pads heads to MMA M. For a CoKernel, fuse the combine via a per-(b,h) "last split finishes" counter, or keep it as a second role |
| | `examples/deepseek_mla/example_mla_decode_persistent.py` | Persistent split + `T.sync_grid()` + combine in one kernel |
| (c) RMSNorm | `examples/norm/rms_norm.py` | fp32, no weight, 128 threads, `tl.disable_tma_lower`. Add `W[d]`. Cross-warp `T.reduce_sum` → named barriers 1/2 |
| (d) Warp specialization | `examples/warp_specialize/example_warp_specialize_gemm_copy_1_gemm_0.py`, `…_barrierpipe_stage2.py` | `T.ws` + `T.alloc_barrier` + `T.tma_copy`; tests on cc 9.0 only |
| | `examples/warp_specialize/example_warp_specialize_flashmla.py` | Two concurrent warpgroups via `if tx < 128 / else` with reduce/Parallel/copy. Hopper-only (wgmma) but the best structural reference for warp-level binding |
| | `tilelang/language/ws_schedule.py` + `examples/aws/gemm.py` | `WSRole(warps_lo, warps_hi, max_nreg)` schedule framework (SM100 examples) |
| (e) Horizontal / multi-op | none | Closest: MLA persistent two-phase; flashmla two-warpgroup; multi-`T.Kernel` prim_funcs (sequential launches) |
| Tile scheduler | `tilelang/language/tile_schedule.py` | Swizzled `coord(tile_id)` decode for atomic-driven loops |

---

## 9. Gaps for a CoKernel prototype (ranked by risk)

1. **Auto-WS on sm_120 vs. two roles / `while` loops** — high risk, cheap to avoid.
   - Only one loop is rewritten; the kernel-wide thread count grows by 128; the other branch is not remapped; `while` ancestors most likely hit an ICHECK; the fallback strips all pipelining.
   - → Always compile CoKernels with `tl.disable_warp_specialized=True`, and optionally `tl.disable_tma_lower=True`.
   - A TMA-based role later means manual `T.tma_copy` + `T.alloc_barrier` pipelines.
2. **Shared memory = sum of roles** — high impact, medium effort.
   - Default liveness keeps all role buffers live across the role `if/while` (§1.6). It kills CTA-level co-residence and may exceed 99 KB even at SM level.
   - Options:
     - (a) `tl.enable_aggressive_shared_memory_merge=True` + debug dump + targeted tests of `while` back-edge safety;
     - (b) manual unioning (`T.view` needs equal byte sizes);
     - (c) a small pass or annotation that treats role branches as exclusive scopes.
   - For warp-level binding the opposite hazard applies: sibling `T.ws` regions may be overlaid. Use a single if/else or `tl.disable_shared_memory_reuse`.
3. **Named-barrier collisions at warp level** — high risk only for warp-level binding.
   - Cross-warp `T.reduce` in both roles uses IDs 1/2 simultaneously.
   - Fix: make `AllReduce`'s barrier IDs a template parameter allocated by the sync pass (e.g. per thread range), or restrict warp-level pairs to roles whose reductions stay within a warp. Also add a ≤ 15 check to `GetOrCreateBarrier`.
4. **`while` loops lack loop-carried sync analysis** (§1.7) — medium. Use explicit `T.sync_threads()` and double-buffered broadcast slots, or `for`+`T.loop_break()`.
5. **`%smid/%nsmid/%globaltimer`** — low risk. `prelude=` + `call_extern` today (§4); header or intrinsic later. Needs the P0 probe for SM id holes.
6. **Resource introspection and register cap** — low risk. Write a helper that reads the cubin; use `annotate_min_blocks_per_sm` for caps; `__maxnreg__` needs a codegen tweak if wanted.
7. **Untested on sm_120:** T.ws + mma.sync, pipelined loops inside thread-subset regions, `setmaxnreg`, CLC. These need small smoke tests once the build is ready.

---

## 10. Recommended minimal implementation path (written directly in TileLang)

Common settings:
- `@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})`.
- Each role's tile body is a `@T.macro` copied from the example (GEMM: `example_gemm.py` body parameterized by `(tm, tn)`; attention: the split kernel body parameterized by `(b, h_blk, split)`).
- Common `threads` (e.g. 256).
- Counters are `int32` tensors zeroed per launch.
- `COTL_SREG` prelude from §4.
- No `T.use_swizzle`.

### (i) SM-level persistent CoKernel (role chosen by `%smid`)

```python
# untested sketch
with T.Kernel(NUM_SM * k, threads=256, prelude=COTL_SREG) as bid:
    <role-A buffers>; <role-B buffers>                       # (sum of smem, gap 2)
    info = T.alloc_shared((4,), T.int32, scope="shared")
    tx = T.get_thread_binding()
    if tx == 0:
        info[0] = sm_role[T.call_extern("int32", "cotl_smid")]     # host table: smid -> role (handles holes / ratio)
    T.sync_threads()
    role = T.alloc_var(T.int32); role = info[0]
    it = T.alloc_var(T.int32, init=0); t = T.alloc_var(T.int32)
    for _ in T.serial(MAX_ITERS):                            # For => loop-carried sync analysis
        if tx == 0:
            info[1 + it % 2] = T.atomic_add(q_ctr[role], 1, return_prev=True)
        T.sync_threads()
        t = info[1 + it % 2]
        if t >= n_tiles[role]:                               # optional: steal from other role instead
            T.loop_break()
        if role == 0:
            gemm_tile(t)       # coords via PersistentTileScheduler(..., stateful=False).coord(t)
        else:
            attn_tile(t)
        it = it + 1
    # completion stamp: after loop, tx==0 does done=atomic_add(done_ctr[role],1,True)+1;
    # the CTA that makes done == #CTAs_in_role writes cotl_globaltimer() to timers[role]
```

- Static variant: grid-stride inside the role once the CTA's rank within the role is known (a second per-role atomic at entry).
- Measure first:
  - whether 1 CTA/SM placement holds (`T.annotate_min_blocks_per_sm(1)` plus large smem guarantees at most one);
  - the atomic cost per tile (chunk it: `atomic_add(q_ctr[role], CHUNK, True)`).

### (ii) CTA-level CoKernel (POD-style, per-SM arrival counter)

```python
# untested sketch; k CTAs per SM must fit => needs gap 2 solved for real co-residence
with T.Kernel(GRID, threads=256, prelude=COTL_SREG) as bid:
    T.annotate_min_blocks_per_sm(k)                          # register cap: 65536/(256*k)
    ...
    if tx == 0:
        sm = T.call_extern("int32", "cotl_smid")
        r = T.atomic_add(sm_ctr[sm], 1, return_prev=True)    # arrival rank on this SM
        info[0] = T.if_then_else(r % (kA + kB) < kA, 0, 1)   # runtime ratio knobs kA:kB
    T.sync_threads()
    role = info[0]
    # same dynamic-queue loop as (i); on exhaustion of own queue, switch role (cross-role takeover)
```

The ratio, SM map, chunk size and priority are all runtime tensors or scalars, so no recompilation is needed.

**Order of work:**
1. Smoke-test a single-role persistent GEMM and a single-role attention with WS disabled on sm_120.
2. Compile the two-role kernel and check generated CUDA:
   - `__launch_bounds__`;
   - `bar.sync` IDs;
   - the smem arena size and offsets (with `tl.debug_merge_shared_memory_allocations`).
3. Correctness vs. PyTorch plus E0 bitwise comparison vs. the solo kernel compiled with the same pass configs.
4. Fix gap 2 before CTA-level experiments.
5. Warp-level binding last, after gap 3.

---

## 11. Suggested first experiments (cheap; most are compile-only once the build exists)

1. `prelude=` + `call_extern("int32","cotl_smid")` compiles; the generated source contains the helper before the kernel.
2. Two-role `if/else` with a `T.Pipelined` GEMM and a `T.Pipelined` attention:
   - with WS disabled: expect two cp.async pipelines, `__syncthreads` inside branches, smem = A + B;
   - with WS enabled: observe the thread count grow to 384, or failure.
3. Same kernel wrapped in `while`: expect success with WS disabled and an ICHECK with WS enabled (confirms §1.5).
4. `tl.enable_aggressive_shared_memory_merge=True` on the above: does the arena shrink to max(A, B)? Inspect offsets.
5. Warp-level: `if tx < 128: rmsnorm-ish reduce else: attention reduce` → grep for `bar.sync 1, 128` in both branches (confirms gap 3).
6. `T.atomic_add(ctr[0], 1, return_prev=True)` vs `T.atomic_add(ctr, 1, return_prev=True)`: the latter should raise.
