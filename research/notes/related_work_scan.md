# Related-work novelty scan for Co-TileLang (2024 → 2026-09)

> Written 2026-09-22 by a research sub-agent. Scope: work that could overlap with or pre-empt proposal v0.1
> (`research/proposal.md`): describing two independent ops by semantics, compiling them into **one** kernel
> with tile-level orchestration (SM/CTA/warp binding; static/dynamic/priority scheduling; resource partitioning),
> and **jointly** searching each op's tile implementation with the orchestration, under a cost budget.
> Nothing was run on the GPU.
>
> **Evidence legend** (applies to every factual claim below):
> **[P]** verified by reading the primary paper text (PDF/HTML) in this scan;
> **[A]** verified from the abstract / official project page only;
> **[C]** verified by reading source code (repo + commit given);
> **[S]** secondary source only (search-engine summary, third-party page), not independently confirmed;
> **[I]** my inference / judgement.

---

## 0. TL;DR

- **No single work found that does the whole thesis**: semantics-driven re-derivation of *two independent* ops'
  tile programs for a *shared* CTA shape / resource contract, co-located in one kernel with a *searchable*
  choice of binding level (SM/CTA/warp) and scheduling policy, jointly optimized for joint completion time with
  controlled search cost. [I, after ~40 searches + reading 8 primary papers and 2 codebases]
- **Every individual mechanism already exists, mostly hand-built**: CTA-level `%smid` binding with per-SM
  tickets (POD-Attention), warp-level split with searched thread partition + register cap (HFuse), compile-time
  tile-level co-scheduling of several ops in one persistent kernel with co-scheduling-aware *selection among
  library variants* and several heterogeneous vEUs per SM (Rammer), SM-level worker runtimes with static/dynamic
  queues and smem paging (MPK, Event Tensor, Hazy megakernels), role-bound SM partitions with runtime-adaptive
  split (TileLink, mKernel).
- **The "solo-optimal ≠ co-run-optimal implementation" phenomenon is already published** for inter-kernel
  concurrency: GOLDYLOC (TACO 2025, concurrent GEMMs, resource-constrained tuning), VELTAIR (ASPLOS'22, CPU,
  multi-version compilation per interference level), NanoFlow (OSDI'25, kernel variants + pairwise interference
  profiling + MILP). The proposal already says the phenomenon is not the contribution; reviewers will still cite
  these against RQ1, so they must be in §1.3/§6.
- **Most threatening**: (1) POD-Attention (+ FlashInfer's port), (2) Rammer (understated in the proposal),
  (3) MPK (OSDI'26, likely) + Event Tensor (MLSys'26) — the Mirage/TVM groups already own an in-kernel tile
  runtime with semantics-derived task implementations and paged shared memory; intra-SM co-residence is a short
  step for them, (4) GOLDYLOC + NanoFlow (co-run-aware implementation choice, inter-kernel), (5) HFuse.
- **Claim checks**: (a) MPK — correct, with caveats; (b) HFuse — correct for tiling, but it *does* search each
  kernel's thread count and the fused register cap; (c) Rammer — **partly wrong**: it has multiple rKernel
  versions and picks among them co-scheduling-aware, and co-locates heterogeneous rTasks on one SM via several vEUs;
  (d) POD `%smid` + per-SM counters — correct; (e) FlashInfer POD on sm_120 — not claimed or tested, no arch
  gate, will JIT-compile, but its SM-aware scheduler assumes `%nsmid == #SMs` and sizes its counter array by
  `cudaDevAttrMultiProcessorCount` → possible out-of-bounds on GPUs with harvested SMs (RTX PRO 6000 = 188/192).

---

## 1. Verification of the five specific claims

### (a) "MPK runs one task per worker at a time and has no design for heterogeneous tasks co-resident on one SM"
**Verdict: correct as stated, but needs two caveats.**
- Paper (arXiv 2512.22219v2, Jun 2026) [P]: "Each worker runs on one physical SM and maintains an independent task
  queue." "Workers execute a lightweight loop that repeatedly dequeues tasks, performs the associated computation or
  communication, and signals task completion." Schedulers are warps ("each SM hosting four scheduler warps") on a
  few dedicated SMs (A100: 104 worker SMs + 4 scheduler SMs).
- Code [C] (`/home/ywc/mirage-compiler/include/mirage/persistent_kernel/persistent_kernel.cuh`, local snapshot,
  tarball dated 2026-09-13, no git metadata): `persistent_kernel` / `worker_kernel` are
  `__launch_bounds__(WORKER_NUM_THREADS, 1)`, `worker_id = blockIdx.x`, one `while (true)` loop that fetches the
  next task and calls `_execute_task(...)` sequentially; dynamic smem is set to `MAX_DYNAMIC_SHARED_MEMORY_SIZE`
  (≈ the whole SM), so one worker CTA occupies one SM.
- Caveat 1: heterogeneous tasks *do* run concurrently on *different* SMs (SM-level heterogeneity is the norm in
  MPK). The claim must be phrased as "no two tasks execute concurrently *on the same SM*".
- Caveat 2 [P]: MPK has **cross-task pipelining** — "MPK opportunistically overlaps the compute phase of the
  current task T₁ with the pre-loading phase of the subsequent task T₂" — enabled by **paged shared memory**
  ("task implementations are modified to operate on pages instead of assuming a monolithic allocation").
  This is a restricted form of intra-SM overlap of two different tasks, and paged smem is exactly the substrate
  that would let MPK host two co-resident tasks. [I] The gap to the proposal is real but small in mechanism terms.
- Side observation [C][I]: in the local snapshot, `runtime_header.h` sets
  `MAX_DYNAMIC_SHARED_MEMORY_SIZE = 207 KB - reserved` for `MPK_TARGET_CC >= 90`, which would include sm_120
  (max ~99 KB/CTA) unless the build sets `MPK_TARGET_CC` differently. Not run; relevant if MPK is a baseline.
- Also: MPK's task implementations come from the **Mirage superoptimizer** at thread-block level [P] ("MPK uses
  the Mirage superoptimizer to search for an optimized thread-block graph"). So the §6 row "从语义推导实现: 部分 /
  自动搜索: 否" is inaccurate: task implementations *are* searched from semantics, with a solo objective.

### (b) "HFuse does not change each kernel's internal tile implementation"
**Verdict: correct for the tiling/algorithm, but incomplete.** [P] (arXiv 2007.01277 = CGO'22 version)
- HFuse is a source-to-source CUDA compiler: it re-maps `threadIdx`/`blockDim` so the two kernels occupy disjoint
  thread ranges of one block, and replaces `__syncthreads()` with `bar.sync id, nthreads` partial barriers.
- Its search (Fig. 6) varies the **block dimension given to the first kernel** (granularity 128 threads; requires
  kernels written for tunable `blockDim`) and profiles each variant **with and without a computed register bound**.
  So it changes each kernel's thread count and the fused kernel's register cap (hence spilling/codegen), but never
  the tile shape, loop structure or decomposition.
- Suggest the §6 cell "为共置改 kernel 内部实现: 否" → "部分（仅线程数与寄存器上限；不改 tile/分解）".
- POD-Attention's §3 measures "warp-parallel (HFuse)" fusion of prefill/decode attention and reports straggler
  problems [P] — useful ammunition for why warp-level binding is not always best.

### (c) "Rammer uses a fixed kernel library and static compile-time scheduling without modeling intra-SM contention"
**Verdict: partly correct, partly misleading — should be rewritten.** [P] (OSDI'20 paper)
- Correct: rKernels are *loaded* from "auto-kernel generators [TVM], hand-tuned kernels, or converted from existing
  operators"; the rProgram is a **static** compile-time plan ("moves the scheduling decision from runtime to compile
  time"); there is no explicit model of intra-SM contention (the profiler gives per-rTask time on a vEU, rTask
  resource usage, and total rProgram time).
- Misleading #1: "One rOperator might have multiple versions of rKernels based on different tiling strategies,
  e.g., trading off between resource efficiency and overall execution time." The policy's `SelectRKernels()` picks,
  per wave, between the *fastest* versions and the *most efficient* ones (smallest runtime × #rTasks), keeping the
  efficient ones only if **profiling the combined wave** shows a win: "This heuristic considers the interplay between
  the inter- and intra-operator scheduling by evaluating the rOperators (and their rTasks) in a wave, instead of
  individually." → Rammer already does *co-scheduling-aware implementation selection among library variants*;
  profiling the whole rProgram captures contention implicitly (not modeled, but measured).
- Misleading #2: "an SM can run multiple vEUs (PTBs) concurrently … the number of vEUs an SM can support depends on
  the most demanding rTask across all the vEUs" and "RAMMER sets the number of threads of a vEU to be the maximum
  number of threads used by an rTask in the vEU. For an rTask with less threads, RAMMER inserts early-exit logic".
  → heterogeneous rTasks from different operators **are co-resident on one SM**, and Rammer already exhibits the
  §1.2(3) max-resource coupling — it just handles it by **idling** the extra threads.
- [I] The defensible differences are: (i) Rammer *selects* among pre-built versions, whereas the proposal
  *re-derives* the partner's tile program for the shared CTA shape (Rammer idles threads instead); (ii) Rammer has no
  contention model and no dynamic/priority scheduling or cross-role takeover; (iii) Rammer targets DAG inter-op
  parallelism at small batch, not compute×memory complementarity.

### (d) "POD-Attention's CTA scheduling is SM-aware via reading %smid and per-SM counters"
**Verdict: correct.** [P] (paper Fig. 9) and [C] (FlashInfer `include/flashinfer/attention/pod.cuh:61-120`,
commit `0d7df3db`, 2026-09-22).
- Leader thread: `mov.u32 %0, %smid;` → `ticket = atomicAdd(&sm_ctr[sm_id], 1) % (prefill_ratio+decode_ratio)` →
  choose PREFILL/DECODE → `cta_id = atomicAdd(&cta_assign[op], 1)`; if that op is exhausted, switch to the other op
  (a simple form of cross-role takeover); broadcast (op, cta_id) through shared memory; `__syncthreads()`.
  Policies: 50:50 and proportional.
- Nuance: the per-SM counter is only the ticket counter; the CTA-id counters are **global per op** in the code
  (the prose mentions "2 more counters" per SM). FlashInfer's port reads `%nsmid` for the array offset.
- Other POD implementation changes for co-location [P]: decode Q-tile 16 (to free Tensor Cores), 2 or 4 CTAs/SM
  configurations chosen at runtime, **virtual decode CTAs** (one warp each, warp-level barriers) to balance smem
  against prefill, limiting prefill KV-splits to ≤ 2 waves, hand-tuned smem for both ops. A persistent-CTA variant
  "performs on par". Evaluated on A100 only; attention speedup up to 59% (mean 28%), end-to-end throughput up to 22%.

### (e) "Does FlashInfer's POD kernel support sm_120?"
**Verdict: not officially; it will probably compile and may run, but it is untested on sm_120 and has a latent
out-of-bounds hazard that must be checked on the RTX PRO 6000 before using it as a baseline.** [C] (FlashInfer
`main` @ `0d7df3db`, 2026-09-22)
- No compute-capability gate: `flashinfer/pod.py` has no `supported_compute_capability` decorator;
  `gen_pod_module` → `gen_jit_spec` → `check_cuda_arch()` only requires ≥ sm75; code is the FA2-style `mma.sync`
  template (`prefill.cuh`), which sm_120 supports. smem budget is derived from
  `cudaDevAttrMaxSharedMemoryPerMultiprocessor`, so it adapts to 100 KB. No sm_120 test or doc claim found
  (`tests/utils/test_pod_kernels.py` has no arch skip; docs say nothing arch-specific).
- Hazard: `pod.cuh:73-76` — `// WARNING: nsmid has only been tested on A100/H100, and matches SM count // No
  guarantee this will work on other GPUs`; the kernel uses `num_SMs = %nsmid` and indexes `tbAssign[%smid]` and
  `tbAssign[num_SMs + op]`, but the host allocates `tbAssign` with `cudaDevAttrMultiProcessorCount + 2` ints
  (`pod.cuh:414`). PTX ISA [S]: "%nsmid may be larger than the physical number of SMs" and `%smid` "is not guaranteed
  to be contiguous". On a 188-of-192-SM GB202, if `%nsmid` reports 192 or `%smid` has holes, the kernel writes
  past the buffer and mis-assigns ops. [I] Must be measured (one tiny kernel reading `%smid/%nsmid`).
- Also: FlashInfer's port does **not** implement virtual decode CTAs (`// TODO_AK: If num_threads dont match, use
  virtual sub-CTAs`) and chooses CTAs/SM by a smem heuristic marked `TODO(Zihao): fix`. It is a simplified POD.

---

## 2. Catalog of related work (grouped; ~32 entries)

### A. Intra-kernel co-location of heterogeneous ops (closest to the thesis)

**A1. POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference** — A. K. Kamath, R. Prabhu,
J. Mohan, S. Peter, R. Ramjee, A. Panwar. ASPLOS 2025. <https://arxiv.org/abs/2410.18038> [P]
One fused kernel computes prefill and decode attention of a hybrid batch. CTA-parallel fusion (not warp-parallel,
which suffers stragglers) plus software **SM-aware CTA scheduling**: after dispatch, each CTA reads `%smid`, takes a
per-SM ticket, and becomes a prefill or decode CTA according to a 50:50 or proportional policy, falling back to the
other op when one is exhausted. Implementation is modified for co-location by hand: small decode tiles, 2 vs 4
CTAs/SM (runtime choice), virtual decode CTAs, limited prefill splits, hand-balanced smem. It is exactly one
hand-found point of the proposal's space (CTA-level binding, dynamic ticket policy, co-run-aware tiles) for one pair.

**A2. FlashInfer POD port and persistent "holistic" BatchAttention** — Z. Ye et al., FlashInfer (MLSys 2025 [S]),
repo `flashinfer-ai/flashinfer` @ `0d7df3db`. <https://github.com/flashinfer-ai/flashinfer> [C]
(i) `PODWithPagedKVCacheWrapper` / `BatchPODWithPagedKVCacheWrapper`: POD's scheduler (see §1(d)), `smem = max(p,d)`,
`threads = max(p,d)`, no virtual CTAs. (ii) `BatchAttention` (`persistent.cuh`, `scheduler.cuh:TwoStageHolisticPlan`):
a cooperative persistent kernel with **two tile programs** (`CTA_TILE_Q = 128` and `16`) in one launch,
2 CTAs/SM (head_dim < 256); the host planner assigns work items to CTAs by a greedy min-heap on
`cost = 2*qo_len + kv_len`; each CTA runs its class-128 list, then its class-16 list, then a grid-sync reduction;
smem = max of both. → a production example of *static, plan-time, cost-balanced tile scheduling of two tile
programs in one kernel* (no `%smid`, no co-location intent, same op family).

**A3. Automatic Horizontal Fusion for GPU Kernels (HFuse)** — A. Li, B. Zheng, G. Pekhimenko, F. Long. CGO 2022.
<https://arxiv.org/abs/2007.01277> [P]
Source-to-source fusion of two CUDA kernels into one block by thread ranges; `__syncthreads` → named partial
barriers; profile-based search over the thread split (step 128) × {no reg bound, computed reg bound}. Reports the
largest gains for memory-intensive × compute-intensive pairs (2.5%–60.8%). Warp-level binding with a small
automatic search; no tile re-derivation, no contention model.

**A4. Tacker: Tensor-CUDA Core Kernel Fusion for Improving the GPU Utilization while Ensuring QoS** — H. Zhao et al.
HPCA 2022. <https://github.com/sjtu-epcc/Tacker> [S]
Fuses a Tensor-Core kernel with a CUDA-core kernel, with a fused-kernel duration predictor and a QoS-aware runtime
manager. Pre-2024; already cited in the proposal.

**A5. Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks** — L. Ma, Z. Xie, Z. Yang, J. Xue,
Y. Miao, W. Cui, W. Hu, F. Yang, L. Zhang, L. Zhou. OSDI 2020. <https://www.usenix.org/conference/osdi20/presentation/ma> [P]
Operators are split into rTasks; a compile-time policy builds a static rProgram mapping rTasks of several operators
onto virtual execution units (vEUs = persistent thread blocks, possibly several per SM) with barrier-rTasks for
dependencies. Multiple rKernel versions per operator; per-wave co-scheduling-aware version selection by profiling;
vEU thread count = max over its rTasks with early-exit of extra threads. See §1(c). Follow-ups in this line (Roller,
Welder) are single-op/subgraph tiling, not co-location.

**A6. Souffle: Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions** — C. Xia, J. Zhao,
Q. Sun, Z. Wang, Y. Wen, T. Yu, X. Feng, H. Cui. ASPLOS 2024. <https://dl.acm.org/doi/10.1145/3617232.3624858> [P]
TE-level global analysis; its **horizontal transformation** merges independent TEs into a single TE (concatenating
outputs, predicates/if-else to select inputs, "similar to Rammer"), then schedules with Ansor. Semantics-driven
horizontal fusion, but the goal is more parallelism/input reuse for a single program; the fused kernel is
homogeneous code, not co-resident heterogeneous roles, and contention is not considered.

**A7. PyTorch Inductor "combo kernels"** (experimental horizontal fusion) — RFC pytorch/pytorch#170268 (author
karthickai), PR #190705. <https://github.com/pytorch/pytorch/issues/170268> [A]/[S]
Fuses independent pointwise/reduction ops into one Triton kernel with per-sub-kernel dispatch branches;
sub-kernel bodies emitted as `noinline` device functions. Production horizontal fusion, no GEMM roles, no
co-location/contention awareness [S/I].

### B. Megakernels / in-kernel tile-task runtimes

**B1. MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs** — X. Cheng, Z. Zhang, Y. Zhou, … T. Chen,
Z. Jia (21 authors). arXiv 2512.22219 (v1 Dec 2025, v2 Jun 2026); acknowledges "the anonymous OSDI reviewers and our
shepherd" → very likely OSDI 2026 [P]. <https://arxiv.org/abs/2512.22219>, <https://github.com/mirage-project/mirage>
Compiles a multi-GPU model into an SM-level task graph and one persistent megakernel: worker CTA per SM with its own
queue, scheduler warps on dedicated SMs doing decentralized event-driven dispatch, paged shared memory, cross-task
pipelining (T₂ preload overlaps T₁ compute), multiple ttGraphs for representative batch sizes. Task implementations
are searched by the Mirage superoptimizer at thread-block level (solo objective). Evaluated on A100/H100/B200.
See §1(a).

**B2. Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel** — H. Jin, B. Hou, G. Wang, R. Lai, …
Z. Jia, T. Chen. MLSys 2026. <https://arxiv.org/abs/2604.13327> [A]+[S from HTML]
Encodes dependencies between tiled tasks (shape- and data-dependent dynamism) and compiles persistent kernels on TVM.
Supports **static** (per-SM pre-assigned queues, counter semaphores) and **dynamic** (event-triggered push to a
scheduler, any SM pops) task scheduling. Task bodies are user-specified (Triton / TVM DSL); no implementation search;
no intra-SM co-residence reported. Same group as MPK/TVM — strong infrastructure for scheduling policies.

**B3. Hazy Research megakernels + ThunderMLA** — B. Spector, J. Juravsky, S. Sul, O. Dugan, D. Lim, D. Fu, S. Arora,
C. Ré. Blogs: "Look Ma, No Bubbles!" (2025-05-27), "We Bought the Whole GPU…" (2025-09-28), "ThunderMLA"
(2025-03-04). <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>,
<https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main> [A]
An on-GPU instruction interpreter (ThunderKittens template): low-latency version uses per-SM instruction schedules
built ahead of time on the host, 13×16 KB smem pages passed between instructions so loads of the next instruction
overlap the current one; the TP Llama-70B throughput version uses a **global atomic work queue** and deliberately
**interleaves different instruction types across SMs** so compute-, memory- and NVLink-bound work run concurrently
on different SMs; loader/consumer/storer warp roles inside an instruction. Hand-written, no search; "different
instruction types do not run simultaneously on individual SMs" [S from page summary]. ThunderMLA used makespan-based
static scheduling (~10% gain) [S].

**B4. Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference** — W. Dong et al.
arXiv 2605.11581 (May 2026). <https://arxiv.org/abs/2605.11581> [A]
Megakernel inside TensorRT-LLM on L20 with a three-dimensional shared-memory constraint model and MLIR-based offline
DAG search that fixes the execution path (no runtime branching). Search over fused execution paths, not over
co-location of independent ops.

**B5. Other 2026 megakernel works (low relevance)** — Fleet (AMD MI350 chiplet tasks, arXiv 2604.15379) [A];
AutoMegaKernel (agent-synthesized megakernels, arXiv 2606.09682) [A]; ForgeMegakernel (agent + milestones,
arXiv 2609.12379) [A]. None addresses co-resident heterogeneous tasks or joint implementation search.

### C. Compute–communication overlap with in-kernel role binding (same machinery, different pair)

**C1. TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives** —
S. Zheng, J. Fang, X. Zheng, Q. Hou, W. Bao, N. Zheng, Z. Jiang, D. Wang, J. Ye, H. Lin, L.-W. Chang, X. Liu.
MLSys 2025. <https://arxiv.org/abs/2503.20313> [A]+[S from HTML]
Tile-centric primitives on Triton link a compute part and a communication part in one fused kernel; communication
mapped to DMA or to a subset of SMs while the rest compute ("we use 20 SMs for communication in this example");
the SM split and tiles are programmer-chosen, no autotuning. Part of Triton-distributed.

**C2. mKernel: Fast Multi-GPU, Multi-Node Fused Kernels** — Z. Mao, Y. Zhang, S. W. Chew, S. Ma, C. Raiciu, Y. Zhou,
S. Shenker, I. Stoica. arXiv 2609.13585 (Sep 2026). <https://arxiv.org/html/2609.13585> [A]+[S]
Persistent kernels where thread blocks are bound to compute / intra-node comm / inter-node send / receive roles,
tile-level readiness handoff, and a **runtime controller that adapts the compute/comm SM split** from measured
progress. Compute tiles from ThunderKittens; no implementation search. Shows "dynamic role re-partitioning inside a
persistent kernel" is current practice.

**C3. ParallelKittens** — S. Sul, S. Arora, B. Spector, C. Ré. arXiv 2511.13940 (MLSys 2026 oral [S]) [A];
**Resource-aware Computation-Communication Overlap for multi-GPU ML Workloads** — M. Cui, M. Pericàs,
ISC 2026 AI-on-HPC workshop, arXiv 2606.09200 [A]: shapes GEMM occupancy via per-block smem so communication kernels
can co-reside — i.e., *changing a kernel's resource shape for co-residence* (inter-kernel, manual/heuristic).
**DeepGEMM** `set_num_sms/get_num_sms` ("set/get the maximum SM count to use"; SM90/SM100 only) [A] — the standard
partial-SM-budget knob.

### D. Co-run-aware implementation selection (pre-empts the RQ1 phenomenon, inter-kernel)

**D1. GOLDYLOC: Global Optimizations & Lightweight Dynamic Logic for Concurrency** — S. Pati, S. Aga, N. Jayasena,
M. D. Sinclair. ACM TACO 2025 (arXiv 2409.02227). <https://dl.acm.org/doi/full/10.1145/3730584> [A]+[S from HTML]
Shows GEMM kernels tuned in isolation "hoard" resources under concurrency. **Resource-constrained tuning**: retune
each GEMM on GPU/2 and GPU/4 (half/quarter CUs and LLC) and benchmark concurrent execution at each concurrency degree
to pick "globally optimized" GO-Kernels (larger tiles or higher occupancy, fewer waves/LLC misses); a
logistic-regression predictor in the command processor picks the concurrency degree and swaps in GO-Kernels at
runtime. Separate kernels on streams; GEMM × GEMM only; AMD MI100. **This is T[co, inter] for GEMM×GEMM.**

**D2. VELTAIR: Towards High-Performance Multi-tenant Deep Learning Services via Adaptive Compilation and Scheduling**
— Z. Liu, J. Leng, Z. Zhang, Q. Chen, C. Li, M. Guo. ASPLOS 2022 (CPU). <https://arxiv.org/abs/2201.06212> [A]
A static multi-version compiler (extends TVM) "can identify different optimal code versions under different
interference levels", and the runtime picks versions by current interference. Pre-2024 but the clearest prior art
for "compilation should see the co-location context"; CPU only.

**D3. NanoFlow: Towards Optimal Large Language Model Serving Throughput** — K. Zhu, Y. Gao, Y. Zhao, et al.
OSDI 2025. <https://arxiv.org/abs/2408.12757> [P from HTML]
Nano-batches duplicate ops so compute-, memory- and network-bound nano-ops overlap as **separate kernels on streams**.
"NanoFlow explores all possible kernel implementations varying the number of thread blocks, the number of warps, and
tile size for GEMM, GEMV, and network kernels"; **pairwise interference profiling** maps a resource share R to
performance P; a **two-stage MILP** picks nano-batch structure then R per nano-op (ΣR ≤ 1). No SM-level co-location
guarantee, no fused kernel.

**D4. ElasticRoom: Multi-Tenant DNN Inference Engine via Co-design with Resource-constrained Compilation and Strong
Priority Scheduling** — HPDC 2024. <https://dl.acm.org/doi/10.1145/3625549.3658654> [S]
"Resource-constrained compilation" for co-located tenants on A100 and MI100; details not verified (ACM page 403).
Worth a full read if the scan is extended.

### E. Inter-kernel spatial sharing in LLM serving (baselines for T[·, inter]; cost models for RQ4)

| Work | Venue | Mechanism | Kernel changes? | Model | Ev. |
|---|---|---|---|---|---|
| **Bullet** — Z. Lin, H. Xu, G. Chen, Z. Chen, Y. Lu, X. Zhang. <https://arxiv.org/abs/2504.19516> | ASPLOS 2026 [S] | libsmctrl stream masks, concurrent prefill/decode streams | No | SM-scaling roofline (SRM) + calibration, SLO feedback | [A]+[S] |
| **Nexus** — X. Shi, C. Cai, J. Du, Z. Jia. <https://arxiv.org/abs/2507.06608> | arXiv 2025 | CUDA green contexts, pre-instantiated partitions | No ("without kernel modification") | analytic per-op max(compute, mem) + contention | [A]+[S] |
| **DuetServe** — L. Gao, C. Jiang, H. E. Zarch, D. Wong, M. Hill, M. Annavaram. <https://arxiv.org/abs/2511.04791> | arXiv 2025/26 | libsmctrl SM partitions, activated when TBT at risk | No | attention-aware roofline, pre-profiled Π_SM(S), B_HBM(S) | [A]+[S] |
| **Drift → MuxWise** (PD-Multiplexing) — Y. Chen, W. Cui, H. Zhao, Z. Xu, …, Q. Chen. <https://arxiv.org/abs/2504.14489> | ASPLOS 2026 [S] | green contexts in SGLang, gang scheduling | No | "contention-free"/"contention-tolerant" estimator | [A] |
| **semi-PD** — K. Hong, …, G. Dai, Y. Liang, Y. Wang. <https://arxiv.org/abs/2504.19867> | arXiv 2025 | SM-level compute controller (MPS per secondary sources) | No | — | [A]+[S] |
| **SpaceServe** — Li et al. <https://neurips.cc/virtual/2025/poster/115356> | NeurIPS 2025 | libsmctrl SM slices for MLLM encoder × decoder | No | cost-model scheduler | [S] |
| **MuxServe** — J. Duan et al. | ICML 2024 | spatial-temporal multiplexing of multiple LLMs | No | — | cited |
| **HyGen** — Sun et al. <https://arxiv.org/abs/2501.14808> | NeurIPS 2025 | request-level online/offline co-location | No | latency predictor | [S] |
| **DynaFlow** — Y. Pan, …, B. Kasikci, S. Wang. <https://arxiv.org/abs/2605.21603> | MLSys 2026 | programmable op scheduling (nano-batching, DBO, TokenWeave) on streams | No (user may swap in fused kernels) | none (user-written schedules) | [A]+[S] |
| **Liger** — Du et al. | PPoPP 2024 | cross-request interleaving on multiple streams, runtime kernel decomposition | No | contention factor | [S] |

Takeaway [I]: all of these treat kernels as black boxes and move SM budgets; their roofline-style models are the
natural competitors to the proposal's v0 analytic model, and green-context/libsmctrl partitioning is the right
T[·, inter] baseline.

### F. GPU sharing runtimes / OS (inter-kernel, black-box kernels)

- **LithOS: An Operating System for Efficient Machine Learning on GPUs** — P. H. Coppock, B. Zhang, E. H. Solomon,
  V. Kypriotis, L. Yang, B. Sharma, D. Schatzberg, T. C. Mowry, D. Skarlatos. SOSP 2025.
  <https://arxiv.org/abs/2504.15465> [A]+[S]: TPC-granularity spatial scheduler with TPC stealing, **transparent
  kernel atomization** (subsets of thread blocks, "without compiler, runtime, source, or PTX changes" [S]),
  hardware right-sizing model. Strongest OS-level baseline; does not touch implementations.
- **SGDRC** — Y. Zhang, …, X. Chu, H. Li. PPoPP 2025. [P] Tidal SM masking + VRAM-channel page colouring for
  LS/BE co-location, offline profiling.
- **KACE** — B. S. Han, T. Paul, Z. Liu, A. Gandhi. SoCC 2024. [S] Predicts co-location interference from
  exclusive kernel metrics with little training data.
- **MeanField surrogate for concurrent heterogeneous inference** — Y. Ennouri, S. Ha. IEEE ESL (arXiv 2609.02109,
  Sep 2026) [A]: linear-in-samples surrogate + GA scheduler; related to the "avoid O(N_A·N_B) co-profiling" argument.
- Already cited in the proposal: Orion (EuroSys'24), Tally (ASPLOS'25), REEF (OSDI'22), KRISP (HPCA'23).

### G. Semantics → tile compilers and warp-specialization compilers (enablers, not competitors)

- **Mirage** (OSDI'25) — cited. **Prism: Symbolic Superoptimization of Tensor Programs** — M. Wu, X. Jiang,
  O. Padon, Z. Jia. arXiv 2604.15272 (Apr 2026) [A]: symbolic sGraph encoding execution parameters for a two-level
  search — the Mirage successor to track for the subgraph phase (§10).
- **Neptune** (PLDI 2026 [S], arXiv 2510.08726) — fusion of reduction chains with schedule/tile optimizers [A].
- **Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References** — H. Chen, B. Fan, …,
  Z. Zhang, V. Grover. CGO 2026. <https://arxiv.org/abs/2510.14719> [A]. **Twill: Optimal Software Pipelining and
  Warp Specialization for Tensor Core GPUs** — R. Soi, R. Yadav, F. Kjolstad, A. Aiken, M. M. Dehnavi, M. Garland,
  M. Bauer. OSDI 2026. <https://arxiv.org/abs/2512.18134> [A]: constraint-solver-optimal SWP+WS schedules. Both
  automate *intra-op* warp roles; the proposal's warp-level binding of *two ops* is an extension of this problem —
  Twill's joint constraint formulation is a candidate technique for the warp-level role split.
- **Kitsune: Enabling Dataflow Execution on GPUs** — M. Davies, N. Crago, K. Sankaralingam, S. W. Keckler (NVIDIA).
  arXiv 2502.18403 / ACM TACO [S]. [P] Spatial pipelines of *dependent* ops via on-chip inter-CTA queues and a
  **modified CTA scheduler** that places complementary SIMT-heavy and TensorCore-heavy CTAs on the same SM; PyTorch
  Dynamo compiler; **evaluated in a simulator** (requires HW changes). Useful evidence that SIMT×TC pairing on one SM
  pays off; not a software competitor.
- Low relevance (dependent fusion): MCFuser (2506.22169), ComFuse (2608.03537), Tile-Level Activation Overlap
  (2607.02521), FlashFuser (2512.12949) [A].

### H. Partial SM budgets, wave quantization, work-centric decomposition

- **Stream-K** (PPoPP'23, cited); **Stream-K++** (Springer 2025, arXiv 2408.11417) — adaptive selection among
  Stream-K variants with Bloom filters [S]. **WaveTune** — K. Zhang et al., arXiv 2604.10187 [A]: wave-aware bilinear
  latency model for tile-based GEMM autotuning (does not model reduced SM budgets or co-location).
- GOLDYLOC's GPU/2, GPU/4 resource-constrained tuning (D1) and DeepGEMM's `set_num_sms` (C3) are the closest
  "tune under a partial SM budget" prior art. [I] The proposal's wave-quantization example (§1.2 point 2) is correct
  arithmetic but not novel as an observation; it becomes interesting only when the budget is *chosen jointly* with
  the partner.

### I. Numerical determinism

- **Defeating Nondeterminism in LLM Inference** — Thinking Machines Lab blog, Sep 2025, with
  `batch_invariant_ops` <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/> [S]; LLM-42
  (arXiv 2601.17768) [S]. Batch-invariance = fix the reduction strategy regardless of batch shape. [I] The
  proposal's E0/E1 "partner-invariance" is the co-location analogue and does not appear in any co-location paper
  found; it is a small but distinctive contribution and should cite batch invariance.

---

## 3. Comparison table (columns as in proposal §6)

Legend: Y = yes, P = partial, N = no, — = not applicable. "Intra-SM" notes whether two different ops can be
co-resident on one SM at the same time.

| Work (venue) | 从语义推导实现 | 为共置改 kernel 内部实现 | kernel 内 tile 级编排 | 自动搜索 | 范围 | Ev. |
|---|---|---|---|---|---|---|
| POD-Attention (ASPLOS'25) | N | Y (manual: decode tile 16, 2/4 CTA/SM, virtual CTAs, split cap) | Y (CTA-level, `%smid` ticket, fallback switch; intra-SM) | N (runtime pick of 2 configs) | prefill × decode attention | P |
| FlashInfer POD port (repo 2025–26) | N | P (inherits POD tiles; no virtual CTAs) | Y (same scheduler; `%nsmid` assumption) | N | prefill × decode attention | C |
| FlashInfer BatchAttention (repo) | N | P (two tile programs 128/16, smem=max) | Y (static plan-time min-heap balancing; sequential per CTA) | P (cost heuristic) | attention only | C |
| HFuse (CGO'22) | N | P (thread count per kernel + reg cap only) | Y (warp/thread-range split, named barriers; intra-SM) | P (profile split × reg bound) | any 2 CUDA kernels w/ tunable blockDim | P |
| Tacker (HPCA'22) | N | N | Y (TC + CUDA-core fusion) | P (duration predictor) | TC kernel + CUDA-core kernel | S |
| Rammer (OSDI'20) | N (library: TVM-generated / hand-tuned) | P (**selects** among rKernel versions per wave; idles extra threads) | Y (static rProgram; several vEUs per SM → intra-SM) | P (profiling-guided heuristic) | DNN inter-op parallelism | P |
| Souffle (ASPLOS'24) | Y (TE-level horizontal/vertical transforms) | N (solo objective) | N (single homogeneous TE kernel) | Y (Ansor) | DNN subprograms | P |
| Inductor combo kernels (PyTorch, exp.) | P (Inductor lowering) | N | P (sub-kernel dispatch branches) | ? | pointwise/reduction ops | S |
| Mirage (OSDI'25) | Y | — | N (all CTAs run the same block graph) | Y | multi-op subgraphs | cited |
| MPK (OSDI'26, likely) | P/Y (task impls by Mirage superoptimizer, solo objective) | N | Y (SM-level task graph; 1 task/SM at a time; T₂-preload/T₁-compute overlap; paged smem) | P (impl search only; no schedule search) | LLM inference, multi-GPU | P+C |
| Event Tensor (MLSys'26) | N (user DSL tasks) | N | Y (static & dynamic tiled-task scheduling) | N | dynamic-shape LLM megakernels | A |
| Hazy megakernels / ThunderMLA (2025 blogs) | N | N (hand-written instructions) | Y (per-SM static queues or global dynamic queue; SM-interleaving of op types; smem paging) | N (makespan heuristics) | Llama-1B/70B, MLA decode | A |
| Ada-MK (arXiv'26) | N | N | Y (megakernel) | Y (offline DAG search, smem constraint model) | LLM inference (TRT-LLM, L20) | A |
| TileLink / Triton-distributed (MLSys'25) | N | N | Y (SM split comp vs comm, tile signals) | N (manual SM split) | compute × communication | A+S |
| mKernel (arXiv'26) | N | N | Y (TB role binding, runtime-adaptive SM split) | P (runtime controller) | compute × multi-node comm | A+S |
| NanoFlow (OSDI'25) | N | P (variants: #TB, warps, tile) | N (streams) | Y (pairwise interference + 2-stage MILP) | LLM serving nano-ops | P |
| GOLDYLOC (TACO'25) | N | Y (RC-tuned GO-Kernels, library) | N (streams) | Y (RC tuning + LR predictor) | concurrent GEMMs (MI100) | A+S |
| VELTAIR (ASPLOS'22) | P (TVM multi-version) | Y (versions per interference level) | N | Y | multi-tenant DL on CPU | A |
| Resource-aware comp-comm overlap (ISC'26 WS) | N | Y (smem occupancy shaping) | N | ? | GEMM × collectives | A |
| Bullet (ASPLOS'26) | N | N | N (libsmctrl) | Y (SRM + feedback) | LLM PD serving | A+S |
| Nexus (arXiv'25) | N | N | N (green ctx) | Y (analytic) | LLM PD serving | A+S |
| DuetServe (arXiv'25/26) | N | N | N (libsmctrl) | Y (roofline + optimizer) | LLM PD serving | A+S |
| Drift/MuxWise (ASPLOS'26) | N | N | N (green ctx) | Y (estimator) | LLM PD serving | A |
| semi-PD (arXiv'25) | N | N | N (SM controller) | P | LLM PD serving | A |
| SpaceServe (NeurIPS'25) | N | N | N (libsmctrl) | Y (cost model) | MLLM enc × dec | S |
| LithOS (SOSP'25) | N | N (atomization splits grids) | N (OS-level TPC scheduling) | P (right-sizing model) | general ML multi-tenant | A+S |
| DynaFlow (MLSys'26) | N | N | N | N (user-programmed) | intra-device parallelism | A+S |
| Kitsune (arXiv'25/TACO) | P (Dynamo compiler) | N | Y (spatial pipelines, SIMT×TC CTAs on same SM) — needs HW change, dependent ops | P | DL dataflow (simulated A100) | P |
| Twill (OSDI'26) / Tawa (CGO'26) | P (from loop program / Triton) | N | intra-op warp roles only | Y (Twill: constraint solver) | single-op WS | A |
| **Co-TileLang (proposal)** | Y (re-derive for shared CTA shape) | Y (auto) | Y (SM/CTA/warp; static/dynamic/priority; takeover) | Y (joint impl × orchestration, cost-controlled) | multiple op-class pairs | — |

---

## 4. Verdict

### 4.1 Already done by someone (do not claim)
1. **Co-locating a compute-bound and a memory-bound op in one kernel and changing their implementations for it**
   — POD-Attention (hand-designed, one pair), shipped in FlashInfer.
2. **Mechanisms of each binding level**: CTA-level `%smid` binding (POD); warp-level split with named barriers and
   searched thread split / register cap (HFuse); SM-level persistent workers with static or dynamic atomic queues,
   op-type interleaving across SMs, smem paging, cross-task prefetch overlap (MPK, Event Tensor, Hazy megakernels);
   runtime-adaptive role split (mKernel); several heterogeneous persistent blocks per SM with max-resource coupling
   (Rammer).
3. **Co-scheduling-aware choice among pre-built implementation variants** — inside one kernel (Rammer) and across
   kernels (GOLDYLOC, NanoFlow, VELTAIR on CPU).
4. **The observation that concurrency-tuned kernels beat solo-tuned ones**, including under reduced SM/LLC budgets
   (GOLDYLOC), and interference-aware search of resource shares (NanoFlow, Bullet, DuetServe, Nexus, MuxWise).
5. **Semantics-driven implementation search** for single programs, including horizontal fusion of independent
   tensor expressions (Mirage, Prism, Souffle); MPK even plugs Mirage-searched task implementations into an
   in-kernel runtime.

### 4.2 Still open (nothing found that does it)
1. **Re-deriving each role's tile program from semantics for a *shared* CTA shape / resource contract** (the
   "resource alignment" of §5.3), as opposed to selecting from solo-tuned libraries (Rammer, GOLDYLOC, NanoFlow),
   idling threads (Rammer early-exit), or hand-balancing (POD virtual CTAs).
2. **One IR/search space spanning SM/CTA/warp binding × static/dynamic/priority policies × cross-role takeover**,
   in which POD, HFuse, Rammer and MPK/Hazy designs are points, with *automatic per-pair selection of the binding
   level*. Every prior system hard-codes one level.
3. **Joint implementation × orchestration search for joint completion time with explicit cost control**
   (per-candidate sensitivity vectors instead of O(N_A·N_B) co-profiling; tile-level model; regret-vs-cost curves).
   Closest: NanoFlow (inter-kernel, pairwise profiling), GOLDYLOC (inter-kernel, GEMM only), Rammer (heuristic,
   library).
4. **The 2×2 decomposition with an interaction term**, and any co-location characterization on **sm_120**
   (100 KB smem/SM, 188 SMs, 128 MB L2). All intra-kernel prior work evaluates A100/H100/B200/MI100.
5. **Partner-invariant numerics (E0/E1)** under co-location — batch-invariance exists, partner-invariance does not.
6. **Generality beyond attention** for intra-kernel co-location (GEMM × RMSNorm, GEMM × GEMV, GEMM × decode) with
   an automatic method — megakernels mix these ops but by hand and one-per-SM.

### 4.3 Novelty risk assessment [I]
- The "automated POD" reading is the default reviewer reaction. If the search mostly returns CTA-level mixes with
  solo-tuned implementations, the contribution collapses to "POD + autotuner". The **interaction term** (value only
  available with semantic re-derivation under a shared CTA contract) is the core claim and must be large on at
  least some pairs.
- **Scoop risk is highest from the Mirage/MPK group** (Z. Jia: MPK + Nexus + Mirage/Prism) and the TVM group
  (Event Tensor): they hold every ingredient (semantic search, in-kernel task runtime, paged smem, intra-GPU PD
  multiplexing). A "two co-resident workers per SM" extension of MPK would pre-empt the orchestration half.
- GOLDYLOC weakens any claim that RQ1's resource effects are new; the proposal must frame RQ1 as quantifying them
  *inside one kernel on sm_120* and separating them via the 2×2.
- Benefit risk: if green-context partitioning + GOLDYLOC-style co-run-tuned kernels (T[co, inter]) gets within a
  few % of T[co, intra], the intra-kernel machinery is hard to justify; POD's own gains are attention-level (mean 28%)
  with ≤22% end-to-end.

---

## 5. Suggestions to sharpen the novelty claim / de-risk

1. **Make the interaction term the headline and isolate it with an extra baseline row.** Extend the 2×2 to a 3×2:
   add T[lib-select, intra] = best co-kernel whose role implementations are *selected* from the union of solo-tuned
   Pareto variants (what Rammer/GOLDYLOC/NanoFlow can reach), and T[lib-select, inter]. The novelty is
   T[lib-select, intra] → T[co-derived, intra]. Also include "idle extra warps" (Rammer) and "virtual CTAs" (POD) as
   explicit resource-alignment baselines. If this delta is small on all pairs, pivot early (the proposal's own §9
   decision point) — better to know in P1 than after building the search.
2. **Express prior designs as points in the CoKernel space and reproduce them as baselines**: POD (CTA-level,
   `%smid` ticket, proportional policy) on POD's own prefill×decode pair; HFuse (warp split + reg cap);
   Rammer-style static rProgram; MPK/Hazy-style SM-level workers with a dynamic queue; green-context partitioning
   (+ GOLDYLOC-style RC-tuned kernels at 1/2 and 1/4 of the SMs). Showing the search matches or beats each on its
   home turf and wins on other pairs turns "crowded area" into evidence of generality. Rewrite §1.3/§6 accordingly
   (see §6 below) and add GOLDYLOC, VELTAIR, MPK-OSDI'26, Event Tensor, Hazy megakernels, TileLink/mKernel,
   Bullet/Nexus/DuetServe/MuxWise, LithOS, Souffle.
3. **Cheap de-risking on sm_120 before any search work** (all measurable in < 1 day of GPU time):
   (i) read `%smid`/`%nsmid` from every CTA of a 188-SM grid — contiguity and `%nsmid` value decide whether the
   FlashInfer POD port is memory-safe on this GPU and how SM-level binding must remap IDs;
   (ii) run FlashInfer POD (it JIT-compiles for sm_120) as the first real intra-kernel baseline, after (i);
   (iii) hand-write one GEMM × decode-attention CoKernel at CTA and SM level and measure the "role compiled in but
   never executed" overhead (register max-coupling, I-cache) — on 64K regs / 100 KB smem per SM this coupling may
   rule out CTA-level co-residence for big GEMM tiles, which would push the design toward SM/warp levels.
4. (Secondary) **Position the cost model against Bullet's SRM, DuetServe's roofline and NanoFlow's pairwise
   R→P tables**: the claim should be "linear-in-candidates characterization with tile-level simulation that ranks
   intra-kernel configurations", evaluated by Kendall τ and regret-vs-cost, with those models as ablations.
5. (Secondary) **Keep E0/E1 partner-invariance** as a distinctive, cheap contribution; cite batch invariance.
6. (Watch list) MPK repo / Mirage group, Event Tensor (TVM), mKernel (UCCL/Berkeley), Hazy megakernels, FlashInfer
   POD/BatchAttention changes, TileLang's persistent tile-scheduler primitives (see capability survey).

---

## 6. Suggested corrections to proposal text (for the main agent)

- §1.3 / §6 **Rammer**: replace "算子实现来自预先生成的 kernel 库，也不建模 SM 内的资源争用" with something like
  "实现来自 kernel 库；每个算子可有多个 rKernel 版本，编译期按 wave 以 profiling 在'最快'与'最省资源'版本间选择；
  多个 vEU 可驻留同一 SM，线程数取最大值、多余线程提前退出；不显式建模 SM 内争用，调度为静态". §6 cells:
  从语义推导实现 N；为共置改实现 **P（在库版本间选择）**；编排 Y（静态，SM 内可共驻）；搜索 P.
- §6 **HFuse**: 为共置改实现 N → **P（线程数、寄存器上限；不改 tile/分解）**.
- §6 **MPK**: 从语义推导实现 → **P/Y（任务实现由 Mirage 超优化器按单跑目标搜索）**; 自动搜索 N → **P（仅任务实现）**;
  §6 prose: "每个 SM 一个 worker 顺序执行任务，仅允许下一任务的预取与当前任务计算重叠（cross-task pipelining，
  基于 paged shared memory）；不支持两个任务在同一 SM 上并发执行". Reference [5] → X. Cheng et al., arXiv 2512.22219
  (v2 2026-06; acknowledges OSDI shepherd, likely OSDI 2026).
- §6 **NanoFlow**: keep P, but the text should say variants over #thread blocks, #warps and tile size + pairwise
  interference profiling + two-stage MILP.
- Add rows: GOLDYLOC, VELTAIR, Event Tensor, Hazy megakernels, TileLink/mKernel, FlashInfer BatchAttention,
  Souffle, LithOS, and one aggregated row for Bullet/Nexus/DuetServe/MuxWise/semi-PD/SpaceServe.
- §5.3 / plan: note the FlashInfer POD `%nsmid` hazard and the PTX statement that `%nsmid` may exceed the SM count.

---

## 7. Sources

Primary papers read (text extracted): POD-Attention <https://arxiv.org/abs/2410.18038>; Rammer
<https://www.usenix.org/system/files/osdi20-ma.pdf>; HFuse <https://arxiv.org/abs/2007.01277>; Souffle
<https://aura.abdn.ac.uk/server/api/core/bitstreams/2688f621-28df-48bf-92a0-231a1f8835fb/content>; SGDRC
<https://people.cs.vt.edu/~huaicheng/p/ppopp25-sgdrc.pdf>; Kitsune <https://arxiv.org/abs/2502.18403>;
MPK <https://arxiv.org/html/2512.22219v2>; NanoFlow <https://arxiv.org/html/2408.12757>.

Code read: FlashInfer <https://github.com/flashinfer-ai/flashinfer> @ `0d7df3db` (2026-09-22): `pod.cuh`,
`batch_pod.cuh`, `persistent.cuh`, `persistent_template.cuh`, `scheduler.cuh`, `flashinfer/jit/core.py`,
`flashinfer/jit/attention/modules.py`, `tests/utils/test_pod_kernels.py`. MPK local snapshot
`/home/ywc/mirage-compiler/include/mirage/persistent_kernel/{persistent_kernel.cuh,runtime_header.h}`.

Abstract / project pages: Event Tensor <https://arxiv.org/abs/2604.13327>; Ada-MK <https://arxiv.org/abs/2605.11581>;
Fleet <https://arxiv.org/abs/2604.15379>; AutoMegaKernel <https://arxiv.org/abs/2606.09682>; ForgeMegakernel
<https://arxiv.org/abs/2609.12379>; mKernel <https://arxiv.org/html/2609.13585>; TileLink
<https://arxiv.org/abs/2503.20313>, <https://arxiv.org/html/2503.20313v1>; ParallelKittens
<https://arxiv.org/abs/2511.13940>; Resource-aware overlap <https://arxiv.org/abs/2606.09200>; GOLDYLOC
<https://arxiv.org/abs/2409.02227>, <https://arxiv.org/html/2409.02227>, <https://dl.acm.org/doi/full/10.1145/3730584>;
VELTAIR <https://arxiv.org/abs/2201.06212>; Bullet <https://arxiv.org/abs/2504.19516>,
<https://arxiv.org/html/2504.19516v4>; Nexus <https://arxiv.org/abs/2507.06608>, <https://arxiv.org/html/2507.06608v5>;
DuetServe <https://arxiv.org/abs/2511.04791>, <https://arxiv.org/html/2511.04791v2>; Drift/MuxWise
<https://arxiv.org/abs/2504.14489>, <https://arxiv.org/abs/2504.14489v2>; semi-PD <https://arxiv.org/abs/2504.19867>;
SpaceServe <https://neurips.cc/virtual/2025/poster/115356>; HyGen <https://arxiv.org/abs/2501.14808>; DynaFlow
<https://arxiv.org/abs/2605.21603>, <https://arxiv.org/html/2605.21603>; Liger
<https://dl.acm.org/doi/10.1145/3627535.3638466>; LithOS <https://arxiv.org/abs/2504.15465>; KACE
<https://dl.acm.org/doi/10.1145/3698038.3698555>; MeanField <https://arxiv.org/abs/2609.02109>; ElasticRoom
<https://dl.acm.org/doi/10.1145/3625549.3658654>; Prism <https://arxiv.org/abs/2604.15272>; Tawa
<https://arxiv.org/abs/2510.14719>; Twill <https://www.usenix.org/conference/osdi26/presentation/soi>; WaveTune
<https://arxiv.org/abs/2604.10187>; Stream-K++ <https://arxiv.org/abs/2408.11417>; ComFuse
<https://arxiv.org/abs/2608.03537>; Tile-Level Activation Overlap <https://arxiv.org/abs/2607.02521>; Hazy blogs
<https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>, <https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main>,
<https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla>; Inductor combo kernels
<https://github.com/pytorch/pytorch/issues/170268>; DeepGEMM <https://github.com/deepseek-ai/DeepGEMM>; MPK repo
<https://github.com/mirage-project/mirage>; ASPLOS'26 list <https://paper.lingyunyang.com/reading-notes/conference/asplos-2026>;
batch invariance <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>; PTX `%smid/%nsmid`
text via <https://blog.csdn.net/dark5669/article/details/77097073> (secondary; confirm in the official PTX ISA).
