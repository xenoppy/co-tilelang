# Co-TileLang：从计算语义出发的 kernel 内 tile 级共置

> 研究提案草稿 v0.1（2026-09-22）。执行计划见 [plan.md](plan.md)，进展与决策记录见 [status.md](status.md)。

## 摘要

GPU 上常有多个相互独立的计算流同时就绪，每个流包含 GEMM、attention、normalization、reduction 等操作。这些操作压在不同的硬件资源上，因此存在共置执行的机会。但一个操作占用多少资源、如何执行，并不只由计算语义决定：tile 大小、线程映射、流水线深度、数据布局和计算分解都会改变实现的资源占用与执行行为。所以，"先分别找到单跑最快的 kernel，再做共置调度"不一定得到最优组合。

本研究的核心思想是：**像 Mirage 那样以计算语义描述操作，由编译器把两个操作编进同一个 kernel，并在 kernel 内以 tile 为粒度编排它们的执行。** 编译器从语义推导出每个操作的 tile 实现空间，把实现选择与 kernel 内编排（空间放置、时间顺序、资源划分）放在一起，以共同执行为目标搜索，同时控制编译和 profiling 成本。

第一阶段聚焦两个单算子之间的共置。长期目标是把描述与编排扩展到子图（每个流贡献一个多算子子图），并引入 Mirage 式的结构搜索。

## 1. 背景与动机

### 1.1 共置的机会

计算密集的操作（大 GEMM、prefill attention）主要消耗 Tensor Core；访存密集的操作（decode attention、normalization、GEMV、reduction）主要消耗 DRAM 带宽和访存流水线。二者同时运行时，有机会同时逼近两类资源的上限。LLM serving 中的 prefill/decode 混合批、MoE 的共享专家与路由专家、多租户推理、投机解码中的 draft 与 target 模型，都会产生这类相互独立、同时就绪的操作。

### 1.2 资源需求属于实现，而不属于语义

同一个 GEMM，换一组 tile 配置，寄存器、shared memory、CTA 数量、DRAM 流量和 Tensor Core 占用都会改变。以本研究的主要平台 RTX PRO 6000 Blackwell（sm_120；188 个 SM；每 SM 100KB shared memory、64K 寄存器、48 个 warp；L2 128MB）为例，下面三种效应都会让"单跑最优"与"共跑最优"分离：

1. **SM 内容量。** fp16 GEMM 用 128×128×64 的 tile、三级流水，占 (128×64 + 64×128) × 2B × 3 = 96KB shared memory，同一 SM 上已放不下任何伙伴 CTA。降为两级流水后占 64KB，能腾出约 35KB 给伙伴，代价是单跑略慢。
2. **波次量化。** 4096×4096 的 GEMM（简化为每 SM 一个 CTA）：128×128 tile 共 1024 块，128×256 tile 共 512 块，在 188 个 SM 上的波次效率都是 90.8%。伙伴占去 34 个 SM、只剩 154 个时，前者为 95.0%，后者降到 83.1%。单跑时打平的两个配置，在共置预算下相差 12 个百分点。
3. **合并进同一个 kernel 后的资源耦合。** 一个 kernel 只有一个寄存器数和一份 shared memory 预留，所以每个 CTA 的占用取两个角色的最大值。若 A 角色需要 168 regs/thread，B 只需要 64，那么每个 256 线程的 CTA 都按 43K 寄存器预留，每个 SM 只能放一个；B 单独运行时（同样 256 线程）本可以放四个。此时选一个单跑稍慢、但资源形状与 B 对齐的 A 实现，收益可能远大于单跑上的损失。

此外，同一种资源"值多少"取决于伙伴争用什么：伙伴吃满 DRAM 时，省流量的实现（大 tile、融合、重算）更值钱；伙伴要在同一 SM 上驻留时，占用小的实现更值钱。这两个方向相反，最优实现只能在共置语境下确定。

### 1.3 现有工作做到哪里

POD-Attention [1] 为 prefill/decode attention 手工修改了 tile、CTA 配置与计算分解，并在 kernel 内做 SM 感知的 CTA 调度，证明了"为共置改实现"的价值，但它只针对一个算子对，依赖人工设计。HFuse [2] 自动把两个给定的 CUDA kernel 在 CTA 内按线程拼接，并搜索融合配置，但不改变各自内部的 tile 实现。NanoFlow [3] 在 LLM serving 中分析不同实现之间的干扰并搜索资源划分，kernel 实现来自库。Rammer [16] 在编译期把多个算子的 tile 级任务静态编排进一个 kernel，但算子实现来自预先生成的 kernel 库，也不建模 SM 内的资源争用。Mirage [4] 从计算语义出发搜索实现，但目标是单跑性能，生成的程序中所有 CTA 执行同一段代码，不涉及共置。详细对比见第 6 节。

尚未回答的问题是：**能否让共置环境参与 kernel 的候选生成、组合编译与性能搜索——从计算语义出发，在保持语义的前提下，自动得到更适合共同执行的实现，并控制编译与 profiling 成本？** 本研究不把"共置有收益"或"为共置改实现有收益"这些现象本身当作创新；贡献在于让这件事能自动、通用、低成本地完成。

## 2. 核心思想

```
语义描述 A、B
   │  推导实现空间：tile 形状、归约分解、CTA 形状、流水深度、布局
   ▼
候选 tile 程序（附资源签名）
   │  联合搜索：实现 × 编排（绑定层级、放置、调度策略、资源划分）
   ▼
CoKernel IR
   │  降级到 TileLang
   ▼
单个 CUDA kernel ──► profiling ──► 更新性能模型，回到搜索
```

三个要点：

1. **从语义出发。** 输入是操作的计算语义（哪些轴并行、哪些轴归约、用什么合并算子），不是写好的 kernel。编译器因此可以重新推导实现：换 tile 形状、换分解方式、为一个共同的 CTA 形状重新 tile。最后一点是共置特有的需求——两个角色编进同一个 kernel 后必须共享线程数和资源预留，只有从语义出发，才能让伙伴"按同一个形状"重新实现。
2. **在 kernel 内以 tile 为粒度编排。** 两个操作的 tile 由 kernel 内统一的分派逻辑执行。编译器显式决定：tile 在哪里执行（SM 级、CTA 级、warp 级），以什么顺序执行（静态划分、动态队列、优先级），资源怎么分（线程、shared memory、寄存器）。与多 stream 并发相比，这带来可控的放置、跨角色的动态负载均衡（一方做完，另一方接手剩余 tile，消除尾部）和优先级控制；代价是资源耦合与分派开销，同样交给编译器权衡。
3. **以共同执行为目标联合搜索。** 实现选择与编排一起搜索，目标是二者共同完成的时间（或带 SLO 的变体）。成本通过资源签名、敏感度刻画、tile 级性能模型和分阶段编译来控制。

## 3. 研究问题

- **RQ1（价值与来源）**：与"各自单跑最优实现 + 最好的共置机制"相比，联合选择实现与 kernel 内编排能带来多少收益？收益来自哪些资源效应（SM 内容量、波次、资源耦合、共享带宽、流水线争用）？
- **RQ2（表示与语义）**：什么样的中间表示能同时表达两个操作的语义、各自的 tile 实现空间和 kernel 内编排，并保证语义保持？
- **RQ3（编排机制）**：kernel 内 tile 级编排有哪些有效形式（SM 级、CTA 级、warp 级绑定；静态、动态、优先级调度），各自适用于什么样的操作组合？
- **RQ4（搜索与成本）**：如何在"实现 × 编排"的乘积空间中，以远低于穷举的编译和 profiling 成本找到接近最优的组合？
- **RQ5（长期：子图）**：当每个流贡献一个多算子子图时，结构搜索（融合边界、切分方式、重算与物化）如何与共置编排联合？

## 4. 问题定义

**输入**：操作 A、B 的语义描述（含形状、数据类型）；目标硬件；可选的优先级或延迟约束。A 与 B 之间没有数据依赖。

**决策变量**：
- A、B 的实现 c_A ∈ C(A)、c_B ∈ C(B)：tile 形状、归约分解（顺序、split、按工作量均分）、CTA 形状、流水深度、布局；
- 编排 o：绑定层级、放置参数（SM 划分、每 SM 的角色配比、warpgroup 分配）、调度策略及其参数、资源划分方式。

**目标**：最小化 A、B 都完成的时间 T(c_A, c_B, o)。两个变体：
- 带约束：B 的完成时间不超过 SLO，最小化 A 的完成时间（或最大化 A 的吞吐）；
- 稳态：A、B 反复执行时的加权吞吐 Σ_i rate_i^co / rate_i^solo。

两者单跑时长相差很大时，完成时间几乎无法改善（例如 2ms 的 GEMM 配 50µs 的 RMSNorm，上限约 2.4%）。因此实验要么按"单跑时长比"扫描，要么在稳态模式下测量。

**约束**：语义保持（见 5.5）；资源可行（每 CTA 占用取角色最大值，每 SM 不超过容量）。

**收益分解**。用 2×2 设计把两种收益分开：

| | 跨 kernel 共置（多 stream / green context 分区） | kernel 内编排（CoKernel） |
|---|---|---|
| 单跑最优实现 | T[solo, inter] | T[solo, intra] |
| 共跑搜索得到的实现 | T[co, inter] | T[co, intra] |

- T[solo, inter] / T[co, inter]：共跑感知实现本身的价值，与编排方式无关；
- T[solo, inter] / T[solo, intra]：kernel 内编排本身的价值；
- 交互项：只有在 kernel 内编排下才划算的实现（例如为共同 CTA 形状重新 tile）带来的额外收益；
- 另报串行时间 T_serial，以及资源下界 LB = max_r (d_A,r + d_B,r) / cap_r 作为参照。

## 5. 技术方案

### 5.1 语义描述

第一阶段支持四类操作：GEMM/GEMV、attention（prefill 与 GQA decode）、normalization（RMSNorm/LayerNorm）、行归约。描述形式是带命名轴的张量表达式，标出并行轴、归约轴和归约的合并算子。草案：

```python
# GEMM：并行轴 m, n；归约轴 k，合并算子 sum
C[m, n] = sum(A[m, k] * B[k, n], over=k)

# GQA decode：并行轴 b, h；g 并入行方向；归约轴 s，合并算子为 online softmax
S[b, h, g, s] = sum(Q[b, h, g, d] * K[b, h, s, d], over=d) * scale
O[b, h, g, :] = softmax_combine(S[b, h, g, s], V[b, h, s, :], over=s)

# RMSNorm：并行轴 t；归约轴 d
r[t] = rsqrt(sum(X[t, d] ** 2, over=d) / D + eps)
Y[t, d] = X[t, d] * r[t] * W[d]
```

从描述推导实现空间的规则：
- 并行轴按 tile 大小切分，得到 tile 下标空间；
- 归约轴有三种处理：tile 内顺序循环（带流水深度）；切成若干段再追加合并（split-K、split-KV，要求合并算子满足结合律）；按工作量均分（Stream-K [17] 式）；
- CTA 形状（线程数、warpgroup 数）、布局与 swizzle 交给 TileLang 的调度参数和 layout 推断。

第一阶段语义层的作用是**让实现空间可以被重新推导**，尤其是"给定一个 CTA 形状和资源预算，推导出每个操作最合适的 tile 程序"；它不追求发现新算法。表示上与 Mirage μGraph 的概念保持对应（tile 映射对应 imap/fmap/omap），为子图阶段接入结构搜索留出接口。

### 5.2 候选 tile 程序与资源签名

一个候选 tile 程序包括：tile 下标空间、每个 tile 的计算体（加载、MMA、归约、收尾）、CTA 形状和调度参数。每个候选附带**资源签名**：
- 每个 CTA 的 shared memory（TileLang lowering 后精确可知）、寄存器（编译后可知，编译前用估计值）、线程数；
- tile 数量，每个 tile 的计算量（Tensor Core / CUDA Core）与访存量（DRAM、L2）；
- 单跑时每个 tile 的时长，以及对各类共享资源压力的敏感度（见 5.4）；
- 数值等级（E0/E1，见 5.5）。

候选生成要主动覆盖单跑启发式会剪掉的区域：更少的流水级、更小的 tile、更细的分解、寄存器上限，以及按给定 CTA 形状重新 tile 的变体。生成后按"单跑时间 × 资源向量"做 Pareto 剪枝。

### 5.3 kernel 内 tile 级编排

**CoKernel 表示**（草案）：

```
CoKernel {
  roles:     [TileProgram_A, TileProgram_B]
  cta:       { threads, smem_layout: alias | partition, reg_cap }
  binding:   SM | CTA | WARP
  placement: { sm_split | per_sm_mix | warpgroup_map }
  schedule:  static | dynamic_queue(policy) | priority(slo)
  knobs:     { ratio, sm_split, priority, chunk_size, ... }   # 运行时参数，调整时不必重新编译
}
```

**绑定层级**：

| 层级 | 角色如何确定 | 资源 | 主要争用 | 约束 |
|---|---|---|---|---|
| SM 级 | 持久化 CTA 读 `%smid`，按 SM 编号决定角色 | 每 CTA 占用取两角色最大值 | L2、DRAM | `%smid` 可能不连续，需要重映射 |
| CTA 级 | 同一 SM 上按到达计数分配角色（POD 式） | 每 SM 放 ⌊容量 / 最大占用⌋ 个 CTA，按配比分给两个角色 | SM 内全部资源 | 两个角色都要小到能共驻 |
| warp 级 | 同一 CTA 内按 warpgroup 分配角色（`T.ws`） | shared memory 相加，寄存器取最大值 | SM 内全部资源 | 角色内只能用 named barrier 同步，不能用全 CTA 的 `__syncthreads` |
| 指令级（远期） | 同一组 warp 交替执行两个角色的 tile 步骤 | 同一组寄存器 | 发射槽 | 需要跨角色的软件流水 |

**调度策略**：
- 静态：按 grid-stride 固定分配 tile，行为可预测，但尾部可能不均；
- 动态队列：每个角色一个全局原子计数器，CTA 做完一个 tile 再取下一个；可以一次取一块 tile，摊薄原子操作开销；
- 跨角色接手：一个角色的队列空了，它的 CTA 转去执行另一角色的 tile，用来消除尾部。SM 级和 CTA 级绑定下可行，因为每个 CTA 本来就按最大占用预留了资源；
- 优先级与配比：延迟敏感的角色优先取 tile，或按目标配比控制两个角色同时在执行的 tile 数。

**资源对齐**：编译器先为两个角色选定共同的 CTA 形状，再从语义为这个形状推导各自的 tile 程序。例如 GEMM 用 256 线程时，decode attention 不让多出来的 warp 空闲，而是让每个 CTA 处理两倍的 head。CTA 级绑定下，两个角色的 shared memory 可以互相复用（同一时刻一个 CTA 只执行一个角色的 tile）；warp 级绑定下则按角色划分。

**降级到 TileLang**：每个角色的 tile 计算体生成为 `T.macro`；分派循环用 `T.atomic_add(..., return_prev=True)` 取 tile 编号，经 shared memory 广播给整个 CTA；warp 级角色用 `T.ws` 划分。TileLang 已有的 `ThreadPartialSyncRewriter` 会把线程范围受限区域内的同步改写成 named barrier，可以作为角色内同步的基础。需要补充或验证的能力：
- 读取 `%smid`/`%nsmid`（TileLang 目前没有）；
- 两个角色都运行完整 tile 算子（gemm、reduce、copy）时，named barrier 的正确性与 ID 分配（硬件每个 CTA 只有 16 个 named barrier）；
- tile 算子在任意 warpgroup 子集上的 layout 推断；
- 跨角色的 shared memory 复用；
- 各角色完成时间的打点（`%globaltimer`）。

### 5.4 搜索与成本控制

搜索空间是"A 的候选 × B 的候选 × CTA 形状 × 绑定层级 × 调度策略 × 运行时参数"。成本控制分三层：

1. **按确定时机给参数分类。** tile 配置、CTA 形状、绑定层级、调度策略的代码形态是编译期参数；配比、SM 划分、优先级、每次取 tile 的块大小是运行期参数，调整时不需要重新编译。
2. **逐个算子刻画，成本随候选数线性增长。** 对每个候选测单跑时长、不同 SM 数下的时长曲线，以及与少量合成压力 kernel（DRAM 流、L2 抖动、Tensor Core 满载、SM 资源占位、ALU 满载）共跑得到的敏感度向量（Bubble-Up [10] 式），代替 O(N_A·N_B) 的两两共跑。
3. **模型排序，少量编译。** 用 tile 级性能模型预测 (c_A, c_B, o) 的共同完成时间：v0 是解析模型（各角色吞吐 × 可用槽位 × 争用系数），v1 是 tile 级离散事件模拟（每 SM 的槽位、依赖共驻情况的 tile 时长、调度策略）。模型只需要把排序做对。排名前 k 的 CoKernel 用 TileLang 的 grouped compilation 编译并实测，再对运行时参数做一维搜索。

评估搜索看三个量：相对穷举最优的差距（regret）随编译与 profiling 成本的变化曲线、模型排序的 Kendall τ、达到最优 5% 以内所需的成本。成本控制的必要性要用实际规模来论证：统计一个目标场景中"（算子, 形状分桶）配对"的实例数，乘以单个实例的穷举成本。

### 5.5 语义保持与数值

两个角色写互不相交的输出，彼此没有数据依赖；原子计数器与工作区在每次启动时正确复位。数值上分两级：
- **E0 逐位不变**：GEMM 的 block_M/N、流水级数、线程数、布局与 swizzle、光栅化方式，以及 SM 划分和调度策略。只要 MMA 指令形状不变，它们就不改变输出元素沿 K 方向的累加顺序。
- **E1 数学等价**：split-K、split-KV、Stream-K，normalization/reduction 的线程映射（归约树变了），以及 attention 的 KV 块大小（online softmax 按块做 rescale）。

如果共置决策会改动 E1 参数，同一个请求的数值结果就会依赖"它和谁一起跑"，这是 serving 中的一种非确定性。系统提供确定性模式：只允许 E0 参数随伙伴变化，并量化这个约束损失了多少收益。验证方式：与高精度参考实现比对；E0 变体还要与同配置的单跑 kernel 做逐位比较。

## 6. 与现有工作的关系

| 工作 | 从语义推导实现 | 为共置改 kernel 内部实现 | kernel 内 tile 级编排 | 自动搜索 | 范围 |
|---|---|---|---|---|---|
| POD-Attention [1] | 否 | 是（手工） | 是（SM 感知的 CTA 调度） | 否 | prefill + decode attention |
| HFuse [2] | 否 | 否 | 是（CTA 内按线程划分） | 部分（融合配置） | 任意两个 CUDA kernel |
| NanoFlow [3] | 否 | 部分（限定 SM 数的变体） | 否（kernel 级并发） | 是（干扰 profiling） | LLM serving 算子 |
| Tacker [7] | 否 | 否 | 是（Tensor Core 与 CUDA Core kernel 融合） | 部分（时长预测） | TC kernel + CUDA Core kernel |
| Rammer [16] | 否（kernel 库） | 否 | 是（编译期静态编排 rTask） | 部分（编译期调度策略） | DNN 推理的算子间并行 |
| Mirage [4] | 是 | 不涉及共置 | 否（所有 CTA 执行同一段代码，单跑目标） | 是 | 多算子子图 |
| MPK [5] | 部分（从模型图生成任务图） | 否 | 是（megakernel 内的任务调度） | 否 | LLM 推理 |
| REEF [13]、Tally [14] | 否 | 否（块级变换，不改 tile 实现） | 部分（kernel padding、块级调度） | 否 | 多租户隔离与抢占 |
| Orion [12]、Abacus [11]、KRISP [15]、MuxServe [18] | 否 | 否 | 否 | 是（调度 / 分区） | 固定 kernel 的共享与调度 |
| Elastic Kernels [8] | 否 | 是（改 grid/block，通用变换） | 否 | 部分 | 通用 kernel |
| Warped-Slicer [9] | — | — | 硬件机制（SM 内资源划分） | — | 硬件 |

本研究落在这几列的交集：从语义推导实现空间，在 kernel 内做多层级的 tile 编排，以共同执行为目标自动搜索，并覆盖多类算子。与最接近的三项工作的区别：
- **POD-Attention**：它手工完成的"改 tile / CTA / 分解 + SM 感知调度"，在本研究中由编译器对任意算子对自动搜索完成；POD 的方案是这个空间里的一个点。
- **Rammer**：它的实现来自固定的 kernel 库，编排在编译期静态完成，也不建模 SM 内共驻时的资源争用。本研究为共同的 CTA 形状重新推导实现，建模 SM 内与共享资源的争用，并支持动态调度。
- **MPK**：它调度的是一个模型内部、实现固定的任务。据我们对其代码的阅读，一个 worker 一次执行一个任务，没有让异构任务在同一 SM 上共驻的设计（写作前需核实）。本研究改变的是实现本身，并允许异构 tile 在 SM 内共驻。

## 7. 预期贡献

1. **刻画研究**：在 Blackwell（sm_120）上系统量化单算子对共置中"共跑感知实现"与"kernel 内编排"各自的收益及其资源来源（2×2 分解）。
2. **表示与编译器**：CoKernel 表示，以及基于 TileLang 的编译器，把两个操作的语义描述编成单个带 tile 级编排的 kernel，支持 SM、CTA、warp 三级绑定和多种调度策略。
3. **搜索方法**：基于资源签名、压力敏感度和 tile 级性能模型的分阶段搜索，在给定成本下接近穷举最优。
4. **评估**：覆盖多种算子对和形状，与多 stream、green context 分区、HFuse 式融合、POD 式手工方案、MPK 等对比，并做消融。
5. **长期**：扩展到子图级描述与编排，结合 Mirage 式结构搜索。

## 8. 评估方案（概要）

- **平台**：RTX PRO 6000 Blackwell Workstation Edition（sm_120）。若能拿到 H100 或 B200，复现关键实验，检验结论是否跨架构成立。
- **算子对**：GEMM × decode attention、GEMM × RMSNorm、GEMM × GEMV（计算 × 访存，主实验）；prefill attention × decode attention（POD 的场景，用于对照）；GEMM × GEMM、decode attention × RMSNorm（同类资源，对照组）。形状取自主流 LLM（如 Llama-3-8B/70B 的层配置），并扫描单跑时长比。
- **基线**：串行；多 stream；green context SM 分区（本机可用，粒度 8 个 SM）；多 stream + 共跑感知配置（持久化 grid、限制占用率）；HFuse 式融合；POD 式手工编排；MPK（在适用的场景）。
- **指标**：共同完成时间与加速比、各操作相对单跑的减速、稳态加权吞吐、搜索成本与 regret、数值一致性。
- **消融**：不做语义重 tile；只用静态调度；逐一比较各绑定层级；不用模型（随机或穷举）；确定性模式。

具体实验设计与判定标准见 [plan.md](plan.md)。

## 9. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| 收益空间小 | 2×2 分解中，共跑感知实现的价值普遍低于 5% | 第一阶段结束设决策点；转向收益集中的场景（单跑时长相当的计算 × 访存对、小批量占不满 GPU 的算子），或提前进入子图级 |
| sm_120 的 SM 内容量小 | 每 SM 只有 100KB shared memory，CTA 级共驻空间有限 | 优先考察 SM 级和 warp 级；搜索中显式覆盖小占用变体 |
| 同一 kernel 内两个角色的副作用 | 寄存器按最大值分配、指令 cache 压力增大、分派开销 | 专门测量这些开销（对照"另一角色编进来但不执行"的 kernel）；按块取 tile |
| TileLang 能力缺口 | 两个角色都跑完整 tile 算子时的 named barrier 正确性；线程子集上的 layout 推断；跨角色 shared memory 复用 | 先手写 CoKernel 原型，确定需要哪些能力，再逐项补齐 |
| 测量噪声 | 共跑结果依赖启动时序；600W 功耗墙下降频；128MB L2 让权重常驻缓存 | 锁频并记录功耗与频率；输入轮换或刷 L2；用 CUDA Graph；多次重复，报中位数与分位数 |
| 场景拥挤 | LLM prefill/decode 共置已有大量工作 | 强调从语义出发的通用性，覆盖多类算子对，并加入结构不同的场景 |
| 数值非确定性 | E1 参数随伙伴变化 | 提供确定性模式，并量化其代价 |

## 10. 长期目标：子图级描述与编排

当每个流贡献一个多算子子图（例如 decode 层的 RMSNorm → QKV GEMM → attention）时，共置不仅改变最优参数，还可能改变最优的程序结构：
- **H1**：伙伴吃满 DRAM 带宽时，省流量的结构（融合、重算、大 tile）更有价值；
- **H2**：伙伴要在同一 SM 驻留时，占用小的结构（不融合、切得更细）更有价值；
- **H3**：SM 预算小或经常变化时，更细粒度的分解（更大的 split-KV、Stream-K）更有价值。

H1 与 H2 方向相反，所以最优结构取决于伙伴。技术路线：用 Mirage [4] 的搜索与等价验证产生结构候选池（保留全部候选，而不是只取单跑最优），把 μGraph 降级到 TileLang 以获得调度参数和 sm_120 支持，再进入本研究的共置编排。初步检查本地的 Mirage 代码，发现要先处理几处问题：sm_120 的 shared memory 容量未定义；转译器的架构判断不统一；每个 μGraph 只实测到一个调度变体；搜索时的 shared memory 上限是编译期常量。细节见 [plan.md](plan.md) 的 P5。

## 11. 参考文献

> 草稿阶段整理，投稿前需逐条核对标题、作者与会议。

1. A. K. Kamath et al. POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference. ASPLOS 2025.
2. A. Li, B. Zheng, G. Pekhimenko, F. Long. Automatic Horizontal Fusion for GPU Kernels. CGO 2022.
3. K. Zhu et al. NanoFlow: Towards Optimal Large Language Model Serving Throughput. OSDI 2025.
4. M. Wu et al. Mirage: A Multi-Level Superoptimizer for Tensor Programs. OSDI 2025.
5. Mirage Persistent Kernel (MPK). mirage-project, 2025.
6. L. Wang et al. TileLang: A Composable Tiled Programming Model for AI Systems. arXiv 2025.
7. H. Zhao et al. Tacker: Tensor-CUDA Core Kernel Fusion for Improving the GPU Utilization while Ensuring QoS. HPCA 2022.
8. S. Pai, M. J. Thazhuthaveetil, R. Govindarajan. Improving GPGPU Concurrency with Elastic Kernels. ASPLOS 2013.
9. Q. Xu et al. Warped-Slicer: Efficient Intra-SM Slicing through Dynamic Resource Partitioning for GPU Multiprogramming. ISCA 2016.
10. J. Mars et al. Bubble-Up: Increasing Utilization in Modern Warehouse Scale Computers via Sensible Co-locations. MICRO 2011.
11. W. Cui et al. Abacus（多 DNN 推理中的确定性算子重叠与时延预测）. SC 2021.
12. F. Strati, X. Ma, A. Klimovic. Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications. EuroSys 2024.
13. M. Han et al. Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences（REEF）. OSDI 2022.
14. W. Zhao, A. Jayarajan, G. Pekhimenko. Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads. ASPLOS 2025.
15. M. Chow, A. Jahanshahi, D. Wong. KRISP: Enabling Kernel-wise RIght-sizing for Spatial Partitioned GPU Inference Servers. HPCA 2023.
16. L. Ma et al. Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks. OSDI 2020.
17. M. Osama et al. Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU. PPoPP 2023.
18. J. Duan et al. MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving. ICML 2024.
