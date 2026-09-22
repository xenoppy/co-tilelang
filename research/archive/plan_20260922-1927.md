# 执行计划

> v0.1（2026-09-22）。研究内容见 [proposal.md](proposal.md)；每周进展与决策记录在 [status.md](status.md)。

## 0. 总览

| 阶段 | 目标 | 对应 | 预计 | 出口标准 |
|---|---|---|---|---|
| P0 | 环境与测量基础设施 | — | 1–2 周 | 示例 kernel 在 sm_120 上跑通；重复测量的变异系数 < 2% |
| P1 | 动机与极限研究，手写 CoKernel 原型 | RQ1 | 3–4 周 | 决策点 D1：继续、收窄或转向 |
| P2 | CoKernel 编译器 MVP | RQ2、RQ3 | 5–6 周 | 自动生成的 CoKernel 与手写原型相差 ≤ 5%；三种绑定层级在 ≥ 2 个算子对上可用 |
| P3 | 搜索与成本控制 | RQ4 | 4–5 周 | ≥ 4 个算子对上，以 ≤ 10% 的穷举成本达到最优的 5% 以内 |
| P4 | 系统评估与论文 | 全部 | 4–6 周 | 论文初稿 |
| P5 | 子图级扩展（长期） | RQ5 | — | — |

时长是粗估，D1 之后按结果调整。

## 1. P0：环境与测量基础设施

### 1.1 环境

- 本机：RTX PRO 6000 Blackwell Workstation Edition，CUDA 12.9，驱动 580.173.02。
- 仓库目前没有 `build/`：`~/mpk-env` 能 import 源码里的 tilelang，但找不到 `build/lib` 和 `build/tvm`。先编译 co-tilelang，再用以下示例做冒烟测试：
  - [examples/gemm](../examples/gemm)
  - [examples/flash_decoding/example_gqa_decode.py](../examples/flash_decoding/example_gqa_decode.py)
  - [examples/norm/rms_norm.py](../examples/norm/rms_norm.py)
  - [examples/warp_specialize](../examples/warp_specialize)
- Python 环境：`~/mpk-env` 已有 torch 2.8 和 triton 3.4，可以直接复用。记录所有版本号。

实测硬件参数（2026-09-22）：

| 参数 | 值 |
|---|---|
| 计算能力 | 12.0（sm_120） |
| SM 数 | 188 |
| 每 SM shared memory | 100KB（每 CTA 最多 99KB） |
| 每 SM 寄存器 | 64K |
| 每 SM 线程数 / CTA 数上限 | 1536 / 24 |
| L2 | 128MB（persisting 区上限 80MB） |
| 显存 | 96GB GDDR7，512-bit |
| 功耗上限 | 600W |
| green context | 可用；SM 划分粒度 8 个（请求 94 个得到 96 个） |

### 1.2 测量协议

- **锁频**：锁定 SM 频率（例如之前 MPK 实验用的 2800 MHz）。测量期间用 NVML 每 10ms 采样一次功耗与实际频率；报告时注明是否锁频。
- **L2**：128MB 的 L2 装得下整块 4096×4096 fp16 权重（32MB）。不加处理的话，重复测量测到的是 L2 命中时的性能，与 serving 中权重从 DRAM 流入的情况不符。默认做法是让输入在多份副本间轮换，总大小超过 L2 的 2 倍；需要测"热"场景时单独标注。
- **计时**：
  - 单 kernel：CUDA event 计时，用 CUDA Graph 捕获多次调用，摊薄启动开销；
  - 多 stream：记录一个公共起始 event 和两条流各自的结束 event，取最后完成的时刻；
  - CoKernel：每个角色的最后一个 tile 完成时，由执行它的 CTA 写入 `%globaltimer`，从而得到各角色的完成时间和整体完成时间。
- **重复**：预热 ≥ 20 次，测量 ≥ 50 次，报告中位数和 p10/p90；变异系数超过 2% 的点重测。
- **稳态模式**：两个角色各自循环执行，在固定时间窗口内统计完成次数，得到各自的进度率。

### 1.3 资源签名采集

- **编译期**：寄存器数（每线程）、每 CTA 的 shared memory（静态 + 动态）、线程数，据此计算理论占用率。
- **Nsight Compute**：DRAM 吞吐、L2 吞吐与命中率、Tensor pipe 利用率、LSU/MIO 利用率、发射槽利用率、achieved occupancy。
- **SM 数曲线**：用 green context 把 SM 数限制为 s ∈ {16, 32, …, 188}，测单跑时长 t(s)。

### 1.4 压力 kernel

用于刻画敏感度，每种取 3–4 档强度：
- **DRAM 流**：顺序读写，用 CTA 数控制占用的带宽比例；
- **L2 抖动**：工作集与 L2 同量级的随机访问；
- **Tensor Core 满载**：只在寄存器里循环执行 mma；
- **SM 资源占位**：占住指定大小的 shared memory 和寄存器后自旋等待，模拟"伙伴占去多少容量"；
- **ALU 满载**：FMA 循环，占用发射槽。

### 1.5 `%smid` 探测

写一个探测 kernel，确认三件事：`%smid` 的取值范围和空洞（与 `%nsmid` 对照）；持久化 grid 下 CTA 在各 SM 上的分布；据此建立的 SM 编号重映射表（SM 级绑定要用）。

### 1.6 产出

- `research/bench/`：测量框架、压力 kernel、探测 kernel；
- `research/bench/README.md`：测量协议；
- `research/results/<日期>_<实验名>/`：原始数据与汇总。

## 2. P1：动机与极限研究（RQ1）

### 2.1 算子与形状

| 算子 | 形状（初选） | 主要资源 |
|---|---|---|
| GEMM（bf16，fp32 累加） | M ∈ {2048, 4096, 8192}（token 数）；(N, K) ∈ {(4096, 4096), (14336, 4096), (4096, 14336)} | Tensor Core |
| GQA decode attention | batch ∈ {16, 32, 64, 128}；KV 长度 ∈ {2K, 8K, 32K}；H_q = 32，H_kv = 8，D = 128 | DRAM |
| RMSNorm | token ∈ {4K, 16K, 64K}；hidden ∈ {4096, 8192} | DRAM |
| GEMV / 小批量 GEMM | batch ∈ {1, 4, 16}；权重 4096 × 14336 | DRAM |
| prefill attention | 序列长度 ∈ {2K, 8K}；H = 32，D = 128，causal | Tensor Core |

### 2.2 算子对

| 编号 | 算子对 | 类型 | 角色 |
|---|---|---|---|
| P1 | GEMM × decode attention | 计算 × 访存 | 主实验 |
| P2 | GEMM × RMSNorm | 计算 × 访存（短） | 主实验 |
| P3 | GEMM × GEMV | 计算 × 访存 | 主实验 |
| P4 | prefill attention × decode attention | 计算 × 访存 | 与 POD 的结论对照 |
| P5 | GEMM × GEMM | 计算 × 计算 | 对照组，预期收益小 |
| P6 | decode attention × RMSNorm | 访存 × 访存 | 对照组，预期收益小 |

选取的形状组合要让两者的单跑时长比落在 0.5–2 之间。P1 另外扫描时长比 {0.25, 0.5, 1, 2, 4}。

### 2.3 配置空间

以 TileLang 示例为基础，每个算子整理 30–50 个有效配置：
- **GEMM**：block_M/N/K、流水级数、线程数、光栅化 / swizzle、split-K；
- **decode attention**：KV 块大小、每 CTA 处理的 head 数、split-KV 份数、线程数、流水级数；
- **RMSNorm**：每 CTA 行数、线程数、向量宽度；
- **GEMV**：块大小、split-K；
- **prefill attention**：block_M/N、流水级数、线程数。

### 2.4 共置方式

- **M0**：串行。
- **M1**：多 stream。
- **M1+**：多 stream + 共跑感知配置。使用持久化 grid、限制占用率，让两个 kernel 的 CTA 能在同一 SM 上共驻。
- **M2**：green context SM 分区，划分比例以 8 个 SM 为步长扫描。
- **M3**：手写 CoKernel 原型。
  - P1 做全部三种绑定层级：先 SM 级（最简单，用来验证分派与测量），再 CTA 级（POD 式），最后 warp 级（顺带验证 `ThreadPartialSyncRewriter` 生成的 named barrier）。
  - P2 至少做 SM 级和 CTA 级。

手写原型同时是 P2 编译器的"金标准"，也用来确定 TileLang 需要补哪些能力。

### 2.5 要算的量

按 proposal 第 4 节的 2×2 设计计算：
- **T[solo, inter]**：单跑最优配置在 M1/M2 下的最好结果。
- **T[co, inter]**：在配置乘积空间上穷举 M1、M1+、M2，取最好结果。这些组合复用已编译的 kernel，只需编译 N_A + N_B 个。
- **T[solo, intra]**：把单跑最优配置放进 CoKernel（只做放进同一 kernel 所必需的最小调整），取最好编排下的结果。
- **T[co, intra]**：在裁剪后的子空间上穷举 CoKernel。每个算子取 10–15 个配置，从"单跑时间 × 资源占用"的 Pareto 前沿上挑选。
- **参照**：T_serial、LB。

### 2.6 归因分析

对 T[co, ·] 选中的配置，报告它在单跑排名中的位置、单跑慢了多少，并借助资源签名和 Nsight 数据归因到具体效应：SM 内容量、波次、资源耦合、DRAM/L2 争用、pipe 争用。挑 2–3 个典型案例做完整剖析。

### 2.7 产出

- 图 1：每个算子对的 2×2 收益分解；
- 图 2：共跑最优配置在单跑排名中的位置与单跑减速；
- 表 1：收益归因；
- 阶段小结写入 [status.md](status.md)。

### 2.8 决策点 D1

- **继续**：P1–P4 中至少 2 对满足下列任一条件：
  - T[solo, inter] / T[co, inter] ≥ 1.10（共跑感知实现的价值）；
  - T[co, inter] / T[co, intra] ≥ 1.10（kernel 内编排的价值）。
- **收窄**：收益只出现在特定形状或时长比下。把范围缩小到这些场景，并刻画收益出现的条件。
- **转向**：收益普遍低于 5%。可选方向：
  - 提前进入子图级（P5），检验改变结构是否带来更大收益；
  - 做成以刻画为主的工作；
  - 转向小批量、占不满 GPU 的场景。

阈值是初设值，可以修订，但必须在看到 P1 数据之前定下来。

## 3. P2：CoKernel 编译器 MVP（RQ2、RQ3）

### 3.1 语义描述 v0

- 支持 GEMM/GEMV、GQA decode attention、prefill attention、RMSNorm 四类操作，描述形式见 proposal 5.1。
- 从描述自动生成 PyTorch 参考实现，用于正确性测试。

### 3.2 候选推导

- 按规则生成 TileLang tile 程序：并行轴切分、归约轴的三种处理、调度参数。
- 给定 CTA 形状和资源预算，推导出符合该形状的 tile 程序。例如 256 线程下，decode attention 让每个 CTA 处理两倍的 head。
- 为每个候选计算资源签名。

### 3.3 CoKernel IR 与降级

- 每个角色的 tile 计算体生成为 `T.macro`。
- 分派循环：
  - 静态分派：grid-stride；
  - 动态队列：用 `T.atomic_add(return_prev=True)` 取 tile 编号，经 shared memory 广播给整个 CTA。
- SM 级：新增 `%smid`/`%nsmid` 的读取。可以先用 `T.call_extern` 调用一个内联 PTX 的设备函数，稳定后再做成 intrinsic。
- warp 级：用 `T.ws` 划分角色。

需要补充或验证的 TileLang 能力（按风险从高到低）：
1. **角色内同步。** [thread_storage_sync.cc](../src/transform/thread_storage_sync.cc) 中的 `ThreadPartialSyncRewriter` 已能把线程范围受限区域内的同步改写成 named barrier（要求线程数是 warp 的整数倍）。需要验证两件事：两个角色都跑完整 tile 算子（gemm、reduce、copy）时，改写仍然正确；两个角色和 `T.gemm` 内部用到的 barrier ID 不冲突（每个 CTA 只有 16 个 named barrier）。
2. **线程子集上的 layout 推断。** 已有示例中 `T.gemm` 能在 `T.ws(1)` 内工作；还需要 `T.reduce`、`T.copy`、`T.Parallel` 在任意 warpgroup 子集上都能正确推断。
3. **跨角色的 shared memory 复用。** CTA 级下两个角色的缓冲区应能互相复用，warp 级下应按角色划分。需要确认 shared memory 合并分配的 pass 在分支之间的行为。
4. **完成时间打点与计数器复位。** 各角色完成时写 `%globaltimer`。原子计数器由最后完成的 CTA 复位，或在每次启动前由 host 清零。

### 3.4 调度策略

实现五种策略：静态、动态队列（可以一次取一块 tile）、跨角色接手、优先级、配比控制。策略参数全部做成运行时参数。

### 3.5 开销测量

- **分派开销**：每取一个 tile 的原子操作延迟，以及一次取一块时的摊薄效果。
- **双角色的副作用**：寄存器数的变化和指令 cache 压力。对照方法：把另一个角色编进 kernel 但不执行，与单角色 kernel 比较。
- **同步开销**：各绑定层级额外引入的同步开销。

### 3.6 测试

- 两个输出都与参考实现比对；
- E0 变体与同配置的单跑 kernel 做逐位比较；
- 动态队列压力测试：不丢 tile、不重复执行 tile，重复启动后计数器状态正确。

### 3.7 出口标准

在 P1、P2 两对上，自动生成的 CoKernel 与手写原型的性能差距 ≤ 5%；三种绑定层级至少在两个算子对上可用。

## 4. P3：搜索与成本控制（RQ4）

1. **逐个算子刻画**：单跑时长、t(s) 曲线、敏感度向量；记录每个候选的刻画成本。
2. **性能模型**：
   - v0 解析模型：各角色吞吐 × 可用槽位 × 争用系数；
   - v1 tile 级离散事件模拟：每 SM 的槽位、依赖共驻情况的 tile 时长、调度策略；
   - 用 P1 的穷举数据评估两者的排序准确度（Kendall τ）。
3. **搜索流水线**：Pareto 剪枝 → 模型排序 → 编译排名前 k 的候选（grouped compilation）→ 实测 → 对运行时参数做一维搜索。
4. **对比的搜索方法**：
   - 穷举（在裁剪后的空间上，作为真值）；
   - 同等预算下的随机搜索；
   - 只用单跑最优配置；
   - "资源形状对齐"启发式；
   - 贝叶斯优化（可选）。
5. **指标**：regret 随成本（编译次数、GPU 时间）变化的曲线；达到最优 5% 以内所需的成本。
6. **规模论证**：选一个目标场景（例如 Llama-3-8B 的 prefill/decode 混合 serving），统计"（算子, 形状分桶）配对"的实例数，乘以单个实例的穷举成本，与本方法的成本对比。

出口标准：至少 4 个算子对上，以 ≤ 10% 的穷举成本达到最优的 5% 以内。

## 5. P4：系统评估与论文

- **扩展**：更多形状；数据类型加入 fp8；更多算子对。
- **案例研究**：
  - LLM serving 的 prefill/decode 混合批，与 POD 对比。先确认 POD（FlashInfer 实现）能否在 sm_120 上运行；不能的话，在本框架中复现 POD 的编排策略作为基线。
  - MoE：共享专家 GEMM ∥ 路由专家 grouped GEMM。
  - 多租户：两个不同模型的算子对。
- **基线**：
  - 多 stream；
  - green context；
  - M1+；
  - HFuse 式融合（相当于 warp 级绑定且不重新 tile）；
  - POD；
  - MPK（之前已在 PRO 6000 上跑过，脚本在 `~/fuser/pdl-comparision`）；
  - Rammer 式的静态编排加固定实现（以消融形式实现）。
- **消融**：不做语义重 tile；只用静态调度；逐一比较各绑定层级；不用模型；确定性模式。
- **跨架构**：若能拿到 H100 或 B200，复现关键实验。

## 6. P5：子图级扩展（长期）

### 6.1 先验证值得做

用 TileLang 手写 2–3 个子图的结构变体：
- **GatedMLP**：融合 / 不融合；
- **GQA decode**：split-KV 取 1/2/4/8，再加 combine；
- **RMSNorm+Linear**：分开 / 融合进 prologue / 把除法移到 matmul 之后（在同一个 K 循环里顺带累加 Σx²）。

每个变体分别配两类伙伴（吃 DRAM 的、占 SM 内资源的），检验 proposal 第 10 节的 H1–H3。

### 6.2 接入 Mirage

本地代码在 `~/mirage-compiler`，初步检查发现以下问题：
- `python/mirage/utils.py` 的 `get_shared_memory_capacity` 没有 cc=120 的分支，会直接 assert 失败（sm_120 应为 99KB）。
- 转译器的架构判断混用 `== H100` 与 `>= H100`（如 `src/transpiler/transpiler_kn.cc` 第 340、710 行），sm_120 会同时走进 Hopper 路径和通用路径。sm_120 没有 wgmma，应显式走 Ampere 的 mma.sync 路径，或者干脆不用它的代码生成。
- `superoptimize()` 在 cc ≥ 90 时为每个 μGraph 枚举流水级数 × warp group 数，但 `compile()` 会在同一个图对象上缓存结果，`self.run` 也会被后编译完的变体覆盖，所以每个 μGraph 实际只测到一个变体。
- 搜索时的 shared memory 上限是编译期常量 `MAX_SMEM_SIZE = 96KB`（`include/mirage/config.h`），用于构造 threadblock 算子时剪枝；做预算约束需要改成运行时参数。

可以直接利用的地方：
- `search()` 返回全部 μGraph（`python/mirage/kernel.py` 第 594 行），最后才按单跑时延取最优（第 688 行）；
- 搜索结果会缓存成 `mirage_cached_mugraphs_*.json`，可以直接导出候选池。

### 6.3 μGraph → TileLang 降级

| μGraph | TileLang |
|---|---|
| grid dims | `T.Kernel` |
| for-loop + fmap | `T.Pipelined` + `T.copy` 的切片 |
| imap / omap | 全局下标 |
| block 级 matmul | `T.gemm` |
| 归约 | `T.reduce_*` |
| 逐元素运算 | `T.Parallel` |

### 6.4 两个子图的 CoKernel 编排

把两个子图的 tile 程序交给第 3 节的 CoKernel 编排。

## 7. 决策点与开放问题

| 编号 | 问题 | 何时决定 | 当前倾向 |
|---|---|---|---|
| D1 | 继续 / 收窄 / 转向 | P1 结束 | 标准见 2.8 |
| D2 | 绑定层级的优先顺序 | P1 结束 | SM 级 → CTA 级 → warp 级 |
| D3 | 目标函数：完成时间、带 SLO、稳态吞吐 | P1 期间 | 以完成时间为主，另报稳态吞吐 |
| D4 | 是否允许 E1 参数随伙伴变化 | P2 开始前 | 允许，同时报告确定性模式 |
| D5 | P4 的案例场景 | P3 结束 | prefill/decode + 一个结构不同的场景 |
| D6 | 目标会议与时间线 | 尽早 | 待定 |

## 8. 代码与数据组织（建议）

- `research/`：proposal.md、plan.md、status.md；
- `research/bench/`：测量框架、压力 kernel、探测 kernel；
- `research/results/`：实验数据，每个实验一个目录；
- 编译器代码放在独立的包里（例如顶层的 `cotile/`），依赖 tilelang。只有必须进入 TileLang 本体的能力（如 `%smid` intrinsic、角色内同步）才改 `tilelang/` 与 `src/`，便于与上游同步。

## 9. 最近两周的任务

- [ ] 编译 co-tilelang（sm_120），跑通 gemm、GQA decode、RMSNorm、warp_specialize 示例
- [ ] 测量框架 v0：event 计时、CUDA Graph、输入轮换、锁频、NVML 采样
- [ ] `%smid` 探测 kernel
- [ ] 压力 kernel v0：DRAM 流、Tensor Core 满载、SM 资源占位
- [ ] 为 GEMM、decode attention、RMSNorm 各整理 30–50 个配置，并完成单跑 profiling
- [ ] P1、P2 两对在 M1、M1+、M2 下的配置乘积空间共跑
- [ ] P1 的 SM 级 CoKernel 手写原型
