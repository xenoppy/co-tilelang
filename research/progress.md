# 进展记录

> 按时间倒序追加。每条记录：时间、现状、结论、关注的问题、下一步。无效尝试记在对应条目下的"回退"小节。

## 2026-09-22 19:30 — 启动 P0

**现状**
- 仓库 `explore` 分支，`research init` 已提交；尚无 `build/`。
- GPU 空闲（nvidia-smi 19:25）。CUDA 12.9 在 `/usr/local/cuda-12.9`（不在 PATH）；`~/mpk-env`：torch 2.8.0+cu128、triton 3.4.0。
- 磁盘只剩约 12GB（/ 占用 100%），编译与数据存放要节省空间。
- 子 agent 定义：`.claude/agents/researcher.md`（opus，effort xhigh；被 .gitignore 忽略）。

- 子 agent 实际调用方式：自定义 agent 类型在会话中途创建后无法加载，改用 `general-purpose` + `model: opus`，在 prompt 里写明规则与"extra-high effort"（Agent 工具无法直接设 effort）。
- 无免密 sudo：不能锁频，测量协议改为 NVML 记录频率 + 交错测量（已写入 plan）。
- `/data`（916G NVMe）可放编译中间产物和数据（用户 19:40 告知），但目前归 root 所有、ywc 不可写，需要用户建一个自己的子目录。

**P0-1 编译与冒烟测试：已完成（20:05 验收）**
- 完成标准：任意目录下 `import tilelang` 可用（`source research/env.sh`）；gemm / GQA decode / RMSNorm / 一个 warp_specialize 示例在 GPU 上通过自带正确性检查；版本记录在 `research/env_versions.md`。
- 结果：开发模式编译（`build/`，ninja，编译约 95 s，总占用约 790MB）。子模块只浅克隆了 tvm、tvm-ffi、cutlass；TVM 下四个不需要的子模块在本地 git config 设为 `update=none`。mpk-env 新装 apache-tvm-ffi 0.1.12 等运行时依赖；z3 保持 4.16（mirage 依赖它）。
- 示例：gemm 1024³ fp16 通过；RMSNorm 8192² 通过（0.36 ms，约 1.49 TB/s）；warp_specialize 五个 GEMM 示例通过（16384³ 约 284 TFLOPS）；flashmla 用 wgmma，sm_120 不支持，符合预期。
- GQA decode 示例默认配置需要 144KB shared memory，超过 sm_120 每 CTA 99KB 上限，启动时报错；用包装脚本换成 block_N=64/stage=2 或 block_N=128/stage=1 后通过。根因是示例的配置启发式只按 sm_89 特判，应该按设备 shared memory 上限选配置（未改上游示例）。
- 有用的发现：
  - sm_120 上 TileLang 会把 `threads=128` 的 gemm 自动改成 256 线程的 warp specialization（128 线程 TMA 生产者 + 128 线程 mma.sync 消费者）。**资源签名必须从编译产物读取，不能用源码参数。**
  - shared memory 超限不会在编译期报错，只在启动时失败；配置生成器要自己检查 101376 B 上限。
  - 上游 CI 对 flash decoding 只测 cc ≤ 8.9，对 warp_specialize 只测 cc == 9.0，sm_120 路径缺少测试覆盖。

**TileLang 能力调研：已完成（20:25）**，报告 `research/notes/tilelang_capability_survey.md`。结论：
- 已有：运行时角色分支（layout 推断、流水、同步插入都能处理 uniform 条件）；`T.Pipelined` 可嵌在 for/if/while 内；`atomic_add(return_prev=True)` 单元素；`T.sync_threads`、named barrier；`T.Persistent`/`loop_break`；`T.ws` 在连续线程子区间上可用 gemm/copy/reduce/parallel；`T.annotate_min_blocks_per_sm` 给出寄存器上限；`par_compile` 批量编译。
- 缺口（按风险）：① sm_120 默认开启自动 warp specialization，会破坏双角色 kernel → CoKernel 一律 `tl.disable_warp_specialized=True`；② shared memory 分配把两个角色的缓冲区都算成整个 if/while 内存活，**两角色 smem 相加、不复用**，CTA 级共驻前必须解决；③ warp 级下跨 warp 的 `T.reduce` 固定用 barrier 1、2，两个角色同时归约会冲突；④ while 循环里没有跨迭代的同步分析；⑤ `%smid`/`%globaltimer` 缺失，可用 `T.Kernel(prelude=...)` + `T.call_extern` 补；⑥ 没有寄存器/smem 查询 API，用 cuobjdump 或 ptxas -v。

**相关工作扫描：已完成（20:50）**，报告 `research/notes/related_work_scan.md`（约 32 项，8 篇精读）。结论与处理：
- 没有工作完整覆盖本研究的主张，但各部件都已单独存在：POD（CTA 级 `%smid` 绑定，手工改实现）、HFuse（warp 级 + 线程划分 / 寄存器上限搜索）、Rammer（多个算子的 tile 共驻同一 SM，库内多版本按 profiling 挑选，多余线程空转）、MPK / Event Tensor / Hazy（SM 级 worker 与动态队列）、mKernel（运行时调角色 SM 划分）、GOLDYLOC / NanoFlow（为共同执行调优实现，跨 kernel）。
- proposal v0.1 对 Rammer、HFuse、MPK 的描述有误或不完整，已修正。
- 仍然空白：共同资源契约下从语义重新推导实现；覆盖三级绑定与多种调度、让已有设计成为其中的点的统一空间；有成本控制的联合搜索；sm_120 上的刻画；与伙伴无关的数值。
- 风险：会被读成"自动化的 POD"；MPK / Mirage 与 Event Tensor 团队最可能先做出"每 SM 两个共驻 worker"。
- 处理：proposal v0.2（658a0019）把核心主张改为"为共同资源契约重新推导实现"，2×2 扩展为 3×2（solo / lib / derived × inter / intra）；plan v0.2（8e10db30）据此修订了 P1 的度量、基线（空转线程、虚拟 CTA、GOLDYLOC 式预算调优、FlashInfer POD）和 D1（在看到任何 P1 数据之前）。
- 附带发现：FlashInfer POD 按 SM 数开计数器数组，而 PTX 规定 `%nsmid` 可能大于 SM 数；本机 188/192 SM，可能越界，使用前要先看探测结果。

**进行中（附完成标准）**
- P1 准备：算子库 v0（`cotile/ops/`：GEMM、GQA decode、RMSNorm）。完成标准：C1 每个算子有 tile_body macro + 普通 grid 版 + 持久化版 + 配置枚举 + 资源签名；C2 每个算子 ≥30 个配置在测试形状上两种版本都通过正确性检查；C3 split-K GEMM 与 split-KV decode 在 kernel 内合并（最后到达的 tile 做合并）并通过重复启动测试；C4 资源签名写入 `research/results/2026-09-22_op_library/signatures.csv`；C5 `cotile/README.md`。
- P0-2 测量框架 v0（`research/bench/cobench/`）。完成标准：单 kernel 计时三种模式（刷 L2 的 event 计时、CUDA Graph + 输入轮换、热模式）；双流共跑计时（公共起点、各自完成时刻）；NVML 采样器；green context 流封装并实测 SM 粒度；`%smid` 探测（取值与空洞、持久化 grid 的 CTA 分布、green context 下用到哪些 SM、`%globaltimer` 分辨率）；matmul 重复 5 次的 CV < 2%（或给出原因）；README。

**关注的问题**
- 基线强度：sm_120 上 TileLang GEMM 走 mma.sync（无 wgmma/tcgen05），单跑性能若明显低于 cuBLAS，共置收益会被"低效 kernel 留下的空闲资源"虚增。P1 必须同时报告 cuBLAS / FlashInfer（或 torch SDPA）单跑时间作为参照，并在 2×2 分解里用最强的单跑实现作为 solo 基线。
- 不能锁频：共跑时功耗更高，可能比单跑更早降频，会低估共置收益或引入噪声；需要在结果里报告每组的频率分布。
