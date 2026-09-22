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
- `/data`（916G NVMe）可放编译中间产物和数据（用户约 19:30 告知），但目前归 root 所有、ywc 不可写，需要用户建一个自己的子目录。

**P0-1 编译与冒烟测试：已完成（19:41 验收）**
- 完成标准：任意目录下 `import tilelang` 可用（`source research/env.sh`）；gemm / GQA decode / RMSNorm / 一个 warp_specialize 示例在 GPU 上通过自带正确性检查；版本记录在 `research/env_versions.md`。
- 结果：开发模式编译（`build/`，ninja，编译约 95 s，总占用约 790MB）。子模块只浅克隆了 tvm、tvm-ffi、cutlass；TVM 下四个不需要的子模块在本地 git config 设为 `update=none`。mpk-env 新装 apache-tvm-ffi 0.1.12 等运行时依赖；z3 保持 4.16（mirage 依赖它）。
- 示例：gemm 1024³ fp16 通过；RMSNorm 8192² 通过（0.36 ms，约 1.49 TB/s）；warp_specialize 五个 GEMM 示例通过（16384³ 约 284 TFLOPS）；flashmla 用 wgmma，sm_120 不支持，符合预期。
- GQA decode 示例默认配置需要 144KB shared memory，超过 sm_120 每 CTA 99KB 上限，启动时报错；用包装脚本换成 block_N=64/stage=2 或 block_N=128/stage=1 后通过。根因是示例的配置启发式只按 sm_89 特判，应该按设备 shared memory 上限选配置（未改上游示例）。
- 有用的发现：
  - sm_120 上 TileLang 会把 `threads=128` 的 gemm 自动改成 256 线程的 warp specialization（128 线程 TMA 生产者 + 128 线程 mma.sync 消费者）。**资源签名必须从编译产物读取，不能用源码参数。**
  - shared memory 超限不会在编译期报错，只在启动时失败；配置生成器要自己检查 101376 B 上限。
  - 上游 CI 对 flash decoding 只测 cc ≤ 8.9，对 warp_specialize 只测 cc == 9.0，sm_120 路径缺少测试覆盖。

**TileLang 能力调研：已完成（19:46）**，报告 `research/notes/tilelang_capability_survey.md`。结论：
- 已有：运行时角色分支（layout 推断、流水、同步插入都能处理 uniform 条件）；`T.Pipelined` 可嵌在 for/if/while 内；`atomic_add(return_prev=True)` 单元素；`T.sync_threads`、named barrier；`T.Persistent`/`loop_break`；`T.ws` 在连续线程子区间上可用 gemm/copy/reduce/parallel；`T.annotate_min_blocks_per_sm` 给出寄存器上限；`par_compile` 批量编译。
- 缺口（按风险）：① sm_120 默认开启自动 warp specialization，会破坏双角色 kernel → CoKernel 一律 `tl.disable_warp_specialized=True`；② shared memory 分配把两个角色的缓冲区都算成整个 if/while 内存活，**两角色 smem 相加、不复用**，CTA 级共驻前必须解决；③ warp 级下跨 warp 的 `T.reduce` 固定用 barrier 1、2，两个角色同时归约会冲突；④ while 循环里没有跨迭代的同步分析；⑤ `%smid`/`%globaltimer` 缺失，可用 `T.Kernel(prelude=...)` + `T.call_extern` 补；⑥ 没有寄存器/smem 查询 API，用 cuobjdump 或 ptxas -v。

**相关工作扫描：已完成（19:53）**，报告 `research/notes/related_work_scan.md`（约 32 项，8 篇精读）。结论与处理：
- 没有工作完整覆盖本研究的主张，但各部件都已单独存在：POD（CTA 级 `%smid` 绑定，手工改实现）、HFuse（warp 级 + 线程划分 / 寄存器上限搜索）、Rammer（多个算子的 tile 共驻同一 SM，库内多版本按 profiling 挑选，多余线程空转）、MPK / Event Tensor / Hazy（SM 级 worker 与动态队列）、mKernel（运行时调角色 SM 划分）、GOLDYLOC / NanoFlow（为共同执行调优实现，跨 kernel）。
- proposal v0.1 对 Rammer、HFuse、MPK 的描述有误或不完整，已修正。
- 仍然空白：共同资源契约下从语义重新推导实现；覆盖三级绑定与多种调度、让已有设计成为其中的点的统一空间；有成本控制的联合搜索；sm_120 上的刻画；与伙伴无关的数值。
- 风险：会被读成"自动化的 POD"；MPK / Mirage 与 Event Tensor 团队最可能先做出"每 SM 两个共驻 worker"。
- 处理：proposal v0.2（658a0019）把核心主张改为"为共同资源契约重新推导实现"，2×2 扩展为 3×2（solo / lib / derived × inter / intra）；plan v0.2（8e10db30）据此修订了 P1 的度量、基线（空转线程、虚拟 CTA、GOLDYLOC 式预算调优、FlashInfer POD）和 D1（在看到任何 P1 数据之前）。
- 附带发现：FlashInfer POD 按 SM 数开计数器数组，而 PTX 规定 `%nsmid` 可能大于 SM 数；本机 188/192 SM，可能越界，使用前要先看探测结果。

**P0-2 测量框架 v0 + `%smid` 探测：已完成（20:14 验收，主 agent 复跑 `research/bench/tests/test_cobench.py` 9/9 通过）**
- API（`research/bench/cobench`，文档 `research/bench/README.md`）：`bench(fn, mode=flush|graph|hot)`、`bench_corun`（公共起点，solo/serial 在每轮内交错测量，发射顺序 ab/ba 交替）、`NvmlSampler`、`split_sms`（green context → torch ExternalStream）、`build_sm_remap`、`CudaKernel`（NVRTC，可在普通流和 green 流上启动）。额外加了"主机闸门"（计时区间不含主机发射延迟）和 GPU 内时钟探针（每 50µs 记录 SM 时钟）。
- 关键数字：bf16 dense 峰值 1024 FLOP/clk/SM，即 2617 MHz 时 504 TFLOP/s；持续负载下 600W 功耗墙把频率压到约 2.18–2.35 GHz，实际峰值约 430–450。torch matmul 4096³ 约 360–390 TFLOP/s（同频率下达峰值的 79–86%）。DRAM 拷贝约 1.46 TB/s（理论 1.79）。跨 5 个进程的 CV 0.17%。
- `%smid`：`%nsmid` = 188，编号 0..187 无空洞（全卡重映射为恒等）；第一波 188 个 CTA 必定落在 188 个不同 SM；块→SM 映射确定但不按 smid 顺序（TPC 成对、8 组轮转）；green context 内 `%nsmid` 仍为 188、编号为物理编号，分区要自己建表。**FlashInfer POD 的 `%nsmid` 越界风险在本机不存在。**
- green context：默认粒度 8 个 SM 且分散在 4 组；加 `IGNORE_SM_COSCHEDULING` 后粒度 2、从 SM 0 连续、数量精确。
- `%globaltimer`：分辨率 32ns、单调，SM 间偏差 ≤ 64ns，可作为 CoKernel 各角色完成时间的设备级时间戳。
- 教训：NVML 在本驱动上约 500ms 才刷新一次，10ms 轮询没有意义（改用 GPU 内时钟探针；功耗可用 `nvmlDeviceGetSamples` 20ms 采样）；功耗墙在负载开始 0.1–0.3s 后降频，预热必须 ≥ 1s；不同模式下频率不同，跨模式比较要用 cycles 而不是 µs。
- 初步观察（非正式）：matmul ∥ copy 在两条普通 stream 上完全串行（加速比 0.998，copy 总是先完成）；green context 94/94 划分为串行时间的 0.81×。说明 M1 基线可能很弱，M2 才是真正要赢的跨 kernel 基线。

**P1 准备：算子库 v0（`cotile/`）：已完成（20:59 验收，主 agent 复跑 `python -m cotile.tests.test_ops` 全部通过；提交 8e60a158）**
- 完成标准：C1 每个算子有 tile_body macro + 普通 grid 版 + 持久化版 + 配置枚举 + 资源签名；C2 每个算子 ≥30 个配置在测试形状上两种版本都通过正确性检查；C3 split-K GEMM 与 split-KV decode 在 kernel 内合并（最后到达的 tile 做合并）并通过重复启动测试；C4 资源签名写入 `research/results/2026-09-22_op_library/signatures.csv`；C5 `cotile/README.md`。
- 结果：GEMM 43、decode 36、RMSNorm 39 个配置全部正确；grid 版与持久化版逐位相同；同一数值键的配置逐位相同（GEMM 的 35 个 split-K=1 配置跨 tile 形状、流水级数、线程数、epilogue、warp specialization 都逐位相同，E0 成立）；split 配置在 kernel 内按固定顺序合并，结果确定。236 个 kernel 冷编译 81.5s（32 并行）。资源签名从 cubin 读取，寄存器与每 SM CTA 数与驱动交叉验证一致。
- 发现：
  - **编译器数值问题**：TileLang 生成的成对 fp32 `mul/sub.rn.f32x2`，在 sm_120a 上会被 ptxas 融合成 FFMA（即使 `-fmad=false`），融合哪一个操作数取决于上下文代码。decode 不同 num_stages 之间差 1–3 个 fp32 ulp。子 agent 在 decode tile 里改写为 `exp2((s−m)·scale)` 规避。**这直接威胁"与伙伴无关的数值"：两个角色编进同一 kernel 后，上下文变化可能改变融合方式。** 需要在 CoKernel 中验证，必要时在 TileLang 里修（`common.h` 对 sm_12x 用标量 `__fmul_rn/__fsub_rn`，或查 #3128 的 PassConfig）。
  - decode 在 ≥8 warp 且每 warp ≥16 列时，TileLang 的 `T.reduce_*` lowering 失败（配置被 validate 拒绝）。
  - `tl.disable_tma_lower=True` 会让 GEMM smem epilogue 的 lowering 崩溃。
  - 持久化版的 shared memory 是所有 scratch 之和（阶段间不复用），19 个 GEMM 配置因此每 SM 少一个 CTA；寄存器也上升（decode 最多 +48，RMSNorm 181→247）。
  - 寄存器耦合会很严重：GEMM 最高 254，decode 56–206，RMSNorm 30–247。
- 技术债：`DeviceSpec` 默认值写死本机参数，应加 `from_current_device()`；`configs()` 的 smem 上限默认值应取自 DeviceSpec。

**P1 基线：FlashInfer 0.7.0：已完成（20:55）**，结果 `research/results/2026-09-22_flashinfer_baselines/`，封装 `research/bench/baselines/flashinfer_ops.py`。
- 安装不改动已有包版本（新增 15 个包，约 0.93GB；JIT 缓存 `~/.cache/flashinfer/0.7.0/120f/`）。decode、prefill、POD 在 sm_120 上都能编译且正确；POD 输出与单独的 FlashInfer kernel 逐位相同。
- GQA decode：1370–1636 GB/s（graph 模式达 DRAM 峰值 84–91%）。TileLang 示例用最好的 ≤99KB 配置（block_N 128、1 stage、不 split）与 FlashInfer 相差约 1%（B16/KV2048 差 5%）。prefill：S=8192 为 331 TFLOP/s（同频率 mma.sync 峰值的 70%）。
- **FlashInfer POD 的 bug**：`pod.cuh:413-415` 用默认流上的 `cudaMemset` 复位调度计数器，而 kernel 跑在 torch 当前流上 → 在非默认流或 green context 流上结果错误；CUDA graph 下第二次回放起不写任何输出（会测出假的约 100× 加速）。封装里禁止了这两种用法；POD 只在默认流 + flush 模式下测。可向上游报告。
- POD（prefill S ∈ {2048, 8192} × decode B ∈ {16, 64} × KV ∈ {2048, 8192}）相对串行 1.02–1.38×；两条 stream 1.04–1.23×；事后挑出的最好 green context 划分 0.97–1.31×。POD 相对最好划分：短 prefill 时基本持平，平衡的重负载情形快 7–15%。划分选错最多慢 5×。
- **功耗墙是共享资源**：张量核与 DRAM 同时忙时 600W 功耗墙压低频率。最重的情形 POD 跑在 2066 MHz，串行为 2606 MHz，cycles 上 1.74× 的优势变成时间上 1.38×。含义：(1) 资源下界 LB 与性能模型必须把功耗（或频率）作为一种资源；(2) 结果要同时报时间与频率；(3) 本平台上共置收益的上限低于只看 SM / DRAM 资源的估计。
- 寄存器耦合可见：POD 里 decode 角色继承 prefill 的 255 寄存器（单跑时 134），但 shared memory 已把两者都限制在每 SM 2 个 block，所以这里没有额外代价。

**P1-M3a CoKernel 构建器 v0（原型）：已完成（22:20 验收；提交 205528ca）**
- 完成标准：K1 通用双角色持久化 kernel（SM 级 / CTA 级绑定，静态 / 动态队列，chunk，跨角色接手，SM 划分与配比为运行时参数）；K2 两角色 scratch 复用（每 CTA smem ≈ max 而非 sum）；K3 `%smid`/`%globaltimer` 与设备端各角色完成时间戳；K4 两个输出正确且与同配置单跑持久化版逐位相同，tile 不丢不重，重复启动无需主机清零；K5 开销测量；K6 README 与生成代码观察。
- 验收：子 agent 的测试 184/184 个可启动设置全部通过（7 个配置对 × 两种绑定 × 两种调度 × 接手开关 × 3–5 种 SM 划分 / 配比，每个启动 3 次，与单跑持久化版逐位相同，每个 tile 恰好执行一次）；3 个不可启动的是故意保留的 `smem="sum"` 对照（173KB）。主 agent 复跑编译部分（92 个 kernel 全部通过）与 lifetime-scope 回归测试、已有 smem 合并测试（全部通过）；GPU 部分因 profiling 占用 GPU，推迟到 profiling 结束后复跑确认。
- 设计：每轮分派由 thread 0 决定（角色, tile），写入双缓冲的共享槽，一次 `__syncthreads()` 后全体读取；这一次同步同时保证两角色 smem 复用的安全。线程少的角色在 `tx < n` 上执行，其余线程空转（Rammer 式）。SM 级按 `%smid` 查主机表；CTA 级按每 SM 到达计数（POD 式）。最后退出的 CTA 发布计时结果并复位所有计数器。
- **核心改动**：`tl.shared_lifetime_scope` 属性（`src/op/builtin.h`、`merge_shared_memory_allocations.cc`，+40 行，含回归测试），让 smem 合并 pass 把一次 tile 执行当作存活边界。GEMM × decode 每 CTA smem 从 173KB（无法启动）降到 73KB（≤ 两者单跑中较大者）。
- **TVM 子模块补丁**：Z3 在 rlimit 耗尽时抛 `canceled`，导致 4 个 kernel 的 layout 推断崩溃；`CanProve` 改为把异常当作"无法证明"。以 `research/patches/tvm_z3_canprove_exception.patch` 保存（子模块远端不归我们）。
- 开销（188 CTA，另一角色编进来但不分配 tile）：GEMM 静态 +2%；decode +3.6%；寄存器 ≈ 较大角色 + 10–40；动态分派的 atomic 被隐藏（RMSNorm chunk 1/4/16 与静态相当）；逐 tile 时间戳开销 0–4%。
- f32x2→FFMA 融合问题在 CoKernel 中没有出现（全部逐位相同）。#3128 的 `tl.enable_fp32x2_reduction` 只影响 `T.reduce_sum/abssum`，不是这个问题的开关。
- **未解决：静态调度在 flush 模式下慢 20–30%**。GEMM 静态 268µs vs 动态 211µs（op 库自己的静态持久化版同样慢，262.5µs）；debug 构建与热 L2 下消失（189 vs 186.5µs）。候选原因：flush 写入的脏 L2 行回写、固定的 SM↔地址映射、L1/smem carve-out。**这可能说明 flush 测量方法对某些 kernel 有偏差，影响所有共跑测量，必须在 3×2 研究之前查清。** 已通知 profiling agent 对代表性配置加测 hot / graph 模式。
- 其他开放问题：寄存器耦合使 RMSNorm 在 GEMM 旁从每 SM 6 个 CTA 降到 1 个；大配置下 CTA 级绑定每 SM 只有 1 个 CTA，两个角色实际上不在同一 SM 共驻，需要更小的或重新推导的实现；warp 级绑定需要 tile body 支持线程偏移、按角色的 named barrier、分区（而非复用）的 smem，decode 归约固定用 barrier 1/2 会冲突。

**GPU 占用规则的解释（22:00）**：用户 qzr 有一个空闲的检索服务（`local_retriever.py --gpu`，持有约 1.1GB 显存，利用率 0%，P8）长期驻留。规则 7 的"被占用"解释为：有外部进程的 SM 利用率 > 0（`nvidia-smi pmon -s u`），而不是仅仅持有上下文；每个计时批次前后采样，被污染的批次重测。若用户不同意此解释需要调整。各测试脚本里"有任何其他进程就等待"的守卫会因此永久阻塞，下一个功能里要统一成共享的守卫。

**进行中（附完成标准）**
- P1 基线：FlashInfer（0.7.0，JIT）。完成标准：不改动现有包版本安装；GQA decode 与 prefill 在 sm_120 上正确并测时（与 TileLang 示例对比）；POD 融合 kernel 能否在 sm_120 编译运行、结果正确，并与串行 / 双流 / green context 划分对比；结果写入 `research/results/2026-09-22_flashinfer_baselines/`，封装 `research/bench/baselines/flashinfer_ops.py`。

- P1-S 单跑 profiling 与 C_lib 目录。完成标准：S1 可复现、可断点续跑的 profiling 脚本（flush 模式，记录频率与功耗→每次调用能耗）；S2 每个算子×形状的最佳 TileLang 配置 vs 参考库（cuBLAS / FlashInfer / torch rms_norm），标出慢 >10% 的形状；S3 持久化版 vs grid 版；S4 94、47 SM 预算下的最优配置变化（GOLDYLOC 效应）；S5 C_lib（Pareto 前沿 ∪ 预算最优）；S6 `cotile/catalog.py` 加载接口；S7 GEMM×decode、GEMM×RMSNorm 的配对表（时长比 0.25–4，T_serial 与资源下界含功耗下界）；S8 结果 < 5MB，GPU 时间 ≤ 约 2.5h。
- proposal v0.3（35be5243）：功耗墙作为共享资源纳入 §1.2（效应 4，待验证推论：低能耗实现在共置下更值钱）、§4 下界、§5.4 模型、§9 风险。

**关注的问题**
- 基线强度：sm_120 上 TileLang GEMM 走 mma.sync（无 wgmma/tcgen05），单跑性能若明显低于 cuBLAS，共置收益会被"低效 kernel 留下的空闲资源"虚增。P1 必须同时报告 cuBLAS / FlashInfer（或 torch SDPA）单跑时间作为参照，并在 3×2 分解里用最强的单跑实现作为 solo 基线。
- 不能锁频：共跑时功耗更高，可能比单跑更早降频，会低估共置收益或引入噪声；需要在结果里报告每组的频率分布。
