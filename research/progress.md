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

**P1-M3a CoKernel 构建器 v0（原型）：已完成（22:08 验收；提交 205528ca）**
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

**P1-S 单跑 profiling 与 C_lib 目录：已完成（23:36 验收；提交 ff257ea5）**，结果 `research/results/2026-09-22_solo_profile/`，加载接口 `cotile/catalog.py`。
- 完成标准（原文）：S1 可复现、可断点续跑的 profiling 脚本（flush 模式，记录频率与功耗→每次调用能耗）；S2 每个算子×形状的最佳 TileLang 配置 vs 参考库（cuBLAS / FlashInfer / torch rms_norm），标出慢 >10% 的形状；S3 持久化版 vs grid 版；S4 94、47 SM 预算下的最优配置变化（GOLDYLOC 效应）；S5 C_lib（Pareto 前沿 ∪ 预算最优）；S6 `cotile/catalog.py` 加载接口；S7 GEMM×decode、GEMM×RMSNorm 的配对表（时长比 0.25–4，T_serial 与资源下界含功耗下界）；S8 结果 < 5MB，GPU 时间 ≤ 约 2.5h。
- 规模：27 个形状 × 1026 个配置 × 4 个测点（grid @188/94/48、persistent @188），2432 个 kernel（冷编译 11 分钟），GPU 约 1h45m，结果 3.2MB。qzr 的空闲进程全程持有上下文、无 SM 活动，0 个批次需要重测。
- S2：**单跑基线是诚实的**。GEMM 最优 TileLang / 最快的正确 cuBLAS = 0.85–1.065；decode ≤ FlashInfer（0.93–0.997）；RMSNorm 0.98–1.04× FlashInfer（torch 2.8 的 `rms_norm` 拆成 8 个 kernel，慢 5–10×，不作参照）。注意：cuBLAS 在更低频率下运行（功耗密度更高），按 cycles 它领先 10–15%；torch 默认允许 cuBLAS 用 bf16 归约 split-K 部分和，M2048_N4096_K14336 上超出 fp32 参考容差。
- S3：**静态持久化的惩罚是真实的**，不是 flush、占用率或代码生成造成的：GEMM persistent/grid 中位数 1.52×（hot、graph 模式下仍 1.20–1.75×），SASS 主循环完全相同，persistent 频率反而更高、功耗更低（在等待）。decode 只在需要循环时出现（中位数 1.10×）。与 CoKernel agent"热 L2 下消失"的说法矛盾，原因未明。
- S4：**GEMM 的 GOLDYLOC 效应显著**：94/48 SM 预算下最优配置在 8/9 个形状上与全卡不同，全卡最优在缩减预算下慢 6–14%（中位数 9.1% / 12.3%）。原因与功耗有关：全卡时功耗墙压频（所有 GEMM 配置中位频率 2015 MHz），大 tile 更省能量而胜出；94/48 SM 时功耗不再受限（2736/2822 MHz），TMA warp-specialized 128×128 胜出。decode、RMSNorm 基本无此效应。
- S5：C_lib 大小 GEMM 9–12、decode 5–10、RMSNorm 4–11。
- S7：**功耗墙是大多数平衡算子对的约束资源**：GEMM×decode 的 63 个时长比在 [0.25, 4] 内的配对中 49 个、GEMM×RMSNorm 24 个中 23 个受功耗约束；T_serial / 下界只有 1.09–1.38。候选研究点：GEMM 4096³ × decode B16×8192（392/343µs，上限 1.22）；GEMM 2048×4096×4096 × decode B32×2048（202/176µs，1.35）；时长比扫描 GEMM 4096³ × decode B64×8192 / B32×8192 / B16×8192 / B32×2048（0.29/0.58/1.14/2.23）；GEMM 4096³ × RMSNorm 16384×8192（1.20）；GEMM 2048×14336×4096 × RMSNorm 65536×4096（1.19）。
- **测量方法问题**：(1) cobench 的写 flush 让 L2 充满脏行，被测 op 要为回写付费（GEMM 慢 1–5.8%，RMSNorm 最多 30%）；先写后读的 flush（冷且干净）没有这个偏差；只读 flush 在本卡上清不干净。两个单跑时间相加会付两次，一次共跑只付一次，所以共跑加速比必须对照 `bench_corun` 自己的 serial 变体。(2) flush 模式下 flush 阶段（约 360W）给功耗控制器留出余量，单跑 GEMM 运行中功耗 608–689W，高于持续运行时；在稳态 serving 中不会有这种余量，所以 flush 模式可能高估单跑、低估共置收益（或相反），必须用持续负载的稳态模式对照。(3) 温度是主要的慢变量：功耗受限的 GEMM 空闲 2 秒后快 2.2–2.6%，测量时要保持 GPU 持续负载。
- 其他：TileLang 的 kernel 缓存以 git commit 为键，每次提交都会让下一次运行重新编译约 11 分钟（2432 个 kernel），缓存增长约 1GB。

**对研究方向的影响（主 agent 判断）**
- 功耗墙把本平台上这些算子对的理想收益限制在 1.09–1.38×，D1 的完整主张（T_inter*/T[derived, intra] ≥ 1.10 且 lib/derived ≥ 1.05）可能只在少数配对上可达。按规则不在看到 P1 共跑数据前改阈值，但这提示 derived 空间应包含"低能耗"实现（大 tile、少 DRAM 流量），因为在功耗受限时降低 E_A + E_B 本身就降低下界——这正是 proposal 效应 4 的推论，可以作为"资源需求属于实现"的一个新的、平台相关的例子。
- 下一功能：测量方法 v1（干净 flush、稳态共跑模式、统一 GPU 守卫、持久化调度检查），在任何 3×2 共跑测量之前完成。
- proposal v0.3（35be5243）：功耗墙作为共享资源纳入 §1.2（效应 4，待验证推论：低能耗实现在共置下更值钱）、§4 下界、§5.4 模型、§9 风险。

**测量方法 v1：已完成（02:20 验收：主 agent 复跑 test_cobench 14/14 通过、CoKernel 套件 184/184 通过）**，结果 `research/results/2026-09-23_methodology_v1/`。
- 完成标准（原文）：M1 先写后读的干净 flush 为默认，用数据证明脏行偏差消失；M2 共享 GPU 守卫 `cobench.guard`（pmon 规则），替换 harness 的旧守卫；M3 稳态共跑测量 `bench_steady`（各变体背靠背持续执行、输入轮换、变体交错、报告每轮时间 / 频率 / 功耗 / 相对 serial 加速比），在 P1 主研究点（GEMM 4096³ × decode B16×8192，单跑最优配置）上做冒烟测试并与 flush 模式对比；M4 静态持久化惩罚的根因（grid / 静态 / 动态，ncu 指标，tile 时间线，假设检验）；M5 README 与结果目录。
- M1：先写后读**并不干净**（读命中 L2 中的脏行不会清掉它们，约一半 L2 仍脏）。新的默认 `clean` flush = 写 2×L2 缓冲后对它执行 `discard.global.L2`。decode B16×8192：旧 flush 342.0µs → clean 330.0µs（背靠背 328.1µs）；RMSNorm 4096²：43.3 → 28.2µs。写密集且输出 ≤ L2 的 op 在 clean flush 下比稳态更快（L2 吸收了自己的写，在计时结束后才回写），已在文档说明。
- M2：`cobench.guard` 用 pmon 流判断外部 SM 活动；qzr 的 ray worker 开始训练（600W / 94GB）时守卫挡住了 46 分钟，并自动重测了一次被污染的稳态测量。
- M3：`bench_steady` 各变体背靠背 1.5s 切片、5 轮交错、输入输出轮换 > 2×L2，温度平稳后开始；3 个进程间时间 CV ≤ 0.28%、加速比 CV ≤ 0.21%。serial 与单跑之和之比：非功耗受限的对 0.988；P1 对 0.958（功耗控制器对 GEMM 与 decode 的功耗做平均）→ 加速比必须对照同一次测量中的 serial。
- **P1 冒烟测试**（GEMM 4096³ × decode B16×8192，单跑最优配置，相对 serial 的加速比，稳态 / clean flush）：两条 stream 1.128 / 1.136；green context GEMM 60/80/100/120/140 SM：0.61/0.78/0.91/1.05/**1.220** 与 0.62/0.79/0.93/1.07/**1.322**；CoKernel 动态 + 接手，GEMM 60/94/128/160 SM：1.010/1.130/1.202/**1.204** 与 1.037/1.219/1.300/**1.320**。**flush 模式对触及功耗墙的变体高估 8–10%**（flush 阶段给控制器留余量：green 140 在 flush 模式下 2469 MHz，稳态 2214 MHz），所以稳态是主模式。两条 stream 的结果由硬件调度决定先跑 GEMM 还是 decode（makespan 约 640 vs 680µs），与主机发射顺序无关。
- **M4：静态持久化惩罚是我们自己的测量假象**。cobench 的时钟探针（常驻一个 warp）让它所在 SM 保持小的 shared memory carveout（8 次中 7 次），SM 只有空闲时才能改 carveout → 大 smem kernel 实际只有 187 个 SM；188 个 CTA 的静态持久化 grid 有一个 CTA 只能在别的 CTA 退出后启动，它的固定 tile 形成串行尾巴（CTA 187 在 259µs 才开始，makespan 488 vs 388µs）。动态队列能吸收缺失的 SM，所以动态总是正常。修复：探针与主机闸门请求最大 carveout。修复后 persistent/grid：GEMM 4096³ 1.010、GEMM 8192×14336×4096 1.003、decode B64 split-KV 1.016。排除了 TPC 配对、atomic 抖动、分派槽与 barrier、flush/L2 状态、代码生成。ncu 需要管理员权限（`ERR_NVGPUCTRPERM`），改用 smem 缓冲的逐 tile 时间戳（扰动 < 0.5%）。
  - **回退 / 作废**：P1-S 的所有 persistent 时间与 `c_lib_persistent` 作废；grid @188 基本不受影响（1026 个点中 17 个处在 187/188 SM 的波次边界）；大 smem 配置的 @94/@48 预算点可能少了一个 SM。CoKernel README 里"静态比动态慢 20–30%"同为此假象，已更正。
  - 独立的真实效应：每个 CTA 分到连续 tile 区间会让 split-head decode 慢 3.95×（失去 K/V 共享）——tile 顺序是编排空间里的一个真实维度。
- **额外修复：CoKernel scratch 未对齐**。16B 的分派槽让每个角色的 swizzle smem 缓冲偏离 128B 边界 16B，所有 CoKernel 角色慢 1.5–1.7×（GEMM 628 vs 373µs）。把槽补齐到 128B（`SLOT_INTS = 32`）后恢复到 grid 速度（0.999–1.006），P1 CoKernel 从 0.84–0.86× serial 变为 1.20×。**根本修复应在 TileLang 的 smem 合并 pass 里对 cp.async/ldmatrix/swizzle 缓冲强制 ≥128B 对齐**（待办）。
- 对 3×2 研究的建议：稳态为主模式，报告每个变体的频率与功耗；双流基线取两种发射顺序 × 两种优先级中最好的，并分别在默认与最大 carveout 下评估（carveout 决定两个 kernel 的 CTA 能否共享 SM）；单跑最优 decode（128 个 tile、每个约 330µs）是很差的共置伙伴，lib / derived 列需要 split-KV 变体。
- 附带（主 agent）：`research/env.sh` 设 `NO_GIT_VERSION=1` 与 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1`，kernel 缓存不再因每次 git 提交失效（以 native 库内容哈希为键）；修改 `src/tl_templates` 后需手动清缓存。
- GPU 环境变化：qzr 的 RL 训练（ray worker，持有 67GB 显存，间歇性满载）从约 00:30 开始，我们只剩约 29GB 显存，测量会被守卫间歇性挡住。

**P1-3x2-A：GEMM × decode 的 solo / lib 两行：已完成（05:00 验收：主 agent 独立重跑主研究对的 F 阶段，各变体加速比与原结果相差 ≤0.6%，排序不变，T_inter*/T[lib,intra] 0.993 vs 0.994）**，结果 `research/results/2026-09-23_p1_3x2_A/`。
- 完成标准（原文）：A1 用修复后的探针重测相关形状的预算点并重建 C_lib；A2 跨 kernel 变体（serial、双流取发射顺序 × 优先级最好者、green context 划分扫描）稳态为主、clean flush 为辅；A3 CoKernel SM 级（动态、接手、chunk、SM 划分扫描）、CTA 级（能共驻时的配比扫描）、静态对照；lib 行用两阶段搜索（flush 筛选 + 稳态确认），并量化筛选的保真度；A4 每对的完整表（T_serial、LB、四个格子、T_inter*、频率 / 功耗、获胜配置及其单跑排名）与收益归因；A5 D1 中期读数（不改阈值）；A6 结果目录与脚本。
- **结论是否定的**：kernel 内编排在 lib 层面没有超过最好的跨 kernel 共置。稳态下（相对同次测量的 serial）：

| 研究对 | T_serial µs | T[solo,inter] | T[lib,inter] | T[solo,intra] | T[lib,intra] | T_inter*/T[lib,intra] |
|---|---|---|---|---|---|---|
| main GEMM 4096³ × B16×8192 | 725.9 | 1.224 | 1.238 | 1.225 | 1.230 | 0.994 |
| GEMM 4096³ × B64×8192 | 1710.9 | 1.171 | 1.187 | 1.192 | 1.192 | 1.004 |
| GEMM 4096³ × B32×8192 | 1061.6 | 1.331 | 1.331 | 1.349 | 1.349 | 1.013 |
| GEMM 4096³ × B32×2048 | 574.2 | 1.133 | 1.137 | 1.131 | 1.131 | 0.995 |
| GEMM 2048×4096² × B32×2048 | 375.0 | 1.188 | 1.197 | 1.171 | 1.182 | 0.987 |

- 从 C_lib 挑选相对 solo-best 只值 ≤1.4%（inter）/ ≤0.9%（intra），两列之间没有交互。D1 弱化主张的条件（T_inter*/T[lib,intra] ≥ 1.10 在 ≥2 对上）在 0/5 对上成立；完整主张需要 derived 行比 T[lib,intra] 再快 8.7–11.4%。
- **归因**：4/5 对受功耗墙约束——所有共置变体都跑在 600W，时间 ≈ 每轮能耗 / 600W；共置降低能耗的原因是静态功耗只付一次、DRAM 受限的 decode 在更低频率下运行，**是"重叠"本身而不是编排方式**，各机制之间能耗差 < 2%。r029 受 DRAM 约束（高于 DRAM 下界 4.8%）。时间与频率并不一致（lib-inter 获胜者频率最低 1836 MHz 但时间最好），以时间为准。
- LB_power 不是有效下界：共置比两者单跑能耗之和少 7–16%（空闲但有时钟的静态功耗约 143W 只付一次、decode 降频）；有效的功耗下界需要"能耗随频率 / 电压变化"的模型。
- 其他：两条 stream 1.05–1.15×；静态调度比动态慢 0.3–2.9%；CTA 级绑定需要小 tile（单跑慢 6–19%），只在第二对上以"部分共驻"获胜；两角色 CoKernel 丢失 warp specialization（第二对 +2.6%）；carveout 对共驻无影响，只有 decode 优先级更高时才共驻。
- 筛选保真度：flush 筛选与稳态的 Spearman 0.77–0.99，稳态获胜者总在筛选前 8 名内，但前 8 名之内（功耗受限对）相关性约为 0，且 flush 会把最优划分推向更多 GEMM SM。
- **TileLang bug 根因修复**：NVRTC / CuTeDSL 后端用 `str(target).startswith("cuda")` 选择 stream，而 `str(Target)` 现在是 JSON，导致所有启动都跑在 legacy 默认流上（`tilelang/jit/adapter/{nvrtc,cutedsl}/adapter.py`）。我们的默认后端（tvm_ffi）不受影响，之前的数据有效。
- qzr 的训练进程会增长到 91–95GB，第一次 A1 运行因 OOM 失败；脚本现在同时等待 GPU 空闲与显存足够。我们的 1–5GB 也可能让对方 OOM——需要协调（应告知用户）。

**主 agent 的判断（04:55）**
- 在本平台（600W 功耗墙）上，对这些大尺寸的计算 × 访存对，"用什么机制共置"几乎不影响结果；收益来自重叠本身。这与 proposal §1.2 假设的主导效应（SM 内容量、波次、资源耦合）不同，功耗是第一位的。
- 但所有 green context 结果都用的是事后挑出的最优划分（oracle）。冒烟测试中，在同一个非最优划分下 CoKernel + 接手为 1.13×，green context 为 0.91×。**kernel 内 tile 级动态调度 + 接手的价值可能在于不需要 oracle 就接近最优**（对划分选择不敏感、能适应变化的负载），这是 green context 做不到、而 mKernel 类工作在别的场景做过的。这需要数据验证，而不是假设。
- 下一步（一个功能）：P1-3x2-B = 针对性的 derived 行 + 无 oracle 鲁棒性读数，然后用同一流程做 P2（GEMM × RMSNorm）与 P4（prefill × decode），在 P1–P4 的证据上做 D1 决策。
- 待办（根因修复）：TileLang smem 合并 pass 应对 cp.async / ldmatrix / swizzle 缓冲强制 ≥128B 对齐（CoKernel 目前靠把分派槽补到 128B 规避）。

- P1-3x2-B（进行中，05:05 启动）：同五个 GEMM × decode 对的 derived 行 + 无 oracle 鲁棒性。完成标准：B1 写明 C_derived 的定义（C_lib ∪ 以伙伴资源契约为条件的变体：契约内的全部 op 库配置含 split-K / split-KV、寄存器上限轴、至少一个"只在共置时才会选"的实现选择——首选 decode K/V 流式加载的 L2 evict-first 提示，作为 decode 的配置轴实现并测试数值逐位相同）；B2 T[derived,intra]、T[derived,inter] 与完整 3×2 表、对获胜者做轴消融；B3 同一组 SM 划分下 green context 与 CoKernel（动态 + 接手）的 oracle / 先验规则划分 / 最坏情况 / regret，以及把 main 上调好的划分迁移到其他对；B4 三行齐全的 D1 中期读数；B5 结果与测试。
- 需要告知用户：qzr 的 RL 训练进程会占到 91–95GB 显存，我们的实验（1–5GB）可能与之争抢显存导致对方 OOM；当前脚本会等待显存足够才启动。

**05:50 暂停（用户要求：集群有其他人在用）**
- 已终止正在运行的 `p1b_study.py main --stages DC1,DC2,DC3,DC4,F,R`（P1-3x2-B 的 main 对，derived 行筛选阶段）；GPU 上只剩其他用户（qhy）的进程。已通知 Part B 子 agent 停止所有 GPU 工作，只做 CPU 侧的整理：写 `research/results/2026-09-23_p1_3x2_B/RESUME.md`（已完成 / 中断位置 / 剩余工作 / 恢复命令 / 未测试的代码改动）。
- 在用户允许之前不启动任何 GPU 负载。恢复时：先按规则 7 查询 GPU 使用情况，再按 RESUME.md 继续 P1-3x2-B。
- 尚未提交的工作：Part B 子 agent 在 `cotile/`（decode 的 L2 提示轴等）与 `research/bench/scripts/p1b_*.py` 中的改动，未验收，暂不提交。

**06:02 恢复（用户："继续做"）**
- 查询 GPU：完全空闲（无任何进程）。已让 Part B 子 agent 恢复（保留上下文），并先收紧 GPU 共享策略（规则 7 的更严格实现）：任何非本用户的计算进程出现在 GPU 上即视为"被占用"（不论 SM 利用率）；启动前若被占用则挂起 30 分钟再查询；运行中若出现外部进程，完成当前测点后让出 GPU、等待 30 分钟再查询并从下一测点继续；被阻塞超过 2 小时则停下来报告（由主 agent 询问用户）。统一实现在 `cobench.guard`，`p1_common.wait_gpu` 与 `cotile/tests/harness.py` 共用。
- Part B 中断前的进展（见 `research/results/2026-09-23_p1_3x2_B/RESUME.md`）：TileLang 的 `T.copy(..., eviction_policy=...)` 此前只对 TMA 生效、对 cp.async 静默忽略，已补上；带 cache hint 时 ptxas 12.9 会错误编译 43 个 GEMM 配置中的 15 个（运行时崩溃），已独立复现并在模板中规避，另在 `cotile/resources.py` 加了 SASS 检查。新配置轴 decode `kv_l2`、GEMM `ab_l2`，输出逐位不变。**初步结果**（待最终交错测量确认）：decode K/V 用 evict-first 提示时 green 共置快 3.0%（568.9 vs 586.0µs，每轮能耗 341 vs 352 mJ）——这是一个"只在共置时才值得选"的实现选择，符合 proposal 的论点，但幅度低于 5% 阈值；GEMM 的 evict-last 提示无效果。
- 注意：Part B 改了 `src/tl_templates/cuda/copy.h`，kernel 缓存键（库哈希）不覆盖模板，已要求子 agent 清缓存或核实。

**06:52 计划调整（用户同意）：plan v0.4（e73b4f5d）**
- 用户问"CoKernel 是否不如 POD 式融合"。结论：目前不能这么说——POD（prefill × decode，旧 flush 模式、测量方法 v1 之前、对照只扫 5 个粗 green 划分）与 CoKernel（GEMM × decode，稳态、细扫且按预算挑配置的 green 对照）不是同条件；同条件下唯一的证据是在 GEMM × decode 上 POD 式 CTA 级混跑反而比 SM 级划分差（主研究对 1.18× vs 1.23×）。
- 决定：P1-3x2-B 完成后先做 P4 与 POD 的同条件对比（算子库加 prefill attention；本地修补 POD 的默认流 memset 以便稳态测量；同一轮稳态比较串行 / 双流 / 细扫 green / POD / CoKernel SM 级与 CTA 级；归因差距来自算子特性还是 tile 实现），再做 P2。执行顺序 P1 → P4 → P2 → P3。

**07:37 GPU 共享策略再调整（用户："qzr 的 workload 只占显存不占计算，继续做实验"）**
- "被占用"改回按计算判断：外部进程在 pmon 中有 SM 活动才算占用；只持有显存的外部进程不阻塞我们（仍检查剩余显存是否足够，保持小显存占用）。占用时仍按规则 7 挂起 30 分钟再查；运行中出现外部 SM 活动则做完当前测点后让出、30 分钟后再查。严格模式（任何外部进程即占用）保留为非默认选项。
- 当时状态：Part B 已完成 main、r029、r058 三对的 FQ/R 阶段；r223 在 07:10 因 qzr 的进程出现（显存 45GB、SM 0%）而按严格策略让出等待。已停止等待中的包装进程，通知子 agent 按新策略立即重启 r223 并继续 second 与其余工作。

## 2026-09-24 03:35 — 会话重启后恢复

**现状**
- 上一会话在 09-23 07:53 之后结束，Part B 子 agent 随之终止。Part B 已完成 main（完整 derived 搜索）、r029、r058（精简协议 FQ + 鲁棒性 R）；r223 在 07:48 因 qzr 的任务真正占用计算（SM 活动、仅剩 1.6GB 显存）让出，阻塞 2 小时后于 09:48 按策略停止；second 未开始。代码改动（L2 提示轴、ptxas 规避、缓存键覆盖模板头、GPU 守卫策略、p1b 脚本）仍未提交。
- GPU：qzr 的 ray worker 持有 32GB，无 SM 活动 → 按用户的规则不算占用。
- 子 agent 类型 `researcher`（opus、effort xhigh）现在可用，从此改用它（符合规则 1）。

**Part B 中期结论（main、r029、r058，稳态）**
- 唯一有效的 derived 轴是 decode K/V 加载的 L2 evict-first 提示：相对最好的 lib 变体，inter / intra 分别快 2.2%/1.6%（main）、4.4%/4.4%（r029）、8.7%/5.6%（r058）；单跑毫无价值（±0.25%），所以是"只在共置时才值得选"的实现。机制与 proposal 效应 4 一致：K/V 流不再把 GEMM 的操作数面板挤出 L2，DRAM 字节与能耗下降（r058 每轮能耗 479 → 440 mJ，时间同比例下降）。
- 但它对 green 分区与 CoKernel 的帮助一样大（甚至 green 更多），所以**不是 kernel 内编排带来的效应**，proposal §4 的交互项 ≈ 0。
- 其余 derived 轴都更差：split-K GEMM −10%，split-KV decode −1%，每 CTA 更少 head −7%，寄存器上限下的 CTA 共驻 −8%，GEMM A/B evict_last 0 到 −2%。
- D1（阈值未改）：完整主张 0/3 对、弱化主张 0/3 对。r058 的重推导比值 1.056 过线，但同一提示让 green 更快，T_inter*/T[derived,intra] = 0.985。
- **B3 鲁棒性假设成立**：8 个划分中 CoKernel（动态 + 接手）最差 ×1.061–1.077，green 最差 ×0.753–0.854（比串行慢）；按单跑时间成比例划分（R1）的 regret：CoKernel 1.000–1.092，green 1.255–1.489；把 main 的最优划分迁移到其他对：CoKernel regret ≤1.003，green 1.048–1.076。但在 oracle 划分下 green 与 CoKernel 相差 ±3% 以内或更好——CoKernel 的价值是鲁棒性，不是更高的最优值。
- 其他发现：ptxas 12.9 在 sm_120 上带 cache policy 的 cp.async 会错误编译约 1/3 的 GEMM 配置（运行时非法指令），已用 64 位共享地址形式规避并加 SASS lint；kernel 缓存键原先不覆盖 `tl_templates` 头文件，已修（`tilelang/cache/build_stamp.py`）。

**P1-3x2-B：derived 行 + 无 oracle 鲁棒性：已完成（04:50 验收：主 agent 复跑 test_ops 全部通过含 L2 提示 32/32、test_cokernel 204/208（4 个为预期对照）、test_cobench 15/15）**，结果 `research/results/2026-09-23_p1_3x2_B/`。
- r223、second 两对于 09-24 03:37–04:16 测完（GPU 空闲，守卫记录干净）；发现中期 README 误称 r029 已按最终精简协议重测，实际没有 → 已重测，与 21 小时前的结果相差 ≤0.2%（独立的复现性证据）。
- 最终结果（稳态，相对串行）：

| 研究对 | T_serial µs | lib,inter | derived,inter | lib,intra | derived,intra | lib/der inter | lib/der intra | T_inter*/T[der,intra] |
|---|---|---|---|---|---|---|---|---|
| main | 726.1 | ×1.238 | ×1.266 | ×1.232 | ×1.252 | 1.022 | 1.016 | 0.989 |
| r029 | 1709.4 | ×1.186 | ×1.238 | ×1.191 | ×1.244 | 1.044 | 1.045 | 1.005 |
| r058 | 1060.4 | ×1.330 | ×1.446 | ×1.349 | ×1.424 | 1.087 | 1.056 | 0.985 |
| r223 | 571.4 | ×1.135 | ×1.157 | ×1.132 | ×1.140 | 1.019 | 1.007 | 0.985 |
| second | 374.0 | ×1.195 | ×1.208 | ×1.179 | ×1.205 | 1.011 | 1.021 | 0.997 |

- derived 获胜者全部是"C_lib 配置对 + decode K/V evict_first"。同一轮 F 中 31 对"带 / 不带提示"的相同旋钮变体，带提示者全部更快（+0.4% 到 +8.7%），每轮能耗低 0.4–8.1%；单跑价值只有 0.1–0.25%。提示对 inter 与 intra 帮助相当（lib/derived 均值 1.037 vs 1.029）。
- **D1（阈值未改）：P1 的五对中，完整主张 0/5、弱化主张 0/5**；重推导阈值（≥1.05）只在 r058 单独成立（1.056）；T_inter*/T[derived,intra] 在 0.985–1.005。
- **B3 鲁棒性（五对全部成立）**：CoKernel 最差划分 ×0.999–1.077，green ×0.590–0.854；R1 regret CoKernel 1.000–1.092，green 1.097–1.490；oracle 划分下 green 相等或更好（−0.1% 到 +3.9%）；不加提示时结论不变（最差 ×0.991–1.071 vs ×0.575–0.829）。r223 的 CoKernel 真实最优在 184 个 GEMM SM，超出 R 的划分集合，其 regret 低估约 2%（README 已标注）。
- 其他：chunk 大小不单调且影响大（r058 在同一划分下 chunk 1/1、1/2、1/4 分别为 765.6 / 811.1 / 971.7µs，+27%）——chunk 必须与划分一起选。
- 测试（子 agent 在最终代码上）：test_ops 全部通过（含 L2 提示 32/32），test_cokernel 204/208（4 个为预期不可启动的 `smem="sum"` 对照），test_cobench 15/15，lifetime-scope pytest 通过。
- GPU 时间：研究共 1.36h（含被取代的运行），测试约 15 分钟。

**P1 阶段小结（主 agent）**
- 在本平台上，GEMM × decode 的共置收益（×1.13–1.45）来自"重叠"本身（共享静态功耗、DRAM 受限的 decode 可降频），各机制在 oracle 划分下相差 ≤4%。
- **proposal 效应 4（资源需求属于实现、价值取决于伙伴）得到一个干净的正例**：L2 evict-first 在单跑时毫无价值、在共置时值 2–9%，机制是减少字节与能耗。但它与编排方式正交——跨 kernel 的 green 分区同样受益，所以不支持"需要 kernel 内编排才能兑现"的核心主张。
- **kernel 内 tile 级动态调度的价值是鲁棒性**：不需要 oracle 划分、不需要逐对调参就能拿到接近最优的收益，而 green 分区选错会比串行更慢。这是 green context 做不到的，但 mKernel 等在别的场景有类似的运行时自适应。
- D1 在 P1–P4 上决定。按 plan v0.4，下一步做 P4（prefill × decode，与 POD 同条件对比），它是唯一可能出现"同 SM 共驻胜过分区"的场景。

**P4-a：P4 的基础设施：已完成（06:10 验收：主 agent 复跑 prefill 29/29 配置、prefill × decode CoKernel 95/95（3 个为预期对照）、流水线回归测试与已有流水线测试 34 通过 3 跳过；提交 c2ca9dcd 等）**，结果 `research/results/2026-09-24_p4_prep/`。
- 完成标准（原文）：P1' 算子库加入 causal prefill attention（与 decode 同一头配置 Hq=32/Hkv=8/D=128；tile 暴露每个 tile 的工作量以便长任务优先；配置枚举、数值键、grid / 持久化版、资源签名；逐配置正确性、grid 与持久化逐位相同；prefill × decode 的 CoKernel 测试）；P2' FlashInfer POD 的计数器改为在 kernel 所在流上按序复位（补丁文件 + 逐位验证：默认流与原版相同，side stream / green 流 / CUDA graph 回放结果正确）；P3' 稳态单跑 profiling（prefill 全配置 + SM 预算曲线、decode 缺失部分、FlashInfer 参照），TileLang prefill 若比 FlashInfer 慢 >10% 必须标出，给出 8 个 P4 对的配对表；P4' 结果目录与脚本，GPU ≤ 2h。
- prefill op（`cotile/ops/prefill_attn.py`）：causal、FA2 式、NHD 布局（与 FlashInfer 共用张量），Hq 32 / Hkv 8 / D 128（POD 要求两者头配置相同，09-22 的 POD 基线也是 32/8/128）；tile =（query 块, query head），warp 全部沿行排布；tile 暴露工作量，默认长任务优先（LPT），自然顺序作对照（188 SM 下慢 1.5–14.8%）；29 个配置；数值 E1，键 `(block_N,)`；用精确 `exp2f` 而非 fast math（fast math 是 pass 配置，会连带改变伙伴算子的比特）。
- **TileLang 核心修复**（`src/transform/inject_pipeline.cc`，c2ca9dcd）：运行时 trip count 的流水循环，其 epilogue 长度未化简，cp.async wait_group 计数依赖循环变量，≥2 级流水的配置全部代码生成失败；加一次 `Simplify` 并附回归测试。
- **POD 补丁**（`research/patches/flashinfer_pod_stream_memset.patch`，已应用于 site-packages）：计数器在 kernel 所在流上 `cudaMemsetAsync`、按设备分配。默认流上与原版逐位相同（24/24）；side stream / green 流 / CUDA graph 回放的错误数从 44/48、48/48、46/48 降到 0/192。剩余限制：两个 POD 同时在不同流上启动仍共用计数器；首次调用须在 stream capture 之外。
- **单跑（稳态）**：TileLang prefill 比 FlashInfer **更快**（S2048 135.7 vs 162.9µs，0.833×；S8192 1699.9 vs 1846.9µs，0.920×），decode 0.96–0.99×。**公平性问题方向相反**：TileLang 串行比 FlashInfer 串行快 3–12%，POD 在三个 prefill 为主的对上比 TileLang 串行还慢。P4 必须让每个系统对照自己的串行，并加归因对照。
- 配对表（稳态）：POD 相对 FlashInfer 串行 1.01–1.37×；只有 P2048_B16_S2048（时长比 1.56）与 P8192_B64_S8192（1.30）落在 0.5–2 之内，两个 0.41 的对勉强；prefill 为主的对（时长比 5–20）上限很低（1.03–1.14）。
- 其他：S=8192 时原有的误差容限对 FlashInfer、TileLang、torch bf16 SDPA 都不通过（长 causal 行输出很小，缩小了容限尺度），改为同时接受"不比 torch bf16 差"。GPU 约 45 分钟测量。`~/.tilelang/cache` 有 2.7GB 旧缓存，未动。

**P4-b：与 POD 的同条件对比：已完成（09:55 验收：主 agent 复跑完整 CoKernel 矩阵 371/378（7 个为预期不可启动对照）、prefill 31/31）**，结果 `research/results/2026-09-24_p4_study/`。
- 完成标准（原文）：Q1 每对一轮最终交错稳态测量，含 TL 串行与 FI 串行、TL 双流、细扫 green（± L2 提示）、打补丁的 POD、CoKernel SM 级与 CTA 级（solo / lib / derived），各自对照自己的串行并给出绝对时间、频率 / 功耗 / 能耗、各角色完成时间；Q2 回答 (a) POD 是否胜过最好的 TileLang 跨 kernel 方案和 / 或最好的 CoKernel，(b) 同 SM 混跑在这对上是否胜过按 SM 划分，(c) 差距归因（算子特性 vs 实现：移植 POD 的 decode tile / 每 SM CTA 数 / 虚拟 CTA；剥离 TileLang 单跑更快的影响）；Q3 P4 的 D1 读数；Q4 鲁棒性；Q5 结果目录与测试，GPU ≤ 3h。
- 新增：CoKernel `binding="tile"`（POD 的每 SM 票号策略放进持久化 kernel，每次取 tile 时选角色，一方耗尽则回落到另一方）；prefill 的 `order="kvhead"`（FlashInfer 的 CTA 顺序，不改比特）。GPU 2.62h，守卫记录全部干净。

| 研究对 | TL / FI 串行 µs | POD µs（对 FI 串行） | 最好跨 kernel（对 TL 串行） | 最好 CoKernel | 最好 CTA 级 | POD / 最好 TL |
|---|---|---|---|---|---|---|
| P2048_B16_S2048 | 217.5 / 246.4 | 207.4（×1.188） | 169.4（×1.284） | 168.6（×1.290） | ×1.209 | 1.231 |
| P8192_B64_S8192 | 2870.7 / 3175.5 | 2370.5（×1.340） | 2234.7（×1.285） | 2183.8（×1.315） | ×1.191 | 1.086 |
| P2048_B16_S8192 | 469.2 / 501.6 | 378.1（×1.326） | 359.8（×1.304） | 358.7（×1.308） | ×1.299 | 1.054 |
| P2048_B64_S2048 | 470.0 / 502.5 | 365.1（×1.376） | 359.2（×1.308） | 359.1（×1.309） | ×1.309 | 1.017 |
| 次要 4 对 | — | ×1.015–1.100 | ×1.012–1.077 | ×1.037–1.095 | ×0.97–1.08 | 1.008–1.110 |

- (a) **绝对时间上 POD 在全部 8 对上都慢于我们最好的方案**（主研究对慢 1.7–23.1%）。按各自串行算，POD 在 3/4 个主研究对上高 1.4–5.1%——因为它的单跑 kernel 更慢（TileLang 串行快 7–12%），"相对自己串行"的比值偏向慢 kernel。
- (b) **同 SM 混跑从未胜过按 SM 分区**：DRAM 受限的两对持平；prefill 主导的两对 CTA 级 −9.0% / −15.7%，POD 策略（tile 绑定）−6.3% / −9.4%。原因：快的 prefill tile（96KB smem、256 线程）无法与 decode CTA 共驻；能共驻的小 tile 单跑慢 1.7–4.8%；混跑的 SM 在功耗墙下每轮能耗更高（111/108 vs 101 mJ；1554/1446 vs 1310 mJ）。寄存器上限共驻在筛选中差 1–13%。
- (c) **归因**：用 POD 的策略 + 类似 FlashInfer 的 tile + FlashInfer 的顺序，TileLang 复现 POD 在 0–5% 以内。绝对差距 = TileLang 单跑更快 × 编排（例：1.227 = 1.123 × 1.092；其余三对编排项 0.95–0.985，即编排本身不占优）。把 POD 的 decode tile 移植进我们的获胜方案反而 −0.9% 到 −2.4%；POD 的"虚拟 decode CTA"在 FlashInfer 实现里只是 TODO。
- **D1（阈值未改）：完整主张 0/8，弱化主张 0/8**；T_inter*/T[derived,intra] 1.000–1.040。唯一有用的 derived 轴仍是 L2 evict-first（只在 DRAM 受限的 B64 对上 +1.5% / +2.6%）。
- **鲁棒性**：CoKernel 最差划分 ×1.021–1.128（从不慢于串行），green ×0.245–0.766；规则划分 regret CoKernel 1.000–1.123，green 1.000–1.482。**与 P1 不同，在 P4 上 CoKernel 的最优划分在全部 8 对上都不差于 green**：平衡与 DRAM 受限的对持平，prefill 为主的对快 3.0–3.9%（接手让短的 decode 借用 SM）。
- 其他：flush 筛选在 P4 上很不可靠（Spearman 0.42–0.71，一个稳态获胜者在筛选中排第 213），每行最好者都送去稳态确认才找到获胜者。kernel 缓存增长到 4.0GB。

**D1 汇总（P1 + P4，共 13 个研究对；阈值在看到数据前设定，未改）**
- 完整主张 0/13，弱化主张 0/13。按 plan §2.8，结论落在"转向"一侧（除非 D1 讨论修订主张）。
- 站得住的发现：(1) 共置本身有 1.1–1.45× 的收益，主要受功耗 / DRAM 约束；(2) kernel 内 tile 级动态调度 + 接手的价值是**鲁棒性**（无需 oracle 划分，从不慢于串行），在 prefill 为主的对上最优值也略优于 green（+3–4%）；(3) 存在"只在共置时才值得选"的实现（L2 evict-first），但与编排方式正交；(4) 在本平台上，粗粒度同 SM 混跑（CTA 级）因资源耦合与功耗而不占优；(5) 测得 TileLang 能复现 POD，POD 的相对优势来自慢的单跑基线。
- **用户提出的新方向（09-24 讨论中）**：像 TileLang 那样在 kernel 内做 tile 步骤级编排——把两个算子的 tile 拆成步骤，放进一条联合软件流水线（分 warp 的生产者 / 消费者，或同 warp 交错），而不是现在"一个 tile 整块执行完再换角色"。这是现有设计最大的缺口（SM 级 = 分区；CTA 级被寄存器耦合卡死；warp 级与步骤级交错从未实现）。主 agent 的建议：先做合成 kernel 的极限测试（纯 MMA 循环 + 纯 DRAM 流，比较 SM 分区 / CTA 共驻 / 分 warp 共驻 / 同 warp 交错），确认在功耗墙下细粒度方式能否超过分区 ≥10%，再决定是否投入手写原型与编译器化。待用户确认。

**极限测试 L（功耗墙下同 SM 共置的天花板）：已完成（12:50 验收：主 agent 复跑 test_cobench 18/18，含新的压力 kernel 测试 14–16）**，结果 `research/results/2026-09-24_limit_study/`，代码 `research/bench/cobench/stress.py`（同时完成 plan 的"压力 kernel v0"的 MMA 与 DRAM 流部分）。
- 过程：10:00 启动，约 10:15 因终端中断、机器重启而停下（一度误记为用户叫停），11:35 用户说"继续"后恢复。GPU 测量约 42 分钟，守卫记录干净。
- 合成 kernel 验证：纯寄存器 MMA 达 97.0% 的 1024 FLOP/clk/SM（2721 MHz，601W）；DRAM 读流 1620 GB/s（最好实测的 99%，理论峰值的 91%），约 40 个 SM 即可打满 DRAM；类 GEMM 主循环（cp.async + swizzle smem + ldmatrix，操作数驻留 L2）达峰值的 80.4%。所有变体都做了工作量"恰好一次"的校验。
- 结果（稳态，相对串行；比较同 SM 共置的最好者与分区的最好者）：

| 场景 | 分区（green / SM 级） | CTA 共驻 | 分 warp | 同 warp 交错 | 同 SM 最好 / 分区最好 |
|---|---|---|---|---|---|
| 纯寄存器 MMA，比例 0.5 / 1 / 2 | 1.511 / 1.488 / 1.283 | 1.511 / 1.540 / 1.313 | 1.512 / 1.527 / 1.306 | 1.477 / 1.419 / 1.243 | **1.000 / 1.035 / 1.023** |
| 类 GEMM，比例 0.5 / 1 / 2 | 1.469 / 1.340 / 1.178 | 1.470 / 1.348 / 1.167 | 1.470 / 1.350 / 1.170 | 1.366 / 1.263 / 1.094 | **1.001 / 1.007 / 0.993** |
| 纯寄存器，94 SM 子设备（功耗不受限） | 1.449 | 1.853 | 1.836 | 1.682 | **1.279** |
| 纯寄存器，每 SM 只 2 个 MMA warp（低强度） | 1.736 | 1.941 | 1.940 | 0.894 | **1.118** |
| 类 GEMM，94 SM 子设备（仍 600W） | 1.445 | 1.510 | 1.562 | 1.484 | **1.081** |

- **结论：在功耗墙下，最好的同 SM 共置方案相对最好的分区最多 +3.5%，从未达到 10%**；只有在 MMA 一侧不受功耗限制（子设备、低强度）时，同 SM 共置才赢 8–28%。所有功耗受限场景中各变体都在 599–603W，耗时 ≈ 每轮能耗 / 600W，最好分区与最好同 SM 方案的能耗只差 0–3.5%。分区让 MMA 用更少的 SM 但频率更高；同 SM 保留了更好的电压 / 频率点（纯寄存器比例 1 下约多 9% 的 SM·MHz），但大部分被 SM 内争用抵消（类 GEMM 情形下，流与 GEMM 的 cp.async / ldmatrix 操作数通路争用）。
- **同 warp 交错在所有场景都是最差的同 SM 方式**（MMA 发射卡在消费流数据的 load 上）；对"步骤级 / 同 warp 交错"方向是负面证据。
- `setmaxnreg`：`sm_120` 不接受，`sm_120a` 可用并正确运行；各角色的全部代码（包括收尾）必须在其 `setmaxnreg` 之后，否则 ptxas 按全局上限分配并溢出。寄存器重分配确实有效（GEMM warp 232、流 warp 40，启动 168，GEMM 路径用到 R215 且无溢出），是分 warp GEMM 布局中最好的，但不改变结论。
- 其他发现：`__launch_bounds__` 存在时 ptxas 静默忽略 `--maxrregcount`（改用 `__maxnreg__`）；驱动 580 不允许对 green context 的拆分结果再拆分（已用 2-SM 组重组规避）。

**对 D1 的含义（主 agent）**
- 设计问题 vs 物理上限：**在这块卡上是物理上限（600W 功耗墙）**。即使理想化的合成 kernel、做得最好的同 SM 共置，也只比分区好 ≤3.5%；我们的 CoKernel 与 POD 的实测结论与此一致。步骤级编排（同 warp 交错）甚至更差。
- 功耗不受限时同 SM 共置确实能赢 8–28%。本卡每 SM 的功耗预算约 600W / 188 ≈ 3.2W；数据中心卡按公开规格约为 H100 SXM 700W / 132 ≈ 5.3W、B200 1000W / 148 ≈ 6.8W（规格待核实）。**论点在功耗预算更宽的平台上可能成立**——这需要硬件验证，目前无法在本机回答。
- 本机上仍可能有价值的场景：不占满 GPU 的算子 / 部分 GPU（多租户各自分到一部分 SM）等功耗不受限的情形；以及 CoKernel 的鲁棒性。
- 需要与用户讨论 D1：是否换平台、是否收窄到"功耗不受限"场景、或转向其他方向。

**关注的问题**
- 基线强度：sm_120 上 TileLang GEMM 走 mma.sync（无 wgmma/tcgen05），单跑性能若明显低于 cuBLAS，共置收益会被"低效 kernel 留下的空闲资源"虚增。P1 必须同时报告 cuBLAS / FlashInfer（或 torch SDPA）单跑时间作为参照，并在 3×2 分解里用最强的单跑实现作为 solo 基线。
- 不能锁频：共跑时功耗更高，可能比单跑更早降频，会低估共置收益或引入噪声；需要在结果里报告每组的频率分布。
