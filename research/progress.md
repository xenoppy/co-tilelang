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

**进行中（附完成标准）**
- 调研：TileLang 中 CoKernel 原型所需能力（角色分支内的 layout/pipeline/smem 合并、持久化循环、`atomic_add(return_prev)`、`%smid`/`%globaltimer`、`T.ws` 与 named barrier、寄存器上限、批量编译）。产出 `research/notes/tilelang_capability_survey.md`。
- P0-2 测量框架 v0（`research/bench/cobench/`）。完成标准：单 kernel 计时三种模式（刷 L2 的 event 计时、CUDA Graph + 输入轮换、热模式）；双流共跑计时（公共起点、各自完成时刻）；NVML 采样器；green context 流封装并实测 SM 粒度；`%smid` 探测（取值与空洞、持久化 grid 的 CTA 分布、green context 下用到哪些 SM、`%globaltimer` 分辨率）；matmul 重复 5 次的 CV < 2%（或给出原因）；README。

**关注的问题**
- 基线强度：sm_120 上 TileLang GEMM 走 mma.sync（无 wgmma/tcgen05），单跑性能若明显低于 cuBLAS，共置收益会被"低效 kernel 留下的空闲资源"虚增。P1 必须同时报告 cuBLAS / FlashInfer（或 torch SDPA）单跑时间作为参照，并在 2×2 分解里用最强的单跑实现作为 solo 基线。
- 不能锁频：共跑时功耗更高，可能比单跑更早降频，会低估共置收益或引入噪声；需要在结果里报告每组的频率分布。
