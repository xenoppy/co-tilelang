# 环境与版本记录（P0-1）

> 2026-09-22 记录。编译方式：源码树内开发者编译（docs/get_started/Installation.md 的 "Working from Source via PYTHONPATH"），不装 wheel，不 `pip install -e`。使用前 `source research/env.sh`。

## 1. 版本

| 项 | 版本 |
|---|---|
| co-tilelang | `0.1.14+cuda.gitb261d5a5`（`tilelang.__version__`，后缀随 HEAD 变）；上游基线 `7e3bbf37`（v0.1.14 之后一个提交） |
| TVM 子模块 | `3rdparty/tvm` = `907a88c8791ccf33b9874821bc875e7abf624367`（TileLang/tvm `tilelang_main`，2026-08-29） |
| tvm-ffi 子模块 | `3rdparty/tvm/3rdparty/tvm-ffi` = `3c35034f`（= tag v0.1.11）；其 dlpack `84d107bf`，libbacktrace `79392187` |
| CUTLASS 子模块 | `3rdparty/cutlass` = `b2dd65dc864e09688245b316ac46c4a6cd07e15c` |
| apache-tvm-ffi（pip，运行时实际加载的 libtvm_ffi.so） | 0.1.12（要求范围 `>=0.1.11,<0.1.13`） |
| CUDA toolkit | 12.9（nvcc V12.9.41），`/usr/local/cuda-12.9`（`/usr/local/cuda` 也指向它） |
| 驱动 | 580.173.02（驱动 CUDA 版本 13.0） |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition，cc 12.0；TileLang 自动选的 target 是 `sm_120a` |
| 每 CTA shared memory 上限 | 101376 B（99 KB，opt-in）；每 SM 102400 B |
| Python | 3.12.3（`~/mpk-env`） |
| torch | 2.8.0+cu128 |
| triton | 3.4.0 |
| numpy | 2.2.6 |
| z3-solver | 4.16.0.0（见 §3.2） |
| Cython | 3.3.0 |
| gcc / g++ | 13.3.0（Ubuntu 13.3.0-6ubuntu2~24.04.1） |
| cmake | 3.28.3（/usr/bin/cmake） |
| ninja | 1.13.2（pip，`~/mpk-env/bin/ninja`） |
| OS / 内核 | Ubuntu 24.04.3 LTS，Linux 6.8.0-139-generic，glibc 2.39 |

## 2. 编译步骤（可复现）

```bash
cd /home/ywc/co-tilelang

# 2.1 子模块：浅克隆，只取需要的（约 2 分钟，共约 270 MB）
git submodule update --init --depth 1 3rdparty/tvm
# TVM 下面这四个子模块在 USE_CUTLASS=OFF / USE_OPENCL=OFF（默认）时用不到，
# 标成 update=none（只改本地 .git/modules/3rdparty/tvm/config，不动任何被跟踪文件），
# 否则 CMake 配置时自动执行的 `git submodule update --init --recursive` 会把它们（含多份 cutlass）完整克隆下来
for s in 3rdparty/cutlass 3rdparty/OpenCL-Headers 3rdparty/cutlass_fpA_intB_gemm 3rdparty/libflash_attn; do
  git -C 3rdparty/tvm config "submodule.$s.update" none
done
git -C 3rdparty/tvm submodule update --init --recursive --depth 1 3rdparty/tvm-ffi
git submodule update --init --depth 1 3rdparty/cutlass
# 撤销上面的跳过设置：git -C 3rdparty/tvm config --unset submodule.<path>.update

# 2.2 ~/mpk-env 补运行时依赖（dry-run 确认不会升级或降级任何已有包）
~/mpk-env/bin/pip install --no-cache-dir "apache-tvm-ffi>=0.1.11,<0.1.13" torch-c-dlpack-ext cloudpickle ml-dtypes psutil
#   装上了 apache-tvm-ffi 0.1.12、torch_c_dlpack_ext 0.1.5、cloudpickle 3.1.2、ml_dtypes 0.6.0、psutil 7.2.2（共约 15 MB）

# 2.3 配置与编译（配置 5 s；编译 95 s，-j96）
mkdir -p build && cd build
export PATH=/home/ywc/mpk-env/bin:/usr/local/cuda-12.9/bin:$PATH CUDA_HOME=/usr/local/cuda-12.9
cmake .. -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=/home/ywc/mpk-env/bin/python -DPython3_EXECUTABLE=/home/ywc/mpk-env/bin/python
ninja -j 96
# 以后改了 C++：source research/env.sh && cmake --build "$CO_TILELANG_ROOT/build"
```

- `build/` 为 506 MB（Release；关键产物在 `build/lib`：libtilelang.so、libtvm_compiler.so、libtvm_runtime.so、libtvm_ffi.so、stub_cuda/cudart/nvrtc、libz3.so.4.16、tilelang_cython_wrapper）。选 Release 是为了省盘；要调试符号就另开一个 RelWithDebInfo 的 build 目录，会大好几倍。
- 没有 ccache，编译也没有产生 `build/` 之外的中间文件。kernel 缓存在 `~/.tilelang/cache`（冒烟测试后约 5 MB）。
- 构建日志：`build/cmake_configure.log`、`build/ninja_build.log`（只有 4 条 TVM 自带的良性 warning）。

## 3. 需要注意的点

### 3.1 运行时的库解析
- `tilelang/env.py` 看到源码树（`tilelang/3rdparty` 不存在）就进入 dev 模式，从 `build/lib` 和 `build/tvm` 加载，打印 `Loading tilelang libs from dev root: .../build`。CUTLASS 头文件用 `3rdparty/cutlass/include`，模板用 `src/`。
- **libtvm_ffi.so 用的是 pip 包里的（0.1.12）**，不是 `build/lib/libtvm_ffi.so`（按子模块 v0.1.11 编的）：Python 先 `import tvm_ffi` 加载了 pip 的库，之后 libtvm_runtime.so 按 SONAME 复用它。进程里只有一份，所以不会出现两份注册表。这和上游 wheel 的做法一致（wheel 排除 libtvm_ffi.so，依赖 pip 包）。
- JIT 用 `$CUDA_HOME/bin/nvcc`（12.9）编 `sm_120a`，执行后端默认是 tvm_ffi。

### 3.2 z3-solver 4.16 超出上游约束 `<4.15.5`
- 上游在 2026-02-09 加的这个上限（#1817，没写原因）。查 PyPI：z3-solver 4.15.5（2026-02-07 发布）**没有 manylinux x86_64 wheel**，pip 会退回源码编译；4.16.0.0 又有了 manylinux_2_27 的 wheel。所以这个上限看起来是打包层面的临时规避，与正确性无关。
- `~/mirage-compiler` 的 `core.*.so` 链接的是 `libz3.so.4.16`，降级会弄坏 Mirage，所以没动。TileLang 编译时链接 z3 4.16，并把 libz3 拷进 `build/lib`（rpath `$ORIGIN`）。编译和全部冒烟测试都通过。

- **补丁（2026-09-22，CoKernel 构建器 v0 时加入）**：`research/patches/tvm_z3_canprove_exception.patch`，作用于子模块 `3rdparty/tvm`（基于 907a88c）的 `src/target/z3/z3_prover_on.cc`。Z3 在资源上限（rlimit）耗尽时有时会抛出 `canceled` 异常，而不是返回 `unknown`，导致 LayoutInference / LowerTileOp 在带 swizzle layout 的 GEMM 下标恒等式上崩溃（CoKernel 的 4 个 kernel 触发）。补丁让 `CanProve` 把任何 Z3 异常当作"无法证明"（它本来就是保守的后备证明器）。是否与 z3 4.16（超出上游约束）有关尚未确认。子模块的远端不归我们，所以以补丁文件形式保存；重新检出子模块后需执行：
  ```bash
  git -C 3rdparty/tvm apply ../../research/patches/tvm_z3_canprove_exception.patch && cmake --build build
  ```

### 3.3 sm_120 相关的观察（对后续研究有用）
- TMA 可用（`TargetHasBulkCopy` 条件是 arch ≥ 90），GEMM 走 `mma.sync`（`ldmatrix` + `mma_sync m16n8k16`），没有 wgmma/tcgen05。
- **基础 gemm 示例（`threads=128`）被自动做了 warp specialization**：生成的 kernel 是 `__launch_bounds__(256, 1)`，前 128 线程作 TMA producer（`warpgroup_reg_dealloc<24>`），后 128 线程作 mma consumer（`warpgroup_reg_alloc<240>`），用 mbarrier 做 3 级流水。实际的线程数和寄存器分配都和源码写的不一样，做资源签名时要以编译产物为准。
- **shared memory 超过 99 KB 的配置在编译期不报错，要到 launch 时才失败**（`Failed to set the allowed dynamic shared memory size to N`）。候选生成需要自己按 101376 B 预先检查。
- `T.wgmma_gemm` 在 sm_120 上会直接报错（例如 flashmla 的 ws 示例）。

## 4. 冒烟测试结果（2026-09-22 19:35–19:39，GPU 空闲，未锁频）

日志与脚本：`research/results/2026-09-22_p0-1_smoke/`（注意：`.gitignore` 有 `*.log`，提交日志需要 `git add -f`）。延迟是示例自己打印的数（各示例的计时方式不同，L2 是热的，也没控制频率），只作为能跑通的佐证，不能当性能数据用。

| 示例 | 结果 | 示例打印的数字 |
|---|---|---|
| `examples/gemm/example_gemm.py`（1024³ fp16） | 通过（`assert_close` rtol/atol 1e-2） | 0.01956 ms（cupti），约 110 TFLOPS；编译约 3 s |
| `examples/flash_decoding/example_gqa_decode.py`（原样运行） | **失败**：launch 时要 147456 B 动态 smem > 101376 B | — |
| 同上，经 `run_gqa_decode.py` 换配置（block_N=64，num_stages=2，其余为示例默认值：block_H=64，num_split=8，threads=128；约 80 KB） | 通过（o_ref diff 1.30e-6，o_ref_split diff 5.37e-6） | TL 0.04 ms / 3.27 TFLOPS（约 41 µs）；torch 参考 0.09 ms |
| 同上（block_N=128，num_stages=1；约 80 KB） | 通过（1.32e-6 / 5.38e-6） | TL 0.04 ms / 3.64 TFLOPS（约 37 µs） |
| `examples/norm/rms_norm.py`（8192×8192 fp32） | 通过（`assert_allclose` 0.01） | TL 0.36 ms（约 1.49 TB/s）；torch 参考 0.86 ms |
| `warp_specialize/..._copy_1_gemm_0.py`（16384³） | 通过 | 30.94 ms（约 284 TFLOPS） |
| `warp_specialize/..._copy_0_gemm_1.py`（1024³） | 通过 | 0.0333 ms |
| `warp_specialize/..._copy_gemm_0_1.py`（128×128×64，两个 warpgroup 各做一半 N） | 通过 | 0.0062 ms |
| `warp_specialize/..._softpipe_stage2.py`（16384³） | 通过 | 31.04 ms |
| `warp_specialize/..._barrierpipe_stage2.py`（16384³） | 通过 | 36.96 ms |
| `warp_specialize/example_warp_specialize_flashmla.py` | 失败（预期之内）：显式调用 `T.wgmma_gemm`，只有 Hopper 能用 | — |

GQA decode 失败的根因：示例的 `get_heuristic_config()` 只对 sm_89 特判，其余 GPU 一律用 `block_N=128, block_H=64, num_split=8, num_stages=2`。这组配置需要 Q 16 KB + K/V 双缓冲 2×2×32 KB = 144 KB，只适合 A100/H100 这类大 smem 的卡；sm_86、sm_89、sm_120 每 CTA 最多 99 KB。示例文件没有改，改用外部包装脚本替换配置后，调用的仍是示例自己的 `main()`，示例自带的两项相似度检查都通过。注意：示例里的 `assert_similar` 默认 `assert_=False`，失败时只打印红字，不会抛异常，也照样会打印 "All checks pass."；因此判断是否通过要看 `passed:` 那两行。

上游 CI 本来就不会覆盖这些情况：`test_example_flash_decoding.py` 限定 cc ≤ 8.9，`test_example_warp_specialize.py` 限定 cc == 9.0。另外 `~/mpk-env` 里没有 pytest，所以示例是直接用 `python` 跑的，没有经过 pytest。

## 5. FlashInfer（P1-baselines，2026-09-22）

用途：GQA decode / prefill 的单跑参考实现，以及 POD-Attention（kernel 内 prefill+decode 编排）的现成基线。测量结果见 `research/results/2026-09-22_flashinfer_baselines/`，包装模块 `research/bench/baselines/flashinfer_ops.py`。

### 5.1 安装

```bash
source research/env.sh
pip install --dry-run --no-cache-dir --report dry.json flashinfer-python==0.7.0   # 先 dry-run：只有新增，没有任何已装包升级/降级
pip install --no-cache-dir flashinfer-python==0.7.0
```

- `flashinfer-python 0.7.0`：纯 Python 的 JIT wheel（`py3-none-any`），CUDA 源码与 CUTLASS/CCCL 头文件随包附带，第一次调用时用本机 nvcc 编译。
- 新增 15 个包（安装前后 `pip list` 逐项对比：**已有包版本全部不变**，torch 仍是 2.8.0+cu128，apache-tvm-ffi 仍是 0.1.12）：flashinfer-python 0.7.0、nvidia-cutlass-dsl 4.8.0（+ libs-base / libs-core / libs-cu12 / libs-cu13 4.8.0）、nvidia-cudnn-frontend 1.29.0、cuda-core 1.2.0、cuda-tile 1.6.0、nccl4py 0.5.0、nccl-extensions 0.1.0、nvidia-ml-py 13.610.43、protobuf 7.36.2、click 8.5.0、tabulate 0.10.0。
- 占盘：根分区少了约 0.93 GB（site-packages 里 `nvidia_cutlass_dsl` 485 MB、`flashinfer` 258 MB、`nccl` 104 MB、`cudnn`（frontend）55 MB、`cuda/core` 18 MB 等）。`--no-cache-dir`，pip 缓存没有增长。
- 卸载：`pip uninstall` 上面 15 个包，再 `rm -rf ~/.cache/flashinfer`。

### 5.2 JIT 与缓存

- 编译器：`$CUDA_HOME/bin/nvcc`（12.9；FlashInfer 按 `CUDA_HOME` → `which nvcc` 的顺序找）。目标：`-gencode=arch=compute_120f,code=sm_120f`（FlashInfer 对 SM 12.x 用 family 后缀 `f`，要求 CUDA ≥ 12.9；可用 `FLASHINFER_CUDA_ARCH_LIST` 覆盖）。
- 缓存目录：`~/.cache/flashinfer/0.7.0/120f/`（`cached_ops/` 放编好的 `.so` 与 `.o`，`generated/` 放模板实例化出的源码；基目录可用 `FLASHINFER_WORKSPACE_BASE` 改）。
- 本任务编了 4 个模块（bf16，head_dim 128，无 RoPE/滑窗/soft-cap），首次编译耗时（96 核）：`batch_decode` 约 8 s，`batch_prefill`（tensor-core decode 走它）约 18 s，`single_prefill` 约 10 s，`pod_with_kv_cache` 约 107 s（16 种 mask 组合的实例化）。4 个模块编完后 `~/.cache/flashinfer` 共 51 MB。（同一任务里 TileLang GQA decode 示例的 48 个配置使 `~/.tilelang/cache` 从 41 MB 涨到 74 MB。）

### 5.3 与现有环境的相互影响

- FlashInfer 0.7.0 要求 `apache-tvm-ffi>=0.1.11,<0.2`，直接复用已装的 0.1.12（与 TileLang 相同）。同一进程先后 `import tilelang, flashinfer` 正常。FlashInfer 的 kernel 通过 tvm-ffi 调用，launch 在 **torch 当前 stream** 上（tvm-ffi 的 DLPack exchange API 在调用时取 torch 的 current stream），所以 `with torch.cuda.stream(s)`（包括 green context 的 ExternalStream）对它生效。
- `nvidia-cutlass-dsl` 装了一个 `nvidia_cutlass_dsl_packages.pth`：每次 Python 启动都会 import `nvidia_cutlass_dsl` 并把其 `dsl_packages/` 插到 `sys.path` 最前面，于是 `import cutlass`（CuTe DSL）现在能成功。TileLang 只有在显式选 `cutedsl` 后端时才会用到它（`tilelang/jit/adapter/cutedsl/checks.py`），默认的 tvm_ffi 后端不受影响。
- `nvidia-ml-py` 提供 `pynvml`；cobench 自己用 ctypes 调 NVML，不受影响。

### 5.4 sm_120 上的已知问题（FlashInfer 0.7.0）

- POD（`PODWithPagedKVCacheWrapper`）能编译、结果正确，但它的 CTA 调度计数器 `tbAssign`（进程级 `static int*`）是用不带 stream 的 `cudaMemset` 清零的（`include/flashinfer/attention/pod.cuh:414-415`），即落在 legacy default stream 上，而 kernel 本身在 torch 当前 stream 上。后果（实测）：在非阻塞 stream（torch 的 side stream、green context stream）上连续调用会得到错误结果；放进 CUDA graph 时只有第一次 replay 正确，之后每次 replay 约 15 µs 就结束、什么都没算。因此 POD 只能在 legacy default stream 上、不用 graph 来测。详见结果目录的 README。
- **2026-09-24 起已在本地打补丁修复**，见 §5.5。

### 5.5 本地补丁：POD 计数器在 kernel 所在流上复位（P4-a，2026-09-24）

- 补丁文件：`research/patches/flashinfer_pod_stream_memset.patch`，作用于 site-packages 里 FlashInfer 0.7.0 附带的头文件 `flashinfer/data/include/flashinfer/attention/pod.cuh`（原文件 md5 `2e7379940684ef9387475e211e82db03`，打补丁后 `0ed77691b03239590b8c58cd418360d7`）。
- 改动：把不带 stream 的 `cudaMemset(tbAssign, ...)` 换成 `cudaMemsetAsync(tbAssign, 0, ..., stream)`（与 kernel 同一条流，按流序执行，也会被 CUDA graph 捕获成 memset 节点）；计数器缓冲区改为每个设备一份（原来是整个进程一个指针）。kernel 代码不变。
- 仍有的限制：同一设备上时间重叠的两个 POD launch（不同流）仍共用计数器，不要并发跑两个 POD；第一次调用不能发生在 stream capture 中（缓冲区在首次调用时 `cudaMalloc`）。`flashinfer_ops.POD.run` 会检查后者。
- 打补丁 / 撤销：
  ```bash
  source research/env.sh
  FI=$(python -c 'import flashinfer,os;print(os.path.dirname(flashinfer.__file__))')
  patch -p1 -d "$FI" < research/patches/flashinfer_pod_stream_memset.patch        # 打补丁
  patch -R -p1 -d "$FI" < research/patches/flashinfer_pod_stream_memset.patch     # 撤销
  python -c "import sys; sys.path.insert(0,'research/bench'); from baselines import flashinfer_ops as fo; print(fo.pod_patch_status())"
  ```
- JIT 缓存会自动重编：FlashInfer 用 ninja（`deps = gcc` 的依赖文件）管理 `~/.cache/flashinfer/0.7.0/120f/cached_ops/pod_with_kv_cache_*`，头文件变了之后第一次加载 POD 模块时重编 16 个实例（约 130 s，96 核），`pod_patch_status()["pod_so"]` 报告 `.so` 是否比头文件新。撤销补丁后同样会自动重编回原版。
- 验证（`research/bench/scripts/p4_pod_patch.py`，结果 `research/results/2026-09-24_p4_prep/pod_patch_{unpatched,patched}.json`）：8 个 P4 配置 × 3 组输入，默认流上补丁版与原版输出逐位相同（SHA-256）；补丁版在 torch side stream（每配置 24 次背靠背调用）、green context stream（96 SM）、CUDA graph（每配置捕获一次、回放 24 次，每次回放前换输入并把输出置 NaN）上共 576 次调用全部与默认流结果逐位相同；原版在同样的测试上 side 44/48、green 48/48、graph 46/48 次错误。
- `pip install --force-reinstall flashinfer-python==0.7.0` 会覆盖补丁，需要重新执行上面的 `patch`。

## 6. flash-attn 2.8.3.post1（claims-repro 子研究 B，2026-09-24）

用途：C2（FlashAttention 前向）的额外参照（独立的 FA2 实现；PyTorch SDPA 的 flash 后端内置的也是 FA2）。结果见 `research/results/2026-09-24_claims_repro/B_attention/`。

```bash
source research/env.sh
# 官方预编译 wheel（GitHub release v2.8.3.post1，256 MB，sha256 9a08775a…3d86e），不从源码编译
curl -L -O https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install --dry-run --no-cache-dir --report dry.json ./flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl   # 只新增 flash_attn
pip install --no-cache-dir --no-deps ./flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

- 安装前后 `pip list` 对比：只多了 `flash_attn 2.8.3.post1`，其余包版本全部不变（torch 2.8.0+cu128、triton 3.4.0、apache-tvm-ffi 0.1.12、flashinfer 0.7.0）。
- `flash_attn_2_cuda.*.so` 含 sm_80 / sm_90 / sm_100 / sm_120 的 SASS（`cuobjdump --list-elf` 各 72 个 cubin），在本机 sm_120 上直接可用。
- 占盘：site-packages 多约 960 MB（`flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so` 955 MB）。wheel 文件已删除，pip 缓存未增长。
- 卸载：`pip uninstall flash_attn`。

## 7. BitBLAS 0.1.0.post1（claims-repro 子研究 A，2026-09-24；已卸载）

用途：C5（反量化 GEMV）的声明就是用 BitBLAS（TileLang 后端）测的，想用原工件复现。结果见 `research/results/2026-09-24_claims_repro/A_gemm/`。

```bash
source research/env.sh
pip install --dry-run --no-cache-dir --report dry.json bitblas          # 只有新增，没有任何已装包升级/降级
pip download --no-deps --no-cache-dir -d whl bitblas==0.1.0.post1       # 82 MB wheel，先检查内容
pip install --no-cache-dir whl/bitblas-0.1.0.post1-py3-none-manylinux1_x86_64.whl
```

- 新增 14 个包（安装前后 `pip list` 对比，已有包版本全部不变）：bitblas 0.1.0.post1、attrs 26.1.0、cffi 2.1.1、cpplint 2.0.2、decorator 5.3.1、docutils 0.23、dtlib 0.0.0.dev2、execnet 2.1.2、pycparser 3.0、pytest-xdist 3.8.0、RapidFuzz 3.14.6、scipy 1.18.1、thefuzz 0.22.1、tornado 6.5.10。site-packages 多 467 MB（bitblas 自带旧版 TVM 153 MB、CUTLASS 95 MB、旧版 TileLang 7 MB，都在 `bitblas/3rdparty` 下；import bitblas 时会把它们插到 `sys.path` 最前面，所以必须在不含本 fork tilelang 的进程里用，脚本用 `env -u PYTHONPATH`）。
- **在 sm_120 上一个 kernel 都编不出来**：（1）按原样（自动检测 target `cuda`，sm_120）：`bitblas/ops/general_matmul/tilelang/dequantize/matmul_dequantize.py` 的 `dispatch_scheduler` 只认 Volta/Ampere/Ada/Hopper，报 `Unsupported architecture`；（2）把 TVM target 改成 `cuda -arch=sm_89`（Ada 调度，nvcc 仍按设备编 `compute_120`）：自带旧 TileLang 的 `tl_templates/cuda/gemm.h` 对 `__CUDA_ARCH_LIST__ >= 900` 一律包含 Hopper 的 `gemm_sm90.h`，nvcc 报 `identifier "warpgroup_wait" is undefined`。证据：`research/results/2026-09-24_claims_repro/A_gemm/bitblas_sm120_build_errors.txt`。
- 因此 bitblas 已卸载（`pip uninstall -y bitblas`，释放约 300 MB）；它的 13 个依赖包仍在（共约 170 MB，scipy 占 140 MB），没有其他代码依赖它们，需要时可 `pip uninstall -y attrs cffi cpplint decorator docutils dtlib execnet pycparser pytest-xdist rapidfuzz scipy thefuzz tornado`（其中 attrs/cffi/pycparser 若被后装的包依赖则保留）。
- 另外：子研究 A 编译了约 1.6 万个 GEMM 配置，`~/.tilelang/cache` 曾增长约 6.5 GB，结束时已按 params.pkl 的形状精确删除这些条目（其他子研究的条目未动）；TileLang 的 cython 执行后端会在 `/tmp` 留下 `tmp*.cu/.so`（`tilelang/jit/adapter/libgen.py` 用 `delete=False`），fp8 编译留下的约 4.6 GB 也已删除。

## 8. mamba-ssm Triton 基线（claims-repro 子研究 C，2026-09-24）

用途：C3 / C6（Mamba-2 chunk-scan / chunk-state）的 Triton 基线。结果见 `research/results/2026-09-24_claims_repro/C_mamba/`。

- **`~/mpk-env` 没有任何改动**（没有安装、升级或卸载任何包）。
- mamba-ssm **没有安装**，而是把 PyPI sdist `mamba_ssm-2.2.6.post3.tar.gz`（sha256 `826a3cdb…a7bf3`，`benchmark/mamba2/README.md` 所用版本）里的 4 个纯 Triton 文件原样拷到 `research/bench/baselines/mamba_ssm_triton/mamba_ssm/ops/triton/`（`ssd_chunk_scan.py`、`ssd_chunk_state.py`、`ssd_bmm.py`、`softplus.py`，md5 见该目录的 `mamba_ssm/__init__.py`），并附 Apache-2.0 LICENSE。自带的 `__init__.py` 不导入 CUDA 扩展（上游的会导入 `selective_scan_cuda` 和依赖 `transformers` 的模型类）。用法：`sys.path.insert(0, "research/bench/baselines/mamba_ssm_triton")` 之后 `from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd`，示例文件里的调用无需修改。
- 其余只在 scratch 目录里用 `pip install --no-deps --no-cache-dir --target <dir>` 临时装过、从不进默认 `sys.path`、用完已删：pillow 12.3.0（读 JPEG 图）、tilelang 0.1.14 wheel + z3-solver 4.15.4.0（fork 与上游对比；wheel 链接 `libz3.so.4.15`，另用独立的 `TILELANG_CACHE_DIR`）、helion 0.2.1（只做了 import 测试）、fla-core 0.5.2（linear attention 的可选基线）。
- `~/.tilelang/cache` 因本子研究多了 935 个 kernel 目录（约 0.55 GB，外加 `cuda-binaries/` 里对应的 cubin）；`~/.tilelang/cache/torch_extensions` 4.4 MB（minference 示例自己编的 index 转换算子）。
- sm_120 上的相关观察：chunk-scan 示例 90 个 autotune 配置中 72 个在 launch 时因 smem > 99 KB 失败（2 个编译失败：layout 单射性检查超出 262144 点上限），只剩 16 个；上游 autotuner 会跳过它们，但每个失败配置都泄漏它刚分配的输入张量，大 shape 时会把后续配置挤成 CUDA OOM（CC4 实测涨到 56 GB）。TileLang 在 sm_120 上默认对这些 kernel 做 warp specialization（256 线程、每线程约 240 寄存器 → 每 SM 1 个 CTA），关掉（`tl.disable_warp_specialized`）后 chunk-scan 快 15–25 %。
