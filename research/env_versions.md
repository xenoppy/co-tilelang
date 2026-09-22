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
