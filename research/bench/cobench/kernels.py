"""Small raw CUDA kernels used by the framework (compiled lazily with NVRTC).

* ``HostGate``: a 1-thread kernel that spins until a host-written counter in pinned
  memory reaches a target. Placed right before the start event, it guarantees the
  host has finished enqueuing all work of a rep before the clock starts.
* ``copy_u4``: vectorised (16B) grid-stride copy, used to validate DRAM GB/s.
* ``mma_peak``: register-only ``mma.sync.m16n8k16`` bf16->fp32 loop; measures the
  tensor-core FLOP/clk/SM to derive the dense bf16 peak independently of datasheets.
"""
from __future__ import annotations

import functools

import numpy as np
import torch

from .cudrv import CudaKernel, device_index

_GATE_SRC = r"""
extern "C" __global__ void host_gate(const volatile unsigned long long* flag,
                                     unsigned long long target,
                                     unsigned long long timeout_ns,
                                     unsigned long long* timeouts) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (*flag < target) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t - t0 > timeout_ns) { atomicAdd(timeouts, 1ull); return; }
    __nanosleep(200);
  }
}
"""


class HostGate:
    """Release GPU work only after the host has enqueued a whole rep.

    ``gate.wait(stream)`` enqueues a spin kernel waiting for ticket ``i``;
    ``gate.release()`` publishes ticket ``i`` (host write to pinned memory).
    Tickets are monotonic, so reps can be pipelined without resets.

    Caveat: anything the host does between ``wait`` and ``release`` that needs a
    context-wide synchronisation deadlocks against the spinning gate until the timeout.
    The common culprit is CUDA lazy module loading on the *first* launch of a kernel
    in a context, so callers must run the workload once, ungated, on the same streams
    first (bench/bench_corun do this). Other culprits: .item()/.cpu(), cudaFree.
    """

    def __init__(self, device=None, timeout_s: float = 2.0):
        self.device = device_index(device)
        self.kernel = _gate_kernel(self.device)
        self.flag = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self._flag_np = self.flag.numpy()
        self.timeouts = torch.zeros(1, dtype=torch.int64, device=f"cuda:{self.device}")
        self.timeout_ns = int(timeout_s * 1e9)
        self._next = 0
        self._seen_timeouts = 0

    def wait(self, stream=None) -> int:
        self._next += 1
        self.kernel(1, 1, self.flag.data_ptr(), self._next, self.timeout_ns, self.timeouts,
                    stream=stream)
        return self._next

    def release(self) -> None:
        self._flag_np[0] = self._next

    def check(self, where: str = "") -> None:
        """Call after synchronisation; raises if a gate timed out since the last check
        (the affected reps are invalid: work started before the host finished enqueuing)."""
        n = int(self.timeouts.item())
        new, self._seen_timeouts = n - self._seen_timeouts, n
        if new:
            raise RuntimeError(
                f"HostGate{(' (' + where + ')') if where else ''}: {new} gate(s) timed out. The "
                "workload probably forces a host-device synchronisation while enqueuing (lazy "
                "module loading of a not-yet-launched kernel, .item()/.cpu(), cudaFree, ...). "
                "Pre-warm it on the same stream, remove the sync, or pass gate=False.")


@functools.lru_cache(maxsize=None)
def _gate_kernel(device: int) -> CudaKernel:
    k = CudaKernel(_GATE_SRC, "host_gate", "pQQp", device=device)
    # max smem carveout: the gate's SM must not need a reconfiguration before a large-smem
    # CTA of the measured kernel can run there (see cobench.clock)
    k.set_carveout(100)
    return k


_COPY_SRC = r"""
extern "C" __global__ void copy_u4(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                   unsigned long long n) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  #pragma unroll 4
  for (; i < n; i += stride) dst[i] = src[i];
}
"""


@functools.lru_cache(maxsize=None)
def _copy_kernel(device: int) -> CudaKernel:
    return CudaKernel(_COPY_SRC, "copy_u4", "ppQ", device=device)


def copy_u4(src: torch.Tensor, dst: torch.Tensor, *, ctas_per_sm: int = 4, threads: int = 512,
            stream=None) -> None:
    """dst[:] = src[:] with 16B vector accesses (both contiguous, nbytes % 16 == 0)."""
    if src.nbytes != dst.nbytes or src.nbytes % 16 or not (src.is_contiguous() and dst.is_contiguous()):
        raise ValueError("copy_u4 needs equal-size contiguous tensors with nbytes % 16 == 0")
    dev = src.device.index
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    _copy_kernel(dev)(nsm * ctas_per_sm, threads, src, dst, src.nbytes // 16, stream=stream)


_READ_SRC = r"""
extern "C" __global__ void read_u4(const uint4* __restrict__ src, unsigned long long n,
                                   unsigned int magic, unsigned int* sink) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  unsigned int acc = 0u;
  #pragma unroll 4
  for (; i < n; i += stride) { const uint4 v = src[i]; acc += v.x ^ v.y ^ v.z ^ v.w; }
  // data-dependent (never true for cobench's flush buffer): keeps every load alive
  if (acc == magic) atomicAdd(sink, 1u);
}
"""


@functools.lru_cache(maxsize=None)
def _read_kernel(device: int) -> CudaKernel:
    return CudaKernel(_READ_SRC, "read_u4", "pQIp", device=device)


@functools.lru_cache(maxsize=None)
def _read_sink(device: int) -> torch.Tensor:
    return torch.zeros(1, dtype=torch.int32, device=f"cuda:{device}")


def read_u4(src: torch.Tensor, *, ctas_per_sm: int = 4, threads: int = 512, stream=None) -> None:
    """Read every byte of ``src`` with 16B vector loads (default caching: lines are allocated
    in L2). Used by the "clean" L2 flush: after a 2xL2 write, reading 2xL2 evicts the dirty
    lines (written back) and leaves only clean lines behind."""
    if src.nbytes % 16 or not src.is_contiguous():
        raise ValueError("read_u4 needs a contiguous tensor with nbytes % 16 == 0")
    dev = src.device.index
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    _read_kernel(dev)(nsm * ctas_per_sm, threads, src, src.nbytes // 16, 0x9E3779B9, _read_sink(dev),
                      stream=stream)


_DISCARD_SRC = r"""
extern "C" __global__ void discard_l2(char* p, unsigned long long nlines) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (; i < nlines; i += stride)
    asm volatile("discard.global.L2 [%0], 128;" :: "l"(p + i * 128) : "memory");
}
"""


@functools.lru_cache(maxsize=None)
def _discard_kernel(device: int) -> CudaKernel:
    return CudaKernel(_DISCARD_SRC, "discard_l2", "pQ", device=device)


def discard_l2(buf: torch.Tensor, *, ctas_per_sm: int = 4, threads: int = 512, stream=None) -> None:
    """``discard.global.L2`` over every 128 B line of ``buf`` (sm_80+): the lines are
    invalidated WITHOUT write-back, so the buffer's contents become undefined. Used by the
    "clean" L2 flush after writing the 2x L2 flush buffer: the write evicts everything else
    (writing back any dirty lines), the discard then drops the flush buffer's own dirty
    lines, leaving an L2 with no dirty lines and none of the op's data."""
    if buf.nbytes % 128 or buf.data_ptr() % 128 or not buf.is_contiguous():
        raise ValueError("discard_l2 needs a contiguous, 128-byte aligned tensor with nbytes % 128 == 0")
    dev = buf.device.index
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    _discard_kernel(dev)(nsm * ctas_per_sm, threads, buf, buf.nbytes // 128, stream=stream)


_MMA_SRC = r"""
template <int NACC>
__device__ void body(float* out, unsigned long long* cycles, int iters) {
  unsigned a0 = 0x3f803f80u ^ threadIdx.x, a1 = 0x3f003f00u ^ (threadIdx.x << 3);
  unsigned a2 = 0x3e803e80u ^ (threadIdx.x << 5), a3 = 0x3f403f40u;
  unsigned b0 = 0x3f803f00u ^ threadIdx.x, b1 = 0x3e003f80u;
  float acc[NACC][4];
  #pragma unroll
  for (int j = 0; j < NACC; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
  __syncthreads();
  unsigned long long c0 = clock64();
  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int j = 0; j < NACC; ++j) {
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  __syncthreads();
  unsigned long long c1 = clock64();
  float s = 0.f;
  #pragma unroll
  for (int j = 0; j < NACC; ++j) s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (threadIdx.x == 0) cycles[blockIdx.x] = c1 - c0;
}
extern "C" __global__ void mma_peak4(float* out, unsigned long long* cycles, int iters) { body<4>(out, cycles, iters); }
extern "C" __global__ void mma_peak8(float* out, unsigned long long* cycles, int iters) { body<8>(out, cycles, iters); }
"""

MMA_FLOP = 2 * 16 * 8 * 16  # one m16n8k16


@functools.lru_cache(maxsize=None)
def _mma_kernel(device: int, nacc: int) -> CudaKernel:
    return CudaKernel(_MMA_SRC, f"mma_peak{nacc}", "ppi", device=device)


def mma_peak(*, warps_per_cta: int = 8, nacc: int = 8, iters: int = 20000, ctas_per_sm: int = 1,
             device=None) -> dict:
    """Measure dense bf16 (fp32 accumulate) mma.sync throughput.

    Returns FLOP/clk/SM (from per-CTA clock64 cycles) and achieved TFLOP/s (from CUDA
    events); peak(f) = flop_per_clk_per_sm * num_sms * f.
    """
    dev = device_index(device)
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    k = _mma_kernel(dev, nacc)
    grid, block = nsm * ctas_per_sm, 32 * warps_per_cta
    out = torch.empty(grid * block, dtype=torch.float32, device=f"cuda:{dev}")
    cyc = torch.empty(grid, dtype=torch.int64, device=f"cuda:{dev}")
    k(grid, block, out, cyc, 50)  # warm (module load, clocks)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    k(grid, block, out, cyc, iters)
    e1.record()
    e1.synchronize()
    ms = e0.elapsed_time(e1)
    c = cyc.cpu().numpy().astype(np.float64)
    flop_per_cta = warps_per_cta * iters * nacc * MMA_FLOP
    total_flop = flop_per_cta * grid
    return {
        "warps_per_cta": warps_per_cta, "nacc": nacc, "iters": iters, "ctas_per_sm": ctas_per_sm,
        "flop_per_clk_per_sm": flop_per_cta * ctas_per_sm / float(np.median(c)),
        "cycles_median": float(np.median(c)),
        "time_ms": ms,
        "tflops": total_flop / (ms * 1e-3) / 1e12,
        # effective SM clock implied by per-CTA cycles over wall time (ns)
        "implied_clock_mhz": float(np.median(c)) / (ms * 1e3),
    }
