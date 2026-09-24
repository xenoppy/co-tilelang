"""cobench: measurement framework v0 for co-tilelang (P0-2).

API summary (all times in microseconds):
  bench(fn, *, make_inputs, mode="flush"|"graph"|"hot", warmup, reps, ...) -> BenchResult
  bench_corun(fn_a, fn_b, *, stream_a, stream_b, flush, flush_kind, extra, ...) -> CorunResult
  bench_variants(variants, *, reference, flush_kind, ...) -> VariantsResult   (flush mode, interleaved)
  bench_steady(variants, *, reference, slice_s, rounds, ...) -> SteadyResult  (primary co-run mode)
  Par((name, stream, fn), ...), Rotation(make_inputs)   variant building blocks
  GpuGuard / wait_until_free() / foreign_activity(t0, t1)   GPU-sharing guard (pmon rule)
  NvmlSampler(interval_ms=10)            context manager; .summary()
  split_sms(n, ignore_coscheduling=False) -> SmPartition (.stream / .rest_stream / .n_sms)
  probe_ctas(grid, block, smem, ...)     %smid/%globaltimer per CTA
  build_sm_remap(stream, expected)       -> SmRemap (smid -> dense index)
  CudaKernel(src, name, signature)       NVRTC + context-independent launch helper
  device_info()                          static device numbers (SMs, L2, DRAM peak, clocks)
"""
from __future__ import annotations

import torch

from .cudrv import CudaKernel, check, device_attr, ensure_init
from .green import GreenContext, SmPartition, query_split, split_sms
from .kernels import HostGate, copy_u4, discard_l2, mma_peak, read_u4
from .nvml import NvmlDevice, NvmlSampler, gpu_state
from .smid import (SmRemap, build_sm_remap, globaltimer_resolution, globaltimer_skew,
                   one_cta_per_sm_smem, probe_ctas)
from .timing import (DEFAULT_FLUSH, FLUSH_KINDS, BenchResult, CorunResult, VariantsResult, bench, bench_corun,
                     bench_variants, l2_bytes, summarize, tensor_bytes)
from .guard import (ComputeProc, GpuBusy, GpuGuard, GpuYield, GuardPolicy, foreign_activity, get_guard, get_policy,
                    free_mib, list_compute_procs, occupancy, occupied, set_policy, wait_loop, wait_until_free)
from .variants import Par, Rotation
from .steady import HostGapError, SteadyResult, bench_steady

__all__ = [
    "bench", "bench_corun", "BenchResult", "CorunResult", "summarize", "tensor_bytes", "l2_bytes",
    "FLUSH_KINDS", "DEFAULT_FLUSH", "bench_variants", "VariantsResult", "bench_steady", "SteadyResult", "HostGapError", "Par", "Rotation",
    "GpuGuard", "GpuBusy", "GpuYield", "GuardPolicy", "ComputeProc", "get_guard", "get_policy", "set_policy",
    "wait_until_free", "wait_loop", "occupancy", "occupied", "list_compute_procs", "foreign_activity", "read_u4", "discard_l2",
    "NvmlSampler", "NvmlDevice", "gpu_state",
    "split_sms", "query_split", "SmPartition", "GreenContext",
    "probe_ctas", "build_sm_remap", "SmRemap", "globaltimer_resolution", "globaltimer_skew",
    "one_cta_per_sm_smem", "CudaKernel", "HostGate", "copy_u4", "mma_peak", "device_info",
]


def device_info(device=None) -> dict:
    """Static device numbers. DRAM peak = bus_width/8 * mem_clock * 2 (DDR signalling)."""
    from cuda.bindings import driver as cu
    idx = torch.cuda.current_device() if device is None else int(device)
    ensure_init(idx)
    p = torch.cuda.get_device_properties(idx)
    A = cu.CUdevice_attribute
    bus_bits = device_attr(A.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, idx)
    mem_khz = device_attr(A.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, idx)
    nv = NvmlDevice(idx)
    return {
        "name": p.name, "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
        "l2_bytes": p.L2_cache_size, "smem_per_sm": p.shared_memory_per_multiprocessor,
        "smem_per_block_optin": p.shared_memory_per_block_optin,
        "regs_per_sm": p.regs_per_multiprocessor, "total_mem": p.total_memory,
        "mem_bus_bits": bus_bits, "mem_clock_mhz_driver": mem_khz / 1e3,
        "dram_peak_gbps": bus_bits / 8 * mem_khz * 1e3 * 2 / 1e9,
        "max_sm_clock_mhz_nvml": nv.max_clock_mhz(1), "max_mem_clock_mhz_nvml": nv.max_clock_mhz(2),
        "power_limit_w": nv.power_limit_w(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    }
