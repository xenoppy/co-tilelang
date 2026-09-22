"""Target device description used by config filtering and occupancy calculation.

The defaults describe the RTX PRO 6000 Blackwell Workstation Edition (sm_120, see
research/plan.md §1.1 and research/env_versions.md). Every number below was measured
on this machine (torch device properties / cuda_occupancy.h for cc 12.0); override
with a different `DeviceSpec` for another GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceSpec:
    name: str = "RTX PRO 6000 Blackwell (sm_120)"
    arch: str = "sm_120a"  # TileLang's auto-detected target arch on this machine
    num_sms: int = 188
    # Shared memory. `smem_per_cta_optin` is the largest static+dynamic *user* smem a
    # CTA may request (cudaDevAttrMaxSharedMemoryPerBlockOptin). The driver additionally
    # reserves `smem_reserved_per_cta` bytes per CTA; on sm_90+ ptxas also reports that
    # reserved KB inside the cubin's static SHARED count (verified: an empty kernel
    # compiled for sm_120a shows SHARED:1024; ptxas -v shows 0 user bytes).
    smem_per_cta_optin: int = 101376
    smem_per_sm: int = 102400
    smem_reserved_per_cta: int = 1024
    smem_alloc_granularity: int = 128  # cudaOccSMemAllocationGranularity, cc 12
    # Registers (cudaOccRegAllocationGranularity / SubPartitions, cc 12).
    regs_per_sm: int = 65536
    regs_per_cta: int = 65536
    reg_alloc_granularity: int = 256  # registers per warp are rounded up to this
    reg_sub_partitions: int = 4
    max_regs_per_thread: int = 255
    # Thread / CTA slots.
    max_threads_per_cta: int = 1024
    max_threads_per_sm: int = 1536
    max_warps_per_sm: int = 48
    max_ctas_per_sm: int = 24
    warp_size: int = 32
    # cudaOccMaxBlocksPerSMBlockBarrierLimit, cc 12: barriers available per SM
    # = max_ctas_per_sm; a kernel using n named barriers fits max_ctas_per_sm // n CTAs.
    barriers_per_sm: int = 24

    @property
    def target(self) -> dict:
        """TileLang/TVM target dict. Passing it explicitly avoids target auto-detection
        (which initialises CUDA through torch) during compile-only work."""
        return {"kind": "cuda", "arch": self.arch}


SM120 = DeviceSpec()
DEFAULT_DEVICE = SM120
