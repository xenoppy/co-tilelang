"""Green-context SM partitioning via the CUDA driver API (cuda.bindings).

``split_sms(n)`` splits the device's SM resource into a group of ~n SMs and the
complementary remainder, creates one green context per part, and exposes a
non-blocking stream in each as ``torch.cuda.ExternalStream`` so torch ops and
``CudaKernel`` launches can target it.

Granularity (measured on RTX PRO 6000 / sm_120, see research/results/2026-09-22_smid_probe):
default flags round the request up to a multiple of 8 SMs; with
``ignore_coscheduling=True`` (CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING) the
split is finer. Always read ``.n_sms`` for the count actually obtained.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from cuda.bindings import driver as cu

from .cudrv import check, device_index, ensure_init

_FLAG_IGNORE_COSCHED = int(cu.CUdevSmResourceSplit_flags.CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING)


def device_sm_resource(device=None) -> cu.CUdevResource:
    dev = ensure_init(device)
    return check(cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))


def query_split(n: int, *, device=None, ignore_coscheduling: bool = False) -> tuple[int, int]:
    """(SMs in the n-group, SMs in the remainder) without creating contexts."""
    res = device_sm_resource(device)
    flags = _FLAG_IGNORE_COSCHED if ignore_coscheduling else 0
    result, nb, rem = check(cu.cuDevSmResourceSplitByCount(1, res, flags, int(n)))
    if nb < 1:
        raise RuntimeError(f"split of {n} SMs produced no group")
    return int(result[0].sm.smCount), int(rem.sm.smCount)


@dataclass
class GreenContext:
    """One green context with one non-blocking stream."""
    n_sms: int
    stream: torch.cuda.ExternalStream
    _gctx: object = field(repr=False, default=None)
    _custream: object = field(repr=False, default=None)
    _closed: bool = field(repr=False, default=False)

    @classmethod
    def create(cls, resource: cu.CUdevResource, device=None, priority: int = 0) -> "GreenContext":
        dev = ensure_init(device)
        desc = check(cu.cuDevResourceGenerateDesc([resource], 1))
        gctx = check(cu.cuGreenCtxCreate(desc, dev,
                                         cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        custream = check(cu.cuGreenCtxStreamCreate(
            gctx, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, int(priority)))
        ts = torch.cuda.ExternalStream(int(custream), device=torch.device("cuda", device_index(device)))
        return cls(n_sms=int(resource.sm.smCount), stream=ts, _gctx=gctx, _custream=custream)

    def sm_count_from_driver(self) -> int:
        r = check(cu.cuGreenCtxGetDevResource(self._gctx, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        return int(r.sm.smCount)

    def close(self) -> None:
        if self._closed:
            return
        self.stream.synchronize()
        check(cu.cuStreamDestroy(self._custream))
        check(cu.cuGreenCtxDestroy(self._gctx))
        self._closed = True


@dataclass
class SmPartition:
    requested: int
    ignore_coscheduling: bool
    part: GreenContext                 # the ~n-SM group
    rest: GreenContext | None          # complementary SMs (None if empty)

    @property
    def n_sms(self) -> int:
        return self.part.n_sms

    @property
    def n_rest(self) -> int:
        return self.rest.n_sms if self.rest else 0

    @property
    def stream(self) -> torch.cuda.ExternalStream:
        return self.part.stream

    @property
    def rest_stream(self) -> torch.cuda.ExternalStream | None:
        return self.rest.stream if self.rest else None

    def close(self) -> None:
        self.part.close()
        if self.rest:
            self.rest.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def split_sms(n: int, *, device=None, ignore_coscheduling: bool = False,
              priority: int = 0) -> SmPartition:
    """Create streams restricted to ~n SMs and to the complementary SMs."""
    res = device_sm_resource(device)
    flags = _FLAG_IGNORE_COSCHED if ignore_coscheduling else 0
    result, nb, rem = check(cu.cuDevSmResourceSplitByCount(1, res, flags, int(n)))
    if nb < 1:
        raise RuntimeError(f"split of {n} SMs produced no group")
    part = GreenContext.create(result[0], device, priority)
    rest = GreenContext.create(rem, device, priority) if rem.sm.smCount > 0 else None
    return SmPartition(requested=int(n), ignore_coscheduling=ignore_coscheduling, part=part, rest=rest)
