"""Thin helpers over ``cuda.bindings``: error checking, NVRTC compilation and
context-independent kernel loading.

Kernels are loaded with ``cuLibraryLoadData`` / ``cuLibraryGetKernel`` (a
``CUkernel``), not ``cuModuleLoadData`` (a context-bound ``CUfunction``).  A
``CUkernel`` is resolved against the context of the launch stream, so the same
kernel object can be launched on ordinary (primary-context) torch streams and on
green-context streams.
"""
from __future__ import annotations

import ctypes
import hashlib
from typing import Iterable, Sequence

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc


class CudaError(RuntimeError):
    pass


def check(ret):
    """Unpack a cuda.bindings return tuple ``(err, *values)``; raise on error.

    Returns None / the single value / a tuple of values.
    """
    if isinstance(ret, tuple):
        err, rest = ret[0], ret[1:]
    else:
        err, rest = ret, ()
    if isinstance(err, cu.CUresult):
        if err != cu.CUresult.CUDA_SUCCESS:
            _, name = cu.cuGetErrorName(err)
            _, desc = cu.cuGetErrorString(err)
            name = name.decode() if isinstance(name, bytes) else str(err)
            desc = desc.decode() if isinstance(desc, bytes) else ""
            raise CudaError(f"{name}: {desc}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, desc = nvrtc.nvrtcGetErrorString(err)
            raise CudaError(f"NVRTC error: {desc.decode() if isinstance(desc, bytes) else err}")
    if not rest:
        return None
    if len(rest) == 1:
        return rest[0]
    return rest


def device_index(device=None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()


def ensure_init(device=None) -> cu.CUdevice:
    """Make sure torch has created/bound the primary context on this thread and
    the driver API is initialised. Returns the CUdevice."""
    idx = device_index(device)
    torch.cuda.init()
    torch.cuda.set_device(idx)
    torch.empty(1, device=f"cuda:{idx}")  # forces primary context creation + current
    check(cu.cuInit(0))
    return check(cu.cuDeviceGet(idx))


def device_attr(attr: cu.CUdevice_attribute, device=None) -> int:
    dev = ensure_init(device)
    return int(check(cu.cuDeviceGetAttribute(attr, dev)))


def arch_string(device=None) -> str:
    major, minor = torch.cuda.get_device_capability(device_index(device))
    return f"sm_{major}{minor}"


_CUBIN_CACHE: dict[str, bytes] = {}


def compile_cubin(src: str, *, arch: str | None = None, options: Iterable[str] = (),
                  name: str = "kernel.cu") -> bytes:
    """Compile CUDA C++ source to a cubin with NVRTC (cached in-process)."""
    arch = arch or arch_string()
    opts = [f"--gpu-architecture={arch}", "-std=c++17", *options]
    key = hashlib.sha1((src + "\0" + "\0".join(opts)).encode()).hexdigest()
    if key in _CUBIN_CACHE:
        return _CUBIN_CACHE[key]
    prog = check(nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], []))
    try:
        bopts = [o.encode() for o in opts]
        ret = nvrtc.nvrtcCompileProgram(prog, len(bopts), bopts)
        log_size = check(nvrtc.nvrtcGetProgramLogSize(prog))
        log = b" " * log_size
        check(nvrtc.nvrtcGetProgramLog(prog, log))
        if ret[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise CudaError(f"NVRTC compilation failed for {name}:\n{log.decode(errors='replace')}")
        size = check(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    _CUBIN_CACHE[key] = cubin
    return cubin


_ARG_TYPES = {
    "p": ctypes.c_void_p,     # pointer (torch.Tensor or int address)
    "i": ctypes.c_int32,
    "I": ctypes.c_uint32,
    "q": ctypes.c_int64,
    "Q": ctypes.c_uint64,
    "f": ctypes.c_float,
    "d": ctypes.c_double,
}


def _stream_handle(stream) -> int:
    if stream is None:
        return torch.cuda.current_stream().cuda_stream
    if isinstance(stream, torch.cuda.Stream):
        return stream.cuda_stream
    if isinstance(stream, cu.CUstream):
        return int(stream)
    return int(stream)


def _dim3(x) -> tuple[int, int, int]:
    if isinstance(x, int):
        return (x, 1, 1)
    x = tuple(int(v) for v in x)
    return x + (1,) * (3 - len(x))


class CudaKernel:
    """A ``__global__`` function compiled with NVRTC and loaded context-independently.

    ``signature`` is one character per kernel parameter:
    p=pointer (torch.Tensor or int), i=int32, I=uint32, q=int64, Q=uint64, f=float, d=double.
    Launch: ``k(grid, block, *args, smem=0, stream=None)``; stream defaults to
    torch's current stream, so kernels are captured by CUDA graphs like torch ops.
    """

    def __init__(self, src: str, name: str, signature: str, *, options: Sequence[str] = (),
                 device=None):
        self.device = device_index(device)
        self._cudev = ensure_init(self.device)
        self.name = name
        self.signature = signature
        for c in signature:
            if c not in _ARG_TYPES:
                raise ValueError(f"bad signature char {c!r}")
        self._types = tuple(_ARG_TYPES[c] for c in signature)
        self.cubin = compile_cubin(src, arch=arch_string(self.device), options=options,
                                   name=f"{name}.cu")
        self._lib = check(cu.cuLibraryLoadData(self.cubin, [], [], 0, [], [], 0))
        self.kernel = check(cu.cuLibraryGetKernel(self._lib, name.encode()))
        self._max_dyn_smem = 48 * 1024

    # -- attributes ---------------------------------------------------------
    def _attr(self, attr: cu.CUfunction_attribute) -> int:
        return int(check(cu.cuKernelGetAttribute(attr, self.kernel, self._cudev)))

    @property
    def num_regs(self) -> int:
        return self._attr(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS)

    @property
    def static_smem(self) -> int:
        return self._attr(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)

    def set_max_dynamic_smem(self, nbytes: int) -> None:
        check(cu.cuKernelSetAttribute(
            cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            int(nbytes), self.kernel, self._cudev))
        self._max_dyn_smem = int(nbytes)

    def occupancy(self, block: int, smem: int = 0) -> int:
        """Max resident CTAs per SM for this block size and dynamic smem."""
        if smem > self._max_dyn_smem:
            self.set_max_dynamic_smem(smem)
        func = check(cu.cuKernelGetFunction(self.kernel))
        return int(check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(func, int(block), int(smem))))

    # -- launch -------------------------------------------------------------
    def __call__(self, grid, block, *args, smem: int = 0, stream=None) -> None:
        if len(args) != len(self._types):
            raise TypeError(f"{self.name} expects {len(self._types)} args, got {len(args)}")
        vals = []
        for a, c in zip(args, self.signature):
            if c == "p":
                vals.append(a.data_ptr() if isinstance(a, torch.Tensor) else int(a))
            elif c in "fd":
                vals.append(float(a))
            else:
                vals.append(int(a))
        if smem > self._max_dyn_smem:
            self.set_max_dynamic_smem(smem)
        g, b = _dim3(grid), _dim3(block)
        check(cu.cuLaunchKernel(self.kernel, *g, *b, int(smem), _stream_handle(stream),
                                (tuple(vals), self._types), 0))

    def __del__(self):
        try:
            if getattr(self, "_lib", None) is not None:
                cu.cuLibraryUnload(self._lib)
        except Exception:
            pass
