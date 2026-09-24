"""Resource signatures read from the *compiled* kernel (not from source parameters).

TileLang has no CUDA resource-query API (JITKernel.n_regs is HIP-only), and on sm_120
it may change the thread count (auto warp specialization adds 128 producer threads).
So every number here comes from compilation products:

* cubin: taken from the kernel's CUDA runtime module (`inspect_source("cubin")`
  returns the raw cubin bytes; the FFI layer fails to UTF-8-decode them and the
  UnicodeDecodeError carries the exact bytes).
  - `cuobjdump --dump-resource-usage`: REG, STACK, LOCAL, SHARED (static smem; on sm_90+
    this *includes* the 1 KB per-CTA reserved smem, see device.py)
  - `cuobjdump -elf` (.nv.info): EIATTR_MAX_THREADS (= __launch_bounds__ = compiled
    threads/CTA), EIATTR_NUM_BARRIERS (named barriers; limits CTAs/SM on cc 12),
    EIATTR_KPARAM_INFO (kernel parameter count)
* host module source: the launch arguments TileLang's host stub passes to the CUDA
  kernel (grid/block extents and the dynamic smem size, as constants).

Occupancy follows cuda_occupancy.h (CUDA 12.9) for compute capability 12.0: warp slots,
CTA slots, per-sub-partition register allocation, smem with reserved KB and 128 B
granularity, and the named-barrier limit. `driver_check` cross-checks registers, static
smem and occupancy against the CUDA driver on a live GPU.

Results are cached in $COTILE_CACHE_DIR/signatures.json (default ~/.cache/cotile),
keyed by sha256(cubin) (parsed cuobjdump results only).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading

from .device import DEFAULT_DEVICE, DeviceSpec

_CACHE_LOCK = threading.Lock()


def _cuda_bin(tool: str) -> str:
    home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    p = os.path.join(home, "bin", tool)
    return p if os.path.exists(p) else tool


def _cache_path() -> str:
    d = os.environ.get("COTILE_CACHE_DIR", os.path.expanduser("~/.cache/cotile"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "signatures.json")


def _load_cache() -> dict:
    try:
        with open(_cache_path()) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    p = _cache_path()
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, p)


# ----------------------------------------------------------------------------------
# compiled artifacts
# ----------------------------------------------------------------------------------


def _iter_modules(mod):
    yield mod
    for sub in mod.imports:
        yield from _iter_modules(sub)


def cubin_bytes(jit_kernel) -> bytes:
    """Raw cubin of a TileLang JITKernel (tvm_ffi execution backend)."""
    ex = jit_kernel.adapter.executable
    if not hasattr(ex, "imports"):  # tvm.runtime.Executable wrapper on a cache miss
        ex = ex.mod
    for m in _iter_modules(ex):
        if m.kind != "cuda":
            continue
        for fmt in ("cubin", "fatbin"):
            try:
                s = m.inspect_source(fmt)
            except UnicodeDecodeError as e:  # binary payload: exact bytes are in e.object
                data = bytes(e.object)
            else:
                data = s.encode("utf-8") if s else b""
            if data:
                if fmt == "cubin" and not data.startswith(b"\x7fELF"):
                    raise RuntimeError("CUDA module payload is not an ELF cubin")
                return data
    raise RuntimeError("no CUDA device module with a cubin/fatbin payload found")


def _run_cuobjdump(args: list[str], cubin: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        r = subprocess.run([_cuda_bin("cuobjdump"), *args, path], capture_output=True, text=True, check=True)
        return r.stdout
    finally:
        os.unlink(path)


def sass(cubin: bytes) -> str:
    """Disassembly (cuobjdump -sass) of a cubin."""
    return _run_cuobjdump(["-sass"], cubin)


def invalid_memory_descriptors(sass_text: str) -> list[str]:
    """Global-memory instructions whose 64-bit memory descriptor (`desc[URn]`) is an odd
    uniform register or one that no instruction of the function writes.

    Guards against a ptxas 12.9 (sm_120/120a) miscompile of cp.async with an L2 cache
    policy: when ptxas folds a uniform shared-memory base into the LDGSTS address
    ([Rx+URy]) it emits `desc[UR1]` with UR0/UR1 never written, and the kernel faults with
    "illegal instruction" (see src/tl_templates/cuda/copy.h, cp_async_gs_l2hint). A
    heuristic over the text disassembly: uniform registers written by any instruction
    (pairs for .64/WIDE writes) count as defined."""
    written, bad = set(), []
    for line in sass_text.splitlines():
        m = re.search(r"/\*[0-9a-f]{4}\*/\s+(.*?);", line)
        if not m:
            continue
        ins = re.sub(r"^@!?U?P\w+\s+", "", m.group(1))
        parts = ins.split(None, 1)
        if len(parts) == 2:
            dst = parts[1].split(",")[0].strip()
            if re.fullmatch(r"UR\d+", dst):
                n = int(dst[2:])
                written.add(n)
                if parts[0].endswith(".64") or "WIDE" in parts[0]:
                    written.add(n + 1)
        for d in re.findall(r"desc\[UR(\d+)\]", ins):
            n = int(d)
            if n % 2 or (n not in written and n + 1 not in written):
                bad.append(re.sub(r"\s+", " ", ins))
    return bad


def parse_resource_usage(text: str) -> dict:
    """cuobjdump --dump-resource-usage -> {function: {REG, STACK, SHARED, LOCAL, ...}}"""
    out, fn = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            fn = m.group(1)
            continue
        if fn and "REG:" in line:
            out[fn] = {k: int(v) for k, v in re.findall(r"(\w+(?:\[\d+\])?):(\d+)", line)}
            fn = None
    return out


def parse_nv_info(text: str) -> dict:
    """cuobjdump -elf -> {function: {"MAX_THREADS": (x,y,z), "NUM_BARRIERS": n, "NUM_PARAMS": n}}"""
    out: dict[str, dict] = {}
    fn = None
    attr = None
    for line in text.splitlines():
        m = re.match(r"^\.nv\.info\.(\S+)$", line.strip())
        if m:
            fn = m.group(1)
            out.setdefault(fn, {"NUM_BARRIERS": 0, "NUM_PARAMS": 0})
            continue
        if fn is None:
            continue
        if line.startswith(".") and not line.startswith(".nv.info."):
            fn = None
            continue
        m = re.match(r"\s*Attribute:\s*(\S+)", line)
        if m:
            attr = m.group(1)
            if attr == "EIATTR_KPARAM_INFO":
                out[fn]["NUM_PARAMS"] += 1
            continue
        m = re.match(r"\s*Value:\s*(.*)$", line)
        if m and attr:
            vals = [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
            if attr == "EIATTR_NUM_BARRIERS" and vals:
                out[fn]["NUM_BARRIERS"] = vals[0]
            elif attr == "EIATTR_MAX_THREADS" and vals:
                out[fn]["MAX_THREADS"] = tuple(vals[:3])
            elif attr == "EIATTR_MAXREG_COUNT" and vals:
                out[fn]["MAXREG_COUNT"] = vals[0]
            attr = None
    return out


_SLOT = re.compile(r"\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_int64\) = \(\(int64_t\)(-?\d+)\);")


def host_launch_values(host_source: str, func_name: str, num_kernel_params: int) -> list[int]:
    """Launch-parameter values the host stub passes to `func_name` (after the kernel
    arguments): the thread-axis extents in TVM launch-tag order, then the dynamic smem
    size if the kernel uses dynamic shared memory."""
    call = re.search(
        r"TVMFFIFunctionCall\(" + re.escape(func_name) + r"_packed, \(TVMFFIAny\*\) stack_ffi_any, (\d+),",
        host_source,
    )
    if call is None:
        raise RuntimeError(f"launch of {func_name} not found in host source")
    nargs = int(call.group(1))
    start = host_source.rfind("TVMFFIFunctionCall(", 0, call.start())
    segment = host_source[start + 1 if start >= 0 else 0 : call.start()]
    slots: dict[int, int] = {}
    for m in _SLOT.finditer(segment):
        slots[int(m.group(1))] = int(m.group(2))
    vals = []
    for i in range(num_kernel_params, nargs):
        if i not in slots:
            raise RuntimeError(f"launch value {i} of {func_name} is not a constant")
        vals.append(slots[i])
    return vals


# ----------------------------------------------------------------------------------
# occupancy (cuda_occupancy.h, cc 12.0)
# ----------------------------------------------------------------------------------


def _round_up(x: int, g: int) -> int:
    return -(-x // g) * g


def occupancy(regs: int, threads: int, smem_user: int, num_barriers: int, dev: DeviceSpec = DEFAULT_DEVICE) -> dict:
    """Theoretical CTAs/SM and the limit imposed by each resource.

    smem_user = static user smem + dynamic smem (bytes, excluding the reserved KB)."""
    warps = -(-threads // dev.warp_size)
    lim_warps = dev.max_warps_per_sm // warps if threads <= dev.max_threads_per_cta else 0
    lim_ctas = dev.max_ctas_per_sm
    regs_per_warp = _round_up(regs * dev.warp_size, dev.reg_alloc_granularity)
    if regs > dev.max_regs_per_thread or regs_per_warp * _round_up(warps, dev.reg_sub_partitions) > dev.regs_per_cta:
        lim_regs = 0
    elif regs_per_warp == 0:
        lim_regs = lim_ctas
    else:
        warps_per_sub = (dev.regs_per_sm // dev.reg_sub_partitions) // regs_per_warp
        lim_regs = warps_per_sub * dev.reg_sub_partitions // warps
    smem_alloc = _round_up(smem_user + dev.smem_reserved_per_cta, dev.smem_alloc_granularity)
    if smem_user > dev.smem_per_cta_optin:
        lim_smem = 0
    else:
        lim_smem = dev.smem_per_sm // smem_alloc
    lim_bars = dev.barriers_per_sm // num_barriers if num_barriers else lim_ctas
    limits = {"warps": lim_warps, "ctas": lim_ctas, "regs": lim_regs, "smem": lim_smem, "barriers": lim_bars}
    ctas = min(limits.values())
    limiters = sorted(k for k, v in limits.items() if v == ctas)
    return {
        "ctas_per_sm": ctas,
        "limit_by": "+".join(limiters),
        "lim_warps": lim_warps,
        "lim_regs": lim_regs,
        "lim_smem": lim_smem,
        "lim_barriers": lim_bars,
        "smem_alloc_per_cta": smem_alloc,
        "warps_per_sm": ctas * warps,
        "occupancy": ctas * warps / dev.max_warps_per_sm,
    }


# ----------------------------------------------------------------------------------
# signature
# ----------------------------------------------------------------------------------


def signature(spec, dev: DeviceSpec = DEFAULT_DEVICE, use_cache: bool = True) -> dict:
    """Resource signature of a compiled KernelSpec (see module docstring)."""
    k = spec.kernel
    if k is None:
        raise RuntimeError(f"{spec.name} is not compiled")
    cubin = cubin_bytes(k)
    host = k.get_host_source()
    dev_src = k.get_kernel_source()
    ci = cubin_info(cubin)
    ru, info = ci["ru"], ci["nv"]
    fn = f"{spec.name}_kernel"
    if fn not in ru:
        if len(ru) != 1:
            raise RuntimeError(f"{fn} not in cubin functions {list(ru)}")
        fn = next(iter(ru))
    r, nv = ru[fn], info.get(fn, {})
    launch = host_launch_values(host, fn, nv.get("NUM_PARAMS", 0))
    uses_dyn = "extern __shared__" in dev_src
    dyn = launch[-1] if uses_dyn else 0
    axes = launch[:-1] if uses_dyn else launch
    mt = nv.get("MAX_THREADS")
    threads = mt[0] * mt[1] * mt[2] if mt else None
    prod = 1
    for a in axes:
        prod *= a
    if threads is None or prod % threads:
        raise RuntimeError(f"cannot split launch extents {axes} of {fn} into grid x {threads} threads")
    grid = prod // threads
    static_total = r["SHARED"]
    static_user = static_total - dev.smem_reserved_per_cta if static_total >= dev.smem_reserved_per_cta else static_total
    smem_user = static_user + dyn
    occ = occupancy(r["REG"], threads, smem_user, nv.get("NUM_BARRIERS", 0), dev)
    num_cls, num_key = spec.op.numerics(spec.cfg)
    sig = {
        "op": spec.op.NAME,
        "config": spec.op.cfg_tag(spec.cfg),
        "build": spec.build,
        "kernel": fn,
        "numerics": num_cls,
        "numerics_key": repr(num_key),
        "num_tiles": spec.num_tiles,
        "grid": grid,
        "grid_src": spec.grid,
        "threads": threads,
        "threads_src": spec.threads,
        "threads_changed": threads != spec.threads,
        "regs": r["REG"],
        "local_bytes": r.get("LOCAL", 0),
        "stack_bytes": r.get("STACK", 0),
        "smem_static": static_user,
        "smem_dynamic": dyn,
        "smem_total": smem_user,
        "smem_estimate": spec.smem_estimate,
        "num_barriers": nv.get("NUM_BARRIERS", 0),
        **occ,
    }
    return sig


_MEM_CACHE: dict | None = None


def cubin_info(cubin: bytes) -> dict:
    """Parsed cuobjdump data for a cubin: {"ru": {...}, "nv": {...}}. Cached on disk
    (keyed by sha256 of the cubin) and in memory."""
    global _MEM_CACHE
    key = hashlib.sha256(cubin).hexdigest()
    with _CACHE_LOCK:
        if _MEM_CACHE is None:
            _MEM_CACHE = _load_cache()
        hit = _MEM_CACHE.get(key)
    if hit is not None:
        return hit
    info = {
        "ru": parse_resource_usage(_run_cuobjdump(["--dump-resource-usage"], cubin)),
        "nv": parse_nv_info(_run_cuobjdump(["-elf"], cubin)),
    }
    with _CACHE_LOCK:
        disk = _load_cache()
        disk.update(_MEM_CACHE)
        disk[key] = info
        _MEM_CACHE = disk
        _save_cache(disk)
    return info


# ----------------------------------------------------------------------------------
# optional: cross-check against the CUDA driver (needs a GPU and a current context)
# ----------------------------------------------------------------------------------

_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
_CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
_CU_FUNC_ATTRIBUTE_NUM_REGS = 4
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


def driver_check(spec, threads: int, dyn_smem: int) -> dict:
    """Load the cubin through the driver API (ctypes) in the current CUDA context and
    return cuFuncGetAttribute values and cuOccupancyMaxActiveBlocksPerMultiprocessor.
    Call after torch has initialised CUDA on this thread."""
    cuda = ctypes.CDLL("libcuda.so.1")

    def ck(res, what):
        if res != 0:
            raise RuntimeError(f"{what} failed with CUresult {res}")

    cubin = cubin_bytes(spec.kernel)
    ru = cubin_info(cubin)["ru"]
    fn = f"{spec.name}_kernel" if f"{spec.name}_kernel" in ru else next(iter(ru))
    mod = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(cubin, len(cubin))
    ck(cuda.cuModuleLoadData(ctypes.byref(mod), buf), "cuModuleLoadData")
    try:
        func = ctypes.c_void_p()
        ck(cuda.cuModuleGetFunction(ctypes.byref(func), mod, fn.encode()), "cuModuleGetFunction")
        out = {}
        for name, attr in (
            ("regs", _CU_FUNC_ATTRIBUTE_NUM_REGS),
            ("smem_static", _CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
            ("local_bytes", _CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
            ("max_threads", _CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        ):
            v = ctypes.c_int()
            ck(cuda.cuFuncGetAttribute(ctypes.byref(v), attr, func), "cuFuncGetAttribute")
            out[name] = v.value
        ck(
            cuda.cuFuncSetAttribute(func, _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ctypes.c_int(dyn_smem)),
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
        )
        n = ctypes.c_int()
        ck(
            cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(n), func, ctypes.c_int(threads), ctypes.c_size_t(dyn_smem)),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )
        out["ctas_per_sm"] = n.value
        return out
    finally:
        cuda.cuModuleUnload(mod)
