"""Generic kernel builders shared by all ops in `cotile.ops`.

Every op module provides the same small protocol (see cotile/README.md):

    validate(shape, cfg)            raise ValueError if (shape, cfg) is unsupported
    tile_space(shape, cfg)          -> TileSpace(num_tiles, decode)
    io_params(shape, cfg)           -> [IOParam]  global tensors of the kernel
    make_io(args)                   -> namedtuple of those buffers (from a name->Buffer dict)
    scratch_spec(shape, cfg)        -> [ScratchBuf]  shared-memory scratch of one tile
    make_tile_body(shape, cfg)      -> @T.macro tile_body(tile_id, io, scratch)
    threads(cfg), pass_configs(cfg, build), smem_bytes(shape, cfg, build), numerics(cfg)

From that, `build_grid` (one CTA per tile) and `build_persistent` (num_ctas CTAs,
static grid-stride over tiles) are written once, here. Both launch exactly the same
`tile_body` macro; only the tile-id source differs. Shared-memory scratch is allocated
by the builder (`alloc_scratch`) and passed into the macro, so a CoKernel builder can
later allocate / alias / partition it for two roles.
"""

from __future__ import annotations

import hashlib
import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable

import tilelang
import tilelang.language as T
from tilelang.language.eager.builder import Builder
from tvm.tirx import Buffer

from .device import DEFAULT_DEVICE, DeviceSpec

# Pass configs every CoKernel-compatible build uses. On sm_120 TileLang auto-enables
# producer/consumer warp specialization (TMA + mbarrier, +128 producer threads), which
# only rewrites the first pipelined loop of a kernel and cannot coexist with a second
# role (research/notes/tilelang_capability_survey.md §1.5). Persistent builds and the
# `ws="off"` grid builds therefore disable it; loads then use cp.async + mma.sync.
PASS_CONFIGS_NO_WS = {"tl.disable_warp_specialized": True}
PASS_CONFIGS_AUTO_WS: dict = {}

BUILDS = ("grid", "persistent")


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def pmin(a, b):
    """min() that works for both Python ints and TIR PrimExprs."""
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return T.min(a, b)


@dataclass(frozen=True)
class IOParam:
    """One global-memory kernel parameter.

    role: "in"   input tensor
          "out"  output tensor (fully overwritten by one launch)
          "ws"   fp32 workspace for split partials (no initialisation needed)
          "ctr"  int32 arrival counters; must be zero when first used. Kernels leave
                 them zero again (the last-arriving tile resets its counter), so
                 zero-initialising once at allocation is enough for any number of
                 launches, as long as no launch is aborted mid-way.
    """

    name: str
    shape: tuple
    dtype: str
    role: str


@dataclass(frozen=True)
class ScratchBuf:
    """One shared-memory scratch buffer of a tile body."""

    name: str
    shape: tuple
    dtype: str

    @property
    def nbytes(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n * _DTYPE_BYTES[self.dtype]


_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "int32": 4, "uint8": 1}


def dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES[dtype]


@dataclass(frozen=True)
class TileSpace:
    """Flat bag of independent tiles. `decode(tile_id)` maps a tile id (Python int or
    TIR PrimExpr) to the op's tile coordinates (a namedtuple).

    `work(tile_id)` (optional, Python ints only): relative cost of one tile (the ops use
    issued MMA FLOPs). None means every tile does the same work. Ops with non-uniform tiles
    (causal prefill attention) expose it so a scheduler can order or balance tiles; the
    tile-id order of such an op is documented with its config (e.g. longest first)."""

    num_tiles: int
    decode: Callable[[Any], tuple]
    work: Callable[[int], float] | None = None

    def works(self) -> list:
        """work(t) for every tile (1 for uniform tile spaces)."""
        if self.work is None:
            return [1] * self.num_tiles
        return [self.work(t) for t in range(self.num_tiles)]


def alloc_scratch(spec: list[ScratchBuf]):
    """Allocate a tile body's shared-memory scratch inside the current T.Kernel and
    return it as a namedtuple (field order = spec order)."""
    Scratch = namedtuple("Scratch", [b.name for b in spec])
    return Scratch(*[T.alloc_shared(b.shape, b.dtype) for b in spec])


def make_io_tuple(params: list[IOParam], args: dict):
    IO = namedtuple("IO", [p.name for p in params])
    return IO(*[args[p.name] for p in params])


def make_prim_func(name: str, params: list[IOParam], body: Callable[[dict], None]):
    """Build a PrimFunc whose parameters are `params` (in order) and whose body is the
    @T.macro `body(args)`, where `args` maps parameter name -> Buffer.

    Uses the eager builder directly so kernel signatures can be assembled
    programmatically (e.g. the union of two roles' parameters in a CoKernel)."""
    builder = Builder()
    with builder.prim_func(name):
        args = {}
        for p in params:
            ann = T.Tensor(p.shape, p.dtype)
            if not isinstance(ann, Buffer) and callable(ann):
                ann = ann()
            args[p.name] = builder.arg(p.name, ann)
        body(args)
    return builder.get()


@dataclass
class KernelSpec:
    """A buildable kernel: PrimFunc + everything needed to compile, run and describe it."""

    op: Any  # op module
    shape: Any
    cfg: Any
    build: str  # "grid" | "persistent"
    name: str
    prim_func: Any
    pass_configs: dict
    params: list
    num_tiles: int
    grid: int  # CTAs launched (source-level)
    threads: int  # source-level threads per CTA
    smem_estimate: int  # per-CTA user smem (static + dynamic) predicted by op.smem_bytes
    kernel: Any = None  # tilelang JITKernel after compile
    compile_error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.op.NAME}:{self.op.cfg_tag(self.cfg)}:{self.build}:{self.grid}"


def _kernel_name(op, cfg, build: str) -> str:
    tag = op.cfg_tag(cfg)
    h = hashlib.sha1(repr((op.NAME, cfg, build)).encode()).hexdigest()[:6]
    safe = "".join(c if c.isalnum() else "_" for c in tag)
    return f"{op.NAME}_{build}_{safe}_{h}"


def build_grid(op, shape, cfg) -> KernelSpec:
    """One CTA per tile: CTA `bid` executes tile `bid`."""
    op.validate(shape, cfg)
    ts = op.tile_space(shape, cfg)
    body = op.make_tile_body(shape, cfg)
    params = op.io_params(shape, cfg)
    sspec = op.scratch_spec(shape, cfg)
    threads = op.threads(cfg)
    num_tiles = ts.num_tiles

    @T.macro
    def kernel_body(args):
        io = op.make_io(args)
        with T.Kernel(num_tiles, threads=threads) as bid:
            scratch = alloc_scratch(sspec)
            body(bid, io, scratch)

    name = _kernel_name(op, cfg, "grid")
    pf = make_prim_func(name, params, kernel_body)
    return KernelSpec(
        op=op,
        shape=shape,
        cfg=cfg,
        build="grid",
        name=name,
        prim_func=pf,
        pass_configs=dict(op.pass_configs(cfg, "grid")),
        params=params,
        num_tiles=num_tiles,
        grid=num_tiles,
        threads=threads,
        smem_estimate=op.smem_bytes(shape, cfg, "grid"),
    )


def build_persistent(op, shape, cfg, num_ctas: int) -> KernelSpec:
    """`num_ctas` CTAs; CTA `bid` executes tiles bid, bid + num_ctas, ... (static
    grid-stride). Tiles never wait on each other (split combines use the
    last-arriving-tile pattern), so any num_ctas >= 1 is deadlock-free."""
    op.validate(shape, cfg)
    if num_ctas < 1:
        raise ValueError("num_ctas must be >= 1")
    ts = op.tile_space(shape, cfg)
    body = op.make_tile_body(shape, cfg)
    params = op.io_params(shape, cfg)
    sspec = op.scratch_spec(shape, cfg)
    threads = op.threads(cfg)
    num_tiles = ts.num_tiles
    waves = ceildiv(num_tiles, num_ctas)

    @T.macro
    def kernel_body(args):
        io = op.make_io(args)
        with T.Kernel(num_ctas, threads=threads) as bid:
            scratch = alloc_scratch(sspec)
            for it in T.serial(waves):
                tile_id = it * num_ctas + bid
                if tile_id < num_tiles:
                    body(tile_id, io, scratch)

    name = _kernel_name(op, cfg, "persistent")
    pf = make_prim_func(name, params, kernel_body)
    return KernelSpec(
        op=op,
        shape=shape,
        cfg=cfg,
        build="persistent",
        name=name,
        prim_func=pf,
        pass_configs=dict(op.pass_configs(cfg, "persistent")),
        params=params,
        num_tiles=num_tiles,
        grid=num_ctas,
        threads=threads,
        smem_estimate=op.smem_bytes(shape, cfg, "persistent"),
    )


def compile_specs(
    specs: list[KernelSpec],
    device: DeviceSpec = DEFAULT_DEVICE,
    num_workers: int | None = 32,
) -> dict:
    """Compile specs with tilelang.par_compile (one call per distinct pass-config set).

    Failures do not abort the batch: the failing spec gets `compile_error` (the
    exception text, obtained by recompiling it alone) and `kernel=None`.
    Returns {"wall_s": ..., "n_ok": ..., "n_fail": ...}."""
    t0 = time.time()
    groups: dict[str, list[KernelSpec]] = {}
    for s in specs:  # pass-config values may be unhashable (e.g. flag lists)
        groups.setdefault(repr(sorted(s.pass_configs.items())), []).append(s)
    for group in groups.values():
        pc_items = sorted(group[0].pass_configs.items())
        kernels = tilelang.par_compile(
            [s.prim_func for s in group],
            target=device.target,
            pass_configs=dict(pc_items),
            num_workers=min(num_workers or len(group), len(group)),
            ignore_error=True,
        )
        for s, k in zip(group, kernels):
            s.kernel = k
            if k is None:
                try:
                    s.kernel = tilelang.compile(s.prim_func, target=device.target, pass_configs=dict(pc_items))
                except Exception as e:  # noqa: BLE001 - recorded, reported by caller
                    s.compile_error = f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:400]}"
    n_ok = sum(1 for s in specs if s.kernel is not None)
    return {"wall_s": time.time() - t0, "n_ok": n_ok, "n_fail": len(specs) - n_ok}


class Runner:
    """Calls a compiled KernelSpec with torch tensors.

    Workspaces ("ws") and counters ("ctr") are allocated on first use and reused by
    every later call, which is what the repeated-launch tests exercise. Pass
    `state=other_runner.state` to share them between kernels of the same op.
    """

    def __init__(self, spec: KernelSpec, state: dict | None = None):
        if spec.kernel is None:
            raise RuntimeError(f"{spec.name} is not compiled ({spec.compile_error})")
        self.spec = spec
        self.state = state if state is not None else {}

    def _persistent_buf(self, p: IOParam, device):
        import torch

        buf = self.state.get(p.name)
        if buf is None or tuple(buf.shape) != tuple(p.shape):
            tdt = getattr(torch, p.dtype)
            buf = torch.zeros(p.shape, dtype=tdt, device=device) if p.role == "ctr" else torch.empty(p.shape, dtype=tdt, device=device)
            self.state[p.name] = buf
        return buf

    def make_args(self, inputs: dict, outputs: dict | None = None) -> tuple[list, dict]:
        import torch

        device = next(iter(inputs.values())).device
        outputs = dict(outputs or {})
        args = []
        for p in self.spec.params:
            if p.role == "in":
                t = inputs[p.name]
                if tuple(t.shape) != tuple(p.shape) or str(t.dtype) != f"torch.{p.dtype}":
                    raise ValueError(f"{p.name}: expected {p.shape} {p.dtype}, got {tuple(t.shape)} {t.dtype}")
                args.append(t)
            elif p.role == "out":
                if p.name not in outputs:
                    outputs[p.name] = torch.empty(p.shape, dtype=getattr(torch, p.dtype), device=device)
                args.append(outputs[p.name])
            else:
                args.append(self._persistent_buf(p, device))
        return args, outputs

    def __call__(self, inputs: dict, outputs: dict | None = None) -> dict:
        args, outputs = self.make_args(inputs, outputs)
        self.spec.kernel(*args)
        return outputs

    def counters(self) -> dict:
        return {p.name: self.state[p.name] for p in self.spec.params if p.role == "ctr" and p.name in self.state}
