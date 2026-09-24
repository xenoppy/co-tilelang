"""Two-role persistent CoKernel builder (v0, plan §3.3 / proposal §5.3).

`build_cokernel(opA, shapeA, cfgA, opB, shapeB, cfgB, orch)` fuses the tile bodies of
two op-library ops (cotile/ops/*, see cotile/kernel.py for the protocol) into ONE
persistent TileLang kernel. Every CTA runs a dispatch loop; in each iteration thread 0
acquires the next (role, tile) pair, broadcasts it through a double-buffered static
shared-memory slot, and the whole CTA executes that tile with the role's `tile_body`
macro. Nothing about the ops is special-cased: roles are described only by the op
protocol (`tile_space`, `io_params`, `scratch_spec`, `make_tile_body`, `threads`).

Orchestration (`Orch`)
----------------------
* binding="sm"  : role = co_sm_role[%smid] (host int32[num_sms] table, a runtime knob:
                  changing the SM split does not recompile).
* binding="cta" : POD-style. r = atomic_add(co_state[SMCTR + %smid], 1) is the CTA's
                  arrival rank on its SM; role = A if r mod (kA+kB) < kA else B, with
                  kA = co_knobs[0], kB = co_knobs[1] runtime knobs.
* binding="tile": POD's scheduling policy in a persistent kernel. The role is drawn per
                  grab, not per CTA: before every grab thread 0 takes the SM's next
                  ticket j = atomic_add(co_state[SMCTR + %smid], 1) and picks role A iff
                  j mod (kA+kB) < kA (FlashInfer pod.cuh draws one ticket per launched
                  CTA, each CTA running one tile). If the drawn role's queue is exhausted
                  it falls back to the other role (as POD does), so takeover is implied;
                  the CTA exits when both queues are exhausted. Needs schedule="dynamic".
                  ctas_A in co_out counts every CTA (there is no per-CTA role).
* schedule="dynamic": per-role global queue head co_state[Q + r]; thread 0 grabs
                  `chunk` tiles with atomic_add(return_prev=True) and hands them out
                  one per dispatch iteration. `chunk` is an int or a per-role pair
                  (chunk_A, chunk_B): big tiles want 1, tiny tiles want more.
* schedule="static" : the CTA takes a rank within its role from a per-role ticket
                  (one atomic per CTA) and runs tiles rank, rank + n_r, ... where
                  n_r = co_knobs[2 + r] is the host-computed number of CTAs of role r
                  (`expected_role_ctas`). The kernel reports the actual per-role CTA
                  count; `CoRunner` raises if the placement assumption was violated.
* takeover=True : after a CTA's own role is exhausted it continues with the other
                  role (dynamic: grabs from the other role's queue; static: steals
                  from the back of the other role's tile range, with per-tile claims
                  -- see below).
* threads       : common CTA size (>= each role's cfg threads). A role with fewer
                  threads runs under `if tx < threads_role` (Rammer-style idle
                  threads); TileLang turns its barriers into `bar.sync id, n`.
* smem="alias"  : every role's tile execution is wrapped in a `tl.shared_lifetime_scope`
                  AttrStmt (co-tilelang core addition, see src/op/builtin.h): no
                  shared-memory value crosses it, so TileLang's merge planner places
                  both roles' scratch in the same bytes of the dynamic-smem arena
                  (per-CTA smem ~ max, not sum). smem="sum" omits the scopes
                  (comparison baseline: TileLang's default plan keeps everything
                  touched inside the dispatch loop live for the whole loop).
* timing=True   : a CTA barrier after every tile, then thread 0 reads %globaltimer
                  (per-role end = max over CTAs of their last tile completion).
* debug=True    : per-tile execution counter (co_dbg), executing SM (co_dbg_sm) and
                  start/end %globaltimer (co_dbg_time; end is after the per-tile
                  barrier when timing=True, else thread 0's own completion).

Static + takeover uses epoch-tagged per-tile claims: co_claim[t] holds the launch
epoch of the last claim; claiming = atomic_max(co_claim[t], epoch) returning a smaller
value. Owners claim each of their tiles before running it, thieves claim from the end
of the range and stop at the first failed claim; since owners always visit all their
tiles, every tile runs exactly once regardless of how far thieves get.

Counters, timestamps, reset
---------------------------
All counters live in `co_state` (int32) and `co_tacc` (int64), role "ctr": zero them
once at allocation. Every CTA increments co_state[EXIT] (acq_rel) as its last action;
the CTA that arrives last copies the results into `co_out` (int64[16], role "out") and
resets every counter to zero (including the per-SM arrival counters) and bumps the
epoch, so repeated launches need no host re-zeroing. Only an aborted launch requires
`CoRunner.reset_state()`.

co_out = [t_start, t_end_A, t_end_B, t_exit, done_A, done_B, ctas_A, ctas_B,
          tickets_A, tickets_B, epoch, steal_A, steal_B, q_A, q_B, smid_errors]
t_start is the min over CTAs of their start %globaltimer (kept as an atomicMax of the
bitwise complement so that the zero-initialised accumulator needs no host init);
t_end_r the max over CTAs of their last role-r tile completion; all in ns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import tilelang.language as T

from .device import DEFAULT_DEVICE, DeviceSpec
from .kernel import PASS_CONFIGS_NO_WS, IOParam, KernelSpec, alloc_scratch, ceildiv, make_prim_func

# ----------------------------------------------------------------------------------
# special registers / 64-bit atomics (prelude + call_extern, no TileLang core change)
# ----------------------------------------------------------------------------------

PRELUDE = r"""
#ifndef COTILE_ORCH_PRELUDE
#define COTILE_ORCH_PRELUDE
__device__ __forceinline__ int cotile_smid() {
  unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return (int)r;
}
__device__ __forceinline__ int cotile_nsmid() {
  unsigned r; asm volatile("mov.u32 %0, %%nsmid;" : "=r"(r)); return (int)r;
}
__device__ __forceinline__ long long cotile_globaltimer() {
  unsigned long long r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return (long long)r;
}
// unsigned 64-bit max at gpu scope (relaxed); p points to an int64 tensor element
__device__ __forceinline__ void cotile_atomic_max_u64(void *p, long long v) {
  atomicMax(reinterpret_cast<unsigned long long *>(p), static_cast<unsigned long long>(v));
}
#endif
"""


def smid():
    """%smid of the executing SM (int32). Needs `prelude=PRELUDE` on the T.Kernel."""
    return T.call_extern("int32", "cotile_smid")


def nsmid():
    return T.call_extern("int32", "cotile_nsmid")


def globaltimer():
    """%globaltimer in ns (int64; 32 ns resolution on this GPU)."""
    return T.call_extern("int64", "cotile_globaltimer")


def atomic_max_u64(elem, value):
    """Unsigned 64-bit atomicMax on an int64 buffer element (fire-and-forget)."""
    return T.call_extern("handle", "cotile_atomic_max_u64", T.address_of(elem), value)


def bounded(tile_id, num_tiles: int):
    """Tile id with a range TileLang's arithmetic analysis can see.

    The dispatched tile id is read back from shared memory, an opaque value for TVM's
    analyzer. The tile bodies' index arithmetic on it (grouped rasterization, split
    decode) then cannot be proven in bounds, and LegalizeSafeMemoryAccess predicates
    the loads (`cp_async_gs_conditional`), which the solo persistent build (bounded
    loop expression + `if tile_id < num_tiles`) does not need. The dispatcher
    therefore both guards the call with `0 <= t < num_tiles` (a constraint the Z3
    fallback prover can use) and passes this clamp (a const-int bound for the
    rewrite simplifier); only the combination removes every predicate (measured:
    clamp alone leaves 4/10 cp.async predicated, guard alone 10/10, both 0/10).
    Both are the identity for every dispatched tile.

    History: without any bound, the Z3 fallback exhausted its resource limit on the
    GEMM body and TVM's Z3Prover::CanProve threw `InternalError: canceled` out of
    LayoutInference/LowerTileOp; co-tilelang now makes CanProve return "not provable"
    on solver exceptions (3rdparty/tvm/src/target/z3/z3_prover_on.cc)."""
    return T.min(T.max(tile_id, 0), num_tiles - 1)


# ----------------------------------------------------------------------------------
# orchestration description
# ----------------------------------------------------------------------------------

BINDINGS = ("sm", "cta", "tile")
SCHEDULES = ("static", "dynamic")
SMEM_MODES = ("alias", "sum")

# co_state layout (int32)
S_Q, S_RANK, S_STEAL, S_DONE, S_CTAS = 0, 2, 4, 6, 8
S_EXIT, S_EPOCH, S_ERR = 10, 11, 12
S_SMCTR = 16  # per-SM arrival (cta binding) / ticket (tile binding) counters, num_sms entries
N_OUT = 16

# int32 entries of the shared dispatch slot. Only 4 are used, but the slot is padded to 128 B:
# the merge planner places the slot at offset 0 of the dynamic-smem arena and the role scratch
# right after it with 16-byte alignment (it forces 1 KB alignment only for TMA/wgmma operands).
# Swizzled cp.async/ldmatrix buffers whose 128-byte rows start 16 bytes off a 128-byte boundary
# run ~1.7x slower (GEMM 4096^3 128x256 direct epilogue inside a CoKernel: 628 vs 373 us; with
# the scratch at 128/256/512/1024 B: 375 us; research/results/2026-09-23_methodology_v1, M4).
SLOT_INTS = 32

# slot values
SKIP = 2
DONE = -1

# AttrStmt key wrapped around one role's tile execution (smem="alias"): a shared-memory
# liveness boundary understood by TileLang's MergeSharedMemoryAllocations
# (src/op/builtin.h, tl::attr::kSharedLifetimeScope; co-tilelang core addition).
SHARED_LIFETIME_SCOPE = "tl.shared_lifetime_scope"


@dataclass(frozen=True)
class Orch:
    binding: str = "sm"
    schedule: str = "dynamic"
    chunk: int | tuple = 1  # tiles per atomic grab (dynamic); int or (chunk_A, chunk_B)
    takeover: bool = False
    num_ctas: int = 188
    threads: int | None = None  # None: max of the two roles' threads
    smem: str = "alias"
    timing: bool = True
    debug: bool = False
    min_blocks_per_sm: int = 1  # __launch_bounds__ 2nd argument (register cap)

    @property
    def chunks(self) -> tuple[int, int]:
        c = self.chunk
        return (int(c), int(c)) if isinstance(c, int) else (int(c[0]), int(c[1]))

    def tag(self) -> str:
        t = (
            f"{self.binding}_{self.schedule}"
            + (f"_c{'-'.join(str(c) for c in self.chunks)}" if self.schedule == "dynamic" else "")
            + ("_to" if self.takeover else "")
            + f"_n{self.num_ctas}"
            + (f"_t{self.threads}" if self.threads else "")
            + ("" if self.smem == "alias" else f"_{self.smem}")
            + ("" if self.timing else "_notime")
            + ("_dbg" if self.debug else "")
            + (f"_mb{self.min_blocks_per_sm}" if self.min_blocks_per_sm != 1 else "")
        )
        return t


@dataclass(frozen=True)
class CoConfig:
    """Compile-time description of a CoKernel (acts as `KernelSpec.cfg`)."""

    op_a: str
    cfg_a: Any
    op_b: str
    cfg_b: Any
    orch: Orch


class _CoOpInfo:
    """Module-like object so cotile.resources.signature() can describe a CoKernel."""

    NAME = "cokernel"

    @staticmethod
    def cfg_tag(cfg: CoConfig) -> str:
        from .ops import OPS

        return f"{cfg.op_a}[{OPS[cfg.op_a].cfg_tag(cfg.cfg_a)}]x{cfg.op_b}[{OPS[cfg.op_b].cfg_tag(cfg.cfg_b)}]:{cfg.orch.tag()}"

    @staticmethod
    def numerics(cfg: CoConfig):
        from .ops import OPS

        a = OPS[cfg.op_a].numerics(cfg.cfg_a)
        b = OPS[cfg.op_b].numerics(cfg.cfg_b)
        return (f"{a[0]}/{b[0]}", (a[1], b[1]))


CO_OP = _CoOpInfo()


@dataclass
class Role:
    idx: int  # 0 = A, 1 = B
    prefix: str
    op: Any
    shape: Any
    cfg: Any
    params: list  # the op's own (unprefixed) io params
    scratch: list
    threads: int
    num_tiles: int
    body: Any = field(repr=False, default=None)

    def io(self, args: dict):
        return self.op.make_io({p.name: args[self.prefix + p.name] for p in self.params})

    @property
    def tag(self) -> str:
        return f"{self.op.NAME}[{self.op.cfg_tag(self.cfg)}]"


def _make_role(idx: int, op, shape, cfg) -> Role:
    op.validate(shape, cfg)
    ts = op.tile_space(shape, cfg)
    return Role(
        idx=idx,
        prefix="ab"[idx] + "_",
        op=op,
        shape=shape,
        cfg=cfg,
        params=op.io_params(shape, cfg),
        scratch=op.scratch_spec(shape, cfg),
        threads=op.threads(cfg),
        num_tiles=ts.num_tiles,
        body=op.make_tile_body(shape, cfg),
    )


def validate_orch(orch: Orch, roles: list[Role]) -> int:
    """Check `orch` against the two roles; returns the CTA thread count."""
    if orch.binding not in BINDINGS:
        raise ValueError(f"binding must be one of {BINDINGS}")
    if orch.schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if orch.smem not in SMEM_MODES:
        raise ValueError(f"smem must be one of {SMEM_MODES}")
    if orch.binding == "tile" and (orch.schedule != "dynamic" or not orch.takeover):
        # the per-grab ticket needs per-role queues, and it always falls back to the other
        # role once the drawn one is exhausted (POD), i.e. takeover is part of the policy
        raise ValueError('binding="tile" needs schedule="dynamic" and takeover=True')
    if isinstance(orch.chunk, (tuple, list)) and len(orch.chunk) != 2:
        raise ValueError("chunk must be an int or a (chunk_A, chunk_B) pair")
    if min(orch.chunks) < 1:
        raise ValueError("chunk >= 1")
    if orch.num_ctas < 1:
        raise ValueError("num_ctas >= 1")
    threads = orch.threads or max(r.threads for r in roles)
    for r in roles:
        if r.threads > threads:
            raise ValueError(f"role {r.tag} needs {r.threads} threads > CoKernel threads {threads}")
    if threads % 32 or threads > 1024:
        raise ValueError("threads must be a multiple of 32 and <= 1024")
    return threads


def orch_params(orch: Orch, roles: list[Role], dev: DeviceSpec) -> list[IOParam]:
    NA, NB = roles[0].num_tiles, roles[1].num_tiles
    ps = [
        IOParam("co_knobs", (8,), "int32", "knob"),
        IOParam("co_state", (S_SMCTR + dev.num_sms,), "int32", "ctr"),
        IOParam("co_tacc", (4,), "int64", "ctr"),
        IOParam("co_out", (N_OUT,), "int64", "out"),
    ]
    if orch.binding == "sm":
        ps.append(IOParam("co_sm_role", (dev.num_sms,), "int32", "knob"))
    if orch.schedule == "static" and orch.takeover:
        ps.append(IOParam("co_claim", (NA + NB,), "int32", "ctr"))
    if orch.debug:
        ps.append(IOParam("co_dbg", (NA + NB,), "int32", "dbg"))
        ps.append(IOParam("co_dbg_sm", (NA + NB,), "int32", "dbg"))
        ps.append(IOParam("co_dbg_time", (2 * (NA + NB),), "int64", "dbg"))
    return ps


# ----------------------------------------------------------------------------------
# builder
# ----------------------------------------------------------------------------------


def build_cokernel(opA, shapeA, cfgA, opB, shapeB, cfgB, orch: Orch, dev: DeviceSpec = DEFAULT_DEVICE) -> KernelSpec:
    roles = [_make_role(0, opA, shapeA, cfgA), _make_role(1, opB, shapeB, cfgB)]
    threads = validate_orch(orch, roles)
    rA, rB = roles
    NA, NB = rA.num_tiles, rB.num_tiles
    num_ctas = orch.num_ctas
    num_sms = dev.num_sms
    alias = orch.smem == "alias"
    dynamic = orch.schedule == "dynamic"
    tile_bind = orch.binding == "tile"
    takeover = orch.takeover
    timing = orch.timing
    debug = orch.debug
    CHA, CHB = orch.chunks
    NPH = 2 if takeover else 1
    # every iteration either runs a tile or makes progress through a phase / claim
    MAX_IT = 2 * (NA + NB) + 2 * NPH + 2

    params = [IOParam(r.prefix + p.name, p.shape, p.dtype, p.role) for r in roles for p in r.params]
    params += orch_params(orch, roles, dev)
    names = [p.name for p in params]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate kernel parameter names: {names}")

    @T.macro
    def run_role(R, t, io, tx, scr):
        # One tile of role R. Under smem="alias" the call is wrapped in a
        # tl.shared_lifetime_scope AttrStmt: the tile body fully rewrites its scratch
        # before reading it (op protocol), so no shared-memory value crosses the
        # scope boundary and TileLang's merge planner may give both roles the same
        # bytes. The CTA barrier at the top of every dispatch iteration makes that
        # reuse safe at run time (all threads have finished the previous tile, of
        # either role, before any thread starts this one).
        if alias:
            # (the scope goes inside the thread guard so the planner sees the body's
            # statements as the scope's direct children -> reuse within the role too)
            if R.threads < threads:
                if tx < R.threads:
                    with T.attr(0, SHARED_LIFETIME_SCOPE, R.idx):
                        R.body(t, io, scr)
            else:
                with T.attr(0, SHARED_LIFETIME_SCOPE, R.idx):
                    R.body(t, io, scr)
        else:
            if R.threads < threads:
                if tx < R.threads:
                    R.body(t, io, scr)
            else:
                R.body(t, io, scr)

    @T.macro
    def flush(st, tacc, cr, cnt, tlast):
        # thread 0: publish the tiles this CTA completed in the phase that just ended
        if cnt > 0:
            T.atomic_add(st[S_DONE + cr], cnt)
            if timing:
                atomic_max_u64(tacc[1 + cr], tlast)

    @T.macro
    def kernel_body(args):
        ioA = rA.io(args)
        ioB = rB.io(args)
        knobs = args["co_knobs"]
        st = args["co_state"]
        tacc = args["co_tacc"]
        out = args["co_out"]
        with T.Kernel(num_ctas, threads=threads, prelude=PRELUDE) as bid:  # noqa: F841
            if orch.min_blocks_per_sm > 1:
                T.annotate_min_blocks_per_sm(orch.min_blocks_per_sm)
            # both roles' scratch at kernel scope; with smem="alias" the lifetime
            # scopes in run_role let the planner overlay them
            scrA = alloc_scratch(rA.scratch)
            scrB = alloc_scratch(rB.scratch)
            # [0..3]: double-buffered (role, tile) broadcast; [0] reused as "last CTA" flag.
            # Dynamic smem: a static __shared__ array would be padded to 1 KB because the
            # dynamic arena is 1024-byte aligned. Padded to SLOT_INTS (128 B) so the role
            # scratch that follows stays 128-byte aligned (see SLOT_INTS).
            slot = T.alloc_shared((SLOT_INTS,), "int32")
            tx = T.get_thread_binding()
            # thread-0 dispatcher state (registers; only thread 0 uses them)
            my_role = T.alloc_var("int32")
            sm_v = T.alloc_var("int32")
            rank = T.alloc_var("int32")
            epoch = T.alloc_var("int32")
            ph = T.alloc_var("int32")
            wi = T.alloc_var("int32")
            nxt = T.alloc_var("int32")
            left = T.alloc_var("int32")
            cnt = T.alloc_var("int32")
            cr = T.alloc_var("int32")
            nr = T.alloc_var("int32")
            o_r = T.alloc_var("int32")
            o_t = T.alloc_var("int32")
            j = T.alloc_var("int32")
            prev = T.alloc_var("int32")
            tlast = T.alloc_var("int64")
            tstart = T.alloc_var("int64")
            cur_r = T.alloc_var("int32")
            cur_t = T.alloc_var("int32")

            if tx == 0:
                atomic_max_u64(tacc[0], T.int64(-1) - globaltimer())
                sm_v = smid()
                if sm_v >= num_sms:
                    T.atomic_add(st[S_ERR], 1)
                    sm_v = sm_v % num_sms
                if orch.binding == "sm":
                    my_role = args["co_sm_role"][sm_v]
                elif orch.binding == "cta":
                    j = T.atomic_add(st[S_SMCTR + sm_v], 1, return_prev=True)
                    my_role = T.if_then_else(j % (knobs[0] + knobs[1]) < knobs[0], 0, 1)
                else:
                    # tile binding: no per-CTA role (drawn per grab); CTAs are counted as A
                    my_role = 0
                    cr = 0
                if not dynamic:
                    rank = T.atomic_add(st[S_RANK + my_role], 1, return_prev=True)
                    if takeover:
                        epoch = st[S_EPOCH] + 1
                ph = 0
                wi = 0
                nxt = 0
                left = 0
                cnt = 0
                tlast = T.int64(0)

            for it in T.serial(MAX_IT):
                if tx == 0:
                    o_r = DONE
                    o_t = 0
                    if tile_bind:
                        # POD policy: draw the role of every grab from the SM's ticket counter;
                        # fall back to the other role if the drawn queue is exhausted; ph = 1
                        # once both are (o_r stays DONE). The finished chunk is published
                        # (flush) before the next draw, so per-role counts and end times stay
                        # exact although the CTA alternates roles.
                        if ph == 0:
                            if left == 0:
                                flush(st, tacc, cr, cnt, tlast)
                                cnt = 0
                                j = T.atomic_add(st[S_SMCTR + sm_v], 1, return_prev=True)
                                cr = T.if_then_else(j % (knobs[0] + knobs[1]) < knobs[0], 0, 1)
                                nr = T.if_then_else(cr == 0, NA, NB)
                                ch = T.if_then_else(cr == 0, CHA, CHB)
                                nxt = T.atomic_add(st[S_Q + cr], ch, return_prev=True)
                                left = T.max(T.min(nr - nxt, ch), 0)
                                if left == 0:
                                    cr = 1 - cr
                                    nr = T.if_then_else(cr == 0, NA, NB)
                                    ch2 = T.if_then_else(cr == 0, CHA, CHB)
                                    nxt = T.atomic_add(st[S_Q + cr], ch2, return_prev=True)
                                    left = T.max(T.min(nr - nxt, ch2), 0)
                            if left > 0:
                                o_r = cr
                                o_t = nxt
                                nxt = nxt + 1
                                left = left - 1
                            else:
                                ph = 1
                    elif ph < NPH:
                        cr = T.if_then_else(ph == 0, my_role, 1 - my_role)
                        nr = T.if_then_else(cr == 0, NA, NB)
                        if dynamic:
                            if left == 0:
                                ch = T.if_then_else(cr == 0, CHA, CHB)
                                nxt = T.atomic_add(st[S_Q + cr], ch, return_prev=True)
                                left = T.max(T.min(nr - nxt, ch), 0)
                            if left > 0:
                                o_r = cr
                                o_t = nxt
                                nxt = nxt + 1
                                left = left - 1
                            else:
                                flush(st, tacc, cr, cnt, tlast)
                                cnt = 0
                                ph = ph + 1
                                o_r = SKIP
                        else:
                            if ph == 0:
                                j = rank + wi * knobs[2 + cr]
                                wi = wi + 1
                                if rank < knobs[2 + cr] and j < nr:
                                    o_t = j
                                    o_r = cr
                                    if takeover:
                                        prev = T.atomic_max(args["co_claim"][cr * NA + j], epoch, return_prev=True)
                                        if prev >= epoch:
                                            o_r = SKIP  # stolen by another CTA
                                else:
                                    flush(st, tacc, cr, cnt, tlast)
                                    cnt = 0
                                    ph = ph + 1
                                    o_r = SKIP
                            if takeover:
                                # (trace-time guard: co_claim exists only with takeover)
                                if ph == 1:
                                    # steal the other role's tiles from the back of its range
                                    # (also right after the own phase ended in this iteration)
                                    cr = 1 - my_role
                                    nr = T.if_then_else(cr == 0, NA, NB)
                                    j = T.atomic_add(st[S_STEAL + cr], 1, return_prev=True)
                                    o_r = SKIP
                                    if j < nr:
                                        prev = T.atomic_max(args["co_claim"][cr * NA + nr - 1 - j], epoch, return_prev=True)
                                        if prev < epoch:
                                            o_r = cr
                                            o_t = nr - 1 - j
                                    if o_r == SKIP:
                                        flush(st, tacc, cr, cnt, tlast)
                                        cnt = 0
                                        ph = ph + 1
                    slot[(it % 2) * 2] = o_r
                    slot[(it % 2) * 2 + 1] = o_t
                T.sync_threads()
                cur_r = slot[(it % 2) * 2]
                cur_t = slot[(it % 2) * 2 + 1]
                if cur_r == DONE:
                    T.loop_break()
                if debug:
                    if tx == 0:
                        tstart = globaltimer()
                # dispatch; see bounded() for why the tile id is guarded AND clamped
                if cur_r == 0:
                    if cur_t >= 0 and cur_t < NA:
                        run_role(rA, bounded(cur_t, NA), ioA, tx, scrA)
                elif cur_r == 1:
                    if cur_t >= 0 and cur_t < NB:
                        run_role(rB, bounded(cur_t, NB), ioB, tx, scrB)
                if cur_r < SKIP:
                    if timing:
                        T.sync_threads()
                    if tx == 0:
                        cnt = cnt + 1
                        if timing:
                            tlast = globaltimer()
                        if debug:
                            T.atomic_add(args["co_dbg"][cur_r * NA + cur_t], 1)
                            args["co_dbg_sm"][cur_r * NA + cur_t] = sm_v
                            args["co_dbg_time"][2 * (cur_r * NA + cur_t)] = tstart
                            args["co_dbg_time"][2 * (cur_r * NA + cur_t) + 1] = globaltimer()

            # ---- exit: the last CTA publishes results and resets all counters ----
            T.sync_threads()
            if tx == 0:
                T.atomic_add(st[S_CTAS + my_role], 1)
                prev = T.atomic_add(st[S_EXIT], 1, memory_order="acq_rel", return_prev=True)
                slot[0] = T.if_then_else(prev == num_ctas - 1, 1, 0)
            T.sync_threads()
            if slot[0] == 1:
                if tx == 0:
                    out[3] = globaltimer()
                    out[0] = T.int64(-1) - tacc[0]
                    out[1] = tacc[1]
                    out[2] = tacc[2]
                    out[4] = T.cast(st[S_DONE], "int64")
                    out[5] = T.cast(st[S_DONE + 1], "int64")
                    out[6] = T.cast(st[S_CTAS], "int64")
                    out[7] = T.cast(st[S_CTAS + 1], "int64")
                    out[8] = T.cast(st[S_RANK], "int64")
                    out[9] = T.cast(st[S_RANK + 1], "int64")
                    out[10] = T.cast(st[S_EPOCH], "int64")
                    out[11] = T.cast(st[S_STEAL], "int64")
                    out[12] = T.cast(st[S_STEAL + 1], "int64")
                    out[13] = T.cast(st[S_Q], "int64")
                    out[14] = T.cast(st[S_Q + 1], "int64")
                    out[15] = T.cast(st[S_ERR], "int64")
                    for k in T.unroll(S_SMCTR):
                        if k != S_EPOCH:
                            st[k] = 0
                    st[S_EPOCH] = st[S_EPOCH] + 1
                    for k in T.unroll(4):
                        tacc[k] = T.int64(0)
                if orch.binding in ("cta", "tile"):
                    for i in T.serial(ceildiv(num_sms, threads)):
                        if i * threads + tx < num_sms:
                            st[S_SMCTR + i * threads + tx] = 0

    cfg = CoConfig(opA.NAME, cfgA, opB.NAME, cfgB, orch)
    h = hashlib.sha1(repr(cfg).encode()).hexdigest()[:8]
    name = f"cokernel_{opA.NAME}_{opB.NAME}_{orch.binding}_{orch.schedule}_{h}"
    pf = make_prim_func(name, params, kernel_body)
    pcs = dict(PASS_CONFIGS_NO_WS)
    for r in roles:
        for k, v in r.op.pass_configs(r.cfg, "persistent").items():
            if k in pcs and pcs[k] != v:
                raise ValueError(f"conflicting pass config {k}: {pcs[k]} vs {v}")
            pcs[k] = v
    sA = opA.smem_bytes(shapeA, cfgA, "persistent")
    sB = opB.smem_bytes(shapeB, cfgB, "persistent")
    spec = KernelSpec(
        op=CO_OP,
        shape=(shapeA, shapeB),
        cfg=cfg,
        build="cokernel",
        name=name,
        prim_func=pf,
        pass_configs=pcs,
        params=params,
        num_tiles=NA + NB,
        grid=num_ctas,
        threads=threads,
        # upper-bound estimate: role footprints as in the solo persistent build
        smem_estimate=(max(sA, sB) if alias else sA + sB),
    )
    spec.extra["co"] = CoInfo(roles=roles, orch=orch, threads=threads, device=dev)
    return spec


@dataclass
class CoInfo:
    roles: list
    orch: Orch
    threads: int
    device: DeviceSpec


# ----------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------


def sm_role_table(n_a: int, dev: DeviceSpec = DEFAULT_DEVICE, order: str = "contiguous") -> list[int]:
    """SM-binding table with `n_a` SMs for role A and the rest for role B.

    order="contiguous": smid < n_a -> A. order="interleave": spread A's SMs evenly
    over the smid range (every TPC pair / GPC group gets a share)."""
    n = dev.num_sms
    if not 0 <= n_a <= n:
        raise ValueError(f"n_a must be in [0, {n}]")
    if order == "contiguous":
        return [0 if s < n_a else 1 for s in range(n)]
    if order == "interleave":
        return [0 if (s * n_a) // n != ((s + 1) * n_a) // n else 1 for s in range(n)]
    raise ValueError("order must be contiguous|interleave")


def expected_role_ctas(orch: Orch, sm_role=None, ratio=None, dev: DeviceSpec = DEFAULT_DEVICE) -> tuple[int, int]:
    """Number of CTAs that take role A / B, assuming the persistent grid is placed
    breadth-first with exactly num_ctas/num_sms co-resident CTAs per SM (measured
    behaviour on this GPU, research/results/2026-09-22_smid_probe). The static
    schedule uses these as strides; the kernel reports the actual counts."""
    n = dev.num_sms
    if orch.num_ctas % n:
        raise ValueError(f"static schedule needs num_ctas % {n} == 0 (one placement per SM), got {orch.num_ctas}")
    c = orch.num_ctas // n
    if orch.binding == "sm":
        na = sum(1 for r in sm_role if r == 0)
        return c * na, c * (n - na)
    ka, kb = ratio
    per_sm_a = sum(1 for a in range(c) if a % (ka + kb) < ka)
    return n * per_sm_a, n * (c - per_sm_a)


class CoRunner:
    """Calls a compiled CoKernel with per-role torch tensors.

    State (ws/ctr buffers of both ops and the orchestration counters) is allocated on
    first use and reused by every later call; counters are zeroed once (the kernel
    leaves them zero). Knobs are runtime: `set_knobs` never recompiles."""

    def __init__(self, spec: KernelSpec, state: dict | None = None):
        if spec.kernel is None:
            raise RuntimeError(f"{spec.name} is not compiled ({spec.compile_error})")
        self.spec = spec
        self.info: CoInfo = spec.extra["co"]
        self.state = state if state is not None else {}
        self.knob_values: dict = {}
        self.expected = None
        self.last_out = None
        self.dbg = {}

    # -- knobs ---------------------------------------------------------------------
    def set_knobs(self, sm_role=None, ratio=(1, 1), device="cuda"):
        import torch

        orch = self.info.orch
        dev = self.info.device
        kn = [0] * 8
        if orch.binding == "sm":
            if sm_role is None:
                sm_role = sm_role_table(dev.num_sms // 2, dev)
            sm_role = [int(r) for r in sm_role]
            if len(sm_role) != dev.num_sms or any(r not in (0, 1) for r in sm_role):
                raise ValueError("sm_role must have num_sms entries in {0, 1}")
            self.state["co_sm_role"] = torch.tensor(sm_role, dtype=torch.int32, device=device)
        else:
            ka, kb = (int(x) for x in ratio)
            if ka < 0 or kb < 0 or ka + kb < 1:
                raise ValueError("ratio (kA, kB) must be >= 0 with kA + kB >= 1")
            kn[0], kn[1] = ka, kb
        if orch.schedule == "static":
            na, nb = expected_role_ctas(orch, sm_role, (kn[0], kn[1]), dev)
            kn[2], kn[3] = na, nb
            self.expected = (na, nb)
        else:
            self.expected = None
        self.knob_values = {"sm_role": sm_role, "ratio": (kn[0], kn[1]), "strides": (kn[2], kn[3])}
        self.state["co_knobs"] = torch.tensor(kn, dtype=torch.int32, device=device)

    # -- launch --------------------------------------------------------------------
    def _buf(self, p: IOParam, device):
        import torch

        buf = self.state.get(p.name)
        if buf is None or tuple(buf.shape) != tuple(p.shape):
            tdt = getattr(torch, p.dtype)
            buf = torch.zeros(p.shape, dtype=tdt, device=device) if p.role == "ctr" else torch.empty(p.shape, dtype=tdt, device=device)
            self.state[p.name] = buf
        return buf

    def make_args(self, inputs: list[dict], outputs: list[dict] | None = None):
        import torch

        roles = self.info.roles
        device = next(iter(inputs[0].values())).device
        if "co_knobs" not in self.state:
            self.set_knobs(device=device)
        outputs = [dict(o) for o in (outputs or [{}, {}])]
        args = []
        for p in self.spec.params:
            role = next((r for r in roles if p.name.startswith(r.prefix)), None)
            if role is not None:
                local = p.name[len(role.prefix) :]
                if p.role == "in":
                    t = inputs[role.idx][local]
                    if tuple(t.shape) != tuple(p.shape) or str(t.dtype) != f"torch.{p.dtype}":
                        raise ValueError(f"{p.name}: expected {p.shape} {p.dtype}, got {tuple(t.shape)} {t.dtype}")
                    args.append(t)
                    continue
                if p.role == "out":
                    if local not in outputs[role.idx]:
                        outputs[role.idx][local] = torch.empty(p.shape, dtype=getattr(torch, p.dtype), device=device)
                    args.append(outputs[role.idx][local])
                    continue
                args.append(self._buf(p, device))
                continue
            if p.role == "knob":
                args.append(self.state[p.name])
            elif p.role == "out":
                self.last_out = torch.empty(p.shape, dtype=getattr(torch, p.dtype), device=device)
                args.append(self.last_out)
            elif p.role == "dbg":
                buf = self._buf(p, device)
                buf.fill_(0 if p.name == "co_dbg" else -1)
                self.dbg[p.name] = buf
                args.append(buf)
            else:
                args.append(self._buf(p, device))
        return args, outputs

    def __call__(self, inputs: list[dict], outputs: list[dict] | None = None) -> list[dict]:
        args, outputs = self.make_args(inputs, outputs)
        self.spec.kernel(*args)
        return outputs

    def reset_state(self):
        """Re-zero all counters (only needed after an aborted launch)."""
        for p in self.spec.params:
            if p.role == "ctr" and p.name in self.state:
                self.state[p.name].zero_()

    # -- results -------------------------------------------------------------------
    def stats(self, check: bool = True) -> dict:
        """Read co_out of the last launch (synchronizes)."""
        o = [int(x) for x in self.last_out.cpu().tolist()]
        roles = self.info.roles
        s = {
            "t_start": o[0],
            "T_A_ns": o[1] - o[0] if o[1] > 0 else None,
            "T_B_ns": o[2] - o[0] if o[2] > 0 else None,
            "exit_ns": o[3] - o[0],
            "makespan_ns": max(o[1], o[2]) - o[0] if max(o[1], o[2]) > 0 else None,
            "done_A": o[4],
            "done_B": o[5],
            "ctas_A": o[6],
            "ctas_B": o[7],
            "tickets_A": o[8],
            "tickets_B": o[9],
            "epoch": o[10],
            "steal_A": o[11],
            "steal_B": o[12],
            "q_A": o[13],
            "q_B": o[14],
            "smid_errors": o[15],
            "tiles_A": roles[0].num_tiles,
            "tiles_B": roles[1].num_tiles,
        }
        if check:
            if s["smid_errors"]:
                raise RuntimeError(f"{s['smid_errors']} CTAs saw %smid >= num_sms")
            if self.expected is not None and (s["tickets_A"], s["tickets_B"]) != self.expected:
                raise RuntimeError(
                    f"static schedule placement assumption violated: expected {self.expected} CTAs per role, "
                    f"got {(s['tickets_A'], s['tickets_B'])}"
                )
        return s
