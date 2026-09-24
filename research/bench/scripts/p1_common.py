"""Shared pieces of the P1 3x2 study (results: research/results/2026-09-23_p1_3x2_A).

* study pairs and SM-budget grids;
* ``OpData``: rotating input/output copies of one op x shape (> 2x L2, so every call is
  DRAM-cold without a flush), shared by every config of that shape;
* launchers: ``OpData.launcher(spec)`` (grid kernel, fn(i)), ``CoVariant`` (one CoKernel launch
  per iteration, own runner = own knobs/counters), ``co_role_times`` (device-side per-role
  completion times of a timing build);
* ``ws_off_twin``: the CoKernel-compatible version of a GEMM config (auto warp specialization
  cannot coexist with a second role, cotile/kernel.py);
* JSON helpers.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "research", "results", "2026-09-23_p1_3x2_A")
SOLO_DIR = os.path.join(OUT, "solo_v1")

import cobench as cb  # noqa: E402
from cotile import catalog  # noqa: E402
from cotile.cokernel import CoRunner  # noqa: E402
from cotile.kernel import compile_specs  # noqa: E402
from cotile.ops import OPS  # noqa: E402

FULL = 188
SHAPES = {"gemm": ["M4096_N4096_K4096", "M2048_N4096_K4096"],
          "gqa_decode": ["B16_S8192", "B64_S8192", "B32_S8192", "B32_S2048"]}
# name -> (GEMM shape, decode shape); duration ratios from the P1-S pairing table
PAIRS = {
    "main": ("M4096_N4096_K4096", "B16_S8192"),      # 392 / 343 us, ratio 1.14
    "r029": ("M4096_N4096_K4096", "B64_S8192"),      # ratio 0.29
    "r058": ("M4096_N4096_K4096", "B32_S8192"),      # ratio 0.58
    "r223": ("M4096_N4096_K4096", "B32_S2048"),      # ratio 2.23
    "second": ("M2048_N4096_K4096", "B32_S2048"),    # 202 / 176 us, ratio 1.15
}
# green-context split grid: GEMM gets n_A SMs [0, n_A), decode the other 188 - n_A
# (IGNORE_SM_COSCHEDULING: exact counts, 2-SM granularity)
GREEN_A = (48, 64, 80, 94, 108, 124, 140, 148, 156, 164, 172, 180)
# solo SM budgets measured per op (A1): the green grid seen from each side, plus the full GPU
BUDGETS = {"gemm": tuple(sorted(set(GREEN_A) | {FULL})),
           "gqa_decode": tuple(sorted({FULL - n for n in GREEN_A} | {FULL}))}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _json_default(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


def save(path: str, obj) -> str:
    path = path if os.path.isabs(path) else os.path.join(OUT, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)
    return path


def load(path: str):
    path = path if os.path.isabs(path) else os.path.join(OUT, path)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def env_meta() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    return {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "torch": torch.__version__,
            "gpu_procs": gpu_procs()}


def gpu_procs() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def wait_gpu(need_bytes: int = 0, margin: int = 1 << 30, poll_s: float | None = None,
             max_wait_s: float | None | str = "policy", policy=None) -> float:
    """Block until the GPU is free under the cobench guard policy (cobench.GuardPolicy,
    research/rules.md 7: foreign SM activity -> wait 30 min and re-check; GpuBusy after 2 h;
    GpuYield instead of waiting if the policy yields) AND
    at least need_bytes + margin of device memory are free. poll_s / max_wait_s override the
    policy's fields. Returns the seconds waited."""
    pol = policy or cb.get_policy()
    kw = {} if poll_s is None else {"poll_s": poll_s}
    if max_wait_s != "policy":
        kw["max_wait_s"] = max_wait_s
    pol = dataclasses.replace(pol, **kw) if kw else pol
    t0 = time.time()
    while True:
        left = None if pol.max_wait_s is None else max(1.0, pol.max_wait_s - (time.time() - t0))
        cb.wait_until_free(policy=dataclasses.replace(pol, max_wait_s=left))
        free, _ = torch.cuda.mem_get_info()
        if free >= need_bytes + margin:
            return time.time() - t0
        if pol.yield_to_caller:
            raise cb.GpuYield(f"only {free / 2**30:.1f} GiB free < {(need_bytes + margin) / 2**30:.1f} GiB")
        if pol.max_wait_s is not None and time.time() - t0 > pol.max_wait_s:
            raise cb.GpuBusy(f"only {free / 2**30:.1f} GiB free after {pol.max_wait_s:.0f}s")
        log(f"[mem] {free / 2**30:.1f} GiB free < {(need_bytes + margin) / 2**30:.1f} GiB; waiting {pol.poll_s:.0f}s")
        time.sleep(pol.poll_s)


def retry_oom(fn, *a, tries: int = 30, poll_s: float | None = None, **kw):
    """Run fn; on a CUDA OOM (foreign process holding the memory), release cached memory, wait
    for the GPU (wait_gpu) and retry. cobench.GpuYield (policy.yield_to_caller) and GpuBusy
    propagate to the caller."""
    import gc
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except torch.OutOfMemoryError as e:
            log(f"[mem] OOM in {getattr(fn, '__name__', fn)} (attempt {i + 1}): {str(e).splitlines()[0][:160]}")
            gc.collect()
            torch.cuda.empty_cache()
            wait_gpu(poll_s=poll_s)
        except cb.GpuBusy:
            raise
        except RuntimeError as e:
            # cobench benches give up after 3 contaminated attempts (foreign SM activity during
            # the timed window): wait for a free GPU and redo the whole step
            if "contaminated" not in str(e):
                raise
            log(f"[guard] {getattr(fn, '__name__', fn)}: {str(e)[:160]}; waiting and redoing")
            wait_gpu(poll_s=poll_s)
    raise RuntimeError(f"{fn}: gave up after {tries} attempts (OOM / contaminated)")


def compile_all(specs, workers: int = 32) -> dict:
    """Compile (kernel-cache hits are cheap); dedupe identical specs; raise on failures.
    Kernel names encode op, config and build but NOT the shape (cotile.kernel._kernel_name,
    cokernel names), so the dedupe key includes the shape, grid and pass configs."""
    uniq = {}
    for s in specs:
        key = (s.name, repr(s.shape), s.grid, repr(sorted(s.pass_configs.items())))
        uniq.setdefault(key, []).append(s)
    todo = [v[0] for v in uniq.values() if v[0].kernel is None]
    t = time.time()
    st = compile_specs(todo, num_workers=workers) if todo else {"n_ok": 0, "n_fail": 0}
    for v in uniq.values():
        for s in v[1:]:
            s.kernel, s.compile_error = v[0].kernel, v[0].compile_error
    st["wall_s"] = time.time() - t
    bad = [s.name for s in specs if s.kernel is None]
    if bad:
        raise RuntimeError(f"compile failures: {[(s.name, s.compile_error) for s in specs if s.kernel is None][:5]}")
    return st


def ws_off_twin(op, cfg):
    """CoKernel-compatible config: GEMM ws='auto' -> 'off' (same tile/stages/threads; the
    auto-WS kernel adds 128 producer threads, the twin uses cp.async + mma.sync). Identity
    for everything else."""
    if op.NAME == "gemm" and cfg.ws != "off":
        return dataclasses.replace(cfg, ws="off")
    return cfg


def cfg_from_tag(op_name: str, shape_tag: str, tag: str):
    op = OPS[op_name]
    shape = catalog.shape_obj(op_name, _shape_fields(op_name, shape_tag))
    for c in op.configs(shape):
        if op.cfg_tag(c) == tag:
            return c
    raise KeyError(f"{op_name} {shape_tag}: no config {tag}")


def _shape_fields(op_name: str, tag: str) -> dict:
    letters = dict(zip(catalog._SHAPE_LETTERS[op_name], catalog._SHAPE_FIELDS[op_name]))
    out = {}
    for part in tag.split("_"):
        out[letters[part[0]]] = int(part[1:])
    return out


def shape_of(op_name: str, shape_tag: str):
    return catalog.shape_obj(op_name, _shape_fields(op_name, shape_tag))


# ----------------------------------------------------------------------------------
# rotating inputs + launchers
# ----------------------------------------------------------------------------------


class OpData:
    """n copies of (inputs, outputs) of one op x shape, n*bytes >= min_bytes (default 2x L2).
    Every config of the shape uses the same copies (their io_params differ only in ws/ctr)."""

    def __init__(self, op_name: str, shape_tag: str, min_bytes: int | None = None):
        self.op = OPS[op_name]
        self.name = op_name
        self.tag = shape_tag
        self.shape = shape_of(op_name, shape_tag)
        cfg0 = self.op.configs(self.shape)[0]
        self.out_params = [p for p in self.op.io_params(self.shape, cfg0) if p.role == "out"]
        min_bytes = int(min_bytes if min_bytes is not None else 2 * cb.l2_bytes())
        self.copies = []
        total = 0
        while total < min_bytes:
            ins = self.op.make_inputs(self.shape, seed=len(self.copies))
            outs = {p.name: torch.zeros(p.shape, dtype=getattr(torch, p.dtype), device="cuda") for p in self.out_params}
            self.copies.append((ins, outs))
            total += cb.tensor_bytes((ins, outs))
        self.bytes_per_copy = total // len(self.copies)
        self.n = len(self.copies)

    def arglists(self, spec, state: dict | None = None) -> list:
        state = {} if state is None else state
        out = []
        for ins, outs in self.copies:
            args = []
            for p in spec.params:
                if p.role == "in":
                    args.append(ins[p.name])
                elif p.role == "out":
                    args.append(outs[p.name])
                else:
                    if p.name not in state:
                        tdt = getattr(torch, p.dtype)
                        state[p.name] = (torch.zeros(p.shape, dtype=tdt, device="cuda") if p.role == "ctr"
                                         else torch.empty(p.shape, dtype=tdt, device="cuda"))
                    args.append(state[p.name])
            out.append(args)
        return out

    def launcher(self, spec, state: dict | None = None):
        """fn(i): one call of the compiled grid/persistent kernel `spec` on copy i % n.
        `state`: shared dict for the ws/ctr buffers (e.g. a StatePool view), so launchers of
        configs with equal workspace shapes do not each allocate their own."""
        k, al, n = spec.kernel, self.arglists(spec, state), self.n

        def fn(i):
            k(*al[i % n])
        fn.spec_name = spec.name
        return fn

    def outputs(self, k: int = 0) -> list:
        return [self.copies[k][1][p.name] for p in self.out_params]


class CoVariant:
    """One CoKernel launch per iteration on copy i % n (both ops' copies; own CoRunner, i.e.
    own knobs and counters, so several knob settings of one kernel can be interleaved)."""

    def __init__(self, spec, da: OpData, db: OpData, sm_role=None, ratio=(1, 1), pool: "StatePool | None" = None):
        self.spec = spec
        self.run = CoRunner(spec)
        if pool is not None:
            # the roles' split workspaces / counters come from the shared pool (they are
            # self-resetting and never used by two launches at the same time: every variant
            # runs its iterations back-to-back on one stream); co_* state stays private
            for p in spec.params:
                for pre, side in (("a_", "a"), ("b_", "b")):
                    if p.name.startswith(pre) and p.role in ("ws", "ctr"):
                        self.run.state[p.name] = pool.get(side, p.name[len(pre):], p.shape, p.dtype, p.role)
        self.run.set_knobs(sm_role=sm_role, ratio=ratio)
        self.n = max(da.n, db.n)
        self.arglists = []
        for k in range(self.n):
            ia, oa = da.copies[k % da.n]
            ib, ob = db.copies[k % db.n]
            args, _ = self.run.make_args([ia, ib], [oa, ob])
            self.arglists.append(args)
        self.kernel = spec.kernel
        self.i_out = [p.name for p in spec.params].index("co_out")

    def __call__(self, i):
        self.kernel(*self.arglists[i % self.n])

    def role_times(self, n: int = 24, warm: int = 800) -> dict:
        """Launch n times back-to-back (rotating copies), each with its own co_out; median
        device-side T_A, T_B (end of the role's last tile), makespan, exit time (us) and the
        takeover counts. Only meaningful for timing=True builds."""
        import numpy as np
        outs = [torch.zeros(16, dtype=torch.int64, device="cuda") for _ in range(n)]
        for j in range(warm):          # back-to-back first: the power controller settles in 0.1-0.3 s
            self(j)
        for j in range(n):
            args = list(self.arglists[j % self.n])
            args[self.i_out] = outs[j]
            self.kernel(*args)
        torch.cuda.synchronize()
        rows = [[int(x) for x in o.cpu().tolist()] for o in outs]
        def med(v):
            v = [x for x in v if x is not None]
            return float(np.median(v)) if v else None
        ta = [(r[1] - r[0]) / 1e3 if r[1] > 0 else None for r in rows]
        tb = [(r[2] - r[0]) / 1e3 if r[2] > 0 else None for r in rows]
        return {"T_A_us": med(ta), "T_B_us": med(tb),
                "makespan_us": med([max(r[1], r[2]) / 1e3 - r[0] / 1e3 for r in rows]),
                "exit_us": med([(r[3] - r[0]) / 1e3 for r in rows]),
                "steal_A": med([r[11] for r in rows]), "steal_B": med([r[12] for r in rows]),
                "done_A": med([r[4] for r in rows]), "done_B": med([r[5] for r in rows]),
                "ctas_A": med([r[6] for r in rows]), "ctas_B": med([r[7] for r in rows]), "n": n}


class StatePool:
    """Split-reduction workspaces and arrival counters shared by every launcher / CoKernel
    of a study, keyed by (side, name, shape, dtype). Counters are zeroed once (kernels leave
    them zero); workspaces need no init. Keeps the memory footprint to one buffer per
    distinct shape (split-K GEMM workspaces are 128-268 MB each)."""

    def __init__(self):
        self.bufs: dict = {}

    def get(self, side, name, shape, dtype, role):
        key = (side, name, tuple(shape), dtype)
        if key not in self.bufs:
            tdt = getattr(torch, dtype)
            self.bufs[key] = (torch.zeros(tuple(shape), dtype=tdt, device="cuda") if role == "ctr"
                              else torch.empty(tuple(shape), dtype=tdt, device="cuda"))
        return self.bufs[key]

    def view(self, side: str, spec) -> "_PoolView":
        """Mapping for OpData.launcher/arglists(spec, state): the ws/ctr params of `spec`
        (one op, unprefixed names) resolved in this pool."""
        return _PoolView(self, side).bind(spec)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.bufs.values())


class _PoolView:
    """Mapping used by OpData.arglists: state[name] for a ws/ctr param of one side."""

    def __init__(self, pool: StatePool, side: str):
        self.pool, self.side, self.params = pool, side, {}

    def bind(self, spec):
        self.params = {p.name: p for p in spec.params}
        return self

    def __contains__(self, name):
        p = self.params[name]
        return (self.side, name, tuple(p.shape), p.dtype) in self.pool.bufs

    def __setitem__(self, name, value):
        p = self.params[name]
        self.pool.bufs[(self.side, name, tuple(p.shape), p.dtype)] = value

    def __getitem__(self, name):
        p = self.params[name]
        return self.pool.get(self.side, name, p.shape, p.dtype, p.role)


def check_same(fn_ref_a, fn_ref_b, da: OpData, db: OpData, variant) -> bool:
    """variant's outputs on copy 0 == the reference kernels' outputs (bitwise)."""
    for x in da.outputs(0) + db.outputs(0):
        x.zero_()
    fn_ref_a(0)
    fn_ref_b(0)
    torch.cuda.synchronize()
    ref = [x.clone() for x in da.outputs(0) + db.outputs(0)]
    for x in da.outputs(0) + db.outputs(0):
        x.zero_()
    variant(0)
    torch.cuda.synchronize()
    return all(torch.equal(x, y) for x, y in zip(da.outputs(0) + db.outputs(0), ref))
