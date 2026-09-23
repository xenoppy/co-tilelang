"""Shared helpers for the methodology-v1 study scripts (research/results/2026-09-23_methodology_v1).

Op launchers for cotile kernels (grid / persistent / schedule variants / CoKernels) whose
argument tuples can be created repeatedly (for cobench.Rotation / graph mode), plus JSON I/O.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "research", "results", "2026-09-23_methodology_v1")

import cobench as cb  # noqa: E402
from cotile import catalog  # noqa: E402
from cotile.kernel import compile_specs  # noqa: E402
from cotile.ops import OPS  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name: str, obj) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    return path


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


def env_meta() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    return {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "device": cb.device_info(), "torch": torch.__version__}


def solo_best(op: str, shape_tag: str):
    """(op module, shape obj, cfg obj, catalog record) of the catalog's solo-best grid config."""
    cat = catalog.load(ops=[op])
    e = cat.get(op, shape_tag)
    return OPS[op], e.shape_obj(), e.solo_best.cfg(), e.solo_best


def cfg_of(op: str, shape_tag: str, tag: str):
    cat = catalog.load(ops=[op])
    e = cat.get(op, shape_tag)
    rec = next(c for c in e.measured() if c.tag == tag)
    return OPS[op], e.shape_obj(), rec.cfg(), rec


class OpLauncher:
    """Launch closures for a compiled cotile KernelSpec.

    ``make_args()`` returns a fresh argument list (new inputs + outputs; ws/ctr/knob/dbg
    buffers shared across copies), ``run(*args)`` launches on torch's current stream.
    ``extra_state`` provides non-op parameters (schedule-variant knobs, CoKernel state)."""

    def __init__(self, spec, inputs_fn=None, state: dict | None = None):
        if spec.kernel is None:
            raise RuntimeError(f"{spec.name}: {spec.compile_error}")
        self.spec = spec
        self.op = spec.op
        self.state = state if state is not None else {}
        self.inputs_fn = inputs_fn

    def _shared(self, p):
        buf = self.state.get(p.name)
        if buf is None:
            tdt = getattr(torch, p.dtype)
            buf = (torch.zeros(p.shape, dtype=tdt, device="cuda") if p.role in ("ctr", "dbg", "knob")
                   else torch.empty(p.shape, dtype=tdt, device="cuda"))
            self.state[p.name] = buf
        return buf

    def make_args(self, inputs: dict | None = None) -> tuple:
        if inputs is None:
            inputs = self.inputs_fn() if self.inputs_fn else self.op.make_inputs(self.spec.shape)
        args = []
        for p in self.spec.params:
            if p.role == "in":
                args.append(inputs[p.name])
            elif p.role == "out":
                args.append(torch.empty(p.shape, dtype=getattr(torch, p.dtype), device="cuda"))
            else:
                args.append(self._shared(p))
        return tuple(args)

    def run(self, *args):
        self.spec.kernel(*args)

    def out_index(self, name: str) -> int:
        return [p.name for p in self.spec.params].index(name)


def compile_all(specs, workers: int = 16) -> dict:
    t = time.time()
    st = compile_specs(list(specs), num_workers=workers)
    st["wall_s"] = time.time() - t
    bad = [s.name for s in specs if s.kernel is None]
    if bad:
        raise RuntimeError(f"compile failures: {bad}: {[s.compile_error for s in specs if s.kernel is None]}")
    return st


def gpu_procs() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]
