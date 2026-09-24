"""Run a resumable GPU study under the GPU-sharing policy (research/rules.md 7; cobench.GuardPolicy).

    python research/bench/scripts/run_guarded.py [--log FILE] -- python p1b_study.py main --stages ...

Loop: wait until the GPU is free (foreign SM activity, or less than --min-free-gib of free
device memory -> wait 30 min, re-check; blocked for more than 2 h -> exit 3 and report; with
--strict any foreign compute process counts, idle or not), then run the command.
The command yields by exiting with YIELD_RC (75) after it has saved its progress (e.g.
p1b_study.py when cobench raises GpuYield between two measurement points, or after a point
that foreign SM activity contaminated); its GPU memory is released with the process. The
wrapper then waits again and relaunches the command, which resumes from its next point.
Any other exit code ends the wrapper with that code.

The wrapper itself never touches CUDA: cobench/guard.py is loaded as a standalone module
(nvidia-smi only), so waiting holds no GPU memory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
YIELD_RC = 75
BLOCKED_RC = 3


def load_guard():
    path = os.path.normpath(os.path.join(HERE, "..", "cobench", "guard.py"))
    spec = importlib.util.spec_from_file_location("cobench_guard_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="JSON-lines record of waits / runs / yields")
    ap.add_argument("--min-free-gib", type=float, default=8.0,
                    help="also wait while less device memory is free (a foreign job may grow)")
    ap.add_argument("--strict", action="store_true", help="any foreign compute process occupies the GPU")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    guard = load_guard()
    # the default policy (foreign SM activity), waiting in-process here, plus free memory
    pol = (guard.GuardPolicy.strict if a.strict else guard.GuardPolicy)(min_free_mib=int(a.min_free_gib * 1024))

    def rec(**kw):
        kw["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[run_guarded] {kw}", flush=True)
        if a.log:
            with open(a.log, "a") as f:
                f.write(json.dumps(kw) + "\n")

    g = guard.GpuGuard()
    try:
        while True:
            try:
                waited = g.wait_until_free(policy=pol, log=lambda m: print(m, flush=True))
            except guard.GpuBusy as e:
                rec(event="blocked", detail=str(e)[:400])
                print(f"[run_guarded] blocked for more than {pol.max_wait_s / 3600:.1f} h: stopping (report back)",
                      flush=True)
                return BLOCKED_RC
            rec(event="launch", waited_s=round(waited, 1), cmd=" ".join(cmd))
            t0 = time.time()
            rc = subprocess.call(cmd)
            rec(event="exit", rc=rc, wall_s=round(time.time() - t0, 1))
            if rc != YIELD_RC:
                return rc
            rec(event="yield")
    finally:
        g.stop()


if __name__ == "__main__":
    sys.exit(main())
