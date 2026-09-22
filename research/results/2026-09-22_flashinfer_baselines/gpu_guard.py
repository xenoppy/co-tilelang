"""GPU-sharing guard for timing runs (shared by the scripts in this directory).

* ``wait_until_free``: before a timed batch, poll for foreign compute processes (any PID
  other than ours) every ``poll_s`` seconds; give up after ``max_wait_s``.
* ``Watchdog``: while a timed batch runs, a thread polls NVML every 0.25 s; any foreign
  compute process seen marks the batch contaminated.
* ``guarded(fn)``: wait -> run under the watchdog -> rerun if contaminated.
"""
from __future__ import annotations

import os
import threading
import time

import pynvml

_INIT = False


def _handle(index: int = 0):
    global _INIT
    if not _INIT:
        pynvml.nvmlInit()
        _INIT = True
    return pynvml.nvmlDeviceGetHandleByIndex(index)


def foreign_pids(index: int = 0) -> list[int]:
    me = os.getpid()
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(_handle(index))
    return sorted({p.pid for p in procs if p.pid != me})


def _cmd(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:160]
    except OSError:
        return "?"


class GpuBusy(RuntimeError):
    pass


def wait_until_free(max_wait_s: float = 900.0, poll_s: float = 150.0, log=print) -> float:
    """Return the seconds waited; raise GpuBusy if foreign processes persist past max_wait_s."""
    t0 = time.time()
    while True:
        p = foreign_pids()
        if not p:
            return time.time() - t0
        waited = time.time() - t0
        if waited >= max_wait_s:
            raise GpuBusy(f"GPU still busy after {waited:.0f}s: " + "; ".join(f"{q}: {_cmd(q)}" for q in p))
        log(f"[gpu_guard] foreign GPU process(es) {[(q, _cmd(q)) for q in p]}; waiting {poll_s:.0f}s "
            f"(waited {waited:.0f}s of {max_wait_s:.0f}s)", flush=True)
        time.sleep(poll_s)


class Watchdog:
    def __init__(self, period_s: float = 0.25):
        self.period_s = period_s
        self.seen: dict[int, str] = {}
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while not self._stop.is_set():
            for p in foreign_pids():
                self.seen.setdefault(p, _cmd(p))
            self._stop.wait(self.period_s)

    def __enter__(self):
        for p in foreign_pids():
            self.seen.setdefault(p, _cmd(p))
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        for p in foreign_pids():
            self.seen.setdefault(p, _cmd(p))
        return False


def guarded(fn, label: str = "", max_attempts: int = 4, log=print, **wait_kw):
    """Run fn() with no foreign GPU process present before, during or after. Returns
    (result, info) where info records waits and contaminated attempts."""
    info = {"attempts": 0, "waited_s": 0.0, "contaminated": []}
    for _ in range(max_attempts):
        info["waited_s"] += wait_until_free(log=log, **wait_kw)
        info["attempts"] += 1
        with Watchdog() as w:
            r = fn()
        if not w.seen:
            return r, info
        info["contaminated"].append({str(k): v for k, v in w.seen.items()})
        log(f"[gpu_guard] {label}: foreign process during timing {w.seen}; re-measuring", flush=True)
    raise GpuBusy(f"{label}: contaminated in all {max_attempts} attempts: {info['contaminated']}")
