"""GPU-sharing guard based on `nvidia-smi pmon -s u`.

Rule (research/rules.md 7, clarified 2026-09-22): the GPU counts as *occupied* only while a
foreign process shows SM utilization > 0 in `nvidia-smi pmon -s u`. A process that merely
holds a CUDA context (pmon prints '-' or 0 for its sm%) does not count. Points measured
while a foreign process was active must be re-measured.

``GpuGuard`` streams `nvidia-smi pmon -s u -d 1` in a background thread (one line per
compute process per ~1 s interval: ``gpu pid type sm% mem% ... command``) and keeps

* every foreign sample with a numeric sm% > 0 (``activity``), and
* the host time of the latest pmon line (``covered_until``): a window [t0, t1] can only be
  judged once pmon has reported past t1 (a line printed at t describes roughly (t-1 s, t]).

"Foreign" excludes this process and its whole process tree (ancestors and descendants), so a
parent script that spawns measurement subprocesses and the subprocesses themselves do not
block each other.

Module-level helpers use one shared guard per process (started on first use)::

    import cobench as cb
    cb.wait_until_free()                         # block while foreign SM activity is seen
    t0 = time.time(); ...measure...; t1 = time.time()
    act = cb.foreign_activity(t0, t1)            # [] -> the window was clean (waits for pmon)
"""
from __future__ import annotations

import atexit
import os
import subprocess
import threading
import time
from dataclasses import dataclass

PMON_CMD = ("nvidia-smi", "pmon", "-s", "u", "-d", "1")
SAMPLE_S = 1.0          # pmon interval
SLACK_S = 1.5           # a pmon line at t covers ~(t - 1 s, t]; widen windows by this much


class GpuBusy(RuntimeError):
    """Raised when the GPU stays occupied by a foreign process past a deadline."""


@dataclass(frozen=True)
class Sample:
    t: float        # host time.time() when the pmon line was read
    pid: int
    sm: int         # SM utilization in % (> 0 for recorded activity)
    cmd: str


def _ppid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
        # comm may contain spaces/parentheses: the fields after the last ')' are fixed
        return int(s[s.rindex(")") + 2:].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _ancestors(pid: int) -> list[int]:
    out, seen = [], set()
    p = _ppid(pid)
    while p and p > 1 and p not in seen:
        seen.add(p)
        out.append(p)
        p = _ppid(p)
    return out


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:120]
    except OSError:
        return "?"


class GpuGuard:
    """Background pmon sampler (see module doc). Use as a context manager or call stop()."""

    def __init__(self, own_pids: tuple[int, ...] = (), gpu: int | None = None):
        self.me = os.getpid()
        self.own_extra = set(int(p) for p in own_pids)
        self.my_ancestors = set(_ancestors(self.me))
        self.gpu = gpu
        self.t_start = time.time()
        self.covered_until = 0.0          # host time of the latest pmon data line
        self.n_lines = 0
        self.activity: list[Sample] = []
        self.present: dict[int, list] = {}   # foreign pid -> [first_t, last_t, cmd]
        self._own_cache: dict[int, bool] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        cmd = list(PMON_CMD) + (["-i", str(gpu)] if gpu is not None else [])
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, bufsize=1)
        except OSError as e:
            raise RuntimeError(f"cannot start {' '.join(cmd)}: {e}") from e
        self._err: str | None = None
        self._t = threading.Thread(target=self._run, daemon=True, name="cobench-pmon")
        self._t.start()

    # -- sampling -------------------------------------------------------------
    def is_own(self, pid: int) -> bool:
        if pid == self.me or pid in self.own_extra or pid in self.my_ancestors:
            return True
        c = self._own_cache.get(pid)
        if c is None:
            c = self.me in _ancestors(pid)   # a descendant of this process
            self._own_cache[pid] = c
        return c

    def _run(self):
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 4:
                continue
            t = time.time()
            with self._lock:
                self.n_lines += 1
                self.covered_until = t
                if not f[1].isdigit():      # "-": no compute process at all this interval
                    continue
                pid = int(f[1])
                if self.is_own(pid):
                    continue
                cmd = " ".join(f[9:]) if len(f) > 9 else ""
                pr = self.present.get(pid)
                if pr is None:
                    self.present[pid] = [t, t, _cmdline(pid) or cmd]
                else:
                    pr[1] = t
                sm = f[3]
                if sm.isdigit() and int(sm) > 0:
                    self.activity.append(Sample(t, pid, int(sm), self.present[pid][2]))
        rc = self._proc.poll()
        if not self._stop.is_set():
            err = self._proc.stderr.read() if self._proc.stderr else ""
            self._err = f"nvidia-smi pmon exited (rc={rc}): {err.strip()[:300]}"

    def _check_alive(self):
        if self._err:
            raise RuntimeError(f"GpuGuard: {self._err}")

    # -- queries --------------------------------------------------------------
    def wait_covered(self, t: float, timeout_s: float = 5.0) -> bool:
        """Wait until pmon has reported past host time t. False on timeout."""
        deadline = time.time() + timeout_s
        while self.covered_until < t:
            self._check_alive()
            if time.time() > deadline:
                return False
            time.sleep(0.05)
        return True

    def activity_between(self, t0: float, t1: float, slack: float = SLACK_S) -> list[Sample]:
        """Foreign samples with SM% > 0 whose ~1 s interval overlaps [t0, t1] (no waiting)."""
        with self._lock:
            return [s for s in self.activity if t0 - slack <= s.t <= t1 + slack]

    def check(self, t0: float, t1: float, *, wait: bool = True, slack: float = SLACK_S,
              timeout_s: float = 6.0) -> dict:
        """Judge the window [t0, t1]: {"clean": bool, "complete": bool, "active": [...]}.
        With wait=True, first waits until pmon has reported past t1 + slack."""
        complete = self.wait_covered(t1 + slack, timeout_s) if wait else self.covered_until >= t1 + slack
        act = self.activity_between(t0, t1, slack)
        return {"clean": not act, "complete": bool(complete), "window": [t0, t1],
                "active": sorted({(s.pid, s.cmd) for s in act}),
                "max_sm": max((s.sm for s in act), default=0)}

    def busy(self, quiet_s: float = 5.0) -> list[tuple[int, str]]:
        """Foreign (pid, cmd) with SM% > 0 in the last quiet_s seconds of pmon data."""
        now = time.time()
        with self._lock:
            return sorted({(s.pid, s.cmd) for s in self.activity if s.t >= now - quiet_s})

    def wait_until_free(self, *, quiet_s: float = 5.0, poll_s: float = 150.0,
                        max_wait_s: float | None = 6 * 3600, log=None) -> float:
        """Block until no foreign process has shown SM% > 0 for quiet_s seconds of pmon data;
        re-check every poll_s while busy. Returns the seconds waited; raises GpuBusy after
        max_wait_s (None = wait forever)."""
        t0 = time.time()
        # need quiet_s of pmon coverage since the guard started, and a recent line (pmon
        # prints every ~1 s; do not idle the GPU waiting for a fresh one)
        need = max(self.t_start + quiet_s, t0 - SAMPLE_S - 0.5)
        if not self.wait_covered(need, timeout_s=quiet_s + 10):
            self._check_alive()
            raise RuntimeError("GpuGuard: nvidia-smi pmon produced no samples")
        while True:
            act = self.busy(quiet_s)
            if not act:
                return time.time() - t0
            waited = time.time() - t0
            if max_wait_s is not None and waited >= max_wait_s:
                raise GpuBusy(f"GPU still occupied after {waited:.0f}s by {act}")
            msg = f"[guard] foreign SM activity from {act}; waiting {poll_s:.0f}s (waited {waited:.0f}s)"
            if log is None:
                print(msg, flush=True)
            else:
                log(msg)
            time.sleep(poll_s if max_wait_s is None else min(poll_s, max(1.0, max_wait_s - waited)))

    def summary(self, t0: float = 0.0, t1: float = float("inf")) -> dict:
        with self._lock:
            act = [(round(s.t, 1), s.pid, s.sm) for s in self.activity if t0 <= s.t <= t1]
            pres = {str(p): [round(a, 1), round(b, 1), c] for p, (a, b, c) in self.present.items()
                    if b >= t0 and a <= t1}
        return {"n_active_samples": len(act), "active": act[:200], "present": pres,
                "n_lines": self.n_lines}

    def stop(self):
        self._stop.set()
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best effort at shutdown
            pass
        self._t.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


_GUARD: GpuGuard | None = None
_GLOCK = threading.Lock()


def get_guard() -> GpuGuard:
    """The process-wide guard (started on first use, stopped at exit)."""
    global _GUARD
    with _GLOCK:
        if _GUARD is None or _GUARD._err:
            _GUARD = GpuGuard()
            atexit.register(_GUARD.stop)
        return _GUARD


def wait_until_free(**kw) -> float:
    """Block while a foreign process shows SM% > 0 (see GpuGuard.wait_until_free)."""
    return get_guard().wait_until_free(**kw)


def foreign_activity(t0: float, t1: float, **kw) -> dict:
    """Judge [t0, t1] with the shared guard (see GpuGuard.check)."""
    return get_guard().check(t0, t1, **kw)
