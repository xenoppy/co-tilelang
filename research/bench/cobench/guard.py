"""GPU-sharing guard based on `nvidia-smi pmon -s u` and the compute-process list.

Rule (research/rules.md 7: query before launching; if occupied, suspend 30 minutes, then
query again), as applied since 2026-09-23 (``GuardPolicy`` defaults; clarified by the user
the same day: a foreign job that only holds GPU memory must not block us):

* **occupied** = a process that is not ours (not in this process tree, not owned by this
  user) shows SM activity in pmon (SM% > 0 within ``quiet_s``); processes that only hold
  memory (SM% '-' or 0) do not count. Optionally (``min_free_mib`` > 0) the GPU also counts
  as occupied while less device memory than that is free (a foreign job can grow to ~95 GB);
* ``GuardPolicy.strict()``: any foreign compute process occupies the GPU, idle or not (the
  stricter reading used between 06:19 and 07:37 on 2026-09-23);
* before launching GPU work (a job, a stage, a measurement point): if occupied, wait
  ``poll_s`` = 30 min, then re-check (no fast polling);
* while a job runs: when the GPU becomes occupied, finish the current measurement point,
  then yield: stop launching GPU work, release GPU memory where feasible, wait 30 min,
  re-check, resume from the next point. With ``yield_to_caller=True`` the guard raises
  ``GpuYield`` instead of waiting in-process, so the caller can exit and release its memory
  (research/bench/scripts/run_guarded.py waits and relaunches the resumable study);
* blocked for more than ``max_wait_s`` = 2 h: ``GpuBusy`` (stop and report instead of waiting);
* points measured while a foreign process showed SM activity are re-measured
  (contamination is judged on SM activity of processes outside this process tree).

``GuardPolicy.legacy()`` is the 2026-09-22 rule: occupied only by SM activity outside this
process tree (same user included), re-checked every 150 s, 6 h deadline.

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
import dataclasses
import getpass
import os
import pwd
import subprocess
import threading
import time
from dataclasses import dataclass

PMON_CMD = ("nvidia-smi", "pmon", "-s", "u", "-d", "1")
APPS_CMD = ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
SAMPLE_S = 1.0          # pmon interval
SLACK_S = 1.5           # a pmon line at t covers ~(t - 1 s, t]; widen windows by this much


class GpuBusy(RuntimeError):
    """Raised when the GPU stays occupied by a foreign process past a deadline."""


class GpuYield(Exception):
    """The GPU is occupied (policy.yield_to_caller): stop launching GPU work, release GPU
    memory, wait and resume from the next measurement point. Not a RuntimeError, so generic
    retry handlers do not swallow it. ``decision`` is the occupancy() record."""

    def __init__(self, msg: str, decision: dict | None = None):
        super().__init__(msg)
        self.decision = decision or {}


@dataclass(frozen=True)
class GuardPolicy:
    """When the GPU counts as occupied and how to wait (see the module doc)."""
    foreign_process_occupies: bool = False  # strict(): any foreign compute process, idle or not
    foreign_sm_occupies: bool = True        # foreign SM% > 0 in pmon within quiet_s
    min_free_mib: int = 0                   # > 0: also occupied while less memory is free
    same_user_is_own: bool = True           # processes of this user count as ours
    poll_s: float = 1800.0                  # re-check interval while occupied
    max_wait_s: float | None = 7200.0       # blocked longer -> GpuBusy (None: wait forever)
    quiet_s: float = 5.0                    # pmon window for the SM-activity check
    yield_to_caller: bool = False           # raise GpuYield instead of waiting in-process

    @classmethod
    def strict(cls, **kw) -> "GuardPolicy":
        """Any foreign compute process occupies the GPU, idle or not."""
        return cls(**{"foreign_process_occupies": True, **kw})

    @classmethod
    def legacy(cls, **kw) -> "GuardPolicy":
        """The 2026-09-22 rule: occupied only by foreign SM activity; 150 s polls; 6 h."""
        d = dict(foreign_process_occupies=False, same_user_is_own=False, poll_s=150.0, max_wait_s=6 * 3600.0)
        d.update(kw)
        return cls(**d)


_POLICY = GuardPolicy()


def get_policy() -> GuardPolicy:
    return _POLICY


def set_policy(policy: GuardPolicy | None = None, **changes) -> GuardPolicy:
    """Set the process-wide policy (used by every bench and wait_until_free); returns the
    previous one. set_policy(yield_to_caller=True) changes single fields."""
    global _POLICY
    prev = _POLICY
    _POLICY = dataclasses.replace(policy or _POLICY, **changes)
    return prev


@dataclass(frozen=True)
class ComputeProc:
    pid: int
    user: str | None        # None: owner unknown (pid not visible in this PID namespace)
    used_mib: int | None
    cmd: str = ""


def _user_of(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Uid:"):
                    uid = int(line.split()[1])
                    try:
                        return pwd.getpwuid(uid).pw_name
                    except KeyError:
                        return str(uid)
    except (OSError, ValueError):
        return None
    return None


def list_compute_procs(gpu: int | None = None) -> list[ComputeProc]:
    """Compute processes on the GPU (nvidia-smi --query-compute-apps), with their owner."""
    cmd = list(APPS_CMD) + (["-i", str(gpu)] if gpu is not None else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.strip()[:300]}")
    out = []
    for line in r.stdout.splitlines():
        f = [x.strip() for x in line.split(",")]
        if not f or not f[0].isdigit():
            continue
        pid = int(f[0])
        mem = int(f[1]) if len(f) > 1 and f[1].isdigit() else None
        out.append(ComputeProc(pid, _user_of(pid), mem, _cmdline(pid)))
    return out


def free_mib(gpu: int | None = None) -> int:
    """Free device memory (MiB) from nvidia-smi (no CUDA context needed)."""
    cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    cmd += ["-i", str(gpu)] if gpu is not None else []
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.strip()[:300]}")
    return int(r.stdout.split()[0])


def occupancy(procs: list[ComputeProc], active: list[tuple[int, str]], *, is_own, own_user: str | None,
              policy: GuardPolicy, user_of=None, free: int | None = None) -> dict:
    """Pure occupancy decision (unit-tested with mocked inputs).

    procs      compute processes on the GPU (list_compute_procs)
    active     (pid, cmd) of processes outside this process tree with SM% > 0 in pmon within
               policy.quiet_s (GpuGuard.busy)
    is_own     pid -> True for this process tree
    own_user   this user's name; with policy.same_user_is_own its processes count as ours
    user_of    pid -> owner (for the pmon pids; default: /proc)
    free       free device memory in MiB (checked against policy.min_free_mib)
    Returns {"occupied", "reasons", "foreign_procs", "foreign_active", "free_mib"}."""
    user_of = user_of or _user_of

    def foreign(pid: int, user: str | None) -> bool:
        if is_own(pid):
            return False
        if policy.same_user_is_own and user is not None and own_user is not None and user == own_user:
            return False
        return True

    fp = [p for p in procs if foreign(p.pid, p.user)]
    fa = [(pid, cmd) for pid, cmd in active if foreign(pid, user_of(pid))]
    reasons = []
    if policy.foreign_process_occupies and fp:
        reasons.append("foreign compute process")
    if policy.foreign_sm_occupies and fa:
        reasons.append("foreign SM activity")
    if policy.min_free_mib > 0 and free is not None and free < policy.min_free_mib:
        reasons.append(f"only {free} MiB free < {policy.min_free_mib} MiB")
    return {"occupied": bool(reasons), "reasons": reasons,
            "foreign_procs": [(p.pid, p.user, p.used_mib, p.cmd[:80]) for p in fp],
            "foreign_active": [(pid, cmd[:80]) for pid, cmd in fa], "free_mib": free}


def wait_loop(probe, policy: GuardPolicy, *, sleep=time.sleep, now=time.time, log=None) -> float:
    """Block until probe() reports a free GPU; returns the seconds waited.

    probe() -> occupancy() record. While occupied: GpuYield if policy.yield_to_caller; GpuBusy
    once blocked >= policy.max_wait_s; otherwise sleep policy.poll_s and re-check."""
    say = log or (lambda m: print(m, flush=True))
    t0 = now()
    while True:
        dec = probe()
        if not dec["occupied"]:
            return now() - t0
        waited = now() - t0
        what = "; ".join(dec["reasons"]) + f": procs {dec['foreign_procs']} active {dec['foreign_active']}"
        if policy.yield_to_caller:
            raise GpuYield(f"GPU occupied ({what}); yielding", dec)
        if policy.max_wait_s is not None and waited >= policy.max_wait_s:
            raise GpuBusy(f"GPU still occupied after {waited:.0f}s ({what})")
        dt = policy.poll_s if policy.max_wait_s is None else min(policy.poll_s, max(1.0, policy.max_wait_s - waited))
        say(f"[guard] GPU occupied ({what}); waiting {dt:.0f}s (waited {waited:.0f}s)")
        sleep(dt)


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
        self.user = getpass.getuser()
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

    def occupancy(self, policy: GuardPolicy | None = None) -> dict:
        """occupancy() of the GPU right now (compute-process list + recent pmon activity)."""
        policy = policy or get_policy()
        # need quiet_s of pmon coverage since the guard started, and a recent line (pmon
        # prints every ~1 s; do not idle the GPU waiting for a fresh one)
        need = max(self.t_start + policy.quiet_s, time.time() - SAMPLE_S - 0.5)
        if not self.wait_covered(need, timeout_s=policy.quiet_s + 10):
            self._check_alive()
            raise RuntimeError("GpuGuard: nvidia-smi pmon produced no samples")
        procs = list_compute_procs(self.gpu) if policy.foreign_process_occupies else []
        if any(p.user is None for p in procs):
            # a listed pid without a /proc entry is either in another PID namespace (foreign)
            # or has just exited (nvidia-smi lags): confirm with a second query
            time.sleep(2.0)
            again = {p.pid for p in list_compute_procs(self.gpu)}
            procs = [p for p in procs if p.user is not None or (p.pid in again and _user_of(p.pid) is None)]
        free = free_mib(self.gpu) if policy.min_free_mib > 0 else None
        return occupancy(procs, self.busy(policy.quiet_s), is_own=self.is_own, own_user=self.user, policy=policy,
                         free=free)

    def wait_until_free(self, *, policy: GuardPolicy | None = None, quiet_s: float | None = None,
                        poll_s: float | None = None, max_wait_s: float | None | str = "policy", log=None) -> float:
        """Block while the GPU is occupied (see GuardPolicy; default: the process-wide policy,
        get_policy()); the keyword arguments override single policy fields. Returns the
        seconds waited; raises GpuBusy past max_wait_s and GpuYield if the policy yields."""
        pol = policy or get_policy()
        ch = {k: v for k, v in (("quiet_s", quiet_s), ("poll_s", poll_s)) if v is not None}
        if max_wait_s != "policy":
            ch["max_wait_s"] = max_wait_s
        pol = dataclasses.replace(pol, **ch) if ch else pol
        return wait_loop(lambda: self.occupancy(pol), pol, log=log)

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
    """Block while the GPU is occupied (see GpuGuard.wait_until_free, GuardPolicy)."""
    return get_guard().wait_until_free(**kw)


def occupied(policy: GuardPolicy | None = None) -> dict:
    """occupancy() record of the GPU right now, with the shared guard."""
    return get_guard().occupancy(policy)


def foreign_activity(t0: float, t1: float, **kw) -> dict:
    """Judge [t0, t1] with the shared guard (see GpuGuard.check)."""
    return get_guard().check(t0, t1, **kw)
