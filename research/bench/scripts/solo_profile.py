#!/usr/bin/env python
"""P1-S solo profiling of the op library (cotile.ops) at the plan's shapes.

For every op x shape (research/plan.md §2.1) and every config of `op.configs(shape)`:
  * grid build on the full GPU, on 94 SMs and on 48 SMs (green contexts with
    IGNORE_SM_COSCHEDULING, contiguous SMs [0, n); a 47-SM request gives 48),
  * persistent build on the full GPU with num_ctas = min(CTAs/SM x 188, tiles), where
    CTAs/SM comes from the compiled persistent kernel's resource signature,
  * the reference libraries (cuBLAS via torch.matmul; FlashInfer decode, CUDA-core and
    tensor-core paths; torch F.rms_norm and FlashInfer rmsnorm) at every budget.

Protocol (research/bench/README.md, cobench "flush" mode):
  per rep: 256 MB (2 x L2) write on a full-GPU side stream F -> host gate -> start event
  on the op stream S -> op -> end event. The rep loop reuses cobench's own flush buffer,
  side stream, host gate and clock probe (cobench.timing internals); on top of cobench it
    - keeps the GPU loaded between points: after the timed reps it enqueues an untimed
      "tail" of identical reps (~tail_s) that runs while the host post-processes, and all
      kernels of all shapes are compiled/loaded before the global warmup;
    - uses a long global warmup (until the GPU temperature is flat) and a shorter
      per-point warmup (`--warm-s`, justified by the `warmup-study` subcommand and by the
      `recheck` subcommand, which re-measures points at the end of the sweep);
    - sizes the timed window to >= `--window-s` (>= 50 reps) so that it holds enough 20 ms
      NVML power samples; power = mean of the samples inside the timed window;
    - records per point: median/p10/p90/mean/CV/min/max, SM clock per rep (clock probe;
      median, min, max, cycles = us x MHz), the drift diagnostic `half_ratio` (median of
      the second half of the reps / first half), average board power, temperature and the
      fraction of NVML polls with the SwPowerCap reason.
  Energy per call (caveat: NVML board power, 20 ms samples, flush-mode duty cycle):
    E_naive  = P_avg x t_median                      (the task's definition)
    E_rep    = P_avg x rep period                    (energy of a whole flush+gate+op rep)
    E_kernel = E_rep - E_rep(flush-only rep)         (energy the op adds; the flush-only
               baseline is measured at the start of every (op, shape) batch)
  Points with CV > 2% are re-measured (`--attempts`, default 2; the lowest-CV attempt is kept;
  short flush-mode points have a ~1-2 us launch-jitter floor, so `spread` = (p90-p10)/median
  is stored as a robust alternative).

GPU sharing (rule as clarified by the coordinator): the GPU counts as occupied only while a
foreign process shows SM utilization > 0 (`nvidia-smi pmon -s u`, streamed in a background
thread, ~1 s samples); an idle process that only holds a CUDA context does not count. Before
each batch the script waits (poll every 150 s) while there is foreign SM activity; every point
whose [warmup start - 2 s, end + 2 s] window contains foreign SM activity is marked
`contaminated` and re-measured after the batch (the batch meta records the foreign activity
and the number of re-measured points).

Results: one JSON per (op, shape) in <out>/points/, written after every budget sweep;
re-running resumes (points already present and clean are skipped).

Usage (source research/env.sh first):
  python research/bench/scripts/solo_profile.py compile            # CPU: build + compile all kernels
  python research/bench/scripts/solo_profile.py run                # GPU sweep (resumable)
  python research/bench/scripts/solo_profile.py warmup-study       # per-point warmup justification
  python research/bench/scripts/solo_profile.py recheck            # re-measure points at the end
  python research/bench/scripts/solo_profile.py energy-check       # flush-subtraction vs graph mode
  python research/bench/scripts/solo_profile.py persistent-modes   # grid vs persistent: flush/read-flush/hot/graph
  python research/bench/scripts/solo_profile.py ref-extra          # cuBLAS without bf16 reduced-precision reduction
  python research/bench/scripts/solo_profile.py flush-bias         # write-flush vs read-flush, solo best per shape
  options: --ops gemm,gqa_decode --shapes M4096_N4096_K4096,... --limit N (configs per shape)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, BENCH)
sys.path.insert(0, REPO)

import cobench as cb  # noqa: E402
from cobench import timing as cbt  # noqa: E402
from cobench.clock import ClockProbe  # noqa: E402
from cobench.nvml import NvmlDevice  # noqa: E402

from cotile import catalog, resources  # noqa: E402
from cotile.device import DEFAULT_DEVICE  # noqa: E402
from cotile.kernel import Runner, compile_specs  # noqa: E402
from cotile.ops import gemm, gqa_decode, rmsnorm  # noqa: E402

OPS = {m.NAME: m for m in (gemm, gqa_decode, rmsnorm)}
FULL = DEFAULT_DEVICE.num_sms
BUDGET_REQUESTS = (94, 47)  # green-context requests; 47 -> 48 SMs (2-SM granularity)
SWPOWERCAP = 0x4
CV_TARGET = 0.02


def log(*a, **kw):
    kw.pop("flush", None)
    print(time.strftime("%H:%M:%S"), *a, flush=True, **kw)


def r5(x):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.5g}")


def plan_shapes() -> dict:
    return {
        "gemm": [gemm.GemmShape(M, N, K) for M in (2048, 4096, 8192) for (N, K) in ((4096, 4096), (14336, 4096), (4096, 14336))],
        "gqa_decode": [gqa_decode.DecodeShape(batch=B, seqlen=S) for B in (16, 32, 64, 128) for S in (2048, 8192, 32768)],
        "rmsnorm": [rmsnorm.RMSNormShape(tokens=T, hidden=H) for T in (4096, 16384, 65536) for H in (4096, 8192)],
    }


def shape_dict(shape) -> dict:
    return {k: v for k, v in shape.__dict__.items()}


def file_sha(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return None


def env_meta() -> dict:
    import tilelang

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    return {
        "tilelang": tilelang.__version__,
        "torch": torch.__version__,
        "git_head": git("rev-parse", "--short", "HEAD"),
        "git_dirty_src": git("diff", "--stat", "--", "src", "cotile", "tilelang"),
        "sha_cotile_kernel_py": file_sha(os.path.join(REPO, "cotile", "kernel.py")),
        "sha_cotile_ops": {m: file_sha(os.path.join(REPO, "cotile", "ops", f"{m}.py")) for m in OPS},
        "sha_libtilelang": file_sha(os.path.join(REPO, "build", "lib", "libtilelang.so")),
        "host": os.uname().nodename,
    }


# ----------------------------------------------------------------------------------
# kernels: build, compile (kernel cache), signatures, persistent occupancy (two passes)
# ----------------------------------------------------------------------------------

SIG_KEYS = ("threads", "regs", "local_bytes", "smem_static", "smem_dynamic", "smem_total", "num_barriers",
            "ctas_per_sm", "limit_by", "grid", "num_tiles", "numerics")


def compact_sig(sig: dict) -> dict:
    d = {k: sig[k] for k in SIG_KEYS if k in sig}
    d["smem"] = d.pop("smem_total")
    d["ctas_sm"] = d.pop("ctas_per_sm")
    return d


class Batch:
    """All kernels of one (op, shape)."""

    def __init__(self, op, shape, limit=None):
        self.op, self.shape = op, shape
        self.tag = catalog.shape_tag(op.NAME, shape)
        cfgs = op.configs(shape)
        self.cfgs = cfgs[:limit] if limit else cfgs
        self.ctag = {op.cfg_tag(c): c for c in self.cfgs}
        self.grid, self.pers, self.p1 = {}, {}, {}
        self.sig: dict = {}  # tag -> {"grid": sig, "persistent": sig}
        self.pinfo: dict = {}

    @property
    def name(self):
        return f"{self.op.NAME}__{self.tag}"


def _sigs(specs, workers=32):
    specs = [s for s in specs if s.kernel is not None]
    with concurrent.futures.ThreadPoolExecutor(min(workers, max(1, len(specs)))) as ex:
        for s, sig in zip(specs, ex.map(resources.signature, specs)):
            s.extra["signature"] = sig


def prepare(batches: list[Batch], workers: int = 64) -> dict:
    """Build + compile every grid and persistent kernel of `batches` (kernel cache hits on
    re-runs). Persistent num_ctas = min(CTAs/SM(persistent kernel) x 188, tiles): pass 1
    compiles with min(188, tiles) CTAs to read the persistent kernel's occupancy, pass 2
    rebuilds with the final count when it differs (and checks the occupancy is unchanged)."""
    t0 = time.time()
    specs1 = []
    for b in batches:
        for tag, cfg in b.ctag.items():
            g = b.op.build_grid(b.shape, cfg)
            p = b.op.build_persistent(b.shape, cfg, min(FULL, g.num_tiles))
            b.grid[tag], b.p1[tag] = g, p
            specs1 += [g, p]
    t_build1 = time.time() - t0
    st1 = compile_specs(specs1, num_workers=workers)
    t1 = time.time()
    _sigs(specs1)
    t_sig1 = time.time() - t1
    specs2 = []
    for b in batches:
        for tag, cfg in b.ctag.items():
            p = b.p1[tag]
            sig = p.extra.get("signature")
            if sig is None:
                b.pers[tag] = p
                b.pinfo[tag] = {"error": p.compile_error}
                continue
            k = sig["ctas_per_sm"]
            n = min(k * FULL, p.num_tiles) if k >= 1 else None
            b.pinfo[tag] = {"k_pass1": k, "num_ctas": n}
            if n is None or n == p.grid:
                b.pers[tag] = p
            else:
                q = b.op.build_persistent(b.shape, cfg, n)
                b.pers[tag] = q
                specs2.append(q)
    t2 = time.time()
    st2 = compile_specs(specs2, num_workers=workers) if specs2 else {"wall_s": 0.0, "n_ok": 0, "n_fail": 0}
    _sigs(specs2)
    # Fixed point: a rebuilt kernel can come out with a different occupancy (the grid-stride
    # trip count changes codegen, e.g. registers); if fewer CTAs/SM fit than were launched,
    # rebuild with the new count (at most 3 more rounds).
    n_extra = 0
    for _ in range(3):
        again = []
        for b in batches:
            for tag, cfg in b.ctag.items():
                q = b.pers[tag]
                sig = q.extra.get("signature")
                if sig is None or q is b.p1[tag]:
                    continue
                k2 = sig["ctas_per_sm"]
                if k2 >= 1 and k2 * FULL < q.grid:
                    n = min(k2 * FULL, q.num_tiles)
                    r = b.op.build_persistent(b.shape, cfg, n)
                    b.pers[tag] = r
                    b.pinfo[tag].setdefault("rebuilds", []).append({"launched": q.grid, "k": k2})
                    again.append(r)
        if not again:
            break
        n_extra += len(again)
        compile_specs(again, num_workers=workers)
        _sigs(again)
    for b in batches:
        for tag in b.ctag:
            g, p = b.grid[tag], b.pers[tag]
            b.sig[tag] = {
                "grid": compact_sig(g.extra["signature"]) if "signature" in g.extra else {"error": g.compile_error},
                "persistent": compact_sig(p.extra["signature"]) if "signature" in p.extra else {"error": p.compile_error},
            }
            ps = b.sig[tag]["persistent"]
            if "ctas_sm" in ps:
                b.pinfo[tag]["k_final"] = ps["ctas_sm"]
                b.sig[tag]["persistent"]["num_ctas"] = p.grid
                b.sig[tag]["persistent"]["k_pass1"] = b.pinfo[tag]["k_pass1"]
    return {
        "n_kernels": len(specs1) + len(specs2) + n_extra,
        "n_pass2": len(specs2),
        "n_fixed_point_rebuilds": n_extra,
        "build_s": t_build1,
        "compile1": st1,
        "sig1_s": t_sig1,
        "compile2": st2,
        "pass2_total_s": time.time() - t2,
        "total_s": time.time() - t0,
    }


def selected_batches(args) -> list[Batch]:
    shapes = plan_shapes()
    out = []
    for opn in args.ops:
        for sh in shapes[opn]:
            tag = catalog.shape_tag(opn, sh)
            if args.shapes and tag not in args.shapes:
                continue
            out.append(Batch(OPS[opn], sh, args.limit))
    return out


def du_bytes(path: str) -> int:
    try:
        return int(subprocess.run(["du", "-sb", path], capture_output=True, text=True, check=True).stdout.split()[0])
    except Exception:  # noqa: BLE001
        return -1


def cmd_compile(args):
    import tilelang

    batches = selected_batches(args)
    cache = os.path.expanduser("~/.tilelang/cache")
    before = du_bytes(cache)
    st = prepare(batches, workers=args.workers)
    after = du_bytes(cache)
    n_fail = sum(1 for b in batches for t in b.ctag for k in ("grid", "persistent") if "error" in b.sig[t][k])
    pm = [(b.name, t, b.pinfo[t]) for b in batches for t in b.ctag if b.pinfo[t].get("k_final") not in (None, b.pinfo[t].get("k_pass1"))]
    rec = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": env_meta(),
        "tilelang_cache_ns": tilelang.__version__,
        "stats": st,
        "n_batches": len(batches),
        "n_configs": sum(len(b.ctag) for b in batches),
        "n_fail": n_fail,
        "persistent_occupancy_changed_in_pass2": pm,
        "cache_bytes_before": before,
        "cache_bytes_after": after,
        "per_batch": {b.name: len(b.ctag) for b in batches},
    }
    os.makedirs(args.out, exist_ok=True)
    name = "compile" if not args.shapes and not args.limit else "compile_partial"
    if sorted(args.ops) != sorted(OPS):
        name += "_" + "_".join(args.ops)
    path = os.path.join(args.out, f"{name}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"compile: {rec['n_configs']} configs, {st['n_kernels']} kernels, {st['total_s']:.0f}s, fail={n_fail}, "
        f"cache {before / 2**20:.0f} -> {after / 2**20:.0f} MB, occupancy changed in pass 2: {len(pm)}")


# ----------------------------------------------------------------------------------
# GPU-side monitors
# ----------------------------------------------------------------------------------


class PowerLog:
    """Background NVML reader: drains the driver's 20 ms power-sample buffer and polls
    temperature / clock-event reasons / SM clock (refreshed ~every 500 ms by the driver)."""

    def __init__(self, dev: int = 0, period_s: float = 0.1):
        self.nv = NvmlDevice(dev)
        self.period_s = period_s
        self.power = deque(maxlen=600_000)  # (t_s, W)
        self.state = deque(maxlen=600_000)  # (t_s, temp_c, reasons, sm_mhz)
        self._last = int(time.time() * 1e6)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="powerlog")
        self._t.start()

    def drain(self):
        with self._lock:
            new = self.nv.power_samples(self._last)
            for ts, w in new:
                self.power.append((ts * 1e-6, w))
            if new:
                self._last = new[-1][0]

    def _run(self):
        while not self._stop.is_set():
            try:
                self.drain()
                self.state.append((time.time(), self.nv.temperature_c(), self.nv.reasons(), self.nv.clock_mhz()))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.period_s)

    def window(self, t0: float, t1: float) -> dict:
        pw = [w for t, w in list(self.power) if t0 <= t <= t1]
        st = [s for s in list(self.state) if t0 <= s[0] <= t1]
        out = {"P_n": len(pw)}
        if pw:
            out["P_w"] = float(np.mean(pw))
        if st:
            out["temp_c"] = max(s[1] for s in st)
            out["pcap_frac"] = sum(1 for s in st if s[2] & SWPOWERCAP) / len(st)
        return out

    def latest_ts(self) -> float:
        return self.power[-1][0] if self.power else 0.0

    def timeline(self, bin_s: float = 5.0) -> list:
        """[(t, mean W, max temp, mean NVML MHz)] in bin_s bins (for the README)."""
        pw, st = list(self.power), list(self.state)
        if not pw:
            return []
        t0 = pw[0][0]
        bins: dict = {}
        for t, w in pw:
            bins.setdefault(int((t - t0) // bin_s), [[], [], []])[0].append(w)
        for t, temp, _, mhz in st:
            k = int((t - t0) // bin_s)
            if k in bins:
                bins[k][1].append(temp)
                bins[k][2].append(mhz)
        return [[round(t0 + k * bin_s, 1), r5(np.mean(v[0])), max(v[1]) if v[1] else None, r5(np.mean(v[2])) if v[2] else None]
                for k, v in sorted(bins.items())]

    def stop(self):
        self._stop.set()
        self._t.join()


class Watchdog:
    """Activity-based GPU-sharing guard (rule clarified by the coordinator, 2026-09-22 22:00):
    the GPU counts as occupied only while a *foreign* process shows nonzero SM utilization;
    an idle process that merely holds a CUDA context does not count.

    A background `nvidia-smi pmon -s u -d 1` stream gives one line per compute process per
    ~1 s interval: `gpu pid type sm% mem% ...` ('-' = no sample). Every line of a foreign
    PID with a numeric sm% > 0 is recorded as activity (t, pid, sm%); every foreign PID seen
    at all is recorded as presence (for the record only)."""

    def __init__(self):
        self.me = os.getpid()
        self.t_start = time.time()
        self.active: list = []  # (t, pid, sm%)
        self.present: dict = {}  # pid -> [first_t, last_t]
        self.n_lines = 0
        self._stop = threading.Event()
        self._proc = subprocess.Popen(["nvidia-smi", "pmon", "-s", "u", "-d", "1"], stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._t = threading.Thread(target=self._run, daemon=True, name="pmon-watch")
        self._t.start()

    def _run(self):
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 4 or not f[1].isdigit():
                continue
            self.n_lines += 1
            pid, sm, t = int(f[1]), f[3], time.time()
            if pid == self.me:
                continue
            pr = self.present.setdefault(pid, [t, t])
            pr[1] = t
            if sm.isdigit() and int(sm) > 0:
                self.active.append((t, pid, int(sm)))

    def between(self, t0: float, t1: float, slack: float = 2.0) -> list:
        """Foreign PIDs with SM% > 0 in [t0 - slack, t1 + slack] (pmon samples are ~1 s wide)."""
        return sorted({pid for t, pid, _ in list(self.active) if t0 - slack <= t <= t1 + slack})

    def wait_quiet(self, poll_s: float = 150.0, quiet_s: float = 5.0, max_wait_s: float = 6 * 3600) -> float:
        """Block while a foreign process showed SM% > 0 within the last `quiet_s` seconds;
        re-check every poll_s. Returns the seconds waited."""
        t0 = time.time()
        if t0 - self.t_start < quiet_s:  # need quiet_s of pmon samples first
            time.sleep(quiet_s - (t0 - self.t_start))
        while True:
            now = time.time()
            act = self.between(now - quiet_s, now, slack=0.0)
            if not act:
                return now - t0
            if now - t0 > max_wait_s:
                raise RuntimeError(f"GPU still busy after {now - t0:.0f}s: foreign SM activity from {act}")
            log(f"[guard] foreign SM activity from {act} "
                f"({', '.join(self._cmd(p) for p in act)}); waiting {poll_s:.0f}s (waited {now - t0:.0f}s)")
            time.sleep(poll_s)

    @staticmethod
    def _cmd(pid: int) -> str:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:100]
        except OSError:
            return "?"

    def summary(self, t0: float = 0.0, t1: float = float("inf")) -> dict:
        act = [(round(t, 1), pid, sm) for t, pid, sm in self.active if t0 <= t <= t1]
        pres = {str(pid): [round(a, 1), round(b, 1), self._cmd(pid)] for pid, (a, b) in self.present.items() if b >= t0 and a <= t1}
        return {"n_active_samples": len(act), "active": act[:200], "present": pres}

    def stop(self):
        self._stop.set()
        self._proc.terminate()
        self._t.join(timeout=5)


# ----------------------------------------------------------------------------------
# flush-mode meter
# ----------------------------------------------------------------------------------


class FlushMeter:
    """cobench flush-mode rep loop (same flush buffer, side stream, host gate and clock
    probe as cobench.bench(mode="flush")), with a sized timed window, NVML power over the
    timed window, and an untimed tail that keeps the GPU loaded between points."""

    def __init__(self, power: PowerLog, dev: int = 0, pool: int = 4096, chunk_s: float = 0.03):
        self.dev = dev
        self.power = power
        self.F = cbt._side_stream(dev, "flush")
        self.buf = cbt._flush_buffer(dev, 2 * cbt.l2_bytes(dev))
        self.g8 = cbt._gate(dev)
        self.ready = torch.cuda.Event()
        self.w0, self.w1 = torch.cuda.Event(), torch.cuda.Event()
        self.wend = [torch.cuda.Event(), torch.cuda.Event()]
        self.starts = cbt._timing_events(pool, self.F)
        self.ends = cbt._timing_events(pool, self.F)
        self.probe = ClockProbe(dev)
        self.chunk_s = chunk_s
        self.i = 0
        self.F.synchronize()

    # "write": cobench's flush (fill 2xL2, leaves dirty lines); "read": read 2xL2 (does NOT fully
    # evict on this GPU: small working sets stay partly L2-resident); "write+read": write then
    # read the same 2xL2 buffer (evicts, then the dirty lines are cleaned before the op starts)
    flush_kind = "write"

    def _rep(self, fn, S, e0, e1, gated: bool):
        F = self.F
        with torch.cuda.stream(F):
            if self.flush_kind in ("write", "write+read"):
                cbt._flush(self.buf, self.i)
            if self.flush_kind in ("read", "write+read"):
                self._rd = self.buf.amax()
            self.i += 1
            if gated:
                self.g8.wait(F)
            self.ready.record(F)
        S.wait_event(self.ready)
        e0.record(S)
        if fn is not None:
            with torch.cuda.stream(S):
                fn()
        e1.record(S)
        F.wait_event(e1)
        if gated:
            self.g8.release()

    def _warm(self, fn, S, seconds: float, min_reps: int, period_hint_s: float):
        """Double-buffered chunks (the GPU always has a queued chunk). Returns
        (reps, seconds, steady period estimate from host-side chunk completions)."""
        chunk = int(min(20, max(1, math.ceil(self.chunk_s / max(period_hint_s, 1e-5)))))
        p0 = time.perf_counter()
        n, k, marks = 0, 0, []
        while True:
            for j in range(chunk):
                self._rep(fn, S, self.w0, self.wend[k % 2] if j == chunk - 1 else self.w1, gated=True)
                n += 1
            if k >= 1:
                self.wend[(k - 1) % 2].synchronize()
                marks.append((time.perf_counter(), n - chunk))
            k += 1
            el = time.perf_counter() - p0
            if n >= min_reps and el >= seconds and len(marks) >= 3:
                break
        self.wend[(k - 1) % 2].synchronize()
        marks.append((time.perf_counter(), n))
        m = marks[-4:]
        per = (m[-1][0] - m[0][0]) / max(1, m[-1][1] - m[0][1])
        return n, time.perf_counter() - p0, per

    def warm(self, fn, S, seconds: float, period_hint_s: float = 1e-3):
        return self._warm(fn, S, seconds, 1, period_hint_s)

    def measure(self, fn, S, *, warm_s: float, window_s: float, min_reps: int = 50, max_reps: int | None = None,
                tail_s: float = 0.1, min_warm_reps: int = 20, period_hint_s: float = 1e-3) -> dict:
        max_reps = min(max_reps or len(self.starts), len(self.starts))
        t_begin = time.time()
        n_warm, t_warm, per = self._warm(fn, S, warm_s, min_warm_reps, period_hint_s)
        reps = int(min(max_reps, max(min_reps, math.ceil(window_s / max(per, 1e-6)))))
        self.probe.start()
        for i in range(reps):
            self._rep(fn, S, self.starts[i], self.ends[i], gated=True)
        n_tail = int(min(400, max(1, math.ceil(tail_s / max(per, 1e-6)))))
        for _ in range(n_tail):  # untimed, ungated: keeps the GPU busy while the host post-processes
            self._rep(fn, S, self.w0, self.w1, gated=False)
        self.ends[reps - 1].synchronize()
        t_end = time.time()
        self.probe.stop()
        self.g8.check("timed reps")
        st, ed = self.starts[:reps], self.ends[:reps]
        samples = np.array([a.elapsed_time(b) for a, b in zip(st, ed)]) * 1e3
        span_us = st[0].elapsed_time(ed[-1]) * 1e3
        period_us = st[0].elapsed_time(st[-1]) * 1e3 / max(1, reps - 1)
        pr = self.probe
        ts = pr.t[0] + np.array([pr.anchor.elapsed_time(e) for e in st]) * 1e6
        te = pr.t[0] + np.array([pr.anchor.elapsed_time(e) for e in ed]) * 1e6
        cov = (ts >= pr.t[0]) & (te <= pr.t[-1]) & (te > ts)
        out = {k: r5(v) for k, v in cbt.summarize(samples).items()}
        h = reps // 2
        out.update({
            "warm_n": n_warm, "warm_s": r5(t_warm), "period_us": r5(period_us), "span_us": r5(span_us),
            "half_ratio": r5(np.median(samples[h:]) / np.median(samples[:h])) if h >= 5 else None,
            "t0": round(t_end - span_us * 1e-6, 4), "t1": round(t_end, 4), "t_begin": round(t_begin, 4),
        })
        if cov.sum() >= 2:
            c0, c1 = np.interp(ts[cov], pr.t, pr.c), np.interp(te[cov], pr.t, pr.c)
            mhz = (c1 - c0) / (te[cov] - ts[cov]) * 1e3
            cyc = samples[cov] * mhz
            out.update({"clk": r5(np.median(mhz)), "clk_min": r5(mhz.min()), "clk_max": r5(mhz.max()),
                        "cycles": r5(np.median(cyc)), "clk_cov": r5(cov.mean())})
            if mhz.std() > 0 and samples[cov].std() > 0:
                out["clk_corr"] = r5(np.corrcoef(samples[cov], mhz)[0, 1])
        return out


def resolve_power(power: PowerLog, points: dict, baseline_key: str | None, wait: bool = True, drop_window: bool = True):
    """Fill P_w / energies of points that still carry their window (t0, t1)."""
    pend = [p for p in points.values() if "t0" in p and "P_w" not in p]
    if not pend:
        return
    need = max(p["t1"] for p in pend) + 0.06
    t_stop = time.time() + 2.0
    while wait and power.latest_ts() < need and time.time() < t_stop:
        time.sleep(0.05)
        power.drain()
    for p in pend:
        # skip the first 20 ms: a 20 ms sample may average activity from before the window
        w = power.window(p["t0"] + 0.02, p["t1"])
        p.update({k: r5(v) for k, v in w.items()})
        if "P_w" in p:
            p["E_rep_uj"] = r5(p["P_w"] * p["period_us"])
            p["E_naive_uj"] = r5(p["P_w"] * p["median"])
    base = points.get(baseline_key) if baseline_key else None
    if base and base.get("E_rep_uj") is not None:
        for p in points.values():
            if p is base or p.get("E_rep_uj") is None:
                continue
            p["E_kernel_uj"] = r5(p["E_rep_uj"] - base["E_rep_uj"])
            p["P_kernel_w"] = r5(p["E_kernel_uj"] / p["median"]) if p["median"] else None
    if drop_window:
        for p in points.values():
            if "P_w" in p or p.get("P_n") == 0:
                p.pop("t0", None)
                p["t1"] = round(p["t1"], 1) if "t1" in p else None


# ----------------------------------------------------------------------------------
# per-op data: inputs, fp32 reference, reference libraries
# ----------------------------------------------------------------------------------


def compare(out, ref, tol) -> dict:
    """|out - ref| <= atol*rms(ref) + rtol*|ref| (cotile.tests.harness.compare)."""
    atol, rtol = tol
    o = out.float()
    finite = bool(torch.isfinite(o).all())
    d = (o - ref).abs()
    rms = ref.pow(2).mean().sqrt()
    worst = float((d / (atol * rms + rtol * ref.abs())).max()) if finite else float("inf")
    return {"ok": finite and worst <= 1.0, "worst": r5(worst), "max_abs": r5(float(d.max())) if finite else None}


REF_NAMES = {"gemm": ["cublas", "cublas_fp32red"], "gqa_decode": ["flashinfer_cc", "flashinfer_tc"], "rmsnorm": ["torch", "flashinfer"]}


class OpData:
    """Inputs, shared output buffers, fp32 reference and reference-library callables."""

    def __init__(self, op, shape):
        self.op, self.shape = op, shape
        self.inputs = op.make_inputs(shape, seed=0)
        self.refs: dict = {}  # name -> (fn, output tensor)
        name = op.NAME
        if name == "gemm":
            A, B = self.inputs["A"], self.inputs["B"]
            self.out_name = "C"
            self.outputs = {"C": torch.empty(shape.M, shape.N, dtype=torch.bfloat16, device="cuda")}
            Cr = torch.empty_like(self.outputs["C"])
            self.refs["cublas"] = (lambda: torch.matmul(A, B.t(), out=Cr), Cr)
            Cr2 = torch.empty_like(self.outputs["C"])

            def mm_fp32red():  # cuBLAS without bf16 reduced-precision (split-K) reduction
                mm = torch.backends.cuda.matmul
                prev = mm.allow_bf16_reduced_precision_reduction
                mm.allow_bf16_reduced_precision_reduction = False
                torch.matmul(A, B.t(), out=Cr2)
                mm.allow_bf16_reduced_precision_reduction = prev

            self.refs["cublas_fp32red"] = (mm_fp32red, Cr2)
        elif name == "gqa_decode":
            from baselines import flashinfer_ops as fo

            Q, K, V = self.inputs["Q"], self.inputs["K"], self.inputs["V"]
            self.out_name = "O"
            self.outputs = {"O": torch.empty_like(Q)}
            ds = fo.DecodeShape(batch=shape.batch, kv_len=shape.seqlen, num_qo_heads=shape.heads,
                                num_kv_heads=shape.kv_heads, head_dim=shape.dim)
            kp = K.view(-1, ds.page_size, shape.kv_heads, shape.dim)  # identity page table: same bytes
            vp = V.view(-1, ds.page_size, shape.kv_heads, shape.dim)
            self._fi = []
            for tc in (False, True):
                d = fo.BatchDecode(ds, use_tensor_cores=tc)
                o = torch.empty_like(Q)
                self._fi.append(d)
                self.refs["flashinfer_tc" if tc else "flashinfer_cc"] = (functools.partial(d.run, Q, kp, vp, o), o)
        elif name == "rmsnorm":
            import flashinfer

            X, W = self.inputs["X"], self.inputs["W"]
            self.out_name = "Y"
            self.outputs = {"Y": torch.empty_like(X)}
            Yr = torch.empty_like(X)
            eps, H = shape.eps, shape.hidden
            holder = {}

            def torch_rms():
                holder["y"] = torch.nn.functional.rms_norm(X, (H,), W, eps)

            self.refs["torch"] = (torch_rms, holder)
            self.refs["flashinfer"] = (lambda: flashinfer.norm.rmsnorm(X, W, eps, out=Yr), Yr)
        else:
            raise KeyError(name)
        self._ref32 = None

    def ref32(self):
        if self._ref32 is None:
            name = self.op.NAME
            if name == "gemm":
                prev = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = False
                self._ref32 = self.op.reference(self.shape, self.inputs)["C"]
                torch.backends.cuda.matmul.allow_tf32 = prev
            elif name == "gqa_decode":
                from baselines import flashinfer_ops as fo

                self._ref32 = fo.decode_reference(self.inputs["Q"], self.inputs["K"], self.inputs["V"])
            else:
                self._ref32 = self.op.reference(self.shape, self.inputs)["Y"]
        return self._ref32

    def ref_output(self, name):
        o = self.refs[name][1]
        return o["y"] if isinstance(o, dict) else o


# ----------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------


class Budgets:
    def __init__(self):
        self.streams = {FULL: torch.cuda.Stream()}
        self.parts = []
        for req in BUDGET_REQUESTS:
            p = cb.split_sms(req, ignore_coscheduling=True)
            self.parts.append(p)
            self.streams[p.n_sms] = p.stream
        self.sms = sorted(self.streams, reverse=True)
        self.requested = {FULL: FULL, **{p.n_sms: p.requested for p in self.parts}}


def points_path(out: str, b: Batch) -> str:
    return os.path.join(out, "points", f"{b.name}.json")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_json(path, obj, indent=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, separators=(",", ":") if indent is None else None)
    os.replace(tmp, path)


def _sync_time(fn, S, reps=3) -> float:
    """Rough seconds per call (ungated, host-timed) -- only used to size warmup chunks."""
    with torch.cuda.stream(S):
        fn()
    S.synchronize()
    t = time.perf_counter()
    with torch.cuda.stream(S):
        for _ in range(reps):
            fn()
    S.synchronize()
    return (time.perf_counter() - t) / reps


class Runner2:
    """Launch closures for one config (grid + persistent share workspaces/counters)."""

    def __init__(self, b: Batch, tag: str, data: OpData):
        self.state: dict = {}
        self.fns = {}
        for build, spec in (("grid", b.grid[tag]), ("persistent", b.pers[tag])):
            if spec.kernel is None:
                continue
            r = Runner(spec, state=self.state)
            args, _ = r.make_args(data.inputs, data.outputs)
            self.fns[build] = functools.partial(spec.kernel, *args)


def run_batch(b: Batch, meter: FlushMeter, bud: Budgets, wd: Watchdog, args) -> dict:
    path = points_path(args.out, b)
    rec = load_json(path) or {}
    if rec.get("schema") != catalog.SCHEMA:
        rec = {}
    rec.setdefault("schema", catalog.SCHEMA)
    rec["op"], rec["shape"], rec["shape_tag"] = b.op.NAME, shape_dict(b.shape), b.tag
    rec.setdefault("points", {})
    rec.setdefault("configs", {})
    rec.setdefault("meta", {}).update({"env": env_meta(), "budgets": {str(k): v for k, v in bud.requested.items()},
                                        "protocol": {"warm_s": args.warm_s, "window_s": args.window_s, "tail_s": args.tail_s,
                                                     "min_reps": 50, "min_warm_reps": 20, "flush_bytes": 2 * cbt.l2_bytes(0)}})
    points = rec["points"]
    todo_keys = set()
    for tag in b.ctag:
        todo_keys |= {f"{tag}|grid|{s}" for s in bud.sms} | {f"{tag}|persistent|{FULL}"}
    for n in REF_NAMES[b.op.NAME]:
        todo_keys |= {f"ref:{n}|{rb}|{s}" for s in bud.sms for rb in ("ref", "ref_end")}
    missing = [k for k in todo_keys if k not in points or points[k].get("contaminated")]
    if not missing and all(t in rec["configs"] for t in b.ctag):
        log(f"[{b.name}] complete, skipped")
        return rec
    data = OpData(b.op, b.shape)
    assert sorted(data.refs) == sorted(REF_NAMES[b.op.NAME])

    # ---- correctness + ungated pre-warm on every stream (lazy loading, sizing) -------
    t_prep = time.time()
    ref = data.ref32()
    runners = {}
    est = {}  # (name, build, sms) -> s per call
    for tag, cfg in b.ctag.items():
        cr = {"cfg": {k: v for k, v in cfg.__dict__.items()}, "sig": b.sig[tag],
              "work": catalog.work(b.op.NAME, b.shape, cfg), "check": {}}
        r = Runner2(b, tag, data)
        runners[tag] = r
        for build, fn in r.fns.items():
            data.outputs[data.out_name].fill_(float("nan"))
            fn()
            torch.cuda.synchronize()
            cr["check"][build] = compare(data.outputs[data.out_name], ref, b.op.TOLERANCE)
            est[(tag, build, FULL)] = _sync_time(fn, bud.streams[FULL])
            if build == "grid":
                for s in bud.sms[1:]:
                    est[(tag, build, s)] = _sync_time(fn, bud.streams[s])
        rec["configs"][tag] = cr
        if not all(c["ok"] for c in cr["check"].values()):
            log(f"[{b.name}] {tag}: CORRECTNESS FAIL {cr['check']}")
    rec["ref_check"] = {}
    for n, (fn, _) in data.refs.items():
        fn()
        torch.cuda.synchronize()
        rec["ref_check"][n] = compare(data.ref_output(n), ref, b.op.TOLERANCE)
        for s in bud.sms:
            est[(f"ref:{n}", "ref", s)] = est[(f"ref:{n}", "ref_end", s)] = _sync_time(fn, bud.streams[s])
    del ref
    data._ref32 = None
    torch.cuda.empty_cache()
    rec["meta"]["prep_s"] = round(time.time() - t_prep, 1)
    log(f"[{b.name}] prep {rec['meta']['prep_s']}s, {len(b.ctag)} configs, {len(missing)} points to measure")

    def fn_of(key):
        name, build, sms = key.split("|")
        if name.startswith("ref:"):
            return data.refs[name[4:]][0], bud.streams[int(sms)]
        return runners[name].fns[build], bud.streams[int(sms)]

    # Config-outer order: the four points of a config (grid @188, persistent @188, grid @94,
    # grid @48) are adjacent, so every config sees the same mix of recent load and the GPU's
    # temperature (a slow variable that moves power-capped clocks by ~10% between ~45 and
    # ~85 C) is quasi-steady across the batch instead of drifting with a budget sweep.
    # References are measured at the start ("ref") and again at the end ("ref_end") of the
    # batch as a drift check.
    order = [f"ref:{n}|ref|{s}" for s in bud.sms for n in data.refs]
    for tag in b.ctag:
        order += [f"{tag}|grid|{FULL}", f"{tag}|persistent|{FULL}"] + [f"{tag}|grid|{s}" for s in bud.sms[1:]]
    order += [f"ref:{n}|ref_end|{s}" for s in bud.sms for n in data.refs]

    wd.wait_quiet()
    t_batch = time.time()
    # batch warmup: replay the first config's point mix (grid/persistent @188, grid @94/@48)
    # until the GPU temperature is flat, so the batch starts in its own thermal steady state
    tag0 = next(iter(b.ctag))
    mix = [(k, fn_of(k), est.get((tag0,) + tuple(k.split("|")[1:2]) + (int(k.split("|")[2]),), 1e-3))
           for k in order if k.startswith(tag0 + "|")]
    rec["meta"]["batch_warmup"] = batch_warmup(meter, mix, args.batch_warm_s, args.batch_warm_max_s)
    base_key = f"flush_only|none|{FULL}"
    if base_key not in points or points[base_key].get("contaminated"):
        points[base_key] = measure_point(meter, wd, None, bud.streams[FULL], args, 0.0, attempts_max=1)

    def run_keys(keys):
        for key in keys:
            fn, S = fn_of(key)
            name, build, sms = key.split("|")
            hint = est.get((name, build, int(sms)), 1e-3)
            points[key] = measure_point(meter, wd, fn, S, args, hint)
            p = points[key]
            if args.verbose:
                log(f"  {key}: {p['median']:.1f}us cv={p['cv'] * 100:.2f}% clk={p.get('clk')} n={p['n']} "
                    f"half={p.get('half_ratio')}{' CONTAMINATED' if p.get('contaminated') else ''}")
            if p.get("contaminated"):  # foreign SM activity: pause until quiet, re-warm, go on
                log(f"[{b.name}] {key}: foreign SM activity {p.get('foreign')}; pausing")
                wd.wait_quiet()
                batch_warmup(meter, mix, args.batch_warm_s, args.batch_warm_max_s)

    keys = [k for k in order if k not in points or points[k].get("contaminated")]
    for i in range(0, len(keys), 40):  # save every 40 points (resumable)
        run_keys(keys[i:i + 40])
        resolve_power(meter.power, points, base_key)
        save_json(path, rec)
        log(f"[{b.name}] {min(i + 40, len(keys))}/{len(keys)} points, {time.time() - t_batch:.0f}s so far")
    # re-measure contaminated points
    remeasured = rec["meta"].setdefault("remeasured", [])
    for rnd in range(3):
        bad = [k for k, p in points.items() if p.get("contaminated")]
        if not bad:
            break
        remeasured.append({"round": rnd + 1, "n": len(bad), "keys": bad[:50],
                           "foreign": sorted({f for k in bad for f in points[k].get("foreign", [])})})
        log(f"[{b.name}] re-measuring {len(bad)} contaminated points (round {rnd + 1})")
        wd.wait_quiet()
        batch_warmup(meter, mix, args.batch_warm_s, args.batch_warm_max_s)
        run_keys([k for k in bad if k != base_key])
        if base_key in bad:
            points[base_key] = measure_point(meter, wd, None, bud.streams[FULL], args, 0.0, attempts_max=1)
        resolve_power(meter.power, points, base_key)
        save_json(path, rec)
    rec["meta"]["batch_gpu_s"] = round(time.time() - t_batch, 1)
    rec["meta"]["foreign"] = wd.summary(t_batch, time.time())
    save_json(path, rec)
    del runners, data
    torch.cuda.empty_cache()
    return rec


def batch_warmup(meter: FlushMeter, mix: list, min_s: float, max_s: float, slice_s: float = 2.0) -> dict:
    """Cycle through `mix` [(key, (fn, S), hint_s)] in slice_s slices for >= min_s, then until
    the max temperature of the last 10 s differs from that of the 10 s before by < 1 C."""
    t0 = time.time()
    temps = []
    i = 0
    while True:
        _, (fn, S), hint = mix[i % len(mix)]
        meter.warm(fn, S, slice_s, hint + 2.2e-4)
        i += 1
        el = time.time() - t0
        temps.append((round(el, 1), meter.power.nv.temperature_c()))
        if el >= max_s:
            break
        if el >= min_s:
            last = [t for x, t in temps if x >= el - 10]
            prev = [t for x, t in temps if el - 20 <= x < el - 10]
            if prev and abs(max(last) - max(prev)) < 1:
                break
    return {"seconds": round(time.time() - t0, 1), "temps": temps}


def measure_point(meter: FlushMeter, wd: Watchdog, fn, S, args, hint_s: float, attempts_max: int | None = None) -> dict:
    best, attempts = None, []
    for _ in range(attempts_max or args.attempts):
        p = meter.measure(fn, S, warm_s=args.warm_s, window_s=args.window_s, tail_s=args.tail_s,
                          period_hint_s=hint_s + 2.2e-4)
        attempts.append(p["cv"])
        foreign = wd.between(p["t_begin"], p["t1"])
        if foreign:  # re-measured after the batch, once the GPU is free again
            p["contaminated"] = True
            p["foreign"] = foreign
            best = p
            break
        if best is None or p["cv"] < best["cv"]:
            best = p
        if p["cv"] <= CV_TARGET:
            break
    if len(attempts) > 1:
        best["attempts"] = attempts
    best["spread"] = r5((best["p90"] - best["p10"]) / best["median"]) if best["median"] else None
    for k in ("t_begin", "std", "mean", "span_us"):
        best.pop(k, None)
    if best.get("clk_cov") == 1.0:
        best.pop("clk_cov")
    return best


def global_warmup(meter: FlushMeter, power: PowerLog, S, min_s: float, max_s: float) -> dict:
    """Flush-mode loop of a large cuBLAS GEMM until the temperature is flat: at least
    min_s, then until the max temperature over the last 20 s exceeds that of the 20 s
    before by < 1 C (or max_s)."""
    M, N, K = 8192, 14336, 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = lambda: torch.matmul(A, B.t(), out=C)  # noqa: E731
    t0 = time.time()
    hint = _sync_time(fn, S)
    temps = []
    while True:
        meter.warm(fn, S, 5.0, hint)
        temps.append((time.time() - t0, power.nv.temperature_c()))
        el = time.time() - t0
        if el >= max_s:
            break
        if el >= min_s:
            last = [t for x, t in temps if x >= el - 20]
            prev = [t for x, t in temps if el - 40 <= x < el - 20]
            if prev and max(last) - max(prev) < 1:
                break
    log(f"global warmup {time.time() - t0:.0f}s, temps {temps[0][1]} -> {temps[-1][1]} C")
    del A, B, C
    return {"seconds": round(time.time() - t0, 1), "temps": temps}


def cmd_run(args):
    batches = selected_batches(args)
    log(f"run: {len(batches)} batches, preparing kernels (kernel cache) ...")
    st = prepare(batches, workers=args.workers)
    log(f"prepared {st['n_kernels']} kernels in {st['total_s']:.0f}s (build {st['build_s']:.0f}s)")
    power = PowerLog()
    wd = Watchdog()
    meter = FlushMeter(power)
    bud = Budgets()
    log(f"budgets: {bud.sms} (requested {bud.requested})")
    wd.wait_quiet()
    runlog_path = os.path.join(args.out, "runlog.json")
    runlog = load_json(runlog_path) or {"runs": []}
    run = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "prepare": st, "batches": {}, "args": vars(args)}
    runlog["runs"].append(run)
    t0 = time.time()
    if not args.no_global_warmup:
        run["global_warmup"] = global_warmup(meter, power, bud.streams[FULL], args.global_min_s, args.global_max_s)
    for b in batches:
        tb = time.time()
        rec = run_batch(b, meter, bud, wd, args)
        run["batches"][b.name] = {"wall_s": round(time.time() - tb, 1), "gpu_s": rec["meta"].get("batch_gpu_s")}
        run["wall_s"] = round(time.time() - t0, 1)
        run["power_timeline"] = power.timeline()
        save_json(runlog_path, runlog, indent=None)
    run["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run["wall_s"] = round(time.time() - t0, 1)
    run["power_timeline"] = power.timeline()
    save_json(runlog_path, runlog)
    wd.stop()
    power.stop()
    log(f"run done in {run['wall_s']:.0f}s")


# ----------------------------------------------------------------------------------
# studies
# ----------------------------------------------------------------------------------


def _study_setup(args, specs: list[tuple[str, str, str, int]]):
    """specs: [(op, shape_tag, cfg_tag or 'ref:<name>', sms)] -> prepared callables."""
    shapes = plan_shapes()
    by_batch: dict = {}
    for opn, stag, ctag, sms in specs:
        sh = next(s for s in shapes[opn] if catalog.shape_tag(opn, s) == stag)
        by_batch.setdefault((opn, stag), (sh, set()))[1].add(ctag)
    batches = []
    for (opn, stag), (sh, tags) in by_batch.items():
        b = Batch(OPS[opn], sh)
        b.ctag = {t: c for t, c in b.ctag.items() if t in tags}
        batches.append(b)
    prepare(batches, workers=args.workers)
    return batches


def cmd_warmup_study(args):
    """Per-point warmup justification: medians / clocks of representative points measured
    with per-point warmup W after three predecessors (idle 2 s; 3 s of a power-capped GEMM
    flush loop; 3 s of the point itself = steady-state reference)."""
    targets = [
        ("gemm", "M8192_N14336_K4096", "128x128x32_s3_t256_k1_wsoff_smem_g8", FULL),
        ("gemm", "M8192_N14336_K4096", "128x128x32_s3_t256_k1_wsoff_smem_g8", 48),
        ("gqa_decode", "B64_S8192", "n128_h4_sp1_t128_s1", FULL),
        ("rmsnorm", "T16384_H4096", "r1_t256_v8", FULL),
    ]
    batches = _study_setup(args, targets)
    power, wd = PowerLog(), Watchdog()
    meter, bud = FlushMeter(power), Budgets()
    wd.wait_quiet()
    datas = {}
    fns = {}
    for b in batches:
        d = OpData(b.op, b.shape)
        datas[b.name] = d
        for tag in b.ctag:
            fns[(b.op.NAME, b.tag, tag)] = Runner2(b, tag, d).fns["grid"]
    # hot predecessor: cuBLAS GEMM 8192x14336x4096 (the global-warmup workload)
    A = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
    Bm = torch.randn(14336, 4096, device="cuda", dtype=torch.bfloat16)
    Cm = torch.empty(8192, 14336, device="cuda", dtype=torch.bfloat16)
    hot = lambda: torch.matmul(A, Bm.t(), out=Cm)  # noqa: E731
    for s in bud.sms:
        _sync_time(hot, bud.streams[s])
    for (opn, stag, ctag, sms) in targets:
        for s in bud.sms:
            _sync_time(fns[(opn, stag, ctag)], bud.streams[s])
    res = {"targets": [], "W": args.study_w, "preds": ["idle", "hot", "self"]}
    gw = global_warmup(meter, power, bud.streams[FULL], args.global_min_s, args.global_max_s)
    res["global_warmup"] = gw
    for (opn, stag, ctag, sms) in targets:
        fn, S = fns[(opn, stag, ctag)], bud.streams[sms]
        hint = _sync_time(fn, S)
        rows = []
        for rep in range(args.study_reps):
            for pred in res["preds"]:
                for W in args.study_w:
                    if pred == "idle":
                        torch.cuda.synchronize()
                        time.sleep(2.0)
                    elif pred == "hot":
                        meter.warm(hot, bud.streams[FULL], 3.0, 2.5e-3)
                    else:
                        meter.warm(fn, S, 3.0, hint + 2.2e-4)
                    p = meter.measure(fn, S, warm_s=W, window_s=args.window_s, tail_s=args.tail_s, min_warm_reps=1,
                                      period_hint_s=hint + 2.2e-4)
                    p["foreign"] = wd.between(p["t_begin"], p["t1"])
                    rows.append({"rep": rep, "pred": pred, "W": W, **{k: p.get(k) for k in (
                        "median", "p10", "p90", "cv", "n", "warm_n", "clk", "clk_min", "clk_max", "cycles", "half_ratio", "foreign")}})
                    log(f"{opn}:{stag}:{ctag}@{sms} rep{rep} pred={pred} W={W}: {p['median']:.1f}us clk={p.get('clk')} "
                        f"half={p.get('half_ratio')} warm_n={p['warm_n']}")
        res["targets"].append({"op": opn, "shape": stag, "cfg": ctag, "sms": sms, "rows": rows})
        save_json(os.path.join(args.out, "studies", "warmup.json"), res, indent=1)
    wd.stop()
    power.stop()


def cmd_recheck(args):
    """Re-measure selected points at the end of the sweep and compare with the stored ones."""
    cat = catalog.load(args.out)
    picks = []
    for opn, stag, sms, rank in args.recheck:
        e = cat.get(opn, stag)
        cur = e.curve(sms)
        picks.append((opn, stag, cur[min(rank, len(cur) - 1)][0], sms))
    batches = _study_setup(args, picks)
    power, wd = PowerLog(), Watchdog()
    meter, bud = FlushMeter(power), Budgets()
    wd.wait_quiet()
    fns = {}
    datas = []
    for b in batches:
        d = OpData(b.op, b.shape)
        datas.append(d)
        for tag in b.ctag:
            fns[(b.op.NAME, b.tag, tag)] = Runner2(b, tag, d).fns["grid"]
    for (opn, stag, ctag, sms) in picks:
        _sync_time(fns[(opn, stag, ctag)], bud.streams[sms])
    gw = global_warmup(meter, power, bud.streams[FULL], args.global_min_s, args.global_max_s)
    rows = []
    for rep in range(args.study_reps):
        for (opn, stag, ctag, sms) in picks:
            fn, S = fns[(opn, stag, ctag)], bud.streams[sms]
            p = measure_point(meter, wd, fn, S, args, _sync_time(fn, S))
            old = cat.get(opn, stag).configs[ctag].meas[("grid", sms)]
            rows.append({"op": opn, "shape": stag, "cfg": ctag, "sms": sms, "rep": rep, "old": old.median, "new": p["median"],
                         "ratio": r5(p["median"] / old.median), "old_clk": old.clock_mhz, "new_clk": p.get("clk"),
                         "old_cv": old.cv, "new_cv": p["cv"], "old_t": old.raw.get("t1"), "new_t": p["t1"],
                         "contaminated": p.get("contaminated", False)})
            log(f"recheck {opn}:{stag}:{ctag}@{sms}: old {old.median:.2f} new {p['median']:.2f} ratio {p['median'] / old.median:.4f} "
                f"clk {old.clock_mhz}->{p.get('clk')}")
    save_json(os.path.join(args.out, "studies", "recheck.json"), {"global_warmup": gw, "rows": rows}, indent=1)
    wd.stop()
    power.stop()


def cmd_energy_check(args):
    """Energy/call two ways for a few configs: (a) flush-mode subtraction (as in the sweep),
    (b) cobench graph mode (back-to-back calls over rotated input copies, ~100% duty cycle,
    power averaged over >= 1 s): E = P_graph x t_graph."""
    cat = catalog.load(args.out)
    picks = []
    for opn, stag, sms, rank in args.energy:
        e = cat.get(opn, stag)
        cur = e.curve(sms)
        picks.append((opn, stag, cur[min(rank, len(cur) - 1)][0], sms))
    batches = _study_setup(args, picks)
    power, wd = PowerLog(), Watchdog()
    meter, bud = FlushMeter(power), Budgets()
    wd.wait_quiet()
    gw = global_warmup(meter, power, bud.streams[FULL], args.global_min_s, args.global_max_s)
    rows = []
    for b in batches:
        for tag in b.ctag:
            sms = next(p[3] for p in picks if p[0] == b.op.NAME and p[1] == b.tag and p[2] == tag)
            S = bud.streams[sms]
            spec = b.grid[tag]
            probe_data = OpData(b.op, b.shape)
            r = Runner(spec, state={})
            args0, _ = r.make_args(probe_data.inputs, probe_data.outputs)
            fn_flush = functools.partial(spec.kernel, *args0)
            base = measure_point(meter, wd, None, bud.streams[FULL], args, 0.0)
            pf = measure_point(meter, wd, fn_flush, S, args, _sync_time(fn_flush, S))
            pts = {"base": base, "p": pf}
            resolve_power(power, pts, "base")

            def mk():
                d = b.op.make_inputs(b.shape, seed=1)
                rr = Runner(spec, state={})
                a, _ = rr.make_args(d)
                return (a,)

            gres = cb.bench(lambda a: spec.kernel(*a), make_inputs=mk, mode="graph", stream=S, nvml=True,
                            clock=True, warmup_s=1.0, reps=max(50, int(1.5 / max(pf["median"] * 1e-6 * 20, 1e-6))),
                            strict=False, label=f"{b.name}:{tag}")
            gp = gres.nvml["power_w"]["mean"]
            row = {"op": b.op.NAME, "shape": b.tag, "cfg": tag, "sms": sms,
                   "flush": {k: pf.get(k) for k in ("median", "clk", "P_w", "period_us", "E_naive_uj", "E_rep_uj", "E_kernel_uj", "P_kernel_w")},
                   "flush_only": {k: base.get(k) for k in ("P_w", "period_us", "E_rep_uj")},
                   "graph": {"median": r5(gres.median), "clk": r5(gres.clock["per_rep"].get("median")), "P_w": r5(gp),
                             "P_n": gres.nvml["power_w"]["n"], "duration_s": r5(gres.nvml["duration_s"]), "k": gres.k,
                             "n_copies": gres.n_copies, "E_uj": r5(gp * gres.median)}}
            rows.append(row)
            log(f"energy {b.name}:{tag}@{sms}: flush E_kernel={pf.get('E_kernel_uj')} uJ (P_k={pf.get('P_kernel_w')} W, "
                f"E_naive={pf.get('E_naive_uj')}) | graph {gres.median:.1f}us P={gp:.0f}W E={gp * gres.median:.0f} uJ")
            del probe_data
            torch.cuda.empty_cache()
    save_json(os.path.join(args.out, "studies", "energy.json"), {"global_warmup": gw, "rows": rows}, indent=1)
    wd.stop()
    power.stop()


def cmd_persistent_modes(args):
    """Grid vs persistent build of the same config under four L2/launch regimes (coordinator
    request, 2026-09-22): flush (cobench: 2xL2 *write* before every call, i.e. L2 full of dirty
    lines), flush-read (2xL2 *read*: L2 full of clean lines), hot (cobench graph of 20 calls on
    one input copy, L2-warm) and graph (cobench graph of back-to-back calls rotating over input
    copies >= 2xL2, cold but no flush). Picks per op at one shape: the solo-best config, the
    config with the largest persistent/grid ratio and the one with the median ratio."""
    cat = catalog.load(args.out)
    picks = []
    for opn, stag in (("gemm", args.pm_gemm), ("gqa_decode", args.pm_decode), ("rmsnorm", args.pm_rmsnorm)):
        if (opn, stag) not in cat.entries:
            log(f"persistent-modes: {opn}:{stag} not in the catalog, skipped")
            continue
        e = cat.get(opn, stag)
        # same code path in both builds: exclude GEMM ws=auto (its persistent build is cp.async)
        same = {t: v for t, v in e.persistent_penalty().items() if e.configs[t].cfg_dict.get("ws", "off") == "off"}
        srt = sorted(same.items(), key=lambda x: x[1])
        best = min(same, key=lambda t: e.configs[t].time())
        chosen = []
        for t in (best, srt[-1][0], srt[len(srt) // 2][0], srt[0][0]):
            if t not in chosen and len(chosen) < 3:
                chosen.append(t)
        picks += [(opn, stag, t, FULL) for t in chosen]
    batches = _study_setup(args, picks)
    power, wd = PowerLog(), Watchdog()
    meter, bud = FlushMeter(power), Budgets()
    S = bud.streams[FULL]
    wd.wait_quiet()
    gw = global_warmup(meter, power, S, args.global_min_s, args.global_max_s)
    rows = []
    for b in batches:
        data = OpData(b.op, b.shape)
        for tag in b.ctag:
            r = Runner2(b, tag, data)
            row = {"op": b.op.NAME, "shape": b.tag, "cfg": tag, "sig": b.sig[tag], "modes": {}}
            for build in ("grid", "persistent"):
                _sync_time(r.fns[build], S)
            for mode in ("flush", "flush_read", "hot", "graph"):
                for build in ("grid", "persistent"):
                    fn = r.fns[build]
                    spec = b.grid[tag] if build == "grid" else b.pers[tag]
                    if mode in ("flush", "flush_read"):
                        meter.flush_kind = "write" if mode == "flush" else "read"
                        p = measure_point(meter, wd, fn, S, args, _sync_time(fn, S))
                        meter.flush_kind = "write"
                        res = {k: p.get(k) for k in ("median", "p10", "p90", "cv", "clk", "cycles", "n", "contaminated")}
                    else:
                        if mode == "hot":
                            mk = None
                        else:
                            def mk(spec=spec, b=b):
                                d = b.op.make_inputs(b.shape, seed=int(time.time() * 1e3) % 100000)
                                a, _ = Runner(spec, state={}).make_args(d)
                                return (a,)
                        t_a = time.time()
                        gres = cb.bench((lambda a, spec=spec: spec.kernel(*a)) if mk else fn, make_inputs=mk,
                                        mode=mode, stream=S, clock=True, warmup_s=1.0, reps=60, strict=False,
                                        label=f"{b.name}:{tag}:{build}:{mode}")
                        c = gres.clock["per_rep"]
                        res = {"median": r5(gres.median), "p10": r5(gres.p10), "p90": r5(gres.p90), "cv": r5(gres.cv),
                               "clk": r5(c.get("median")), "cycles": r5(c.get("cycles_median")), "k": gres.k,
                               "n_copies": gres.n_copies, "contaminated": bool(wd.between(t_a, time.time()))}
                        torch.cuda.empty_cache()
                    row["modes"].setdefault(mode, {})[build] = res
                    log(f"pm {b.name}:{tag} {mode:10s} {build:10s} {res['median']:.1f}us clk={res.get('clk')} cyc={res.get('cycles')}")
                g, pp = row["modes"][mode]["grid"], row["modes"][mode]["persistent"]
                row["modes"][mode]["ratio_time"] = r5(pp["median"] / g["median"])
                if g.get("cycles") and pp.get("cycles"):
                    row["modes"][mode]["ratio_cycles"] = r5(pp["cycles"] / g["cycles"])
            rows.append(row)
            save_json(os.path.join(args.out, "studies", "persistent_modes.json"), {"global_warmup": gw, "rows": rows}, indent=1)
        del data
        torch.cuda.empty_cache()
    wd.stop()
    power.stop()


def cmd_flush_bias(args):
    """cobench's flush writes 2xL2, leaving L2 full of dirty lines: every line the timed op
    allocates (its loads and its output writes) first evicts a dirty line, so write-backs land
    inside the timed window. For every op x shape, measure the solo-best TileLang config (grid,
    188 SMs) and the fastest correct reference with the write-flush and with a write+read flush
    (write 2xL2, then read it: evicted and clean), adjacent in time; writes
    studies/flush_bias.json. (A read-only flush does not evict completely on this GPU.)"""
    cat = catalog.load(args.out)
    batches = selected_batches(args)
    for b in batches:
        e = cat.get(b.op.NAME, b.tag)
        b.ctag = {t: c for t, c in b.ctag.items() if t == e.solo_best.tag}
    prepare(batches, workers=args.workers)
    power, wd = PowerLog(), Watchdog()
    meter, bud = FlushMeter(power), Budgets()
    S = bud.streams[FULL]
    wd.wait_quiet()
    gw = global_warmup(meter, power, S, args.global_min_s, args.global_max_s)
    rows = []
    for b in batches:
        e = cat.get(b.op.NAME, b.tag)
        data = OpData(b.op, b.shape)
        rb = e.ref_best(FULL)
        fns = {e.solo_best.tag: Runner2(b, e.solo_best.tag, data).fns["grid"]}
        if rb:
            fns[f"ref:{rb[0]}"] = data.refs[rb[0]][0]
        for name, fn in fns.items():
            hint = _sync_time(fn, S)
            res = {}
            for kind in ("write", "write+read"):
                meter.flush_kind = kind
                p = measure_point(meter, wd, fn, S, args, hint)
                res[kind] = {k: p.get(k) for k in ("median", "p10", "p90", "cv", "clk", "contaminated")}
            meter.flush_kind = "write"
            row = {"op": b.op.NAME, "shape": b.tag, "name": name, **res,
                   "clean_over_write": r5(res["write+read"]["median"] / res["write"]["median"]),
                   "delta_us": r5(res["write"]["median"] - res["write+read"]["median"])}
            rows.append(row)
            log(f"flush-bias {b.name} {name}: write {res['write']['median']:.1f} write+read {res['write+read']['median']:.1f} "
                f"ratio {row['clean_over_write']}")
        save_json(os.path.join(args.out, "studies", "flush_bias.json"), {"global_warmup": gw, "rows": rows}, indent=1)
        del data
        torch.cuda.empty_cache()
    wd.stop()
    power.stop()


def cmd_ref_extra(args):
    """cuBLAS with torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    (torch's default True lets cuBLAS reduce split-K partials in bf16; at M2048_N4096_K14336
    its output then misses the fp32-reference tolerance). Stored as `ref:cublas_fp32red` in the
    GEMM point files (same protocol; measured after the sweep in its own process, so not
    interleaved with the batch), with its own correctness check."""
    batches = [b for b in selected_batches(args) if b.op.NAME == "gemm"]
    power, wd = PowerLog(), Watchdog()
    meter, bud = FlushMeter(power), Budgets()
    wd.wait_quiet()
    gw = global_warmup(meter, power, bud.streams[FULL], args.global_min_s, args.global_max_s)
    for b in batches:
        path = points_path(args.out, b)
        rec = load_json(path)
        A, B = (t for t in b.op.make_inputs(b.shape, seed=0).values())
        C = torch.empty(b.shape.M, b.shape.N, dtype=torch.bfloat16, device="cuda")
        prev = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        try:
            fn = lambda: torch.matmul(A, B.t(), out=C)  # noqa: E731
            fn()
            torch.cuda.synchronize()
            ref = b.op.reference(b.shape, {"A": A, "B": B})["C"]
            rec.setdefault("ref_check", {})["cublas_fp32red"] = compare(C, ref, b.op.TOLERANCE)
            del ref
            base_key = f"flush_only|none|{FULL}"
            pts = {base_key: rec["points"][base_key]}
            for sms in bud.sms:
                S = bud.streams[sms]
                hint = _sync_time(fn, S)
                if sms == FULL:  # a fresh flush-only baseline in this process for the energy estimate
                    pts[base_key] = measure_point(meter, wd, None, bud.streams[FULL], args, 0.0, attempts_max=1)
                key = f"ref:cublas_fp32red|ref|{sms}"
                pts[key] = measure_point(meter, wd, fn, S, args, hint)
                log(f"{b.name} {key}: {pts[key]['median']:.1f}us clk={pts[key].get('clk')} "
                    f"(cublas default: {rec['points'][f'ref:cublas|ref|{sms}']['median']:.1f})")
            resolve_power(power, pts, base_key)
            for k, v in pts.items():
                if k != base_key:
                    v["note"] = "measured after the sweep (separate process)"
                    rec["points"][k] = v
            rec["meta"]["ref_extra_global_warmup_s"] = gw["seconds"]
            save_json(path, rec)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev
    wd.stop()
    power.stop()


def _parse_picks(s: str) -> list:
    out = []
    for item in s.split(";"):
        opn, stag, sms, rank = item.split(":")
        out.append((opn, stag, int(sms), int(rank)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["compile", "run", "warmup-study", "recheck", "energy-check", "persistent-modes", "ref-extra", "flush-bias"])
    ap.add_argument("--pm-gemm", default="M4096_N4096_K4096")
    ap.add_argument("--pm-decode", default="B64_S8192")
    ap.add_argument("--pm-rmsnorm", default="T16384_H4096")
    ap.add_argument("--ops", default="gemm,gqa_decode,rmsnorm")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--limit", type=int, default=None, help="first N configs per shape (smoke tests)")
    ap.add_argument("--out", default=catalog.DEFAULT_DIR)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--warm-s", type=float, default=0.3, help="per-point warmup (s)")
    ap.add_argument("--window-s", type=float, default=0.25, help="minimum timed window (s)")
    ap.add_argument("--tail-s", type=float, default=0.1)
    ap.add_argument("--batch-warm-s", type=float, default=12.0, help="batch warmup: minimum (s)")
    ap.add_argument("--batch-warm-max-s", type=float, default=60.0, help="batch warmup: maximum (s)")
    ap.add_argument("--global-min-s", type=float, default=60.0)
    ap.add_argument("--global-max-s", type=float, default=240.0)
    ap.add_argument("--no-global-warmup", action="store_true")
    ap.add_argument("--attempts", type=int, default=2, help="attempts per point while CV > 2%%")
    ap.add_argument("--study-w", default="0.05,0.1,0.2,0.3,0.5,1.0,2.0")
    ap.add_argument("--study-reps", type=int, default=2)
    ap.add_argument("--recheck", default="gemm:M8192_N14336_K4096:188:0;gemm:M4096_N4096_K4096:48:0;"
                                          "gqa_decode:B64_S8192:188:0;gqa_decode:B128_S2048:94:0;rmsnorm:T65536_H8192:188:0")
    ap.add_argument("--energy", default="gemm:M8192_N14336_K4096:188:0;gemm:M2048_N4096_K4096:188:0;"
                                         "gqa_decode:B64_S8192:188:0;gqa_decode:B16_S2048:188:0;"
                                         "rmsnorm:T65536_H8192:188:0;rmsnorm:T4096_H4096:188:0")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    args.ops = [o for o in args.ops.split(",") if o]
    args.shapes = [s for s in args.shapes.split(",") if s]
    args.study_w = [float(x) for x in args.study_w.split(",")]
    args.recheck = _parse_picks(args.recheck)
    args.energy = _parse_picks(args.energy)
    torch.backends.cuda.matmul.allow_tf32 = False
    {"compile": cmd_compile, "run": cmd_run, "warmup-study": cmd_warmup_study, "recheck": cmd_recheck,
     "energy-check": cmd_energy_check, "persistent-modes": cmd_persistent_modes, "ref-extra": cmd_ref_extra, "flush-bias": cmd_flush_bias}[args.cmd](args)


if __name__ == "__main__":
    main()
