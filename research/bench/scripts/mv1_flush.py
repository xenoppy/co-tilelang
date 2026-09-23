"""M1 (methodology v1): clean L2 flush vs the v0 write-only flush.

    source research/env.sh
    python research/bench/scripts/mv1_flush.py [--passes 2]

Parts (results -> research/results/2026-09-23_methodology_v1/flush.json):
  A  mechanism: (i) eviction -- a 32 MB buffer read in the previous rep is read again after
     each flush kind (evicted => DRAM rate); (ii) cleanliness -- a 64 MB write-only kernel
     after each flush kind (clean L2 => the writes are absorbed without write-backs).
  B  ops: bf16 matmul 4096^3 (torch), copy 512 MB -> 512 MB (copy_u4), RMSNorm 4096x4096
     and GQA decode B16xS8192 (catalog solo-best TileLang configs) under flush-write,
     flush-clean, flush-read, graph (rotation, back-to-back) and hot; time, SM clock,
     cycles, power, rep period.
  C  flush cost / floor: duration of each flush kind; an empty kernel under each mode.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

import mv1_common as C
from mv1_common import cb, log

MB = 1 << 20
MODES = [("flush", "write"), ("flush", "write+read"), ("flush", "clean"), ("graph", None), ("hot", None)]
FLUSHES = [("flush", "write"), ("flush", "write+read"), ("flush", "clean"), ("flush", "read")]


def row(r, extra=None):
    per = (r.clock or {}).get("per_rep", {})
    d = {"median_us": r.median, "p10_us": r.p10, "p90_us": r.p90, "cv": r.cv, "n": r.n,
         "clock_mhz": per.get("median"), "kcycles": (per.get("cycles_median") or 0) / 1e3 or None,
         "power_w": (r.nvml or {}).get("power_w", {}).get("mean"), "period_us": r.period_us,
         "gbps": r.gbps, "tflops": r.tflops, "k": r.k, "n_copies": r.n_copies,
         "guard_clean": (r.guard or {}).get("clean")}
    if extra:
        d.update(extra)
    return d


def bench_modes(name, fn, make_inputs, modes, **kw):
    out = {}
    for mode, fk in modes:
        key = mode if fk is None else f"{mode}-{fk}"
        r = cb.bench(fn, make_inputs=make_inputs, mode=mode, flush_kind=fk or "clean", clock=True, nvml=True,
                     label=f"{name}:{key}", **kw)
        log("   ", r)
        out[key] = row(r)
    return out


def part_a():
    res = {}
    X = torch.empty(32 * MB // 4, dtype=torch.int32, device="cuda").fill_(3)
    rd = lambda: cb.read_u4(X)  # noqa: E731
    res["evict_read32MB"] = bench_modes("read32MB", rd, None, FLUSHES + [("hot", None)], nbytes=32 * MB)
    # cold reference: rotate 32 MB buffers (>= 2x L2 in total), back-to-back
    res["evict_read32MB"]["graph"] = row(cb.bench(lambda x: cb.read_u4(x), make_inputs=lambda: (
        torch.empty(32 * MB // 4, dtype=torch.int32, device="cuda").fill_(3),), mode="graph", clock=True, nvml=True,
        nbytes=32 * MB, label="read32MB:graph"))
    Y = torch.empty(64 * MB // 4, dtype=torch.int32, device="cuda")
    wr = lambda: Y.fill_(7)  # noqa: E731
    res["clean_write64MB"] = bench_modes("write64MB", wr, None, FLUSHES + [("hot", None)], nbytes=64 * MB)
    res["clean_write64MB"]["graph"] = row(cb.bench(lambda y: y.fill_(7), make_inputs=lambda: (
        torch.empty(64 * MB // 4, dtype=torch.int32, device="cuda"),), mode="graph", clock=True, nvml=True,
        nbytes=64 * MB, label="write64MB:graph"))
    return res


def part_a2():
    """Eviction stress: the buffer is read 3x (untimed) right before each flush; compare with
    a never-touched buffer (rotation over > 2x L2 of buffers, clean flush before)."""
    from cobench.kernels import HostGate
    from cobench.timing import _flush, _flush_buffer
    import numpy as np
    buf = _flush_buffer(0, 2 * cb.l2_bytes())
    g8 = HostGate()
    F, Sx = torch.cuda.Stream(), torch.cuda.Stream()

    def run(kind, prep, op, reps=60):
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps + 20)]
        fe = torch.cuda.Event()
        with torch.cuda.stream(F):
            prep()
            if kind:
                _flush(buf, 0, kind)
        with torch.cuda.stream(Sx):
            op()
        torch.cuda.synchronize()
        for rep in range(reps + 20):
            with torch.cuda.stream(F):
                prep()
                if kind:
                    _flush(buf, rep, kind)
                g8.wait(F)
                fe.record(F)
            Sx.wait_event(fe)
            evs[rep][0].record(Sx)
            with torch.cuda.stream(Sx):
                op()
            evs[rep][1].record(Sx)
            F.wait_event(evs[rep][1])
            g8.release()
        torch.cuda.synchronize()
        g8.check()
        return float(np.median([a.elapsed_time(b) * 1e3 for a, b in evs[20:]]))

    res = {}
    for size in (32, 96):
        X = torch.empty(size * MB // 4, dtype=torch.int32, device="cuda").fill_(3)
        nrot = max(3, 2 * cb.l2_bytes() // (size * MB) + 1)
        Xs = [torch.empty(size * MB // 4, dtype=torch.int32, device="cuda").fill_(3) for _ in range(nrot)]
        rd = lambda: cb.read_u4(X)  # noqa: E731
        prep3 = lambda: (cb.read_u4(X), cb.read_u4(X), cb.read_u4(X))  # noqa: E731
        r = {}
        for kind in cb.FLUSH_KINDS:
            r[kind] = {"read_once_before_us": run(kind, lambda: None, rd), "read_3x_before_us": run(kind, prep3, rd)}
        k = [0]

        def rot():
            k[0] += 1
            cb.read_u4(Xs[k[0] % len(Xs)])
        r["never_touched_rotation_us"] = run("clean", lambda: None, rot)
        r["no_flush_us"] = run(None, lambda: None, rd)
        log(f"   evict {size}MB: {r}")
        res[f"read{size}MB"] = r
        del X, Xs
    return res


def ops_setup():
    L = {}
    MM = 4096

    def mk_mm():
        return (torch.randn(MM, MM, device="cuda", dtype=torch.bfloat16),
                torch.randn(MM, MM, device="cuda", dtype=torch.bfloat16),
                torch.empty(MM, MM, device="cuda", dtype=torch.bfloat16))
    L["matmul4096"] = (lambda a, b, c: torch.matmul(a, b, out=c), mk_mm, {"flops": 2 * MM ** 3})

    def mk_copy():
        s = torch.empty(256 * MB, device="cuda", dtype=torch.bfloat16).normal_()
        return s, torch.empty_like(s)
    L["copy512MB"] = (lambda s, d: cb.copy_u4(s, d), mk_copy, {"nbytes": 2 * 512 * MB})
    specs = {}
    for op, st in (("rmsnorm", "T4096_H4096"), ("gqa_decode", "B16_S8192")):
        mod, shape, cfg, rec = C.solo_best(op, st)
        specs[(op, st)] = (mod.build_grid(shape, cfg), rec)
    C.compile_all([s for s, _ in specs.values()])
    for (op, st), (spec, rec) in specs.items():
        la = C.OpLauncher(spec)
        w = C.catalog.work(op, spec.shape, spec.cfg)
        L[f"{op}_{st}"] = (la.run, la.make_args, {"nbytes": w["bytes_min"]}, rec.tag)
    return L


def part_b(passes: int):
    L = ops_setup()
    res = {}
    for p in range(passes):
        names = list(L) if p % 2 == 0 else list(L)[::-1]
        modes = MODES if p % 2 == 0 else MODES[::-1]
        for name in names:
            fn, mk, kw = L[name][:3]
            log(f"pass {p}: {name}")
            r = bench_modes(name, fn, mk, modes, **kw)
            for k, v in r.items():
                res.setdefault(name, {}).setdefault(k, []).append(v)
    cfg_tags = {n: v[3] for n, v in L.items() if len(v) > 3}
    return res, cfg_tags


def part_c():
    res = {}
    buf = torch.empty(2 * cb.l2_bytes() // 4, dtype=torch.int32, device="cuda")
    from cobench.timing import _flush
    for kind in cb.FLUSH_KINDS:
        r = cb.bench(lambda: _flush(buf, 1, kind), mode="hot", clock=True, nvml=True, label=f"flush-{kind}",
                     nbytes=(2 if kind == "write+read" else 1) * buf.nbytes)
        log("   ", r)
        res[f"flush_{kind}"] = row(r)
    noop = cb.CudaKernel('extern "C" __global__ void noop(int* p) { if (p == 0) p[threadIdx.x] = 1; }',
                         "noop", "p")
    sink = torch.zeros(32, dtype=torch.int32, device="cuda")
    f = lambda: noop(1, 32, sink)  # noqa: E731
    res["noop"] = bench_modes("noop", f, None, FLUSHES[:3] + [("hot", None)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--parts", default="abc")
    args = ap.parse_args()
    out = {"meta": C.env_meta(), "gpu_procs_start": C.gpu_procs()}
    cb.wait_until_free()
    t0 = time.time()
    if "a" in args.parts:
        log("== part A: mechanism")
        out["A"] = part_a()
        out["A2"] = part_a2()
    if "b" in args.parts:
        log("== part B: ops")
        out["B"], out["B_cfg"] = part_b(args.passes)
    if "c" in args.parts:
        log("== part C: flush cost / floor")
        out["C"] = part_c()
    out["wall_s"] = time.time() - t0
    out["gpu_procs_end"] = C.gpu_procs()
    out["guard"] = cb.get_guard().summary(t0 - 5, time.time() + 5)
    log("saved", C.save("flush.json", out))


if __name__ == "__main__":
    sys.exit(main())
