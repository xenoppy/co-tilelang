"""M3 (methodology v1): steady-state co-run measurement (cobench.bench_steady) on the P1 study
point, compared with clean-flush mode (cobench.bench_corun), plus its validations.

    source research/env.sh
    python research/bench/scripts/mv1_steady.py p1        [--rounds 5 --slice 1.5]
    python research/bench/scripts/mv1_steady.py validate  # (a) serial ~ sum of solos
    python research/bench/scripts/mv1_steady.py repro --procs 3   # (b) across processes

P1 point: GEMM 4096x4096x4096 x GQA decode B16xS8192 (Hq 32, Hkv 8, D 128), catalog solo-best
grid configs for both ops (methodology smoke test, not the 3x2 study). Variants:
  serial                 A then B on one stream
  solo_a / solo_b        A alone / B alone (back-to-back)
  streams_ab / _ba       A on s1 || B on s2, joined per iteration (host launch order A,B / B,A)
  green_<nA>             A on an nA-SM green context || B on the complement
                         (CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING: contiguous SMs)
  co_to_<nA>             CoKernel, SM binding, dynamic queue (chunk 1), takeover, nA SMs -> A
  co_<nA>                same without takeover
Results -> research/results/2026-09-23_methodology_v1/steady_*.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

import mv1_common as C
from mv1_common import cb, log

GREEN = (60, 80, 100, 120, 140)
CO_TO = (60, 94, 128, 160)
CO_NOTO = (94, 128)


def _io(launcher, args):
    ins, outs = {}, {}
    for p, a in zip(launcher.spec.params, args):
        if p.role == "in":
            ins[p.name] = a
        elif p.role == "out":
            outs[p.name] = a
    return ins, outs


def setup_pair(a_op, a_shape, b_op, b_shape, green=GREEN, co_to=CO_TO, co_noto=CO_NOTO, small=False):
    from cotile.cokernel import CoRunner, Orch, build_cokernel, sm_role_table

    ga, gs, gc, grec = C.solo_best(a_op, a_shape)
    da, ds, dc, drec = C.solo_best(b_op, b_shape)
    specs = {"a": ga.build_grid(gs, gc), "b": da.build_grid(ds, dc)}
    orchs = {}
    if co_to:
        orchs["to"] = Orch(binding="sm", schedule="dynamic", chunk=1, takeover=True, num_ctas=188, timing=False)
    if co_noto:
        orchs["noto"] = Orch(binding="sm", schedule="dynamic", chunk=1, takeover=False, num_ctas=188, timing=False)
    for k, o in orchs.items():
        specs["co_" + k] = build_cokernel(ga, gs, gc, da, ds, dc, o)
    C.compile_all(list(specs.values()))
    LA, LB = C.OpLauncher(specs["a"]), C.OpLauncher(specs["b"])
    ra, rb = cb.Rotation(LA.make_args), cb.Rotation(LB.make_args)
    fa = lambda i: LA.run(*ra[i])  # noqa: E731
    fb = lambda i: LB.run(*rb[i])  # noqa: E731
    V = {}

    def serial(i):
        fa(i)
        fb(i)
    V["serial"] = serial
    V["solo_a"] = fa
    V["solo_b"] = fb
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    V["streams_ab"] = cb.Par(("a", s1, fa), ("b", s2, fb))
    V["streams_ba"] = cb.Par(("a", s1, fa), ("b", s2, fb), order="reverse")
    parts = []
    for n in green:
        p = cb.split_sms(n, ignore_coscheduling=True)
        parts.append(p)
        V[f"green_{p.n_sms}"] = cb.Par(("a", p.stream, fa), ("b", p.rest_stream, fb))
    # CoKernels: one runner (own knobs + counters) per split; args prebuilt per rotation copy
    ncopy = max(ra.n, rb.n)
    co_meta = {}
    for kind, splits in (("to", co_to), ("noto", co_noto)):
        for n in splits:
            spec = specs["co_" + kind]
            run = CoRunner(spec)
            run.set_knobs(sm_role=sm_role_table(n))
            arglists = []
            for k in range(ncopy):
                ia, oa = _io(LA, ra[k])
                ib, ob = _io(LB, rb[k])
                args, _ = run.make_args([ia, ib], [oa, ob])
                arglists.append(args)
            name = f"co_to_{n}" if kind == "to" else f"co_{n}"
            V[name] = CoVariant(run, arglists)
            co_meta[name] = {"n_a": n, "orch": run.info.orch.tag(), "copies": len(arglists)}
    # correctness: every variant's outputs == the grid kernels' outputs (copy 0)
    ref_a = _outs(LA, ra[0], fa)
    ref_b = _outs(LB, rb[0], fb)
    for name, f in V.items():
        if name.startswith("co_"):
            _zero_outs(LA, ra[0])
            _zero_outs(LB, rb[0])
            f(0)
            torch.cuda.synchronize()
            oa, ob = _get_outs(LA, ra[0]), _get_outs(LB, rb[0])
            if not (all(torch.equal(x, y) for x, y in zip(oa, ref_a)) and all(torch.equal(x, y) for x, y in zip(ob, ref_b))):
                raise RuntimeError(f"{name}: outputs differ from the grid kernels")
    info = {"a": {"op": a_op, "shape": a_shape, "cfg": grec.tag, "catalog_us": grec.time()},
            "b": {"op": b_op, "shape": b_shape, "cfg": drec.tag, "catalog_us": drec.time()},
            "rotation": {"a_copies": ra.n, "a_bytes_per_copy": ra.bytes_per_copy, "b_copies": rb.n,
                         "b_bytes_per_copy": rb.bytes_per_copy},
            "green": {f"green_{p.n_sms}": {"requested": p.requested, "n_a": p.n_sms, "n_b": p.n_rest} for p in parts},
            "cokernel": co_meta}
    if orchs:
        from cotile import resources
        info["cokernel_sig"] = {k: {kk: v for kk, v in resources.signature(specs[k]).items()
                                    if kk in ("regs", "smem_dynamic", "ctas_per_sm", "threads")}
                                for k in specs if k.startswith("co_")}
    return V, info, parts, (fa, fb)


class CoVariant:
    """One CoKernel launch per iteration; argument lists prebuilt per rotation copy (own
    runner = own knobs and counters, so several SM splits can be interleaved)."""

    def __init__(self, runner, arglists):
        self.runner, self.kernel, self.arglists = runner, runner.spec.kernel, arglists

    def __call__(self, i):
        self.kernel(*self.arglists[i % len(self.arglists)])


def _outs(L, args, f):
    _zero_outs(L, args)
    f(0)
    torch.cuda.synchronize()
    return [x.clone() for x in _get_outs(L, args)]


def _get_outs(L, args):
    return [a for p, a in zip(L.spec.params, args) if p.role == "out"]


def _zero_outs(L, args):
    for x in _get_outs(L, args):
        x.zero_()


def steady_summary(r):
    return {"variants": r.variants, "derived": r.derived, "config": r.config, "guard": {"clean": (r.guard or {}).get("clean"),
            "attempts": (r.guard or {}).get("attempts")}, "clock": r.clock, "nvml": r.nvml, "slices": r.slices}


def cmd_p1(args):
    V, info, parts, (fa, fb) = setup_pair("gemm", "M4096_N4096_K4096", "gqa_decode", "B16_S8192")
    out = {"meta": C.env_meta(), "setup": info, "gpu_procs": C.gpu_procs()}
    log("steady:", list(V))
    r = cb.bench_steady(V, reference="serial", slice_s=args.slice, settle_s=args.settle, rounds=args.rounds,
                        warmup_s=args.warmup, label="P1 steady")
    log("\n" + str(r))
    out["steady"] = steady_summary(r)
    C.save("steady_p1.json", out)
    # clean-flush mode: bench_corun (serial, solo_a, solo_b, corun = streams with alternating
    # launch order) + the green / CoKernel variants as extras, all interleaved per rep
    extra = {k: v for k, v in V.items() if k.startswith(("green_", "co_"))}
    extra["streams_ab"] = V["streams_ab"]
    extra["streams_ba"] = V["streams_ba"]
    rc = cb.bench_corun(lambda: fa(0), lambda: fb(0), extra=extra, flush_kind="clean", clock=True, nvml=True,
                        reps=args.flush_reps, label="P1 clean flush")
    log("\n" + str(rc))
    out["flush"] = rc.to_dict()
    out["flush_clock_per_variant"] = {k: {kk: v.get(kk) for kk in ("median", "min", "max", "cycles_median")}
                                      for k, v in (rc.clock or {}).get("per_variant", {}).items()}
    log("saved", C.save("steady_p1.json", out))
    for p in parts:
        p.close()


def cmd_validate(args):
    """(a) serial vs sum of solos: a non-power-capped pair (decode B16_S8192 + RMSNorm
    T16384_H4096, both memory-bound, below the cap) and the P1 pair (from cmd_p1's run)."""
    V, info, parts, _ = setup_pair("gqa_decode", "B16_S8192", "rmsnorm", "T16384_H4096", green=(), co_to=(), co_noto=())
    V = {k: V[k] for k in ("serial", "solo_a", "solo_b", "streams_ab")}
    r = cb.bench_steady(V, reference="serial", slice_s=args.slice, settle_s=args.settle, rounds=args.rounds,
                        warmup_s=args.warmup, label="decode+rmsnorm steady")
    log("\n" + str(r))
    v = r.variants
    out = {"meta": C.env_meta(), "setup": info, "steady": steady_summary(r),
           "serial_over_sum_solo": v["serial"]["t_iter_us"] / (v["solo_a"]["t_iter_us"] + v["solo_b"]["t_iter_us"])}
    log("serial / (solo_a + solo_b) =", out["serial_over_sum_solo"])
    log("saved", C.save("steady_validate.json", out))


def cmd_prio(args):
    """Two-stream co-runs are bimodal (which kernel the hardware dispatches first). Does a
    stream priority make the order deterministic? streams_{ab,ba}: equal priority, host launch
    order A,B / B,A; prio_a / prio_b: A (resp. B) on a high-priority stream."""
    V, info, parts, (fa, fb) = setup_pair("gemm", "M4096_N4096_K4096", "gqa_decode", "B16_S8192", green=(),
                                          co_to=(), co_noto=())
    lo, hi = torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=-1)
    lo2, hi2 = torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=-1)
    V = {"serial": V["serial"], "streams_ab": V["streams_ab"], "streams_ba": V["streams_ba"],
         "prio_a": cb.Par(("a", hi, fa), ("b", lo, fb)), "prio_b": cb.Par(("a", lo2, fa), ("b", hi2, fb))}
    r = cb.bench_steady(V, reference="serial", slice_s=args.slice, settle_s=args.settle, rounds=args.rounds,
                        warmup_s=args.warmup, label="stream priority")
    log("\n" + str(r))
    rc = cb.bench_variants(V, reference="serial", clock=True, reps=args.flush_reps, label="stream priority, clean flush")
    log("\n" + str(rc))
    out = {"meta": C.env_meta(), "setup": info, "steady": steady_summary(r), "flush": rc.to_dict(),
           "priority_range": torch.cuda.Stream.priority_range()}
    log("saved", C.save("steady_stream_priority.json", out))


def cmd_child(args):
    V, info, parts, _ = setup_pair("gemm", "M4096_N4096_K4096", "gqa_decode", "B16_S8192", green=(100,), co_to=(128,),
                                   co_noto=())
    keep = ["serial", "solo_a", "solo_b", "streams_ab", "green_100", "co_to_128"]
    V = {k: V[k] for k in keep}
    r = cb.bench_steady(V, reference="serial", slice_s=args.slice, settle_s=args.settle, rounds=args.rounds,
                        warmup_s=args.warmup, label=f"repro {os.getpid()}")
    print("RESULT " + json.dumps({"variants": {k: {kk: x[kk] for kk in ("t_iter_us", "cv_slices", "clock_mhz", "power_w")}
                                               for k, x in r.variants.items()},
                                  "speedup": {k: x["ratio_of_medians"] for k, x in r.derived["speedup"].items()},
                                  "guard_clean": r.guard["clean"], "wall": time.time()}), flush=True)


def cmd_repro(args):
    runs = []
    for i in range(args.procs):
        cmd = [sys.executable, __file__, "child", "--rounds", str(args.rounds), "--slice", str(args.slice),
               "--settle", str(args.settle), "--warmup", str(args.warmup)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
        if p.returncode != 0 or not line:
            raise RuntimeError(f"child {i} failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        runs.append(json.loads(line[0][7:]))
        log(f"process {i}: " + ", ".join(f"{k} {v['t_iter_us']:.1f}" for k, v in runs[-1]["variants"].items()))

    def cv(x):
        a = np.asarray(x, float)
        return float(a.std(ddof=1) / a.mean())
    names = list(runs[0]["variants"])
    out = {"meta": C.env_meta(), "processes": len(runs), "runs": runs,
           "cv_across_processes": {"t_iter": {k: cv([r["variants"][k]["t_iter_us"] for r in runs]) for k in names},
                                   "speedup": {k: cv([r["speedup"][k] for r in runs]) for k in names if k != "serial"},
                                   "clock": {k: cv([r["variants"][k]["clock_mhz"] for r in runs]) for k in names}}}
    log(json.dumps(out["cv_across_processes"], indent=1))
    log("saved", C.save("steady_repro.json", out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["p1", "validate", "repro", "child", "prio"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--slice", type=float, default=1.5)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--flush-reps", type=int, default=60)
    ap.add_argument("--procs", type=int, default=3)
    args = ap.parse_args()
    cb.wait_until_free()
    {"p1": cmd_p1, "validate": cmd_validate, "repro": cmd_repro, "child": cmd_child, "prio": cmd_prio}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
