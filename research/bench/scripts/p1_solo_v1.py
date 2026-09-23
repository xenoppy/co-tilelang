"""P1 3x2 study, A1: re-validated solo catalog for the study shapes (fixed ClockProbe carveout,
clean flush), C_lib rebuilt, SM-budget curves for the green-split grid, and a steady-mode check
of the solo-best choice.

    source research/env.sh
    python research/bench/scripts/p1_solo_v1.py [--phases 123]

Phase 1  every config of op.configs(shape), grid build, at 188 / 94 / 48 SMs (clean-flush mode,
         the P1-S protocol otherwise: solo_profile.FlushMeter, batch warmup, config-outer order,
         >= 0.3 s + 20 reps warmup and >= 0.25 s + 50 reps per point, pmon guard, flush-only
         baseline for energy). -> C_lib = Pareto(full-GPU time x resource vector) U best@94 U best@48.
Phase 2  C_lib configs at the remaining budgets of the green-split grid (p1_common.BUDGETS).
Phase 3  steady mode (cobench.bench_steady, back-to-back, rotation > 2x L2), full GPU: the top-6
         configs of phase 1 per shape (+ cuBLAS for GEMM). solo-best = fastest in steady mode
         (the study's primary mode); it is added to C_lib if the Pareto front misses it.
Output: <OUT>/solo_v1/points/<op>__<shape>.json (catalog format: `catalog.load(SOLO_DIR)`),
        <OUT>/solo_v1/summary.json.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

import p1_common as C
from p1_common import FULL, cb, catalog, log

import solo_profile as SP  # FlushMeter, PowerLog, Watchdog, measure_point, batch_warmup, ...
from cotile import resources  # noqa: E402

TOP_STEADY = 6


def budget_streams(sms_list):
    """{n_sms: stream}; exact counts via IGNORE_SM_COSCHEDULING (SMs [0, n))."""
    streams, parts = {FULL: torch.cuda.Stream()}, []
    for n in sorted(set(sms_list) - {FULL}):
        p = cb.split_sms(n, ignore_coscheduling=True)
        if p.n_sms != n:
            raise RuntimeError(f"requested {n} SMs, got {p.n_sms}")
        parts.append(p)
        streams[n] = p.stream
    return streams, parts


class Shape:
    def __init__(self, op_name, tag):
        self.op_name, self.tag = op_name, tag
        self.op = C.OPS[op_name]
        self.shape = C.shape_of(op_name, tag)
        self.cfgs = {self.op.cfg_tag(c): c for c in self.op.configs(self.shape)}
        self.specs = {t: self.op.build_grid(self.shape, c) for t, c in self.cfgs.items()}
        self.path = os.path.join(C.SOLO_DIR, "points", f"{op_name}__{tag}.json")


def prepare(shapes):
    specs = [s for sh in shapes for s in sh.specs.values()]
    st = C.compile_all(specs)
    SP._sigs(specs)
    return st


def run_points(sh: Shape, data: C.OpData, keys, streams, meter, wd, args, rec):
    """Measure `keys` = [(tag, sms)] (config-outer order), resumable, contaminated points redone."""
    pts = rec["points"]
    fns = {t: data.launcher(sh.specs[t]) for t in {k[0] for k in keys}}
    todo = [k for k in keys if f"{k[0]}|grid|{k[1]}" not in pts or pts[f"{k[0]}|grid|{k[1]}"].get("contaminated")]
    if not todo:
        return
    est = {}
    for t, n in todo:
        f = fns[t]
        est[(t, n)] = SP._sync_time(lambda: f(0), streams[n])
    wd.wait_quiet()
    t0 = todo[0][0]
    mix = [(k, (lambda f=fns[t0]: f(0), streams[k[1]]), est[k]) for k in todo if k[0] == t0][:4]
    rec["meta"].setdefault("batch_warmup", []).append(SP.batch_warmup(meter, mix, args.batch_warm_s, args.batch_warm_max_s))
    base = f"flush_only|none|{FULL}"
    if base not in pts or pts[base].get("contaminated"):
        pts[base] = SP.measure_point(meter, wd, None, streams[FULL], args, 0.0, attempts_max=1)
    for rnd in range(3):
        for i, (t, n) in enumerate(todo):
            f = fns[t]
            key = f"{t}|grid|{n}"
            pts[key] = SP.measure_point(meter, wd, lambda f=f: f(0), streams[n], args, est[(t, n)])
            if pts[key].get("contaminated"):
                log(f"[{sh.tag}] {key}: foreign SM activity; pausing")
                wd.wait_quiet()
                SP.batch_warmup(meter, mix, args.batch_warm_s, args.batch_warm_max_s)
            if i % 40 == 39:
                SP.resolve_power(meter.power, pts, base)
                SP.save_json(sh.path, rec)
        SP.resolve_power(meter.power, pts, base)
        SP.save_json(sh.path, rec)
        todo = [k for k in todo if pts[f"{k[0]}|grid|{k[1]}"].get("contaminated")]
        if not todo:
            break
        rec["meta"].setdefault("remeasured", []).append(len(todo))


def new_rec(sh: Shape):
    rec = SP.load_json(sh.path) or {}
    if rec.get("schema") != catalog.SCHEMA:
        rec = {"schema": catalog.SCHEMA, "points": {}, "meta": {}}
    rec["op"], rec["shape"], rec["shape_tag"] = sh.op_name, SP.shape_dict(sh.shape), sh.tag
    rec["configs"] = {t: {"cfg": dict(c.__dict__), "sig": {"grid": SP.compact_sig(sh.specs[t].extra["signature"])},
                          "work": catalog.work(sh.op_name, sh.shape, c)} for t, c in sh.cfgs.items()}
    rec["meta"].update({"env": SP.env_meta(), "flush_kind": "clean", "probe": "ClockProbe carveout 100 (methodology v1)",
                        "protocol": {"warm_s": args_g.warm_s, "window_s": args_g.window_s, "tail_s": args_g.tail_s}})
    return rec


def correctness(sh: Shape, data: C.OpData):
    ref = sh.op.reference(sh.shape, data.copies[0][0])
    out_name = data.out_params[0].name
    res = {}
    for t, s in sh.specs.items():
        data.outputs(0)[0].fill_(float("nan"))
        data.launcher(s)(0)
        torch.cuda.synchronize()
        res[t] = SP.compare(data.outputs(0)[0], ref[out_name], sh.op.TOLERANCE)
    bad = [t for t, r in res.items() if not r["ok"]]
    if bad:
        raise RuntimeError(f"{sh.tag}: incorrect configs {bad}")
    return res


def steady_solo(sh: Shape, data: C.OpData, tags, args):
    V = {t: data.launcher(sh.specs[t]) for t in tags}
    if sh.op_name == "gemm":  # cuBLAS reference (torch.matmul, fp32 accumulation, rotation)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        cps = data.copies

        def cublas(i):
            ins, outs = cps[i % len(cps)]
            torch.matmul(ins["A"], ins["B"].t(), out=outs["C"])
        V["ref:cublas_fp32red"] = cublas
    r = cb.bench_steady(V, reference=None, slice_s=args.steady_slice, settle_s=0.4, rounds=args.steady_rounds,
                        warmup_s=5.0, thermal_window_s=10.0, label=f"solo steady {sh.tag}")
    log("\n" + str(r))
    return {"variants": {k: {kk: v.get(kk) for kk in ("t_iter_us", "cv_slices", "clock_mhz", "power_w", "energy_mj_per_iter",
                                                     "iter_p10_us", "iter_p90_us", "n_iter")} for k, v in r.variants.items()},
            "config": r.config, "guard": {"clean": r.guard["clean"], "attempts": r.guard["attempts"]}}


def summarize(shapes, steady: dict) -> dict:
    new = catalog.load(C.SOLO_DIR)
    old = catalog.load()
    out = {"meta": C.env_meta(), "definitions": {
        "times": "clean-flush grid-build medians (us), fixed ClockProbe; 94/48 SMs via IGNORE_SM_COSCHEDULING green contexts",
        "solo_best": "fastest in steady mode (back-to-back, rotation, power-capped) among the top-6 clean-flush configs @188",
        "c_lib": "Pareto(time@188 x (smem, regs*threads, threads, -CTAs/SM)) U best@94 U best@48 U {solo_best}",
    }, "shapes": {}}
    for sh in shapes:
        e = new.get(sh.op_name, sh.tag)
        eo = old.get(sh.op_name, sh.tag)
        st = steady.get(sh.tag, {})
        sv = {k: v["t_iter_us"] for k, v in st.get("variants", {}).items() if not k.startswith("ref:")}
        solo_steady = min(sv, key=sv.get) if sv else e.solo_best.tag
        lib, why = e._c_lib("grid")
        why = {k: list(v) for k, v in why.items()}
        if solo_steady not in why:
            why[solo_steady] = ["solo_best(steady)"]
        else:
            why[solo_steady].append("solo_best(steady)")
        rank = [t for t, _ in e.curve(FULL)]
        d = {
            "solo_best": solo_steady, "solo_best_flush": e.solo_best.tag, "solo_best_p1s": eo.solo_best.tag,
            "flush_rank": {t: i + 1 for i, t in enumerate(rank)},
            "t188": {t: e.configs[t].time("grid", FULL) for t in rank},
            "steady": st,
            "c_lib": sorted(why, key=lambda t: e.configs[t].time("grid", FULL)), "c_lib_reasons": why,
            "c_lib_p1s": [c.tag for c in eo.c_lib], "c_lib_reasons_p1s": eo.c_lib_reasons,
            "budget_best": {str(n): e.budget_best(n).tag for n in e.budgets if n != FULL},
            "budget_best_p1s": {str(n): eo.budget_best(n).tag for n in eo.budgets if n != FULL},
            "budgets": {t: {str(n): e.configs[t].time("grid", n) for n in e.budgets if e.configs[t].time("grid", n)}
                        for t in why},
        }
        # changes vs P1-S at the shared points (ratio new / old), per budget
        ch = {}
        for n in (FULL, 94, 48):
            r = [e.configs[t].time("grid", n) / eo.configs[t].time("grid", n) for t in e.configs
                 if e.configs[t].time("grid", n) and t in eo.configs and eo.configs[t].time("grid", n)]
            if r:
                ch[str(n)] = {"median": float(np.median(r)), "min": float(np.min(r)), "max": float(np.max(r)), "n": len(r)}
        d["new_over_p1s"] = ch
        d["c_lib_added"] = sorted(set(d["c_lib"]) - set(d["c_lib_p1s"]))
        d["c_lib_removed"] = sorted(set(d["c_lib_p1s"]) - set(d["c_lib"]))
        out["shapes"][sh.tag] = d
    return out


def main():
    global args_g
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="123")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--warm-s", type=float, default=0.3)
    ap.add_argument("--window-s", type=float, default=0.25)
    ap.add_argument("--tail-s", type=float, default=0.1)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--batch-warm-s", type=float, default=8.0)
    ap.add_argument("--batch-warm-max-s", type=float, default=40.0)
    ap.add_argument("--global-min-s", type=float, default=45.0)
    ap.add_argument("--steady-slice", type=float, default=1.0)
    ap.add_argument("--steady-rounds", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--redo-steady", action="store_true")
    args = ap.parse_args()
    args_g = args
    sel = [s for s in args.shapes.split(",") if s]
    shapes = [Shape(op, t) for op, tags in C.SHAPES.items() for t in tags if not sel or t in sel]
    log(f"compiling {sum(len(s.specs) for s in shapes)} grid kernels")
    log("compile", prepare(shapes))
    power, wd = SP.PowerLog(), SP.Watchdog()
    meter = SP.FlushMeter(power)
    meter.flush_kind = "clean"
    all_budgets = sorted({n for op in C.BUDGETS for n in C.BUDGETS[op]} | {94, 48})
    streams, parts = budget_streams(all_budgets)
    log("budgets", sorted(streams))
    wd.wait_quiet()
    t_start = time.time()
    runlog = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "phases": {}}
    if not os.path.exists(os.path.join(C.SOLO_DIR, "points")):
        os.makedirs(os.path.join(C.SOLO_DIR, "points"))
    def with_data(sh, fn):
        """fn(data) with the shape's rotation copies; waits for a free GPU (SM idle and enough
        memory) first, releases the copies afterwards, retries after an OOM."""
        def once():
            C.wait_gpu(4 << 30)
            data = C.OpData(sh.op_name, sh.tag)
            try:
                return fn(data)
            finally:
                del data
                torch.cuda.empty_cache()
        return C.retry_oom(once)

    def phase1(sh):
        def fn(data):
            rec = new_rec(sh)
            rec["check"] = correctness(sh, data)
            keys = [(t, n) for t in sh.cfgs for n in (FULL, 94, 48)]
            log(f"[{sh.tag}] phase 1: {len(keys)} points")
            run_points(sh, data, keys, streams, meter, wd, args, rec)
            rec["meta"]["foreign"] = wd.summary(t_start, time.time())
            SP.save_json(sh.path, rec)
        with_data(sh, fn)

    def phase2(sh, cat):
        e = cat.get(sh.op_name, sh.tag)
        lib = [c.tag for c in e.c_lib]
        top = [t for t, _ in e.curve(FULL)[:TOP_STEADY]]   # solo-best candidates, see phase 3
        keys = [(t, n) for t in dict.fromkeys(lib + top) for n in C.BUDGETS[sh.op_name] if n not in (FULL, 94, 48)]

        def fn(data):
            rec = SP.load_json(sh.path)
            log(f"[{sh.tag}] phase 2: {len(keys)} points (C_lib {len(lib)} + top-{TOP_STEADY})")
            run_points(sh, data, keys, streams, meter, wd, args, rec)
            SP.save_json(sh.path, rec)
        with_data(sh, fn)

    if "1" in args.phases or "2" in args.phases:
        C.wait_gpu(2 << 30)
        gw = SP.global_warmup(meter, power, streams[FULL], args.global_min_s, 180.0)
        runlog["global_warmup_s"] = gw["seconds"]
    if "1" in args.phases:
        t1 = time.time()
        for sh in shapes:
            phase1(sh)
        runlog["phases"]["1"] = time.time() - t1
    if "2" in args.phases:
        t2 = time.time()
        cat = catalog.load(C.SOLO_DIR)
        for sh in shapes:
            phase2(sh, cat)
        runlog["phases"]["2"] = time.time() - t2
    steady_path = os.path.join(C.SOLO_DIR, "steady_solo.json")
    steady = SP.load_json(steady_path) or {}
    if "3" in args.phases:
        t3 = time.time()
        cat = catalog.load(C.SOLO_DIR)
        for sh in shapes:
            if sh.tag in steady and not args.redo_steady:
                continue
            e = cat.get(sh.op_name, sh.tag)
            top = [t for t, _ in e.curve(FULL)[:TOP_STEADY]]
            steady[sh.tag] = with_data(sh, lambda data: steady_solo(sh, data, top, args))
            SP.save_json(steady_path, steady, indent=1)
        runlog["phases"]["3"] = time.time() - t3
    runlog["gpu_s"] = time.time() - t_start
    runlog["power_timeline"] = power.timeline()
    old = SP.load_json(os.path.join(C.SOLO_DIR, "runlog.json")) or {"runs": []}
    old["runs"].append(runlog)
    SP.save_json(os.path.join(C.SOLO_DIR, "runlog.json"), old)
    summ = summarize(shapes, steady)
    C.save(os.path.join(C.SOLO_DIR, "summary.json"), summ)
    for tag, d in summ["shapes"].items():
        log(f"{tag}: solo_best {d['solo_best']} (flush {d['solo_best_flush']}, P1-S {d['solo_best_p1s']}); "
            f"C_lib {len(d['c_lib'])} (+{d['c_lib_added']} -{d['c_lib_removed']}); new/P1-S {d['new_over_p1s']}")
    wd.stop()
    power.stop()
    for p in parts:
        p.close()


args_g = None
if __name__ == "__main__":
    sys.exit(main())
