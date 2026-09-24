"""P4-a / P3': solo profiling for the P4 study (causal prefill attention x GQA decode).

    source research/env.sh
    python research/bench/scripts/run_guarded.py -- \\
        python research/bench/scripts/p4_solo.py --phases 1234        # resumable
    python research/bench/scripts/p4_report.py                         # tables

Shapes: prefill S in {2048, 8192} (one request, causal, Hq 32 / Hkv 8 / D 128); decode
B in {16, 64} x KV in {2048, 8192}. SM budgets: the green split grid p4_common.GREEN_P for the
prefill, its complement 188 - n_P for the decode (IGNORE_SM_COSCHEDULING green contexts,
exact SM counts).

Phase 1  clean-flush sweep (A1 protocol: solo_profile.FlushMeter, batch warmup, config-outer
         order, >= 0.3 s + 20 reps warmup and >= 0.25 s + 50 reps per point, pmon guard,
         contaminated points re-measured): every config of the new shapes at 188 / 94 / 48 SMs;
         every config checked against the fp32 reference first. Prefill: every config at every
         budget of its grid (the op is new; its budget curves are part of the deliverable).
Phase 2  decode: C_lib U top-6 at the remaining budgets of the new decode shapes; the A1 points
         of B16_S8192 / B64_S8192 are reused (same 36 configs; copied, marked reused) and only
         the budget A1 did not measure (156 SMs) is added, plus a spot check (A1's top-3 @188 and
         the best @48 re-measured). FlashInfer references at every budget: single prefill,
         batch decode on the CUDA-core and the tensor-core path.
Phase 3  steady mode (cobench.bench_steady, back-to-back, rotation > 2x L2), full GPU, per
         shape: the top-6 clean-flush configs + the FlashInfer references. solo-best = fastest
         TileLang config in steady mode (the P1 definition).
Phase 4  steady mode per P4 pair (8 runs): serial = TileLang prefill solo-best then TileLang
         decode solo-best (reference), the two TileLang solos, serial_fi = FlashInfer prefill
         then the faster FlashInfer decode path, the two FlashInfer solos, and FlashInfer POD
         (patched, research/patches/flashinfer_pod_stream_memset.patch). POD's outputs after the
         run are checked bitwise against a default-stream call.
Output (research/results/2026-09-24_p4_prep): solo/points/<op>__<shape>.json (catalog format),
solo/steady_solo.json, pairs_steady.json, solo/runlog.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

import p4_common as C
from p4_common import FULL, cb, catalog, log

import solo_profile as SP  # FlushMeter, PowerLog, Watchdog, measure_point, batch_warmup, ...
from p1_solo_v1 import budget_streams

TOP_STEADY = 6
YIELD_RC = 75  # run_guarded.py: wait (GuardPolicy) and relaunch
FI_PREFILL = ("fi_prefill",)
FI_DECODE = ("fi_decode_cc", "fi_decode_tc")


class Shape:
    def __init__(self, op_name, tag):
        self.op_name, self.tag = op_name, tag
        self.op = C.P1.OPS[op_name]
        self.shape = C.shape_of(op_name, tag)
        self.cfgs = {self.op.cfg_tag(c): c for c in self.op.configs(self.shape)}
        self.specs = {t: self.op.build_grid(self.shape, c) for t, c in self.cfgs.items()}
        self.path = os.path.join(C.SOLO_DIR, "points", f"{op_name}__{tag}.json")
        self.reused = op_name == "gqa_decode" and tag in C.DECODE_REUSED

    @property
    def refs(self):
        return FI_PREFILL if self.op_name == "prefill_attn" else FI_DECODE

    def fi_data(self, ref: str) -> C.FIData:
        if ref == "fi_prefill":
            return C.FIData("prefill", prefill_tag=self.tag)
        return C.FIData("decode_cc" if ref == "fi_decode_cc" else "decode_tc", decode_tag=self.tag)


def new_rec(sh: Shape, args):
    rec = SP.load_json(sh.path) or {}
    if rec.get("schema") != catalog.SCHEMA:
        rec = {"schema": catalog.SCHEMA, "points": {}, "meta": {}}
    rec["op"], rec["shape"], rec["shape_tag"] = sh.op_name, SP.shape_dict(sh.shape), sh.tag
    rec["configs"] = {t: {"cfg": dict(c.__dict__), "sig": {"grid": SP.compact_sig(sh.specs[t].extra["signature"])},
                          "work": catalog.work(sh.op_name, sh.shape, c)} for t, c in sh.cfgs.items()}
    rec["meta"].update({"env": SP.env_meta(), "flush_kind": "clean", "probe": "ClockProbe carveout 100 (methodology v1)",
                        "protocol": {"warm_s": args.warm_s, "window_s": args.window_s, "tail_s": args.tail_s}})
    return rec


def import_a1(sh: Shape, args):
    """Reused decode shape: start from A1's point file (every config @188/94/48, C_lib U top-6
    at 188 - GREEN_A)."""
    if os.path.exists(sh.path):
        return
    src = os.path.join(C.A1_SOLO_DIR, "points", os.path.basename(sh.path))
    rec = SP.load_json(src)
    rec["meta"]["reused_from"] = os.path.relpath(src, C.ROOT)
    rec["meta"]["reused_note"] = ("A1 points (2026-09-23, same 36 configs); p4_solo adds the missing budgets and a "
                                  "spot check (rec['spot_check'])")
    # refresh config signatures from the current compiled kernels (same configs)
    for t in rec["configs"]:
        rec["configs"][t]["sig"] = {"grid": SP.compact_sig(sh.specs[t].extra["signature"])}
    os.makedirs(os.path.dirname(sh.path), exist_ok=True)
    SP.save_json(sh.path, rec)


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


def ensure_free(wd, rec=None, path=None, quiet_s: float = 5.0):
    """GPU-sharing policy between two measurement points (cobench.GuardPolicy with
    yield_to_caller): if a foreign process showed SM activity within the last quiet_s, save and
    raise cb.GpuYield; run_guarded.py waits 30 min and relaunches (the sweep resumes)."""
    el = time.time() - wd.t_start
    if el < quiet_s:  # need quiet_s of pmon samples first
        time.sleep(quiet_s - el)
    now = time.time()
    act = wd.between(now - quiet_s, now, slack=0.0)
    if act:
        if rec is not None:
            SP.save_json(path, rec)
        raise cb.GpuYield(f"foreign SM activity from {act} ({', '.join(wd._cmd(p) for p in act)})")


def run_points(sh: Shape, fns: dict, keys, streams, meter, wd, args, rec, pts_key="points"):
    """Measure keys = [(name, sms)] with fns[name] = fn() (config-outer order); resumable;
    contaminated points are re-measured (up to 3 rounds)."""
    pts = rec.setdefault(pts_key, {})

    def pkey(name, n):
        return f"{name}|ref|{n}" if name.startswith("ref:") else f"{name}|grid|{n}"

    todo = [k for k in keys if pkey(*k) not in pts or pts[pkey(*k)].get("contaminated")]
    if not todo:
        return
    ensure_free(wd, rec, sh.path)
    est = {k: SP._sync_time(fns[k[0]], streams[k[1]]) for k in todo}
    n0 = todo[0][0]
    mix = [(k, (fns[n0], streams[k[1]]), est[k]) for k in todo if k[0] == n0][:4]
    rec["meta"].setdefault("batch_warmup", []).append(SP.batch_warmup(meter, mix, args.batch_warm_s, args.batch_warm_max_s))
    base = f"flush_only|none|{FULL}"
    if pts_key == "points" and (base not in pts or pts[base].get("contaminated")):
        pts[base] = SP.measure_point(meter, wd, None, streams[FULL], args, 0.0, attempts_max=1)
    for rnd in range(3):
        for i, k in enumerate(todo):
            key = pkey(*k)
            ensure_free(wd, rec, sh.path)
            pts[key] = SP.measure_point(meter, wd, fns[k[0]], streams[k[1]], args, est[k])
            if pts[key].get("contaminated"):
                # yield; the point stays marked contaminated and is re-measured on resume
                log(f"[{sh.tag}] {key}: foreign SM activity; yielding")
                SP.resolve_power(meter.power, rec["points"], base)
                SP.save_json(sh.path, rec)
                raise cb.GpuYield(f"{sh.tag} {key}: contaminated by foreign SM activity {pts[key].get('foreign')}")
            if i % 40 == 39:
                SP.resolve_power(meter.power, rec["points"], base)
                SP.save_json(sh.path, rec)
        SP.resolve_power(meter.power, rec["points"], base)
        if pts_key != "points":
            SP.resolve_power(meter.power, pts, None)
        SP.save_json(sh.path, rec)
        todo = [k for k in todo if pts[pkey(*k)].get("contaminated")]
        if not todo:
            break
        rec["meta"].setdefault("remeasured", []).append(len(todo))


def with_data(sh: Shape, fn, need_gib: float = 4.0):
    """fn(data) with the shape's TileLang rotation copies; waits for a free GPU (SM idle and
    enough memory) first, releases the copies afterwards, retries after an OOM."""
    def once():
        C.wait_gpu(int(need_gib * (1 << 30)))
        data = C.OpData(sh.op_name, sh.tag)
        try:
            return fn(data)
        finally:
            del data
            torch.cuda.empty_cache()
    return C.retry_oom(once)


def summarize_shape(sh: Shape, steady: dict) -> dict:
    cat = catalog.load(C.SOLO_DIR)
    e = cat.get(sh.op_name, sh.tag)
    st = steady.get(sh.tag, {})
    sv = {k: v["t_iter_us"] for k, v in st.get("variants", {}).items() if not k.startswith("ref:")}
    solo_steady = min(sv, key=sv.get) if sv else e.solo_best.tag
    lib, why = e._c_lib("grid")
    why = {k: list(v) for k, v in why.items()}
    why.setdefault(solo_steady, []).append("solo_best(steady)")
    rank = [t for t, _ in e.curve(FULL)]
    refs = {n: {str(s): m.median for s, m in d.items()} for n, d in e.refs.items()}
    return {
        "solo_best": solo_steady, "solo_best_flush": e.solo_best.tag,
        "flush_rank": {t: i + 1 for i, t in enumerate(rank)},
        "t188": {t: e.configs[t].time("grid", FULL) for t in rank},
        "steady": st,
        "c_lib": sorted(why, key=lambda t: e.configs[t].time("grid", FULL)), "c_lib_reasons": why,
        "budget_best": {str(n): (e.budget_best(n).tag, e.budget_best(n).time("grid", n)) for n in e.budgets},
        "budget_loss_of_full_best": {str(n): e.budget_loss(n) for n in e.budgets if n != FULL},
        "budgets": {t: {str(n): e.configs[t].time("grid", n) for n in e.budgets if e.configs[t].time("grid", n)}
                    for t in why},
        "refs_flush": refs,
        "reused_from": e.meta.get("reused_from"),
    }


def steady_solo(sh: Shape, data: C.OpData, tags, args):
    V = {t: data.launcher(sh.specs[t]) for t in tags}
    fis = {}
    for r in sh.refs:
        fis[r] = sh.fi_data(r)
        V["ref:" + r] = fis[r].launcher()
    res = cb.bench_steady(V, reference=None, slice_s=args.steady_slice, settle_s=0.4, rounds=args.steady_rounds,
                          warmup_s=5.0, thermal_window_s=10.0, label=f"solo steady {sh.tag}")
    log("\n" + str(res))
    out = {"variants": {k: {kk: v.get(kk) for kk in ("t_iter_us", "cv_slices", "clock_mhz", "power_w",
                                                     "energy_mj_per_iter", "iter_p10_us", "iter_p90_us", "n_iter",
                                                     "kcycles_per_iter")} for k, v in res.variants.items()},
           "config": res.config, "guard": {"clean": res.guard["clean"], "attempts": res.guard["attempts"]},
           "fi_check": {r: C.fi_check(d, sh.op.TOLERANCE) for r, d in fis.items()}}
    del fis
    return out


def pair_steady(pair: str, summ: dict, steady: dict, args) -> dict:
    """One steady run per P4 pair: TileLang serial (reference) and solos, FlashInfer serial and
    solos, FlashInfer POD (patched)."""
    ptag, dtag = C.PAIRS[pair]
    shp = Shape("prefill_attn", ptag)
    shd = Shape("gqa_decode", dtag)
    tp, td = summ[ptag]["solo_best"], summ[dtag]["solo_best"]
    fiv = steady[dtag]["variants"]
    fi_dec = min(FI_DECODE, key=lambda r: fiv["ref:" + r]["t_iter_us"])
    C.wait_gpu(12 << 30)
    dp, dd = C.OpData("prefill_attn", ptag), C.OpData("gqa_decode", dtag)
    C.compile_all([shp.specs[tp], shd.specs[td]])
    A, B = dp.launcher(shp.specs[tp]), dd.launcher(shd.specs[td])
    fp, fd = shp.fi_data("fi_prefill"), shd.fi_data(fi_dec)
    pod = C.FIData("pod", prefill_tag=ptag, decode_tag=dtag)
    FA, FB, POD = fp.launcher(), fd.launcher(), pod.launcher()

    def serial(i):
        A(i)
        B(i)

    def serial_fi(i):
        FA(i)
        FB(i)

    V = {"serial": serial, "prefill": A, "decode": B, "serial_fi": serial_fi, "prefill_fi": FA, "decode_fi": FB,
         "pod": POD}
    res = cb.bench_steady(V, reference="serial", slice_s=args.pair_slice, settle_s=0.4, rounds=args.pair_rounds,
                          warmup_s=5.0, thermal_window_s=10.0, label=f"P4 pair {pair}")
    log("\n" + str(res))
    # POD outputs after the run: copy 0 on the default stream vs a steady-stream call
    x = pod.copies[0]
    torch.cuda.synchronize()
    o1 = pod.w.run(*x)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        o2 = pod.w.run(*x)
    torch.cuda.synchronize()
    pod_ok = bool(torch.equal(o1[0], o2[0]) and torch.equal(o1[1], o2[1]))
    out = {"pair": pair, "prefill": ptag, "decode": dtag, "cfg_prefill": tp, "cfg_decode": td, "fi_decode_path": fi_dec,
           "variants": {k: {kk: v.get(kk) for kk in ("t_iter_us", "cv_slices", "clock_mhz", "power_w",
                                                     "energy_mj_per_iter", "iter_p10_us", "iter_p90_us", "n_iter",
                                                     "kcycles_per_iter")} for k, v in res.variants.items()},
           "speedup_vs_serial": res.derived.get("speedup"), "config": res.config,
           "guard": {"clean": res.guard["clean"], "attempts": res.guard["attempts"]},
           "pod_side_stream_bitwise_eq_default": pod_ok, "copies": {"tl_prefill": dp.n, "tl_decode": dd.n,
                                                                     "fi_prefill": fp.n, "fi_decode": fd.n, "pod": pod.n}}
    del dp, dd, fp, fd, pod, A, B, FA, FB, POD, V
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="1234")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--warm-s", type=float, default=0.3)
    ap.add_argument("--window-s", type=float, default=0.25)
    ap.add_argument("--tail-s", type=float, default=0.1)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--batch-warm-s", type=float, default=8.0)
    ap.add_argument("--batch-warm-max-s", type=float, default=40.0)
    ap.add_argument("--global-min-s", type=float, default=45.0)
    ap.add_argument("--steady-slice", type=float, default=1.0)
    ap.add_argument("--steady-rounds", type=int, default=3)
    ap.add_argument("--pair-slice", type=float, default=1.0)
    ap.add_argument("--pair-rounds", type=int, default=3)
    ap.add_argument("--redo-steady", action="store_true")
    ap.add_argument("--redo-pairs", action="store_true")
    args = ap.parse_args()
    cb.set_policy(yield_to_caller=True)  # run_guarded.py waits and relaunches after a yield
    try:
        return _main(args)
    except cb.GpuYield as e:
        log(f"yielding the GPU: {str(e)[:300]}")
        rl = SP.load_json(os.path.join(C.SOLO_DIR, "runlog.json")) or {"runs": []}
        rl.setdefault("yields", []).append({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "detail": str(e)[:300]})
        SP.save_json(os.path.join(C.SOLO_DIR, "runlog.json"), rl)
        return YIELD_RC


def _main(args):
    sel = [s for s in args.shapes.split(",") if s]
    shapes = [Shape(op, t) for op, tags in C.SHAPES.items() for t in tags if not sel or t in sel]
    specs = [s for sh in shapes for s in sh.specs.values()]
    log(f"compiling {len(specs)} grid kernels")
    log("compile", C.compile_all(specs))
    SP._sigs(specs)
    os.makedirs(os.path.join(C.SOLO_DIR, "points"), exist_ok=True)
    runlog = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "phases": {}, "args": vars(args)}
    t_start = time.time()
    power = wd = meter = None
    parts = []
    streams = None
    if "1" in args.phases or "2" in args.phases:
        power, wd = SP.PowerLog(), SP.Watchdog()
        meter = SP.FlushMeter(power)
        meter.flush_kind = "clean"
        all_budgets = sorted({n for op in C.BUDGETS for n in C.BUDGETS[op]})
        streams, parts = budget_streams(all_budgets)
        log("budgets", sorted(streams))
        C.wait_gpu(4 << 30)
        ensure_free(wd)
        gw = SP.global_warmup(meter, power, streams[FULL], args.global_min_s, 180.0)
        runlog["global_warmup_s"] = gw["seconds"]

    def ref_fns(sh):
        fis = {r: sh.fi_data(r) for r in sh.refs}
        return fis, {"ref:" + r: (lambda d=d: d.w.run(*d.copies[0])) for r, d in fis.items()}

    if "1" in args.phases:
        t1 = time.time()
        for sh in shapes:
            if sh.reused:
                continue

            def fn(data, sh=sh):
                rec = new_rec(sh, args)
                if "check" not in rec:
                    rec["check"] = correctness(sh, data)
                budgets = C.BUDGETS[sh.op_name] if sh.op_name == "prefill_attn" else C.PHASE1_BUDGETS
                keys = [(t, n) for t in sh.cfgs for n in budgets]
                fns = {t: (lambda f=data.launcher(sh.specs[t]): f(0)) for t in sh.cfgs}
                log(f"[{sh.tag}] phase 1: {len(keys)} points")
                run_points(sh, fns, keys, streams, meter, wd, args, rec)
                rec["meta"]["foreign"] = wd.summary(t_start, time.time())
                SP.save_json(sh.path, rec)
            with_data(sh, fn)
        runlog["phases"]["1"] = time.time() - t1
    if "2" in args.phases:
        t2 = time.time()
        for sh in shapes:
            if sh.reused:
                import_a1(sh, args)
            cat = catalog.load(C.SOLO_DIR)
            e = cat.get(sh.op_name, sh.tag)
            lib = [c.tag for c in e.c_lib]
            top = [t for t, _ in e.curve(FULL)[:TOP_STEADY]]

            def fn(data, sh=sh, e=e, lib=lib, top=top):
                rec = SP.load_json(sh.path)
                fns = {t: (lambda f=data.launcher(sh.specs[t]): f(0)) for t in sh.cfgs}
                keys = []
                if sh.op_name == "gqa_decode":
                    keys = [(t, n) for t in dict.fromkeys(lib + top) for n in C.BUDGETS[sh.op_name]
                            if n not in C.PHASE1_BUDGETS]
                log(f"[{sh.tag}] phase 2: {len(keys)} TileLang points (C_lib {len(lib)} + top-{TOP_STEADY}), "
                    f"missing ones only")
                run_points(sh, fns, keys, streams, meter, wd, args, rec)
                if sh.reused:
                    # spot check of the reused A1 points: top-3 @188 and the best @48
                    spot = [(t, FULL) for t in top[:3]] + [(e.budget_best(48).tag, 48)]
                    run_points(sh, fns, spot, streams, meter, wd, args, rec, pts_key="spot_check")
                fis, rfns = ref_fns(sh)
                rkeys = [(r, n) for r in rfns for n in C.BUDGETS[sh.op_name]]
                rec["ref_check"] = {r[4:]: C.fi_check(fis[r[4:]], sh.op.TOLERANCE) for r in rfns}
                bad = [r for r, v in rec["ref_check"].items() if not v["ok"]]
                if bad:
                    raise RuntimeError(f"{sh.tag}: FlashInfer reference outputs out of tolerance: {bad}")
                run_points(sh, rfns, rkeys, streams, meter, wd, args, rec)
                del fis, rfns
                SP.save_json(sh.path, rec)
            with_data(sh, fn, need_gib=8.0)
        runlog["phases"]["2"] = time.time() - t2
    for p in parts:
        p.close()
    if wd is not None:
        wd.stop()
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
            steady[sh.tag] = with_data(sh, lambda data, sh=sh, top=top: steady_solo(sh, data, top, args), need_gib=8.0)
            SP.save_json(steady_path, steady, indent=1)
        runlog["phases"]["3"] = time.time() - t3
    summ = {sh.tag: summarize_shape(sh, steady) for sh in shapes}
    C.P1.save(os.path.join(C.SOLO_DIR, "summary.json"), {"meta": C.P1.env_meta(), "shapes": summ})
    if "4" in args.phases:
        t4 = time.time()
        pairs_path = os.path.join(C.OUT, "pairs_steady.json")
        pr = SP.load_json(pairs_path) or {}
        psel = [p for p in args.pairs.split(",") if p] or list(C.PAIRS)
        for p in psel:
            if p in pr and not args.redo_pairs:
                continue
            pr[p] = C.retry_oom(pair_steady, p, summ, steady, args)
            SP.save_json(pairs_path, pr, indent=1)
        runlog["phases"]["4"] = time.time() - t4
    runlog["gpu_s"] = time.time() - t_start
    if power is not None:
        runlog["power_timeline"] = power.timeline()
        power.stop()
    old = SP.load_json(os.path.join(C.SOLO_DIR, "runlog.json")) or {"runs": []}
    old["runs"].append(runlog)
    SP.save_json(os.path.join(C.SOLO_DIR, "runlog.json"), old)
    log("done", json.dumps({k: round(v, 1) for k, v in runlog["phases"].items()}), f"gpu_s {runlog['gpu_s']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
