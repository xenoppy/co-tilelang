"""Timing of the FlashInfer baselines on the RTX PRO 6000 (sm_120) with cobench (criteria 2-4).

  decode   BatchDecode (CUDA-core / tensor-core), B in {16, 64}, KV in {2048, 8192}: flush + graph
  prefill  SinglePrefill causal, S in {2048, 8192}, H_kv in {8, 32}: flush + graph
  pod      for prefill S in {2048, 8192} x decode (B, KV) in {16, 64} x {2048, 8192}:
             POD            one fused launch, cobench flush mode on the legacy default stream
                            (graph mode is invalid for POD, see README)
             serial         prefill then decode on one stream (cobench flush), with the fastest
                            standalone decode ("best") and with the tensor-core decode ("tc",
                            the kernel POD embeds)
             streams        bench_corun(prefill, decode_best) on two torch side streams
             green          bench_corun on a green-context split: prefill on n SMs
                            (IGNORE_SM_COSCHEDULING, SMs [0, n)), decode on the other 188-n
Every timed batch runs under gpu_guard (no foreign GPU process before/during/after, else
re-measured); every point is re-measured up to 3x if CV > 2%. After each timed batch the
last outputs are checked against a reference computed before timing (catches stream races).
Writes timing.json (incrementally).
Run: source research/env.sh && python research/results/2026-09-22_flashinfer_baselines/run_timing.py [sections]
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "bench")))
sys.path.insert(0, HERE)
import cobench as cb  # noqa: E402
from baselines import flashinfer_ops as fo  # noqa: E402
import gpu_guard  # noqa: E402

OUT = os.path.join(HERE, "timing.json")
DECODE_SHAPES = [(16, 2048), (16, 8192), (64, 2048), (64, 8192)]
PREFILL_SEQS = [2048, 8192]
GREEN_PREFILL_SMS = [28, 60, 96, 128, 160]


def log(*a, **k):
    k.pop("flush", None)
    print(time.strftime("%H:%M:%S"), *a, **k, flush=True)


def load():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def save(res):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, OUT)


def bstats(r):
    c = (r.clock or {}).get("per_rep", {})
    return {"median_us": r.median, "p10_us": r.p10, "p90_us": r.p90, "cv": r.cv, "cv_ok": r.cv_ok,
            "gbps": r.gbps, "tflops": r.tflops, "mode": r.mode, "l2": r.l2, "k": r.k, "n_copies": r.n_copies,
            "clock_mhz_median": c.get("median"), "clock_mhz_min": c.get("min"), "clock_mhz_max": c.get("max"),
            "cycles_median": c.get("cycles_median"), "corr_time_vs_clock": c.get("corr_time_vs_clock")}


def cstats(c):
    clk = (c.clock or {}).get("per_variant", {})
    d = {"makespan": c.corun["makespan"], "a_end": c.corun["a_end"], "b_end": c.corun["b_end"],
         "by_order": c.corun["by_order"], "solo_a": c.solo_a, "solo_b": c.solo_b, "serial": c.serial,
         "derived": c.derived,
         "clock_mhz_median": {v: x.get("median") for v, x in clk.items()}}
    return d


def retry(measure, label, cv_of, attempts=3):
    """guarded measurement, re-measured while CV > 2% (keeps the lowest-CV attempt)."""
    best, tries = None, []
    for _ in range(attempts):
        r, info = gpu_guard.guarded(measure, label=label, log=log)
        cv = cv_of(r)
        tries.append({"cv": cv, "guard": info})
        if best is None or cv < cv_of(best):
            best = r
        if cv <= 0.02:
            break
    return best, tries


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# ---------------------------------------------------------------------------
def section_decode(res):
    out = res.setdefault("decode", {})
    for B, S in DECODE_SHAPES:
        for tc in (False, True):
            key = f"B{B}_S{S}_{'tc' if tc else 'cc'}"
            if key in out:
                continue
            gpu_guard.wait_until_free(log=log)      # also before untimed setup work
            d = fo.BatchDecode(fo.DecodeShape(batch=B, kv_len=S), use_tensor_cores=tc)
            rec = {"describe": d.describe()}
            x = d.make_inputs(seed=1)
            ref = d.run(*x).clone()
            torch.cuda.synchronize()
            for mode in ("flush", "graph"):
                r, tries = retry(lambda: cb.bench(d.run, make_inputs=d.make_inputs, mode=mode, nbytes=d.nbytes,
                                                  flops=d.flops, clock=True, label=key),
                                 f"decode {key} {mode}", lambda r: r.cv)
                rec[mode] = {**bstats(r), "tries": tries}
                log(f"decode {key} {mode}: {r}")
            o = d.run(*x)
            torch.cuda.synchronize()
            rec["post_check_max_diff"] = maxdiff(o, ref)
            out[key] = rec
            save(res)
            del d, x, ref, o
            torch.cuda.empty_cache()


def section_prefill(res):
    out = res.setdefault("prefill", {})
    for S in PREFILL_SEQS:
        for hkv in (8, 32):
            key = f"S{S}_Hkv{hkv}"
            if key in out:
                continue
            gpu_guard.wait_until_free(log=log)
            p = fo.SinglePrefill(fo.PrefillShape(seq_len=S, num_kv_heads=hkv))
            rec = {"describe": p.describe()}
            for mode in ("flush", "graph"):
                r, tries = retry(lambda: cb.bench(p.run, make_inputs=p.make_inputs, mode=mode, flops=p.flops,
                                                  nbytes=p.nbytes, clock=True, label=key),
                                 f"prefill {key} {mode}", lambda r: r.cv)
                rec[mode] = {**bstats(r), "tries": tries}
                log(f"prefill {key} {mode}: {r}")
            out[key] = rec
            save(res)
            del p
            torch.cuda.empty_cache()


def best_decode_is_tc(res, B, S):
    """fastest standalone decode path in flush mode (from the decode section)."""
    dd = res["decode"]
    cc, tc = dd[f"B{B}_S{S}_cc"]["flush"]["median_us"], dd[f"B{B}_S{S}_tc"]["flush"]["median_us"]
    return tc < cc


class Stash:
    """wraps a callable and keeps a reference to its last return value (no sync)."""

    def __init__(self, fn):
        self.fn, self.last = fn, None

    def __call__(self, *a):
        self.last = self.fn(*a)
        return self.last


def section_pod(res):
    out = res.setdefault("pod", {})
    for Sp in PREFILL_SEQS:
        for B, S in DECODE_SHAPES:
            key = f"P{Sp}_B{B}_S{S}"
            rec = out.setdefault(key, {})
            if rec.get("done"):
                continue
            gpu_guard.wait_until_free(log=log)
            ps, ds = fo.PrefillShape(seq_len=Sp), fo.DecodeShape(batch=B, kv_len=S)
            pod = fo.POD(ps, ds)
            pre = fo.SinglePrefill(ps)
            use_tc = best_decode_is_tc(res, B, S)
            dec_best = fo.BatchDecode(ds, use_tensor_cores=use_tc)
            dec_tc = dec_best if use_tc else fo.BatchDecode(ds, use_tensor_cores=True)
            rec["best_decode"] = "tc" if use_tc else "cc"
            rec["flops"], rec["nbytes"] = pod.flops, pod.nbytes
            # one input set shared by every variant (flush mode: inputs are not rotated)
            x = pod.make_inputs(seed=7)
            xp, xd = pod.split_inputs(x)
            out_d = torch.empty_like(xd[0])
            ref_p, ref_d = pod.run(*x)
            ref_p, ref_d = ref_p.clone(), ref_d.clone()
            torch.cuda.synchronize()

            # ---- POD (single launch) --------------------------------------------------
            if "pod" not in rec:
                f = Stash(pod.run)
                r, tries = retry(lambda: cb.bench(f, make_inputs=lambda: x, mode="flush", flops=pod.flops,
                                                  nbytes=pod.nbytes, clock=True, label=key + "-pod"),
                                 key + " pod", lambda r: r.cv)
                torch.cuda.synchronize()
                rec["pod"] = {**bstats(r), "tries": tries,
                              "post_check_max_diff": [maxdiff(f.last[0], ref_p), maxdiff(f.last[1], ref_d)]}
                log(f"{key} POD: {r}")
                save(res)

            # ---- serial (one stream) --------------------------------------------------
            for tag, dec in (("best", dec_best), ("tc", dec_tc)):
                name = f"serial_{tag}"
                if name in rec or (tag == "tc" and use_tc):
                    continue
                fp = Stash(pre.run)

                def ser(*_):
                    fp(*xp)
                    dec.run(*xd, out_d)
                r, tries = retry(lambda: cb.bench(ser, make_inputs=lambda: (), mode="flush", flops=pod.flops,
                                                  nbytes=pod.nbytes, clock=True, label=key + "-" + name),
                                 key + " " + name, lambda r: r.cv)
                torch.cuda.synchronize()
                rec[name] = {**bstats(r), "tries": tries, "decode_path": "tc" if dec.use_tensor_cores else "cc",
                             "post_check_max_diff": [maxdiff(fp.last, ref_p), maxdiff(out_d, ref_d)]}
                log(f"{key} {name}: {r}")
                save(res)
            if use_tc:
                rec["serial_tc"] = {"same_as": "serial_best"}

            # ---- two streams / green contexts ------------------------------------------
            fa = Stash(lambda: pre.run(*xp))

            def fb():
                dec_best.run(*xd, out_d)

            def corun(stream_a=None, stream_b=None, label=""):
                return retry(lambda: cb.bench_corun(fa, fb, stream_a=stream_a, stream_b=stream_b, clock=True,
                                                    label=label),
                             label, lambda c: c.corun["makespan"]["cv"])

            if "streams" not in rec:
                c, tries = corun(label=key + " streams")
                torch.cuda.synchronize()
                rec["streams"] = {**cstats(c), "tries": tries,
                                  "post_check_max_diff": [maxdiff(fa.last, ref_p), maxdiff(out_d, ref_d)]}
                log(f"{key} streams: {c}")
                save(res)
            green = rec.setdefault("green", {})
            for n in GREEN_PREFILL_SMS:
                gk = str(n)
                if gk in green:
                    continue
                part = cb.split_sms(n, ignore_coscheduling=True)
                try:
                    c, tries = corun(part.stream, part.rest_stream, label=f"{key} green {part.n_sms}/{part.n_rest}")
                    torch.cuda.synchronize()
                    green[gk] = {"prefill_sms": part.n_sms, "decode_sms": part.n_rest, **cstats(c), "tries": tries,
                                 "post_check_max_diff": [maxdiff(fa.last, ref_p), maxdiff(out_d, ref_d)]}
                    log(f"{key} green {part.n_sms}/{part.n_rest}: {c}")
                finally:
                    part.close()
                save(res)
            rec["done"] = True
            save(res)
            del pod, pre, dec_best, dec_tc, x, xp, xd, out_d, ref_p, ref_d, fa
            torch.cuda.empty_cache()


def main():
    sections = sys.argv[1:] or ["decode", "prefill", "pod"]
    gpu_guard.wait_until_free(log=log)          # before this process touches CUDA at all
    res = load()
    res.setdefault("meta", {})
    res["meta"].update({"device": cb.device_info(), "versions": fo.versions(),
                        "last_run": time.strftime("%Y-%m-%d %H:%M")})
    save(res)
    for s in sections:
        {"decode": section_decode, "prefill": section_prefill, "pod": section_pod}[s](res)
    log("done")


if __name__ == "__main__":
    main()
