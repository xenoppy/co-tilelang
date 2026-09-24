"""P4-a / P2': verification of the FlashInfer 0.7.0 POD stream-memset patch
(research/patches/flashinfer_pod_stream_memset.patch).

    source research/env.sh
    python research/bench/scripts/p4_pod_patch.py --phase unpatched   # BEFORE applying the patch
    patch -p1 -d "$(python -c 'import flashinfer,os;print(os.path.dirname(flashinfer.__file__))')" \\
          < research/patches/flashinfer_pod_stream_memset.patch
    python research/bench/scripts/p4_pod_patch.py --phase patched     # JIT rebuilds POD on load

For each of the 8 P4 configurations (causal prefill S in {2048, 8192} x decode B in {16, 64} x
KV in {2048, 8192}; Hq 32, Hkv 8, D 128, bf16) and NS input sets (fixed seeds):
  default  eager call on the legacy default stream: SHA-256 of both outputs, fp32-reference
           check (flashinfer_ops.err_stats); the patched phase compares the hashes with the
           unpatched phase's (bitwise equality of the patched build on the default stream);
  side     N back-to-back calls on a torch side stream (non-blocking), inputs cycling over the
           NS sets, no host sync in between; every output compared bitwise with the
           default-stream result of its input set;
  green    the same on a green-context stream (cobench.split_sms, 96 SMs);
  graph    one CUDA graph captured on a side stream, then R replays; before every replay the
           static inputs are overwritten with input set r % NS and the static outputs are
           poisoned with NaN; after it both outputs are compared bitwise with the
           default-stream result.
The unpatched phase runs side / green / graph only on --demo configs, to record the bug
(bypassing the wrapper's stream guard with the raw FlashInfer wrapper).
Output: research/results/2026-09-24_p4_prep/pod_patch_<phase>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "research", "results", "2026-09-24_p4_prep")

import torch  # noqa: E402

import cobench as cb  # noqa: E402
from baselines import flashinfer_ops as fo  # noqa: E402

CONFIGS = [(p, b, kv) for p in (2048, 8192) for b in (16, 64) for kv in (2048, 8192)]


def tag(c):
    return f"P{c[0]}_B{c[1]}_S{c[2]}"


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def raw_run(pod: fo.POD, x):
    """FlashInfer's own run (no stream guard of the wrapper)."""
    q_p, k_p, v_p, q_d, k_c, v_c = x
    return pod.wrapper.run(q_p, k_p, v_p, q_d, (k_c, v_c), causal_p=True, kv_layout_p="NHD")


def eq(a, b) -> bool:
    return bool(torch.equal(a, b))


def check_stream(pod, xs, expect, stream, n, run):
    """n back-to-back calls on `stream` (inputs cycle over xs), then compare every output."""
    outs = []
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        for i in range(n):
            outs.append((i % len(xs), run(pod, xs[i % len(xs)])))
    torch.cuda.synchronize()
    bad = [i for i, (k, (op, od)) in enumerate(outs)
           if not (eq(op, expect[k][0]) and eq(od, expect[k][1]))]
    maxdiff = max(float((op.float() - expect[k][0].float()).abs().nan_to_num(1e9).max()) for k, (op, od) in outs)
    return {"calls": n, "wrong": len(bad), "wrong_idx": bad[:20], "max_abs_diff_prefill": maxdiff}


def check_graph(pod, xs, expect, replays, run):
    """Capture one call on a side stream, replay `replays` times with rotating inputs and
    NaN-poisoned outputs."""
    static = [t.clone() for t in xs[0]]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            run(pod, static)            # eager warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=s):
            o_p, o_d = run(pod, static)
    except Exception as e:  # noqa: BLE001 - recorded
        return {"captured": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    rows = []
    times = []
    for r in range(replays):
        k = r % len(xs)
        for dst, src in zip(static, xs[k]):
            dst.copy_(src)
        o_p.fill_(float("nan"))
        o_d.fill_(float("nan"))
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1e3)
        rows.append(eq(o_p, expect[k][0]) and eq(o_d, expect[k][1]))
    del g
    return {"captured": True, "replays": replays, "wrong": rows.count(False),
            "wrong_idx": [i for i, ok in enumerate(rows) if not ok][:20],
            "replay_us_median": sorted(times)[len(times) // 2], "replay_us_min": min(times),
            "replay_us_max": max(times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("unpatched", "patched"), required=True)
    ap.add_argument("--sets", type=int, default=3, help="input sets per configuration")
    ap.add_argument("--calls", type=int, default=24, help="back-to-back calls per stream test")
    ap.add_argument("--replays", type=int, default=24, help="CUDA graph replays")
    ap.add_argument("--demo", default="P8192_B64_S8192,P2048_B16_S2048",
                    help="unpatched phase: configs on which the bug is demonstrated")
    ap.add_argument("--green-sms", type=int, default=96)
    a = ap.parse_args()

    st = fo.pod_patch_status()
    want = a.phase == "patched"
    if st["header_patched"] != want:
        raise SystemExit(f"phase {a.phase} but the installed header is "
                         f"{'patched' if st['header_patched'] else 'unpatched'} ({st['header']})")
    cb.wait_until_free(log=log)
    free, _ = torch.cuda.mem_get_info()
    if free < 12 << 30:
        raise SystemExit(f"only {free / 2**30:.1f} GiB free")
    so_before = {p["path"]: os.path.getmtime(p["path"]) for p in st["pod_so"]}
    res = {"phase": a.phase, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "status_before": st,
           "versions": fo.versions(), "torch": torch.__version__, "configs": {}}
    prev = None
    if want:
        with open(os.path.join(OUT, "pod_patch_unpatched.json")) as f:
            prev = json.load(f)
    part = cb.split_sms(a.green_sms, ignore_coscheduling=True)
    side = torch.cuda.Stream()
    t_start = time.time()
    for c in CONFIGS:
        t = tag(c)
        pod = fo.POD(fo.PrefillShape(seq_len=c[0]), fo.DecodeShape(batch=c[1], kv_len=c[2]),
                     unsafe_stream_ok=True)
        xs = [pod.make_inputs(seed=100 + 10 * k) for k in range(a.sets)]
        # default stream, eager
        expect = []
        rec = {"default": []}
        for k, x in enumerate(xs):
            o_p, o_d = raw_run(pod, x)
            o_p2, o_d2 = raw_run(pod, x)   # repeat: deterministic on the default stream?
            torch.cuda.synchronize()
            expect.append((o_p.clone(), o_d.clone()))
            d = {"sha_prefill": sha(o_p), "sha_decode": sha(o_d),
                 "repeat_bitwise": eq(o_p, o_p2) and eq(o_d, o_d2)}
            if k == 0:
                rp, rd = pod.reference(*x)
                d["err_prefill"] = fo.err_stats(o_p, rp)
                d["err_decode"] = fo.err_stats(o_d, rd)
                d["ref_ok"] = bool((o_p.float() - rp).abs().max() <= 2e-2 + 2e-2 * rp.abs().max()
                                   and (o_d.float() - rd).abs().max() <= 1e-2 + 1e-2 * rd.abs().max())
                del rp, rd
            if prev is not None:
                pd = prev["configs"][t]["default"][k]
                d["bitwise_eq_unpatched"] = (d["sha_prefill"] == pd["sha_prefill"]
                                             and d["sha_decode"] == pd["sha_decode"])
            rec["default"].append(d)
        run_ok = want or t in a.demo.split(",")
        if run_ok:
            rec["side"] = check_stream(pod, xs, expect, side, a.calls, raw_run)
            rec["green"] = check_stream(pod, xs, expect, part.stream, a.calls, raw_run)
            rec["graph"] = check_graph(pod, xs, expect, a.replays, raw_run)
            if want:
                # the research wrapper (flashinfer_ops.POD.run with its patch-aware guard) on a
                # side stream and in a graph
                pod2 = fo.POD(pod.prefill_shape, pod.decode_shape, require_patch=True)
                rec["wrapper_side"] = check_stream(pod2, xs, expect, side, 8, lambda p, x: p.run(*x))
                rec["wrapper_graph"] = check_graph(pod2, xs, expect, 20, lambda p, x: p.run(*x))
                del pod2
        res["configs"][t] = rec
        log(t, json.dumps({k: (v if k == "default" else {kk: vv for kk, vv in v.items() if kk != "wrong_idx"})
                           for k, v in rec.items()}, default=str)[:900])
        del xs, expect, pod
        torch.cuda.empty_cache()
    res["gpu_s"] = time.time() - t_start
    st_after = fo.pod_patch_status()
    res["status_after"] = st_after
    res["pod_so_rebuilt"] = {p["path"]: (so_before.get(p["path"]) is None
                                         or os.path.getmtime(p["path"]) > so_before[p["path"]])
                             for p in st_after["pod_so"]}
    part.close()
    # summary
    cfgs = res["configs"].values()
    summ = {"default_repeat_bitwise": all(d["repeat_bitwise"] for r in cfgs for d in r["default"]),
            "default_ref_ok": all(r["default"][0]["ref_ok"] for r in cfgs)}
    if want:
        summ["default_bitwise_eq_unpatched"] = all(d["bitwise_eq_unpatched"] for r in cfgs for d in r["default"])
    for mode in ("side", "green", "graph", "wrapper_side", "wrapper_graph"):
        rs = [r[mode] for r in cfgs if mode in r]
        if rs:
            summ[mode] = {"configs": len(rs), "calls": sum(x.get("calls", x.get("replays", 0)) for x in rs),
                          "wrong": sum(x.get("wrong", 0) for x in rs),
                          "captured": all(x.get("captured", True) for x in rs)}
    res["summary"] = summ
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"pod_patch_{a.phase}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    log("summary", json.dumps(summ))


if __name__ == "__main__":
    main()
