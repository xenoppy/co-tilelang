"""TileLang GQA decode (examples/flash_decoding/example_gqa_decode.py, unmodified) at the
P1 decode shapes, for comparison with FlashInfer's decode (criterion 2).

The example's own flashattn() is built with explicit configs from its config space
(block_H=64, threads=128, block_N in {64,128}, num_stages in {1,2}, num_split in {1,2,4,8}),
keeping only those within sm_120's 99 KB/CTA shared memory:
  (block_N, num_stages) = (64,1) ~48 KB, (64,2) ~80 KB, (128,1) ~80 KB.
(64,2) and (128,1) with num_split=8 are the two configs of
research/results/2026-09-22_p0-1_smoke/run_gqa_decode.py.

The example is fp16 (hard-coded) and takes a uint8 [B, S, H_kv] mask; K/V are dense
[B, S, H_kv, D] (byte-identical to FlashInfer's identity-paged NHD cache). Timing uses an
all-ones mask; correctness is checked against the fp32 GQA reference (all-ones mask) and,
for the first config per shape, against the example's own ref_program with a random mask.
GB/s counts K+V+Q+O (the mask adds 1/(4*D) = 1/512 of the K+V bytes on top).
Writes tilelang_decode.json.
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "research", "bench"))
sys.path.insert(0, os.path.join(REPO, "examples", "flash_decoding"))
sys.path.insert(0, HERE)
import cobench as cb  # noqa: E402
from baselines import flashinfer_ops as fo  # noqa: E402
import gpu_guard  # noqa: E402
import example_gqa_decode as ex  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
SHAPES = [(16, 2048), (16, 8192), (64, 2048), (64, 8192)]
CONFIGS = [(bn, st, ns) for bn, st in ((64, 1), (64, 2), (128, 1)) for ns in (1, 2, 4, 8)]
H, G, D = 32, 8, 128
SMEM_LIMIT = 101376   # sm_120 max dynamic smem per CTA (opt-in); see research/env_versions.md


def smem_est(bn, st):
    return 2 * (64 * D + 2 * max(st, 1) * bn * D) + 2 * (H // G) * D   # Q + K/V stages + O_shared


def make_inputs_fn(B, S):
    def mk():
        q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, S, G, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, S, G, D, device="cuda", dtype=torch.float16)
        m = torch.ones(B, S, G, device="cuda", dtype=torch.uint8)
        return q, k, v, m
    return mk


def stats(r):
    d = r.to_dict()
    c = (r.clock or {}).get("per_rep", {})
    return {"median_us": r.median, "p10_us": r.p10, "p90_us": r.p90, "cv": r.cv, "cv_ok": r.cv_ok,
            "gbps": r.gbps, "n_copies": r.n_copies, "k": r.k, "clock_mhz_median": c.get("median"),
            "cycles_median": c.get("cycles_median"), "l2": d["l2"], "mode": d["mode"]}


def bench_retry(fn, label, attempts=3, **kw):
    best = None
    for _ in range(attempts):
        r, info = gpu_guard.guarded(lambda: cb.bench(fn, clock=True, label=label, **kw), label=label)
        if best is None or r.cv < best[0].cv:
            best = (r, info)
        if r.cv_ok:
            break
    return best


def main():
    gpu_guard.wait_until_free()
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "device": cb.device_info(), "smem_limit": SMEM_LIMIT,
           "note": "fp16, example_gqa_decode.flashattn unmodified; GB/s from K+V+Q+O", "shapes": []}
    for B, S in SHAPES:
        gpu_guard.wait_until_free()
        nbytes = fo._decode_bytes(fo.DecodeShape(batch=B, kv_len=S, dtype=torch.float16))
        mk = make_inputs_fn(B, S)
        q, k, v, m = mk()
        ref = fo.decode_reference(q, k, v)
        entry = {"batch": B, "kv_len": S, "nbytes": nbytes, "configs": []}
        for i, (bn, st, ns) in enumerate(CONFIGS):
            cfg = dict(block_N=bn, block_H=64, num_split=ns, num_stages=st, threads=128)
            rec = {"config": cfg, "smem_est": smem_est(bn, st)}
            try:
                t0 = time.time()
                kern = ex.flashattn(B, H, G, S, D, **cfg)
                rec["compile_s"] = time.time() - t0
                gpu_guard.wait_until_free()
                o = kern(q, k, v, m)
                torch.cuda.synchronize()
                rec["vs_fp32"] = fo.err_stats(o, ref)
                if i == 0:
                    mr = torch.randint(0, 2, (B, S, G), device="cuda", dtype=torch.uint8)
                    o2 = kern(q, k, v, mr)
                    rec["vs_example_ref_program_random_mask"] = fo.err_stats(o2, ex.ref_program(q, k, v, mr))
                r, info = bench_retry(kern, f"tl-B{B}-S{S}-{bn}/{st}/{ns}", make_inputs=mk, mode="flush",
                                      nbytes=nbytes)
                rec["flush"] = stats(r)
                rec["guard"] = info
            except Exception as e:  # launch failures (smem) are recorded, not hidden
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(json.dumps({"B": B, "S": S, **{k2: rec.get(k2) for k2 in ("config", "error")},
                              "us": rec.get("flush", {}).get("median_us"), "gbps": rec.get("flush", {}).get("gbps"),
                              "err": rec.get("vs_fp32", {}).get("max_abs_err")}), flush=True)
            entry["configs"].append(rec)
        ok = [c for c in entry["configs"] if "flush" in c and c["vs_fp32"]["max_abs_err"] < 1e-2]
        best = min(ok, key=lambda c: c["flush"]["median_us"])
        entry["best_config"] = best["config"]
        # graph mode for the best config and the two smoke-test configs
        for cfg in [best["config"], dict(block_N=64, block_H=64, num_split=8, num_stages=2, threads=128),
                    dict(block_N=128, block_H=64, num_split=8, num_stages=1, threads=128)]:
            rec = next(c for c in entry["configs"] if c["config"] == cfg)
            if "graph" in rec or "flush" not in rec:
                continue
            try:
                kern = ex.flashattn(B, H, G, S, D, **cfg)
                r, info = bench_retry(kern, f"tl-graph-B{B}-S{S}", make_inputs=mk, mode="graph", nbytes=nbytes)
                rec["graph"] = stats(r)
            except Exception as e:
                rec["graph_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print("graph", cfg, rec.get("graph", {}).get("median_us"), rec.get("graph_error"), flush=True)
        out["shapes"].append(entry)
        del q, k, v, m, ref
        torch.cuda.empty_cache()
    with open(os.path.join(HERE, "tilelang_decode.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
