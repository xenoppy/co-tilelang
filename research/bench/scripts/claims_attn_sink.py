"""Optional claim (examples/attention_sink/README.md): GQA attention-sink forward, TileLang vs the
benchmark's Triton kernel (gpt-oss style), bf16, batch 1, 64 query heads, 8 KV heads, causal,
seq 2048..16384, head_dim 64/128. README: TileLang 1.21-1.35x faster on H800.

As shipped the benchmark does not run: examples/attention_sink/benchmark_gqa_sink_fwd.py imports
`example_gqa_sink_fwd_bhsd`, which does not exist in the tree (upstream #1909, ded6a992, changed the
import but deleted example_gqa_sink_fwd_bhsd_wgmma_pipelined.py, the kernel the README numbers were
produced with, #885). This wrapper registers that deleted file, read verbatim from git
(ded6a992^), under the missing module name, then imports the benchmark unmodified.

TileLang variants: `tl_fixed` = benchmark main()'s config (block 128/128, 2 stages, 256 threads)
when it fits 99 KB (head_dim 64 only; head_dim 128 needs 160 KB); `tl_tuned` = the example's own
autotuner (its configs: block 128/128, stages {0,1,2}, threads {128,256}; unlaunchable ones fail
and are skipped by the tuner). Triton: benchmark's triton_program (BLOCK 64/64).

  python research/bench/scripts/claims_attn_sink.py --seq 4096 --dim 128   (under the GPU lock)
"""
from __future__ import annotations

import argparse
import importlib.util
import linecache
import math
import os
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_attn_common as C  # noqa: E402

SINK_DIR = os.path.join(C.REPO, "examples", "attention_sink")
GIT_REV = "ded6a992^"
GIT_PATH = "examples/attention_sink/example_gqa_sink_fwd_bhsd_wgmma_pipelined.py"
B, H, GROUPS = 1, 64, 8
README = {  # (seq, dim): (Triton TFLOPs, TileLang TFLOPs, speedup) -- examples/attention_sink/README.md, H800
    (2048, 64): (232.98, 281.89, 1.21), (2048, 128): (321.55, 417.98, 1.30),
    (4096, 64): (280.70, 349.47, 1.25), (4096, 128): (369.61, 497.13, 1.35),
    (8192, 64): (299.04, 385.56, 1.29), (8192, 128): (399.39, 507.93, 1.27),
    (16384, 64): (309.46, 400.62, 1.29), (16384, 128): (418.99, 549.11, 1.31),
}


def load_modules():
    src = subprocess.run(["git", "-C", C.REPO, "show", f"{GIT_REV}:{GIT_PATH}"], capture_output=True, text=True,
                         check=True).stdout
    name = "example_gqa_sink_fwd_bhsd"
    fname = f"{GIT_REV}:{GIT_PATH}"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=fname)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = fname
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    exec(compile(src, fname, "exec", dont_inherit=True), mod.__dict__)  # noqa: S102
    sys.modules[name] = mod
    if SINK_DIR not in sys.path:
        sys.path.insert(0, SINK_DIR)
    import benchmark_gqa_sink_fwd as bench
    return mod, bench, {"git_rev": GIT_REV, "path": GIT_PATH, "sha256": __import__("hashlib").sha256(src.encode()).hexdigest()}


def reference(q, k, v, sinks):
    """fp32 causal GQA attention with a per-head sink logit, chunked over queries:
    out = softmax([q.k/sqrt(d), sink]) @ [v, 0]  ->  [B, H, S, D] fp32."""
    import torch
    Bq, Hq, S, D = q.shape
    G = Hq // k.shape[1]
    out = torch.empty(Bq, Hq, S, D, dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(D)
    step = 1024
    kpos = torch.arange(S, device=q.device)
    for h in range(Hq):
        kf, vf = k[0, h // G].float(), v[0, h // G].float()
        s_h = sinks[h].float()
        for q0 in range(0, S, step):
            q1 = min(S, q0 + step)
            lg = (q[0, h, q0:q1].float() @ kf.T) * scale
            lg.masked_fill_(kpos[None, :] > torch.arange(q0, q1, device=q.device)[:, None], float("-inf"))
            m = torch.maximum(lg.max(-1, keepdim=True).values, s_h)
            p = torch.exp(lg - m)
            den = p.sum(-1, keepdim=True) + torch.exp(s_h - m)
            out[0, h, q0:q1] = (p @ vf) / den
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    import torch
    res = {"shape": {"batch": B, "heads": H, "groups": GROUPS, "seq": args.seq, "dim": args.dim, "dtype": "bfloat16",
                     "causal": True}, "env": C.env_info(), "wait": C.wait_gpu(min_free_gib=8), "variants": {},
           "readme_claim": dict(zip(("triton_tflops", "tilelang_tflops", "speedup"), README[(args.seq, args.dim)]))}
    import cobench as cb
    import tilelang.language as T
    mod, bench, prov = load_modules()
    res["restored_module"] = prov
    torch.manual_seed(0)
    q, k, v, sinks = mod.gen_inputs(B, H, args.seq, args.seq, args.dim, GROUPS, dtype=torch.bfloat16)
    ref = reference(q, k, v, sinks)
    flops = 2 * (2.0 * B * H * args.seq * args.seq * args.dim * 0.5)   # benchmark main()'s count
    V = {}
    # TileLang, benchmark main()'s fixed config
    rec = {"cfg": [128, 128, 2, 256]}
    try:
        kern = mod.flashattn(B, H, args.seq, args.seq, args.dim, GROUPS, None, block_M=128, block_N=128, num_stages=2,
                             threads=256, dtype=T.bfloat16)
        rec["resources"] = C.tl_resources(kern)
        if all(r["fits_99KB"] for r in rec["resources"].values()):
            V["tl_fixed"] = lambda: kern(q, k, v, sinks)
        else:
            rec["skipped"] = "smem > 99 KB (fails at launch on sm_120)"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["variants"]["tl_fixed"] = rec
    # TileLang, the example's autotuner as shipped
    rec = {}
    try:
        tuned = mod.flashattn(B, H, args.seq, args.seq, args.dim, GROUPS, None, dtype=T.bfloat16)
        rec["config"] = tuned.config
        rec["tuner_latency_ms"] = tuned.latency
        rec["resources"] = C.tl_resources(tuned)
        V["tl_tuned"] = lambda: tuned(q, k, v, sinks)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["traceback"] = traceback.format_exc()[-1500:]
        rec["cleared_last_error"] = C.clear_cuda_error()
    res["variants"]["tl_tuned"] = rec
    V["triton"] = lambda: bench.triton_program(q, k, v, sinks, None)
    timed = {}
    for n, f in V.items():
        r = res["variants"].setdefault(n, {})
        try:
            o = f()
            torch.cuda.synchronize()
            r["check"] = C.err_stats(o, ref, atol=2e-2, rtol=2e-2)   # bf16 output
            if r["check"]["ok"]:
                timed[n] = f
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            r["cleared_last_error"] = C.clear_cuda_error()
        print(n, r.get("check", {}).get("ok"), r.get("check", {}).get("max_abs"), r.get("error", "")[:200], flush=True)
    vr = cb.bench_variants(dict(timed), reference="triton" if "triton" in timed else None, reps=args.reps,
                           clock=True, nvml=True, keep_samples=True, label=f"sink s{args.seq} d{args.dim}")
    res["cobench"] = {"summary": C.variants_summary(vr), "derived": vr.derived, "nvml": vr.nvml, "guard": vr.guard,
                      "samples_us": vr.samples}
    from tilelang.profiler import do_bench
    for n, f in timed.items():
        res["variants"][n]["upstream_tl_do_bench_us"] = 1e3 * float(do_bench(f, warmup=500))   # benchmark's timer
    C.save_json(res, os.path.join(C.RESULTS, "raw", f"sink_s{args.seq}_d{args.dim}.json"))
    s = res["cobench"]["summary"]
    for n, d in s.items():
        print(f"{n:10s} {d['median_us']:9.1f} us {flops / d['median_us'] / 1e6:7.1f} TFLOPS  "
              f"speedup vs triton {s['triton']['median_us'] / d['median_us'] if 'triton' in s else 0:.2f}  "
              f"up {res['variants'][n].get('upstream_tl_do_bench_us', 0):.1f}")


if __name__ == "__main__":
    main()
