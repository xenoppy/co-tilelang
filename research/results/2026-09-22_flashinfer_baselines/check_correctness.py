"""Correctness of the FlashInfer baselines on sm_120 at the P1 shapes (criteria 2-4).

decode : BatchDecode (CUDA-core and tensor-core paths) vs fp32 torch GQA reference
prefill: SinglePrefill (causal) vs fp32 torch SDPA (yardstick: bf16 SDPA vs fp32 SDPA)
POD    : PODWithPagedKVCacheWrapper outputs vs the same fp32 references and vs the standalone
         FlashInfer kernels; 10 back-to-back calls on the default stream must be bitwise equal.
Pass rule (per element): |out - ref| <= atol + rtol*|ref|, atol = rtol = 1e-2 (bf16 output).
Writes correctness.json next to this file.
Run: source research/env.sh && python research/results/2026-09-22_flashinfer_baselines/check_correctness.py
"""
import json, os, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench"))
from baselines import flashinfer_ops as fo  # noqa: E402
sys.path.insert(0, HERE)
import gpu_guard  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ATOL = RTOL = 1e-2


def check(out, ref):
    st = fo.err_stats(out, ref)
    d = (out.float() - ref.float()).abs()
    st["n_violations"] = int((d > ATOL + RTOL * ref.float().abs()).sum().item())
    st["bf16_rounding_floor"] = (ref.to(torch.bfloat16).float() - ref.float()).abs().max().item()
    st["pass"] = st["finite"] and st["n_violations"] == 0
    return st


def main():
    gpu_guard.wait_until_free()
    watch = gpu_guard.Watchdog().__enter__()
    res = {"versions": fo.versions(), "tolerance": {"atol": ATOL, "rtol": RTOL},
           "gpu_procs_start": subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                              "--format=csv,noheader"], capture_output=True, text=True).stdout,
           "decode": [], "prefill": [], "pod": []}
    for B in (16, 64):
        for S in (2048, 8192):
            shp = fo.DecodeShape(batch=B, kv_len=S)
            ref = None
            for tc in (False, True):
                d = fo.BatchDecode(shp, use_tensor_cores=tc)
                x = d.make_inputs(seed=1000 + B + S)
                o = d.run(*x)
                torch.cuda.synchronize()
                if ref is None:
                    ref = d.reference(*x)
                r = {"batch": B, "kv_len": S, "use_tensor_cores": tc, **check(o, ref)}
                print("decode", r, flush=True)
                res["decode"].append(r)
                del d, x, o
            del ref
            torch.cuda.empty_cache()
    for S in (2048, 8192):
        for hkv in (8, 32):
            p = fo.SinglePrefill(fo.PrefillShape(seq_len=S, num_kv_heads=hkv))
            x = p.make_inputs(seed=2000 + S + hkv)
            o = p.run(*x)
            torch.cuda.synchronize()
            ref = p.reference(*x)
            ref_bf16 = p.reference(*x, dtype=torch.bfloat16)
            r = {"seq_len": S, "num_kv_heads": hkv, **check(o, ref),
                 "yardstick_bf16_sdpa_vs_fp32": fo.err_stats(ref_bf16, ref)}
            print("prefill", r, flush=True)
            res["prefill"].append(r)
            del p, x, o, ref, ref_bf16
            torch.cuda.empty_cache()
    for Sp in (2048, 8192):
        for B in (16, 64):
            for S in (2048, 8192):
                pod = fo.POD(fo.PrefillShape(seq_len=Sp), fo.DecodeShape(batch=B, kv_len=S))
                x = pod.make_inputs(seed=3000 + Sp + B + S)
                o_p, o_d = pod.run(*x)
                torch.cuda.synchronize()
                reps = [pod.run(*x) for _ in range(10)]
                torch.cuda.synchronize()
                det = max(max((a.float() - o_p.float()).abs().max().item(), (b.float() - o_d.float()).abs().max().item())
                          for a, b in reps)
                r_p, r_d = pod.reference(*x)
                # standalone FlashInfer kernels on the same inputs
                (xp, xd) = pod.split_inputs(x)
                sp = fo.SinglePrefill(pod.prefill_shape).run(*xp)
                dtc = fo.BatchDecode(pod.decode_shape, use_tensor_cores=True)
                sd = dtc.run(*xd, torch.empty_like(xd[0]))
                torch.cuda.synchronize()
                r = {"prefill_seq": Sp, "decode_batch": B, "decode_kv_len": S,
                     "prefill_vs_fp32": check(o_p, r_p), "decode_vs_fp32": check(o_d, r_d),
                     "prefill_vs_standalone_max_abs": (o_p.float() - sp.float()).abs().max().item(),
                     "decode_vs_standalone_tc_max_abs": (o_d.float() - sd.float()).abs().max().item(),
                     "repeat10_max_abs_diff": det}
                r["pass"] = r["prefill_vs_fp32"]["pass"] and r["decode_vs_fp32"]["pass"] and det == 0.0
                print("pod", json.dumps(r), flush=True)
                res["pod"].append(r)
                del pod, x, o_p, o_d, reps, r_p, r_d, sp, sd, dtc
                torch.cuda.empty_cache()
    watch.__exit__(None, None, None)
    res["foreign_gpu_procs_seen_during_run"] = {str(k): v for k, v in watch.seen.items()}
    res["gpu_procs_end"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    res["all_pass"] = all(r["pass"] for k in ("decode", "prefill", "pod") for r in res[k])
    with open(os.path.join(HERE, "correctness.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("ALL PASS" if res["all_pass"] else "SOME FAILED")


if __name__ == "__main__":
    main()
