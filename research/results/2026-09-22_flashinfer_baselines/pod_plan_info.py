"""POD work decomposition per configuration (explains the timing results).

For each (prefill S, decode B, KV) prints, from FlashInfer's own plan and the dispatch
rules of include/flashinfer/attention/pod.cuh (0.7.0) on this device:
  prefill CTAs = ceil(S*G/128) * H_kv   (CTA_TILE_Q=128; no KV split: max_num_kv_chunks = 0 here)
  decode CTAs  = padded_batch_size * H_kv (from the FA2 plan POD uses; split-KV chunks)
  POD's per-SM ticket ratio (prefill:decode) and the standalone TC-decode plan for comparison.
Writes pod_plan_info.json.
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "bench")))
sys.path.insert(0, HERE)
from baselines import flashinfer_ops as fo  # noqa: E402
import gpu_guard  # noqa: E402


def main():
    gpu_guard.wait_until_free()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = []
    for Sp in (2048, 8192):
        for B in (16, 64):
            for S in (2048, 8192):
                ps, ds = fo.PrefillShape(seq_len=Sp), fo.DecodeShape(batch=B, kv_len=S)
                pod = fo.POD(ps, ds)
                pi = list(pod.wrapper._plan_info)
                tc = fo.BatchDecode(ds, use_tensor_cores=True)
                ti = list(tc.wrapper._plan_info)
                p_ctas = math.ceil(Sp * ps.group / 128) * ps.num_kv_heads
                d_ctas = pi[0] * ds.num_kv_heads
                if p_ctas <= d_ctas:
                    ratio = f"1:{d_ctas // p_ctas}"
                else:
                    ratio = f"{p_ctas // d_ctas}:1"
                rows.append({"prefill_seq": Sp, "decode_batch": B, "decode_kv_len": S,
                             "prefill_ctas": p_ctas, "decode_ctas": d_ctas, "grid": p_ctas + d_ctas,
                             "waves_at_2_per_sm": (p_ctas + d_ctas) / (2 * sms),
                             "ticket_ratio_prefill_to_decode": ratio,
                             "pod_decode_plan": {"padded_batch_size": pi[0], "cta_tile_q": pi[3], "split_kv": bool(pi[14])},
                             "standalone_tc_decode_plan": {"padded_batch_size": ti[0], "cta_tile_q": ti[3],
                                                           "split_kv": bool(ti[14])}})
                print(rows[-1], flush=True)
                del pod, tc
    with open(os.path.join(HERE, "pod_plan_info.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
