"""Shared pieces of the P4 study (prefill attention x GQA decode, POD's scenario).
Results: research/results/2026-09-24_p4_prep (P4-a: infrastructure + solo profiling).

* shapes, the 8 P4 pairs and the SM-budget grids (green-context split sweep: the prefill
  gets n_P SMs [0, n_P), the decode the other 188 - n_P);
* FlashInfer references (research/bench/baselines/flashinfer_ops.py) as launchers over
  rotating input copies (> 2x L2, like p1_common.OpData for the TileLang ops);
* re-exports of p1_common helpers (OpData, launchers, GPU waiting, JSON).

Head configuration of both ops: Hq 32, Hkv 8, D 128, bf16 (FlashInfer's POD requires the
prefill and the decode to share it; cotile/ops/prefill_attn.py docstring).
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p1_common as P1  # noqa: E402
from p1_common import FULL, OpData, cb, catalog, compile_all, log, retry_oom, wait_gpu  # noqa: E402,F401

from baselines import flashinfer_ops as fo  # noqa: E402

ROOT = P1.ROOT
OUT = os.path.join(ROOT, "research", "results", "2026-09-24_p4_prep")
SOLO_DIR = os.path.join(OUT, "solo")
A1_SOLO_DIR = P1.SOLO_DIR  # research/results/2026-09-23_p1_3x2_A/solo_v1

PREFILL_SHAPES = ["S2048", "S8192"]
DECODE_SHAPES = ["B16_S2048", "B16_S8192", "B64_S2048", "B64_S8192"]
# decode shapes whose A1 points (every config @188/94/48, C_lib U top-6 at 188 - GREEN_A)
# are reused (same config set, same code path; spot-checked in p4_solo.py phase 2)
DECODE_REUSED = ["B16_S8192", "B64_S8192"]
SHAPES = {"prefill_attn": PREFILL_SHAPES, "gqa_decode": DECODE_SHAPES}

# the 8 P4 pairs: (prefill shape, decode shape); name P<S>_B<b>_S<kv> as in the 2026-09-22
# FlashInfer POD baseline
PAIRS = {f"P{p[1:]}_{d}": (p, d) for p in PREFILL_SHAPES for d in DECODE_SHAPES}

# green split grid, prefill side (n_P SMs); the decode gets 188 - n_P. It is the P1 grid
# (GREEN_A, so the A1 decode budgets are reused) plus 32 for decode-heavy pairs
# (P2048 + B64_S8192: 165 vs 1360 us solo).
GREEN_P = tuple(sorted(set(P1.GREEN_A) | {32}))
BUDGETS = {"prefill_attn": tuple(sorted(set(GREEN_P) | {FULL})),
           "gqa_decode": tuple(sorted({FULL - n for n in GREEN_P} | {FULL}))}
PHASE1_BUDGETS = (FULL, 94, 48)


def shape_of(op_name: str, tag: str):
    return P1.shape_of(op_name, tag)


def fi_prefill_shape(tag: str) -> fo.PrefillShape:
    return fo.PrefillShape(seq_len=int(tag[1:]))


def fi_decode_shape(tag: str) -> fo.DecodeShape:
    b, s = tag.split("_")
    return fo.DecodeShape(batch=int(b[1:]), kv_len=int(s[1:]))


class FIData:
    """Rotating input copies (> min_bytes, default 2x L2) of one FlashInfer workload and a
    launcher fn(i) running it on copy i % n on the current stream.

    kind: "prefill" (single_prefill_with_kv_cache), "decode_cc" / "decode_tc" (batch decode,
    CUDA-core / tensor-core path), "pod" (patched POD, prefill + decode in one launch)."""

    def __init__(self, kind: str, prefill_tag: str | None = None, decode_tag: str | None = None,
                 min_bytes: int | None = None):
        self.kind = kind
        if kind == "prefill":
            self.w = fo.SinglePrefill(fi_prefill_shape(prefill_tag))
        elif kind in ("decode_cc", "decode_tc"):
            self.w = fo.BatchDecode(fi_decode_shape(decode_tag), use_tensor_cores=kind == "decode_tc")
        elif kind == "pod":
            self.w = fo.POD(fi_prefill_shape(prefill_tag), fi_decode_shape(decode_tag), require_patch=True)
        else:
            raise ValueError(kind)
        min_bytes = int(min_bytes if min_bytes is not None else 2 * cb.l2_bytes())
        self.copies, total = [], 0
        while total < min_bytes:
            x = self.w.make_inputs(seed=1000 + len(self.copies))
            self.copies.append(x)
            total += cb.tensor_bytes(x)
            if kind == "prefill":  # the output (allocated per call) is written too
                total += x[0].numel() * x[0].element_size()
        self.n = len(self.copies)

    def launcher(self):
        w, cps, n = self.w, self.copies, self.n

        def fn(i):
            w.run(*cps[i % n])
        fn.spec_name = f"fi:{self.kind}"
        return fn

    def reference(self, k: int = 0):
        return self.w.reference(*self.copies[k])


def torch_bf16_reference(data: FIData, x):
    """The same attention computed by torch in bf16 (SDPA; K/V heads expanded): the accuracy
    a plain bf16 implementation reaches on these inputs."""
    if data.kind == "prefill":
        return fo.prefill_reference(x[0], x[1], x[2], causal=True, dtype=torch.bfloat16).float()
    q, k_c, v_c = x[0], *data.w.dense_kv(x[1], x[2])
    G = q.shape[1] // k_c.shape[2]
    kh = k_c.repeat_interleave(G, dim=2).transpose(1, 2)            # [B, Hq, S, D]
    vh = v_c.repeat_interleave(G, dim=2).transpose(1, 2)
    o = torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(2), kh, vh)   # [B, Hq, 1, D]
    return o.squeeze(2).float()


def fi_check(data: FIData, tol) -> dict:
    """FlashInfer output of copy 0 vs the fp32 reference with the op library's tolerance rule
    (|out - ref| <= atol*rms(ref) + rtol*|ref|, cotile.tests.harness.compare; tol = the matching
    cotile op's TOLERANCE). FlashInfer's inputs are unscaled randn: at S=8192 the long causal rows
    average to tiny outputs, the global rms is small, and even torch's own bf16 SDPA exceeds that
    rule (worst ratio 1.47 on the prefill S=8192 inputs, identical to FlashInfer's and TileLang's).
    ok = within the rule, or no worse than torch's bf16 computation of the same attention
    (worst ratio <= 1.05 x torch bf16's)."""
    from cotile.tests.harness import compare

    x = data.copies[0]
    out = data.w.run(*x)
    torch.cuda.synchronize()
    ref = data.w.reference(*x)
    if data.kind == "pod":
        raise ValueError("fi_check: POD has two outputs; check them separately")
    if data.kind.startswith("decode"):
        out = x[3]
    c = compare(out, ref.float(), tol)
    cb16 = compare(torch_bf16_reference(data, x), ref.float(), tol)
    ok = c["ok"] or (bool(torch.isfinite(out.float()).all()) and c["worst_ratio"] <= 1.05 * cb16["worst_ratio"])
    return {"ok": ok, "worst": c["worst_ratio"], "max_abs": c["max_abs_err"], "within_rule": c["ok"],
            "torch_bf16_worst": cb16["worst_ratio"], "torch_bf16_max_abs": cb16["max_abs_err"],
            **fo.err_stats(out, ref)}
