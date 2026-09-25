"""Claim C4: MLA decode, TileLang vs FlashMLA / FlashInfer / Triton / Torch
(examples/deepseek_mla/figures/bs{64,128}_float16.png, examples/deepseek_mla/README.md).

Workload = examples/deepseek_mla/benchmark_mla.py `shape_configs` (the figures add batch 64):
  batch b in {64, 128}; KV context L in {1024, ..., 32768}; per-request lengths L + 2*i (i < b);
  s_q = 1, h_q = 128, h_kv = 1, d = 576 (512 latent + 64 rope), dv = 512, fp16, page size 64,
  identity block table, KV cache padded to max_seqlen_pad = ceil(max_len / 256) * 256.
  FLOP count as in benchmark_mla.py: s_q * sum(lengths) * h_q * (d + dv) * 2.

Implementations
  TileLang  examples/deepseek_mla/example_mla_decode_paged.mla_decode_tilelang (what
            benchmark_mla.py times), shipped config BLOCK_N=64, BLOCK_H=64, num_split=1:
            needs 226 KB shared memory (num_stages=2 is hard-coded in the example's
            T.Pipelined loops) -> cannot launch on sm_120 (99 KB/CTA). The example's parameters
            cannot bring it under 99 KB either (block_N < 64 fails layout inference on the
            mma.sync path, block_H=16 still needs 166 KB). Adapted here: the example source is
            loaded unmodified and exactly the two `T.Pipelined(loop_range, num_stages=2)` are
            re-parameterised (derived module; with num_stages=2 it reproduces the example's
            kernel source bit-for-bit, checked at run time); the first config that fits
            99 KB in the order num_stages (2, 1) x block_H (64, 32, 16) is used (block_N=64,
            num_split=1 kept). A small num_split sweep is reported as sensitivity.
  Triton    benchmark_mla.py's mla_decode_triton (BLOCK_H=16, BLOCK_N=64, 32 KV splits); if its
            default num_stages exceeds 99 KB, the same kernels are launched with the largest
            num_stages that compiles (launcher copy, kernels imported from benchmark_mla.py).
  FlashInfer BatchMLAPagedAttentionWrapper as in benchmark_mla.py, which asks for backend "fa3"
            (Hopper-only); here "fa2" (FlashInfer's default on non-sm90) and "cutile" (the
            wrapper's sm_120 backend); plus xqa_batch_decode_with_kv_cache_mla (FlashInfer's
            sm_120 MLA decode kernel; bf16/fp8 only -> run in bf16, an extra reference).
  Torch     benchmark_mla.run_torch_mla (per-request fp32 loop), timed with its own
            triton.testing.do_bench only (it synchronises the host, so it cannot run under
            cobench's host gate).
  FlashMLA  not testable on sm_120 (dense decoding kernels are sm_90a-only).

Usage (repo root, `source research/env.sh`; one process per shape, under the GPU lock):
  python research/bench/scripts/claims_attn_mla.py --phase compile --batch 64 --seqlen 1024
  flock <gpu.lock> python research/bench/scripts/claims_attn_mla.py --phase run --batch 64 --seqlen 1024
Output: research/results/2026-09-24_claims_repro/B_attention/raw/mla_b<b>_s<L>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import linecache
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_attn_common as C  # noqa: E402

MLA_DIR = os.path.join(C.REPO, "examples", "deepseek_mla")
H_Q, H_KV, D, DV, BLOCK_SIZE = 128, 1, 576, 512, 64
DPE = D - DV
SHIPPED = dict(block_N=64, block_H=64, num_split=1, num_stages=2)
ADAPT_ORDER = [(st, bh) for st in (2, 1) for bh in (64, 32, 16)]
SPLIT_SWEEP = (2, 4)
PIPE_STR = "T.Pipelined(loop_range, num_stages=2)"


def _path():
    if MLA_DIR not in sys.path:
        sys.path.insert(0, MLA_DIR)


def example_module():
    _path()
    import example_mla_decode_paged as P
    return P


_DERIVED = {}


def derived_module(num_stages: int):
    """example_mla_decode_paged.py with its two hard-coded `num_stages=2` KV pipelines
    replaced by `num_stages` (nothing else changes; the file itself is not modified)."""
    if num_stages in _DERIVED:
        return _DERIVED[num_stages]
    path = os.path.join(MLA_DIR, "example_mla_decode_paged.py")
    src = open(path).read()
    n = src.count(PIPE_STR)
    if n != 2:
        raise RuntimeError(f"expected 2 occurrences of {PIPE_STR!r} in {path}, found {n}")
    new = src.replace(PIPE_STR, f"T.Pipelined(loop_range, num_stages={int(num_stages)})")
    name = f"example_mla_decode_paged_stages{num_stages}"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    fname = f"{path} [num_stages={num_stages}]"
    linecache.cache[fname] = (len(new), None, new.splitlines(True), fname)   # for inspect.getsource
    # dont_inherit: this file uses `from __future__ import annotations`, which must not leak into
    # the example (TileLang evaluates its T.Tensor annotations eagerly)
    exec(compile(new, fname, "exec", dont_inherit=True), mod.__dict__)  # noqa: S102
    sys.modules[name] = mod
    _DERIVED[num_stages] = mod
    return mod


def shape_of(b, seqlen):
    import torch
    lens = [seqlen + 2 * i for i in range(b)]
    max_pad = math.ceil(max(lens) / 256) * 256
    return {"batch": b, "seqlen": seqlen, "lens": lens, "max_seqlen_pad": max_pad,
            "total": sum(lens), "flops": 1 * sum(lens) * H_Q * (D + DV) * 2,
            "bytes": (sum(lens) * H_KV * D + b * H_Q * D + b * H_Q * DV) * 2, "dtype": torch.float16}


def tl_kernel(sh, block_N, block_H, num_split, num_stages):
    mod = example_module() if num_stages == 2 else derived_module(num_stages)
    return mod.mla_decode_tilelang(sh["batch"], H_Q, H_KV, sh["max_seqlen_pad"], DV, DPE, block_N, block_H,
                                   num_split, BLOCK_SIZE)


def pick_adapted(sh):
    """First config (num_stages desc, block_H desc) whose kernels fit 99 KB; records all tried."""
    tried = []
    for st, bh in ADAPT_ORDER:
        rec = {"num_stages": st, "block_H": bh, "block_N": 64, "num_split": 1}
        try:
            k = tl_kernel(sh, 64, bh, 1, st)
            r = C.tl_resources(k)
            rec["smem"] = max(x["smem_total"] for x in r.values())
            rec["fits"] = rec["smem"] <= C.SMEM_PER_CTA
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rec["fits"] = False
        tried.append(rec)
        if rec["fits"]:
            return rec, tried
    raise RuntimeError(f"no TileLang MLA config fits 99 KB: {tried}")


def check_derived_faithful(sh) -> dict:
    """derived_module(2) must generate exactly the example's program (TIR script compared as text;
    TileLang's kernel cache is keyed on the same script, so the kernels are then identical too)."""
    args = (sh["batch"], H_Q, H_KV, sh["max_seqlen_pad"], DV, DPE, 64, 64, 1, BLOCK_SIZE)
    a = example_module().mla_decode_tilelang.get_tir(*args).script(show_meta=True)
    b = derived_module(2).mla_decode_tilelang.get_tir(*args).script(show_meta=True)
    c = derived_module(1).mla_decode_tilelang.get_tir(*args).script(show_meta=True)
    return {"identical_kernel_source": a == b, "tir_len": len(a), "stages1_differs": a != c}


def compile_phase(sh) -> dict:
    out = {"faithful": check_derived_faithful(sh)}
    k = tl_kernel(sh, **SHIPPED)
    out["shipped"] = {**SHIPPED, "resources": C.tl_resources(k)}
    rec, tried = pick_adapted(sh)
    out["adapted"], out["adapt_tried"] = rec, tried
    out["split_sweep"] = {}
    for ns in SPLIT_SWEEP:
        k = tl_kernel(sh, 64, rec["block_H"], ns, rec["num_stages"])
        out["split_sweep"][ns] = C.tl_resources(k)
    return out


# ---------------------------------------------------------------------------------------------
def make_inputs(sh):
    """Exactly benchmark_mla.compare_a's inputs (seed 0, default dtype fp16), on cuda."""
    import torch
    b, mp = sh["batch"], sh["max_seqlen_pad"]
    torch.manual_seed(0)
    dev = "cuda"
    cache_seqlens = torch.tensor(sh["lens"], dtype=torch.int32, device=dev)
    q = torch.randn(b, 1, H_Q, D, dtype=torch.float16, device=dev)
    block_table = torch.arange(b * mp // BLOCK_SIZE, dtype=torch.int32, device=dev).view(b, mp // BLOCK_SIZE)
    blocked_k = torch.randn(block_table.numel(), BLOCK_SIZE, H_KV, D, dtype=torch.float16, device=dev)
    q_nope, q_pe = q[..., :DV].contiguous(), q[..., DV:].contiguous()
    k_nope, k_pe = blocked_k[..., :DV].contiguous(), blocked_k[..., DV:].contiguous()
    return dict(q=q, cache_seqlens=cache_seqlens, block_table=block_table, blocked_k=blocked_k,
                q_nope=q_nope, q_pe=q_pe, k_nope=k_nope, k_pe=k_pe)


def reference(sh, inp, chunk=8):
    """fp32 attention over the valid prefix of each request -> [b, h_q, dv] fp32."""
    import torch
    b, mp = sh["batch"], sh["max_seqlen_pad"]
    q = inp["q"][:, 0].float()                                         # [b, h, d]
    kv = inp["blocked_k"].view(b, mp, D)                                 # identity block table
    lens = inp["cache_seqlens"]
    out = torch.empty(b, H_Q, DV, dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(D)
    for b0 in range(0, b, chunk):
        b1 = min(b, b0 + chunk)
        k = kv[b0:b1].float()
        s = torch.einsum("bhd,bsd->bhs", q[b0:b1], k) * scale
        pos = torch.arange(mp, device=q.device)
        s.masked_fill_(pos[None, None, :] >= lens[b0:b1, None, None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[b0:b1] = torch.einsum("bhs,bsd->bhd", p, k[..., :DV])
    return out


def build_variants(sh, inp, adapted, res):
    import torch
    import triton
    _path()
    import benchmark_mla as BM
    b = sh["batch"]
    V = {}

    # --- TileLang (adapted) + num_split sensitivity ---
    def tl_fn(kern, ns):
        glse = torch.empty(b, H_Q, ns, dtype=torch.float16, device="cuda")
        part = torch.empty(b, H_Q, ns, DV, dtype=torch.float16, device="cuda")
        args = (inp["q_nope"].view(-1, H_Q, DV), inp["q_pe"].view(-1, H_Q, DPE), inp["k_nope"].view(-1, H_KV, DV),
                inp["k_pe"].view(-1, H_KV, DPE), inp["block_table"], inp["cache_seqlens"], glse, part)
        return lambda: kern(*args)
    st, bh = adapted["num_stages"], adapted["block_H"]
    for ns in (1,) + SPLIT_SWEEP:
        name = "tl_adapted" if ns == 1 else f"tl_adapted_split{ns}"
        k = tl_kernel(sh, 64, bh, ns, st)
        V[name] = tl_fn(k, ns)
        res["variants"][name] = {"cfg": {"block_N": 64, "block_H": bh, "num_split": ns, "num_stages": st},
                                 "resources": C.tl_resources(k)}

    # --- Triton (benchmark_mla.py kernels) ---
    # _mla_attn_kernel forms element offsets kv_loc * stride in int32: past 2**31 elements of the
    # nope cache (batch 128 x 32 K context) they wrap and the kernel faults (illegal address, which
    # kills the CUDA context) -> only attempted in a child process there.
    n_elem = inp["k_nope"].numel()
    if n_elem >= 2 ** 31:
        tri = {"skipped": f"nope cache has {n_elem} elements >= 2**31: int32 offset overflow in _mla_attn_kernel",
               "isolated_attempt": C.run_isolated([sys.executable, os.path.abspath(__file__), "--phase", "triton_probe",
                                                   "--batch", str(sh["batch"]), "--seqlen", str(sh["seqlen"])])}
    else:
        tri = triton_select(sh, inp, V)
    res["triton_mla"] = tri

    # --- FlashInfer ---
    import flashinfer
    from flashinfer.mla import MLAPlanMetadata
    lens = inp["cache_seqlens"]
    npages = (lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_indptr = torch.zeros(b + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(npages, 0)
    kv_indices = torch.cat([inp["block_table"][i, :int(npages[i])] for i in range(b)]).to(torch.int32)
    qo_indptr = torch.arange(b + 1, dtype=torch.int32, device="cuda")
    ckv = inp["k_nope"].view(-1, BLOCK_SIZE, DV)
    kpe = inp["k_pe"].view(-1, BLOCK_SIZE, DPE)
    qn, qp = inp["q_nope"].view(-1, H_Q, DV), inp["q_pe"].view(-1, H_Q, DPE)
    fi = {}
    # XQA (sm_120 MLA decode kernel, bf16)
    rec = {}
    try:
        qb = inp["q"].to(torch.bfloat16)                                    # [b, 1, h, 576]
        kvb = inp["blocked_k"].view(-1, BLOCK_SIZE, D).to(torch.bfloat16)    # [pages, 64, 576]
        wsx = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        maxlen = max(sh["lens"])
        f = (lambda: flashinfer.mla.xqa_batch_decode_with_kv_cache_mla(
            qb, kvb, wsx, 128, DV, DPE, inp["block_table"], lens, maxlen, bmm1_scale=1 / math.sqrt(D),
            bmm2_scale=1.0).view(b, H_Q, DV))
        f()
        torch.cuda.synchronize()
        V["flashinfer_xqa_bf16"] = f
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        rec["cleared_last_error"] = C.clear_cuda_error()
    fi["xqa_bf16"] = rec
    # fa3 (the backend benchmark_mla.py uses; sm_90a-only) is attempted in a child process: its failed
    # launch leaves cudaErrorNoKernelImageForDevice as the runtime's "last error", which the next
    # TileLang launch in this process then reported as its own failure (observed 2026-09-24)
    fi["fa3"] = C.run_isolated([sys.executable, os.path.abspath(__file__), "--phase", "fi_fa3",
                                "--batch", str(sh["batch"]), "--seqlen", str(sh["seqlen"])])
    for backend in ("fa2", "cutile"):
        rec = {}
        # benchmark_mla.py plans with causal=True; cuTile rejects causal, which is a no-op for
        # single-token decode (the query sits at the last KV position), so it gets causal=False
        causal = backend != "cutile"
        rec["causal"] = causal
        try:
            ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
            w = flashinfer.mla.BatchMLAPagedAttentionWrapper(ws, backend=backend)
            md = MLAPlanMetadata.csr(qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
                                     kv_len_arr=lens)
            w.plan(metadata=md, num_heads=H_Q, head_dim_ckv=DV, head_dim_kpe=DPE, page_size=BLOCK_SIZE,
                   causal=causal, sm_scale=1 / math.sqrt(D), q_data_type=torch.float16,
                   kv_data_type=torch.float16, query_layout="split", kv_cache_layout="split")
            f = (lambda w=w: w.run(query=(qn, qp), kv_cache=(ckv, kpe)))
            f()
            torch.cuda.synchronize()
            V[f"flashinfer_mla_{backend}"] = f
            rec["ok"] = True
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            rec["cleared_last_error"] = C.clear_cuda_error()
        fi[backend] = rec
    res["flashinfer"] = fi
    return V


def run_phase(sh, args) -> dict:
    import torch
    # fp16 cache + nope/pe copies + bf16 copy + temps; torch_mla adds fp32 K and V repeated over 128 heads
    need = (sh["batch"] * sh["max_seqlen_pad"] * D * 2) * 5 / 2 ** 30 + 4
    if not args.skip_torch:
        need += 2 * H_Q * sh["max_seqlen_pad"] * D * 4 / 2 ** 30
    res = {"shape": {k: v for k, v in sh.items() if k != "lens"}, "env": C.env_info(), "timings_s": {},
           "variants": {}, "wait": C.wait_gpu(min_free_gib=need), "pmon_before": C.pmon_snapshot()}
    import cobench as cb
    res["faithful"] = check_derived_faithful(sh)
    if not res["faithful"]["identical_kernel_source"]:
        raise RuntimeError("derived module (num_stages=2) does not reproduce the example's kernel")
    inp = make_inputs(sh)
    with C.timed("reference", res["timings_s"]):
        ref = reference(sh, inp)

    # shipped TileLang config: compiles, fails at launch (recorded). Attempted in a child process:
    # after this failed launch, the next TileLang module loaded in the same process failed with
    # CUDA_ERROR_NO_BINARY_FOR_GPU (observed 2026-09-24; the same kernel runs fine in a clean process).
    res["tl_shipped"] = C.run_isolated([sys.executable, os.path.abspath(__file__), "--phase", "shipped",
                                        "--batch", str(sh["batch"]), "--seqlen", str(sh["seqlen"])])

    adapted, tried = pick_adapted(sh)
    res["tl_adapt"] = {"picked": adapted, "tried": tried}
    V = build_variants(sh, inp, adapted, res)
    timed_fns = {}
    for name, fn in V.items():
        r = res["variants"].setdefault(name, {})
        try:
            o = fn()
            torch.cuda.synchronize()
            r["check"] = C.err_stats(o.view(sh["batch"], H_Q, DV), ref)
            if r["check"]["ok"]:
                timed_fns[name] = fn
            else:
                r["excluded"] = "failed correctness check"
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            r["traceback"] = traceback.format_exc()[-1500:]
            r["cleared_last_error"] = C.clear_cuda_error()
        print(f"{name}: {r.get('check', {}).get('ok')} maxabs {r.get('check', {}).get('max_abs')} "
              f"{r.get('error', '')[:200]}", flush=True)

    ref_name = "tl_adapted" if "tl_adapted" in timed_fns else None
    with C.timed("bench_variants", res["timings_s"]):
        vr = cb.bench_variants(dict(timed_fns), reference=ref_name, reps=args.reps, clock=True, nvml=True, keep_samples=True,
                               label=f"C4 b{sh['batch']} L{sh['seqlen']}")
    res["cobench"] = {"summary": C.variants_summary(vr), "derived": vr.derived, "config": vr.config,
                      "nvml": vr.nvml, "clock_window": vr.clock, "guard": vr.guard,
                      "samples_us": vr.samples}
    with C.timed("upstream timers", res["timings_s"]):
        for name, fn in timed_fns.items():
            which = ("tilelang",) if name.startswith("tl_") else ("triton",)
            res["variants"][name]["upstream"] = C.upstream_timers(fn, which=which)
    # Torch: benchmark_mla.run_torch_mla with its own timer (triton.testing.do_bench)
    if not args.skip_torch:
        _path()
        import benchmark_mla as BM
        with C.timed("torch", res["timings_s"]), torch.device("cuda"):
            try:
                out_t, _, t_ms = BM.run_torch_mla(inp["q"], inp["block_table"], inp["blocked_k"],
                                                  sh["max_seqlen_pad"], BLOCK_SIZE, sh["batch"], 1,
                                                  inp["cache_seqlens"], H_Q, H_KV, D, DV, True, torch.float16)
                res["variants"]["torch_mla"] = {"check": C.err_stats(out_t.view(sh["batch"], H_Q, DV), ref),
                                                "upstream": {"triton_do_bench_us": 1e3 * t_ms}}
            except Exception as e:  # noqa: BLE001
                res["variants"]["torch_mla"] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
    res["pmon_after"] = C.pmon_snapshot()
    return res


TRITON_NSPLIT = 32


def triton_shipped_fn(sh, inp):
    """benchmark_mla.mla_decode_triton exactly as the script calls it (default num_stages)."""
    import torch
    _path()
    import benchmark_mla as BM
    b = sh["batch"]
    q_nope, q_pe = inp["q_nope"].view(-1, H_Q, DV), inp["q_pe"].view(-1, H_Q, DPE)
    kvn, kvp = inp["k_nope"].view(-1, DV), inp["k_pe"].view(-1, DPE)

    def f():
        o = torch.empty(b, H_Q, DV, dtype=torch.float16, device="cuda")
        logits = torch.empty(b, H_Q, TRITON_NSPLIT, DV + 1, dtype=torch.float16, device="cuda")
        BM.mla_decode_triton(q_nope, q_pe, kvn, kvp, o, inp["block_table"], inp["cache_seqlens"], logits,
                             TRITON_NSPLIT, 1 / math.sqrt(D), BLOCK_SIZE)
        return o
    return f


def triton_launch_fn(sh, inp, num_stages, num_warps=4):
    """Same kernels, launched like benchmark_mla._mla_attn but with an explicit num_stages."""
    import torch
    import triton
    _path()
    import benchmark_mla as BM
    b = sh["batch"]
    q_nope, q_pe = inp["q_nope"].view(-1, H_Q, DV), inp["q_pe"].view(-1, H_Q, DPE)
    kvn, kvp = inp["k_nope"].view(-1, DV), inp["k_pe"].view(-1, DPE)

    def f():
        o = torch.empty(b, H_Q, DV, dtype=torch.float16, device="cuda")
        logits = torch.empty(b, H_Q, TRITON_NSPLIT, DV + 1, dtype=torch.float16, device="cuda")
        BLOCK_H, BLOCK_N = 16, 64          # benchmark_mla._mla_attn's constants
        grid = (triton.cdiv(H_Q, BLOCK_H), b, TRITON_NSPLIT)
        BM._mla_attn_kernel[grid](q_nope, q_pe, kvn, kvp, inp["block_table"], inp["cache_seqlens"], logits,
                                  1 / math.sqrt(D), q_nope.stride(0), q_nope.stride(1), q_pe.stride(0),
                                  q_pe.stride(1), kvn.stride(-2), kvp.stride(-2), inp["block_table"].stride(0),
                                  logits.stride(0), logits.stride(1), logits.stride(2), BLOCK_H=BLOCK_H,
                                  BLOCK_N=BLOCK_N, NUM_KV_SPLITS=TRITON_NSPLIT, PAGE_SIZE=BLOCK_SIZE,
                                  HEAD_DIM_CKV=DV, HEAD_DIM_KPE=DPE, num_stages=num_stages, num_warps=num_warps)
        BM._mla_softmax_reducev(logits, o, inp["cache_seqlens"], TRITON_NSPLIT)
        return o
    return f


def triton_select(sh, inp, V) -> dict:
    """As shipped if it launches, else the largest num_stages that fits; adds V["triton_mla"]."""
    import torch
    tri = {"shipped_error": None}
    try:
        f = triton_shipped_fn(sh, inp)
        f()
        torch.cuda.synchronize()
        V["triton_mla"] = f
        tri["used"] = "benchmark_mla.mla_decode_triton as shipped (default num_stages)"
        return tri
    except Exception as e:  # noqa: BLE001 - compile-time OutOfResources, no CUDA state involved
        tri["shipped_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    for st in (2, 1):
        try:
            f = triton_launch_fn(sh, inp, st)
            f()
            torch.cuda.synchronize()
            V["triton_mla"] = f
            tri["used"] = f"same kernels, num_stages={st} (default exceeds 99 KB)"
            break
        except Exception as e:  # noqa: BLE001
            tri[f"num_stages{st}_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return tri


def triton_probe_phase(sh) -> dict:
    """Child: try the Triton MLA kernel once on this shape and check it."""
    import torch
    inp = make_inputs(sh)
    V = {}
    rec = triton_select(sh, inp, V)
    if "triton_mla" in V:
        try:
            rec["check"] = C.err_stats(V["triton_mla"]().view(sh["batch"], H_Q, DV), reference(sh, inp))
        except Exception as e:  # noqa: BLE001
            rec["check_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return rec


def fi_fa3_phase(sh) -> dict:
    """FlashInfer's MLA wrapper with backend="fa3" (what benchmark_mla.py uses), attempted once."""
    import torch
    import flashinfer
    from flashinfer.mla import MLAPlanMetadata
    inp = make_inputs(sh)
    b = sh["batch"]
    lens = inp["cache_seqlens"]
    npages = (lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_indptr = torch.zeros(b + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(npages, 0)
    kv_indices = torch.cat([inp["block_table"][i, :int(npages[i])] for i in range(b)]).to(torch.int32)
    rec = {"causal": True}
    try:
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        w = flashinfer.mla.BatchMLAPagedAttentionWrapper(ws, backend="fa3")
        md = MLAPlanMetadata.csr(qo_indptr=torch.arange(b + 1, dtype=torch.int32, device="cuda"),
                                 kv_indptr=kv_indptr, kv_indices=kv_indices, kv_len_arr=lens)
        w.plan(metadata=md, num_heads=H_Q, head_dim_ckv=DV, head_dim_kpe=DPE, page_size=BLOCK_SIZE, causal=True,
               sm_scale=1 / math.sqrt(D), q_data_type=torch.float16, kv_data_type=torch.float16,
               query_layout="split", kv_cache_layout="split")
        o = w.run(query=(inp["q_nope"].view(-1, H_Q, DV), inp["q_pe"].view(-1, H_Q, DPE)),
                  kv_cache=(inp["k_nope"].view(-1, BLOCK_SIZE, DV), inp["k_pe"].view(-1, BLOCK_SIZE, DPE)))
        torch.cuda.synchronize()
        rec["ok"] = True
        rec["check"] = C.err_stats(o.view(b, H_Q, DV), reference(sh, inp))
    except Exception as e:  # noqa: BLE001 - expected on sm_120, recorded
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    return rec


def shipped_phase(sh) -> dict:
    """Try the example's shipped config (BLOCK_N=64, BLOCK_H=64, num_split=1, 2 stages) once."""
    import torch
    inp = make_inputs(sh)
    rec = dict(SHIPPED)
    try:
        k = tl_kernel(sh, **SHIPPED)
        rec["resources"] = C.tl_resources(k)
        glse = torch.empty(sh["batch"], H_Q, 1, dtype=torch.float16, device="cuda")
        part = torch.empty(sh["batch"], H_Q, 1, DV, dtype=torch.float16, device="cuda")
        o = k(inp["q_nope"].view(-1, H_Q, DV), inp["q_pe"].view(-1, H_Q, DPE), inp["k_nope"].view(-1, H_KV, DV),
              inp["k_pe"].view(-1, H_KV, DPE), inp["block_table"], inp["cache_seqlens"], glse, part)
        torch.cuda.synchronize()
        rec["launch"] = "succeeded"
        rec["check"] = C.err_stats(o, reference(sh, inp))
    except Exception as e:  # noqa: BLE001 - expected, recorded
        rec["launch"] = f"failed: {type(e).__name__}: {str(e)[:400]}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("compile", "run", "shipped", "fi_fa3", "triton_probe"), required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seqlen", type=int, required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()
    sh = shape_of(args.batch, args.seqlen)
    tag = f"b{args.batch}_s{args.seqlen}"
    if args.phase == "compile":
        C.save_json(compile_phase(sh), os.path.join(C.RESULTS, "raw", f"mla_{tag}_compile.json"))
        return
    if args.phase == "shipped":           # child of the run phase (already under the GPU lock)
        C.emit_isolated(shipped_phase(sh))
        return
    if args.phase == "triton_probe":      # child of the run phase (already under the GPU lock)
        C.emit_isolated(triton_probe_phase(sh))
        return
    if args.phase == "fi_fa3":            # child of the run phase (already under the GPU lock)
        C.emit_isolated(fi_fa3_phase(sh))
        return
    res = run_phase(sh, args)
    C.save_json(res, os.path.join(C.RESULTS, "raw", f"mla_{tag}.json"))
    s = res["cobench"]["summary"]
    ref = s.get("tl_adapted", {}).get("median_us")
    print(f"\nMLA {tag}")
    for n, d in sorted(s.items(), key=lambda kv: kv[1]["median_us"]):
        up = res["variants"][n].get("upstream", {})
        tf = sh["flops"] / (d["median_us"] * 1e-6) / 1e12
        print(f"  {n:24s} {d['median_us']:10.1f} us {tf:7.1f} TFLOPS cv {d['cv']:.3f} "
              f"{d.get('clock_mhz_median') or 0:6.0f} MHz  {d['median_us'] / ref if ref else 0:6.2f}x  up {up}")
    t = res["variants"].get("torch_mla", {})
    print("  torch_mla", t.get("upstream"), t.get("check", {}).get("ok"), t.get("error", ""))


if __name__ == "__main__":
    main()
