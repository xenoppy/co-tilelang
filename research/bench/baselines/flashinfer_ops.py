"""FlashInfer baselines for cobench (plan item P1-baselines).

Three workloads, each exposing ``make_inputs()`` (a fresh argument tuple, safe to call
repeatedly for cobench's graph-mode rotation) and ``run(*inputs)`` (launches on torch's
current stream; FlashInfer 0.7.0 binds through tvm-ffi, which launches on the torch
current stream of the tensors' device):

  BatchDecode    GQA decode over a paged KV cache
                 (flashinfer.BatchDecodeWithPagedKVCacheWrapper).
  SinglePrefill  causal prefill of one request (flashinfer.single_prefill_with_kv_cache).
  POD            prefill + decode fused into one launch
                 (flashinfer.PODWithPagedKVCacheWrapper, the POD-Attention port).

Usage with cobench (run from the repo root, env from research/env.sh):

    sys.path.insert(0, "research/bench")
    import cobench as cb
    from baselines import flashinfer_ops as fo
    dec = fo.BatchDecode(fo.DecodeShape(batch=64, kv_len=8192))
    r = cb.bench(dec.run, make_inputs=dec.make_inputs, mode="graph", nbytes=dec.nbytes)
    pre = fo.SinglePrefill(fo.PrefillShape(seq_len=8192))
    a, b = pre.make_inputs(), dec.make_inputs()
    c = cb.bench_corun(lambda: pre.run(*a), lambda: dec.run(*b))

KV-cache layout (decode and the decode half of POD):
  * NHD pages, K and V as two tensors of shape [num_pages, page_size, H_kv, D];
  * request b owns pages [b*P, (b+1)*P) with P = ceil(kv_len / page_size), and the page
    table is the identity. The cache is therefore byte-identical to a dense
    [batch, P*page_size, H_kv, D] tensor (``BatchDecode.dense_kv`` gives that view),
    which is the layout of TileLang's examples/flash_decoding/example_gqa_decode.py.
  * GQA: query head i reads KV head i // (H_q / H_kv) (FlashInfer's convention).

POD caveat (FlashInfer 0.7.0, include/flashinfer/attention/pod.cuh:414-415): the kernel's
CTA-scheduler counters (``static int* tbAssign``, one buffer per process) are zeroed with a
plain ``cudaMemset``, i.e. on the *legacy default stream*, while the kernel is launched on
the current torch stream. Measured consequences (results/2026-09-22_flashinfer_baselines):
  * on a non-blocking stream (torch side stream, green-context stream) back-to-back calls
    return wrong outputs (a later call's memset lands while an earlier kernel is running);
  * under CUDA-graph capture the memset executes once, eagerly, and is not captured: the
    first replay is correct, every later replay finds the counters exhausted, all CTAs exit
    at once (~15 us instead of ~2 ms) and the outputs are left unwritten.
``POD.run`` therefore raises on any stream other than the legacy default stream (unless
``unsafe_stream_ok=True``) and always raises during stream capture.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

import flashinfer

WORKSPACE_BYTES = 128 << 20      # FlashInfer's recommended float workspace (split-KV partials)


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecodeShape:
    batch: int
    kv_len: int
    num_qo_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    page_size: int = 16
    dtype: torch.dtype = torch.bfloat16

    @property
    def group(self) -> int:
        return self.num_qo_heads // self.num_kv_heads

    @property
    def pages_per_req(self) -> int:
        return math.ceil(self.kv_len / self.page_size)

    @property
    def last_page_len(self) -> int:
        return self.kv_len - (self.pages_per_req - 1) * self.page_size

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dtype"] = _dtype_name(self.dtype)
        return d


@dataclass(frozen=True)
class PrefillShape:
    seq_len: int                      # qo_len == kv_len (one request, no KV prefix)
    num_qo_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    causal: bool = True
    dtype: torch.dtype = torch.bfloat16

    @property
    def group(self) -> int:
        return self.num_qo_heads // self.num_kv_heads

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dtype"] = _dtype_name(self.dtype)
        return d


def _check_heads(num_qo_heads: int, num_kv_heads: int) -> None:
    if num_qo_heads % num_kv_heads:
        raise ValueError(f"num_qo_heads={num_qo_heads} is not a multiple of num_kv_heads={num_kv_heads}")


def _gen(seed, device):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _randn(shape, dtype, device, gen):
    return torch.randn(shape, dtype=torch.float32, device=device, generator=gen).to(dtype)


# ---------------------------------------------------------------------------
# fp32 references
# ---------------------------------------------------------------------------
def decode_reference(q: torch.Tensor, k_dense: torch.Tensor, v_dense: torch.Tensor,
                     sm_scale: float | None = None, chunk: int = 8) -> torch.Tensor:
    """fp32 GQA decode. q [B, Hq, D]; k_dense/v_dense [B, S, Hkv, D] -> [B, Hq, D] (fp32)."""
    B, Hq, D = q.shape
    Hkv = k_dense.shape[2]
    G = Hq // Hkv
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    out = torch.empty(B, Hq, D, dtype=torch.float32, device=q.device)
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        qf = q[b0:b1].float().view(b1 - b0, Hkv, G, D)
        kf = k_dense[b0:b1].float()
        vf = v_dense[b0:b1].float()
        s = torch.einsum("bhgd,bshd->bhgs", qf, kf) * scale
        p = torch.softmax(s, dim=-1)
        out[b0:b1] = torch.einsum("bhgs,bshd->bhgd", p, vf).reshape(b1 - b0, Hq, D)
    return out


def prefill_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                      sm_scale: float | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """torch SDPA reference. q [S, Hq, D], k/v [S, Hkv, D] (NHD) -> [S, Hq, D] in ``dtype``.
    KV heads are expanded with repeat_interleave (query head i -> KV head i // G)."""
    S, Hq, D = q.shape
    G = Hq // k.shape[1]
    qh = q.to(dtype).transpose(0, 1).unsqueeze(0)                              # [1, Hq, S, D]
    kh = k.to(dtype).repeat_interleave(G, dim=1).transpose(0, 1).unsqueeze(0)
    vh = v.to(dtype).repeat_interleave(G, dim=1).transpose(0, 1).unsqueeze(0)
    o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=causal, scale=sm_scale)
    return o.squeeze(0).transpose(0, 1).contiguous()


def err_stats(out: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (out.float() - ref.float()).abs()
    rmax = ref.float().abs().max().item()
    return {"max_abs_err": d.max().item(), "mean_abs_err": d.mean().item(),
            "max_abs_ref": rmax, "max_rel_err_vs_max_ref": d.max().item() / rmax if rmax else float("nan"),
            "finite": bool(torch.isfinite(out.float()).all().item())}


# ---------------------------------------------------------------------------
# paged KV helpers
# ---------------------------------------------------------------------------
def _page_table(s: DecodeShape, device):
    P = s.pages_per_req
    indptr = torch.arange(0, (s.batch + 1) * P, P, dtype=torch.int32, device=device)
    indices = torch.arange(s.batch * P, dtype=torch.int32, device=device)
    last = torch.full((s.batch,), s.last_page_len, dtype=torch.int32, device=device)
    return indptr, indices, last


def _kv_cache(s: DecodeShape, device, gen):
    shape = (s.batch * s.pages_per_req, s.page_size, s.num_kv_heads, s.head_dim)
    return _randn(shape, s.dtype, device, gen), _randn(shape, s.dtype, device, gen)


def _dense(s: DecodeShape, cache: torch.Tensor) -> torch.Tensor:
    """[num_pages, page, Hkv, D] -> [B, kv_len, Hkv, D] view (identity page table)."""
    return cache.view(s.batch, s.pages_per_req * s.page_size, s.num_kv_heads, s.head_dim)[:, : s.kv_len]


def _decode_bytes(s: DecodeShape) -> int:
    e = torch.tensor([], dtype=s.dtype).element_size()
    kv = 2 * s.batch * s.kv_len * s.num_kv_heads * s.head_dim * e
    qo = 2 * s.batch * s.num_qo_heads * s.head_dim * e
    return kv + qo


def _decode_flops(s: DecodeShape) -> float:
    return 4.0 * s.batch * s.num_qo_heads * s.kv_len * s.head_dim


def _prefill_bytes(s: PrefillShape) -> int:
    e = torch.tensor([], dtype=s.dtype).element_size()
    return (2 * s.seq_len * s.num_qo_heads + 2 * s.seq_len * s.num_kv_heads) * s.head_dim * e


def _prefill_flops(s: PrefillShape) -> float:
    """QK^T + PV, 2 FLOP per MAC; causal counts half of the S x S score matrix
    (the plan's convention 2*2*S^2*D*H/2; the exact causal count is S(S+1)/2 pairs)."""
    f = 4.0 * s.seq_len * s.seq_len * s.head_dim * s.num_qo_heads
    return f / 2 if s.causal else f


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------
class BatchDecode:
    """GQA decode, one query token per request, full paged KV cache.

    use_tensor_cores=False -> FlashInfer's CUDA-core batch-decode kernel (decode.cuh);
    use_tensor_cores=True  -> the FA2 batch-prefill kernel with a 16-row Q tile
                              (the same code path as the decode half of POD).
    The plan (split-KV schedule) is computed once in __init__ for the full device; it is
    shape-only, so every input copy from make_inputs() shares it.
    """

    def __init__(self, shape: DecodeShape, *, use_tensor_cores: bool = False, device="cuda",
                 backend: str = "auto"):
        _check_heads(shape.num_qo_heads, shape.num_kv_heads)
        self.shape, self.device = shape, torch.device(device)
        self.use_tensor_cores = use_tensor_cores
        self.workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, "NHD", use_tensor_cores=use_tensor_cores, backend=backend)
        self.page_table = _page_table(shape, self.device)
        self.wrapper.plan(*self.page_table, shape.num_qo_heads, shape.num_kv_heads, shape.head_dim,
                          shape.page_size, q_data_type=shape.dtype, kv_data_type=shape.dtype)
        self.nbytes = _decode_bytes(shape)
        self.flops = _decode_flops(shape)

    def make_inputs(self, seed=None):
        s = self.shape
        g = _gen(seed, self.device)
        q = _randn((s.batch, s.num_qo_heads, s.head_dim), s.dtype, self.device, g)
        k, v = _kv_cache(s, self.device, g)
        out = torch.empty_like(q)
        return q, k, v, out

    def run(self, q, k_cache, v_cache, out):
        return self.wrapper.run(q, (k_cache, v_cache), out=out)

    def dense_kv(self, k_cache, v_cache):
        return _dense(self.shape, k_cache), _dense(self.shape, v_cache)

    def reference(self, q, k_cache, v_cache, out=None):
        return decode_reference(q, *self.dense_kv(k_cache, v_cache))

    def describe(self) -> dict:
        return {"api": "BatchDecodeWithPagedKVCacheWrapper", "kv_layout": "NHD paged, separate K/V, identity page table",
                "use_tensor_cores": self.use_tensor_cores, "backend": getattr(self.wrapper, "_backend", None),
                "shape": self.shape.to_dict(), "nbytes": self.nbytes, "flops": self.flops}


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------
class SinglePrefill:
    """Causal prefill of one request (qo_len == kv_len), NHD q/k/v, FlashInfer backend 'auto'
    (FA2 on sm_120). single_prefill_with_kv_cache allocates its output and a 32 MB split-KV
    scratch per call (torch caching allocator; host-side only)."""

    def __init__(self, shape: PrefillShape, *, device="cuda", backend: str = "auto"):
        _check_heads(shape.num_qo_heads, shape.num_kv_heads)
        self.shape, self.device, self.backend = shape, torch.device(device), backend
        self.nbytes = _prefill_bytes(shape)
        self.flops = _prefill_flops(shape)

    def make_inputs(self, seed=None):
        s = self.shape
        g = _gen(seed, self.device)
        q = _randn((s.seq_len, s.num_qo_heads, s.head_dim), s.dtype, self.device, g)
        k = _randn((s.seq_len, s.num_kv_heads, s.head_dim), s.dtype, self.device, g)
        v = _randn((s.seq_len, s.num_kv_heads, s.head_dim), s.dtype, self.device, g)
        return q, k, v

    def run(self, q, k, v):
        return flashinfer.single_prefill_with_kv_cache(q, k, v, causal=self.shape.causal, kv_layout="NHD",
                                                       backend=self.backend)

    def reference(self, q, k, v, dtype=torch.float32):
        return prefill_reference(q, k, v, causal=self.shape.causal, dtype=dtype)

    def describe(self) -> dict:
        return {"api": "single_prefill_with_kv_cache", "kv_layout": "NHD ragged (one request)",
                "backend": self.backend, "shape": self.shape.to_dict(), "nbytes": self.nbytes, "flops": self.flops}


# ---------------------------------------------------------------------------
# POD (prefill + decode in one kernel)
# ---------------------------------------------------------------------------
class POD:
    """FlashInfer's POD-Attention port: one launch computes a causal single-request prefill and
    a batch decode; each CTA picks its role at run time from a per-SM ticket (%smid).
    Prefill and decode must share num_qo_heads, num_kv_heads and head_dim (checked by FlashInfer).
    The decode half is the FA2 tensor-core kernel with a 16-row Q tile (plan() of the
    batch-prefill module, like BatchDecode(use_tensor_cores=True)).
    """

    def __init__(self, prefill: PrefillShape, decode: DecodeShape, *, device="cuda",
                 unsafe_stream_ok: bool = False):
        for f in ("num_qo_heads", "num_kv_heads", "head_dim", "dtype"):
            if getattr(prefill, f) != getattr(decode, f):
                raise ValueError(f"POD needs equal {f} for prefill and decode")
        _check_heads(decode.num_qo_heads, decode.num_kv_heads)
        self.prefill_shape, self.decode_shape = prefill, decode
        self.device = torch.device(device)
        self.unsafe_stream_ok = unsafe_stream_ok
        self.workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
        self.wrapper = flashinfer.PODWithPagedKVCacheWrapper(self.workspace, "NHD")
        self.page_table = _page_table(decode, self.device)
        self.wrapper.plan(*self.page_table, decode.num_qo_heads, decode.num_kv_heads, decode.head_dim,
                          decode.page_size, q_data_type=decode.dtype, kv_data_type=decode.dtype)
        self.nbytes = _prefill_bytes(prefill) + _decode_bytes(decode)
        self.flops = _prefill_flops(prefill) + _decode_flops(decode)

    def make_inputs(self, seed=None):
        g = _gen(seed, self.device)
        p, d = self.prefill_shape, self.decode_shape
        q_p = _randn((p.seq_len, p.num_qo_heads, p.head_dim), p.dtype, self.device, g)
        k_p = _randn((p.seq_len, p.num_kv_heads, p.head_dim), p.dtype, self.device, g)
        v_p = _randn((p.seq_len, p.num_kv_heads, p.head_dim), p.dtype, self.device, g)
        q_d = _randn((d.batch, d.num_qo_heads, d.head_dim), d.dtype, self.device, g)
        k_c, v_c = _kv_cache(d, self.device, g)
        return q_p, k_p, v_p, q_d, k_c, v_c

    def _check_stream(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer 0.7.0 POD cannot be captured in a CUDA graph: its scheduler counters are "
                "reset by an uncaptured cudaMemset, so every replay after the first computes nothing.")
        if self.unsafe_stream_ok:
            return
        s = torch.cuda.current_stream(self.device)
        if s.cuda_stream != 0:
            raise RuntimeError(
                "FlashInfer 0.7.0 POD zeroes its CTA-scheduler counters with cudaMemset on the legacy "
                "default stream; on a non-blocking stream (torch side stream, green context) that "
                "memset is unordered w.r.t. the kernel. Run POD on the default stream, or pass "
                "unsafe_stream_ok=True knowingly.")

    def run(self, q_p, k_p, v_p, q_d, k_cache, v_cache):
        self._check_stream()
        return self.wrapper.run(q_p, k_p, v_p, q_d, (k_cache, v_cache), causal_p=self.prefill_shape.causal,
                                kv_layout_p="NHD")

    def reference(self, q_p, k_p, v_p, q_d, k_cache, v_cache):
        d = self.decode_shape
        o_p = prefill_reference(q_p, k_p, v_p, causal=self.prefill_shape.causal)
        o_d = decode_reference(q_d, _dense(d, k_cache), _dense(d, v_cache))
        return o_p, o_d

    def split_inputs(self, inputs):
        """(prefill args for SinglePrefill.run, decode args minus out for BatchDecode.run)."""
        q_p, k_p, v_p, q_d, k_c, v_c = inputs
        return (q_p, k_p, v_p), (q_d, k_c, v_c)

    def describe(self) -> dict:
        return {"api": "PODWithPagedKVCacheWrapper", "prefill": self.prefill_shape.to_dict(),
                "decode": self.decode_shape.to_dict(), "nbytes": self.nbytes, "flops": self.flops}


def serial(prefill: SinglePrefill, decode: BatchDecode):
    """fn(*inputs) running prefill then decode on the current stream; inputs = prefill args + decode args."""
    def fn(q_p, k_p, v_p, q_d, k_c, v_c, out_d):
        prefill.run(q_p, k_p, v_p)
        decode.run(q_d, k_c, v_c, out_d)
    return fn


def serial_inputs(prefill: SinglePrefill, decode: BatchDecode, seed=None):
    a = prefill.make_inputs(seed)
    b = decode.make_inputs(None if seed is None else seed + 1)
    return (*a, *b)


def versions() -> dict:
    from flashinfer.jit import env as jit_env
    return {"flashinfer": flashinfer.__version__, "torch": torch.__version__,
            "jit_workspace_dir": str(jit_env.FLASHINFER_WORKSPACE_DIR)}
