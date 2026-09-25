"""Shared helpers for the Mamba-2 claims sub-study (claims C3 / C6-mamba).

Results: research/results/2026-09-24_claims_repro/C_mamba/.

* TileLang kernels are imported, unmodified, from
    examples/linear_attention/example_mamba_chunk_scan.py   (kind "scan_ex",  C3 chunk-scan)
    examples/linear_attention/example_mamba_chunk_state.py  (kind "state_ex", C3 chunk-state)
    benchmark/mamba2/benchmark_mamba_chunk_scan.py          (kind "scan_bm",  C6)
  benchmark_mamba_chunk_scan.py imports helion at module level and raises ImportError without it; when
  helion is not installed, a stub module is put in sys.modules *only* so that the TileLang kernel and
  the Triton wrapper of that file can be imported (the helion kernel itself is never run). The stub
  raises if anything in it is called.
* The Triton baseline is mamba-ssm 2.2.6.post3's pure-Triton SSD kernels, vendored under
  research/bench/baselines/mamba_ssm_triton (see mamba_ssm/__init__.py there for provenance).
* Inputs are generated with realistic Mamba-2 statistics (dt = softplus(.) > 0, A < 0, dA_cumsum the
  within-chunk cumulative sum of dt*A, cb = C B^T per chunk). With these inputs the exp(min(., 0))
  clamps that mamba-ssm >= 2.2.5 added are inactive in the causal region, so TileLang and Triton
  compute the same function (checked numerically against an fp32 torch reference).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH = os.path.join(ROOT, "research", "bench")
VENDOR = os.path.join(BENCH, "baselines", "mamba_ssm_triton")
RESULTS = os.path.join(ROOT, "research", "results", "2026-09-24_claims_repro", "C_mamba")
for p in (BENCH, VENDOR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

SMEM_OPTIN = 101376  # bytes per CTA on sm_120

# --------------------------------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------------------------------
# C3: tilelang-benchmark README "Table 3: Linear Attention shapes" = paper arXiv:2504.17577 Table 4
# (appendix A). Figure (images/mha_performance_h100.png = paper Fig. 12) shows CC0-CC4 / CT0-CT4;
# CC5/CT5 are in the table only. chunk_size and ngroups are not given anywhere -> the example
# defaults (chunk_size=256, ngroups=1; also the Mamba-2 defaults) are used.
_C3_BL = [(1, 1024), (1, 2048), (1, 8192), (64, 1024), (64, 2048), (64, 8192)]
SHAPES = {}
for i, (b, L) in enumerate(_C3_BL):
    for pre, kind in (("CC", "scan_ex"), ("CT", "state_ex")):
        SHAPES[f"{pre}{i}"] = dict(kind=kind, batch=b, nheads=64, seqlen=L, headdim=64, dstate=128,
                                   chunk_size=256, ngroups=1)
# C6: benchmark/mamba2/README.md (batch 8, heads 80, groups 1, chunk 256, dim 64, dstate 128).
C6_SEQ = [1024, 2048, 4096, 8192, 16384, 32768]
for L in C6_SEQ:
    SHAPES[f"B{L}"] = dict(kind="scan_bm", batch=8, nheads=80, seqlen=L, headdim=64, dstate=128,
                           chunk_size=256, ngroups=1)


def flops(shape: dict) -> float:
    """FLOP counts used by the upstream scripts (examples' / benchmark README's total_flops)."""
    b, L, h, p, n, cs = (shape[k] for k in ("batch", "seqlen", "nheads", "headdim", "dstate", "chunk_size"))
    if shape["kind"] in ("scan_ex", "scan_bm"):
        return 2 * b * L * cs * h * p * 0.5 + 2 * b * L * h * p * n
    return 2 * b * L * h * p * n


def min_bytes(shape: dict) -> float:
    """Compulsory DRAM traffic (each input read once, output written once, fp16) of one call."""
    b, L, h, p, n, cs, g = (shape[k] for k in ("batch", "seqlen", "nheads", "headdim", "dstate", "chunk_size", "ngroups"))
    nc = L // cs
    x, dt_da = b * L * h * p, 2 * b * h * L
    if shape["kind"] in ("scan_ex", "scan_bm"):
        return 2.0 * (b * nc * g * cs * cs + x + dt_da + b * L * g * n + b * nc * h * p * n + h + x)  # cb x dt dA C states D | out
    return 2.0 * (b * L * g * n + x + dt_da + b * nc * h * p * n)  # B x dt dA | states


# --------------------------------------------------------------------------------------------------
# Module loading
# --------------------------------------------------------------------------------------------------
_MODS = {}
HELION_STUBBED = False


def _load(path: str, name: str):
    if name in _MODS:
        return _MODS[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODS[name] = mod
    return mod


def _install_helion_stub():
    global HELION_STUBBED
    try:
        import helion  # noqa: F401
        return
    except ImportError:
        pass

    def _unavailable(*a, **k):
        raise RuntimeError("helion is not installed; this is an import stub (claims_mamba_common)")

    h = types.ModuleType("helion")
    h.kernel = _unavailable
    t = types.ModuleType("helion._testing")
    t.run_example = _unavailable
    lang = types.ModuleType("helion.language")  # only imported at module level; hl.* is used inside the
    # never-called helion kernel. (No module __getattr__: it would also answer __file__ & co. and break
    # inspect.getmodule, which torch._dynamo runs over sys.modules.)
    h._testing, h.language = t, lang
    sys.modules.update({"helion": h, "helion._testing": t, "helion.language": lang})
    HELION_STUBBED = True


def module(kind: str):
    if kind == "scan_ex":
        return _load(os.path.join(ROOT, "examples/linear_attention/example_mamba_chunk_scan.py"), "tl_ex_mamba_scan")
    if kind == "state_ex":
        return _load(os.path.join(ROOT, "examples/linear_attention/example_mamba_chunk_state.py"), "tl_ex_mamba_state")
    if kind == "scan_bm":
        _install_helion_stub()
        return _load(os.path.join(ROOT, "benchmark/mamba2/benchmark_mamba_chunk_scan.py"), "tl_bm_mamba_scan")
    raise ValueError(kind)


def triton_modules():
    from mamba_ssm.ops.triton import ssd_chunk_scan, ssd_chunk_state
    return ssd_chunk_scan, ssd_chunk_state


def shape_args(shape: dict) -> dict:
    return dict(batch=shape["batch"], seqlen=shape["seqlen"], chunk_size=shape["chunk_size"],
                ngroups=shape["ngroups"], nheads=shape["nheads"], headdim=shape["headdim"], dstate=shape["dstate"])


def tilelang_kernel(shape: dict, config: dict | None, threads: int = 128):
    """Compile (or fetch from the kernel cache) the upstream TileLang kernel.
    config None -> run the upstream autotuner as shipped (returns the tuned kernel)."""
    mod = module(shape["kind"])
    fn = mod.chunk_scan_fwd if shape["kind"] in ("scan_ex", "scan_bm") else mod.chunk_state_fwd
    sa = shape_args(shape)
    if shape["kind"] == "scan_bm":
        pos = (sa["batch"], sa["seqlen"], sa["chunk_size"], sa["ngroups"], sa["nheads"], sa["headdim"], sa["dstate"])
        if config is None:
            return fn(*pos)
        return fn.compile(*pos, **config, threads=threads)
    if config is None:
        return fn.compile(**sa)
    return fn.compile(**sa, **config, threads=threads)


# --------------------------------------------------------------------------------------------------
# Inputs, calls, references
# --------------------------------------------------------------------------------------------------
def make_inputs(shape: dict, seed: int = 0, device: str = "cuda") -> dict:
    b, L, h, p, n, cs, g = (shape[k] for k in ("batch", "seqlen", "nheads", "headdim", "dstate", "chunk_size", "ngroups"))
    assert L % cs == 0, "the TileLang kernels assume seqlen % chunk_size == 0"
    nc = L // cs
    gen = torch.Generator(device=device).manual_seed(seed)
    f16 = torch.float16

    def rn(*s, dtype=torch.float32):
        return torch.randn(*s, generator=gen, device=device, dtype=dtype)

    x = rn(b, L, h, p, dtype=f16)
    dt = torch.nn.functional.softplus(rn(b, h, nc, cs) - 2.0)                  # > 0, mean ~0.13
    A = -torch.exp(rn(h) * 0.5)                                              # < 0
    dA_cumsum = torch.cumsum(dt * A[None, :, None, None], dim=-1)           # non-increasing within a chunk
    Bm = rn(b, L, g, n) * n ** -0.25
    Cm = rn(b, L, g, n) * n ** -0.25
    out = dict(x=x, dt=dt.to(f16), dA_cumsum=dA_cumsum.to(f16), B=Bm.to(f16))
    if shape["kind"] in ("scan_ex", "scan_bm"):
        cb = torch.einsum("bclgn,bcsgn->bcgls", Cm.view(b, nc, cs, g, n), Bm.view(b, nc, cs, g, n))
        out.update(cb=cb.to(f16).contiguous(), C=Cm.to(f16), states=rn(b, nc, h, p, n, dtype=f16) * 0.1,
                   D=rn(h).to(f16))
        del cb
    del Bm, Cm, dt, dA_cumsum
    return out


def call_args(shape: dict, inp: dict) -> tuple:
    if shape["kind"] in ("scan_ex", "scan_bm"):
        return (inp["cb"], inp["x"], inp["dt"], inp["dA_cumsum"], inp["C"], inp["states"], inp["D"])
    return (inp["B"], inp["x"], inp["dt"], inp["dA_cumsum"])


def triton_fn(shape: dict):
    """The Triton baseline exactly as the upstream TileLang files call it."""
    mod = module(shape["kind"])
    if shape["kind"] in ("scan_ex", "scan_bm"):
        return mod.chunk_scan_triton          # _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D)[0]
    return mod.chunk_state_triton             # _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=False)


def triton_best_config(shape: dict) -> str | None:
    scan, state = triton_modules()
    k = scan._chunk_scan_fwd_kernel if shape["kind"] in ("scan_ex", "scan_bm") else state._chunk_state_fwd_kernel
    return str(k.best_config) if getattr(k, "best_config", None) is not None else None


def reference_error(shape: dict, inp: dict, outs: dict, max_chunks: int = 16) -> dict:
    """Compare each output in `outs` (name -> tensor) with the upstream torch reference (the example's
    ref_program) evaluated in fp32, sliced over (batch, chunk-block) to bound memory. Metrics per
    output: max_abs, rel_l2, mismatch fraction at atol = rtol = 1e-2 (the examples' tolerances)."""
    mod = module(shape["kind"])
    ref_prog = mod.ref_program
    b, L, cs = shape["batch"], shape["seqlen"], shape["chunk_size"]
    nc = L // cs
    acc = {k: dict(max_abs=0.0, se=0.0, mism=0, n=0) for k in outs}
    ref_sq, ref_max = 0.0, 0.0
    f32 = lambda t: t.float()  # noqa: E731
    for bi in range(b):
        for c0 in range(0, nc, max_chunks):
            c1 = min(nc, c0 + max_chunks)
            sb, sl = slice(bi, bi + 1), slice(c0 * cs, c1 * cs)
            if shape["kind"] in ("scan_ex", "scan_bm"):
                ref = ref_prog(f32(inp["cb"][sb, c0:c1]), f32(inp["x"][sb, sl]), f32(inp["dt"][sb, :, c0:c1]),
                               f32(inp["dA_cumsum"][sb, :, c0:c1]), f32(inp["C"][sb, sl]),
                               f32(inp["states"][sb, c0:c1]), f32(inp["D"]))
                get = lambda o: o[sb, sl]  # noqa: E731
            else:
                ref = ref_prog(f32(inp["B"][sb, sl]), f32(inp["x"][sb, sl]), f32(inp["dt"][sb, :, c0:c1]),
                               f32(inp["dA_cumsum"][sb, :, c0:c1]))
                get = lambda o: o[sb, c0:c1]  # noqa: E731
            ref_sq += float(ref.double().pow(2).sum())
            ref_max = max(ref_max, float(ref.abs().max()))
            for k, o in outs.items():
                d = (get(o).float() - ref).abs()
                a = acc[k]
                a["max_abs"] = max(a["max_abs"], float(d.max()))
                a["se"] += float(d.double().pow(2).sum())
                a["mism"] += int((d > 1e-2 + 1e-2 * ref.abs()).sum())
                a["n"] += d.numel()
                a["nonfinite"] = a.get("nonfinite", 0) + int((~torch.isfinite(get(o))).sum())
            del ref
    res = {}
    for k, a in acc.items():
        rel = math.sqrt(a["se"] / ref_sq) if ref_sq > 0 else float("nan")
        mism = a["mism"] / max(1, a["n"])
        res[k] = dict(max_abs=a["max_abs"], rel_l2=rel, mismatch_frac=mism, nonfinite=a["nonfinite"],
                      ref_max_abs=ref_max, ok=bool(a["nonfinite"] == 0 and rel <= 1e-2 and mism <= 0.01))
    return res


def pair_error(a: torch.Tensor, b: torch.Tensor, chunk: int = 1 << 26) -> dict:
    """max |a - b| and ||a - b|| / ||b||, accumulated over flat chunks (outputs reach 4 GB at CC5/CT5)."""
    assert a.shape == b.shape, (a.shape, b.shape)
    fa, fb = a.reshape(-1), b.reshape(-1)
    mx, se, sb = 0.0, 0.0, 0.0
    for i in range(0, fa.numel(), chunk):
        x, y = fa[i:i + chunk].float(), fb[i:i + chunk].float()
        d = (x - y).abs()
        mx = max(mx, float(d.max()))
        se += float(d.double().pow(2).sum())
        sb += float(y.double().pow(2).sum())
    return dict(max_abs=mx, rel_l2=(se ** 0.5) / (sb ** 0.5) if sb > 0 else float("nan"))


# --------------------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------------------
class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def capture_stdout():
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    try:
        yield buf
    finally:
        sys.stdout = old


def dump(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def env_info() -> dict:
    import triton
    import tilelang
    import mamba_ssm
    return dict(torch=torch.__version__, triton=triton.__version__, tilelang=tilelang.__version__,
                mamba_ssm=mamba_ssm.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), helion_stubbed=HELION_STUBBED)
