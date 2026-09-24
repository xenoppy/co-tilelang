"""Test/characterisation harness for the op library (used by test_ops.py).

Phases:
  1. build + compile every config of an op (grid and persistent builds) with
     tilelang.par_compile — CPU only, explicit target, no CUDA context;
  2. resource signatures from the compiled kernels (CPU only);
  3. GPU (after checking no other compute process is present): correctness of both
     builds against a torch fp32 reference, bitwise grid-vs-persistent identity,
     bitwise identity within numerics-key groups, repeated launches of split
     configs (outputs and workspaces poisoned with NaN before every launch, counters
     checked to be zero after every launch), and a driver cross-check of registers /
     static smem / CTAs-per-SM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

from cotile import resources
from cotile.device import DEFAULT_DEVICE
from cotile.kernel import Runner, ceildiv, compile_specs
from cotile.ops import gemm, gqa_decode, rmsnorm

TEST_SHAPES = {
    gemm.NAME: gemm.GemmShape(M=1024, N=4096, K=4096),
    gqa_decode.NAME: gqa_decode.DecodeShape(batch=16, seqlen=2048, heads=32, kv_heads=8, dim=128),
    rmsnorm.NAME: rmsnorm.RMSNormShape(tokens=4096, hidden=4096),
}
OPS = {m.NAME: m for m in (gemm, gqa_decode, rmsnorm)}


def persistent_ctas(num_tiles: int, dev=DEFAULT_DEVICE) -> int:
    """CTA count for the persistent test build: at most one per SM, and few enough that
    each CTA runs ~3 tiles when the tile count is small (exercises the grid-stride loop
    and a partial last wave)."""
    return max(1, min(dev.num_sms, ceildiv(num_tiles, 3)))


@dataclass
class OpRun:
    op: object
    shape: object
    configs: list
    specs: dict = field(default_factory=dict)  # (cfg, build) -> KernelSpec
    compile_stats: dict = field(default_factory=dict)
    signatures: list = field(default_factory=list)
    results: list = field(default_factory=list)  # per (cfg) dict
    groups: dict = field(default_factory=dict)


def build_and_compile(op, shape, configs=None, num_workers: int = 32) -> OpRun:
    configs = list(configs if configs is not None else op.configs(shape))
    run = OpRun(op=op, shape=shape, configs=configs)
    t0 = time.time()
    for cfg in configs:
        g = op.build_grid(shape, cfg)
        run.specs[(cfg, "grid")] = g
        run.specs[(cfg, "persistent")] = op.build_persistent(shape, cfg, persistent_ctas(g.num_tiles))
    t_build = time.time() - t0
    stats = compile_specs(list(run.specs.values()), num_workers=num_workers)
    stats["build_s"] = t_build
    stats["n_kernels"] = len(run.specs)
    run.compile_stats = stats
    return run


def check_tile_space(op, shape, cfg) -> bool:
    """decode() must map tile ids 0..num_tiles-1 bijectively onto distinct in-range
    coordinates (evaluated with Python ints)."""
    ts = op.tile_space(shape, cfg)
    coords = [tuple(int(v) for v in ts.decode(t)) for t in range(ts.num_tiles)]
    if len(set(coords)) != ts.num_tiles:
        return False
    if op.NAME == "gemm":
        m_t, n_t = shape.M // cfg.block_M, shape.N // cfg.block_N
        return all(0 <= tm < m_t and 0 <= tn < n_t and 0 <= ks < cfg.split_k and mn == tm * n_t + tn for tm, tn, ks, mn in coords)
    if op.NAME == "gqa_decode":
        HB = shape.group // cfg.heads_per_cta
        return all(
            0 <= b < shape.batch and 0 <= kvh < shape.kv_heads and 0 <= hb < HB and 0 <= s < cfg.num_split and g == (b * shape.kv_heads + kvh) * HB + hb
            for b, kvh, hb, s, g in coords
        )
    if op.NAME == "rmsnorm":
        return sorted(c[0] for c in coords) == list(range(0, shape.tokens, cfg.rows_per_cta))
    return True


def extract_signatures(run: OpRun) -> None:
    for (cfg, build), spec in run.specs.items():
        if spec.kernel is None:
            continue
        sig = resources.signature(spec)
        spec.extra["signature"] = sig
        run.signatures.append(sig)


# ----------------------------------------------------------------------------------
# GPU phase
# ----------------------------------------------------------------------------------


def other_gpu_processes() -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    me = os.getpid()
    rows = [r.strip() for r in out.splitlines() if r.strip()]
    return [r for r in rows if int(r.split(",")[0]) != me]


def _cobench():
    """cobench lives in research/bench (not an installed package)."""
    bench = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "research", "bench"))
    if bench not in sys.path:
        sys.path.insert(0, bench)
    import cobench

    return cobench


def wait_for_gpu(max_wait_s: float | None = None, poll_s: float | None = None) -> bool:
    """GPU-sharing rule (research/rules.md 7, cobench.GuardPolicy as applied since 2026-09-23):
    the GPU is occupied while a foreign process (another user's) shows SM activity in
    `nvidia-smi pmon`; processes that only hold memory do not count; wait policy.poll_s
    (30 min) between checks. Uses the shared guard (cobench.guard). max_wait_s / poll_s
    override the policy (None: the policy's 2 h / 30 min). Returns True when free, False if
    still occupied after max_wait_s."""
    guard = _cobench().guard
    kw = {} if poll_s is None else {"poll_s": poll_s}
    if max_wait_s is not None:
        kw["max_wait_s"] = max_wait_s
    try:
        guard.wait_until_free(log=lambda m: print(f"[gpu] {m}", flush=True), **kw)
        return True
    except guard.GpuBusy as e:
        print(f"[gpu] {e}", flush=True)
        return False


def compare(out, ref, tol) -> dict:
    """|out - ref| <= atol*rms(ref) + rtol*|ref| elementwise (tol = (atol, rtol))."""
    import torch

    atol, rtol = tol
    o = out.float()
    finite = bool(torch.isfinite(o).all())
    d = (o - ref).abs()
    rms = ref.pow(2).mean().sqrt()
    bound = atol * rms + rtol * ref.abs()
    worst = float((d / bound).max()) if finite else float("inf")
    return {
        "ok": finite and worst <= 1.0,
        "max_abs_err": float(d.max()) if finite else float("inf"),
        "max_err_over_rms": float(d.max() / rms) if finite else float("inf"),
        "worst_ratio": worst,
    }


def _poison(runner: Runner, outputs: dict) -> None:
    for t in outputs.values():
        t.fill_(float("nan"))
    for p in runner.spec.params:
        if p.role == "ws" and p.name in runner.state:
            runner.state[p.name].fill_(float("nan"))


def _launch(runner: Runner, inputs: dict, outputs: dict | None = None) -> dict:
    import torch

    if outputs is None:
        _, outputs = runner.make_args(inputs)
    _poison(runner, outputs)
    runner(inputs, outputs)
    torch.cuda.synchronize()
    return outputs


def gpu_tests(run: OpRun, repeat: int = 3, driver_check: bool = True) -> None:
    import torch

    op, shape = run.op, run.shape
    inputs = op.make_inputs(shape, seed=0)
    ref = op.reference(shape, inputs)
    out_name = next(p.name for p in op.io_params(shape, run.configs[0]) if p.role == "out")
    ref_t = ref[out_name]
    group_ref = {}  # numerics key -> (tag, tensor)
    for cfg in run.configs:
        cls, key = op.numerics(cfg)
        res = {"op": op.NAME, "config": op.cfg_tag(cfg), "numerics": cls, "numerics_key": repr(key)}
        res["tile_space_ok"] = check_tile_space(op, shape, cfg)
        outs = {}
        for build in ("grid", "persistent"):
            spec = run.specs[(cfg, build)]
            if spec.kernel is None:
                res[f"{build}_ok"] = False
                res[f"{build}_error"] = spec.compile_error
                continue
            runner = Runner(spec)
            try:
                o = _launch(runner, inputs)[out_name]
            except Exception as e:  # noqa: BLE001 - launch failures are results
                res[f"{build}_ok"] = False
                res[f"{build}_error"] = f"launch: {type(e).__name__}: {str(e).splitlines()[-1][:300]}"
                continue
            cmp = compare(o, ref_t, op.TOLERANCE)
            res[f"{build}_ok"] = cmp["ok"]
            res[f"{build}_max_abs_err"] = cmp["max_abs_err"]
            res[f"{build}_max_err_over_rms"] = cmp["max_err_over_rms"]
            res[f"{build}_worst_ratio"] = cmp["worst_ratio"]
            outs[build] = o
            has_ctr = any(p.role == "ctr" for p in spec.params)
            if has_ctr:
                rep_ok, zero_ok = True, True
                for _ in range(repeat):
                    o2 = _launch(runner, inputs)[out_name]
                    rep_ok &= bool(torch.equal(o2, o)) and compare(o2, ref_t, op.TOLERANCE)["ok"]
                    zero_ok &= all(bool((c == 0).all()) for c in runner.counters().values())
                res[f"{build}_repeat_ok"] = rep_ok
                res[f"{build}_counters_zero"] = zero_ok
            if driver_check:
                sig = spec.extra.get("signature")
                if sig is not None:
                    try:
                        drv = resources.driver_check(spec, sig["threads"], sig["smem_dynamic"])
                        sig.update({f"drv_{k}": v for k, v in drv.items()})
                    except Exception as e:  # noqa: BLE001
                        sig["drv_error"] = str(e)[:200]
        if "grid" in outs and "persistent" in outs:
            res["grid_eq_persistent"] = bool(torch.equal(outs["grid"], outs["persistent"]))
            # alternate grid/persistent launches on one shared counter/workspace state
            g, p = run.specs[(cfg, "grid")], run.specs[(cfg, "persistent")]
            if any(q.role == "ctr" for q in g.params):
                rg = Runner(g)
                rp = Runner(p, state=rg.state)
                ok = True
                for r in (rg, rp, rg, rp):
                    o3 = _launch(r, inputs)[out_name]
                    ok &= bool(torch.equal(o3, outs["grid"])) and all(bool((c == 0).all()) for c in r.counters().values())
                res["shared_state_alternate_ok"] = ok
        if outs:
            first = outs.get("grid", outs.get("persistent"))
            if key not in group_ref:
                group_ref[key] = (op.cfg_tag(cfg), first)
                res["bitwise_eq_group_ref"] = True
            else:
                res["bitwise_eq_group_ref"] = bool(torch.equal(first, group_ref[key][1]))
            res["group_ref"] = group_ref[key][0]
        run.results.append(res)
        del outs
    run.groups = {repr(k): v[0] for k, v in group_ref.items()}
    del inputs, ref, ref_t
    torch.cuda.empty_cache()


def summarize(run: OpRun) -> dict:
    rs = run.results
    n = len(rs)

    def cnt(k):
        return sum(1 for r in rs if r.get(k) is True)

    split = [r for r in rs if "grid_repeat_ok" in r or "persistent_repeat_ok" in r]
    return {
        "op": run.op.NAME,
        "configs": n,
        "kernels": run.compile_stats.get("n_kernels"),
        "compile_ok": run.compile_stats.get("n_ok"),
        "compile_fail": run.compile_stats.get("n_fail"),
        "compile_wall_s": round(run.compile_stats.get("wall_s", 0), 1),
        "tile_space_ok": cnt("tile_space_ok"),
        "grid_ok": cnt("grid_ok"),
        "persistent_ok": cnt("persistent_ok"),
        "both_ok": sum(1 for r in rs if r.get("grid_ok") and r.get("persistent_ok")),
        "grid_eq_persistent": cnt("grid_eq_persistent"),
        "group_bitwise_ok": cnt("bitwise_eq_group_ref"),
        "numerics_groups": len(run.groups),
        "split_configs": len(split),
        "repeat_ok": sum(1 for r in split if r.get("grid_repeat_ok") and r.get("persistent_repeat_ok")),
        "counters_zero": sum(1 for r in split if r.get("grid_counters_zero") and r.get("persistent_counters_zero")),
        "shared_state_ok": cnt("shared_state_alternate_ok"),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def log(*a):
    print(*a, flush=True, file=sys.stdout)
