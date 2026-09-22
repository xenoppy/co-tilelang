"""K5 overhead probe for CoKernels (short GPU runs, cobench flush mode).

    source research/env.sh
    python -m cotile.tests.probe_cokernel_overhead --out research/results/2026-09-22_cokernel

(i)  Compile-time coupling: a CoKernel whose role B is compiled in but gets no tiles
     (SM binding, every SM -> role A, takeover off) vs the solo persistent build of A
     (same CTA count). Variants: static schedule without timestamps (closest to the
     solo grid-stride loop), static with per-tile timestamps, dynamic chunk 1. The
     "A x A" CoKernel (partner = A itself) separates dispatch cost from coupling to a
     different partner (registers, thread count, smem).
(ii) Dispatch overhead for tiny tiles: RMSNorm (16384 x 4096, one row per tile) as the
     only active role of an RMSNorm x RMSNorm CoKernel, static vs dynamic with chunk
     1/4/16, with and without per-tile timestamps, vs the solo persistent RMSNorm.

Every row reports cobench's flush-mode event time (median over 50 reps, L2 cold, host
gate, >= 1 s warmup) plus the clock-normalised cycles, and the compiled resource
signature (regs/thread, smem/CTA, CTAs/SM). The CoKernel's own %globaltimer stamps of
the last rep (makespan_ns) are reported next to the event time as a K3 cross-check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

from cotile import resources
from cotile.cokernel import CoRunner, Orch, build_cokernel, sm_role_table
from cotile.device import DEFAULT_DEVICE
from cotile.kernel import Runner, compile_specs
from cotile.ops import gemm, gqa_decode, rmsnorm
from cotile.tests import harness as H

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "research", "bench"))

NSM = DEFAULT_DEVICE.num_sms
GS = gemm.GemmShape(2048, 4096, 4096)
DS = gqa_decode.DecodeShape(batch=16, seqlen=8192)
RS = rmsnorm.RMSNormShape(16384, 4096)
GC = gemm.GemmConfig(128, 128, 64, 2, 128)
DC = gqa_decode.DecodeConfig(64, 4, 1, 128, 2)
RC = rmsnorm.RMSNormConfig(1, 256, 8)

# (label, (opA, shapeA, cfgA), (opB, shapeB, cfgB), num_ctas)
COUPLING = [
    ("gemm|decode", (gemm, GS, GC), (gqa_decode, DS, DC), NSM),
    ("gemm|gemm", (gemm, GS, GC), (gemm, GS, GC), NSM),
    ("decode|gemm", (gqa_decode, DS, DC), (gemm, GS, GC), NSM),
    ("decode|decode", (gqa_decode, DS, DC), (gqa_decode, DS, DC), NSM),
    ("gemm|rmsnorm", (gemm, GS, GC), (rmsnorm, RS, RC), NSM),
    ("rmsnorm|gemm", (rmsnorm, RS, RC), (gemm, GS, GC), NSM),
    ("rmsnorm|rmsnorm", (rmsnorm, RS, RC), (rmsnorm, RS, RC), NSM),
]
COUPLING_ORCH = [
    ("static_notime", Orch(schedule="static", timing=False)),
    ("static", Orch(schedule="static")),
    ("dynamic_c1", Orch(schedule="dynamic", chunk=1)),
]
# (ii) RMSNorm x RMSNorm with 4 CTAs/SM
DISPATCH_CTAS = 4 * NSM
DISPATCH_ORCH = [("static_notime", Orch(schedule="static", timing=False, num_ctas=DISPATCH_CTAS)),
                 ("static", Orch(schedule="static", num_ctas=DISPATCH_CTAS))]
for c in (1, 4, 16):
    DISPATCH_ORCH.append((f"dynamic_c{c}_notime", Orch(schedule="dynamic", chunk=c, timing=False, num_ctas=DISPATCH_CTAS)))
    DISPATCH_ORCH.append((f"dynamic_c{c}", Orch(schedule="dynamic", chunk=c, num_ctas=DISPATCH_CTAS)))


def build():
    specs = []
    items = []  # (group, label, variant, spec)
    solo = {}
    for label, (oa, sa, ca), (ob, sb, cb), n in COUPLING:
        key = (oa.NAME, repr(ca), n)
        if key not in solo:
            solo[key] = oa.build_persistent(sa, ca, n)
            items.append(("coupling", label.split("|")[0], "solo_persistent", solo[key]))
        for vname, o in COUPLING_ORCH:
            o = replace(o, num_ctas=n)
            items.append(("coupling", label, vname, build_cokernel(oa, sa, ca, ob, sb, cb, o)))
    items.append(("dispatch", "rmsnorm", "solo_persistent", rmsnorm.build_persistent(RS, RC, DISPATCH_CTAS)))
    for vname, o in DISPATCH_ORCH:
        items.append(("dispatch", "rmsnorm|rmsnorm", vname, build_cokernel(rmsnorm, RS, RC, rmsnorm, RS, RC, o)))
    uniq = {}
    for it in items:
        uniq.setdefault(it[3].name, it[3])
    specs = list(uniq.values())
    st = compile_specs(specs)
    for it in items:
        u = uniq[it[3].name]
        it[3].kernel, it[3].compile_error = u.kernel, u.compile_error
    return items, st


def _inputs_for(op, shape, cache):
    if op.NAME not in cache:
        cache[op.NAME] = op.make_inputs(shape, seed=3)
    return cache[op.NAME]


def measure(items, reps=50):
    import torch
    import cobench as cb

    rows = []
    inputs = {}
    shapes = {gemm.NAME: GS, gqa_decode.NAME: DS, rmsnorm.NAME: RS}
    ops = {gemm.NAME: gemm, gqa_decode.NAME: gqa_decode, rmsnorm.NAME: rmsnorm}
    for group, label, vname, spec in items:
        if spec.kernel is None:
            rows.append({"group": group, "label": label, "variant": vname, "error": spec.compile_error})
            continue
        sig = resources.signature(spec)
        row = {
            "group": group,
            "label": label,
            "variant": vname,
            "threads": sig["threads"],
            "regs": sig["regs"],
            "smem_per_cta": sig["smem_total"],
            "ctas_per_sm": sig["ctas_per_sm"],
            "limit_by": sig["limit_by"],
            "grid": sig["grid"],
            "local_bytes": sig["local_bytes"],
        }
        if spec.build == "cokernel":
            info = spec.extra["co"]
            ra, rb = info.roles
            ia = _inputs_for(ra.op, shapes[ra.op.NAME], inputs)
            ib = _inputs_for(rb.op, shapes[rb.op.NAME], inputs)
            runner = CoRunner(spec)
            runner.set_knobs(sm_role=sm_role_table(NSM))  # every SM -> role A, B gets 0 tiles
            args, _ = runner.make_args([ia, ib])
            fn = lambda args=args, k=spec.kernel: k(*args)  # noqa: E731
        else:
            op = ops[spec.op.NAME]
            ia = _inputs_for(op, shapes[op.NAME], inputs)
            runner = Runner(spec)
            args, _ = runner.make_args(ia)
            fn = lambda args=args, k=spec.kernel: k(*args)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        r = cb.bench(fn, mode="flush", reps=reps, clock=True, label=f"{label}:{vname}", strict=True)
        row.update({"median_us": r.median, "p10_us": r.p10, "p90_us": r.p90, "cv": r.cv, "cv_ok": r.cv_ok})
        clk = (r.clock or {}).get("per_rep") or {}
        row["mhz_median"] = clk.get("median")
        row["cycles_median"] = clk.get("cycles_median")
        if spec.build == "cokernel":
            torch.cuda.synchronize()
            s = runner.stats()
            row["done_A"], row["done_B"], row["tiles_A"] = s["done_A"], s["done_B"], s["tiles_A"]
            row["kernel_T_A_us"] = s["T_A_ns"] / 1e3 if s["T_A_ns"] else None
            row["kernel_exit_us"] = s["exit_ns"] / 1e3
        rows.append(row)
        H.log(
            f"  {group:8s} {label:16s} {vname:18s} {r.median:9.1f} us (cv {r.cv:.3f}) regs={row['regs']} "
            f"smem={row['smem_per_cta']} ctas/SM={row['ctas_per_sm']} "
            + (f"T_A(kernel)={row.get('kernel_T_A_us')}" if spec.build == "cokernel" else "")
        )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args(argv)
    t0 = time.time()
    items, st = build()
    H.log(f"compiled {st}")
    if not H.wait_for_gpu():
        raise RuntimeError("GPU busy for 15 min; aborting")
    rows = measure(items, a.reps)
    H.log(f"wall {time.time() - t0:.1f}s")
    if a.out:
        H.write_csv(os.path.join(a.out, "overhead_probe.csv"), rows)
        with open(os.path.join(a.out, "overhead_probe.json"), "w") as f:
            json.dump({"compile": st, "rows": rows}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
