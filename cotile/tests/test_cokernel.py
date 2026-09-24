"""CoKernel builder tests (P1-M3a: K1-K4 correctness, K2 shared-memory aliasing, K6
generated-code observations). Runnable with plain python:

    source research/env.sh
    python -m cotile.tests.test_cokernel                    # full matrix
    python -m cotile.tests.test_cokernel --pairs gd_e0      # one config pair
    python -m cotile.tests.test_cokernel --no-gpu           # compile + signatures only
    python -m cotile.tests.test_cokernel --out research/results/2026-09-22_cokernel

What is checked (per CoKernel x runtime-knob setting, 3 launches each):
  * both outputs match the fp32 torch reference (op.TOLERANCE),
  * both outputs are bitwise identical to the same op/cfg's solo persistent build,
  * debug builds: every tile executed exactly once (per-tile execution counter), and
    with SM binding and takeover off every tile ran on an SM of its own role,
  * co_out reports done_A/done_B == tile counts and the per-role CTA counts the
    static schedule expected,
  * after every launch all orchestration counters (co_state except the epoch, co_tacc)
    and the ops' split counters are zero again (no host re-zeroing between launches);
    outputs and split workspaces are poisoned with NaN before every launch.
K2: per-CTA shared memory of every CoKernel (from the compiled cubin) vs the two roles'
solo persistent builds and vs the smem="sum" build of the same pair.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from cotile import resources
from cotile.cokernel import S_EPOCH, CoRunner, Orch, build_cokernel, sm_role_table
from cotile.device import DEFAULT_DEVICE
from cotile.kernel import Runner, compile_specs
from cotile.ops import gemm, gqa_decode, rmsnorm
from cotile.tests import harness as H

G, D, R = gemm.GemmConfig, gqa_decode.DecodeConfig, rmsnorm.RMSNormConfig

SHAPES = {
    gemm.NAME: gemm.GemmShape(M=2048, N=4096, K=4096),
    gqa_decode.NAME: gqa_decode.DecodeShape(batch=16, seqlen=8192, heads=32, kv_heads=8, dim=128),
    rmsnorm.NAME: rmsnorm.RMSNormShape(tokens=16384, hidden=4096),
}

# name -> (opA, cfgA, opB, cfgB, ctas_per_sm for CTA binding (None: not co-resident))
PAIRS = {
    # equal threads, E0 GEMM, no split
    "gd_e0": (gemm, G(128, 128, 64, 2, 128), gqa_decode, D(64, 4, 1, 128, 2), None),
    # split-K GEMM x split-KV decode (in-kernel last-arriver combines in both roles)
    "gd_split": (gemm, G(128, 128, 64, 2, 128, split_k=2), gqa_decode, D(64, 4, 4, 128, 2), None),
    # 256-thread GEMM, decode runs on 128 of 256 threads (named barriers + AllReduce)
    "gd_thr": (gemm, G(128, 128, 64, 2, 256), gqa_decode, D(64, 4, 2, 128, 2), None),
    # small tiles: 2 CTAs/SM fit -> CTA (POD-style) binding; decode on 64 of 128 threads
    "gd_small": (gemm, G(64, 64, 64, 2, 128), gqa_decode, D(32, 4, 2, 64, 2), 2),
    # GEMM on 128 of 256 threads, RMSNorm 1 row per tile (16384 tiny tiles)
    "gr_e0": (gemm, G(128, 128, 64, 2, 128), rmsnorm, R(1, 256, 8), None),
    # split-K GEMM (256 thr) x RMSNorm on 128 of 256 threads
    "gr_split": (gemm, G(128, 128, 64, 2, 256, split_k=2), rmsnorm, R(4, 128, 8), None),
    # small: 4 CTAs/SM -> CTA binding with ratios 1:1, 1:3, 3:1
    "gr_small": (gemm, G(64, 64, 32, 2, 128), rmsnorm, R(2, 128, 8), 4),
    # L2 eviction-policy axes: GEMM A/B loads evict_last (a config that hits the ptxas
    # desc[UR1] miscompile with the 32-bit shared address), decode K/V loads evict_first
    "gd_l2": (gemm, G(128, 128, 64, 2, 128, ab_l2="evict_last"), gqa_decode, D(64, 4, 1, 64, 1, kv_l2="evict_first"),
              None),
}


def orch_variants(pair: str, debug: bool) -> list[Orch]:
    """Compile-time orchestration variants tested for a pair."""
    opA, cfgA, opB, cfgB, c = PAIRS[pair]
    # per-role chunk: 1 for GEMM / decode tiles, 4 for the tiny RMSNorm tiles
    chunk = tuple(4 if op is rmsnorm else 1 for op in (opA, opB))
    out = []
    for binding in ("sm", "cta"):
        if binding == "cta" and not c:
            continue
        per_sm = c or 1
        for schedule in ("static", "dynamic"):
            for takeover in (False, True):
                out.append(
                    Orch(
                        binding=binding,
                        schedule=schedule,
                        chunk=chunk if schedule == "dynamic" else 1,
                        takeover=takeover,
                        num_ctas=DEFAULT_DEVICE.num_sms * per_sm,
                        min_blocks_per_sm=per_sm,
                        debug=debug,
                    )
                )
    return out


def knob_settings(orch: Orch, c: int | None) -> list[dict]:
    """Runtime knob settings (no recompilation). Without takeover both roles must get
    CTAs; with takeover also degenerate splits where one role has no CTA at all."""
    n = DEFAULT_DEVICE.num_sms
    if orch.binding == "sm":
        ks = [
            {"sm_role": sm_role_table(94), "label": "sm94"},
            {"sm_role": sm_role_table(140, order="interleave"), "label": "sm140i"},
            {"sm_role": sm_role_table(40), "label": "sm40"},
        ]
        if orch.takeover:
            ks += [{"sm_role": sm_role_table(n), "label": "sm188(allA)"}, {"sm_role": sm_role_table(0), "label": "sm0(allB)"}]
        return ks
    ratios = [(1, 1)] if c == 2 else [(1, 1), (1, 3), (3, 1)]
    ks = [{"ratio": r, "label": f"r{r[0]}:{r[1]}"} for r in ratios]
    if orch.takeover:
        ks += [{"ratio": (1, 0), "label": "r1:0(allA)"}, {"ratio": (0, 1), "label": "r0:1(allB)"}]
    return ks


# ----------------------------------------------------------------------------------


def build_all(pairs: list[str], workers: int = 32):
    specs = {}  # key -> spec
    for p in pairs:
        opA, cfgA, opB, cfgB, c = PAIRS[p]
        sA, sB = SHAPES[opA.NAME], SHAPES[opB.NAME]
        specs[(p, "soloA")] = opA.build_persistent(sA, cfgA, DEFAULT_DEVICE.num_sms)
        specs[(p, "soloB")] = opB.build_persistent(sB, cfgB, DEFAULT_DEVICE.num_sms)
        for debug in (True, False):
            for o in orch_variants(p, debug):
                specs[(p, o)] = build_cokernel(opA, sA, cfgA, opB, sB, cfgB, o)
        o_sum = Orch(binding="sm", schedule="dynamic", smem="sum", num_ctas=DEFAULT_DEVICE.num_sms)
        specs[(p, "sum")] = build_cokernel(opA, sA, cfgA, opB, sB, cfgB, o_sum)
    # solo builds of identical (op, cfg) are shared between pairs: dedupe by name
    uniq = {}
    for s in specs.values():
        uniq.setdefault(s.name, s)
    t0 = time.time()
    st = compile_specs(list(uniq.values()), num_workers=workers)
    for k, s in specs.items():
        u = uniq[s.name]
        s.kernel, s.compile_error = u.kernel, u.compile_error
    st["wall_s"] = time.time() - t0
    st["n_kernels"] = len(uniq)
    return specs, st


def code_observations(spec) -> dict:
    """Facts read from the generated CUDA source (K6)."""
    src = spec.kernel.get_kernel_source()
    lb = re.search(r"__launch_bounds__\(([^)]*)\)", src)
    partial = sorted({(int(a), int(b)) for a, b in re.findall(r"__sync_thread_partial\((\d+), (\d+)\)", src)})
    named = sorted({(int(a), int(b)) for a, b in re.findall(r"bar\.sync (\d+), (\d+)", src)})
    allreduce = sorted(set(re.findall(r"NamedBarrier<(\d+)>", src)))
    offs = [int(x) for x in re.findall(r"buf_dyn_shmem \+ (\d+)\)", src)]
    return {
        "launch_bounds": lb.group(1) if lb else None,
        "n_syncthreads": src.count("__syncthreads()"),
        "partial_barriers(id,n)": partial,
        "bar_sync(id,n)": named,
        "allreduce_named_barrier_threads": allreduce,
        "smem_view_offsets": sorted(set(offs)),
        "cp_async_conditional": src.count("cp_async_gs_conditional"),
        "cp_async": src.count("cp_async_gs<") + src.count("cp_async_gs_conditional"),
    }


def signatures(specs: dict) -> list[dict]:
    rows = []
    by_pair: dict = {}
    for (p, kind), s in specs.items():
        if s.kernel is None:
            continue
        sig = resources.signature(s)
        s.extra["signature"] = sig
        by_pair.setdefault(p, {})[kind] = sig
    for (p, kind), s in specs.items():
        if s.kernel is None or kind in ("soloA", "soloB"):
            continue
        sig = s.extra["signature"]
        a, b = by_pair[p]["soloA"], by_pair[p]["soloB"]
        row = {
            "pair": p,
            "orch": kind.tag() if isinstance(kind, Orch) else kind,
            "threads": sig["threads"],
            "regs": sig["regs"],
            "local_bytes": sig["local_bytes"],
            "smem_total": sig["smem_total"],
            "smem_static": sig["smem_static"],
            "smem_dynamic": sig["smem_dynamic"],
            "soloA_smem": a["smem_total"],
            "soloB_smem": b["smem_total"],
            "soloA_regs": a["regs"],
            "soloB_regs": b["regs"],
            "max_solo_smem": max(a["smem_total"], b["smem_total"]),
            "sum_solo_smem": a["smem_total"] + b["smem_total"],
            "num_barriers": sig["num_barriers"],
            "ctas_per_sm": sig["ctas_per_sm"],
            "limit_by": sig["limit_by"],
            # ptxas miscompile guard (cp.async with an L2 cache policy, resources.py)
            "invalid_mem_desc": len(resources.invalid_memory_descriptors(resources.sass(resources.cubin_bytes(s.kernel)))),
        }
        row.update(code_observations(s))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------
# GPU
# ----------------------------------------------------------------------------------


def _poison(outputs: list[dict], runner) -> None:
    for o in outputs:
        for t in o.values():
            t.fill_(float("nan"))
    for p in runner.spec.params:
        if p.role == "ws" and p.name in runner.state:
            runner.state[p.name].fill_(float("nan"))


def _counters_clean(runner) -> tuple[bool, dict]:
    import torch

    bad = {}
    for p in runner.spec.params:
        if p.role != "ctr" or p.name not in runner.state:
            continue
        t = runner.state[p.name]
        if p.name == "co_state":
            z = t.clone()
            z[S_EPOCH] = 0
            if bool((z != 0).any()):
                bad[p.name] = torch.nonzero(z).flatten().tolist()[:8]
        elif p.name == "co_claim":
            continue  # epoch-tagged, never reset by design
        elif bool((t != 0).any()):
            bad[p.name] = torch.nonzero(t).flatten().tolist()[:8]
    return (not bad), bad


def run_gpu(specs: dict, pairs: list[str], repeat: int = 3) -> list[dict]:
    import torch

    rows = []
    for p in pairs:
        opA, cfgA, opB, cfgB, c = PAIRS[p]
        sA, sB = SHAPES[opA.NAME], SHAPES[opB.NAME]
        ia, ib = opA.make_inputs(sA, seed=1), opB.make_inputs(sB, seed=2)
        outA = next(q.name for q in opA.io_params(sA, cfgA) if q.role == "out")
        outB = next(q.name for q in opB.io_params(sB, cfgB) if q.role == "out")
        refA, refB = opA.reference(sA, ia)[outA], opB.reference(sB, ib)[outB]
        soloA = Runner(specs[(p, "soloA")])(ia)[outA]
        soloB = Runner(specs[(p, "soloB")])(ib)[outB]
        torch.cuda.synchronize()
        solo_ok = (H.compare(soloA, refA, opA.TOLERANCE)["ok"], H.compare(soloB, refB, opB.TOLERANCE)["ok"])
        kinds = [k for (pp, k) in specs if pp == p and isinstance(k, Orch)] + ["sum"]
        for kind in kinds:
            spec = specs[(p, kind)]
            orch = spec.extra["co"].orch
            base = {"pair": p, "orch": orch.tag(), "solo_ok": all(solo_ok)}
            if spec.kernel is None:
                rows.append({**base, "ok": False, "error": spec.compile_error})
                continue
            sig = spec.extra.get("signature")
            if sig is not None and sig["smem_total"] > DEFAULT_DEVICE.smem_per_cta_optin:
                # e.g. smem="sum" GEMM x decode: not launchable at all on this GPU
                rows.append({**base, "knobs": "-", "ok": True, "launchable": False,
                             "error": f"not launchable: {sig['smem_total']} B smem/CTA > {DEFAULT_DEVICE.smem_per_cta_optin}"})
                H.log(f"  {p:9s} {orch.tag():34s} not launchable ({sig['smem_total']} B smem/CTA)")
                continue
            runner = CoRunner(spec)
            ks = knob_settings(orch, c) if orch.debug or kind == "sum" else knob_settings(orch, c)[:1]
            for kn in ks:
                runner.set_knobs(sm_role=kn.get("sm_role"), ratio=kn.get("ratio", (1, 1)))
                row = dict(base, knobs=kn["label"])
                agg = {
                    "refA_ok": True, "refB_ok": True, "bitA": True, "bitB": True, "exec_once": True,
                    "role_placement_ok": True, "done_ok": True, "ctr_clean": True, "expected_ctas_ok": True,
                }
                stats = []
                errs = []
                epochs = []
                for rep in range(repeat):
                    _, outs = runner.make_args([ia, ib])
                    _poison(outs, runner)
                    try:
                        outs = runner([ia, ib], outs)
                        torch.cuda.synchronize()
                        s = runner.stats(check=False)
                    except Exception as e:  # noqa: BLE001 - reported as a result
                        errs.append(f"{type(e).__name__}: {str(e).splitlines()[-1][:200]}")
                        agg = {k: False for k in agg}
                        break
                    a, b = outs[0][outA], outs[1][outB]
                    agg["refA_ok"] &= H.compare(a, refA, opA.TOLERANCE)["ok"]
                    agg["refB_ok"] &= H.compare(b, refB, opB.TOLERANCE)["ok"]
                    agg["bitA"] &= bool(torch.equal(a, soloA))
                    agg["bitB"] &= bool(torch.equal(b, soloB))
                    agg["done_ok"] &= s["done_A"] == s["tiles_A"] and s["done_B"] == s["tiles_B"] and s["smid_errors"] == 0
                    if runner.expected is not None:
                        agg["expected_ctas_ok"] &= (s["tickets_A"], s["tickets_B"]) == runner.expected
                    if orch.debug:
                        dbg = runner.dbg["co_dbg"]
                        agg["exec_once"] &= bool((dbg == 1).all())
                        if orch.binding == "sm" and not orch.takeover:
                            table = runner.state["co_sm_role"]
                            sm = runner.dbg["co_dbg_sm"].long()
                            NA = s["tiles_A"]
                            role_of_tile = torch.cat(
                                [torch.zeros(NA, dtype=torch.int32, device=sm.device), torch.ones(s["tiles_B"], dtype=torch.int32, device=sm.device)]
                            )
                            agg["role_placement_ok"] &= bool((sm >= 0).all()) and bool((table[sm.clamp(min=0)] == role_of_tile).all())
                    ok, bad = _counters_clean(runner)
                    agg["ctr_clean"] &= ok
                    if not ok:
                        errs.append(f"dirty counters {bad}")
                    epochs.append(s["epoch"])
                    stats.append(s)
                row.update(agg)
                row["epoch_increments"] = all(e2 == e1 + 1 for e1, e2 in zip(epochs, epochs[1:])) if len(epochs) > 1 else False
                row["launches"] = len(stats)
                if stats:
                    last = stats[-1]
                    for k in ("T_A_ns", "T_B_ns", "makespan_ns", "exit_ns", "ctas_A", "ctas_B", "steal_A", "steal_B", "q_A", "q_B"):
                        row[k] = last[k]
                row["error"] = "; ".join(errs)
                row["ok"] = all(agg.values()) and row["epoch_increments"] and not errs and row["launches"] == repeat
                row["launchable"] = True
                rows.append(row)
                H.log(
                    f"  {p:9s} {orch.tag():34s} {kn['label']:11s} ok={row['ok']} "
                    + " ".join(k for k, v in agg.items() if not v)
                    + (f" T_A={row.get('T_A_ns')} T_B={row.get('T_B_ns')}" if stats else "")
                    + (f" ERR {row['error']}" if errs else "")
                )
        del ia, ib, refA, refB, soloA, soloB
        torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------------


def _run(pairs, gpu=True, workers=32):
    specs, cst = build_all(pairs, workers)
    H.log(f"compiled {cst}")
    for (p, k), s in specs.items():
        if s.compile_error:
            H.log(f"  COMPILE-FAIL {p} {k if isinstance(k, str) else k.tag()}: {s.compile_error}")
    sigs = signatures(specs)
    rows = []
    if gpu:
        if not H.wait_for_gpu():
            raise RuntimeError("GPU occupied past the guard deadline (cobench.GuardPolicy); aborting GPU tests")
        rows = run_gpu(specs, pairs)
    return specs, cst, sigs, rows


def test_cokernel_all():
    specs, cst, sigs, rows = _run(list(PAIRS))
    assert cst["n_fail"] == 0, cst
    assert all(r["invalid_mem_desc"] == 0 for r in sigs), [r for r in sigs if r["invalid_mem_desc"]][:3]
    assert rows and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]][:5]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    pairs = a.pairs.split(",")
    t0 = time.time()
    specs, cst, sigs, rows = _run(pairs, gpu=not a.no_gpu, workers=a.workers)
    for r in sigs:
        H.log(
            f"  SIG {r['pair']:9s} {r['orch']:34s} thr={r['threads']} regs={r['regs']} (solo {r['soloA_regs']}/{r['soloB_regs']}) "
            f"smem={r['smem_total']} (soloA {r['soloA_smem']}, soloB {r['soloB_smem']}, sum {r['sum_solo_smem']}) "
            f"ctas/SM={r['ctas_per_sm']} ({r['limit_by']}) bars={r['num_barriers']} lb={r['launch_bounds']} "
            f"partial={r['partial_barriers(id,n)']} allreduce={r['allreduce_named_barrier_threads']} local={r['local_bytes']}"
        )
    summary = {
        "compile": cst,
        "cokernels_tested": len({(r["pair"], r["orch"]) for r in rows}),
        "knob_runs": len(rows),
        "knob_runs_ok": sum(1 for r in rows if r.get("ok") and r.get("launchable")),
        "not_launchable": sorted({(r["pair"], r["orch"]) for r in rows if r.get("launchable") is False}),
        "failed": [(r["pair"], r["orch"], r.get("knobs"), r.get("error")) for r in rows if not r.get("ok")],
        "wall_s": round(time.time() - t0, 1),
    }
    H.log(json.dumps({k: v for k, v in summary.items() if k != "failed"}))
    for f in summary["failed"][:20]:
        H.log(f"  FAIL {f}")
    if a.out:
        H.write_csv(os.path.join(a.out, "cokernel_tests.csv"), rows)
        H.write_csv(os.path.join(a.out, "cokernel_signatures.csv"), sigs)
        with open(os.path.join(a.out, "cokernel_summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)
    bad_desc = [(r["pair"], r["orch"]) for r in sigs if r["invalid_mem_desc"]]
    if bad_desc:
        H.log(f"  INVALID MEMORY DESCRIPTORS (ptxas miscompile) in {bad_desc[:10]}")
    ok = cst["n_fail"] == 0 and not bad_desc and (a.no_gpu or (rows and all(r.get("ok") for r in rows)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
