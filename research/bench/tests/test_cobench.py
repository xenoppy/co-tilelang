"""Acceptance tests for cobench (P0-2 + methodology v1). Run directly:

    source research/env.sh
    python research/bench/tests/test_cobench.py [name-filter ...] [--out DIR]
    python -m pytest research/bench/tests/test_cobench.py          # also works (no JSON report)

Each test_* function asserts the behaviour listed in the completion criteria and records its
key numbers; a PASS/FAIL table and a JSON with the numbers are written at the end
(default: research/results/2026-09-23_methodology_v1/test_output.json; the P0-2 run is in
research/results/2026-09-22_cobench_validation/).

test_1..7   P0-2 (modes, co-run, NVML, green contexts, %smid, reproducibility, README)
test_8      M1 clean flush: eviction + cleanliness of each flush kind, API
test_9      M2 GPU guard: own/descendant activity ignored, a foreign process detected,
            wait_until_free, cotile harness wait_for_gpu, bench/bench_corun guard records
test_10     M3 bench_steady: serial ~ sum of solos (non-power-capped pair), Par per-op
            times, speedups, host-gap detection, bench_variants
test_11     ClockProbe carveout regression (the probe must not keep an SM from hosting a
            large-smem CTA; root cause of the P1-S "static persistent penalty")
test_12     M3 repeatability across 3 processes (steady mode; reads the study JSON)
test_13     GPU-sharing policy (2026-09-23): occupancy decision and wait loop on mocked
            process lists / pmon activity / clock (no GPU work, no foreign job needed)
test_14..16 stress kernels (cobench.stress, 2026-09-24 limit study): work-completion checks of
            every co-location layout, criteria V (MMA / DRAM), setmaxnreg on sm_120(a)
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, ".."))
RESULTS = os.path.normpath(os.path.join(BENCH, "../results"))
sys.path.insert(0, BENCH)
import cobench as cb  # noqa: E402

REPORT: dict = {}
OUT_DIR = os.path.join(RESULTS, "2026-09-23_methodology_v1")   # main(--out) overrides
INFO = cb.device_info()
NSM = INFO["sms"]
DRAM = INFO["dram_peak_gbps"]
L2 = INFO["l2_bytes"]
MM = 4096
MM_FLOPS = 2 * MM ** 3


def mk_mm():
    return (torch.randn(MM, MM, device="cuda", dtype=torch.bfloat16),
            torch.randn(MM, MM, device="cuda", dtype=torch.bfloat16),
            torch.empty(MM, MM, device="cuda", dtype=torch.bfloat16))


def mm(a, b, c):
    torch.matmul(a, b, out=c)


def mk_copy(nbytes):
    def f():
        s = torch.empty(nbytes // 2, device="cuda", dtype=torch.bfloat16).normal_()
        return s, torch.empty_like(s)
    return f


def other_gpu_procs() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip() and not l.startswith(f"{os.getpid()},")]


# ---------------------------------------------------------------------------
def test_1a_matmul_modes():
    """bench() modes flush/graph/hot on bf16 matmul 4096^3; TFLOP/s vs measured peak."""
    pk = cb.mma_peak(warps_per_cta=8, nacc=8, iters=20000)
    fpc = pk["flop_per_clk_per_sm"]
    assert 950 < fpc < 1100, f"mma.sync FLOP/clk/SM {fpc}"
    rep = {"mma_peak": pk, "peak_tflops_at_2617MHz": fpc * NSM * 2617e6 / 1e12,
           "peak_tflops_at_max_clock": fpc * NSM * INFO["max_sm_clock_mhz_nvml"] * 1e6 / 1e12}
    for mode in ("flush", "graph", "hot"):
        r = cb.bench(mm, make_inputs=mk_mm, mode=mode, flops=MM_FLOPS, nvml=True, clock=True,
                     label=f"matmul4096-{mode}")
        print("   ", r)
        clk = r.clock["per_rep"]
        assert clk["covered"], "clock probe did not cover every rep"
        peak_med = fpc * NSM * clk["median"] * 1e6 / 1e12
        peak_max = fpc * NSM * clk["max"] * 1e6 / 1e12
        assert r.n == 50 and r.l2 == {"flush": "cold-flush-clean", "graph": "cold-rotate", "hot": "hot"}[mode]
        assert 0.5 * peak_med < r.tflops <= 1.0 * peak_max, (r.tflops, peak_med, peak_max)
        if mode == "graph":
            assert r.n_copies * r.bytes_per_copy >= 2 * L2 and r.k % r.n_copies == 0
        rep[mode] = {"median_us": r.median, "p10_us": r.p10, "p90_us": r.p90, "cv": r.cv,
                     "tflops": r.tflops, "clock_mhz_median": clk["median"],
                     "peak_tflops_at_that_clock": peak_med, "efficiency": r.tflops / peak_med,
                     "k": r.k, "n_copies": r.n_copies, "power_w": r.nvml["power_w"]["mean"],
                     "throttle": r.nvml["throttle"]["reasons_seen"]}
    REPORT["1a_matmul"] = rep


def test_1b_copy_modes():
    """Copy kernels: DRAM GB/s plausibility; flush/rotation really make it cold; flush time
    is excluded."""
    rep = {"dram_peak_gbps": DRAM}
    big = 512 << 20
    for name, f in (("torch_copy_", lambda s, d: d.copy_(s)), ("copy_u4", lambda s, d: cb.copy_u4(s, d))):
        for mode in ("flush", "graph"):
            r = cb.bench(f, make_inputs=mk_copy(big), mode=mode, nbytes=2 * big, clock=True,
                         label=f"{name}-512MB-{mode}")
            print("   ", r)
            assert 0.6 * DRAM < r.gbps <= DRAM, r.gbps
            rep[f"{name}_512MB_{mode}"] = {"median_us": r.median, "gbps": r.gbps, "cv": r.cv,
                                           "frac_of_peak": r.gbps / DRAM}
    small = 16 << 20  # 32 MB footprint: fits in L2
    res = {}
    for mode in ("flush", "graph", "hot"):
        r = cb.bench(lambda s, d: cb.copy_u4(s, d), make_inputs=mk_copy(small), mode=mode,
                     nbytes=2 * small, label=f"copy_u4-16MB-{mode}")
        print("   ", r)
        res[mode] = r
        rep[f"copy_u4_16MB_{mode}"] = {"median_us": r.median, "gbps": r.gbps, "cv": r.cv}
    # graph (back-to-back rotation): reads and write-backs both hit DRAM
    assert res["graph"].gbps <= DRAM, "graph mode exceeds DRAM peak"
    # clean flush (methodology v1): the 16 MB source is read from DRAM, but the 16 MB of
    # writes are absorbed by the (clean) L2 and written back after the end event -> check
    # the read side only, and that it is far from L2-hot
    assert small / (res["flush"].median * 1e-6) / 1e9 <= DRAM, "flush-mode reads exceed DRAM peak"
    assert res["flush"].median > 2 * res["hot"].median, "flush mode is not cold"
    assert res["hot"].gbps > 2 * DRAM, "hot mode should run from L2"
    # the clean flush takes ~136 us; a 16 MB copy ~14-22 us -> a leaked flush would show
    assert res["flush"].median < 2 * res["graph"].median, "flush time leaked into timing"
    REPORT["1b_copy"] = rep


def test_1c_api_contract():
    try:
        cb.bench(mm, mode="graph")
        raise AssertionError("graph mode without make_inputs must raise")
    except ValueError:
        pass
    try:
        cb.bench(lambda: None, warmup=5, reps=10)
        raise AssertionError("strict protocol must reject warmup<20/reps<50")
    except ValueError:
        pass
    r = cb.bench(mm, make_inputs=mk_mm, mode="hot", flops=MM_FLOPS, warmup_s=0.2)
    json.dumps(r.to_dict())
    REPORT["1c_api"] = "ok"


def test_2_corun():
    """bench_corun: common start, per-op completion, makespan, solo/serial references."""
    a, b, c = mk_mm()
    nb = 256 << 20
    s, d = mk_copy(nb)()
    fa = lambda: torch.matmul(a, b, out=c)  # noqa: E731
    fb = lambda: cb.copy_u4(s, d)  # noqa: E731
    rep = {}
    r = cb.bench_corun(fa, fb, nvml=True, clock=True, label="matmul || copy256MB (streams)")
    print("   ", r)
    for o, a_end, b_end, mk in r.samples["corun"]:
        assert mk == max(a_end, b_end)
    co, sa, sb = r.corun, r.solo_a, r.solo_b
    assert co["a_end"]["median"] > 0.9 * sa["median"] and co["b_end"]["median"] > 0.9 * sb["median"]
    assert co["makespan"]["median"] >= 0.95 * max(sa["median"], sb["median"])
    assert abs(r.serial["total"]["median"] / (sa["median"] + sb["median"]) - 1) < 0.15
    assert set(co["by_order"]) == {"ab", "ba"} and r.config["flush"]
    rep["streams"] = {**r.derived, "makespan_us": co["makespan"]["median"],
                      "makespan_cv": co["makespan"]["cv"], "a_end_us": co["a_end"]["median"],
                      "b_end_us": co["b_end"]["median"], "solo_a_us": sa["median"],
                      "solo_b_us": sb["median"], "serial_us": r.serial["total"]["median"],
                      "by_order": co["by_order"],
                      "clock_mhz": {k: v.get("median") for k, v in r.clock["per_variant"].items()}}
    with cb.split_sms(94, ignore_coscheduling=True) as p:
        r2 = cb.bench_corun(fa, fb, stream_a=p.stream, stream_b=p.rest_stream, clock=True,
                            label=f"matmul@{p.n_sms}SM || copy@{p.n_rest}SM (green)")
        print("   ", r2)
        assert r2.solo_a["median"] > 1.3 * sa["median"], "green partition did not restrict matmul"
        rep["green_94_94"] = {**r2.derived, "makespan_us": r2.corun["makespan"]["median"],
                              "solo_a_us": r2.solo_a["median"], "solo_b_us": r2.solo_b["median"],
                              "serial_us": r2.serial["total"]["median"]}
    REPORT["2_corun"] = rep


def test_3_nvml_sampler():
    a, b, c = mk_mm()
    with cb.NvmlSampler(interval_ms=10) as smp:
        t = time.perf_counter()
        ev = torch.cuda.Event()
        while time.perf_counter() - t < 1.5:
            for _ in range(50):
                mm(a, b, c)
            ev.record()
            ev.synchronize()
    sm = smp.summary()
    print("    NVML summary:", {k: sm[k] for k in ("n_samples", "interval_ms_median", "refresh_ms_observed",
                                                    "sm_clock_mhz", "power_w", "temp_c", "throttle")})
    assert 0.7 * 150 < sm["n_samples"] < 1.3 * 150
    for key in ("sm_clock_mhz", "power_w", "temp_c", "throttle"):
        assert key in sm
    assert sm["power_w"]["max"] > 200, "no load seen in power samples"
    r = cb.bench(mm, make_inputs=mk_mm, mode="hot", nvml=True, warmup_s=0.2)
    assert r.nvml is not None and "throttle" in r.nvml
    REPORT["3_nvml"] = {k: sm[k] for k in ("n_samples", "interval_ms_median", "refresh_ms_observed",
                                           "sm_clock_mhz", "power_w", "temp_c", "throttle")}


def test_4_green_contexts():
    rep = {}
    a, b, c = mk_mm()
    for flag in (False, True):
        for n in (8, 16, 32, 64, 94, 128, 180):
            with cb.split_sms(n, ignore_coscheduling=flag) as p:
                assert p.n_sms >= n and p.n_sms + p.n_rest == NSM
                assert p.part.sm_count_from_driver() == p.n_sms
                if not flag:
                    assert p.n_sms % 8 == 0 or p.n_sms == NSM
                with torch.cuda.stream(p.stream):   # torch op on the green stream
                    mm(a, b, c)
                p.stream.synchronize()
                ra = cb.build_sm_remap(p.stream, expected=p.n_sms)   # raises if not restricted
                rb = cb.build_sm_remap(p.rest_stream, expected=p.n_rest) if p.rest else None
                if rb is not None:
                    assert not set(ra.smids.tolist()) & set(rb.smids.tolist())
                rep[f"{'ignoreCosched' if flag else 'default'}_{n}"] = {"got": p.n_sms, "rest": p.n_rest}
    print("    green (requested -> got/rest):", rep)
    REPORT["4_green"] = rep


def test_5_smid():
    remap = cb.build_sm_remap()
    assert remap.n == NSM and remap.holes == [] and remap.nsmid == NSM
    t = remap.to_tensor()
    assert t.dtype == torch.int32 and t.numel() == remap.table.size
    r = cb.probe_ctas(NSM, 128, smem=cb.one_cta_per_sm_smem(), spin_ns=50_000)
    assert len(set(r["smid"].tolist())) == NSM and r["occupancy"] == 1
    res = cb.globaltimer_resolution()
    sk = cb.globaltimer_skew()
    sk.pop("offsets_by_smid")
    print("    globaltimer:", res["delta_ns"], "decreases", res["decreases_total"], "| skew", sk["offset_vs_cta0_ns"])
    assert res["delta_ns"]["min"] <= 1000 and res["decreases_total"] == 0
    assert sk["negative_one_way_samples"] == 0 and sk["offset_vs_cta0_ns"]["abs_max"] <= 1000
    art = os.path.join(RESULTS, "2026-09-22_smid_probe")
    for fn in ("summary.json", "README.md"):
        assert os.path.exists(os.path.join(art, fn)), f"missing {fn} (run scripts/smid_probe.py)"
    REPORT["5_smid"] = {"nsmid": remap.nsmid, "n": remap.n, "holes": remap.holes,
                        "globaltimer_delta_ns": res["delta_ns"], "skew": sk}


def test_6_reproducibility():
    out = os.path.join(OUT_DIR, "repro_matmul_graph.json")
    p = subprocess.run([sys.executable, os.path.join(BENCH, "scripts", "repro_matmul.py"),
                        "--runs", "5", "--out", out], capture_output=True, text=True)
    print("\n".join("    " + l for l in p.stdout.splitlines() if l.startswith("run ")))
    assert p.returncode == 0, p.stderr[-2000:]
    s = json.load(open(out))
    print(f"    CV across 5 processes: {s['cv_across_runs']*100:.2f}%  (clock CV "
          f"{s['cv_clock_across_runs']*100:.2f}%, cycles/call CV {s['cv_cycles_per_call_across_runs']*100:.3f}%)")
    assert s["cv_across_runs"] < 0.02
    REPORT["6_repro"] = {k: v for k, v in s.items() if k != "per_run"}


def test_8_flush_kinds():
    """M1: "clean" flush evicts (a buffer read before the flush reads like a never-touched
    one) and leaves no dirty lines (a 64 MB write-only kernel is absorbed); "write" leaves
    dirty lines; flush kinds are selectable and labelled."""
    MB = 1 << 20
    assert cb.DEFAULT_FLUSH == "clean" and set(cb.FLUSH_KINDS) >= {"clean", "write", "write+read", "read"}
    try:
        cb.bench(lambda: None, flush_kind="bogus", strict=False)
        raise AssertionError("unknown flush kind must raise")
    except ValueError:
        pass
    X = torch.empty(32 * MB // 4, dtype=torch.int32, device="cuda").fill_(3)
    rep = {}
    for fk in ("clean", "write", "write+read"):
        r = cb.bench(lambda: cb.read_u4(X), mode="flush", flush_kind=fk, nbytes=32 * MB, label=f"read32MB-{fk}")
        assert r.l2 == f"cold-flush-{fk}" and r.flush_kind == fk
        rep[f"read32MB_{fk}"] = r.median
    # never-touched reference: same flush-mode structure (same launch/event floor), but every
    # rep reads a different buffer of a rotation > 2x L2
    bufs = [torch.empty(32 * MB // 4, dtype=torch.int32, device="cuda").fill_(3)
            for _ in range(2 * L2 // (32 * MB) + 2)]
    k = [0]

    def rot():
        k[0] += 1
        cb.read_u4(bufs[k[0] % len(bufs)])
    cold = cb.bench(rot, mode="flush", flush_kind="clean", nbytes=32 * MB, label="read32MB-rotation")
    rep["read32MB_rotation_cold"] = cold.median
    del bufs
    Y = torch.empty(64 * MB // 4, dtype=torch.int32, device="cuda")
    for fk in ("clean", "write", "write+read"):
        rep[f"write64MB_{fk}"] = cb.bench(lambda: Y.fill_(7), mode="flush", flush_kind=fk, label=f"write64MB-{fk}").median
    rep["write64MB_hot"] = cb.bench(lambda: Y.fill_(7), mode="hot", label="write64MB-hot").median
    print("    ", {k: round(v, 2) for k, v in rep.items()})
    # evicted: the clean flush reads as fast as a never-touched rotation (and not faster)
    assert abs(rep["read32MB_clean"] / rep["read32MB_rotation_cold"] - 1) < 0.04, rep
    assert rep["read32MB_clean"] > 2.5 * cb.bench(lambda: cb.read_u4(X), mode="hot").median
    # the write flush makes the same read pay write-backs of dirty lines
    assert rep["read32MB_write"] > 1.03 * rep["read32MB_clean"], rep
    # clean: a 64 MB write is absorbed (close to hot); dirty: DRAM write-back bound
    assert rep["write64MB_clean"] < 1.3 * rep["write64MB_hot"], rep
    assert rep["write64MB_write"] > 1.8 * rep["write64MB_clean"], rep
    REPORT["8_flush_kinds"] = rep


_FOREIGN = r"""
import os, sys, time
if os.fork() > 0:
    os._exit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)
import torch
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
t = time.time()
while time.time() - t < float(sys.argv[1]):
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
"""


def test_9_guard():
    """M2: pmon guard ignores this process (and its children), flags a foreign process with
    SM% > 0, wait_until_free returns once it stops; harness.wait_for_gpu uses it; bench and
    bench_corun record a clean guard window."""
    g = cb.get_guard()
    cb.wait_until_free(max_wait_s=1800)
    rep = {}
    # own activity is not foreign
    a, b, c = mk_mm()
    t0 = time.time()
    while time.time() - t0 < 3.0:
        for _ in range(20):
            mm(a, b, c)
        torch.cuda.synchronize()
    own = g.check(t0, time.time())
    assert own["clean"] and own["complete"], own
    # a child process (descendant) is not foreign either
    t0 = time.time()
    subprocess.run([sys.executable, "-c", _FOREIGN.replace("os.fork() > 0", "False"), "3"], check=True)
    child = g.check(t0, time.time())
    assert child["clean"], child
    # a detached (double-forked, not a descendant) GPU process is foreign
    subprocess.run([sys.executable, "-c", _FOREIGN, "8"], check=True)
    time.sleep(6.0)
    t_mid = time.time()
    busy = g.check(t_mid - 4.0, t_mid)
    assert not busy["clean"] and busy["active"], busy
    # the detached process belongs to this user: the default policy (same_user_is_own) treats it
    # as ours, so the tree-only (legacy) policy is used to exercise the pmon wait here
    waited = cb.wait_until_free(policy=cb.GuardPolicy.legacy(poll_s=2.0, max_wait_s=120),
                                log=lambda m: print("    " + m))
    assert waited > 0.5, waited
    rep.update({"own_clean": own["clean"], "child_clean": child["clean"], "foreign_detected": busy["active"],
                "foreign_max_sm": busy["max_sm"], "waited_s": waited})
    # cotile harness uses the guard (signature and bool result unchanged)
    root = os.path.normpath(os.path.join(BENCH, "..", ".."))
    sys.path.insert(0, root)
    from cotile.tests import harness as H
    import inspect
    sig = inspect.signature(H.wait_for_gpu)
    # defaults None = the guard policy's (2 h deadline, 30 min between checks)
    assert list(sig.parameters) == ["max_wait_s", "poll_s"] and sig.parameters["max_wait_s"].default is None
    assert H.wait_for_gpu(max_wait_s=60, poll_s=5) is True
    assert "guard" in inspect.getsource(H.wait_for_gpu)
    # bench / bench_corun record the guard window
    r = cb.bench(mm, make_inputs=mk_mm, mode="flush", label="guarded")
    assert r.guard and r.guard["clean"] and r.guard["complete"], r.guard
    s, d = mk_copy(64 << 20)()
    rc = cb.bench_corun(lambda: mm(a, b, c), lambda: cb.copy_u4(s, d), label="guarded corun")
    assert rc.guard and rc.guard["clean"], rc.guard
    rep["bench_guard"] = r.guard
    REPORT["9_guard"] = rep


def test_10_steady():
    """M3: bench_steady mechanics on a non-power-capped pair (copy 128 MB -> 128 MB, read
    256 MB): serial ~ solo_a + solo_b, Par per-op completion, speedups, no host gaps; a
    host-starved variant raises HostGapError; bench_variants (flush-mode counterpart)."""
    MB = 1 << 20

    def mk_cp():
        x = torch.empty(64 * MB, device="cuda", dtype=torch.bfloat16).normal_()
        return x, torch.empty_like(x)
    ra = cb.Rotation(mk_cp)
    rb = cb.Rotation(lambda: (torch.empty(64 * MB, device="cuda", dtype=torch.int32).fill_(1),))
    assert ra.total_bytes >= 2 * L2 and rb.total_bytes >= 2 * L2
    fa = lambda i: cb.copy_u4(*ra[i])  # noqa: E731
    fb = lambda i: cb.read_u4(*rb[i])  # noqa: E731
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    def serial(i):
        fa(i)
        fb(i)
    V = {"serial": serial, "solo_a": fa, "solo_b": fb, "streams": cb.Par(("a", s1, fa), ("b", s2, fb))}
    r = cb.bench_steady(V, slice_s=1.0, settle_s=0.3, rounds=3, warmup_s=1.5, thermal=False, label="copy+read")
    print("    " + str(r).replace("\n", "\n    "))
    v = r.variants
    ssum = v["solo_a"]["t_iter_us"] + v["solo_b"]["t_iter_us"]
    ratio = v["serial"]["t_iter_us"] / ssum
    assert abs(ratio - 1) < 0.03, (ratio, v)
    assert all(x["gaps"] == 0 for x in v.values())
    assert set(v["streams"]["ops"]) == {"a", "b"}
    assert max(o["median_us"] for o in v["streams"]["ops"].values()) <= 1.05 * v["streams"]["t_iter_us"]
    assert r.derived["speedup"]["serial"]["ratio_of_medians"] == 1.0
    assert all(x["clock_mhz"] and x["power_w"] for x in v.values())
    assert r.guard and r.guard["clean"]
    # a variant the host cannot keep ahead of must be rejected
    tiny = lambda: cb.read_u4(rb[0][0][:1024])  # noqa: E731

    def starved():
        time.sleep(0.0005)
        tiny()
    try:
        cb.bench_steady({"serial": starved}, slice_s=0.5, settle_s=0.1, rounds=1, warmup_s=0.2, thermal=False,
                        guard=False)
        raise AssertionError("host-starved variant must raise HostGapError")
    except cb.HostGapError:
        pass
    # flush-mode counterpart
    rv = cb.bench_variants({"serial": lambda: (fa(0), fb(0)), "streams": cb.Par(("a", s1, fa), ("b", s2, fb))},
                           reference="serial", label="copy+read flush")
    print("    " + str(rv).replace("\n", "\n    "))
    assert set(rv.variants["streams"]["ops"]) == {"a", "b"}
    REPORT["10_steady"] = {"serial_over_sum_solo": ratio, "t_iter_us": {k: x["t_iter_us"] for k, x in v.items()},
                           "cv_slices": {k: x["cv_slices"] for k, x in v.items()},
                           "clock_mhz": {k: x["clock_mhz"] for k, x in v.items()},
                           "power_w": {k: x["power_w"] for k, x in v.items()},
                           "streams_ops": v["streams"]["ops"],
                           "flush_mode": {k: x["total"]["median"] for k, x in rv.variants.items()}}


def test_11_probe_carveout():
    """While the ClockProbe runs, a grid of 188 one-CTA-per-SM CTAs with ~96 KB smem must use
    all 188 SMs (before the fix the probe's SM kept a small smem carveout and hosted none)."""
    from cobench.clock import ClockProbe
    smem = 96 * 1024
    pk = ClockProbe()
    assert pk.k.carveout == 100
    pk.start()
    try:
        res = cb.probe_ctas(NSM, 128, smem=smem, spin_ns=100_000)
    finally:
        pk.stop()
    used = set(res["smid"].tolist())
    print(f"     probe on SM {pk.smid}; distinct SMs used by the 96 KB-smem grid: {len(used)}")
    assert res["occupancy"] == 1
    assert len(used) == NSM and pk.smid in used, (len(used), pk.smid)
    REPORT["11_probe_carveout"] = {"probe_smid": pk.smid, "distinct_sms": len(used)}


def test_12_steady_repro():
    """M3 (b): steady-mode repeatability across 3 processes (run by
    research/bench/scripts/mv1_steady.py repro): CV of per-variant times and speedups < 2%."""
    # produced by the study script in the canonical results directory (independent of --out)
    path = os.path.join(RESULTS, "2026-09-23_methodology_v1", "steady_repro.json")
    assert os.path.exists(path), "run research/bench/scripts/mv1_steady.py repro first"
    d = json.load(open(path))
    cvs = d["cv_across_processes"]
    print("    ", cvs)
    assert d["processes"] >= 3
    assert all(v < 0.02 for v in cvs["t_iter"].values()), cvs
    assert all(v < 0.02 for v in cvs["speedup"].values()), cvs
    REPORT["12_steady_repro"] = cvs


def test_13_guard_policy():
    """GPU-sharing policy (research/rules.md 7 as clarified on 2026-09-23): occupied = foreign
    SM activity (another user's process with SM% > 0), not a foreign process that only holds
    memory; optional free-memory floor; GuardPolicy.strict() = any foreign compute process;
    30 min between checks; GpuBusy after 2 h; GpuYield when the policy yields; the legacy
    (pmon, tree-only) rule stays available. Mocked process lists, pmon activity and clock."""
    P = cb.GuardPolicy
    dflt, strict, legacy = P(), P.strict(), P.legacy()
    assert (dflt.poll_s, dflt.max_wait_s, dflt.foreign_process_occupies, dflt.foreign_sm_occupies,
            dflt.same_user_is_own, dflt.min_free_mib) == (1800.0, 7200.0, False, True, True, 0)
    assert strict.foreign_process_occupies and strict.poll_s == 1800.0
    assert (legacy.poll_s, legacy.foreign_process_occupies, legacy.foreign_sm_occupies) == (150.0, False, True)
    me, child, other_same_user, qzr, hidden = 100, 101, 200, 300, 400
    users = {me: "ywc", child: "ywc", other_same_user: "ywc", qzr: "qzr", hidden: None}
    own = {me, child}.__contains__
    CP = cb.ComputeProc

    def dec(procs, active, pol, free=None):
        return cb.occupancy([CP(p, users[p], 1000, f"proc{p}") for p in procs], [(p, f"proc{p}") for p in active],
                            is_own=own, own_user="ywc", policy=pol, user_of=users.get, free=free)
    rep = {}
    # our own process tree and this user's other processes: free under every rule but legacy's tree-only view
    d = dec([me, child, other_same_user], [child, other_same_user], dflt)
    assert not d["occupied"], d
    assert not dec([me, child, other_same_user], [child, other_same_user], strict)["occupied"]
    # a foreign (other user) process that only holds memory: NOT occupied by default, occupied when strict
    d = dec([me, qzr], [], dflt)
    assert not d["occupied"] and d["reasons"] == [], d
    d = dec([me, qzr], [], strict)
    assert d["occupied"] and d["reasons"] == ["foreign compute process"] and d["foreign_procs"][0][:2] == (qzr, "qzr"), d
    assert not dec([me, qzr], [], legacy)["occupied"]
    rep["idle_foreign_default"] = dec([me, qzr], [], dflt)
    # foreign SM activity: occupied under every rule
    d = dec([me, qzr], [qzr], dflt)
    assert d["occupied"] and d["reasons"] == ["foreign SM activity"], d
    assert set(dec([me, qzr], [qzr], strict)["reasons"]) == {"foreign compute process", "foreign SM activity"}
    assert dec([qzr], [qzr], legacy)["occupied"]
    # a process whose owner is not visible (other PID namespace) counts as foreign
    assert dec([hidden], [hidden], dflt)["occupied"] and dec([hidden], [], strict)["occupied"]
    assert not dec([hidden], [], dflt)["occupied"]
    # the legacy rule judges this user's other processes by the process tree only
    assert dec([], [other_same_user], legacy)["occupied"] and not dec([], [other_same_user], dflt)["occupied"]
    # free-memory floor: a foreign job that only holds memory blocks us once too little is free
    lowmem = P(min_free_mib=8192)
    d = dec([me, qzr], [], lowmem, free=4096)
    assert d["occupied"] and d["reasons"] == ["only 4096 MiB free < 8192 MiB"] and d["free_mib"] == 4096, d
    assert not dec([me, qzr], [], lowmem, free=50000)["occupied"]
    assert not dec([me, qzr], [], dflt, free=10)["occupied"]          # floor off by default
    # policy knobs
    assert not dec([], [qzr], P(foreign_sm_occupies=False))["occupied"]
    busy = dec([qzr], [qzr], dflt)
    free = dec([me, qzr], [], dflt)

    # wait loop on a fake clock: occupied for 3 checks -> 3 x 30 min, then free
    class Clock:
        def __init__(self):
            self.t, self.sleeps = 0.0, []

        def now(self):
            return self.t

        def sleep(self, dt):
            self.sleeps.append(dt)
            self.t += dt
    seq = iter([busy, busy, busy, free])
    c = Clock()
    waited = cb.wait_loop(lambda: next(seq), dflt, sleep=c.sleep, now=c.now, log=lambda m: None)
    assert c.sleeps == [1800.0] * 3 and waited == 5400.0, (c.sleeps, waited)
    # blocked for good: checks at 0, 30, 60, 90 min; GpuBusy at the 2 h check (no further wait)
    c = Clock()
    try:
        cb.wait_loop(lambda: busy, dflt, sleep=c.sleep, now=c.now, log=lambda m: None)
        raise AssertionError("expected GpuBusy")
    except cb.GpuBusy as e:
        assert c.sleeps == [1800.0] * 4 and c.t == 7200.0, (c.sleeps, c.t)
        rep["busy_msg"] = str(e)[:120]
    # yield_to_caller: GpuYield right away, carrying the decision; no sleep
    c = Clock()
    try:
        cb.wait_loop(lambda: busy, P(yield_to_caller=True), sleep=c.sleep, now=c.now, log=lambda m: None)
        raise AssertionError("expected GpuYield")
    except cb.GpuYield as e:
        assert c.sleeps == [] and e.decision["occupied"], e.decision
    assert not issubclass(cb.GpuYield, RuntimeError)       # generic retry handlers must not swallow it
    # free right away: no wait at all
    c = Clock()
    assert cb.wait_loop(lambda: free, dflt, sleep=c.sleep, now=c.now) == 0.0 and c.sleeps == []
    # legacy timing: 150 s between checks; max_wait None waits until free
    seq = iter([busy, free])
    c = Clock()
    cb.wait_loop(lambda: next(seq), P.legacy(), sleep=c.sleep, now=c.now, log=lambda m: None)
    assert c.sleeps == [150.0], c.sleeps
    seq = iter([busy] * 10 + [free])
    c = Clock()
    cb.wait_loop(lambda: next(seq), P(max_wait_s=None), sleep=c.sleep, now=c.now, log=lambda m: None)
    assert c.sleeps == [1800.0] * 10
    # process-wide policy: set_policy changes single fields and returns the previous policy
    prev = cb.set_policy(yield_to_caller=True)
    try:
        assert cb.get_policy().yield_to_caller and cb.get_policy().poll_s == prev.poll_s
    finally:
        cb.set_policy(prev)
    assert cb.get_policy() == prev
    # the live process list parses (whatever is on the GPU right now)
    live = cb.list_compute_procs()
    rep["live_compute_procs"] = [(p.pid, p.user, p.used_mib) for p in live]
    REPORT["13_guard_policy"] = rep


def _stress_quick(fn, secs: float = 0.6):
    """Back-to-back timing (events) with the ClockProbe: (us per call, MHz)."""
    from cobench.clock import ClockProbe
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        s.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        n = max(5, int(secs * 1e6 / (e0.elapsed_time(e1) * 1e3)))
        pr = ClockProbe(period_us=100, capacity=100_000, timeout_s=60).start()
        for _ in range(n // 4):
            fn()
        e0.record()
        for _ in range(n):
            fn()
        e1.record()
        e1.synchronize()
        pr.stop()
    return e0.elapsed_time(e1) * 1e3 / n, pr.mhz_between(e0, e1)


def test_14_stress_work_completion():
    """Stress kernels (cobench.stress, plan §1.4 v0 / limit study): every co-location layout --
    serial, green partition (full device and nested in a 94-SM sub-device), SM-level roles with
    takeover, CTA-level co-residence, warp-specialised (incl. setmaxnreg), same-warp
    interleaving -- does every A (MMA) and B (stream) unit exactly once per launch, with checksums
    equal to the reference, for both MMA families; the checker catches skipped units; GEMM-like
    tiles equal torch; cp.async / bulk / read+write streams are correct."""
    from cobench import stress as S
    remap = cb.build_sm_remap().to_tensor()
    rep = {}

    def chk(v, k=2, **kw):
        r = v.verify(k, **kw)
        rep[f"{v.wl.family}:{v.name}"] = r["ok"]
        assert r["ok"], (v.name, r["errors"])

    stream = S.KCfg(ak="none", mode="solo_b", su=4, maxreg=48)
    # ---- register-only MMA family
    wl = S.Workload("reg", stream_bytes=64 << 20, a_iters=64)
    ra = S.KCfg(ak="reg", maxreg=64)
    S.reference_sum_a(wl, ra)
    chk(S.v_serial(wl, S.solo_a(wl, ra), S.solo_b(wl, stream)))
    with cb.split_sms(96, ignore_coscheduling=True) as part:
        chk(S.v_green(wl, part, ra, stream))
    chk(S.v_sm(wl, ra, 120, remap))
    chk(S.v_cta(wl, S.KCfg(ak="reg", warps=4, maxreg=64), 2, 2))
    chk(S.v_ws(wl, ra, 8, 4))
    chk(S.v_sw(wl, S.KCfg(ak="reg", maxreg=96)))
    # nested split of the 94-SM sub-device [0, 94): A on [0, 48), B on [48, 94), disjoint
    inner = S.split_nested(94, 48)
    try:
        assert (inner.n_sms, inner.n_rest) == (48, 46), (inner.n_sms, inner.n_rest)
        sa, sb = (set(cb.build_sm_remap(st, expected=n).smids) for st, n in ((inner.stream, 48), (inner.rest_stream, 46)))
        assert sa == set(range(48)) and sb == set(range(48, 94)), (sorted(sa)[:4], sorted(sb)[:4])
        chk(S.v_green(wl, inner, ra, stream, name="green_nested_48"))
    finally:
        inner.close()
    # the checker catches skipped work: start the B tickets at n_b/2 -> half the chunks never done
    L = S.solo_b(wl, stream)
    bad = S.Variant("skip_half", wl, [L], lambda i=0: (L.ctl[1].fill_(wl.n_b // 2), L.launch()), {})
    r = bad.verify(1)
    assert not r["ok"] and r["bad_units_b"] == wl.n_b // 2, r
    rep["negative_control_bad_units"] = r["bad_units_b"]
    # stream kinds
    for name, cfg in (("cpasync", S.KCfg(ak="none", mode="solo_b", skind=1, su=8)),
                      ("bulk", S.KCfg(ak="none", mode="solo_b", skind=2, su=2, sbulk=8192, warps=2))):
        chk(S.v_single(f"stream_{name}", S.solo_b(wl, cfg), {}))
    wrw = S.Workload("reg", stream_bytes=64 << 20, a_iters=16, rw=True)
    chk(S.v_single("stream_rw", S.solo_b(wrw, S.KCfg(ak="none", mode="solo_b", su=4, srw=1, maxreg=64)), {}))
    assert torch.equal(wrw.dbuf, wrw.sbuf)
    # ---- GEMM-like family
    wg = S.Workload("gemm", stream_bytes=64 << 20, n_tiles=256)
    g4 = S.KCfg(ak="gemm", wm=2, wn=2, nt=8, warps=4, stages=3, maxreg=232)
    S.reference_sum_a(wg, g4)
    chk(S.v_serial(wg, S.solo_a(wg, g4, ctas_per_sm=2), S.solo_b(wg, stream)))
    chk(S.v_sm(wg, g4, 140, remap, ctas_per_sm=2))
    chk(S.v_cta(wg, S.KCfg(ak="gemm", wm=2, wn=2, nt=4, warps=4, stages=2, maxreg=128), 2, 1))  # 128x64 tiles (3 CTAs/SM fit)
    chk(S.v_ws(wg, S.KCfg(ak="gemm", stages=4, maxreg=128), 8, 4))
    chk(S.v_ws(wg, S.KCfg(ak="gemm", wm=2, wn=2, nt=8, stages=3, nga=2, smaxnreg=1, ra=232, rb=40, su=2), 8, 4))
    chk(S.v_sw(wg, S.KCfg(ak="gemm", stages=3, spk=1, maxreg=168)))
    # numerics: C of every tile vs torch (fp32 accumulation)
    for cfg in (S.KCfg(ak="gemm", stages=4, maxreg=128, cdbg=1), dataclasses.replace(g4, cdbg=1)):
        wd = S.Workload("gemm", stream_bytes=64 << 20, n_tiles=64, rw=True)
        Ld = S.solo_a(wd, cfg)
        Ld.reset()
        Ld.launch()
        torch.cuda.synchronize()
        C = wd.dbuf.view(torch.float32)[: 1024 * 1024].view(1024, 1024)
        ref = wd.aop.float() @ wd.bop.float().T
        rel = ((C - ref).abs().max() / ref.abs().max()).item()
        rep[f"gemm_rel_err_{cfg.warps}w"] = rel
        assert rel < 1e-4, rel
    print("    ", rep)
    REPORT["14_stress_work_completion"] = rep


def test_15_stress_validation():
    """Stress kernels, criteria V (quick, back-to-back): register-only MMA >= 85% of the measured
    1024 FLOP/clk/SM at its measured clock; the study's ldg stream >= 85% of the best measured
    DRAM read bandwidth (ldg / cp.async / bulk / cobench read_u4). The steady-state numbers are in
    research/results/2026-09-24_limit_study (stage V)."""
    from cobench import stress as S
    wl = S.Workload("reg", stream_bytes=512 << 20, a_iters=256)
    La = S.solo_a(wl, S.KCfg(ak="reg", maxreg=64))
    t, mhz = _stress_quick(La.launch)
    fpc = wl.flops_a / (t * 1e-6) / (mhz * 1e6) / NSM
    rep = {"mma_reg_us": t, "mma_reg_mhz": mhz, "mma_reg_flop_per_clk_per_sm": fpc}
    gb = {}
    for name, cfg, cps in (("ldg", S.KCfg(ak="none", mode="solo_b", su=4, maxreg=48), 2),
                           ("cpasync", S.KCfg(ak="none", mode="solo_b", skind=1, su=8), 2),
                           ("bulk", S.KCfg(ak="none", mode="solo_b", skind=2, su=2, sbulk=8192, warps=2), 2)):
        L = S.solo_b(wl, cfg, ctas_per_sm=cps)
        gb[name] = wl.bytes_b / _stress_quick(L.launch)[0] / 1e3
    gb["read_u4"] = wl.bytes_b / _stress_quick(lambda: cb.read_u4(wl.sbuf))[0] / 1e3
    rep["stream_gbps"] = gb
    rep["ldg_frac_of_best"] = gb["ldg"] / max(gb.values())
    print(f"     MMA {fpc:.0f} FLOP/clk/SM ({fpc / 1024 * 100:.1f}%) at {mhz:.0f} MHz; stream {gb} GB/s")
    assert fpc >= 0.85 * 1024, fpc
    assert rep["ldg_frac_of_best"] >= 0.85, rep
    assert max(gb.values()) >= 0.85 * DRAM, gb
    REPORT["15_stress_validation"] = rep


def test_16_setmaxnreg():
    """setmaxnreg on sm_120: rejected by ptxas for sm_120, accepted for sm_120a; with it the
    warp-specialised GEMM (2 groups x 4 MMA warps at 232 regs + 4 stream warps at 40) launches
    at 168 registers/thread without spills, while a uniform 168-register cap spills."""
    from cobench import stress as S
    smax = S.KCfg(ak="gemm", mode="ws", wm=2, wn=2, nt=8, stages=3, nga=2, wa=8, warps=12,
                  smaxnreg=1, ra=232, rb=40, su=2)
    src = smax.defines() + S.SRC
    plain = S._ptxas_log(src, "sm_120")
    arch_a = S._ptxas_log(src, "sm_120a")
    uniform = S.ptxas_info(dataclasses.replace(smax, smaxnreg=0, ra=0, rb=0, maxreg=168))
    print(f"     sm_120: {plain.get('error', 'compiled')[:120]!r}; sm_120a: {arch_a}; uniform 168: {uniform}")
    assert "error" in plain and "setmaxnreg" in plain["error"], plain
    assert "error" not in arch_a and arch_a["regs"] <= 168 and arch_a["spill_stores"] == 0, arch_a
    assert uniform["spill_stores"] > 0, uniform
    k = S.kernel(smax)
    assert k.arch.endswith("a") and k.num_regs <= 168
    REPORT["16_setmaxnreg"] = {"sm_120": plain.get("error", "")[:300], "sm_120a": arch_a, "uniform_168": uniform}


def test_7_readme():
    path = os.path.join(BENCH, "README.md")
    assert os.path.exists(path) and os.path.getsize(path) > 1000
    REPORT["7_readme"] = path


def _key(item):
    n = item[0]
    num = n.split("_")[1]
    return (int(num[:-1]) if num[-1].isalpha() else int(num), num)


def main():
    global OUT_DIR
    argv = sys.argv[1:]
    if "--out" in argv:
        i = argv.index("--out")
        OUT_DIR = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    out_dir = OUT_DIR
    tests = sorted([(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)], key=_key)
    if argv:
        tests = [(n, f) for n, f in tests if any(k in n for k in argv)]
    before = other_gpu_procs()
    print(f"device: {INFO['name']} sms={NSM} L2={L2 >> 20}MB DRAM peak={DRAM:.0f} GB/s "
          f"max SM clock={INFO['max_sm_clock_mhz_nvml']} MHz; other GPU procs at start: {before or 'none'}")
    status = {}
    for name, f in tests:
        print(f"== {name}: {(f.__doc__ or '').strip().splitlines()[0] if f.__doc__ else ''}", flush=True)
        t = time.time()
        try:
            f()
            status[name] = "PASS"
        except Exception as e:  # report and continue
            traceback.print_exc()
            status[name] = f"FAIL: {e!r}"[:300]
        print(f"   -> {status[name]} ({time.time() - t:.1f}s)", flush=True)
    after = other_gpu_procs()
    REPORT["_meta"] = {"device": INFO, "status": status, "other_gpu_procs_start": before,
                       "other_gpu_procs_end": after, "date": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_output.json"), "w") as fh:
        json.dump(REPORT, fh, indent=1, default=str)
    print("\nSUMMARY")
    for n, s in status.items():
        print(f"  {n:28s} {s}")
    print(f"other GPU procs at end: {after or 'none'}")
    sys.exit(0 if all(s == "PASS" for s in status.values()) else 1)


if __name__ == "__main__":
    main()
