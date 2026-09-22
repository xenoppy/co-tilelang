"""Acceptance tests for cobench (P0-2). Run directly:

    CUDA_HOME=/usr/local/cuda-12.9 ~/mpk-env/bin/python research/bench/tests/test_cobench.py

Each test_* function asserts the behaviour listed in the P0-2 completion criteria and
records its key numbers; a PASS/FAIL table and a JSON with the numbers are written at
the end (research/results/2026-09-22_cobench_validation/test_output.json).
"""
from __future__ import annotations

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
        assert r.n == 50 and r.l2 == {"flush": "cold-flush", "graph": "cold-rotate", "hot": "hot"}[mode]
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
    assert res["flush"].gbps <= DRAM and res["graph"].gbps <= DRAM, "cold modes exceed DRAM peak"
    assert res["hot"].gbps > 2 * DRAM, "hot mode should run from L2"
    # the 256 MB flush write takes >150 us; a 16 MB copy takes ~22 us -> excluded if close
    assert abs(res["flush"].median / res["graph"].median - 1) < 0.25, "flush time leaked into timing"
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
    out = os.path.join(RESULTS, "2026-09-22_cobench_validation", "repro_matmul_graph.json")
    p = subprocess.run([sys.executable, os.path.join(BENCH, "scripts", "repro_matmul.py"),
                        "--runs", "5", "--out", out], capture_output=True, text=True)
    print("\n".join("    " + l for l in p.stdout.splitlines() if l.startswith("run ")))
    assert p.returncode == 0, p.stderr[-2000:]
    s = json.load(open(out))
    print(f"    CV across 5 processes: {s['cv_across_runs']*100:.2f}%  (clock CV "
          f"{s['cv_clock_across_runs']*100:.2f}%, cycles/call CV {s['cv_cycles_per_call_across_runs']*100:.3f}%)")
    assert s["cv_across_runs"] < 0.02
    REPORT["6_repro"] = {k: v for k, v in s.items() if k != "per_run"}


def test_7_readme():
    path = os.path.join(BENCH, "README.md")
    assert os.path.exists(path) and os.path.getsize(path) > 1000
    REPORT["7_readme"] = path


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    if len(sys.argv) > 1:
        tests = [(n, f) for n, f in tests if any(k in n for k in sys.argv[1:])]
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
    os.makedirs(os.path.join(RESULTS, "2026-09-22_cobench_validation"), exist_ok=True)
    with open(os.path.join(RESULTS, "2026-09-22_cobench_validation", "test_output.json"), "w") as fh:
        json.dump(REPORT, fh, indent=1, default=str)
    print("\nSUMMARY")
    for n, s in status.items():
        print(f"  {n:28s} {s}")
    print(f"other GPU procs at end: {after or 'none'}")
    sys.exit(0 if all(s == "PASS" for s in status.values()) else 1)


if __name__ == "__main__":
    main()
