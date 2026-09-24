"""Co-location limit study (Task L, plan §1.4 stress kernels v0): the performance ceiling of
fine-grained intra-SM co-location vs SM partitioning on the RTX PRO 6000 (600 W cap), with
idealised synthetic kernels (research/bench/cobench/stress.py).

    source research/env.sh
    python research/bench/scripts/run_guarded.py -- python research/bench/scripts/limit_study.py run
    python research/bench/scripts/limit_study.py report          # tables for the README

Stages (each saved to research/results/2026-09-24_limit_study/limit_study.json as soon as it
is measured; a rerun skips finished stages, --redo S1,S2 re-measures them):
  V      validation: register-only MMA FLOP/clk/SM, stream GB/s (ldg / cp.async / bulk / read+write
         / cobench read_u4) incl. stream GB/s vs SM count, GEMM-like solo, constant vs toggling
         MMA operands (power)
  C      calibration: steady solo times -> work amounts for MMA-alone : stream-alone = 0.5 / 1 / 2
  reg_r, gemm_r (r = 0.5, 1, 2)   one cell per MMA family x work ratio, all doing the same work:
         solo_a, solo_b, serial, green splits, SM-level persistent (dynamic + takeover), CTA-level
         co-residence, warp-specialised (incl. setmaxnreg), same-warp interleaving.
         Every variant passes the work-completion check (k=2) first. Screening: one steady round
         of every variant / split / ratio; final: 4 interleaved steady rounds of serial, the solos
         and the best two of every kind.
  nocap_reg, nocap_gemm   ratio 1 inside a 94-SM green sub-device (the same kernels with half the
         SMs -> below the power cap): separates the power-cap effect from in-SM contention.
  lowint_reg   ratio 1 on the full device with a reduced-intensity MMA (2 MMA warps per SM, below
         the cap): the same question for an op that leaves the SM under-occupied.

GPU sharing (research/rules.md 7, cobench.GuardPolicy): bench_steady waits for a free GPU and
raises GpuYield on foreign SM activity; this script then saves and exits with 75, and
run_guarded.py waits 30 min and relaunches it (resuming at the next stage).
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.normpath(os.path.join(BENCH, "..", ".."))
sys.path.insert(0, BENCH)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cobench as cb  # noqa: E402
from cobench import stress as S  # noqa: E402
from cobench.stress import KCfg  # noqa: E402

OUT = os.path.join(ROOT, "research", "results", "2026-09-24_limit_study")
RES = os.path.join(OUT, "limit_study.json")
YIELD_RC = 75
NSM = 188
PEAK_FPC = 1024.0                                    # mma.sync bf16 FLOP/clk/SM (2026-09-22)
RATIOS = (0.5, 1.0, 2.0)
MB = 1 << 20

# ---------------------------------------------------------------- kernel configurations
# A (reg): 8 warps x 8 accumulator chains, 2 alternating operand sets, <= 64 registers
REG_A = KCfg(ak="reg", maxreg=64)
# A (gemm): 128x128 tile by 4 warps (64x64 warp tiles), 3 stages (48 KB), 2 CTAs/SM (best solo)
GEMM_A = KCfg(ak="gemm", wm=2, wn=2, nt=8, warps=4, stages=3, maxreg=232)
GEMM_A_CTAS = 2
# B: ldg streaming, 4 x 16 B per lane per batch (double-buffered), L2 evict-first, 8 warps x 2 CTAs/SM
STREAM = KCfg(ak="none", mode="solo_b", su=4, maxreg=48)
STREAM_CTAS = 2
# reduced intensity (the no-power-cap regime on the full device): 2 MMA warps per SM
REG_A_LOW = KCfg(ak="reg", warps=2, maxreg=64)
REG_UNITS_BYTES = 512 * MB                           # reg family: 512 MB stream, 16384 x 32 KB chunks
GEMM_TILES = 2048                                    # gemm family: 2048 tiles of 128x128x2048
GREEN_A = (94, 124, 140, 148, 156, 160, 164, 168, 172, 180)   # A's SMs in the green sweep
SM_A = (124, 140, 156, 164, 172)                               # A-first SMs in the SM-level sweep


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdefault(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    return str(o)


def load() -> dict:
    if os.path.exists(RES):
        with open(RES) as f:
            return json.load(f)
    return {"stages": {}, "runs": []}


def save(res: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    tmp = RES + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1, default=jdefault)
    os.replace(tmp, RES)


def env_meta() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    return {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "device": cb.device_info(), "torch": torch.__version__}


# ---------------------------------------------------------------- measurement helpers
def summarize(r: cb.SteadyResult, extra: dict | None = None) -> dict:
    out = {"config": r.config, "guard": {"clean": (r.guard or {}).get("clean"),
                                         "attempts": len((r.guard or {}).get("attempts", []))},
           "throttle": (r.nvml or {}).get("throttle"), "variants": {}}
    sp = r.derived.get("speedup", {})
    for n, v in r.variants.items():
        d = {k: v.get(k) for k in ("t_iter_us", "cv_slices", "clock_mhz", "power_w", "energy_mj_per_iter",
                                   "kcycles_per_iter", "n_iter", "iter_p10_us", "iter_p90_us")}
        d["slices_us"] = v.get("slices_us")
        if v.get("ops"):
            d["ops"] = {k: o["median_us"] for k, o in v["ops"].items()}
        if n in sp:
            d["speedup"] = sp[n]["ratio_of_medians"]
            d["speedup_paired"] = [sp[n]["paired_median"], sp[n]["paired_min"], sp[n]["paired_max"]]
        if extra and n in extra:
            d.update(extra[n])
        out["variants"][n] = d
    return out


def steady(variants: dict, *, kind: str, reference: str | None = "serial", label: str = "") -> cb.SteadyResult:
    fns = {n: v.fn for n, v in variants.items()}
    if kind == "screen":
        kw = dict(slice_s=0.5, settle_s=0.25, rounds=1, warmup_s=2.0)
    elif kind == "final":
        kw = dict(slice_s=1.2, settle_s=0.4, rounds=4, warmup_s=3.0)
    elif kind == "quick":
        kw = dict(slice_s=0.6, settle_s=0.3, rounds=2, warmup_s=2.0)
    else:
        raise ValueError(kind)
    log(f"[{label}] bench_steady {kind}: {len(fns)} variants")
    t = time.time()
    r = cb.bench_steady(fns, reference=reference, label=label, **kw)
    log(f"[{label}] done in {time.time() - t:.0f}s")
    return r


def on_stream(v: S.Variant, stream, name: str | None = None) -> S.Variant:
    """Run variant v on another stream (e.g. a green sub-device), forked from / joined into the
    current stream."""
    par = cb.Par(("x", stream, v.fn))
    return S.Variant(name or v.name, v.wl, v.launches, par, dict(v.desc, stream="sub-device"))


class Parts:
    """Green partitions, created once: full device splits (A = [0, n)) and the 94-SM sub-device."""

    def __init__(self):
        self.full: dict = {}
        self.sub = None
        self.inner: dict = {}
        self.sub_remap = None

    def split(self, n_a: int):
        if n_a not in self.full:
            self.full[n_a] = cb.split_sms(n_a, ignore_coscheduling=True)
        return self.full[n_a]

    def sub_device(self, n: int = 94):
        if self.sub is None:
            self.sub = cb.split_sms(n, ignore_coscheduling=True)
            self.sub_remap = cb.build_sm_remap(self.sub.stream, expected=self.sub.n_sms).to_tensor()
        return self.sub

    def nested(self, n_outer: int, n_a: int):
        if n_a not in self.inner:
            self.inner[n_a] = S.split_nested(n_outer, n_a)     # [0, n_a) + [n_a, n_outer) = self.sub's SMs
        return self.inner[n_a]

    def close(self):
        for p in list(self.full.values()) + list(self.inner.values()) + ([self.sub] if self.sub else []):
            try:
                p.close()
            except Exception:
                pass


# ---------------------------------------------------------------- variant sets
def add(V: dict, skipped: dict, name: str, make):
    try:
        v = make()
        v.name = name
        V[name] = v
    except (ValueError, RuntimeError) as e:
        skipped[name] = str(e)[:200]


def build_variants_lowint(wl: S.Workload, parts: Parts, remap) -> tuple[dict, dict]:
    """Reduced-intensity register-only MMA (2 MMA warps per SM: <= 50% of the tensor pipe, below
    the power cap) co-located with the same stream: every variant keeps the A intensity at 2
    warps per SM (an op that cannot use more warps)."""
    A = REG_A_LOW
    V, skipped = {}, {}
    la = S.solo_a(wl, A, name="solo_a")
    lb = S.solo_b(wl, STREAM, ctas_per_sm=STREAM_CTAS, name="solo_b")
    V["solo_a"] = S.v_single("solo_a", la, {"kind": "solo_a", "launch": la.info()})
    V["solo_b"] = S.v_single("solo_b", lb, {"kind": "solo_b", "launch": lb.info()})
    V["serial"] = S.v_serial(wl, la, lb)
    for na in GREEN_A:
        part = parts.split(na)
        add(V, skipped, f"green_{part.n_sms}", lambda part=part: S.v_green(wl, part, A, STREAM, ctas_a=1,
                                                                            ctas_b=STREAM_CTAS))
    for na in SM_A:        # one 2-warp CTA per SM for both roles (keeps the A intensity)
        add(V, skipped, f"sm_{na}", lambda na=na: S.v_sm(wl, A, na, remap, ctas_per_sm=1))
    for kb in (1, 2, 4):   # 1 A CTA (2 warps) + kb B CTAs (2 warps) per SM
        add(V, skipped, f"cta_w2_1_{kb}", lambda kb=kb: S.v_cta(wl, A, 1, kb))
    for wb in (2, 4, 8):
        add(V, skipped, f"ws_2_{wb}", lambda wb=wb: S.v_ws(wl, KCfg(ak="reg", maxreg=64), 2, wb))
    for sd, mr in ((8, 128), (16, 168)):
        add(V, skipped, f"sw_w2sd{sd}", lambda sd=sd, mr=mr: S.v_sw(wl, KCfg(ak="reg", warps=2, sd=sd, maxreg=mr)))
    return V, skipped


def build_variants(wl: S.Workload, parts: Parts, remap, *, sub: bool = False) -> tuple[dict, dict]:
    """All co-location variants of one cell (full device, or inside the 94-SM sub-device)."""
    fam = wl.family
    A = REG_A if fam == "reg" else GEMM_A
    a_ctas = 1 if fam == "reg" else GEMM_A_CTAS
    n = parts.sub.n_sms if sub else NSM
    V, skipped = {}, {}

    def place(v):
        return on_stream(v, parts.sub.stream) if sub else v

    la = S.solo_a(wl, A, n_sms=n, ctas_per_sm=a_ctas, name="solo_a")
    lb = S.solo_b(wl, STREAM, n_sms=n, ctas_per_sm=STREAM_CTAS, name="solo_b")
    V["solo_a"] = place(S.v_single("solo_a", la, {"kind": "solo_a", "launch": la.info()}))
    V["solo_b"] = place(S.v_single("solo_b", lb, {"kind": "solo_b", "launch": lb.info()}))
    V["serial"] = place(S.v_serial(wl, la, lb))
    # 2. green-context partitions (A on [0, n_a), B on the rest)
    greens = (40, 48, 54, 62, 70) if sub else GREEN_A
    for na in greens:
        part = parts.nested(n, na) if sub else parts.split(na)
        add(V, skipped, f"green_{part.n_sms}", lambda part=part: S.v_green(wl, part, A, STREAM, ctas_a=a_ctas,
                                                                            ctas_b=STREAM_CTAS))
    # 3. one persistent kernel, SM-level roles, dynamic queues + takeover
    sm_splits = (48, 62, 76) if sub else SM_A
    rm = parts.sub_remap if sub else remap
    for na in sm_splits:
        add(V, skipped, f"sm_{na}", lambda na=na: place(S.v_sm(wl, A, na, rm, ctas_per_sm=a_ctas, n_sms=n)))
    # 4. CTA-level co-residence (one kernel, k_a A CTAs + k_b B CTAs on every SM)
    if fam == "reg":
        ctas = [("w4", KCfg(ak="reg", warps=4, maxreg=64), 2, 1), ("w4", KCfg(ak="reg", warps=4, maxreg=64), 2, 2),
                ("w4", KCfg(ak="reg", warps=4, maxreg=64), 3, 1), ("w4", KCfg(ak="reg", warps=4, maxreg=64), 4, 2),
                ("w8", KCfg(ak="reg", warps=8, maxreg=64), 1, 1)]
    else:
        g64 = KCfg(ak="gemm", wm=2, wn=2, nt=4, warps=4, maxreg=128)       # 128x64 tiles, 126 regs
        ctas = [("t128x64s2", dataclasses.replace(g64, stages=2), 2, 1),
                ("t128x64s2", dataclasses.replace(g64, stages=2), 2, 2),
                ("t128x64s2", dataclasses.replace(g64, stages=2), 3, 1),
                ("t128x128s3", GEMM_A, 1, 1),
                ("t128x128w8s2", KCfg(ak="gemm", stages=2, maxreg=128), 1, 1)]
    for tag, cfg, ka, kb in ctas:
        add(V, skipped, f"cta_{tag}_{ka}_{kb}", lambda cfg=cfg, ka=ka, kb=kb: place(S.v_cta(wl, cfg, ka, kb, n_sms=n)))
    # 5. warp-specialised (one CTA per SM); setmaxnreg rebalancing where it matters
    if fam == "reg":
        wss = [("", KCfg(ak="reg", maxreg=64), 4, 2), ("", KCfg(ak="reg", maxreg=64), 4, 4),
               ("", KCfg(ak="reg", maxreg=64), 8, 2), ("", KCfg(ak="reg", maxreg=64), 8, 4),
               ("", KCfg(ak="reg", maxreg=64), 8, 8), ("", KCfg(ak="reg", maxreg=64), 12, 4)]
    else:
        g8 = KCfg(ak="gemm", stages=4, maxreg=128)                         # 1 group, 8 warps, 64 KB
        g4x2 = KCfg(ak="gemm", wm=2, wn=2, nt=8, stages=3, nga=2)          # 2 groups x 4 warps, 96 KB
        wss = [("g8", g8, 8, 2), ("g8", g8, 8, 4), ("g8", g8, 8, 8),
               ("g4", KCfg(ak="gemm", wm=2, wn=2, nt=8, stages=4, maxreg=232), 4, 4),
               ("g4x2", dataclasses.replace(g4x2, smaxnreg=1, ra=232, rb=40, su=2), 8, 4),
               ("g4x2", dataclasses.replace(g4x2, smaxnreg=1, ra=200, rb=40, su=2), 8, 8),
               ("g8", dataclasses.replace(g8, maxreg=0, smaxnreg=1, ra=160, rb=40, su=2), 8, 16)]
    for tag, cfg, wa, wb in wss:
        nm = f"ws_{tag + '_' if tag else ''}{wa}_{wb}" + (f"_smax{cfg.ra}_{cfg.rb}" if cfg.smaxnreg else "")
        add(V, skipped, nm, lambda cfg=cfg, wa=wa, wb=wb: place(S.v_ws(wl, cfg, wa, wb, n_sms=n)))
    # 6. same-warp interleaving
    if fam == "reg":
        sws = [("w8", KCfg(ak="reg", warps=8, maxreg=96), 1), ("w16", KCfg(ak="reg", warps=16, maxreg=96), 1),
               ("w8sd8", KCfg(ak="reg", warps=8, sd=8, maxreg=128), 1), ("w4x2", KCfg(ak="reg", warps=4, maxreg=96), 2),
               ("w8sd16", KCfg(ak="reg", warps=8, sd=16, maxreg=168), 1),
               ("w16sd8", KCfg(ak="reg", warps=16, sd=8, maxreg=120), 1)]
    else:
        sws = []
        nk = wl.K // 32
        for tag, base, cps in (("g8s3", KCfg(ak="gemm", stages=3, maxreg=168), 1),
                               ("g8s4", KCfg(ak="gemm", stages=4, maxreg=168), 1),
                               ("g4s2x2", KCfg(ak="gemm", wm=2, wn=2, nt=8, warps=4, stages=2, maxreg=232), 2),
                               ("g8s2x2", KCfg(ak="gemm", stages=2, maxreg=128), 2)):
            try:
                P = wl.sw_packets(dataclasses.replace(base, mode="sw"))
            except ValueError as e:
                skipped[f"sw_{tag}"] = str(e)[:200]
                continue
            sws.append((tag, dataclasses.replace(base, spk=-(-P // nk)), cps))
    for tag, cfg, cps in sws:
        add(V, skipped, f"sw_{tag}", lambda cfg=cfg, cps=cps: place(S.v_sw(wl, cfg, ctas_per_sm=cps, n_sms=n)))
    return V, skipped


KINDS = ("green", "sm", "cta", "ws", "sw")


def kind_of(name: str) -> str:
    return name.split("_")[0] if name.split("_")[0] in KINDS else name


def verify_all(V: dict, label: str) -> dict:
    out = {}
    for n, v in V.items():
        r = v.verify(2)
        out[n] = {"ok": r["ok"], "errors": r["errors"], "units_a": r.get("units_a"), "units_b": r.get("units_b")}
        if not r["ok"]:
            raise RuntimeError(f"[{label}] work-completion check failed for {n}: {r['errors']}")
    log(f"[{label}] work-completion check: {len(out)} variants OK (every unit exactly 2x, checksums = 2x reference)")
    return out


def variant_desc(V: dict) -> dict:
    return {n: v.desc for n, v in V.items()}


# ---------------------------------------------------------------- stages
def stage_V(res, parts):
    wl = S.Workload("reg", stream_bytes=1024 * MB, a_iters=256)
    wl_rw = S.Workload("reg", stream_bytes=512 * MB, a_iters=16, rw=True)
    wg = S.Workload("gemm", stream_bytes=64 * MB, n_tiles=GEMM_TILES)
    V = {}
    L = {}

    def single(name, launch, desc=None):
        L[name] = launch
        V[name] = S.v_single(name, launch, desc or {"launch": launch.info()})

    single("mma_reg_8w", S.solo_a(wl, REG_A))
    single("mma_reg_4w_x2", S.solo_a(wl, dataclasses.replace(REG_A, warps=4), ctas_per_sm=2))
    single("mma_reg_8w_constops", S.solo_a(wl, dataclasses.replace(REG_A, nsets=1)))
    single("stream_ldg", S.solo_b(wl, STREAM, ctas_per_sm=STREAM_CTAS))
    single("stream_ldg_nohint", S.solo_b(wl, dataclasses.replace(STREAM, shint=0), ctas_per_sm=STREAM_CTAS))
    single("stream_ldg_su1_x1", S.solo_b(wl, dataclasses.replace(STREAM, su=1), ctas_per_sm=1))
    single("stream_cpasync", S.solo_b(wl, KCfg(ak="none", mode="solo_b", skind=1, su=8), ctas_per_sm=2))
    single("stream_bulk", S.solo_b(wl, KCfg(ak="none", mode="solo_b", skind=2, su=2, sbulk=8192, warps=2),
                                   ctas_per_sm=2))
    single("stream_rw", S.solo_b(wl_rw, dataclasses.replace(STREAM, srw=1, maxreg=64), ctas_per_sm=STREAM_CTAS))
    single("gemm_4w_s3_x2", S.solo_a(wg, GEMM_A, ctas_per_sm=GEMM_A_CTAS))
    single("gemm_8w_s3_x2", S.solo_a(wg, KCfg(ak="gemm", stages=3, maxreg=128), ctas_per_sm=2))
    single("gemm_8w_s4_x1", S.solo_a(wg, KCfg(ak="gemm", stages=4, maxreg=128), ctas_per_sm=1))
    buf = wl.sbuf
    V["read_u4"] = S.Variant("read_u4", wl, [], lambda i=0: cb.read_u4(buf), {"kind": "cobench.read_u4"})
    # stream GB/s vs SMs (green partitions)
    for n in (8, 16, 24, 32, 40, 48, 64, 94):
        p = parts.split(n)
        lb = S.solo_b(wl, STREAM, n_sms=p.n_sms, ctas_per_sm=STREAM_CTAS)
        L[f"stream_sms{n}"] = lb
        V[f"stream_sms{n}"] = S.Variant(f"stream_sms{n}", wl, [lb], cb.Par(("x", p.stream, lambda lb=lb: lb.launch())),
                                        {"launch": lb.info(), "sms": p.n_sms})
    # work-completion check; MMA checksums against the reference of the study configs (the reg
    # hash depends only on the unit, the gemm hash is tiling independent), except the
    # constant-operand variant (different operands by design)
    S.reference_sum_a(wl, REG_A)
    S.reference_sum_a(wg, GEMM_A)
    checks = {}
    for n, v in V.items():
        if v.launches:
            r = v.verify(1, check_a=(n != "mma_reg_8w_constops"))
            checks[n] = r["ok"]
            if not r["ok"]:
                raise RuntimeError(f"V: {n}: {r['errors']}")
    r = steady(V, kind="quick", reference=None, label="V")
    s = summarize(r)
    best_read = 0.0
    for n, d in s["variants"].items():
        v = V[n]
        if n.startswith("mma") or n.startswith("gemm"):
            d["tflops"] = v.wl.flops_a / d["t_iter_us"] / 1e6
            if d.get("clock_mhz"):
                d["flop_per_clk_per_sm"] = v.wl.flops_a / (d["t_iter_us"] * 1e-6) / (d["clock_mhz"] * 1e6) / NSM
                d["frac_of_1024"] = d["flop_per_clk_per_sm"] / PEAK_FPC
        else:
            d["gbps"] = v.wl.bytes_b / d["t_iter_us"] / 1e3
            if n != "stream_rw" and not n.startswith("stream_sms"):
                best_read = max(best_read, d["gbps"])
    for n, d in s["variants"].items():
        if "gbps" in d and n != "stream_rw":
            d["frac_of_best_read"] = d["gbps"] / best_read
            d["frac_of_dram_peak"] = d["gbps"] / cb.device_info()["dram_peak_gbps"]
    vr = s["variants"]
    s["validation"] = {
        "mma_reg_frac_of_1024": vr["mma_reg_8w"]["frac_of_1024"],
        "mma_reg_pass": vr["mma_reg_8w"]["frac_of_1024"] >= 0.85,
        "best_read_gbps": best_read,
        "stream_ldg_frac_of_best": vr["stream_ldg"]["frac_of_best_read"],
        "stream_pass": vr["stream_ldg"]["frac_of_best_read"] >= 0.85,
        "dram_peak_gbps": cb.device_info()["dram_peak_gbps"],
    }
    s["checks"] = checks
    s["launches"] = {n: l.info() for n, l in L.items()}
    res["stages"]["V"] = s
    log(f"V: {s['validation']}")
    for w in (wl, wl_rw, wg):
        w.free()


def stage_C(res, parts):
    """Steady solo times of the default work amounts -> work amounts per ratio."""
    out = {}
    for fam in ("reg", "gemm"):
        if fam == "reg":
            wl = S.Workload("reg", stream_bytes=REG_UNITS_BYTES, a_iters=256)
            S.reference_sum_a(wl, REG_A)
            la = S.solo_a(wl, REG_A)
        else:
            wl = S.Workload("gemm", stream_bytes=GEMM_TILES * 8 * 32 * 1024, n_tiles=GEMM_TILES)  # 8 chunks/tile
            S.reference_sum_a(wl, GEMM_A)
            la = S.solo_a(wl, GEMM_A, ctas_per_sm=GEMM_A_CTAS)
        lb = S.solo_b(wl, STREAM, ctas_per_sm=STREAM_CTAS)
        V = {"solo_a": S.v_single("solo_a", la, {}), "solo_b": S.v_single("solo_b", lb, {}),
             "serial": S.v_serial(wl, la, lb)}
        verify_all(V, f"C/{fam}")
        s = summarize(steady(V, kind="quick", label=f"C/{fam}"))
        ta, tb = s["variants"]["solo_a"]["t_iter_us"], s["variants"]["solo_b"]["t_iter_us"]
        if fam == "reg":
            # T_A scales with a_iters (per-unit overhead is small); stream fixed at 512 MB
            amounts = {str(r): int(max(64, round(256 * r * tb / ta / 8) * 8)) for r in RATIOS}
        else:
            # T_B scales with the stream chunks per tile (8 measured); tiles fixed
            amounts = {str(r): int(max(1, round(8 * ta / (r * tb)))) for r in RATIOS}
        out[fam] = {"t_a_us": ta, "t_b_us": tb, "ratio_measured": ta / tb, "amounts": amounts, "steady": s,
                    "workload": wl.describe()}
        log(f"C/{fam}: T_A {ta:.1f} us, T_B {tb:.1f} us -> amounts {amounts}")
        wl.free()
    res["stages"]["C"] = out


def make_workload(fam: str, r: float, calib: dict) -> S.Workload:
    amt = calib[fam]["amounts"][str(r)]
    if fam == "reg":
        wl = S.Workload("reg", stream_bytes=REG_UNITS_BYTES, a_iters=amt)
        S.reference_sum_a(wl, REG_A)
    else:
        wl = S.Workload("gemm", stream_bytes=GEMM_TILES * amt * 32 * 1024, n_tiles=GEMM_TILES)
        S.reference_sum_a(wl, GEMM_A)
    return wl


def run_cell(res, parts, remap, fam: str, r: float, *, sub: bool = False, wl: S.Workload | None = None,
             stage: str = "", builder=None):
    wl = wl or make_workload(fam, r, res["stages"]["C"])
    V, skipped = (builder or build_variants)(wl, parts, remap, **({} if builder else {"sub": sub}))
    checks = verify_all(V, stage)
    part = res["stages"].setdefault("partial", {}).setdefault(stage, {})
    if "screen" not in part:
        s1 = summarize(steady(V, kind="screen", label=f"{stage}/screen"))
        part["screen"] = s1
        save(res)
    s1 = part["screen"]
    # finalists: the two best (screening time) of every kind
    fin = ["serial", "solo_a", "solo_b"]
    for k in KINDS:
        cand = sorted((d["t_iter_us"], n) for n, d in s1["variants"].items() if kind_of(n) == k)
        fin += [n for _, n in cand[:2]]
    s2 = summarize(steady({n: V[n] for n in fin}, kind="final", label=f"{stage}/final"))
    out = {"family": fam, "ratio": r, "sub_device_sms": parts.sub.n_sms if sub else None,
           "workload": wl.describe(), "screen": s1, "final": s2, "skipped": skipped, "checks": checks,
           "variants": variant_desc(V)}
    out["table"] = cell_table(out)
    res["stages"][stage] = out
    res["stages"]["partial"].pop(stage, None)
    wl.free()
    return out


def stage_nocap(res, parts, fam: str, stage: str):
    """Ratio 1 inside the 94-SM sub-device: calibrate the A work there, then the full cell."""
    sub = parts.sub_device(94)
    calib = res["stages"]["C"][fam]
    if fam == "reg":
        wl = S.Workload("reg", stream_bytes=REG_UNITS_BYTES, a_iters=calib["amounts"]["1.0"])
        S.reference_sum_a(wl, REG_A)
        la = S.solo_a(wl, REG_A, n_sms=sub.n_sms)
    else:
        wl = S.Workload("gemm", stream_bytes=GEMM_TILES * calib["amounts"]["1.0"] * 32 * 1024, n_tiles=GEMM_TILES)
        S.reference_sum_a(wl, GEMM_A)
        la = S.solo_a(wl, GEMM_A, n_sms=sub.n_sms, ctas_per_sm=GEMM_A_CTAS)
    lb = S.solo_b(wl, STREAM, n_sms=sub.n_sms, ctas_per_sm=STREAM_CTAS)
    V = {"solo_a": on_stream(S.v_single("solo_a", la, {}), sub.stream),
         "solo_b": on_stream(S.v_single("solo_b", lb, {}), sub.stream),
         "serial": on_stream(S.v_serial(wl, la, lb), sub.stream)}
    verify_all(V, f"{stage}/calib")
    s = summarize(steady(V, kind="quick", label=f"{stage}/calib"))
    ta, tb = s["variants"]["solo_a"]["t_iter_us"], s["variants"]["solo_b"]["t_iter_us"]
    if fam == "reg":
        a_iters = int(max(64, round(wl.a_iters * tb / ta / 8) * 8))
        wl2 = S.Workload("reg", stream_bytes=REG_UNITS_BYTES, a_iters=a_iters)
        S.reference_sum_a(wl2, REG_A)
    else:
        c = int(max(1, round(calib["amounts"]["1.0"] * ta / tb)))
        wl2 = S.Workload("gemm", stream_bytes=GEMM_TILES * c * 32 * 1024, n_tiles=GEMM_TILES)
        S.reference_sum_a(wl2, GEMM_A)
    wl.free()
    log(f"{stage}: 94-SM solo T_A {ta:.1f} T_B {tb:.1f} -> workload {wl2.describe()}")
    out = run_cell(res, parts, None, fam, 1.0, sub=True, wl=wl2, stage=stage)
    out["calib"] = s


def stage_lowint(res, parts, remap, stage: str):
    """Ratio 1 on the full device with the reduced-intensity MMA (below the power cap)."""
    calib = res["stages"]["C"]["reg"]
    wl = S.Workload("reg", stream_bytes=REG_UNITS_BYTES, a_iters=calib["amounts"]["1.0"] // 4)
    S.reference_sum_a(wl, REG_A_LOW)
    la = S.solo_a(wl, REG_A_LOW)
    lb = S.solo_b(wl, STREAM, ctas_per_sm=STREAM_CTAS)
    V = {"solo_a": S.v_single("solo_a", la, {}), "solo_b": S.v_single("solo_b", lb, {}),
         "serial": S.v_serial(wl, la, lb)}
    verify_all(V, f"{stage}/calib")
    s = summarize(steady(V, kind="quick", label=f"{stage}/calib"))
    ta, tb = s["variants"]["solo_a"]["t_iter_us"], s["variants"]["solo_b"]["t_iter_us"]
    a_iters = int(max(16, round(wl.a_iters * tb / ta / 8) * 8))
    wl.free()
    wl2 = S.Workload("reg", stream_bytes=REG_UNITS_BYTES, a_iters=a_iters)
    S.reference_sum_a(wl2, REG_A_LOW)
    log(f"{stage}: 2-warp MMA solo T_A {ta:.1f} T_B {tb:.1f} -> a_iters {a_iters}")
    out = run_cell(res, parts, remap, "reg", 1.0, wl=wl2, stage=stage, builder=build_variants_lowint)
    out["calib"] = s
    out["regime"] = "reduced intensity: 2 MMA warps per SM"


# ---------------------------------------------------------------- analysis
def cell_table(cell: dict) -> dict:
    """Best variant per kind (final run): speed-up vs serial, clock, power, energy; best intra-SM
    (cta/ws/sw) vs best partition (green/sm); contention-free prediction for every variant:
    T_pred = max(T_A_solo * f_A_solo / f_variant, T_B_solo) (A time scales with 1/clock at a
    fixed SM count; the DRAM-bound stream does not)."""
    v = cell["final"]["variants"]
    ser, sa, sb = v["serial"], v["solo_a"], v["solo_b"]
    rows = {}
    for k in ("serial", "solo_a", "solo_b") + KINDS:
        names = [n for n in v if (n == k if k in ("serial", "solo_a", "solo_b") else kind_of(n) == k)]
        if not names:
            continue
        n = min(names, key=lambda x: v[x]["t_iter_us"])
        d = v[n]
        row = {"variant": n, "t_us": d["t_iter_us"], "speedup": ser["t_iter_us"] / d["t_iter_us"],
               "clock_mhz": d["clock_mhz"], "power_w": d["power_w"], "energy_mj": d["energy_mj_per_iter"],
               "cv": d["cv_slices"]}
        if sa.get("clock_mhz") and d.get("clock_mhz"):
            row["t_pred_contention_free_us"] = max(sa["t_iter_us"] * sa["clock_mhz"] / d["clock_mhz"], sb["t_iter_us"])
            row["t_over_pred"] = d["t_iter_us"] / row["t_pred_contention_free_us"]
        if d.get("ops"):
            row["ops_us"] = d["ops"]
        rows[k] = row
    part = min((rows[k] for k in ("green", "sm") if k in rows), key=lambda x: x["t_us"])
    intra = min((rows[k] for k in ("cta", "ws", "sw") if k in rows), key=lambda x: x["t_us"])
    ideal = max(sa["t_iter_us"], sb["t_iter_us"])
    return {"rows": rows, "best_partition": part["variant"], "best_intra": intra["variant"],
            "intra_over_partition": part["t_us"] / intra["t_us"],
            "ratio_measured": sa["t_iter_us"] / sb["t_iter_us"],
            "ideal_overlap_speedup": ser["t_iter_us"] / ideal,
            "sum_of_solos_over_serial": (sa["t_iter_us"] + sb["t_iter_us"]) / ser["t_iter_us"]}


def report(res: dict) -> str:
    lines = []
    V = res["stages"].get("V")
    if V:
        va = V["validation"]
        lines.append(f"Validation: reg MMA {va['mma_reg_frac_of_1024'] * 100:.1f}% of 1024 FLOP/clk/SM "
                     f"({'PASS' if va['mma_reg_pass'] else 'FAIL'}); stream ldg {va['stream_ldg_frac_of_best'] * 100:.1f}% "
                     f"of best read {va['best_read_gbps']:.0f} GB/s ({'PASS' if va['stream_pass'] else 'FAIL'})")
        lines.append("| kernel | t us | clock MHz | power W | TFLOP/s or GB/s | FLOP/clk/SM or % best |")
        lines.append("|---|---|---|---|---|---|")
        for n, d in V["variants"].items():
            perf = f"{d['tflops']:.0f} TFLOP/s" if "tflops" in d else f"{d['gbps']:.0f} GB/s"
            eff = (f"{d['flop_per_clk_per_sm']:.0f} ({d['frac_of_1024'] * 100:.1f}%)" if "flop_per_clk_per_sm" in d
                   else (f"{d['frac_of_best_read'] * 100:.1f}%" if "frac_of_best_read" in d else ""))
            lines.append(f"| {n} | {d['t_iter_us']:.1f} | {d['clock_mhz'] or 0:.0f} | {d['power_w'] or 0:.0f} | {perf} | {eff} |")
    for st, cell in res["stages"].items():
        if not isinstance(cell, dict) or "table" not in cell:
            continue
        t = cell["table"]
        lines.append("")
        lines.append(f"### {st}: {cell['family']} ratio {cell['ratio']} (measured T_A/T_B {t['ratio_measured']:.2f}"
                     + (f", {cell['sub_device_sms']}-SM sub-device" if cell.get("sub_device_sms") else "") + ")")
        lines.append(f"ideal overlap x{t['ideal_overlap_speedup']:.3f}; best intra / best partition = "
                     f"{t['intra_over_partition']:.3f} ({t['best_intra']} vs {t['best_partition']})")
        lines.append("| kind | best variant | t us | x serial | clock MHz | power W | energy mJ/it | t / contention-free pred |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for k, row in t["rows"].items():
            lines.append(f"| {k} | {row['variant']} | {row['t_us']:.1f} | {row['speedup']:.3f} | {row['clock_mhz'] or 0:.0f} | "
                         f"{row['power_w'] or 0:.0f} | {row['energy_mj'] or 0:.1f} | {row.get('t_over_pred', 0):.3f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
STAGES = ["V", "C"] + [f"{f}_{r}" for f in ("reg", "gemm") for r in RATIOS] + ["nocap_reg", "nocap_gemm",
                                                                                 "lowint_reg"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--redo", default="")
    a = ap.parse_args()
    res = load()
    if a.cmd == "report":
        print(report(res))
        return 0
    cb.set_policy(yield_to_caller=True)
    stages = [s for s in a.stages.split(",") if s]
    redo = {s for s in a.redo.split(",") if s}
    for s in redo:
        res["stages"].pop(s, None)
        res["stages"].get("partial", {}).pop(s, None)
    res.setdefault("runs", []).append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "meta": env_meta(), "stages": stages})
    save(res)
    parts = Parts()
    remap = cb.build_sm_remap().to_tensor()
    cur = None
    try:
        for s in stages:
            cur = s
            if s in res["stages"]:
                log(f"{s}: done, skipped")
                continue
            t0 = time.time()
            log(f"=== stage {s}")
            if s == "V":
                stage_V(res, parts)
            elif s == "C":
                stage_C(res, parts)
            elif s.startswith("nocap_"):
                stage_nocap(res, parts, s.split("_")[1], s)
            elif s == "lowint_reg":
                stage_lowint(res, parts, remap, s)
            else:
                fam, r = s.split("_")
                run_cell(res, parts, remap, fam, float(r), stage=s)
            res["stages"][s]["wall_s"] = time.time() - t0
            res["runs"][-1].setdefault("done", []).append((s, round(time.time() - t0, 1)))
            save(res)
            if s in res["stages"] and "table" in res["stages"][s]:
                log("\n" + report({"stages": {s: res["stages"][s]}}))
            gc.collect()
            torch.cuda.empty_cache()
    except cb.GpuYield as e:
        res["runs"][-1]["yield"] = {"stage": cur, "msg": str(e)[:300]}
        save(res)
        log(f"yielding the GPU at stage {cur}: {str(e)[:200]}")
        return YIELD_RC
    except Exception:
        res["runs"][-1]["error"] = {"stage": cur, "tb": traceback.format_exc()[-3000:]}
        save(res)
        raise
    finally:
        parts.close()
    res["runs"][-1]["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
