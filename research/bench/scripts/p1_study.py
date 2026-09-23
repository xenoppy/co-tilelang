"""P1 3x2 study (A2-A4): solo and lib rows x inter-kernel and intra-kernel (CoKernel) columns
for one GEMM x GQA-decode pair.

    source research/env.sh
    python research/bench/scripts/p1_study.py <pair> [--stages I1,I2,I3,C1,C2,C3,C4,F,P,K] [--compile-only]

Inputs: A1 (p1_solo_v1.py) -> <OUT>/solo_v1/summary.json: solo-best configs, C_lib, budget curves.
Pairs: p1_common.PAIRS. Output: <OUT>/<pair>/study.json (one entry per stage, resumable).

Rows. solo = the steady-mode solo-best config of each op. lib = any C_lib config (A1). In the
intra column a GEMM ws='auto' config becomes its ws='off' twin (auto warp specialization cannot
host a second role); the CTA thread count is the max of the roles (idle threads, Rammer-style).

Stages (steady = cobench.bench_steady, primary; flush = clean-flush cobench.bench_variants):
  I1  inter, flush screen: streams (equal / A-high / B-high priority x host order AB / BA),
      green splits (GEMM on n_A SMs [0, n_A), IGNORE_SM_COSCHEDULING) for the solo configs
      and for budget-chosen lib configs (per split: the C_lib config with the lowest solo time
      at that budget, per side), streams for every C_lib x C_lib pair.
  I2  inter, steady search: the I1 set except the lib-streams pairs (top 3 kept).
  I3  inter, steady refinement: green splits +-2/4/6 SMs around the I2 best (solo, lib), the
      2nd-best lib configs per side at the best lib split, order x priority for the best
      lib-streams pair.
  C1  intra, flush screen. solo row: SM binding, dynamic, takeover on/off, 12 splits; chunk
      (1,2), (1,4); static schedule; CTA binding if the CoKernel fits 2 CTAs/SM. lib row:
      every C_lib x C_lib pair x SM/dynamic/takeover x 5 splits, + CTA binding (2 CTAs/SM,
      1:1) where feasible.
  C2  intra, steady confirmation: lib top-8 of the screen + 6 random others + solo top-3;
      screening fidelity (Spearman / Kendall between screen and steady, winner in top-k).
  C3  intra, steady refinement around the lib and solo winners: split +-4/8/12, takeover off,
      chunk variants, static schedule, CTA binding with 188+k CTAs.
  C4  intra, CTA binding in steady mode: the screen's 3 best CTA-binding variants and the
      decode-CTA count sweep (188 + k CTAs, k SMs host a decode CTA next to a GEMM CTA).
  P   idle-but-clocked board power (1-thread sleeping kernel), for the power-bound discussion.
  Winners (`winners`): per cell over I2-C4, times normalized by each stage's serial; a variant is
  in the solo row iff it uses the solo-best configs, whichever search found it.
  F   final table: serial, solo_a, solo_b and the winners of every cell, steady (full
      protocol: 1.5 s slices, 5 rounds) + clean flush; timing builds of the CoKernel winners
      give per-role completion times.
  K   carveout: two-stream co-runs with the kernels' default vs maximum shared-memory
      carveout (NVRTC-backend builds, cuKernelSetAttribute).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch

import p1_common as C
from p1_common import FULL, cb, log
from cotile import catalog, resources
from cotile.cokernel import Orch, build_cokernel, sm_role_table
from cotile.ops import gemm, gqa_decode

GREEN = C.GREEN_A
SCREEN_SPLITS = (64, 94, 124, 148, 172)     # lib-row CoKernel screen (SM binding)
SOLO_SPLITS = GREEN                          # solo-row CoKernel sweep
TOPK, NRAND = 8, 6
LIGHT = dict(slice_s=1.0, settle_s=0.4, rounds=3, warmup_s=5.0, thermal_window_s=10.0)
FULLP = dict(slice_s=1.5, settle_s=0.5, rounds=5, warmup_s=5.0, thermal_window_s=15.0)


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall(x, y) -> float:
    n, c, d = len(x), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            c += s > 0
            d += s < 0
    return float((c - d) / max(1, c + d))


SEARCH_STAGES = ("I2", "I3", "C2", "C3", "C4")


def winners(res: dict) -> dict:
    """Best variant per cell over the steady search stages, as {cell: (t / t_serial of its
    stage, variant, stage)}. Times are normalized by the serial of their own stage (stages run
    at different times). A variant belongs to the solo row iff it uses the solo-best configs
    (the GEMM's ws='off' twin in the intra column), whichever search stage produced it; every
    variant belongs to the lib row (solo-best configs are C_lib members). Extra rows: the best
    two-stream (solo) variant, static-schedule and CTA-binding CoKernels."""
    solo_a, solo_b = res["solo"]["a"], res["solo"]["b"]
    twin_a = res["twin"].get(solo_a, solo_a)
    pool = {}
    for st in SEARCH_STAGES:
        s = res["stages"].get(st)
        if not s or "steady" not in s:
            continue
        ser = s["steady"]["serial"]["t_iter_us"]
        for n, v in s["steady"].items():
            pool.setdefault(n, []).append((v["t_iter_us"] / ser, st, s["desc"][n]))
    best = {}

    def put(cell, x, n, st):
        if cell not in best or x < best[cell][0]:
            best[cell] = (x, n, st)

    for n, lst in pool.items():
        x, st, d = min(lst, key=lambda e: e[0])
        if d["kind"] not in ("co", "green", "streams"):
            continue
        col = "intra" if d["kind"] == "co" else "inter"
        is_solo = d["b"] == solo_b and d["a"] == (twin_a if col == "intra" else solo_a)
        put(f"lib_{col}", x, n, st)
        if is_solo:
            put(f"solo_{col}", x, n, st)
            if d["kind"] == "streams":
                put("solo_streams", x, n, st)
        if d["kind"] == "co" and d["binding"] == "sm_static":
            put("static", x, n, st)
        if d["kind"] == "co" and d["binding"] == "cta_dyn":
            put("cta", x, n, st)
    return best


class Study:
    def __init__(self, pair: str, args):
        self.pair, self.args = pair, args
        self.A, self.B = C.PAIRS[pair]
        summ = C.load(os.path.join(C.SOLO_DIR, "summary.json"))
        if summ is None:
            raise RuntimeError("run p1_solo_v1.py first (A1)")
        sa, sb = summ["shapes"][self.A], summ["shapes"][self.B]
        self.cat = {"a": sa, "b": sb}
        self.solo = {"a": sa["solo_best"], "b": sb["solo_best"]}
        self.lib = {"a": list(sa["c_lib"]), "b": list(sb["c_lib"])}
        self.budget = {"a": sa["budgets"], "b": sb["budgets"]}      # tag -> {str(n): us}
        self.cfg = {"a": {t: C.cfg_from_tag("gemm", self.A, t) for t in self.lib["a"]},
                    "b": {t: C.cfg_from_tag("gqa_decode", self.B, t) for t in self.lib["b"]}}
        self.op = {"a": gemm, "b": gqa_decode}
        self.shape = {"a": C.shape_of("gemm", self.A), "b": C.shape_of("gqa_decode", self.B)}
        # intra-column GEMM configs: ws='off' twins (deduplicated)
        self.twin = {t: gemm.cfg_tag(C.ws_off_twin(gemm, c)) for t, c in self.cfg["a"].items()}
        for t in set(self.twin.values()) - set(self.cfg["a"]):
            self.cfg["a"][t] = C.cfg_from_tag("gemm", self.A, t)
        self.lib_intra_a = list(dict.fromkeys(self.twin[t] for t in self.lib["a"]))
        ids_a = list(dict.fromkeys(self.lib["a"] + self.lib_intra_a))
        self.id = {"a": {t: f"a{i}" for i, t in enumerate(ids_a)}, "b": {t: f"b{i}" for i, t in enumerate(self.lib["b"])}}
        self.path = os.path.join(C.OUT, pair, "study.json")
        self.res = C.load(self.path) or {"pair": pair, "A": self.A, "B": self.B, "stages": {}}
        self.res.update({"solo": self.solo, "lib": self.lib, "lib_intra_a": self.lib_intra_a, "twin": self.twin,
                         "ids": self.id})
        self.grid = {s: {t: self.op[s].build_grid(self.shape[s], c) for t, c in self.cfg[s].items()} for s in "ab"}
        self.co_specs: dict = {}
        self.V: dict = {}
        self.D: dict = {}
        self.parts: dict = {}
        self.streams = {"s1": torch.cuda.Stream(), "s2": torch.cuda.Stream(),
                        "hi1": torch.cuda.Stream(priority=-1), "lo1": torch.cuda.Stream(priority=0),
                        "hi2": torch.cuda.Stream(priority=-1), "lo2": torch.cuda.Stream(priority=0)}
        self.da = self.db = None

    # ------------------------------------------------------------------ bookkeeping
    def save(self):
        C.save(self.path, self.res)

    def stage_done(self, s):
        return s in self.res["stages"] and not self.args.redo

    def put(self, stage, obj):
        obj["wall_s"] = obj.get("wall_s")
        self.res["stages"][stage] = obj
        self.save()

    def name_cfg(self, s, t):
        return self.id[s][t]

    # ------------------------------------------------------------------ kernels
    def co_spec(self, ta, tb, orch: Orch):
        key = (ta, tb, orch)
        if key not in self.co_specs:
            self.co_specs[key] = build_cokernel(gemm, self.shape["a"], self.cfg["a"][ta], gqa_decode, self.shape["b"],
                                                self.cfg["b"][tb], orch)
        return self.co_specs[key]

    @staticmethod
    def orch(kind, to=True, chunk=(1, 1), num_ctas=FULL, timing=False):
        if kind == "sm_dyn":
            return Orch(binding="sm", schedule="dynamic", chunk=tuple(chunk), takeover=to, num_ctas=FULL, timing=timing)
        if kind == "sm_static":
            return Orch(binding="sm", schedule="static", takeover=to, num_ctas=FULL, timing=timing)
        if kind == "cta_dyn":
            return Orch(binding="cta", schedule="dynamic", chunk=tuple(chunk), takeover=to, num_ctas=num_ctas, timing=timing)
        raise ValueError(kind)

    def compile(self, specs):
        st = C.compile_all(list(specs))
        log(f"compiled {len(specs)} specs: {st}")
        self.res["compile_s"] = self.res.get("compile_s", 0.0) + st["wall_s"]
        return st

    def ensure_data(self):
        if self.da is None:
            self.compile([s for d in self.grid.values() for s in d.values()])
            self.da = C.OpData("gemm", self.A)
            self.db = C.OpData("gqa_decode", self.B)
            self.fa = {t: self.da.launcher(s) for t, s in self.grid["a"].items()}
            self.fb = {t: self.db.launcher(s) for t, s in self.grid["b"].items()}
            self.res["rotation"] = {"a": {"copies": self.da.n, "bytes": self.da.bytes_per_copy},
                                    "b": {"copies": self.db.n, "bytes": self.db.bytes_per_copy}}
            sa, sb = self.solo["a"], self.solo["b"]
            fa, fb = self.fa[sa], self.fb[sb]

            def serial(i):
                fa(i)
                fb(i)
            self.add("serial", serial, kind="serial", row="ref", a=sa, b=sb)
            self.add("solo_a", fa, kind="solo", row="ref", a=sa)
            self.add("solo_b", fb, kind="solo", row="ref", b=sb)
            if self.twin[sa] != sa:
                self.add("solo_a_wsoff", self.fa[self.twin[sa]], kind="solo", row="ref", a=self.twin[sa])

    def add(self, name, fn, **desc):
        self.V[name] = fn
        self.D[name] = desc
        return name

    def part(self, n):
        if n not in self.parts:
            p = cb.split_sms(n, ignore_coscheduling=True)
            if p.n_sms != n or p.n_rest != FULL - n:
                raise RuntimeError(f"green split {n}: got {p.n_sms}/{p.n_rest}")
            self.parts[n] = p
        return self.parts[n]

    # ------------------------------------------------------------------ inter variants
    def v_streams(self, ta, tb, prio="eq", order="ab", row="solo"):
        name = f"st_{prio}_{order}" + ("" if row == "solo" else f"_{self.id['a'][ta]}{self.id['b'][tb]}")
        if name in self.V:
            return name
        s = self.streams
        sa, sb = {"eq": (s["s1"], s["s2"]), "pA": (s["hi1"], s["lo2"]), "pB": (s["lo1"], s["hi2"])}[prio]
        order_arg = {"ab": "given", "ba": "reverse", "alt": "alternate"}[order]
        return self.add(name, cb.Par(("a", sa, self.fa[ta]), ("b", sb, self.fb[tb]), order=order_arg),
                        kind="streams", row=row, a=ta, b=tb, prio=prio, order=order)

    def lib_choice(self, side, n, rank=0):
        """C_lib config of `side` with the rank-th lowest solo time at n SMs (nearest measured budget)."""
        best = []
        for t in self.lib[side]:
            b = self.budget[side].get(t, {})
            if not b:
                continue
            m = min(b, key=lambda k: (abs(int(k) - n), int(k)))
            best.append((b[m], t, int(m)))
        best.sort()
        return best[min(rank, len(best) - 1)]

    def v_green(self, n, row="solo", ranks=(0, 0)):
        if row == "solo":
            ta, tb = self.solo["a"], self.solo["b"]
            name = f"gr_s_{n}"
        else:
            ta = self.lib_choice("a", n, ranks[0])[1]
            tb = self.lib_choice("b", FULL - n, ranks[1])[1]
            name = f"gr_l_{n}" + ("" if ranks == (0, 0) else f"_r{ranks[0]}{ranks[1]}")
        if name in self.V:
            return name
        p = self.part(n)
        return self.add(name, cb.Par(("a", p.stream, self.fa[ta]), ("b", p.rest_stream, self.fb[tb])),
                        kind="green", row=row, a=ta, b=tb, n_a=n, ranks=list(ranks))

    # ------------------------------------------------------------------ intra variants
    def v_co(self, ta, tb, kind="sm_dyn", n=None, to=True, chunk=(1, 1), num_ctas=FULL, timing=False, row="lib",
             check=True):
        o = self.orch(kind, to, chunk, num_ctas, timing)
        spec = self.co_spec(ta, tb, o)
        knob = f"n{n}" if kind.startswith("sm") else f"g{num_ctas}"
        name = (f"co_{kind}{'T' if to else ''}_c{chunk[0]}-{chunk[1]}_{self.id['a'][ta]}{self.id['b'][tb]}_{knob}"
                + ("_tm" if timing else ""))
        if name in self.V:
            return name
        if spec.kernel is None:
            self.compile([spec])
        var = C.CoVariant(spec, self.da, self.db, sm_role=sm_role_table(n) if kind.startswith("sm") else None,
                          ratio=(1, 1))
        if check and not C.check_same(self.fa[ta], self.fb[tb], self.da, self.db, var):
            raise RuntimeError(f"{name}: outputs differ from the grid kernels")
        return self.add(name, var, kind="co", row=row, a=ta, b=tb, binding=kind, n_a=n, takeover=to,
                        chunk=list(chunk), num_ctas=num_ctas, timing=timing, spec=spec.name)

    def v_co_many(self, kws: list) -> list:
        """Create several CoKernel variants (v_co keyword dicts); missing kernels are compiled in
        one batch first."""
        specs = [self.co_spec(k["ta"], k["tb"], self.orch(k.get("kind", "sm_dyn"), k.get("to", True),
                                                          tuple(k.get("chunk", (1, 1))), k.get("num_ctas", FULL),
                                                          k.get("timing", False)))
                 for k in kws]
        todo = [s for s in specs if s.kernel is None]
        if todo:
            self.compile(todo)
        return [self.v_co(**k) for k in kws]

    def co_sig(self, ta, tb, kind="sm_dyn", **kw):
        spec = self.co_spec(ta, tb, self.orch(kind, **kw))
        if spec.kernel is None:
            self.compile([spec])
        if "sig" not in spec.extra:
            sig = resources.signature(spec)
            src = spec.kernel.get_kernel_source()
            import re
            offs = sorted({int(x) for x in re.findall(r"buf_dyn_shmem \+ (\d+)\)", src)})
            spec.extra["sig"] = {"regs": sig["regs"], "smem": sig["smem_total"], "threads": sig["threads"],
                                 "ctas_per_sm": sig["ctas_per_sm"], "limit_by": sig.get("limit_by"),
                                 "smem_offsets_mod128": sorted({o % 128 for o in offs})}
        return spec.extra["sig"]

    # ------------------------------------------------------------------ measurement
    def flush(self, names, label, reps=50, group=24):
        """Clean-flush screen in groups of <= group variants + serial (reference of each group)."""
        out = {}
        names = [n for n in names if n != "serial"]
        t0 = time.time()
        for g in range(0, len(names), group):
            sub = names[g:g + group]
            vs = {"serial": self.V["serial"], **{n: self.V[n] for n in sub}}
            r = cb.bench_variants(vs, reference="serial", reps=reps, clock=True, label=f"{self.pair} {label} {g}")
            for n in sub:
                v = r.variants[n]
                out[n] = {"t_us": v["total"]["median"], "p10": v["total"]["p10"], "p90": v["total"]["p90"],
                          "cv": v["total"]["cv"], "speedup": r.derived["speedup"][n],
                          "clock_mhz": (v.get("clock") or {}).get("median"),
                          "ops": {k: o["median"] for k, o in (v.get("ops") or {}).items()}}
            out.setdefault("_serial", []).append(r.variants["serial"]["total"]["median"])
            out.setdefault("_guard", []).append(r.guard)
        out["_wall_s"] = time.time() - t0
        return out

    def steady(self, names, label, proto=LIGHT):
        names = list(dict.fromkeys(["serial"] + [n for n in names if n != "serial"]))
        t0 = time.time()
        r = cb.bench_steady({n: self.V[n] for n in names}, reference="serial", label=f"{self.pair} {label}", **proto)
        log("\n" + str(r))
        out = {}
        for n in names:
            v = r.variants[n]
            out[n] = {k: v.get(k) for k in ("t_iter_us", "cv_slices", "clock_mhz", "power_w", "energy_mj_per_iter",
                                            "kcycles_per_iter", "iter_p10_us", "iter_p90_us", "n_iter")}
            out[n]["speedup"] = r.derived["speedup"][n]["ratio_of_medians"]
            out[n]["paired"] = [r.derived["speedup"][n][k] for k in ("paired_median", "paired_min", "paired_max")]
            if v.get("ops"):
                out[n]["ops"] = {k: o["median_us"] for k, o in v["ops"].items()}
        meta = {"config": r.config, "guard": {"clean": r.guard["clean"], "attempts": r.guard["attempts"]},
                "wall_s": time.time() - t0}
        return out, meta

    def desc(self, names):
        return {n: dict(self.D[n]) for n in dict.fromkeys(["serial"] + list(names))}

    # ------------------------------------------------------------------ stages: inter
    def stage_I1(self):
        sa, sb = self.solo["a"], self.solo["b"]
        names = [self.v_streams(sa, sb, p, o) for p in ("eq", "pA", "pB") for o in ("ab", "ba")]
        names += [self.v_green(n, "solo") for n in GREEN]
        names += [self.v_green(n, "lib") for n in GREEN]
        names += [self.v_streams(ta, tb, "eq", "alt", row="lib") for ta in self.lib["a"] for tb in self.lib["b"]]
        r = self.flush(names, "I1")
        self.put("I1", {"flush": r, "desc": self.desc(names), "wall_s": r["_wall_s"]})

    def stage_I2(self):
        i1 = self.res["stages"]["I1"]["flush"]
        sa, sb = self.solo["a"], self.solo["b"]
        names = ["solo_a", "solo_b"] + [self.v_streams(sa, sb, p, o) for p in ("eq", "pA", "pB") for o in ("ab", "ba")]
        names += [self.v_green(n, "solo") for n in GREEN] + [self.v_green(n, "lib") for n in GREEN]
        stl = sorted((k for k in i1 if k.startswith("st_eq_alt_")), key=lambda k: -i1[k]["speedup"])[:3]
        for k in stl:
            d = self.res["stages"]["I1"]["desc"][k]
            names.append(self.v_streams(d["a"], d["b"], "eq", "ab", row="lib"))
        if "solo_a_wsoff" in self.V:
            names.append("solo_a_wsoff")
        r, meta = self.steady(names, "I2")
        self.put("I2", {"steady": r, "meta": meta, "desc": self.desc(names), "stl_top3": stl, "wall_s": meta["wall_s"]})

    def _best(self, stage, pred):
        """Fastest variant of a steady stage whose descriptor satisfies pred(desc)."""
        s = self.res["stages"][stage]
        c = [(v["t_iter_us"], k) for k, v in s["steady"].items() if pred(s["desc"][k])]
        return min(c)[1] if c else None

    def stage_I3(self):
        i2 = self.res["stages"]["I2"]
        self._rebuild(list(i2["steady"]), "I2")
        gs = self._best("I2", lambda d: d["kind"] == "green" and d["row"] == "solo")
        gl = self._best("I2", lambda d: d["kind"] == "green" and d["row"] == "lib")
        ss = self._best("I2", lambda d: d["kind"] == "streams" and d["row"] == "solo")
        sl = self._best("I2", lambda d: d["kind"] == "streams" and d["row"] == "lib")
        names = [gs, gl, ss]
        for base, row in ((gs, "solo"), (gl, "lib")):
            n0 = self.D[base]["n_a"]
            for dn in (-6, -4, -2, 2, 4, 6):
                n = n0 + dn
                if 2 <= n <= FULL - 2 and n not in GREEN:
                    names.append(self.v_green(n, row))
        n0 = self.D[gl]["n_a"]
        for rk in ((1, 0), (0, 1), (1, 1)):
            names.append(self.v_green(n0, "lib", rk))
        if sl:
            d = self.D[sl]
            names += [self.v_streams(d["a"], d["b"], p, o, row="lib") for p in ("eq", "pA", "pB") for o in ("ab", "ba")]
        names = list(dict.fromkeys(n for n in names if n))
        r, meta = self.steady(names, "I3")
        self.put("I3", {"steady": r, "meta": meta, "desc": self.desc(names), "wall_s": meta["wall_s"]})

    # ------------------------------------------------------------------ stages: intra
    def solo_intra_specs(self):
        ta, tb = self.twin[self.solo["a"]], self.solo["b"]
        out = [(ta, tb, "sm_dyn", dict(to=to, chunk=ch)) for to in (True, False) for ch in ((1, 1),)]
        out += [(ta, tb, "sm_dyn", dict(to=True, chunk=ch)) for ch in ((1, 2), (1, 4))]
        out += [(ta, tb, "sm_static", dict(to=to)) for to in (True, False)]
        return out

    def lib_pairs(self):
        return [(ta, tb) for ta in self.lib_intra_a for tb in self.lib["b"]]

    def precompile(self):
        """Every CoKernel the screen (C1) needs (CPU only)."""
        specs = [self.co_spec(ta, tb, self.orch(k, **kw)) for ta, tb, k, kw in self.solo_intra_specs()]
        specs += [self.co_spec(ta, tb, self.orch("sm_dyn")) for ta, tb in self.lib_pairs()]
        specs += [s for d in self.grid.values() for s in d.values()]
        self.compile(specs)
        feas = []
        for ta, tb in self.lib_pairs() + [(self.twin[self.solo["a"]], self.solo["b"])]:
            if self.co_sig(ta, tb)["ctas_per_sm"] >= 2:
                feas.append((ta, tb))
        cspecs = [self.co_spec(ta, tb, self.orch("cta_dyn", num_ctas=2 * FULL)) for ta, tb in feas]
        if cspecs:
            self.compile(cspecs)
        return feas

    def stage_C1(self):
        feas = self.precompile()
        sigs = {f"{self.id['a'][ta]}{self.id['b'][tb]}": self.co_sig(ta, tb) for ta, tb in self.lib_pairs()}
        ta0, tb0 = self.twin[self.solo["a"]], self.solo["b"]
        sigs["solo"] = self.co_sig(ta0, tb0)
        bad = {k: s for k, s in sigs.items() if any(o % 128 for o in s["smem_offsets_mod128"])}
        if bad:
            log(f"WARNING: role smem buffers not 128-B aligned: {bad}")
        names_solo = []
        for ta, tb, kind, kw in self.solo_intra_specs():
            splits = SOLO_SPLITS if (kind == "sm_dyn" and kw.get("chunk") == (1, 1)) else (94, 124, 148, 172)
            names_solo += self.v_co_many([dict(ta=ta, tb=tb, kind=kind, n=n, row="solo", **kw) for n in splits])
        cta_solo = (ta0, tb0) in feas
        if cta_solo:
            names_solo.append(self.v_co(ta0, tb0, "cta_dyn", num_ctas=2 * FULL, row="solo"))
        names_lib = self.v_co_many([dict(ta=ta, tb=tb, kind="sm_dyn", n=n) for ta, tb in self.lib_pairs()
                                    for n in SCREEN_SPLITS])
        names_lib += self.v_co_many([dict(ta=ta, tb=tb, kind="cta_dyn", num_ctas=2 * FULL) for ta, tb in feas])
        r = self.flush(names_solo + names_lib, "C1")
        self.put("C1", {"flush": r, "desc": self.desc(names_solo + names_lib), "names_solo": names_solo,
                        "names_lib": names_lib, "sigs": sigs, "cta_feasible": [f"{self.id['a'][a]}{self.id['b'][b]}" for a, b in feas],
                        "cta_solo_feasible": cta_solo, "wall_s": r["_wall_s"]})

    def _rebuild(self, names, stage):
        """Re-create variants recorded in an earlier stage (after a restart)."""
        co = []
        for n in names:
            if n in self.V:
                continue
            d = self.res["stages"][stage]["desc"][n]
            if d["kind"] == "co":
                co.append(dict(ta=d["a"], tb=d["b"], kind=d["binding"], n=d["n_a"], to=d["takeover"],
                               chunk=tuple(d["chunk"]), num_ctas=d["num_ctas"], timing=d["timing"], row=d["row"]))
            elif d["kind"] == "green":
                self.v_green(d["n_a"], d["row"], tuple(d["ranks"]))
            elif d["kind"] == "streams":
                self.v_streams(d["a"], d["b"], d["prio"], d["order"], d["row"])
        if co:
            self.v_co_many(co)

    def stage_C2(self):
        c1 = self.res["stages"]["C1"]
        f = c1["flush"]
        # screen score = t / t_serial of the same bench_variants group (drift-normalized)
        lib = sorted(c1["names_lib"], key=lambda n: -f[n]["speedup"])
        top = lib[:TOPK]
        rng = random.Random(f"{self.pair}-C2")
        rand = rng.sample(lib[TOPK:], min(NRAND, len(lib) - TOPK))
        solo = sorted(c1["names_solo"], key=lambda n: -f[n]["speedup"])[:3]
        names = top + rand + solo
        self._rebuild(names, "C1")
        r, meta = self.steady(names, "C2")
        conf = top + rand
        xs = [1.0 / f[n]["speedup"] for n in conf]
        ys = [r[n]["t_iter_us"] for n in conf]
        win = min(conf, key=lambda n: r[n]["t_iter_us"])
        fid = {"n": len(conf), "spearman": spearman(xs, ys), "kendall": kendall(xs, ys),
               "steady_winner": win, "winner_screen_rank": lib.index(win) + 1, "winner_in_topk": win in top,
               "screen_top1": lib[0], "regret_of_screen_top1": r[lib[0]]["t_iter_us"] / r[win]["t_iter_us"] - 1,
               "spearman_topk": spearman(xs[:TOPK], ys[:TOPK]) if TOPK > 2 else None}
        log(f"screening fidelity: {fid}")
        self.put("C2", {"steady": r, "meta": meta, "desc": self.desc(names), "top": top, "rand": rand, "solo": solo,
                        "fidelity": fid, "wall_s": meta["wall_s"]})

    def _refine_kws(self, base, row):
        d = self.D[base]
        ta, tb = d["a"], d["b"]
        kws = []
        if d["binding"] in ("sm_dyn", "sm_static"):
            n0, kind, ch0 = d["n_a"], d["binding"], tuple(d["chunk"])
            for dn in (-12, -8, -4, 4, 8, 12):
                n = n0 + dn
                if 8 <= n <= FULL - 4:
                    kws.append(dict(ta=ta, tb=tb, kind=kind, n=n, to=d["takeover"], chunk=ch0, row=row))
            kws.append(dict(ta=ta, tb=tb, kind=kind, n=n0, to=not d["takeover"], chunk=ch0, row=row))
            for ch in ((1, 1), (1, 2), (1, 4), (2, 1)):
                kws.append(dict(ta=ta, tb=tb, kind="sm_dyn", n=n0, to=True, chunk=ch, row=row))
            for to in (True, False):
                kws.append(dict(ta=ta, tb=tb, kind="sm_static", n=n0, to=to, row=row))
        if self.co_sig(ta, tb)["ctas_per_sm"] >= 2:
            for k in (47, 94, 141, 188):
                kws.append(dict(ta=ta, tb=tb, kind="cta_dyn", num_ctas=FULL + k, row=row))
        return kws

    def stage_C3(self):
        c2 = self.res["stages"]["C2"]["steady"]
        self._rebuild(list(c2), "C2")
        win_lib = min(self.res["stages"]["C2"]["top"] + self.res["stages"]["C2"]["rand"], key=lambda n: c2[n]["t_iter_us"])
        win_solo = min(self.res["stages"]["C2"]["solo"], key=lambda n: c2[n]["t_iter_us"])
        # compile what the refinement needs in one batch
        names = [win_lib, win_solo] + self.v_co_many(self._refine_kws(win_lib, "lib") + self._refine_kws(win_solo, "solo"))
        names = list(dict.fromkeys(names))
        # the lib winner's CTA-binding variant and the runner-up config pair (different pair) as anchors
        r2 = sorted((n for n in c2 if self.D[n]["row"] == "lib"), key=lambda n: c2[n]["t_iter_us"])
        other = next((n for n in r2 if (self.D[n]["a"], self.D[n]["b"]) != (self.D[win_lib]["a"], self.D[win_lib]["b"])), None)
        if other:
            names.append(other)
        r, meta = self.steady(names, "C3")
        self.put("C3", {"steady": r, "meta": meta, "desc": self.desc(names), "win_lib_c2": win_lib,
                        "win_solo_c2": win_solo, "wall_s": meta["wall_s"]})

    def stage_C4(self):
        """intra, CTA binding in steady mode (POD-style: one GEMM and one decode CTA per SM):
        the 3 best CTA-binding variants of the screen, and for the best of them the decode-CTA
        count sweep (188 + k CTAs: every SM hosts a GEMM CTA, k SMs also a decode CTA), with the
        lib-row intra winner so far as anchor."""
        c1 = self.res["stages"]["C1"]
        f = c1["flush"]
        cta = sorted((n for n in c1["names_lib"] + c1["names_solo"] if c1["desc"][n].get("binding") == "cta_dyn"),
                     key=lambda n: -f[n]["speedup"])
        if not cta:
            self.put("C4", {"feasible": False, "wall_s": 0.0})
            return
        top = cta[:3]
        self._rebuild(top, "C1")
        d = self.D[top[0]]
        sweep = self.v_co_many([dict(ta=d["a"], tb=d["b"], kind="cta_dyn", num_ctas=FULL + k, row=d["row"])
                                for k in (47, 94, 141)])
        anchor = None
        for st in ("C3", "C2"):
            if st in self.res["stages"]:
                anchor = self._best(st, lambda x: x["kind"] == "co" and x["row"] == "lib")
                self._rebuild([anchor], st)
                break
        names = list(dict.fromkeys(top + sweep + ([anchor] if anchor else [])))
        r, meta = self.steady(names, "C4")
        self.put("C4", {"steady": r, "meta": meta, "desc": self.desc(names), "top": top, "sweep": sweep,
                        "screen": {n: f[n]["speedup"] for n in cta}, "wall_s": meta["wall_s"]})

    # ------------------------------------------------------------------ final
    def winners(self):
        return winners(self.res)

    def stage_F(self):
        for st in ("I2", "I3", "C2", "C3", "C4"):
            if "steady" in self.res["stages"].get(st, {}):
                self._rebuild(list(self.res["stages"][st]["steady"]), st)
        best = self.winners()
        names = ["solo_a", "solo_b"] + (["solo_a_wsoff"] if "solo_a_wsoff" in self.V else [])
        names += list(dict.fromkeys(v[1] for v in best.values()))
        timing = {}
        for cell in ("solo_intra", "lib_intra"):
            if cell in best:
                d = self.D[best[cell][1]]
                tn = self.v_co(d["a"], d["b"], d["binding"], d["n_a"], d["takeover"], tuple(d["chunk"]), d["num_ctas"],
                               True, d["row"])
                timing[cell] = tn
                names.append(tn)
        names = list(dict.fromkeys(names))
        r, meta = self.steady(names, "F", FULLP)
        roles = {c: self.V[n].role_times() for c, n in timing.items()}
        rf = self.flush(names, "F-flush", reps=60, group=len(names))
        self.put("F", {"steady": r, "meta": meta, "flush": rf, "desc": self.desc(names),
                       "best": {k: list(v) for k, v in best.items()}, "timing": timing, "role_times": roles,
                       "wall_s": meta["wall_s"] + rf["_wall_s"]})

    # ------------------------------------------------------------------ power reference
    def stage_P(self):
        """Board power with the GPU clocked up but its SMs idle: back-to-back launches of a
        1-thread kernel that sleeps (%globaltimer + nanosleep) for ~200 us, interleaved with
        serial / solo runs. Gives the static share P_s of the solo energies behind LB_power."""
        src = r'''
extern "C" __global__ void idle_spin(unsigned long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { __nanosleep(2000); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}'''
        k = cb.CudaKernel(src, "idle_spin", "Q")
        self.add("idle_spin", lambda i: k(1, 1, 200_000), kind="idle", row="ref")
        names = ["solo_a", "solo_b", "idle_spin"]
        r, meta = self.steady(names, "P")
        self.put("P", {"steady": r, "meta": meta, "desc": self.desc(names), "wall_s": meta["wall_s"]})

    # ------------------------------------------------------------------ carveout
    def stage_K(self):
        from p1_carveout import carveout_study
        self.put("K", carveout_study(self))

    def close(self):
        for p in self.parts.values():
            p.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(C.PAIRS))
    ap.add_argument("--stages", default="I1,I2,I3,C1,C2,C3,F")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    st = Study(args.pair, args)
    if args.compile_only:
        feas = st.precompile()
        log(f"{args.pair}: precompiled; CTA-feasible pairs: {len(feas)}")
        return
    C.wait_gpu(6 << 30)
    C.retry_oom(st.ensure_data)
    st.res.setdefault("meta", []).append(C.env_meta())
    for s in [x for x in args.stages.split(",") if x]:
        if st.stage_done(s):
            log(f"{args.pair} {s}: done, skipped")
            continue
        t = time.time()
        log(f"{args.pair} {s} ...")
        C.retry_oom(getattr(st, f"stage_{s}"))
        st.res.setdefault("stage_wall_s", {})[s] = time.time() - t
        st.save()
        log(f"{args.pair} {s}: {time.time() - t:.0f}s")
    st.close()


if __name__ == "__main__":
    sys.exit(main())
