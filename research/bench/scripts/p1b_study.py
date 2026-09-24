"""P1 3x2 study, part B: the derived row (B1/B2) and the no-oracle robustness readout (B3)
for the five GEMM x GQA-decode pairs of part A (research/results/2026-09-23_p1_3x2_B).

    source research/env.sh
    python research/bench/scripts/run_guarded.py -- python p1b_study.py <pair> [--stages DI1,...,DC4,F,R | FQ,R]
    python research/bench/scripts/p1b_study.py <pair> --compile-only

Inputs: part A (p1_common.OUT): solo_v1/summary.json (C_lib, solo-best, budget curves) and
<pair>/study.json (the part-A winners, re-measured here as anchors).
Output: <OUT_B>/<pair>/study.json, one entry per stage (resumable).

Derived space v0 (README "B1"). C_derived(X | partner) = C_lib(X) U contract-conditioned variants:
  (a) every op-library config (cotile/ops/*.configs: 43 GEMM, 36 decode per shape, incl. split-K
      GEMM and split-KV / fewer-heads decode, i.e. every tile granularity) -- the configs a
      role may take under an SM-share contract (SM binding / green partition: any config fits
      1 CTA/SM; which one is best depends on the share);
  (b) register-capped variants (Orch.min_blocks_per_sm = 2 -> __launch_bounds__(threads, 2))
      for the CTA-level co-residence contract (a GEMM CTA and a decode CTA per SM): configs
      whose smem fits the contract (<= 48 KB each) and that fit 2 CTAs/SM only with the cap;
  (c) co-location-specific L2 eviction priorities: decode K/V loads evict_first (kv_l2),
      GEMM A/B loads evict_last (ab_l2) -- no solo value (the decode reads K/V once; a GEMM
      alone keeps its panels anyway), so solo tuning never selects them.
Rows: a variant is "lib" iff both configs are C_lib members (intra: their ws=off twins), no
L2 hint and no register cap; otherwise "derived". lib (B search) = best lib variant found by
this search (same effort as the derived search), reported next to part A's T[lib,.].

Search (per column; flush = clean-flush cobench.bench_variants screen, steady = bench_steady):
  DI1/DC1  flush, factorized: GEMM axis (every GEMM config, decode fixed to part A's lib winner)
           and decode axis (every decode config x {normal, evict_first}, GEMM fixed) x 5 splits
           around part A's optimum (inter: green, intra: SM binding, dynamic, takeover).
           DC1 also screens the CTA-binding register-cap axis.
  DI2/DC2  flush: top-4 GEMM bases x {normal, evict_last} x top-4 decode bases x {normal,
           evict_first} x the 3 best splits of DI1/DC1.
  DI3/DC3  steady confirmation: top-8 + 4 random of DI2/DC2, hint-flipped twins of the top-2,
           part-A anchors (+ the best register-capped CTA variants for DC3).
  DI4/DC4  steady refinement around the derived and the lib (B search) winners: split, and
           for intra takeover / chunk; hint ablations of the derived winner.
  F        final interleaved steady run (full protocol): serial, solo runs, part A's four
           winners, the derived and lib (B search) winners, ablations of the derived winners
           (each derived axis reverted), solo runs of the derived winners' configs; timing
           builds give per-role completion times.
  FQ       reduced protocol (used for the pairs after main, whose full search found only the
           decode K/V hint and < 3% derived gain): the hint confirmed on part A's winners, with
           the hinted winner's split / chunk refinement, one interleaved steady run stored as F.
  R        B3 robustness: green split and CoKernel (SM binding, dynamic, takeover) at the same
           GEMM shares (SPLITS, plus the rule splits and, for the other pairs, main's oracle
           splits), each with its candidate configs for that split (budget-best C_lib pair,
           the column's derived winner, part A's lib winner; a CTA-binding winner is replaced
           by the best SM-binding variant of its row); steady.

GPU sharing (research/rules.md 7, cobench.GuardPolicy): the study never waits in-process
while holding GPU memory. cobench raises GpuYield before a measurement point when the GPU is
occupied and after a point that foreign SM activity contaminated; the study saves (every
flush group is saved as soon as it is measured), exits with YIELD_RC = 75, and
run_guarded.py waits 30 min, re-checks and relaunches it; it resumes at its next point.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import random
import re
import sys
import time

import numpy as np
import torch

import p1_common as C
import p1_study as P
from p1_common import FULL, cb, log
from cotile import resources
from cotile.cokernel import Orch, build_cokernel, sm_role_table
from cotile.kernel import compile_specs
from cotile.ops import gemm, gqa_decode

OUT_B = os.path.join(C.ROOT, "research", "results", "2026-09-23_p1_3x2_B")
YIELD_RC = 75          # exit code of a GPU-sharing yield (run_guarded.py relaunches)
SPLITS = (64, 80, 94, 108, 124, 140, 156, 172)   # B3 GEMM shares: A1 measured both sides' budgets here
LIGHT = P.LIGHT
FULLP = P.FULLP
TOPB, NRAND = 4, 4
HINT_ID = {"normal": "", "evict_first": "F", "evict_last": "L"}


def even(n: int) -> int:
    return int(2 * round(n / 2))


def clip(n: int, lo: int = 8, hi: int = FULL - 8) -> int:
    return max(lo, min(hi, n))


class StudyB(P.Study):
    def __init__(self, pair: str, args):
        super().__init__(pair, args)
        self.A_res = self.res                                  # part A study.json (loaded by Study)
        self.path = os.path.join(OUT_B, pair, "study.json")
        self.res = C.load(self.path) or {"pair": pair, "A": self.A, "B": self.B, "stages": {}}
        self.pool = C.StatePool()
        # --- config universe: every op-library config (+ L2-hint variants on demand)
        ga = gemm.configs(self.shape["a"])
        db = gqa_decode.configs(self.shape["b"])
        self.base = {"a": [gemm.cfg_tag(c) for c in ga], "b": [gqa_decode.cfg_tag(c) for c in db]}
        for c in ga:
            self.cfg["a"].setdefault(gemm.cfg_tag(c), c)
        for c in db:
            self.cfg["b"].setdefault(gqa_decode.cfg_tag(c), c)
        # intra GEMM universe: ws=off twins, deduplicated
        self.intra_a = list(dict.fromkeys(gemm.cfg_tag(C.ws_off_twin(gemm, self.cfg["a"][t])) for t in self.base["a"]))
        for t in self.intra_a:
            self.cfg["a"].setdefault(t, C.cfg_from_tag("gemm", self.A, t))
        # ids: g<i>/d<i> by op-library order (+ hint letter); stable across runs
        self.id = {"a": {}, "b": {}}
        for i, t in enumerate(list(dict.fromkeys(self.base["a"] + self.intra_a))):
            self.id["a"][t] = f"g{i}"
        for i, t in enumerate(self.base["b"]):
            self.id["b"][t] = f"d{i}"
        self.lib_set = {"a": set(self.lib["a"]) | set(self.lib_intra_a), "b": set(self.lib["b"])}
        self.res.update({"solo": self.solo, "lib": self.lib, "lib_intra_a": self.lib_intra_a, "twin": self.twin,
                         "universe": {"a": self.base["a"], "intra_a": self.intra_a, "b": self.base["b"]}})
        self.grid = {"a": {}, "b": {}}
        self.fa, self.fb = {}, {}
        self.failed: dict = {}

    # ------------------------------------------------------------------ configs / ids
    def hinted(self, side: str, tag: str, pol: str) -> str:
        """Tag of `tag`'s config with L2 policy `pol` (registers the config)."""
        c = self.cfg_of(side, tag)
        field = "ab_l2" if side == "a" else "kv_l2"
        c2 = dataclasses.replace(c, **{field: pol})
        t2 = (gemm if side == "a" else gqa_decode).cfg_tag(c2)
        self.cfg[side].setdefault(t2, c2)
        return t2

    def base_of(self, side: str, tag: str) -> str:
        return re.sub(r"_l2e[fl]$", "", tag)

    def hint_of(self, side: str, tag: str) -> str:
        c = self.cfg_of(side, tag)
        return c.ab_l2 if side == "a" else c.kv_l2

    def cfg_of(self, side: str, tag: str):
        if tag not in self.cfg[side]:
            base = self.base_of(side, tag)
            pol = {"_l2ef": "evict_first", "_l2el": "evict_last"}[tag[len(base):]]
            self.hinted(side, base, pol)
        return self.cfg[side][tag]

    def sid(self, side: str, tag: str) -> str:
        base = self.base_of(side, tag)
        if base not in self.id[side]:
            self.id[side][base] = f"{'gd'[side == 'b']}x{len(self.id[side])}"
        return self.id[side][base] + HINT_ID[self.hint_of(side, tag)]

    def row_of(self, ta: str, tb: str, mbps: int = 1) -> str:
        lib = (self.base_of("a", ta) == ta and self.base_of("b", tb) == tb and ta in self.lib_set["a"]
               and tb in self.lib_set["b"] and mbps == 1)
        return "lib" if lib else "derived"

    def axes_of(self, ta: str, tb: str, mbps: int = 1) -> dict:
        ca, cbb = self.cfg_of("a", ta), self.cfg_of("b", tb)
        return {"gemm_nonlib": self.base_of("a", ta) not in self.lib_set["a"], "gemm_splitk": ca.split_k > 1,
                "gemm_hint": ca.ab_l2 != "normal", "decode_nonlib": self.base_of("b", tb) not in self.lib_set["b"],
                "decode_splitkv": cbb.num_split > 1, "decode_hint": cbb.kv_l2 != "normal", "regcap": mbps > 1}

    # ------------------------------------------------------------------ kernels / launchers
    def compile_tolerant(self, specs) -> list:
        """Compile; failures are recorded (self.failed) and returned, not raised."""
        uniq = {}
        for s in specs:
            uniq.setdefault((s.name, repr(s.shape), s.grid, repr(sorted(s.pass_configs.items()))), []).append(s)
        todo = [v[0] for v in uniq.values() if v[0].kernel is None and v[0].compile_error is None]
        if todo:
            t = time.time()
            st = compile_specs(todo, num_workers=32)
            self.res["compile_s"] = self.res.get("compile_s", 0.0) + time.time() - t
            log(f"compiled {len(todo)}: {st}")
        bad = []
        for v in uniq.values():
            for s in v[1:]:
                s.kernel, s.compile_error = v[0].kernel, v[0].compile_error
            if v[0].kernel is None:
                bad.append(v[0])
                self.failed[v[0].name] = v[0].compile_error
        return bad

    def lint(self, spec) -> list:
        """ptxas miscompile guard for kernels with L2 hints (cotile.resources)."""
        if "lint" not in spec.extra:
            spec.extra["lint"] = resources.invalid_memory_descriptors(resources.sass(resources.cubin_bytes(spec.kernel)))
        return spec.extra["lint"]

    def grid_spec(self, side: str, tag: str):
        if tag not in self.grid[side]:
            op = self.op[side]
            self.grid[side][tag] = op.build_grid(self.shape[side], self.cfg_of(side, tag))
        return self.grid[side][tag]

    def launcher(self, side: str, tag: str):
        f = self.fa if side == "a" else self.fb
        if tag not in f:
            spec = self.grid_spec(side, tag)
            if spec.kernel is None:
                if self.compile_tolerant([spec]):
                    raise RuntimeError(f"grid {tag}: {spec.compile_error}")
            if self.lint(spec):
                raise RuntimeError(f"grid {tag}: invalid memory descriptors {spec.extra['lint'][:1]}")
            d = self.da if side == "a" else self.db
            f[tag] = d.launcher(spec, self.pool.view(side, spec))
        return f[tag]

    def ensure_data(self):
        if self.da is not None:
            return
        if getattr(self.args, "universe", True):
            # batch-compile every grid kernel the screens need (the reduced protocol compiles
            # its few kernels lazily)
            specs = [self.grid_spec("a", t) for t in self.base["a"] + self.intra_a] + [self.grid_spec("b", t) for t in self.base["b"]]
            specs += [self.grid_spec("b", self.hinted("b", t, "evict_first")) for t in self.base["b"]]
            bad = self.compile_tolerant(specs)
            if bad:
                log(f"grid compile failures: {[(s.name, s.compile_error) for s in bad][:4]}")
        self.da = C.OpData("gemm", self.A)
        self.db = C.OpData("gqa_decode", self.B)
        self.res["rotation"] = {"a": {"copies": self.da.n, "bytes": self.da.bytes_per_copy},
                                "b": {"copies": self.db.n, "bytes": self.db.bytes_per_copy}}
        sa, sb = self.solo["a"], self.solo["b"]
        fa, fb = self.launcher("a", sa), self.launcher("b", sb)

        def serial(i):
            fa(i)
            fb(i)
        self.add("serial", serial, kind="serial", row="ref", a=sa, b=sb)
        self.add("solo_a", fa, kind="solo", row="ref", a=sa)
        self.add("solo_b", fb, kind="solo", row="ref", b=sb)

    # ------------------------------------------------------------------ variants
    @staticmethod
    def orch(kind, to=True, chunk=(1, 1), num_ctas=FULL, timing=False, mbps=1):
        o = P.Study.orch(kind, to, chunk, num_ctas, timing)
        return dataclasses.replace(o, min_blocks_per_sm=mbps) if mbps != 1 else o

    def co_spec(self, ta, tb, orch):
        key = (ta, tb, orch)
        if key not in self.co_specs:
            self.co_specs[key] = build_cokernel(gemm, self.shape["a"], self.cfg_of("a", ta), gqa_decode, self.shape["b"],
                                                self.cfg_of("b", tb), orch)
        return self.co_specs[key]

    def co_name(self, ta, tb, kind, n, to, chunk, num_ctas, timing, mbps):
        knob = f"n{n}" if kind.startswith("sm") else f"g{num_ctas}"
        return (f"co_{kind}{'T' if to else ''}_c{chunk[0]}-{chunk[1]}_{self.sid('a', ta)}{self.sid('b', tb)}_{knob}"
                + (f"_mb{mbps}" if mbps != 1 else "") + ("_tm" if timing else ""))

    def v_co(self, ta, tb, kind="sm_dyn", n=None, to=True, chunk=(1, 1), num_ctas=FULL, timing=False, row=None,
             check=True, mbps=1):
        chunk = tuple(chunk)
        name = self.co_name(ta, tb, kind, n, to, chunk, num_ctas, timing, mbps)
        if name in self.V:
            return name
        spec = self.co_spec(ta, tb, self.orch(kind, to, chunk, num_ctas, timing, mbps))
        if spec.kernel is None and self.compile_tolerant([spec]):
            return None
        if self.lint(spec):
            self.failed[spec.name] = f"invalid memory descriptors: {spec.extra['lint'][:1]}"
            log(f"SKIP {name}: {self.failed[spec.name]}")
            return None
        var = C.CoVariant(spec, self.da, self.db, sm_role=sm_role_table(n) if kind.startswith("sm") else None,
                          ratio=(1, 1), pool=self.pool)
        if check and not C.check_same(self.launcher("a", ta), self.launcher("b", tb), self.da, self.db, var):
            raise RuntimeError(f"{name}: outputs differ from the grid kernels")
        return self.add(name, var, kind="co", row=row or self.row_of(ta, tb, mbps), a=ta, b=tb, binding=kind, n_a=n,
                        takeover=to, chunk=list(chunk), num_ctas=num_ctas, timing=timing, mbps=mbps, spec=spec.name,
                        axes=self.axes_of(ta, tb, mbps))

    def v_co_many(self, kws: list) -> list:
        self.prep_grids([(k["ta"], k["tb"]) for k in kws])
        specs = [self.co_spec(k["ta"], k["tb"], self.orch(k.get("kind", "sm_dyn"), k.get("to", True),
                                                          tuple(k.get("chunk", (1, 1))), k.get("num_ctas", FULL),
                                                          k.get("timing", False), k.get("mbps", 1))) for k in kws]
        self.compile_tolerant([s for s in specs if s.kernel is None])
        out = [self.v_co(**k) for k in kws]
        return [n for n in out if n]

    def v_green_cfg(self, n, ta, tb, row=None):
        name = f"gr_{self.sid('a', ta)}{self.sid('b', tb)}_{n}"
        if name in self.V:
            return name
        try:
            fa, fb = self.launcher("a", ta), self.launcher("b", tb)
        except RuntimeError as e:
            log(f"SKIP {name}: {e}")
            return None
        p = self.part(n)
        return self.add(name, cb.Par(("a", p.stream, fa), ("b", p.rest_stream, fb)), kind="green",
                        row=row or self.row_of(ta, tb), a=ta, b=tb, n_a=n, axes=self.axes_of(ta, tb))

    def mark(self, v, axis):
        if v:
            a = self.D[v].get("axis", "")
            if axis not in a:
                self.D[v]["axis"] = a + axis
        return v

    def prep_grids(self, pairs):
        """Compile the grid kernels of (ta, tb) pairs in one batch (check_same / green need them)."""
        specs = [self.grid_spec("a", ta) for ta, _ in pairs] + [self.grid_spec("b", tb) for _, tb in pairs]
        self.compile_tolerant([s for s in specs if s.kernel is None])

    def variant_tag(self, side, tag, **changes) -> str | None:
        """Tag of `tag`'s config with fields changed (registered), None if invalid."""
        op = self.op[side]
        c = dataclasses.replace(self.cfg_of(side, tag), **changes)
        try:
            op.validate(self.shape[side], c)
        except ValueError:
            return None
        t = op.cfg_tag(c)
        self.cfg[side].setdefault(t, c)
        return t

    def v_solo(self, side, tag):
        name = f"solo_{self.sid(side, tag)}"
        if name not in self.V:
            self.add(name, self.launcher(side, tag), kind="solo", row="ref", **{side: tag})
        return name

    def rebuild(self, names, stage_desc: dict):
        """Re-create variants from their descriptors (after a restart / from part A)."""
        out = []
        for n in names:
            d = stage_desc[n]
            if n in self.V:
                out.append(n)
                continue
            k = d["kind"]
            if k == "co":
                out.append(self.v_co(d["a"], d["b"], d["binding"], d["n_a"], d["takeover"], tuple(d["chunk"]), d["num_ctas"],
                                     d["timing"], d.get("row"), mbps=d.get("mbps", 1)))
            elif k == "green":
                out.append(self.v_green_cfg(d["n_a"], d["a"], d["b"], d.get("row")))
            elif k == "solo":
                side = "a" if "a" in d else "b"
                out.append(self.v_solo(side, d[side]))
            elif k in ("serial",):
                out.append(n)
        return [x for x in out if x]

    # ------------------------------------------------------------------ part-A anchors
    def A_winner(self, cell: str):
        """(name, desc) of part A's final winner of a cell."""
        F = self.A_res["stages"]["F"]
        best = P.winners(self.A_res)
        if cell not in best:
            return None, None
        n = best[cell][1]
        return n, F["desc"].get(n) or next(s["desc"][n] for s in self.A_res["stages"].values()
                                           if isinstance(s, dict) and n in s.get("desc", {}))

    def A_best_sm(self):
        """Part A's best lib-row SM-binding CoKernel over its steady stages (split anchor)."""
        best = None
        for st in P.SEARCH_STAGES:
            s = self.A_res["stages"].get(st)
            if not s or "steady" not in s or not st.startswith("C"):
                continue
            ser = s["steady"]["serial"]["t_iter_us"]
            for n, v in s["steady"].items():
                d = s["desc"].get(n, {})
                if d.get("kind") == "co" and d.get("binding") == "sm_dyn":
                    x = v["t_iter_us"] / ser
                    if best is None or x < best[0]:
                        best = (x, n, d)
        return best[2]

    def anchors(self, col: str) -> list:
        """Part A's solo and lib winners of a column, rebuilt here."""
        out = []
        for cell in (f"solo_{col}", f"lib_{col}"):
            n, d = self.A_winner(cell)
            if d is None or d["kind"] == "streams":
                continue
            if d["kind"] == "green":
                v = self.v_green_cfg(d["n_a"], d["a"], d["b"], row=f"A_{cell}")
            else:
                v = self.v_co(d["a"], d["b"], d["binding"], d["n_a"], d["takeover"], tuple(d["chunk"]), d["num_ctas"],
                              False, row=f"A_{cell}")
            if v:
                self.D[v]["A_cell"] = cell          # (last cell wins; A_cells keeps all)
                self.D[v].setdefault("A_cells", [])
                if cell not in self.D[v]["A_cells"]:
                    self.D[v]["A_cells"].append(cell)
                out.append(v)
        return out

    # ------------------------------------------------------------------ measurement points
    def flush(self, names, label, reps=50, group=24):
        """Clean-flush screen in groups of <= group variants + serial (as part A). Every group
        is one measurement point: its result is saved at once (res["partial"][label]), so a
        stage interrupted by a GPU-sharing yield resumes at its next group."""
        part = self.res.setdefault("partial", {}).setdefault(label, {})
        out = {"_serial": [], "_guard": []}
        names = [n for n in names if n != "serial"]
        t0 = time.time()
        for g in range(0, len(names), group):
            sub = names[g:g + group]
            rec = part.get(str(g))
            if not (rec and rec["names"] == sub):
                vs = {"serial": self.V["serial"], **{n: self.V[n] for n in sub}}
                r = cb.bench_variants(vs, reference="serial", reps=reps, clock=True, label=f"{self.pair} {label} {g}")
                o = {}
                for n in sub:
                    v = r.variants[n]
                    o[n] = {"t_us": v["total"]["median"], "p10": v["total"]["p10"], "p90": v["total"]["p90"],
                            "cv": v["total"]["cv"], "speedup": r.derived["speedup"][n],
                            "clock_mhz": (v.get("clock") or {}).get("median"),
                            "ops": {k: x["median"] for k, x in (v.get("ops") or {}).items()}}
                rec = {"names": sub, "out": o, "serial": r.variants["serial"]["total"]["median"], "guard": r.guard,
                       "wall_s": None}
                part[str(g)] = rec
                self.save()
            out.update(rec["out"])
            out["_serial"].append(rec["serial"])
            out["_guard"].append(rec["guard"])
        out["_wall_s"] = time.time() - t0
        return out

    def put(self, stage, obj):
        super().put(stage, obj)
        if self.res.get("partial", {}).pop(stage, None) is not None:
            self.save()

    # ------------------------------------------------------------------ scoring helpers
    def flush_scores(self, stage: str) -> dict:
        f = self.res["stages"][stage]["flush"]
        return {k: v["speedup"] for k, v in f.items() if not k.startswith("_")}

    def top_bases(self, stage, side, k=TOPB):
        """Top-k config bases of `side` by their best screen speedup in `stage`."""
        sc, desc = self.flush_scores(stage), self.res["stages"][stage]["desc"]
        best = {}
        for n, x in sc.items():
            d = desc[n]
            if side not in d.get("axis", "ab"):
                continue
            t = self.base_of(side, d[side])
            best[t] = max(best.get(t, 0.0), x)
        return [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]]

    def top_splits(self, stage, k=3):
        sc, desc = self.flush_scores(stage), self.res["stages"][stage]["desc"]
        best = {}
        for n, x in sc.items():
            d = desc[n]
            if d.get("n_a") is None:
                continue
            best[d["n_a"]] = max(best.get(d["n_a"], 0.0), x)
        return [s for s, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]]

    def stage_best(self, stages, pred):
        """(t / t_serial, name, stage) of the best steady variant over `stages` with pred(desc)."""
        best = None
        for st in stages:
            s = self.res["stages"].get(st)
            if not s or "steady" not in s:
                continue
            ser = s["steady"]["serial"]["t_iter_us"]
            for n, v in s["steady"].items():
                d = s["desc"].get(n)
                if d and pred(d):
                    x = v["t_iter_us"] / ser
                    if best is None or x < best[0]:
                        best = (x, n, st)
        return best

    def ensure_rebuilt(self, stages):
        for st in stages:
            s = self.res["stages"].get(st)
            if s and "steady" in s:
                self.rebuild([n for n in s["steady"] if n not in ("serial", "solo_a", "solo_b")], s["desc"])

    # ------------------------------------------------------------------ inter (green)
    def inter_anchor(self):
        _, d = self.A_winner("lib_inter")
        return d

    def stage_DI1(self):
        d0 = self.inter_anchor()
        n0, ta0, tb0 = d0["n_a"], d0["a"], d0["b"]
        splits = sorted({even(clip(n0 + dn)) for dn in (-16, -8, 0, 8, 16)})
        names = []
        for ta in self.base["a"]:
            for n in splits:
                names.append(self.mark(self.v_green_cfg(n, ta, tb0), "a"))
        for tb in self.base["b"]:
            for pol in ("normal", "evict_first"):
                t = self.hinted("b", tb, pol)
                for n in splits:
                    names.append(self.mark(self.v_green_cfg(n, ta0, t), "b"))
        names = list(dict.fromkeys(x for x in names if x))
        r = self.flush(names, "DI1")
        self.put("DI1", {"flush": r, "desc": self.desc(names), "splits": splits, "fixed": {"a": ta0, "b": tb0},
                         "wall_s": r["_wall_s"]})

    def combos(self, stage, col):
        ga, gb = self.top_bases(stage, "a"), self.top_bases(stage, "b")
        splits = self.top_splits(stage)
        out = []
        for ta in ga:
            for pa in ("normal", "evict_last"):
                for tb in gb:
                    for pb in ("normal", "evict_first"):
                        out.append((self.hinted("a", ta, pa), self.hinted("b", tb, pb)))
        return out, splits, ga, gb

    def stage_DI2(self):
        pairs, splits, ga, gb = self.combos("DI1", "inter")
        self.prep_grids(pairs)
        names = [v for ta, tb in pairs for n in splits for v in [self.v_green_cfg(n, ta, tb)] if v]
        r = self.flush(names, "DI2")
        self.put("DI2", {"flush": r, "desc": self.desc(names), "splits": splits, "top_a": ga, "top_b": gb,
                         "wall_s": r["_wall_s"]})

    def flipped(self, name):
        """Hint-flipped twins (decode hint, GEMM hint) of a variant, same knobs."""
        d = self.D[name]
        out = []
        for side, pol in (("b", "evict_first"), ("a", "evict_last")):
            t = d[side]
            t2 = self.hinted(side, self.base_of(side, t), "normal" if self.hint_of(side, t) != "normal" else pol)
            ta, tb = (t2, d["b"]) if side == "a" else (d["a"], t2)
            if d["kind"] == "green":
                out.append(self.v_green_cfg(d["n_a"], ta, tb))
            else:
                out.append(self.v_co(ta, tb, d["binding"], d["n_a"], d["takeover"], tuple(d["chunk"]), d["num_ctas"],
                                     False, mbps=d.get("mbps", 1)))
        return [x for x in out if x]

    def confirm(self, screen_stage, label, extra=()):
        sc = self.flush_scores(screen_stage)
        self.rebuild(list(sc), self.res["stages"][screen_stage]["desc"])
        ranked = sorted(sc, key=lambda n: -sc[n])
        top = ranked[:8]
        rng = random.Random(f"{self.pair}-{label}")
        rand = rng.sample(ranked[8:], min(NRAND, max(0, len(ranked) - 8)))
        flips = [f for n in top[:2] for f in self.flipped(n)]
        names = list(dict.fromkeys(top + rand + flips + list(extra)))
        r, meta = self.steady(names, label)
        conf = top + rand
        xs = [1.0 / sc[n] for n in conf]
        ys = [r[n]["t_iter_us"] for n in conf]
        fid = {"n": len(conf), "spearman": P.spearman(xs, ys), "kendall": P.kendall(xs, ys),
               "winner": min(conf, key=lambda n: r[n]["t_iter_us"])}
        fid["winner_screen_rank"] = ranked.index(fid["winner"]) + 1
        self.put(label, {"steady": r, "meta": meta, "desc": self.desc(names), "top": top, "rand": rand, "flips": flips,
                         "extra": list(extra), "fidelity": fid, "wall_s": meta["wall_s"]})

    def stage_DI3(self):
        self.confirm("DI2", "DI3", extra=self.anchors("inter"))

    def refine_inter(self, name):
        d = self.D[name]
        out = []
        for dn in (-6, -4, -2, 2, 4, 6):
            n = d["n_a"] + dn
            if 8 <= n <= FULL - 8:
                out.append(self.v_green_cfg(n, d["a"], d["b"]))
        return out

    def stage_DI4(self):
        self.ensure_rebuilt(["DI3"])
        wd = self.stage_best(["DI3"], lambda d: d["kind"] == "green")[1]
        wl = self.stage_best(["DI3"], lambda d: d["kind"] == "green" and d["row"] in ("lib", "A_lib_inter", "A_solo_inter"))
        names = [wd] + self.refine_inter(wd) + self.flipped(wd)
        if wl:
            names += [wl[1]] + self.refine_inter(wl[1])
        names = list(dict.fromkeys(x for x in names if x))
        r, meta = self.steady(names, "DI4")
        self.put("DI4", {"steady": r, "meta": meta, "desc": self.desc(names), "wall_s": meta["wall_s"]})

    # ------------------------------------------------------------------ intra (CoKernel)
    def stage_DC1(self):
        d0 = self.A_best_sm()
        n0, ta0, tb0 = d0["n_a"], d0["a"], d0["b"]
        splits = sorted({clip(n0 + dn) for dn in (-16, -8, 0, 8, 16)})
        kwa = [dict(ta=ta, tb=tb0, n=n, kind="sm_dyn", to=True) for ta in self.intra_a for n in splits]
        kwb = [dict(ta=ta0, tb=self.hinted("b", tb, pol), n=n, kind="sm_dyn", to=True) for tb in self.base["b"]
               for pol in ("normal", "evict_first") for n in splits]
        names_sm = [self.mark(v, "a") for v in self.v_co_many(kwa)] + [self.mark(v, "b") for v in self.v_co_many(kwb)]
        names_sm = list(dict.fromkeys(names_sm))
        names_cta, cta_info = self.cta_regcap_candidates()
        r = self.flush(names_sm + names_cta, "DC1")
        self.put("DC1", {"flush": r, "desc": self.desc(names_sm + names_cta), "splits": splits,
                         "fixed": {"a": ta0, "b": tb0, "n": n0}, "names_sm": names_sm, "names_cta": names_cta,
                         "cta_info": cta_info, "wall_s": r["_wall_s"]})

    def cta_regcap_candidates(self):
        """CTA binding (one GEMM + one decode CTA per SM on k SMs) under the co-residence
        contract: both roles <= 48 KB smem. mbps=1 (as in part A) and mbps=2 (register cap:
        __launch_bounds__(threads, 2)); a candidate is kept iff the compiled CoKernel fits
        2 CTAs/SM and mbps=2 is needed for that (otherwise it is part A's space)."""
        # pre-filter on the grid-build footprint model (a CoKernel role's lifetime scope allows
        # the same intra-role reuse, e.g. the GEMM's C staging over its A/B pipeline); the
        # compiled CoKernel's CTAs/SM (below) is authoritative
        lim = 49 * 1024
        ga = [t for t in self.intra_a if gemm.smem_bytes(self.shape["a"], self.cfg_of("a", t), "grid") <= lim]
        # decode: the solo-fastest small configs (A1 rank) + their evict_first twins
        s = C.load(os.path.join(C.SOLO_DIR, "summary.json"))["shapes"][self.B]["t188"]
        gb = [t for t in sorted(self.base["b"], key=lambda t: s.get(t, 1e9))
              if gqa_decode.smem_bytes(self.shape["b"], self.cfg_of("b", t), "grid") <= lim][:3]
        info, kws = {}, []
        specs = []
        for ta in ga:
            for tb in gb:
                for mb in (1, 2):
                    specs.append((ta, tb, mb, self.co_spec(ta, tb, self.orch("cta_dyn", num_ctas=2 * FULL, mbps=mb))))
        self.compile_tolerant([x[3] for x in specs])
        for ta, tb, mb, sp in specs:
            if sp.kernel is None:
                continue
            sig = resources.signature(sp)
            info[f"{self.sid('a', ta)}{self.sid('b', tb)}_mb{mb}"] = {"regs": sig["regs"], "ctas_per_sm": sig["ctas_per_sm"],
                                                                      "threads": sig["threads"], "smem": sig["smem_total"],
                                                                      "local": sig.get("local_bytes")}
        for ta in ga:
            for tb in gb:
                i1 = info.get(f"{self.sid('a', ta)}{self.sid('b', tb)}_mb1", {})
                i2 = info.get(f"{self.sid('a', ta)}{self.sid('b', tb)}_mb2", {})
                if i2.get("ctas_per_sm", 0) >= 2 and i1.get("ctas_per_sm", 0) < 2:
                    for k in (94, 188):
                        for pol in ("normal", "evict_first"):
                            kws.append(dict(ta=ta, tb=self.hinted("b", tb, pol), kind="cta_dyn", num_ctas=FULL + k, mbps=2))
        names = self.v_co_many(kws)
        for v in names:
            self.D[v]["axis"] = "cta"
        return names, info

    def stage_DC2(self):
        pairs, splits, ga, gb = self.combos("DC1", "intra")
        kws = [dict(ta=ta, tb=tb, n=n, kind="sm_dyn", to=True) for ta, tb in pairs for n in splits]
        names = self.v_co_many(kws)
        r = self.flush(names, "DC2")
        self.put("DC2", {"flush": r, "desc": self.desc(names), "splits": splits, "top_a": ga, "top_b": gb,
                         "wall_s": r["_wall_s"]})

    def stage_DC3(self):
        c1 = self.res["stages"]["DC1"]
        sc = {n: c1["flush"][n]["speedup"] for n in c1["names_cta"] if n in c1["flush"]}
        cta = sorted(sc, key=lambda n: -sc[n])[:2]
        self.rebuild(cta, c1["desc"])
        self.confirm("DC2", "DC3", extra=cta + self.anchors("intra"))

    def refine_intra(self, name, full=True):
        d = self.D[name]
        out = []
        if d["binding"] != "sm_dyn":
            for k in (47, 94, 141, 188):
                out.append(self.v_co(d["a"], d["b"], "cta_dyn", None, d["takeover"], tuple(d["chunk"]), FULL + k, False,
                                     mbps=d.get("mbps", 1)))
            return out
        n0 = d["n_a"]
        for dn in (-12, -8, -4, 4, 8, 12):
            n = n0 + dn
            if 8 <= n <= FULL - 4:
                out.append(self.v_co(d["a"], d["b"], "sm_dyn", n, d["takeover"], tuple(d["chunk"]), FULL, False))
        if full:
            out.append(self.v_co(d["a"], d["b"], "sm_dyn", n0, not d["takeover"], tuple(d["chunk"]), FULL, False))
            for ch in ((1, 2), (1, 4), (2, 1)):
                out.append(self.v_co(d["a"], d["b"], "sm_dyn", n0, True, ch, FULL, False))
        return out

    def stage_DC4(self):
        self.ensure_rebuilt(["DC3"])
        wd = self.stage_best(["DC3"], lambda d: d["kind"] == "co")[1]
        wl = self.stage_best(["DC3"], lambda d: d["kind"] == "co" and d["row"] in ("lib", "A_lib_intra", "A_solo_intra"))
        names = [wd] + self.refine_intra(wd) + self.flipped(wd)
        if wl:
            names += [wl[1]] + self.refine_intra(wl[1], full=False)
        names = list(dict.fromkeys(x for x in names if x))
        r, meta = self.steady(names, "DC4")
        self.put("DC4", {"steady": r, "meta": meta, "desc": self.desc(names), "wall_s": meta["wall_s"]})

    # ------------------------------------------------------------------ winners, F
    def ranked(self, stages, pred, k) -> list:
        """Up to k distinct variants of `stages` with pred(desc), best first by t / t_serial of
        their own stage (best stage per variant)."""
        best = {}
        for st in stages:
            s = self.res["stages"].get(st)
            if not s or "steady" not in s:
                continue
            ser = s["steady"]["serial"]["t_iter_us"]
            for n, v in s["steady"].items():
                d = s["desc"].get(n)
                if d and pred(d) and not d.get("timing"):
                    x = v["t_iter_us"] / ser
                    if n not in best or x < best[n][0]:
                        best[n] = (x, n, st)
        return sorted(best.values())[:k]

    def winners_b(self) -> dict:
        """{cell: (t/t_serial, name, stage)} over the B steady stages. derived_* = best variant
        of any row (derived space includes lib); lib_B_* = best lib-row variant (incl. part A's
        anchors, re-measured)."""
        out = {}
        for col, kind, stages in (("inter", "green", ["DI3", "DI4"]), ("intra", "co", ["DC3", "DC4"])):
            self.ensure_rebuilt(stages)
            b = self.stage_best(stages, lambda d, k=kind: d["kind"] == k)
            if b:
                out[f"derived_{col}"] = b
            b = self.stage_best(stages, lambda d, k=kind: d["kind"] == k and (d["row"] == "lib" or d["row"].startswith("A_lib")
                                                                              or d["row"].startswith("A_solo")))
            if b:
                out[f"lib_B_{col}"] = b
        return out

    def ablations(self, name) -> dict:
        """Revert each derived axis of a winner (all else fixed) -> {axis: variant}."""
        d = self.D[name]
        ta, tb = d["a"], d["b"]
        ax = d.get("axes") or self.axes_of(ta, tb, d.get("mbps", 1))
        alts = {}
        mk = (lambda a, b, **kw: self.v_green_cfg(d["n_a"], a, b)) if d["kind"] == "green" else \
             (lambda a, b, mbps=d.get("mbps", 1): self.v_co(a, b, d["binding"], d["n_a"], d["takeover"], tuple(d["chunk"]),
                                                            d["num_ctas"], False, mbps=mbps))
        if ax["decode_hint"]:
            alts["decode_hint"] = mk(ta, self.base_of("b", tb))
        if ax["gemm_hint"]:
            alts["gemm_hint"] = mk(self.base_of("a", ta), tb)
        if ax["decode_hint"] and ax["gemm_hint"]:
            alts["both_hints"] = mk(self.base_of("a", ta), self.base_of("b", tb))
        if ax["decode_splitkv"]:
            # tile granularity only: the same decode tile without split-KV
            t1 = self.variant_tag("b", tb, num_split=1)
            if t1:
                alts["decode_split"] = mk(ta, t1)
        if ax["gemm_splitk"]:
            t1 = self.variant_tag("a", ta, split_k=1)
            if t1:
                alts["gemm_split"] = mk(t1, tb)
        if ax["decode_nonlib"]:
            # the partner-agnostic choice for this side: the lib winner's decode config of this column
            ref = self.lib_ref_cfg(d["kind"], "b")
            alts["decode_cfg"] = mk(ta, self.hinted("b", ref, self.hint_of("b", tb)))
        if ax["gemm_nonlib"]:
            ref = self.lib_ref_cfg(d["kind"], "a")
            alts["gemm_cfg"] = mk(self.hinted("a", ref, self.hint_of("a", ta)), tb)
        if ax["regcap"] and d["kind"] == "co":
            alts["regcap"] = mk(ta, tb, mbps=1)
        return {k: v for k, v in alts.items() if v}

    def lib_ref_cfg(self, kind, side):
        """C_lib config used as the 'not derived' reference for one side: the part-A lib
        winner's config of the column (intra: ws=off twin)."""
        col = "inter" if kind == "green" else "intra"
        _, d = self.A_winner(f"lib_{col}")
        return d[side]

    def stage_F(self):
        best = self.winners_b()
        names = ["solo_a", "solo_b"] + self.anchors("inter") + self.anchors("intra")
        names += [v[1] for v in best.values()]
        # cross-stage drift (thermal state) moves power-capped variants by ~1.5% between steady
        # stages, so F re-measures the top-3 derived and top-2 lib candidates of each column
        # together; T[derived,c] / T[lib,c] are then taken over F (p1b_report.py)
        cands = {}
        for col, kind, stages in (("inter", "green", ["DI3", "DI4"]), ("intra", "co", ["DC3", "DC4"])):
            cands[f"derived_{col}"] = [n for _, n, _ in self.ranked(stages, lambda d, k=kind: d["kind"] == k, 3)]
            cands[f"lib_B_{col}"] = [n for _, n, _ in self.ranked(
                stages, lambda d, k=kind: d["kind"] == k and (d["row"] == "lib" or d["row"].startswith("A_")), 2)]
            names += cands[f"derived_{col}"] + cands[f"lib_B_{col}"]
        abl = {}
        for cell in ("derived_inter", "derived_intra"):
            if cell in best:
                a = self.ablations(best[cell][1])
                abl[cell] = a
                names += list(a.values())
        solo = {}
        for cell in ("derived_inter", "derived_intra"):
            if cell in best:
                d = self.D[best[cell][1]]
                for side in "ab":
                    t = d[side]
                    solo[f"{cell}_{side}"] = self.v_solo(side, t)
                    if self.base_of(side, t) != t:
                        solo[f"{cell}_{side}_unhinted"] = self.v_solo(side, self.base_of(side, t))
        names += list(solo.values())
        timing = {}
        for cell in ("derived_intra", "lib_B_intra"):
            if cell in best:
                d = self.D[best[cell][1]]
                if d["binding"] == "sm_dyn" or d["binding"] == "cta_dyn":
                    tn = self.v_co(d["a"], d["b"], d["binding"], d["n_a"], d["takeover"], tuple(d["chunk"]), d["num_ctas"],
                                   True, mbps=d.get("mbps", 1))
                    if tn:
                        timing[cell] = tn
                        names.append(tn)
        a = abl.get("derived_intra", {})
        for ax in ("decode_hint", "both_hints"):
            if ax in a:
                d = self.D[a[ax]]
                tn = self.v_co(d["a"], d["b"], d["binding"], d["n_a"], d["takeover"], tuple(d["chunk"]), d["num_ctas"], True,
                               mbps=d.get("mbps", 1))
                if tn:
                    timing[f"abl_{ax}"] = tn
                    names.append(tn)
                break
        names = list(dict.fromkeys(x for x in names if x))
        r, meta = self.steady(names, "F", FULLP)
        roles = {c: self.V[n].role_times() for c, n in timing.items()}
        self.put("F", {"steady": r, "meta": meta, "desc": self.desc(names), "best": {k: list(v) for k, v in best.items()},
                       "candidates": cands, "ablations": abl, "solo": solo, "timing": timing, "role_times": roles, "wall_s": meta["wall_s"]})

    # ------------------------------------------------------------------ reduced protocol
    def like(self, name, ta=None, tb=None, n=None, num_ctas=None, timing=False):
        """Variant with `name`'s knobs and some configs / split replaced."""
        d = self.D[name]
        ta, tb = ta or d["a"], tb or d["b"]
        if d["kind"] == "green":
            return self.v_green_cfg(n or d["n_a"], ta, tb)
        return self.v_co(ta, tb, d["binding"], n or d["n_a"], d["takeover"], tuple(d["chunk"]),
                         num_ctas or d["num_ctas"], timing, mbps=d.get("mbps", 1))

    def stage_FQ(self):
        """Reduced protocol for the pairs after main (main's full search found no derived axis
        beyond the decode K/V evict_first hint, and < 3% derived gain; README): confirmation of
        that axis on part A's winners, both columns, in ONE interleaved steady run (full
        protocol), stored as stage F with F's schema (so R and p1b_report work unchanged).
        Per column: part A's solo and lib winners, each with and without the decode hint; the
        lib winner with the hint at the neighbouring splits (+-8 SMs, or +-47 decode CTAs for
        CTA binding) and with GEMM evict_last added; for intra also part A's best SM-binding
        variant +- hint when the lib winner uses CTA binding. Timing builds of the intra lib
        winner +- hint give role completion times."""
        names = ["solo_a", "solo_b"]
        pairs = {"inter": [], "intra": []}
        extra = []
        for col in ("inter", "intra"):
            anc = self.anchors(col)
            names += anc
            for n in anc:
                h = self.like(n, tb=self.hinted("b", self.D[n]["b"], "evict_first"))
                if h:
                    pairs[col].append((h, n))
                    names.append(h)
            nl = next((n for n in anc if self.D[n].get("A_cell") == f"lib_{col}"), None)
            if nl is None:
                continue
            d = self.D[nl]
            tbh = self.hinted("b", d["b"], "evict_first")
            if d["kind"] == "green" or d["binding"] == "sm_dyn":
                # the hinted winner gets the split refinement part A gave the lib winner (C3/I3:
                # +-4/8/12 intra, +-2..6 green); unhinted twins at +-8 as the same-run reference
                steps = (-12, -8, -4, 4, 8, 12) if d["kind"] == "co" else (-8, -6, -4, -2, 2, 4, 6, 8)
                for dn in steps:
                    if 8 <= d["n_a"] + dn <= FULL - 8:
                        extra.append(self.like(nl, tb=tbh, n=d["n_a"] + dn))
                        if abs(dn) == 8:
                            extra.append(self.like(nl, n=d["n_a"] + dn))
                if d["kind"] == "co":
                    for ch in ((1, 1), (1, 2), (1, 4)):
                        if tuple(d["chunk"]) != ch:
                            extra.append(self.v_co(d["a"], tbh, d["binding"], d["n_a"], d["takeover"], ch, d["num_ctas"],
                                                   False))
            else:
                for dk in (-47, 47):
                    if FULL < d["num_ctas"] + dk <= 2 * FULL:
                        extra += [self.like(nl, tb=tb_, num_ctas=d["num_ctas"] + dk) for tb_ in (tbh, d["b"])]
            extra.append(self.like(nl, ta=self.hinted("a", d["a"], "evict_last"), tb=tbh))
            if col == "intra" and d["binding"] != "sm_dyn":
                dsm = self.A_best_sm()
                v = self.v_co(dsm["a"], dsm["b"], "sm_dyn", dsm["n_a"], dsm["takeover"], tuple(dsm["chunk"]), FULL,
                              False, row="A_best_sm")
                vh = self.like(v, tb=self.hinted("b", dsm["b"], "evict_first")) if v else None
                if v and vh:
                    pairs[col].append((vh, v))
                    names += [v, vh]
        names += [x for x in extra if x]
        solo = {}
        tbh = self.hinted("b", self.solo["b"], "evict_first")
        solo["solo_b_hinted"] = self.v_solo("b", tbh)
        names.append(solo["solo_b_hinted"])
        timing = {}
        _, dl = self.A_winner("lib_intra")
        if dl:
            for cell, tb in (("lib_intra", dl["b"]), ("lib_hint_intra", self.hinted("b", dl["b"], "evict_first"))):
                tn = self.v_co(dl["a"], tb, dl["binding"], dl["n_a"], dl["takeover"], tuple(dl["chunk"]), dl["num_ctas"],
                               True)
                if tn:
                    timing[cell] = tn
                    names.append(tn)
        names = list(dict.fromkeys(x for x in names if x))
        r, meta = self.steady(names, "F", FULLP)
        roles = {c: self.V[n].role_times() for c, n in timing.items()}
        ser = r["serial"]["t_iter_us"]
        best, abl = {}, {}
        for col, kind in (("inter", "green"), ("intra", "co")):
            cand = [n for n in names if self.D[n]["kind"] == kind and not self.D[n].get("timing")]
            hinted = [n for n in cand if self.D[n].get("axes", {}).get("decode_hint")]
            plain = [n for n in cand if n not in hinted and not self.D[n].get("axes", {}).get("gemm_hint")]
            if hinted:
                w = min(hinted, key=lambda n: r[n]["t_iter_us"])
                best[f"derived_{col}"] = [r[w]["t_iter_us"] / ser, w, "F"]
                tw = self.like(w, tb=self.base_of("b", self.D[w]["b"]))
                if tw in r:
                    abl[f"derived_{col}"] = {"decode_hint": tw}
            if plain:
                w = min(plain, key=lambda n: r[n]["t_iter_us"])
                best[f"lib_B_{col}"] = [r[w]["t_iter_us"] / ser, w, "F"]
        self.put("F", {"steady": r, "meta": meta, "desc": self.desc(names), "best": best, "ablations": abl,
                       "solo": solo, "timing": timing, "role_times": roles, "protocol": "reduced (FQ)",
                       "hint_pairs": pairs, "wall_s": meta["wall_s"]})

    def stage_done(self, s):
        # FQ writes stage F
        return super().stage_done("F" if s == "FQ" else s)

    # ------------------------------------------------------------------ B3 robustness
    def rule_splits(self) -> dict:
        """A-priori splits from solo information only.
        R1 (proportional): n_A = 188 * t_A / (t_A + t_B), full-GPU steady solo times (part A F).
        R2 (equal finish): argmin_n max(t_A(n), t_B(188 - n)), t_X(n) = best C_lib config's solo
            time at n SMs (A1 budget curves, clean flush), linear interpolation between budgets."""
        F = self.A_res["stages"]["F"]["steady"]
        ta, tb = F["solo_a"]["t_iter_us"], F["solo_b"]["t_iter_us"]
        r1 = even(FULL * ta / (ta + tb))

        def curve(side):
            pts = {}
            for t in self.lib[side]:
                for k, v in self.budget[side].get(t, {}).items():
                    pts[int(k)] = min(pts.get(int(k), 1e18), v)
            xs = sorted(pts)
            return xs, [pts[x] for x in xs]
        xa, ya = curve("a")
        xb, yb = curve("b")
        best = None
        for n in range(max(xa[0], FULL - xb[-1]), min(xa[-1], FULL - xb[0]) + 1, 2):
            pa = float(np.interp(n, xa, ya))
            pb = float(np.interp(FULL - n, xb, yb))
            if best is None or max(pa, pb) < best[0]:
                best = (max(pa, pb), n, pa, pb)
        return {"R1": clip(r1), "R2": clip(best[1]), "R1_inputs": {"t_A": ta, "t_B": tb},
                "R2_pred": {"t_A": best[2], "t_B": best[3]}}

    def stage_R(self):
        rules = self.rule_splits()
        splits = set(SPLITS) | {rules["R1"], rules["R2"]}
        transfer = {}
        if self.pair != "main":
            m = C.load(os.path.join(OUT_B, "main", "study.json"))
            if not m or "R" not in m["stages"]:
                raise RuntimeError("run main's stage R first (cross-pair transfer)")
            transfer = m["stages"]["R"]["summary"]["oracle_split"]
            splits |= set(transfer.values())
        splits = sorted(splits)
        # candidate configs per mechanism
        self.ensure_rebuilt(["F"])
        Fb = self.res["stages"]["F"]["best"]
        dI = self.D[Fb["derived_inter"][1]] if "derived_inter" in Fb else None
        dC = self.D[Fb["derived_intra"][1]] if "derived_intra" in Fb else None
        _, aI = self.A_winner("lib_inter")
        _, aC = self.A_winner("lib_intra")
        # The CoKernel mechanism of B3 is SM binding. A column winner that uses CTA binding
        # (e.g. part A's lib-intra winner of `second`) has no SM split, so the best SM-binding
        # variant of the same row takes its place as the config source: for the derived winner
        # the best decode-hinted SM-binding CoKernel of F, for part A's lib winner part A's best
        # SM-binding CoKernel (A_best_sm).
        subst = {}
        if dC and dC["binding"] != "sm_dyn":
            FS, FD = self.res["stages"]["F"]["steady"], self.res["stages"]["F"]["desc"]
            sm = [n for n, d in FD.items() if n in FS and d.get("kind") == "co" and d.get("binding") == "sm_dyn"
                  and not d.get("timing") and (d.get("axes") or {}).get("decode_hint")]
            w = min(sm, key=lambda n: FS[n]["t_iter_us"]) if sm else None
            subst["derived"] = {"winner": Fb["derived_intra"][1], "sm_source": w}
            dC = self.D[w] if w else None
        if aC and aC["binding"] != "sm_dyn":
            aC = self.A_best_sm()
            subst["A_lib"] = {"winner_binding": "cta_dyn", "sm_source": {k: aC.get(k) for k in ("a", "b", "n_a", "chunk")}}
        names, cand = [], {}
        for n in splits:
            gl = []
            ga0, gb0 = self.lib_choice("a", n)[1], self.lib_choice("b", FULL - n)[1]
            gl.append(("budget", ga0, gb0))
            if dI:
                gl.append(("derived", dI["a"], dI["b"]))
            if aI:
                gl.append(("A_lib", aI["a"], aI["b"]))
            cl = [("budget", self.twin.get(ga0, gemm.cfg_tag(C.ws_off_twin(gemm, self.cfg_of("a", ga0)))), gb0, (1, 1))]
            if dC and dC["binding"] == "sm_dyn":
                cl.append(("derived", dC["a"], dC["b"], tuple(dC["chunk"])))
            if aC and aC["binding"] == "sm_dyn":
                cl.append(("A_lib", aC["a"], aC["b"], tuple(aC["chunk"])))
            seen = set()
            for src, ta, tb in gl:
                v = self.v_green_cfg(n, ta, tb)
                if v and v not in seen:
                    seen.add(v)
                    cand.setdefault(v, {"mech": "green", "n": n, "src": []})["src"].append(src)
            kws = [dict(ta=ta, tb=tb, n=n, kind="sm_dyn", to=True, chunk=ch) for _, ta, tb, ch in cl]
            for (src, *_), v in zip(cl, [self.v_co(**k) for k in kws]):
                if v:
                    cand.setdefault(v, {"mech": "co", "n": n, "src": []})["src"].append(src)
        names = list(cand)
        r, meta = self.steady(names, "R")
        summ = self.robustness_summary(r, cand, rules, transfer, splits)
        self.put("R", {"steady": r, "meta": meta, "desc": self.desc(names), "cand": cand, "rules": rules,
                       "transfer_from_main": transfer, "splits": splits, "summary": summ, "cta_substitutes": subst,
                       "wall_s": meta["wall_s"]})

    @staticmethod
    def robustness_summary(r, cand, rules, transfer, splits) -> dict:
        out = {"per_split": {}, "oracle_split": {}}
        for mech in ("green", "co"):
            per = {}
            for v, c in cand.items():
                if c["mech"] != mech:
                    continue
                s = r[v]["speedup"]
                if c["n"] not in per or s > per[c["n"]][0]:
                    per[c["n"]] = (s, v, c["src"])
                if "budget" in c["src"]:
                    per.setdefault(("budget", c["n"]), (s, v))
            curve = {n: per[n][0] for n in splits if n in per}
            orc = max(curve, key=curve.get)
            sweep = [curve[n] for n in SPLITS if n in curve]
            m = {"curve": {str(k): v for k, v in curve.items()}, "best_cfg": {str(n): per[n][1] for n in splits if n in per},
                 "budget_only": {str(k[1]): v[0] for k, v in per.items() if isinstance(k, tuple)},
                 "oracle": curve[orc], "oracle_split": orc, "worst_sweep": min(sweep), "mean_sweep": float(np.mean(sweep)),
                 "R1": curve.get(rules["R1"]), "R2": curve.get(rules["R2"])}
            m["regret_R1"] = m["oracle"] / m["R1"] if m["R1"] else None
            m["regret_R2"] = m["oracle"] / m["R2"] if m["R2"] else None
            m["regret_worst"] = m["oracle"] / m["worst_sweep"]
            if transfer:
                t = transfer.get(mech)
                m["transfer_split"] = t
                m["transfer"] = curve.get(t)
                m["regret_transfer"] = m["oracle"] / curve[t] if curve.get(t) else None
            out[mech] = m
            out["oracle_split"][mech] = orc
        return out

    def compile_only(self):
        """CPU: every grid kernel and the DI1/DC1 screen CoKernels."""
        specs = [self.grid_spec("a", t) for t in self.base["a"] + self.intra_a] + [self.grid_spec("b", t) for t in self.base["b"]]
        specs += [self.grid_spec("b", self.hinted("b", t, "evict_first")) for t in self.base["b"]]
        d0 = self.A_best_sm()
        n0, ta0, tb0 = d0["n_a"], d0["a"], d0["b"]
        o = self.orch("sm_dyn", True)
        specs += [self.co_spec(ta, tb0, o) for ta in self.intra_a]
        specs += [self.co_spec(ta0, self.hinted("b", tb, pol), o) for tb in self.base["b"] for pol in ("normal", "evict_first")]
        bad = self.compile_tolerant(specs)
        lint = [s.name for s in specs if s.kernel is not None and self.lint(s)]
        log(f"{self.pair}: compiled {len(specs)}; failures {len(bad)}; invalid-descriptor kernels {len(lint)}")
        for s in bad[:10]:
            log(f"  FAIL {s.name}: {s.compile_error}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(C.PAIRS))
    ap.add_argument("--stages", default="DI1,DI2,DI3,DI4,DC1,DC2,DC3,DC4,F,R")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    args.universe = any(x in args.stages.split(",") for x in ("DI1", "DI2", "DC1", "DC2"))
    st = StudyB(args.pair, args)
    if args.compile_only:
        st.compile_only()
        return
    # GPU-sharing policy (research/rules.md 7, cobench.GuardPolicy): never wait in-process
    # while holding GPU memory -- raise GpuYield between measurement points (and after a point
    # that foreign SM activity contaminated), save, exit YIELD_RC; run_guarded.py waits
    # 30 min and relaunches, and the study resumes at its next point.
    cb.set_policy(yield_to_caller=True)
    s, t = None, time.time()
    try:
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
            st.res.setdefault("stage_runs", []).append({"stage": s, "wall_s": time.time() - t, "done": True})
            st.res["failed"] = st.failed
            st.res["pool_bytes"] = st.pool.nbytes()
            st.save()
            log(f"{args.pair} {s}: {time.time() - t:.0f}s (pool {st.pool.nbytes() / 2**20:.0f} MiB)")
    except cb.GpuYield as e:
        st.res.setdefault("stage_runs", []).append({"stage": s, "wall_s": time.time() - t, "done": False,
                                                    "yield": str(e)[:300]})
        st.res.setdefault("yields", []).append({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": s,
                                                "detail": str(e)[:300]})
        st.save()
        log(f"{args.pair}: yielding the GPU at stage {s}: {str(e)[:200]}")
        return YIELD_RC
    finally:
        st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
