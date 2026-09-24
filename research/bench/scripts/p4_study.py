"""P4-b: same-condition comparison of CoKernel vs FlashInfer POD vs inter-kernel co-location on
P4 = causal prefill attention (one request, Hq 32 / Hkv 8 / D 128) x GQA decode
(research/results/2026-09-24_p4_study).

    source research/env.sh
    G="python research/bench/scripts/run_guarded.py --log research/results/2026-09-24_p4_study/guard_log.jsonl --"
    $G python research/bench/scripts/p4_study.py P2048_B16_S2048 --stages SA,SB,SC,SD,F,R     # primary pairs
    $G python research/bench/scripts/p4_study.py P8192_B16_S2048 --stages R,FQ               # secondary pairs
    $G python research/bench/scripts/p4_study.py P2048_B16_S2048 --stages AT                 # attribution (primary)
    python research/bench/scripts/p4_study.py <pair> --compile-only                          # CPU only

Inputs (P4-a, research/results/2026-09-24_p4_prep): solo/summary.json (solo-best, C_lib, budget
curves per shape), pairs_steady.json (FlashInfer decode path per pair, solo times for the rules).
Output: <OUT>/<pair>/study.json, one entry per stage (resumable; every flush group is saved as
soon as it is measured).

Roles: A = prefill (tensor-core bound), B = decode (DRAM bound). Green splits give the prefill
SMs [0, n) and the decode the other 188 - n; SM-binding CoKernels use the same %smid ranges.

Rows (the 3x2 design, plan §2.5):
  solo     both configs are the steady-mode solo-best of P4-a;
  lib      both configs are C_lib members of P4-a (plain library configs: LPT order, no hint,
           no register cap);
  derived  C_derived = C_lib U contract-conditioned variants: every op-library config (all tile
           sizes, split-KV decode, 256-thread tiles), decode K/V L2 evict_first (kv_l2), the
           prefill tile order (natural = shortest first, FlashInfer's order, vs LPT), register-
           capped CTA co-residence (min_blocks_per_sm = 2), FlashInfer-like tiles.
  T[row, col] = best variant of column col (inter: streams, green; intra: CoKernel) whose
  configs are in the row's set (solo c lib c derived).

Mechanisms:
  inter  serial (reference), two streams (priority eq / prefill-high / decode-high x host order),
         green-context split (per split the configs chosen by the row's rule), both with and
         without the decode hint.
  intra  CoKernel (cotile.cokernel): SM binding (dynamic queue, takeover, chunk, split),
         CTA binding (POD-like co-residence: a prefill and a decode CTA per SM; 188 + k CTAs =
         k SMs host a decode CTA), tile binding (POD's per-SM ticket policy per grab, runtime
         ratio kP:kD; 376 CTAs = 2 per SM, 188 = 1 per SM).
  POD    FlashInfer 0.7.0 PODWithPagedKVCacheWrapper with the local stream-memset patch; its
         own serial is serial_fi (FlashInfer prefill, then the faster FlashInfer decode path).

Stages (flush = clean-flush cobench.bench_variants screen, groups of <= 24 + serial;
steady = cobench.bench_steady, primary):
  SA  flush, coarse: streams (solo, every C_lib pair); green over the full split grid for the
      solo pair, the budget-best C_lib pair (+/- hint); SM-binding CoKernels: solo pair (+/- hint)
      over the grid, every C_lib pair at 5 splits; tile binding at 1 CTA/SM (solo pair);
      CTA and tile binding at 2 CTAs/SM for every pair of contract-fitting configs.
  SB  flush, derived factorized around SA's best split per column: decode axis (every decode
      config x {normal, evict_first}) and prefill axis (every prefill config, natural twins of
      the best four), 2 splits; hint / natural twins of the best CTA-level variants.
  SC  steady confirmation: top-8 per mechanism class of SA U SB, the best per row, hint twins
      of the top-2, the best streams, 3 random.
  SD  steady refinement around the winners: green split +-2/4/6, CoKernel split +-4/8/12,
      takeover flip, chunk, static schedule; CTA binding k sweep; tile-binding ratio sweep;
      streams order x priority.
  F   final interleaved steady run (full protocol) + clean-flush view: serial, solos, serial_fi,
      FlashInfer solos, POD; the top candidates of every cell; hint ablations; attribution
      variants (POD emulated in our framework: tile binding + FlashInfer-like tiles, FI-like
      TileLang serial, our winner with POD's decode tile, ...); timing builds (per-role ends).
  R   no-oracle robustness (B3 protocol): green vs CoKernel (SM binding, dynamic, takeover) at
      the same prefill shares, each with its candidate configs; rules R1 (proportional) and R2
      (equal finish from solo budget curves), transfer of the first pair's oracle split.
  AT  attribution run (primary pairs, after F): POD and its own serial, POD's policy emulated
      in our framework (tile binding, POD's ratio rule, 2 CTAs/SM, FlashInfer-like tiles in
      FlashInfer's / natural / LPT order, each against the FI-like TileLang serial), POD's
      policy with our own tiles, FI-like tiles under SM binding and on streams, the F winners
      re-measured, the intra winner with POD's decode tile; timing builds.
  FQ  secondary pairs: one interleaved steady run (full protocol) with serial / POD references,
      streams, R's best green and CoKernel (+/- hint, split neighbours), CTA / tile binding,
      POD emulation; stored as stage F.

GPU sharing (research/rules.md 7, cobench.GuardPolicy): cobench raises GpuYield before a point
when a foreign process shows SM activity and after a contaminated point; the study saves, exits
with YIELD_RC = 75, and run_guarded.py waits 30 min and relaunches it (resumes at its next point).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time

import numpy as np
import torch

import p4_common as C4
import p1_common as C
from p1_common import FULL, cb, log
from p1_study import FULLP, LIGHT, kendall, spearman
from cotile import resources
from cotile.cokernel import Orch, build_cokernel, sm_role_table
from cotile.kernel import compile_specs
from cotile.ops import gqa_decode as GD
from cotile.ops import prefill_attn as PA

OUT = os.path.join(C.ROOT, "research", "results", "2026-09-24_p4_study")
PREP = C4.OUT
YIELD_RC = 75
PRIMARY = ["P2048_B16_S2048", "P8192_B64_S8192", "P2048_B16_S8192", "P2048_B64_S2048"]
SECONDARY = ["P8192_B16_S2048", "P8192_B16_S8192", "P8192_B64_S2048", "P2048_B64_S8192"]
TRANSFER_SRC = PRIMARY[0]
GRID = tuple(n for n in C4.GREEN_P)                 # prefill SMs: 32 ... 180
SCREEN_SPLITS = (48, 80, 108, 140, 172)             # C_lib-pair CoKernel screen (SM binding)
R_SPLITS = (32, 48, 64, 80, 94, 108, 124, 140, 156, 172)   # B3 sweep (both sides' budgets measured)
CTA_SMEM = 49 * 1024                                # per-role smem contract for 2 CTAs/SM
TOPK, NRAND = 8, 3


def even(n) -> int:
    return int(2 * round(n / 2))


def clip(n, lo=8, hi=FULL - 8) -> int:
    return max(lo, min(hi, int(n)))


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"), default=C._json_default)
    os.replace(tmp, path)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------------------------------
# config naming (deterministic short ids)
# ----------------------------------------------------------------------------------


def sid_a(c: PA.PrefillConfig) -> str:
    return (f"p{c.block_M}.{c.block_N}.{c.num_stages}.{c.threads}" + ("d" if c.epilogue == "direct" else "")
            + {"lpt": "", "natural": "N", "kvhead": "K"}[c.order])


def sid_b(c: GD.DecodeConfig) -> str:
    return (f"d{c.block_N}.{c.heads_per_cta}.{c.num_split}.{c.threads}.{c.num_stages}"
            + {"normal": "", "evict_first": "F", "evict_last": "L"}[c.kv_l2])


def pod_ratio(na: int, nb: int) -> tuple[int, int]:
    """FlashInfer pod.cuh ticket rule: prefill:decode = 1:(D/P) if P <= D else (P/D):1."""
    if na <= nb:
        return (1, max(1, nb // na))
    return (max(1, na // nb), 1)


class Study:
    def __init__(self, pair: str, args):
        self.pair, self.args = pair, args
        self.P, self.Dt = C4.PAIRS[pair]
        summ = load_json(os.path.join(C4.SOLO_DIR, "summary.json"))
        if summ is None:
            raise RuntimeError("P4-a solo summary missing (p4_solo.py)")
        sp, sd = summ["shapes"][self.P], summ["shapes"][self.Dt]
        self.solo = {"a": sp["solo_best"], "b": sd["solo_best"]}
        self.lib = {"a": list(sp["c_lib"]), "b": list(sd["c_lib"])}
        self.budget = {"a": sp["budgets"], "b": sd["budgets"]}
        self.t188 = {"a": sp["t188"], "b": sd["t188"]}
        self.steady_solo = {"a": {k: v["t_iter_us"] for k, v in sp["steady"]["variants"].items()},
                            "b": {k: v["t_iter_us"] for k, v in sd["steady"]["variants"].items()}}
        self.prep_pair = load_json(os.path.join(PREP, "pairs_steady.json"))[pair]
        self.op = {"a": PA, "b": GD}
        self.shape = {"a": C4.shape_of("prefill_attn", self.P), "b": C4.shape_of("gqa_decode", self.Dt)}
        self.cfg = {"a": {}, "b": {}}
        self.base = {"a": [], "b": []}
        for side in "ab":
            for c in self.op[side].configs(self.shape[side]):
                t = self.op[side].cfg_tag(c)
                self.cfg[side][t] = c
                self.base[side].append(t)
        self.lib_set = {"a": set(self.lib["a"]), "b": set(self.lib["b"])}
        self.path = os.path.join(OUT, pair, "study.json")
        self.res = load_json(self.path) or {"pair": pair, "prefill": self.P, "decode": self.Dt, "stages": {}}
        self.res.update({"solo": self.solo, "lib": self.lib, "universe": self.base})
        self.pool = C.StatePool()
        self.grid: dict = {"a": {}, "b": {}}
        self.fl: dict = {"a": {}, "b": {}}
        self.co_specs: dict = {}
        self.V: dict = {}
        self.D: dict = {}
        self.parts: dict = {}
        self.failed: dict = {}
        self.sigs: dict = {}
        self.streams = {"s1": torch.cuda.Stream(), "s2": torch.cuda.Stream(),
                        "hi1": torch.cuda.Stream(priority=-1), "lo1": torch.cuda.Stream(priority=0),
                        "hi2": torch.cuda.Stream(priority=-1), "lo2": torch.cuda.Stream(priority=0)}
        self.da = self.db = None
        self.fi: dict = {}

    # ------------------------------------------------------------------ bookkeeping
    def save(self):
        self.res["failed"] = self.failed
        save_json(self.path, self.res)

    def stage_done(self, s):
        s = "F" if s == "FQ" else s
        return s in self.res["stages"] and not self.args.redo

    def put(self, stage, obj):
        self.res["stages"][stage] = obj
        self.res.get("partial", {}).pop(stage, None)
        self.save()

    def add(self, name, fn, **desc):
        self.V[name] = fn
        self.D[name] = desc
        return name

    def desc(self, names):
        return {n: dict(self.D[n]) for n in dict.fromkeys(["serial"] + list(names))}

    # ------------------------------------------------------------------ configs
    def cfg_of(self, side, tag):
        if tag not in self.cfg[side]:
            raise KeyError(f"{side}: unknown config {tag}")
        return self.cfg[side][tag]

    def variant_tag(self, side, tag, **changes):
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

    def hinted(self, tb, pol="evict_first"):
        return self.variant_tag("b", tb, kv_l2=pol)

    def unhinted(self, tb):
        return self.variant_tag("b", tb, kv_l2="normal")

    def natural(self, ta, order="natural"):
        return self.variant_tag("a", ta, order=order)

    def sid(self, side, tag):
        return sid_a(self.cfg_of("a", tag)) if side == "a" else sid_b(self.cfg_of("b", tag))

    def register(self, d: dict):
        """Re-register the configs of a stored descriptor (derived tags are created lazily)."""
        for side, key in (("a", "a"), ("b", "b")):
            t = d.get(key)
            if t and t not in self.cfg[side]:
                self._register_tag(side, t)

    def _register_tag(self, side, tag):
        # derived tags = a base tag + hint / order suffix; rebuild them from the base
        if side == "b":
            for suf, pol in (("_l2ef", "evict_first"), ("_l2el", "evict_last")):
                if tag.endswith(suf) and tag[:-len(suf)] in self.cfg["b"]:
                    self.hinted(tag[:-len(suf)], pol)
                    return
            # a non-library decode tile (e.g. POD-like n64 sp2 s1): parse the tag
            p = tag.split("_")
            c = GD.DecodeConfig(block_N=int(p[0][1:]), heads_per_cta=int(p[1][1:]), num_split=int(p[2][2:]),
                                threads=int(p[3][1:]), num_stages=int(p[4][1:]),
                                kv_l2={"": "normal", "l2ef": "evict_first", "l2el": "evict_last"}[p[5] if len(p) > 5 else ""])
            GD.validate(self.shape["b"], c)
            self.cfg["b"][GD.cfg_tag(c)] = c
            return
        p = tag.split("_")
        kw = dict(block_M=int(p[0][1:]), block_N=int(p[1][1:]), num_stages=int(p[2][1:]), threads=int(p[3][1:]))
        for x in p[4:]:
            if x in PA.EPILOGUES:
                kw["epilogue"] = x
            elif x in PA.ORDERS:
                kw["order"] = x
        c = PA.PrefillConfig(**kw)
        PA.validate(self.shape["a"], c)
        self.cfg["a"][PA.cfg_tag(c)] = c

    def row_of(self, ta, tb, mbps=1) -> str:
        if mbps == 1 and ta == self.solo["a"] and tb == self.solo["b"]:
            return "solo"
        if mbps == 1 and ta in self.lib_set["a"] and tb in self.lib_set["b"]:
            return "lib"
        return "derived"

    def axes_of(self, ta, tb, mbps=1) -> dict:
        ca, cb_ = self.cfg_of("a", ta), self.cfg_of("b", tb)
        base_a = self.natural(ta, "lpt")
        return {"prefill_nonlib": base_a not in self.lib_set["a"], "prefill_natural": ca.order == "natural",
                "prefill_order": ca.order,
                "decode_nonlib": self.unhinted(tb) not in self.lib_set["b"], "decode_splitkv": cb_.num_split > 1,
                "decode_hint": cb_.kv_l2 != "normal", "regcap": mbps > 1}

    def pod_like_decode(self) -> str:
        """FlashInfer POD's decode tile on this GPU: 16-row Q tile (the 4 heads of a KV group),
        64-row KV steps, 128 threads, K/V single-buffered (36 KB smem), 2-way KV split at B16
        (padded batch 32), none at B64 (research/results/2026-09-22_flashinfer_baselines §4.1)."""
        sp = 2 if self.shape["b"].batch <= 16 else 1
        return self.variant_tag("b", "n64_h4_sp1_t128_s1", num_split=sp)

    def fi_like_prefill(self, order="kvhead") -> str:
        """FlashInfer's prefill tile: 128 query rows x 32 KV rows, 128 threads, 48 KB smem, in
        FlashInfer's CTA order (order="kvhead": KV head outermost, query blocks ascending; the
        stages run before 2026-09-24 07:00 used "natural", TileLang's globally shortest-first
        order). FlashInfer's tile packs 32 positions x the 4 heads of a KV group, ours 128
        positions x 1 head: same MMA shape and smem."""
        return self.variant_tag("a", "m128_n32_s1_t128", order=order)

    def ntiles(self, side, tag) -> int:
        return self.op[side].tile_space(self.shape[side], self.cfg_of(side, tag)).num_tiles

    # ------------------------------------------------------------------ kernels / launchers
    def compile_tolerant(self, specs) -> list:
        uniq = {}
        for s in specs:
            uniq.setdefault((s.name, repr(s.shape), s.grid, repr(sorted(s.pass_configs.items()))), []).append(s)
        todo = [v[0] for v in uniq.values() if v[0].kernel is None and v[0].compile_error is None]
        if todo:
            t = time.time()
            st = compile_specs(todo, num_workers=48)
            self.res["compile_s"] = self.res.get("compile_s", 0.0) + time.time() - t
            log(f"compiled {len(todo)}: {st}")
        bad = []
        for v in uniq.values():
            for s in v[1:]:
                s.kernel, s.compile_error = v[0].kernel, v[0].compile_error
            if v[0].kernel is None:
                bad.append(v[0])
                self.failed[v[0].name] = str(v[0].compile_error)[:300]
        return bad

    def lint(self, spec) -> list:
        if "lint" not in spec.extra:
            spec.extra["lint"] = resources.invalid_memory_descriptors(resources.sass(resources.cubin_bytes(spec.kernel)))
        return spec.extra["lint"]

    def grid_spec(self, side, tag):
        if tag not in self.grid[side]:
            self.grid[side][tag] = self.op[side].build_grid(self.shape[side], self.cfg_of(side, tag))
        return self.grid[side][tag]

    def launcher(self, side, tag):
        f = self.fl[side]
        if tag not in f:
            spec = self.grid_spec(side, tag)
            if spec.kernel is None and self.compile_tolerant([spec]):
                raise RuntimeError(f"grid {tag}: {spec.compile_error}")
            if side == "b" and self.cfg_of("b", tag).kv_l2 != "normal" and self.lint(spec):
                raise RuntimeError(f"grid {tag}: invalid memory descriptors {spec.extra['lint'][:1]}")
            d = self.da if side == "a" else self.db
            f[tag] = d.launcher(spec, self.pool.view(side, spec))
        return f[tag]

    def prep_grids(self, pairs):
        specs = [self.grid_spec("a", ta) for ta, _ in pairs] + [self.grid_spec("b", tb) for _, tb in pairs]
        self.compile_tolerant([s for s in specs if s.kernel is None])

    def ensure_data(self):
        if self.da is not None:
            return
        self.da = C.OpData("prefill_attn", self.P)
        self.db = C.OpData("gqa_decode", self.Dt)
        self.res["rotation"] = {"a": {"copies": self.da.n, "bytes": self.da.bytes_per_copy},
                                "b": {"copies": self.db.n, "bytes": self.db.bytes_per_copy}}
        sa, sb = self.solo["a"], self.solo["b"]
        self.prep_grids([(sa, sb)])
        fa, fb = self.launcher("a", sa), self.launcher("b", sb)

        def serial(i):
            fa(i)
            fb(i)
        self.add("serial", serial, kind="serial", row="ref", a=sa, b=sb)
        self.add("solo_a", fa, kind="solo", row="ref", a=sa)
        self.add("solo_b", fb, kind="solo", row="ref", b=sb)

    def ensure_fi(self):
        """FlashInfer references: prefill, the faster decode path of P4-a, patched POD."""
        if self.fi:
            return
        path = self.prep_pair["fi_decode_path"]
        self.fi["a"] = C4.FIData("prefill", prefill_tag=self.P)
        self.fi["b"] = C4.FIData("decode_cc" if path == "fi_decode_cc" else "decode_tc", decode_tag=self.Dt)
        self.fi["pod"] = C4.FIData("pod", prefill_tag=self.P, decode_tag=self.Dt)
        FA, FB, POD = self.fi["a"].launcher(), self.fi["b"].launcher(), self.fi["pod"].launcher()

        def serial_fi(i):
            FA(i)
            FB(i)
        self.add("serial_fi", serial_fi, kind="serial_fi", row="ref")
        self.add("fi_a", FA, kind="fi_solo", row="ref")
        self.add("fi_b", FB, kind="fi_solo", row="ref", path=path)
        self.add("pod", POD, kind="pod", row="ref")
        self.res["fi"] = {"decode_path": path, "copies": {k: v.n for k, v in self.fi.items()}}

    def part(self, n):
        if n not in self.parts:
            p = cb.split_sms(n, ignore_coscheduling=True)
            if p.n_sms != n or p.n_rest != FULL - n:
                raise RuntimeError(f"green split {n}: got {p.n_sms}/{p.n_rest}")
            self.parts[n] = p
        return self.parts[n]

    def close(self):
        for p in self.parts.values():
            p.close()

    # ------------------------------------------------------------------ variants: inter
    def v_serial(self, ta, tb, row=None):
        name = f"ser_{self.sid('a', ta)}_{self.sid('b', tb)}"
        if name in self.V:
            return name
        fa, fb = self.launcher("a", ta), self.launcher("b", tb)

        def fn(i):
            fa(i)
            fb(i)
        return self.add(name, fn, kind="serial_cfg", row=row or "ref", a=ta, b=tb)

    def v_solo(self, side, tag):
        name = f"solo_{self.sid(side, tag)}"
        if name not in self.V:
            self.add(name, self.launcher(side, tag), kind="solo", row="ref", **{side: tag})
        return name

    def v_streams(self, ta, tb, prio="eq", order="ab", row=None):
        name = f"st_{prio}_{order}_{self.sid('a', ta)}_{self.sid('b', tb)}"
        if name in self.V:
            return name
        try:
            fa, fb = self.launcher("a", ta), self.launcher("b", tb)
        except RuntimeError as e:
            log(f"SKIP {name}: {e}")
            return None
        s = self.streams
        sa, sb = {"eq": (s["s1"], s["s2"]), "pA": (s["hi1"], s["lo2"]), "pB": (s["lo1"], s["hi2"])}[prio]
        order_arg = {"ab": "given", "ba": "reverse", "alt": "alternate"}[order]
        return self.add(name, cb.Par(("a", sa, fa), ("b", sb, fb), order=order_arg), kind="streams",
                        row=row or self.row_of(ta, tb), a=ta, b=tb, prio=prio, order=order, axes=self.axes_of(ta, tb))

    def v_green(self, n, ta, tb, row=None):
        name = f"gr_{n}_{self.sid('a', ta)}_{self.sid('b', tb)}"
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

    def lib_choice(self, side, n, rank=0):
        """C_lib config of `side` with the rank-th lowest solo time at n SMs (nearest measured
        budget; P4-a clean-flush budget curves)."""
        best = []
        for t in self.lib[side]:
            b = self.budget[side].get(t, {})
            if not b:
                continue
            m = min(b, key=lambda k: (abs(int(k) - n), int(k)))
            best.append((b[m], t, int(m)))
        best.sort()
        return best[min(rank, len(best) - 1)][1]

    def budget_pair(self, n):
        return self.lib_choice("a", n), self.lib_choice("b", FULL - n)

    # ------------------------------------------------------------------ variants: intra
    @staticmethod
    def orch(kind, to=True, chunk=(1, 1), num_ctas=FULL, timing=False, mbps=1):
        chunk = tuple(chunk)
        if kind == "sm_dyn":
            o = Orch(binding="sm", schedule="dynamic", chunk=chunk, takeover=to, num_ctas=FULL, timing=timing)
        elif kind == "sm_static":
            o = Orch(binding="sm", schedule="static", takeover=to, num_ctas=FULL, timing=timing)
        elif kind == "cta_dyn":
            o = Orch(binding="cta", schedule="dynamic", chunk=chunk, takeover=to, num_ctas=num_ctas, timing=timing)
        elif kind == "tile_dyn":
            o = Orch(binding="tile", schedule="dynamic", chunk=chunk, takeover=True, num_ctas=num_ctas, timing=timing)
        else:
            raise ValueError(kind)
        return dataclasses.replace(o, min_blocks_per_sm=mbps) if mbps != 1 else o

    def co_spec(self, ta, tb, orch):
        key = (ta, tb, orch)
        if key not in self.co_specs:
            self.co_specs[key] = build_cokernel(PA, self.shape["a"], self.cfg_of("a", ta), GD, self.shape["b"],
                                                self.cfg_of("b", tb), orch)
        return self.co_specs[key]

    def co_sig(self, spec) -> dict:
        if spec.name not in self.sigs:
            s = resources.signature(spec)
            self.sigs[spec.name] = {k: s.get(k) for k in ("regs", "smem_total", "threads", "ctas_per_sm", "limit_by",
                                                          "local_bytes")}
        return self.sigs[spec.name]

    def co_name(self, ta, tb, kind, n, to, chunk, num_ctas, ratio, timing, mbps):
        knob = f"n{n}" if kind.startswith("sm") else f"g{num_ctas}r{ratio[0]}-{ratio[1]}"
        return (f"co_{kind}{'T' if to else ''}_c{chunk[0]}-{chunk[1]}_{self.sid('a', ta)}_{self.sid('b', tb)}_{knob}"
                + (f"_mb{mbps}" if mbps != 1 else "") + ("_tm" if timing else ""))

    def v_co(self, ta, tb, kind="sm_dyn", n=None, to=True, chunk=(1, 1), num_ctas=FULL, ratio=(1, 1), timing=False,
             row=None, check=True, mbps=1):
        chunk, ratio = tuple(chunk), tuple(ratio)
        if kind == "tile_dyn":
            to = True
        name = self.co_name(ta, tb, kind, n, to, chunk, num_ctas, ratio, timing, mbps)
        if name in self.V:
            return name
        spec = self.co_spec(ta, tb, self.orch(kind, to, chunk, num_ctas, timing, mbps))
        if spec.kernel is None and self.compile_tolerant([spec]):
            return None
        if self.cfg_of("b", tb).kv_l2 != "normal" and self.lint(spec):
            self.failed[spec.name] = f"invalid memory descriptors: {spec.extra['lint'][:1]}"
            log(f"SKIP {name}: {self.failed[spec.name]}")
            return None
        sig = self.co_sig(spec)
        if not kind.startswith("sm") and sig["ctas_per_sm"] * FULL < num_ctas and kind == "cta_dyn":
            # CTA binding assumes every CTA is resident (roles by arrival rank per SM)
            log(f"SKIP {name}: {sig['ctas_per_sm']} CTAs/SM < {num_ctas}/{FULL}")
            return None
        try:
            fa, fb = self.launcher("a", ta), self.launcher("b", tb)
        except RuntimeError as e:
            log(f"SKIP {name}: {e}")
            return None
        var = C.CoVariant(spec, self.da, self.db, sm_role=sm_role_table(n) if kind.startswith("sm") else None,
                          ratio=ratio, pool=self.pool)
        if check and not C.check_same(fa, fb, self.da, self.db, var):
            raise RuntimeError(f"{name}: outputs differ from the grid kernels")
        return self.add(name, var, kind="co", row=row or self.row_of(ta, tb, mbps), a=ta, b=tb, binding=kind, n_a=n,
                        takeover=to, chunk=list(chunk), num_ctas=num_ctas, ratio=list(ratio), timing=timing,
                        mbps=mbps, spec=spec.name, sig=sig, axes=self.axes_of(ta, tb, mbps))

    def v_co_many(self, kws):
        pairs = [(k["ta"], k["tb"]) for k in kws]
        self.prep_grids(list(dict.fromkeys(pairs)))
        specs = [self.co_spec(k["ta"], k["tb"], self.orch(k.get("kind", "sm_dyn"), k.get("to", True),
                                                          tuple(k.get("chunk", (1, 1))), k.get("num_ctas", FULL),
                                                          k.get("timing", False), k.get("mbps", 1))) for k in kws]
        self.compile_tolerant([s for s in specs if s.kernel is None])
        return [n for n in (self.v_co(**k) for k in kws) if n]

    def like(self, name, **ch):
        """Variant with `name`'s knobs and some fields replaced (a, b, n, to, chunk, num_ctas,
        ratio, timing, mbps)."""
        d = self.D[name]
        a, b = ch.get("a", d["a"]), ch.get("b", d["b"])
        if a is None or b is None:
            return None
        if d["kind"] == "green":
            return self.v_green(ch.get("n", d["n_a"]), a, b)
        if d["kind"] == "streams":
            return self.v_streams(a, b, ch.get("prio", d["prio"]), ch.get("order", d["order"]))
        if d["kind"] != "co":
            return None
        return self.v_co(a, b, d["binding"], ch.get("n", d["n_a"]), ch.get("to", d["takeover"]),
                         tuple(ch.get("chunk", d["chunk"])), ch.get("num_ctas", d["num_ctas"]),
                         tuple(ch.get("ratio", d["ratio"])), ch.get("timing", False), mbps=ch.get("mbps", d.get("mbps", 1)))

    def rebuild(self, names, stage_desc):
        out = []
        co = []
        need = []
        for n in names:
            d = stage_desc.get(n)
            if n not in self.V and d is not None and d.get("a") and d.get("b"):
                self.register(d)
                need.append((d["a"], d["b"]))
        if need:
            self.prep_grids(list(dict.fromkeys(need)))
        for n in names:
            if n in self.V:
                out.append(n)
                continue
            d = stage_desc.get(n)
            if d is None:
                continue
            self.register(d)
            k = d["kind"]
            if k == "co":
                co.append(dict(ta=d["a"], tb=d["b"], kind=d["binding"], n=d["n_a"], to=d["takeover"],
                               chunk=tuple(d["chunk"]), num_ctas=d["num_ctas"], ratio=tuple(d["ratio"]),
                               timing=d["timing"], row=d.get("row"), mbps=d.get("mbps", 1)))
            elif k == "green":
                out.append(self.v_green(d["n_a"], d["a"], d["b"], d.get("row")))
            elif k == "streams":
                out.append(self.v_streams(d["a"], d["b"], d["prio"], d["order"], d.get("row")))
            elif k == "solo" and ("a" in d or "b" in d):
                side = "a" if "a" in d else "b"
                out.append(self.v_solo(side, d[side]))
            elif k == "serial_cfg":
                out.append(self.v_serial(d["a"], d["b"], d.get("row")))
            elif k in ("serial_fi", "fi_solo", "pod"):
                self.ensure_fi()
                out.append(n)
        if co:
            out += self.v_co_many(co)
        return [x for x in dict.fromkeys(out) if x and x in self.V]

    def ensure_rebuilt(self, stages):
        for st in stages:
            s = self.res["stages"].get(st)
            if s:
                names = list(s.get("steady", {})) or [k for k in s.get("flush", {}) if not k.startswith("_")]
                self.rebuild([n for n in names if n not in ("serial", "solo_a", "solo_b")], s["desc"])

    # ------------------------------------------------------------------ measurement
    def flush(self, names, label, reps=None, group=24):
        """Clean-flush screen in groups of <= group variants + serial; every group is saved as
        soon as it is measured (resumable after a GPU-sharing yield)."""
        part = self.res.setdefault("partial", {}).setdefault(label, {})
        if reps is None:
            reps = 50          # cobench protocol minimum (strict)
        out = {"_serial": [], "_guard": []}
        names = [n for n in dict.fromkeys(names) if n and n != "serial"]
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
                    o[n] = {"t_us": v["total"]["median"], "cv": v["total"]["cv"], "speedup": r.derived["speedup"][n],
                            "clock_mhz": (v.get("clock") or {}).get("median"),
                            "ops": {k: x["median"] for k, x in (v.get("ops") or {}).items()}}
                rec = {"names": sub, "out": o, "serial": r.variants["serial"]["total"]["median"],
                       "guard": {"clean": r.guard.get("clean"), "attempts": r.guard.get("attempts")}}
                part[str(g)] = rec
                self.save()
            out.update(rec["out"])
            out["_serial"].append(rec["serial"])
            out["_guard"].append(rec["guard"])
        out["_wall_s"] = time.time() - t0
        return out

    def steady(self, names, label, proto=LIGHT):
        names = list(dict.fromkeys(["serial"] + [n for n in names if n and n != "serial"]))
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

    # ------------------------------------------------------------------ helpers
    def fits_cta(self, side, tag) -> bool:
        return self.op[side].smem_bytes(self.shape[side], self.cfg_of(side, tag), "persistent") <= CTA_SMEM

    def cta_sets(self):
        """Contract for 2 CTAs/SM (one prefill + one decode CTA per SM): per-role smem <= 49 KB;
        256-thread CTAs need the register cap (min_blocks_per_sm = 2)."""
        pa = [t for t in self.base["a"] if self.fits_cta("a", t) and self.cfg_of("a", t).order == "lpt"
              and self.cfg_of("a", t).epilogue == "smem"]
        fi_like = self.fi_like_prefill()
        pa = list(dict.fromkeys(pa + [fi_like]))
        fit_b = [t for t in self.base["b"] if self.fits_cta("b", t)]
        rank_b = sorted(fit_b, key=lambda t: self.t188["b"].get(t, 1e9))
        pb = [t for t in self.lib["b"] if t in fit_b] or rank_b[:3]
        return pa, list(dict.fromkeys(pb + [self.pod_like_decode()]))

    def mbps_for(self, ta, tb) -> int:
        thr = max(self.cfg_of("a", ta).threads, self.cfg_of("b", tb).threads)
        return 2 if thr > 128 else 1

    def pod_ratio_of(self, ta, tb):
        return pod_ratio(self.ntiles("a", ta), self.ntiles("b", tb))

    def cta_variants(self, ta, tb, hint=False, tile_ratios=None):
        """CTA-level co-residence variants of one config pair at 2 CTAs/SM: CTA binding (1:1
        per SM) and tile binding (POD's ratio rule, 1:1)."""
        tb2 = self.hinted(tb) if hint else tb
        mb = self.mbps_for(ta, tb2)
        kws = [dict(ta=ta, tb=tb2, kind="cta_dyn", num_ctas=2 * FULL, mbps=mb)]
        ratios = tile_ratios or list(dict.fromkeys([self.pod_ratio_of(ta, tb2), (1, 1)]))
        kws += [dict(ta=ta, tb=tb2, kind="tile_dyn", num_ctas=2 * FULL, ratio=r, mbps=mb) for r in ratios]
        return kws

    def cta_feasible(self, pairs) -> list:
        """(ta, tb) pairs whose compiled CTA-binding CoKernel fits 2 CTAs/SM."""
        specs = [(ta, tb, self.co_spec(ta, tb, self.orch("cta_dyn", num_ctas=2 * FULL, mbps=self.mbps_for(ta, tb))))
                 for ta, tb in pairs]
        self.compile_tolerant([s for *_, s in specs if s.kernel is None])
        out, info = [], {}
        for ta, tb, s in specs:
            if s.kernel is None:
                continue
            sig = self.co_sig(s)
            info[f"{self.sid('a', ta)}_{self.sid('b', tb)}"] = sig
            if sig["ctas_per_sm"] >= 2 and not sig.get("local_bytes"):
                out.append((ta, tb))
        self.res["cta_info"] = info
        return out

    # ------------------------------------------------------------------ SA: coarse flush screen
    def stage_SA(self):
        sa, sb = self.solo["a"], self.solo["b"]
        names = [self.v_streams(sa, sb, p, o) for p in ("eq", "pA", "pB") for o in ("ab", "ba")]
        self.prep_grids([(ta, tb) for ta in self.lib["a"] for tb in self.lib["b"]])
        names += [self.v_streams(ta, tb, "eq", "alt") for ta in self.lib["a"] for tb in self.lib["b"]]
        # two streams with co-residable configs (<= 48 KB each: the hardware scheduler can put a
        # prefill and a decode CTA on one SM): FlashInfer-like prefill x POD-like decode
        pf, dp = self.fi_like_prefill("lpt"), self.pod_like_decode()
        self.prep_grids([(pf, dp)] + [self.budget_pair(n) for n in GRID]
                        + [(self.budget_pair(n)[0], self.hinted(self.budget_pair(n)[1])) for n in GRID])
        names += [self.v_streams(pf, dp, p, "ab") for p in ("eq", "pA", "pB")]
        for n in GRID:
            ga, gb = self.budget_pair(n)
            names += [self.v_green(n, sa, sb), self.v_green(n, ga, gb), self.v_green(n, ga, self.hinted(gb))]
        # SM binding: solo pair (+hint) over the grid, C_lib pairs at 5 splits
        kws = [dict(ta=sa, tb=tb, kind="sm_dyn", n=n) for tb in (sb, self.hinted(sb)) for n in GRID]
        kws += [dict(ta=ta, tb=tb, kind="sm_dyn", n=n) for ta in self.lib["a"] for tb in self.lib["b"] for n in SCREEN_SPLITS]
        # tile binding at 1 CTA/SM with the solo pair: POD rule and 1:1
        kws += [dict(ta=sa, tb=sb, kind="tile_dyn", num_ctas=FULL, ratio=r)
                for r in dict.fromkeys([self.pod_ratio_of(sa, sb), (1, 1)])]
        names += self.v_co_many(kws)
        # CTA-level co-residence at 2 CTAs/SM: every contract-fitting pair
        pa, pb = self.cta_sets()
        feas = self.cta_feasible([(ta, tb) for ta in pa for tb in pb])
        ck = [k for ta, tb in feas for k in self.cta_variants(ta, tb, tile_ratios=[self.pod_ratio_of(ta, tb)])]
        names_cta = self.v_co_many(ck)
        names += names_cta
        names = list(dict.fromkeys(x for x in names if x))
        r = self.flush(names, "SA")
        self.put("SA", {"flush": r, "desc": self.desc(names), "cta_sets": {"a": pa, "b": pb},
                        "cta_feasible": [list(x) for x in feas], "wall_s": r["_wall_s"]})

    # ------------------------------------------------------------------ SB: derived factorized screen
    def best_flush(self, stages, pred, k=1):
        pool = {}
        for st in stages:
            s = self.res["stages"].get(st)
            if not s or "flush" not in s:
                continue
            for n, v in s["flush"].items():
                if n.startswith("_"):
                    continue
                d = s["desc"].get(n)
                if d and pred(d):
                    pool[n] = max(pool.get(n, 0.0), v["speedup"])
        return sorted(pool, key=lambda n: -pool[n])[:k], pool

    def split_pair(self, stage, kind):
        """Best and second-best split (by the best variant at each split) of a mechanism in a
        flush stage."""
        s = self.res["stages"][stage]
        best = {}
        for n, v in s["flush"].items():
            if n.startswith("_"):
                continue
            d = s["desc"][n]
            if d.get("kind") != kind[0] or (kind[0] == "co" and d.get("binding") != kind[1]):
                continue
            best[d["n_a"]] = max(best.get(d["n_a"], 0.0), v["speedup"])
        order = sorted(best, key=lambda x: -best[x])
        n0 = order[0]
        n1 = next((x for x in order[1:] if abs(x - n0) <= 24), clip(n0 + 8 if n0 < 140 else n0 - 8))
        return n0, n1

    def stage_SB(self):
        names = []
        sa_desc = self.res["stages"]["SA"]["desc"]
        (wg,), _ = self.best_flush(["SA"], lambda d: d["kind"] == "green")
        (wc,), _ = self.best_flush(["SA"], lambda d: d["kind"] == "co" and d["binding"] == "sm_dyn")
        top_c, _ = self.best_flush(["SA"], lambda d: d["kind"] == "co" and d["binding"] in ("cta_dyn", "tile_dyn"), 3)
        self.rebuild([wg, wc] + top_c, sa_desc)
        # inter: green, factorized around the best split
        dg = self.D[wg]
        ng = self.split_pair("SA", ("green",))
        tb0 = dg["b"]
        pairs = [(dg["a"], tb) for tb in self.base["b"]] + [(dg["a"], self.hinted(tb)) for tb in self.base["b"]]
        pairs += [(ta, tb0) for ta in self.base["a"]]
        self.prep_grids(pairs)
        for n in ng:
            names += [self.mark(self.v_green(n, ta, tb), "b" if ta == dg["a"] else "a") for ta, tb in pairs]
        # intra: SM binding, factorized around the best CoKernel split
        dc = self.D[wc]
        nc = self.split_pair("SA", ("co", "sm_dyn"))
        base_a = self.natural(dc["a"], "lpt")
        top_a = [t for t in sorted(self.base["a"], key=lambda t: self.t188["a"].get(t, 1e9))
                 if self.cfg_of("a", t).order == "lpt"][:4]
        kws = [dict(ta=dc["a"], tb=tb, kind="sm_dyn", n=n) for tb in self.base["b"] for n in nc]
        kws += [dict(ta=dc["a"], tb=self.hinted(tb), kind="sm_dyn", n=n) for tb in self.base["b"] for n in nc]
        kws += [dict(ta=ta, tb=dc["b"], kind="sm_dyn", n=n) for ta in self.base["a"] for n in nc]
        kws += [dict(ta=self.natural(ta), tb=dc["b"], kind="sm_dyn", n=n) for ta in dict.fromkeys(top_a + [base_a])
                for n in nc]
        names += self.v_co_many(kws)
        # CTA-level: hint and natural twins of the best 3 CTA/tile variants
        ck = []
        for n in top_c:
            d = self.D[n]
            for ta, tb in ((d["a"], self.hinted(d["b"])), (self.natural(d["a"]), d["b"])):
                if ta and tb:
                    ck.append(dict(ta=ta, tb=tb, kind=d["binding"], num_ctas=d["num_ctas"], ratio=tuple(d["ratio"]),
                                   mbps=self.mbps_for(ta, tb)))
        names += self.v_co_many(ck)
        names = list(dict.fromkeys(x for x in names if x))
        r = self.flush(names, "SB")
        self.put("SB", {"flush": r, "desc": self.desc(names), "splits": {"green": list(ng), "co": list(nc)},
                        "fixed": {"green": wg, "co": wc}, "top_cta": top_c, "wall_s": r["_wall_s"]})

    def mark(self, v, axis):
        if v:
            self.D[v]["axis"] = axis
        return v

    # ------------------------------------------------------------------ SC: steady confirmation
    def stage_SC(self):
        st = ["SA", "SB"]
        all_desc = {**self.res["stages"]["SA"]["desc"], **self.res["stages"]["SB"]["desc"]}
        classes = {
            "green": lambda d: d["kind"] == "green",
            "co_sm": lambda d: d["kind"] == "co" and d["binding"] == "sm_dyn",
            "co_cta": lambda d: d["kind"] == "co" and d["binding"] in ("cta_dyn", "tile_dyn"),
        }
        picks, allsc = {}, {}
        for cname, pred in classes.items():
            k = TOPK if cname != "co_cta" else 4
            top, pool = self.best_flush(st, pred, k)
            picks[cname] = top
            allsc.update(pool)
        rows = {}
        for col, kinds in (("inter", ("green", "streams")), ("intra", ("co",))):
            for row in ("solo", "lib"):
                top, _ = self.best_flush(st, lambda d, r=row, ks=kinds: d["kind"] in ks and d["row"] in
                                         (("solo",) if r == "solo" else ("solo", "lib")), 2)
                rows[f"{row}_{col}"] = top
        streams, _ = self.best_flush(st, lambda d: d["kind"] == "streams", 3)
        ranked = sorted(allsc, key=lambda n: -allsc[n])
        chosen = set(x for v in picks.values() for x in v) | set(x for v in rows.values() for x in v)
        rest = [n for n in ranked if n not in chosen]
        rng = random.Random(f"{self.pair}-SC")
        rand = rng.sample(rest, min(NRAND, len(rest)))
        self.rebuild(list(chosen) + streams + rand, all_desc)
        flips = []
        for c in ("green", "co_sm", "co_cta"):
            for n in picks[c][:2]:
                d = self.D[n]
                tb = self.unhinted(d["b"]) if d["axes"]["decode_hint"] else self.hinted(d["b"])
                flips.append(self.like(n, b=tb))
        names = [x for v in picks.values() for x in v] + [x for v in rows.values() for x in v] + streams + flips + rand
        names = ["solo_a", "solo_b"] + list(dict.fromkeys(x for x in names if x))
        r, meta = self.steady(names, "SC")
        conf = [n for n in names if n in allsc]
        xs = [1.0 / allsc[n] for n in conf]
        ys = [r[n]["t_iter_us"] / r["serial"]["t_iter_us"] for n in conf]
        fid = {"n": len(conf), "spearman": spearman(xs, ys), "kendall": kendall(xs, ys),
               "winner": min(conf, key=lambda n: r[n]["t_iter_us"])}
        fid["winner_screen_rank"] = ranked.index(fid["winner"]) + 1
        self.put("SC", {"steady": r, "meta": meta, "desc": self.desc(names), "picks": picks, "rows": rows,
                        "streams": streams, "flips": [f for f in flips if f], "rand": rand, "fidelity": fid,
                        "wall_s": meta["wall_s"]})

    # ------------------------------------------------------------------ SD: steady refinement
    def stage_best(self, stages, pred, k=1):
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

    def refine_green(self, name, steps=(-6, -4, -2, 2, 4, 6)):
        d = self.D[name]
        return [self.v_green(d["n_a"] + dn, d["a"], d["b"]) for dn in steps if 8 <= d["n_a"] + dn <= FULL - 8]

    def refine_co(self, name, full=True):
        d = self.D[name]
        out = []
        if d["binding"] == "sm_dyn":
            for dn in (-12, -8, -4, 4, 8, 12):
                if 8 <= d["n_a"] + dn <= FULL - 4:
                    out.append(self.like(name, n=d["n_a"] + dn))
            if full:
                out.append(self.like(name, to=not d["takeover"]))
                for ch in ((1, 2), (1, 4), (2, 1)):
                    if tuple(d["chunk"]) != ch:
                        out.append(self.like(name, chunk=ch))
                out.append(self.v_co(d["a"], d["b"], "sm_static", d["n_a"], True))
        elif d["binding"] == "cta_dyn":
            for k in (47, 94, 141, 188):
                if FULL + k != d["num_ctas"]:
                    out.append(self.like(name, num_ctas=FULL + k))
            if full:
                # the same pair with tile binding (POD policy)
                out.append(self.v_co(d["a"], d["b"], "tile_dyn", None, True, (1, 1), 2 * FULL,
                                     self.pod_ratio_of(d["a"], d["b"]), mbps=d.get("mbps", 1)))
        else:  # tile binding: ticket ratio sweep (runtime knob) and CTAs/SM
            pr = self.pod_ratio_of(d["a"], d["b"])
            for r in dict.fromkeys([pr, (1, 1), (2, 1), (1, 2), (4, 1), (3, 2)]):
                if tuple(d["ratio"]) != r:
                    out.append(self.like(name, ratio=r))
            if full:
                out.append(self.v_co(d["a"], d["b"], "cta_dyn", None, True, (1, 1), 2 * FULL, (1, 1), mbps=d.get("mbps", 1)))
        return out

    def stage_SD(self):
        self.ensure_rebuilt(["SC"])
        S = ["SC"]
        names = []
        for row, row_ok in (("derived", lambda d: True), ("lib", lambda d: d["row"] in ("solo", "lib")),
                            ("solo", lambda d: d["row"] == "solo")):
            g = self.stage_best(S, lambda d, f=row_ok: d["kind"] == "green" and f(d))
            if g:
                names += [g[0][1]] + self.refine_green(g[0][1], (-6, -4, -2, 2, 4, 6) if row != "solo" else (-4, 4))
            c = self.stage_best(S, lambda d, f=row_ok: d["kind"] == "co" and d["binding"] == "sm_dyn" and f(d))
            if c:
                names += [c[0][1]] + self.refine_co(c[0][1], full=row == "derived")
            if row == "solo":
                continue
            t = self.stage_best(S, lambda d, f=row_ok: d["kind"] == "co" and d["binding"] in ("cta_dyn", "tile_dyn") and f(d))
            if t:
                names += [t[0][1]] + self.refine_co(t[0][1], full=True)
        s = self.stage_best(S, lambda d: d["kind"] == "streams")
        if s:
            d = self.D[s[0][1]]
            names += [self.v_streams(d["a"], d["b"], p, o) for p in ("eq", "pA", "pB") for o in ("ab", "ba")]
        # hint flips of the derived winners (same knobs)
        for pred in (lambda d: d["kind"] == "green", lambda d: d["kind"] == "co"):
            w = self.stage_best(S, pred)
            if w:
                d = self.D[w[0][1]]
                names.append(self.like(w[0][1], b=self.unhinted(d["b"]) if d["axes"]["decode_hint"] else self.hinted(d["b"])))
        names = list(dict.fromkeys(x for x in names if x))
        r, meta = self.steady(names, "SD")
        self.put("SD", {"steady": r, "meta": meta, "desc": self.desc(names), "wall_s": meta["wall_s"]})

    # ------------------------------------------------------------------ F
    CELLS = {
        "solo_inter": lambda d: d["kind"] in ("green", "streams") and d["row"] == "solo",
        "lib_inter": lambda d: d["kind"] in ("green", "streams") and d["row"] in ("solo", "lib"),
        "derived_inter": lambda d: d["kind"] in ("green", "streams"),
        "solo_intra": lambda d: d["kind"] == "co" and d["row"] == "solo",
        "lib_intra": lambda d: d["kind"] == "co" and d["row"] in ("solo", "lib"),
        "derived_intra": lambda d: d["kind"] == "co",
        "streams": lambda d: d["kind"] == "streams",
        "green_nohint": lambda d: d["kind"] == "green" and not d["axes"]["decode_hint"],
        "co_sm": lambda d: d["kind"] == "co" and d["binding"] == "sm_dyn",
        "co_cta": lambda d: d["kind"] == "co" and d["binding"] == "cta_dyn",
        "co_tile": lambda d: d["kind"] == "co" and d["binding"] == "tile_dyn",
        "co_nohint": lambda d: d["kind"] == "co" and not d["axes"]["decode_hint"],
    }

    def attribution_variants(self, w_intra=None) -> dict:
        """Q2(c) controls. POD emulated in our framework (tile binding at 2 CTAs/SM, POD's ratio
        rule, FlashInfer-like tiles: prefill 128x32 / 128 threads, natural order; decode 16-row
        Q tile, 64-row KV steps, 128 threads, POD's KV split), the same with LPT order and with
        SM binding; the FI-like TileLang serial (the emulation's own serial); our intra winner
        with POD's decode tile."""
        out = {}
        pf, pl, dp = self.fi_like_prefill("kvhead"), self.fi_like_prefill("lpt"), self.pod_like_decode()
        self.prep_grids([(pf, dp), (pl, dp)])
        out["ser_filike"] = self.v_serial(pf, dp)
        out["solo_filike_a"] = self.v_solo("a", pf)
        out["solo_filike_b"] = self.v_solo("b", dp)
        r = self.pod_ratio_of(pf, dp)
        kws = [dict(ta=pf, tb=dp, kind="tile_dyn", num_ctas=2 * FULL, ratio=r),
               dict(ta=pl, tb=dp, kind="tile_dyn", num_ctas=2 * FULL, ratio=r),
               dict(ta=pf, tb=dp, kind="cta_dyn", num_ctas=2 * FULL),
               dict(ta=pf, tb=self.hinted(dp), kind="tile_dyn", num_ctas=2 * FULL, ratio=r)]
        v = self.v_co_many(kws)
        tags = ["pod_emu", "pod_emu_lpt", "pod_emu_cta", "pod_emu_hint"]
        for t, kw in zip(tags, kws):
            n = self.co_name(kw["ta"], kw["tb"], kw["kind"], None, True, (1, 1), kw["num_ctas"], kw.get("ratio", (1, 1)),
                             False, 1)
            if n in v:
                out[t] = n
        # POD-emulation tiles in the SM-binding mechanism (same split as our SM winner)
        if w_intra:
            d = self.D[w_intra]
            if d["binding"] == "sm_dyn":
                out["sm_filike"] = self.v_co(pf, dp, "sm_dyn", d["n_a"], True)
                # our intra winner with POD's decode tile (hint kept as in the winner)
                tb = self.hinted(dp) if d["axes"]["decode_hint"] else dp
                out["win_pod_decode"] = self.like(w_intra, b=tb)
            elif self.fits_cta("a", d["a"]):
                tb = self.hinted(dp) if d["axes"]["decode_hint"] else dp
                out["win_pod_decode"] = self.like(w_intra, b=tb, mbps=self.mbps_for(d["a"], tb))
        return {k: v for k, v in out.items() if v}

    def stage_F(self):
        self.ensure_fi()
        S = ["SC", "SD"]
        self.ensure_rebuilt(S)
        names = ["solo_a", "solo_b", "serial_fi", "fi_a", "fi_b", "pod"]
        cands = {}
        for cell, pred in self.CELLS.items():
            k = 3 if cell.startswith("derived") else 2 if cell.startswith("lib") else 1
            cands[cell] = [n for _, n, _ in self.stage_best(S, pred, k)]
            names += cands[cell]
        # hint ablations of the column winners (same knobs)
        abl = {}
        for col in ("derived_inter", "derived_intra"):
            if cands[col]:
                w = cands[col][0]
                d = self.D[w]
                tb = self.unhinted(d["b"]) if d["axes"]["decode_hint"] else self.hinted(d["b"])
                abl[col] = {"hint_flip": self.like(w, b=tb)}
                if d["axes"]["prefill_natural"]:
                    abl[col]["order_flip"] = self.like(w, a=self.natural(d["a"], "lpt"))
                names += list(abl[col].values())
        w_intra = cands["derived_intra"][0] if cands["derived_intra"] else None
        att = self.attribution_variants(w_intra)
        names += list(att.values())
        # solo runs of the winners' configs
        solo = {}
        for col in ("derived_inter", "derived_intra"):
            if cands[col]:
                d = self.D[cands[col][0]]
                solo[f"{col}_a"] = self.v_solo("a", d["a"])
                solo[f"{col}_b"] = self.v_solo("b", d["b"])
        names += list(solo.values())
        timing = {}
        for tag, n in (("derived_intra", w_intra), ("lib_intra", (cands["lib_intra"] or [None])[0]),
                       ("pod_emu", att.get("pod_emu")), ("co_cta", (cands["co_cta"] or [None])[0]),
                       ("co_tile", (cands["co_tile"] or [None])[0])):
            if n:
                tn = self.like(n, timing=True)
                if tn:
                    timing[tag] = tn
                    names.append(tn)
        names = list(dict.fromkeys(x for x in names if x))
        r, meta = self.steady(names, "F", FULLP)
        roles = {c: self.V[n].role_times() for c, n in timing.items()}
        rf = self.flush([n for n in names if not self.D[n].get("timing")], "F-flush", reps=60, group=64)
        pod_ok = self.pod_check()
        self.put("F", {"steady": r, "meta": meta, "flush": rf, "desc": self.desc(names), "candidates": cands,
                       "ablations": abl, "attribution": att, "solo": solo, "timing": timing, "role_times": roles,
                       "pod_side_stream_bitwise_eq_default": pod_ok, "wall_s": meta["wall_s"] + rf["_wall_s"]})

    def stage_AT(self):
        """Attribution run (Q2c), one interleaved steady run (full protocol) after F: POD and
        its own serial; POD's policy emulated in our framework (tile binding at 2 CTAs/SM,
        POD's ticket ratio, FlashInfer-like tiles in FlashInfer's CTA order / natural / LPT
        order, the FI-like TileLang serial as its own serial) with the ratio and CTA-binding
        variants; POD's policy with our own tiles (tile binding, 1 CTA/SM, the solo pair);
        FI-like tiles under SM binding and on two streams; our F winners (re-measured) and the
        intra winner with POD's decode tile."""
        self.ensure_fi()
        F = self.res["stages"]["F"]
        FS, FD = F["steady"], F["desc"]
        pick = {}
        for cell, pred in (("inter", lambda d: d.get("kind") in ("green", "streams")),
                           ("intra", lambda d: d.get("kind") == "co" and d.get("binding") == "sm_dyn"),
                           ("cta", lambda d: d.get("kind") == "co" and d.get("binding") == "cta_dyn"),
                           ("tile", lambda d: d.get("kind") == "co" and d.get("binding") == "tile_dyn"
                            and d.get("a") in self.lib_set["a"] | {self.fi_like_prefill("lpt")})):
            c = [n for n, d in FD.items() if n in FS and pred(d) and not d.get("timing")]
            if c:
                pick[cell] = min(c, key=lambda n: FS[n]["t_iter_us"])
        self.rebuild(list(pick.values()), FD)
        names = ["solo_a", "solo_b", "serial_fi", "fi_a", "fi_b", "pod"] + list(pick.values())
        pk, pn, pl, dp = (self.fi_like_prefill("kvhead"), self.fi_like_prefill("natural"), self.fi_like_prefill("lpt"),
                          self.pod_like_decode())
        self.prep_grids([(pk, dp), (pn, dp), (pl, dp)])
        att = {"ser_filike": self.v_serial(pk, dp), "ser_filike_lpt": self.v_serial(pl, dp),
               "ser_filike_natural": self.v_serial(pn, dp),
               "solo_filike_a": self.v_solo("a", pk), "solo_filike_a_lpt": self.v_solo("a", pl),
               "solo_filike_b": self.v_solo("b", dp)}
        r = self.pod_ratio_of(pk, dp)
        sa, sb = self.solo["a"], self.solo["b"]
        kws = {"pod_emu": dict(ta=pk, tb=dp, kind="tile_dyn", num_ctas=2 * FULL, ratio=r),
               "pod_emu_natural": dict(ta=pn, tb=dp, kind="tile_dyn", num_ctas=2 * FULL, ratio=r),
               "pod_emu_lpt": dict(ta=pl, tb=dp, kind="tile_dyn", num_ctas=2 * FULL, ratio=r),
               "pod_emu_r11": dict(ta=pk, tb=dp, kind="tile_dyn", num_ctas=2 * FULL, ratio=(1, 1)),
               "pod_emu_cta": dict(ta=pk, tb=dp, kind="cta_dyn", num_ctas=2 * FULL),
               "pod_emu_hint": dict(ta=pk, tb=self.hinted(dp), kind="tile_dyn", num_ctas=2 * FULL, ratio=r),
               "tile_ourtiles": dict(ta=sa, tb=sb, kind="tile_dyn", num_ctas=FULL, ratio=self.pod_ratio_of(sa, sb))}
        if "intra" in pick:
            d = self.D[pick["intra"]]
            kws["sm_filike"] = dict(ta=pk, tb=dp, kind="sm_dyn", n=d["n_a"])
            kws["sm_filike_lpt"] = dict(ta=pl, tb=dp, kind="sm_dyn", n=d["n_a"])
            tb = self.hinted(dp) if d["axes"]["decode_hint"] else dp
            kws["win_pod_decode"] = dict(ta=d["a"], tb=tb, kind="sm_dyn", n=d["n_a"], to=d["takeover"],
                                         chunk=tuple(d["chunk"]))
        made = self.v_co_many(list(kws.values()))
        for tag, kw in kws.items():
            n = self.co_name(kw["ta"], kw["tb"], kw["kind"], kw.get("n"), True if kw["kind"] == "tile_dyn" else kw.get("to", True),
                             tuple(kw.get("chunk", (1, 1))), kw.get("num_ctas", FULL), tuple(kw.get("ratio", (1, 1))), False, 1)
            if n in made:
                att[tag] = n
        att["streams_filike"] = self.v_streams(pl, dp, "eq", "ab")
        att["streams_filike_pB"] = self.v_streams(pl, dp, "pB", "ab")
        names += list(att.values())
        timing = {}
        for tag in ("pod_emu", "pod_emu_lpt", "tile_ourtiles"):
            if att.get(tag):
                tn = self.like(att[tag], timing=True)
                if tn:
                    timing[tag] = tn
                    names.append(tn)
        names = list(dict.fromkeys(x for x in names if x))
        res, meta = self.steady(names, "AT", FULLP)
        roles = {c: self.V[n].role_times() for c, n in timing.items()}
        self.put("AT", {"steady": res, "meta": meta, "desc": self.desc(names), "picks": pick, "attribution": att,
                        "timing": timing, "role_times": roles, "pod_side_stream_bitwise_eq_default": self.pod_check(),
                        "wall_s": meta["wall_s"]})

    def pod_check(self) -> bool:
        pod = self.fi["pod"]
        x = pod.copies[0]
        torch.cuda.synchronize()
        o1 = pod.w.run(*x)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            o2 = pod.w.run(*x)
        torch.cuda.synchronize()
        return bool(torch.equal(o1[0], o2[0]) and torch.equal(o1[1], o2[1]))

    # ------------------------------------------------------------------ FQ (secondary pairs)
    def stage_FQ(self):
        """Secondary pairs: one interleaved steady run (full protocol) + clean-flush view, from
        R's curves: serial / POD references, streams (solo pair, 3 priorities), R's best green
        and best CoKernel (+/- decode hint, split neighbours), CTA and tile binding of the
        fastest contract-fitting C_lib pair, the POD emulation, timing builds."""
        self.ensure_fi()
        R = self.res["stages"]["R"]
        self.ensure_rebuilt(["R"])
        rs = R["steady"]
        names = ["solo_a", "solo_b", "serial_fi", "fi_a", "fi_b", "pod"]
        sa, sb = self.solo["a"], self.solo["b"]
        names += [self.v_streams(sa, sb, p, "ab") for p in ("eq", "pA", "pB")]
        best = {}
        for mech in ("green", "co"):
            cand = [n for n, c in R["cand"].items() if c["mech"] == mech]
            w = min(cand, key=lambda n: rs[n]["t_iter_us"])
            best[mech] = w
            d = self.D[w]
            tb = self.unhinted(d["b"]) if d["axes"]["decode_hint"] else self.hinted(d["b"])
            names += [w, self.like(w, b=tb)]
            steps = (-8, -4, 4, 8) if mech == "green" else (-12, -6, 6, 12)
            names += [self.like(w, n=d["n_a"] + dn) for dn in steps if 8 <= d["n_a"] + dn <= FULL - 8]
        # CTA-level: fastest contract-fitting C_lib pair (flush solo times), +/- hint
        pa, pb = self.cta_sets()
        la = [t for t in pa if t in self.lib_set["a"]] or pa
        lb = [t for t in pb if t in self.lib_set["b"]] or pb
        ta = min(la, key=lambda t: self.t188["a"].get(t, 1e9))
        tb = min(lb, key=lambda t: self.t188["b"].get(t, 1e9))
        if self.cta_feasible([(ta, tb)]):
            names += self.v_co_many(self.cta_variants(ta, tb) + self.cta_variants(ta, tb, hint=True))
        att = self.attribution_variants(best["co"])
        names += list(att.values())
        timing = {}
        for tag, n in (("co", best["co"]), ("pod_emu", att.get("pod_emu"))):
            if n:
                tn = self.like(n, timing=True)
                if tn:
                    timing[tag] = tn
                    names.append(tn)
        names = list(dict.fromkeys(x for x in names if x))
        r, meta = self.steady(names, "F", FULLP)
        roles = {c: self.V[n].role_times() for c, n in timing.items()}
        rf = self.flush([n for n in names if not self.D[n].get("timing")], "F-flush", reps=60, group=64)
        self.put("F", {"steady": r, "meta": meta, "flush": rf, "desc": self.desc(names), "protocol": "quick (FQ)",
                       "r_best": best, "attribution": att, "timing": timing, "role_times": roles,
                       "pod_side_stream_bitwise_eq_default": self.pod_check(), "wall_s": meta["wall_s"] + rf["_wall_s"]})

    # ------------------------------------------------------------------ R: robustness
    def rule_splits(self) -> dict:
        """R1: n_P = 188 t_P / (t_P + t_D) (P4-a steady solo times of the pair run);
        R2: argmin_n max(t_P(n), t_D(188 - n)) over the C_lib budget curves (clean flush, linear
        interpolation)."""
        v = self.prep_pair["variants"]
        ta, tb = v["prefill"]["t_iter_us"], v["decode"]["t_iter_us"]
        r1 = even(FULL * ta / (ta + tb))

        def curve(side):
            pts = {}
            for t in self.lib[side]:
                for k, x in self.budget[side].get(t, {}).items():
                    pts[int(k)] = min(pts.get(int(k), 1e18), x)
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
        return {"R1": clip(r1), "R2": clip(best[1]), "R1_inputs": {"t_P": ta, "t_D": tb},
                "R2_pred": {"t_P": best[2], "t_D": best[3]}}

    def stage_R(self):
        rules = self.rule_splits()
        splits = set(R_SPLITS) | {rules["R1"], rules["R2"]}
        transfer = {}
        if self.pair != TRANSFER_SRC:
            m = load_json(os.path.join(OUT, TRANSFER_SRC, "study.json"))
            if m and "R" in m["stages"]:
                transfer = m["stages"]["R"]["summary"]["oracle_split"]
                splits |= set(transfer.values())
        splits = sorted(splits)
        F = self.res["stages"].get("F")
        dI = dC = lI = lC = None
        subst = {}
        if F and "candidates" in F:
            cz = F["candidates"]
            FS, FD = F["steady"], F["desc"]

            def best_of(cell, pred=lambda d: True):
                c = [n for n in cz.get(cell, []) if n in FS and pred(FD[n])]
                return FD[min(c, key=lambda n: FS[n]["t_iter_us"])] if c else None
            dI = best_of("derived_inter", lambda d: d["kind"] == "green") or best_of("green_nohint")
            lI = best_of("lib_inter", lambda d: d["kind"] == "green")
            dC = best_of("derived_intra")
            if dC and dC["binding"] != "sm_dyn":
                subst["derived"] = {"winner_binding": dC["binding"]}
                dC = best_of("co_sm")
            lC = best_of("lib_intra")
            if lC and lC["binding"] != "sm_dyn":
                subst["lib"] = {"winner_binding": lC["binding"]}
                lC = best_of("lib_intra", lambda d: d["binding"] == "sm_dyn")
        sa, sb = self.solo["a"], self.solo["b"]
        cand = {}
        for n in splits:
            ga, gb = self.budget_pair(n)
            gl = [("budget", ga, gb), ("budget_hint", ga, self.hinted(gb))]
            cl = [("budget", ga, gb, (1, 1)), ("budget_hint", ga, self.hinted(gb), (1, 1)), ("solo", sa, sb, (1, 1))]
            if dI:
                gl.append(("derived", dI["a"], dI["b"]))
            if lI:
                gl.append(("lib", lI["a"], lI["b"]))
            if dC:
                cl.append(("derived", dC["a"], dC["b"], tuple(dC["chunk"])))
            if lC:
                cl.append(("lib", lC["a"], lC["b"], tuple(lC["chunk"])))
            self.prep_grids([(a, b) for _, a, b in gl] + [(a, b) for _, a, b, _ in cl])
            for src, ta, tb in gl:
                v = self.v_green(n, ta, tb)
                if v:
                    cand.setdefault(v, {"mech": "green", "n": n, "src": []})["src"].append(src)
            kws = [dict(ta=ta, tb=tb, n=n, kind="sm_dyn", to=True, chunk=ch) for _, ta, tb, ch in cl]
            self.v_co_many(kws)
            for (src, ta, tb, ch) in cl:
                v = self.co_name(ta, tb, "sm_dyn", n, True, ch, FULL, (1, 1), False, 1)
                if v in self.V:
                    cand.setdefault(v, {"mech": "co", "n": n, "src": []})["src"].append(src)
        names = list(cand)
        r, meta = self.steady(names, "R")
        summ = self.robustness_summary(r, cand, rules, transfer, splits)
        self.put("R", {"steady": r, "meta": meta, "desc": self.desc(names), "cand": cand, "rules": rules,
                       "transfer_from": TRANSFER_SRC if transfer else None, "transfer_splits": transfer,
                       "splits": splits, "summary": summ, "cta_substitutes": subst, "wall_s": meta["wall_s"]})

    @staticmethod
    def robustness_summary(r, cand, rules, transfer, splits) -> dict:
        out = {"oracle_split": {}}
        for mech in ("green", "co"):
            per = {}
            per_src: dict = {}
            for v, c in cand.items():
                if c["mech"] != mech:
                    continue
                s = r[v]["speedup"]
                if c["n"] not in per or s > per[c["n"]][0]:
                    per[c["n"]] = (s, v, c["src"])
                for src in c["src"]:
                    per_src.setdefault(src, {})[str(c["n"])] = s
            curve = {n: per[n][0] for n in splits if n in per}
            orc = max(curve, key=curve.get)
            sweep = [curve[n] for n in R_SPLITS if n in curve]
            m = {"curve": {str(k): v for k, v in curve.items()}, "best_cfg": {str(n): per[n][1] for n in splits if n in per},
                 "per_src": per_src, "oracle": curve[orc], "oracle_split": orc, "worst_sweep": min(sweep),
                 "worst_split": min((n for n in R_SPLITS if n in curve), key=curve.get),
                 "mean_sweep": float(np.mean(sweep)), "R1": curve.get(rules["R1"]), "R2": curve.get(rules["R2"])}
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

    # ------------------------------------------------------------------ compile only
    def compile_only(self, stages):
        """CPU: every grid kernel of the universe (+ hinted / natural twins) and the CoKernels of
        SA (the SB set depends on SA's results)."""
        specs = [self.grid_spec("a", t) for t in self.base["a"]] + [self.grid_spec("b", t) for t in self.base["b"]]
        specs += [self.grid_spec("b", self.hinted(t)) for t in self.base["b"]]
        specs += [self.grid_spec("a", self.natural(t)) for t in self.base["a"] if self.natural(t)]
        specs += [self.grid_spec("a", self.fi_like_prefill()), self.grid_spec("b", self.pod_like_decode())]
        sa, sb = self.solo["a"], self.solo["b"]
        o = self.orch("sm_dyn")
        cos = [self.co_spec(sa, sb, o), self.co_spec(sa, self.hinted(sb), o)]
        cos += [self.co_spec(ta, tb, o) for ta in self.lib["a"] for tb in self.lib["b"]]
        cos.append(self.co_spec(sa, sb, self.orch("tile_dyn", num_ctas=FULL)))
        pa, pb = self.cta_sets()
        for ta in pa:
            for tb in pb:
                mb = self.mbps_for(ta, tb)
                cos += [self.co_spec(ta, tb, self.orch("cta_dyn", num_ctas=2 * FULL, mbps=mb)),
                        self.co_spec(ta, tb, self.orch("tile_dyn", num_ctas=2 * FULL, mbps=mb))]
        if "R" in stages:
            for n in R_SPLITS:
                ga, gb = self.budget_pair(n)
                cos += [self.co_spec(ga, gb, o), self.co_spec(ga, self.hinted(gb), o)]
        bad = self.compile_tolerant(specs + cos)
        log(f"{self.pair}: compiled {len(specs)} grid + {len(cos)} CoKernel specs; failures {len(bad)}")
        for s in bad[:10]:
            log(f"  FAIL {s.name}: {str(s.compile_error)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(C4.PAIRS))
    ap.add_argument("--stages", default="SA,SB,SC,SD,F,R")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    stages = [x for x in args.stages.split(",") if x]
    st = Study(args.pair, args)
    if args.compile_only:
        st.compile_only(stages)
        return 0
    cb.set_policy(yield_to_caller=True)
    s, t = None, time.time()
    try:
        C.wait_gpu(12 << 30)
        C.retry_oom(st.ensure_data)
        st.res.setdefault("meta", []).append(C.env_meta())
        for s in stages:
            if st.stage_done(s):
                log(f"{args.pair} {s}: done, skipped")
                continue
            t = time.time()
            log(f"{args.pair} {s} ...")
            C.retry_oom(getattr(st, f"stage_{s}"))
            st.res.setdefault("stage_wall_s", {})[s] = time.time() - t
            st.res.setdefault("stage_runs", []).append({"stage": s, "wall_s": time.time() - t, "done": True,
                                                        "end": time.strftime("%Y-%m-%d %H:%M:%S")})
            st.res["pool_bytes"] = st.pool.nbytes()
            st.save()
            log(f"{args.pair} {s}: {time.time() - t:.0f}s")
    except cb.GpuYield as e:
        st.res.setdefault("stage_runs", []).append({"stage": s, "wall_s": time.time() - t, "done": False,
                                                    "yield": str(e)[:300]})
        st.save()
        log(f"{args.pair}: yielding the GPU at stage {s}: {str(e)[:200]}")
        return YIELD_RC
    finally:
        st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
