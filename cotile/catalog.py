"""Solo-profile catalog (plan P1-S): measured configs per op x shape, `solo_best`,
`c_lib`, SM-budget curves, and the pairing table used to pick P1 study points.

Data come from `research/bench/scripts/solo_profile.py`, which writes one JSON file
per (op, shape) into `<results>/points/`. Every measurement there is a cobench
flush-mode point (L2 flushed before every call): median/p10/p90/CV of the kernel
time, the SM clock during the timed reps (GPU clock probe), board power from NVML
20 ms samples, and energy estimates (see `Meas`).

    from cotile import catalog
    cat = catalog.load()                                   # default results dir
    e = cat.get("gemm", "M4096_N4096_K4096")               # or a shape dataclass
    e.solo_best.tag, e.solo_best.time()                    # fastest grid build, full GPU
    e.budget_best(48), e.budget_loss(48)                   # GOLDYLOC effect
    [c.cfg() for c in e.c_lib]                             # config dataclasses of C_lib
    rows = cat.pairs("gemm", "gqa_decode")                 # pairing table (S7)

Definitions (also in the results README):

* times are medians in microseconds; `sms` is the SM count actually obtained
  (a 47-SM request gives 48 with IGNORE_SM_COSCHEDULING, which has 2-SM granularity);
* `solo_best` = fastest *grid* build on the full GPU (188 SMs);
* resource vector of a config = (smem/CTA, regs/CTA = regs/thread x threads,
  threads/CTA, CTAs/SM) of the compiled kernel; the first three are minimized,
  CTAs/SM is maximized (finer-grained, smaller SM footprint per CTA);
* `c_lib` = Pareto front over (grid time @188, grid-kernel resource vector)
  U best grid config at each reduced SM budget (94, 48) -- proposal section 4;
  `c_lib_persistent` = same with the persistent build's time and resources;
* energy: `e_kernel_uj` = (P_avg x rep period) - (same for a flush-only rep), i.e. the
  energy one call adds on top of the L2 flush and gate; `e_naive_uj` = P_avg x t.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO, "research", "results", "2026-09-22_solo_profile")
FULL_SMS = 188
POWER_CAP_W = 600.0
SCHEMA = 1

# ----------------------------------------------------------------------------------
# shapes
# ----------------------------------------------------------------------------------

_SHAPE_FIELDS = {
    "gemm": ("M", "N", "K"),
    "gqa_decode": ("batch", "seqlen"),
    "rmsnorm": ("tokens", "hidden"),
}
_SHAPE_LETTERS = {
    "gemm": ("M", "N", "K"),
    "gqa_decode": ("B", "S"),
    "rmsnorm": ("T", "H"),
}


def _op_module(op: str):
    from .ops import OPS

    return OPS[op]


def shape_tag(op: str, shape) -> str:
    """'M4096_N4096_K4096' / 'B16_S2048' / 'T4096_H4096'."""
    if isinstance(shape, str):
        return shape
    d = shape if isinstance(shape, dict) else shape.__dict__
    return "_".join(f"{l}{d[f]}" for l, f in zip(_SHAPE_LETTERS[op], _SHAPE_FIELDS[op]))


def shape_obj(op: str, d: dict):
    """Shape dataclass of `op` from its field dict."""
    m = _op_module(op)
    cls = {"gemm": "GemmShape", "gqa_decode": "DecodeShape", "rmsnorm": "RMSNormShape"}[op]
    return getattr(m, cls)(**d)


def cfg_obj(op: str, d: dict):
    m = _op_module(op)
    cls = {"gemm": "GemmConfig", "gqa_decode": "DecodeConfig", "rmsnorm": "RMSNormConfig"}[op]
    return getattr(m, cls)(**d)


# ----------------------------------------------------------------------------------
# work model: FLOPs and global-memory bytes of one call
# ----------------------------------------------------------------------------------


def work(op: str, shape, cfg) -> dict:
    """Per-call work of (shape, cfg), from the tile program's structure.

    flops        useful tensor FLOPs (2 per MAC)
    flops_mma    tensor FLOPs actually issued (decode pads the heads of a tile to 16 MMA rows)
    bytes_min    compulsory DRAM bytes: every input read once, every output written once
                 (a lower bound for ANY implementation of the op at this shape)
    bytes_ws     split-reduction workspace traffic (write partials + combine read, fp32);
                 an upper bound: partials may be served from L2
    bytes_tiles  global->SM load traffic of all tiles + stores (redundant tile loads
                 included); DRAM traffic lies between bytes_min + (0..bytes_ws) and this,
                 depending on L2 hits
    """
    s = shape if isinstance(shape, dict) else shape.__dict__
    c = cfg if isinstance(cfg, dict) else cfg.__dict__
    if op == "gemm":
        M, N, K = s["M"], s["N"], s["K"]
        bM, bN, S = c["block_M"], c["block_N"], c["split_k"]
        ws = 2 * S * M * N * 4 if S > 1 else 0
        return {
            "flops": 2.0 * M * N * K,
            "flops_mma": 2.0 * M * N * K,
            "bytes_min": 2.0 * (M * K + N * K + M * N),
            "bytes_ws": float(ws),
            "bytes_tiles": 2.0 * M * N * K * (bM + bN) / (bM * bN) + 2.0 * M * N + ws,
        }
    if op == "gqa_decode":
        B, S, Hq, Hkv, D = s["batch"], s["seqlen"], s.get("heads", 32), s.get("kv_heads", 8), s.get("dim", 128)
        G = Hq // Hkv
        hpc, nsp = c["heads_per_cta"], c["num_split"]
        HB = G // hpc
        kv = 2.0 * B * S * Hkv * D * 2
        qo = 2.0 * B * Hq * D * 2
        ws = 2.0 * B * Hq * nsp * (D + 1) * 4 if nsp > 1 else 0.0
        return {
            "flops": 4.0 * B * Hq * S * D,
            "flops_mma": 4.0 * B * Hkv * HB * 16 * S * D,
            "bytes_min": kv + qo,
            "bytes_ws": ws,
            "bytes_tiles": HB * kv + nsp * B * Hq * D * 2 + B * Hq * D * 2 + ws,
        }
    if op == "rmsnorm":
        T, H = s["tokens"], s["hidden"]
        R = c["rows_per_cta"]
        return {
            "flops": 0.0,
            "flops_mma": 0.0,
            "bytes_min": 2.0 * T * H * 2 + H * 2,
            "bytes_ws": 0.0,
            "bytes_tiles": 2.0 * T * H * 2 + (T // R) * H * 2,
        }
    raise KeyError(op)


def work_min(op: str, shape) -> dict:
    """Implementation-independent work (flops, bytes_min) of an op at a shape."""
    s = shape if isinstance(shape, dict) else shape.__dict__
    dummy = {
        "gemm": {"block_M": 1, "block_N": 1, "split_k": 1},
        "gqa_decode": {"heads_per_cta": 1, "num_split": 1},
        "rmsnorm": {"rows_per_cta": 1},
    }[op]
    w = work(op, s, dummy)
    return {"flops": w["flops"], "bytes_min": w["bytes_min"]}


# ----------------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------------


@dataclass
class Meas:
    """One measured point (cobench flush mode). Times in us, clock in MHz, power in W,
    energy in uJ. `raw` keeps every stored field."""

    median: float
    p10: float
    p90: float
    cv: float
    n: int
    clock_mhz: float | None
    cycles: float | None
    power_w: float | None
    e_kernel_uj: float | None
    e_naive_uj: float | None
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_raw(cls, d: dict) -> "Meas":
        return cls(
            median=d["median"],
            p10=d["p10"],
            p90=d["p90"],
            cv=d["cv"],
            n=d["n"],
            clock_mhz=d.get("clk"),
            cycles=d.get("cycles"),
            power_w=d.get("P_w"),
            e_kernel_uj=d.get("E_kernel_uj"),
            e_naive_uj=d.get("E_naive_uj"),
            raw=d,
        )

    @property
    def ok(self) -> bool:
        return not self.raw.get("contaminated", False) and self.raw.get("check_ok", True) is not False


@dataclass
class ConfigRecord:
    op: str
    tag: str
    cfg_dict: dict
    sig: dict  # {"grid": {...}, "persistent": {...}}
    work: dict
    meas: dict = field(default_factory=dict)  # (build, sms) -> Meas
    check: dict = field(default_factory=dict)

    def cfg(self):
        """The op's config dataclass (imports cotile.ops, i.e. tilelang)."""
        return cfg_obj(self.op, self.cfg_dict)

    def time(self, build: str = "grid", sms: int = FULL_SMS) -> float | None:
        m = self.meas.get((build, sms))
        return m.median if m is not None else None

    def resources(self, build: str = "grid") -> dict:
        s = self.sig[build]
        return {
            "smem": s["smem"],
            "regs_cta": s["regs"] * s["threads"],
            "threads": s["threads"],
            "ctas_sm": s["ctas_sm"],
        }

    @property
    def numerics(self) -> str:
        return self.sig["grid"].get("numerics", "?")


# resource objectives: (key, +1 minimize / -1 maximize)
RESOURCE_OBJECTIVES = (("smem", 1), ("regs_cta", 1), ("threads", 1), ("ctas_sm", -1))


def dominates(a: tuple, b: tuple) -> bool:
    """a dominates b (all objectives minimized)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_front(items: list, key) -> list:
    """Non-dominated items; key(item) -> tuple of minimized objectives."""
    vals = [key(i) for i in items]
    return [it for it, v in zip(items, vals) if not any(dominates(w, v) for w in vals if w is not v)]


@dataclass
class Entry:
    """All measurements of one op x shape."""

    op: str
    shape: dict
    tag: str
    configs: dict  # tag -> ConfigRecord
    refs: dict  # ref name -> {sms: Meas}
    baseline: dict  # "none|188" -> Meas (flush-only rep)
    meta: dict
    refs_end: dict = field(default_factory=dict)  # same references re-measured at the end of the batch
    ref_check: dict = field(default_factory=dict)  # ref name -> fp32-reference check {"ok", "worst", "max_abs"}

    def shape_obj(self):
        return shape_obj(self.op, self.shape)

    @property
    def budgets(self) -> list[int]:
        s = sorted({sms for c in self.configs.values() for (b, sms) in c.meas if b == "grid"}, reverse=True)
        return s

    def measured(self, build: str = "grid", sms: int = FULL_SMS) -> list[ConfigRecord]:
        return [c for c in self.configs.values() if (build, sms) in c.meas and c.meas[(build, sms)].ok]

    def curve(self, sms: int = FULL_SMS, build: str = "grid") -> list[tuple[str, float]]:
        """[(tag, median us)] sorted fastest first."""
        return sorted(((c.tag, c.time(build, sms)) for c in self.measured(build, sms)), key=lambda x: x[1])

    def budget_best(self, sms: int, build: str = "grid") -> ConfigRecord | None:
        cur = self.curve(sms, build)
        return self.configs[cur[0][0]] if cur else None

    @property
    def solo_best(self) -> ConfigRecord | None:
        return self.budget_best(FULL_SMS, "grid")

    @property
    def solo_best_persistent(self) -> ConfigRecord | None:
        return self.budget_best(FULL_SMS, "persistent")

    def budget_loss(self, sms: int, build: str = "grid") -> float | None:
        """t(full-GPU best at `sms`) / t(best at `sms`) - 1 (>= 0)."""
        full, best = self.solo_best, self.budget_best(sms, build)
        if full is None or best is None or full.time(build, sms) is None:
            return None
        return full.time(build, sms) / best.time(build, sms) - 1.0

    def pareto(self, build: str = "grid", objectives=RESOURCE_OBJECTIVES) -> list[ConfigRecord]:
        items = self.measured(build, FULL_SMS)

        def key(c):
            r = c.resources(build)
            return (c.time(build, FULL_SMS),) + tuple(sign * r[k] for k, sign in objectives)

        return sorted(pareto_front(items, key), key=lambda c: c.time(build, FULL_SMS))

    def _c_lib(self, build: str) -> tuple[list[ConfigRecord], dict]:
        why: dict[str, list[str]] = {}
        for c in self.pareto(build):
            why.setdefault(c.tag, []).append("pareto")
        for sms in self.budgets:
            if sms == FULL_SMS:
                continue
            b = self.budget_best(sms, "grid")
            if b is not None:
                why.setdefault(b.tag, []).append(f"best@{sms}")
        recs = sorted((self.configs[t] for t in why), key=lambda c: c.time(build, FULL_SMS) or math.inf)
        return recs, why

    @property
    def c_lib(self) -> list[ConfigRecord]:
        return self._c_lib("grid")[0]

    @property
    def c_lib_reasons(self) -> dict:
        return self._c_lib("grid")[1]

    @property
    def c_lib_persistent(self) -> list[ConfigRecord]:
        return self._c_lib("persistent")[0]

    def ref_best(self, sms: int = FULL_SMS, correct_only: bool = True) -> tuple[str, Meas] | None:
        """Fastest reference at `sms` (by default only references whose output passed the
        op's fp32-reference tolerance)."""
        c = [(n, m[sms]) for n, m in self.refs.items() if sms in m
             and (not correct_only or self.ref_check.get(n, {}).get("ok", True))]
        return min(c, key=lambda x: x[1].median) if c else None

    def flush_phase(self) -> tuple[float, float] | None:
        """(energy uJ, duration us) of the flush+gate phase of one flush-mode rep, from the
        batch's flush-only baseline point."""
        m = self.baseline.get(f"none|{FULL_SMS}")
        if m is None or m.power_w is None:
            return None
        per = m.raw["period_us"]
        return m.power_w * per, per

    def persistent_penalty(self) -> dict:
        """tag -> t(persistent) / t(grid) at full GPU."""
        out = {}
        for c in self.configs.values():
            g, p = c.time("grid"), c.time("persistent")
            if g and p:
                out[c.tag] = p / g
        return out


class Catalog:
    def __init__(self, entries: dict, path: str):
        self.entries = entries  # (op, shape_tag) -> Entry
        self.path = path

    def ops(self) -> list[str]:
        return sorted({op for op, _ in self.entries})

    def shapes(self, op: str) -> list[str]:
        return [t for o, t in self.entries if o == op]

    def get(self, op: str, shape) -> Entry:
        return self.entries[(op, shape_tag(op, shape))]

    def __iter__(self):
        return iter(self.entries.values())

    # -- measured machine rates (for the naive resource bound) ------------------
    def rates(self, sms: int = FULL_SMS) -> dict:
        """Best measured tensor FLOP/s (GEMM configs and references) and DRAM bytes/s
        (compulsory bytes / time over every op) at `sms` SMs."""
        tc, dram = (0.0, None), (0.0, None)
        for e in self:
            wm = work_min(e.op, e.shape)
            cands = [(f"{c.tag}|grid", c.meas[("grid", sms)]) for c in e.measured("grid", sms)]
            cands += [(f"{c.tag}|persistent", c.meas[("persistent", sms)]) for c in e.measured("persistent", sms)]
            cands += [(f"ref:{n}", m[sms]) for n, m in e.refs.items() if sms in m]
            for name, m in cands:
                t = m.median * 1e-6
                if wm["flops"] and e.op == "gemm" and wm["flops"] / t > tc[0]:
                    tc = (wm["flops"] / t, f"{e.op}:{e.tag}:{name}")
                if wm["bytes_min"] / t > dram[0]:
                    dram = (wm["bytes_min"] / t, f"{e.op}:{e.tag}:{name}")
        return {"tflops": tc[0] / 1e12, "tflops_at": tc[1], "gbps": dram[0] / 1e9, "gbps_at": dram[1]}

    # -- pairing table ------------------------------------------------------------
    def pairs(self, op_a: str, op_b: str, ratio=(0.25, 4.0), use_ref: bool = False) -> list[dict]:
        """All (shape_a, shape_b) pairs whose best-solo duration ratio t_a / t_b lies in
        `ratio`, with the serial time and naive lower bounds:

        LB_tc    = (flops_mma_a + flops_mma_b) / best measured tensor rate
        LB_dram  = (bytes_min_a + bytes_min_b) / best measured DRAM rate
        LB_power_ss    = (E_a + E_b) / 600 W, E = energy per call (E_kernel) of the solo-best
                   config: the bound for back-to-back (steady-state) execution;
        LB_power_flush = (E_a + E_b + E_f) / 600 W - t_f, where (E_f, t_f) = energy and
                   duration of the flush+gate phase of a flush-mode rep (flush-only baseline,
                   mean of the two batches): flush-mode reps (cobench, bench_corun) interleave
                   a ~315 W flush phase that lends the power controller headroom, so a
                   flush-mode co-run can beat LB_power_ss (solo GEMMs do: P_kernel > 600 W);
        LB       = max(LB_tc, LB_dram, LB_power_flush); also max_solo = max(t_a, t_b)
        (a co-run cannot finish before its longer member does solo on the whole GPU).
        `use_ref`: if a reference library is faster than the best TileLang config, use it
        as the solo time (energy then stays the TileLang config's)."""
        rates = self.rates()
        rows = []
        for ea in (e for e in self if e.op == op_a):
            for eb in (e for e in self if e.op == op_b):
                sa, sb = ea.solo_best, eb.solo_best
                if sa is None or sb is None:
                    continue
                ta, tb = sa.time(), sb.time()
                ra = ea.ref_best() if use_ref else None
                rb = eb.ref_best() if use_ref else None
                if ra and ra[1].median < ta:
                    ta = ra[1].median
                if rb and rb[1].median < tb:
                    tb = rb[1].median
                r = ta / tb
                if not ratio[0] <= r <= ratio[1]:
                    continue
                wa, wb = sa.work, sb.work
                lb_tc = (wa["flops_mma"] + wb["flops_mma"]) / (rates["tflops"] * 1e12) * 1e6
                lb_dram = (work_min(op_a, ea.shape)["bytes_min"] + work_min(op_b, eb.shape)["bytes_min"]) / (
                    rates["gbps"] * 1e9
                ) * 1e6
                Ea = sa.meas[("grid", FULL_SMS)].e_kernel_uj
                Eb = sb.meas[("grid", FULL_SMS)].e_kernel_uj
                lb_ss = lb_p = None
                if Ea is not None and Eb is not None:
                    lb_ss = (Ea + Eb) / POWER_CAP_W
                    fl = [e.flush_phase() for e in (ea, eb)]
                    fl = [f for f in fl if f is not None]
                    if fl:
                        ef = sum(f[0] for f in fl) / len(fl)
                        tf = sum(f[1] for f in fl) / len(fl)
                        lb_p = max(0.0, (Ea + Eb + ef) / POWER_CAP_W - tf)
                lb = max(x for x in (lb_tc, lb_dram, lb_p) if x is not None)
                t_serial = ta + tb
                bound = max(lb, ta, tb)
                rows.append(
                    {
                        "a": ea.tag,
                        "b": eb.tag,
                        "cfg_a": sa.tag,
                        "cfg_b": sb.tag,
                        "t_a": ta,
                        "t_b": tb,
                        "ratio": r,
                        "t_serial": t_serial,
                        "max_solo": max(ta, tb),
                        "lb_tc": lb_tc,
                        "lb_dram": lb_dram,
                        "lb_power": lb_p,
                        "lb_power_ss": lb_ss,
                        "lb": lb,
                        "lb_binding": ("tc", "dram", "power")[[lb_tc, lb_dram, lb_p or -1].index(lb)],
                        "speedup_bound": t_serial / bound,
                        "e_a_uj": Ea,
                        "e_b_uj": Eb,
                    }
                )
        return sorted(rows, key=lambda x: -x["speedup_bound"])


# ----------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------


def _entry_from_file(d: dict) -> Entry:
    op = d["op"]
    configs = {}
    for tag, c in d["configs"].items():
        configs[tag] = ConfigRecord(op=op, tag=tag, cfg_dict=c["cfg"], sig=c["sig"], work=c["work"], check=c.get("check", {}))
    refs: dict = {}
    refs_end: dict = {}
    baseline = {}
    for key, p in d.get("points", {}).items():
        name, build, sms = key.split("|")
        m = Meas.from_raw(p)
        if name.startswith("ref:") and build == "ref_end":
            refs_end.setdefault(name[4:], {})[int(sms)] = m
        elif name.startswith("ref:"):
            refs.setdefault(name[4:], {})[int(sms)] = m
        elif name == "flush_only":
            baseline[f"{build}|{sms}"] = m
        elif name in configs:
            configs[name].meas[(build, int(sms))] = m
    return Entry(op=op, shape=d["shape"], tag=d["shape_tag"], configs=configs, refs=refs, baseline=baseline, meta=d.get("meta", {}),
                 refs_end=refs_end, ref_check=d.get("ref_check", {}))


def load(path: str | None = None, ops: Iterable[str] | None = None) -> Catalog:
    """Load every `<path>/points/<op>__<shape>.json` (default: DEFAULT_DIR)."""
    path = path or DEFAULT_DIR
    pdir = os.path.join(path, "points")
    entries = {}
    for fn in sorted(os.listdir(pdir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(pdir, fn)) as f:
            d = json.load(f)
        if ops is not None and d["op"] not in ops:
            continue
        e = _entry_from_file(d)
        entries[(e.op, e.tag)] = e
    return Catalog(entries, path)


def summary_dict(cat: Catalog) -> dict[str, Any]:
    """Compact derived summary (written to <results>/summary.json by solo_analyze.py)."""
    out: dict[str, Any] = {"schema": SCHEMA, "rates": cat.rates(), "entries": {}}
    for e in cat:
        sb = e.solo_best
        d = {
            "solo_best": sb.tag if sb else None,
            "solo_best_us": sb.time() if sb else None,
            "solo_best_persistent": (e.solo_best_persistent.tag if e.solo_best_persistent else None),
            "refs": {n: {str(k): v.median for k, v in m.items()} for n, m in e.refs.items()},
            "budgets": {},
            "c_lib": {c.tag: e.c_lib_reasons[c.tag] for c in e.c_lib},
            "c_lib_persistent": [c.tag for c in e.c_lib_persistent],
            "pareto_grid": [c.tag for c in e.pareto("grid")],
            "n_configs": len(e.configs),
        }
        for sms in e.budgets:
            b = e.budget_best(sms)
            d["budgets"][str(sms)] = {
                "best": b.tag if b else None,
                "best_us": b.time("grid", sms) if b else None,
                "full_best_us": sb.time("grid", sms) if sb else None,
                "loss_of_full_best": e.budget_loss(sms),
            }
        out["entries"][f"{e.op}:{e.tag}"] = d
    return out
