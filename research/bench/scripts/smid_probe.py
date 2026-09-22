"""P0-2 %smid / %globaltimer / green-context probe.

Writes research/results/<date>_smid_probe/{summary.json, placement_raw.json}.
Usage: python research/bench/scripts/smid_probe.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cobench as cb  # noqa: E402
from cobench.smid import probe_ctas  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "../../results/2026-09-22_smid_probe"))


def ranges(xs) -> str:
    """[0,1,2,5,6] -> '0-2,5-6'"""
    xs = sorted(int(x) for x in xs)
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(f"{xs[i]}" if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


# ---------------------------------------------------------------------------
# (ii) CTA -> SM placement
# ---------------------------------------------------------------------------
CONFIGS = [  # name, block, dynamic smem
    ("smem60K_b128", 128, 60 * 1024),
    ("smem40K_b128", 128, 40 * 1024),
    ("smem20K_b128", 128, 20 * 1024),
    ("thr1024", 1024, 0),
    ("thr512", 512, 0),
    ("thr128", 128, 0),
    ("thr32", 32, 0),
]
GRIDS = [188, 376, 752]


def analyze_placement(r: dict, nsm: int, canon: np.ndarray | None) -> dict:
    sm, tk, t0, t1 = r["smid"], r["ticket"], r["t_start"], r["t_end"]
    G = sm.size
    first_end = t1.min()
    wave1 = t0 < first_end               # started before any CTA finished
    cnt_all = np.bincount(sm, minlength=nsm)
    cnt_w1 = np.bincount(sm[wave1], minlength=nsm)
    used_w1 = cnt_w1[cnt_w1 > 0]
    blk = np.arange(G)
    out = {
        "grid": int(G), "occupancy_calc": int(r["occupancy"]),
        "wave1_ctas": int(wave1.sum()),
        "wave1_sms_used": int((cnt_w1 > 0).sum()),
        "wave1_per_sm_hist": {int(k): int(v) for k, v in zip(*np.unique(cnt_w1, return_counts=True))},
        "total_per_sm_hist": {int(k): int(v) for k, v in zip(*np.unique(cnt_all, return_counts=True))},
        "wave1_max_per_sm_eq_occupancy": bool(used_w1.max() <= r["occupancy"]),
        # breadth-first: the first min(G, nsm) CTAs by blockIdx go to distinct SMs
        "first_nsm_blocks_distinct_sms": int(np.unique(sm[: min(G, nsm)]).size),
        "ticket_eq_block_frac": float((tk == blk).mean()),
        "ticket_block_max_absdiff": int(np.abs(tk - blk).max()),
        "wave1_start_spread_us": float((t0[wave1].max() - t0[wave1].min()) / 1e3),
        "block_order_ok_frac": float((np.diff(t0[np.argsort(blk)]) >= 0).mean()),
        "warpid_values_head": sorted({int(x) for x in r["warpid"]})[:12],
        "nwarpid": sorted({int(x) for x in r["nwarpid"]}),
        "nsmid": sorted({int(x) for x in r["nsmid"]}),
    }
    if canon is not None:
        m = min(nsm, G)
        out["first_nsm_match_canonical_frac"] = float((sm[:m] == canon[:m]).mean())
        if G >= 2 * nsm:
            out["block_b_plus_nsm_same_sm_frac"] = float((sm[nsm:2 * nsm] == sm[:nsm]).mean())
    return out


def run_placement(nsm: int, repeats: int = 3, spin_ns: int = 200_000):
    canon = None
    res, raw = {}, {}
    # canonical order = 1 CTA/SM, grid = nsm
    rc = probe_ctas(nsm, 128, smem=60 * 1024, spin_ns=spin_ns)
    canon = rc["smid"].copy()
    for name, block, smem in CONFIGS:
        for G in GRIDS:
            key = f"{name}_g{G}"
            runs = []
            smids = []
            for rep in range(repeats):
                r = probe_ctas(G, block, smem=smem, spin_ns=spin_ns)
                runs.append(analyze_placement(r, nsm, canon))
                smids.append(r["smid"])
                if rep == 0:
                    raw[key] = {"smid": r["smid"].tolist(), "ticket": r["ticket"].tolist(),
                                "t_start_ns": (r["t_start"] - r["t_start"].min()).tolist()}
            a = runs[0]
            a["repeat_identical_mapping_frac"] = float(np.mean([(s == smids[0]).mean() for s in smids[1:]]))
            a["repeats"] = repeats
            res[key] = a
    return canon, res, raw


def describe_canonical(canon: np.ndarray) -> dict:
    """Structure of the breadth-first dispatch order (blockIdx -> smid for grid=nsm)."""
    pairs = canon.reshape(-1, 2) if canon.size % 2 == 0 else None
    tpc_pairs = bool(pairs is not None and np.all(pairs[:, 1] == pairs[:, 0] + 1) and np.all(pairs[:, 0] % 2 == 0))
    return {
        "head_48": canon[:48].tolist(),
        "consecutive_blocks_form_even_odd_smid_pairs": tpc_pairs,
        "smid_sorted": bool(np.all(np.diff(canon) > 0)),
        "full": canon.tolist(),
    }


# ---------------------------------------------------------------------------
# (iii) green contexts
# ---------------------------------------------------------------------------
def run_green(nsm: int, ns=(8, 16, 32, 64, 94, 128, 180)):
    out = {}
    for flag in (False, True):
        for n in ns:
            key = f"n{n}_{'ignoreCosched' if flag else 'default'}"
            with cb.split_sms(n, ignore_coscheduling=flag) as p:
                rp = cb.build_sm_remap(p.stream, expected=p.n_sms)
                rr = cb.build_sm_remap(p.rest_stream, expected=p.n_rest) if p.rest else None
                # %nsmid seen inside the green context
                r = probe_ctas(p.n_sms, 32, smem=cb.one_cta_per_sm_smem(), spin_ns=10_000,
                               stream=p.stream)
                a = set(rp.smids.tolist())
                b = set(rr.smids.tolist()) if rr else set()
                ent = {
                    "requested": n, "got": p.n_sms, "rest": p.n_rest,
                    "driver_count_part": p.part.sm_count_from_driver(),
                    "part_smids": ranges(a), "rest_smids": ranges(b),
                    "disjoint": not (a & b), "union_is_all": (a | b) == set(range(nsm)),
                    "nsmid_inside": sorted({int(x) for x in r["nsmid"]}),
                    "part_contiguous": len(ranges(a).split(",")) == 1,
                    "part_n_runs": len(ranges(a).split(",")),
                }
            # reproducibility: same request again -> same SMs?
            with cb.split_sms(n, ignore_coscheduling=flag) as p2:
                rp2 = cb.build_sm_remap(p2.stream, expected=p2.n_sms)
                ent["same_sms_on_recreate"] = rp2.smids.tolist() == rp.smids.tolist()
            out[key] = ent
            print(f"  green {key}: got {ent['got']} rest {ent['rest']} part={ent['part_smids']}", flush=True)
    # granularity sweep (no contexts needed)
    gran = {}
    for flag in (False, True):
        gran["ignoreCosched" if flag else "default"] = {
            n: cb.query_split(n, ignore_coscheduling=flag) for n in range(1, nsm + 1)}
    return out, gran


def granularity_summary(gran: dict) -> dict:
    s = {}
    for k, m in gran.items():
        gots = sorted({g for g, _ in m.values()})
        s[k] = {"distinct_sizes": gots,
                "examples": {n: m[n] for n in (1, 2, 3, 7, 8, 9, 15, 16, 17, 94, 95, 180, 184, 185, 188) if n in m}}
    return s


# ---------------------------------------------------------------------------
# (iv) %globaltimer + CUDA event resolution
# ---------------------------------------------------------------------------
def event_resolution(n: int = 2000) -> dict:
    """Quantisation of cudaEventElapsedTime: events around tiny kernels, enqueued behind a
    long sleep kernel so the GPU executes them back-to-back (no host gaps)."""
    x = torch.zeros(1, device="cuda")
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for e in evs:  # torch creates the CUDA event lazily on first record (~40 us host)
        e.record()
    torch.cuda.synchronize()
    torch.cuda._sleep(400_000_000)  # ~140 ms of GPU time: host enqueues everything first
    for e in evs:
        e.record()
        x.add_(1)
    torch.cuda.synchronize()
    d = np.round(np.array([evs[i].elapsed_time(evs[i + 1]) * 1e6 for i in range(n - 1)])).astype(np.int64)
    nz = d[d > 0]
    return {"quantum_ns_gcd": int(np.gcd.reduce(nz)) if nz.size else None,
            "min_nonzero_ns": int(nz.min()) if nz.size else None,
            "note": "gcd of GPU-side spacings of back-to-back record+tiny-kernel pairs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.cuda.init()
    info = cb.device_info()
    nsm = info["sms"]
    t_all = time.time()
    summary = {"device": info, "gpu_state_start": cb.gpu_state(), "date": time.strftime("%Y-%m-%d %H:%M")}

    print("(i) smid range / holes", flush=True)
    remap = cb.build_sm_remap()
    summary["smid_range"] = {"nsmid": remap.nsmid, "n_observed": remap.n,
                             "min": int(remap.smids.min()), "max": int(remap.smids.max()),
                             "holes": remap.holes, "table_len": int(remap.table.size),
                             "identity_mapping": bool(np.all(remap.table == np.arange(remap.table.size)))}
    print("  ", summary["smid_range"], flush=True)

    print("(ii) placement", flush=True)
    canon, placement, raw = run_placement(nsm)
    summary["canonical_order"] = describe_canonical(canon)
    summary["placement"] = placement
    for k, v in placement.items():
        print(f"  {k}: occ={v['occupancy_calc']} wave1={v['wave1_ctas']} sms={v['wave1_sms_used']} "
              f"hist={v['wave1_per_sm_hist']} distinct_first={v['first_nsm_blocks_distinct_sms']} "
              f"ticket==blk {v['ticket_eq_block_frac']:.2f} canon {v.get('first_nsm_match_canonical_frac', 0):.2f} "
              f"b+nsm {v.get('block_b_plus_nsm_same_sm_frac', float('nan')):.2f} "
              f"repeat_same {v['repeat_identical_mapping_frac']:.2f} spread {v['wave1_start_spread_us']:.1f}us",
              flush=True)

    print("(iii) green contexts", flush=True)
    green, gran = run_green(nsm)
    summary["green"] = green
    summary["green_granularity"] = granularity_summary(gran)

    print("(iv) globaltimer", flush=True)
    summary["globaltimer_resolution"] = cb.globaltimer_resolution()
    print("  ", summary["globaltimer_resolution"], flush=True)
    sk = cb.globaltimer_skew()
    by = sk.pop("offsets_by_smid")
    summary["globaltimer_skew"] = sk
    summary["globaltimer_skew"]["offsets_by_smid_nonzero"] = {k: v for k, v in by.items() if v != 0}
    print("  ", sk, flush=True)
    summary["event_resolution"] = event_resolution()
    print("  event", summary["event_resolution"], flush=True)
    summary["gpu_state_end"] = cb.gpu_state()
    summary["elapsed_s"] = time.time() - t_all

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(os.path.join(args.out, "placement_raw.json"), "w") as f:
        json.dump(raw, f, separators=(",", ":"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
