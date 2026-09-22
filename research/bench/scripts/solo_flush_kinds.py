"""Run from the repo root (source research/env.sh). Discriminate dirty-writeback vs residual-L2-hit effects of the flush kind (scratch experiment,
results copied to research/results/2026-09-22_solo_profile/studies/flush_kinds.json)."""
import sys, os, json, time
sys.argv = ["x", "run"]
sys.path.insert(0, "research/bench/scripts")
import solo_profile as sp
import torch
from cotile import catalog
args = sp.argparse.Namespace(attempts=2, warm_s=0.3, window_s=0.25, tail_s=0.1)
cat = catalog.load()
picks = [("rmsnorm", "T4096_H4096"), ("rmsnorm", "T16384_H4096"), ("rmsnorm", "T65536_H8192"),
         ("gqa_decode", "B16_S2048"), ("gqa_decode", "B64_S8192"), ("gemm", "M4096_N4096_K4096"), ("gemm", "M2048_N14336_K4096")]
batches = []
for op, st in picks:
    e = cat.get(op, st)
    b = next(x for x in sp.selected_batches(sp.argparse.Namespace(ops=[op], shapes=[st], limit=None)))
    b.ctag = {t: c for t, c in b.ctag.items() if t == e.solo_best.tag}
    batches.append(b)
sp.prepare(batches, workers=8)
power, wd = sp.PowerLog(), sp.Watchdog()
meter, bud = sp.FlushMeter(power), sp.Budgets()
S = bud.streams[sp.FULL]
wd.wait_quiet()
gw = sp.global_warmup(meter, power, S, 60, 120)
rows = []
for b in batches:
    data = sp.OpData(b.op, b.shape)
    tag = next(iter(b.ctag))
    fn = sp.Runner2(b, tag, data).fns["grid"]
    hint = sp._sync_time(fn, S)
    res = {}
    for rep in range(2):
        for kind in ("write", "read", "write+read"):
            meter.flush_kind = kind
            p = sp.measure_point(meter, wd, fn, S, args, hint)
            res.setdefault(kind, []).append(p["median"])
    meter.flush_kind = "write"
    row = {"op": b.op.NAME, "shape": b.tag, "cfg": tag, **{k: sum(v) / len(v) for k, v in res.items()}, "raw": res}
    rows.append(row)
    print(f"{b.name} {tag}: write {row['write']:.1f} read {row['read']:.1f} write+read {row['write+read']:.1f}", flush=True)
    del data; torch.cuda.empty_cache()
json.dump({"global_warmup": gw, "rows": rows}, open("research/results/2026-09-22_solo_profile/studies/flush_kinds.json", "w"), indent=1)
wd.stop(); power.stop()
