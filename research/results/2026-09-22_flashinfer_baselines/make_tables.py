"""Print the README's markdown tables from timing.json / tilelang_decode.json / correctness.json /
pod_plan_info.json (so every number in the README comes from a JSON file in this directory)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def j(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def decode_table(t):
    print("| B | KV | path | flush µs | flush GB/s | MHz | graph µs | graph GB/s | MHz | CV flush/graph |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, r in t["decode"].items():
        s = r["describe"]["shape"]
        fl, gr = r["flush"], r["graph"]
        path = "tensor-core (FA2, 16-row Q tile)" if r["describe"]["use_tensor_cores"] else "CUDA-core"
        print(f"| {s['batch']} | {s['kv_len']} | {path} | {f(fl['median_us'])} | {f(fl['gbps'], 0)} | "
              f"{f(fl['clock_mhz_median'], 0)} | {f(gr['median_us'])} | {f(gr['gbps'], 0)} | "
              f"{f(gr['clock_mhz_median'], 0)} | {fl['cv']*100:.2f}% / {gr['cv']*100:.2f}% |")


def tilelang_table(tl, t):
    print("| B | KV | TileLang config (block_N/stages/num_split) | flush µs | GB/s | graph µs | graph GB/s | "
          "FlashInfer best flush µs | FI µs / TL µs (flush; >1 = TileLang faster) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e in tl["shapes"]:
        B, S = e["batch"], e["kv_len"]
        fi = min(t["decode"][f"B{B}_S{S}_cc"]["flush"]["median_us"], t["decode"][f"B{B}_S{S}_tc"]["flush"]["median_us"])
        shown = []
        for c in e["configs"]:
            cfg = c["config"]
            tag = (cfg["block_N"], cfg["num_stages"], cfg["num_split"])
            is_best = cfg == e["best_config"]
            is_smoke = cfg["num_split"] == 8 and (cfg["block_N"], cfg["num_stages"]) in ((64, 2), (128, 1))
            if not (is_best or is_smoke):
                continue
            if "flush" not in c:
                print(f"| {B} | {S} | {tag} | error: {c.get('error')} | | | | | |")
                continue
            g = c.get("graph", {})
            label = f"{tag[0]}/{tag[1]}/{tag[2]}" + (" (best of 12)" if is_best else "") + (" (smoke cfg)" if is_smoke else "")
            print(f"| {B} | {S} | {label} | {f(c['flush']['median_us'])} | {f(c['flush']['gbps'], 0)} | "
                  f"{f(g.get('median_us'))} | {f(g.get('gbps'), 0)} | {f(fi)} | {fi / c['flush']['median_us']:.2f} |")
            shown.append(tag)


def prefill_table(t):
    print("| S | H_q/H_kv | flush µs | flush TFLOP/s | MHz | graph µs | graph TFLOP/s | MHz | % of mma.sync peak at MHz (graph) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, r in t["prefill"].items():
        s = r["describe"]["shape"]
        fl, gr = r["flush"], r["graph"]
        peak = 188 * 1024 * gr["clock_mhz_median"] * 1e6 / 1e12
        print(f"| {s['seq_len']} | {s['num_qo_heads']}/{s['num_kv_heads']} | {f(fl['median_us'])} | {f(fl['tflops'])} | "
              f"{f(fl['clock_mhz_median'], 0)} | {f(gr['median_us'])} | {f(gr['tflops'])} | {f(gr['clock_mhz_median'], 0)} | "
              f"{100 * gr['tflops'] / peak:.0f}% |")


def pod_tables(t):
    print("| prefill S | decode B×KV | solo prefill µs | solo decode µs | serial µs | POD µs (MHz) | POD vs serial | "
          "streams makespan µs | streams vs serial | best green (P/D SMs) µs | green vs serial | POD vs best alternative |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for key, r in t["pod"].items():
        if not r.get("done"):
            continue
        Sp, B, S = (int(x[1:]) for x in key.split("_"))
        ser = r["serial_best"]["median_us"]
        pod = r["pod"]["median_us"]
        st = r["streams"]
        sp = st["solo_a"]["median"]
        sd = st["solo_b"]["median"]
        mk = st["makespan"]["median"]
        gbest_k = min(r["green"], key=lambda n: r["green"][n]["makespan"]["median"])
        g = r["green"][gbest_k]
        gm = g["makespan"]["median"]
        best_alt = min(ser, mk, gm)
        print(f"| {Sp} | {B}×{S} | {f(sp)} | {f(sd)} | {f(ser)} | {f(pod)} ({f(r['pod']['clock_mhz_median'], 0)}) | "
              f"{ser / pod:.2f}× | {f(mk)} | {st['derived']['speedup_vs_serial']:.2f}× | {f(gm)} "
              f"({g['prefill_sms']}/{g['decode_sms']}) | {g['derived']['speedup_vs_serial']:.2f}× | {best_alt / pod:.2f}× |")
        rows.append(key)
    print()
    print("Green-context sweep: makespan µs (speedup vs serial measured in the same bench_corun run); "
          "columns = prefill SMs / decode SMs")
    ks = None
    for key, r in t["pod"].items():
        if not r.get("done"):
            continue
        if ks is None:
            ks = sorted(r["green"], key=int)
            print("| config | " + " | ".join(f"{r['green'][k]['prefill_sms']}/{r['green'][k]['decode_sms']}" for k in ks) + " |")
            print("|---|" + "---|" * len(ks))
        cells = [f"{r['green'][k]['makespan']['median']:.0f} ({r['green'][k]['derived']['speedup_vs_serial']:.2f}×)" for k in ks]
        print(f"| {key} | " + " | ".join(cells) + " |")
    print()
    print("Green partitions, solo times: prefill alone on its P SMs / decode alone on its D SMs (µs)")
    for key, r in t["pod"].items():
        if not r.get("done"):
            continue
        cells = [f"{r['green'][k]['solo_a']['median']:.0f} / {r['green'][k]['solo_b']['median']:.0f}" for k in ks]
        print(f"| {key} | " + " | ".join(cells) + " |")
    print()
    print("Clock (ClockProbe per-rep median MHz) and clock-normalised comparison (cycles = µs × MHz)")
    print("| config | serial µs @ MHz | POD µs @ MHz | POD vs serial: time | POD vs serial: cycles | streams makespan @ MHz (corun) | streams vs serial: cycles |")
    print("|---|---|---|---|---|---|---|")
    for key, r in t["pod"].items():
        if not r.get("done"):
            continue
        s, p, st = r["serial_best"], r["pod"], r["streams"]
        sc, pc = s["median_us"] * s["clock_mhz_median"], p["median_us"] * p["clock_mhz_median"]
        cm = st["clock_mhz_median"]
        stc = st["makespan"]["median"] * cm["corun"]
        sersc = st["serial"]["total"]["median"] * cm["serial"]
        print(f"| {key} | {s['median_us']:.0f} @ {s['clock_mhz_median']:.0f} | {p['median_us']:.0f} @ {p['clock_mhz_median']:.0f} | "
              f"{s['median_us'] / p['median_us']:.2f}× | {sc / pc:.2f}× | {st['makespan']['median']:.0f} @ {cm['corun']:.0f} | "
              f"{sersc / stc:.2f}× |")
    print()
    print("serial variants: flush-mode bench (best decode path) / (tc decode) / bench_corun's serial on a side stream")
    for key, r in t["pod"].items():
        if not r.get("done"):
            continue
        tc = r["serial_tc"].get("median_us", r["serial_best"]["median_us"])
        print(f"| {key} | best={r['best_decode']} {r['serial_best']['median_us']:.1f} | tc {tc:.1f} | "
              f"corun-serial {r['streams']['serial']['total']['median']:.1f} | POD post-check {r['pod']['post_check_max_diff']} |")


def main():
    t, tl = j("timing.json"), j("tilelang_decode.json")
    if t and "decode" in t:
        print("## decode\n")
        decode_table(t)
        print()
    if tl and t:
        print("## tilelang vs flashinfer decode\n")
        tilelang_table(tl, t)
        print()
    if t and "prefill" in t:
        print("## prefill\n")
        prefill_table(t)
        print()
    if t and "pod" in t:
        print("## pod\n")
        pod_tables(t)
    pi = j("pod_plan_info.json")
    if pi:
        print("\n## pod plan\n")
        print("| config | prefill CTAs | decode CTAs (POD plan) | grid | waves @2/SM | ticket P:D | standalone TC-decode padded batch |")
        print("|---|---|---|---|---|---|---|")
        for r in pi:
            print(f"| P{r['prefill_seq']}_B{r['decode_batch']}_S{r['decode_kv_len']} | {r['prefill_ctas']} | {r['decode_ctas']} | "
                  f"{r['grid']} | {r['waves_at_2_per_sm']:.2f} | {r['ticket_ratio_prefill_to_decode']} | "
                  f"{r['standalone_tc_decode_plan']['padded_batch_size']} |")


if __name__ == "__main__":
    main()
