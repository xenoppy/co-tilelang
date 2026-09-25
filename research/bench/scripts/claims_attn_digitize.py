"""Read the claimed numbers off the upstream figures (no data files exist for them).

  images/mha_performance_h100.png (top panel, FA0-FA4): bar heights = latency normalised to
      TileLang (TileLang = 1). Output: ratio baseline_latency / TileLang_latency per shape.
      Cross-check: the geometric means reproduce the TileLang paper's text (arXiv 2504.17577 v2
      sec. 5.2: "speedups of 1.36x, 1.41x, and 1.70x" over FA3, Triton, PyTorch).
  examples/deepseek_mla/figures/bs{64,128}_float16.png: marker centres -> TFLOPS per series.

Pure-Python PNG decoding (no PIL in ~/mpk-env). Pixel -> value calibration from the figures'
own axes: FA uses the TileLang bars (height 1.0) and the zero line; MLA uses the grid lines
(100 TFLOPS apart). Reading precision is about +-1 px (FA: 1 px = 0.014; MLA: 1 px = 0.6 TFLOPS).

  python research/bench/scripts/claims_attn_digitize.py   # writes B_attention/claims_digitized.json
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_attn_common as C  # noqa: E402


def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, hdr = 8, b"", None
    while pos < len(data):
        ln, = struct.unpack(">I", data[pos:pos + 4])
        typ, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if typ == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
    w, h, bd, ct, _, _, il = hdr
    assert bd == 8 and il == 0, hdr
    ch = {2: 3, 6: 4, 0: 1, 4: 2}[ct]
    raw, stride = zlib.decompress(idat), w * ch
    img, prev, i = [], bytearray(stride), 0
    for _ in range(h):
        f = raw[i]
        i += 1
        line = bytearray(raw[i:i + stride])
        i += stride
        for x in range(stride):
            a = line[x - ch] if x >= ch else 0
            b = prev[x]
            c = prev[x - ch] if x >= ch else 0
            if f == 1:
                line[x] = (line[x] + a) & 255
            elif f == 2:
                line[x] = (line[x] + b) & 255
            elif f == 3:
                line[x] = (line[x] + ((a + b) >> 1)) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if (pa <= pb and pa <= pc) else (b if pb <= pc else c))) & 255
        img.append([tuple(line[x * ch:(x + 1) * ch][:3]) if ch >= 3 else (line[x],) * 3 for x in range(w)])
        prev = line
    return w, h, img


def fa_figure():
    path = os.path.join(C.REPO, "images", "mha_performance_h100.png")
    w, h, img = read_png(path)
    dark = lambda p: sum(p) < 300  # noqa: E731
    # top panel: first long dark horizontal line from the top that spans most of the width = frame
    rows = [sum(1 for x in range(w) if dark(img[y][x])) for y in range(h)]
    long_rows = [y for y in range(h) if rows[y] > 0.8 * w]
    frame_top = long_rows[0]
    while frame_top + 1 in long_rows:       # the frame line is 2-3 px thick: take its last row
        frame_top += 1
    y0 = next(y for y in long_rows if y > frame_top + 50)       # x axis (value 0) of the top panel
    # bar edges: dark vertical runs rising from the axis
    yb = y0 - 3
    cols = [x for x in range(w) if dark(img[yb][x])]
    groups = []
    for x in cols:
        if groups and x - groups[-1][-1] <= 1:
            groups[-1].append(x)
        else:
            groups.append([x])

    def run_len(x):
        y = y0 - 2
        while y > frame_top and dark(img[y][x]):
            y -= 1
        return y0 - y

    # bar edges rise > 40 px; the panel frame's own vertical sides run the full panel height
    edges = [g[0] for g in groups if 40 < max(run_len(x) for x in g) < (y0 - frame_top) - 3]
    assert len(edges) == 25, f"expected 5 groups x 5 bar edges, found {len(edges)}: {edges}"

    def bar_top(xl, xr):
        xs = range(xl + 4, xr - 4)
        for y in range(frame_top + 2, y0):
            if sum(1 for x in xs if dark(img[y][x])) >= 0.9 * len(xs):
                # centre of the (2-3 px) top edge line
                y1 = y
                while sum(1 for x in xs if dark(img[y1 + 1][x])) >= 0.9 * len(xs):
                    y1 += 1
                return (y + y1) / 2
        raise RuntimeError("no bar top")

    names = ["TileLang", "FA3", "Triton", "PyTorch"]
    out, px = {}, {}
    for g in range(5):
        e = edges[g * 5:(g + 1) * 5]
        tops = [bar_top(e[b], e[b + 1]) for b in range(4)]
        hts = [y0 + 0.5 - t for t in tops]
        out[f"FA{g}"] = {n: round(hts[i] / hts[0], 3) for i, n in enumerate(names)}
        px[f"FA{g}"] = {n: hts[i] for i, n in enumerate(names)}
    gm = {n: round(math.exp(sum(math.log(out[s][n]) for s in out) / 5), 3) for n in names[1:]}
    return {"source": "images/mha_performance_h100.png (top panel)",
            "meaning": "latency / TileLang latency (bar heights)", "ratios": out, "bar_px": px,
            "geomean": gm, "paper_text_geomean": {"FA3": 1.36, "Triton": 1.41, "PyTorch": 1.70},
            "calibration": {"axis_y": y0, "frame_top": frame_top, "unit": "TileLang bar height"}}


MLA_COLORS = {"FlashMLA": (31, 119, 180), "Torch": (255, 127, 14), "Triton": (44, 160, 44),
              "FlashInfer": (128, 0, 128), "TileLang": (255, 0, 0)}
CTX = [1024, 2048, 4096, 8192, 16384, 32768]


def mla_figure(bs):
    path = os.path.join(C.REPO, "examples", "deepseek_mla", "figures", f"bs{bs}_float16.png")
    w, h, img = read_png(path)
    dark = lambda p: sum(p) < 150  # noqa: E731
    colsd = [sum(1 for y in range(h) if dark(img[y][x])) for x in range(w)]
    rowsd = [sum(1 for x in range(w) if dark(img[y][x])) for y in range(h)]
    xs = sorted(range(w), key=lambda x: -colsd[x])[:6]
    ys = sorted(range(h), key=lambda y: -rowsd[y])[:6]
    x_left, x_right = min(xs), max(x for x in xs if x < w * 0.85)
    y_top, y_bot = min(ys), max(ys)
    gray = lambda p: 225 <= p[0] <= 245 and abs(p[0] - p[2]) < 4  # noqa: E731
    grid_rows = [y for y in range(y_top + 3, y_bot - 2)
                 if sum(1 for x in range(x_left + 5, x_right - 5) if gray(img[y][x])) > (x_right - x_left) * 0.3]
    grid_cols = [x for x in range(x_left + 3, x_right - 3)
                 if sum(1 for y in range(y_top + 5, y_bot - 5) if gray(img[y][x])) > (y_bot - y_top) * 0.3]

    def clusters(v):
        cl = []
        for x in v:
            if cl and x - cl[-1][-1] <= 2:
                cl[-1].append(x)
            else:
                cl.append([x])
        return [sum(c) / len(c) for c in cl]

    gr, gc = clusters(grid_rows), clusters(grid_cols)
    assert len(gc) == 6, gc
    # visible grid lines are 100..500 TFLOPS (0 is hidden under the Torch line): top = 500
    assert len(gr) == 5, gr
    px_per_100 = (gr[-1] - gr[0]) / 4
    y_zero = gr[-1] + px_per_100
    near = lambda p, c: all(abs(p[i] - c[i]) <= 40 for i in range(3))  # noqa: E731
    out = {}
    for name, c in MLA_COLORS.items():
        vals = []
        for xc in gc:
            ys_ = sorted(y for y in range(y_top, y_bot + 1) for x in range(int(xc) - 12, int(xc) + 13)
                         if near(img[y][x], c))
            best = max(((sum(1 for y in ys_ if y0 <= y < y0 + 26), y0) for y0 in ys_), default=None)
            if best is None:
                vals.append(None)
                continue
            sel = [y for y in ys_ if best[1] <= y < best[1] + 26]
            yc = sum(sel) / len(sel)
            vals.append(round((y_zero - yc) / px_per_100 * 100, 1))
        out[name] = dict(zip(CTX, vals))
    return {"source": f"examples/deepseek_mla/figures/bs{bs}_float16.png", "meaning": "TFLOPS (marker centres)",
            "tflops": out, "calibration": {"grid_rows_px": gr, "px_per_100_tflops": px_per_100, "y_zero": y_zero}}


def main():
    res = {"fa": fa_figure(), "mla": {str(bs): mla_figure(bs) for bs in (64, 128)}}
    ratios = {}
    for bs, d in res["mla"].items():
        t = d["tflops"]
        ratios[bs] = {ctx: {
            "TileLang_over_FlashMLA": round(t["TileLang"][ctx] / t["FlashMLA"][ctx], 3),
            "TileLang_over_FlashInfer": round(t["TileLang"][ctx] / t["FlashInfer"][ctx], 3),
            "TileLang_over_Triton": round(t["TileLang"][ctx] / t["Triton"][ctx], 2)} for ctx in CTX}
    res["mla_ratios"] = ratios
    C.save_json(res, os.path.join(C.RESULTS, "claims_digitized.json"))
    print(json.dumps(res["fa"]["ratios"], indent=0))
    print("geomean", res["fa"]["geomean"])
    for bs in ratios:
        for ctx, r in ratios[bs].items():
            print(bs, ctx, {n: res["mla"][bs]["tflops"][n][ctx] for n in MLA_COLORS}, r)


if __name__ == "__main__":
    main()
