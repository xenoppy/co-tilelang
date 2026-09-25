"""Digitize the claimed per-shape speed-ups from the upstream bar charts (claims C1, C5).

The upstream repos ship no per-shape numbers for these figures (tilelang-benchmark's
data_*.py files parse logs that were never committed), so the claimed ratios are read off
the PNGs. Calibration: the y tick marks left of the left spine (0, step, 2*step, ...) give a
least-squares pixel->value map; the dashed cuBLAS line (value 1) is reported as a
cross-check. Each bar's top edge is found per pixel column, bars are segmented by fill
colour and grouped by the white gaps between shape groups. Pure numpy + zlib (no PIL in
~/mpk-env).

    python research/bench/scripts/claims_gemm_figures.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import zlib

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def read_png(path: str) -> np.ndarray:
    """Minimal PNG decoder: 8-bit gray/RGB/RGBA, non-interlaced. Returns HxWx3 uint8."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, idat, hdr = 8, [], None
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif typ == b"IDAT":
            idat.append(body)
        elif typ == b"IEND":
            break
        pos += 12 + ln
    w, h, depth, ctype, _, _, interlace = hdr
    assert depth == 8 and interlace == 0 and ctype in (0, 2, 6), hdr
    bpp = {0: 1, 2: 3, 6: 4}[ctype]
    raw = zlib.decompress(b"".join(idat))
    stride = w * bpp
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    p = 0
    for y in range(h):
        ft = raw[p]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=p + 1).astype(np.int32)
        p += 1 + stride
        if ft == 0:
            cur = line
        elif ft == 2:
            cur = (line + prev) & 0xFF
        else:  # filters 1, 3, 4 depend on the left neighbour: byte by byte
            cur = np.zeros(stride, dtype=np.int32)
            for x in range(stride):
                a = cur[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                if ft == 1:
                    pr = a
                elif ft == 3:
                    pr = (a + b) >> 1
                else:
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[x] = (line[x] + pr) & 0xFF
        out[y] = cur
        prev = cur
    img = out.reshape(h, w, bpp)
    if bpp == 1:
        img = np.repeat(img, 3, axis=2)
    return img[:, :, :3]


def axes_frame(img):
    """(left, right, top, band_top, ticks). Horizontal spines and the dashed line are rows
    that are > 60 % dark; band_top = first row of the bottom spine band; left/right spines =
    columns > 95 % dark between the spines; y ticks = short dark marks 5 px left of the
    left spine (centres, bottom first)."""
    dark = img.max(axis=2) < 60
    h, w = dark.shape
    rows = np.where(dark.sum(axis=1) > 0.6 * w)[0]
    rowset = set(rows.tolist())
    top, bottom = int(rows.min()), int(rows.max())
    band_top = bottom
    while band_top - 1 in rowset:
        band_top -= 1
    frac = dark[top + 2:band_top - 1].mean(axis=0)
    vert = np.where(frac > 0.95)[0]
    left, right = int(vert.min()), int(vert.max())
    ys = np.where(dark[:, left - 5])[0]
    groups = []
    for y in ys:
        if groups and y - groups[-1][-1] <= 1:
            groups[-1].append(int(y))
        else:
            groups.append([int(y)])
    ticks = sorted((float(np.mean(g)) for g in groups if len(g) <= 8), reverse=True)
    return left, right, top, band_top, ticks


def bar_columns(img, left, right, top, band_top, white=235):
    """Per column: top edge row of the non-white run standing on the bottom spine, and its
    fill colour (median of the non-dark pixels of the run)."""
    wmask = img.min(axis=2) >= white
    res = []
    for x in range(left + 3, right - 2):
        y = band_top - 1
        if wmask[y, x]:
            res.append((x, None, None))
            continue
        while y > top and not wmask[y, x]:
            y -= 1
        top_edge = y + 1
        run = img[top_edge + 4:band_top - 1, x].astype(int)
        light = run[run.max(axis=1) >= 90]
        col = tuple(int(v) for v in np.median(light, axis=0)) if len(light) else (0, 0, 0)
        res.append((x, top_edge, col))
    return res


def segment_bars(cols, min_width=6, color_tol=20):
    """Group consecutive columns into bars by fill colour; drop outline-only runs (black)
    and slivers."""
    bars, cur = [], []

    def flush():
        if len(cur) >= min_width:
            tops = [c[1] for c in cur]
            cs = np.array([c[2] for c in cur])
            inner = slice(2, len(cur) - 2) if len(cur) > 6 else slice(0, len(cur))
            color = tuple(int(v) for v in np.median(cs[inner], axis=0))
            if max(color) >= 90:
                bars.append({"x0": cur[0][0], "x1": cur[-1][0], "tops": list(tops[inner]),
                             "color": color})

    for c in cols:
        if c[1] is None:
            flush()
            cur.clear()
            continue
        if cur and max(abs(a - b) for a, b in zip(c[2], cur[-1][2])) > color_tol:
            flush()
            cur.clear()
        cur.append(c)
    flush()
    # a hatch line running vertically through a bar (e.g. the '+' hatch) splits it: re-join
    # neighbours with the same colour and top edge separated by a few outline columns
    merged = []
    for b in bars:
        m = merged[-1] if merged else None
        if (m and b["x0"] - m["x1"] <= 6 and
                max(abs(p - q) for p, q in zip(b["color"], m["color"])) <= color_tol):
            m["x1"] = b["x1"]
            m["tops"] += b["tops"]
        else:
            merged.append(dict(b))
    # A column can only over-estimate a bar's height (dark marks sitting on its top, e.g.
    # the dashed y=1 line when the bar ends just below it); never under-estimate (hatches
    # have no white). Hence the bar top = the 80th percentile of the per-column top rows.
    for m in merged:
        m["top"] = float(np.percentile(m.pop("tops"), 80))
    return merged


def digitize(path, n_groups, series, group_labels, tick_step):
    img = read_png(path)
    left, right, top, band_top, ticks = axes_frame(img)
    vals = np.arange(len(ticks)) * tick_step
    slope, icpt = np.polyfit(vals, np.array(ticks), 1)          # y = icpt + slope * value
    resid = float(np.max(np.abs(icpt + slope * vals - np.array(ticks))))
    bars = sorted(segment_bars(bar_columns(img, left, right, top, band_top)), key=lambda b: b["x0"])
    groups, g = [], [bars[0]]
    for b in bars[1:]:
        if b["x0"] - g[-1]["x1"] > 15:
            groups.append(g)
            g = [b]
        else:
            g.append(b)
    groups.append(g)
    assert len(groups) == n_groups, f"{path}: found {len(groups)} groups, expected {n_groups}"
    out = {}
    for lab, g in zip(group_labels, groups):
        assert len(g) == len(series), f"{path} {lab}: {len(g)} bars, expected {len(series)}: {g}"
        # the top edge outline is ~3 px thick and centred on the value: +1 px to its centre
        out[lab] = {s: round((b["top"] + 1.0 - icpt) / slope, 3) for s, b in zip(series, g)}
        out[lab]["_colors"] = [b["color"] for b in g]
    dark = img.max(axis=2) < 60
    rows = [y for y in range(top + 3, band_top - 3) if dark[y].sum() > 0.6 * img.shape[1]]
    meta = {"frame_px": [left, right, top, band_top], "ticks_px": ticks, "tick_step": tick_step,
            "px_per_unit": float(-slope), "tick_fit_max_resid_px": resid,
            "dashed_line_value": float((np.mean(rows) - icpt) / slope) if rows else None,
            "resolution_units": float(1.0 / -slope), "size": list(img.shape[:2])}
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    gemm_series = ["Triton-RTX4090", "TileLang-RTX4090", "Triton-A100", "TileLang-A100",
                   "Triton-H100", "TileLang-H100", "Triton-MI300X", "TileLang-MI300X"]
    d, meta = digitize(os.path.join(REPO, "images/op_benchmark_consistent_gemm_fp16.png"), 8,
                       gemm_series, [f"M{i}" for i in range(8)], 1.0)
    res["C1_gemm_fp16"] = {"figure": "images/op_benchmark_consistent_gemm_fp16.png", "meta": meta,
                           "speedup_vs_cublas": d}
    gemv_series = ["CUTLASS-W_INT4A_FP16", "Marlin-W_INT4A_FP16", "BitsAndBytes-W_NF4A_FP16",
                   "BitBLAS-TileLang-W_INT4A_FP16", "BitBLAS-TileLang-W_INT2A_FP16",
                   "BitBLAS-TileLang-W_INT2A_INT8", "BitBLAS-TileLang-W_NF4A_FP16"]
    d, meta = digitize(os.path.join(REPO, "images/op_benchmark_a100_wq_gemv.png"), 7,
                       gemv_series, [f"V{i}" for i in range(7)], 2.0)
    res["C5_gemv_a100"] = {"figure": "images/op_benchmark_a100_wq_gemv.png", "meta": meta,
                           "speedup_vs_cublas_fp16": d}
    for k, v in res.items():
        print(k, v["meta"])
        tab = v.get("speedup_vs_cublas") or v.get("speedup_vs_cublas_fp16")
        for lab, row in tab.items():
            print(" ", lab, " ".join(f"{s}={x:.3f}" for s, x in row.items() if not s.startswith("_")))
            print("    colors", row["_colors"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
