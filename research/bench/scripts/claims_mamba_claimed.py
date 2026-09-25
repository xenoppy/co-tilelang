"""Recover the claimed Mamba-2 numbers from the upstream figures (claims C3 / C6-mamba). CPU only.

    source research/env.sh
    python research/bench/scripts/claims_mamba_claimed.py [--workdir /tmp/x]
    # the C6 figure is a JPEG; reading it needs Pillow (not in ~/mpk-env). Without it the C6 part
    # is skipped; with a throw-away install:  pip install --target /tmp/pil pillow; PYTHONPATH=/tmp/pil ...

C3: images/mha_performance_h100.png is a raster copy of Fig. 12 of arXiv:2504.17577; the arXiv source
    (e-print v2) ships the vector original figures/mha_performance_h100.pdf (matplotlib). Its content
    stream holds every bar as a rectangle; ratio = bar height / TileLang bar height of the same group
    (TileLang bars are exactly 1.0 axis units, checked against the y tick "2").
C6: benchmark/mamba2/README.md gives TileLang latency/TFLOPS per seq_len (exact). The Triton and
    Helion bars of benchmark/mamba2/mamba_benchmark_result.png (a JPEG despite the name) are read
    by pixel colour; the pixel->TFLOPS map is fitted on the six TileLang bars against the README values.
Writes research/results/2026-09-24_claims_repro/C_mamba/claimed.json.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tarfile
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402

ARXIV = "https://arxiv.org/e-print/2504.17577v2"


def c3_from_pdf(pdf: bytes) -> dict:
    page = None
    for m in re.finditer(rb"(\d+) 0 obj\s*<<(.*?)>>\s*stream\r?\n", pdf, re.S):
        s = m.end()
        e = pdf.find(b"endstream", s)
        try:
            t = zlib.decompress(pdf[s:e]).decode("latin1")
        except zlib.error:
            continue
        if "BT" in t and "/Pattern cs" in t:  # the page stream (text + hatched bars); pattern tiles have no text
            page = t
            break
    assert page is not None, "page content stream not found"
    bars = re.findall(r"/Pattern cs /(H\d) scn\s+([\d.]+) ([\d.]+) m\s+([\d.]+) [\d.]+ l\s+[\d.]+ ([\d.]+) l", page)
    legend = {"H1": "TileLang", "H2": "FlashAttention-3", "H3": "Triton", "H4": "PyTorch"}
    panels = {}
    for h, x0, y0, x1, y1 in bars:
        panels.setdefault(float(y0), []).append((legend[h], float(x0), float(y1) - float(y0)))
    ticks = {float(y): lab for y, lab in re.findall(
        r"37\.713813 ([\d.]+) m\s+34\.213813 [\d.]+ l\s+B\s+1 w\s+q\s+1 0 -0 1 [\d.]+ [\d.]+ cm\s+BT\s+/F1 10 Tf\s+0 0 Td\s+\[ \((\d)\) \] TJ",
        page)}
    out = {}
    names = iter(["FA", "CC", "CT"])  # panels top to bottom
    for y0 in sorted((y for y in panels if any(r[0] == "TileLang" for r in panels[y])), reverse=True):
        tag = next(names)
        rows = panels[y0]
        tl = sorted((r for r in rows if r[0] == "TileLang"), key=lambda r: r[1])
        two = [y for y, lab in ticks.items() if lab == "2" and 0 < y - y0 < 60]
        unit = (two[0] - y0) / 2 if two else None
        entry = {"tilelang_height_units": [round(r[2] / unit, 4) for r in tl] if unit else None}
        for name in ("FlashAttention-3", "Triton", "PyTorch"):
            rr = sorted((r for r in rows if r[0] == name), key=lambda r: r[1])
            if rr:
                entry[name] = [round(r[2] / tl[i][2], 3) for i, r in enumerate(rr)]
        out[tag] = entry
    return out


def c6_from_jpeg(path: str) -> dict | None:
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    img = np.asarray(Image.open(path).convert("RGB")).astype(int)
    colours = {"TileLang": (66, 133, 244), "Triton": (234, 67, 53), "Helion": (251, 188, 4)}
    tops = {}
    for name, c in colours.items():
        m = np.abs(img - np.array(c)).sum(-1) < 60
        xs = np.where(m.sum(0) > 30)[0]
        runs, start = [], xs[0]
        for a, b in zip(xs[:-1], xs[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, xs[-1]))
        runs = [r for r in runs if r[1] - r[0] > 10]
        tops[name] = [float(np.where(m[:, x0 + 3:x1 - 2].mean(1) > 0.5)[0].min()) for x0, x1 in runs]
    readme_tl = [126.477, 130.195, 133.054, 134.362, 135.711, 135.379]
    A = np.vstack([tops["TileLang"], np.ones(6)]).T
    (a, b), *_ = np.linalg.lstsq(A, np.array(readme_tl), rcond=None)
    fit_res = float(np.abs(A @ np.array([a, b]) - readme_tl).max())
    val = {n: [round(float(a * t + b), 1) for t in tops[n]] for n in tops}
    return dict(seq_len=C.C6_SEQ, tilelang_tflops_readme=readme_tl, tilelang_latency_ms_readme=[0.169, 0.329, 0.645, 1.278, 2.531, 5.076],
                triton_tflops_read=val["Triton"], helion_tflops_read=val["Helion"],
                tilelang_tflops_read=val["TileLang"], fit_max_residual_tflops=round(fit_res, 2),
                ratio_tilelang_over_triton=[round(t / r, 3) for t, r in zip(readme_tl, val["Triton"])],
                ratio_tilelang_over_helion=[round(t / r, 3) for t, r in zip(readme_tl, val["Helion"])],
                platform="H800 SXM (README) / figure caption says H100", versions="Triton 3.5.0, mamba-ssm 2.2.6.post3, Helion 0.2.1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp")
    a = ap.parse_args()
    tgz = os.path.join(a.workdir, "arxiv_2504.17577v2.tgz")
    if not os.path.exists(tgz):
        urllib.request.urlretrieve(ARXIV, tgz)
    with tarfile.open(tgz) as tf:
        pdf = tf.extractfile("figures/mha_performance_h100.pdf").read()
        exp = tf.extractfile("Experiment.tex").read().decode()
    c3 = c3_from_pdf(pdf)
    text = re.search(r"Linear Attention Performance\.\}(.*?)\n", exp).group(1).strip()
    res = dict(
        c3=dict(source=f"{ARXIV} figures/mha_performance_h100.pdf (vector original of Fig. 12 = images/mha_performance_h100.png)",
                chunk_scan_triton_over_tilelang=c3["CC"]["Triton"], chunk_state_triton_over_tilelang=c3["CT"]["Triton"],
                labels_scan=["CC0", "CC1", "CC2", "CC3", "CC4"], labels_state=["CT0", "CT1", "CT2", "CT3", "CT4"],
                mean_scan=round(sum(c3["CC"]["Triton"]) / 5, 3), mean_state=round(sum(c3["CT"]["Triton"]) / 5, 3),
                paper_text=text, flashattention_panel=c3["FA"]),
        c6=c6_from_jpeg(os.path.join(C.ROOT, "benchmark/mamba2/mamba_benchmark_result.png")),
    )
    C.dump(res, os.path.join(C.RESULTS, "claimed.json"))
    print(res)


if __name__ == "__main__":
    main()
