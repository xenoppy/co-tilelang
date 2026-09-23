"""Stage K of p1_study.py: does the shared-memory carveout change two-stream co-residence?

An SM's L1/shared-memory carveout is chosen when a CTA is placed on an idle SM and can change
only while the SM is idle (methodology v1, §M4). A kernel whose CTAs need little smem may leave
SMs with a small carveout on which a large-smem partner cannot start. TileLang's default
(tvm_ffi) launch path exposes no CUfunction handle, so the attribute cannot be set there; the
NVRTC backend loads the same generated CUDA source with cuLibraryLoadData and exposes CUkernel
handles, on which cuKernelSetAttribute(PREFERRED_SHARED_MEMORY_CARVEOUT) works. Every kernel is
built twice under distinct symbols (driver default vs 100 %), so both settings can be interleaved.

Pairs: the solo-best configs, and the lib pair with the best I1 two-stream screen among the pairs
whose CTAs can co-reside on one SM (smem_a + smem_b + 2 KB reserve <= 100 KB, registers and
threads within the SM). Variants: streams with priority {eq, A-high, B-high} x host order {AB, BA}.
"""
from __future__ import annotations

import time

import tilelang
from cuda.bindings import driver as cu

import p1_common as C
from p1_common import log
from cotile.device import DEFAULT_DEVICE

SM_SMEM = 100 * 1024
SM_REGS = 64 * 1024
SM_THREADS = 1536


def nvrtc_kernel(spec, suffix: str):
    pf = spec.prim_func.with_attr("global_symbol", spec.name + suffix)
    return tilelang.compile(pf, target=DEFAULT_DEVICE.target, pass_configs=dict(spec.pass_configs),
                            execution_backend="nvrtc")


def set_carveout(kernel, pct: int):
    for name, h in kernel.adapter.kernels.items():
        (res,) = cu.cuKernelSetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                         int(pct), h, cu.CUdevice(0))
        if res != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuKernelSetAttribute(carveout={pct}) on {name}: {res}")


def co_residable(sa: dict, sb: dict) -> bool:
    return (sa["smem"] + sb["smem"] + 2048 <= SM_SMEM and sa["regs"] * sa["threads"] + sb["regs"] * sb["threads"] <= SM_REGS
            and sa["threads"] + sb["threads"] <= SM_THREADS)


def carveout_study(st) -> dict:
    t0 = time.time()
    from cotile import catalog
    cat = catalog.load(C.SOLO_DIR)
    ea, eb = cat.get("gemm", st.A), cat.get("gqa_decode", st.B)
    sig = lambda e, t: e.configs[t].sig["grid"]  # noqa: E731
    pairs = {"solo": (st.solo["a"], st.solo["b"])}
    i1 = st.res["stages"].get("I1", {}).get("flush", {})
    cands = []
    for ta in st.lib["a"]:
        for tb in st.lib["b"]:
            if co_residable(sig(ea, ta), sig(eb, tb)):
                n = f"st_eq_alt_{st.id['a'][ta]}{st.id['b'][tb]}"
                cands.append((-(i1.get(n, {}).get("speedup") or 0.0), ta, tb))
    if cands:
        cands.sort()
        pairs["coresident"] = (cands[0][1], cands[0][2])
    names, info = [], {"pairs": {k: {"a": a, "b": b, "sig_a": sig(ea, a), "sig_b": sig(eb, b),
                                      "co_residable": co_residable(sig(ea, a), sig(eb, b))} for k, (a, b) in pairs.items()},
                       "n_coresidable_lib_pairs": len(cands)}
    for pk, (ta, tb) in pairs.items():
        for cv in ("def", "max"):
            ka = nvrtc_kernel(st.grid["a"][ta], f"_cv{cv}")
            kb = nvrtc_kernel(st.grid["b"][tb], f"_cv{cv}")
            if cv == "max":
                set_carveout(ka, 100)
                set_carveout(kb, 100)
            ala, alb = st.da.arglists(st.grid["a"][ta]), st.db.arglists(st.grid["b"][tb])
            fa = (lambda k, al: (lambda i: k(*al[i % len(al)])))(ka, ala)
            fb = (lambda k, al: (lambda i: k(*al[i % len(al)])))(kb, alb)
            s = st.streams
            for prio, (sa, sb) in {"eq": (s["s1"], s["s2"]), "pA": (s["hi1"], s["lo2"]), "pB": (s["lo1"], s["hi2"])}.items():
                for order in ("ab", "ba"):
                    n = f"K_{pk}_{cv}_{prio}_{order}"
                    st.add(n, C.cb.Par(("a", sa, fa), ("b", sb, fb), order="given" if order == "ab" else "reverse"),
                           kind="streams_nvrtc", row="K", a=ta, b=tb, prio=prio, order=order, carveout=cv, pair=pk)
                    names.append(n)
            # nvrtc serial with the default build as a same-backend reference
            if cv == "def":
                def ser(i, fa=fa, fb=fb):
                    fa(i)
                    fb(i)
                st.add(f"K_{pk}_serial_nvrtc", ser, kind="serial_nvrtc", row="K", a=ta, b=tb, pair=pk)
                names.append(f"K_{pk}_serial_nvrtc")
    # tvm_ffi two-stream anchors (the study's own solo streams)
    names += [st.v_streams(st.solo["a"], st.solo["b"], p, o) for p in ("eq", "pA", "pB") for o in ("ab", "ba")]
    r, meta = st.steady(names, "K")
    rf = st.flush(names, "K-flush", reps=50, group=len(names))
    log(f"K done in {time.time() - t0:.0f}s")
    return {"steady": r, "meta": meta, "flush": rf, "desc": st.desc(names), "info": info, "wall_s": time.time() - t0}
