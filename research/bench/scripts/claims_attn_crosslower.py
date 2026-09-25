"""CPU-only: lower the shipped TileLang configs of C2 / C4 for sm_90a (H100) and sm_120a (this GPU) and
compare dynamic shared memory and the MMA instruction family in the generated CUDA source. Shows that
the 99 KB problem is a property of the configs (same bytes on both targets) and that the H100 build
uses wgmma where the sm_120 build uses mma.sync; block_N=32 MLA tiles compile only on sm_90a.

  python research/bench/scripts/claims_attn_crosslower.py   # -> B_attention/crosslower.json
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_attn_common as C  # noqa: E402


def lower_info(func, arch):
    import tilelang
    import tvm
    target = {"kind": "cuda", "arch": arch}
    with tvm.transform.PassContext(config={"tl.enable_fast_math": True}), tvm.target.Target(target):
        art = tilelang.lower(func, target=target)
    dyn = {}
    for gv, f in art.device_mod.functions.items():
        a = f.attrs
        dyn[gv.name_hint] = int(a["dyn_shared_memory_buf"]) if a is not None and "dyn_shared_memory_buf" in a else None
    src = art.kernel_source
    return {"dyn_smem": dyn, "wgmma": "wgmma" in src, "mma_sync": bool(re.search(r"mma_sync|tl::mma|gemm_ss|gemm_rs", src)),
            "tma": "tma_load" in src, "launch_bounds": re.findall(r"__launch_bounds__\((\d+)", src)}


def main():
    import claims_attn_fa as FA
    import claims_attn_mla as MLA
    exb, _ = FA.import_examples()
    out = {}
    sh = FA.SHAPES["FA2"]
    for arch in ("sm_90a", "sm_120a"):
        f = exb.flashattn.jit_impl.get_tir(sh["batch"], sh["heads"], sh["seq"], sh["seq"], sh["dim"], sh["causal"],
                                           block_M=128, block_N=128, num_stages=2, threads=256)
        out[f"FA bhsd 128x128 2-stage 256thr ({arch})"] = lower_info(f, arch)
    msh = MLA.shape_of(64, 1024)
    P = MLA.example_module()
    for arch in ("sm_90a", "sm_120a"):
        for bn, bh in ((64, 64), (32, 64), (64, 16)):
            key = f"MLA paged block_N={bn} block_H={bh} 2-stage ({arch})"
            try:
                f = P.mla_decode_tilelang.get_tir(msh["batch"], MLA.H_Q, MLA.H_KV, msh["max_seqlen_pad"], MLA.DV, MLA.DPE,
                                                  bn, bh, 1, MLA.BLOCK_SIZE)
                out[key] = lower_info(f, arch)
            except Exception as e:  # noqa: BLE001 - recorded
                out[key] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    C.save_json(out, os.path.join(C.RESULTS, "crosslower.json"))
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
