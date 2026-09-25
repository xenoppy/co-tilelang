"""Resident CTAs/SM of the TileLang kernels in the warp-specialisation diagnostic, from the CUDA occupancy
API (no timing). For each (shape, top-k config, {shipped, no_ws}) of diag/<shape>.json the generated CUDA
source is recompiled to a cubin with the options tilelang/cuda/backend.py uses (claims_mamba_diag.
cubin_resources), loaded with the driver API, and cuOccupancyMaxActiveBlocksPerMultiprocessor is asked
with the kernel's real block size and dynamic smem (from the generated host code).

    source research/env.sh
    python research/bench/scripts/claims_mamba_occupancy.py --compile-only     # CPU: cubins
    flock $GPU_LOCK python research/bench/scripts/claims_mamba_occupancy.py    # occupancy queries only

Writes C_mamba/diag/occupancy.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_mamba_common as C  # noqa: E402

CUBIN_DIR = os.path.join(C.RESULTS, "diag", "cubins")


def build(shape_name, j, vname):
    import claims_mamba_diag as D
    from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
    d = json.load(open(os.path.join(C.RESULTS, "diag", f"{shape_name}.json")))
    v = d["variants"].get(f"top{j}_{vname}")
    if not v or "config" not in v:
        return None
    k = D.compile_variant(C.SHAPES[shape_name], v["config"], D.VARIANTS[vname])
    li = D.launch_info(k)
    src = k.get_kernel_source()
    fast = bool((getattr(k.adapter, "pass_configs", None) or {}).get("tl.enable_fast_math", False))
    os.makedirs(CUBIN_DIR, exist_ok=True)
    base = os.path.join(CUBIN_DIR, f"{shape_name}_top{j}_{vname}")
    with open(base + ".cu", "w") as f:
        f.write(src)
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cmd = [os.path.join(cuda, "bin", "nvcc"), "-cubin", "-arch=sm_120a", "-std=c++20", "-I" + TILELANG_TEMPLATE_PATH,
           "-I" + CUTLASS_INCLUDE_DIR, "-o", base + ".cubin", base + ".cu"] + (["--use_fast_math"] if fast else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return dict(error=r.stderr[-300:])
    name = [ln.split("(")[0].split()[-1] for ln in src.splitlines() if ln.startswith('extern "C" __global__')][0]
    return dict(shape=shape_name, top=j, variant=vname, config=v["config"], cubin=base + ".cubin", func=name,
                block=li["block"], dyn_smem=li["dyn_smem"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()
    shapes = ["CC1", "CC4", "B8192", "CT1", "CT4"]
    meta_path = os.path.join(CUBIN_DIR, "meta.pkl")
    if a.compile_only:
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(s, j, v) for s in shapes for j in range(4) for v in ("shipped", "no_ws")]
        with ProcessPoolExecutor(8) as ex:
            metas = [m for m in ex.map(build, *zip(*jobs)) if m]
        pickle.dump(metas, open(meta_path, "wb"))
        print(len(metas), "cubins;", sum(1 for m in metas if "error" in m), "errors")
        return
    from cuda.bindings import driver as cu
    import torch
    torch.cuda.init()
    torch.empty(1, device="cuda")  # primary context current
    metas = pickle.load(open(meta_path, "rb"))
    out = []
    for m in metas:
        if "error" in m:
            out.append(m)
            continue
        err, mod = cu.cuModuleLoadData(open(m["cubin"], "rb").read())
        assert int(err) == 0, err
        err, fn = cu.cuModuleGetFunction(mod, m["func"].encode())
        assert int(err) == 0, err
        err, = cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, m["dyn_smem"])
        err2, occ = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, m["block"][0], m["dyn_smem"])
        _, regs = cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)
        _, sstat = cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn)
        out.append(dict(m, set_attr_err=int(err), occ_err=int(err2), ctas_per_sm=int(occ), regs=int(regs), static_smem=int(sstat)))
        cu.cuModuleUnload(mod)
        print(m["shape"], m["top"], m["variant"], m["config"], "block", m["block"][0], "dyn", m["dyn_smem"], "static", sstat,
              "regs", regs, "-> CTAs/SM", occ, "(set_attr err %d)" % int(err))
    C.dump(out, os.path.join(C.RESULTS, "diag", "occupancy.json"))


if __name__ == "__main__":
    main()
