"""Run examples/flash_decoding/example_gqa_decode.py:main() with an explicit config.

Why: the example's get_heuristic_config() only special-cases sm_89; every other
GPU gets block_N=128, block_H=64, num_split=8, num_stages=2, which needs 144 KB
of dynamic shared memory per CTA (Q 16 KB + double-buffered K/V 2x2x32 KB).
sm_120 (like sm_86/sm_89) allows at most 99 KB (101376 B) per CTA, so the
unmodified example fails at launch with
"Failed to set the allowed dynamic shared memory size to 147456".

This wrapper leaves the example file untouched: it replaces the module-level
get_heuristic_config() with one that returns the config given on the command
line, then calls the example's own main(), so the example's built-in
correctness checks (vs. ref_program and ref_split_program) and benchmark run
unchanged.

Usage:
  python run_gqa_decode.py [--block_N 64] [--block_H 64] [--num_split 8]
                           [--num_stages 2] [--threads 128] [example main() args]
"""

import argparse
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "examples", "flash_decoding"))

import example_gqa_decode as ex  # noqa: E402


def smem_bytes_estimate(block_N, block_H, dim, num_stages, dtype_bytes=2):
    """Rough dynamic-smem need of flashattn_gqa_decode_split (Q + multi-buffered K/V)."""
    stages = max(num_stages, 1)
    return dtype_bytes * (block_H * dim + 2 * stages * block_N * dim)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--block_N", type=int, default=64)
    p.add_argument("--block_H", type=int, default=64)
    p.add_argument("--num_split", type=int, default=8)
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--kv_seqlen", type=int, default=8192)
    p.add_argument("--dim", type=int, default=128)
    return p.parse_args()


def main():
    a = parse()
    cfg = dict(block_N=a.block_N, block_H=a.block_H, num_split=a.num_split, num_stages=a.num_stages, threads=a.threads)
    props = torch.cuda.get_device_properties(0)
    need = smem_bytes_estimate(a.block_N, a.block_H, a.dim, a.num_stages)
    print(f"device={props.name} cc={props.major}.{props.minor} smem_optin_per_block={props.shared_memory_per_block_optin}")
    print(f"config={cfg} est_dynamic_smem={need} B")
    sm_version = props.major * 10 + props.minor

    def heuristic():
        return cfg, sm_version

    ex.get_heuristic_config = heuristic
    ex.main(a.batch, a.heads, a.groups, a.kv_seqlen, a.dim, tune=False)


if __name__ == "__main__":
    main()
