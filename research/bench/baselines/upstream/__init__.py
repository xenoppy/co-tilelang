"""Verbatim copies of upstream baseline sources (not modified; imported by the claims scripts).

tlbench_triton_mha.py
    tile-ai/tilelang-benchmark @ b658f7e9f326156d11a09dff1e9825fa6d9a8767 (2026-06-10),
    hopper_benchmark/flashattention/2.triton_benchmark/benchmark_triton_mha.py (MIT).
    The Triton FlashAttention forward the TileLang authors benchmarked on H100 (C2): a copy of
    Triton's tutorial 06 with a TMA-descriptor path (used when capability >= 9, so also on sm_120)
    and warp_specialize=True in its benchmark. sha256 06b1ca80...fe6.

triton340_tutorial06_fused_attention.py
    triton-lang/triton tag v3.4.0, python/tutorials/06-fused-attention.py (MIT): the fused
    attention tutorial matching the installed triton 3.4.0. sha256 5f312a05...6a7.

Both files ``import pytest`` for test decorators; ~/mpk-env has no pytest, so the claims scripts
register a no-op stub first (claims_attn_common.install_pytest_stub).
"""
