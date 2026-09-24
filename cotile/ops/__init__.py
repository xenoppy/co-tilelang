"""Op library v0. Each module follows the protocol documented in cotile/kernel.py."""

from . import gemm, gqa_decode, prefill_attn, rmsnorm  # noqa: F401

OPS = {m.NAME: m for m in (gemm, gqa_decode, rmsnorm, prefill_attn)}
