"""Vendored subset of mamba-ssm 2.2.6.post3 (Tri Dao, Albert Gu; Apache-2.0, see ../LICENSE).

Only the pure-Triton SSD kernels are vendored, so that the Triton baseline used by TileLang's
Mamba-2 examples/benchmark (``mamba_ssm.ops.triton.ssd_chunk_scan._chunk_scan_fwd`` and
``mamba_ssm.ops.triton.ssd_chunk_state._chunk_state_fwd``) can be imported without building
mamba-ssm's CUDA extension and without its ``transformers`` / ``causal_conv1d`` dependencies.

Source: PyPI sdist ``mamba_ssm-2.2.6.post3.tar.gz`` (uploaded 2025-10-10,
sha256 826a3cdb651959f191dac64502f8a29627d9116fe6bb7c57e4f562da1aea7bf3). This is the version
named in ``benchmark/mamba2/README.md`` ("Triton: v3.5.0, mamba-ssm: v2.2.6.post3").

Files copied byte-for-byte from ``mamba_ssm/ops/``:
    ops/__init__.py, ops/triton/__init__.py (both empty),
    ops/triton/ssd_chunk_scan.py   md5 4fed822f9199858c37f5ed5db9a62448
    ops/triton/ssd_chunk_state.py  md5 4a0b1f1e52bd9902fd1ea249bede4830
    ops/triton/ssd_bmm.py          md5 e82f986caeb24385d7cf52ffa8d36714  (imported by ssd_chunk_scan)
    ops/triton/softplus.py         md5 3bbdcd7416588a1587e6c42b0eee24d8  (imported by ssd_chunk_state)

This file replaces upstream ``mamba_ssm/__init__.py``, which imports the compiled
``selective_scan_cuda`` extension and the model classes (not needed for the SSD kernels).
Use: ``sys.path.insert(0, "research/bench/baselines/mamba_ssm_triton")``.
"""

__version__ = "2.2.6.post3+vendored-triton-ssd"
