"""Content stamp of the device-side template headers that generated CUDA sources include.

The kernel caches (``kernel_cache.py``, ``cuda_binary_cache.py``) key entries on the
generated source and, with ``TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1``, on a content hash of
the native TileLang libraries. The generated source only *includes* the ``tl_templates``
headers (``#include <tl_templates/cuda/copy.h>``), so an edit to a header changes the
compiled binary without changing either key: a stale cubin compiled against the old header
would be reused. The lib stamp therefore also covers every file of the template tree.
"""

from __future__ import annotations

import functools
import os
from hashlib import sha256

TEMPLATE_SUFFIXES = (".h", ".hpp", ".cuh", ".inl")


@functools.cache
def template_stamp() -> str | None:
    """``tl_templates:<sha256>`` over the relative path and content of every header below
    ``<TILELANG_TEMPLATE_PATH>/tl_templates`` (sorted, so the stamp is machine independent),
    or None if the template tree cannot be found."""
    try:
        from tilelang import env
    except Exception:  # noqa: BLE001 - no env, no stamp
        return None
    base = getattr(env, "TILELANG_TEMPLATE_PATH", None)
    if not base:
        return None
    root = os.path.join(str(base), "tl_templates")
    if not os.path.isdir(root):
        return None
    files = []
    for d, _, names in os.walk(root):
        for n in names:
            if n.endswith(TEMPLATE_SUFFIXES):
                p = os.path.join(d, n)
                files.append((os.path.relpath(p, root), p))
    if not files:
        return None
    h = sha256()
    for rel, p in sorted(files):
        h.update(rel.encode())
        h.update(b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return f"tl_templates:{h.hexdigest()}"
