"""cotile: tile-program library and (later) CoKernel compiler on top of TileLang.

v0 contents: parameterized single-op tile programs (cotile.ops), generic grid /
persistent kernel builders (cotile.kernel) and compiled-kernel resource signatures
(cotile.resources). See cotile/README.md.
"""

from .device import DEFAULT_DEVICE, SM120, DeviceSpec  # noqa: F401
