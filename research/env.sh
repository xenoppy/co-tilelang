# co-tilelang developer environment (source this file; do not execute it).
#
#   source research/env.sh
#   python -c "import tilelang; print(tilelang.__version__)"
#
# Uses the in-place developer build documented in
# docs/get_started/Installation.md ("Working from Source via PYTHONPATH"):
# the source-tree `tilelang` package is put on PYTHONPATH, and tilelang/env.py
# then loads the native libraries from <repo>/build/lib and <repo>/build/tvm.
# Only documented variables are set: PYTHONPATH (Installation.md) and
# CUDA_HOME (tilelang/env.py). Rebuild the C++ side with:
#   cmake --build "$CO_TILELANG_ROOT/build"      (ninja, see research/env_versions.md)

# Repo root = parent of the directory containing this file (works from any cwd).
CO_TILELANG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export CO_TILELANG_ROOT

# CUDA 12.9 toolkit (not on the default PATH on this machine). TileLang JIT-compiles
# kernels with $CUDA_HOME/bin/nvcc; ncu/nsys also live here.
export CUDA_HOME=/usr/local/cuda-12.9

# Python env with torch 2.8.0+cu128 / triton 3.4 (also provides ninja for rebuilds).
MPK_ENV=/home/ywc/mpk-env

case ":$PATH:" in
  *":$CUDA_HOME/bin:"*) ;;
  *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac
case ":$PATH:" in
  *":$MPK_ENV/bin:"*) ;;
  *) export PATH="$MPK_ENV/bin:$PATH" ;;
esac

# Source-tree tilelang (dev mode: libs come from $CO_TILELANG_ROOT/build).
case ":${PYTHONPATH:-}:" in
  *":$CO_TILELANG_ROOT:"*) ;;
  *) export PYTHONPATH="$CO_TILELANG_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
