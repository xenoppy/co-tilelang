#!/usr/bin/env bash
# Mamba-2 claims sub-study (C3 / C6-mamba): compile -> autotune -> bench, one shape at a time.
#
#   GPU_LOCK=/path/to/gpu.lock bash research/bench/scripts/claims_mamba_run.sh [SHAPE ...]
#
# Default shapes: CC0-CC5 CT0-CT5 B1024-B32768 (see claims_mamba_common.SHAPES).
# TUNE_ARGS="--filter-ref CC0" drops configs that cannot launch on sm_120 before tuning (claims_mamba_tune.py).
# Used as: CC0-CC4, CT0, B1024 without filter; CC5, B2048-B32768 with --filter-ref CC0; CT1-CT5 with --filter-ref CT0.
# compile: CPU only, no lock. tune + bench: under the GPU-sharing guard (run_guarded.py: foreign SM
# activity -> wait 30 min; blocked > 2 h -> exit 3) and one hold of the shared GPU flock.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT/research/env.sh"
: "${GPU_LOCK:?set GPU_LOCK to the shared flock file}"
LOGS="${LOGS:-$ROOT/research/results/2026-09-24_claims_repro/C_mamba/logs}"
mkdir -p "$LOGS"
S="$ROOT/research/bench/scripts"
SHAPES=("$@")
[ ${#SHAPES[@]} -eq 0 ] && SHAPES=(CC0 CC1 CC2 CC3 CC4 CC5 CT0 CT1 CT2 CT3 CT4 CT5 B1024 B2048 B4096 B8192 B16384 B32768)
for s in "${SHAPES[@]}"; do
  if [ "${SKIP_COMPILE:-0}" != 1 ]; then
    python "$S/claims_mamba_compile.py" --shape "$s" --workers "${WORKERS:-16}" > "$LOGS/compile_$s.log" 2>&1
    grep -a "\[compile" "$LOGS/compile_$s.log" || true
  fi
  # tune + bench share one lock hold (~1-3 min) to halve the waits on the shared lock
  if [ "${SKIP_TUNE:-0}" != 1 ]; then
    python "$S/run_guarded.py" -- flock "$GPU_LOCK" bash -c \
      "python '$S/claims_mamba_tune.py' --shape '$s' ${TUNE_ARGS:-} > '$LOGS/tune_$s.log' 2>&1 && python '$S/claims_mamba_bench.py' --shape '$s' > '$LOGS/bench_$s.log' 2>&1"
    grep -a "\[tune" "$LOGS/tune_$s.log" || true
  else
    python "$S/run_guarded.py" -- flock "$GPU_LOCK" python "$S/claims_mamba_bench.py" --shape "$s" > "$LOGS/bench_$s.log" 2>&1
  fi
  grep -a "\] TL " "$LOGS/bench_$s.log" || true
done
