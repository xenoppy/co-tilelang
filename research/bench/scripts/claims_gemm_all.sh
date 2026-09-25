#!/usr/bin/env bash
# Sub-study A of the claims reproduction (research/results/2026-09-24_claims_repro/A_gemm):
# the exact job sequence. Compile phases are CPU-only; every GPU job runs under the shared
# GPU lock (one job per lock hold) so concurrent sub-studies never time kernels together.
#
#   bash research/bench/scripts/claims_gemm_all.sh compile   # CPU: fill ~/.tilelang/cache (~45 min at 24 workers)
#   bash research/bench/scripts/claims_gemm_all.sh run       # C1/C6 GPU jobs, each under flock
#   bash research/bench/scripts/claims_gemm_all.sh xcheck    # cuBLAS 12.9 (LD_PRELOAD) + cuBLAS NN-layout cross-checks
#   bash research/bench/scripts/claims_gemm_all.sh steady    # bench_steady cross-check (C1, C6 fp16)
#   bash research/bench/scripts/claims_gemm_all.sh gemv      # C5 GPU jobs, each under flock
#   python research/bench/scripts/claims_gemm_report.py --md tables.md   # tables + summary.json
#
# LOCK defaults to the orchestrator's lock file; LOGS to a scratch directory. Run from the repo
# root: the TileLang autotuner writes ./autotuner.log (gitignored) into the working directory.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$REPO/research/env.sh"
LOCK=${LOCK:-/tmp/claude-1008/-home-ywc-co-tilelang/d36518d2-9f4d-4232-868c-46d863e88350/scratchpad/gpu.lock}
LOGS=${LOGS:-/tmp/claude-1008/-home-ywc-co-tilelang/d36518d2-9f4d-4232-868c-46d863e88350/scratchpad/A_gemm/logs}
S="$REPO/research/bench/scripts"
mkdir -p "$LOGS"
C1="M0,M1,M2,M3,M4,M5,M6,M7"
KS="K256,K512,K1024,K2048,K4096,K16384"   # K8192 == M5 (same 8192^3 problem)
KS8="K256,K512,K1024,K2048,K4096,K8192,K16384"
V="V0,V1,V2,V3,V4,V5,V6,V7"
CUBLAS129=/usr/local/cuda-12.9/lib64/libcublasLt.so.12:/usr/local/cuda-12.9/lib64/libcublas.so.12

gpu() {  # gpu <log> <cmd...>: one job under the GPU lock (the job itself applies the rules.md 7 guard)
  local log=$1; shift
  echo "$(date '+%F %T') start $*" >> "$LOGS/queue.log"
  flock "$LOCK" timeout 5400 "$@" > "$log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') exit $rc $*" >> "$LOGS/queue.log"
}

case "${1:-}" in
  compile)
    python "$S/claims_gemm_tl.py" compile --suite fp32acc --shapes "$C1,$KS" --workers 24 > "$LOGS/compile_fp32acc.log" 2>&1
    python "$S/claims_gemm_tl.py" compile --suite fp16acc --shapes "$C1" --workers 24 > "$LOGS/compile_fp16acc.log" 2>&1
    python "$S/claims_gemm_tl.py" compile --suite fp8 --shapes "$KS8" --workers 24 > "$LOGS/compile_fp8.log" 2>&1
    # the fp8 suite uses TileLang's cython backend, whose libgen leaves tmp*.cu/.so in /tmp
    # (~1.1 MB per config); delete them afterwards if disk is short.
    ;;
  run)
    for sh in ${C1//,/ } ${KS//,/ }; do gpu "$LOGS/run_fp32acc_$sh.log" python "$S/claims_gemm_tl.py" run --suite fp32acc --shape "$sh"; done
    for sh in ${C1//,/ }; do gpu "$LOGS/run_fp16acc_$sh.log" python "$S/claims_gemm_tl.py" run --suite fp16acc --shape "$sh"; done
    for sh in ${KS8//,/ }; do gpu "$LOGS/run_fp8_$sh.log" python "$S/claims_gemm_tl.py" run --suite fp8 --shape "$sh"; done
    ;;
  xcheck)
    for sh in ${C1//,/ }; do gpu "$LOGS/run_fp32acc_${sh}_cublas129.log" env LD_PRELOAD=$CUBLAS129 \
        python "$S/claims_gemm_tl.py" run --suite fp32acc --shape "$sh" --tag _cublas129 --cublas-nn; done
    for sh in M0 M5 M7; do gpu "$LOGS/run_fp32acc_${sh}_cublas128nn.log" \
        python "$S/claims_gemm_tl.py" run --suite fp32acc --shape "$sh" --tag _cublas128nn --cublas-nn; done
    ;;
  steady)  # cobench.bench_steady cross-check of the flush-mode winners (reads raw/<suite>_<shape>.json)
    python "$S/claims_gemm_steady.py" compile --suite fp32acc --shapes "$C1,$KS" > "$LOGS/steady_compile_fp32acc.log" 2>&1
    python "$S/claims_gemm_steady.py" compile --suite fp16acc --shapes "$C1" > "$LOGS/steady_compile_fp16acc.log" 2>&1
    gpu "$LOGS/steady_fp32acc_M0.log" python "$S/claims_gemm_steady.py" run --suite fp32acc --shape M0
    for grp in M1,M2 M3,M4 M5,M6 M7,K256 K512,K1024 K2048,K4096 K16384; do
      gpu "$LOGS/steady_fp32acc_${grp//,/_}.log" python "$S/claims_gemm_steady.py" run --suite fp32acc --shape "$grp"; done
    for grp in M0,M1 M2,M3 M4,M5 M6,M7; do
      gpu "$LOGS/steady_fp16acc_${grp//,/_}.log" python "$S/claims_gemm_steady.py" run --suite fp16acc --shape "$grp"; done
    ;;
  gemv)
    for sh in ${V//,/ }; do gpu "$LOGS/gemv_tl_$sh.log" python "$S/claims_gemv_tl.py" --shape "$sh"; done
    # BitBLAS 0.1.0.post1 (uninstalled again; reinstall: research/env_versions.md section 7) brings its
    # own TVM + TileLang: keep the fork's tilelang off sys.path. Both runs record the build failure.
    gpu "$LOGS/gemv_bitblas_V0.log" env -u PYTHONPATH python "$S/claims_gemv_bitblas.py" --shape V0 --skip-secondary
    gpu "$LOGS/gemv_bitblas_V0_ada.log" env -u PYTHONPATH python "$S/claims_gemv_bitblas.py" --shape V0 \
        --skip-secondary --target "cuda -arch=sm_89" --tag _ada_target
    ;;
  *) echo "usage: $0 compile|run|xcheck|steady|gemv"; exit 2 ;;
esac
