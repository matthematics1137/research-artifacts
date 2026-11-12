#!/bin/bash
# P9 phase 1, M4 (research/prereg/2026-11-05-p9-phase1.md): Nsight Compute counters for the one-row kernel's full form
# and its loop alone, on the laptop. GPU counters need root here (RmProfilingAdminOnly: 1), so Matt runs it:
#   sudo bash /opt/p8/repository/tools/turing/p9_ncu.sh
# It builds the phase kernels into a fresh root-owned temporary directory, profiles the eight launches of
# p9_ncu_target.py (mlp_gate and gdn_qkv, K = 1 and 2, full and loop only) with the issue, warp-state, compute, memory
# and instruction sections, exports the raw metrics as CSV with a short report, and hands the results to user
# somebody. While it profiles, Nsight Compute holds the GPU at its base clocks (its default) and restores them after;
# nothing else changes, and it reads no file outside the repository and the CUDA toolkit. About two minutes.
set -euo pipefail
R=/opt/p8/repository
OUT=$R/testsuite/results/ada/p9/ncu-$(date +%Y%m%dT%H%M%S)
E=$R/engines
CUDA=/usr/local/cuda-12.8
SRC=$(ls -d "$E"/exl3-1.6.5-w13sm89/src-* | head -1)/exllamav3/exllamav3_ext
mkdir -p "$OUT"
X=$(mktemp -d /tmp/p9-ncu-ext.XXXXXX)
env PYTHONPATH="$E/exl3-1.6.5-w13sm89/site:$R/tools/turing" CUDA_HOME=$CUDA PATH="$CUDA/bin:$E/exl3-env/bin:/usr/bin:/bin" \
  CC=gcc-13 CXX=g++-13 MAX_JOBS=4 TORCH_EXTENSIONS_DIR="$X" \
  "$CUDA/bin/ncu" --kernel-name regex:phases_ --section SpeedOfLight --section SchedulerStats --section WarpStateStats \
    --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section InstructionStats --section Occupancy \
    --export "$OUT/p9_ncu" \
    "$E/exl3-env/bin/python" "$R/tools/turing/p9_ncu_target.py" --src "$SRC" --tag w13sm89 2>&1 | tee "$OUT/ncu.log"
"$CUDA/bin/ncu" --import "$OUT/p9_ncu.ncu-rep" --page raw --csv > "$OUT/raw.csv"
"$CUDA/bin/ncu" --import "$OUT/p9_ncu.ncu-rep" --page details --csv > "$OUT/details.csv"
python3 "$R/tools/turing/p9_ncu_report.py" "$OUT" > /dev/null || echo "report failed; raw.csv is complete"
chown -R somebody:somebody "$OUT"
echo "done: $OUT"
