#!/usr/bin/env bash
# Yield watcher: kills OUR GPU process (pid $1) within ~5 s if any GPU compute process that is not ours appears
# (Matt or another agent starting a model). Our pids are listed in $EFF/our_pids.txt. Writes YIELDED to $EFF/yield.flag.
#   yield_watch.sh <our_pid> [min_foreign_mib=300]
set -u
EFF=/path/to/scratch/eff
pid=$1; minmib=${2:-300}
while kill -0 "$pid" 2>/dev/null; do
  while IFS=, read -r fp fm _; do
    fp=$(echo "$fp" | tr -d ' '); fm=$(echo "$fm" | tr -dc '0-9')
    [ -z "$fp" ] && continue
    if ! grep -qx "$fp" "$EFF/our_pids.txt" 2>/dev/null && [ "${fm:-0}" -ge "$minmib" ]; then
      echo "$(date -Is) YIELDED: foreign GPU pid $fp using ${fm} MiB -> stopping our pid $pid" | tee -a "$EFF/yield.log"
      echo YIELDED > "$EFF/yield.flag"
      kill -INT "$pid" 2>/dev/null; sleep 4; kill -9 "$pid" 2>/dev/null
      exit 0
    fi
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  sleep 5
done
