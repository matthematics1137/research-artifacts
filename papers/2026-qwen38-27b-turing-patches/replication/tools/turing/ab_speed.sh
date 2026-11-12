#!/bin/bash
# Paper 8: interleaved A/B decode-speed repeats on the 2080 Ti, no profiler. Each arm runs in a fresh process (the
# EXL3_* switches are read once per process), in a rotated order every round (A B C, B C A, C A B, ...), so heat and
# clock drift spread over the arms instead of landing on one. census.py does the measuring (its profiler calls are
# no-ops without nsys): warm-up on the three smoke prompts, then 256 decoded tokens per prompt, one row each.
#   bash ab_speed.sh TAG ROUNDS 'NAME|ENV=1 ENV=2|extra exllamav3 args' 'NAME|...' ...
# An empty ENV or args field is fine ('base||'). Rows go to $L/runs/ab/TAG.jsonl, tagged TAG-NAME-rROUND; each
# row records the exllamav3 version, its path and the EXL3_* switches it saw, so an overlay arm is verifiable.
# Power arms (L2): TURING_POWER_W=N in an arm's ENV sets the board limit with `sudo -n nvidia-smi -pl N` before that
# arm (the sudoers rule allows 260, 290 and 320 only). Once any arm names a limit, arms without one run at the 260 W
# default, and the limit goes back to 260 W when the script exits for any reason.
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
TAG=$1; ROUNDS=$2; shift 2
ARMS=("$@")
D=$L/runs/ab
mkdir -p "$D"
M=(-m "$L/models/qwen38-exl3-2.0" -cs 16384 -cq 8,8 -rcs 2 -chunk_size 1024)
n=${#ARMS[@]}
power=0
[[ " ${ARMS[*]} " == *TURING_POWER_W=* ]] && power=1
set_power() { sudo -n /usr/bin/nvidia-smi -pl "$1" | grep -i 'power limit'; }
[ "$power" = 1 ] && trap 'set_power 260 || echo "$(date -Is) WARNING: could not reset the power limit to 260 W"' EXIT
for r in $(seq 1 "$ROUNDS"); do
  for i in $(seq 0 $((n - 1))); do
    spec=${ARMS[$(( (i + r - 1) % n ))]}
    name=${spec%%|*}; rest=${spec#*|}; envs=${rest%%|*}; extra=${rest#*|}
    echo "$(date -Is) round $r arm $name"
    if [ "$power" = 1 ]; then
      w=260; [[ " $envs " =~ \ TURING_POWER_W=([0-9]+)\  ]] && w=${BASH_REMATCH[1]}
      set_power "$w"
    fi
    # shellcheck disable=SC2086  # envs and extra are deliberately split into words
    env $envs "$L/venv/bin/python" "$L/tools/census.py" "${M[@]}" $extra --tag "$TAG-$name-r$r" --out "$D/$TAG.jsonl"
  done
done
