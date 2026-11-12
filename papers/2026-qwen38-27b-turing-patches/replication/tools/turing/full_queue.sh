#!/bin/bash
# Paper 8 run queue on the desktop (2026-10-30). Runs each step as a capped turing-run-<label> unit, one at a time,
# in the order of the steps file. Before each try it waits until the GPU is free of other programs, no turing-run
# unit is active and the GPU is below 65 C. A step is done only when it exits 0 (marker in runs/queue/). If the
# guard stops a step (heat, memory, another GPU program), the queue waits and retries; quality runs, numerics and the
# census resume where they stopped. A guard stop (a STOP line naming the step's unit in logs/guard.log) does not use
# up a try. After 30 failed tries (2 for a label starting with x-, an experimental step), or 30 guard stops, a step
# gets a .gaveup marker and the queue moves on.
# The queue itself runs as unit turing-queue, which the guard does not stop. Nothing is deleted.
#   systemd-run --user --unit=turing-queue --collect -p MemoryMax=512M -p Nice=10 bash $L/tools/full_queue.sh
#
# Steps file: $L/runs/queue/steps.txt, one "label|command" per line ('#' comments). It is read again before every
# step, so steps can be added, reordered or dropped while the queue runs: write the new file beside it and rename
# it into place. When it is missing, the queue writes the original 2026-10-30 list below into it. Restarting the
# queue unit while a step runs is safe: the new queue waits for that unit and then finds its marker.
set -u
L=${TURING_LAB:-/opt/p8}
Q=$L/runs/queue
LOG=$L/logs/queue.log
STEPS_FILE=${TURING_STEPS:-$Q/steps.txt}
mkdir -p "$Q"
say() { echo "$(date -Is) $*" >> "$LOG"; }
RQ="bash $L/tools/quality/run_quality.sh"
ROWS="--rows $L/evals/laptop/EFF-h38a_mtp/gsm8k_500.jsonl $L/evals/laptop/EFF-h38a_mtp/mmlu_500.jsonl"
NUM="$L/venv/bin/python $L/tools/numerics.py -m $L/models/qwen38-exl3-2.0 -cq 8,8 $ROWS --repeat-default --full-logits-stride 16 --tag w4b-e47-n50"
SEED=(
  "q-exl3-mtp-gsm8k_500|$RQ exl3-mtp gsm8k_500"
  "q-exl3-nodraft-gsm8k_500|$RQ exl3-nodraft gsm8k_500"
  "q-exl3-mtp-mmlu_500|$RQ exl3-mtp mmlu_500"
  "q-exl3-nodraft-mmlu_500|$RQ exl3-nodraft mmlu_500"
  "numerics-w4b|$NUM"
  "speed-all|bash $L/tools/speed_all.sh full-20261030"
  "sweep|bash $L/tools/llamacpp_sweep.sh"
  "q-exl3-mtp-fp32-gsm8k_500|$RQ exl3-mtp-fp32 gsm8k_500"
  "q-lcpp-iq2xxs-dflash2-gsm8k_500|$RQ lcpp-iq2xxs-dflash2 gsm8k_500"
  "q-lcpp-iq2s-mtp-gsm8k_500|$RQ lcpp-iq2s-mtp gsm8k_500"
  "q-lcpp-iq2xxs-gsm8k_500|$RQ lcpp-iq2xxs gsm8k_500"
  "q-lcpp-iq2s-gsm8k_500|$RQ lcpp-iq2s gsm8k_500"
  "q-lcpp-iq2xxs-dflash2-mmlu_500|$RQ lcpp-iq2xxs-dflash2 mmlu_500"
  "q-lcpp-iq2s-mtp-mmlu_500|$RQ lcpp-iq2s-mtp mmlu_500"
  "q-lcpp-iq2xxs-mmlu_500|$RQ lcpp-iq2xxs mmlu_500"
  "q-lcpp-iq2s-mmlu_500|$RQ lcpp-iq2s mmlu_500"
)
[ -f "$STEPS_FILE" ] || printf '%s\n' "${SEED[@]}" > "$STEPS_FILE"

finished() { [ -f "$Q/$1.done" ] || [ -f "$Q/$1.gaveup" ]; }
next_step() {  # the first step in the file that is neither done nor given up
  local line
  while IFS= read -r line; do
    case $line in ''|'#'*) continue ;; esac
    finished "${line%%|*}" || { echo "$line"; return 0; }
  done < "$STEPS_FILE"
  return 1
}

gpu_ready() {
  systemctl --user list-units --state=active --plain --no-legend 'turing-run*' | grep -q . && return 1
  local others temp
  others=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -v -c sunshine)
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  [ "$others" -eq 0 ] && [ "$temp" -lt 65 ]
}

GUARD_LOG=$L/logs/guard.log
GUARD_STOPS_MAX=30
# whether the guard logged a STOP of this step's unit after line $2 of its log (the guard yields the GPU to other
# programs and stops on heat or memory; such a stop is not the step's failure, so it does not use up a try)
guard_stopped() { tail -n +$(($2 + 1)) "$GUARD_LOG" 2>/dev/null | grep ' STOP ' | grep -qF "turing-run-$1.service"; }

say "queue start: steps from $STEPS_FILE"
while entry=$(next_step); do
  label=${entry%%|*}; cmd=${entry#*|}
  tries=30
  case $label in x-*) tries=2 ;; esac  # experimental steps (new engine builds, new tools) fail fast instead of retrying
  try=0; stops=0
  while [ "$try" -lt "$tries" ] && [ "$stops" -lt "$GUARD_STOPS_MAX" ]; do
    until gpu_ready; do sleep 60; done
    [ -f "$Q/$label.done" ] && break  # a run started by an earlier queue instance finished it meanwhile
    run=$cmd
    case $label in *numerics*)  # numerics resumes when its run directory holds a stored run (default --out-dir)
      tag=$(sed -n 's/.*--tag \([^ ]*\).*/\1/p' <<< "$cmd")
      [ -n "$tag" ] && [ -f "$L/runs/numerics/$tag/run.json" ] && run="$cmd --resume" ;;
    esac
    g0=$(wc -l < "$GUARD_LOG" 2>/dev/null || echo 0)
    say "start $label (try $((try + 1)) of $tries, guard stops so far $stops)"
    bash "$L/tools/desktop_run.sh" cmd "$label" bash -c "$run && date -Is > $Q/$label.done" >> "$LOG" 2>&1
    sleep 10
    while systemctl --user is-active --quiet "turing-run-$label"; do sleep 30; done
    if [ -f "$Q/$label.done" ]; then say "done $label"; break; fi
    if guard_stopped "$label" "$g0"; then
      stops=$((stops + 1)); say "stopped by the guard: $label (stop $stops); waiting before retry"
    else
      try=$((try + 1)); say "not finished: $label (try $try of $tries); waiting before retry"
    fi
    sleep 120
  done
  [ -f "$Q/$label.done" ] || { date -Is > "$Q/$label.gaveup"; say "GAVE UP $label after $try failed tries and $stops guard stops"; }
done
say "queue finished"
