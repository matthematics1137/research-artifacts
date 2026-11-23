#!/bin/bash
# Speed of every main lane with the common client (tools/turing/speed_bench.py), one lane at a time.
# Run through the launcher: bash $L/tools/desktop_run.sh cmd speed-all bash $L/tools/speed_all.sh [TAG]
set -u
L=${TURING_LAB:-/opt/p8}
TAG=${1:-full-20261030}
rc=0
for lane in exl3-nodraft exl3-mtp exl3-mtp-fp32 lcpp-iq2xxs lcpp-iq2s lcpp-iq2s-mtp lcpp-iq2xxs-dflash2; do
  echo "== $(date -Is) $lane"
  if bash "$L/tools/lanes.sh" start "$lane"; then
    python3 "$L/tools/speed_bench.py" --lane "$lane" --repeats 3 --tag "$TAG" || rc=1
  else
    echo "lane $lane did not start"; rc=1
  fi
  bash "$L/tools/lanes.sh" stop "$lane"
done
exit $rc
