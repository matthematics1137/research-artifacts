#!/bin/bash
# Whole-machine power, 1 Hz, from the Corsair power supply's own monitor (corsairpsu-hid), beside turing-telemetry's
# GPU log. Columns: timestamp (the GPU log's format), input V, total DC output W, +12 V W, +5 V W, +3.3 V W, VRM °C,
# case °C. DC output, not wall draw: wall ~ output / PSU efficiency (~0.9 at this load).
#   systemd-run --user --unit turing-psu-telemetry -p Nice=10 bash $L/tools/psu_telemetry.sh
L=${TURING_LAB:-/opt/p8}
out=$L/logs/psu_telemetry.csv
chip=$(sensors 2>/dev/null | grep -m1 -o '^corsairpsu-[a-z0-9-]*')
[ -n "$chip" ] || { echo "no corsairpsu sensor found" >&2; exit 1; }
while :; do
  ts=$(date '+%Y/%m/%d %H:%M:%S.%3N')
  sensors -u "$chip" 2>/dev/null | awk -v ts="$ts" '
    /^power total:/ {k="tot"} /^power \+12v:/ {k="p12"} /^power \+5v:/ {k="p5"} /^power \+3.3v:/ {k="p3"}
    /^vrm temp:/ {k="vrm"} /^case temp:/ {k="case"} /^v_in:/ {k="vin"}
    /_input:/ && k != "" {v[k]=$2; k=""}
    END {printf "%s, %s, %s, %s, %s, %s, %s, %s\n", ts, v["vin"], v["tot"], v["p12"], v["p5"], v["p3"], v["vrm"], v["case"]}' >> "$out"
  sleep 1
done
