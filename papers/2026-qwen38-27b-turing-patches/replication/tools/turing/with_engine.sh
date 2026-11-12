#!/bin/bash
# Run a command with a patched exllamav3 overlay (build_exl3.sh) in front of the lab venv's own exllamav3, after
# checking, without touching the GPU, that the overlay is what Python will import. A missing or wrong overlay
# would otherwise fall back silently to the stock wheel and mislabel the run.
#   bash with_engine.sh TAG [VAR=VALUE...] COMMAND...      e.g. with_engine.sh w7 EXL3_INT8_GEMV_FIT=0 bash census.sh w7-fit0
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
TAG=$1; shift
O=${EXL3_OVERLAY:-$L/overlay/exl3-$TAG}
[ -f "$O/exllamav3/version.py" ] || { echo "with_engine: no overlay at $O" >&2; exit 2; }
export PYTHONPATH=$O${PYTHONPATH:+:$PYTHONPATH}
got=$(CUDA_VISIBLE_DEVICES= "$L/venv/bin/python" -c \
  'import os, exllamav3; from exllamav3.version import __version__ as v; print(v, os.path.dirname(exllamav3.__file__))' 2>/dev/null)
case $got in
  *"+$TAG $O/exllamav3") echo "with_engine: exllamav3 $got" ;;
  *) echo "with_engine: expected 1.6.5+$TAG from $O, Python imports: $got" >&2; exit 2 ;;
esac
exec env "$@"
