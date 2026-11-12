#!/bin/bash
# Build the W12 sqmb microbenchmark extension ahead of its queue step (CPU only), against the w11b source tree.
L=/opt/p8
SRC=$L/src/exllamav3-1.6.5-w11b-20261030T202316/exllamav3/exllamav3_ext
cd $L/tools && exec $L/venv/bin/python -c "import sys, w12_sqmb; w12_sqmb.build(sys.argv[1]); print('build ok')" "$SRC"
