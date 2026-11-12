#!/bin/bash
# P9 phase 1: build the same patched exllamav3 1.6.5 extension as tools/turing/build_exl3.sh, for the laptop's RTX 4080
# Laptop GPU only (sm_89), CPU only, and unpack it into an overlay directory that PYTHONPATH puts in front of
# engines/exl3-env's own exllamav3. Same source tarball (git archive of v1.6.5 from the laptop's mirror, the
# desktop's tarball byte for byte) and the same patch files, so the kernels differ from the desktop's builds only in
# their target. With no patch it builds stock 1.6.5 with the same toolchain, to set against the official wheel.
#   bash build_exl3_ada.sh TAG PATCH...      e.g. build_exl3_ada.sh w13sm89 tools/turing/patches/exllamav3-1.6.5-w7-...
# The laptop's toolchain needs no copy: CUDA 12.8 with glibc 2.39 and gcc 13.3 builds as is.
# Result: engines/exl3-1.6.5-TAG/site (use PYTHONPATH=that), wheel in engines/exl3-1.6.5-TAG/wheelhouse, log on stdout.
# Run it as its own capped unit:
#   systemd-run --user --unit=p9-build-TAG --collect -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=400% -p Nice=15 \
#     -p IOSchedulingClass=idle -p OOMScoreAdjust=800 -p StandardOutput=append:LOG -p StandardError=append:LOG \
#     bash tools/turing/build_exl3_ada.sh TAG PATCH...
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
TAG=$1; shift
E=$R/engines/exl3-1.6.5-$TAG
SRC_TAR=$E/exllamav3-v1.6.5.tar.gz
SRC_SHA=07b70fc139447b9f9633e5ec164987de20a9e6526b17f1aed4e875b37bfc6ae8
PY=$R/engines/exl3-env/bin/python
CUDA=/usr/local/cuda-12.8
export TMPDIR=$E/tmp
mkdir -p "$TMPDIR" "$E/wheelhouse"
say() { echo "$(date -Is) $*"; }

[ -f "$SRC_TAR" ] || git -C "$R/engines/src-mirror/turboderp-org__exllamav3.git" archive --format=tar.gz -o "$SRC_TAR" v1.6.5
echo "$SRC_SHA  $SRC_TAR" | sha256sum -c -
W=$E/src-$(date +%Y%m%dT%H%M%S)
mkdir -p "$W"
tar -xzf "$SRC_TAR" -C "$W"
for p in "$@"; do say "patch: $p ($(sha256sum "$p" | cut -c1-16))"; patch -p1 -d "$W" < "$p"; done
# A local version label, so run records name the build: 1.6.5+TAG
sed -i -E "s/^__version__ = \"([0-9.]+)\"/__version__ = \"\\1+$TAG\"/" "$W/exllamav3/version.py"
grep -n __version__ "$W/exllamav3/version.py"

say "build: sm_89 only, 4 jobs"
cd "$W"
env CUDA_HOME="$CUDA" PATH="$CUDA/bin:$R/engines/exl3-env/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 \
    CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 \
    "$PY" setup.py -q bdist_wheel -d "$E/wheelhouse"
WHL=$(ls -t "$E/wheelhouse"/*.whl | head -1)
say "wheel: $WHL $(sha256sum "$WHL" | cut -c1-16)"
O=$E/site
[ -e "$O" ] && O=$O-$(date +%Y%m%dT%H%M%S)  # never overwrite an overlay a run may be using
"$PY" -m zipfile -e "$WHL" "$O"
SO=$(ls "$O"/exllamav3_ext*.so)
"$CUDA/bin/cuobjdump" --list-elf "$SO" | awk '{print $NF}' | sed -E 's/.*\.(sm_[0-9]+)\..*/\1/' | sort | uniq -c
say "overlay ready: PYTHONPATH=$O"
