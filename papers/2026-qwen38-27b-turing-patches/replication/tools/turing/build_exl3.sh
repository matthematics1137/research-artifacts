#!/bin/bash
# Paper 8: build a patched exllamav3 1.6.5 extension for the 2080 Ti only (sm_75) on the desktop, CPU only, and
# unpack it into an overlay directory that PYTHONPATH puts in front of the lab venv's own exllamav3. The lab venv
# and the official wheel stay untouched, so every run can choose its engine with one environment variable.
#   bash build_exl3.sh TAG PATCH...      e.g. build_exl3.sh w7 $L/patches/exllamav3-1.6.5-w7-int8-gemv-fit.patch
# Another source tree (e.g. a fork, for a comparison lane): EXL3_SRC_TAR=$L/src/OTHER.tar.gz EXL3_SRC_LABEL=1.5.3
# (default the v1.6.5 tarball, label 1.6.5); the tarball must unpack to the repository root.
# Result: $L/overlay/exl3-TAG (use PYTHONPATH=$L/overlay/exl3-TAG), wheel in $L/wheelhouse/exl3-TAG, log on stdout.
# Run it as its own capped unit (not turing-run-*, so the GPU guard and the queue ignore it):
#   systemd-run --user --unit=turing-build-TAG --collect -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=300% \
#     -p Nice=15 -p IOSchedulingClass=idle -p OOMScoreAdjust=800 -p StandardOutput=append:$L/logs/build-TAG.log \
#     -p StandardError=append:$L/logs/build-TAG.log bash $L/tools/build_exl3.sh TAG PATCH...
#
# Toolchain. PyTorch 2.10.0+cu128 needs a CUDA 12 nvcc. The desktop's CUDA 12.5 headers predate glibc 2.41, whose
# math.h declares rsqrt, rsqrtf, sinpi, sinpif, cospi and cospif noexcept; nvcc then stops on the mismatched
# declarations (seen on 2026-10-30 building llama.cpp). The script makes a copy of CUDA 12.5 inside the lab, adds
# noexcept(true) to those six declarations in the copy (the change CUDA 12.9 made), and builds with gcc-13, the
# newest host compiler CUDA 12.5 supports. /usr/local/cuda-12.5 itself is never modified. Nothing is deleted:
# every build gets a fresh, stamped source tree.
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
TAG=$1; shift
SRC_TAR=${EXL3_SRC_TAR:-$L/src/exllamav3-v1.6.5.tar.gz}
SRC_LABEL=${EXL3_SRC_LABEL:-1.6.5}
SRC_SHA=${EXL3_SRC_SHA:-}
TC=$L/toolchains/cuda-12.5-glibc243
export TMPDIR=$L/tmp XDG_CACHE_HOME=$L/cache PIP_CACHE_DIR=$L/cache/pip
mkdir -p "$L/toolchains" "$L/tmp" "$L/wheelhouse/exl3-$TAG" "$L/overlay"
say() { echo "$(date -Is) $*"; }

if [ ! -x "$TC/bin/nvcc" ]; then
  say "toolchain: copying /usr/local/cuda-12.5 to $TC"
  cp -a /usr/local/cuda-12.5 "$TC.partial"
  H=$TC.partial/targets/x86_64-linux/include/crt/math_functions.h
  sed -E -i 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +(rsqrt|rsqrtf|sinpi|sinpif|cospi|cospif)\((double|float) +x\));/\1 noexcept (true);/' "$H"
  n=$(grep -c -E '(rsqrt|rsqrtf|sinpi|sinpif|cospi|cospif)\((double|float) +x\) noexcept \(true\);' "$H")
  [ "$n" -eq 6 ] || { say "toolchain: expected 6 patched declarations, found $n"; exit 1; }
  mv "$TC.partial" "$TC"
fi
"$TC/bin/nvcc" --version | tail -2

[ -f "$SRC_TAR" ] || { say "missing $SRC_TAR (git archive from the laptop mirror)"; exit 1; }
[ -z "$SRC_SHA" ] || echo "$SRC_SHA  $SRC_TAR" | sha256sum -c -
W=$L/src/exllamav3-$SRC_LABEL-$TAG-$(date +%Y%m%dT%H%M%S)
mkdir -p "$W"
tar -xzf "$SRC_TAR" -C "$W"
for p in "$@"; do say "patch: $p ($(sha256sum "$p" | cut -c1-16))"; patch -p1 -d "$W" < "$p"; done
# A local version label, so run records name the build: 1.6.5+TAG
sed -i -E "s/^__version__ = \"([0-9.]+)\"/__version__ = \"\\1+$TAG\"/" "$W/exllamav3/version.py"
grep -n __version__ "$W/exllamav3/version.py"

say "build: sm_75 only, 3 jobs"
cd "$W"
env CUDA_HOME="$TC" PATH="$TC/bin:$L/venv/bin:$PATH" TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=3 \
    CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 \
    "$L/venv/bin/python" setup.py -q bdist_wheel -d "$L/wheelhouse/exl3-$TAG"
WHL=$(ls -t "$L/wheelhouse/exl3-$TAG"/*.whl | head -1)
say "wheel: $WHL $(sha256sum "$WHL" | cut -c1-16)"
O=$L/overlay/exl3-$TAG
[ -e "$O" ] && O=$O-$(date +%Y%m%dT%H%M%S)  # never overwrite an overlay a run may be using
"$L/venv/bin/python" -m zipfile -e "$WHL" "$O"
SO=$(ls "$O"/exllamav3_ext*.so)
"$TC/bin/cuobjdump" --list-elf "$SO" | awk '{print $NF}' | sed -E 's/.*\.(sm_[0-9]+)\..*/\1/' | sort | uniq -c
say "overlay ready: PYTHONPATH=$O"
