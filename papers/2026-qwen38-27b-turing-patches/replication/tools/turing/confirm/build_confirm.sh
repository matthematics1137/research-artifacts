#!/bin/bash
# Paper 8 confirmation speed run (research/prereg/2026-11-04-p8-confirmation-speed.md): build every engine the run needs
# on the desktop, CPU only, before the measuring step. Each engine builds in its own directory inside the lab, at the
# registered commit; nothing outside $L is written and nothing is deleted. A failed engine is logged and the script goes
# on, so the run can drop that arm by its registered rule. The run step waits for the line "confirm builds done".
# Order: the arms the registration keeps first (exllamav3 1.6.6 stock and w13, NInferno, llama.cpp v0.9.0), then
# ik_llama.cpp and buun-llama-cpp.
# Run it as one capped unit (not turing-run-*, so the GPU guard and the queue ignore it):
#   systemd-run --user --unit=turing-build-confirm --collect -p Nice=19 -p CPUQuota=400% -p MemoryMax=16G \
#     -p MemorySwapMax=0 -p IOSchedulingClass=idle -p OOMScoreAdjust=800 \
#     -p StandardOutput=append:$L/logs/build-confirm.log -p StandardError=append:$L/logs/build-confirm.log \
#     bash $L/tools/confirm/build_confirm.sh
# ONLY=label,label (passed with --setenv) builds just those steps, e.g. the 2026-11-04 rebuild of FFmpeg and NInferno
# into logs/build-confirm-2.log after the first FFmpeg configure failed.
set -uo pipefail
L=${TURING_LAB:-/opt/p8}
SRC=$L/src
CUDA=/usr/local/cuda-13.4
FF=$L/deps/ffmpeg-6.1.2
EXL3_166_SHA=05943b12bde35bb487e3ac74d318d5f98bff0d9182345641fb5d70535362889d   # git archive v1.6.6 (bd1f5191)
WHEEL=exllamav3-1.6.6+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl
export PATH=$L/venv/bin:$PATH XDG_CACHE_HOME=$L/cache CCACHE_DISABLE=1 TMPDIR=$L/tmp PIP_CACHE_DIR=$L/cache/pip
mkdir -p "$L/deps" "$L/tmp" "$L/logs"
say() { echo "$(date -Is) $*"; }
summary=()
step() {  # label command...
  local label=$1; shift
  if [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$label,"* ]]; then summary+=("$label=skipped"); return 0; fi
  say "== $label: start"
  if ( "$@" ); then say "== $label: ok"; summary+=("$label=ok"); else say "== $label: FAILED"; summary+=("$label=failed"); fi
}

pin() {  # dir commit
  git -C "$1" checkout -q "$2" && [ "$(git -C "$1" rev-parse HEAD)" = "$(git -C "$1" rev-parse "$2^{commit}")" ] &&
    echo "$(basename "$1") at $(git -C "$1" rev-parse HEAD), $(git -C "$1" status --short | grep -vc '^??') modified files"
}

llama_family() {  # dir commit: the llama.cpp CMake flags of tools/build_llamacpp.sh, sm_75, CUDA 13.4, gcc-13
  pin "$1" "$2" || return 1
  cmake -S "$1" -B "$1/build-sm75-cu134" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF || return 1
  cmake --build "$1/build-sm75-cu134" -j 4 --target llama-server || return 1
  LD_LIBRARY_PATH=$CUDA/lib64 "$1/build-sm75-cu134/bin/llama-server" --version 2>&1 | tail -2
}

exl3_stock166() {  # the official wheel, unpacked into its own overlay; the lab venv is not touched
  local w=$L/wheelhouse/exl3-stock166 o=$L/overlay/exl3-stock166
  mkdir -p "$w"
  if [ ! -f "$w/$WHEEL" ]; then
    curl -fsSL -o "$w/$WHEEL.part" "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.6/${WHEEL/+/%2B}" &&
      mv "$w/$WHEEL.part" "$w/$WHEEL" || return 1
  fi
  sha256sum "$w/$WHEEL"
  if [ ! -d "$o/exllamav3" ]; then
    "$L/bootstrap/bin/uv" pip install --quiet --python "$L/venv/bin/python" --no-deps --target "$o" "$w/$WHEEL" || return 1
  fi
  grep -n __version__ "$o/exllamav3/version.py"
}

exl3_w13on166() {  # the w13 patch chain, unchanged, on the v1.6.6 source (its decode-path files equal 1.6.5's)
  [ -d "$L/overlay/exl3-w13on166" ] && { echo "overlay exl3-w13on166 already built"; return 0; }
  local p=$L/patches/exllamav3-1.6.5
  EXL3_SRC_TAR=$SRC/exllamav3-v1.6.6.tar.gz EXL3_SRC_LABEL=1.6.6 EXL3_SRC_SHA=$EXL3_166_SHA \
    bash "$L/tools/build_exl3.sh" w13on166 "$p-w7-int8-gemv-fit.patch" "$p-w9-int8-sq-fit.patch" \
      "$p-w10-int8-sq-m4.patch" "$p-w10b-int8-sq-rowcap.patch" "$p-w11-fd75d-decode.patch" \
      "$p-w11b-fd75d-live-splits.patch" "$p-w12p-int8-stepped-pointer.patch" "$p-w13-gemv-sk.patch"
}

ffmpeg_build() {  # NInferno links libavformat/libavcodec/libavutil/libswscale; the desktop has no -dev packages and
  # this lab uses no sudo, so a minimal shared FFmpeg (Ubuntu 24.04's 6.1 series, no codecs, no network) goes to $FF.
  # The desktop has no nasm or yasm, so the hand-written x86 assembly is off (--disable-x86asm; without it configure
  # stopped on 2026-11-04). It speeds only FFmpeg's own media code, which no text-decode path uses.
  [ -f "$FF/lib/pkgconfig/libavformat.pc" ] && { echo "ffmpeg already in $FF"; return 0; }
  [ -d "$SRC/ffmpeg-n6.1.2" ] || git clone -q --depth 1 --branch n6.1.2 https://github.com/FFmpeg/FFmpeg.git \
    "$SRC/ffmpeg-n6.1.2" || return 1
  cd "$SRC/ffmpeg-n6.1.2" && ./configure --prefix="$FF" --cc=gcc-13 --enable-shared --disable-static --disable-x86asm \
    --disable-programs --disable-doc --disable-network --disable-autodetect --disable-everything \
    --enable-avcodec --enable-avformat --enable-avutil --enable-swscale >/dev/null &&
    make -j 4 >/dev/null && make install >/dev/null && ls "$FF/lib/pkgconfig"
}

ninferno_build() {
  local s=$SRC/NInferno
  pin "$s" 1d3b43e3fa94c533427261868b42fac4b7e300ba || return 1
  PKG_CONFIG_PATH=$FF/lib/pkgconfig cmake -S "$s" -B "$s/build-sm75" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 -DNINFERNO_UI_CHAT=OFF \
    -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_BUILD_RPATH="$FF/lib" || return 1
  cmake --build "$s/build-sm75" -j 4 --target ninferno-serve || return 1
  ls -la "$s/build-sm75/apps/ninferno-serve"
}

say "== confirm builds: start (nproc $(nproc); $(free -g | awk '/Mem:/ {print $7}') GB available)"
[ -f "$SRC/exllamav3-v1.6.6.tar.gz" ] || say "missing $SRC/exllamav3-v1.6.6.tar.gz (git archive v1.6.6 from the laptop mirror)"
step exl3-stock166 exl3_stock166
step exl3-w13on166 exl3_w13on166
step ffmpeg ffmpeg_build
step ninferno ninferno_build
step llamacpp-v0.9.0 llama_family "$SRC/llama.cpp-v0.9.0" cd959bae99ff4bed22bef2604f45241c09bbb8c8
step ik_llama.cpp llama_family "$SRC/ik_llama.cpp" 009dce19afdd3f1e4b5039bcd8853256726c322b
step buun-llama-cpp llama_family "$SRC/buun-llama-cpp" a8022d241929dcc7ab99d90af36bdaa8176cc311
say "summary: ${summary[*]}"
say "confirm builds done"
