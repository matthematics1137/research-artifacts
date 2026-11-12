#!/bin/bash
# Build the llama.cpp baselines for the 2080 Ti (sm_75) inside the lab on the desktop's ssd2 drive.
#   mainline : ggml-org/llama.cpp at the commit the laptop mirror held on 2026-10-30
#   mtpshared: v0.4.0 plus the lab's shared-MTP patch (research/patches/llamacpp-v0.4.0-qwen35-mtp-shared-embd.patch),
#              needed for the stripped 0.28 GB MTP sidecar
# CUDA 13.4 with gcc-13 as host compiler: CUDA 12.5's math headers clash with this glibc (rsqrtf noexcept),
# which 13.x fixes; sm_75 SASS is still supported. Run with LD_LIBRARY_PATH=$CUDA/lib64. Nothing is deleted.
# Run in a capped unit, for example:
#   systemd-run --user --unit=turing-build --collect -p Nice=15 -p CPUQuota=400% -p MemoryMax=12G \
#     -p IOSchedulingClass=idle bash /opt/p8/tools/build_llamacpp.sh
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
SRC=$L/src
MAINLINE=d63fbfe794b62990086043fa9e05fd6e8367b4d6
CUDA=${CUDA_HOME:-/usr/local/cuda-13.4}
B=build-sm75-cu$(basename "$CUDA" | tr -dc 0-9)
export PATH=$L/venv/bin:$PATH XDG_CACHE_HOME=$L/cache CCACHE_DISABLE=1
exec >> "$L/logs/build-llamacpp.log" 2>&1
echo "== $(date -Is) build start"
command -v cmake >/dev/null && command -v ninja >/dev/null || "$L/bootstrap/bin/uv" pip install --python "$L/venv/bin/python" cmake ninja

build() {  # dir label
  cmake -S "$1" -B "$1/$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=ON
  cmake --build "$1/$B" -j 4 --target llama-server llama-bench llama-cli
  LD_LIBRARY_PATH=$CUDA/lib64 "$1/$B/bin/llama-server" --version 2>&1 | tail -2
  echo "$2 built: $(git -C "$1" rev-parse HEAD) $(git -C "$1" status --short | grep -vc '^??') modified files"
}

if [ ! -d "$SRC/llama.cpp/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$SRC/llama.cpp"
fi
git -C "$SRC/llama.cpp" checkout --quiet "$MAINLINE"
build "$SRC/llama.cpp" mainline

if [ ! -d "$SRC/llama.cpp-v0.4.0-mtpshared/.git" ]; then
  git clone --branch v0.4.0 https://github.com/ggml-org/llama.cpp.git "$SRC/llama.cpp-v0.4.0-mtpshared"
  git -C "$SRC/llama.cpp-v0.4.0-mtpshared" apply "$L/patches/llamacpp-v0.4.0-qwen35-mtp-shared-embd.patch"
fi
build "$SRC/llama.cpp-v0.4.0-mtpshared" mtpshared
echo "== $(date -Is) build done"
