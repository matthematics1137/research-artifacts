#!/bin/bash
# Turing lab bootstrap on p8-host (RTX 2080 Ti). Approved by Matt 2026-10-30.
# Everything lives in /opt/p8; nothing outside it is changed, nothing is deleted.
# Runs at low priority inside a capped systemd user unit (see tools/turing/README.md).
set -euo pipefail
# Everything in one place on the desktop's ssd2 drive; nothing on the 500 GB main drive (Matt, 2026-10-30).
L=${TURING_LAB:-/opt/p8}
mkdir -p "$L"/{bootstrap,python,src,models,logs,wheelhouse,runs,cache}
export PIP_CACHE_DIR=$L/cache/pip XDG_CACHE_HOME=$L/cache
exec >> "$L/logs/setup.log" 2>&1
echo "== $(date -Is) setup start on $(hostname)"

floor_gb=${TURING_DISK_FLOOR_GB:-50}
avail=$(df -BG --output=avail "$L" | tail -1 | tr -dc 0-9)
echo "disk free: ${avail} GB (floor ${floor_gb} GB)"
[ "$avail" -ge "$floor_gb" ] || { echo "STOP: free disk below the floor"; exit 1; }

WHEEL=$L/wheelhouse/exllamav3-1.6.5+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl
echo "e184693383192e1b971ba895f911abe68818bca158e8cad297712c26cb0c112d  $WHEEL" | sha256sum -c -

# uv in its own bootstrap venv, then a standalone Python 3.12 (the desktop's system Python is 3.14).
[ -x "$L/bootstrap/bin/uv" ] || { python3 -m venv "$L/bootstrap"; "$L/bootstrap/bin/pip" install --quiet uv; }
UV=$L/bootstrap/bin/uv
"$UV" --version
export UV_PYTHON_INSTALL_DIR=$L/python UV_NO_CACHE=1
"$UV" python install 3.12
[ -x "$L/venv/bin/python" ] || "$UV" venv "$L/venv" --python 3.12
PY=$L/venv/bin/python

# torch 2.10.0 cu128 (brings its pinned triton) exactly as on the laptop's exl3-env, then the verified wheel.
"$UV" pip install --python "$PY" torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$PY" "$WHEEL"

# Upstream TabbyAPI (Turing allowed since PR #463) in its own checkout, separate from the lab's copy.
if [ ! -d "$L/src/tabbyAPI/.git" ]; then
  git clone https://github.com/theroyallab/tabbyAPI.git "$L/src/tabbyAPI"
fi
git -C "$L/src/tabbyAPI" log -1 --format='tabbyAPI %H %cs'
"$UV" pip install --python "$PY" -e "$L/src/tabbyAPI"

"$PY" - <<'EOF'
import torch, triton, exllamav3
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("exllamav3", exllamav3.__version__ if hasattr(exllamav3, "__version__") else "?")
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
EOF
"$UV" pip freeze --python "$PY" > "$L/logs/freeze-$(date +%Y%m%d).txt"
df -BG --output=avail "$L" | tail -1
echo "== $(date -Is) setup done"
