#!/usr/bin/env python3
"""Yield the desktop to Matt's own work. Runs on p8-host as the user unit turing-guard.

Every 15 s it checks the machine. When any trigger fires, it stops every turing-run* user unit (our
own model processes only; nothing is deleted) and logs why. Triggers (adjustable environment
defaults, not study parameters):
  - MemAvailable below TURING_MIN_AVAIL_GIB (default 10) on two samples, or below half that at once;
  - memory stall: PSI "full" avg10 at or above TURING_MAX_MEM_FULL (default 15);
  - free disk on the lab's drive (ssd2) below TURING_DISK_FLOOR_GB (default 50);
  - GPU temperature at or above TURING_MAX_TEMP_C (default 87: the 2080 Ti holds its own 84 C target by trimming
    clocks, so 84 would stop every long run; 87 is below its 89 C max operating, 91 C slowdown, 94 C shutdown);
  - any other compute process on the GPU (not ours, not Sunshine): his own GPU work comes first.
It never restarts anything itself; a run is relaunched by hand after a stop.
"""
import os
import re
import shutil
import subprocess
import time

LAB = os.environ.get("TURING_LAB", "/opt/p8")  # never the main drive
LOG = os.path.join(LAB, "logs/guard.log")
MIN_AVAIL = float(os.environ.get("TURING_MIN_AVAIL_GIB", 10))
MAX_FULL = float(os.environ.get("TURING_MAX_MEM_FULL", 15))
DISK_FLOOR = float(os.environ.get("TURING_DISK_FLOOR_GB", 50))
MAX_TEMP = float(os.environ.get("TURING_MAX_TEMP_C", 87))
ALLOWED_GPU = {"sunshine", "Xorg", "gnome-shell"}


def log(text):
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S%z ") + text + "\n")


def avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return 0.0


def mem_full_avg10():
    for line in open("/proc/pressure/memory"):
        if line.startswith("full"):
            return float(re.search(r"avg10=([\d.]+)", line).group(1))
    return 0.0


def disk_free_gb():
    return shutil.disk_usage(LAB).free / 1e9


def gpu_temp():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    return float(out.splitlines()[0]) if out else 0.0


def ours(pid):
    try:
        return re.search(r"/turing-[A-Za-z0-9_.-]*\.service", open(f"/proc/{pid}/cgroup").read()) is not None
    except OSError:
        return False


def foreign_gpu():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20).stdout
    found = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pid, name = [x.strip() for x in line.split(",", 1)]
        if os.path.basename(name) in ALLOWED_GPU or (pid.isdigit() and ours(int(pid))):
            continue
        found.append(f"{pid} {name}")
    return found


def running_units():
    out = subprocess.run(["systemctl", "--user", "list-units", "--state=active", "--plain", "--no-legend",
                          "turing-run*"], capture_output=True, text=True).stdout
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def main():
    log(f"guard start: min_avail={MIN_AVAIL} GiB, mem_full<{MAX_FULL}, disk_floor={DISK_FLOOR} GB, temp<{MAX_TEMP} C")
    low = 0
    while True:
        reasons = []
        try:
            a = avail_gib()
            low = low + 1 if a < MIN_AVAIL else 0
            if low >= 2 or a < MIN_AVAIL / 2:
                reasons.append(f"MemAvailable {a:.1f} GiB")
            if (f := mem_full_avg10()) >= MAX_FULL:
                reasons.append(f"memory stall {f}")
            if (d := disk_free_gb()) < DISK_FLOOR:
                reasons.append(f"disk free {d:.1f} GB")
            if (t := gpu_temp()) >= MAX_TEMP:
                reasons.append(f"GPU {t:.0f} C")
            if other := foreign_gpu():
                reasons.append("another GPU program: " + "; ".join(other))
        except Exception as exc:  # a broken check must not leave a run unguarded
            reasons.append(f"guard check failed: {exc!r}")
        units = running_units()
        if reasons and units:
            log("STOP " + ", ".join(units) + " because " + "; ".join(reasons))
            subprocess.run(["systemctl", "--user", "stop", *units])
        time.sleep(15)


if __name__ == "__main__":
    main()
