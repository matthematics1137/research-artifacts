#!/usr/bin/env python3
"""Print or execute CPU compilation recipes. No model downloads or GPU queries.

Execution is for a separately authorized staging/build step, not the GPU queue.
Existing lab source trees, overlays and environments are never checked out or rebuilt.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from run_refresh import digest, load, RC


def commands(plan, lab):
    cuda = "/usr/local/cuda-13.4"
    cmake = str(Path(lab) / "venv/bin/cmake")
    common = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
              "-DCMAKE_CUDA_COMPILER=" + cuda + "/bin/nvcc", "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13",
              "-DCMAKE_C_COMPILER=/usr/bin/gcc-13", "-DCMAKE_CXX_COMPILER=/usr/bin/g++-13"]
    out = {}
    for key in ("ninferno", "llama"):
        e = plan["engines"][key]
        build = str(Path(e["source"]) / "build-sm75-cu134-refresh")
        extra = (["-DNINFERNO_UI_CHAT=OFF", "-DBUILD_TESTING=OFF", "-DNINFERNO_BUILD_BENCHMARKS=OFF",
                  "-DCMAKE_BUILD_RPATH=" + lab + "/deps/ffmpeg-6.1.2/lib"] if key == "ninferno" else
                 ["-DGGML_CUDA=ON", "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF", "-DGGML_NATIVE=ON", "-DLLAMA_BUILD_TESTS=OFF"])
        targets = ["ninferno-serve", "ninferno-convert"] if key == "ninferno" else ["llama-server", "llama-tokenize"]
        out[key] = dict(configure=[cmake, "-S", e["source"], "-B", build] + common + extra,
                        compile=[cmake, "--build", build, "-j", "4", "--target"] + targets,
                        targets=targets, build=build)
    return out


def execute(plan, lab):
    # All compiler work is CPU-only, including CMake compiler detection.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", CCACHE_DISABLE="1",
               PKG_CONFIG_PATH=lab + "/deps/ffmpeg-6.1.2/lib/pkgconfig", TMPDIR=lab + "/tmp")
    env["PATH"] = lab + "/venv/bin:/usr/local/cuda-13.4/bin:" + env.get("PATH", "")
    nvcc = subprocess.check_output(["/usr/local/cuda-13.4/bin/nvcc", "--version"], text=True, env=env)
    if "release 13.4," not in nvcc or "V13.4.92" not in nvcc:
        raise ValueError("requires the pinned CUDA 13.4 CPU compiler, not the laptop's CUDA 12.8")
    versions = dict(cuda=nvcc, gxx=subprocess.check_output(["/usr/bin/g++-13", "--version"], text=True, env=env),
                    cmake=subprocess.check_output([lab + "/venv/bin/cmake", "--version"], text=True, env=env))
    if "13.4.0" not in versions["gxx"] or "cmake version 4.4.4" not in versions["cmake"]:
        raise ValueError("target CPU tool versions changed; amend the frozen build rather than substitute")
    recipes = commands(plan, lab)
    receipt = dict(schema="p8-refresh-builds-v1", tool_versions=versions, engines={})
    for key, recipe in recipes.items():
        e = plan["engines"][key]
        source = Path(e["source"])
        if not source.is_dir():
            raise ValueError("stage exact public source/submodules first: " + str(source))
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
        if head != e["commit"] or dirty:
            raise ValueError("wrong or modified source tree: " + key)
        subprocess.run(recipe["configure"], env=env, check=True)
        subprocess.run(recipe["compile"], env=env, check=True)
        binaries = []
        for target in recipe["targets"]:
            binary = Path(recipe["build"]) / ("apps" if key == "ninferno" else "bin") / target
            binaries.append(dict(path=str(binary).replace(lab, "{L}", 1), sha256=digest(binary)))
        cache = Path(recipe["build"]) / "CMakeCache.txt"
        receipt["engines"][key] = dict(commit=head, arch="75", cuda_release="13.4", binaries=binaries,
                                      cmake_cache_sha256=digest(cache), **recipe)
    destination = Path(lab) / "builds/p8-refresh-76b5"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "builds.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", default=RC.L)
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    plan, _ = load(lab=args.lab)
    if args.build:
        execute(plan, args.lab)
    else:
        print(json.dumps(commands(plan, args.lab), indent=2))


if __name__ == "__main__":
    main()
