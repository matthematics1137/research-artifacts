#!/usr/bin/env python3
"""P9 phase 1, M3 (research/prereg/2026-11-05-p9-phase1.md): the machine code of the one-row int8 sq GEMV's main loop.

Dumps one build's exllamav3_ext with cuobjdump -sass for one target, keeps the exl3_gemv_int8_sq_kernel instances at
m = 1 with fp16 output and no residual (K = 1-4), and writes each kernel's SASS to OUT_DIR/LABEL.sm_XX.K<k>.sass. The
main loop is the innermost loop (a backward branch's range) holding IDP.4A instructions. Each IDP.4A multiplies four
weights for one lane, so the weights per lane per iteration are four times their count, and the instructions per
weight are the loop's instructions over that. The counts go to OUT_DIR/LABEL.sm_XX.json, by opcode (the mnemonic
before its first dot) and by throughput class: the rows of the CUDA C++ Best Practices Guide's table of native
arithmetic throughput (research/raw/2026-11-05-p9-prior-art/cuda-arithmetic-throughput-table.txt). An opcode the
table does not cover is "unassigned". p9_analyze.py turns counts into floors with the measured SM clocks.
  python3 p9_sass.py --so EXT.so --arch sm_75 --label w13 --out-dir DIR [--cuobjdump PATH]
"""
import argparse
import collections
import json
import os
import re
import subprocess

FUNC_RE = re.compile(r"Function : (_Z24exl3_gemv_int8_sq_kernelILi(\d)ELi1ELb0ELb0ELb0EE\S*)")
INSN_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)(.*?);")
LABEL_RE = re.compile(r"^\s*(\.L_x_\d+):")
TARGET_RE = re.compile(r"`\((\.L_x_\d+)\)")
ABS_TARGET_RE = re.compile(r"\bBRA(?:\.[A-Z0-9_]+)*\s+(?:!?U?P[T0-9]+\s*,\s*)?(0x[0-9a-f]+)")

# Per-PTX rows of the throughput table (results per clock per SM), mapped to SASS mnemonics. {class: (7.5, 8.9)}
CLASSES = {
    "int add 3-operand (IADD3)": (64, 64),
    "int add 2-operand (IADD)": (128, 128),
    "int multiply-add (IMAD)": (64, 64),
    "int multiply-add wide (IMAD.WIDE, IMAD.HI)": (32, 32),
    "int shift (SHF, SHL, SHR)": (64, 64),
    "bitwise (LOP3, LOP)": (64, 64),
    "int min/max (IMNMX)": (64, 64),
    "bit reverse (BREV)": (64, 64),
    "leading zeros (FLO)": (16, 16),
    "population count (POPC)": (16, 16),
    "warp shuffle (SHFL)": (16, 32),
    "fp32 add/mul/fma (FADD, FMUL, FFMA)": (64, 128),
    "fp32 min/max (FMNMX)": (64, 64),
    "fp16x2 add/mul/fma (HADD2, HMUL2, HFMA2)": (64, 64),
    "int to fp32 or back (I2F, F2I)": (16, 64),
    "fp16 to fp32 or back (F2F)": (64, 64),
}


def classify(op, full):
    if op == "IMAD":
        return "int multiply-add wide (IMAD.WIDE, IMAD.HI)" if (".WIDE" in full or ".HI" in full) else "int multiply-add (IMAD)"
    table = {"IADD3": "int add 3-operand (IADD3)", "IADD": "int add 2-operand (IADD)", "IMUL": "int multiply-add (IMAD)",
             "SHF": "int shift (SHF, SHL, SHR)", "SHL": "int shift (SHF, SHL, SHR)", "SHR": "int shift (SHF, SHL, SHR)",
             "LOP3": "bitwise (LOP3, LOP)", "LOP": "bitwise (LOP3, LOP)", "IMNMX": "int min/max (IMNMX)",
             "BREV": "bit reverse (BREV)", "FLO": "leading zeros (FLO)", "POPC": "population count (POPC)",
             "SHFL": "warp shuffle (SHFL)", "FADD": "fp32 add/mul/fma (FADD, FMUL, FFMA)",
             "FMUL": "fp32 add/mul/fma (FADD, FMUL, FFMA)", "FFMA": "fp32 add/mul/fma (FADD, FMUL, FFMA)",
             "FMNMX": "fp32 min/max (FMNMX)", "HADD2": "fp16x2 add/mul/fma (HADD2, HMUL2, HFMA2)",
             "HMUL2": "fp16x2 add/mul/fma (HADD2, HMUL2, HFMA2)", "HFMA2": "fp16x2 add/mul/fma (HADD2, HMUL2, HFMA2)",
             "I2F": "int to fp32 or back (I2F, F2I)", "F2I": "int to fp32 or back (I2F, F2I)",
             "F2F": "fp16 to fp32 or back (F2F)"}
    return table.get(op, "unassigned")


def kernels(so, arch, cuobjdump):
    """{K: (mangled name, [sass lines])} for the sq kernel instances at m = 1, fp16 output, no residual."""
    p = subprocess.Popen([cuobjdump, "-sass", "-arch", arch, so], stdout=subprocess.PIPE, text=True, errors="replace")
    found, cur = {}, None
    for line in p.stdout:
        if "Function : " in line:
            m = FUNC_RE.search(line)
            cur = None
            if m:
                cur = int(m[2])
                found[cur] = (m[1], [])
            continue
        if cur is not None:
            found[cur][1].append(line.rstrip("\n"))
    p.wait()
    return found


def main_loop(lines):
    insns, labels, pending = [], {}, []
    for line in lines:
        m = LABEL_RE.match(line)
        if m:
            pending.append(m[1])
            continue
        m = INSN_RE.search(line)
        if not m:
            continue
        addr = int(m[1], 16)
        for lab in pending:
            labels[lab] = addr
        pending = []
        insns.append((addr, m[3], m[3] + m[4], line))
    loops = []
    for addr, op, full, line in insns:
        if op == "BRA":
            t, u = TARGET_RE.search(line), ABS_TARGET_RE.search(line)
            target = labels.get(t[1]) if t else (int(u[1], 16) if u else None)
            if target is not None and target <= addr:
                loops.append((target, addr))
    best = None
    for lo, hi in loops:
        body = [i for i in insns if lo <= i[0] <= hi]
        n_idp = sum(1 for i in body if i[2].startswith("IDP.4A"))
        if n_idp and (best is None or hi - lo < best[1] - best[0]):
            best = (lo, hi, body, n_idp)
    return insns, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so", required=True)
    ap.add_argument("--arch", required=True, help="sm_75 or sm_89")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuobjdump", default="cuobjdump")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    col = 0 if a.arch == "sm_75" else 1
    result = {"so": os.path.abspath(a.so), "arch": a.arch, "label": a.label, "kernels": {}}
    for K, (name, lines) in sorted(kernels(a.so, a.arch, a.cuobjdump).items()):
        with open(os.path.join(a.out_dir, f"{a.label}.{a.arch}.K{K}.sass"), "w") as f:
            f.write(f"Function : {name}\n" + "\n".join(lines) + "\n")
        insns, best = main_loop(lines)
        row = {"function": name, "instructions_total": len(insns)}
        if best:
            lo, hi, body, n_idp = best
            ops = collections.Counter(i[1] for i in body)
            classes = collections.Counter(classify(i[1], i[2]) for i in body)
            wpl = 4 * n_idp
            row.update({
                "loop": [hex(lo), hex(hi)], "loop_instructions": len(body), "idp4a": n_idp, "weights_per_lane": wpl,
                "instructions_per_weight": round(len(body) / wpl, 4),
                "opcodes": dict(ops.most_common()),
                "classes_per_weight": {c: round(n / wpl, 4) for c, n in classes.most_common()},
                "class_throughput_per_clock_per_sm": {c: CLASSES[c][col] for c in classes if c in CLASSES},
                "loads_global": sum(1 for i in body if i[1] in ("LDG", "LD")),
                "loads_shared": sum(1 for i in body if i[1] == "LDS"),
            })
        else:
            row["loop"] = None
        result["kernels"][K] = row
    with open(os.path.join(a.out_dir, f"{a.label}.{a.arch}.json"), "w") as f:
        json.dump(result, f, indent=1)
    for K, row in result["kernels"].items():   # after the file is written, so a closed stdout loses nothing
        print(f"{a.label} {a.arch} K={K}: " + (f"loop {row['loop_instructions']} instructions, {row['idp4a']} IDP.4A, "
              f"{row['instructions_per_weight']} per weight; top {dict(list(row['opcodes'].items())[:6])}"
              if row["loop"] else "no IDP.4A loop found"), flush=True)


if __name__ == "__main__":
    main()
