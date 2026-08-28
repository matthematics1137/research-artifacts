#!/usr/bin/env python3
"""Generate all figures for the paper from the campaign's raw JSONL data.

Inputs (read-only):
  ../testsuite/results/phase1.jsonl            - Phase 1 speed matrix
  ../testsuite/evals/results/summary.jsonl     - Phase 2 suite summaries
  ../testsuite/evals/results/*/math25.jsonl    - per-item hard-math rows
  ../testsuite/evals/rescore_math25.py         - alternate MATH rescore (invoked)
  ../testsuite/results/server_EXL3-2.0.log     - TabbyAPI per-request metrics
  ../testsuite/results/server_bart_hard.log    - llama.cpp per-request metrics

Outputs: ../figures/fig_{ladder,frontier,artifact,offload}.pdf
With --dark: dark-mode variants into ../figures/dark/ (same filenames), for
the paper's dark build (`make annotated-dark`). Surface matches the dark
page (#111216); series colors are the dataviz palette's *dark steps* (a
selected dark palette, not a flipped one), validated on that surface with
the skill's validate_palette.js: blue #3987e5 / orange #d95926 CVD dE 26.8,
blue / gray dE 15.9, all marks >= 3:1; two-shade blue ramp (#1c5cab ->
#3987e5) passes the ordinal checks (monotone L, light end 2.8:1).
Also prints the derived statistics used in the LaTeX tables (Wilson CIs,
effective bpw, and matched-workload throughput).

Style follows the dataviz skill: light surface, palette blue #2a78d6 as the
emphasis hue, muted gray #898781 as de-emphasis context, orange #eb6834 as the
second categorical slot (blue-orange adjacent pair validated), one-hue-two-shade
for before/after, hairline solid gridlines, direct labels, no chartjunk.
"""
import json, math, os, re, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

DARK = "--dark" in sys.argv[1:]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # local_llms/
TS = os.path.join(ROOT, "testsuite")
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
if DARK:
    FIG = os.path.join(FIG, "dark")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- palette
if not DARK:
    BLUE = "#2a78d6"      # categorical slot 1 / emphasis hue (full-GPU)
    BLUE_L = "#86b6ef"    # sequential step 250 (the "before" shade)
    ORANGE = "#eb6834"    # categorical slot 2 (model overlay)
    GRAY = "#898781"      # de-emphasis context (partial offload / CPU)
    INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
    GRID = "#e1e0d9"; AXIS = "#c3c2b7"; SURF = "#ffffff"
else:
    # Dark mode is *selected*, not flipped: same hues, the palette's dark
    # steps, re-derived chrome for the page surface (#111216 = the PDF page,
    # so figures sit seamlessly on it). The "before" shade of the dumbbell
    # goes DARKER (seq. step 550), keeping raw recessive / lenient emphatic
    # on a dark ground; text is the page's light gray, never pure white.
    BLUE = "#3987e5"      # categorical slot 1, dark step
    BLUE_L = "#1c5cab"    # sequential step 550 (recessive "before" on dark)
    ORANGE = "#d95926"    # categorical slot 2, dark step
    GRAY = "#898781"      # de-emphasis neutral (mode-invariant per palette)
    INK = "#e1e4e8"; INK2 = "#b7bdc5"; MUTED = "#8b9099"
    GRID = "#272b32"; AXIS = "#454c59"; SURF = "#111216"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "pdf.fonttype": 42,
})
if DARK:
    # legend/free text defaults to black otherwise (fine on light, not here)
    plt.rcParams.update({"text.color": INK})

N_LAYERS = 65
PARAMS = 26_895_998_464  # exact main-language parameter count at pinned revision

# ---------------------------------------------------------------- raw data
def jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

phase1 = jsonl(os.path.join(TS, "results", "phase1.jsonl"))
summary = jsonl(os.path.join(TS, "evals", "results", "summary.jsonl"))

FILE_BYTES = {  # ls -l over ~/models/qwen38*, 2026-08-25 (see paper table)
    "UD-IQ2_XXS": 7266070528, "UD-IQ2_S": 8371970048,
    "UD-Q2_K_XL": 9828981664, "bartowski IQ2_S": 10295330400,
    "UD-IQ3_XXS": 10934860704, "UD-IQ3_S": 12040883104,
    "UD-Q4_K_XL": 17559178144,
    "EXL3-2.0bpw": 8573967630 + 1622158601,
}
GIB = 2**30

print("== effective bpw (file_bytes*8 / 26,895,998,464 language parameters) ==")
for k, b in FILE_BYTES.items():
    print(f"  {k:16s} {b/GIB:6.2f} GiB  {b*8/PARAMS:5.2f} bpw")

# Alternate rescore (run the campaign's own script). The original score remains
# visible in every output; the alternate rules were written after failure audit.
res = subprocess.run([sys.executable, os.path.join(TS, "evals", "rescore_math25.py")],
                     capture_output=True, text=True, check=True)
rescore = {}
for line in res.stdout.splitlines()[1:]:
    m = re.match(r"(\S+)\s+(\d+)/25\s+(\d+)/25\s+(\d+)\s+(\d+)", line)
    if m:
        rescore[m.group(1)] = dict(original=int(m.group(2)), alternate=int(m.group(3)),
                                   artifacts=int(m.group(4)), trunc=int(m.group(5)))
# Refuse to plot partial runs.
for label in list(rescore):
    p = os.path.join(TS, "evals", "results", label, "math25.jsonl")
    n = sum(1 for _ in open(p))
    if n < 25:
        print(f"  [skip] {label}: only {n}/25 math25 rows on disk")
        del rescore[label]
print("== MATH-25 original and alternate scores (complete runs) ==")
for k, v in rescore.items():
    print(f"  {k}: {v}")

# Matched-workload generation throughput. The logs contain 25 MATH requests
# followed by 20 HumanEval+ requests. These are same-suite engine metrics, not
# a controlled tg128 comparison.
log = open(os.path.join(TS, "results", "server_EXL3-2.0.log")).read().replace("\n", " ")
reqs = re.findall(r"(\d+) tokens generated in ([\d.]+) seconds\s*\(.*?Generate:\s*([\d.]+)\s*T/s", log)
toks = [int(t) for t, s, g in reqs]
def weighted_rate(records):
    return sum(int(t) for t, _, _ in records) / sum(int(t) / float(g) for t, _, g in records)
if len(reqs) != 45:
    raise RuntimeError(f"expected 45 EXL3 request metrics, found {len(reqs)}")
exl3_math_gen = weighted_rate(reqs[:25])
exl3_he_gen = weighted_rate(reqs[25:])

bart_log = open(os.path.join(TS, "results", "server_bart_hard.log")).read()
bart_reqs = re.findall(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens", bart_log)
if len(bart_reqs) != 45:
    raise RuntimeError(f"expected 45 bartowski request metrics, found {len(bart_reqs)}")
def llama_weighted_rate(records):
    return 1000 * sum(int(t) for _, t in records) / sum(float(ms) for ms, _ in records)
bart_math_gen = llama_weighted_rate(bart_reqs[:25])
bart_he_gen = llama_weighted_rate(bart_reqs[25:])
print("== matched-workload token-weighted generation throughput ==")
print(f"  MATH-25:    EXL3 {exl3_math_gen:.2f} T/s; bartowski {bart_math_gen:.2f} T/s; "
      f"ratio {exl3_math_gen/bart_math_gen:.2f}x")
print(f"  HumanEval+: EXL3 {exl3_he_gen:.2f} T/s; bartowski {bart_he_gen:.2f} T/s; "
      f"ratio {exl3_he_gen/bart_he_gen:.2f}x")

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0.0, c - h), min(1.0, c + h)

print("== Wilson 95% CIs used in the paper ==")
for k, n in [(12,25),(14,25),(16,25),(17,25),(18,25),(20,25),(21,25),(22,25),(23,25),(24,25),
             (11,20),(14,20),(16,20),(18,20),(19,20),(20,20),
             (47,50),(48,50),(49,50)]:
    lo, hi = wilson(k, n)
    print(f"  {k}/{n} = {100*k/n:5.1f}%  CI [{100*lo:.1f}, {100*hi:.1f}]")

# ---------------------------------------------------------------- helpers
def rounded_hbar(ax, y, w, h, color, r_frac=0.35):
    """Horizontal bar, square at baseline, rounded at the data end."""
    r = h * r_frac
    r = min(r, w / 2 if w > 0 else r)
    y0, y1 = y - h/2, y + h/2
    # scale r in x units so the arc looks circular regardless of axis scale
    trans = ax.transData
    p0 = trans.transform((0, 0)); p1 = trans.transform((1, 1))
    rx = r * abs((p1[1]-p0[1]) / (p1[0]-p0[0])) if p1[0] != p0[0] else r
    verts = [(0, y0), (w - rx, y0), (w, y0), (w, y0 + r), (w, y1 - r),
             (w, y1), (w - rx, y1), (0, y1), (0, y0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
             MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none"))

def grid_x(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

# ================================================================ FIG 1: ladder
# Controlled llama-bench tg128 numbers (server tg agreed within +-0.4).
ladder = [
    # label, tok/s, residency ('full'/'partial'), note
    ("UD-IQ2_XXS + MTP (GPU draft)", 30.9, "full", "48–50 on math/code"),
    ("UD-IQ2_XXS",                    25.5, "full", ""),
    ("UD-IQ2_S",                      24.4, "full", ""),
    ("UD-Q2_K_XL  (61/65)",           16.6, "partial", ""),
    ("bartowski IQ2_S  (59/65)",      13.8, "partial", ""),
    ("UD-IQ3_XXS  (54/65)",           11.3, "partial", ""),
    ("UD-IQ3_S  (49/65)",              8.2, "partial", ""),
    ("UD-Q4_K_XL  (33/65)",            4.9, "partial", ""),
    ("CPU floor: UD-IQ2_S  (0/65)",    3.3, "partial", ""),
]
ladder.sort(key=lambda r: -r[1])
fig, ax = plt.subplots(figsize=(6.3, 3.1))
ys = list(range(len(ladder)))[::-1]
for y, (lab, v, res_, note) in zip(ys, ladder):
    c = BLUE if res_ == "full" else GRAY
    rounded_hbar(ax, y, v, 0.62, c)
    txt = f"{v:.1f}"
    ax.text(v + 0.45, y, txt, va="center", ha="left", fontsize=8, color=INK)
    if note:  # MTP content-dependence range marker
        ax.plot([48, 50], [y, y], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot([v + 2.6, 47.2], [y, y], color=AXIS, lw=0.7)
        ax.text(50.9, y, note, va="center", ha="left", fontsize=7.5, color=INK2)
ax.set_yticks(ys)
ax.set_yticklabels([l for l, *_ in ladder], fontsize=8)
ax.set_xlabel("decode speed (tokens/s), ctx 8192, KV q8_0")
ax.set_xlim(0, 69)
ax.set_ylim(-0.65, len(ladder) - 0.35)
grid_x(ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=BLUE, label="all 65 layers on GPU"),
                   Patch(fc=GRAY, label="partial offload (layers on GPU shown)")],
          loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ladder.pdf"))
plt.close(fig)

# ================================================================ FIG 2: frontier
# x = benchmark items/min from logged HTTP-response wall time;
# y = alternate-normalizer accuracy. Post-response local scoring is excluded.
S = {(r["label"], r["suite"]): r for r in summary}
def items_per_min(label):
    return 60 / S[(label, "math25")]["mean_wall_s"]

pts = [
    # label, x items/min, alternate k/25, GiB, residency, label placement
    ("UD-IQ2_XXS+MTP", items_per_min("UD-IQ2_XXS-mtpgpu-effmed"),
     rescore["UD-IQ2_XXS-mtpgpu-effmed"]["alternate"], FILE_BYTES["UD-IQ2_XXS"]/GIB,
     "full", (-9, -18, "right")),
    ("UD-IQ2_S", items_per_min("UD-IQ2_S-effmed"),
     rescore["UD-IQ2_S-effmed"]["alternate"], FILE_BYTES["UD-IQ2_S"]/GIB,
     "full", (0, -28, "center")),
    ("EXL3 2.0 bpw", items_per_min("EXL3-2.0bpw-effmed"),
     rescore["EXL3-2.0bpw-effmed"]["alternate"], FILE_BYTES["EXL3-2.0bpw"]/GIB,
     "full", (9, 7, "left")),
    ("bartowski IQ2_S", items_per_min("bart-IQ2_S-ngl59-effmed"),
     rescore["bart-IQ2_S-ngl59-effmed"]["alternate"], FILE_BYTES["bartowski IQ2_S"]/GIB,
     "partial", (0, -28, "center")),
    ("UD-IQ3_XXS", items_per_min("UD-IQ3_XXS-ngl54-effmed"),
     rescore["UD-IQ3_XXS-ngl54-effmed"]["alternate"], FILE_BYTES["UD-IQ3_XXS"]/GIB,
     "partial", (-9, 7, "right")),
    ("UD-Q4_K_XL", items_per_min("UD-Q4_K_XL-ngl33-effmed"),
     rescore["UD-Q4_K_XL-ngl33-effmed"]["alternate"], FILE_BYTES["UD-Q4_K_XL"]/GIB,
     "partial", (0, -28, "center")),
]
fig, ax = plt.subplots(figsize=(6.3, 3.9))
for lab, x, k, gib, res_, (dx, dy, ha) in pts:
    c = BLUE if res_ == "full" else GRAY
    lo, hi = wilson(k, 25)
    y = 100 * k / 25
    ax.plot([x, x], [100*lo, 100*hi], color=AXIS, lw=1.0, zorder=1)
    ax.scatter([x], [y], s=22 * gib, color=c, edgecolor=SURF, linewidth=1.4, zorder=3)
    correct_min = x * k / 25
    ax.annotate(f"{lab}\n{y:.0f}%  ·  {correct_min:.2f}/min", (x, y),
                textcoords="offset points", xytext=(dx, dy),
                ha=ha, fontsize=7.5, color=INK)
ax.set_xlim(0.12, 1.18)
ax.set_ylim(40, 104)
ax.set_xlabel("MATH-25 items per minute (logged HTTP-response time)")
ax.set_ylabel("MATH-25 alternate-normalizer accuracy (%)")
ax.grid(axis="both", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(fc=BLUE, label="fully GPU-resident"),
                   Patch(fc=GRAY, label="partial offload")],
          loc="lower left", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_frontier.pdf"))
plt.close(fig)

# ================================================================ FIG 3: original->alternate
order = [  # bottom-to-top display order by effective bpw
    ("UD-Q4_K_XL (5.1 bpw)",        "UD-Q4_K_XL-ngl33-effmed"),
    ("UD-IQ3_XXS (3.2 bpw)",        "UD-IQ3_XXS-ngl54-effmed"),
    ("bartowski IQ2_S (3.0 bpw)",   "bart-IQ2_S-ngl59-effmed"),
    ("EXL3 2.0 bpw (3.03 eff.)",    "EXL3-2.0bpw-effmed"),
    ("UD-IQ2_S, xhigh (2.5 bpw)",   "UD-IQ2_S-effxhigh"),
    ("UD-IQ2_S (2.5 bpw)",          "UD-IQ2_S-effmed"),
    ("UD-IQ2_XXS+MTP (2.2 bpw)",    "UD-IQ2_XXS-mtpgpu-effmed"),
]
fig, ax = plt.subplots(figsize=(6.3, 2.9))
for i, (lab, key) in enumerate(order):
    r = rescore[key]
    original, alternate = 100*r["original"]/25, 100*r["alternate"]/25
    ax.plot([original, alternate], [i, i], color=BLUE_L, lw=2.4, zorder=2,
            solid_capstyle="round")
    ax.scatter([original], [i], s=52, color=BLUE_L, edgecolor=SURF, lw=1.2, zorder=3)
    ax.scatter([alternate], [i], s=52, color=BLUE, edgecolor=SURF, lw=1.2, zorder=3)
    ax.annotate(f"+{r['artifacts']} upgraded", ((original+alternate)/2, i),
                textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=7.2, color=INK2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([l for l, _ in order], fontsize=8)
ax.set_xlabel("MATH-25 accuracy (%), n = 25")
ax.set_xlim(40, 100)
ax.set_ylim(-0.6, len(order) - 0.15)
grid_x(ax)
ax.legend(handles=[
    plt.Line2D([], [], marker="o", ls="none", mfc=BLUE_L, mec=SURF, ms=8,
               label="original parser"),
    plt.Line2D([], [], marker="o", ls="none", mfc=BLUE, mec=SURF, ms=8,
               label="alternate normalizer")],
    loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
    fontsize=8, borderaxespad=0.1)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_artifact.pdf"))
plt.close(fig)

# ================================================================ FIG 4: offload decay
NAME = {"Qwen3.8-27B-UD-IQ2_XXS.gguf": "UD-IQ2_XXS", "Qwen3.8-27B-UD-IQ2_S.gguf": "UD-IQ2_S",
        "Qwen3.8-27B-UD-Q2_K_XL.gguf": "UD-Q2_K_XL", "Qwen3.8-27B-IQ2_S.gguf": "bartowski IQ2_S",
        "Qwen3.8-27B-UD-IQ3_XXS.gguf": "UD-IQ3_XXS", "Qwen3.8-27B-UD-IQ3_S.gguf": "UD-IQ3_S",
        "Qwen3.8-27B-UD-Q4_K_XL.gguf": "UD-Q4_K_XL"}
bench = {}
for r in phase1:
    if r.get("stage") == "bench" and r.get("test") == "tg" and r.get("ok"):
        name = NAME[r["model"]]
        ngl = min(r["ngl"], N_LAYERS)
        bench[(name, ngl)] = r["tok_s"]
obs = []
for (name, ngl), tps in bench.items():
    b = FILE_BYTES[name]
    on = b * ngl / N_LAYERS
    obs.append((name, ngl, tps, on, b - on))
# least-squares fit of 1/tps = on/BWg + off/BWc, weighted for RELATIVE error
# (unweighted, the slow partial-offload rows dominate and the full-GPU end
#  is fit poorly; weighting each row by tok/s equalizes relative residuals)
import numpy as np
A = np.array([[on * t, off * t] for _, _, t, on, off in obs])
y = np.ones(len(obs))
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
BWg, BWc = 1.0 / coef[0] / 1e9, 1.0 / coef[1] / 1e9
pred = {(n, g): 1.0 / (on * coef[0] + off * coef[1])
        for (n, g, t, on, off) in obs}
print(f"== offload model fit (relative-error weighted): "
      f"BW_gpu={BWg:.0f} GB/s, BW_cpu={BWc:.0f} GB/s ==")
for (n, g, t, on, off) in sorted(obs, key=lambda o: o[1]):
    p = pred[(n, g)]
    print(f"  {n:16s} ngl={g:2d}  measured {t:5.2f}  model {p:5.2f}  ({100*(p/t-1):+.0f}%)")

fig, ax = plt.subplots(figsize=(6.3, 3.5))
srt = sorted(obs, key=lambda o: o[1])
mx = [g for _, g, *_ in srt]
my = [t for _, _, t, _, _ in srt]
py = [pred[(n, g)] for n, g, *_ in srt]
ax.plot(mx, py, color=ORANGE, lw=1.6, zorder=2, label="two-bandwidth model (linear in bytes)")
ax.scatter(mx, py, marker="D", s=26, facecolor=SURF, edgecolor=ORANGE, lw=1.3, zorder=3)
ax.scatter(mx, my, s=42, color=BLUE, edgecolor=SURF, lw=1.2, zorder=4, label="measured (llama-bench tg128)")
lab_off = {"UD-IQ2_XXS": (-8, 3, "right"), "UD-IQ2_S": (-8, -12, "right"),
           "UD-Q2_K_XL": (-8, 2, "right"), "bartowski IQ2_S": (7, -13, "left"),
           "UD-IQ3_XXS": (-9, 2, "right"), "UD-IQ3_S": (-9, 0, "right"),
           "UD-Q4_K_XL": (0, 10, "center")}
for n, g, t, on, off in srt:
    if (n, g) != ("UD-IQ2_S", 0):
        dx, dy, ha = lab_off[n]
        ax.annotate(f"{n}\n{FILE_BYTES[n]/GIB:.1f} GiB", (g, t),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=7.0, color=INK2)
    else:
        ax.annotate("CPU floor (UD-IQ2_S, ngl 0)", (g, t),
                    textcoords="offset points", xytext=(8, -3), ha="left",
                    fontsize=7.0, color=INK2)
ax.set_xlabel("transformer layers resident on GPU (of 65)")
ax.set_ylabel("decode speed (tokens/s)")
ax.set_xlim(-3, 70)
ax.set_ylim(0, 30)
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_offload.pdf"))
plt.close(fig)

print("figures written to", FIG)
