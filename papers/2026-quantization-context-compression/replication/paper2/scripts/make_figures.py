#!/usr/bin/env python3
"""Build Paper 2's manuscript figures (light; --dark for dark variants).

The sole quantitative input is paper2/derived/claims.json — this script never
reads evaluation rows, window files, or any other suite data, so the figures
are privacy-safe by construction (the claims file is aggregates only), and
test_claims.py enforces that this stays the script's only file read. Palette and chrome match paper 1's figure system: light surface with
palette blue #2a78d6 / orange #eb6834; dark mode is *selected* dark steps
(#3987e5 / #d95926) on the dark page surface #111216, never a flipped light
figure.
"""

import json, os, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DARK = "--dark" in sys.argv[1:]
SELF_TEST = "--self-test" in sys.argv[1:]
HERE = os.path.dirname(os.path.abspath(__file__))


def padded_limits(values, *, include_zero=False, nonnegative=False):
    """Outcome-derived plot bounds with enough headroom for annotations."""
    vals = [float(value) for value in values]
    if not vals:
        raise ValueError("axis limits require at least one value")
    lo, hi = min(vals), max(vals)
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = hi - lo
    pad = max(1.0, span * 0.10, max(abs(lo), abs(hi), 1.0) * 0.03)
    lower = lo - pad
    upper = hi + pad
    if nonnegative:
        lower = 0.0
    return lower, upper


def self_test_axis_policy():
    # A fresh corrective result may reverse the old ordering or lie far beyond
    # the invalid P2G draft's hand-picked 0..70/0..7.6 ranges.
    lo, hi = padded_limits([-12.0, 95.0], include_zero=True)
    assert lo < -12.0 and hi > 95.0 and lo <= 0 <= hi
    lo, hi = padded_limits([0.0, 25.0], nonnegative=True)
    assert lo == 0.0 and hi > 25.0


if SELF_TEST:
    self_test_axis_policy()
    print("Paper 2 figure axis-policy synthetic reversal/range test passed")
    raise SystemExit(0)

# A direct Python invocation is a supported mutation entry point, so it must
# enforce the same promotion/aggregate-claims gate as `make figures` before it
# creates directories or output files.
try:
    subprocess.run(
        [sys.executable, os.path.join(HERE, "check_release_state.py"), "--pre-figure"],
        check=True,
    )
except subprocess.CalledProcessError as exc:
    raise SystemExit("STOP: Paper 2 pre-figure release gate failed") from exc
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
if DARK:
    FIG = os.path.join(FIG, "dark")
os.makedirs(FIG, exist_ok=True)

if not DARK:
    BLUE = "#2a78d6"; BLUE_L = "#86b6ef"; ORANGE = "#eb6834"; ORANGE_L = "#f4a380"
    GRAY = "#898781"
    INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
    GRID = "#e1e0d9"; AXIS = "#c3c2b7"; SURF = "#ffffff"
else:
    BLUE = "#3987e5"; BLUE_L = "#1c5cab"; ORANGE = "#d95926"; ORANGE_L = "#8c3a1a"
    GRAY = "#898781"
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
    plt.rcParams.update({"text.color": INK})

CLAIMS = json.load(open(os.path.join(HERE, "..", "derived", "claims.json")))
GRID_D = CLAIMS["grid"]

TIERS = [("Qf", "Q4_K_XL 5.1 bpw", BLUE),
         ("Ex", "EXL3 2.0 bpw", ORANGE),
         ("Iq", "IQ2_S 2.5 bpw", GRAY)]
CONDS = [("O", "original"), ("L", "LLMLingua-2"),
         ("A", "stopword rung"), ("E", "full ladder")]
CORPORA = [("Wc", "WildChat (public)"), ("Tsi", "private corpus (tsi)")]

META = {"Title": "Paper 2 figure", "Author": "Matthew Schwartz",
        "Creator": "paper2/scripts/make_figures.py",
        "CreationDate": None, "ModDate": None}


def cell(corpus, tier, cond):
    return GRID_D[corpus]["cells"][f"{tier}-{cond}"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), metadata=META)
    plt.close(fig)
    print(("dark " if DARK else "") + name)


def grouped_panels(value, err=None, ylabel="", name="", ylim=None, fmt="{:.0f}"):
    """Two corpus panels; condition groups on x, one bar per tier."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    width = 0.24
    for ax, (corpus, ctitle) in zip(axes, CORPORA):
        for j, (tier, tlabel, color) in enumerate(TIERS):
            xs = [i + (j - 1) * width for i in range(len(CONDS))]
            ys = [value(corpus, tier, cond) for cond, _ in CONDS]
            ax.bar(xs, ys, width=width * 0.92, color=color, zorder=2,
                   label=tlabel if corpus == "Wc" else None)
            if err is not None:
                for x, (cond, _) in zip(xs, CONDS):
                    lo, hi = err(corpus, tier, cond)
                    ax.plot([x, x], [lo, hi], color=INK2, lw=0.9, zorder=3)
            for x, y in zip(xs, ys):
                ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                            xytext=(0, 2.5), ha="center", fontsize=6.6,
                            color=INK2, zorder=4)
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([lbl for _, lbl in CONDS], fontsize=7.3)
        ax.set_title(ctitle, fontsize=8.5, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        if ylim == "auto-nonnegative":
            all_values = [
                value(c, tier, cond)
                for c, _ in CORPORA for tier, _, _ in TIERS for cond, _ in CONDS
            ]
            ax.set_ylim(*padded_limits(all_values, nonnegative=True))
        elif ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=7.3, loc="upper right",
                   handlelength=1.0, borderaxespad=0.1)
    fig.tight_layout(pad=0.4)
    save(fig, name)


# ---- Figure 1: the 24-cell grid -------------------------------------------
grouped_panels(
    value=lambda c, t, d: cell(c, t, d)["acc"] * 100,
    err=lambda c, t, d: [v * 100 for v in cell(c, t, d)["wilson"]],
    ylabel="exact accuracy (%)", name="fig_grid.pdf", ylim=(0, 108))

# ---- Figure 2: damage vs. retention ---------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.0))
groups = {"L": (BLUE, "LLMLingua-2"), "A": (ORANGE, "stopword rung"),
          "E": (GRAY, "full ladder")}
ratio_xs_all, ratio_ys_all = [], []
for cond, (color, label) in groups.items():
    xs, ys = [], []
    for corpus, _ in CORPORA:
        keep = CLAIMS["corpora"][corpus]["char_retention_vs_O"][cond] * 100
        for tier, _, _ in TIERS:
            pv = cell(corpus, tier, cond)["paired_vs_O"]
            xs.append(keep)
            ys.append(pv["delta"] * 100)
    ratio_xs_all.extend(xs)
    ratio_ys_all.extend(ys)
    marks = ["s", "s", "s", "o", "o", "o"]  # square = WildChat, circle = tsi
    for x, y, m in zip(xs, ys, marks):
        ax.scatter([x], [y], s=34, marker=m, color=color, edgecolor=SURF,
                   linewidth=1.0, zorder=3)
    ax.annotate(label, (sum(xs) / 6 + 1.6, sum(ys) / 6),
                fontsize=8, color=color, va="center")
ax.set_xlabel("characters retained (% of original window)")
ax.set_ylabel("original minus compressed accuracy (pp)")
ax.set_xlim(*padded_limits(ratio_xs_all))
ax.set_ylim(*padded_limits(ratio_ys_all, include_zero=True))
ax.grid(axis="both", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.scatter([], [], marker="s", color=INK2, label="WildChat")
ax.scatter([], [], marker="o", color=INK2, label="tsi (private)")
ax.legend(frameon=False, fontsize=7.3, loc="lower right")
fig.tight_layout(pad=0.4)
save(fig, "fig_ratio.pdf")

# ---- Figure 3: accuracy per wall-clock minute -----------------------------
grouped_panels(
    value=lambda c, t, d: cell(c, t, d)["acc_per_min"],
    ylabel="correct answers / minute", name="fig_accmin.pdf",
    ylim="auto-nonnegative", fmt="{:.1f}")

# ---- Figure 4: answer-survival decomposition (healthy tier) ---------------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
for ax, (corpus, ctitle) in zip(axes, CORPORA):
    sv = CLAIMS["survival"][corpus]
    o_acc = cell(corpus, "Qf", "O")["acc"] * 100
    bars = []
    for cond, color, light, label in (("L", BLUE, BLUE_L, "LLMLingua-2"),
                                      ("E", ORANGE, ORANGE_L, "full ladder")):
        e = sv[cond]; t = e["tiers"]["Qf"]
        bars.append((f"{label}\nkept (n={t['surv_n']})",
                     t["surv_correct"], t["surv_n"], color))
        bars.append((f"{label}\ndeleted (n={t['dest_n']})",
                     t["dest_correct"], t["dest_n"], light))
    xs = range(len(bars))
    for x, (lbl, k, m, color) in zip(xs, bars):
        acc = k / m * 100 if m else 0.0
        ax.bar([x], [acc], width=0.62, color=color, zorder=2)
        ax.annotate(f"{acc:.0f}", (x, acc), textcoords="offset points",
                    xytext=(0, 2.5), ha="center", fontsize=6.8, color=INK2)
    ax.axhline(o_acc, color=AXIS, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"original: {o_acc:.0f}", (3.45, o_acc), fontsize=6.8,
                color=INK2, va="bottom", ha="right")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.0)
    ax.set_title(ctitle, fontsize=8.5, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)
axes[0].set_ylabel("exact accuracy (%)")
fig.tight_layout(pad=0.4)
save(fig, "fig_survival.pdf")
