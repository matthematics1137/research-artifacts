"""Allowlisted, unchanged presentation definitions; no private input loading."""
# Full source SHA256: 44ede405b6ab480f9d0c393ffb60cfc23754dd056e455b9d4948fa079727a77a
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reporting import EVIDENCE_METRICS, evidence_percent, first_measured
DARK = False
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
    "axes.grid": False, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "pdf.fonttype": 42,
})

CELLS = {}  # populated by main(); importing this module never reads results


META = {"Title": "Paper 5 figure", "Author": "Matthew Schwartz",
        "Creator": "paper5/scripts/make_figures.py", "CreationDate": None, "ModDate": None}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), metadata=META, bbox_inches="tight")
    plt.close(fig)
    print(("dark " if DARK else "") + name)


def cell(arm, budget):
    if str(budget).lower() == "none":
        return CELLS.get(f"{arm}@None", CELLS.get(f"{arm}@none"))
    return CELLS.get(f"{arm}@{budget}")


def pct(c, key="accuracy"):
    return 100 * c[key]


def err(c):
    lo, hi = c["wilson95"]
    return [[100 * (c["accuracy"] - lo)], [100 * (hi - c["accuracy"])]]


def fig_arms():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    for ax, budget in zip(axes, (1024, 4096)):
        xs, labels = [], []
        for i, (arm, label, color) in enumerate(ARMS):
            c = cell(arm, budget)
            if not c:
                continue
            ax.bar(i, pct(c), yerr=err(c), color=color, width=0.66, capsize=2.5,
                   error_kw={"elinewidth": 0.9, "ecolor": AXIS})
            ax.text(i, pct(c) + 1.2, f"{pct(c):.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
            xs.append(i); labels.append(label)
        # A margin right of the last bar holds the reference-line labels clear of the bars.
        right = len(xs) - 0.5 + 0.95
        ax.set_xlim(-0.6, right)
        e = cell("E", "none")
        if e:
            ax.axhline(pct(e), color=MUTED, lw=0.9, ls=":")
            ax.text(right - 0.03, pct(e) + 0.6, "no history", fontsize=6.5, color=MUTED, ha="right")
        if budget == 4096:
            o = cell("O", 4096)
            if o:
                ax.axhline(pct(o), color=MUTED, lw=0.9, ls="--")
                ax.text(right - 0.03, pct(o) + 0.6, f"oracle ({o['n']} fitting items)",
                        fontsize=6.5, color=MUTED, ha="right")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right", rotation_mode="anchor")
        ax.set_title(f"budget {budget:,} tokens", fontsize=8.5, color=INK2)
        ax.yaxis.grid(True)
    axes[0].set_ylabel("judged accuracy (%)")
    save(fig, "fig_arms.pdf")


def fig_evidence():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, (metric, xlabel, eligible) in zip(axes, EVIDENCE_METRICS):
        for arm, label, color in ARMS + VARIANTS:
            if arm not in eligible:
                continue
            for budget, marker in ((1024, "o"), (4096, "s")):
                c = cell(arm, budget)
                if not c:
                    continue
                ev = evidence_percent(c, metric)
                if ev is None:
                    continue
                ax.scatter(ev, pct(c), color=color, marker=marker, s=26, zorder=3,
                           label=f"{arm}, {budget:,}")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=6, loc="best", frameon=False,
                      labelcolor=INK2)
    axes[0].set_ylabel("Judged accuracy (%)")
    fig.tight_layout()
    save(fig, "fig_evidence.pdf")


def fig_cost():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    ax = axes[0]
    for i, (arm, label, color) in enumerate(ARMS):
        for j, budget in enumerate((1024, 4096)):
            c = cell(arm, budget)
            if not c or c.get("delivered_prompt_tokens_mean") is None:
                continue
            ax.bar(i + (j - 0.5) * 0.36, c["delivered_prompt_tokens_mean"], width=0.34,
                   color=color, alpha=1.0 if budget == 4096 else 0.55)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([a[1] for a in ARMS], fontsize=7, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel("delivered prompt tokens (mean)")
    ax.set_title("what each policy actually delivers\n(pale: 1,024 budget; solid: 4,096)", fontsize=7.5, color=INK2)
    ax.yaxis.grid(True)

    ax = axes[1]
    for i, (arm, label, color) in enumerate(ARMS):
        c = cell(arm, 4096)
        if not c:
            continue
        gen = c.get("generation_seconds_mean")
        pol = first_measured(c.get("diag_means") or {}, "policy_seconds", "selection_seconds")
        if gen is not None:
            ax.bar(i, gen, color=color, width=0.66)
        if gen is not None and pol is not None:
            ax.bar(i, pol, bottom=gen, color=color, width=0.66, alpha=0.45)
        if gen is None or pol is None:
            ax.text(i, (gen or 0) + 0.3, "cost incomplete", ha="center", va="bottom",
                    fontsize=5.5, color=INK2)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([a[1] for a in ARMS], fontsize=7, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel("recorded seconds per question (mean)")
    ax.set_title("generation + recorded selection\n(accounting differs by policy)", fontsize=7.5, color=INK2)
    ax.yaxis.grid(True)
    save(fig, "fig_cost.pdf")


def render_figures(claims, directory, dark=False):
    """Draw supplied aggregates; the CLI is the evidence gate, tests use temporary paths."""
    global CELLS, FIG, DARK
    palette = ("#3987e5", "#1c5cab", "#d95926", "#8c3a1a", "#898781", "#e1e4e8", "#b7bdc5", "#8b9099",
               "#272b32", "#454c59", "#111216") if dark else (
               "#2a78d6", "#86b6ef", "#eb6834", "#f4a380", "#898781", "#0b0b0b", "#52514e", "#898781",
               "#e1e0d9", "#c3c2b7", "#ffffff")
    globals().update(zip(("BLUE", "BLUE_L", "ORANGE", "ORANGE_L", "GRAY", "INK", "INK2", "MUTED", "GRID", "AXIS", "SURF"), palette))
    global ARMS, VARIANTS
    ARMS = [("T", "truncation", BLUE), ("S", "summaries", ORANGE),
            ("C", "LLMLingua-2", GRAY), ("M", "retrieval", BLUE_L)]
    VARIANTS = [("Cw", "LLMLingua-2\n(recent window)", ORANGE_L), ("Mbm25", "retrieval\n(BM25)", GRAY)]
    plt.rcParams.update({"text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                         "grid.color": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF})
    CELLS, FIG, DARK = claims["cells"], str(directory), dark
    fig_arms()
    fig_evidence()
    fig_cost()

