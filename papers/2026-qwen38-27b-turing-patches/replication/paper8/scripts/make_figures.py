#!/usr/bin/env python3
"""P8 figures from the stage reports' results files; no model execution.

fig_reach.pdf        (a) peak GPU memory against filled context per KV-cache format, with the pass limit and the card;
                     (b) graded retrieval accuracy per arm and length, with 95% Wilson intervals
fig_comparisons.pdf  the paired differences with their 90% bootstrap intervals, against the registered bounds
                     (usable: lower bound above -10 points; equivalent: within +-5)
fig_speed.pdf        stock versus the selected Turing-patched build in F2 (testsuite/results/turing/ab/final2.jsonl): pooled
                     tokens/s over three rounds with the range of the round values, undrafted and with MTP drafting,
                     at an empty context and at 28K


Light versions go to figures/, dark versions (--dark) to figures/dark/. Colours follow the validated categorical
palette of the dataviz reference, as in paper6/scripts/make_figures.py; identity is also carried by marker shape and
labels. Run after build_claims.py, whose generated macros supply the paired 96K-64K gap.
"""
import argparse
import json
import os
import re

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER)
R = os.path.join(REPO, "testsuite", "results", "turing", "longctx")
FORMATS = ("8,8", "8,4", "4,4", "fp16")
LABEL = {"8,8": "8,8 (8-bit)", "8,4": "8,4 (split)", "4,4": "4,4 (4-bit)", "fp16": "fp16"}
MARKER = {"8,8": "o", "8,4": "D", "4,4": "^", "fp16": "s"}
STYLE = {"8,8": "-", "8,4": "--", "4,4": "-.", "fp16": ":"}
LIMIT_MIB, CARD_MIB = 10752, 11264


def theme(dark):
    if dark:
        return {"ink": "#e1e4e8", "muted": "#a7adb7", "bg": "#111216", "grid": "#343a45", "band": "#3a404c",
                "band2": "#1d2129", "hatch": "#5a6170", "series": ("#3987e5", "#d95926", "#199e70", "#c98500")}
    return {"ink": "#151515", "muted": "#5f5e5a", "bg": "#ffffff", "grid": "#e3e3e0", "band": "#c9c6bd",
            "band2": "#f1efe9", "hatch": "#8f8b80", "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")}


def context(t):
    return plt.rc_context({"font.size": 8, "text.color": t["ink"], "axes.labelcolor": t["ink"],
                           "xtick.color": t["ink"], "ytick.color": t["ink"], "axes.edgecolor": t["grid"],
                           "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
                           "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                           "legend.frameon": False})


def load(*parts):
    with open(os.path.join(R, *parts)) as f:
        return json.load(f)


def colour(t, fmt):
    return t["series"][FORMATS.index(fmt)]


def claims():
    with open(os.path.join(PAPER, "derived", "claims.tex")) as f:
        return dict(re.findall(r"\\newcommand\{\\(Pe[A-Za-z]+)\}\{(.*)\}", f.read()))


def value(text):
    return float(re.sub(r"[^0-9.+-]", "", text.replace("\\ensuremath", "")))


def reach_panel(ax, t):
    s = load("stage1", "results.json")
    for fmt in FORMATS:
        steps = sorted((x for x in s["steps"] if x["format"] == fmt), key=lambda x: x["length"])
        xs = [x["length"] / 1024 for x in steps]
        ys = [x["peak_mib"] for x in steps]
        c = colour(t, fmt)
        ax.plot(xs, ys, linestyle=STYLE[fmt], color=c, linewidth=1.4)
        for x, y, st in zip(xs, ys, steps):
            fail = st["verdict"] != "pass"
            ax.plot(x, y, marker="X" if fail else MARKER[fmt], markersize=7 if fail else 6, linestyle="none",
                    markerfacecolor=t["bg"] if fail else c, markeredgecolor=c, markeredgewidth=1.4 if fail else .8)
        ax.plot([], [], marker=MARKER[fmt], linestyle=STYLE[fmt], color=c, label=LABEL[fmt], markersize=5)
    ax.axhline(CARD_MIB, color=t["muted"], linewidth=.9)
    ax.axhline(LIMIT_MIB, color=t["muted"], linewidth=.9, linestyle=(0, (4, 2)))
    ax.text(140, CARD_MIB + 25, "card, 11,264 MiB", ha="right", va="bottom", fontsize=7, color=t["muted"])
    ax.text(140, LIMIT_MIB - 30, "pass limit, 10,752 MiB", ha="right", va="top", fontsize=7, color=t["muted"])
    ax.annotate("8,8 at 80K: 10,777 MiB,\nover the limit", xy=(80, 10777), xytext=(84, 10930), fontsize=7,
                color=t["ink"], arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": .7})
    ax.set_xlim(26, 146)
    ax.set_ylim(8800, 11450)
    ax.set_xticks([32, 48, 64, 80, 96, 112, 128, 144])
    ax.set_xlabel("Filled context (K tokens)")
    ax.set_ylabel("Peak card memory (MiB)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("(a) Reach: peak memory per cache format", fontsize=9, color=t["ink"], loc="left")
    ax.grid(axis="y", color=t["grid"], linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7, handlelength=2.6)


def accuracy_panel(ax, t):
    s2, s3, arm = load("stage2", "results.json"), load("stage3", "results.json"), load("fd75d-arm", "results.json")
    a2 = s2["arms"]
    rows = [  # (group x, offset, format, accuracy record, open marker for the Triton-read 8-bit cache)
        (0, -.24, "fp16", a2["s2-32k-fp16"]["accuracy"], False),
        (0, -.08, "8,8", a2["s2-32k-88"]["accuracy"], False),
        (0, .24, "4,4", a2["s2-32k-44"]["accuracy"], False),
        (1, -.24, "8,8", a2["s2-64k-88"]["accuracy"], False),
        (1, -.08, "8,8", arm["arms"]["s2-64k-88-triton"]["accuracy"], True),
        (1, .08, "8,4", a2["s2-64k-84"]["accuracy"], False),
        (1, .24, "4,4", a2["s2-64k-44"]["accuracy"], False),
        (2, .08, "8,4", s3["arms"]["s3-96k"]["accuracy"], False),
    ]
    for gx, off, fmt, acc, hollow in rows:
        c = colour(t, fmt)
        y = 100 * acc["k"] / acc["n"]
        lo, hi = (100 * v for v in acc["wilson95"])
        ax.errorbar([gx + off], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=c, elinewidth=1.2, capsize=0)
        ax.plot(gx + off, y, marker=MARKER[fmt], markersize=6, linestyle="none", markeredgecolor=c,
                markerfacecolor=t["bg"] if hollow else c, markeredgewidth=1.4 if hollow else .8)
    ax.text(1 - .24, 59.5, "fd75d", ha="center", va="top", fontsize=6.5, color=t["muted"])
    ax.text(1 - .08, 56.0, "Triton", ha="center", va="top", fontsize=6.5, color=t["muted"])
    ax.text(2 + .08, 41.5, "48 items", ha="center", va="top", fontsize=6.5, color=t["muted"])
    for fmt in FORMATS:
        ax.plot([], [], marker=MARKER[fmt], color=colour(t, fmt), linestyle="none", markersize=5, label=LABEL[fmt])
    ax.plot([], [], marker="o", color=colour(t, "8,8"), markerfacecolor=t["bg"], linestyle="none", markersize=5,
            label="8,8 read by Triton")
    ax.legend(loc="lower left", fontsize=6.5, handlelength=1.2, ncol=1)
    ax.set_xticks([0, 1, 2], ["32K", "64K", "96K"])
    ax.set_xlim(-.55, 2.55)
    ax.set_ylim(35, 95)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Items fully right (%)")
    ax.set_title("(b) Graded accuracy, 95% Wilson", fontsize=9, color=t["ink"], loc="left")
    ax.grid(axis="y", color=t["grid"], linewidth=.5)
    ax.set_axisbelow(True)


def draw_reach(dark=False):
    t = theme(dark)
    with context(t):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
        reach_panel(axes[0], t)
        accuracy_panel(axes[1], t)
        fig.tight_layout(w_pad=1.6)
        save(fig, "fig_reach.pdf", dark)


def draw_comparisons(dark=False):
    t = theme(dark)
    s2, arm, m = load("stage2", "results.json"), load("fd75d-arm", "results.json"), claims()
    c = s2["comparisons"]
    rows = [  # label, difference, (lo, hi), registered comparison
        ("H1: 8,8 $-$ fp16 at 32K", c["h1"]["diff_points"], c["h1"]["ci90"], True),
        ("H3: 4,4 $-$ fp16 at 32K", c["h3"]["diff_points"], c["h3"]["ci90"], True),
        ("H2 (primary): 4,4 $-$ 8,8 at 64K", c["h2"]["diff_points"], c["h2"]["ci90"], True),
        ("H2s: 8,4 $-$ 8,8 at 64K", c["h2s"]["diff_points"], c["h2s"]["ci90"], True),
        ("K1: 8,8 by Triton $-$ 8,8 by fd75d, 64K", arm["K1"]["diff_points"], arm["K1"]["ci90"], True),
        ("K2: 8,4 $-$ 8,8, both by Triton, 64K", arm["K2"]["diff_points"], arm["K2"]["ci90"], True),
        ("8,4: 96K $-$ 64K, 48 items (descriptive)", value(m["PeGapDiff"]),
         (value(m["PeGapLo"]), value(m["PeGapHi"])), False),
    ]
    with context(t):
        fig, ax = plt.subplots(figsize=(7, 2.75))
        ax.axvspan(-45, -10, facecolor=t["band"], edgecolor=t["hatch"], hatch="///", linewidth=0)
        ax.axvspan(-5, 5, color=t["band2"], linewidth=0)
        ax.axvline(0, color=t["muted"], linewidth=.8)
        ax.axvline(-10, color=t["muted"], linewidth=.8, linestyle=(0, (4, 2)))
        ys, labels = [], []
        for i, (label, d, (lo, hi), registered) in enumerate(rows):
            y = len(rows) - 1 - i if registered else -.45
            ink = t["ink"] if registered else t["muted"]
            ax.plot([lo, hi], [y, y], color=ink, linewidth=1.6, solid_capstyle="butt")
            ax.plot(d, y, marker="o" if registered else "D", markersize=5.5, linestyle="none", markeredgecolor=ink,
                    markerfacecolor=ink if registered else t["bg"], markeredgewidth=1.2)
            text = f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]".replace("+0.0", "0.0").replace("-0.0", "0.0")
            ax.text(22, y, text.replace("-", "$-$"), ha="left", va="center", fontsize=7, color=ink)
            ys.append(y)
            labels.append(label)
        ax.axhline(.3, color=t["grid"], linewidth=.6)
        ax.set_yticks(ys, labels, fontsize=7.2)
        for tick, (*_, registered) in zip(ax.get_yticklabels(), rows):
            tick.set_color(t["ink"] if registered else t["muted"])
        ax.tick_params(axis="y", length=0)
        top = len(rows) - .35
        ax.text(-10.6, top, "a lower bound here:\nnot usable", ha="right", va="bottom", fontsize=6.5,
                color=t["ink"], bbox={"facecolor": t["bg"], "edgecolor": "none", "pad": 1.2})   # legible on the hatching
        ax.text(5.6, top, "equivalent if the whole\ninterval is within $\\pm$5 points", ha="left", va="bottom",
                fontsize=6.5, color=t["muted"])
        ax.set_xlim(-45, 36)
        ax.set_ylim(-1.05, len(rows) + .55)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([-40, -30, -20, -10, -5, 0, 5, 10, 20])
        ax.set_xlabel("Paired difference in items fully right (points), with its 90% bootstrap interval")
        fig.tight_layout()
        save(fig, "fig_comparisons.pdf", dark)


SPEED_ROWS = (("stock", "stock 1.6.5"), ("w13s2", "Turing patches"))


def speed_rounds():
    """Pooled tokens/s per final2 arm, overall and per round (total decoded tokens over total decode time)."""
    rows = [json.loads(x) for x in open(os.path.join(REPO, "testsuite", "results", "turing", "ab", "final2.jsonl"))
            if x.strip()]
    acc = {}
    for r in rows:
        arm, rnd = re.fullmatch(r"final2-(.+)-r(\d)", r["tag"]).groups()
        build = {"w13": "w13s1"}.get(arm.split("-")[0], arm.split("-")[0])
        key = (build, r["mtp"], r["fill"])
        tok, sec = acc.setdefault(key, {}).get(rnd, (0, 0.0))
        acc[key][rnd] = (tok + r["new_tokens"], sec + r["decode_wall_s"])
    out = {}
    for key, rounds in acc.items():
        pooled = sum(t for t, _ in rounds.values()) / sum(s for _, s in rounds.values())
        out[key] = (pooled, [t / s for t, s in rounds.values()])
    return out


def draw_speed(dark=False):
    t = theme(dark)
    data = speed_rounds()
    with context(t):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.35), sharey=True)
        for ax, mtp, title in ((axes[0], False, "(a) Undrafted"), (axes[1], True, "(b) With MTP drafting")):
            for i, (build, _) in enumerate(SPEED_ROWS):
                y = len(SPEED_ROWS) - 1 - i
                # Mode 2 changes verify passes only; its undrafted code and
                # registered observation are the w13 mode-1 undrafted arm.
                measured_build = "w13s1" if build == "w13s2" and not mtp else build
                for fill, dy, hollow in ((0, .17, False), (28000, -.17, True)):
                    if (measured_build, mtp, fill) not in data:
                        continue
                    pooled, rounds = data[(measured_build, mtp, fill)]
                    ax.plot([min(rounds), max(rounds)], [y + dy, y + dy], color=t["muted"], linewidth=1.6,
                            solid_capstyle="butt")
                    ax.plot(pooled, y + dy, marker="o", markersize=5.5, linestyle="none", markeredgecolor=t["ink"],
                            markerfacecolor=t["bg"] if hollow else t["ink"], markeredgewidth=1.2)
                    ax.text(max(max(rounds), pooled) + 2.4, y + dy, f"{pooled:.2f}", ha="left", va="center",
                            fontsize=6.5, color=t["ink"])
            ax.set_title(title, fontsize=9, color=t["ink"], loc="left")
            ax.set_xlim(0, 86)
            ax.set_xticks([0, 20, 40, 60, 80])
            ax.grid(axis="x", color=t["grid"], linewidth=.5)
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel("Decode speed (tokens/s)")
        axes[0].set_yticks(range(len(SPEED_ROWS)), [label for _, label in reversed(SPEED_ROWS)], fontsize=7.2)
        axes[0].set_ylim(-.6, len(SPEED_ROWS) - .4)
        axes[0].plot([], [], marker="o", markersize=5, linestyle="none", color=t["ink"], label="empty context")
        axes[0].plot([], [], marker="o", markersize=5, linestyle="none", markeredgecolor=t["ink"],
                     markerfacecolor=t["bg"], label="28K-token context")
        axes[0].plot([], [], color=t["muted"], linewidth=1.6, label="range of the three rounds")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01),
                   fontsize=6.5, handlelength=1.6, ncol=3)
        fig.tight_layout(w_pad=1.2, rect=(0, .16, 1, 1))
        save(fig, "fig_speed.pdf", dark)


def save(fig, name, dark):
    folder = os.path.join(PAPER, "figures", "dark" if dark else "")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, name), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()
    draw_reach(args.dark)
    draw_comparisons(args.dark)
    draw_speed(args.dark)


if __name__ == "__main__":
    main()
