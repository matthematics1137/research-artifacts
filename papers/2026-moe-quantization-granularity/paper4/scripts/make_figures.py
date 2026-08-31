#!/usr/bin/env python3
"""Build Paper 4's supplementary, analysis-only figures.

The sole quantitative input is ``paper4/derived/claims.json``. In particular,
this module does not read evaluation summaries or raw result rows and does not
reconstruct any of the claims checker statistics. The figures are not included
in the manuscript; they are diagnostic views of its frozen, audited claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CLAIMS = PAPER_DIR / "derived" / "claims.json"
DEFAULT_FIGURES = PAPER_DIR / "figures"

MODEL_SPECS = (
    (
        "Mixtral-8x7B",
        (
            ("Q4 K M", "mixtral_q4_k_m", "MIXTRAL-Q4KM-n100/gsm8k"),
            ("IQ2 XXS", "mixtral_iq2_xxs", "MIXTRAL-IQ2XXS-n100/gsm8k"),
            ("IQ1 M", "mixtral_iq1_m", "MIXTRAL-IQ1M-n100/gsm8k"),
        ),
    ),
    (
        "Qwen3-30B-A3B",
        (
            ("Q4 K M", "qwen3moe_q4_k_m", "QWEN3MOE-Q4KM/gsm8k"),
            ("IQ2 XXS", "qwen3moe_iq2_xxs", "QWEN3MOE-IQ2XXS/gsm8k"),
            ("IQ1 M", "qwen3moe_iq1_m", "QWEN3MOE-IQ1M/gsm8k"),
        ),
    ),
)

COMPARISON_SPECS = (
    ("Mixtral IQ1 M", "mixtral_iq1_vs_q4", "mixtral"),
    ("Mixtral IQ2 XXS", "mixtral_iq2_vs_q4", "mixtral"),
    ("Qwen IQ1 M", "qwen3moe_iq1_vs_q4", "qwen"),
    ("Qwen IQ2 XXS", "qwen3moe_iq2_vs_q4", "qwen"),
)

PDF_METADATA = {
    "Title": "Paper 4 supplementary analysis figure",
    "Author": "Matthew Schwartz",
    "Creator": "paper4/scripts/make_figures.py",
    "CreationDate": None,
    "ModDate": None,
}


def load_claims(path: Path = DEFAULT_CLAIMS) -> dict[str, Any]:
    """Load the checker-produced claims artifact and no other evidence file."""
    with path.open("r", encoding="utf-8") as handle:
        claims = json.load(handle)
    if claims.get("schema_version") not in {"1.0.0", "paper4-p4r1-claims-v1"}:
        raise ValueError(f"unsupported claims schema: {claims.get('schema_version')!r}")
    return claims


def extract_plot_data(claims: dict[str, Any]) -> dict[str, Any]:
    """Select the audited values used by all three figures."""
    panels: list[dict[str, Any]] = []
    for model_label, tier_specs in MODEL_SPECS:
        tiers: list[dict[str, Any]] = []
        for tier_label, model_key, cell_key in tier_specs:
            model = claims["models"][model_key]
            cell = claims["eval_cells"][cell_key]
            tiers.append(
                {
                    "label": tier_label,
                    "effective_bpw": float(model["effective_file_bpw"]),
                    "correct": int(cell["correct"]),
                    "n": int(cell["n"]),
                    "accuracy": float(cell["accuracy"]),
                    "wilson_95": tuple(float(value) for value in cell["wilson_95"]),
                }
            )
        panels.append({"model": model_label, "tiers": tiers})

    differences: list[dict[str, Any]] = []
    for label, comparison_key, family in COMPARISON_SPECS:
        paired = claims["comparisons"][comparison_key]["paired"]
        differences.append(
            {
                "label": label,
                "family": family,
                "n": int(paired["n"]),
                "difference": float(paired["risk_difference_healthy_minus_low"]),
                "interval": tuple(
                    float(value)
                    for value in paired["risk_difference_95_newcombe_method_10"]
                ),
            }
        )

    data = {"panels": panels, "differences": differences}
    validate_plot_data(data)
    return data


def validate_plot_data(data: dict[str, Any]) -> None:
    """Fail closed if the selected claim cells are malformed or inconsistent."""
    if len(data["panels"]) != 2 or any(len(panel["tiers"]) != 3 for panel in data["panels"]):
        raise ValueError("expected two models with exactly three quantization tiers each")
    for panel in data["panels"]:
        labels = [tier["label"] for tier in panel["tiers"]]
        if labels != ["Q4 K M", "IQ2 XXS", "IQ1 M"]:
            raise ValueError(f"unexpected tier order for {panel['model']}: {labels}")
        for tier in panel["tiers"]:
            if tier["n"] <= 0 or not 0 <= tier["correct"] <= tier["n"]:
                raise ValueError(f"invalid score count: {tier}")
            if not math.isclose(
                tier["accuracy"], tier["correct"] / tier["n"], rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"accuracy/count mismatch: {tier}")
            lo, hi = tier["wilson_95"]
            if not (0 <= lo <= tier["accuracy"] <= hi <= 1):
                raise ValueError(f"invalid Wilson interval: {tier}")
            if tier["effective_bpw"] <= 0:
                raise ValueError(f"invalid effective bpw: {tier}")
    if len(data["differences"]) != 4:
        raise ValueError("expected four paired Q4-versus-low comparisons")
    for comparison in data["differences"]:
        lo, hi = comparison["interval"]
        if not (lo <= comparison["difference"] <= hi):
            raise ValueError(f"paired difference outside its interval: {comparison}")


def theme(dark: bool) -> dict[str, Any]:
    if dark:
        palette = {
            "surface": "#111216",
            "ink": "#e1e4e8",
            "muted": "#9399a3",
            "blue": "#58a6ff",
            "orange": "#ff7b52",
            "gray": "#9ba2ad",
        }
    else:
        palette = {
            "surface": "white",
            "ink": "#1a1a1a",
            "muted": "#74716b",
            "blue": "#2a78d6",
            "orange": "#d95825",
            "gray": "#777b83",
        }
    plt.rcParams.update(
        {
            "figure.facecolor": palette["surface"],
            "axes.facecolor": palette["surface"],
            "savefig.facecolor": palette["surface"],
            "text.color": palette["ink"],
            "axes.edgecolor": palette["muted"],
            "axes.labelcolor": palette["ink"],
            "xtick.color": palette["ink"],
            "ytick.color": palette["ink"],
            "font.size": 10,
            "axes.grid": True,
            "grid.color": palette["muted"],
            "grid.alpha": 0.23,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return palette


def save_pdf(fig: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata=PDF_METADATA)
    plt.close(fig)


def plot_within_model_scores(
    data: dict[str, Any], output_dir: Path, palette: dict[str, Any]
) -> None:
    """Plot Q4/IQ2/IQ1 scores in separate model panels."""
    tier_colors = [palette["gray"], palette["blue"], palette["orange"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.6), sharey=True)
    for index, (ax, panel) in enumerate(zip(axes, data["panels"])):
        for x, (tier, color) in enumerate(zip(panel["tiers"], tier_colors)):
            accuracy = tier["accuracy"] * 100
            lo, hi = (value * 100 for value in tier["wilson_95"])
            ax.bar(x, accuracy, width=0.62, color=color, zorder=3)
            ax.errorbar(
                x,
                accuracy,
                yerr=[[accuracy - lo], [hi - accuracy]],
                fmt="none",
                ecolor=palette["ink"],
                elinewidth=1.0,
                capsize=3,
                zorder=4,
            )
            ax.text(
                x,
                min(103, hi + 2.5),
                f"{tier['correct']}/{tier['n']}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(range(3), [tier["label"] for tier in panel["tiers"]])
        ax.set_title(panel["model"], fontsize=11)
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)
        if index == 0:
            ax.set_ylabel("GSM8K accuracy (%)")
    fig.suptitle("GSM8K scores across quantization tiers within each model", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Bars use the same 100 items; whiskers are 95% Wilson intervals.",
        ha="center",
        fontsize=8.5,
        color=palette["muted"],
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    save_pdf(fig, output_dir / "fig_matched.pdf")


def plot_paired_differences(
    data: dict[str, Any], output_dir: Path, palette: dict[str, Any]
) -> None:
    """Plot audited paired risk differences and Newcombe intervals."""
    fig, ax = plt.subplots(figsize=(6.5, 3.75))
    for y, comparison in enumerate(data["differences"]):
        estimate = comparison["difference"] * 100
        lo, hi = (value * 100 for value in comparison["interval"])
        color = palette["orange"] if comparison["family"] == "mixtral" else palette["blue"]
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - lo], [hi - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=6,
            elinewidth=1.6,
            capsize=3,
            zorder=4,
        )
        ax.text(hi + 1.3, y, f"{estimate:.0f} pp", va="center", fontsize=9)
    ax.axvline(0, color=palette["ink"], linewidth=0.9, linestyle="--", zorder=2)
    ax.axhline(1.5, color=palette["muted"], linewidth=0.6, alpha=0.45)
    ax.set_yticks(range(4), [item["label"] for item in data["differences"]])
    ax.invert_yaxis()
    ax.set_xlim(-12, 62)
    ax.set_xlabel("Q4 accuracy minus low-tier accuracy (percentage points)")
    ax.set_title("Paired Q4-minus-low differences", fontsize=12)
    ax.grid(axis="y", visible=False)
    fig.text(
        0.5,
        0.015,
        "Whiskers are paired Newcombe method 10 95% intervals; n=100 per comparison.",
        ha="center",
        fontsize=8.5,
        color=palette["muted"],
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_pdf(fig, output_dir / "fig_delta.pdf")


def plot_within_model_ladders(
    data: dict[str, Any], output_dir: Path, palette: dict[str, Any]
) -> None:
    """Plot two separate bpw/score ladders, never an expert-count regression."""
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.85), sharey=True)
    panel_colors = [palette["orange"], palette["blue"]]
    label_positions = {
        "Mixtral-8x7B": {
            "IQ1 M": ((-5, -30), "right"),
            "IQ2 XXS": ((8, 11), "left"),
            "Q4 K M": ((0, 11), "center"),
        },
        "Qwen3-30B-A3B": {
            "IQ1 M": ((-5, 16), "right"),
            "IQ2 XXS": ((0, -34), "center"),
            "Q4 K M": ((0, 11), "center"),
        },
    }
    for index, (ax, panel, color) in enumerate(zip(axes, data["panels"], panel_colors)):
        tiers = sorted(panel["tiers"], key=lambda tier: tier["effective_bpw"])
        xs = [tier["effective_bpw"] for tier in tiers]
        ys = [tier["accuracy"] * 100 for tier in tiers]
        lower = [y - tier["wilson_95"][0] * 100 for y, tier in zip(ys, tiers)]
        upper = [tier["wilson_95"][1] * 100 - y for y, tier in zip(ys, tiers)]
        ax.plot(xs, ys, color=color, linewidth=1.5, zorder=3)
        ax.errorbar(
            xs,
            ys,
            yerr=[lower, upper],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=6,
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )
        for tier, x, y in zip(tiers, xs, ys):
            offset, alignment = label_positions[panel["model"]][tier["label"]]
            ax.annotate(
                f"{tier['label']}\n{x:.2f} bpw, {y:.0f}%",
                (x, y),
                textcoords="offset points",
                xytext=offset,
                ha=alignment,
                fontsize=8,
            )
        margin = max(0.18, (max(xs) - min(xs)) * 0.10)
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(0, 108)
        ax.set_xlabel("Effective file bits per weight")
        ax.set_title(panel["model"], fontsize=11)
        if index == 0:
            ax.set_ylabel("GSM8K accuracy (%)")
    fig.suptitle("Separate within-model quantization ladders", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Descriptive within-model paths only—not a cross-model or expert-count dose-response.",
        ha="center",
        fontsize=8.5,
        color=palette["muted"],
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    save_pdf(fig, output_dir / "fig_staircase.pdf")


def generate_figures(claims_path: Path, output_dir: Path, dark: bool = False) -> list[Path]:
    """Generate the three requested PDFs and return their paths."""
    data = extract_plot_data(load_claims(claims_path))
    palette = theme(dark)
    plot_within_model_scores(data, output_dir, palette)
    plot_paired_differences(data, output_dir, palette)
    plot_within_model_ladders(data, output_dir, palette)
    return [
        output_dir / "fig_matched.pdf",
        output_dir / "fig_delta.pdf",
        output_dir / "fig_staircase.pdf",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dark", action="store_true", help="use the dark-paper palette")
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override figures/ (or figures/dark with --dark), primarily for tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (DEFAULT_FIGURES / "dark" if args.dark else DEFAULT_FIGURES)
    outputs = generate_figures(args.claims, output_dir, args.dark)
    print("wrote " + ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
