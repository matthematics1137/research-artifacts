"""Allowlisted, unchanged presentation definitions; no private input loading."""
# Full source SHA256: 079ee98e6a0b2292c033c004093c2b4c4df00940f066559a7260ab8fb332fb44
import math
EVIDENCE_METRICS = (
    ("evidence_turns_kept", "Evidence turns retained (%)", ("T", "M", "Mbm25")),
    ("evidence_sessions_summarized", "Evidence sessions summarized (%)", ("S",)),
    ("evidence_word_recall", "Evidence-word recall (%)", ("C", "Cw")),
)


BUDGET_TAGS = {"1024": "One", "4096": "Four", "none": "No"}


ARM_TEX = {"T": "T", "S": "S", "C": "C", "M": "M", "Cw": "Cw", "Mbm25": "Mb", "O": "O", "E": "E"}


def secondary_tag(name):
    """Keep both arm contrast AND budget in the TeX command's identity."""
    try:
        contrast, budget = name.rsplit("@", 1)
        left, right = contrast.split("_vs_")
        # TeX command names cannot contain digits. Use the same explicit arm
        # mapping as the cell macros, including the declared BM25 ablation.
        return f"{ARM_TEX[left]}v{ARM_TEX[right]}At{BUDGET_TAGS[budget.lower()]}"
    except (AttributeError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid contrast: {name}") from exc


def evidence_percent(cell, metric):
    value = (cell.get("diag_means") or {}).get(metric)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid evidence fraction for {metric}: {value!r}")
    return 100 * value


def first_measured(mapping, *keys):
    """Zero is a measurement; None is not. Never substitute a later value for zero."""
    return next((mapping[key] for key in keys if mapping.get(key) is not None), None)

