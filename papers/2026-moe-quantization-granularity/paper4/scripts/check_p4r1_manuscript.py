#!/usr/bin/env python3
"""Fail closed until Paper 4 prose is fully converted from history to P4R1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parent.parent
SOURCES = (PAPER / "main.tex", PAPER / "explainer.tex", PAPER / "brief.tex")
CLAIMS_BUILDER = PAPER / "scripts" / "build_p4r1_claims.py"
PROFILE_MARKER = "% P4R1-PROTOCOL-PROFILE: corrective-exploratory-v1"
FORBIDDEN = {
    r"\b8192\b": "historical 8,192-token context",
    r"one stochastic completion|P4R1[^.]{0,80}stochastic": "P4R1 described as unseeded/stochastic",
    r"sampler seed (?:was )?not set|no (?:inference )?sampler seed": "historical missing-seed wording",
    r"auto-fit placements? (?:were|was) not preserved|placement[^.]{0,80}not auditable": "historical unpreserved-placement wording",
    r"\\PfourQwenMoeHealthy(?:Math|Code)": "historical Qwen supplementary result macro",
    r"\\Pfour(?:Flash|GptOss|Dense|MixtralTemp|MixtralPilot|MixtralHealthyMath)": "historical supplementary/forensic result macro",
}
REQUIRED_EACH = (
    r"P4R1",
    r"corrective",
    r"not independent",
)
SIGNED_OUTCOMES = {
    "mixtral_iq1_vs_q4": (
        "P4R1-MIXTRAL-IQ1M", "P4R1-MIXTRAL-Q4KM",
        r"\PfourMixtralIqOneSignedDirection", "mixtral", "iq1",
    ),
    "mixtral_iq2_vs_q4": (
        "P4R1-MIXTRAL-IQ2XXS", "P4R1-MIXTRAL-Q4KM",
        r"\PfourMixtralIqTwoSignedDirection", "mixtral", "iq2",
    ),
    "qwen3moe_iq1_vs_q4": (
        "P4R1-QWEN3MOE-IQ1M", "P4R1-QWEN3MOE-Q4KM",
        r"\PfourQwenMoeIqOneSignedDirection", "qwen", "iq1",
    ),
    "qwen3moe_iq2_vs_q4": (
        "P4R1-QWEN3MOE-IQ2XXS", "P4R1-QWEN3MOE-Q4KM",
        r"\PfourQwenMoeIqTwoSignedDirection", "qwen", "iq2",
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require_rendered_claims_projection(claims_path: Path, claims: dict[str, Any]) -> str:
    """Bind the TeX macro projection used by rendering to reviewed JSON."""
    tex_path = claims_path.with_suffix(".tex")
    if tex_path.is_symlink() or not tex_path.is_file():
        raise RuntimeError("missing or unsafe P4R1 claims TeX projection")
    spec = importlib.util.spec_from_file_location(
        "paper4_p4r1_claims_projection", CLAIMS_BUILDER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the P4R1 claims projection builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.render_p4r1_tex(claims)
    observed = tex_path.read_text(encoding="utf-8")
    if observed != expected:
        raise RuntimeError("P4R1 claims.tex is not the exact projection of claims.json")
    return sha256_file(tex_path)


def section(text: str, title_pattern: str) -> str:
    match = re.search(
        rf"\\section\*?\{{[^}}]*(?:{title_pattern})[^}}]*\}}(.*?)(?=\\section\*?\{{|\\end\{{document\}})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"missing manuscript section matching {title_pattern!r}")
    return match.group(1)


def require_provenance_disclosure(text: str, context: str) -> None:
    for pattern, label in (
        (r"P4R1", "P4R1"),
        (r"historical", "historical campaign"),
        (r"provenance|endpoint|model identity", "provenance/endpoint failure"),
    ):
        if re.search(pattern, text, re.IGNORECASE) is None:
            raise RuntimeError(f"{context} omits {label} disclosure")


def validate_signed_outcome_narrative(
    claims: dict[str, Any], main_text: str, explainer_text: str, brief_text: str,
) -> None:
    signature = claims.get("outcome_signature")
    if not isinstance(signature, dict):
        raise RuntimeError("claims have no P4R1 outcome signature")
    deltas = signature.get("q4_minus_low_delta_pp")
    branches = signature.get("signed_result_branch")
    cell_correct = signature.get("cell_correct")
    if (
        not isinstance(deltas, dict)
        or not isinstance(branches, dict)
        or not isinstance(cell_correct, dict)
        or set(deltas) != set(SIGNED_OUTCOMES)
        or set(branches) != set(SIGNED_OUTCOMES)
    ):
        raise RuntimeError("claims outcome signature lacks exact signed P4R1 branches")
    combined = "\n".join((main_text, explainer_text, brief_text))
    for comparison, (low_alias, q4_alias, macro, family, tier) in SIGNED_OUTCOMES.items():
        low_correct = cell_correct.get(low_alias)
        q4_correct = cell_correct.get(q4_alias)
        if (
            not isinstance(low_correct, int) or isinstance(low_correct, bool)
            or not isinstance(q4_correct, int) or isinstance(q4_correct, bool)
            or not 0 <= low_correct <= 100 or not 0 <= q4_correct <= 100
        ):
            raise RuntimeError(f"{comparison}: cell counts are malformed")
        expected_delta = float(q4_correct - low_correct)
        delta = deltas[comparison]
        if (
            not isinstance(delta, (int, float)) or isinstance(delta, bool)
            # Proportion arithmetic can leave harmless IEEE-754 residue (for
            # example 42.00000000000001 for an exact 42/100 difference).
            # The signed branch remains bound to exact integer cell counts.
            or not math.isclose(
                float(delta), expected_delta, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise RuntimeError(f"{comparison}: signed delta differs from exact cell counts")
        expected_branch = (
            "q4_higher" if expected_delta > 0
            else "low_tier_higher" if expected_delta < 0
            else "tied"
        )
        if branches[comparison] != expected_branch:
            raise RuntimeError(f"{comparison}: result branch reverses the signed delta")
        for source_name, source_text in (("main", main_text), ("explainer", explainer_text)):
            if macro not in source_text:
                raise RuntimeError(
                    f"{source_name}: missing signed result macro {macro} for {comparison}"
                )
        family_tier = rf"(?:{family}[^.\n]{{0,100}}{tier}|{tier}[^.\n]{{0,100}}{family})"
        if expected_branch == "low_tier_higher":
            stale = re.compile(
                family_tier
                + r"[^.\n]{0,220}(?:collapse|degrad|\b(?:loss|lose|loses|lost)\b|"
                  r"q4[^.\n]{0,80}(?:higher|preserv|beat|outperform))",
                re.IGNORECASE,
            )
            if stale.search(combined):
                raise RuntimeError(
                    f"{comparison}: stale Q4-preservation/low-tier-loss prose reverses the result"
                )
        elif expected_branch == "q4_higher":
            reversed_claim = re.compile(
                family_tier
                + r"[^.\n]{0,220}(?:low[^.\n]{0,80}(?:higher|beat|outperform)|"
                  r"q4[^.\n]{0,80}(?:lower|lost|degrad))",
                re.IGNORECASE,
            )
            if reversed_claim.search(combined):
                raise RuntimeError(
                    f"{comparison}: prose reverses the observed Q4-minus-low direction"
                )
    for family, family_keys in (
        ("mixtral", ("mixtral_iq1_vs_q4", "mixtral_iq2_vs_q4")),
        ("qwen", ("qwen3moe_iq1_vs_q4", "qwen3moe_iq2_vs_q4")),
    ):
        if any(branches[key] == "low_tier_higher" for key in family_keys) and re.search(
            rf"{family}[^.\n]{{0,240}}(?:collapse|degrad|\b(?:loss|lose|loses|lost)\b|"
            r"q4[^.\n]{0,80}(?:higher|preserv|beat|outperform))",
            combined,
            re.IGNORECASE,
        ):
            raise RuntimeError(
                f"{family}: family-level Q4-preservation/low-tier-loss prose is unsafe under a signed reversal"
            )
    if max(abs(float(value)) for value in deltas.values()) < 1.0 and re.search(
        r"collapse|sharp degradation|whereas|in contrast", combined, re.IGNORECASE
    ):
        raise RuntimeError("near-tied P4R1 outcomes retain a stale collapse/contrast narrative")


def check(
    claims_path: Path, expected_claims_sha256: str,
    narrative_review_path: Path, expected_narrative_review_sha256: str,
) -> dict[str, Any]:
    if claims_path.is_symlink() or not claims_path.is_file():
        raise RuntimeError("missing or unsafe P4R1 claims file")
    actual_claims_sha256 = sha256_file(claims_path)
    if actual_claims_sha256 != expected_claims_sha256:
        raise RuntimeError("P4R1 claims file differs from the explicitly reviewed SHA-256")
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims_tex_sha256 = require_rendered_claims_projection(claims_path, claims)
    validation = claims.get("p4r1_validation") if isinstance(claims, dict) else None
    if (
        claims.get("schema_version") != "paper4-p4r1-claims-v1"
        or not isinstance(claims.get("outcome_signature"), dict)
        or
        not isinstance(validation, dict)
        or validation.get("status") != "promoted-private-evidence"
        or not str(validation.get("independence", "")).startswith("not independent")
        or validation.get("context_tokens_all_cells") != 4096
    ):
        raise RuntimeError("claims file is not a promoted P4R1 corrective-validation artifact")
    if narrative_review_path.is_symlink() or not narrative_review_path.is_file():
        raise RuntimeError("missing or unsafe P4R1 narrative-review receipt")
    if sha256_file(narrative_review_path) != expected_narrative_review_sha256:
        raise RuntimeError("narrative-review receipt differs from its explicit SHA-256")
    review = json.loads(narrative_review_path.read_text(encoding="utf-8"))
    required_attestations = {
        "historical_provenance_failure_disclosed_in_abstract_methods_and_limitations": True,
        "corrective_exploratory_not_independent_status_disclosed": True,
        "qualitative_claims_reviewed_against_exact_outcome_signature": True,
        "explainer_and_brief_reviewed": True,
    }
    if (
        not isinstance(review, dict)
        or set(review)
        != {
            "schema", "status", "reviewer", "claims_sha256",
            "outcome_signature_sha256", "sources", "attestations",
            "reviewed_at_utc", "method",
        }
        or review.get("schema") != "paper4-p4r1-narrative-review-v1"
        or review.get("status") != "author-reviewed-after-p4r1-results"
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or review.get("claims_sha256") != actual_claims_sha256
        or review.get("outcome_signature_sha256")
        != canonical_sha256(claims["outcome_signature"])
        or review.get("attestations") != required_attestations
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00",
            str(review.get("reviewed_at_utc")),
        )
        is None
        or review.get("method") != "interactive-exact-challenge"
    ):
        raise RuntimeError("narrative-review receipt does not bind the exact claims/outcomes")
    review_sources = review.get("sources")
    if not isinstance(review_sources, list):
        raise RuntimeError("narrative-review source inventory is absent")
    review_by_path = {
        item.get("path"): item for item in review_sources if isinstance(item, dict)
    }
    if set(review_by_path) != {path.name for path in SOURCES} or len(review_by_path) != len(
        review_sources
    ):
        raise RuntimeError("narrative-review source inventory differs")
    checked = []
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        record = review_by_path[path.name]
        if (
            set(record) != {"path", "bytes", "sha256"}
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"{path.name}: source changed after narrative review")
        if PROFILE_MARKER not in text:
            raise RuntimeError(f"{path.name}: missing reviewed P4R1 protocol-profile marker")
        for pattern, description in FORBIDDEN.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                raise RuntimeError(f"{path.name}:{line}: {description} remains")
        for pattern in REQUIRED_EACH:
            if re.search(pattern, text, re.IGNORECASE) is None:
                raise RuntimeError(f"{path.name}: required P4R1 disclosure is absent: {pattern}")
        checked.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    main_text = (PAPER / "main.tex").read_text(encoding="utf-8")
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        main_text,
        re.IGNORECASE | re.DOTALL,
    )
    if abstract_match is None:
        raise RuntimeError("main.tex has no abstract")
    require_provenance_disclosure(abstract_match.group(1), "abstract")
    require_provenance_disclosure(section(main_text, "design|methods"), "methods")
    limitations = section(main_text, "limitations|threats")
    require_provenance_disclosure(limitations, "limitations")
    if re.search(r"not independent|outcomes? (?:were )?(?:already )?known", limitations, re.IGNORECASE) is None:
        raise RuntimeError("limitations omit the post-outcome/non-independent status")
    explainer_text = (PAPER / "explainer.tex").read_text(encoding="utf-8")
    require_provenance_disclosure(section(explainer_text, "Abstract walkthrough"), "explainer abstract")
    require_provenance_disclosure(section(explainer_text, "design|methods"), "explainer methods")
    require_provenance_disclosure(section(explainer_text, "limitations|threats"), "explainer limitations")

    brief_text = (PAPER / "brief.tex").read_text(encoding="utf-8")
    validate_signed_outcome_narrative(claims, main_text, explainer_text, brief_text)
    combined = "\n".join((main_text, explainer_text, brief_text))
    deltas = claims["outcome_signature"]["q4_minus_low_delta_pp"]
    combined_lower = combined.lower()
    if max(abs(float(deltas[key])) for key in ("mixtral_iq1_vs_q4", "mixtral_iq2_vs_q4")) < 5 and re.search(
        r"mixtral[^.]{0,160}(collapse|sharp|failed|degrad)", combined_lower
    ):
        raise RuntimeError("Mixtral outcomes no longer support the retained failure narrative")
    if max(abs(float(deltas[key])) for key in ("qwen3moe_iq1_vs_q4", "qwen3moe_iq2_vs_q4")) >= 5 and re.search(
        r"qwen[^.]{0,160}(stable|unchanged|preserv)", combined_lower
    ):
        raise RuntimeError("Qwen outcomes no longer support the retained stability narrative")
    for required in (
        r"4\{,\}200\{,\}000|4,200,000|4200000",
        r"same (?:per-item )?seed|shared across (?:all )?(?:six )?artifacts",
        r"offloaded[^\n]{0,60}layers|placement readback",
        r"shutdown receipt",
        r"MATH[^.]{0,120}outside P4R1|outside P4R1[^.]{0,120}MATH",
    ):
        if re.search(required, combined, re.IGNORECASE) is None:
            raise RuntimeError(f"manuscript set omits required P4R1 protocol disclosure: {required}")
    return {
        "status": "pass",
        "profile": "corrective-exploratory-v1",
        "claims_sha256": actual_claims_sha256,
        "claims_tex_sha256": claims_tex_sha256,
        "narrative_review_sha256": expected_narrative_review_sha256,
        "outcome_signature_sha256": canonical_sha256(claims["outcome_signature"]),
        "sources": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--claims-sha256", required=True)
    parser.add_argument("--narrative-review", required=True, type=Path)
    parser.add_argument("--narrative-review-sha256", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{64}", args.claims_sha256) is None:
        parser.error("--claims-sha256 must be 64 lowercase hexadecimal characters")
    if re.fullmatch(r"[0-9a-f]{64}", args.narrative_review_sha256) is None:
        parser.error("--narrative-review-sha256 must be 64 lowercase hexadecimal characters")
    print(
        json.dumps(
            check(
                args.claims, args.claims_sha256,
                args.narrative_review, args.narrative_review_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
