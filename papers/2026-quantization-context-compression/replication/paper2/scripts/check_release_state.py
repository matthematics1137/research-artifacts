#!/usr/bin/env python3
"""Fail-closed canonical-build gate for Paper 2.

`INVALIDATED.md` is permanent historical provenance. Its presence alone does
not block a future corrected release; canonical targets open only when a
hash-valid P2R1 promotion, rebuilt claims, and an explicit result-specific
narrative audit all agree. A forensic build remains available while blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT = PAPER_DIR.parent
R1 = ROOT / "testsuite" / "paper2" / "r1"
PROMOTION = R1 / "private" / "promotion.json"
CLAIMS_JSON = PAPER_DIR / "derived" / "claims.json"
CLAIMS_TEX = PAPER_DIR / "derived" / "claims.tex"
NARRATIVE_AUDIT = PAPER_DIR / "derived" / "postrun_narrative_audit.json"
PUBLIC_ALLOWLIST = PAPER_DIR / "derived" / "public_release_allowlist.json"
FIGURE_PATHS = (
    PAPER_DIR / "figures" / "fig_grid.pdf",
    PAPER_DIR / "figures" / "fig_ratio.pdf",
    PAPER_DIR / "figures" / "fig_accmin.pdf",
    PAPER_DIR / "figures" / "fig_survival.pdf",
    PAPER_DIR / "figures" / "dark" / "fig_grid.pdf",
    PAPER_DIR / "figures" / "dark" / "fig_ratio.pdf",
    PAPER_DIR / "figures" / "dark" / "fig_accmin.pdf",
    PAPER_DIR / "figures" / "dark" / "fig_survival.pdf",
)
REVIEW_PDF_PATHS = (
    PAPER_DIR / "review" / "light" / "main.pdf",
    PAPER_DIR / "review" / "light" / "explainer.pdf",
    PAPER_DIR / "review" / "dark" / "main_dark_review.pdf",
    PAPER_DIR / "review" / "dark" / "explainer_dark_review.pdf",
)
ALLOWED_PUBLIC_SUFFIXES = {
    ".bib", ".cff", ".cls", ".csv", ".json", ".lock", ".md", ".pdf",
    ".py", ".sh", ".sty", ".svg", ".tex", ".toml", ".txt", ".yaml",
    ".yml",
}
ALLOWED_EXTENSIONLESS_BASENAMES = {"LICENSE", "Makefile", "paper-Makefile"}


# These are distinctive facts from the invalid P2G draft.  Merely removing the
# red invalidation banner must never turn that draft into a canonical P2R1
# manuscript.  A future historical discussion may describe P2G, but it must not
# repeat these stale design/results assertions as part of the promoted paper.
STALE_P2G_PATTERNS = {
    "unseeded question generation": r"\bunseeded\b",
    "old public 105/15/0 stratum count": (
        r"\b105\s+short\b|\b15\s+long(?:[/ -]+no[- ]?code)?\b|"
        r"\b(?:zero|no)\s+long[/ -]?code\b"
    ),
    "old private 103/14 length count": r"\b103\s+short\b.{0,120}\b14\s+long\b",
    "stale prepared-release claim": (
        r"\bprepared\s+(?:public\s+)?(?:bundle|artifact|package|release)\b"
    ),
    "old CR1 finite-cluster formula": r"g/\(g-1\)",
    "old fixed t-60 reference": r"\bt_?60\b",
    "old fixed 61-window clustering": (
        r"\b61\s+source[- ](?:window|conversation)(?:s|\s+clusters)?\b|"
        r"\b61\b.{0,80}\bper corpus\b"
    ),
    "tier-randomization inference": r"\b(?:zero-effect|specified)\s+randomization model\b",
    "sign flip described as primary": (
        r"(?:primary\s+(?:test|analysis|calculation)\s+(?:is\s+)?(?:the\s+)?"
        r"sign[- ]?flip|sign[- ]?flip\s+(?:test|analysis|calculation)?\s*"
        r"(?:is|was|as)\s+(?:the\s+)?primary)"
    ),
}


def _semantic_projection(text: str) -> str:
    """Flatten harmless TeX punctuation before exact protocol-fact checks."""
    value = text.casefold()
    value = re.sub(r"[{}$\\]", "", value)
    value = re.sub(r"-{2,}", "-", value)
    value = value.replace(",", "")
    return re.sub(r"\s+", " ", value)


def validate_p2r1_protocol_narrative(
    claims: dict, main_text: str, explainer_text: str
) -> None:
    """Require the recovery design and its measurement limits in both sources.

    This is deliberately a semantic tripwire, not a substitute for human prose
    review.  It prevents the invalid P2G body from passing after a banner-only
    edit and binds the future paper/explainer to the prospective P2R1 protocol.
    """
    headline = claims.get("headline", {})
    hardware = claims.get("hardware", {})
    if (
        headline.get("n_cells") != 24
        or headline.get("n_interaction_tests") != 12
        or hardware.get("gpu_name") != "NVIDIA GeForce RTX 4080 Laptop GPU"
        or hardware.get("gpu_memory_total_mib") != 12282.0
        or hardware.get("gpu_current_power_limit_w") != 80.0
    ):
        raise SystemExit(
            "STOP: claims lack the P2R1 grid or promoted host facts required "
            "for narrative review"
        )

    required_patterns = {
        "P2R1 corrective chronology": (
            r"\bp2r1\b", r"corrective exploratory rerun",
            r"not an independent confirmation",
        ),
        "official corpus counts": (
            r"(?:\b117\b.{0,100}\btsi\b|\btsi\b.{0,100}\b117\b)",
            r"(?:\b120\b.{0,100}\bwildchat\b|\bwildchat\b.{0,100}\b120\b)",
        ),
        "stratified quotas": (
            r"stratif", r"37\D{0,20}37\D{0,20}6\D{0,20}37",
            r"30\D{0,20}30\D{0,20}30\D{0,20}30",
        ),
        "predeclared retry schedule": (
            r"predeclared", r"retry seed", r"1000000", r"2000000",
        ),
        "timing estimand": (
            r"end-to-end", r"chat-completion",
            r"(?:preprocessing.{0,80}excluded|exclud\w*.{0,80}preprocessing)",
            r"(?:no|without|not).{0,80}(?:prefill|decode).{0,80}(?:split|timing)",
        ),
        "rights and privacy boundary": (
            r"wildchat", r"odc-by", r"raw.{0,100}private",
        ),
        "deployment interpretation": (
            r"artifact", r"engine", r"placement",
            r"(?:not|no).{0,120}(?:weight[- ]bit|weight[- ]precision).{0,120}causal",
        ),
        "character rather than memory-saving estimand": (
            r"character", r"kv-cache", r"(?:not|no|unmeasured).{0,100}(?:vram|kv-cache)",
        ),
        "familywise inference": (
            r"\bcr2\b", r"satterthwaite", r"holm", r"\b12\b",
            r"(?:primary.{0,140}holm|holm.{0,140}(?:primary|result branch))",
            r"pointwise", r"model[- ]assisted|approximate",
            r"sign[- ]?flip.{0,140}secondary",
            r"(?:conditional.{0,140}sign|sign[- ]?flip.{0,140}conditional)",
            r"source[- ]conversation", r"independent.{0,80}(?:source|cluster)",
        ),
        "conditional inference scope": (
            r"(?:one|single|frozen).{0,100}(?:q4[- ]generated|generated qa|question set)",
            r"(?:one|single).{0,80}seeded decode",
            r"(?:does not|not|without).{0,120}(?:alternate|other).{0,80}question[- ]generation seed",
            r"(?:does not|not|without).{0,120}run[- ]to[- ]run variation",
        ),
        "reproducibility chain": (
            r"receipt", r"hash", r"(?:rerun|reproduc)",
        ),
    }

    for source_name, original in (("main.tex", main_text), ("explainer.tex", explainer_text)):
        projected = _semantic_projection(original)
        for description, pattern in STALE_P2G_PATTERNS.items():
            if re.search(pattern, projected, flags=re.S):
                raise SystemExit(
                    f"STOP: {source_name} retains {description} from the invalid P2G draft"
                )
        for description, patterns in required_patterns.items():
            missing = [pattern for pattern in patterns if not re.search(pattern, projected, flags=re.S)]
            if missing:
                raise SystemExit(
                    f"STOP: {source_name} lacks reviewed P2R1 {description} language"
                )
        for description, count, phrase in (
            (
                "evaluation request-error disclosure",
                headline.get("evaluation_request_errors"),
                r"evaluation request errors?",
            ),
            (
                "question-generation request-error disclosure",
                headline.get("question_generation_request_errors"),
                r"question[- ]generation request errors?",
            ),
        ):
            if type(count) is not int or count < 0:
                raise SystemExit(f"STOP: claims lack {description} count")
            exact = rf"(?:\b{count}\b.{{0,100}}{phrase}|{phrase}.{{0,100}}\b{count}\b)"
            if not re.search(exact, projected, flags=re.S):
                raise SystemExit(f"STOP: {source_name} lacks exact {description}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"STOP: required canonical-release evidence is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"STOP: expected a JSON object: {path}")
    return value


def verify_promoted_supporting_evidence(promotion: dict) -> None:
    """Keep private raw evidence immutable through the canonical PDF build."""
    supporting = promotion.get("supporting_evidence_sha256")
    if not isinstance(supporting, dict) or not supporting:
        raise SystemExit("STOP: P2R1 promotion lacks retained supporting evidence")
    for raw_path, expected_hash in supporting.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise SystemExit(
                f"STOP: promoted private evidence changed before build: {path}"
            )


def require_aggregate_only_public_claims(claims: dict) -> None:
    """Reject per-item or deterministically linkable WildChat material."""
    expected_top_level = {
        "schema", "study_status", "promotion_sha256", "attempt_id",
        "promotion_completed_at", "analysis_harness_sha256",
        "analysis_implementation", "analysis_implementation_sha256",
        "artifacts", "corpora", "grid", "interaction", "survival",
        "headline", "protocol", "hardware", "software",
    }
    if set(claims) != expected_top_level:
        raise SystemExit("STOP: public claims top-level schema changed without audit")
    forbidden_keys = {
        "qid", "window_id", "source_cluster_id", "conversation_id",
        "upstream_conversation_hash", "source_row_index",
        "question", "answer", "response", "raw_response",
    }
    forbidden_collection_names = re.compile(
        r"(?:^|_)(?:ids?|rows?|records?|observations?|samples?|per_item|item_records?)(?:$|_)"
    )
    allowed_dict_list_paths = {
        ("artifacts", "Ex", "upstream_weight_shards"):
            {"filename", "bytes", "sha256"},
    }

    def walk(value: object, trail: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if (
                    key in forbidden_keys
                    or (
                        forbidden_collection_names.search(key)
                        and not (trail == () and key == "attempt_id")
                        # a scalar SHA-256 of a deployment's artifact record, not a row collection
                        and not (
                            len(trail) == 2 and trail[0] == "artifacts"
                            and key == "artifact_record_sha256"
                            and isinstance(child, str)
                            and re.fullmatch(r"[0-9a-f]{64}", child) is not None
                        )
                    )
                    or re.fullmatch(r"(?:wc\d{4}|cwc\d{5})", key)
                ):
                    raise SystemExit(
                        "STOP: public claims contain per-item/private field "
                        + ".".join((*trail, key))
                    )
                walk(child, (*trail, key))
        elif isinstance(value, list):
            if any(isinstance(child, dict) for child in value):
                allowed_keys = allowed_dict_list_paths.get(trail)
                if (
                    len(trail) == 4
                    and trail[0] == "interaction"
                    and trail[3] == "cluster_n_items_raw_z_sum_histogram"
                ):
                    allowed_keys = {
                        "cluster_n_items", "raw_z_sum", "multiplicity"
                    }
                if allowed_keys is None or any(
                    not isinstance(child, dict) or set(child) != allowed_keys
                    for child in value
                ):
                    raise SystemExit(
                        "STOP: public claims contain an unaudited row-like collection at "
                        + ".".join(trail)
                    )
            if any(isinstance(child, str) for child in value):
                raise SystemExit(
                    "STOP: public claims contain an unaudited string/ID collection at "
                    + ".".join(trail)
                )
            for index, child in enumerate(value):
                walk(child, (*trail, str(index)))
        elif isinstance(value, str) and re.search(r"\b(?:wc\d{4}|cwc\d{5})\b", value):
            raise SystemExit("STOP: public claims expose a linkable WildChat local ID")

    walk(claims)


def validate_public_allowlist(promotion: dict) -> None:
    value = load_object(PUBLIC_ALLOWLIST)
    files = value.get("files")
    release_roots = value.get("release_roots")
    if (
        value.get("schema") != "paper2-r1-public-release-allowlist-v2"
        or value.get("promotion_sha256") != sha256_file(PROMOTION)
        or not isinstance(files, list)
        or not files
        or not isinstance(release_roots, list)
        or not release_roots
    ):
        raise SystemExit("STOP: public release allowlist is missing, stale, or malformed")
    forbidden_roots = (
        (R1 / "private").resolve(),
        (R1 / "private_inputs").resolve(),
        (ROOT / "testsuite" / "evals" / "results" / "paper2-r1").resolve(),
        (ROOT / ".publication" / "state").resolve(),
    )
    listed_paths: set[Path] = set()
    for entry in files:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry["path"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"]))
        ):
            raise SystemExit("STOP: public allowlist entries require path+SHA-256")
        relative = entry["path"]
        if Path(relative).is_absolute():
            raise SystemExit("STOP: public allowlist paths must be relative text")
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise SystemExit("STOP: public allowlist path escapes repository") from exc
        if not path.is_file() or any(path == root or root in path.parents for root in forbidden_roots):
            raise SystemExit(f"STOP: public allowlist contains private/missing path: {path}")
        if path in listed_paths or sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"STOP: public allowlist has duplicate/stale hash: {path}")
        listed_paths.add(path)

    required_public_inputs = {
        CLAIMS_JSON.resolve(), CLAIMS_TEX.resolve(),
        (PAPER_DIR / "main.tex").resolve(),
        (PAPER_DIR / "explainer.tex").resolve(),
        (PAPER_DIR / "references.bib").resolve(),
        (PAPER_DIR / "VERIFICATION.md").resolve(),
        (PAPER_DIR / "BIBLIOGRAPHY_AUDIT.md").resolve(),
        *(path.resolve() for path in FIGURE_PATHS),
        *(path.resolve() for path in REVIEW_PDF_PATHS),
    }
    if not required_public_inputs <= listed_paths:
        missing = sorted(str(path) for path in required_public_inputs - listed_paths)
        raise SystemExit(f"STOP: public allowlist omits canonical release input: {missing}")

    # The staged public package lives under one or more dedicated
    # publication/.../public roots. Every regular file there must be listed;
    # unlisted extras therefore cannot evade the privacy scan.
    inventoried: set[Path] = set()
    seen_roots: set[Path] = set()
    for relative in release_roots:
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise SystemExit("STOP: public release roots must be relative text")
        release_root = (ROOT / relative).resolve()
        try:
            release_root.relative_to((ROOT / "publication" / "papers").resolve())
        except ValueError as exc:
            raise SystemExit("STOP: public release root is outside publication/papers") from exc
        if release_root.name != "public" or not release_root.is_dir() or release_root in seen_roots:
            raise SystemExit("STOP: public release roots must be distinct existing public directories")
        seen_roots.add(release_root)
        for path in release_root.rglob("*"):
            if path.is_symlink():
                raise SystemExit(f"STOP: symlink forbidden in staged public tree: {path}")
            if path.is_file():
                if (
                    path.suffix.casefold() not in ALLOWED_PUBLIC_SUFFIXES
                    and path.name not in ALLOWED_EXTENSIONLESS_BASENAMES
                ):
                    raise SystemExit(f"STOP: unaudited file type in public tree: {path}")
                inventoried.add(path.resolve())
    if inventoried != {path for path in listed_paths if any(
        path == root or root in path.parents for root in seen_roots
    )}:
        raise SystemExit("STOP: staged public release tree differs from exact allowlist inventory")


def validate_result_narrative(
    claims: dict, narrative: dict, main_text: str, explainer_text: str
) -> None:
    """Bind qualitative outcomes to explicit, reviewed source assertions."""
    main_lower = main_text.casefold()
    if (
        "corrective exploratory rerun" not in main_lower
        or "not an independent confirmation" not in main_lower
        or "ai assistance" not in main_lower
    ):
        raise SystemExit(
            "STOP: canonical manuscript lacks the corrective/non-independent/"
            "AI-assistance disclosures required by the P2R1 protocol"
        )
    result_branch = claims.get("headline", {}).get("interaction_result_branch")
    qualitative = claims.get("headline", {}).get("qualitative_assertions")
    if (
        result_branch not in {"interaction-detected", "no-interaction-detected"}
        or not isinstance(qualitative, dict)
        or qualitative.get("interaction_primary_branch") != result_branch
    ):
        raise SystemExit("STOP: claims lack a coherent qualitative result branch")
    branch_marker = f"P2R1_RESULT_BRANCH: {result_branch}"
    assertions = narrative.get("result_specific_source_assertions")
    def matches_branch(value: str) -> bool:
        words = set(re.findall(r"[a-z]+", value.casefold()))
        has_result_terms = "interaction" in words and any(
            word.startswith("detect") for word in words
        )
        negated = bool({"no", "not", "none"} & words)
        return has_result_terms and (
            negated if result_branch == "no-interaction-detected" else not negated
        )

    assertions_valid = (
        isinstance(assertions, dict)
        and set(assertions) == {"title", "abstract", "conclusion", "explainer"}
        and all(isinstance(value, str) and len(value.strip()) >= 15
                for value in assertions.values())
        and assertions["title"] in main_text
        and assertions["abstract"] in main_text
        and assertions["conclusion"] in main_text
        and assertions["explainer"] in explainer_text
        and all(matches_branch(value) for value in assertions.values())
    )
    if (
        narrative.get("interaction_result_branch") != result_branch
        or narrative.get("qualitative_assertions") != qualitative
        or branch_marker not in main_text
        or branch_marker not in explainer_text
        or not assertions_valid
    ):
        raise SystemExit(
            "STOP: post-result narrative audit disagrees with the observed "
            "qualitative result branch or exact source assertions"
        )


def candidate_source_gate() -> None:
    """Open only unmistakable review-candidate builds, never canonical output."""
    canonical_gate(pre_figure=True)
    claims = load_object(CLAIMS_JSON)
    main_path = PAPER_DIR / "main.tex"
    explainer_path = PAPER_DIR / "explainer.tex"
    references = PAPER_DIR / "references.bib"
    for path in (CLAIMS_TEX, main_path, explainer_path, references):
        if not path.is_file():
            raise SystemExit(f"STOP: review-candidate source is missing: {path}")
    main_text = main_path.read_text()
    explainer_text = explainer_path.read_text()
    for path, text in ((main_path, main_text), (explainer_path, explainer_text)):
        if "INVALIDATED DRAFT" in text or "DO NOT CITE, DISTRIBUTE, OR PUBLISH" in text:
            raise SystemExit(f"STOP: {path.name} is still the invalid P2G draft")
    validate_p2r1_protocol_narrative(claims, main_text, explainer_text)
    branch = claims.get("headline", {}).get("interaction_result_branch")
    marker = f"P2R1_RESULT_BRANCH: {branch}"
    if branch not in {"interaction-detected", "no-interaction-detected"} or any(
        marker not in text for text in (main_text, explainer_text)
    ):
        raise SystemExit("STOP: review sources do not declare the observed result branch")
    print("P2R1 review-candidate source gate passed")


def canonical_gate(*, pre_figure: bool = False) -> None:
    promotion = load_object(PROMOTION)
    if (
        promotion.get("schema") != "paper2-r1-private-promotion-v1"
        or promotion.get("status") != "validated-for-claims-rebuild-not-publication"
        or promotion.get("legacy_p2g_accepted") is not False
        or promotion.get("official_runs") != 6
        or promotion.get("official_condition_cells") != 24
        or len(promotion.get("runs", [])) != 6
    ):
        raise SystemExit("STOP: P2R1 promotion receipt is malformed or incomplete")
    attempt_id = promotion.get("attempt_id")
    protocol = load_object(PAPER_DIR / "R1_PROTOCOL.json")
    attempt_receipt = Path(promotion.get("campaign", "")) / "attempt.json"
    expected_result_root = (
        ROOT / "testsuite" / "evals" / "results" / "paper2-r1" / str(attempt_id)
    ).resolve()
    expected_question_root = (
        R1 / "private" / "attempts" / str(attempt_id) / "questions"
    ).resolve()
    attempt_host_binding = expected_result_root / "HOST_BINDING.json"
    if (
        not isinstance(attempt_id, str)
        or attempt_id != protocol.get("official_attempt_id")
        or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{5,79}", attempt_id)
        or promotion.get("result_root") != str(expected_result_root)
        or promotion.get("question_root") != str(expected_question_root)
        or (expected_result_root / "ATTEMPT_INVALIDATED.json").exists()
        or not attempt_receipt.is_file()
        or promotion.get("attempt_namespace_sha256") != sha256_file(attempt_receipt)
        or not attempt_host_binding.is_file()
        or promotion.get("attempt_host_binding_sha256")
        != sha256_file(attempt_host_binding)
    ):
        raise SystemExit("STOP: P2R1 promotion has a stale or invalid attempt namespace")
    verify_promoted_supporting_evidence(promotion)

    claims = load_object(CLAIMS_JSON)
    if (
        claims.get("schema") != "paper2-r1-claims-v3"
        or claims.get("promotion_sha256") != sha256_file(PROMOTION)
        or claims.get("attempt_id") != promotion.get("attempt_id")
        or claims.get("analysis_harness_sha256")
        != promotion.get("analysis_harness_sha256")
        or claims.get("analysis_implementation") != "paper4/scripts/verify_claims.py"
        or claims.get("analysis_implementation_sha256")
        != sha256_file(ROOT / "paper4" / "scripts" / "verify_claims.py")
        or "not independent confirmation" not in claims.get("study_status", "")
    ):
        raise SystemExit("STOP: derived claims are not bound to the validated P2R1 promotion")
    require_aggregate_only_public_claims(claims)

    if pre_figure:
        print("P2R1 pre-figure evidence/claims gate passed")
        return
    validate_public_allowlist(promotion)

    main_tex = PAPER_DIR / "main.tex"
    explainer_tex = PAPER_DIR / "explainer.tex"
    references = PAPER_DIR / "references.bib"
    for path in (CLAIMS_TEX, main_tex, explainer_tex, references):
        if not path.is_file():
            raise SystemExit(f"STOP: canonical source is missing: {path}")
    for path in (main_tex, explainer_tex):
        text = path.read_text()
        if "INVALIDATED DRAFT" in text or "DO NOT CITE, DISTRIBUTE, OR PUBLISH" in text:
            raise SystemExit(
                f"STOP: {path.name} is still the visibly invalidated P2G-era draft"
            )
    public_text_paths = (
        CLAIMS_JSON, CLAIMS_TEX, main_tex, explainer_tex, references,
        PAPER_DIR / "VERIFICATION.md", PAPER_DIR / "BIBLIOGRAPHY_AUDIT.md",
    )
    for path in public_text_paths:
        text = path.read_text()
        if re.search(r"/home/[^/\s]+", text):
            raise SystemExit(f"STOP: public Paper 2 source exposes a home-directory path: {path}")
    main_text = main_tex.read_text()
    explainer_text = explainer_tex.read_text()

    narrative = load_object(NARRATIVE_AUDIT)
    for path in FIGURE_PATHS:
        if not path.is_file():
            raise SystemExit(f"STOP: canonical quantitative figure is missing: {path}")
    figure_hashes = {
        str(path.relative_to(PAPER_DIR)): sha256_file(path)
        for path in FIGURE_PATHS
    }
    review_pdf_hashes = {
        str(path.relative_to(PAPER_DIR)): sha256_file(path)
        for path in REVIEW_PDF_PATHS
        if path.is_file()
    }
    if len(review_pdf_hashes) != len(REVIEW_PDF_PATHS):
        raise SystemExit("STOP: all four reviewed light/dark PDFs are required")
    expected_hashes = {
        "promotion_sha256": sha256_file(PROMOTION),
        "claims_json_sha256": sha256_file(CLAIMS_JSON),
        "claims_tex_sha256": sha256_file(CLAIMS_TEX),
        "main_tex_sha256": sha256_file(main_tex),
        "explainer_tex_sha256": sha256_file(explainer_tex),
        "references_bib_sha256": sha256_file(references),
        "public_allowlist_sha256": sha256_file(PUBLIC_ALLOWLIST),
    }
    validate_result_narrative(claims, narrative, main_text, explainer_text)
    validate_p2r1_protocol_narrative(claims, main_text, explainer_text)
    if (
        narrative.get("schema") != "paper2-r1-postrun-narrative-audit-v1"
        or narrative.get("status") != "approved-after-result-specific-review"
        or narrative.get("title_abstract_conclusion_reviewed") is not True
        or narrative.get("explainer_reviewed") is not True
        or narrative.get("ai_assistance_disclosure_verified") is not True
        or narrative.get("p2r1_protocol_facts_reviewed") is not True
        or narrative.get("rights_and_privacy_reviewed") is not True
        or narrative.get("timing_boundary_reviewed") is not True
        or narrative.get("multiplicity_reviewed") is not True
        or narrative.get("reproducibility_chain_reviewed") is not True
        or narrative.get("host_provenance_reviewed") is not True
        or narrative.get("quantitative_figures_reviewed") is not True
        or narrative.get("figure_sha256") != figure_hashes
        or narrative.get("review_pdf_sha256") != review_pdf_hashes
        or any(narrative.get(key) != value for key, value in expected_hashes.items())
    ):
        raise SystemExit(
            "STOP: post-result narrative audit is absent, stale, or disagrees "
            "with the observed qualitative result branch"
        )

    print(
        "P2R1 canonical build gate passed; INVALIDATED.md remains as historical provenance"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forensic-invalidated-build",
        action="store_true",
        help="permit a visibly invalidated PDF build for audit only",
    )
    parser.add_argument(
        "--pre-figure",
        action="store_true",
        help="verify promoted evidence and aggregate claims before mutating figures",
    )
    parser.add_argument(
        "--candidate-sources",
        action="store_true",
        help="validate rewritten sources before building noncanonical review PDFs",
    )
    args = parser.parse_args()
    if args.forensic_invalidated_build:
        print("FORENSIC ONLY: building a visibly invalidated Paper 2 draft")
        return 0
    if args.candidate_sources:
        candidate_source_gate()
        return 0
    canonical_gate(pre_figure=args.pre_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
