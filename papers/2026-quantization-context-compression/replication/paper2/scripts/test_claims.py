#!/usr/bin/env python3
"""Regression and privacy checks for Paper 2's derived quantitative record.

Run via `make audit-check`. Two jobs:
1. The derived outputs (claims.json / claims.tex) match a fresh rebuild from
   the frozen evidence, and the headline invariants the manuscript's claims
   rest on actually hold in the data.
2. PRIVACY: nothing from the private corpus (window text, question text, gold
   answers) leaks into any released file (derived outputs, main.tex, figure
   scripts). The figures are additionally constrained by construction: their
   script must read only derived/claims.json.
"""

from __future__ import annotations

import copy
import json
import hashlib
import importlib.util
import base64
import os
import re
import random
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parent
R1_PRIVATE = REPO_ROOT / "testsuite" / "paper2" / "r1" / "private_inputs"
PROMOTION = REPO_ROOT / "testsuite" / "paper2" / "r1" / "private" / "promotion.json"
CLAIMS_JSON = PAPER_DIR / "derived" / "claims.json"
CLAIMS_TEX = PAPER_DIR / "derived" / "claims.tex"
PUBLIC_ALLOWLIST = PAPER_DIR / "derived" / "public_release_allowlist.json"
NARRATIVE_AUDIT = PAPER_DIR / "derived" / "postrun_narrative_audit.json"
SCANNABLE_TEXT_SUFFIXES = {
    ".bib", ".cff", ".cls", ".csv", ".json", ".lock", ".md", ".py",
    ".sh", ".sty", ".svg", ".tex", ".toml", ".txt", ".yaml", ".yml",
}
SCANNABLE_EXTENSIONLESS = {"LICENSE", "Makefile", "paper-Makefile"}


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def distinctive_ngrams(value: str, width: int = 48) -> list[str]:
    """Every overlapping normalized n-gram; no offset or short-leak gaps."""
    text = normalized_text(value)
    if len(text) < width:
        return []
    return [text[start:start + width] for start in range(len(text) - width + 1)]


def public_text_views() -> dict[Path, str]:
    """Read the release allowlist when present, else claims-stage sources."""
    if PUBLIC_ALLOWLIST.is_file():
        if not PROMOTION.is_file():
            raise AssertionError("public release allowlist exists without a promotion")
        allowlist = json.loads(PUBLIC_ALLOWLIST.read_text())
        if (
            allowlist.get("schema") != "paper2-r1-public-release-allowlist-v2"
            or allowlist.get("promotion_sha256") != hashlib.sha256(
                PROMOTION.read_bytes()
            ).hexdigest()
            or not isinstance(allowlist.get("files"), list)
            or not allowlist["files"]
        ):
            raise AssertionError("public release allowlist is malformed or stale")
        # Reuse the release gate's independent path/hash/exact-tree inventory
        # validation, then scan the bytes of every listed textual/PDF file.
        release_gate.validate_public_allowlist(json.loads(PROMOTION.read_text()))
        paths = []
        for entry in allowlist["files"]:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "sha256"}
            ):
                raise AssertionError("public allowlist entry lacks path+SHA-256")
            relative = entry["path"]
            path = (REPO_ROOT / relative).resolve()
            try:
                path.relative_to(REPO_ROOT.resolve())
            except ValueError as exc:
                raise AssertionError("public allowlist path escapes repository") from exc
            if not path.is_file():
                raise AssertionError(f"public allowlist file is missing: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise AssertionError(f"public allowlist file hash changed: {path}")
            paths.append(path)
    else:
        # The claims stage owns only the generated JSON/TeX pair. Manuscripts,
        # references, PDFs, and staged release files enter the strict scan once
        # the exact allowlist exists. Keeping those stages separate also avoids
        # treating ordinary phrases in a bibliography as leaked corpus text.
        paths = [path for path in (CLAIMS_JSON, CLAIMS_TEX) if path.is_file()]
    views: dict[Path, str] = {}
    for path in sorted(set(paths)):
        if path.suffix == ".pdf":
            result = subprocess.run(
                ["pdftotext", str(path), "-"], capture_output=True, text=True,
                check=True,
            )
            views[path] = normalized_text(result.stdout)
        elif (
            path.suffix in SCANNABLE_TEXT_SUFFIXES
            or path.name in SCANNABLE_EXTENSIONLESS
        ):
            views[path] = normalized_text(path.read_text(errors="replace"))
    return views

_gate_spec = importlib.util.spec_from_file_location(
    "paper2_release_gate", PAPER_DIR / "scripts" / "check_release_state.py"
)
release_gate = importlib.util.module_from_spec(_gate_spec)
_gate_spec.loader.exec_module(release_gate)

_claims_spec = importlib.util.spec_from_file_location(
    "paper2_claim_builder", PAPER_DIR / "scripts" / "build_claims.py"
)
claim_builder = importlib.util.module_from_spec(_claims_spec)
_claims_spec.loader.exec_module(claim_builder)

_allow_spec = importlib.util.spec_from_file_location(
    "paper2_allowlist_builder", PAPER_DIR / "scripts" / "build_public_allowlist.py"
)
allowlist_builder = importlib.util.module_from_spec(_allow_spec)
_allow_spec.loader.exec_module(allowlist_builder)


class FamilywiseRuleTests(unittest.TestCase):
    def test_student_t_reference_probability(self) -> None:
        self.assertAlmostEqual(claim_builder.student_t_two_sided(0.0, 20), 1.0)
        # The conventional 0.975 critical value for 20 df is about 2.086.
        self.assertAlmostEqual(
            claim_builder.student_t_two_sided(2.085963, 20), 0.05, places=4
        )
        references = {
            2.6033: 3.475186228426227,
            5.0: 2.570581835636314,
            20.0: 2.0859634472658364,
            60.0: 2.00029782201426,
        }
        for df, expected in references.items():
            with self.subTest(df=df):
                self.assertAlmostEqual(
                    claim_builder.t_critical_975(df), expected, places=8
                )
                self.assertAlmostEqual(
                    claim_builder.student_t_two_sided(
                        claim_builder.t_critical_975(df), df
                    ),
                    0.05,
                    places=10,
                )

    def test_cr2_satterthwaite_fixed_cluster_sizes(self) -> None:
        self.assertAlmostEqual(
            claim_builder.cr2_satterthwaite_df([2] * 56 + [1] * 5),
            58.77738950065, places=9,
        )
        self.assertAlmostEqual(
            claim_builder.cr2_satterthwaite_df([2] * 59 + [1] * 2),
            59.50423758990, places=9,
        )
        self.assertAlmostEqual(
            claim_builder.cr2_satterthwaite_df([2] * 10), 9.0, places=10
        )

    @staticmethod
    def _matrix_reference(values: list[float], clusters: list[str]) -> tuple[float, float, float]:
        """Direct Bell--McCaffrey score/P-matrix reference for an intercept."""
        n = len(values)
        mean = sum(values) / n
        groups: dict[str, list[float]] = {}
        for value, cluster in zip(values, clusters):
            groups.setdefault(cluster, []).append(value)
        sizes = [len(group) for group in groups.values()]
        # For X=1, A_g=(I-H_gg)^(-1/2). Its score in the all-ones
        # direction is S_g/sqrt(1-h_g); orthogonal components sum to zero.
        adjusted_scores = []
        for group in groups.values():
            h = len(group) / n
            score = sum(value - mean for value in group)
            adjusted_scores.append(score / ((1.0 - h) ** 0.5))
        variance = sum(score * score for score in adjusted_scores) / n**2
        h = [size / n for size in sizes]
        p = []
        for g, hg in enumerate(h):
            row = []
            for j, hj in enumerate(h):
                row.append(
                    hg / n if g == j else
                    -hg * hj / (n * ((1.0 - hg) * (1.0 - hj)) ** 0.5)
                )
            p.append(row)
        trace = sum(p[i][i] for i in range(len(p)))
        df = trace * trace / sum(value * value for row in p for value in row)
        return mean, variance, df

    def test_cr2_matches_general_score_and_p_matrix(self) -> None:
        rng = random.Random(240831)
        for _ in range(200):
            sizes = [rng.randint(1, 5) for _ in range(rng.randint(6, 14))]
            values = []
            clusters = []
            for index, size in enumerate(sizes):
                for _ in range(size):
                    values.append(rng.choice((-1.0, 0.0, 1.0)))
                    clusters.append(f"g{index}")
            reference_mean, reference_variance, reference_df = self._matrix_reference(
                values, clusters
            )
            if reference_df < 5 - 1e-12:
                continue
            # Skip the deliberately fail-closed nonzero/zero-variance case.
            if reference_variance == 0 and reference_mean != 0:
                continue
            observed = claim_builder.cr2_intercept_stats(values, clusters)
            self.assertAlmostEqual(observed["estimate"], reference_mean, places=12)
            self.assertAlmostEqual(
                observed["variance_cr2"], reference_variance, places=12
            )
            self.assertAlmostEqual(
                observed["cluster_reference_df_satterthwaite"], reference_df,
                places=12,
            )

    def test_cr2_intercept_identities_and_invariance(self) -> None:
        singleton_values = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        singleton_clusters = [f"s{i}" for i in range(len(singleton_values))]
        singleton = claim_builder.cr2_intercept_stats(
            singleton_values, singleton_clusters
        )
        mean = sum(singleton_values) / len(singleton_values)
        sample_variance = sum((value - mean) ** 2 for value in singleton_values) / (
            len(singleton_values) - 1
        )
        self.assertAlmostEqual(
            singleton["variance_cr2"], sample_variance / len(singleton_values)
        )
        self.assertAlmostEqual(
            singleton["cluster_reference_df_satterthwaite"],
            len(singleton_values) - 1,
        )

        cluster_means = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0]
        base = claim_builder.cr2_intercept_stats(
            cluster_means, [f"g{i}" for i in range(len(cluster_means))]
        )
        replicated_values = [value for value in cluster_means for _ in range(3)]
        replicated_clusters = [f"g{i}" for i in range(len(cluster_means)) for _ in range(3)]
        replicated = claim_builder.cr2_intercept_stats(
            replicated_values, replicated_clusters
        )
        for key in (
            "estimate", "cluster_se_cr2", "cluster_reference_df_satterthwaite",
            "p_cluster_cr2_satterthwaite",
        ):
            self.assertAlmostEqual(base[key], replicated[key], places=12)

        order = list(reversed(range(len(replicated_values))))
        reordered = claim_builder.cr2_intercept_stats(
            [replicated_values[i] for i in order],
            ["renamed-" + replicated_clusters[i] for i in order],
        )
        for key in (
            "estimate", "cluster_se_cr2", "cluster_reference_df_satterthwaite",
            "p_cluster_cr2_satterthwaite",
        ):
            self.assertAlmostEqual(replicated[key], reordered[key], places=12)

        negated = claim_builder.cr2_intercept_stats(
            [-value for value in replicated_values], replicated_clusters
        )
        self.assertAlmostEqual(negated["estimate"], -replicated["estimate"])
        self.assertAlmostEqual(negated["cluster_se_cr2"], replicated["cluster_se_cr2"])
        self.assertAlmostEqual(
            negated["p_cluster_cr2_satterthwaite"],
            replicated["p_cluster_cr2_satterthwaite"],
        )
        self.assertAlmostEqual(
            negated["ci_cluster_95"][0], -replicated["ci_cluster_95"][1]
        )
        self.assertAlmostEqual(
            negated["ci_cluster_95"][1], -replicated["ci_cluster_95"][0]
        )

    def test_cr2_zero_standard_error_policy(self) -> None:
        clusters = [f"g{i}" for i in range(6)]
        zero = claim_builder.cr2_intercept_stats([0.0] * 6, clusters)
        self.assertEqual(zero["cluster_se_cr2"], 0.0)
        self.assertEqual(zero["t_cluster_cr2"], 0.0)
        self.assertEqual(zero["p_cluster_cr2_satterthwaite"], 1.0)
        self.assertEqual(zero["ci_cluster_95"], [0.0, 0.0])
        with self.assertRaisesRegex(AssertionError, "nonzero estimate with zero"):
            claim_builder.cr2_intercept_stats([1.0] * 6, clusters)

    def test_conditional_sign_sensitivity_cannot_set_primary_branch(self) -> None:
        # Six same-sign clusters give a two-sided conditional sign calculation
        # below .05, while twelve model-assisted primary p=.10 values do not
        # yield a Holm rejection. The branch must follow the latter only.
        sensitivity_p = claim_builder.cluster_sign_symmetry_sensitivity([1] * 6)
        self.assertLess(sensitivity_p, 0.05)
        family = claim_builder.holm_family({f"h{i}": 0.10 for i in range(12)})
        self.assertEqual(
            claim_builder.interaction_result_branch(family),
            "no-interaction-detected",
        )

    def test_holm_step_down_is_frozen_and_monotone(self) -> None:
        raw = {f"h{i}": p for i, p in enumerate(
               (0.001, 0.004, 0.01, 0.02, 0.03, 0.04, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0))}
        adjusted = claim_builder.holm_family(raw)
        ordered = sorted(adjusted.values(), key=lambda row: row["p_raw"])
        self.assertEqual(len(ordered), 12)
        self.assertEqual(
            [row["p_holm"] for row in ordered],
            sorted(row["p_holm"] for row in ordered),
        )
        self.assertTrue(ordered[0]["reject_fwer_005"])
        self.assertFalse(ordered[-1]["reject_fwer_005"])


class ClaimsSchemaContractTests(unittest.TestCase):
    @staticmethod
    def _emit_shell(corpus: dict) -> dict:
        """Smallest complete claims object accepted by the TeX emitter."""
        headline = {
            "naive_delta_pp_min": 1.0,
            "naive_delta_pp_max": 2.0,
            "l2_delta_pp_min": 3.0,
            "l2_delta_pp_max": 4.0,
            "interaction_p_min": 0.5,
            "interaction_p_max": 0.9,
            "interaction_delta_pp_range": [-1.0, 1.0],
            "interaction_ci_pp_envelope": [-2.0, 2.0],
            "interaction_max_abs_ci_endpoint_pp": 2.0,
            "interaction_holm_rejections_fwer_005": 0,
            "interaction_pointwise_ci_excluding_zero": 0,
            "interaction_unadjusted_signflip_p_below_005": 0,
            "interaction_unadjusted_cluster_t_p_below_005": 0,
            "n_interaction_tests": 12,
            "n_cells": 24,
            "n_scored_requests": 948,
            "evaluation_request_errors": 0,
            "question_generation_request_errors": 0,
            "o_acc_range": [0.4, 0.5],
            "l_acc_range": [0.3, 0.4],
            "apm_o_range": [1.0, 2.0],
            "apm_l_range": [0.8, 1.8],
            "apm_naive_range": [0.5, 1.5],
            "wall_o_range": [1.0, 2.0],
            "wall_naive_range": [1.1, 2.1],
            "tier_delta_spread_pp_max": 1.0,
            "surv_acc_l_range": None,
            "surv_acc_e_range": None,
            "destroyed_trials": 0,
            "destroyed_item_condition_pairs": 0,
            "destroyed_recovered": 0,
        }
        return {
            "artifacts": {
                "Qf": {
                    "file_bytes": 2**30,
                    "effective_file_bpw": 4.0,
                    "stored_parameters_from_gguf_tensor_shapes": 27e9,
                },
                "Iq": {"file_bytes": 2**30, "effective_file_bpw": 2.0},
                "Ex": {"weight_safetensors_bytes": 2**30},
            },
            "corpora": {"Wc": corpus},
            "grid": {},
            "interaction": {},
            "survival": {},
            "headline": headline,
        }

    def test_corpus_builder_output_flows_through_tex_emitter(self) -> None:
        windows = [{
            "chars_O": 100,
            "chars_A": 50,
            "chars_L2": 60,
            "chars_E": 40,
            "stratum": "S",
            "has_code": False,
        }]
        question_manifest = {
            "accepted_by_verifier_before_target_truncation": 2,
            "rejected_or_unparsed": 1,
            "written_items": 2,
        }
        attempts = [{
            "schema": "paper2-r1-generation-terminal-v1",
            "status": "success",
            "attempt": {
                "pairs_parsed": 3,
                "accepted": 2,
                "rejected_or_unparsed": 1,
            },
        }]
        corpus = claim_builder.build_corpus_claim(
            windows, question_manifest, attempts, [{}, {}],
        )
        self.assertNotIn("verifier_rejection_rate", corpus)
        self.assertAlmostEqual(
            corpus["candidate_pair_rejection_rate_after_successful_requests"],
            1 / 3,
        )
        emitted = claim_builder.emit_tex(self._emit_shell(corpus))
        self.assertIn(r"\newcommand{\PtwoWcRejPct}{33}", emitted)
        self.assertIn(r"\newcommand{\PtwoInterPMax}{0.90}", emitted)

    @staticmethod
    def _private_cuda_runtime() -> dict:
        return {
            "schema": "paper2-r1-llama-cuda-runtime-private-v1",
            "toolkit_release": "12.6",
            "libraries": {
                soname: {
                    "soname": soname,
                    "loader_path": f"/private/cuda/{expected['filename']}",
                    "resolved_path": f"/private/cuda/{expected['filename']}",
                    **expected,
                }
                for soname, expected
                in claim_builder.CUDA_RUNTIME_CONTRACT.items()
            },
        }

    def test_cuda_public_projection_is_exact_and_path_free(self) -> None:
        private = self._private_cuda_runtime()
        canonical = claim_builder.sanitize_llama_cuda_runtime(private)
        self.assertEqual(
            claim_builder.require_canonical_public_cuda_runtime(
                private, canonical,
            ),
            canonical,
        )
        self.assertNotIn("/private/", json.dumps(canonical, sort_keys=True))

        mutations = []
        extra_top_level = json.loads(json.dumps(canonical))
        extra_top_level["private_library_directory"] = "/home/research/cuda"
        mutations.append(extra_top_level)
        extra_library_field = json.loads(json.dumps(canonical))
        first_soname = sorted(extra_library_field["libraries"])[0]
        extra_library_field["libraries"][first_soname]["resolved_path"] = (
            "/private/cuda/library.so"
        )
        mutations.append(extra_library_field)
        changed_privacy = json.loads(json.dumps(canonical))
        changed_privacy["privacy"] = "/home/research/cuda"
        mutations.append(changed_privacy)
        for public in mutations:
            with self.subTest(public=public), self.assertRaisesRegex(
                AssertionError, "canonical public projection",
            ):
                claim_builder.require_canonical_public_cuda_runtime(
                    private, public,
                )


class PromotionBindingTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[dict, dict[str, dict], dict[str, dict]]:
        tier_values = {
            "Q4KXL": {
                "artifact_sha256": "1" * 64,
                "artifact_record_sha256": "2" * 64,
                "engine": "llama.cpp",
                "engine_revision": "q4-revision",
                "server_model_id": f"P2R1-Q4KXL-{'1' * 12}",
            },
            "EXL3": {
                "artifact_sha256": "3" * 64,
                "artifact_record_sha256": "4" * 64,
                "engine": "ExLlamaV3/tabbyAPI",
                "engine_revision": "exl3-revision",
                "server_model_id": "qwen38-exl3-2.0",
            },
            "IQ2S": {
                "artifact_sha256": "5" * 64,
                "artifact_record_sha256": "6" * 64,
                "engine": "llama.cpp",
                "engine_revision": "iq2-revision",
                "server_model_id": f"P2R1-IQ2S-{'5' * 12}",
            },
        }
        run_contracts: dict[str, dict] = {}
        runs = []
        for tier, values in tier_values.items():
            artifact = {
                "kind": "file" if tier != "EXL3" else "directory",
                "path": f"/private/models/{tier}",
                "sha256": values["artifact_sha256"],
            }
            evidence = {
                "tier": tier,
                "artifact": artifact,
                "artifact_record": f"/private/records/{tier}.json",
                "artifact_record_sha256": values["artifact_record_sha256"],
            }
            for corpus in ("TSI", "WC"):
                label = f"P2R1-{tier}-{corpus}"
                run_contracts[label] = {
                    "schema": "paper2-r1-run-contract-v1",
                    "label": label,
                    "deployment": {"tier": tier, **values},
                }
                runs.append({"label": label, "artifact_evidence": copy.deepcopy(evidence)})
        q4 = tier_values["Q4KXL"]
        generator_projection = {
            "model_id": q4["server_model_id"],
            "engine": q4["engine"],
            "engine_revision": q4["engine_revision"],
            "artifact_sha256": q4["artifact_sha256"],
            "artifact_record_sha256": q4["artifact_record_sha256"],
        }
        generation_contracts = {
            corpus: {
                "schema": "paper2-r1-question-generation-contract-v1",
                "corpus": corpus,
                "q4_deployment": copy.deepcopy(generator_projection),
            }
            for corpus in ("TSI", "WC")
        }
        q4_evidence = next(
            row["artifact_evidence"]
            for row in runs if row["label"] == "P2R1-Q4KXL-TSI"
        )
        promotion = {
            "runs": runs,
            "post_campaign_artifacts": {
                "Q4KXL": {
                    "artifact": copy.deepcopy(q4_evidence["artifact"]),
                    "artifact_record": q4_evidence["artifact_record"],
                    "artifact_record_sha256": q4_evidence[
                        "artifact_record_sha256"
                    ],
                    "full_hash_verified": True,
                }
            },
        }
        return promotion, run_contracts, generation_contracts

    def test_exact_official_bindings_pass(self) -> None:
        claim_builder.validate_deployment_bindings(*self._fixture())

    def test_every_label_rejects_every_other_contract_tier(self) -> None:
        tiers = ("Q4KXL", "EXL3", "IQ2S")
        for corpus in ("TSI", "WC"):
            for expected in tiers:
                for replacement in tiers:
                    if replacement == expected:
                        continue
                    promotion, contracts, generators = self._fixture()
                    label = f"P2R1-{expected}-{corpus}"
                    contracts[label]["deployment"]["tier"] = replacement
                    with self.subTest(
                        label=label, replacement=replacement
                    ), self.assertRaisesRegex(
                        AssertionError, "deployment tier disagree"
                    ):
                        claim_builder.validate_deployment_bindings(
                            promotion, contracts, generators
                        )

    def test_label_binding_rejects_missing_or_malformed_deployment(self) -> None:
        malformed = (None, [], "Q4KXL", {}, {"tier": None}, {"tier": {}})
        for deployment in malformed:
            promotion, contracts, generators = self._fixture()
            contracts["P2R1-Q4KXL-TSI"]["deployment"] = deployment
            with self.subTest(deployment=deployment), self.assertRaisesRegex(
                AssertionError, "deployment tier disagree"
            ):
                claim_builder.validate_deployment_bindings(
                    promotion, contracts, generators
                )

    def test_label_to_promoted_tier_mutation_is_rejected(self) -> None:
        promotion, contracts, generators = self._fixture()
        next(
            row for row in promotion["runs"]
            if row["label"] == "P2R1-IQ2S-WC"
        )["artifact_evidence"]["tier"] = "Q4KXL"
        with self.assertRaisesRegex(AssertionError, "artifact tier disagree"):
            claim_builder.validate_deployment_bindings(
                promotion, contracts, generators
            )

    def test_each_generator_projection_field_and_corpus_is_bound(self) -> None:
        replacements = {
            "model_id": "P2R1-Q4KXL-ffffffffffff",
            "engine": "other-engine",
            "engine_revision": "other-revision",
            "artifact_sha256": "a" * 64,
            "artifact_record_sha256": "b" * 64,
        }
        for corpus in ("TSI", "WC"):
            for field, replacement in replacements.items():
                promotion, contracts, generators = self._fixture()
                generators[corpus]["q4_deployment"][field] = replacement
                with self.subTest(corpus=corpus, field=field), self.assertRaisesRegex(
                    AssertionError, "question generator differs"
                ):
                    claim_builder.validate_deployment_bindings(
                        promotion, contracts, generators
                    )

    def test_generator_projection_rejects_extra_fields(self) -> None:
        promotion, contracts, generators = self._fixture()
        generators["TSI"]["q4_deployment"]["tier"] = "Q4KXL"
        with self.assertRaisesRegex(AssertionError, "exact five-field"):
            claim_builder.validate_deployment_bindings(
                promotion, contracts, generators
            )

    def test_generator_projection_rejects_each_missing_field(self) -> None:
        for corpus in ("TSI", "WC"):
            for field in claim_builder.GENERATOR_DEPLOYMENT_FIELDS:
                promotion, contracts, generators = self._fixture()
                del generators[corpus]["q4_deployment"][field]
                with self.subTest(corpus=corpus, field=field), self.assertRaisesRegex(
                    AssertionError, "exact five-field"
                ):
                    claim_builder.validate_deployment_bindings(
                        promotion, contracts, generators
                    )

    def test_two_q4_evaluation_deployments_must_match(self) -> None:
        promotion, contracts, generators = self._fixture()
        contracts["P2R1-Q4KXL-WC"]["deployment"]["engine_revision"] = "changed"
        with self.assertRaisesRegex(AssertionError, "evaluation deployments disagree"):
            claim_builder.validate_deployment_bindings(
                promotion, contracts, generators
            )

    def test_q4_evaluation_projection_is_exact(self) -> None:
        promotion, contracts, generators = self._fixture()
        contracts["P2R1-Q4KXL-TSI"]["deployment"]["unregistered"] = "value"
        with self.assertRaisesRegex(AssertionError, "exact six-field"):
            claim_builder.validate_deployment_bindings(
                promotion, contracts, generators
            )
        for field in (
            "server_model_id", "engine", "engine_revision",
            "artifact_sha256", "artifact_record_sha256",
        ):
            promotion, contracts, generators = self._fixture()
            del contracts["P2R1-Q4KXL-TSI"]["deployment"][field]
            with self.subTest(field=field), self.assertRaisesRegex(
                AssertionError, "exact six-field|artifact evidence disagree"
            ):
                claim_builder.validate_deployment_bindings(
                    promotion, contracts, generators
                )

    def test_both_generator_corpora_are_required(self) -> None:
        promotion, contracts, generators = self._fixture()
        del generators["WC"]
        with self.assertRaisesRegex(AssertionError, "both corpus"):
            claim_builder.validate_deployment_bindings(
                promotion, contracts, generators
            )

    def test_each_post_campaign_q4_binding_must_match(self) -> None:
        for field in (
            "artifact", "artifact_record", "artifact_record_sha256",
            "full_hash_verified",
        ):
            promotion, contracts, generators = self._fixture()
            post = promotion["post_campaign_artifacts"]["Q4KXL"]
            if field == "artifact":
                post[field]["sha256"] = "c" * 64
            elif field == "artifact_record":
                post[field] = "/private/records/other.json"
            elif field == "artifact_record_sha256":
                post[field] = "c" * 64
            else:
                post[field] = False
            with self.subTest(field=field), self.assertRaisesRegex(
                AssertionError, "post-campaign artifact evidence"
            ):
                claim_builder.validate_deployment_bindings(
                    promotion, contracts, generators
                )

    def test_promotion_header_binds_status_chronology_and_protocol(self) -> None:
        digest = "d" * 64
        base = {
            "schema": "paper2-r1-private-promotion-v1",
            "status": claim_builder.PROMOTION_STATUS,
            "study_status": claim_builder.PROMOTION_STUDY_STATUS,
            "protocol_sha256": digest,
            "legacy_p2g_accepted": False,
            "official_runs": 6,
            "official_condition_cells": 24,
        }
        claim_builder.validate_promotion_header(base, protocol_sha256=digest)
        mutations = {
            "status": "publication-approved",
            "study_status": "independent confirmation",
            "protocol_sha256": "e" * 64,
        }
        for field, replacement in mutations.items():
            changed = copy.deepcopy(base)
            changed[field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(
                AssertionError, "status, chronology, protocol, or shape"
            ):
                claim_builder.validate_promotion_header(
                    changed, protocol_sha256=digest
                )
        for field in tuple(base):
            changed = copy.deepcopy(base)
            del changed[field]
            with self.subTest(missing=field), self.assertRaisesRegex(
                AssertionError, "status, chronology, protocol, or shape"
            ):
                claim_builder.validate_promotion_header(
                    changed, protocol_sha256=digest
                )


class ClaimsOutputSafetyTests(unittest.TestCase):
    def test_atomic_pair_is_owner_only_and_leaves_no_staging_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paper2-claims-test-") as raw:
            directory = Path(raw)
            outputs = {
                directory / "claims.json": b'{"schema":"test"}\n',
                directory / "claims.tex": b"% test\n",
            }
            claim_builder.atomic_write_claim_outputs(outputs)
            for path, expected in outputs.items():
                self.assertEqual(path.read_bytes(), expected)
                self.assertEqual(
                    stat.S_IMODE(path.lstat().st_mode),
                    claim_builder.CLAIMS_OUTPUT_MODE,
                )
            self.assertEqual(
                sorted(path.name for path in directory.iterdir()),
                ["claims.json", "claims.tex"],
            )

    def test_atomic_pair_replaces_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paper2-claims-test-") as raw:
            directory = Path(raw)
            first = directory / "claims.json"
            second = directory / "claims.tex"
            first.write_bytes(b"old-json")
            second.write_bytes(b"old-tex")
            first.chmod(0o600)
            second.chmod(0o600)
            outputs = {first: b"new-json", second: b"new-tex"}
            claim_builder.atomic_write_claim_outputs(outputs)
            self.assertEqual(first.read_bytes(), outputs[first])
            self.assertEqual(second.read_bytes(), outputs[second])
            self.assertEqual(
                sorted(path.name for path in directory.iterdir()),
                ["claims.json", "claims.tex"],
            )

    def test_each_symlink_output_is_rejected_without_touching_target(self) -> None:
        for symlink_name in ("claims.json", "claims.tex"):
            with self.subTest(symlink_name=symlink_name), tempfile.TemporaryDirectory(
                prefix="paper2-claims-test-"
            ) as raw:
                directory = Path(raw)
                victim = directory / "victim"
                victim.write_bytes(b"private")
                symlink = directory / symlink_name
                symlink.symlink_to(victim)
                other_name = (
                    "claims.tex" if symlink_name == "claims.json" else "claims.json"
                )
                with self.assertRaisesRegex(AssertionError, "output path is unsafe"):
                    claim_builder.atomic_write_claim_outputs({
                        symlink: b"changed",
                        directory / other_name: b"other",
                    })
                self.assertEqual(victim.read_bytes(), b"private")
                self.assertTrue(symlink.is_symlink())

    def test_symlink_and_group_writable_parent_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paper2-claims-test-") as raw:
            root = Path(raw)
            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(AssertionError, "directory is unsafe"):
                claim_builder.atomic_write_claim_outputs({
                    linked_parent / "claims.json": b"json",
                    linked_parent / "claims.tex": b"tex",
                })
            real_parent.chmod(0o720)
            with self.assertRaisesRegex(AssertionError, "not owner-controlled"):
                claim_builder.atomic_write_claim_outputs({
                    real_parent / "claims.json": b"json",
                    real_parent / "claims.tex": b"tex",
                })

    def test_group_writable_existing_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paper2-claims-test-") as raw:
            directory = Path(raw)
            first = directory / "claims.json"
            first.write_bytes(b"old")
            first.chmod(0o620)
            with self.assertRaisesRegex(AssertionError, "output path is unsafe"):
                claim_builder.atomic_write_claim_outputs({
                    first: b"new", directory / "claims.tex": b"tex"
                })
            self.assertEqual(first.read_bytes(), b"old")

    def test_second_replace_failure_restores_prior_pair(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paper2-claims-test-") as raw:
            directory = Path(raw)
            first = directory / "claims.json"
            second = directory / "claims.tex"
            first.write_bytes(b"old-json")
            second.write_bytes(b"old-tex")
            first.chmod(0o600)
            second.chmod(0o600)
            real_replace = os.replace
            calls = 0

            def replace_with_one_failure(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic second replacement failure")
                return real_replace(source, destination)

            with mock.patch.object(
                claim_builder.os, "replace", side_effect=replace_with_one_failure
            ), self.assertRaisesRegex(OSError, "synthetic second"):
                claim_builder.atomic_write_claim_outputs({
                    first: b"new-json", second: b"new-tex"
                })
            self.assertEqual(first.read_bytes(), b"old-json")
            self.assertEqual(second.read_bytes(), b"old-tex")
            self.assertEqual(
                sorted(path.name for path in directory.iterdir()),
                ["claims.json", "claims.tex"],
            )

    def test_second_replace_failure_from_absent_state_leaves_no_pair(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paper2-claims-test-") as raw:
            directory = Path(raw)
            first = directory / "claims.json"
            second = directory / "claims.tex"
            real_replace = os.replace
            calls = 0

            def replace_with_one_failure(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic second replacement failure")
                return real_replace(source, destination)

            with mock.patch.object(
                claim_builder.os, "replace", side_effect=replace_with_one_failure
            ), self.assertRaisesRegex(OSError, "synthetic second"):
                claim_builder.atomic_write_claim_outputs({
                    first: b"new-json", second: b"new-tex"
                })
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertEqual(list(directory.iterdir()), [])

    def test_cli_modes_are_mutually_exclusive_and_promotion_path_is_fixed(self) -> None:
        script = PAPER_DIR / "scripts" / "build_claims.py"
        self_test = subprocess.run(
            [sys.executable, str(script), "--self-test"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(self_test.returncode, 0, self_test.stderr)
        both = subprocess.run(
            [sys.executable, str(script), "--self-test", "--check"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(both.returncode, 0)
        self.assertIn("not allowed with argument", both.stderr)
        override = subprocess.run(
            [sys.executable, str(script), "--promotion", "/tmp/not-canonical"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(override.returncode, 0)
        self.assertIn("unrecognized arguments", override.stderr)


class PromotionGateTests(unittest.TestCase):
    @staticmethod
    def _aggregate_claim_shell() -> dict:
        keys = {
            "schema", "study_status", "promotion_sha256", "attempt_id",
            "promotion_completed_at", "analysis_harness_sha256",
            "analysis_implementation", "analysis_implementation_sha256",
            "artifacts", "corpora", "grid", "interaction", "survival",
            "headline", "protocol", "hardware", "software",
        }
        return {key: None for key in keys}

    @staticmethod
    def _protocol_text() -> str:
        return """
        P2R1 is a corrective exploratory rerun, not an independent confirmation.
        The exact stratified quotas are TSI 37, 37, 6, and 37 (117 total) and
        WildChat 30, 30, 30, and 30 (120 total). Predeclared retry seed offsets
        are 0, 1,000,000, and 2,000,000. End-to-end chat-completion timing excludes
        preprocessing and has no prefill/decode timing split. WildChat is ODC--By;
        raw content remains private. These are artifact--engine--placement tiers,
        not a weight-precision causal intervention. Character retention is the
        input estimand; no KV-cache or VRAM saving was measured. The primary
        result branch applies Holm across 12 model-assisted approximate CR2
        source-conversation-cluster tests with Satterthwaite degrees of freedom;
        source conversations are treated as independent clusters. Pointwise
        intervals are descriptive. The sign-flip calculation is secondary and
        exact only conditional on sign symmetry. Inference conditions on one
        frozen Q4-generated question set and one seeded decode per cell; it does
        not include alternate question-generation seeds or run-to-run variation.
        There were 0 evaluation request errors and 0 question-generation request
        errors.
        Hash-bound receipts and the rerun harness provide the reproducibility chain.
        """

    def test_builder_is_fail_closed_before_r1_promotion(self) -> None:
        if PROMOTION.exists():
            self.skipTest("P2R1 promotion exists; current-claims tests cover the open gate")
        result = subprocess.run(
            [sys.executable, str(PAPER_DIR / "scripts" / "build_claims.py"), "--check"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("promotion receipt is missing", result.stderr + result.stdout)

    def test_direct_figure_and_pdf_entry_points_are_gated(self) -> None:
        makefile = (PAPER_DIR / "Makefile").read_text()
        for target in (
            "main.pdf", "main_dark.pdf", "explainer.pdf", "explainer_dark.pdf"
        ):
            self.assertRegex(
                makefile, rf"(?m)^{re.escape(target)}:\s+audit-check\b"
            )
        self.assertIn("review-candidates: figures", makefile)
        self.assertIn("--candidate-sources", makefile)
        for reviewed in (
            "review/light/main.pdf", "review/light/explainer.pdf",
            "review/dark/main_dark_review.pdf",
            "review/dark/explainer_dark_review.pdf",
        ):
            self.assertIn(reviewed, makefile)
        canonical_section = makefile.split("main.pdf: audit-check", 1)[1].split(
            "# Forensic outputs", 1
        )[0]
        self.assertNotIn("$(TECTONIC)", canonical_section)
        self.assertEqual(canonical_section.count("\tcp --preserve=mode,timestamps"), 4)
        if not PROMOTION.exists():
            result = subprocess.run(
                [sys.executable, str(PAPER_DIR / "scripts" / "make_figures.py")],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "required canonical-release evidence is missing",
                result.stderr + result.stdout,
            )
        subprocess.run(
            [
                sys.executable,
                str(PAPER_DIR / "scripts" / "build_public_allowlist.py"),
                "--self-test",
            ],
            check=True,
        )

    def test_semantic_gate_rejects_opposite_result_branch(self) -> None:
        qualitative = {"interaction_primary_branch": "interaction-detected"}
        claims = {"headline": {
            "interaction_result_branch": "interaction-detected",
            "qualitative_assertions": qualitative,
        }}
        detected = "A deployed-tier interaction was detected in the corrective grid."
        main = (
            "corrective exploratory rerun; not an independent confirmation; "
            "AI assistance disclosure. P2R1_RESULT_BRANCH: interaction-detected "
            + detected
        )
        explainer = "P2R1_RESULT_BRANCH: interaction-detected " + detected
        narrative = {
            "interaction_result_branch": "interaction-detected",
            "qualitative_assertions": qualitative,
            "result_specific_source_assertions": {
                "title": detected,
                "abstract": detected,
                "conclusion": detected,
                "explainer": detected,
            },
        }
        release_gate.validate_result_narrative(claims, narrative, main, explainer)
        opposite = dict(narrative)
        no_result = "No deployed-tier interaction was detected in the corrective grid."
        opposite["result_specific_source_assertions"] = {
            key: no_result for key in ("title", "abstract", "conclusion", "explainer")
        }
        with self.assertRaises(SystemExit):
            release_gate.validate_result_narrative(
                claims, opposite, main + no_result, explainer + no_result
            )

        no_qualitative = {"interaction_primary_branch": "no-interaction-detected"}
        no_claims = {"headline": {
            "interaction_result_branch": "no-interaction-detected",
            "qualitative_assertions": no_qualitative,
        }}
        no_main = (
            "corrective exploratory rerun; not an independent confirmation; "
            "AI assistance disclosure. "
            "P2R1_RESULT_BRANCH: no-interaction-detected " + no_result
        )
        no_explainer = "P2R1_RESULT_BRANCH: no-interaction-detected " + no_result
        no_narrative = {
            "interaction_result_branch": "no-interaction-detected",
            "qualitative_assertions": no_qualitative,
            "result_specific_source_assertions": {
                key: no_result for key in ("title", "abstract", "conclusion", "explainer")
            },
        }
        release_gate.validate_result_narrative(
            no_claims, no_narrative, no_main, no_explainer
        )

    def test_protocol_gate_rejects_banner_only_legacy_rewrite(self) -> None:
        claims = {
            "headline": {
                "n_cells": 24,
                "n_interaction_tests": 12,
                "evaluation_request_errors": 0,
                "question_generation_request_errors": 0,
            },
            "hardware": {
                "gpu_name": "NVIDIA GeForce RTX 4080 Laptop GPU",
                "gpu_memory_total_mib": 12282.0,
                "gpu_current_power_limit_w": 80.0,
            },
        }
        good = self._protocol_text()
        release_gate.validate_p2r1_protocol_narrative(claims, good, good)
        for stale in (
            "The original generator was unseeded.",
            "The public arm has 105 short items and 15 long/no-code items.",
            "The private arm contains 103 short and 14 long items.",
            "This is included in the prepared public artifact.",
        ):
            with self.subTest(stale=stale), self.assertRaises(SystemExit):
                release_gate.validate_p2r1_protocol_narrative(
                    claims, good + stale, good
                )

    def test_aggregate_gate_rejects_aliased_row_collections(self) -> None:
        claims = self._aggregate_claim_shell()
        claims["records"] = [{"a": 1, "b": 2}]
        with self.assertRaises(SystemExit):
            release_gate.require_aggregate_only_public_claims(claims)

        claims = self._aggregate_claim_shell()
        claims["grid"] = {"Wc": {"cells": {"Qf-O": {
            "observations": [{"a": 1, "b": 2}],
        }}}}
        with self.assertRaises(SystemExit):
            release_gate.require_aggregate_only_public_claims(claims)

        claims = self._aggregate_claim_shell()
        claims["grid"] = {"aliases": ["synthetic-link-id-a", "synthetic-link-id-b"]}
        with self.assertRaises(SystemExit):
            release_gate.require_aggregate_only_public_claims(claims)

        claims = self._aggregate_claim_shell()
        claims["grid"] = {"attempt_id": "nested-linkable-id"}
        with self.assertRaises(SystemExit):
            release_gate.require_aggregate_only_public_claims(claims)

        claims = self._aggregate_claim_shell()
        claims["interaction"] = {"Wc": {"A-Ex": {
            "cluster_n_items_raw_z_sum_histogram": [
                {"cluster_n_items": 2, "raw_z_sum": -1, "multiplicity": 4}
            ]
        }}}
        release_gate.require_aggregate_only_public_claims(claims)

    def test_every_allowed_public_type_is_privacy_scanned(self) -> None:
        self.assertEqual(
            release_gate.ALLOWED_PUBLIC_SUFFIXES,
            SCANNABLE_TEXT_SUFFIXES | {".pdf"},
        )
        self.assertEqual(
            allowlist_builder.ALLOWED_PUBLIC_SUFFIXES,
            release_gate.ALLOWED_PUBLIC_SUFFIXES,
        )
        self.assertEqual(
            release_gate.ALLOWED_EXTENSIONLESS_BASENAMES,
            SCANNABLE_EXTENSIONLESS,
        )
        self.assertEqual(
            allowlist_builder.ALLOWED_EXTENSIONLESS_BASENAMES,
            SCANNABLE_EXTENSIONLESS,
        )


class ClaimsCurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not PROMOTION.exists():
            raise unittest.SkipTest("P2R1 rerun has not passed the promotion validator")
        cls.claims = json.loads(CLAIMS_JSON.read_text())

    def test_derived_outputs_are_current(self) -> None:
        subprocess.run(
            [sys.executable, str(PAPER_DIR / "scripts" / "build_claims.py"),
             "--check"], check=True)

    def test_grid_shape(self) -> None:
        h = self.claims["headline"]
        promotion = json.loads(PROMOTION.read_text())
        self.assertEqual(self.claims["attempt_id"], promotion["attempt_id"])
        self.assertEqual(h["n_cells"], 24)
        self.assertEqual(h["n_interaction_tests"], 12)
        self.assertEqual(self.claims["grid"]["Tsi"]["n_items"], 117)
        self.assertEqual(self.claims["grid"]["Wc"]["n_items"], 120)

    def test_interaction_reanalysis_is_paired_and_bounded(self) -> None:
        h = self.claims["headline"]
        self.assertIn("no equivalence margin", h["interaction_inference"].casefold())
        self.assertNotIn("establishes independence", h["interaction_inference"])
        ci_detected = 0
        signflip_detected = 0
        cluster_t_detected = 0
        holm_detected = 0
        for corpus, comparisons in self.claims["interaction"].items():
            for key, result in comparisons.items():
                self.assertGreaterEqual(result["n_source_clusters"], 3,
                                        f"{corpus}/{key}")
                lo, hi = result["ci_cluster_95"]
                self.assertLessEqual(lo, result["delta_low_minus_q4_reference"])
                self.assertGreaterEqual(hi, result["delta_low_minus_q4_reference"])
                self.assertGreaterEqual(result["p_cluster_cr2_satterthwaite"], 0.0)
                self.assertLessEqual(result["p_cluster_cr2_satterthwaite"], 1.0)
                self.assertGreaterEqual(result["p_sign_symmetry_sensitivity"], 0.0)
                self.assertLessEqual(result["p_sign_symmetry_sensitivity"], 1.0)
                histogram = result["cluster_n_items_raw_z_sum_histogram"]
                n_hist = sum(
                    row["cluster_n_items"] * row["multiplicity"]
                    for row in histogram
                )
                g_hist = sum(row["multiplicity"] for row in histogram)
                raw_total = sum(
                    row["raw_z_sum"] * row["multiplicity"]
                    for row in histogram
                )
                self.assertEqual(n_hist, result["n_items"])
                self.assertEqual(g_hist, result["n_source_clusters"])
                estimate = raw_total / n_hist
                variance = sum(
                    row["multiplicity"]
                    * (row["raw_z_sum"] - row["cluster_n_items"] * estimate) ** 2
                    / (1.0 - row["cluster_n_items"] / n_hist)
                    for row in histogram
                ) / n_hist**2
                self.assertAlmostEqual(estimate, result["delta_low_minus_q4_reference"])
                self.assertAlmostEqual(variance, result["variance_cr2"])
                expanded_sums = [
                    row["raw_z_sum"]
                    for row in histogram for _ in range(row["multiplicity"])
                ]
                self.assertAlmostEqual(
                    claim_builder.cluster_sign_symmetry_sensitivity(expanded_sums),
                    result["p_sign_symmetry_sensitivity"],
                )
                self.assertGreaterEqual(result["familywise"]["p_holm"], 0.0)
                self.assertLessEqual(result["familywise"]["p_holm"], 1.0)
                ci_detected += int(lo > 0 or hi < 0)
                cluster_t_detected += int(
                    result["p_cluster_cr2_satterthwaite"] < 0.05
                )
                signflip_detected += int(
                    result["p_sign_symmetry_sensitivity"] < 0.05
                )
                holm_detected += int(result["familywise"]["reject_fwer_005"])
        self.assertEqual(h["interaction_pointwise_ci_excluding_zero"], ci_detected)
        self.assertEqual(
            h["interaction_unadjusted_signflip_p_below_005"], signflip_detected
        )
        self.assertEqual(
            h["interaction_unadjusted_cluster_t_p_below_005"], cluster_t_detected
        )
        self.assertEqual(h["interaction_holm_rejections_fwer_005"], holm_detected)
        expected_branch = (
            "interaction-detected" if holm_detected else "no-interaction-detected"
        )
        self.assertEqual(h["interaction_result_branch"], expected_branch)
        self.assertEqual(
            h["qualitative_assertions"]["interaction_primary_branch"],
            expected_branch,
        )

    def test_analysis_implementation_is_bound(self) -> None:
        implementation = REPO_ROOT / "paper4" / "scripts" / "verify_claims.py"
        digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
        self.assertEqual(self.claims["analysis_implementation_sha256"], digest)

    def test_actual_evaluation_strata_are_recorded(self) -> None:
        self.assertEqual(self.claims["grid"]["Tsi"]["n_source_windows"], 117)
        self.assertEqual(self.claims["grid"]["Wc"]["n_source_windows"], 120)
        self.assertEqual(
            self.claims["grid"]["Tsi"]["item_strata_counts"],
            {"S-nocode": 37, "S-code": 37, "L-nocode": 6, "L-code": 37})
        self.assertEqual(
            self.claims["grid"]["Wc"]["item_strata_counts"],
            {"S-nocode": 30, "S-code": 30, "L-nocode": 30, "L-code": 30})

    def test_paired_compression_contrasts_are_well_formed(self) -> None:
        for corpus, g in self.claims["grid"].items():
            for key, cell in g["cells"].items():
                if "paired_vs_O" in cell:
                    paired = cell["paired_vs_O"]
                    self.assertLessEqual(paired["ci"][0], paired["delta"])
                    self.assertGreaterEqual(paired["ci"][1], paired["delta"])
                    self.assertGreaterEqual(paired["p_mcnemar"], 0.0)
                    self.assertLessEqual(paired["p_mcnemar"], 1.0)

    def test_timing_metrics_are_positive(self) -> None:
        for grid in self.claims["grid"].values():
            for cell in grid["cells"].values():
                self.assertGreater(cell["mean_wall_s"], 0)
                self.assertGreaterEqual(cell["acc_per_min"], 0)

    def test_request_outcome_aggregates_are_complete(self) -> None:
        total_errors = 0
        for grid in self.claims["grid"].values():
            for cell in grid["cells"].values():
                outcomes = cell["request_outcomes"]
                self.assertEqual(
                    outcomes["success"]
                    + outcomes["identity_preserved_request_errors"],
                    outcomes["total"],
                )
                self.assertEqual(outcomes["total"], cell["n"])
                self.assertEqual(
                    sum(outcomes["request_error_type_counts"].values()),
                    outcomes["identity_preserved_request_errors"],
                )
                total_errors += outcomes["identity_preserved_request_errors"]
        self.assertEqual(
            total_errors, self.claims["headline"]["evaluation_request_errors"]
        )
        generator_errors = 0
        for corpus in self.claims["corpora"].values():
            summary = corpus["question_generation"]
            self.assertEqual(
                sum(summary["request_error_type_counts"].values()),
                summary["identity_preserved_request_errors"],
            )
            generator_errors += summary["identity_preserved_request_errors"]
        self.assertEqual(
            generator_errors,
            self.claims["headline"]["question_generation_request_errors"],
        )

    def test_survival_partition_is_consistent(self) -> None:
        for corpus, sc in self.claims["survival"].items():
            for cond, entry in sc.items():
                for tier, tv in entry["tiers"].items():
                    self.assertEqual(
                        tv["surv_n"] + tv["dest_n"], entry["n"],
                        f"{corpus}/{cond}/{tier} survival split inconsistent")

    def test_tex_macros_unique(self) -> None:
        names = re.findall(r"\\newcommand\{(\\Ptwo\w+)\}",
                           CLAIMS_TEX.read_text())
        self.assertEqual(len(names), len(set(names)))
        self.assertGreater(len(names), 250)


class ReleaseStateTests(unittest.TestCase):
    """Author/public-package gates intentionally run after claims review."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (PUBLIC_ALLOWLIST.is_file() and NARRATIVE_AUDIT.is_file()):
            raise unittest.SkipTest(
                "release allowlist and author narrative audit are not complete"
            )

    def test_result_specific_manuscript_and_explainer_gate(self) -> None:
        subprocess.run(
            [sys.executable, str(PAPER_DIR / "scripts" / "check_release_state.py")],
            check=True,
        )


class PrivacyTests(unittest.TestCase):
    """No private-corpus text in any released file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.released = public_text_views()
        cls.released_ngrams = {
            path: set(distinctive_ngrams(text))
            for path, text in cls.released.items()
        }

    def _assert_absent(self, needle: str, why: str) -> None:
        needle = normalized_text(needle)
        for path, hay in self.released.items():
            self.assertNotIn(needle, hay,
                             f"private {why} found in released file {path}")

    def _assert_distinctive_substrings_absent(
        self, value: str, why: str, *, scan_all_short: bool = False,
    ) -> None:
        text = normalized_text(value)
        if 12 <= len(text) < 20 and (
            scan_all_short
            or any(char.isdigit() for char in text)
            or len(text.split()) >= 3
        ):
            self._assert_absent(text, why)
            return
        if 20 <= len(text) < 48:
            self._assert_absent(text, why)
            return
        for chunk in distinctive_ngrams(text):
            for path, grams in self.released_ngrams.items():
                self.assertNotIn(
                    chunk, grams, f"private {why} found in released file {path}"
                )

    def test_distinctive_ngram_scanner_covers_every_offset(self) -> None:
        value = "  Alpha\n" + "0123456789abcdefghijklmnopqrstuvwxyz" * 3 + " omega  "
        normalized = normalized_text(value)
        grams = distinctive_ngrams(value)
        self.assertEqual(len(grams), len(normalized) - 48 + 1)
        self.assertEqual(grams[0], normalized[:48])
        self.assertEqual(grams[1], normalized[1:49])
        self.assertEqual(grams[-1], normalized[-48:])
        self.assertEqual(distinctive_ngrams("x" * 48), ["x" * 48])
        self.assertEqual(distinctive_ngrams("x" * 47), [])

    def test_short_question_scanner_has_no_13_to_19_character_gap(self) -> None:
        old_released = self.released
        try:
            self.released = {
                Path("synthetic-public.txt"): "prefix who won today? suffix"
            }
            with self.assertRaises(AssertionError):
                self._assert_distinctive_substrings_absent(
                    "Who won today?", "question text", scan_all_short=True,
                )
        finally:
            self.released = old_released

    def test_no_private_gold_answers(self) -> None:
        # Single common words (e.g. a gold answer that happens to be
        # "chatgpt") are not identifying and collide with ordinary prose;
        # the scan targets distinctive strings: multi-word or digit-bearing
        # answers, and full question sentences.
        if not PROMOTION.exists():
            self.skipTest("P2R1 promotion/attempt-specific question set not present")
        promotion = json.loads(PROMOTION.read_text())
        question_root = Path(promotion["question_root"])
        qfiles = [question_root / corpus / "questions.jsonl" for corpus in ("tsi", "wildchat")]
        if not any(path.exists() for path in qfiles):
            self.skipTest("private corpus not present on this machine")
        for qfile in qfiles:
            if not qfile.exists():
                continue
            with qfile.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    ans = row["answer"].strip()
                    if len(ans) >= 6 and (" " in ans or any(ch.isdigit() for ch in ans)):
                        self._assert_absent(ans, "gold answer")
                    self._assert_distinctive_substrings_absent(
                        row["question"], "question text", scan_all_short=True,
                    )

    def test_no_private_window_text(self) -> None:
        wfiles = [R1_PRIVATE / corpus / "windows.jsonl" for corpus in ("tsi", "wildchat")]
        if not any(path.exists() for path in wfiles):
            self.skipTest("private corpus not present on this machine")
        for wfile in wfiles:
            if not wfile.exists():
                continue
            with wfile.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    for key in ("text_O", "text_A", "text_L2", "text_E"):
                        self._assert_distinctive_substrings_absent(
                            row.get(key, ""), f"window {key} text chunk"
                        )

    def test_no_private_deterministic_ids(self) -> None:
        values: set[str] = set()
        upstream_hashes: set[str] = set()
        source_rows: set[int] = set()
        for corpus in ("tsi", "wildchat"):
            windows = R1_PRIVATE / corpus / "windows.jsonl"
            if windows.is_file():
                with windows.open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        values.update(
                            str(row[key]) for key in ("window_id", "source_cluster_id")
                            if row.get(key)
                        )
            source_manifest = R1_PRIVATE / corpus / "source_manifest.json"
            if source_manifest.is_file():
                source = json.loads(source_manifest.read_text())
                for provenance in source.get("private_window_provenance", []):
                    upstream_hash = provenance.get("upstream_conversation_hash")
                    source_row = provenance.get("source_row_index")
                    if isinstance(upstream_hash, str) and upstream_hash:
                        upstream_hashes.add(upstream_hash)
                    if isinstance(source_row, int) and not isinstance(source_row, bool):
                        source_rows.add(source_row)
        if PROMOTION.exists():
            promotion = json.loads(PROMOTION.read_text())
            question_root = Path(promotion["question_root"])
            for corpus in ("tsi", "wildchat"):
                questions = question_root / corpus / "questions.jsonl"
                if questions.is_file():
                    with questions.open() as handle:
                        for line in handle:
                            row = json.loads(line)
                            values.update(
                                str(row[key])
                                for key in ("qid", "window_id", "source_cluster_id")
                                if row.get(key)
                            )
        if not (values or upstream_hashes or source_rows):
            self.skipTest("private deterministic IDs are not present")
        for value in sorted(values):
            if len(value) >= 4:
                self._assert_absent(value, "deterministic item/source ID")
        for value in sorted(upstream_hashes):
            self._assert_absent(value, "upstream conversation hash")
        # Bare small integers are not identifying in a quantitative paper.
        # Scan labeled forms that would expose the private snapshot row while
        # avoiding false positives on ordinary counts such as 117 or 120.
        for value in sorted(source_rows):
            for labeled in (
                f'"source_row_index": {value}',
                f"source_row_index {value}",
                f"source row index {value}",
            ):
                self._assert_absent(labeled, "upstream source-row index")

    def test_no_private_response_text(self) -> None:
        if not PROMOTION.exists():
            self.skipTest("P2R1 promotion/result streams not present")
        promotion = json.loads(PROMOTION.read_text())
        for run in promotion.get("runs", []):
            outdir = Path(run["result_directory"])
            with (outdir / "p2grid.jsonl").open() as handle:
                for line in handle:
                    row = json.loads(line)
                    self._assert_distinctive_substrings_absent(
                        row.get("response", ""), "evaluation response"
                    )
                    evidence = row.get("api_evidence", {})
                    self._assert_distinctive_substrings_absent(
                        evidence.get("raw_message_content", ""),
                        "raw evaluation response",
                    )
                    raw_b64 = evidence.get("raw_api_response_base64")
                    if isinstance(raw_b64, str):
                        try:
                            raw_body = base64.b64decode(raw_b64, validate=True)
                        except Exception as exc:
                            raise AssertionError("invalid retained evaluation body base64") from exc
                        self._assert_distinctive_substrings_absent(
                            raw_body.decode("utf-8", errors="replace"),
                            "raw evaluation HTTP body",
                        )
        question_root = Path(promotion["question_root"])
        for corpus in ("tsi", "wildchat"):
            attempts = question_root / corpus / "question_attempts.jsonl"
            with attempts.open() as handle:
                for line in handle:
                    event = json.loads(line)
                    attempt = event.get("attempt", {})
                    self._assert_distinctive_substrings_absent(
                        attempt.get("raw_message_content", ""),
                        "raw generator response",
                    )
                    raw_b64 = attempt.get("raw_api_response_base64")
                    if isinstance(raw_b64, str):
                        try:
                            raw_body = base64.b64decode(raw_b64, validate=True)
                        except Exception as exc:
                            raise AssertionError("invalid retained generator body base64") from exc
                        self._assert_distinctive_substrings_absent(
                            raw_body.decode("utf-8", errors="replace"),
                            "raw generator HTTP body",
                        )

    def test_figures_read_only_claims(self) -> None:
        src = (PAPER_DIR / "scripts" / "make_figures.py").read_text()
        self.assertNotIn("testsuite", src)
        self.assertNotIn("jsonl", src)
        opens = [ln for ln in src.splitlines() if "open(" in ln]
        self.assertEqual(len(opens), 1, f"unexpected file reads: {opens}")
        self.assertIn("claims.json", opens[0])

    def test_figure_axes_accept_reversal_and_large_fresh_results(self) -> None:
        subprocess.run(
            [sys.executable, str(PAPER_DIR / "scripts" / "make_figures.py"),
             "--self-test"],
            check=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=1)
