"""Canonical source files frozen for a P2R1 campaign."""

from __future__ import annotations

import subprocess
from pathlib import Path

from identity_guard import sha256_file
from r1_safety import RunSafetyError


# These files can change the requests, inputs, scoring rows, timing, or live
# deployment identity. Their exact hashes are prospective measurement
# evidence and must match every question/run contract forever.
MEASUREMENT_HARNESS_FILES = (
    "methodology/EMPIRICAL_RESEARCH_LOOP.md",
    "paper2/R1_PROTOCOL.json",
    "paper2/RERUN_PLAN.md",
    "publication/papers/qwen38_27b_12gb/public/environment/model_artifacts.json",
    "testsuite/paper2/build_wildchat_r1.py",
    "testsuite/paper2/campaign_guard.py",
    "testsuite/paper2/freeze_tsi_r1.py",
    "testsuite/paper2/gen_questions.py",
    "testsuite/paper2/identity_guard.py",
    "testsuite/paper2/ladder.py",
    "testsuite/paper2/make_conditions_r1.py",
    "testsuite/paper2/p2_grid_r1.sh",
    "testsuite/paper2/prepare_tabby_config_r1.py",
    "testsuite/paper2/protocol_bindings.py",
    "testsuite/paper2/r1_safety.py",
    "testsuite/paper2/runtime_provenance.py",
    "testsuite/paper2/requirements-r1.txt",
    "testsuite/paper2/run_grid.py",
    "testsuite/paper2/selftest_r1.py",
    "testsuite/paper2/tabby_p2r1_config.template.yml",
    "testsuite/paper2/validate_inputs_r1.py",
    "testsuite/paper2/warmup_r1.py",
)

# These files interpret already-frozen rows. They are captured at promotion
# and release, but are deliberately not part of the inference run contract:
# a reviewed statistics/checker bug fix must not invalidate sound requests.
ANALYSIS_HARNESS_FILES = (
    "paper2/Makefile",
    "paper2/scripts/build_claims.py",
    "paper2/scripts/build_public_allowlist.py",
    "paper2/scripts/check_release_state.py",
    "paper2/scripts/make_figures.py",
    "paper2/scripts/test_claims.py",
    "paper2/scripts/validate_r1.py",
    "paper4/scripts/verify_claims.py",
)


def hashes_for(root: Path, relative_files: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for relative in relative_files:
        path = root / relative
        if not path.is_file():
            raise RunSafetyError(f"frozen harness source is missing: {relative}")
        values[relative] = sha256_file(path)
    return values


def source_hashes(root: Path) -> dict[str, str]:
    """Backward-compatible name for the prospective measurement bundle."""
    return hashes_for(root, MEASUREMENT_HARNESS_FILES)


def analysis_source_hashes(root: Path) -> dict[str, str]:
    return hashes_for(root, ANALYSIS_HARNESS_FILES)


def require_files_committed_clean(root: Path, relative_files: tuple[str, ...]) -> None:
    for relative in relative_files:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", relative],
            capture_output=True,
        )
        if tracked.returncode:
            raise RunSafetyError(f"frozen harness source is not committed: {relative}")
    for args, description in (
        (["git", "-C", str(root), "diff", "--quiet", "--", *relative_files], "unstaged"),
        (["git", "-C", str(root), "diff", "--cached", "--quiet", "--", *relative_files], "staged"),
    ):
        result = subprocess.run(args)
        if result.returncode:
            raise RunSafetyError(f"frozen harness has {description} changes")


def require_committed_clean(root: Path) -> None:
    """Require the prospective measurement bundle to be committed and clean."""
    require_files_committed_clean(root, MEASUREMENT_HARNESS_FILES)


def require_analysis_committed_clean(root: Path) -> None:
    """Require the post-run analysis/promoter bundle to be auditably frozen."""
    require_files_committed_clean(root, ANALYSIS_HARNESS_FILES)
