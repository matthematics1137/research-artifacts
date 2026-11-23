# Experimental-source preparation component

This is preparation, NOT a complete results artifact or approved public release.
No measurements, model weights, benchmark conversations/questions/answers,
private logs, existing execution/approval receipts, or private reading companions
are included. The separately built results component is the analysis-only route.

SOURCE_MANIFEST.json records every original and exported source hash, its
evidence boundary, and each path-only transformation. Most source is byte-for-byte
unchanged. The original live/frozen files were never edited by this export.
Existing preparation observations are not completed experimental execution.

Run `python3 -B replication_source.py` to check the complete inventory and Python
syntax. `--self-test` additionally runs the inspected synthetic measurement
tests in separate isolated Python processes, away from the original repository.
NumPy and SciPy are needed for these tests. They block Python-level network and
process launch operations and real model/cache/device access as an accidental
launch guard, not an adversarial-code security sandbox. No inference is run.

ASSETS.json and replication_check.py supply the separate existence/size plan and
explicit content-hashing route. Pass your own repo/models/hf-cache roots; a full
weight check reads many GB and needs a live machine-availability check first.

ENVIRONMENT.json and RUNTIME.md distinguish P5's dated runtime observation,
current package/build metadata and P6's still-prospective execution. Installed
package versions are not a hash-locked installer or evidence of earlier imports.

The source layout under replication/ preserves the experiment's relative paths.
Follow FULL_RERUN.md for prerequisites and the original workflow. The original
P5 guarded execution adapter binds a private dated baseline intentionally not
shipped here; it cannot be used to fabricate original observations from a new
run. A separate rerun must make its own records. P6's prospective harness keeps
its hardware and evidence gates. External-model/runtime setup and any local
path configuration remain operator steps; this component is not an installer.

P6 additionally includes prepare_p6_rerun.py. It creates a new private source
layout with explicit asset/engine paths and a six-model expected-identity
catalog, preserving this component and all original journals. Its RERUN.md
explains the separate data rebuild, dry-run, hardware and measurement gates.
It does not download, install, run inference or claim old execution evidence.

P5 includes verify_p5_runtime_wheel.py and prepare_p5_runtime_overlay.py,
plus separate dated byte-verification records under runtime-evidence/.
They reproduce the exact upstream wheel and three historically modified files
in a new directory, without importing/installing the engine. The third patch
is an inherited withdrawn conversion workaround, not an inference improvement.
RUNTIME.md distinguishes this byte reconstruction from dependency setup and
actual execution. ENVIRONMENT.json alone remains a metadata projection.
prepare_p5_rerun.py creates a separate private layout with explicit asset,
cache and engine paths, keeping scoring/request code and scientific settings.
It does not manufacture a new execution baseline or transplant the original
one. Its RERUN.md identifies the still-required new-campaign execution path.
The separate freeze_p5_rerun.py becomes freeze_inputs.py in the new workspace:
read-only input checking by default, explicit create-only freezing, and later
verification. It checks the complete pilot/cap/prompt/source inventory but does
not attest runtime/model/hardware ownership or authorize inference. Ordinary
legacy generation entrypoints are not safety preflights.

The archival original paths in documentation/configuration become explicit
generic path markers; the GSM8K scorer's unrelated code-execution scratch path
uses its already-published temporary-directory fallback. All transformations are
listed, not represented as the original executed bytes. Do not point any rerun
at the original study's work directory or weaken its evidence guards.

Third-party license text/notices are included for the identified source/template
portions. Final rights/reference review, complete real data, findings-specific
editing, release-controller integration and human approval remain outstanding.
Third-party notices do not grant a license for unreviewed lab artifacts, models
or benchmark data. Do not upload this preparation component as a results release.
