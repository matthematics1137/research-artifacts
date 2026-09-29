# P4R1 replication-source layout

The top-level `data/` directory and `check_claims.py` are the fast,
standard-library-only analysis reproduction. A full six-cell inference repeat
also needs the source-layout snapshot included in this artifact, the six pinned
GGUF files, the pinned llama.cpp build, and compatible CUDA hardware.

The P4R1 campaign guard deliberately binds a prospective plan to a clean Git
commit and to fixed repository-relative source paths. To preserve that safety
property outside the working repository, this package carries the required
files at those original relative paths:

- `publication/papers/moe-quantization-granularity/public/` contains the
  frozen protocol, runbook, harness, promoter, checker, and tests;
- `paper4/scripts/` contains the result-building and manuscript-validation
  sources named by the campaign plan; and
- `paper4/Makefile` completes that plan's source inventory.

The `paper4/Makefile` is an inventory record of the lab editing workflow, not
the portable paper-build entry point: its private review/controller targets
require the original workspace. Use the separate shipped `paper/Makefile`
(`make -B -C paper`) for analysis-only PDF reproduction. No editing controller
or private companion-building scripts are needed for that command.

After extracting the artifact, create a local-only Git snapshot before making
a new prospective plan:

```bash
git init
git add publication/papers/moe-quantization-granularity/public paper4
git -c user.name='Replication operator' \
    -c user.email='operator@invalid.example' \
    commit -m 'Freeze local P4R1 replication source'
cd publication/papers/moe-quantization-granularity/public
```

Then follow `FULL_RERUN.md` from that directory. Its relative commands and the
plan guard resolve against the reconstructed artifact root. The approval made
there authorizes only the operator's new inference campaign; it does not
approve or publish this paper.

The replication scripts and tests in the nested public directory are the exact
files bound into the completed P4R1 evidence. The accompanying `paper4`
narrative-validation sources are the local release-candidate versions. This
distinction preserves the signed experiment evidence while allowing the final
manuscript gates to remain fail-closed.

No model weights, benchmark text, gold answers, response prose, private logs,
or prior approval receipts are included. The benchmark builder retrieves the
pinned upstream revision and reconstructs the required 100-item subset.
