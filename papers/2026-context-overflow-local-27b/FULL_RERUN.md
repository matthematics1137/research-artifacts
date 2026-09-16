# Full experimental replication

This private candidate packages an existing completed benchmark's sanitized analysis and historical source snapshots. It does not represent a newly executed experiment.

The exact asset catalog is `replication/experiment/ASSETS.json` (30 assets, immutable upstream revisions or generated-artifact declarations). The pinned environment and runtime limitations are in `replication/experiment/ENVIRONMENT.json` and `RUNTIME.md`. The full historical instructions, serving settings, pilot, oracle correction, 6,157 primary requests and both follow-ups are in `replication/experiment/FULL_RERUN.md`; those instructions are historical context and do not authorize a launch.

For a new machine, start with `prepare_p5_rerun.py` as documented there. It writes new path-bound source/configuration without copying models, datasets or old execution receipts. `freeze_p5_rerun.py` checks prospective inputs, not model execution. The documented prospective pilot/runtime/execution gate remains incomplete. Do not bypass snapshot-bound historical wrappers to run a new campaign. New answers need not be byte-identical because generation is stochastic.

Remaining release-review work: independently check the full prospective commands, runtime installation, hardware/headroom, model and dataset access terms, expected output counts, cost estimates and statistical tolerances. Full-run documentation remains a pending gate. No new GPU run is part of this release-preparation task.

For the independently runnable public scored-data replay, use the top-level README. LoCoMo's public aggregate projection cannot regenerate its conversation-bootstrap token-F1 intervals.
