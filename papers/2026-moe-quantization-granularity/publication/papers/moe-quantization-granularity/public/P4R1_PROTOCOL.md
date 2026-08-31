# P4R1 corrective validation protocol

Status: prospectively specified before P4R1 inference. This is a corrective,
exploratory validation, not an independent confirmation. The 100 item IDs,
historical outcomes, and paper hypothesis were already known when this protocol
was written.

## Why P4R1 exists

The historical six-cell GSM8K result files are internally coherent, but no
primary cell independently proves that every saved response came from its
intended model endpoint. The launchers could accept a pre-existing healthy
server, and shared server-log paths were overwritten. One cell retains more
direct identity evidence than the other five, but it still does not meet the
fail-closed publication contract adopted after the audit. Those files remain
historical audit material; they are not publication evidence. P4R1 replaces all
six primary cells with a single prospective campaign whose
artifact, process, endpoint, placement, request, response, and shutdown
identities are retained per cell.

## Scope and question

Within each of two public MoE artifact ladders, compare IQ1_M and IQ2_XXS with
that ladder's Q4_K_M anchor on the same 100 GSM8K items. The estimand is the
paired change in numeric-answer success for the deployed artifact--engine pair.
The study does not identify expert count, quantizer family, or architecture as
a cause and does not rank MoE families generally.

## Frozen six-cell matrix

All cells use llama.cpp commit
`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, context 4,096, one request at a
time, and a distinct immutable output/log directory.

| Alias | Artifact | SHA-256 | GPU-layer request |
|---|---|---|---:|
| `P4R1-MIXTRAL-IQ1M` | `Mixtral-8x7B-Instruct-v0.1.i1-IQ1_M.gguf` | `7e8d75c23bd4de55582ac91084a0575f081c1de374b693f5ea7ffa6fa66a2e3c` | 29 |
| `P4R1-MIXTRAL-IQ2XXS` | `Mixtral-8x7B-Instruct-v0.1.i1-IQ2_XXS.gguf` | `db62b07756f50d2f046b0a00dc594f1aa73dba82ebdc182d9f55c96474655f03` | 26 |
| `P4R1-MIXTRAL-Q4KM` | `Mixtral-8x7B-Instruct-v0.1.i1-Q4_K_M.gguf` | `7f2f98f301a74b645d0a4c2bd64c98e9811f2dfa26dbae4787c280eb1183632c` | 12 |
| `P4R1-QWEN3MOE-IQ1M` | `Qwen3-30B-A3B-Instruct-2507-UD-IQ1_M.gguf` | `d527a854db2a1582a3ce746a17b1f42d860334ece18d385ede9e2e395058b39e` | automatic fit; no override |
| `P4R1-QWEN3MOE-IQ2XXS` | `Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf` | `aa0b04ac05f71aa94b542b074500b3de956608365e74f0e1d97ae0d6a6380269` | automatic fit; no override |
| `P4R1-QWEN3MOE-Q4KM` | `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` | `6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0` | automatic fit; no override |

## Data and generation contract

- GSM8K test source revision:
  `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc`.
- Source SHA-256:
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
- Item selection: `sorted(random.Random(42).sample(range(1319), 100))`.
- Subset SHA-256:
  `184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37`.
- One completion per item; concurrency 1.
- Per-item inference seed: `4,200,000 + n` for item ID `gsm8k_n`.
  The same item receives the same seed in all six cells.
- Temperature 1.0; top-p 0.95; top-k 20; requested reasoning effort
  `medium`; maximum 2,048 generated tokens.
- Numeric success: absolute error at most `1e-4`, or relative error at most
  `1e-4` for a nonzero expected value.
- A request that fails while endpoint identity remains proven is retained as an
  incorrect row with exact private error telemetry. Loss of process, listener,
  model, artifact, or endpoint identity quarantines the entire cell.
- The harness reads at most 1 MiB of response bytes before JSON decoding. A
  malformed HTTP 200 response retains those exact private bytes, status,
  headers, length, and hash as error telemetry; an oversized response
  quarantines the cell rather than becoming an unmeasured incorrect row.
- Output-limit responses are scored by the same parser and separately marked
  truncated.

## Identity and timing contract

The six GGUFs are fully SHA-256 hashed once before the campaign. After the
operator allows page-cache and thermal state to normalize, live cells verify
the immutable manifest by path, size, device, inode, and mtime without rereading
weight contents. Before and after every request, the harness proves the same
PID/start-time owns the port and exposes the exact alias as the sole model.
Every completion response must repeat that alias.

After all six timed cells have shut down, the campaign fully SHA-256 hashes all
six artifacts again. The post-campaign receipt must match every content hash,
byte count, and pre-campaign stat identity before a campaign receipt can be
created. This post-timing read cannot bias the recorded request intervals.

The live argv must contain exactly the frozen model, alias, host, port,
4,096-token context, and trace verbosity `-lv 4`. Mixtral must contain exactly
its specified `-ngl`; Qwen must contain no GPU-layer override. After endpoint
readiness and before the first benchmark request, an early log probe must find
one unambiguous `offloaded X/Y layers to GPU` readback; for explicit `-ngl`,
`X` must equal the request. A missing, ambiguous, or mismatched readback fails
the cell before evaluation, and the campaign directory is preserved as a
failed attempt rather than resumed. Each cell retains separate stdout/stderr,
the exact readback, PID-scoped GPU-memory snapshots, startup and completion
identities, and a shutdown receipt proving the same PID/start-time is gone and
the port is free.

`wall_s` measures only the HTTP completion request/response interval using a
monotonic clock. Identity-guard overhead is recorded separately. Neither field
includes model loading, shutdown, or either full artifact-verification pass.

## Analysis contract

Report every cell's numerator, denominator, accuracy, and Wilson 95% interval.
For IQ1-versus-Q4 and IQ2-versus-Q4 within each ladder, retain the paired 2x2
table, Q4-minus-low risk difference with Newcombe method-10 interval, and exact
two-sided McNemar p-value. Apply Holm adjustment across exactly those four
McNemar tests. Report truncations, API errors, chat-template retries, placement,
request time, and identity-guard time without exclusion.

MATH and HumanEval+ are outside P4R1. Historical supplementary cells must not
appear as validated P4R1 evidence unless a separately frozen protocol reruns
them with the same identity controls.

## Prospective freeze and deviations

`prepare_campaign_plan.py` may create the plan receipt only when this protocol,
every inference/scoring source, and every listed result-interpreting analysis
source is tracked and clean in both the Git index and worktree. The two source
inventories are separate. The latter includes the promoter, public checker,
tests, claims analysis/builder, figure builder, manuscript gate, and Makefile.
The author then signs an interactive approval bound to the exact plan SHA-256.
`run_matrix.sh` verifies both immediately before launch and binds both to the
campaign.

Any listed inference, scoring, promotion, or analysis-source change after the
freeze requires a new commit, protocol, plan, approval, and campaign directory
under the implemented strict gate. No amendment path is implemented. Raw files
from a blocked or superseded campaign may remain forensic material, but they
cannot be promoted. No silent post-outcome source drift is accepted. A failed
or partial cell is quarantined; it is not repaired in place. All deviations
must be documented before a replacement campaign.
