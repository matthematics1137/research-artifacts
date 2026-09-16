# Design pre-registration — Paper 5: Same Budget, Same Questions

Status: frozen 2026-09-01, before any harness code, download, or measurement.
Prior art: `prior-art-overflow-policies.md` (same date). Campaign may not
start until the P2R1 campaign has fully completed and validated; nothing in
this design runs beside it.

## Question

At a fixed prompt-token budget B per turn, which context-overflow policy
class preserves the most downstream task accuracy over long, evolving
conversations: truncate-oldest (llama.cpp context-shift semantics), rolling
summarization, learned prompt compression (LLMLingua-2), or a minimal
hierarchical external memory? Secondary: does the ranking change with B?

This is the budget-matched comparison that AgentMemBench (arXiv:2608.00009)
explicitly does not perform and MemDelta (arXiv:2606.29914) explicitly calls
for, plus the first published accuracy evaluation of llama.cpp's deployed
context-shift default.

## Arms

Budget-matched arms (total prompt = system + managed history + current
question, tokenizer-counted, must be <= B; verified and logged per turn):

1. **T** — truncate-oldest with protected prefix, replicating llama.cpp
   context-shift semantics (`n_keep` system prefix; oldest half of the
   remainder discarded on overflow).
2. **S** — rolling summarization: the generator model itself summarizes
   evicted spans into a running summary held within B (fixed summarization
   prompt, seeded decode, summary length cap frozen in the harness).
3. **C** — LLMLingua-2 compression of the full history to fit B (pinned
   `llmlingua-2-xlm-roberta-large-meetingbank`, CPU, same environment as
   paper 2).
4. **M** — minimal hierarchical memory: evicted turns stored verbatim in an
   external store; per-question retrieval (frozen embedding model on CPU,
   top-k by cosine) re-inserts retrieved spans, capped so the total prompt
   stays <= B.

Controls (not budget-matched, contextualize the arms):

5. **O** — full-context oracle upper bound (no budget, capped only by the
   model's context window).
6. **E** — empty-history lower bound (system + question only).

## Task and scoring

- Primary corpus: **LongMemEval-S** (public), full ~500-question set,
  evolving multi-session histories, benchmark-native QA scoring
  (exact/F1 as defined upstream).
- Secondary corpus: **LoCoMo** subset (~200 questions) for comparability with
  AgentMemBench/Mem0; run only the winning-policy-vs-T contrast.
- Public data only — no private corpus anywhere in this study, so the full
  per-item release is possible for the first time in this series.
- Two generator models from different families, both fully GPU-resident on
  the 12 GiB card with the required context (target ~8B class, e.g. a Qwen3
  8B GGUF and a Llama/Gemma-class 8-9B GGUF; exact artifacts pinned by
  repo+revision+sha256 in an amendment before any download, per house rules).
- Fixed seeds, temperature 0.2, top-p 0.95, completion cap 2048 (P2R1
  amendment lesson: never let the cap truncate the reasoning of a thinking
  model; verify the chat template's reasoning behavior per model before
  freeze), 600 s timeout.
- Budgets: B = 1024 and B = 4096 tokens (primary analysis at 4096).

## Pre-registered outcomes (all publishable)

1. Primary: per-item accuracy of S, C, M each vs T at B=4096 on the primary
   model — paired McNemar, Holm across the 3 contrasts at familywise 0.05.
   Any direction is publishable; "truncation is not beaten at this budget" is
   as reportable as any winner.
2. Budget interaction: does the policy ranking flip between B=1024 and
   B=4096 (descriptive, pointwise intervals)?
3. Oracle gap: accuracy loss of the best budget-matched arm vs O (how much
   the budget costs at all).
4. Cross-model agreement (secondary model at B=4096 only).
5. Wall-time per turn as a descriptive secondary (policy overheads included:
   summarization passes, compression CPU time, retrieval; reported
   separately from generation time).

## Plan

- Cells: 500 items x 4 arms x 2 budgets + controls (500 x 2) on the primary
  model = ~5,000 scored requests; secondary model 500 x 4 at B=4096 = 2,000;
  LoCoMo replication ~400. Estimated 35-50 GPU-hours as armed overnights.
- Harness reuses the P2R1 discipline: identity-verified endpoint bound to
  PID/config/artifact hash, seeded decode, append-only hash-chained journals,
  request intents, cool gates, scoped shutdown, full artifact hashing before
  and after. Reuse `testsuite/paper2/identity_guard.py` etc. directly.
- Budget-parity is a hard gate: any turn whose realized prompt exceeds B
  aborts the arm's session fail-closed (no silent overage).
- Statistics implemented against the paper-4 audited stats module
  (Wilson/McNemar/Newcombe) plus cluster bootstrap by conversation for F1.

## Constraints and notes

- Starts only after P2R1 completes; downloads (datasets ~small, two ~6 GB
  GGUFs, LLMLingua-2 already local) staged before the campaign, disk budget
  ~15 GiB against 131 GiB free.
- Scoop risk is the schedule driver (see prior-art sweep): design freeze now,
  harness while P2R1 evaluates, campaign immediately after, preprint fast.
- Not exotic by construction: same hardware, same engine family, same
  statistical house style; the novelty is the controlled budget-matched
  design, not new machinery.
- The context governor tool idea (2026-08-31 discussion) becomes this
  paper's applied artifact: the winning policy is what the governor should
  implement; the tool ships after the paper, citing it.

## Amendments

### 2026-09-01 — dataset-inspection amendments (before any harness code or measurement)

Staging (`testsuite/paper5_data/MANIFEST.md`, pinned revisions + sha256)
surfaced four facts requiring pre-data decisions:

1. **Cleaned variant.** The original `xiaowu0162/longmemeval` release is
   author-deprecated in favor of `longmemeval-cleaned` (noisy sessions
   interfere with answer correctness). Frozen choice: the cleaned variant
   (`longmemeval_s_cleaned.json`, revision `98d7416`) is primary; same 500
   question ids and type mix.
2. **Oracle arm redefinition.** Median history is ~122k estimated tokens, so
   a full-context oracle cannot run on the 12 GiB card with an 8B-class
   resident model. Arm O becomes the upstream **oracle variant**
   (`longmemeval_oracle.json`: evidence sessions only) — a perfect-retrieval
   upper bound that fits any budget. The paper describes it as
   perfect-retrieval, not full-context; no full-context arm exists in this
   study and no claim about one will be made.
3. **LoCoMo subset rule.** LoCoMo has 1,986 questions, not ~200. Frozen
   deterministic subset: 200 questions from categories 1 (multi-hop),
   2 (temporal), 4 (single-hop) at 70/70/60, selected by ascending
   SHA-256(question_id) within category; category 5 (adversarial, no
   canonical answer field) and category 3 are excluded from the replication
   contrast. Image turns contribute their text and `blip_caption` only.
   **Licensing:** LoCoMo is CC BY-NC 4.0 — the LoCoMo portion releases
   aggregates only; the full per-item public release applies to the
   LongMemEval portion (MIT).
4. **Answer normalization.** 32 LongMemEval answers are integers; the scorer
   normalizes all answers to strings before benchmark-native scoring. The 30
   `_abs` abstention items are scored by the benchmark's own abstention rule
   and additionally reported as a separate subgroup.

### 2026-09-19 — feasibility amendment (before any harness code, pilot, or measurement)

Authority: Matt, 2026-09-19: "proceed with full finished drafts for all in
2-7 'bring it home' then i can review and edit". The owner delegated the
open decisions in `research/DECISIONS-PAPERS-5-6-2026-09-19.md`, so they are
taken here at the recommended defaults, except where noted. The prior-art
refresh `research/prior-art-overflow-policies-2026-09-19.md` sets the
framing. No LongMemEval generation, scoring, or pilot had run when this was
written; only token counts of the staged data had been measured (19,195
unique sessions, median 2,255 tokens; item histories 100,756–112,043
Qwen tokens).

**1. Generator.** One primary generator: Qwen3.8-27B EXL3 SC_2.00bpw_H3
(`turboderp/Qwen3.8-27B-exl3` @ `d7e02d63dddd4e047a8d397acb22aeb2b0bedcec`,
shards `cf8966eb…`, `2a72afda…`), with its own MTP draft head. It is served
by TabbyAPI `4a4f9f4` with the lab's exllamav3 1.4.7 copy, which is the
`serving/` daily-driver configuration; the P5 config differs only in port and
host-RAM stash. The two-family 8B design is dropped, because no second
family is on disk and the owner took the no-download default for the
generator. The chat template's default thinking mode is kept, as in local
deployment. Decode: temperature 0.2, top-p 0.95. Seed per request is the
first 8 hex digits of SHA-256(`item_id|arm|budget`), as an integer.
Completion cap 8,192 tokens; timeout 900 s. The cap is subject to the pilot
rule in item 9.

**2. Rendering and budget parity.** Every prompt is one user message in the
upstream LongMemEval reader template (commit `d6dc8b50…`,
`run_generation.py`, the non-CoT "history chats" template, `nl` history
format, session blocks `### Session i: / Session Date: / Session
Content:`). There is no system prompt. The harness renders the prompt with
the model's own `chat_template.jinja` and sends it as a raw completion.

- **Budget.** B counts every token of that rendered string with the served
  `tokenizer.json`: template tokens and the generation prompt included.
- **Hard gate.** A request is sent only if the count is ≤ B. The server's
  reported prompt-token count must equal the harness count. Any mismatch
  stops the run fail-closed.
- **Fill rule.** Every budget-matched arm fills toward B. Delivered tokens
  are recorded per request and reported per arm.

**3. Arms (redefined; the class names are unchanged).**

- **T — recency truncation, message-level, fill to B.** Drop the oldest
  whole messages until the prompt fits, keeping each surviving message's
  session header. This is the prompt-side semantics of Ollama's `chatPrompt`
  truncation; on a card under 24 GiB, Ollama's default context is 4,096. The
  llama.cpp framing is withdrawn: context shift is off by default in the
  pinned build, and the server rejects over-long prompts. No claim about
  llama.cpp's default, or about Ollama itself, is made.
- **S — summaries of whole sessions.**
  - The generator itself summarizes each needed session once, cached by
    session id and question-agnostic.
  - Summarizer settings: thinking off, temperature 0, fixed seed, at most
    200 visible tokens, and a frozen prompt that keeps user facts,
    assistant-provided specifics, and every date or time expression. The
    session date is given to the summarizer and shown in the rendered
    header.
  - S fills B with whole summaries, newest first, rendered in chronological
    order.
  - Sessions are summarized lazily, newest first per item, until S at
    B=4,096 is full. The decision file's cost estimate replaces the rolling
    summary of all 48 sessions.
- **C — LLMLingua-2 over the full rendered history.**
  - Model `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` @
    `ebaba9b0e874dadd3003ffcff828e4397e568089`, the same as P2R1.
  - Keep-probabilities are computed once per item by the library's own
    chunking and classifier: llmlingua 0.2.2, default arguments, as in
    P2R1.
  - The library's per-chunk percentile keep rule is then applied at the
    largest attainable rate whose prompt fits B.
  - The library's integer-percentile rule cannot keep less than about 1% per
    chunk. When even that exceeds B, which is expected at B=1,024, the
    oldest compressed chunks are dropped until the prompt fits. This
    fallback is counted and reported.
  - The delivered compression rate is a reported quantity. Near-floor
    accuracy is the expected outcome at 25–100× compression, far outside
    the 2–5× the method was validated for.
- **M — retrieval over evicted rounds.**
  - Unit: a round, meaning a user turn plus the following assistant turn.
  - Retriever: dense, `sentence-transformers/all-MiniLM-L6-v2` @
    `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. It runs on CPU with mean
    pooling and cosine similarity, and truncates inputs at 256 word pieces,
    as the model does. The query is the question text.
  - Fill: the highest-ranked rounds are added until the next does not fit,
    rendered chronologically under their sessions' headers.
  - Every round is evictable, since no item's history fits either budget.
    This arm is named "retrieval over evicted turns", not "hierarchical
    memory".
- **O — perfect-retrieval control.** Evidence sessions from the oracle file,
  fully rendered, only for the items whose O prompt fits B=4,096. It is
  reported on that subset only, labelled as such, and never truncated.
- **E — empty-history control.** The upstream no-retrieval template with the
  question date. It is budget-independent and run once.
- **Declared secondary variants** (descriptive, outside the Holm family):
  - **C-w**: LLMLingua-2 on only the most recent whole chunks, totalling
    about 4×B tokens, compressed to fit B. That is inside the method's
    validated range and is how compressors are deployed. Run at both
    budgets.
  - **M-bm25**: M with Okapi BM25 (k1 = 1.2, b = 0.75, lowercase word
    tokens) instead of the dense retriever. Run at B=4,096 only.

**4. Scoring.** The frozen "exact/F1 as upstream" is corrected. LongMemEval's
upstream metric is an LLM judge with per-type prompts, and McNemar needs a
yes/no label.

- **Primary.**
  - Prompts: the upstream judge prompts verbatim, from `evaluate_qa.py` @
    `d6dc8b50…` (SHA-256
    `ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251`, MIT).
  - Judge: a local model from a different developer than the generator,
    `ggml-org/gpt-oss-20b-GGUF` `gpt-oss-20b-MXFP4.gguf` @
    `ef9b12f2ff56c69cf32153a02784e7a3c88bf524`, SHA-256 verified at
    download.
  - Judge settings: the pinned llama.cpp build, temperature 0, reasoning
    effort low.
  - Label: `'yes' in content.lower()` of the final channel, the upstream
    rule.
  - The judge sees only the generator's visible answer, with its thinking
    removed. An unterminated thinking block leaves an empty answer.
  - A judge reply with no final content is retried once with a larger
    completion cap. If it still fails, it scores incorrect and is counted.
- **Why not the official judge.** It is `gpt-4o-2024-08-06`, and no OpenAI
  credential exists on the lab machine. If one is supplied before the
  analysis freeze, the unmodified official judge is run over every label and
  reported as a validation of the primary judge. It never replaces the
  primary.
- **Secondary scorers.**
  - The generator as a second judge: same prompts, thinking off,
    temperature 0.
  - Normalized containment on short-answer items (the preference and `_abs`
    items are excluded): lowercase, punctuation and articles removed, gold
    contained in the answer.
  - Per-arm agreement (raw and Cohen's kappa) is reported between each pair
    of scorers.

**5. Statistics.**

- **Primary (as frozen).** Exact two-sided McNemar tests of S, C, and M
  against T at B=4,096. Primary judge labels, all 500 items, Holm across the
  three contrasts at a familywise 0.05. Paired differences are reported with
  Newcombe method-10 intervals.
- **Secondary (descriptive, pointwise intervals, no confirmatory claims):**
  - the same contrasts at B=1,024;
  - the C-w and M-bm25 variants;
  - the ranking flip between budgets;
  - the O gap on the fitting subset;
  - E;
  - per question type and the abstention subgroup;
  - evidence retention: whether the evidence (`has_answer`) turns or
    sessions reach the prompt in full, in part, or only as a summary;
  - delivered tokens and compression rates;
  - wall time split into policy cost (summarization, compression,
    retrieval) and generation.

**6. Cross-model check (outcome 4, secondary).**

- Model: Qwen3-8B, a standard-GQA model from the same developer
  (`unsloth/Qwen3-8B-GGUF` `Qwen3-8B-UD-Q4_K_XL.gguf` @
  `a6adef130ffb23ddaf1a62fec9dced968c9bc482`, downloaded for Paper 6).
- Setup: the pinned llama.cpp build, the four arms at B=4,096, prompts
  rebuilt with that model's own tokenizer, and the same S summaries.
- Condition: it runs only after the primary grid. If it is not run, it is
  reported as not run.

**7. LoCoMo leg (replication).**

- Contrast: the best of S, C, and M at B=4,096 by primary-judge accuracy
  against T at B=4,096. Ties break in the order S, M, C.
- Items: the frozen 200-item subset, with speaker names as roles.
- Scoring: the primary judge with the upstream generic template, plus token
  F1.
- Release: aggregates only (CC BY-NC).
- Framing: a replication of the Sleeping Agent's T < S at about 4K, not a
  first.

**8. Framing (from the refresh).**

- BudgetBench (arXiv:2609.13149) is the nearest neighbour, and "first
  fixed-budget comparison" is not claimed.
- The novelty claim is the refresh's E1 wording: all four classes, the
  LLMLingua-2 arm, the full LongMemEval-S haystack, 1,024 and 4,096 tokens,
  local serving, enforced parity, paired tests.
- Prior results that predict the budget-interaction direction are cited as
  expectations.

**9. Pilot (operational, before the grid).**

- Items: 20 LoCoMo category-3 questions, a category excluded from every
  analysis, run under T at B=4,096. The pilot checks the harness end to end
  and measures completion lengths.
- Cap rule: if any pilot completion reaches 6,144 tokens, the completion cap
  becomes 16,384 for the whole study. The choice is recorded before the
  grid.
- No LongMemEval item is used in any pilot.

**10. Operational rules** (adjustable, not hash-bound, per the lab rule after
P2R1):

- a pause file;
- a thermal pause at a GPU temperature of 87 °C or above, resuming below
  75 °C;
- the server is stopped while paused, so the owner can use the GPU.

Journals are append-only. Each row records the prompt's SHA-256, the
harness-file hashes, and the server identity. The model files are hashed
before and after the campaign.

**Estimated cost.** About 10–13k session summaries (13–17 GPU-h); 6,139
primary-model generation requests (T, S, C, and M at two budgets, C-w at two
budgets, M-bm25, O, and E; 20–30 GPU-h); two judge passes (about 8 GPU-h);
the cross-model check and LoCoMo leg (about 6 GPU-h). About 50–60 GPU-hours
in total.

**Clarification 2026-09-19 13:40 (before any harness run or pilot).** "The chat
template's default thinking mode" in item 1 means thinking enabled. The
template's default `reasoning_effort` is `xhigh`, which injects a system
message. The study instead pins `reasoning_effort: medium`, which injects
nothing, for the generator's answering requests. That keeps item 2's "no
system prompt" literally true, and it matches the lab evaluation harness
(`testsuite/evals/run_eval.py`) used for Papers 1 and 7. Summaries and the
generator-as-judge use `enable_thinking: false`.

**Clarification 2026-09-19 14:05 (before any generation).** TabbyAPI `4a4f9f4`
has no per-request seed parameter, and exllamav3 1.4.7's sampler draws from
torch's global generator. The per-request seed in item 1 is therefore
recorded in every journal row but cannot be applied. Sampled requests
(temperature 0.2) are not byte-reproducible, and the paper states this. The
judge requests (llama.cpp, temperature 0) and the summaries (temperature 0)
are not affected. The same limitation applied, unstated, to P2R1's `"seed":
42`; it is noted for the lab ledger, and it does not change any P2R1
analysis.

**Clarification 2026-09-19 15:20 (pilot run; no LongMemEval item measured).** The
pilot showed that TabbyAPI returns an empty `usage` block for raw
completions, so the amendment's parity gate ("the server's reported prompt
count must equal the harness count") had nothing to compare against, and the
pilot's completion lengths were unavailable. Two operational corrections,
both applied before any LongMemEval generation:

1. **Parity** is now checked against the server's own tokenizer through
   `/v1/token/encode` (`add_bos_token` false, special tokens encoded) for
   every grid request. A mismatch with the harness count, or a count above
   the budget, still stops the run fail-closed. This is the same quantity the
   amendment asked for, from the same tokenizer, by a different route.
2. **Completion length** is counted by the harness from the returned text
   with the served tokenizer, and the pre-registered pilot cap rule reads
   that count. The first pilot is therefore rerun; its journal is kept as
   `pilot.firstrun.jsonl`.

The first pilot's 20 requests (LoCoMo category 3, excluded from every
analysis) otherwise passed: prompts inside the budget, thinking blocks
closed, answers present, 7--37 s per request.

**Operational note 2026-09-19 22:35 (summary preparation; before the primary
grid).** At the owner's request, the campaign was paused at a completed
request boundary with 2,830 successful summaries saved. A minimum ten-second
idle interval between live completion attempts was added for lower average
GPU activity. This uses the adjustable operational rules in item 10; it
does not change the model, hardware, prompts, budgets, sampling, or cached
outputs, and the completed pilot is not rerun. Existing journal rows remain
unchanged. New rows record the pacing setting, actual idle wait, and updated
harness hashes. The delay is excluded from the per-request HTTP timer but
included in elapsed campaign time. Since pacing can affect thermal/clock
conditions, timing analyses must distinguish operating modes rather than
treating mixed timings as a continuous maximum-throughput run. Full
checkpoint and restart details are in `CAMPAIGN-P5-P6-2026-09-19.md`.
