# Paper 5 evaluation data manifest

Staged 2026-09-01 on the frozen-campaign machine (no GPU, no inference; network
limited to huggingface.co + github.com). Design reference:
`research/design-overflow-policies.md` (frozen 2026-09-01).

All sha256 values below were computed locally with `sha256sum` and, for the
Hugging Face files, verified to match the server-declared Git-LFS sha256 oids
at the pinned revisions (exact match in all three cases).

## 1. LongMemEval-S (Wu et al., arXiv:2410.10813)

Upstream code repo: https://github.com/xiaowu0162/LongMemEval (data itself is
distributed via the HF Hub; the GitHub repo hosts code + a Drive mirror, which
is outside the allowed network policy, so the HF Hub copies were used).

**Deprecation notice (important):** the original HF dataset
`xiaowu0162/longmemeval` is marked deprecated by the author, replaced by
`xiaowu0162/longmemeval-cleaned`, which "removes noisy history sessions that
interfere with the answer correctness". Both S variants are staged so the
freeze/amendment can pick one explicitly. Same 500 question_ids in both, same
question_type distribution; the cleaned variant drops ~1-2 haystack sessions
per item (median 50 -> 48). License: MIT (per HF dataset card).

| file | source URL (pinned revision) | repo revision | bytes | sha256 |
|---|---|---|---|---|
| `longmemeval/longmemeval_s.json` | https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/2ec2a557f339b6c0369619b1ed5793734cc87533/longmemeval_s | `2ec2a557f339b6c0369619b1ed5793734cc87533` | 278,025,796 | `08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894` |
| `longmemeval/longmemeval_s_cleaned.json` | https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json | `98d7416c24c778c2fee6e6f3006e7a073259d48f` | 277,383,467 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `longmemeval/longmemeval_oracle.json` | https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/2ec2a557f339b6c0369619b1ed5793734cc87533/longmemeval_oracle | `2ec2a557f339b6c0369619b1ed5793734cc87533` | 15,388,478 | `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` |

Notes:
- In the original repo the S/M/oracle files are stored WITHOUT a `.json`
  extension (`longmemeval_s`); they are plain JSON and were renamed with the
  extension locally. Content is byte-identical to upstream (sha256 verified).
- The oracle file is byte-identical in the original and cleaned repos (same
  LFS sha256), so it was downloaded once from the original revision.
- **Skipped:** `longmemeval_m` (2,745,274,681 B) and
  `longmemeval_m_cleaned.json` (2,737,100,077 B) — each alone exceeds the
  2 GB budget for this staging task and the design uses only the S variant.

## 2. LoCoMo (Maharana et al., ACL 2024)

Source: `git clone https://github.com/snap-research/locomo` (shallow),
commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`
("Updated Readme", 2024-08-12 23:21:21 -0400, current HEAD of default branch).
Only the released conversation+QA file was staged. License: CC BY-NC 4.0
(repo `LICENSE.txt`) — note for the planned full per-item public release:
attribution + non-commercial terms apply to redistributed LoCoMo content.

| file | source URL | commit | bytes | sha256 |
|---|---|---|---|---|
| `locomo/locomo10.json` | https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json | `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` | 2,805,274 | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |

Total staged: 573,603,015 bytes (~547 MiB), well under the 2 GB budget.

---

## Schema notes (from local inspection, 2026-09-01)

### LongMemEval-S — both variants share the identical schema

Top-level: a JSON array of **500 items** (one per question). Per-item keys:

- `question_id` (str) — 30 ids end in `_abs`: abstention items where the
  correct behavior is to say the information is not present.
- `question_type` (str), distribution (identical in original and cleaned):
  `multi-session` 133, `temporal-reasoning` 133, `knowledge-update` 78,
  `single-session-user` 70, `single-session-assistant` 56,
  `single-session-preference` 30.
- `question` (str), `question_date` (str, e.g. `"2023/05/30 (Tue) 23:40"`).
- `answer` — str for 468 items, **int for 32 items** (harness must
  str-coerce before scoring).
- `haystack_sessions` — list of sessions; each session is a list of turns
  `{role: "user"|"assistant", content: str}`; evidence turns carry an extra
  `has_answer: true` flag.
- `haystack_session_ids`, `haystack_dates` — parallel lists (one id and one
  date string per session, e.g. `"2023/05/20 (Sat) 02:21"`).
- `answer_session_ids` — list of ids of the evidence sessions (subset of
  `haystack_session_ids`).

History sizes (original / cleaned nearly identical):
- Sessions per item: min 39 / 38, median 50 / 48, max 66 / 62.
- Turns per item: ~400-620 (median ~490).
- History chars per item: min ~455-462k, median ~490k, max ~514k
  => rough chars/4 estimate ~113k-128k tokens per item (median ~122k).
  Matches the paper's "~115k tokens" claim for the S variant. **Every item's
  full history exceeds 100k tokens** — see design implications below.

`longmemeval_oracle.json`: same 500 items and identical schema, but
`haystack_sessions` contains ONLY the evidence sessions (2-5 per item) —
useful for scoring sanity checks and debugging, not an arm.

Harness mapping: haystack = `haystack_sessions` (+ per-session
`haystack_dates` timestamps), question = `question` + `question_date`,
answer = `answer`, type = `question_type`, evidence = `answer_session_ids` +
`has_answer` turn flags. Scoring is QA-judge/exact-match per upstream repo
(github.com/xiaowu0162/LongMemEval `src/evaluation/`).

### LoCoMo — `locomo10.json`

Top-level: JSON array of **10 conversations**. Per-conversation keys:
`sample_id`, `conversation`, `qa`, plus annotation extras (`event_summary`,
`observation`, `session_summary`) that the harness does not need.

- `conversation` is a dict: `speaker_a`, `speaker_b`, then
  `session_1` ... `session_N` (N = 19-32 per conversation) with parallel
  `session_K_date_time` strings (e.g. `"1:56 pm on 8 May, 2023"`).
- Each session is a list of turns `{speaker, dia_id, text}`; some turns are
  multimodal image-sharing turns with extra `img_url`, `blip_caption`,
  `query` keys (77 image turns in conv 1 alone). A text-only harness must
  decide: substitute `blip_caption` for the image (what the upstream eval
  does) or drop image turns — freeze this in an amendment.
- `qa`: list of QA dicts, **1,986 questions total** across the 10
  conversations (105-199+ per conversation). Keys: `question`, `answer`
  (present on 1,542), `evidence` (list of `dia_id` refs like `"D1:3"`),
  `category` (int), and `adversarial_answer` on the 446 category-5 items
  (category 5 = adversarial/unanswerable; `answer` is absent on ~444 of
  them, so the harness must branch on category).
- Category distribution: 1 (multi-hop) 282, 2 (temporal) 321,
  3 (open-domain/commonsense) 96, 4 (single-hop) 841, 5 (adversarial) 446.
- Conversation sizes are much smaller than LongMemEval: ~44-90k chars
  (~11-22k est tokens) per full conversation.

### Format findings that affect the pre-registered design

1. **Two S variants exist.** The pre-registration names "LongMemEval-S"; the
   original is deprecated in favor of `longmemeval-cleaned` (noisy history
   sessions removed, answer-correctness fix). Both are staged and pinned.
   The choice must be frozen by amendment before any measurement; the
   cleaned variant is the author-recommended one, and question ids/counts
   are identical, so scoring comparability with older published numbers is
   the only argument for the original.
2. **500 questions confirmed** — matches the pre-registered "~500".
   30 are abstention (`_abs`) items; decide in the amendment whether the
   benchmark-native scoring treats them separately (upstream does).
3. **Histories are ~113-128k tokens per item (median ~122k).** This exceeds
   the ~100k flag in the staging brief and directly stresses arm O (full
   context oracle): an 8B model fully GPU-resident on the 12 GiB card
   cannot hold a ~125k-token KV cache. Arm O will need KV quantization,
   partial CPU KV offload, or an explicit documented cap below the full
   history — the design's "capped only by the model's context window" needs
   a VRAM-realistic amendment.
4. **LoCoMo has 1,986 questions, not ~200.** The pre-registered "~200
   subset" requires an explicit, frozen selection rule (e.g. category
   filter and/or seeded sample; note category 4 alone is 841 items and
   category 5 is adversarial with a different answer format).
5. **Mixed answer dtypes** (int/str) in LongMemEval and **missing `answer`
   on LoCoMo category-5** items are the only schema irregularities found.
6. **LoCoMo is CC BY-NC 4.0** — compatible with the study, but the planned
   full per-item public release must carry attribution and the NC term for
   the LoCoMo-derived portion (LongMemEval is MIT).
