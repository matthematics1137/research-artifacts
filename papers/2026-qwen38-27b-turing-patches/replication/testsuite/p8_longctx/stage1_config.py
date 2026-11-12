"""P8 long context, stage 1: the TabbyAPI config of one ladder step (Amendment 1, item 3).

Lane exl3-mtp's config (tools/turing/tabby/exl3-mtp.yml, upstream TabbyAPI b1c2253 schema) with only the registered
changes: cache_size = max_seq_len = L; cache_mode per format, and the MTP layer's draft cache in the same format
(the anchors' loader gives the draft cache the main cache's bits, and the fit projection assumes it);
sysmem_recurrent_cache 2048 MB, TabbyAPI's spelling of census.py's -rcs 2. Everything else is the lane's: chunk 1024,
one slot, prompt checkpoints every 1,024 tokens, MTP drafting at 6 tokens with dynamic length, full draft head,
vision off, auth and request logging off. The allocator is left to exllamav3's default (no cuda_malloc_async), as in
the anchors.
"""
import stage1_spec as S

TEMPLATE = """\
# P8 long context stage 1, step {step} (research/prereg/2026-11-01-p8-longctx-stage1.md, Amendment 1, item 3).
# Lane exl3-mtp's config (tools/turing/tabby/exl3-mtp.yml) with this step's context, cache format and recurrent
# cache. Written by testsuite/p8_longctx/stage1_config.py for this one server start.
network:
  host: localhost
  port: {port}
  disable_auth: true
  allowed_origins: []
  web_ui: false
logging:
  log_prompt: false
  log_generation_params: false
  log_requests: false
model:
  model_dir: {model_dir}
  model_name: qwen38-exl3-2.0
  backend: exllamav3
  max_seq_len: {length}
  cache_size: {length}
  cache_mode: "{mode}"
  chunk_size: 1024
  max_batch_size: 1
  recurrent_checkpoint_interval_pp: 1024
  vision: false
draft_model:
  draft_mode: mtp
  draft_cache_mode: "{mode}"
  dynamic_draft: true
  draft_num_tokens: 6
  sparse_draft_head: 0
memory:
  sysmem_recurrent_cache: 2048
"""


def render(fmt: str, length: int, port: int, model_dir: str) -> str:
    if fmt not in S.CACHE_MODE:
        raise ValueError(f"unknown cache format {fmt!r}")
    if length % 256:
        raise ValueError(f"TabbyAPI needs a context that is a multiple of 256 (got {length})")
    return TEMPLATE.format(step=S.step_id(fmt, length), port=port, model_dir=model_dir, length=length,
                           mode=S.CACHE_MODE[fmt])


def check(text: str, fmt: str, length: int, port: int) -> list[str]:
    """The lines a step's config must carry; returns the missing ones (empty = ok)."""
    mode = S.CACHE_MODE[fmt]
    want = [f"  port: {port}", f"  max_seq_len: {length}", f"  cache_size: {length}", f'  cache_mode: "{mode}"',
            f'  draft_cache_mode: "{mode}"', "  max_batch_size: 1", "  chunk_size: 1024", "  draft_mode: mtp",
            "  draft_num_tokens: 6", "  dynamic_draft: true", "  sparse_draft_head: 0", "  vision: false",
            "  sysmem_recurrent_cache: 2048"]
    lines = set(text.splitlines())
    return [w for w in want if w not in lines]
