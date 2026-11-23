"""P8 long context, stage 1: the registered ladder, limits and projections in one place.

Source: research/prereg/2026-11-01-p8-longctx-stage1.md (registered 2026-11-01 21:50 EST, commit c75ee2e) and its
Amendment 1. Every number here is copied from the registration; change none of them without a dated amendment.
"""

K = 1024
TOTAL_MIB = 11264                   # nvidia-smi memory.total on the desktop 2080 Ti
HEADROOM_MIB = 512
PASS_LIMIT_MIB = TOTAL_MIB - HEADROOM_MIB   # 10,752: a completed step passes at or below this peak
PROMPT_MARGIN = 512                 # each step sends one filled prompt of L - 512 tokens
DECODE_TOKENS = 64
HANG_S = 25 * 60                    # a step that hangs is stopped 25 minutes after its server starts
BUDGET_S = 90 * 60                  # at most 1.5 GPU hours for the whole stage
P1_BAND = (-150, 250)               # P1: measured peak within -150..+250 MiB of the projected range

# Registered order: 8,8, then 4,4, then 8,4, then fp16. Each format stops at its first failure.
LADDER = (
    ("8,8", (64 * K, 72 * K, 80 * K)),
    ("4,4", (64 * K, 96 * K, 128 * K, 136 * K)),
    ("8,4", (64 * K, 88 * K, 96 * K)),
    ("fp16", (32 * K, 36 * K, 40 * K)),
)

# Projected peak (MiB, low-high) and the registered prediction for each step, from fit_projection.py.
PROJECTED = {
    ("8,8", 64 * K): (10303, 10450, "pass"),
    ("8,8", 72 * K): (10592, 10739, "pass"),
    ("8,8", 80 * K): (10881, 11028, "fail"),
    ("8,4", 64 * K): (9759, 9906, "pass"),
    ("8,4", 88 * K): (10422, 10569, "pass"),
    ("8,4", 96 * K): (10643, 10790, "fail"),
    ("4,4", 64 * K): (9237, 9384, "pass"),
    ("4,4", 96 * K): (9849, 9996, "pass"),
    ("4,4", 128 * K): (10461, 10608, "pass"),
    ("4,4", 136 * K): (10614, 10761, "fail"),
    ("fp16", 32 * K): (10167, 10186, "pass"),
    ("fp16", 36 * K): (10439, 10474, "pass"),
    ("fp16", 40 * K): (10711, 10762, "fail"),
}

# TabbyAPI cache_mode strings (backends/exllamav3/model.py create_cache: "FP16", or "k,v" bits)
CACHE_MODE = {"8,8": "8,8", "8,4": "8,4", "4,4": "4,4", "fp16": "FP16"}

# The harness check of Amendment 1: the 8,8 server at 16K with a short prompt; recorded, never judged.
CHECK_STEP = ("8,8", 16 * K)
CHECK_PROMPT_TOKENS = 1024
CHECK_DECODE_TOKENS = 16


def step_id(fmt: str, length: int) -> str:
    return f"{fmt.replace(',', '')}-{length // K}k"


def lengths() -> list[int]:
    """Every prompt length the ladder needs, once each (one prompt per length, shared across formats)."""
    return sorted({length for _, ls in LADDER for length in ls})
