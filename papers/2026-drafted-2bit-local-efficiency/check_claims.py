#!/usr/bin/env python3
"""Re-derive Paper 7's reported numbers from the sanitized evidence bundle.

Analysis-only: standard library, no model, no GPU, no network. The checker
first verifies every exported file against ``data/EXPORT_MANIFEST.json`` (whose
own SHA-256 is pinned below), then recomputes the abstract, conclusion,
Table 1, Table 2, and hardening-block (H38, E47-E51) numbers from the
row-level data and compares each with the value printed in the paper at the
paper's own rounding. Any mismatch fails the run.

Figures that rest on campaign text or logs without a row-level derivation are
listed in CLAIM_EVIDENCE_MAP.md, not asserted here. Where the paper prints a
value that the rows do not reproduce exactly, the check asserts the
re-derived value and prints the paper's figure under "documented
discrepancies" so the difference stays visible.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "data").is_dir() else ROOT / "generated" / "data"
EXPORT_MANIFEST_SHA256 = "4f21e309febda5de3f73a19262a31070405e55fb9445a6f394658cef9c0e97ea"

FAILURES: list[str] = []
PASSED: list[str] = []
DISCREPANCIES: list[str] = []


# ----------------------------------------------------------------- utilities
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILURES).append(name if condition else f"{name}: {detail}")


def same(name: str, actual: Any, expected: Any) -> None:
    check(name, actual == expected, f"re-derived {actual!r}, paper {expected!r}")


def rounds_to(name: str, actual: float, printed: float, digits: int) -> None:
    """The paper printed `printed` with `digits` decimals; `actual` must round to it."""
    check(name, round(actual + 0.0, digits) == round(printed, digits) or
          abs(actual - printed) <= 0.5 * 10 ** (-digits) + 1e-9,
          f"re-derived {actual:.6g}, paper {printed}")


def pct(name: str, fraction: float, printed: float, digits: int = 1) -> None:
    rounds_to(name, 100.0 * fraction, printed, digits)


def load_json(relative: str) -> Any:
    return json.loads((DATA / relative).read_text(encoding="utf-8"))


def load_jsonl(relative: str) -> list[dict[str, Any]]:
    with (DATA / relative).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def items(label: str, suite: str) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in load_jsonl(f"items/{label}/{suite}.jsonl")}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def binomial_two_sided(k: int, n: int) -> float:
    return mcnemar_exact(k, n - k)


ORDER = None


def order(suite: str) -> list[str]:
    global ORDER
    if ORDER is None:
        ORDER = load_json("dataset_item_order.json")
    return ORDER[suite]


def paired(label_a: str, label_b: str, suite: str, limit: int = 0) -> dict[str, Any]:
    """The hardening_analyze.py estimands on one pair of labels."""
    a, b = items(label_a, suite), items(label_b, suite)
    ids = [i for i in order(suite) if i in a and i in b]
    if limit:
        ids = [i for i in order(suite)[:limit] if i in a and i in b]
    ok = [i for i in ids if not a[i]["errored"] and not b[i]["errored"]]
    n = len(ok)
    acc_a = sum(1 for i in ok if a[i]["correct"])
    acc_b = sum(1 for i in ok if b[i]["correct"])
    a_only = sum(1 for i in ok if a[i]["correct"] and not b[i]["correct"])
    b_only = sum(1 for i in ok if b[i]["correct"] and not a[i]["correct"])
    identical = [i for i in ok if a[i].get("content_sha256") and a[i]["content_sha256"] == b[i].get("content_sha256")]
    same_tokens = sum(1 for i in ok if a[i]["completion_tokens"] == b[i]["completion_tokens"])
    return {
        "a": a, "b": b, "ids": ok, "n": n, "acc_a": acc_a, "acc_b": acc_b, "a_only": a_only, "b_only": b_only,
        "p": mcnemar_exact(a_only, b_only), "identical": identical, "k_identical": len(identical),
        "token_identity": same_tokens / n,
        "mean_tok_a": statistics.mean(a[i]["completion_tokens"] for i in ok),
        "mean_tok_b": statistics.mean(b[i]["completion_tokens"] for i in ok),
        "wall_a": statistics.mean(a[i]["wall_s"] for i in ok),
        "wall_b": statistics.mean(b[i]["wall_s"] for i in ok),
        "trunc_a": sum(1 for i in ok if a[i]["truncated"]),
        "trunc_b": sum(1 for i in ok if b[i]["truncated"]),
    }


def ci_pct(k: int, n: int) -> list[float]:
    lo, hi = wilson(k, n)
    return [round(100 * lo, 1), round(100 * hi, 1)]


# ---------------------------------------------------------------- integrity
def verify_integrity() -> None:
    manifest_path = DATA / "EXPORT_MANIFEST.json"
    if sha256(manifest_path) != EXPORT_MANIFEST_SHA256:
        raise SystemExit("changed or missing evidence: data/EXPORT_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        if not name.startswith("data/"):
            continue  # replication sources are checked by their own digests below
        path = DATA / name.removeprefix("data/")
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise SystemExit(f"changed or missing evidence: {name}")
    replication = ROOT / "replication" if (ROOT / "replication").is_dir() else ROOT / "generated" / "replication"
    if replication.is_dir():
        for name, record in manifest["files"].items():
            if name.startswith("replication/"):
                path = replication / name.removeprefix("replication/")
                if not path.is_file() or sha256(path) != record["sha256"]:
                    raise SystemExit(f"changed or missing replication source: {name}")
    check("integrity: every exported file matches EXPORT_MANIFEST.json", True)


# ------------------------------------------------------------- E7 (Table A1)
def check_e7() -> None:
    rows = load_jsonl("rows/e7_rows.jsonl")

    def mean_rate(label: str, prompt: str) -> float:
        return statistics.mean(r["e2e_tok_s"] for r in rows if r["label"] == label and r["prompt"] == prompt)

    def joules_per_token(label: str, prompt: str) -> float:
        sel = [r for r in rows if r["label"] == label and r["prompt"] == prompt]
        return statistics.mean(r["gpu"]["power_w_mean"] for r in sel) / statistics.mean(r["e2e_tok_s"] for r in sel)

    undrafted = {p: mean_rate("exl3_147_nodraft", p) for p in ("code", "math", "prose", "rewrite")}
    drafted = {p: mean_rate("exl3_147_mtpdyn", p) for p in ("code", "math", "prose", "rewrite")}
    fixed3 = {p: mean_rate("exl3_147_mtp3", p) for p in ("code", "math", "prose", "rewrite")}
    for p, v in zip(("code", "math", "prose", "rewrite"), (29.2, 29.0, 29.3, 28.6)):
        rounds_to(f"E7 undrafted 1.4.7 {p} tok/s", undrafted[p], v, 1)
    for p, v in zip(("code", "math", "prose", "rewrite"), (48.7, 50.4, 35.7, 49.5)):
        rounds_to(f"E7 MTP dynamic {p} tok/s (abstract, conclusion)", drafted[p], v, 1)
    for p, v in zip(("code", "math", "prose", "rewrite"), (47.5, 49.6, 34.7, 49.5)):
        rounds_to(f"E7 MTP fixed-3 {p} tok/s", fixed3[p], v, 1)
    same("E7 drafted rows are means over two sessions (4 rows each)",
         sorted({len([r for r in rows if r["label"] == "exl3_147_mtpdyn" and r["prompt"] == p]) for p in drafted}), [4])
    rounds_to("E7 undrafted 'from 29' (min)", min(undrafted[p] for p in ("code", "math", "prose")), 29.0, 0)
    for label, pairs in (("exl3_147_nodraft", ((2.73, "code"), (2.72, "prose"))),
                         ("exl3_147_mtpdyn", ((1.64, "code"), (2.24, "prose"))),
                         ("exl3_147_mtp3", ((1.68, "code"), (2.30, "prose")))):
        for printed, p in pairs:
            rounds_to(f"E7 J/token {label} {p}", joules_per_token(label, p), printed, 2)
    ratio_code = drafted["code"] / undrafted["code"]
    ratio_math = drafted["math"] / undrafted["math"]
    ratio_prose = drafted["prose"] / undrafted["prose"]
    rounds_to("E7 prose speed-up 1.2x (title, 3.1)", ratio_prose, 1.2, 1)
    rounds_to("title 1.2-1.7x more tokens per joule: lower end (prose)", ratio_prose, 1.2, 1)
    rounds_to("title 1.2-1.7x more tokens per joule: upper end (math)", max(ratio_code, ratio_math), 1.7, 1)
    check("E7 code/math speed-up as printed, 1.67-1.74x (corrected 2026-09-19)", (round(ratio_code, 2), round(ratio_math, 2)) == (1.67, 1.74),
          f"{ratio_code:.3f}, {ratio_math:.3f}")


# --------------------------------------------------------- E25 (Table 2 top)
def check_e25() -> None:
    rows = load_jsonl("rows/tpj_rows.jsonl")
    summaries = {r["label"]: r for r in rows if r.get("phase") == "summary"}
    requests = [r for r in rows if "req" in r and "tokens" in r]
    expected = {
        "exl3_mtpdyn12k_30min": dict(minutes=30.4, tokens=81331, tok_s=44.6, mean_w=79.8, idle_w=16.4,
                                     gross=0.559, net=0.704, gpu="rows/tpj_gpu_exl3_mtpdyn12k_30min.jsonl",
                                     capped=98, sm=1215, cmax=64),
        "iq2s_mtp3_15min": dict(minutes=15.1, tokens=41437, tok_s=45.8, mean_w=79.7, idle_w=17.6,
                                gross=0.577, net=0.740, gpu="rows/tpj_gpu_iq2s_mtp3_15min.jsonl",
                                capped=99, sm=1380, cmax=63),
    }
    for label, e in expected.items():
        s = summaries[label]
        req = [r for r in requests if r["label"] == label]
        same(f"E25 {label} tokens = sum of per-request rows", sum(r["tokens"] for r in req), e["tokens"])
        same(f"E25 {label} tokens (Table 2)", s["tokens"], e["tokens"])
        rounds_to(f"E25 {label} minutes", s["minutes"], e["minutes"], 1)
        rounds_to(f"E25 {label} tok/s incl. prefill and gaps", s["tok_s"], e["tok_s"], 1)
        rounds_to(f"E25 {label} mean W", s["mean_w"], e["mean_w"], 1)
        rounds_to(f"E25 {label} idle W", s["idle_w"], e["idle_w"], 1)
        rounds_to(f"E25 {label} tok/J gross = tokens / energy", s["tokens"] / (1000 * s["energy_kj"]), e["gross"], 3)
        rounds_to(f"E25 {label} tok/J net (recorded)", s["tok_per_j_net"], e["net"], 3)
        net_from_minutes = s["tokens"] / (1000 * s["energy_kj"] - s["idle_w"] * 60 * s["minutes"])
        check(f"E25 {label} tok/J net consistent with energy, idle and minutes (+/-0.002)",
              abs(net_from_minutes - e["net"]) <= 0.002, f"{net_from_minutes:.4f}")
        gpu = load_jsonl(e["gpu"])
        # "Capped": samples whose throttle reason is the software power cap (0x4) alone, over all samples.
        capped = sum(1 for g in gpu if int(g["throttle"], 16) == 0x4) / len(gpu)
        running = [g for g in gpu if g["util"] > 0]
        capped_running = sum(1 for g in running if int(g["throttle"], 16) == 0x4) / len(running)
        check(f"E25 {label} capped share of all samples inside the text's 96-98%", 0.955 <= capped <= 0.985, f"{capped:.4f}")
        check(f"E25 {label} over 99% of samples once generation started are capped", capped_running > 0.99, f"{capped_running:.4f}")
        if label.startswith("exl3"):
            rounds_to(f"E25 {label} Table 2 'Capped' 98%", 100 * capped, e["capped"], 0)
        else:
            rounds_to(f"E25 {label} Table 2 'Capped' 96% (all samples, corrected 2026-09-19)", 100 * capped, 96, 0)
        run = [g for g in gpu if g["util"] > 0 or int(g["throttle"], 16) & 0x4]
        sm_median = statistics.median(g["sm_mhz"] for g in gpu if int(g["throttle"], 16) & 0x4)
        check(f"E25 {label} SM clock median under the cap ~{e['sm']} MHz (+/-10)", abs(sm_median - e["sm"]) <= 10,
              f"{sm_median}")
        same(f"E25 {label} GPU max temperature", max(g["gpu_c"] for g in gpu), e["cmax"])
        check(f"E25 {label} run samples exist", len(run) > 0)
    for prompt, (exl3, iq2s) in {"math": (0.75, 0.75), "rewrite": (0.73, 0.75), "code": (0.64, 0.65), "prose": (0.50, 0.52)}.items():
        for label, printed in (("exl3_mtpdyn12k_30min", exl3), ("iq2s_mtp3_15min", iq2s)):
            sel = [r["tok_per_j"] for r in requests if r["label"] == label and r["prompt"] == prompt]
            rounds_to(f"E25 per-prompt tok/J {label} {prompt} (request mean)", statistics.mean(sel), printed, 2)
    gross = sorted(round(s["tokens"] / (1000 * s["energy_kj"]), 2) for s in summaries.values())
    same("abstract 0.56-0.58 tokens per joule (E25)", gross, [0.56, 0.58])
    exl3 = summaries["exl3_mtpdyn12k_30min"]
    windows = [w["tok_s"] for w in exl3["windows"]][:6]
    same("E25 EXL3 cold-start step 46.6 -> 44.2 tok/s (first vs last 5-min window)", (windows[0], windows[-1]), (46.6, 44.2))


# --------------------------------------------------------- E29 (Table 2 low)
def check_e29() -> None:
    rows = load_jsonl("rows/e29_rows.jsonl")
    summaries = {r["label"]: r for r in rows if r.get("phase") == "summary"}
    table = {  # label: (clients, aggregate, per-stream, mean W, tok/J)
        "e29_mtp_c1": (1, 43.0, 43.7, 79.6, 0.54),
        "e29_mtp_c2": (2, 68.5, 35.6, 79.5, 0.86),
        "e29_mtp_c4": (4, 91.0, 23.5, 79.7, 1.15),
        "e29_nomtp_c1": (1, 28.9, 28.9, 79.7, 0.36),
        "e29_nomtp_c4": (4, 75.6, 19.1, 79.7, 0.95),
    }
    for label, (clients, aggregate, per_stream, watts, tpj) in table.items():
        s = summaries.get(label)
        if s is None:
            check(f"E29 summary row {label}", False, f"missing; have {sorted(summaries)}")
            continue
        same(f"E29 {label} clients", s["concurrency"], clients)
        rounds_to(f"E29 {label} aggregate tok/s", s["tok_s"], aggregate, 1)
        rounds_to(f"E29 {label} per-stream tok/s", s["per_stream_tok_s"] if not isinstance(s["per_stream_tok_s"], list)
                  else statistics.mean(s["per_stream_tok_s"]), per_stream, 1)
        rounds_to(f"E29 {label} mean W", s["mean_w"], watts, 1)
        rounds_to(f"E29 {label} tok/J", s["tok_per_j_gross"], tpj, 2)
    mtp1 = summaries["e29_mtp_c1"]["tok_s"]
    rounds_to("E29 x1.59 at two drafted clients", summaries["e29_mtp_c2"]["tok_s"] / mtp1, 1.59, 2)
    rounds_to("E29 x2.12 at four drafted clients", summaries["e29_mtp_c4"]["tok_s"] / mtp1, 2.12, 2)
    rounds_to("E29 x2.61 at four undrafted clients",
              summaries["e29_nomtp_c4"]["tok_s"] / summaries["e29_nomtp_c1"]["tok_s"], 2.61, 2)


# ------------------------------------------------ Table 1 and E27/E28 VRAM
def vram_lines() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for entry in load_json("logs/vram_log_extracts.json"):
        out.setdefault(entry["source"].rsplit("/", 1)[1], []).append(entry["log_line"])
    return out


def fit(lines: list[str], name: str) -> tuple[int, int | None]:
    for line in lines:
        m = re.match(rf"FIT {re.escape(name)} load=(\d+)MiB(?: warm=(\d+)MiB)?$", line)
        if m:
            return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)
    raise KeyError(name)


def check_table1_and_vram() -> None:
    audit = {e["local_name"].rsplit("/", 1)[-1]: e for e in load_json("hash_audit_2026-09-06.json")
             if not e["local_name"].endswith(".safetensors")}
    shards = [e for e in load_json("hash_audit_2026-09-06.json") if e["local_name"].endswith(".safetensors")]
    same("hash audit: all 11 artifacts MATCH the Hub", (len(audit) + len(shards), {e["status"] for e in [*audit.values(), *shards]}), (11, {"MATCH"}))
    exl3_20 = [e for e in shards if any(h["branch"] == "SC_2.00bpw_H3" for h in e["hub"])]
    exl3_22 = [e for e in shards if any(h["branch"] == "SC_2.20bpw_H3_V3" for h in e["hub"])]
    same("Table 1 EXL3 SC_2.00bpw_H3 bytes (two shards)", sum(e["bytes"] for e in exl3_20), 10_196_126_231)
    same("Table 1 EXL3 SC_2.20bpw_H3_V3 bytes (two shards)", sum(e["bytes"] for e in exl3_22), 10_280_960_404)
    gguf = {"Qwen3.8-27B-UD-IQ2_S.gguf": (8_371_970_048, 2.49), "Qwen3.8-27B-UD-IQ2_XXS.gguf": (7_266_070_528, 2.16),
            "Qwen3.8-27B-UD-Q4_K_XL.gguf": (17_559_178_144, 5.22)}
    params = []
    for name, (size, bpw) in gguf.items():
        same(f"Table 1 bytes {name}", audit[name]["bytes"], size)
        params.append(size * 8 / bpw)
    check("Table 1 GGUF eff. bpw = whole-file bits per parameter at one common count (~26.9e9, +/-0.5%)",
          max(params) / min(params) < 1.005 and 26.8e9 < statistics.mean(params) < 27.0e9,
          f"{[round(p / 1e9, 3) for p in params]}")
    same("Table 1 MTP sidecar bytes", audit["mtp-Qwen3.8-27B-Q4_0.gguf"]["bytes"], 1_369_590_656)
    same("Table 1 DFlash2 drafter bytes", audit["Qwen3.8-27B-DFlash2-Q4_K_M.gguf"]["bytes"], 1_143_006_816)
    sidecar_gib = audit["mtp-Qwen3.8-27B-Q4_0.gguf"]["bytes"] / 2 ** 30
    rounds_to("Section 2 sidecar 1.28 GiB (corrected 2026-09-19)", sidecar_gib, 1.28, 2)

    logs = vram_lines()
    load_nomtp, warm_nomtp = fit(logs["e27.out"], "nomtp_12k")
    load_mtp, warm_mtp = fit(logs["e27.out"], "mtp_12k")
    load_mtp8, warm_mtp8 = fit(logs["e27.out"], "mtp_8k")
    load_fp16, warm_fp16 = fit(logs["e27.out"], "mtp_12k_fp16draft")
    same("E27 Table A4 load/warm rows", [(load_nomtp, warm_nomtp), (load_mtp, warm_mtp), (load_mtp8, warm_mtp8), (load_fp16, warm_fp16)],
         [(7794, 7834), (10226, 10270), (10098, 10142), (10258, 10302)])
    same("abstract: the drafter's 2,432 MiB (E27: 10,226 - 7,794)", load_mtp - load_nomtp, 2432)
    load_mbs1, _ = fit(logs["e28.out"], "mbs1_12k")
    load_default, _ = fit(logs["e28.out"], "default_12k")
    load_32k, _ = fit(logs["e28.out"], "mbs1_32k")
    same("abstract/conclusion: one config line 10,226 -> 8,050 MiB (E28)", (load_default, load_mbs1), (10226, 8050))
    same("E28 2,176 MiB reclaimed", load_default - load_mbs1, 2176)
    same("E28 32K drafted with max_batch_size 1 loads at 8,754 MiB", load_32k, 8754)
    same("E28 warm 8,468 MiB and 9,588 MiB after a 30K prompt",
         [int(re.search(r"(\d+)MiB$", l).group(1)) for l in logs["e28.out"] if "warm vram" in l][:1] +
         [int(re.search(r"(\d+)MiB$", l).group(1)) for l in logs["e28.out"] if "30K probe" in l], [8468, 9588])
    mib = 1024 * 1024
    per_layer_step = 48 * 128 * 128 * 4
    same("E28 recurrent state: 48 x 128 x 128 fp32 = 3 MiB per layer and history step", per_layer_step, 3 * mib)
    same("E28 4 slots x (4+1) history x 48 layers x 3 MiB = 2,880 MiB drafted; 576 MiB undrafted",
         (4 * 5 * 48 * per_layer_step // mib, 4 * 1 * 48 * per_layer_step // mib), (2880, 576))
    same("E28 ~148 MiB per sequence (48 layers x 3 MiB = 144 MiB of state, plus conv history)",
         48 * per_layer_step // mib, 144)
    reserved = {}
    for name, lines in logs.items():
        if name.startswith("memsnap"):
            reserved[name] = [int(re.search(r"reserved (\d+) MiB", l).group(1)) for l in lines if "reserved" in l]
    same("E28 bare API torch reserved, no draft 1 slot (after generate, as printed; corrected 2026-09-19)", reserved["memsnap_none_1_0.out"][-1], 7260)
    same("E28 bare API MTP 1 slot / MTP 2 slots / no draft 4 slots (after generate)",
         (reserved["memsnap_mtp.out"][-1], reserved["memsnap_mtp_2_4.out"][-1], reserved["memsnap_none_4_0.out"][-1]),
         (8050, 8732, 7914))
    same("E28 MTP 4 slots does not fit the bare API's split", sum("Insufficient VRAM" in l for l in logs["memsnap_mtp_4_4.out"]) > 0, True)
    e23c = [int(re.search(r"(\d+) MiB", l).group(1)) for l in logs["e23c.out"]]
    same("Table 1 SC_2.20bpw resident 8,242 MiB at load (E23c)", e23c[0], 8242)
    fp = {r["label"]: r for r in load_jsonl("rows/footprint_rows.jsonl")}
    same("Table 1 UD-IQ2_S resident 8,356 MiB (E3)", fp["iq2s_none_ub512"]["vram_load_mib"], 8356)
    xxs = [r for r in fp.values() if r["label"].startswith("iq2xxs_none")]
    same("Table 1 UD-IQ2_XXS resident 7,248 MiB (E3)", [r["vram_load_mib"] for r in xxs], [7248])
    mtp_side = [r for r in fp.values() if r["label"].startswith("iq2s_mtp") and r.get("vram_load_mib") == 9480]
    same("Table 1 MTP sidecar resident 1,124 MiB = 9,480 - 8,356 (E3)", (9480 - 8356) if mtp_side else None, 1124)
    dflash = [r for r in fp.values() if r["label"].startswith("iq2xxs_dflash") and r.get("vram_load_mib")]
    same("Table 1 DFlash2 resident 2,462 MiB = 9,710 - 7,248 (E3)",
         [r["vram_load_mib"] - 7248 for r in dflash], [2462])


# ---------------------------------------------------------------- E35, E37
def check_e35_e37() -> None:
    rows = load_jsonl("rows/spec_meas.jsonl")
    same("E35 drafted positions", len(rows), 9821)
    ranges = {}
    for line in vram_lines()["b6.out"]:
        m = re.match(r"RANGE (\w+):(\d+):(\d+)$", line)
        ranges[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    table = {"prose": (2708, 0.541, 0.539, 0.688, 0.557, 0.018), "code": (2797, 0.705, 0.699, 0.816, 0.685, -0.015),
             "math": (2544, 0.732, 0.726, 0.831, 0.715, -0.011), "rewrite": (1772, 0.738, 0.737, 0.814, 0.702, -0.035)}
    for name, (n, acc, p_star, q_star, sum_min, gain) in table.items():
        lo, hi = ranges[name]
        r = rows[lo:hi]
        same(f"E35 {name} positions", len(r), n)
        rounds_to(f"E35 {name} observed acceptance", statistics.mean(x["acc"] for x in r), acc, 3)
        rounds_to(f"E35 {name} p(x*)", statistics.mean(x["p_star"] for x in r), p_star, 3)
        rounds_to(f"E35 {name} q(x*)", statistics.mean(x["q_star"] for x in r), q_star, 3)
        rounds_to(f"E35 {name} sum min(p,q)", statistics.mean(x["sum_min"] for x in r), sum_min, 3)
        rounds_to(f"E35 {name} predicted gain", statistics.mean(x["sum_min"] for x in r) - statistics.mean(x["p_star"] for x in r), gain, 3)
        gap = abs(statistics.mean(x["acc"] for x in r) - statistics.mean(x["p_star"] for x in r))
        check(f"E35 {name}: p(x*) predicts observed acceptance within 0.007", gap < 0.007, f"{gap:.4f}")
    by_pos = {}
    for x in rows:
        by_pos.setdefault(x["i"], []).append(x["p_star"])
    same("E35 decay by draft position 0.77/0.66/0.53/0.37 (mean p(x*))",
         [round(statistics.mean(by_pos[i]), 2) for i in sorted(by_pos)], [0.77, 0.66, 0.53, 0.37])
    sampled = load_jsonl("rows/spec_meas_b7_sampled.jsonl")
    same("E36 positions under the sampled rule", len(sampled), 19074)
    rounds_to("E36 acceptance under the sampled rule", statistics.mean(x["acc"] for x in sampled), 0.592, 3)
    rounds_to("E36 that run's sum min(p,q)", statistics.mean(x["sum_min"] for x in sampled), 0.597, 3)

    profile = {r["step"]: r for r in load_jsonl("rows/prefill_profile.jsonl") if r["step"] in ("profile",)}["profile"]
    buckets = profile["buckets"]
    pct("E37 prefill GPU time in cuBLAS fp16 GEMM (abstract 74%)", buckets["gemm"], 73.8)
    pct("E37 trellis-to-fp16 reconstruction", buckets["reconstruct"], 10.7)
    pct("E37 copies", buckets["copies"], 4.8)
    pct("E37 Gated-DeltaNet chunk kernels", buckets["gdn"], 3.5)
    pct("E37 paged attention", buckets["attention"], 2.9)
    rounds_to("E37 remainder 'about 4% norms and miscellany'",
              100 * (1 - sum(buckets[k] for k in ("gemm", "reconstruct", "copies", "gdn", "attention"))), 4.0, 0)
    chunks = {r["chunk"]: statistics.mean([r["tok_s"], r["tok_s_rep2"]]) for r in load_jsonl("rows/prefill_profile.jsonl") if r["step"] == "chunk"}
    same("E37 prefill tok/s at chunk 512 / 1,024 / 2,048 (535 / 583 / 592)",
         (round(chunks[512]), round(profile["tok_s"]), round(chunks[2048])), (536, 583, 592))


# --------------------------------------------------------------------- E40
def check_e40() -> None:
    stats = load_jsonl("traces/e40_trace_stats.jsonl")
    pairs = load_jsonl("traces/e40_pairs.jsonl")

    def sel(label: str, suite: str) -> dict[str, dict[str, Any]]:
        return {r["id"]: r for r in stats if r["label"] == label and r["suite"] == suite}

    q_m, e_m = sel("q4kxl", "math25"), sel("exl3_20", "math25")
    q_h, e_h = sel("q4kxl", "humaneval_plus"), sel("exl3_20", "humaneval_plus")
    same("E40 paired items (25 math25, 20 HumanEval+)", (len(set(q_m) & set(e_m)), len(set(q_h) & set(e_h))), (25, 20))
    ids = sorted(set(q_m) & set(e_m))
    mean_q = statistics.mean(q_m[i]["completion_tokens"] for i in ids)
    mean_e = statistics.mean(e_m[i]["completion_tokens"] for i in ids)
    rounds_to("E40 math25 mean completion tokens Q4_K_XL", mean_q, 1242, 0)
    rounds_to("E40 math25 mean completion tokens EXL3 2.00", mean_e, 1592, 0)
    rounds_to("abstract/conclusion: 2-bit reasons 1.28x longer on 25 MATH items (E40)", mean_e / mean_q, 1.28, 2)
    rounds_to("E40 median per-item ratio 1.16",
              statistics.median(e_m[i]["completion_tokens"] / max(1, q_m[i]["completion_tokens"]) for i in ids), 1.16, 2)
    mq = statistics.mean(q_m[i]["n_markers"] for i in ids)
    me = statistics.mean(e_m[i]["n_markers"] for i in ids)
    rounds_to("E40 'Wait'-class transitions per math trace, Q4", mq, 2.0, 1)
    rounds_to("E40 'Wait'-class transitions per math trace, 2-bit", me, 4.7, 1)
    rounds_to("E40 transitions ratio 2.4x", me / mq, 2.4, 1)
    same("E40 explicit self-corrections (math totals) 1 vs 4",
         (sum(q_m[i]["n_corrections"] for i in ids), sum(e_m[i]["n_corrections"] for i in ids)), (1, 4))
    pct("E40 tokens chosen with p < 0.5, Q4", statistics.mean(q_m[i]["low_conf_share"] for i in ids), 2.3)
    pct("E40 tokens chosen with p < 0.5, 2-bit", statistics.mean(e_m[i]["low_conf_share"] for i in ids), 1.4)
    rounds_to("E40 transition-token log-probability median, Q4",
              statistics.median(v for i in ids for v in q_m[i]["marker_logprobs"]), -0.22, 2)
    rounds_to("E40 transition-token log-probability median, 2-bit",
              statistics.median(v for i in ids for v in e_m[i]["marker_logprobs"]), -0.17, 2)
    rounds_to("E40 math25 reasoning tokens (estimate), Q4", statistics.mean(q_m[i]["reason_tokens_est"] or 0 for i in ids), 840, 0)
    rounds_to("E40 math25 reasoning tokens (estimate), 2-bit", statistics.mean(e_m[i]["reason_tokens_est"] or 0 for i in ids), 1033, 0)
    hids = sorted(set(q_h) & set(e_h))
    rounds_to("E40 HumanEval+ mean tokens Q4", statistics.mean(q_h[i]["completion_tokens"] for i in hids), 663, 0)
    rounds_to("E40 HumanEval+ mean tokens 2-bit", statistics.mean(e_h[i]["completion_tokens"] for i in hids), 647, 0)
    he_ratio = statistics.mean(e_h[i]["completion_tokens"] for i in hids) / statistics.mean(q_h[i]["completion_tokens"] for i in hids)
    rounds_to("E40 HumanEval+ ratio of means 0.98 (Table A10, corrected 2026-09-19)", he_ratio, 0.98, 2)
    rounds_to("E40 HumanEval+ median per-item ratio 0.97",
              statistics.median(e_h[i]["completion_tokens"] / q_h[i]["completion_tokens"] for i in hids), 0.97, 2)
    rounds_to("E40 HumanEval+ transitions per trace Q4", statistics.mean(q_h[i]["n_markers"] for i in hids), 1.95, 2)
    rounds_to("E40 HumanEval+ transitions per trace 2-bit", statistics.mean(e_h[i]["n_markers"] for i in hids), 1.50, 2)
    div = [p["first_divergence_tokens"] for p in pairs if p["suite"] == "math25"]
    same("E40 median two matching tokens on math items", statistics.median(div), 2)
    same("E40 divergence within the first 30 tokens on 24 of 25 math items", sum(1 for d in div if d < 30), 24)


# ------------------------------------------------ levers (E26, E32-E45)
def mean_tokens(label: str, suite: str, ids: list[str] | None = None) -> float:
    rows = items(label, suite)
    keys = ids if ids is not None else list(rows)
    return statistics.mean(rows[i]["completion_tokens"] for i in keys)


def pair_counts(label_a: str, label_b: str, suite: str) -> tuple[int, int, int, int, list[str]]:
    a, b = items(label_a, suite), items(label_b, suite)
    ids = [i for i in a if i in b]
    return (sum(1 for i in ids if a[i]["correct"]), sum(1 for i in ids if b[i]["correct"]),
            sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"]),
            sum(1 for i in ids if b[i]["correct"] and not a[i]["correct"]), ids)


def token_change(label_ref: str, label_new: str, suite: str) -> float:
    ids = [i for i in items(label_ref, suite) if i in items(label_new, suite)]
    return mean_tokens(label_new, suite, ids) / mean_tokens(label_ref, suite, ids) - 1


def check_levers() -> None:
    # E26 noise floor: same-config re-roll of GSM8K-100.
    acc_a, acc_b, lost, gained, _ = pair_counts("EFF-exl3_147_mtpdyn", "EFF-exl3_147_mtpdyn-r2", "gsm8k")
    same("E26 noise floor: 98 -> 95, 3 lost / 0 gained", (acc_a, acc_b, lost, gained), (98, 95, 3, 0))
    rounds_to("E26 exact McNemar p = 0.25", mcnemar_exact(lost, gained), 0.25, 2)
    # E42: off-policy against self-calibrated (E32 conversion).
    pooled_self, pooled_off = 0, 0
    for suite, (acc_self, acc_off, d_self, d_off, tok) in {"humaneval_plus": (18, 17, 1, 0, -1.5), "math25": (18, 15, 3, 0, 6.3),
                                                         "gsm8k": (96, 92, 5, 1, 15.6)}.items():
        a, b, a_only, b_only, _ = pair_counts("EFF-exl3_20cot_147_nomtp8k", "EFF-exl3_20off_147_nomtp8k", suite)
        same(f"E42 {suite} accuracy self vs off and discordant", (a, b, a_only, b_only), (acc_self, acc_off, d_self, d_off))
        change = token_change("EFF-exl3_20cot_147_nomtp8k", "EFF-exl3_20off_147_nomtp8k", suite)
        if suite == "gsm8k":
            pct("E42 GSM8K completion tokens change +15.6% (corrected 2026-09-19)", change, 15.6)
        else:
            pct(f"E42 {suite} completion tokens change", change, tok)
        pooled_self += a_only
        pooled_off += b_only
    same("E42 pooled discordant 9 to 1", (pooled_self, pooled_off), (9, 1))
    rounds_to("E42 exact binomial p = 0.021", binomial_two_sided(pooled_off, pooled_self + pooled_off), 0.021, 3)
    trunc = [sum(1 for r in items(label, "math25").values() if r["truncated"])
             for label in ("EFF-exl3_20cot_147_nomtp8k", "EFF-exl3_20off_147_nomtp8k")]
    same("E42 math25 cap truncations 1 -> 5", trunc, [1, 5])
    rounds_to("E42 GSM8K-100 exact McNemar p = 0.22", mcnemar_exact(5, 1), 0.22, 2)
    # E41: variant A against the E32 conversion.
    for suite, (acc_stock, acc_var, d_stock, d_var, tok) in {"humaneval_plus": (18, 17, 2, 1, -7), "math25": (18, 15, 3, 0, 2),
                                                            "gsm8k": (96, 98, 0, 2, 3)}.items():
        a, b, a_only, b_only, _ = pair_counts("EFF-exl3_20cot_147_nomtp8k", "EFF-exl3_20varA_147_nomtp8k", suite)
        same(f"E41 {suite} accuracy stock recipe vs variant A and discordant", (a, b, a_only, b_only), (acc_stock, acc_var, d_stock, d_var))
        pct(f"E41 {suite} completion tokens change", token_change("EFF-exl3_20cot_147_nomtp8k", "EFF-exl3_20varA_147_nomtp8k", suite), tok, 0)
    net = sum(pair_counts("EFF-exl3_20cot_147_nomtp8k", "EFF-exl3_20varA_147_nomtp8k", s)[1] -
              pair_counts("EFF-exl3_20cot_147_nomtp8k", "EFF-exl3_20varA_147_nomtp8k", s)[0] for s in ("humaneval_plus", "math25", "gsm8k"))
    same("E41 net -2 items over 145 paired", net, -2)
    # E32: own-trace recalibration against the stock quant (cross-day comparators).
    for suite, ref, printed in (("humaneval_plus", "EFF-exl3_147_mtp4", 1.5), ("math25", "EFF-exl3_147_mtp4", -1),
                                ("gsm8k", "EFF-exl3_147_mtpdyn", -3), ("gsm8k", "EFF-exl3_147_mtpdyn-r2", -6)):
        change = token_change(ref, "EFF-exl3_20cot_147_nomtp8k", suite)
        pct(f"E32 {suite} tokens vs {ref}", change, printed, 1 if printed == 1.5 else 0)
        check(f"E32 {suite} tokens within +/-6%", abs(change) <= 0.06 + 1e-9, f"{change:.4f}")
    same("E32 accuracy HE+ / math25 / GSM8K (18, 18, 96)",
         tuple(sum(1 for r in items("EFF-exl3_20cot_147_nomtp8k", s).values() if r["correct"]) for s in ("humaneval_plus", "math25", "gsm8k")),
         (18, 18, 96))
    # E45: marker-penalty strength on the EXL3 lane.
    base = "EFF-b10_base"
    same("E45 baseline math25 19/25 and GSM8K 98",
         (sum(r["correct"] for r in items(base, "math25").values()), sum(r["correct"] for r in items(base, "gsm8k").values())), (19, 98))
    for strength, printed in (("0.5", -7.5), ("1.0", -4.6), ("2.0", -5.0), ("3.0", -4.8)):
        pct(f"E45 math25 tokens at -{strength}", token_change(base, f"EFF-b10_pen{strength}", "math25"), printed)
    change = token_change(base, "EFF-b10_pen1.0", "gsm8k")
    pct("E45 GSM8K tokens at -1.0, -5.2% (corrected 2026-09-19)", change, -5.2)
    pct("E45 GSM8K tokens at -2.0", token_change(base, "EFF-b10_pen2.0", "gsm8k"), -5.4)
    changes = [token_change(base, f"EFF-b10_pen{s}", "math25") for s in ("0.5", "1.0", "2.0", "3.0")]
    check("abstract: marker suppression trims by about 5% (E45 math25 -4.6 to -7.5%)",
          all(-0.08 < c < -0.04 for c in changes), f"{changes}")


# -------------------------------------------------------- hardening (H38)
def check_hardening() -> None:
    # E47 (H38a): 1,000 paired greedy items.
    gsm = paired("EFF-h38a_nodraft", "EFF-h38a_mtp", "gsm8k_500")
    mmlu = paired("EFF-h38a_nodraft", "EFF-h38a_mtp", "mmlu_500")
    for name, r, acc, ci_a, ci_b, disc, p, ident, ident_ci, tok_ident, toks, walls, speed in (
        ("GSM8K-500", gsm, (97.2, 97.6), [95.4, 98.3], [95.9, 98.6], (0, 2), 0.50, 318, [59.3, 67.7], 68.4, (327, 332), (11.61, 6.49), 1.79),
        ("MMLU-500", mmlu, (86.2, 86.8), [82.9, 88.9], [83.6, 89.5], (6, 9), 0.61, 105, [17.7, 24.8], 24.8, (725, 705), (23.78, 16.02), 1.48),
    ):
        same(f"E47 {name} n", r["n"], 500)
        rounds_to(f"E47 {name} undrafted accuracy", 100 * r["acc_a"] / r["n"], acc[0], 1)
        rounds_to(f"E47 {name} drafted accuracy", 100 * r["acc_b"] / r["n"], acc[1], 1)
        same(f"E47 {name} Wilson CI undrafted", ci_pct(r["acc_a"], r["n"]), ci_a)
        same(f"E47 {name} Wilson CI drafted", ci_pct(r["acc_b"], r["n"]), ci_b)
        same(f"E47 {name} discordant undrafted-only / drafted-only", (r["a_only"], r["b_only"]), disc)
        rounds_to(f"E47 {name} exact McNemar p", r["p"], p, 2)
        same(f"E47 {name} byte-identical outputs", r["k_identical"], ident)
        same(f"E47 {name} identity Wilson CI", ci_pct(r["k_identical"], r["n"]), ident_ci)
        pct(f"E47 {name} token-count identical", r["token_identity"], tok_ident)
        same(f"E47 {name} mean completion tokens undrafted / drafted", (round(r["mean_tok_a"]), round(r["mean_tok_b"])), toks)
        same(f"E47 {name} wall per item undrafted / drafted", (round(r["wall_a"], 2), round(r["wall_b"], 2)), walls)
        rounds_to(f"E47 {name} drafted speed-up", r["wall_a"] / r["wall_b"], speed, 2)
    same("E47 truncations MMLU 5 undrafted / 3 drafted; GSM8K none",
         (mmlu["trunc_a"], mmlu["trunc_b"], gsm["trunc_a"], gsm["trunc_b"]), (5, 3, 0, 0))
    pooled_a = gsm["acc_a"] + mmlu["acc_a"]
    pooled_b = gsm["acc_b"] + mmlu["acc_b"]
    rounds_to("E47 pooled accuracy undrafted 91.7%", pooled_a / 10, 91.7, 1)
    rounds_to("E47 pooled accuracy drafted 92.2%", pooled_b / 10, 92.2, 1)
    same("E47 pooled discordant 6 / 11", (gsm["a_only"] + mmlu["a_only"], gsm["b_only"] + mmlu["b_only"]), (6, 11))
    rounds_to("E47 pooled exact p = 0.33", mcnemar_exact(6, 11), 0.33, 2)
    same("E47 pooled byte-identical 423/1,000", gsm["k_identical"] + mmlu["k_identical"], 423)
    check("abstract: 'McNemar p >= 0.50' on both suites", min(gsm["p"], mmlu["p"]) >= 0.50, f"{gsm['p']}, {mmlu['p']}")
    rounds_to("E47 hazard GSM8K = -ln(0.636)/280", -math.log(gsm["k_identical"] / 500) / 280 * 1000, 1.6, 1)
    rounds_to("E47 hazard MMLU = -ln(0.21)/600", -math.log(mmlu["k_identical"] / 500) / 600 * 1000, 2.6, 1)
    for name, r, terciles, edges in (("GSM8K", gsm, (0.76, 0.69, 0.46), (227, 339)), ("MMLU", mmlu, (0.35, 0.23, 0.05), (450, 708))):
        ranked = sorted((r["a"][i]["completion_tokens"], i) for i in r["ids"])
        t = len(ranked) // 3
        parts = (ranked[:t], ranked[t:2 * t], ranked[2 * t:])
        identical = set(r["identical"])
        same(f"E47 {name} identity by undrafted-length tercile",
             tuple(round(sum(1 for _, i in part if i in identical) / len(part), 2) for part in parts), terciles)
        same(f"E47 {name} tercile edges", (parts[0][-1][0], parts[1][-1][0]), edges)
        diff = [i for i in r["ids"] if i not in identical]
        same_answer = sum(1 for i in diff if r["a"][i].get("final_answer_sha256") is not None
                          and r["a"][i]["final_answer_sha256"] == r["b"][i].get("final_answer_sha256"))
        parsed = sum(1 for i in diff if r["a"][i].get("final_answer_sha256") is not None and r["b"][i].get("final_answer_sha256") is not None)
        agree = sum(1 for i in diff if r["a"][i]["correct"] == r["b"][i]["correct"])
        deltas = [r["b"][i]["completion_tokens"] - r["a"][i]["completion_tokens"] for i in diff]
        if name == "GSM8K":
            same("E47 GSM8K differing pairs 182; same final answer 179; same correctness 180", (len(diff), same_answer, agree), (182, 179, 180))
            same("E47 GSM8K median length difference 1 token", statistics.median(deltas), 1)
        else:
            same("E47 MMLU differing pairs 395; parsed 387; same answer 374; same correctness 380",
                 (len(diff), parsed, same_answer, agree), (395, 387, 374, 380))
            same("E47 MMLU median length difference -1 (mean -25)", (statistics.median(deltas), round(statistics.mean(deltas))), (-1, -25))

    # E48 (H38b): first 200 GSM8K-500 items, 4-bit offloaded vs 2-bit drafted.
    b = paired("EFF-h38b_q4kxl_mtp", "EFF-h38a_mtp", "gsm8k_500", limit=200)
    same("E48 n = 200 paired", b["n"], 200)
    same("E48 accuracy 98.0 vs 99.0", (100 * b["acc_a"] / 200, 100 * b["acc_b"] / 200), (98.0, 99.0))
    same("E48 Wilson CIs", (ci_pct(b["acc_a"], 200), ci_pct(b["acc_b"], 200)), ([95.0, 99.2], [96.4, 99.7]))
    same("E48 discordant 1 (4-bit only) / 3 (2-bit only)", (b["a_only"], b["b_only"]), (1, 3))
    rounds_to("abstract: E48 exact McNemar p = 0.63", b["p"], 0.63, 2)
    same("E48 mean completion tokens 321.9 vs 322.0", (round(b["mean_tok_a"], 1), round(b["mean_tok_b"], 1)), (321.9, 322.0))
    same("E48 wall per item 30.24 vs 6.30 s", (round(b["wall_a"], 2), round(b["wall_b"], 2)), (30.24, 6.30))
    rounds_to("E48 x4.8 wall ratio (recorded, not claimed)", b["wall_a"] / b["wall_b"], 4.8, 1)
    same("E48 truncations 0 / 0", (b["trunc_a"], b["trunc_b"]), (0, 0))
    same_answer = sum(1 for i in b["ids"] if b["a"][i].get("final_answer_sha256") is not None
                      and b["a"][i]["final_answer_sha256"] == b["b"][i].get("final_answer_sha256"))
    same("E48 same final answer on 195 of 200", same_answer, 195)
    longer = sum(1 for i in b["ids"] if b["b"][i]["completion_tokens"] > 1.2 * b["a"][i]["completion_tokens"])
    shorter = sum(1 for i in b["ids"] if b["b"][i]["completion_tokens"] < 0.8 * b["a"][i]["completion_tokens"])
    same("E48 2-bit trace >20% longer on 34 items, >20% shorter on 16", (longer, shorter), (34, 16))
    lo_diff = (b["acc_b"] - b["acc_a"]) / 200
    check("conclusion: 'within +/-3 points of a 4-bit GGUF' (both Wilson intervals inside a 3-point band of each other's point estimate)",
          abs(lo_diff) <= 0.03 and max(abs(x - 98.0) for x in ci_pct(b["acc_a"], 200)) <= 3.0 + 1e-9, f"{lo_diff}")

    # E49 (H38c) and E50 (H38d): first 100 GSM8K-500 items.
    def ident(a: str, bb: str) -> dict[str, Any]:
        return paired(a, bb, "gsm8k_500", limit=100)

    i_ = ident("EFF-h38a_mtp", "EFF-h38c_mtp_rerun")
    du = ident("EFF-h38a_nodraft", "EFF-h38a_mtp")
    iii = ident("EFF-h38a_nodraft", "EFF-h38c_nodraft_rerun")
    for name, r, k, ci, hazard, acc in (("(i) drafted vs drafted re-run", i_, 91, [83.8, 95.2], 3.4e-4, (99, 99, 0)),
                                        ("drafted vs undrafted (E47 pairs)", du, 63, [53.2, 71.8], 1.7e-3, (98, 99, 1)),
                                        ("(iii) undrafted vs undrafted re-run", iii, 50, [40.4, 59.6], 2.5e-3, (98, 98, 0))):
        same(f"E49 {name} byte-identical", r["k_identical"], k)
        same(f"E49 {name} Wilson CI", ci_pct(r["k_identical"], 100), ci)
        same(f"E49 {name} hazard -ln(identity)/280", f"{-math.log(r['k_identical'] / 100) / 280:.1e}", f"{hazard:.1e}")
        same(f"E49 {name} accuracy and discordant", (r["acc_a"], r["acc_b"], r["a_only"] + r["b_only"]), acc)
    rounds_to("E49 hazard ratio undrafted/drafted ~7x", (-math.log(0.50)) / (-math.log(0.91)), 7.3, 1)
    diff = [i for i in iii["ids"] if i not in set(iii["identical"])]
    same("E49 (iii) all 50 differing pairs reach the same answer",
         sum(1 for i in diff if iii["a"][i]["final_answer_sha256"] == iii["b"][i]["final_answer_sha256"]), 50)
    same("E49 walls: drafted 6.39 / 5.87 s; undrafted 11.74 / 9.86 s",
         (round(i_["wall_a"], 2), round(i_["wall_b"], 2), round(iii["wall_a"], 2), round(iii["wall_b"], 2)), (6.39, 5.87, 11.74, 9.86))
    rounds_to("E50 run-to-run spread of the drafted pair ~8%", 100 * (i_["wall_a"] / i_["wall_b"] - 1), 8.9, 1)
    pairs = load_jsonl("traces/h38c_pairs.jsonl")
    divergent = [p for p in pairs if not p["identical"]]
    same("E49 (ii) 25 paired traces: 20 identical / 5 divergent", (len(pairs), len(pairs) - len(divergent)), (25, 20))
    same("E49 (ii) top-1/top-2 gaps at the first divergent token (nat, 1/64 steps)",
         sorted(round(p["gap_nat"], 3) for p in divergent), [0.0, 0.016, 0.016, 0.016, 0.031])
    same("E49 (ii) drafted token is the runner-up in 5 of 5", sum(p["other_is_runner_up"] for p in divergent), 5)
    divs = sorted(p["first_div"] for p in divergent)
    same("E49 (ii) first divergences at tokens 39-357, median 92", (divs[0], divs[-1], statistics.median(divs)), (39, 357, 92))
    fracs = sorted(p["frac"] for p in divergent)
    same("E49 (ii) 4-61% into the trace", (round(100 * fracs[0]), round(100 * fracs[-1])), (4, 61))
    gaps_rows = {r["label"]: {} for r in load_jsonl("traces/h38c_trace_gaps.jsonl")}
    for r in load_jsonl("traces/h38c_trace_gaps.jsonl"):
        gaps_rows[r["label"]][r["id"]] = r
    same("E49 (ii) 23 of 25 pairs end with the same answer (last 60 characters)",
         sum(1 for i, r in gaps_rows["h38c_nodraft"].items() if r["content_tail60_sha256"] == gaps_rows["h38c_mtp"][i]["content_tail60_sha256"]), 23)
    all_gaps = [g for r in gaps_rows["h38c_nodraft"].values() for g in r["top1_top2_gap_nat"] if g is not None]
    same("E49 8,475 positions in the undrafted traces", sum(len(r["top1_top2_gap_nat"]) for r in gaps_rows["h38c_nodraft"].values()), 8475)
    pct("E49 gap < 0.05 nat at 0.35% of positions", sum(1 for g in all_gaps if g < 0.05) / 8475, 0.35, 2)
    pct("E49 gap < 0.02 nat at 0.13% of positions", sum(1 for g in all_gaps if g < 0.02) / 8475, 0.13, 2)

    clean = ident("EFF-h38d_nodraft_clean1", "EFF-h38d_nodraft_clean2")
    same("abstract/E50: undrafted clean vs clean byte-identical 64/100 [54.2, 72.7]",
         (clean["k_identical"], ci_pct(clean["k_identical"], 100)), (64, [54.2, 72.7]))
    same("E50 clean accuracy 98 / 99 (0 / 1)", (clean["acc_a"], clean["acc_b"], clean["a_only"], clean["b_only"]), (98, 99, 0, 1))
    same("E50 clean walls 10.81 / 10.53 s", (round(clean["wall_a"], 2), round(clean["wall_b"], 2)), (10.81, 10.53))
    same("E50 clean run 1 vs mixed A / mixed B: 58 / 61",
         (ident("EFF-h38a_nodraft", "EFF-h38d_nodraft_clean1")["k_identical"],
          ident("EFF-h38c_nodraft_rerun", "EFF-h38d_nodraft_clean1")["k_identical"]), (58, 61))
    drafted = ident("EFF-h38a_mtp", "EFF-h38d_mtp_clean1")
    same("E50 drafted mixed vs drafted clean 91/100 [83.8, 95.2]", (drafted["k_identical"], ci_pct(drafted["k_identical"], 100)), (91, [83.8, 95.2]))
    same("E50 drafted walls 6.39 / 5.92 s", (round(drafted["wall_a"], 2), round(drafted["wall_b"], 2)), (6.39, 5.92))

    dev = ident("EFF-h38e_nodraft_dev1", "EFF-h38e_nodraft_dev2")
    same("abstract/E51: dev branch 100 of 100 byte-identical [96.3, 100]", (dev["k_identical"], ci_pct(dev["k_identical"], 100)), (100, [96.3, 100.0]))
    same("E51 accuracy 97 / 97", (dev["acc_a"], dev["acc_b"]), (97, 97))
    cross = ident("EFF-h38d_nodraft_clean1", "EFF-h38e_nodraft_dev1")
    same("E51 1.4.7 clean vs dev 67/100 [57.3, 75.4], accuracy 98 / 97",
         (cross["k_identical"], ci_pct(cross["k_identical"], 100), cross["acc_a"], cross["acc_b"]), (67, [57.3, 75.4], 98, 97))

    rel = ident("EFF-h38f_nodraft_rel1", "EFF-h38f_nodraft_rel2")
    same("abstract/E52: released 1.5.0, 100 of 100 byte-identical [96.3, 100]", (rel["k_identical"], ci_pct(rel["k_identical"], 100)), (100, [96.3, 100.0]))
    same("E52 accuracy 98 / 98, no discordant items", (rel["acc_a"], rel["acc_b"], rel["a_only"], rel["b_only"]), (98, 98, 0, 0))
    dev_rel = ident("EFF-h38e_nodraft_dev1", "EFF-h38f_nodraft_rel1")
    same("E52 dev branch (E51) vs release 72/100 [62.5, 79.9], accuracy 97 / 98",
         (dev_rel["k_identical"], ci_pct(dev_rel["k_identical"], 100), dev_rel["acc_a"], dev_rel["acc_b"]), (72, [62.5, 79.9], 97, 98))

    # The committed per-comparison summaries agree with the recomputation.
    for name, r in (("h38a_gsm8k_500", gsm), ("h38a_mmlu_500", mmlu), ("h38b_gsm8k", b), ("h38c_same_config", i_),
                    ("h38c_same_config_undrafted", iii), ("h38d_clean_vs_clean", clean), ("h38d_mtp_clean_vs_mixed", drafted),
                    ("h38e_dev_vs_dev", dev), ("h38e_147clean_vs_dev", cross),
                    ("h38f_rel_vs_rel", rel), ("h38f_dev_vs_rel", dev_rel)):
        s = load_json(f"rows/{name}.json")
        same(f"summary row {name} agrees with the recomputation",
             (s["n"], s["A_only_correct"], s["B_only_correct"], round(s["identity_rate"] * s["n"])),
             (r["n"], r["a_only"], r["b_only"], r["k_identical"]))
    libs = [ROOT for _ in ()]  # placeholder to keep the section self-contained
    for name in ("EFF-h38d_nodraft_clean1", "EFF-h38d_nodraft_clean2", "EFF-h38d_mtp_clean1",
                 "EFF-h38e_nodraft_dev1", "EFF-h38e_nodraft_dev2", "EFF-h38f_nodraft_rel1", "EFF-h38f_nodraft_rel2"):
        lines = [l for l in (DATA / "rows" / f"{name}_cuda_libs.txt").read_text().splitlines() if l.strip()]
        check(f"E50-E52 {name}: six bundled CUDA libraries, none from the system toolkit",
              len(lines) == 6 and all("site-packages/nvidia/" in l for l in lines) and not any("cuda-12.1" in l for l in lines),
              f"{lines}")
    del libs


def main() -> int:
    verify_integrity()
    sections: list[tuple[str, Callable[[], None]]] = [
        ("E7", check_e7), ("E25", check_e25), ("E29", check_e29), ("Table 1 / E27 / E28", check_table1_and_vram),
        ("E35 / E36 / E37", check_e35_e37), ("E40", check_e40), ("levers", check_levers), ("hardening", check_hardening),
    ]
    for name, function in sections:
        try:
            function()
        except Exception as exc:  # a missing field is a failed check, never a pass
            FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
    print(f"{len(PASSED)} checks passed")
    if DISCREPANCIES:
        print("documented discrepancies (the re-derived value is asserted; the paper's figure is reported):")
        for line in DISCREPANCIES:
            print(f"  - {line}")
    if FAILURES:
        print(f"{len(FAILURES)} checks FAILED:")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("ALL PAPER 7 CLAIM CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
