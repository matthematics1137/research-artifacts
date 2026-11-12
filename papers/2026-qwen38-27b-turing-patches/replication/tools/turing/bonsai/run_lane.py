#!/usr/bin/env python3
"""Future queue payload. Fresh directory only; a failed attempt is never resumed.

Use only after the preregistration and runtime holds in staging.md are cleared.
Creates/fsyncs an intent before every generation, reuses the legacy row/scorer,
keeps native and normalized responses, stops on the first transport/shape error.
The outer queue step owns the global six-hour wall cap and server lifecycle.
"""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from adapter import Adapter, REPO, digest


def append_sync(f, value):
    f.write(json.dumps(value, ensure_ascii=False) + '\n')
    f.flush()
    os.fsync(f.fileno())


def run(base_url, mode, datasets, out, preflight, check_only=False):
    contract = json.loads((REPO / 'research/raw/2026-11-12-p8-bonsai-lane/contract.json').read_text())
    harness = REPO / 'tools/turing/quality/run_eval_turing.py'
    if digest(harness.read_bytes()) != contract['legacy_harness_sha256']:
        raise ValueError('legacy harness changed; registration review needed')
    receipt = json.loads(preflight.read_text())
    for key, want in contract['runtime_required'].items():
        if receipt.get(key) != want:
            raise ValueError('unqualified preflight field: ' + key)
    if receipt.get('mode') != mode or not receipt.get('evidence_paths'):
        raise ValueError('mode-specific runtime evidence required')
    binary_sha = receipt.get('binary_sha256')
    qualification_s = receipt.get('qualification_elapsed_s')
    if (not isinstance(binary_sha, str) or len(binary_sha) != 64
            or any(c not in '0123456789abcdef' for c in binary_sha)
            or type(qualification_s) not in (int, float)
            or not 0 < qualification_s <= 900):
        raise ValueError('qualified binary hash and elapsed time required')
    adapter = Adapter(base_url)
    legacy = adapter.legacy
    suites = {}
    for suite in ('gsm8k_500', 'mmlu_500'):
        ok, got, want = legacy.verify_dataset(suite, str(datasets))
        if not ok:
            raise ValueError('dataset hash mismatch: ' + suite)
        items = legacy.load_dataset(suite, 0, str(datasets))
        ids = [x['id'] for x in items]
        if len(items) != 500 or len(set(ids)) != 500 or digest(json.dumps(ids, separators=(',', ':')).encode()) != contract['datasets'][suite]['ids_sha256']:
            raise ValueError('dataset item identity/order mismatch')
        suites[suite] = items
    if out.exists():
        raise ValueError('fresh result directory required')
    if check_only:
        return 0
    out.mkdir(parents=True, exist_ok=False)
    (out / 'preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
    legacy.chat_completion = adapter.chat_completion
    args = SimpleNamespace(base_url=base_url, temperature=0, max_tokens=4096,
                           reasoning_effort='medium', extra_body=None, encode_count=False,
                           label='TURING-bonsai-ninfer-' + mode)
    started = time.monotonic()
    for suite, items in suites.items():
        with (out / (suite + '.jsonl')).open('x') as rows, (out / (suite + '.full.jsonl')).open('x') as full, (out / (suite + '.intents.jsonl')).open('x') as intents:
            for item in items:
                append_sync(intents, {'id': item['id'], 'state': 'intent', 'attempt': 1,
                                     'elapsed_s': time.monotonic() - started})
                row, sidecar = legacy.run_item_full(suite, item, args)
                sidecar['ninfer_exchange'] = adapter.last_exchange
                sidecar['token_basis'] = 'native_usage_generated_tokens'
                sidecar['shape_fidelity'] = 'logical_inline_normalization; byte fidelity unverified'
                append_sync(full, sidecar)
                append_sync(rows, row)
                append_sync(intents, {'id': item['id'], 'state': 'recorded', 'attempt': 1})
                if 'error' in row or 'sidecar_error' in sidecar:
                    return 3
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--mode', choices=('none', 'mtp3'), required=True)
    p.add_argument('--datasets-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--check-only', action='store_true', help='validate local files only; no HTTP or writes')
    a = p.parse_args()
    raise SystemExit(run(a.base_url, a.mode, a.datasets_dir, a.output_dir, a.preflight, a.check_only))
