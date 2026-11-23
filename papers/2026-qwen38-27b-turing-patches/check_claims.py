#!/usr/bin/env python3
"""Offline replay from exported P8 evidence. No model, tokenizer or network.

Default: integrity, all generated claims/fragments, quality counts/pairs,
registered Bonsai statistics and fleet projections. --figures additionally
redraws every used light figure with the pinned Matplotlib environment.
Original experiment sources remain under replication/.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import statistics
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GENERATED = ROOT if (ROOT/'replication').is_dir() else ROOT/'generated'
ANALYSIS = GENERATED/'replication'
INVENTORY_SHA256 = '7a37f8939738d3aa8a42d049fa61ad0720ae87c6c852464fcb4fd796c239ef3f'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().split('\n') if line.strip()]


def quality():
    directory = ANALYSIS/'testsuite/results/turing/quality'
    report = json.loads((directory/'compare_lcpp_fp32.json').read_text())
    by = {}
    for suite, lanes in report['lanes'].items():
        for lane in lanes:
            path = directory/lane['label']/(suite+'.jsonl')
            if not path.is_file():
                continue
            values = rows(path)
            ids = [r['id'] for r in values]
            assert len(values) == len(set(ids)) == lane['n'] == 500, path
            assert all(type(r['correct']) is bool for r in values), path
            assert sum(r['correct'] for r in values) == lane['correct'], path
            assert round(100*sum(r['correct'] for r in values)/len(values), 1) == lane['acc'], path
            assert round(statistics.mean(r['wall_s'] for r in values), 2) == lane['mean_wall'], path
            assert round(statistics.mean(r['completion_tokens'] for r in values), 1) == lane['mean_tok'], path
            by[(suite,lane['label'])] = {r['id']:r for r in values}
    count = 0
    for pair in report['pairs']:
        if pair['group'] != 'turing-internal':
            continue
        a = by[(pair['suite'],pair['A'])]
        b = by[(pair['suite'],pair['B'])]
        assert set(a)==set(b) and len(a)==500
        st=pair['stats']
        assert sum(a[k]['correct'] and not b[k]['correct'] for k in a)==st['A_only_correct']
        assert sum(b[k]['correct'] and not a[k]['correct'] for k in a)==st['B_only_correct']
        count+=1
    assert count==14
    print('Quality: '+str(len(by))+' complete 500-item lanes; 14 paired discordance counts match.')


def semantic_attribution():
    """Independently select the named arm; do not reuse the claim builder's loops."""
    source=rows(ANALYSIS/'testsuite/results/turing/ab/final2.jsonl')
    selected=[r for r in source if re.fullmatch(r'final2-w13s2-mtp-0-r[123]',r['tag'])]
    macros=dict(re.findall(r'\\newcommand\{\\(Pe[A-Za-z]+)\}\{([^}]+)\}',
                           (ANALYSIS/'paper8/derived/claims.tex').read_text()))
    failures=[]
    for task in ('code','math','prose'):
        values=[r for r in selected if r['task']==task]
        assert len(values)==3 and len({r['tag'] for r in values})==3, task
        expected=f"{sum(r['new_tokens'] for r in values)/sum(r['decode_wall_s'] for r in values):.1f}"
        name='PeFtPpSTwo'+task.capitalize()
        actual=macros[name]
        if expected!=actual:
            failures.append({'macro':name,'printed':actual,'frozen_named_arm':expected})
    if failures:
        print('RELEASE HOLD: per-prompt prose macro attribution disagrees with the named frozen arm:')
        print(json.dumps(failures,indent=2))
        raise SystemExit(1)
    print('Semantic attribution: mode2 empty-context per-prompt macros match their own arm.')


def fleet():
    """Check complete numeric fixtures, fixed row order and paired IDs independently.

    These are archived verdicts, not response re-scoring or controlled seeds.
    Card labels are logical fixtures, never a route to a private machine.
    """
    base = ANALYSIS / 'research/raw/2026-11-12-fleet-lanes/results-2026-11-14-c4d8/available/quality-rows'
    for card in ('card-4070', 'card-4080-super'):
        values = rows(ANALYSIS / 'fixtures/fleet' / card / 'runs/ns-20261112-p8-speed/journal.jsonl')
        assert len(values) == len({r['id'] for r in values}) == 36
        conditions = [(False, 0), (True, 0), (False, 28000), (True, 28000)]
        expected = [(rnd, draft, fill, task) for rnd in range(3)
                    for draft, fill in conditions[rnd:] + conditions[:rnd] for task in ('code', 'math', 'prose')]
        assert [(r['round'], r['drafted'], r['fill'], r['task']) for r in values] == expected
        assert all(type(r['drafted']) is bool and type(r['new_tokens']) is int and r['new_tokens'] > 0
                   and math.isfinite(r['decode_wall_s']) and r['decode_wall_s'] > 0 for r in values)
        for suite in ('gsm8k_500', 'mmlu_500'):
            pairs = [rows(base / f'FLEET-{card}-w13-{mode}' / (suite + '.jsonl')) for mode in ('nd', 'mtp')]
            reference = rows(ANALYSIS / 'testsuite/results/turing/quality/TURING-exl3-nodraft' / (suite + '.jsonl'))
            expected_ids = [r['id'] for r in reference]
            for arm in pairs:
                assert len(arm) == len({r['id'] for r in arm}) == 500
                assert [r['id'] for r in arm] == expected_ids
                assert all(type(r['correct']) is bool and not r.get('error') and
                           type(r['completion_tokens']) is int and r['completion_tokens'] >= 0 and
                           math.isfinite(r['wall_s']) and r['wall_s'] > 0 for r in arm)
    print('Fleet: two complete rotated 36-request speed fixtures and eight paired 500-item verdict lanes; numeric projections only.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--figures',action='store_true')
    args=ap.parse_args()
    inventory=GENERATED/'EXPORT_MANIFEST.json'
    assert sha(inventory)==INVENTORY_SHA256, 'export inventory changed'
    data=json.loads(inventory.read_text())
    for row in data['files']:
        path=GENERATED/row['destination']
        assert path.is_file() and sha(path)==row['export_sha256'], row['destination']
    print('Integrity: '+str(len(data['files']))+' exported files match the inventory.')
    paper=ANALYSIS/'paper8'
    names=sorted(p.name for p in (ROOT/'paper/derived').glob('*.tex'))
    assert {'claims.tex','f2_per_prompt.tex','model_files.tex','bonsai_counterpoint.tex','bonsai_sentence.tex','fleet_replication.tex'} <= set(names)
    before={name:(paper/'derived'/name).read_bytes() for name in names}
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    subprocess.run([sys.executable,'-B','scripts/build_claims.py'],cwd=paper,env=env,check=True)
    subprocess.run([sys.executable,'-B','scripts/check_bonsai.py','--write-derived'],cwd=paper,env=env,check=True)
    for name,value in before.items():
        assert (paper/'derived'/name).read_bytes()==value, 'changed '+name
    quality()
    fleet()
    if args.figures:
        import matplotlib
        assert matplotlib.__version__=='3.10.6', 'Matplotlib 3.10.6 is required for exact figure replay'
        subprocess.run([sys.executable,'-B','scripts/make_figures.py'],cwd=paper,env=env,check=True)
        for name in sorted(p.name for p in (ROOT/'paper/figures').glob('*.pdf')):
            assert (paper/'figures'/name).read_bytes()==(ROOT/'paper/figures'/name).read_bytes(), name
        print('Figures: all canonical light PDFs are byte-identical.')
    semantic_attribution()
    print('PASS: frozen outputs replay exactly; this is verification, not author approval.')


if __name__=='__main__':
    main()
