#!/usr/bin/env python3
"""Replay the registered Bonsai row from archived numeric outcomes, without response text or inference.

Arithmetic is the unchanged registered implementation in bonsai_stats.py. Compact
projections retain IDs/order, correctness, request wall time, native tokens and
length finishes. They do not rescore answers or establish byte/token identity.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import bonsai_stats as S

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'testsuite/results/turing/bonsai'
PAPER = ROOT / 'paper8'


def replay():
    expected = json.loads((DATA / 'registered-analysis.json').read_text())
    ids = json.loads((DATA / 'expected-ids.json').read_text())
    for suite, values in ids.items():
        assert len(values) == len(set(values)) == 500, suite
        # Preserve the registration's compact-JSON ordered-ID hash.
        assert hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest() == expected['contract']['datasets'][suite]['ids_sha256'], suite
    loaded, summaries, comparisons = {}, {}, []
    for suite, original, bonsai in S.FAMILY:
        for label in (original, bonsai):
            key = (suite, label)
            if key in loaded:
                continue
            path = (DATA if 'bonsai' in label else ROOT / 'testsuite/results/turing/quality') / label / (suite + '.jsonl')
            rows = S.load_rows(path, set(ids[suite]))
            assert list(rows) == ids[suite], 'missing, reordered or incomplete rows: ' + str(path)
            assert all('error' not in value for value in rows.values()), path
            loaded[key] = rows
            summaries[suite + ':' + label] = S.summarize(rows, 500)
        comparisons.append(dict(suite=suite, a=original, b=bonsai,
                                **S.paired(loaded[suite, original], loaded[suite, bonsai], ids[suite])))
    for row, adjusted in zip(comparisons, S.holm([row['p'] for row in comparisons])):
        row['p_holm_family5'] = adjusted
    assert summaries == expected['summaries'], 'registered summary mismatch'
    assert comparisons == expected['comparisons'], 'registered paired/bootstrap/family mismatch'
    assert all(row['complete'] for row in summaries.values())
    assert all(row['p_holm_family5'] >= .05 for row in comparisons)
    return summaries, comparisons


def render(summaries, comparisons):
    order = [('gsm8k_500', 'TURING-bonsai-ninfer-none'),
             ('mmlu_500', 'TURING-bonsai-ninfer-none'),
             ('gsm8k_500', 'TURING-bonsai-ninfer-mtp3'),
             ('mmlu_500', 'TURING-bonsai-ninfer-mtp3')]

    def cell(suite, label):
        row = summaries[suite + ':' + label]
        low, high = row['accuracy_wilson95']
        return (f"$n={row['n']}$\\newline {100*row['accuracy']:.1f}\\%\\newline "
                f"[{100*low:.1f}, {100*high:.1f}]\\%\\newline "
                f"{row['mean_completion_tokens']:.1f} tokens\\newline "
                f"{row['mean_wall_s']:.2f} s/answer\\newline "
                f"{row['correct_per_minute']:.2f} correct/min")

    # Caption finish counts and the legacy-correct capped answer are themselves checked.
    assert [summaries[s + ':' + label]['truncated'] for s, label in order] == [0, 4, 0, 3]
    assert summaries['mmlu_500:TURING-bonsai-ninfer-mtp3']['legacy_correct_truncated'] == 1
    layout = (PAPER / 'data/bonsai-row-layout.tex').read_text()
    assert layout.count('@@BONSAI_ROW@@') == 1
    row = 'Bonsai 2 27B/NInfer (ternary derivative; none / MTP-3) & ' + ' & '.join(cell(*key) for key in order) + r'\\'
    table = layout.replace('@@BONSAI_ROW@@', row)

    def clause(index, first):
        comparison = comparisons[index]
        suite = comparison['suite']
        value = summaries[suite + ':' + comparison['b']]
        original = summaries[suite + ':' + comparison['a']]
        suite_name = 'GSM8K' if suite == 'gsm8k_500' else 'MMLU'
        control = 'archived w13 MTP' if comparison['a'] == 'TURING-exl3-mtp-w13' else 'stock EXL3 MTP'
        return (('On' if first else 'on') + f" our fixed 500-item {suite_name} slice, Bonsai 2 27B/NInfer MTP-3 yielded "
                f"{value['correct_per_minute']:.2f} correct answers/min versus {original['correct_per_minute']:.2f} for {control}, "
                f"at {100*value['accuracy']:.1f}% versus {100*original['accuracy']:.1f}% accuracy "
                f"(paired difference {comparison['delta_accuracy_pp']:+.1f} points, Holm-adjusted p={comparison['p_holm_family5']:.3f})")
    sentence = clause(4, True) + '; ' + clause(3, False) + '.'
    latex = re.sub(r'Holm-adjusted p=([0-9.]+)', r'Holm-adjusted $p=\1$', sentence.replace('%', r'\%')) + '\n'
    return {'bonsai_counterpoint.tex': table, 'bonsai_sentence.tex': latex}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-derived', action='store_true')
    args = parser.parse_args()
    summaries, comparisons = replay()
    outputs = render(summaries, comparisons)
    for name, text in outputs.items():
        path = PAPER / 'derived' / name
        if args.write_derived:
            path.write_text(text)
        else:
            assert path.read_text() == text, 'changed ' + name
    # Bind the generated evidence to the actual manuscript, including the curated public layout.
    manuscript = PAPER / 'main.tex'
    if not manuscript.exists():
        manuscript = ROOT.parent / 'paper/main.tex'
    text = manuscript.read_text()
    assert r'\input{derived/bonsai_counterpoint}' in text
    assert r'\input{derived/bonsai_sentence}' in text
    print('Bonsai: nine 500-item numeric lanes; all registered summaries, five paired tests/Holm and 5000-replicate intervals match; two generated TeX outputs match.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
