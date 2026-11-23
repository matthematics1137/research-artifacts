"""Portable arithmetic extracted unchanged from the registered P8 Bonsai analyzer.

Source: tools/turing/bonsai/analyze.py at e42ef4a8dd7d0577afb05dc7e5e5ec7e38bbec16.
The lab's registered family, rounded Wilson z, bootstrap seed and 5000 replicates are preserved.
Only acquisition/legacy-harness imports are omitted; this module reads archived numeric outcomes.
"""
import json
import math
import random
import statistics


FAMILY = [('gsm8k_500', 'TURING-exl3-nodraft', 'TURING-bonsai-ninfer-none'),
          ('mmlu_500', 'TURING-exl3-nodraft', 'TURING-bonsai-ninfer-none'),
          ('gsm8k_500', 'TURING-exl3-mtp', 'TURING-bonsai-ninfer-mtp3'),
          ('mmlu_500', 'TURING-exl3-mtp', 'TURING-bonsai-ninfer-mtp3'),
          ('gsm8k_500', 'TURING-exl3-mtp-w13', 'TURING-bonsai-ninfer-mtp3')]


def load_rows(path, wanted):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        ident = r['id']
        if ident in out or ident not in wanted:
            raise ValueError('duplicate or unknown id: ' + str(ident))
        if type(r.get('correct')) is not bool or (r.get('error') and r['correct']):
            raise ValueError('invalid correctness/error')
        wall = r.get('wall_s')
        # Legacy rounds to centiseconds: a fast transport error can truthfully be 0.00.
        if (type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0
                or (wall == 0 and 'error' not in r)):
            raise ValueError('invalid/nonfinite wall time')
        ct = r.get('completion_tokens')
        if ct is not None and (type(ct) is not int or ct < 0):
            raise ValueError('invalid token usage')
        out[ident] = r
    return out


def wilson(k, n):
    if not n:
        return None
    p=k/n; z=1.96; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h, c+h]


def mcnemar(b, c):
    n=b+c
    return min(1.0, 2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n)


def holm(ps):
    adjusted=[None]*len(ps); previous=0.0
    # Missing comparisons stay in the registered family as p=1; never shrink m.
    ordered=sorted(range(len(ps)), key=lambda i: 1.0 if ps[i] is None else ps[i])
    for rank,i in enumerate(ordered):
        v=1.0 if ps[i] is None else ps[i]
        previous=max(previous, min(1.0, (len(ps)-rank)*v))
        if ps[i] is not None:
            adjusted[i]=previous
    return adjusted


def summarize(rows, expected):
    rs=list(rows.values()); n=len(rs); correct=sum(r['correct'] for r in rs)
    wall=sum(r['wall_s'] for r in rs); tokens=[r['completion_tokens'] for r in rs if r.get('completion_tokens') is not None]
    errors=sum('error' in r for r in rs)
    unfinished=sum('error' in r or bool(r.get('truncated')) for r in rs)
    return {'n':n, 'expected':expected, 'errors':errors, 'missing':expected-n,
            'complete':n==expected and errors==0, 'correct':correct,
            'accuracy':correct/n if n else None, 'accuracy_wilson95':wilson(correct,n),
            'total_wall_s':wall, 'mean_wall_s':wall/n if n else None,
            'correct_per_minute':60*correct/wall if wall else None,
            'mean_completion_tokens':statistics.mean(tokens) if tokens else None,
            'tokens_present':len(tokens), 'truncated':sum(bool(r.get('truncated')) for r in rs),
            'finished_correct':sum(r['correct'] and 'error' not in r and not r.get('truncated') for r in rs),
            'finished_wrong':sum(not r['correct'] and 'error' not in r and not r.get('truncated') for r in rs),
            'unfinished_including_missing':unfinished+expected-n,
            'legacy_correct_truncated':sum(r['correct'] and bool(r.get('truncated')) for r in rs)}


def paired(a, b, ids, reps=5000):
    if set(a)!=set(ids) or set(b)!=set(ids) or any('error' in x for x in [*a.values(),*b.values()]):
        return {'status':'incomplete', 'p':None}
    n=len(ids); ar=[a[i] for i in ids]; br=[b[i] for i in ids]
    ao=sum(x['correct'] and not y['correct'] for x,y in zip(ar,br))
    bo=sum(y['correct'] and not x['correct'] for x,y in zip(ar,br))
    rng=random.Random(20261112); deltas=[]; rates=[]
    for _ in range(reps):
        ix=[rng.randrange(n) for _ in range(n)]
        ka=sum(ar[i]['correct'] for i in ix); kb=sum(br[i]['correct'] for i in ix)
        wa=sum(ar[i]['wall_s'] for i in ix); wb=sum(br[i]['wall_s'] for i in ix)
        deltas.append(100*(kb-ka)/n)
        rates.append(60*kb/wb-60*ka/wa)
    def ci(values):
        values.sort()
        return [values[int((reps-1)*.025)],values[int((reps-1)*.975)]]
    return {'status':'complete','n':n,'a_only':ao,'b_only':bo,'delta_accuracy_pp':100*(bo-ao)/n,
            'p':mcnemar(ao,bo),'delta_accuracy_pp_bootstrap95':ci(deltas),
            'delta_correct_per_minute_bootstrap95':ci(rates),
            'interval_note':f'unadjusted descriptive paired percentile bootstrap; fixed seed 20261112; {reps} replicates'}
