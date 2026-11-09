#!/usr/bin/env python3
"""cn2_nonrigid_collect.py -- one deduplicated, re-verified file of every non-rigid order-2 number we have found.

    python3 cn2_nonrigid_collect.py [--out cn2_nonrigid_all.tsv.gz] [--procs 24]

Sources (each record carries its generator L0, its type-p prime(s) and every value found -- all values were stored):
  cn2_nonrigid_runs.jsonl, cn2_box_nonrigid.jsonl, cn2_xbox_nonrigid.jsonl, cn2_box_nonrigid_smallp.jsonl,
  cn2_ey_top10_runs.jsonl   (one type-p prime),  cn2_twin_runs.jsonl (two),  cn2_triple_runs.jsonl (three).
Each n is factorised by dividing out the pool primes of its generator and its type-p primes (every prime of a number
found by Howe's construction is one of these), then checked from scratch: squarefree, the product is n, and
n mod (p^2-1) is 1 or p for every prime; the type-p primes are recomputed and must equal the searched set.

Output, sorted by n, gzip-compressed tab-separated, one row per distinct n:
  log10  k  type_p  L_R  times_found  n  primes
where type_p is the comma-separated type-p primes, L_R = lcm(q^2-1 : q a rigid prime of n) (the rigid part),
times_found counts the search records (generator, type-p set) that produced n, and primes is space-separated.
A summary goes to the .summary.json beside it.
"""
import argparse, gzip, json, math, os, sys
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sympy import factorint                      # noqa: E402
from cn2_nonrigid_scout import pool             # noqa: E402

SOURCES = ['cn2_nonrigid_runs.jsonl', 'cn2_box_nonrigid.jsonl', 'cn2_xbox_nonrigid.jsonl',
           'cn2_box_nonrigid_smallp.jsonl', 'cn2_ey_top10_runs.jsonl', 'cn2_twin_runs.jsonl', 'cn2_triple_runs.jsonl']


def lcm(a, b):
    return a // math.gcd(a, b) * b


def check(job):
    """job = (candidate primes, type-p set, values) -> list of (n, primes, type_p) or ('BAD', n, reason)"""
    cands, tset, values = job
    out = []
    for v in values:
        n = m = int(v)
        ps = []
        for q in cands:
            if m % q == 0:
                m //= q
                if m % q == 0:
                    out.append(('BAD', n, f'not squarefree at {q}')); break
                ps.append(q)
        else:
            if m != 1:
                out.append(('BAD', n, f'cofactor {m} outside the pool')); continue
            ps.sort()
            types = [p for p in ps if n % (p * p - 1) == p % (p * p - 1)]
            rigid_ok = all(n % (p * p - 1) in (1, p) for p in ps)
            if not rigid_ok or len(ps) < 3 or n % 2 == 0:
                out.append(('BAD', n, 'definition fails')); continue
            if types != tset:
                out.append(('BAD', n, f'type-p primes {types} != searched {tset}')); continue
            out.append((n, ps, types))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'cn2_nonrigid_all.tsv.gz'))
    ap.add_argument('--procs', type=int, default=24)
    a = ap.parse_args()
    jobs, pools = [], {}
    for f in SOURCES:
        for l in open(os.path.join(HERE, f)):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            vals = r.get('values') or []
            if not vals:
                continue
            L0 = int(r['L0'])
            tset = sorted(r['p0s']) if 'p0s' in r else [int(r['p0'])]
            if L0 not in pools:
                pools[L0] = pool(factorint(L0))
            cands = sorted(set(pools[L0]) | set(tset))
            for i in range(0, len(vals), 2000):
                jobs.append((cands, tset, vals[i:i + 2000]))
    print(f"{len(jobs):,} chunks from {len(pools):,} generators", flush=True)
    rows, times, bad = {}, Counter(), []
    with Pool(a.procs) as pl:
        for res in pl.imap_unordered(check, jobs, chunksize=4):
            for x in res:
                if x[0] == 'BAD':
                    bad.append((str(x[1]), x[2]))
                    continue
                n, ps, types = x
                times[n] += 1
                rows.setdefault(n, (ps, types))
    if bad:
        print(f"{len(bad)} values FAILED verification, e.g. {bad[:3]}")
    with gzip.open(a.out, 'wt') as fh:
        fh.write("log10\tk\ttype_p\tL_R\ttimes_found\tn\tprimes\n")
        for n in sorted(rows):
            ps, types = rows[n]
            LR = 1
            for q in ps:
                if q not in types:
                    LR = lcm(LR, q * q - 1)
            fh.write(f"{math.log10(n):.3f}\t{len(ps)}\t{','.join(map(str, types))}\t{LR}\t{times[n]}\t{n}\t"
                     f"{' '.join(map(str, ps))}\n")
    by_t = Counter(len(t) for _, t in rows.values())
    by_p = Counter(t[0] for _, t in rows.values() if len(t) == 1)
    ns = sorted(rows)
    summ = dict(file=os.path.basename(a.out), distinct=len(rows), by_number_of_type_p_primes=dict(sorted(by_t.items())),
                single_type_p_by_prime=dict(sorted(by_p.items())), failed_verification=len(bad),
                smallest=str(ns[0]), smallest_log10=round(math.log10(ns[0]), 3), largest_log10=round(math.log10(ns[-1]), 3),
                below_howe=sum(1 for n in ns if n < 392000251605356793349050844538065236557716721692385776886401),
                sources=SOURCES)
    json.dump(summ, open(a.out.replace('.tsv.gz', '.summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
