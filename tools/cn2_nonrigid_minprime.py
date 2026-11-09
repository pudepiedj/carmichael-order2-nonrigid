#!/usr/bin/env python3
"""cn2_nonrigid_minprime.py -- distribution of the SMALLEST prime factor over every non-rigid order-2 number found.

    python3 cn2_nonrigid_minprime.py [--json cn2_nonrigid_minprime.json]

Reads the same five result files as cn2_nonrigid_afile.py, deduplicates the values, and finds each number's smallest
prime by trial division in increasing order (every prime of n is a pool prime, so this stops after a few steps).
Prints the distribution overall and by size decade band, next to the rigid census below 1e30 for comparison.
"""
import argparse, json, math
from collections import Counter, defaultdict
from sympy import primerange, factorint

SOURCES = ['cn2_nonrigid_runs.jsonl', 'cn2_box_nonrigid.jsonl', 'cn2_xbox_nonrigid.jsonl',
           'cn2_box_nonrigid_smallp.jsonl', 'cn2_ey_top10_runs.jsonl']
PR = list(primerange(5, 200000))


def smallest(n):
    for p in PR:
        if n % p == 0:
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', default='cn2_nonrigid_minprime.json')
    a = ap.parse_args()
    vals = set()
    for f in SOURCES:
        for line in open(f):
            r = json.loads(line)
            for v in r.get('values', []):
                vals.add(int(v))
    sp = {n: smallest(n) for n in vals}
    tot = Counter(sp.values())
    band = defaultdict(Counter)
    for n, p in sp.items():
        band[10 * (int(math.log10(n)) // 10)][p] += 1
    rigid = Counter(min(factorint(int(l))) for l in open('cn2x/gpu/known_1e30.txt'))
    N = len(vals)
    print(f"{N:,} distinct non-rigid numbers; smallest prime not found below 2e5: {tot.get(None, 0)}")
    print(f"\n{'p_1':>6} {'non-rigid':>10} {'share':>7} {'cum':>7}   {'rigid<1e30':>10}")
    cum = 0
    for p in sorted(k for k in set(tot) | set(rigid) if k):
        cum += tot.get(p, 0)
        if tot.get(p, 0) / N >= 0.001 or rigid.get(p):
            print(f"{p:>6} {tot.get(p, 0):>10,} {tot.get(p, 0) / N:>7.2%} {cum / N:>7.2%}   {rigid.get(p, 0):>10}")
    print(f"\nby size band (rows: 10^b <= n < 10^(b+10)); columns: share with smallest prime p")
    cols = [11, 13, 17, 19, 23, 29, 31, 37]
    print(f"{'band':>6} {'count':>9} " + ' '.join(f"{c:>6}" for c in cols) + f" {'>37':>6}")
    for b in sorted(band):
        c = band[b]; m = sum(c.values())
        print(f"{b:>6} {m:>9,} " + ' '.join(f"{c.get(q, 0) / m:>6.1%}" for q in cols)
              + f" {sum(v for k, v in c.items() if k and k > 37) / m:>6.1%}")
    json.dump(dict(n=N, overall={str(k): v for k, v in sorted(tot.items(), key=lambda kv: kv[0] or 0)},
                   by_band={str(b): {str(k): v for k, v in c.items()} for b, c in band.items()},
                   rigid_below_1e30={str(k): v for k, v in rigid.items()}), open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
