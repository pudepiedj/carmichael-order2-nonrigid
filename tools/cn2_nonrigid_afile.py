#!/usr/bin/env python3
"""cn2_nonrigid_afile.py -- union of every single-type-p non-rigid search, and regeneration of Part 1 of the a-file.

    python3 cn2_nonrigid_afile.py [--write]

Reads the single-p0 result files (values lists), forms the distinct union, ranks Howe's example, and re-verifies every
number below Howe's from scratch (squarefree; k == 1 or p mod p^2-1 for every p | k; at least one type-p prime).
With --write, Part 1 of a175531_nonrigid_examples.txt is rebuilt; its header lines and Parts 2-3 are kept verbatim.
"""
import json, math, os, re, sys
from sympy import factorint

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ['cn2_nonrigid_runs.jsonl', 'cn2_box_nonrigid.jsonl', 'cn2_xbox_nonrigid.jsonl',
           'cn2_box_nonrigid_smallp.jsonl', 'cn2_ey_top10_runs.jsonl']
HOWE = 392000251605356793349050844538065236557716721692385776886401
AFILE = os.path.join(HERE, 'a175531_nonrigid_examples.txt')


def check(k):
    """(type-p primes, prime factors) if k is a non-rigid order-2 Carmichael number, else None"""
    f = factorint(k, limit=10 ** 7)
    if any(e > 1 for e in f.values()) or not all(p < 10 ** 14 for p in f) or math.prod(f) != k or len(f) < 3:
        return None
    ps = sorted(f)
    T = []
    for p in ps:
        r = k % (p * p - 1)
        if r == p:
            T.append(p)
        elif r != 1:
            return None
    return (T, ps) if T else None


def main():
    allv, per = set(), {}
    for s in SOURCES:
        vs = set()
        for l in open(os.path.join(HERE, s)):
            try: r = json.loads(l)
            except ValueError: continue
            vs.update(int(v) for v in r.get('values', []))
        per[s] = vs
        allv |= vs
        print(f"{s:34} {len(vs):>9,} distinct  smallest 10^{math.log10(min(vs)):.2f}" if vs else s)
    sub = sorted(v for v in allv if v < HOWE)
    print(f"union {len(allv):,} distinct; {len(sub)} below Howe's 10^{math.log10(HOWE):.2f}, which ranks {len(sub) + 1}")
    rows = []
    for k in sub:
        c = check(k)
        if c is None:
            sys.exit(f"VERIFICATION FAILED: {k}")
        rows.append(f"{k} {math.log10(k):.2f} {'.'.join(map(str, c[0]))} {' '.join(map(str, c[1]))}")
    print(f"all {len(rows)} sub-Howe numbers verified")
    if '--write' in sys.argv:
        lines = open(AFILE).read().split('\n')
        i1 = next(i for i, l in enumerate(lines) if l.startswith('# Part 1'))
        i2 = next(i for i, l in enumerate(lines) if l.startswith('# Part 2'))
        head = lines[:i1] + [re.sub(r'the \d+ non-rigid', f'the {len(rows)} non-rigid', lines[i1]), lines[i1 + 1]]
        open(AFILE, 'w').write('\n'.join(head + rows + ['#'] + lines[i2:]))
        print(f"wrote {AFILE}")


if __name__ == '__main__':
    main()
