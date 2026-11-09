#!/usr/bin/env python3
"""cn2_nonrigid_scout.py -- which generators could carry a Howe-style NON-RIGID order-2 Carmichael number?

Howe (1998, section "Examples of non-rigid Carmichael numbers"): take a pool generator L0 and a prime p0
with p0 !| L0 and gcd(L0, p0^2-1) | p0-1.  Then every squarefree n0 built from P(2, L0) with
    n0 == 1 (mod p0^2-1)   and   p0*n0 == 1 (mod L0)
gives n = p0*n0 with n == 1 (mod q^2-1) for q | n0 and n == p0 (mod p0^2-1): order 2, not rigid.
With L = lcm(L0, p0^2-1) the expected number of such n is 2^|P| / phi(L) = 2^(F(L0) - Delta),
    F(L0)  = |P(2,L0)| - log2 phi(L0)          (Howe's fecundity; 2^F rigid CN2s expected)
    Delta  = log2(phi(L) / phi(L0))            (the price of the type-p prime, in bits)
The gcd condition, prime by prime: p0 == 1 (mod 2^v2(L0)), and no odd prime of L0 divides p0+1
(the odd part of p0+1 is "private").

For each generator this lists every admissible p0 below a bound, prices it, and reports the best
p0 and the total expectation.  Sources:
    --master [--fmin F]     generators in cn2_master.json with stored fecundity > F
    --result result.json    the canonical L = lcm(q^2-1) of every n found by a split run
    --L L [L ...]           explicit generators
"""
import argparse
import json
import math
import os
from math import gcd, isqrt, lcm, log2

import numpy as np
from sympy import factorint, isprime

HERE = os.path.dirname(os.path.abspath(__file__))


def divisors_from(f):
    ds = [1]
    for p, e in f.items():
        ds = [d * p ** k for d in ds for k in range(e + 1)]
    return ds


def pool(f):
    """P(2, L): primes q !| L with q^2 - 1 | L (Howe's pool; a q | L could never divide n == c mod L)"""
    P = []
    for d in divisors_from(f):
        q = isqrt(d + 1)
        if q * q == d + 1 and q > 2 and q not in f and isprime(q):   # q | L0 can never divide n
            P.append(q)
    return sorted(P)


def log2phi(f):
    return sum(log2(p - 1) + (e - 1) * log2(p) for p, e in f.items())


def merge(f, g):
    h = dict(f)
    for p, e in g.items():
        h[p] = max(h.get(p, 0), e)
    return h


_SIEVE = {}


def primes_upto(n):
    if n not in _SIEVE:
        s = np.ones(n + 1, dtype=bool)
        s[:2] = False
        for i in range(2, isqrt(n) + 1):
            if s[i]:
                s[i * i::i] = False
        _SIEVE.clear()
        _SIEVE[n] = np.nonzero(s)[0].astype(np.int64)
    return _SIEVE[n]


def scout(L0, fL0=None, bound=None, pmax_cap=2 * 10 ** 7):
    f = fL0 or factorint(L0)
    P = pool(f)
    lp = log2phi(f)
    F = len(P) - lp
    # Delta >= log2 phi((p0+1)/2) ~ log2 p0 - 1, so p0 beyond 2^(F+6) cannot give E > 2^-5
    B = bound or int(min(pmax_cap, max(1000, 2 ** (F + 6))))
    pr = primes_upto(max(B, 3))
    pr = pr[pr <= B]
    e2 = f.get(2, 0)
    ok = (pr % (1 << e2) == 1) & (L0 % pr != 0)
    for r in f:
        if r != 2:
            ok &= (pr % r) != (r - 1)            # r | p0+1 is forbidden
    cands = []
    for p0 in pr[ok].tolist():
        fa, fb = factorint(p0 - 1), factorint(p0 + 1)
        g2 = merge(fa, fb)                       # odd parts of p0-1 and p0+1 are coprime ...
        g2[2] = fa.get(2, 0) + fb.get(2, 0)      # ... but their 2-powers multiply
        fl = merge(f, g2)
        assert (p0 - 1) % gcd(L0, p0 * p0 - 1) == 0
        delta = log2phi(fl) - lp
        cands.append((F - delta, p0, delta))
    cands.sort(reverse=True)
    tot = sum(2 ** x[0] for x in cands)
    return dict(L0=L0, P=len(P), F=F, bound=B, n_p0=len(cands), E_total=tot,
                best=[(p0, round(d, 2), round(2 ** e, 3)) for e, p0, d in cands[:5]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--master', action='store_true')
    ap.add_argument('--fmin', type=float, default=3.0)
    ap.add_argument('--result')
    ap.add_argument('--L', type=int, nargs='*')
    ap.add_argument('--bound', type=int)
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--json')
    a = ap.parse_args()
    jobs = []                                    # (label, L0, factors or None)
    if a.master:
        g = json.load(open(os.path.join(HERE, 'cn2_master.json')))['generators']
        for k, v in g.items():
            if v['fecundity'] > a.fmin:
                jobs.append((f"count {v['count']}", int(k), {int(p): e for p, e in v['factors'].items()}))
    if a.result:
        for s in json.load(open(a.result))['found']:
            n = int(s)
            f = {}
            for q in factorint(n):
                f = merge(f, factorint(q * q - 1))
            jobs.append((f"n={s}", math.prod(p ** e for p, e in f.items()), f))
    for L in a.L or []:
        jobs.append(('cli', L, None))
    seen, rows = set(), []
    for label, L0, f in jobs:
        if L0 in seen:
            continue
        seen.add(L0)
        r = scout(L0, f, a.bound)
        r['label'] = label
        rows.append(r)
    rows.sort(key=lambda r: -r['E_total'])
    print(f"{'L0':>20} {'|P|':>4} {'F':>6} {'#p0':>5} {'E_total':>9}  best p0 (Delta, E)            source")
    for r in rows[:a.top]:
        b = '  '.join(f"{p}({d},{e})" for p, d, e in r['best'][:3])
        print(f"{r['L0']:>20} {r['P']:>4} {r['F']:>6.2f} {r['n_p0']:>5} {r['E_total']:>9.3f}  {b:<40} {r['label'][:40]}")
    print(f"\n{len(rows)} generators; sum of E_total = {sum(r['E_total'] for r in rows):.2f}")
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(rows, fh, indent=1, default=str)


if __name__ == '__main__':
    main()
