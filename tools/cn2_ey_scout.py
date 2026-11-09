#!/usr/bin/env python3
"""cn2_ey_scout.py -- size-resolved expectation E_Y for non-rigid (L0, p0) pairs: where are numbers below Y likely?

    python3 cn2_ey_scout.py --p0 9601,14593,... [--Y 4.63e36] [--l2max 50] [--box cn2_box_fecundity.csv]
                            [--out cn2_ey_pairs.json] [--procs 30]

Total fecundity predicts how MANY members a family has; it says nothing about how SMALL the smallest is.  For a pair
(L0, p0) the expected number of non-rigid n = p0*n0 < Y is

    E_Y = #{ nonempty S subset of P' : prod(S) < Y/p0 } / phi(L),     L = lcm(L0, p0^2-1),

with P' the pool of L0 minus primes dividing p0^2-1 (the uniformity model that predicts total counts to within a few
percent).  The numerator is a subset-sum count over log2 p, done by dynamic programming on a grid of 1/RES bit; it
GROWS with the pool (more primes mean more small subsets too), so large pools are not penalised a priori - only phi(L)
weighs against them.  Output: every admissible pair with its E_Y, pool size and total expectation, sorted by E_Y.
"""
import argparse
import csv
import json
import math
import os
from multiprocessing import Pool

import numpy as np
from sympy import factorint

from cn2_nonrigid_scout import pool, log2phi, merge

HERE = os.path.dirname(os.path.abspath(__file__))
RES = 64                                   # grid cells per bit of log2


def small_subsets(primes, log2_bound):
    """number of nonempty subsets with log2(product) < log2_bound (DP on a 1/RES-bit grid; rounding up, so a
    subset is counted only if it is below the bound even after rounding: a slight undercount, never an overcount)"""
    cap = int(math.floor(log2_bound * RES))
    if cap <= 0:
        return 0.0
    w = [int(math.ceil(math.log2(q) * RES)) for q in primes]
    ways = np.zeros(cap, dtype=np.float64)                     # ways[s] = #subsets with weight exactly s (s < cap)
    ways[0] = 1.0
    for x in sorted(w):
        if x >= cap:
            break
        ways[x:] += ways[:cap - x].copy()
    return float(ways.sum() - 1.0)                              # drop the empty subset


def work(args):
    L0, p0s, logY = args
    f = factorint(L0)
    P = pool(f)
    lp = log2phi(f)
    out = []
    for p0 in p0s:
        if L0 % p0 == 0 or (p0 - 1) % math.gcd(L0, p0 * p0 - 1):
            continue
        fa, fb = factorint(p0 - 1), factorint(p0 + 1)
        g = merge(fa, fb)
        g[2] = fa.get(2, 0) + fb.get(2, 0)
        Pe = [q for q in P if (p0 * p0 - 1) % q]
        l2phiL = log2phi(merge(f, g))
        n_small = small_subsets(Pe, logY - math.log2(p0))
        EY = n_small / 2.0 ** l2phiL if n_small else 0.0
        out.append(dict(L0=str(L0), p0=p0, P=len(Pe), EY=EY, E=2.0 ** (len(Pe) - l2phiL)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--p0', required=True)
    ap.add_argument('--Y', type=float, default=4.63e36)
    ap.add_argument('--l2max', type=float, default=50)
    ap.add_argument('--box', default=os.path.join(HERE, 'cn2_box_fecundity.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'cn2_ey_pairs.json'))
    ap.add_argument('--procs', type=int, default=30)
    a = ap.parse_args()
    p0s = [int(x) for x in a.p0.split(',')]
    logY = math.log2(a.Y)
    Ls = [int(r['L']) for r in csv.DictReader(open(a.box)) if float(r['log2L']) <= a.l2max]
    with Pool(a.procs) as pl:
        res = [x for chunk in pl.imap_unordered(work, [(L, p0s, logY) for L in Ls], chunksize=256) for x in chunk]
    res.sort(key=lambda d: -d['EY'])
    json.dump(res, open(a.out, 'w'))
    tot = sum(d['EY'] for d in res)
    print(f"{len(Ls):,} generators, {len(res):,} admissible pairs; total E_Y (Y = {a.Y:.3g}) = {tot:.4f}; "
          f"total E (any size) = {sum(d['E'] for d in res):.1f}")
    by_P = {}
    for d in res:
        b = d['P'] // 10 * 10
        by_P[b] = by_P.get(b, 0.0) + d['EY']
    print("E_Y by pool size:", {f"{b}-{b + 9}": round(v, 4) for b, v in sorted(by_P.items())})
    acc = 0.0
    for i, d in enumerate(res):
        acc += d['EY']
        if acc >= 0.99 * tot:
            print(f"top {i + 1:,} pairs carry 99% of E_Y (smallest kept E_Y {d['EY']:.2e})")
            break
    for d in res[:8]:
        print(f"  E_Y={d['EY']:.4f}  E={d['E']:9.2f}  |P'|={d['P']:>3}  p0={d['p0']:<6} L0={d['L0']}")


if __name__ == '__main__':
    main()
