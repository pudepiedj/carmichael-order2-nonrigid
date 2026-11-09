#!/usr/bin/env python3
"""cn2_box_fecundity.py -- pool size and fecundity for EVERY generator in an exponent box, without divisor loops.

    python3 cn2_box_fecundity.py                         # the box spanned by cn2_master.json's 31-smooth generators
    python3 cn2_box_fecundity.py --caps 2:16,3:4,5:3     # widen some exponents (others keep their defaults)
    python3 cn2_box_fecundity.py --top 40 --out cn2_box_fecundity.csv

The box is every L = prod p^e_p over p in PRIMES with lo_p <= e_p <= hi_p.  The pool P(2, L) is the set of primes
q !| L with q^2-1 | L.  Every such q has q^2-1 | Lmax (the top corner of the box), so the candidate q's are found
once, from the divisors d of Lmax with d+1 a prime square; each carries its exponent vector v(q).  Then for every
box point, q is in the pool iff v(q) <= e(L) componentwise and q !| L -- a vectorised comparison, no per-L divisor
enumeration (the "divisor explosion" that stopped the earlier scripts).

    F(L) = |P(2,L)| - log2 phi(L)          (Howe's fecundity; about 2^F rigid CN2 children expected)

Output: a summary by log2 L band (box size, how many are in cn2_master.json, how many have F > 0 and are missing),
the highest-F generators the master lacks, and optionally a CSV of every box point.  A self-check compares F with
the fecundity stored in cn2_master.json for every master generator inside the box.
"""
import argparse
import itertools
import json
import math
import os

import numpy as np
from sympy import isprime

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
DEFAULT = {2: (4, 11), 3: (2, 4), 5: (1, 3), 7: (0, 2), 11: (0, 2), 13: (0, 2), 17: (0, 2), 19: (0, 2),
           23: (0, 2), 29: (0, 1), 31: (0, 1)}


def parse_caps(s):
    box = dict(DEFAULT)
    for part in filter(None, (s or '').split(',')):
        p, _, r = part.partition(':')
        lo, _, hi = r.partition('-')
        p = int(p)
        box[p] = (int(lo), int(hi)) if hi else (box.get(p, (0, 0))[0], int(lo))
    return box


def pool_candidates(box):
    """primes q (not in PRIMES) with q^2-1 | Lmax, and their exponent vectors over PRIMES"""
    divs = [(1, (0,) * len(PRIMES))]
    for i, p in enumerate(PRIMES):
        hi = box[p][1]
        new = []
        for d, v in divs:
            pk = 1
            for k in range(hi + 1):
                w = list(v); w[i] = k
                new.append((d * pk, tuple(w)))
                pk *= p
        divs = new
    qs = []
    for d, v in divs:
        q = math.isqrt(d + 1)
        if q * q == d + 1 and q > 2 and isprime(q):
            qs.append((q, v))
    qs.sort()
    return qs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--caps', help="exponent ranges, e.g. '2:4-16,3:2-4' (a single number sets the upper bound)")
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--out', help='CSV of every box point: L, log2L, P, F, in_master')
    ap.add_argument('--extra', type=int, help='one extra prime r > 31, present to the first power in every L')
    ap.add_argument('--l2max', type=float, help='keep only L <= 2^l2max')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    global PRIMES
    box = parse_caps(a.caps)
    if a.extra:
        PRIMES = PRIMES + [a.extra]
        box[a.extra] = (1, 1)
    PR = [p for p in PRIMES if box[p][1] > 0]

    qs = pool_candidates(box)
    Q = np.array([v for _, v in qs], dtype=np.int8)                # (nq, 11)
    qdiv = np.array([PRIMES.index(q) if q in PRIMES else -1 for q, _ in qs])   # a box prime q is in the pool only if q !| L
    print(f"box: {', '.join(f'{p}^{box[p][0]}..{box[p][1]}' for p in PRIMES)}")
    print(f"pool candidates (q^2-1 | Lmax): {len(qs)}, largest {qs[-1][0]}")

    ranges = [range(box[p][0], box[p][1] + 1) for p in PRIMES]
    E = np.array(list(itertools.product(*ranges)), dtype=np.int8)  # (nL, len(PRIMES))
    lp = np.log2(np.array(PRIMES, dtype=float))
    if a.l2max:
        E = E[E @ lp <= a.l2max + 1e-9]
    lpm1 = np.log2(np.array(PRIMES, dtype=float) - 1)
    log2L = E @ lp
    log2phi = ((E > 0) * (lpm1 - lp) + E * lp).sum(axis=1)
    P = np.zeros(len(E), dtype=np.int32)
    for i in range(0, len(E), 4096):                               # |P| = #{q : v(q) <= e(L)}
        blk = E[i:i + 4096]
        fits = (Q[None, :, :] <= blk[:, None, :]).all(axis=2)
        for j in np.nonzero(qdiv >= 0)[0]:
            fits[:, j] &= blk[:, qdiv[j]] == 0
        P[i:i + 4096] = fits.sum(axis=1)
    F = P - log2phi
    Ls = [math.prod(p ** int(e) for p, e in zip(PRIMES, row)) for row in E]

    g = json.load(open(os.path.join(HERE, 'cn2_master.json')))['generators']
    inM = np.array([str(L) in g for L in Ls])
    # self-check against the master's stored fecundity
    diffs = [abs(F[i] - g[str(Ls[i])]['fecundity']) for i in np.nonzero(inM)[0]]
    print(f"box points: {len(E):,}; in master: {int(inM.sum()):,}; "
          + (f"self-check vs stored fecundity: max |dF| = {max(diffs):.2e} over {len(diffs):,}" if diffs
             else "self-check: no master generators in this box"))

    print(f"\n{'log2 L':>7} {'box':>7} {'master':>7} {'F>0':>7} {'F>0 missing':>11} {'F>4 missing':>11}")
    band = np.floor(log2L).astype(int)
    for b in range(band.min(), band.max() + 1):
        s = band == b
        if not s.any():
            continue
        print(f"{b:>7} {int(s.sum()):>7} {int((s & inM).sum()):>7} {int((s & (F > 0)).sum()):>7} "
              f"{int((s & (F > 0) & ~inM).sum()):>11} {int((s & (F > 4) & ~inM).sum()):>11}")
    tot = lambda m: int(m.sum())
    print(f"{'all':>7} {len(E):>7} {tot(inM):>7} {tot(F > 0):>7} {tot((F > 0) & ~inM):>11} {tot((F > 4) & ~inM):>11}")

    miss = np.nonzero(~inM)[0]
    order = miss[np.argsort(-F[miss])][:a.top]
    print(f"\nhighest-fecundity generators NOT in the master:")
    print(f"{'L':>22} {'log2 L':>7} {'|P|':>4} {'F':>7}  factorisation")
    for i in order:
        fac = '.'.join(f"{p}^{int(e)}" if e > 1 else str(p) for p, e in zip(PRIMES, E[i]) if e)
        print(f"{Ls[i]:>22} {log2L[i]:>7.2f} {P[i]:>4} {F[i]:>7.2f}  {fac}")
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write('L,log2L,P,F,in_master\n')
            for i in range(len(E)):
                fh.write(f"{Ls[i]},{log2L[i]:.4f},{P[i]},{F[i]:.4f},{int(inM[i])}\n")
        print(f"\nwrote {a.out}")


if __name__ == '__main__':
    main()
