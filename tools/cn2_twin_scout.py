#!/usr/bin/env python3
"""cn2_twin_scout.py -- price every compatible PAIR of type-p primes on every box generator (two-type-p numbers).

    python3 cn2_twin_scout.py [--pmax 85] [--p0max 100000] [--emin 0.001] [--out cn2_twin_pairs.json]

For a generator L0 and type-p primes p0 < p1 the number n = p0*p1*n0 must satisfy n == 1 (mod L0), n == p0 (mod
p0^2-1), n == p1 (mod p1^2-1); these are compatible iff every pairwise gcd of the moduli divides the difference of
the residues.  Expected count 2^|P'| / phi(lcm(L0, p0^2-1, p1^2-1)), |P'| = pool minus primes dividing p0^2-1 or
p1^2-1 (the same model that predicted 180,777 vs 180,938 for single type-p primes).
"""
import argparse, csv, json, math, os
import numpy as np
from sympy import factorint, primerange

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pmax', type=int, default=85)
    ap.add_argument('--p0max', type=int, default=100000)
    ap.add_argument('--emin', type=float, default=0.001)
    ap.add_argument('--out', default=os.path.join(HERE, 'cn2_twin_pairs.json'))
    a = ap.parse_args()
    from cn2_nonrigid_scout import pool, log2phi, merge
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, 'cn2_box_fecundity.csv'))) if int(r['P']) <= a.pmax]
    # a pair costs >= ~15 bits, so only generators with F > 15 + log2(emin) can reach emin
    rows = [r for r in rows if float(r['F']) > 15 + math.log2(a.emin)]
    cand = [p for p in primerange(3, a.p0max) if p % 8 == 1 and p % 3 == 1]   # p == 1 mod 24 is necessary
    fac = {p: merge(factorint(p - 1), factorint(p + 1)) for p in cand}
    for p in cand:
        fac[p][2] = factorint(p - 1).get(2, 0) + factorint(p + 1).get(2, 0)
    print(f"{len(rows):,} generators with |P| <= {a.pmax} and F > {15 + math.log2(a.emin):.1f}; "
          f"{len(cand):,} candidate type-p primes < {a.p0max}", flush=True)
    out, tot = [], 0.0
    for k, r in enumerate(rows):
        L0 = int(r['L']); f = factorint(L0); P = pool(f); lp = log2phi(f); F = float(r['F'])
        adm = [p for p in cand if L0 % p and (p - 1) % math.gcd(L0, p * p - 1) == 0]
        # single-prime price first; a pair costs at least the sum minus shared parts, so prune by the cheaper pair bound
        d1 = {p: log2phi(merge(f, fac[p])) - lp for p in adm}
        adm = [p for p in adm if F - d1[p] > math.log2(a.emin) - 0]      # each alone must be affordable
        for i, p0 in enumerate(adm):
            for p1 in adm[i + 1:]:
                g0, g1 = p0 * p0 - 1, p1 * p1 - 1
                if (p1 - p0) % math.gcd(g0, g1) or g0 % p1 == 0 or g1 % p0 == 0:
                    continue
                fl = merge(merge(f, fac[p0]), fac[p1])
                Pe = sum(1 for q in P if g0 % q and g1 % q and q not in (p0, p1))
                E = 2.0 ** (Pe - log2phi(fl))
                tot += E
                if E >= a.emin:
                    out.append(dict(L0=str(L0), p0s=[p0, p1], P=Pe, E=E, F=F))
        if (k + 1) % 2000 == 0:
            print(f"  [{k + 1:,}/{len(rows):,}] pairs kept {len(out):,}, total E {tot:.2f}", flush=True)
    out.sort(key=lambda d: -d['E'])
    json.dump(out, open(a.out, 'w'), indent=0)
    print(f"total expected two-type-p numbers over all compatible pairs: {tot:.2f}; kept {len(out):,} with E >= "
          f"{a.emin} (sum {sum(d['E'] for d in out):.2f}) -> {a.out}")
    for d in out[:15]:
        print(f"  E={d['E']:8.3f}  |P'|={d['P']:>3}  F={d['F']:6.2f}  p0s={d['p0s']}  L0={d['L0']}")


if __name__ == '__main__':
    main()
