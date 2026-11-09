#!/usr/bin/env python3
"""cn2_nonrigid_search.py -- Howe's construction of NON-RIGID order-2 Carmichael numbers, by Schroeppel-Shamir.

    python3 cn2_nonrigid_search.py L0 p0 [p0 ...] [--show 5] [--json runs.jsonl]
    python3 cn2_nonrigid_search.py 2^7.3^3.5^2.7.11.13.17.19.29.31 1153      # Howe 1998: 53 found

Howe (1998), "Examples of non-rigid Carmichael numbers": given a pool generator L0 and a prime p0 !| L0
with gcd(L0, p0^2-1) | p0-1, find every squarefree n0 over P(2, L0) with
    n0 == 1 (mod p0^2-1)   and   p0*n0 == 1 (mod L0).
Then n = p0*n0 == 1 (mod q^2-1) for q | n0 and n == p0 (mod p0^2-1): order 2 (Howe), not rigid.
With L = lcm(L0, p0^2-1) and c the CRT residue (c == 1 mod L0, c == p0 mod p0^2-1) this is one
subset-product problem  prod(S) == c * p0^-1 (mod L)  over the pool, solved by the four-list engine
cn2_systematic_fast_SS.py with a target in place of 1.  Pool primes dividing p0^2-1 are removed
(n == -1 or 1 modulo them would contradict q | n).

Every hit is re-verified from scratch (squarefree, n == 1 or p mod p^2-1 at every prime, exactly
one type-p prime).  Expected count 2^|P| / phi(L) (cf. cn2_nonrigid_scout.py).
"""
import argparse
import json
import math
import time

from sympy import factorint, isprime
from sympy.ntheory.modular import crt

from cn2_systematic_fast_SS import evaluate_generator, calculate_P, parse_generator, phi_from_factors


def howe_types(n, ps):
    """{p: '1' | 'p' | None} for n == 1 / p (mod p^2-1)"""
    out = {}
    for p in ps:
        r = n % (p * p - 1)
        out[p] = '1' if r == 1 else ('p' if r == p else None)
    return out


def search(L0, L0f, p0, collect=64, verbose=False, min_P_size=4):
    if not isprime(p0) or L0 % p0 == 0 or (p0 - 1) % math.gcd(L0, p0 * p0 - 1):
        raise ValueError(f"p0 = {p0} is not admissible for L0 = {L0}")
    g = p0 * p0 - 1
    L = math.lcm(L0, g)
    Lf = factorint(L)
    P = [q for q in calculate_P(L0f) if g % q]
    c = int(crt([L0, g], [1, p0])[0])                # the residue of n mod L
    target = c * pow(p0, -1, L) % L                  # ... of n0 = n / p0
    t0 = time.time()
    r = evaluate_generator(L, Lf, target=target, P_override=P, collect_hits=collect, verbose=verbose,
                           min_P_size=min_P_size)
    found = []
    for n0 in r.cn2_values:
        n = p0 * n0
        ps = sorted([q for q in P if n0 % q == 0] + [p0])   # n0 is a product of pool primes by construction;
                                                           # the prod(ps) == n test below confirms it is complete
        ty = howe_types(n, ps)
        ok = math.prod(ps) == n and all(ty.values()) and [p for p in ps if ty[p] == 'p'] == [p0]
        found.append((n, ps, ok))
    found.sort()
    return dict(L0=L0, p0=p0, L=L, P=len(P), predicted=2.0 ** len(P) / phi_from_factors(Lf),
                count=r.cn2_count, collected=len(found), verified=sum(ok for _, _, ok in found),
                seconds=time.time() - t0, found=found)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('L0', help="generator: 25113661372800 or 2^7.3^3.5^2...")
    ap.add_argument('p0', type=int, nargs='+')
    ap.add_argument('--show', type=int, default=3)
    ap.add_argument('--collect', type=int, default=64, help='hits kept per engine chunk')
    ap.add_argument('--json', help='append one JSON line per (L0, p0)')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    L0, L0f = parse_generator(a.L0)
    for p0 in a.p0:
        r = search(L0, L0f, p0, a.collect, a.verbose)
        print(f"L0 = {L0}  p0 = {p0}  L = {r['L']}  |P| = {r['P']}  predicted {r['predicted']:.2f}  "
              f"found {r['count']}  (collected {r['collected']}, verified {r['verified']})  {r['seconds']:.1f}s")
        if r['collected'] < r['count']:
            print(f"  WARNING: only {r['collected']} of {r['count']} hits collected; raise --collect")
        for n, ps, ok in r['found'][:a.show]:
            print(f"  {'OK ' if ok else 'BAD'} 10^{math.log10(n):.2f}  k={len(ps)}  n = {n}")
            print(f"      {', '.join(map(str, ps))}")
        if a.json:
            with open(a.json, 'a') as fh:
                rec = {k: v for k, v in r.items() if k != 'found'}
                rec['smallest'] = str(r['found'][0][0]) if r['found'] else None
                rec['values'] = [str(n) for n, _, _ in r['found']]
                fh.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()


def search_multi(L0, L0f, p0s, collect=64, verbose=False, min_P_size=1):
    """Howe's construction with SEVERAL type-p primes: n = prod(p0s) * n0, n0 squarefree over P(2, L0), with
    n == 1 (mod q^2-1) for every q | n0 and n == p (mod p^2-1) for every p in p0s.  The residue of n modulo
    L = lcm(L0, p^2-1 for p in p0s) is fixed by CRT; the conditions are compatible iff every pairwise gcd of the
    moduli divides the difference of the residues (checked; raises ValueError otherwise).  Every hit is verified
    exactly: complete factorisation into distinct primes, and the set of type-p primes is exactly p0s."""
    p0s = sorted(p0s)
    mods, res = [L0], [1]
    for p in p0s:
        if not isprime(p) or L0 % p == 0:
            raise ValueError(f"{p} is not prime or divides L0")
        mods.append(p * p - 1)
        res.append(p)
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            if (res[i] - res[j]) % math.gcd(mods[i], mods[j]):
                raise ValueError(f"incompatible: moduli {mods[i]}, {mods[j]}")
    for p in p0s:                                       # a type-p prime may not divide another's p^2-1
        if any((q * q - 1) % p == 0 for q in p0s if q != p):
            raise ValueError(f"{p} divides another type-p prime's p^2-1")
    L = math.lcm(*mods)
    Lf = factorint(L)
    P = [q for q in calculate_P(L0f) if all((p * p - 1) % q for p in p0s) and q not in p0s]
    c = int(crt(mods, res)[0])
    m = math.prod(p0s)
    target = c * pow(m, -1, L) % L
    t0 = time.time()
    r = evaluate_generator(L, Lf, target=target, P_override=P, collect_hits=collect, verbose=verbose,
                           min_P_size=min_P_size)
    found = []
    for n0 in r.cn2_values:
        n = m * n0
        ps = sorted([q for q in P if n0 % q == 0] + p0s)
        ty = howe_types(n, ps)
        ok = math.prod(ps) == n and all(ty.values()) and [p for p in ps if ty[p] == 'p'] == p0s
        found.append((n, ps, ok))
    found.sort()
    return dict(L0=L0, p0s=p0s, L=L, P=len(P), predicted=2.0 ** len(P) / phi_from_factors(Lf),
                count=r.cn2_count, collected=len(found), verified=sum(ok for _, _, ok in found),
                seconds=time.time() - t0, found=found)
