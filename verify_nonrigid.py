#!/usr/bin/env python3
"""verify_nonrigid.py -- check every number in the non-rigid dataset from scratch, using only the standard library.

    python3 verify_nonrigid.py cn2_nonrigid_all.tsv.gz [--afile data/a175531_nonrigid_examples.txt] [--procs N]

The dataset is the release asset cn2_nonrigid_all.tsv.gz: gzip-compressed, tab-separated, one header line, then one
row per number, sorted by n:

    log10  k  type_p  L_R  times_found  n  primes

For every row this checks that:
  - the listed primes are distinct and each is prime (deterministic Miller-Rabin; every prime here is far below 3.3e24);
  - k is the number of primes, their product is n, and log10 is log10(n) to 3 decimals;
  - n is a Carmichael number of order 2 (Howe's definition): n == 1 or p (mod p^2 - 1) for every prime p | n;
  - the type-p primes (those with n == p, not 1, mod p^2 - 1) are exactly the listed type_p, and there is at least one,
    so n is non-rigid;
  - L_R is lcm(q^2 - 1) over the other (rigid) primes q;
  - the rows are strictly increasing in n, so no number is listed twice.
With --afile it also checks that every number in the a-file is in the dataset with the same type-p primes.
Prints a summary and exits non-zero on any failure.
"""
import argparse, gzip, json, math, sys
from collections import Counter
from multiprocessing import Pool

MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)   # deterministic for n < 3.3e24


def is_prime(n):
    if n < 2:
        return False
    for p in MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def check_row(line):
    """-> (n, type_p tuple, None) if the row is good, else (n or None, None, reason)."""
    try:
        f = line.rstrip('\n').split('\t')
        if len(f) != 7:
            return None, None, f'{len(f)} fields'
        lg, k, tp, lr, times, n, ps = f
        n = int(n)
        primes = [int(x) for x in ps.split()]
        tset = tuple(sorted(int(x) for x in tp.split(',')))
        if len(set(primes)) != len(primes):
            return n, None, 'repeated prime'
        if not all(p < 3 * 10**24 and is_prime(p) for p in primes):
            return n, None, 'a listed factor is not prime'
        if int(k) != len(primes):
            return n, None, 'k is not the number of primes'
        if math.prod(primes) != n:
            return n, None, 'product of primes is not n'
        if f'{math.log10(n):.3f}' != lg:
            return n, None, f'log10 {lg} != {math.log10(n):.3f}'
        typep, L = [], 1
        for p in primes:
            r = n % (p * p - 1)
            if r == p:
                typep.append(p)
            elif r == 1:
                L = math.lcm(L, p * p - 1)
            else:
                return n, None, f'not of order 2 at p = {p}'
        if not typep:
            return n, None, 'rigid (no type-p prime)'
        if tuple(sorted(typep)) != tset:
            return n, None, f'type-p primes {sorted(typep)} != listed {tp}'
        if L != int(lr):
            return n, None, 'L_R is not lcm(q^2-1) over the rigid primes'
        if int(times) < 1:
            return n, None, 'times_found < 1'
        return n, tset, None
    except Exception as e:                                        # malformed row
        return None, None, f'unparseable: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--afile')
    ap.add_argument('--procs', type=int, default=None)
    a = ap.parse_args()
    with gzip.open(a.data, 'rt') as fh:
        header = fh.readline().rstrip('\n').split('\t')
        rows = fh.readlines()
    if header != ['log10', 'k', 'type_p', 'L_R', 'times_found', 'n', 'primes']:
        sys.exit(f'unexpected header {header}')
    with Pool(a.procs) as pool:
        res = pool.map(check_row, rows, chunksize=2000)
    bad = [(i + 2, n, why) for i, (n, _, why) in enumerate(res) if why]
    prev, order = 0, 0
    for n, _, why in res:
        if n is not None and not why:
            order += n <= prev
            prev = n
    good = {n: t for n, t, why in res if not why}
    by_count = Counter(len(t) for t in good.values())
    single = Counter(t[0] for t in good.values() if len(t) == 1)
    smallest = min(good) if good else None
    print(json.dumps({'rows': len(rows), 'verified': len(good), 'failed': len(bad), 'order_violations': order,
                      'by_number_of_type_p_primes': dict(sorted(by_count.items())),
                      'distinct_single_type_p_primes': len(single),
                      'smallest': str(smallest), 'smallest_log10': round(math.log10(smallest), 3) if smallest else None,
                      'largest_log10': round(math.log10(max(good)), 3) if good else None}, indent=1))
    for line, n, why in bad[:20]:
        print(f'FAIL line {line}: n = {n}: {why}')
    afail = 0
    if a.afile:
        seen = 0
        for line in open(a.afile):
            if not line.strip() or line.startswith('#'):
                continue
            t = line.split()
            n, tset = int(t[0]), tuple(sorted(int(x) for x in t[2].split('.')))
            seen += 1
            if good.get(n) != tset:
                afail += 1
                print(f'A-FILE: {t[0][:30]}... not in the dataset with type-p primes {t[2]}')
        print(f'a-file: {seen} numbers, {seen - afail} found in the dataset with the same type-p primes')
    ok = not bad and not order and not afail
    print('ALL ROWS VERIFIED' if ok else 'VERIFICATION FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
