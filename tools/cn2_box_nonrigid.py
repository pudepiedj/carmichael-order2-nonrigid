#!/usr/bin/env python3
"""cn2_box_nonrigid.py -- Howe's non-rigid construction on EVERY generator of the exponent box, any fecundity.

    python3 cn2_box_nonrigid.py run    [--l2max 50] [--p0 193,673,1153,2017]    # resumable, smallest L0 first
    python3 cn2_box_nonrigid.py report [--top 20]

Generators come from cn2_box_fecundity.csv.  For each generator L0 and each listed type-p prime p0 that is
admissible (p0 !| L0, gcd(L0, p0^2-1) | p0-1), cn2_nonrigid_search.search() finds every squarefree n0 over
P(2, L0) with n = p0*n0 order 2 and p0 its only type-p prime, and re-verifies each from its factorisation.
No fecundity or expectation filter: coverage is exactly (box generators with log2 L0 <= l2max) x (listed p0).
One JSON line per pair in cn2_box_nonrigid.jsonl.
"""
import argparse
import csv
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOX = os.path.join(HERE, 'cn2_box_fecundity.csv')
RUNS = os.path.join(HERE, 'cn2_box_nonrigid.jsonl')
NCH = 1920


def done_set():
    out = set()
    if os.path.exists(RUNS):
        for line in open(RUNS):
            try:
                r = json.loads(line)
                out.add((int(r['L0']), r['p0']))
            except (ValueError, KeyError):
                pass
    return out


def run(l2max, p0s):
    from sympy import factorint
    from cn2_nonrigid_search import search
    from numba import get_num_threads
    global NCH
    NCH = get_num_threads() * 64
    rows = sorted((int(r['L']) for r in csv.DictReader(open(BOX)) if float(r['log2L']) <= l2max))
    done = done_set()
    todo = []
    for L0 in rows:
        for p0 in p0s:
            if (L0, p0) in done or L0 % p0 == 0 or (p0 - 1) % math.gcd(L0, p0 * p0 - 1):
                continue
            todo.append((L0, p0))
    print(f"{len(rows):,} generators, p0 in {p0s}: {len(todo):,} admissible pairs to run ({len(done):,} done)",
          flush=True)
    t0, found, best = time.time(), 0, None
    for i, (L0, p0) in enumerate(todo):
        f = factorint(L0)
        rec = dict(L0=str(L0), p0=p0)
        try:
            cap = 64
            while True:
                r = search(L0, f, p0, cap, min_P_size=1)
                if r['collected'] >= r['count'] or cap >= 1 << 22:
                    break
                cap = max(cap * 4, 1 << math.ceil(math.log2(4 * r['count'] / NCH + 1)))   # 4x the mean per chunk
            lg = [round(math.log10(n), 3) for n, _, _ in r['found']]
            rec.update(count=r['count'], smallest_log10=lg[0] if lg else None, P=r['P'], predicted=r['predicted'],
                       collected=r['collected'], verified=r['verified'],
                       log10=lg, values=[str(n) for n, _, _ in r['found']])   # readable sizes before raw values
            found += r['count']
            if r['found'] and (best is None or r['found'][0][0] < best):
                best = r['found'][0][0]
                print(f"  new smallest 10^{math.log10(best):.2f}  L0={L0} p0={p0}  {best}", flush=True)
        except Exception as e:
            rec.update(count=None, values=[], error=repr(e))
        with open(RUNS, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        if (i + 1) % 5000 == 0 or i + 1 == len(todo):
            print(f"[{i + 1:,}/{len(todo):,}] log2 L0 {math.log2(L0):.1f}  non-rigid so far {found}  "
                  f"smallest {'10^%.2f' % math.log10(best) if best else '-'}  {(time.time() - t0) / 60:.1f} min",
                  flush=True)


def report(top):
    rows = []
    for line in open(RUNS):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    errs = [r for r in rows if r.get('error')]
    bad = [r for r in rows if r.get('count') and (r['verified'] != r['count'] or r['collected'] != r['count'])]
    src = {}
    for r in rows:
        for v in r.get('values', []):
            src.setdefault(int(v), (r['L0'], r['p0']))
    print(f"{len(rows):,} pairs; {sum(r.get('count') or 0 for r in rows):,} non-rigid found "
          f"(predicted {sum(r.get('predicted') or 0 for r in rows):,.1f}); {len(src):,} distinct; "
          f"{len(errs)} errors; {len(bad)} pairs with unverified/uncollected hits")
    for n in sorted(src)[:top]:
        L0, p0 = src[n]
        print(f"  10^{math.log10(n):6.2f}  p0={p0:<5} L0={L0:<22} {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('--l2max', type=float, default=50)
    ap.add_argument('--p0', default='193,673,1153,2017')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--box', help='generator CSV (default cn2_box_fecundity.csv)')
    ap.add_argument('--runs', help='results JSONL (default cn2_box_nonrigid.jsonl)')
    a = ap.parse_args()
    global BOX, RUNS
    BOX = a.box or BOX
    RUNS = a.runs or RUNS
    if a.cmd == 'run':
        run(a.l2max, [int(x) for x in a.p0.split(',')])
    else:
        report(a.top)


if __name__ == '__main__':
    main()
