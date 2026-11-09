#!/usr/bin/env python3
"""cn2_twin_run.py -- time-limited search for numbers with TWO type-p primes, over pairs from cn2_twin_scout.py.

    python3 cn2_twin_run.py [--pmax 72] [--emin 0.05] [--minutes 40]

Pairs are taken smallest pool first (cheapest), each searched exhaustively by search_multi() and verified; one
JSON line per pair in cn2_twin_runs.jsonl (resumable).  Coverage is exactly the pairs completed before the limit.
"""
import argparse, json, math, os, time
from sympy import factorint
from cn2_nonrigid_search import search_multi

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'cn2_twin_runs.jsonl')

ap = argparse.ArgumentParser()
ap.add_argument('--pmax', type=int, default=72)
ap.add_argument('--emin', type=float, default=0.05)
ap.add_argument('--minutes', type=float, default=40)
a = ap.parse_args()
pairs = [d for d in json.load(open(os.path.join(HERE, 'cn2_twin_pairs.json'))) if d['P'] <= a.pmax and d['E'] >= a.emin]
pairs.sort(key=lambda d: (d['P'], -d['E']))
done = set()
if os.path.exists(RUNS):
    for l in open(RUNS):
        try: r = json.loads(l); done.add((r['L0'], tuple(r['p0s'])))
        except ValueError: pass
todo = [d for d in pairs if (d['L0'], tuple(d['p0s'])) not in done]
print(f"{len(pairs):,} pairs (|P'| <= {a.pmax}, E >= {a.emin}); {len(todo):,} to run; limit {a.minutes} min", flush=True)
t0, best, found = time.time(), None, 0
for i, d in enumerate(todo):
    if time.time() - t0 > 60 * a.minutes:
        print(f"time limit: stopped after {i:,} pairs (last |P'| {todo[i-1]['P']})", flush=True); break
    L0 = int(d['L0'])
    r = search_multi(L0, factorint(L0), d['p0s'], 256)
    rec = {k: v for k, v in r.items() if k != 'found'}; rec['L0'] = d['L0']; rec['E'] = d['E']
    rec['values'] = [str(n) for n, _, _ in r['found']]
    open(RUNS, 'a').write(json.dumps(rec) + '\n')
    found += r['count']
    if r['found'] and (best is None or r['found'][0][0] < best):
        best = r['found'][0][0]
        print(f"  new smallest 10^{math.log10(best):.2f}  p0s={d['p0s']} L0={L0} (verified {r['verified']}/{r['count']})  {best}", flush=True)
    if (i + 1) % 50 == 0:
        print(f"[{i + 1}/{len(todo)}] |P'| {d['P']}  found {found}  {(time.time() - t0) / 60:.1f} min", flush=True)
print(f"done: {found} two-type-p numbers; smallest {'10^%.2f' % math.log10(best) if best else '-'}", flush=True)
