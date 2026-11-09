#!/usr/bin/env python3
"""cn2_pairs_run.py -- exhaustively search a list of (L0, p0) pairs (e.g. the E_Y-filtered pairs of cn2_ey_scout.py).

    python3 cn2_pairs_run.py PAIRS.json [--frac 0.99] [--runs OUT.jsonl]

Takes the pairs carrying `frac` of the total E_Y (the file is sorted by E_Y), runs them smallest pool first with
cn2_nonrigid_search.search (every hit verified), one JSON line per pair; resumable; logs each new smallest.
"""
import argparse, json, math, os, time
from sympy import factorint
from cn2_nonrigid_search import search

ap = argparse.ArgumentParser()
ap.add_argument('pairs')
ap.add_argument('--frac', type=float, default=0.99)
ap.add_argument('--runs', required=True)
a = ap.parse_args()
P = json.load(open(a.pairs))
tot, acc, sel = sum(d['EY'] for d in P), 0.0, []
for d in P:
    if acc >= a.frac * tot:
        break
    sel.append(d); acc += d['EY']
done, best, found = set(), None, 0
if os.path.exists(a.runs):                       # resume: skip done pairs, and carry over the running minimum/total
    for l in open(a.runs):
        try: r = json.loads(l)
        except ValueError: continue
        done.add((str(r['L0']), r['p0']))
        found += r.get('count') or 0
        for v in r.get('values', []):
            if best is None or int(v) < best:
                best = int(v)
todo = sorted((d for d in sel if (d['L0'], d['p0']) not in done), key=lambda d: d['P'])
print(f"{len(sel):,} pairs carry {acc / tot:.1%} of E_Y = {tot:.4f}; {len(todo):,} to run"
      + (f"; resumed with {found} found, smallest 10^{math.log10(best):.2f}" if best else ''), flush=True)
t0 = time.time()
for i, d in enumerate(todo):
    L0 = int(d['L0'])
    r = search(L0, factorint(L0), d['p0'], 256, min_P_size=1)
    lg = [round(math.log10(n), 3) for n, _, _ in r['found']]
    rec = {'L0': d['L0'], 'p0': d['p0'], 'count': r['count'], 'smallest_log10': lg[0] if lg else None, 'EY': d['EY']}
    rec.update({k: v for k, v in r.items() if k not in ('found', 'L0', 'p0', 'count')})
    rec['log10'] = lg                                         # readable sizes BEFORE the long raw values
    rec['values'] = [str(n) for n, _, _ in r['found']]
    with open(a.runs, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    found += r['count']
    if r['found'] and (best is None or r['found'][0][0] < best):
        best = r['found'][0][0]
        print(f"  new smallest 10^{math.log10(best):.2f}  p0={d['p0']} L0={L0} (verified {r['verified']}/{r['count']})  {best}", flush=True)
    if (i + 1) % 10000 == 0 or i + 1 == len(todo):
        print(f"[{i + 1:,}/{len(todo):,}] pool {d['P']}  found {found}  smallest {'10^%.2f' % math.log10(best) if best else '-'}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
