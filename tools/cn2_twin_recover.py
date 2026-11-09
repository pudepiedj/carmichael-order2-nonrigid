#!/usr/bin/env python3
"""cn2_twin_recover.py -- re-run the one two-type-p search whose results were printed but never stored (30 Sep 2026).

    python3 cn2_twin_recover.py

L0 = 433210658680800 with T = {673, 2017} (pool 69) gave 18 numbers against 13.2 predicted, the smallest
2014054071452486938916691582437070523247423980591422448730122005483055750184232583201 (about 10^84.30).  This re-runs
search_multi exactly as cn2_twin_run.py does, checks that it reproduces those two facts, and appends the record to
cn2_twin_runs.jsonl in the same format.  It refuses to append twice.
"""
import json, math, os, sys
from sympy import factorint
from cn2_nonrigid_search import search_multi

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'cn2_twin_runs.jsonl')
L0, P0S = 433210658680800, [673, 2017]
SMALLEST = 2014054071452486938916691582437070523247423980591422448730122005483055750184232583201

if __name__ == '__main__':
    for l in open(RUNS):
        r = json.loads(l)
        if int(r['L0']) == L0 and sorted(r['p0s']) == P0S:
            sys.exit("already in cn2_twin_runs.jsonl; nothing to do")
    r = search_multi(L0, factorint(L0), P0S, 256)
    vals = sorted(n for n, _, _ in r['found'])
    print(f"count {r['count']}, collected {r.get('collected')}, verified {r.get('verified')}, "
          f"smallest 10^{math.log10(vals[0]):.2f}" if vals else "nothing found")
    if r['count'] != 18 or len(vals) != 18 or vals[0] != SMALLEST:
        sys.exit("does NOT reproduce the 30 Sep result (18 found, smallest 10^84.30); not appended")
    rec = {k: v for k, v in r.items() if k != 'found'}
    rec['L0'] = str(L0)
    rec['values'] = [str(n) for n in vals]
    rec['note'] = 'recovered 7 Oct 2026 by cn2_twin_recover.py (the 30 Sep run printed but did not store its values)'
    open(RUNS, 'a').write(json.dumps(rec) + '\n')
    print(f"appended to {RUNS}")
