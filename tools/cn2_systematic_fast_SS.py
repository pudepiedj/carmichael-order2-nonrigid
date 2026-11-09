#!/usr/bin/env python3
"""
CN2 generator evaluation by four-list modular Schroeppel–Shamir.

Successor to cn2_systematic_fast_splitL.py.  The two-list MITM there holds
2^(|P|/2) residues (plus argsort copies) in RAM, which is ~2 TiB at |P| = 72.
Here the pool is cut into four quarters Q1..Q4 (~2^(|P|/4) subsets each) and
the modulus into coprime parts

    L = A * G * E_1 * ... * E_m        (every part < 2^31)

For each residue class t of the subgroup H = <P mod A>:

    list12 = {(i,j) : Q1[i]*Q2[j] == t      (mod A)}   keyed by  Q1*Q2      mod G
    list34 = {(k,l) : Q3[k]*Q4[l] == t^-1   (mod A)}   keyed by (Q3*Q4)^-1  mod G

Equal keys give products == 1 mod A*G; the E_g (usually none) are checked
per candidate, so every accepted (i,j,k,l) is exactly n == 1 (mod L).
Each subset lands in exactly one class, so every CN2 is found exactly once
and no deduplication is needed.

Memory per thread ~ 2^(|P|/2) / |H| list entries; A is chosen to balance the
|H| * 2^(|P|/4) class lookups against the sorting work.

Only statistics are kept (count, factor-count and log10-size distributions,
timing); the CN2 values themselves are never materialised.  Factor counts
are popcounts of the subset masks, not factorint().

Output statistics (table columns and --json fields):

    cn2_count        number of CN2s n == 1 (mod L) built from distinct pool primes
    P                the pool, sorted
    fecundity        |P| - log2 phi(L)
    predicted_count  2^fecundity = 2^|P| / phi(L): the expected count if subset
                     products are uniform on (Z/L)* and <P> has index 1.  An
                     index-k subgroup would multiply the count by k.
    count_ratio      cn2_count / predicted_count ("Obs/Pred"); ~1 confirms both
                     uniformity and index 1 (1.0003 at |P|=72, 1.0033 for L2)
    omega_hist       NOTE: keys are NOT primes.  key w = omega(n), the number of
                     (distinct) prime factors; value = how many CN2s have exactly
                     w factors.  {"13": 1, "14": 3, ...} means one CN2 with 13
                     prime factors, three with 14.  min/max keys = MinF/MaxF.
                     (Named factor_hist in early cn2_SS_runs.jsonl lines.)
    prime_counts     keys ARE the pool primes: p -> number of CN2s divisible by p.
                     Zeros are kept, so a pool prime never used shows as 0.
    prime_frac_min/max  min and max of prime_counts / cn2_count ("p-fraction
                     range"); ~0.5 for every p when products are equidistributed
                     on <P> (this alone does not detect index > 1)
    log10_hist       key d -> number of CN2s with floor(log10 n) = d
    min/max_log10    exact size range (sums of log10 p, float precision)
    outliers         primes multiplied into the base generator by --outliers
                     ([] = the base itself); first table column, "base" or "+37"
    A, H_size, G, extra_moduli, max_list, candidates
                     search plan and work diagnostics; candidates = key matches
                     mod A*G, equal to cn2_count when there are no extra moduli

Usage:
    python cn2_systematic_fast_SS.py 2^7.3^3.5^2.7.11.13.17.19.29.31
    python cn2_systematic_fast_SS.py 577614211574400 --json runs.jsonl
    python cn2_systematic_fast_SS.py 2^7.3^3.5^2.7.11.13.17.19.29.31 --outliers 23,37,41
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange, get_num_threads
from sympy import isprime

GROUP_BITS = 31          # every modulus < 2^31, so a*b fits in int64
MAX_A = 1 << 24          # bitmap / CSR tables over Z/A stay <= ~128 MB
MAX_H = 1 << 24          # abandon subgroup BFS beyond this
LIST_CAP = 1 << 26       # per-class list entries (x ~24 bytes) per thread
N_LOG_BINS = 512         # log10(n) histogram, 1-decade bins

# ============================================================================
# NUMBER THEORY ON L (Python ints)
# ============================================================================

def factorint(n: int) -> Dict[int, int]:
    """Trial division -- only ever applied to L, never to CN2s."""
    d, out = 2, {}
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out

def divisors(fact: Dict[int, int]) -> List[int]:
    divs = [1]
    for p, e in fact.items():
        divs = [d * p**k for d in divs for k in range(e + 1)]
    return divs

def calculate_P(L_factors: Dict[int, int]) -> List[int]:
    P = []
    for d in divisors(L_factors):
        s = d + 1
        root = math.isqrt(s)
        if root * root == s and root > 1 and root not in L_factors and isprime(root):
            P.append(root)
    return sorted(P)

def phi_from_factors(fact: Dict[int, int]) -> int:
    return math.prod((p - 1) * p**(e - 1) for p, e in fact.items())

def parse_generator(spec: str) -> Tuple[int, Dict[int, int]]:
    """'577614211574400' or '2^7.3^3.5^2.7.11' (also accepts '*' or 'x')."""
    s = spec.strip().replace('*', '.').replace('x', '.')
    if s.isdigit():
        L = int(s)
        return L, factorint(L)
    fact: Dict[int, int] = {}
    for part in s.split('.'):
        if not part:
            continue
        p, _, e = part.partition('^')
        p, e = int(p), int(e) if e else 1
        if not isprime(p):
            raise ValueError(f"{p} in '{spec}' is not prime")
        fact[p] = fact.get(p, 0) + e
    return math.prod(p**e for p, e in fact.items()), fact

def format_L_factors(fact: Dict[int, int]) -> str:
    return '.'.join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(fact.items()))

def split_coprime(fact: Dict[int, int], max_bits: int) -> List[int]:
    """Greedy first-fit packing of prime powers into moduli < 2^max_bits,
    largest first, so groups[0] is the largest (used as the join modulus)."""
    groups: List[int] = []
    for pe in sorted((p**e for p, e in fact.items()), reverse=True):
        for g in range(len(groups)):
            if groups[g] * pe < (1 << max_bits):
                groups[g] *= pe
                break
        else:
            groups.append(pe)
    return sorted(groups, reverse=True)

# ============================================================================
# NUMBA KERNELS
# ============================================================================

@njit(cache=True)
def modinv(a, m):
    if m == 1:
        return 0
    a %= m
    r0, r1, x0, x1 = m, a, 0, 1
    while r1 != 0:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        x0, x1 = x1, x0 - q * x1
    if x0 < 0:
        x0 += m
    return x0

@njit(cache=True)
def subset_residues(primes, M):
    """out[mask] = prod_{bit b in mask} primes[b]  mod M."""
    q = len(primes)
    out = np.empty(1 << q, np.int64)
    out[0] = 1 % M
    for b in range(q):
        pb = primes[b] % M
        half = 1 << b
        for m in range(half):
            out[half + m] = out[m] * pb % M
    return out

@njit(cache=True)
def invert_table(res, M):
    out = np.empty(len(res), np.int64)
    for i in range(len(res)):
        out[i] = modinv(res[i], M)
    return out

@njit(cache=True)
def subset_logs(primes):
    q = len(primes)
    out = np.empty(1 << q, np.float64)
    out[0] = 0.0
    for b in range(q):
        lb = math.log10(primes[b])
        half = 1 << b
        for m in range(half):
            out[half + m] = out[m] + lb
    return out

@njit(cache=True)
def subset_popcounts(q):
    out = np.empty(1 << q, np.int64)
    out[0] = 0
    for b in range(q):
        half = 1 << b
        for m in range(half):
            out[half + m] = out[m] + 1
    return out

@njit(cache=True)
def subgroup_elements(gens, A, limit):
    """BFS of <gens> in (Z/A)*.  Returns empty array if |H| > limit."""
    seen = np.zeros(A, np.uint8)
    queue = np.empty(limit, np.int64)
    s = 1 % A
    seen[s] = 1
    queue[0] = s
    head, tail = 0, 1
    while head < tail:
        x = queue[head]
        head += 1
        for g in gens:
            y = x * g % A
            if seen[y] == 0:
                if tail == limit:
                    return np.empty(0, np.int64)
                seen[y] = 1
                queue[tail] = y
                tail += 1
    return queue[:tail].copy()

@njit(cache=True)
def bucket_by_residue(res, A):
    """CSR grouping of subset indices by residue mod A."""
    off = np.zeros(A + 1, np.int64)
    for r in res:
        off[r + 1] += 1
    for r in range(A):
        off[r + 1] += off[r]
    order = np.empty(len(res), np.int32)
    pos = off[:-1].copy()
    for idx in range(len(res)):
        r = res[idx]
        order[pos[r]] = idx
        pos[r] += 1
    return off, order

@njit(cache=True)
def _build_list(t, A, iaX, offY, ordY, gX, gY, G, s, I, J):
    """Fill s[m] = key<<32 | m with key = gX[i]*gY[j] mod G over all (i,j)
    with resX[i]*resY[j] == t mod A; sort s[:m].  Returns m."""
    m = 0
    for i in range(len(iaX)):
        r = t * iaX[i] % A
        gi = gX[i]
        for u in range(offY[r], offY[r + 1]):
            j = ordY[u]
            s[m] = ((gi * gY[j] % G) << 32) | m
            I[m] = i
            J[m] = j
            m += 1
    v = s[:m]
    v.sort()
    return m

@njit(cache=True)
def _count_list(t, A, iaX, offY):
    n = 0
    for i in range(len(iaX)):
        r = t * iaX[i] % A
        n += offY[r + 1] - offY[r]
    return n

@njit(cache=True)
def _add_bits(row, mask, offset):
    """row[offset + b] += 1 for every set bit b of mask."""
    b = offset
    while mask:
        if mask & 1:
            row[b] += 1
        mask >>= 1
        b += 1

@njit(parallel=True, cache=True)
def run_chunks(chunk_lo, chunk_hi, classes, A,
               ia1, off2, ord2, ia3, off4, ord4,
               g1, g2, ig3, ig4, G,
               e1, e2, e3, e4, EM, TA, TE,
               p1, p2, p3, p4, l1, l2, l3, l4,
               qoff, nbits, nlog, hit_cap):
    nch = len(chunk_lo)
    ng = len(EM)
    MASK32 = (1 << 32) - 1
    primecnt = np.zeros((nch, nbits), np.int64)
    count = np.zeros(nch, np.int64)
    pophist = np.zeros((nch, nbits + 1), np.int64)
    loghist = np.zeros((nch, nlog), np.int64)
    minlog = np.full(nch, np.inf)
    maxlog = np.full(nch, -np.inf)
    maxlist = np.zeros(nch, np.int64)
    candidates = np.zeros(nch, np.int64)
    hit_masks = np.zeros((nch, hit_cap, 4), np.int64)      # subset masks of the first hit_cap hits per chunk
    hit_n = np.zeros(nch, np.int64)

    for c in prange(nch):
        cap = 0
        s12 = np.empty(0, np.int64); I12 = np.empty(0, np.int32); J12 = np.empty(0, np.int32)
        s34 = np.empty(0, np.int64); I34 = np.empty(0, np.int32); J34 = np.empty(0, np.int32)
        for ci in range(chunk_lo[c], chunk_hi[c]):
            t = classes[ci]
            tinv = TA * modinv(t, A) % A          # Q3*Q4 == T * t^-1 (mod A); T = 1 for CN2s
            n12 = _count_list(t, A, ia1, off2)
            n34 = _count_list(tinv, A, ia3, off4)
            if n12 == 0 or n34 == 0:
                continue
            need = max(n12, n34)
            if need > maxlist[c]:
                maxlist[c] = need
            if need > cap:
                cap = need + need // 4
                s12 = np.empty(cap, np.int64); I12 = np.empty(cap, np.int32); J12 = np.empty(cap, np.int32)
                s34 = np.empty(cap, np.int64); I34 = np.empty(cap, np.int32); J34 = np.empty(cap, np.int32)
            _build_list(t, A, ia1, off2, ord2, g1, g2, G, s12, I12, J12)
            _build_list(tinv, A, ia3, off4, ord4, ig3, ig4, G, s34, I34, J34)

            a = 0
            b = 0
            while a < n12 and b < n34:
                ka = s12[a] >> 32
                kb = s34[b] >> 32
                if ka < kb:
                    a += 1
                elif ka > kb:
                    b += 1
                else:
                    a2 = a
                    while a2 < n12 and (s12[a2] >> 32) == ka:
                        a2 += 1
                    b2 = b
                    while b2 < n34 and (s34[b2] >> 32) == ka:
                        b2 += 1
                    for x in range(a, a2):
                        pa = s12[x] & MASK32
                        i = I12[pa]
                        j = J12[pa]
                        for y in range(b, b2):
                            pb = s34[y] & MASK32
                            k = I34[pb]
                            l = J34[pb]
                            if i == 0 and j == 0 and k == 0 and l == 0:
                                continue            # empty product n = 1
                            candidates[c] += 1
                            ok = True
                            for g in range(ng):
                                M = EM[g]
                                if (e1[g, i] * e2[g, j] % M) * (e3[g, k] * e4[g, l] % M) % M != TE[g]:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            pop = p1[i] + p2[j] + p3[k] + p4[l]
                            lg = l1[i] + l2[j] + l3[k] + l4[l]
                            if hit_n[c] < hit_cap:
                                hit_masks[c, hit_n[c], 0] = i
                                hit_masks[c, hit_n[c], 1] = j
                                hit_masks[c, hit_n[c], 2] = k
                                hit_masks[c, hit_n[c], 3] = l
                            hit_n[c] += 1
                            count[c] += 1
                            row = primecnt[c]
                            _add_bits(row, i, qoff[0])
                            _add_bits(row, j, qoff[1])
                            _add_bits(row, k, qoff[2])
                            _add_bits(row, l, qoff[3])
                            pophist[c, pop] += 1
                            bn = int(lg)
                            if bn >= nlog:
                                bn = nlog - 1
                            loghist[c, bn] += 1
                            if lg < minlog[c]:
                                minlog[c] = lg
                            if lg > maxlog[c]:
                                maxlog[c] = lg
                    a = a2
                    b = b2
    return count, pophist, loghist, minlog, maxlog, maxlist, candidates, primecnt, hit_masks, hit_n

# ============================================================================
# PLANNING: quarters and the class modulus A
# ============================================================================

def quarter_sizes(n: int) -> List[int]:
    base, rem = divmod(n, 4)
    return [base + (1 if k >= 4 - rem else 0) for k in range(4)]

@dataclass
class Plan:
    A: int
    A_factors: Dict[int, int]
    classes: np.ndarray          # elements of H = <P mod A>
    G: int                       # join modulus
    extra: List[int]             # remaining coprime moduli, checked per candidate
    est_list: float              # expected per-class list length
    est_cost: float

def _cost(h: float, qs: List[int], phiG: int, n_extra: int) -> Tuple[float, float]:
    e12 = 2.0 ** (qs[0] + qs[1])
    e34 = 2.0 ** (qs[2] + qs[3])
    lookups = 3.0 * h * (2.0 ** qs[0] + 2.0 ** qs[2])
    per_class = max(e12, e34) / h
    sorting = (e12 + e34) * math.log2(max(2.0, per_class))
    cands = 2.0 ** sum(qs) / (h * phiG) * n_extra * 2.0
    return lookups + sorting + cands, per_class

def choose_plan(L: int, L_factors: Dict[int, int], P: List[int],
                qs: List[int], group_bits: int = GROUP_BITS,
                n_exact: int = 32) -> Plan:
    comps = sorted(L_factors.items())
    for p, e in comps:
        if p**e >= (1 << group_bits):
            raise ValueError(f"prime power {p}^{e} >= 2^{group_bits}; not supported")

    rough = []
    for mask in range(1 << len(comps)):
        Af = {p: e for b, (p, e) in enumerate(comps) if mask >> b & 1}
        A = math.prod(p**e for p, e in Af.items())
        if A > MAX_A:
            continue
        rest = {p: e for p, e in comps if p not in Af}
        groups = split_coprime(rest, group_bits) if rest else [1]
        phiG = phi_from_factors({p: e for p, e in rest.items()
                                 if groups[0] % p == 0}) if rest else 1
        cost, _ = _cost(float(phi_from_factors(Af)), qs, max(phiG, 1), len(groups) - 1)
        rough.append((cost, A, Af, groups, phiG))
    rough.sort(key=lambda r: r[0])

    best: Optional[Plan] = None
    for _, A, Af, groups, phiG in rough[:n_exact]:
        gens = np.array(sorted({p % A for p in P}), dtype=np.int64)
        H = subgroup_elements(gens, A, min(MAX_H, phi_from_factors(Af) + 1))
        if len(H) == 0:
            continue
        cost, per_class = _cost(float(len(H)), qs, max(phiG, 1), len(groups) - 1)
        if per_class > LIST_CAP:
            continue
        if best is None or cost < best.est_cost:
            best = Plan(A, Af, H, groups[0], groups[1:], per_class, cost)
    if best is None:
        raise RuntimeError("no class modulus A keeps per-class lists under LIST_CAP")
    return best

# ============================================================================
# EVALUATION
# ============================================================================

@dataclass
class GeneratorResult:
    L: int
    L_factors: Dict[int, int]
    P_size: int
    fecundity: float
    cn2_count: int = 0
    min_factors: Optional[int] = None
    max_factors: Optional[int] = None
    min_log10: Optional[float] = None
    max_log10: Optional[float] = None
    elapsed: float = 0.0
    A: int = 1
    H_size: int = 1
    G: int = 1
    extra_moduli: List[int] = field(default_factory=list)
    max_list: int = 0
    candidates: int = 0
    omega_hist: Dict[int, int] = field(default_factory=dict)   # omega -> count
    log10_hist: Dict[int, int] = field(default_factory=dict)    # floor(log10 n) -> count
    P: List[int] = field(default_factory=list)
    prime_counts: Dict[int, int] = field(default_factory=dict)  # p -> #CN2s divisible by p
    cn2_values: List[int] = field(default_factory=list)         # populated only with collect_hits > 0
    predicted_count: float = 0.0      # 2^|P| / phi(L) = 2^fecundity
    count_ratio: Optional[float] = None   # cn2_count / predicted_count
    prime_frac_min: Optional[float] = None
    prime_frac_max: Optional[float] = None
    outliers: List[int] = field(default_factory=list)   # primes multiplied into the base spec
    skipped: bool = False
    skip_reason: str = ""

def evaluate_generator(L: int, L_factors: Dict[int, int],
                       max_P_size: int = 90, min_P_size: int = 4,
                       group_bits: int = GROUP_BITS,
                       n_chunks: Optional[int] = None,
                       collect_hits: int = 0,
                       verbose: bool = True,
                       target: int = 1,
                       P_override: Optional[List[int]] = None) -> GeneratorResult:
    """target != 1: count subsets with product == target (mod L) instead of == 1 (Howe's non-rigid
    construction, see cn2_nonrigid_search.py); P_override replaces the pool P(2, L)."""
    t0 = time.time()
    P = sorted(P_override) if P_override is not None else calculate_P(L_factors)
    n = len(P)
    fec = n - math.log2(phi_from_factors(L_factors))
    res = GeneratorResult(L=L, L_factors=L_factors, P_size=n, fecundity=fec, P=P,
                          predicted_count=2.0 ** fec)
    if not (min_P_size <= n <= max_P_size):
        res.skipped, res.skip_reason = True, f"|P|={n} outside [{min_P_size},{max_P_size}]"
        return res

    qs = quarter_sizes(n)
    plan = choose_plan(L, L_factors, P, qs, group_bits)
    A, G, H = plan.A, plan.G, plan.classes
    res.A, res.H_size, res.G, res.extra_moduli = A, len(H), G, plan.extra
    if verbose:
        print(f"  L = {L} = {format_L_factors(L_factors)}  ({L.bit_length()} bits)")
        print(f"  |P| = {n}, quarters {qs}, fecundity {fec:.4f}")
        print(f"  A = {A} = {format_L_factors(plan.A_factors) or '1'}, |H| = 2^{math.log2(len(H)):.2f}, "
              f"G = {G}, extra moduli {plan.extra or 'none'}")
        print(f"  expected list length/class ~2^{math.log2(max(plan.est_list, 1)):.1f}, "
              f"work ~2^{math.log2(plan.est_cost):.1f}")

    bounds = np.cumsum([0] + qs)
    quarters = [np.array(P[bounds[k]:bounds[k + 1]], dtype=np.int64) for k in range(4)]
    rA = [subset_residues(q, A) for q in quarters]
    rG = [subset_residues(q, G) for q in quarters]
    ia1, ia3 = invert_table(rA[0], A), invert_table(rA[2], A)
    off2, ord2 = bucket_by_residue(rA[1], A)
    off4, ord4 = bucket_by_residue(rA[3], A)
    ig3, ig4 = invert_table(rG[2], G), invert_table(rG[3], G)
    if target % G != 1 % G:
        ig3 = ig3 * (target % G) % G                 # join key T * (Q3*Q4)^-1 (mod G)
    EM = np.array(plan.extra, dtype=np.int64)
    TA = target % A
    TE = np.array([target % M for M in plan.extra], dtype=np.int64)
    ext = [np.stack([subset_residues(q, M) for M in plan.extra]) if plan.extra
           else np.empty((0, 1 << qs[k]), np.int64)
           for k, q in enumerate(quarters)]
    pops = [subset_popcounts(qs[k]) for k in range(4)]
    logs = [subset_logs(q) for q in quarters]
    del rA

    threads = get_num_threads()
    if n_chunks is None:
        n_chunks = min(len(H), threads * 64)
    edges = np.linspace(0, len(H), n_chunks + 1).astype(np.int64)
    per_call = threads * 4

    qoff = bounds[:4].astype(np.int64)
    hit_cap = max(0, int(collect_hits))
    hits_found: List[int] = []
    count = 0
    primecnt = np.zeros(n, np.int64)
    pophist = np.zeros(n + 1, np.int64)
    loghist = np.zeros(N_LOG_BINS, np.int64)
    minlog, maxlog, maxlist, cands = math.inf, -math.inf, 0, 0
    t_search = time.time()
    for s in range(0, n_chunks, per_call):
        e = min(n_chunks, s + per_call)
        c, ph, lh, mn, mx, ml, cd, pc, hm, hn = run_chunks(
            edges[s:e], edges[s + 1:e + 1], H, A,
            ia1, off2, ord2, ia3, off4, ord4,
            rG[0], rG[1], ig3, ig4, G,
            ext[0], ext[1], ext[2], ext[3], EM, TA, TE,
            pops[0], pops[1], pops[2], pops[3],
            logs[0], logs[1], logs[2], logs[3],
            qoff, n, N_LOG_BINS, max(hit_cap, 1))
        count += int(c.sum())
        if hit_cap:
            for ci in range(len(hn)):
                for hi in range(min(int(hn[ci]), hit_cap)):
                    masks = hm[ci, hi]
                    val = 1
                    for qi in range(4):
                        mask = int(masks[qi])
                        for b in range(qs[qi]):
                            if mask >> b & 1:
                                val *= int(quarters[qi][b])
                    hits_found.append(val)
        primecnt += pc.sum(axis=0)
        pophist += ph.sum(axis=0)
        loghist += lh.sum(axis=0)
        minlog, maxlog = min(minlog, float(mn.min())), max(maxlog, float(mx.max()))
        maxlist, cands = max(maxlist, int(ml.max())), cands + int(cd.sum())
        if verbose and n_chunks > per_call:
            done = e / n_chunks
            el = time.time() - t_search
            print(f"    {100*done:5.1f}%  {count:>12,} CN2s  {el:8.1f}s  "
                  f"ETA {el/done - el:8.1f}s", flush=True)

    res.cn2_count = count
    res.elapsed = time.time() - t0
    res.max_list = maxlist
    res.candidates = cands
    res.omega_hist = {w: int(v) for w, v in enumerate(pophist) if v}
    res.log10_hist = {d: int(v) for d, v in enumerate(loghist) if v}
    res.prime_counts = {p: int(v) for p, v in zip(P, primecnt)}   # zeros kept
    res.cn2_values = sorted(set(hits_found))                      # only when collect_hits > 0
    res.count_ratio = count / res.predicted_count
    if count:
        res.min_factors, res.max_factors = min(res.omega_hist), max(res.omega_hist)
        res.min_log10, res.max_log10 = minlog, maxlog
        res.prime_frac_min = float(primecnt.min()) / count
        res.prime_frac_max = float(primecnt.max()) / count
    if verbose:
        print(f"  => {count:,} CN2s in {res.elapsed:.2f}s "
              f"(max list {maxlist:,}, {cands:,} G-joins)")
    return res

# ============================================================================
# OUTPUT
# ============================================================================

def fmt_opt(v, fmt: str) -> str:
    return format(v, fmt) if v is not None else "--"

def format_outliers(r: GeneratorResult, sep: str = ',') -> str:
    return '+' + sep.join(map(str, r.outliers)) if r.outliers else 'base'

def print_results_table(results: List[GeneratorResult]):
    ow = max([8] + [len(format_outliers(r)) for r in results])
    w = 146 + ow + 1
    print(f"\n{'='*w}")
    print(f"{'Outlier':<{ow}} {'L':>24} {'|P|':>4} {'Fec':>8} {'CN2s':>12} {'Predicted':>14} {'Obs/Pred':>9} "
          f"{'MinF':>4} {'MaxF':>4} {'log10 range':>15} {'p-fraction range':>17} "
          f"{'|H|':>9} {'Time':>9}")
    print('-' * w)
    for r in results:
        if r.skipped:
            print(f"{format_outliers(r):<{ow}} {r.L:>24} {r.P_size:>4} {r.fecundity:>8.3f}  "
                  f"skipped: {r.skip_reason}")
            continue
        rng = (f"{r.min_log10:.2f}-{r.max_log10:.2f}" if r.cn2_count else "--")
        prng = (f"{r.prime_frac_min:.5f}-{r.prime_frac_max:.5f}" if r.cn2_count else "--")
        print(f"{format_outliers(r):<{ow}} {r.L:>24} {r.P_size:>4} {r.fecundity:>8.3f} {r.cn2_count:>12,} "
              f"{r.predicted_count:>14,.1f} {r.count_ratio:>9.5f} "
              f"{fmt_opt(r.min_factors, '>4')} {fmt_opt(r.max_factors, '>4')} "
              f"{rng:>15} {prng:>17} {r.H_size:>9,} {r.elapsed:>8.2f}s")

def print_omega_histograms(results: List[GeneratorResult]):
    for r in results:
        if r.skipped or not r.cn2_count:
            continue
        print(f"\n  L = {r.L}: CN2s by number of prime factors")
        peak = max(r.omega_hist.values())
        for w, v in sorted(r.omega_hist.items()):
            print(f"    {w:>3} {v:>12,}  {'#' * max(1, round(50 * v / peak))}")

def print_prime_counts(results: List[GeneratorResult]):
    for r in results:
        if r.skipped or not r.cn2_count:
            continue
        zeros = [p for p, v in r.prime_counts.items() if v == 0]
        print(f"\n  L = {r.L}: CN2s containing each pool prime "
              f"(fraction of {r.cn2_count:,})")
        items = list(r.prime_counts.items())
        for s in range(0, len(items), 3):
            print('   ' + ''.join(f"{p:>9} {v:>12,} {v / r.cn2_count:6.3f}   "
                                  for p, v in items[s:s + 3]))
        print(f"  never used: {zeros if zeros else 'none -- every pool prime occurs'}")

def print_latex_table(results: List[GeneratorResult]):
    print(r"\begin{table}[htbp]\centering")
    print(r"\begin{tabular}{lrcrrrrrrrrrr}")
    print(r"\toprule")
    print(r"Outlier & $L$ & $L_{\mathrm{factors}}$ & $|P|$ & CN2s & $2^{|P|}/\varphi(L)$ & obs/pred & "
          r"MinF & MaxF & $\log_{\min}$ & $\log_{\max}$ & $p$-fraction & time \\")
    print(r"\midrule")
    for r in results:
        if r.skipped:
            continue
        prng = (f"{r.prime_frac_min:.5f}--{r.prime_frac_max:.5f}" if r.cn2_count else "--")
        print(f"{format_outliers(r, ', ')} & ${r.L}$ & ${format_L_factors(r.L_factors)}$ & ${r.P_size}$ & ${r.cn2_count}$ & "
              f"${r.predicted_count:.1f}$ & ${r.count_ratio:.5f}$ & "
              f"{fmt_opt(r.min_factors, 'd')} & {fmt_opt(r.max_factors, 'd')} & "
              f"{fmt_opt(r.min_log10, '.2f')} & {fmt_opt(r.max_log10, '.2f')} & "
              f"{prng} & {r.elapsed:.2f}s \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{CN2 search results (four-list Schroeppel--Shamir).}\label{tab:cn2ss}")
    print(r"\end{table}")

def result_to_json(r: GeneratorResult) -> str:
    d = asdict(r)
    d['L'] = str(r.L)
    d['L_factors'] = {str(p): e for p, e in r.L_factors.items()}
    d['omega_hist'] = {str(k): v for k, v in r.omega_hist.items()}
    d['log10_hist'] = {str(k): v for k, v in r.log10_hist.items()}
    d['prime_counts'] = {str(p): v for p, v in r.prime_counts.items()}
    d['cn2_values'] = [str(v) for v in r.cn2_values]
    return json.dumps(d)

# ============================================================================
# MAIN
# ============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('generators', nargs='+',
                    help="L as an integer or as a factor spec like 2^7.3^3.5^2.7.11")
    ap.add_argument('--outliers', default='',
                    help="comma-separated primes; each is also tried multiplied into every generator")
    ap.add_argument('--max-P', type=int, default=90)
    ap.add_argument('--min-P', type=int, default=4)
    ap.add_argument('--group-bits', type=int, default=GROUP_BITS,
                    help="max bits per coprime modulus (lower it to exercise the extra-moduli check)")
    ap.add_argument('--latex', action='store_true')
    ap.add_argument('--json', metavar='PATH', help="append one JSON line per generator")
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    jobs = []
    outliers = [int(o) for o in args.outliers.split(',') if o.strip()]
    for spec in args.generators:
        L, fact = parse_generator(spec)
        jobs.append((L, fact, []))
        for o in outliers:
            if o in fact:
                print(f"  outlier {o} already divides {spec}; not added")
                continue
            f2 = dict(fact)
            f2[o] = 1
            jobs.append((L * o, f2, [o]))

    print(f"CN2 four-list Schroeppel–Shamir: {len(jobs)} generator(s), "
          f"{get_num_threads()} threads")
    results = []
    for idx, (L, fact, added) in enumerate(jobs):
        print(f"\n[{idx+1}/{len(jobs)}] {'+' + ','.join(map(str, added)) if added else 'base'}")
        r = evaluate_generator(L, fact, args.max_P, args.min_P, args.group_bits,
                               verbose=not args.quiet)
        r.outliers = added
        if r.skipped:
            print(f"  skipped: {r.skip_reason}")
        results.append(r)
        if args.json:
            with open(args.json, 'a') as fh:
                fh.write(result_to_json(r) + '\n')

    print_results_table(results)
    print_omega_histograms(results)
    print_prime_counts(results)
    if args.latex:
        print()
        print_latex_table(results)
    return results


if __name__ == '__main__':
    main()
