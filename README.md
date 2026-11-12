# Non-rigid Carmichael numbers of order 2

This repository publishes **1,135,606 non-rigid Carmichael numbers of order 2** (OEIS [A175531](https://oeis.org/A175531)). They were found with Howe's construction, and every one has been verified from scratch. They range from about $4.63\times10^{36}$ to $10^{174.45}$.

**These are examples, not a complete list.** A construction finds the numbers on the generators and type-$p$ primes it is given; it does not sweep everything below a bound. Nothing here says how many non-rigid numbers there are below any bound, or that the smallest one found is the smallest that exists.

**What is proved:**
- No non-rigid number lies below $10^{30}$. This comes from the certified census of Chebyshev–Carmichael numbers below $10^{30}$ ([cheb-cn2-census-2](https://github.com/pudepiedj/cheb-cn2-census-2)), which contains every Carmichael number of order 2 below $10^{30}$. All 95 of them are rigid.
- So the smallest non-rigid number lies in $(10^{30},\ 4.63\times10^{36}]$, and the upper end is the smallest one in this dataset.

## What is a non-rigid number?

A **Carmichael number of order 2** (Howe 2000) is a squarefree composite $n$ with

$$n \equiv 1 \ \text{ or } \ n \equiv p \pmod{p^2-1} \quad\text{for every prime } p \mid n .$$

| at a prime $p \mid n$ | condition | the map $x \mapsto x^n$ on $\mathbb F_{p^2}$ |
|---|---|---|
| rigid | $n \equiv 1 \pmod{p^2-1}$ | the identity |
| type $p$ | $n \equiv p \pmod{p^2-1}$ | conjugation, $a+b\sqrt d \mapsto a-b\sqrt d$ |

A number is **rigid** if every prime is rigid, and **non-rigid** if at least one prime is of type $p$. Its type-$p$ primes form its **type-$p$ set** $T(n)$. Its **rigid part** $L_R(n)$ is $\mathrm{lcm}(q^2-1)$ over its rigid primes $q$, and $n \equiv 1 \pmod{L_R(n)}$.

Howe's own example, in his §5, is about $3.92\times10^{59}$, with type-$p$ prime 1153. It is in this dataset, and **289 of the numbers here are smaller than it**.

## The smallest examples

| $n$ | $\log_{10} n$ | type-$p$ | prime factors |
|---|---|---|---|
| 4626955899583332366834728873711784001 | 36.67 | 193 | 23 · 37 · 43 · 61 · 79 · 109 · 151 · **193** · 307 · 373 · 727 · 883 · 1871 · 2549 · 23561 |
| 87980197053523993194874215980503779281023681 | 43.94 | 193 | 37 · 59 · 71 · 83 · 89 · 181 · **193** · 233 · 419 · 991 · 1559 · 1871 · 4523 · 5279 · 15313 · 21319 |
| 131183934137807514121702103390897001253718401 | 44.12 | 1153 | 23 · 53 · 67 · 89 · 113 · 151 · 199 · 433 · 911 · 991 · **1153** · 1301 · 4159 · 5851 · 15809 · 23561 |
| 1357270055042801903716418620368722966717379361 | 45.13 | 673 | 23 · 29 · 37 · 89 · 103 · 109 · 131 · 239 · 281 · 307 · 311 · 379 · **673** · 883 · 1429 · 1861 · 3079 · 35531 |
| 4875744419959186574801326065156868496290845601 | 45.69 | 673 | 41 · 43 · 47 · 53 · 79 · 109 · 113 · 131 · 151 · 229 · 239 · 271 · 419 · **673** · 911 · 919 · 1427 · 2851 · 4049 |

For the first, $n \equiv 1 \pmod{q^2-1}$ at fourteen of its primes and $n \equiv 193 \pmod{193^2-1}$ at the fifteenth. Anyone can check this from the factorisation.

## What the dataset holds

| type-$p$ primes per number | numbers | smallest |
|---|---|---|
| one | 1,135,572 | $10^{36.67}$ |
| two | 32 | $10^{77.32}$ (type-$p$ primes 193 and 27457) |
| three | 2 | $10^{123.29}$ (type-$p$ primes 673, 2017 and 2593) |
| **all** | **1,135,606** | |

As far as we know, the numbers with two or three type-$p$ primes are the first of their kind found. The two with three are the only two found.

The single type-$p$ numbers use 93 different type-$p$ primes. The counts reflect where we searched, not how such numbers are distributed:

| type-$p$ prime | numbers |
|---|---|
| 193 | 769,225 |
| 1153 | 186,342 |
| 14593 | 42,554 |
| 6337 | 23,430 |
| 26881 | 22,913 |
| 9601 | 20,961 |
| 673 | 20,304 |
| 86 others | 49,843 |

`data/cn2_nonrigid_all.summary.json` has the full breakdown.

## Files

| file | contents |
|---|---|
| **release asset** `cn2_nonrigid_all.tsv.gz` (100 MB) | the whole dataset; see below |
| `data/SHA256SUMS` | the SHA-256 hash of the release asset |
| `data/cn2_nonrigid_all.summary.json` | counts by number of type-$p$ primes and by type-$p$ prime |
| `data/a175531_nonrigid_examples.txt` | a 303-number selection, suitable for an OEIS a-file: the 289 found below Howe's example, the 12 smallest with two type-$p$ primes and the 2 with three |
| `verify_nonrigid.py` | checks every number from scratch, standard library only |
| `tools/` | the search and collection programs, as used (below) |

The dataset is too large for the repository itself, so it is attached to the [release](https://github.com/pudepiedj/carmichael-order2-nonrigid/releases). It is gzip-compressed and tab-separated, with one header line and then one row per number, sorted by $n$:

| column | meaning |
|---|---|
| `log10` | $\log_{10} n$, to 3 decimals |
| `k` | the number of prime factors |
| `type_p` | the type-$p$ primes, comma-separated |
| `L_R` | the rigid part, $\mathrm{lcm}(q^2-1)$ over the rigid primes $q$ |
| `times_found` | how many searches (generator, type-$p$ set) produced $n$ |
| `n` | the number |
| `primes` | its prime factors, space-separated |

## Checking it

Download the release asset, then:

```bash
shasum -a 256 -c data/SHA256SUMS
python3 verify_nonrigid.py cn2_nonrigid_all.tsv.gz --afile data/a175531_nonrigid_examples.txt
```

For every row, `verify_nonrigid.py` checks:
- the listed factors are distinct primes, and their product is $n$;
- $n \equiv 1$ or $p \pmod{p^2-1}$ at every prime, so $n$ is of order 2;
- the type-$p$ primes are exactly those listed, and there is at least one, so $n$ is non-rigid;
- `L_R` and `log10` are correct;
- the rows are strictly increasing, so no number appears twice.

It also checks that every a-file number is in the dataset with the same type-$p$ primes. It needs only Python 3; on 24 cores it takes about 20 seconds. The last run gave:

```
"rows": 1135606, "verified": 1135606, "failed": 0, "order_violations": 0
a-file: 303 numbers, 303 found in the dataset with the same type-p primes
ALL ROWS VERIFIED
```

## How they were found

Howe's construction fixes a **search modulus** $L_0$ and a type-$p$ set $T$. It then looks for

$$n = \Big(\prod_{p \in T} p\Big)\, n_0, \qquad n_0 \text{ squarefree over the pool } P(2,L_0) = \lbrace q \nmid L_0 : q^2-1 \mid L_0\rbrace,$$

with $n \equiv 1 \pmod{L_0}$, $n \equiv 1 \pmod{q^2-1}$ at every $q \mid n_0$, and $n \equiv p \pmod{p^2-1}$ at every $p \in T$. A meet-in-the-middle (Schroeppel–Shamir) engine solves each such problem exhaustively. **For a fixed $(L_0, T)$, the search is complete:** any non-rigid $n$ with $T(n) = T$ and $L_R(n) = L_0$ is found. The dataset is incomplete only because just some pairs $(L_0, T)$ were searched:

| search | pairs $(L_0, T)$ | numbers found |
|---|---|---|
| the "box": all 151,133 generators $L_0 \le 2^{50}$ over the primes up to 31, with $T$ = {193}, {673}, {1153} or {2017} | 233,907 | 1,016,196 |
| ten more type-$p$ primes (6337 … 26881), on the pairs carrying 99% of the expected count of small numbers | 139,462 | 163,832 |
| earlier sweeps of individual generators | 3,851 | 1,465 |
| five small type-$p$ primes (97, 241, 337, 433, 577) on the box | 60,694 | 53 |
| an extended box: one extra prime from 37 to 251, $L_0 \le 2^{46}$ | 4,511,844 | 215 |
| two type-$p$ primes: one generator with {673, 2017}, plus 100 small pairs | 101 | 33 |
| three type-$p$ primes: {673, 2017, 2593} on one generator | 1 | 2 |

The counts are hits. A number found by more than one search is counted once in the dataset, and `times_found` records how often it turned up.

A simple count model predicts the hits closely, for example 1,013,982 predicted against 1,016,196 found in the box. The model gives $2^{\lvert P'\rvert}/\varphi(L)$ per pair, where $L = \mathrm{lcm}(L_0, p^2-1 : p \in T)$ and $P'$ is the pool less any prime dividing some $p^2-1$.

The programs in `tools/` are the ones used, copied as they were run:

| program | role |
|---|---|
| `cn2_systematic_fast_SS.py` | the meet-in-the-middle engine (takes a target residue) |
| `cn2_nonrigid_search.py` | Howe's construction on one $(L_0, T)$: `search`, `search_multi` |
| `cn2_nonrigid_scout.py` | pools and admissibility of type-$p$ primes |
| `cn2_box_fecundity.py`, `cn2_box_nonrigid.py` | the box and the extended box |
| `cn2_ey_scout.py`, `cn2_pairs_run.py` | the ten further type-$p$ primes |
| `cn2_twin_scout.py`, `cn2_twin_run.py`, `cn2_twin_recover.py` | two type-$p$ primes |
| `cn2_nonrigid_collect.py` | merges every search's results, re-verifies them, and writes the dataset |
| `cn2_nonrigid_afile.py`, `cn2_nonrigid_minprime.py` | the a-file selection; smallest-prime statistics |

They need `sympy` and `numpy`, and they expect to run in one working folder, where they read and write their run records. The raw run records (about 0.8 GB) are not included. Everything in them that bears on the numbers is in the dataset. For example, Howe's 53 examples and the smallest number above are reproduced by:

```bash
cd tools
python3 cn2_nonrigid_search.py 25113661372800 1153     # Howe's 53, in about 8 seconds
python3 cn2_nonrigid_search.py 16670275221600 193      # the 10^36.67 number
```

## Licence

MIT. See `LICENSE`.
