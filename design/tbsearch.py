# Parameter search for design/twoblock.py: for one puzzle (n, L) and a set of band thicknesses (N1, N2), random restarts
# plus hill climbing over the band lengths h1, h2, the lattice phases q1, q2, the wall offsets x01, x02 in {0, 1} and the
# mirror flags m1, m2 (rows up-right or down-right) and the bend positions ys1, ys2 (0 = straight band).
# Usage: python3 tbsearch.py N L SECONDS SEED N1LIST N2LIST OUTPREFIX [INIT_best.json]   (N1LIST like 5,6,7; writes
# OUTPREFIX_best.json and the best packings OUTPREFIX_<count>_<k>.csv in competition format; INIT continues an earlier run;
# TB_LEDGES=1 in the environment adds twoblock's ledge candidates, 5x slower per assembly)
import json, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twoblock as T


def evaluate(n, L, prm, cache):
    key = tuple(round(v, 4) for v in prm)
    if key in cache:
        return cache[key][0]
    N1, h1, q1, N2, h2, q2, x01, x02, m1, m2, ys1, ys2 = prm
    cnt, P, info = T.build(n, L, int(N1), h1, q1, int(N2), h2, q2, None, None, x01, x02, int(m1), int(m2), time_limit=10.0,
                           ys1=ys1 if ys1 > 0 else None, ys2=ys2 if ys2 > 0 else None, ledges=os.environ.get('TB_LEDGES') == '1')
    cache[key] = (cnt, P)
    return cnt


def search(n, L, secs, seed, n1s, n2s, prefix, init=None):
    rng = random.Random(seed)
    cache = {}
    best = {}  # (N1, N2) -> (count, prm)
    if init:  # continue from the results of an earlier run (its _best.json)
        for e in json.load(open(init))['best']:
            p = list(e['prm']) + [0.0] * (12 - len(e['prm']))
            if (int(p[0]), int(p[3])) not in best and e['count'] > 0:
                c0 = evaluate(n, L, p, cache)  # re-evaluated: the earlier run may have used another L
                if c0 > 0:
                    best[(int(p[0]), int(p[3]))] = (c0, p)
    t0 = time.time()
    it = 0
    f = L - math.floor(L + 1e-9)

    def rnd(N1, N2):
        c1 = math.cos(T.band_angle(N1, f)); c2 = math.cos(T.band_angle(N2, f))
        return [N1, rng.uniform(0.45 * L, L - 1.0), rng.uniform(0, 1 / c1), N2, rng.uniform(0.45 * L, L - 1.0), rng.uniform(0, 1 / c2),
                1.0 if rng.random() < 0.15 else 0.0, 1.0 if rng.random() < 0.15 else 0.0,
                float(rng.random() < 0.5), float(rng.random() < 0.5),
                rng.uniform(2, 0.8 * L) if rng.random() < 0.5 else 0.0, rng.uniform(2, 0.8 * L) if rng.random() < 0.5 else 0.0]
    combos = [(a, b) for a in n1s for b in n2s]
    while time.time() - t0 < secs:
        N1, N2 = rng.choice(combos)
        c1 = math.cos(T.band_angle(N1, f)); c2 = math.cos(T.band_angle(N2, f))
        if (N1, N2) in best and rng.random() < 0.6:
            cur = list(best[(N1, N2)][1])
        else:
            cur = rnd(N1, N2)
        cc = evaluate(n, L, cur, cache); it += 1
        stall = 0
        while stall < 25 and time.time() - t0 < secs:
            cand = list(cur)
            k = rng.choice([1, 2, 4, 5, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 10, 11])
            if k in (1, 4):
                cand[k] += rng.gauss(0, 0.5) if rng.random() < 0.7 else rng.uniform(-3, 3)
                cand[k] = min(L - 0.5, max(2.0, cand[k]))
            elif k == 2:
                cand[k] = (cand[k] + rng.gauss(0, 0.15)) % (1 / c1)
            elif k == 5:
                cand[k] = (cand[k] + rng.gauss(0, 0.15)) % (1 / c2)
            elif k in (10, 11):
                if cand[k] <= 0 or rng.random() < 0.2:
                    cand[k] = rng.uniform(2, 0.8 * L) if (cand[k] <= 0 or rng.random() < 0.5) else 0.0
                else:
                    cand[k] = min(0.9 * L, max(2.0, cand[k] + rng.gauss(0, 0.7)))
            else:
                cand[k] = 1.0 - cand[k]
            # a bent band slides one column towards the wall above the bend: the lower part must sit one column off
            # the wall (rows up-right) or at the wall (rows down-right)
            for kb, kx, km in ((10, 6, 8), (11, 7, 9)):
                if cand[kb] > 0:
                    cand[kx] = 0.0 if cand[km] > 0.5 else 1.0
            c2v = evaluate(n, L, cand, cache); it += 1
            if c2v >= cc:
                if c2v > cc:
                    stall = 0
                else:
                    stall += 1
                cur, cc = cand, c2v
            else:
                stall += 1
            if (N1, N2) not in best or cc > best[(N1, N2)][0]:
                best[(N1, N2)] = (cc, list(cur))
                print('%.0fs it %d best %s -> %d  prm %s' % (time.time() - t0, it, (N1, N2), cc, [round(v, 3) for v in cur]), flush=True)
    # write results
    allbest = sorted(best.values(), key=lambda v: -v[0])
    res = dict(n=n, L=L, seed=seed, iters=it, best=[dict(count=c, prm=p) for c, p in allbest])
    with open(prefix + '_best.json', 'w') as fh:
        json.dump(res, fh, indent=1)
    for k, (c, p) in enumerate(allbest[:6]):
        key = tuple(round(v, 4) for v in p)
        P = cache[key][1]
        if P is None:
            P = T.build(n, L, int(p[0]), p[1], p[2], int(p[3]), p[4], p[5], None, None, p[6], p[7], int(p[8]), int(p[9]),
                        ys1=p[10] if p[10] > 0 else None, ys2=p[11] if p[11] > 0 else None, ledges=os.environ.get('TB_LEDGES') == '1')[1]
        T.write_csv(P, '%s_%d_%d.csv' % (prefix, c, k), n)
    print('DONE n %d L %.4f iters %d best %s' % (n, L, it, [(c, [round(v, 3) for v in p]) for c, p in allbest[:8]]), flush=True)


if __name__ == '__main__':
    n, L, secs, seed = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    n1s = [int(v) for v in sys.argv[5].split(',')]
    n2s = [int(v) for v in sys.argv[6].split(',')]
    search(n, L, secs, seed, n1s, n2s, sys.argv[7], sys.argv[8] if len(sys.argv) > 8 else None)
