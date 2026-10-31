# Lattice-band search: a band of tilted unit squares on a perfect lattice (angle th, touching rows), along the diagonal
# or anti-diagonal of a box of side s, width w, offset u0, lattice phase (p1, p2); the rest filled exactly with axis
# squares (Evan Daniel's fillfast, corner lattices). Count = tilted + axis. Random search + hill climbing.
import os, sys, math, time, random, json, numpy as np
sys.path.insert(0, os.environ.get('EVAND_PACKER', '/content/evand/search/packer'))
import fillfast
H = 0.5

def band(s, th, w, u0, p1, p2, anti, gap=1e-6):
    t = math.radians(th); c, sn = math.cos(t), math.sin(t)
    e1 = np.array([c, sn]) * (1 + gap); e2 = np.array([-sn, c]) * (1 + gap)
    d = np.array([1, -1]) / math.sqrt(2) if anti else np.array([1, 1]) / math.sqrt(2)
    dp = np.array([-d[1], d[0]])
    ctr = np.array([s / 2, s / 2])
    m = int(s * 1.5) + 2
    I, J = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
    P = ctr + (I.ravel() + p1)[:, None] * e1 + (J.ravel() + p2)[:, None] * e2
    u = (P - ctr) @ dp
    hw = 0.5 * (abs(c) + abs(sn))  # half extent of a tilted square along x or y
    ok = (np.abs(u - u0) <= w / 2) & (P[:, 0] >= hw) & (P[:, 1] >= hw) & (P[:, 0] <= s - hw) & (P[:, 1] <= s - hw)
    return [(float(x), float(y), th % 90) for x, y in P[ok]]

def count(s, prm):
    T = band(s, *prm)
    if not T:
        return 0, T, []
    X = fillfast.fill(s, T, 'exact', time_limit=5)
    return len(X), T, X

def search(n, s, secs, seed, ths):
    rng = random.Random(seed)
    best = (-1, None)
    t0 = time.time(); it = 0
    def rnd():
        return [rng.choice(ths) + rng.gauss(0, 0.3) * (rng.random() < 0.3), rng.uniform(1.5, 8.0), rng.uniform(-2, 2),
                rng.random(), rng.random(), rng.random() < 0.5]
    cur = rnd(); cc = count(s, cur)[0]
    while time.time() - t0 < secs:
        it += 1
        if rng.random() < 0.15:
            cand = rnd()
        else:
            cand = list(cur); k = rng.randrange(5)
            cand[k] += rng.gauss(0, [0.3, 0.3, 0.2, 0.1, 0.1][k])
            cand[3] %= 1; cand[4] %= 1
            cand[1] = max(0.8, cand[1])
        c2 = count(s, cand)[0]
        if c2 >= cc:
            cur, cc = cand, c2
        if c2 > best[0]:
            best = (c2, list(cand))
    return best, it

if __name__ == '__main__':
    n, s, secs = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    ths = [math.degrees(math.atan(x)) for x in (1/2, 1/3, 2/3, 3/4, 1/4, 2/5, 3/5)] + [26.9, 63.1, 18.0, 27.0]
    best, it = search(n, s, secs, int(sys.argv[4]) if len(sys.argv) > 4 else 1, ths)
    print(json.dumps(dict(n=n, s=s, best=best[0], need=n, params=best[1], iters=it)))
