# Thick corner-to-corner band seeds for one puzzle (pjmathematician's recipe, forum 755628, 4 Nov 2026): one band of
# tilted squares along a diagonal of the box, its width and offset treated as search variables, the rest filled exactly
# with upright squares (Evan Daniel's fillfast). Several rigid seeds per n; an annealing-style optimiser (fq quench +
# packer hop at temp 1.0, hop/sq10_hop.py quenchhop) then relaxes each one, which is where the multi-angle band forms.
# The band search is design/bandsearch.py (straight lattice band, any tilt, width, offset, phase), here with the band
# width bounded per seed so the seeds really differ in width. If the exact fill holds fewer than n squares, the tolerant
# fill adds slightly overlapping squares (tau) and the optimiser has to repair them.
# Usage: python3 bandseed.py N S SECONDS SEED WMIN WMAX OUT.csv     (writes competition format, angles in radians)
import json, math, os, random, sys, time
sys.path.insert(0, os.environ.get('EVAND_PACKER', '/content/evand/search/packer'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fillfast
import bandsearch as B

THS = [math.degrees(math.atan(x)) for x in (1 / 2, 1 / 3, 2 / 3, 3 / 4, 1 / 4, 2 / 5, 3 / 5)] + [45.0, 26.9, 63.1, 18.0, 30.0, 40.0]


def search(n, s, secs, seed, wmin, wmax):
    rng = random.Random(seed)
    best = (-1, None)
    t0 = time.time(); it = 0

    def rnd():
        return [rng.choice(THS) + rng.gauss(0, 0.5) * (rng.random() < 0.4), rng.uniform(wmin, wmax), rng.uniform(-2.5, 2.5),
                rng.random(), rng.random(), rng.random() < 0.5]
    cur = rnd(); cc = B.count(s, cur)[0]
    while time.time() - t0 < secs:
        it += 1
        if rng.random() < 0.15:
            cand = rnd()
        else:
            cand = list(cur); k = rng.randrange(5)
            cand[k] += rng.gauss(0, [0.3, 0.3, 0.2, 0.1, 0.1][k])
            cand[3] %= 1; cand[4] %= 1
            cand[1] = min(wmax, max(wmin, cand[1]))
        c2 = B.count(s, cand)[0]
        if c2 >= cc:
            cur, cc = cand, c2
        if c2 > best[0]:
            best = (c2, list(cand))
    return best, it


def main():
    n, s, secs, seed = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    wmin, wmax, out = float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
    (c, prm), it = search(n, s, secs, seed, wmin, wmax)
    T = B.band(s, *prm)
    X = fillfast.fill(s, T, 'exact', time_limit=10)
    tau = 0.0
    for t in (0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16):
        if len(X) >= n:
            break
        tau = t
        X = fillfast.fill(s, T, 'exact', tau=t, time_limit=10)
    info = dict(n=n, s=s, seed=seed, wmin=wmin, wmax=wmax, exact_count=c, iters=it, tau=tau, final_count=len(X),
                tilted=len(T), params=[round(v, 4) if isinstance(v, float) else v for v in prm])
    print('BANDSEED ' + json.dumps(info), flush=True)
    if len(X) < n:
        print('no seed: fewer than n squares even with the tolerant fill', flush=True)
        return
    X = list(X)
    # keep all tilted squares, drop surplus uprights (the ones nearest a wall go last, so drop from the end)
    tilted = [q for q in X if abs((q[2] % 90)) > 1e-6]
    upright = [q for q in X if abs((q[2] % 90)) <= 1e-6]
    keep = (tilted + upright)[:n]
    with open(out, 'w') as f:
        f.write('id,n,square_id,x,y,angle\n')
        for j, (x, y, d) in enumerate(keep):
            f.write('%d,%d,%d,%.12f,%.12f,%.12f\n' % (j, n, j, x, y, math.radians(d)))


if __name__ == '__main__':
    main()
