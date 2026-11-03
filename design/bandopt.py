# Band surgery with exact refill: keep the tilted squares T of our packing, transform them (band translate / rotate /
# retilt / stretch along the band, drop or add end squares), and recompute the maximum upright fill exactly (Evan Daniel's
# fillfast). s*(T) = smallest side at which |T| + fill >= n (bisection). Hill climb on s*. Uprights are re-chosen
# combinatorially at every step, which a continuous quench (fq) cannot do.
import os, sys, math, time, random, json, csv, numpy as np
sys.path.insert(0, os.environ.get('EVAND_PACKER', '/content/evand/search/packer'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hop'))
import fillfast, gen, sq09_requench as R

def count(s, T):
    hw = 0.5 * (np.abs(np.cos(np.radians(T[:, 2]))) + np.abs(np.sin(np.radians(T[:, 2]))))
    if (T[:, 0] - hw < -1e-9).any() or (T[:, 1] - hw < -1e-9).any() or (T[:, 0] + hw > s + 1e-9).any() or (T[:, 1] + hw > s + 1e-9).any():
        return -1, None
    # tilted squares must not overlap each other (the fill only checks uprights against them)
    P = T[:, :2]; D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    I, J = np.nonzero(np.triu(D < 1.4143, 1))
    for i, j in zip(I, J):
        if gen.pen(tuple(T[i]), tuple(T[j])) > 1e-9:
            return -1, None
    try:
        X = fillfast.fill(s, [tuple(map(float, q)) for q in T], 'exact', time_limit=5)
    except Exception:
        return -1, None
    return len(X), X

def sstar(T, n, lo, hi, tol=2e-5):
    c, X = count(hi, T)
    if c < n:
        return 9e9, None
    best = X
    while hi - lo > tol:
        m = 0.5 * (lo + hi); c, X = count(m, T)
        if c >= n: hi, best = m, X
        else: lo = m
    return hi, best

def tilted_set(rs):
    a = R.arr(rs); h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    a = a - [(a[:, 0] - h).min(), (a[:, 1] - h).min(), 0]
    th = np.mod(a[:, 2], math.pi / 2); tl = np.abs(np.mod(th + math.pi / 4, math.pi / 2) - math.pi / 4) > 0.02
    T = a[tl].copy(); T[:, 2] = np.degrees(T[:, 2]) % 90
    return T, R.side_arr(a)

def move(T, rng, s):
    T = T.copy(); k = rng.randrange(7); c = T[:, :2].mean(0)
    if k == 0:   # translate the band
        T[:, :2] += rng.gauss(0, 0.05), rng.gauss(0, 0.05)
    elif k == 1: # rotate the band rigidly (positions and angles)
        a = math.radians(rng.gauss(0, 0.3)); R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        T[:, :2] = c + (T[:, :2] - c) @ R2.T; T[:, 2] = (T[:, 2] + math.degrees(a)) % 90
    elif k == 2: # retilt squares in place
        T[:, 2] = (T[:, 2] + rng.gauss(0, 0.3)) % 90
    elif k == 3: # stretch along the principal axis
        cc = T[:, :2] - c; w, v = np.linalg.eigh(cc.T @ cc); d = v[:, 1]; t = cc @ d
        T[:, :2] = c + cc + np.outer(t * rng.gauss(0, 0.004), d)
    elif k == 4: # move one square
        i = rng.randrange(len(T)); T[i, :2] += rng.gauss(0, 0.05), rng.gauss(0, 0.05); T[i, 2] = (T[i, 2] + rng.gauss(0, 0.5)) % 90
    elif k == 5 and len(T) > 5: # drop a band square at an end
        cc = T[:, :2] - c; w, v = np.linalg.eigh(cc.T @ cc); t = cc @ v[:, 1]
        T = np.delete(T, int(np.argmax(t) if rng.random() < 0.5 else np.argmin(t)), axis=0)
    else:        # shift a contiguous half of the band (bend point)
        cc = T[:, :2] - c; w, v = np.linalg.eigh(cc.T @ cc); t = cc @ v[:, 1]; cut = rng.uniform(t.min(), t.max())
        m = t > cut; T[m, :2] += rng.gauss(0, 0.04), rng.gauss(0, 0.04)
    return T

def main():
    n, secs, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    src = sys.argv[4] if len(sys.argv) > 4 else os.environ.get('OURS_CSV', 'ours.csv')
    rows = R.load(src); T, s0 = tilted_set(rows[n]); rng = random.Random(seed)
    lo = s0 - 0.06
    st, X = sstar(T, n, lo, s0 + 0.002)
    print(f'n {n} ours {s0:.6f} |T| {len(T)} s*(T) {st:.6f}', flush=True)
    best, bestT, bestX = st, T, X; cur, curT = st, T; t0 = time.time(); it = 0
    while time.time() - t0 < secs:
        it += 1
        T2 = move(curT, rng, cur)
        s2, X2 = sstar(T2, n, lo, cur + 0.001)
        if s2 <= cur + 1e-6 or rng.random() < 0.02 and s2 < cur + 0.003:
            curT, cur = T2, s2
            if s2 < best - 1e-6:
                best, bestT, bestX = s2, T2, X2
                print(f'  it {it} {time.time()-t0:.0f}s s* {best:.6f} ({best - s0:+.5f}) |T| {len(T2)}', flush=True)
    print(json.dumps(dict(n=n, ours=s0, best=best, gain=s0 - best, iters=it)))
    if bestX is not None and best < s0 - 1e-5:
        with open(f'seed_{n}_{seed}.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
            for i, (x, y, d) in enumerate(bestX[:n]):
                w.writerow([i, n, i, repr(float(x)), repr(float(y)), repr(math.radians(float(d)))])

if __name__ == '__main__':
    main()
