# At the board side B (+eps), scale our tilted band into the smaller box and count tilted + exact upright fill; then hill
# climb (band moves + delete/insert band squares) to maximise the count. Deficit = n - count says how far the design is.
import os, sys, math, time, random, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bandopt import count, tilted_set, move, R
def main():
    n, secs, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    B = {int(l.split()[0]): float(l.split()[1]) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hop', 'board_29oct.txt')) if l.strip() and not l.startswith('#')}
    rows = R.load(os.environ.get('OURS_CSV', 'ours.csv'))
    T, s0 = tilted_set(rows[n]); s = B[n] + 0.002; rng = random.Random(seed)
    T = T.copy(); T[:, :2] -= (s0 - s) / 2
    hw = 0.5 * (np.abs(np.cos(np.radians(T[:, 2]))) + np.abs(np.sin(np.radians(T[:, 2]))))
    T = T[(T[:, 0] - hw >= 0) & (T[:, 1] - hw >= 0) & (T[:, 0] + hw <= s) & (T[:, 1] + hw <= s)]
    c, X = count(s, T); best, cur, curT = c, c, T
    print(f'n {n} ours {s0:.5f} board {B[n]:.5f} at s {s:.5f}: count {c} of {n} (deficit {n - c})', flush=True)
    t0 = time.time(); it = 0
    while time.time() - t0 < secs:
        it += 1
        T2 = move(curT, rng, s)
        if rng.random() < 0.15:  # insert a band square next to a random band square along the row direction
            i = rng.randrange(len(T2)); a = math.radians(T2[i, 2]); d = rng.choice([(math.cos(a), math.sin(a)), (-math.cos(a), -math.sin(a)), (-math.sin(a), math.cos(a)), (math.sin(a), -math.cos(a))])
            T2 = np.vstack([T2, [T2[i, 0] + d[0] * 1.0005, T2[i, 1] + d[1] * 1.0005, T2[i, 2]]])
        c2, X2 = count(s, T2)
        if c2 >= cur or rng.random() < 0.05 and c2 >= cur - 1:
            cur, curT = c2, T2
            if c2 > best:
                best = c2; print(f'  it {it} {time.time()-t0:.0f}s count {best} (deficit {n - best}) |T| {len(T2)}', flush=True)
    print(json.dumps(dict(n=n, board=B[n], best=best, deficit=n - best, iters=it)))
main()
