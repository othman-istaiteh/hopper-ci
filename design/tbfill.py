# Fill a rigid two-block assembly up to n squares: each missing square goes to the spot of largest clearance (the centre
# of the biggest empty region, found on a fine grid by the smallest penetration depth against the placed squares and the
# walls, over a few angles). Small overlaps are left for the relaxer (fq quench repairs about 0.1).
# Usage: python3 tbfill.py IN.csv N L OUT.csv [--rows]  (competition format in and out, angles in radians; --rows inserts
# each missing square into the longest upright row instead, re-spacing the row so the overlap is spread thinly)
import math, sys
import numpy as np
from scipy.spatial import cKDTree


def half_ext(t, ang):
    d = t - ang
    return 0.5 * (abs(math.cos(d)) + abs(math.sin(d)))


def penetration(sq, others):
    # largest penetration depth of unit square sq against the squares in others (SAT depth, 0 if separated)
    worst = 0.0
    x, y, t = sq
    for ox, oy, ot in others:
        dx, dy = ox - x, oy - y
        if dx * dx + dy * dy >= 2.0:
            continue
        depth = 9.0
        for base in (t, ot):
            for ang in (base, base + math.pi / 2):
                ux, uy = math.cos(ang), math.sin(ang)
                d = half_ext(t, ang) + half_ext(ot, ang) - abs(dx * ux + dy * uy)
                if d < depth:
                    depth = d
        if depth > worst:
            worst = depth
    return worst


def wall_pen(sq, L):
    x, y, t = sq
    h = half_ext(t, 0.0)
    return max(0.0, h - x, h - y, x + h - L, y + h - L)


def fill(P, n, L, angles=(0.0, math.pi / 4, 0.3, 0.6, 0.9, 1.2), step=0.05, verbose=True):
    P = [tuple(p) for p in P]
    added = []
    while len(P) < n:
        A = np.array(P)
        tree = cKDTree(A[:, :2])
        g = np.arange(0.5, L - 0.5 + 1e-9, step)
        X, Y = np.meshgrid(g, g)
        pts = np.c_[X.ravel(), Y.ravel()]
        dist, _ = tree.query(pts)
        order = np.argsort(-dist)[:400]
        best = None
        for k in order:
            px, py = pts[k]
            idx = tree.query_ball_point([px, py], 1.6)
            others = [P[i] for i in idx]
            for t in angles:
                sq = (float(px), float(py), t)
                pen = max(penetration(sq, others), 100.0 * wall_pen(sq, L))  # the box is hard: stay inside
                if best is None or pen < best[0]:
                    best = (pen, sq)
        pen, sq = best
        # local refinement around the best spot
        for _ in range(2):
            px, py, t = sq
            idx = tree.query_ball_point([px, py], 2.0)
            others = [P[i] for i in idx]
            for ddx in np.linspace(-step, step, 5):
                for ddy in np.linspace(-step, step, 5):
                    for dt in (-0.1, 0.0, 0.1):
                        s2 = (px + ddx, py + ddy, (t + dt) % (math.pi / 2))
                        p2 = max(penetration(s2, others), 100.0 * wall_pen(s2, L))
                        if p2 < pen:
                            pen, sq = p2, s2
            step = step / 2
        P.append(sq)
        added.append((sq, pen))
        if verbose:
            print('added square %d at (%.3f, %.3f, %.3f) penetration %.4f' % (len(P), sq[0], sq[1], sq[2], pen), flush=True)
    return P, added


def fill_rows(P, n, L, verbose=True):
    # variant: each missing square is inserted into the longest horizontal run of upright squares (same y, unit pitch),
    # the run re-spaced to pitch (m - 1) / m: the overlap is spread thinly along the row instead of one deep hole
    P = [tuple(p) for p in P]
    used = set()
    while len(P) < n:
        ups = [(i, p) for i, p in enumerate(P) if abs(p[2]) < 1e-9 or abs(p[2] - math.pi / 2) < 1e-9]
        rows = {}
        for i, p in ups:
            rows.setdefault(round(p[1], 6), []).append((p[0], i))
        best = None
        for y, lst in rows.items():
            if y in used:
                continue
            lst.sort()
            # maximal runs of consecutive unit-spaced squares
            run = [lst[0]]
            for a, b in zip(lst, lst[1:]):
                if abs(b[0] - a[0] - 1.0) < 1e-6:
                    run.append(b)
                else:
                    if best is None or len(run) > len(best[1]):
                        best = (y, list(run))
                    run = [b]
            if best is None or len(run) > len(best[1]):
                best = (y, list(run))
        y, run = best
        used.add(y)
        m = len(run)
        x0 = run[0][0] - 0.5
        pitch = (m - 1.0) / m  # m + 1 unit squares in a span of m
        for k, (x, i) in enumerate(run):
            P[i] = (x0 + 0.5 + k * pitch, y, P[i][2])
        P.append((x0 + 0.5 + m * pitch, y, 0.0))
        if verbose:
            print('row y=%.3f: %d -> %d squares, pitch %.4f (overlap %.4f)' % (y, m, m + 1, pitch, 1 - pitch), flush=True)
    return P


def read_csv(path):
    P = []
    with open(path) as fh:
        next(fh)
        for line in fh:
            r = line.strip().split(',')
            if len(r) >= 6:
                P.append((float(r[3]), float(r[4]), float(r[5])))
    return P


def write_csv(P, path, n):
    with open(path, 'w') as fh:
        fh.write('id,n,square_id,x,y,angle\n')
        for k, (x, y, t) in enumerate(P):
            fh.write('%d,%d,%d,%.12f,%.12f,%.12f\n' % (k, n, k, x, y, t))


if __name__ == '__main__':
    inp, n, L, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    P = read_csv(inp)
    print('input %d squares, need %d' % (len(P), n))
    if '--rows' in sys.argv:
        P = fill_rows(P, n, L)
        write_csv(P, out, n)
        print('wrote %s with %d squares' % (out, len(P)))
    else:
        P, added = fill(P, n, L)
        write_csv(P, out, n)
        print('wrote %s with %d squares, max penetration %.4f' % (out, len(P), max([a[1] for a in added], default=0.0)))
