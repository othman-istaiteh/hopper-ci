# Two-block design (the #1 team's structure seen in their published n = 299): two thick wall bands of tilted unit squares,
# one standing against the left wall from the bottom corner upward, one lying against the top wall from the right corner
# leftward, the rest filled with upright squares on the four wall-aligned unit grids (exact fill by a small MILP).
#
# Wall band: a brick lattice of unit squares at angle t, rows of N squares along e1 = (cos t, sin t), consecutive rows
# stacked straight along the wall with step 1 / cos t (offset (tan t, 1) in the square frame, so rows touch without
# overlap and every row covers the same strip of width W = N cos t + sin t across the wall). The angle is set by the
# strip: N cos t + sin t = N - 1 + f', so that the band replaces N - 1 upright columns plus the fractional part f' of the
# box and K - N + 1 upright columns fit beside it exactly. The band's ends are staircases (lattice squares fully inside
# the box and below the band length h); the MILP decides the end squares together with the upright fillers.
#
# Usage: python3 twoblock.py N L N1 h1 q1 N2 h2 q2 [f1p f2p x01 x02 m1 m2 ys1 ys2] [--out FILE.csv]
#   N1, N2 rows across the left / top band; h1, h2 band lengths along their walls; q1, q2 lattice phases along the wall
#   (in [0, 1 / cos t)); f1p, f2p effective fractional widths (default f = L - floor(L)); x01, x02 band offsets from the
#   wall (default 0); m1, m2 = 1 mirror the band (rows running down-right instead of up-right); ys1, ys2 > 0 bend the
#   band at that distance along the wall (the leader's one-column slide, see band_squares).
#   Prints the count; --out writes the packing in competition format (angles in radians, box from 0).
import math, sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

TOL = 1e-9


def band_angle(N, fp):
    # solve N cos t + sin t = N - 1 + fp for t in (0, 45 deg)
    lo, hi = 0.0, math.pi / 4
    g = lambda t: N * math.cos(t) + math.sin(t) - (N - 1 + fp)
    if g(hi) > 0:
        return hi
    for _ in range(100):
        m = 0.5 * (lo + hi)
        if g(m) > 0:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def band_squares(L, N, fp, h, q, x0, mirror=0, ext=0.0, ysplit=None):
    # left-wall band in wall coordinates (X across the wall, Y along it, from the corner); returns [(X, Y, t)] and t.
    # mirror = 1: rows run down-right instead of up-right (the band reflected along the wall), angle 90 - t.
    # ysplit (< h): the leader's bend. The rows below ysplit are the lattice at offset x0 from the wall; above it the
    # same brick lattice shifted one column towards the wall (x0 - 1, needs x0 >= 1; for a mirrored band away from the
    # wall, x0 + 1), its first row sitting on the last lower row with the along-wall offset (1 - sin t) / cos t: every
    # pair of squares of the two rows then touches along e2 exactly (the e2 separation s + dY cos t does not depend on
    # the column), so the junction is as dense as the lattice and the band's slack column changes side at the bend.
    t = band_angle(N, fp)
    c, s = math.cos(t), math.sin(t)
    hx = 0.5 * (c + s)
    step = 1.0 / c
    ta = (math.pi / 2 - t) if mirror else t
    sg = -1.0 if mirror else 1.0
    out = []
    ymax = min(h + ext, L)

    def rows(x0r, qr, blo, bhi, ylo, yhi):
        for a in range(N):
            X = x0r + hx + a * c
            if X + hx > L + TOL or X - hx < -TOL:
                continue
            y0 = qr + sg * a * s
            for b in range(blo, bhi + 1):
                Y = y0 + b * step
                if Y - hx >= max(0.0, ylo) - TOL and Y + hx <= yhi + TOL:
                    out.append((X, Y, ta))
    bmin = math.floor((-N * s - 1.0 - q) / step) - 1
    bmax = math.ceil((ymax + N * s + 1.0 - q) / step) + 1
    if ysplit is None or ysplit >= h:
        rows(x0, q, bmin, bmax, 0.0, ymax)
        return out, t
    # bent band: lower rows b with q + b step < ysplit (row position of column a = 0), upper lattice above
    blast = math.floor((ysplit - q - 1e-9) / step)
    rows(x0, q, bmin, blast, 0.0, ymax)
    x0u = x0 + (1.0 if mirror else -1.0)
    qu = q + blast * step + (1.0 - s) / c
    rows(x0u, qu, 0, bmax, 0.0, ymax)
    return out, t


def to_top(sq, L):
    # map wall coordinates of the top band (X down from the top wall, Y leftward from the right corner) to box coordinates
    return [(L - Y, L - X, (math.pi / 2 - t) % (math.pi / 2)) for X, Y, t in sq]


def half_extent(t, axis_angle):
    d = t - axis_angle
    return 0.5 * (abs(math.cos(d)) + abs(math.sin(d)))


def overlap(a, b, tol=TOL):
    # separating-axis test for two unit squares a = (x, y, t), b = (x, y, t); True if they overlap by more than tol
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx * dx + dy * dy >= 2.0:
        return False
    for t in (a[2], b[2]):
        for ang in (t, t + math.pi / 2):
            ux, uy = math.cos(ang), math.sin(ang)
            proj = abs(dx * ux + dy * uy)
            if proj >= half_extent(a[2], ang) + half_extent(b[2], ang) - tol:
                return False
    return True


def upright_candidates(L, f, offsets=None):
    # unit grids aligned to the walls (offsets 0 and f) or to extra fractional offsets; returns [(x, y, 0.0)] and the
    # witness coordinates (midpoints of centre pairs closer than 1: squares overlapping in x share one of them)
    K = int(math.floor(L + 1e-9))
    offs = [0.0, f] if offsets is None else offsets
    xs = sorted(set(round(i + 0.5 + o, 9) for i in range(K + 1) for o in offs if i + 1 + o <= L + 1e-9))
    pts = [(x, y, 0.0) for x in xs for y in xs]
    w = set()
    for i, a in enumerate(xs):
        for b in xs[i + 1:]:
            if b - a >= 1 - 1e-9:
                break
            w.add(round(0.5 * (a + b), 9))
    return pts, sorted(w)


def build(n, L, N1, h1, q1, N2, h2, q2, f1p=None, f2p=None, x01=0.0, x02=0.0, m1=0, m2=0, fix_margin=2.5, time_limit=20.0,
          offsets=None, ext=0.0, ys1=None, ys2=None, verbose=False):
    f = L - math.floor(L + 1e-9)
    if f1p is None:
        f1p = f
    if f2p is None:
        f2p = f
    B1, t1 = band_squares(L, N1, f1p, h1, q1, x01, m1, ext, ys1)
    B2w, t2 = band_squares(L, N2, f2p, h2, q2, x02, m2, ext, ys2)
    B2 = to_top(B2w, L)
    # fixed band interiors: squares away from the band ends along the wall
    fixed, free = [], []
    for X, Y, t in B1:
        (fixed if fix_margin <= Y <= h1 - fix_margin else free).append((X, Y, t))
    for (X, Y, t), sq in zip(B2w, B2):
        (fixed if fix_margin <= Y <= h2 - fix_margin else free).append(sq)
    # upright candidates not overlapping fixed squares, inside the box
    cands = []
    hx = 0.5
    upc, wx = upright_candidates(L, f, offsets)
    for x, y, t in upc:
        if x - hx < -TOL or x + hx > L + TOL or y - hx < -TOL or y + hx > L + TOL:
            continue
        cands.append((x, y, 0.0))
    # spatial hashing for overlap checks
    def cell(p):
        return (int(math.floor(p[0])), int(math.floor(p[1])))
    grid = {}
    for i, sq in enumerate(fixed):
        grid.setdefault(cell(sq), []).append(i)

    def hits_fixed(sq):
        cx, cy = cell(sq)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for i in grid.get((cx + dx, cy + dy), []):
                    if overlap(sq, fixed[i]):
                        return True
        return False
    # the two bands must not overlap each other: reject the layout if two fixed squares overlap
    for i, sq in enumerate(fixed):
        cx, cy = cell(sq)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for j in grid.get((cx + dx, cy + dy), []):
                    if j > i and overlap(sq, fixed[j]):
                        return -1, [], dict(t1=t1, t2=t2, fail='fixed overlap')
    # free band squares that hit fixed squares are dropped
    fx = [sq for sq in free if not hits_fixed(sq)]
    cands = [sq for sq in cands if not hits_fixed(sq)]
    var = fx + cands
    nf = len(fx)
    m = len(var)
    if m == 0:
        return len(fixed), fixed, dict(t1=t1, t2=t2)
    # conflicts: witness cliques for upright squares, pairwise constraints otherwise
    gridv = {}
    for i, sq in enumerate(var):
        gridv.setdefault(cell(sq), []).append(i)
    rows, cols = [], []
    ncons = 0
    pair_seen = set()
    for i in range(nf):
        a = var[i]
        cx, cy = cell(a)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for j in gridv.get((cx + dx, cy + dy), []):
                    if j <= i and j < nf:
                        continue
                    if overlap(a, var[j]):
                        rows += [ncons, ncons]; cols += [i, j]; ncons += 1
    # witness points for upright-upright cliques
    up_idx = {}
    for j in range(nf, m):
        up_idx.setdefault((var[j][0], var[j][1]), j)
    xs = sorted(set(v[0] for v in var[nf:]))
    seen_cl = set()
    for px in wx:
        xin = [x for x in xs if abs(x - px) < 0.5 - 1e-9]
        if not xin:
            continue
        for py in wx:
            members = tuple(sorted(up_idx[(x, y)] for x in xin for y in xs if abs(y - py) < 0.5 - 1e-9 and (x, y) in up_idx))
            if len(members) >= 2 and members not in seen_cl:
                seen_cl.add(members)
                for j in members:
                    rows.append(ncons); cols.append(j)
                ncons += 1
    if ncons == 0:
        A = None
    from scipy.sparse import csr_matrix
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(ncons, m))
    c = -np.ones(m)
    cons = LinearConstraint(A, -np.inf, 1.0) if ncons else []
    res = milp(c, constraints=cons, integrality=np.ones(m), bounds=Bounds(0, 1), options=dict(time_limit=time_limit, disp=False))
    if res.x is None:
        return len(fixed), fixed, dict(t1=t1, t2=t2, fail=True)
    chosen = [var[i] for i in range(m) if res.x[i] > 0.5]
    packing = fixed + chosen
    info = dict(t1=t1, t2=t2, nfixed=len(fixed), nfree=nf, ncand=len(cands), ncons=ncons, status=res.status,
                mip_gap=getattr(res, 'mip_gap', None))
    return len(packing), packing, info


def check(packing, L):
    # full pairwise check (slow, for validation); returns the number of overlapping pairs and out-of-box squares
    bad = 0
    out = 0
    for x, y, t in packing:
        h = half_extent(t, 0.0)
        if x - h < -1e-7 or x + h > L + 1e-7 or y - h < -1e-7 or y + h > L + 1e-7:
            out += 1
    P = sorted(packing)
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if P[j][0] - P[i][0] > 1.5:
                break
            if overlap(P[i], P[j], 1e-7):
                bad += 1
    return bad, out


def write_csv(packing, path, n_id=None):
    with open(path, 'w') as fh:
        fh.write('id,n,square_id,x,y,angle\n')
        nn = len(packing) if n_id is None else n_id
        for k, (x, y, t) in enumerate(packing):
            fh.write('%d,%d,%d,%.12f,%.12f,%.12f\n' % (k, nn, k, x, y, t))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = None
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
        args = [a for a in args if a != out]
    offsets = None
    if '--fine' in sys.argv:  # extra upright grids at half and quarter fractional offsets (slower MILP, for a final polish)
        fv = float(args[1]) - math.floor(float(args[1]) + 1e-9)
        offsets = [0.0, fv / 4, fv / 2, 3 * fv / 4, fv]
    ext = 1.5 if '--ext' in sys.argv else 0.0  # band lattice squares up to 1.5 beyond the band length as free candidates
    n, L = int(args[0]), float(args[1])
    N1, h1, q1, N2, h2, q2 = int(args[2]), float(args[3]), float(args[4]), int(args[5]), float(args[6]), float(args[7])
    extra = [float(a) for a in args[8:16]]
    f1p = extra[0] if len(extra) > 0 and extra[0] > 0 else None
    f2p = extra[1] if len(extra) > 1 and extra[1] > 0 else None
    x01 = extra[2] if len(extra) > 2 else 0.0
    x02 = extra[3] if len(extra) > 3 else 0.0
    m1 = int(extra[4]) if len(extra) > 4 else 0
    m2 = int(extra[5]) if len(extra) > 5 else 0
    ys1 = extra[6] if len(extra) > 6 and extra[6] > 0 else None
    ys2 = extra[7] if len(extra) > 7 and extra[7] > 0 else None
    t0 = time.time()
    cnt, P, info = build(n, L, N1, h1, q1, N2, h2, q2, f1p, f2p, x01, x02, m1, m2, offsets=offsets, ext=ext, ys1=ys1, ys2=ys2,
                         time_limit=120.0)
    bad, oob = check(P, L)
    print('count %d (need %d) t1 %.2f deg t2 %.2f deg overlaps %d outside %d time %.1fs info %s' %
          (cnt, n, math.degrees(info['t1']), math.degrees(info['t2']), bad, oob, time.time() - t0, info))
    if out:
        write_csv(P, out, n)
