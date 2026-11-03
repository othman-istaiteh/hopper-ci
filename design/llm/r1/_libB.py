# ---- shared helpers (direction B): geometry, SAT, slab fill, strip rows, deficit insertion ----
import numpy as np, math, time

def sq_corners(x, y, t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[x + dx * c - dy * s, y + dx * s + dy * c] for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))])

def sat_depth(P, ti, Q, tj):
    d = 9.0
    for t in (ti, tj):
        for ax in ((math.cos(t), math.sin(t)), (-math.sin(t), math.cos(t))):
            p = P @ ax; q = Q @ ax
            d = min(d, min(p.max(), q.max()) - max(p.min(), q.min()))
            if d <= 0: return d
    return d

def inside_box(P, S, tol=1e-9):
    return P.min() >= -tol and P.max() <= S + tol

def clip_yrange(P, x0, x1):
    """y-range of convex polygon P intersected with slab x0<=x<=x1 (None if empty)."""
    if P[:, 0].max() <= x0 + 1e-9 or P[:, 0].min() >= x1 - 1e-9: return None
    pts = list(map(tuple, P))
    for sgn, xc in ((1, x0), (-1, x1)):
        out = []; m = len(pts)
        for k in range(m):
            a = pts[k]; b = pts[(k + 1) % m]
            ia = sgn * (a[0] - xc) >= 0; ib = sgn * (b[0] - xc) >= 0
            if ia: out.append(a)
            if ia != ib:
                u = (xc - a[0]) / (b[0] - a[0]); out.append((xc, a[1] + u * (b[1] - a[1])))
        pts = out
        if not pts: return None
    ys = [p[1] for p in pts]
    return min(ys), max(ys)

class Packing:
    def __init__(self, S):
        self.S = S; self.sq = []; self.P = []; self.bx = []; self.up = []
    def add(self, x, y, t):
        self.sq.append((x, y, t)); P = sq_corners(x, y, t); self.P.append(P)
        self.bx.append((P[:, 0].min(), P[:, 0].max()))
    def arr(self):
        return np.array(self.sq, float).reshape(-1, 3)
    def blocked(self, x0, x1):
        r = []
        for (a, b), P in zip(self.bx, self.P):
            if b <= x0 + 1e-9 or a >= x1 - 1e-9: continue
            if b - a < 1 + 1e-9 and abs(P[0, 1] - P[1, 1]) < 1e-12:
                r.append((P[:, 1].min(), P[:, 1].max())); continue
            yr = clip_yrange(P, x0, x1)
            if yr: r.append(yr)
        return sorted(r)

def free_intervals(blk, S):
    out = []; cur = 0.0
    for a, b in blk:
        if a > cur + 1e-9: out.append((cur, a))
        cur = max(cur, b)
    if S > cur + 1e-9: out.append((cur, S))
    return out

def slab_fill(pk, xs, keep, from_top):
    """For each slab [x0,x0+1] in xs, stack uprights in free y-intervals accepted by keep(xc, lo, hi)."""
    S = pk.S
    for x0 in xs:
        if x0 < -1e-9 or x0 + 1 > S + 1e-9: continue
        for lo, hi in free_intervals(pk.blocked(x0, x0 + 1), S):
            if not keep(x0 + .5, lo, hi): continue
            m = int(math.floor(hi - lo + 1e-9))
            for k in range(m):
                y = hi - .5 - k if from_top else lo + .5 + k
                pk.add(x0 + .5, y, 0.0)

def transpose(a):
    b = a.copy(); b[:, 0], b[:, 1] = a[:, 1].copy(), a[:, 0].copy(); b[:, 2] = np.pi / 2 - a[:, 2]; return b

def row_spacing(beta):
    return 1.0 / max(abs(math.cos(beta)), abs(math.sin(beta)))

def make_row(c, phi, off, alpha, phase, S, kmax=80):
    """squares with orientation alpha centred on the line c + off*nrm + (phase + k*s)*d, kept fully inside the box."""
    d = np.array([math.cos(phi), math.sin(phi)]); nr = np.array([-d[1], d[0]])
    s = row_spacing(alpha - phi); out = []
    for k in range(-kmax, kmax + 1):
        p = c + off * nr + (phase + k * s) * d
        P = sq_corners(p[0], p[1], alpha)
        if inside_box(P, S): out.append((p[0], p[1], alpha))
    return out

def rows_conflict(A, B, tol=1e-9):
    PA = [sq_corners(*q) for q in A]; PB = [sq_corners(*q) for q in B]
    ca = np.array([q[:2] for q in A]); cb = np.array([q[:2] for q in B])
    if len(A) == 0 or len(B) == 0: return False
    D = np.hypot(ca[:, None, 0] - cb[None, :, 0], ca[:, None, 1] - cb[None, :, 1])
    for i, j in zip(*np.where(D < 1.4143)):
        if sat_depth(PA[i], A[i][2], PB[j], B[j][2]) > tol: return True
    return False

def stack_offset(c, phi, off1, row1, alpha2, phase2, S, gap=0.0, sign=1):
    """smallest |offset| (beyond off1 on side sign) for a row of alpha2 not overlapping row1, plus gap."""
    lo, hi = 0.0, 1.6
    for _ in range(22):
        m = .5 * (lo + hi)
        r2 = make_row(c, phi, off1 + sign * m, alpha2, phase2, S)
        if rows_conflict(row1, r2): lo = m
        else: hi = m
    return off1 + sign * (hi + gap)

def depth_total(a):
    P = [sq_corners(*q) for q in a]; tot = 0.0
    D = np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1])
    for i, j in zip(*np.where(np.triu(D < 1.4143, 1))):
        d = sat_depth(P[i], a[i, 2], P[j], a[j, 2])
        if d > 1e-9: tot += d
    return tot

def clearance_map(a, S, h=0.05):
    g = np.arange(0.5, S - 0.5 + 1e-9, h)
    X, Y = np.meshgrid(g, g); X = X.ravel(); Y = Y.ravel(); best = np.full(X.size, 9.0)
    for x, y, t in a:
        c, s = math.cos(t), math.sin(t)
        dx = X - x; dy = Y - y
        m = (np.abs(dx) < 3) & (np.abs(dy) < 3)
        u = np.abs(dx[m] * c + dy[m] * s); v = np.abs(-dx[m] * s + dy[m] * c)
        best[m] = np.minimum(best[m], np.maximum(u, v) - .5)
    return X, Y, best

def insert_deficit(a, n, S, angles=(0.0,)):
    """add squares at largest-clearance points until len == n, choosing orientation with least SAT depth."""
    a = [tuple(q) for q in a]
    while len(a) < n:
        A = np.array(a); X, Y, cl = clearance_map(A, S)
        idx = np.argsort(-cl)[:40]; best = None
        for i in idx:
            for t in angles:
                P = sq_corners(X[i], Y[i], t)
                if not inside_box(P, S, 1e-6):
                    continue
                dsum = 0.0
                near = np.where(np.hypot(A[:, 0] - X[i], A[:, 1] - Y[i]) < 1.4143)[0]
                for j in near:
                    d = sat_depth(P, t, sq_corners(*A[j]), A[j, 2])
                    if d > 0: dsum += d
                if best is None or dsum < best[0]: best = (dsum, X[i], Y[i], t)
        if best is None: best = (0, X[idx[0]], Y[idx[0]], 0.0)
        a.append((best[1], best[2], best[3]))
    return np.array(a[:n])

def trim_excess(a, n, S):
    """drop squares (prefer uprights touching most neighbours? keep simple: those furthest in a corner) until n."""
    if len(a) <= n: return a
    # remove squares whose removal is cheapest: uprights at the top-right corner region first
    c = S / 2; key = -np.hypot(a[:, 0] - c, a[:, 1] - c)
    keep = np.sort(np.argsort(key)[:n])
    return a[keep]

def to_centred(a, S):
    b = a.copy(); b[:, 0] -= S / 2; b[:, 1] -= S / 2; return b
# ---- end helpers ----
# ---- vectorized SAT by centres/angles and a greedy channel filler ----
def hext(dt):
    return .5 * (np.abs(np.cos(dt)) + np.abs(np.sin(dt)))

def depth_vs(cx, cy, t, O):
    """max SAT depth of candidate squares (cx,cy arrays, angle t) against squares O (k,3). <=0 means free."""
    if len(O) == 0: return np.full(cx.shape, -1.0)
    dx = cx[:, None] - O[None, :, 0]; dy = cy[:, None] - O[None, :, 1]; to = O[None, :, 2]
    dep = np.full(dx.shape, 9.0)
    for th in (np.full(to.shape, t), np.full(to.shape, t + np.pi / 2), to, to + np.pi / 2):
        ov = hext(t - th) + hext(to - th) - np.abs(dx * np.cos(th) + dy * np.sin(th))
        dep = np.minimum(dep, ov)
    return dep.max(1)

def greedy_fill(S, placed, cand_xy, key, angles, tol=1e-7, maxadd=10000):
    """repeatedly add the feasible candidate (point, angle) with smallest key; placed: list of (x,y,t)."""
    pl = [tuple(q) for q in placed]
    o = np.argsort(key); X = cand_xy[o, 0]; Y = cand_xy[o, 1]
    alive = {t: np.ones(len(X), bool) for t in angles}
    for t in angles:
        h = hext(np.array(t)); alive[t] &= (X - h >= -1e-9) & (X + h <= S + 1e-9) & (Y - h >= -1e-9) & (Y + h <= S + 1e-9)
    added = 0; B = 3000
    while added < maxadd:
        A = np.array(pl).reshape(-1, 3); best = None
        for t in angles:
            idx = np.where(alive[t])[0]
            for s in range(0, len(idx), B):
                ii = idx[s:s + B]
                if best is not None and ii[0] >= best[0]: break
                near = A[(A[:, 0] > X[ii].min() - 1.5) & (A[:, 0] < X[ii].max() + 1.5) & (A[:, 1] > Y[ii].min() - 1.5) & (A[:, 1] < Y[ii].max() + 1.5)]
                ok = depth_vs(X[ii], Y[ii], t, near) <= tol
                alive[t][ii[~ok]] = False
                if ok.any():
                    j = ii[np.argmax(ok)]
                    if best is None or j < best[0]: best = (j, t)
                    break
        if best is None: break
        j, t = best; pl.append((X[j], Y[j], t)); added += 1
        for tt in angles:  # kill candidates now overlapping the new square
            near = np.where(alive[tt] & (np.abs(X - X[j]) < 1.5) & (np.abs(Y - Y[j]) < 1.5))[0]
            if len(near):
                ok = depth_vs(X[near], Y[near], tt, np.array([[X[j], Y[j], t]])) <= tol
                alive[tt][near[~ok]] = False
    return pl

def lattice_UL(S, K1, h):
    """top-left anchored lattice cells with centre above line y - x = h"""
    out = []
    for i in range(K1):
        for j in range(K1):
            x, y = i + .5, S - .5 - j
            if y - x > h: out.append((x, y, 0.0))
    return out

def lattice_LR(S, K1, h):
    out = []
    for i in range(K1):
        for j in range(K1):
            x, y = S - .5 - i, .5 + j
            if y - x < h: out.append((x, y, 0.0))
    return out

def depth_total2(a):
    tot = 0.0
    for i in range(len(a) - 1):
        O = a[i + 1:]; m = (np.abs(O[:, 0] - a[i, 0]) < 1.5) & (np.abs(O[:, 1] - a[i, 1]) < 1.5)
        if m.any():
            dx = a[i, 0] - O[m, 0]; dy = a[i, 1] - O[m, 1]; to = O[m, 2]; t = a[i, 2]; dep = np.full(dx.shape, 9.)
            for th in (np.full(to.shape, t), np.full(to.shape, t + np.pi / 2), to, to + np.pi / 2):
                dep = np.minimum(dep, hext(t - th) + hext(to - th) - np.abs(dx * np.cos(th) + dy * np.sin(th)))
            tot += dep[dep > 1e-9].sum()
    return tot
