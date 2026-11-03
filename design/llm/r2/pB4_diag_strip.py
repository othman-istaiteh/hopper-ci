# pB4 diag_strip: straight width-2 diagonal channel between wall-flush upright staircases (top-left / bottom-right
# lattices cut at y - x = h1 / h2), filled bottom-left-first with tilted squares (exact SAT, slide-to-contact),
# leftover pockets filled at any angle; bisection on the side S for the smallest S holding n (all feasible, gap 1e-9).
# ---- shared helpers (round 2, feasible constructions): exact SAT, wall-flush slab fill, raster pocket fill ----
import numpy as np, math, time
from scipy.ndimage import distance_transform_edt

GAP = 2e-9

def hext(dt):
    return .5 * (np.abs(np.cos(dt)) + np.abs(np.sin(dt)))

def depth_vs(cx, cy, t, O):
    """max SAT depth of candidates (arrays cx, cy, scalar angle t) against squares O (k,3). <0 = separated."""
    cx = np.atleast_1d(cx); cy = np.atleast_1d(cy)
    if len(O) == 0: return np.full(cx.shape, -1.0)
    dx = cx[:, None] - O[None, :, 0]; dy = cy[:, None] - O[None, :, 1]; to = O[None, :, 2]
    dep = np.full(dx.shape, 9.0)
    for th in (np.full(to.shape, t), np.full(to.shape, t + np.pi / 2), to, to + np.pi / 2):
        dep = np.minimum(dep, hext(t - th) + hext(to - th) - np.abs(dx * np.cos(th) + dy * np.sin(th)))
    return dep.max(1)

class Pack:
    """growing list of squares with a coarse spatial hash for neighbour queries"""
    def __init__(self, S):
        self.S = S; self.A = np.zeros((0, 3)); self.cell = {}
    def near(self, x, y, r=1.5):
        ids = []
        for i in range(int(math.floor(x - r)), int(math.floor(x + r)) + 1):
            for j in range(int(math.floor(y - r)), int(math.floor(y + r)) + 1):
                ids += self.cell.get((i, j), [])
        return self.A[ids] if ids else np.zeros((0, 3))
    def fits(self, x, y, t, gap=GAP):
        h = float(hext(t)); S = self.S
        if x - h < gap or x + h > S - gap or y - h < gap or y + h > S - gap: return False
        return depth_vs(x, y, t, self.near(x, y))[0] <= -gap
    def add(self, x, y, t):
        k = len(self.A); self.A = np.vstack([self.A, [x, y, t]])
        self.cell.setdefault((int(math.floor(x)), int(math.floor(y))), []).append(k)
    def try_add(self, x, y, t):
        if self.fits(x, y, t): self.add(x, y, t); return True
        return False

def ylimits_in_slab(pk, x0, x1):
    """blocked y-intervals of slab [x0,x1] by existing squares (exact for convex squares)."""
    out = []
    xm = .5 * (x0 + x1)
    for x, y, t in pk.near(xm, pk.S / 2, r=pk.S):
        h = float(hext(t))
        if x + h <= x0 or x - h >= x1: continue
        c, s = math.cos(t), math.sin(t)
        P = np.array([[x + dx * c - dy * s, y + dx * s + dy * c] for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))])
        pts = list(map(tuple, P))
        for sgn, xc in ((1, x0), (-1, x1)):
            o = []; m = len(pts)
            for k in range(m):
                a = pts[k]; b = pts[(k + 1) % m]
                ia = sgn * (a[0] - xc) >= 0; ib = sgn * (b[0] - xc) >= 0
                if ia: o.append(a)
                if ia != ib:
                    u = (xc - a[0]) / (b[0] - a[0]); o.append((xc, a[1] + u * (b[1] - a[1])))
            pts = o
            if not pts: break
        if pts:
            ys = [p[1] for p in pts]; out.append((min(ys), max(ys)))
    return sorted(out)

def column_fill(pk, x0, from_top, stop=None):
    """stack uprights in slab [x0,x0+1] from the top (or bottom) wall down to the first obstacle."""
    S = pk.S
    if x0 < GAP or x0 + 1 > S - GAP: return 0
    blk = ylimits_in_slab(pk, x0 + GAP / 2, x0 + 1 - GAP / 2)
    if from_top:
        lo = 0.0
        for a, b in blk:
            lo = max(lo, b)
        hi = S
    else:
        hi = S
        for a, b in blk:
            hi = min(hi, a)
        lo = 0.0
    if stop is not None:
        if from_top: lo = max(lo, stop)
        else: hi = min(hi, stop)
    m = int(math.floor((hi - lo - 2 * GAP) / (1 + GAP)))
    for k in range(max(m, 0)):
        y = (hi - GAP - .5 - k * (1 + GAP)) if from_top else (lo + GAP + .5 + k * (1 + GAP))
        pk.add(x0 + .5, y, 0.0)
    return max(m, 0)

def colx(S, i, left):
    return i * (1 + 2 * GAP) + GAP if left else S - GAP - (i + 1) * (1 + 2 * GAP)

def row_fill(pk, y0, from_right, stop=None):
    """transpose of column_fill: horizontal slab [y0,y0+1], fill from the right (or left) wall."""
    S = pk.S
    T = Pack(S)
    for x, y, t in pk.A: T.add(y, x, np.pi / 2 - t)
    before = len(T.A)
    m = column_fill(T, y0, from_right, stop)
    for x, y, t in T.A[before:]: pk.add(y, x, 0.0)
    return m

def raster_free(pk, h):
    S = pk.S; N = int(math.ceil(S / h)); g = (np.arange(N) + .5) * h
    occ = np.zeros((N, N), bool)
    for x, y, t in pk.A:
        r = float(hext(t)); i0 = max(int((x - r) / h) - 1, 0); i1 = min(int((x + r) / h) + 2, N)
        j0 = max(int((y - r) / h) - 1, 0); j1 = min(int((y + r) / h) + 2, N)
        X, Y = np.meshgrid(g[i0:i1], g[j0:j1], indexing='ij')
        dx = X - x; dy = Y - y; c, s = math.cos(t), math.sin(t)
        occ[i0:i1, j0:j1] |= (np.abs(dx * c + dy * s) <= .5 + h) & (np.abs(-dx * s + dy * c) <= .5 + h)
    pad = np.ones((N + 2, N + 2), bool); pad[1:-1, 1:-1] = ~occ
    pad[0, :] = pad[-1, :] = pad[:, 0] = pad[:, -1] = False
    D = distance_transform_edt(pad)[1:-1, 1:-1] * h
    return g, D

def pocket_fill(pk, angles, h=0.03, rmin=0.47, key=None, maxadd=200, tlim=None, jit=(0.0,)):
    """greedy: add squares (any angle in `angles`) in the free pockets until none fits."""
    t0 = time.time(); added = 0
    while added < maxadd:
        if tlim and time.time() - t0 > tlim: break
        g, D = raster_free(pk, h)
        I, J = np.where(D >= rmin)
        if len(I) == 0: break
        xs = g[I]; ys = g[J]
        o = np.argsort(D[I, J]) if key is None else np.argsort(key(xs, ys))
        done = False
        for k in o[:4000]:
            for t in angles:
                for dx in jit:
                    for dy in jit:
                        if pk.try_add(xs[k] + dx, ys[k] + dy, t):
                            done = True; break
                    if done: break
                if done: break
            if done: break
        if not done: break
        added += 1
    return added

def verify(A, S, gap=0.0):
    """indices of squares that overlap (SAT depth > -gap... >1e-12) or leave the box; greedy removal set."""
    bad = set()
    h = hext(A[:, 2])
    out = (A[:, 0] - h < -1e-12) | (A[:, 0] + h > S + 1e-12) | (A[:, 1] - h < -1e-12) | (A[:, 1] + h > S + 1e-12)
    bad |= set(np.where(out)[0].tolist())
    for i in range(len(A)):
        if i in bad: continue
        O = A[i + 1:]; m = (np.abs(O[:, 0] - A[i, 0]) < 1.5) & (np.abs(O[:, 1] - A[i, 1]) < 1.5)
        idx = np.where(m)[0] + i + 1
        idx = [j for j in idx if j not in bad]
        if idx:
            d = depth_vs(A[i, 0], A[i, 1], A[i, 2], A[idx])
            if d[0] > 1e-12: bad.add(i)
    return bad

def finalize(A, n, S):
    A = np.asarray(A, float)
    bad = verify(A, S)
    if bad: A = A[[i for i in range(len(A)) if i not in bad]]
    if len(A) > n:
        # drop uprights touching the top-right corner first (cheap), keep tilted structure
        key = (A[:, 0] + A[:, 1]) + 100 * (np.abs(np.sin(2 * A[:, 2])) > 1e-6)
        A = A[np.sort(np.argsort(key)[:n])]
    return A

def centred(A, S):
    B = A.copy(); B[:, 0] -= S / 2; B[:, 1] -= S / 2; return B

def row_line(pk, c, d, alpha, phase, off, pitch=None, s_range=None):
    """place squares of angle alpha at c + off*nrm + (phase + k*pitch)*d wherever they fit (exact)."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d); nr = np.array([-d[1], d[0]])
    if pitch is None:
        b = alpha - math.atan2(d[1], d[0]); pitch = 1.0 / max(abs(math.cos(b)), abs(math.sin(b))) + 4 * GAP
    L = 2 * pk.S; cnt = 0
    for k in range(int(-L / pitch) - 1, int(L / pitch) + 2):
        s = phase + k * pitch
        if s_range and not (s_range[0] <= s <= s_range[1]): continue
        p = np.asarray(c) + off * nr + s * d
        cnt += pk.try_add(p[0], p[1], alpha)
    return cnt

def min_offset(pk, c, d, alpha, phase, off0, sign, pitch=None, s_range=None, hi=1.8):
    """smallest offset beyond off0 (direction sign) at which a full row would not hit existing squares
    in its central part; returns offset (bisection on conflict count of the row's interior)."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d); nr = np.array([-d[1], d[0]])
    if pitch is None:
        b = alpha - math.atan2(d[1], d[0]); pitch = 1.0 / max(abs(math.cos(b)), abs(math.sin(b))) + 4 * GAP
    def conflicts(off):
        bad = 0; tot = 0
        for k in range(int(-2 * pk.S / pitch) - 1, int(2 * pk.S / pitch) + 2):
            s = phase + k * pitch
            if s_range and not (s_range[0] <= s <= s_range[1]): continue
            p = np.asarray(c) + off * nr + s * d
            hh = float(hext(alpha))
            if p[0] - hh < 0 or p[0] + hh > pk.S or p[1] - hh < 0 or p[1] + hh > pk.S: continue
            tot += 1
            O = pk.near(p[0], p[1])
            if len(O) and depth_vs(p[0], p[1], alpha, O)[0] > -GAP: bad += 1
        return bad
    lo, h2 = 0.0, hi
    for _ in range(25):
        m = .5 * (lo + h2)
        if conflicts(off0 + sign * m) > 0: lo = m
        else: h2 = m
    return off0 + sign * h2
# ---- end helpers ----

def slide(pk, x, y, t, dirs, step=0.04):
    """push a feasible square along each unit direction in dirs as far as it stays feasible (bisection)."""
    for dx, dy in dirs:
        lo, hi = 0.0, step
        if pk.fits(x + hi * dx, y + hi * dy, t):
            lo = hi; hi = 2 * step
            while hi < 0.6 and pk.fits(x + hi * dx, y + hi * dy, t): lo = hi; hi *= 2
        for _ in range(30):
            m = .5 * (lo + hi)
            if pk.fits(x + m * dx, y + m * dy, t): lo = m
            else: hi = m
        x += lo * dx; y += lo * dy
    return x, y

def greedy_first_fit(pk, X, Y, key, angles, dirs=None, B=1500):
    """single pass over candidates in key order (equivalent to repeated smallest-key greedy); exact SAT with GAP."""
    o = np.argsort(key, kind='stable'); X = X[o]; Y = Y[o]; S = pk.S; added = 0
    for t in angles if False else [None]:
        pass
    N = len(X); p = 0
    hs = {t: float(hext(t)) for t in angles}
    while p < N:
        ii = np.arange(p, min(p + B, N)); xs = X[ii]; ys = Y[ii]
        A = pk.A
        msk = (A[:, 0] > xs.min() - 1.5) & (A[:, 0] < xs.max() + 1.5) & (A[:, 1] > ys.min() - 1.5) & (A[:, 1] < ys.max() + 1.5)
        O = A[msk]; first = None
        for t in angles:
            h = hs[t]
            inb = (xs - h >= GAP) & (xs + h <= S - GAP) & (ys - h >= GAP) & (ys + h <= S - GAP)
            ok = inb & (depth_vs(xs, ys, t, O) <= -GAP)
            if ok.any():
                j = int(np.argmax(ok))
                if first is None or j < first[0]: first = (j, t)
        if first is None: p += B; continue
        j, t = first; x, y = xs[j], ys[j]
        if dirs: x, y = slide(pk, x, y, t, dirs)
        pk.add(x, y, t); added += 1; p = p + j + 1
    return added

def lattice_blocks(pk, h1, h2, fUL=None, fLR=None):
    """wall-flush upright blocks: top-left anchored cells with y - F(x) > h1, bottom-right with y - F(x) < h2."""
    S = pk.S; K1 = int(math.floor(S)); F = (lambda x: x) if fUL is None else fUL; G = F if fLR is None else fLR
    for i in range(K1):
        for j in range(K1):
            x = colx(S, i, True) + .5; y = S - GAP - .5 - j * (1 + 2 * GAP)
            if y - F(x) > h1: pk.add(x, y, 0.0)
    for i in range(K1):
        for j in range(K1):
            x = colx(S, i, False) + .5; y = GAP + .5 + j * (1 + 2 * GAP)
            if y - G(x) < h2 and pk.fits(x, y, 0.0, gap=0.0): pk.add(x, y, 0.0)

def base_side(a):
    h = hext(a[:, 2])
    return max((a[:, 0] + h).max() - (a[:, 0] - h).min(), (a[:, 1] + h).max() - (a[:, 1] - h).min())

def search_side(n, base, rng, ev, cfgs, tbudget=34.0, steps=9):
    """ev(S, cfg, n) -> Pack (or None). Bisection on S in [ours, K-0.025]; at each S try configs
    (last winners first, then rng-shuffled others) until one holds n squares feasibly."""
    t0 = time.time(); K = math.ceil(math.sqrt(n) - 1e-12)
    lo = base_side(base) - 0.002; hi = K - 0.021
    order = [cfgs[i] for i in rng.permutation(len(cfgs))]
    win = []; best = None
    def attempt(S, budget):
        tried = 0
        for c in win[::-1] + [c for c in order if c not in win]:
            if time.time() - t0 > budget: break
            pk = ev(S, c, n); tried += 1
            if pk is not None and len(pk.A) >= n:
                if c in win: win.remove(c)
                win.append(c); return pk
        return None
    pk = attempt(hi, tbudget)
    if pk is None:
        return None
    best = (hi, pk)
    for _ in range(steps):
        if time.time() - t0 > tbudget: break
        m = .5 * (lo + hi)
        pk = attempt(m, min(tbudget, time.time() - t0 + tbudget / steps))
        if pk is not None: hi = m; best = (m, pk)
        else: lo = m
    return best

def result(n, base, best, fallback):
    if best is None: return fallback
    S, pk = best
    A = finalize(pk.A, n, S)
    if len(A) < n: return fallback
    return centred(A, S)
def ev_strip(S, c, n):
    h1, h2, al, lam = c
    pk = Pack(S); lattice_blocks(pk, h1, h2)
    g = np.arange(0, S, 0.02); X, Y = np.meshgrid(g, g); X = X.ravel(); Y = Y.ravel()
    m = (Y - X < h1 + 1.0) & (Y - X > h2 - 1.0); X = X[m]; Y = Y[m]
    u = (X + Y) / 2 ** .5; v = (Y - X) / 2 ** .5
    a = math.radians(al); r2 = 2 ** -.5
    greedy_first_fit(pk, X, Y, u + lam * v, [a], dirs=[(-r2, -r2), (r2, -r2)])
    if n - 10 <= len(pk.A) < n:
        pocket_fill(pk, [0.0, a] + [math.radians(x) for x in range(5, 90, 10)], maxadd=n - len(pk.A))
    return pk

def build(n, base, board, rng):
    cfgs = [(h1, h2, al, lam) for h1 in (1.5, 1.75, 2.0, 2.25, 2.5, 2.75) for h2 in (-0.75, -1.25)
            for al in (22.0, 26.6, 30.0, 33.0) for lam in (0.15, 0.4)]
    return result(n, base, search_side(n, base, rng, ev_strip, cfgs), base)
