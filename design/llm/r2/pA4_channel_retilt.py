# pA4 region rebuild: delete a window of the tilted channel plus the upright notch cells beside it, refill it
# greedily (tilted squares at a NEW tilt = old +-1..3 deg, then uprights), opening the box (eps<=0.004, then a cut
# of d) only as far as needed for exactly n squares; falls back to base if no feasible rebuild is found.
# ---------------- region-rebuild toolkit (self-contained, numpy/scipy) ----------------
import math, time
import numpy as np
from scipy.spatial import cKDTree

TOL = -2e-7  # a candidate is free if its max SAT depth vs every neighbour is <= TOL (tiny positive gap)


def hext(t):
    return 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))


def side(a):
    h = hext(a[:, 2])
    return max((a[:, 0] + h).max() - (a[:, 0] - h).min(), (a[:, 1] + h).max() - (a[:, 1] - h).min())


def normalize(a):
    a = a.copy(); a[:, 2] = np.mod(a[:, 2], np.pi / 2); h = hext(a[:, 2])
    a[:, 0] -= (a[:, 0] - h).min(); a[:, 1] -= (a[:, 1] - h).min()
    return a


def is_tilt(a, tol=0.02):
    return np.abs(np.sin(2 * a[:, 2])) > tol


def pdepth(ax, ay, at, bx, by, bt):
    """elementwise SAT penetration depth of square pairs (positive = overlap)."""
    dx = ax - bx; dy = ay - by; dep = np.full(np.shape(dx), 9.0)
    for th in (at, at + np.pi / 2, bt, bt + np.pi / 2):
        ov = hext(at - th) + hext(bt - th) - np.abs(dx * np.cos(th) + dy * np.sin(th))
        dep = np.minimum(dep, ov)
    return dep


def depth_sum(a):
    pr = cKDTree(a[:, :2]).query_pairs(1.4143, output_type='ndarray')
    if len(pr) == 0: return 0.0
    d = pdepth(a[pr[:, 0], 0], a[pr[:, 0], 1], a[pr[:, 0], 2], a[pr[:, 1], 0], a[pr[:, 1], 1], a[pr[:, 1], 2])
    return float(d[d > 1e-9].sum())


def in_box(C, S):
    h = hext(C[:, 2])
    return (C[:, 0] - h >= 1e-9) & (C[:, 0] + h <= S - 1e-9) & (C[:, 1] - h >= 1e-9) & (C[:, 1] + h <= S - 1e-9)


def free_vs(C, O):
    """bool per candidate row of C (x,y,t): no overlap with any square in O."""
    ok = np.ones(len(C), bool)
    if len(O) == 0 or len(C) == 0: return ok
    nb = cKDTree(O[:, :2]).query_ball_point(C[:, :2], 1.4143)
    L = np.array([len(x) for x in nb]); I = np.repeat(np.arange(len(C)), L)
    if len(I) == 0: return ok
    J = np.concatenate([np.asarray(x, int) for x in nb if len(x)])
    d = pdepth(C[I, 0], C[I, 1], C[I, 2], O[J, 0], O[J, 1], O[J, 2])
    bad = np.zeros(len(C), bool); np.logical_or.at(bad, I, d > TOL)
    return ~bad


def greedy(S, kept, C, key, need, alive=None):
    """bottom-left style greedy: repeatedly accept the free candidate (row of C) with the smallest key.
    returns the accepted squares (k,3), k <= need."""
    o = np.argsort(key, kind='stable'); C = C[o]
    ok = in_box(C, S) if alive is None else alive[o] & in_box(C, S)
    ok &= free_vs(C, kept)
    tr = cKDTree(C[:, :2]); out = []; p = 0
    while len(out) < need:
        while p < len(C) and not ok[p]: p += 1
        if p >= len(C): break
        q = C[p]; out.append(q.copy()); ok[p] = False
        nb = np.asarray(tr.query_ball_point(q[:2], 1.4143), int)
        nb = nb[ok[nb]]
        if len(nb):
            d = pdepth(C[nb, 0], C[nb, 1], C[nb, 2], q[0], q[1], q[2])
            ok[nb[d > TOL]] = False
    return np.array(out).reshape(-1, 3)


def scale_about(a, f):
    a = a.copy(); a[:, :2] *= f
    return a


def grid_pts(R, rad, h, rot=0.0, anchor=(0.0, 0.0)):
    """grid points of step h in a frame rotated by rot and anchored at `anchor`, kept within rad of a centre of R."""
    cs, sn = math.cos(rot), math.sin(rot); ax_, ay_ = anchor
    u = (R[:, 0] - ax_) * cs + (R[:, 1] - ay_) * sn; v = -(R[:, 0] - ax_) * sn + (R[:, 1] - ay_) * cs
    gu = np.arange(math.floor((u.min() - rad) / h), math.ceil((u.max() + rad) / h) + 1) * h
    gv = np.arange(math.floor((v.min() - rad) / h), math.ceil((v.max() + rad) / h) + 1) * h
    U, V = np.meshgrid(gu, gv); U = U.ravel(); V = V.ravel()
    P = np.column_stack([ax_ + U * cs - V * sn, ay_ + U * sn + V * cs])
    return P[near_mask(P, R, rad)]


def near_mask(P, R, rad):
    """points of P within rad of any centre in R."""
    if len(R) == 0: return np.zeros(len(P), bool)
    d, _ = cKDTree(R[:, :2]).query(P, distance_upper_bound=rad)
    return np.isfinite(d)


def components(R, r=1.6):
    """connected components (index lists) of centres R under distance r."""
    n = len(R); lab = -np.ones(n, int); k = 0
    if n == 0: return []
    tr = cKDTree(R[:, :2])
    for s in range(n):
        if lab[s] >= 0: continue
        st = [s]; lab[s] = k
        while st:
            i = st.pop()
            for j in tr.query_ball_point(R[i, :2], r):
                if lab[j] < 0: lab[j] = k; st.append(j)
        k += 1
    return [np.where(lab == i)[0] for i in range(k)]


def pca_dir(P):
    P = P - P.mean(0)
    w, v = np.linalg.eigh(P.T @ P)
    d = v[:, -1]
    return math.atan2(d[1], d[0])


def verify(a, n, K):
    return a.shape == (n, 3) and depth_sum(a) == 0.0 and side(a) < K - 0.02
# ---------------- end toolkit ----------------


def refill(kept, R, S, need, rng, angles, h, keyang, wv, rad=0.95, anchors=None):
    """greedy re-fill of the freed region (around removed centres R), one phase per angle in `angles`
    (tilted first, uprights last). key = gravity along direction keyang (weight wv) + transverse coordinate."""
    if anchors is None: anchors = [R[0, :2] + rng.uniform(0, h, 2)]
    cs, sn = math.cos(keyang), math.sin(keyang); out = np.zeros((0, 3)); placed = kept
    for t in angles:
        if len(out) >= need: break
        C = np.vstack([np.column_stack([P, np.full(len(P), t)]) for P in (grid_pts(R, rad, h, rot=t, anchor=an) for an in anchors)])
        key = (C[:, 0] * cs + C[:, 1] * sn) * wv + (-C[:, 0] * sn + C[:, 1] * cs)
        q = greedy(S, placed, C, key, need - len(out))
        out = np.vstack([out, q]); placed = np.vstack([placed, q])
    return out


def try_rebuild(a, rem, n, rng, eps, angles, h, keyang, wv, rad=0.95, d=(0.0, 0.0), cut=None):
    """remove rem; scale kept by 1+eps; open the box by d[0] (x) / d[1] (y) along cut lines through `cut`
    (squares beyond the line are translated); greedy refill. returns (packing or None, shortfall)."""
    S = side(a) * (1 + eps)
    kept = scale_about(a[~rem], 1 + eps); R = scale_about(a[rem], 1 + eps)
    if cut is None: cut = R[:, :2].mean(0)
    cut = np.asarray(cut) * (1 + eps)
    for k in (0, 1):
        if d[k] > 0:
            kept[kept[:, k] > cut[k], k] += d[k]; R[R[:, k] > cut[k], k] += d[k] / 2
    if d[0] > 0 or d[1] > 0:
        # squares that collide after opening join the rebuild region
        for _ in range(5):
            pr = cKDTree(kept[:, :2]).query_pairs(1.4143, output_type='ndarray')
            if len(pr) == 0: break
            dd = pdepth(kept[pr[:, 0], 0], kept[pr[:, 0], 1], kept[pr[:, 0], 2], kept[pr[:, 1], 0], kept[pr[:, 1], 1], kept[pr[:, 1], 2])
            bad = np.unique(pr[dd > TOL].ravel())
            if len(bad) == 0: break
            R = np.vstack([R, kept[bad]]); kept = np.delete(kept, bad, 0)
        # extra freed points along the opened cut slabs inside the region
        S = S + max(d)
    need = n - len(kept)
    if need <= 0: return None, 99
    anchors = [R[0, :2] + rng.uniform(0, h, 2), (0.5 + 1e-6, 0.5 + 1e-6) if rng.random() < 0.5 else (S - 0.5 - 1e-6, S - 0.5 - 1e-6)]
    q = refill(kept, R, S, need, rng, angles, h, keyang, wv, rad, anchors)
    if len(q) < need: return None, need - len(q)
    return normalize(np.vstack([kept, q])), 0


def ladder(a, rem, n, rng, K, angles, cut=None, budget=50.0, per=8, t_start=None, keyang=None):
    """try eps = 0..0.004 (uniform scale), then open the box by d at the cut, until a refill reaches n.
    returns the smallest-side feasible result found at the first successful level, or None."""
    t0 = t_start or time.time(); S0 = side(a)
    levels = [(e, 0.0) for e in (0.0, 0.004)]
    dmax = (K - 0.025) - S0 * 1.004
    levels += [(0.004, d) for d in np.arange(0.04, dmax, 0.04)]
    for eps, d in levels:
        best = None
        for _ in range(per):
            if time.time() - t0 > budget: return best
            h = (1 + 3e-6) / int(rng.integers(32, 48))
            res, sh = try_rebuild(a, rem, n, rng, eps, angles, h, (rng.uniform(0, 2 * math.pi) if keyang is None else keyang + rng.choice([0, math.pi])), rng.uniform(2, 10),
                                  d=(d, d) if d > 0 else (0.0, 0.0), cut=cut)
            if res is not None and verify(res, n, K) and (best is None or side(res) < side(best)): best = res
        if best is not None: return best
    return None


# ---- pA4: channel window retilt ----
def build(n, base, board, rng):
    t0 = time.time(); K = math.ceil(math.sqrt(n)); a = normalize(base); T = is_tilt(a)
    Tc = a[T]; comps = components(Tc, 1.6)
    comp = comps[int(rng.integers(len(comps)))]; C = Tc[comp]
    phi = pca_dir(C); cs, sn = math.cos(phi), math.sin(phi)
    u = C[:, 0] * cs + C[:, 1] * sn
    L = rng.uniform(2.5, 4.5); uc = rng.uniform(u.min() + L / 2, max(u.min() + L / 2, u.max() - L / 2) + 1e-9)
    # window along the channel: tilted squares + upright notch cells next to them
    ua = a[:, 0] * cs + a[:, 1] * sn
    dT, _ = cKDTree(Tc[:, :2]).query(a[:, :2])
    nr = rng.uniform(1.05, 1.4)
    rem = (np.abs(ua - uc) < L / 2) & (T | (dT < nr))
    if not (rem & T).any(): rem = T & (np.abs(ua - uc) < L)
    tt = float(np.median(np.mod(a[rem & T, 2], np.pi / 2)))
    best = None
    for dt in rng.permutation([-2.5, -1.2, 1.2, 2.5])[:1]:  # small tilt scan
        tn = tt + math.radians(dt + rng.uniform(-0.4, 0.4))
        r = ladder(a, rem, n, rng, K, [tn, 0.0], cut=a[rem, :2].mean(0), keyang=phi + math.pi / 2, budget=45.0, t_start=time.time())
        if r is not None and (best is None or side(r) < side(best)): best = r
        if time.time() - t0 > 40: break
    return best if best is not None else a
