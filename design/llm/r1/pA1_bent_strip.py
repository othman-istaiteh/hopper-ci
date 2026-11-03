# pA1 bent strip: each tilted band/strip (cluster of squares with theta mod 90 away from 0) gets a smooth
# angle profile dtheta(s) = A sin(pi k s + phi) plus a matching small perpendicular bow, so different rows of the
# staircase meet different notch phases; uprights near the interface and the strip are then pushed apart locally.
# ---- shared helpers (inlined) ----
import numpy as np
from scipy.spatial import cKDTree
HP = np.pi / 2

def tilt(a):
    t = np.mod(a[:, 2], HP); return np.minimum(t, HP - t)

def halfext(a):
    return 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))

def box(a):
    h = halfext(a)
    return (a[:, 0] - h).min(), (a[:, 0] + h).max(), (a[:, 1] - h).min(), (a[:, 1] + h).max()

def corners(a):
    c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
    off = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    X = a[:, None, 0] + off[None, :, 0] * c[:, None] - off[None, :, 1] * s[:, None]
    Y = a[:, None, 1] + off[None, :, 0] * s[:, None] + off[None, :, 1] * c[:, None]
    return np.stack([X, Y], -1)

def mtv(a, I, J):
    """SAT min penetration depth and unit push direction (for j away from i) per pair."""
    P = corners(a); best = np.full(len(I), 9.0); bx = np.zeros((len(I), 2))
    for src in (I, J):
        t = a[src, 2]
        for ax in (np.stack([np.cos(t), np.sin(t)], 1), np.stack([-np.sin(t), np.cos(t)], 1)):
            pi = np.einsum('kcd,kd->kc', P[I], ax); pj = np.einsum('kcd,kd->kc', P[J], ax)
            d = np.minimum(pi.max(1), pj.max(1)) - np.maximum(pi.min(1), pj.min(1))
            m = d < best; best[m] = d[m]; bx[m] = ax[m]
    dc = a[J, :2] - a[I, :2]
    sg = np.sign(np.einsum('kd,kd->k', dc, bx)); sg[sg == 0] = 1
    return best, bx * sg[:, None]

def push(a, movable, c, L, iters=300, step=0.55):
    """Iterative pairwise push-apart of overlaps; only `movable` squares move; clamp into [lo,hi]^2 box."""
    a = a.copy(); mv = np.zeros(len(a), bool); mv[movable] = True
    for it in range(iters):
        pairs = cKDTree(a[:, :2]).query_pairs(1.4143, output_type='ndarray')
        if len(pairs):
            k = mv[pairs[:, 0]] | mv[pairs[:, 1]]; pairs = pairs[k]
        disp = np.zeros((len(a), 2)); worst = 0.0
        if len(pairs):
            I, J = pairs[:, 0], pairs[:, 1]
            d, u = mtv(a, I, J); m = d > 1e-7; I, J, d, u = I[m], J[m], d[m], u[m]
            if len(d):
                worst = d.max()
                wi = mv[I].astype(float); wj = mv[J].astype(float); tot = wi + wj
                fi = wi / tot; fj = wj / tot
                np.add.at(disp, I, -(step * d * fi)[:, None] * u)
                np.add.at(disp, J, (step * d * fj)[:, None] * u)
        disp[~mv] = 0; a[:, :2] += disp
        h = halfext(a)
        for k in (0, 1):
            a[mv, k] = np.clip(a[mv, k], c[k] - L / 2 + h[mv], c[k] + L / 2 - h[mv])
        if worst < 1e-6 and it > 5: break
    return a

def scale_about(a, s):
    x0, x1, y0, y1 = box(a); c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    b = a.copy(); b[:, :2] = c + (a[:, :2] - c) * s
    L = max(x1 - x0, y1 - y0) * s
    return b, c, L
# ---- end helpers ----

def clusters(a, T, r=1.25):
    """connected components of tilted squares T (index array) -> list of index arrays."""
    if len(T) == 0: return []
    pr = cKDTree(a[T, :2]).query_pairs(r, output_type='ndarray')
    par = list(range(len(T)))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in pr: par[f(i)] = f(j)
    g = {}
    for i in range(len(T)): g.setdefault(f(i), []).append(T[i])
    return [np.array(v) for v in g.values() if len(v) >= 4]

def axis(a, C):
    p = a[C, :2]; m = p.mean(0); w, v = np.linalg.eigh(np.cov((p - m).T)); u = v[:, 1]
    return m, u, np.array([-u[1], u[0]])

def interface(a, T, r=1.6):
    U = np.setdiff1d(np.arange(len(a)), T)
    d, _ = cKDTree(a[T, :2]).query(a[U, :2]); return U[d < r]

def finish(a, n, c, L):
    a = np.asarray(a, float)
    assert a.shape == (n, 3)
    return a

def build(n, base, board, rng):
    a, c, L = scale_about(np.asarray(base, float), 1.0 + rng.uniform(0.0015, 0.003))
    T = np.where(tilt(a) > 0.01)[0]
    if len(T) < 4: return a
    for C in clusters(a, T):
        m, u, v = axis(a, C)
        s = (a[C, :2] - m) @ u; s = (s - s.min()) / max(np.ptp(s), 1e-9)
        k = rng.choice([1, 2, 3]); phi = rng.uniform(0, 2 * np.pi)
        A = np.radians(rng.uniform(1.0, 3.5)) * rng.choice([-1, 1])
        B = rng.uniform(0.03, 0.12) * rng.choice([-1, 1])
        prof = np.sin(np.pi * k * s + phi)
        a[C, 2] += A * prof
        a[C, :2] += (B * np.sin(np.pi * s))[:, None] * v[None, :]
    M = np.union1d(T, interface(a, T, 1.6))
    a = push(a, M, c, L, iters=400)
    return finish(a, n, c, L)
