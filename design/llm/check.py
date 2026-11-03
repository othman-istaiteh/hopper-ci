# Local checker for LLM generator programs (pure numpy, no fq): python3 check.py prog.py [n ...]
# Prints, per n: count ok, bounding side, number of overlapping pairs, sum of SAT penetration depths, vs ours and board.
# Loads our packings from base.csv (competition format) and the board from ../../hop/board_29oct.txt.
import sys, os, csv, importlib.util, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

def load_base(path=os.path.join(HERE, 'base.csv')):
    B = {}
    for r in csv.DictReader(open(path)):
        B.setdefault(int(r['n']), []).append((int(r['square_id']), float(r['x']), float(r['y']), float(r['angle'])))
    return {n: np.array([v[1:] for v in sorted(rs)]) for n, rs in B.items()}

def board():
    B = {}
    for l in open(os.path.join(HERE, '..', '..', 'hop', 'board_29oct.txt')):
        if l.strip() and not l.startswith('#'):
            n, L = l.split()[:2]; B[int(n)] = float(L)
    return B

def side(a):
    h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    return max((a[:, 0] + h).max() - (a[:, 0] - h).min(), (a[:, 1] + h).max() - (a[:, 1] - h).min())

def corners(a):
    c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
    off = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    X = a[:, None, 0] + off[None, :, 0] * c[:, None] - off[None, :, 1] * s[:, None]
    Y = a[:, None, 1] + off[None, :, 0] * s[:, None] + off[None, :, 1] * c[:, None]
    return np.stack([X, Y], -1)

def overlaps(a):
    """pairs with positive SAT penetration depth and the sum of depths (0 = feasible)."""
    P = corners(a); n = len(a); cnt = 0; tot = 0.0
    d = np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1])
    I, J = np.where(np.triu(d < 1.4143, 1))
    for i, j in zip(I, J):
        depth = 9.0
        for t in (a[i, 2], a[j, 2]):
            for ax in ((np.cos(t), np.sin(t)), (-np.sin(t), np.cos(t))):
                pi = P[i] @ ax; pj = P[j] @ ax
                depth = min(depth, min(pi.max(), pj.max()) - max(pi.min(), pj.min()))
        if depth > 1e-9:
            cnt += 1; tot += depth
    return cnt, tot

if __name__ == '__main__':
    prog = sys.argv[1]; ns = [int(x) for x in sys.argv[2:]] or [299, 412, 498, 544]
    spec = importlib.util.spec_from_file_location('prog', prog); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    base = load_base(); B = board()
    for n in ns:
        a = np.asarray(m.build(n, base[n].copy(), B[n], np.random.default_rng(n)), float)
        ok = a.shape == (n, 3)
        c, t = overlaps(a) if ok else (-1, -1)
        print('n=%d shape_ok=%s side=%.5f ours=%.5f board=%.5f overlap_pairs=%d depth_sum=%.4f' % (n, ok, side(a), side(base[n]), B[n], c, t))
