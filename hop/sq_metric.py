# Faithful re-implementation of the official Squares in Squares metric (metric notebook v20, Apache 2.0,
# Kaggle Competition Metrics): per-puzzle parse rules, float SAT with undecided band, interval-arithmetic escalation,
# L measured in mpmath from the decimal strings. evaluate(n, rows) -> (ok, L_float, reason).
import math, re
import mpmath

L_DPS = 60
_MAX_LITERAL_CHARS = 100
_MAX_COORD = 1e4
_MAX_ANGLE = 100.0
_AABB_PAD = 1e-7
_ESC = [50, 100, 200, 500]
_DEC = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


class _WD:
    def __init__(self, d): self.d = d
    def __enter__(self):
        self.o = (mpmath.iv.dps, mpmath.mp.dps); mpmath.iv.dps = self.d; mpmath.mp.dps = self.d
    def __exit__(self, *a):
        mpmath.iv.dps, mpmath.mp.dps = self.o


def _chk(s):
    s = s.strip()
    if not (len(s) <= _MAX_LITERAL_CHARS and _DEC.match(s)):
        raise ValueError('not a decimal number: %r' % s[:40])
    return s


def _pf(s):
    v = float(_chk(s))
    if not math.isfinite(v):
        raise ValueError('out of range')
    return v


def _pmp(s, dps=L_DPS):
    o = mpmath.mp.dps
    try:
        mpmath.mp.dps = dps; return mpmath.mpf(_chk(s))
    finally:
        mpmath.mp.dps = o


def _corners(cx, cy, a, cos=math.cos, sin=math.sin, h=0.5):
    c = cos(a); s = sin(a)
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    return [(cx + dx1, cy + dy1), (cx + dx2, cy + dy2), (cx - dx1, cy - dy1), (cx - dx2, cy - dy2)]


def _ivmin(v): return mpmath.iv.mpf([min(x.a for x in v), min(x.b for x in v)])
def _ivmax(v): return mpmath.iv.mpf([max(x.a for x in v), max(x.b for x in v)])


def _ivle(a, b):
    try:
        return a <= b
    except ValueError:
        return None


def _sat_float(c1, c2, a1, a2, tol=1e-8):
    axes = [(math.cos(a1), math.sin(a1)), (-math.sin(a1), math.cos(a1)), (math.cos(a2), math.sin(a2)), (-math.sin(a2), math.cos(a2))]
    mo = float('inf')
    for ax, ay in axes:
        p1 = [ax * x + ay * y for x, y in c1]; p2 = [ax * x + ay * y for x, y in c2]
        ov = min(max(p1), max(p2)) - max(min(p1), min(p2))
        if ov < -tol:
            return 'separated'
        mo = min(mo, ov)
    return 'overlapping' if mo > tol else 'undecided'


def _sat_iv(ci, cj, ai, aj):
    c1, s1 = mpmath.iv.cos(ai), mpmath.iv.sin(ai); c2, s2 = mpmath.iv.cos(aj), mpmath.iv.sin(aj)
    und = False
    for ax, ay in [(c1, s1), (-s1, c1), (c2, s2), (-s2, c2)]:
        p1 = [ax * x + ay * y for x, y in ci]; p2 = [ax * x + ay * y for x, y in cj]
        ov = _ivmin([_ivmax(p1), _ivmax(p2)]) - _ivmax([_ivmin(p1), _ivmin(p2)])
        r = _ivle(ov, mpmath.iv.mpf('0'))
        if r is True:
            return True
        if r is None:
            und = True
    return None if und else False


def evaluate(n, rows):
    """rows: list of dicts with string fields square_id, x, y, angle (as they will appear in the CSV)."""
    if len(rows) != n:
        return False, None, 'count'
    try:
        by = {int(r['square_id']): r for r in rows}
    except Exception:
        return False, None, 'square_id'
    if set(by) != set(range(n)):
        return False, None, 'ids'
    raw = []
    try:
        for sid in range(n):
            r = by[sid]; xs, ys, as_ = str(r['x']).strip(), str(r['y']).strip(), str(r['angle']).strip()
            xf, yf, af = _pf(xs), _pf(ys), _pf(as_)
            if abs(xf) > _MAX_COORD or abs(yf) > _MAX_COORD or abs(af) > _MAX_ANGLE:
                return False, None, 'bounds'
            raw.append((xs, ys, as_))
    except ValueError as e:
        return False, None, 'parse: %s' % e
    x0 = _pmp(raw[0][0]); y0 = _pmp(raw[0][1]); tp = 2 * mpmath.pi
    sq = []
    for xs, ys, as_ in raw:
        sq.append((float(_pmp(xs) - x0), float(_pmp(ys) - y0), float(_pmp(as_) % tp)))
    # L in mp
    o = mpmath.mp.dps
    try:
        mpmath.mp.dps = L_DPS
        X = []; Y = []
        for xs, ys, as_ in raw:
            for x, y in _corners(_pmp(xs) - x0, _pmp(ys) - y0, _pmp(as_), mpmath.cos, mpmath.sin, mpmath.mpf(1) / 2):
                X.append(x); Y.append(y)
        L = max(max(X) - min(X), max(Y) - min(Y))
    finally:
        mpmath.mp.dps = o
    # sweep
    cs = [_corners(x, y, a) for x, y, a in sq]
    bb = sorted(((min(p[0] for p in c) - _AABB_PAD, min(p[1] for p in c) - _AABB_PAD, max(p[0] for p in c) + _AABB_PAD,
                  max(p[1] for p in c) + _AABB_PAD), i) for i, c in enumerate(cs))
    und = []
    for k in range(len(bb)):
        a, i = bb[k]
        for m in range(k + 1, len(bb)):
            b, j = bb[m]
            if b[0] > a[2]:
                break
            if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                r = _sat_float(cs[i], cs[j], sq[i][2], sq[j][2])
                if r == 'overlapping':
                    return False, float(L), 'overlap float %d %d' % (i, j)
                if r == 'undecided':
                    und.append((i, j))
    if und:
        res = None
        for d in _ESC:
            with _WD(d):
                still = []; bad = False
                for i, j in und:
                    ivs = []
                    for t in (i, j):
                        xs, ys, as_ = raw[t]
                        ivs.append((mpmath.iv.mpf(_chk(xs)), mpmath.iv.mpf(_chk(ys)), mpmath.iv.mpf(_chk(as_))))
                    ci = _corners(ivs[0][0], ivs[0][1], ivs[0][2], mpmath.iv.cos, mpmath.iv.sin, mpmath.iv.mpf('0.5'))
                    cj = _corners(ivs[1][0], ivs[1][1], ivs[1][2], mpmath.iv.cos, mpmath.iv.sin, mpmath.iv.mpf('0.5'))
                    r = _sat_iv(ci, cj, ivs[0][2], ivs[1][2])
                    if r is False:
                        bad = True; break
                    if r is None:
                        still.append((i, j))
            if bad:
                return False, float(L), 'overlap certified'
            if not still:
                res = True; break
            und = still
        if res is None:
            return False, float(L), 'undecidable %d pairs' % len(und)
    if float(L) < math.sqrt(n) - 1e-7:
        return False, float(L), 'below bound'
    return True, float(L), 'ok'
