# Convert exact rational certificates (format: "n side_fraction", then one line per square "x_frac y_frac t_frac"
# with centre (x, y) and t = tan(theta / 2)) from public GitHub repositories into competition rows, keep per n the
# best certificate that is below our current best side, certify it with the metric copy, and write a results CSV.
# Usage: python3 certs2csv.py OUR_BEST.csv OUT.csv DIR [DIR ...]   (DIR holds *.cert files, searched recursively)
import csv, glob, math, os, sys
from fractions import Fraction
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sq_metric import evaluate


def parse(path):
    lines = [l for l in open(path).read().splitlines() if l.strip() and not l.startswith('#')]
    n, side = lines[0].split()[:2]; n = int(n); side = float(Fraction(side))
    rows = []
    for l in lines[1:1 + n]:
        x, y, t = (Fraction(v) for v in l.split()[:3])
        th = 2 * math.atan(float(t))
        rows.append((float(x), float(y), th))
    if len(rows) != n:
        raise ValueError('row count')
    return n, side, rows


def bbox_side(rows):
    a = np.array(rows); h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    return max((a[:, 0] + h).max() - (a[:, 0] - h).min(), (a[:, 1] + h).max() - (a[:, 1] - h).min())


def our_sides(path):
    S = {}
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            S.setdefault(int(row[1]), []).append((float(row[3]), float(row[4]), float(row[5])))
    return {n: bbox_side(v) for n, v in S.items()}


def main():
    ours = our_sides(sys.argv[1]); out = sys.argv[2]
    cands = {}
    for d in sys.argv[3:]:
        for p in glob.glob(os.path.join(d, '**', '*.cert'), recursive=True):
            try:
                n, side, rows = parse(p)
            except Exception as e:
                continue
            if n < 1 or n > 1000: continue
            s = bbox_side(rows)
            if n in ours and s < ours[n] - 1e-9 and (n not in cands or s < cands[n][0]):
                cands[n] = (s, p, rows)
    print(len(cands), 'certificates below our side:', sorted((n, round(ours[n] - c[0], 7), c[1].split('/')[3]) for n, c in cands.items()))
    kept = []
    for n in sorted(cands):
        s, p, rows = cands[n]
        rs = [dict(square_id=str(i), x=repr(x), y=repr(y), angle=repr(a)) for i, (x, y, a) in enumerate(rows)]
        ok, L, why = evaluate(n, rs)
        if not ok:  # try a tiny uniform scaling about the centre (as the merge does for contact failures)
            cx = sum(r[0] for r in rows) / n; cy = sum(r[1] for r in rows) / n
            for dd in (1e-12, 1e-11, 1e-10, 1e-9):
                rs2 = [dict(square_id=str(i), x=repr(cx + (x - cx) * (1 + dd)), y=repr(cy + (y - cy) * (1 + dd)), angle=repr(a)) for i, (x, y, a) in enumerate(rows)]
                ok, L, why2 = evaluate(n, rs2)
                if ok and L < ours[n] - 1e-9:
                    rs = rs2; break
        print(n, 'ours %.9f cert %.9f certified %s %s %s' % (ours[n], s, ok, L, why if not ok else ''), flush=True)
        if ok and L < ours[n] - 1e-9:
            kept.append((n, L, rs, p))
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle']); k = 0
        for n, L, rs, p in kept:
            for r in rs:
                w.writerow([k, n, r['square_id'], r['x'], r['y'], r['angle']]); k += 1
    gain = sum((ours[n] - L) / math.sqrt(n) for n, L, rs, p in kept)
    print(f'kept {len(kept)} puzzles, score gain {gain:.6f}:', [(n, p.split('/')[3]) for n, L, rs, p in kept])


if __name__ == '__main__':
    main()
