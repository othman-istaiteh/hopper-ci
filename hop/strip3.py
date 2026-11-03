# Three-row strip seeds (H2 of design_plan_0794.md): our two-row diagonal strip packings get a third row, a copy of one
# zigzag row shifted by the row offset vector, on side A or B of the strip; block squares overlapping the new row are
# removed and the square count is brought back to n (extra block squares farthest from the strip dropped, missing
# ones inserted at the largest-clearance points). Overlapping seed, relaxed by relax.yml (fq quench / shrink).
# Usage: python3 strip3.py merged.csv OUT_DIR n1,n2,...
import csv, gzip, math, os, sys
import numpy as np
import sq09_requench as R
from sq10_hop import insert_squares


def strip_frame(a):
    th = np.mod(a[:, 2], math.pi / 2)
    tilted = np.abs(np.mod(th + math.pi / 4, math.pi / 2) - math.pi / 4) > 0.02
    phis = np.degrees(th[tilted]) - 45
    phi0 = float(np.median(phis))
    sel = tilted.copy(); sel[tilted] = np.abs(phis - phi0) < 5
    C = a[sel, :2]; cc = C - C.mean(0); w, v = np.linalg.eigh(cc.T @ cc); d = v[:, 1]; dp = np.array([-d[1], d[0]])
    return sel, C.mean(0), d, dp


def third_row(a, s, side):
    """side +1: new row beyond the max-u row; -1: beyond the min-u row. Returns the (n x 3) seed."""
    n = len(a)
    sel, c0, d, dp = strip_frame(a)
    S = a[sel]; t = (S[:, :2] - c0) @ d; u = (S[:, :2] - c0) @ dp
    # the two zigzag rows: split u at its median
    um = np.median(u); hi = u > um; lo = ~hi
    off = np.array([np.mean(t[hi]) - np.mean(t[lo]), np.mean(u[hi]) - np.mean(u[lo])])  # (dt, du) from the low row to the high row
    src = S[hi] if side > 0 else S[lo]
    shift = (off if side > 0 else -off)
    new = src.copy(); new[:, :2] += shift[0] * d + shift[1] * dp
    new = new[(new[:, 0] > 0.3) & (new[:, 1] > 0.3) & (new[:, 0] < s - 0.3) & (new[:, 1] < s - 0.3)]
    # drop block squares that overlap the new row (centre distance < 0.95 to a new square)
    blocks = a[~sel]
    dmin = np.min(np.hypot(blocks[:, None, 0] - new[None, :, 0], blocks[:, None, 1] - new[None, :, 1]), axis=1)
    keep = blocks[dmin >= 0.95]
    out = np.vstack([a[sel], new, keep])
    if len(out) > n:
        # too many: drop the block squares farthest from the strip axis (corners), keeping n
        ub = np.abs((keep[:, :2] - c0) @ dp)
        order = np.argsort(-ub)
        keep = keep[np.sort(order[len(out) - n:])]
        out = np.vstack([a[sel], new, keep])
    elif len(out) < n:
        out = insert_squares(out, s, n - len(out), 7)
    return out, int(sel.sum()), len(new), len(blocks) - len(keep)


def main():
    src, outdir, ns = sys.argv[1:4]
    rows = R.load(src)
    os.makedirs(outdir, exist_ok=True)
    for side, name in ((1, 'A'), (-1, 'B')):
        path = os.path.join(outdir, 'row3_%s.csv.gz' % name)
        with gzip.open(path, 'wt', newline='') as f:
            w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle']); rid = 0
            for n in [int(x) for x in ns.split(',')]:
                a = R.arr(rows[n]); s = R.side_arr(a)
                h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
                a = a - [(a[:, 0] - h).min(), (a[:, 1] - h).min(), 0.0]
                b, k, added, removed = third_row(a, s, side)
                assert len(b) == n, (n, len(b))
                for i, (x, y, t) in enumerate(b):
                    w.writerow([rid, n, i, repr(float(x)), repr(float(y)), repr(float(t))]); rid += 1
                print('side %s n %d: strip %d, third row %d squares, %d block squares removed, bbox %.4f' % (name, n, k, added, removed, R.side_arr(b)))


if __name__ == '__main__':
    main()
