# Re-tilt seeds for the diagonal tilted-strip family (30 Oct geometer test 2): take our packing of n, rotate every strip
# square (tilted by about phi0 from the diagonal) to a new tilt phi, optionally re-space the strip rows along the diagonal
# by cos(phi0) / cos(phi), and write one slice per (variant, phi) under OUT_DIR as <variant>_<phi>.csv.gz. The seeds
# overlap; fq quench --loosen (engine quench) or fq --shrink relaxes them on GitHub (relax.yml).
# Usage: python3 retilt.py merged.csv OUT_DIR n1,n2,... phi1,phi2,...
import csv, gzip, math, os, sys
import numpy as np
import sq09_requench as R


def retilt(a, phi_new, respace):
    th = np.mod(a[:, 2], math.pi / 2)
    tilted = np.abs(np.mod(th + math.pi / 4, math.pi / 2) - math.pi / 4) > 0.02
    phis = np.degrees(th[tilted]) - 45
    phi0 = float(np.median(phis))
    sel = tilted.copy(); sel[tilted] = np.abs(phis - phi0) < 5  # the main strip only (747 holds a few other tilts)
    b = a.copy()
    b[sel, 2] = a[sel, 2] + math.radians(math.copysign(phi_new, phi0) - phi0)
    if respace:
        C = a[sel, :2]; cc = C - C.mean(0); w, v = np.linalg.eigh(cc.T @ cc); d = v[:, 1]
        t = cc @ d
        f = math.cos(math.radians(abs(phi0))) / math.cos(math.radians(phi_new))
        b[sel, :2] = C.mean(0) + np.outer(t * f, d) + (cc - np.outer(t, d))
    return b, phi0, int(sel.sum())


def main():
    src, out, ns, phis = sys.argv[1:5]
    rows = R.load(src)
    os.makedirs(out, exist_ok=True)
    for variant in ('rot', 'resp'):
        for phi in [float(x) for x in phis.split(',')]:
            path = os.path.join(out, '%s_%g.csv.gz' % (variant, phi))
            with gzip.open(path, 'wt', newline='') as f:
                w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle']); rid = 0
                for n in [int(x) for x in ns.split(',')]:
                    a = R.arr(rows[n])
                    b, phi0, k = retilt(a, phi, variant == 'resp')
                    for i, (x, y, t) in enumerate(b):
                        w.writerow([rid, n, i, repr(float(x)), repr(float(y)), repr(float(t))]); rid += 1
                    print('%s phi %g: n %d strip %d squares phi0 %.2f side %.4f -> bbox %.4f' % (variant, phi, n, k, phi0, R.side_arr(a), R.side_arr(b)))


if __name__ == '__main__':
    main()
