# CSV <-> packer txt for Evan's explorer.
#   python3 ex_io.py start SUB.csv N OUT.txt          our packing of n (radians) -> packer txt (degrees), prints its side
#   python3 ex_io.py basins RUN_DIR N SIDE OUT_DIR    the 6 best basins b*.txt with side < SIDE -> OUT_DIR/x<n>_<j>.csv (radians)
import csv, glob, math, os, sys

sys.path.insert(0, os.environ.get('EVAND_PACKER', '/content/evand/search/packer'))
import gen  # noqa: E402


def side(sq):
    xs, ys = [], []
    for q in sq:
        for p in gen.corners(*q):
            xs.append(p[0]); ys.append(p[1])
    return max(max(xs) - min(xs), max(ys) - min(ys)), min(xs), min(ys)


def start(sub, n, out):
    sq = [(float(r['x']), float(r['y']), math.degrees(float(r['angle']))) for r in csv.DictReader(open(sub)) if int(r['n']) == n]
    s, mx, my = side(sq)
    with open(out, 'w') as f:
        f.write(f'{n} {s!r}\n' + ''.join(f'{x - mx!r} {y - my!r} {d!r}\n' for x, y, d in sq))
    print(f'{s:.10f}')


def basins(run, n, lim, out):
    found = []
    for p in glob.glob(os.path.join(run, 'b*.txt')):
        L = [l.split() for l in open(p) if l.strip()]
        sq = [(float(a), float(b), float(c)) for a, b, c, *_ in L[1:n + 1]]
        if len(sq) == n:
            s = side(sq)[0]
            if s < lim - 1e-9:
                found.append((s, sq))
    found.sort(key=lambda t: t[0])
    os.makedirs(out, exist_ok=True)
    for j, (s, sq) in enumerate(found[:6]):
        with open(os.path.join(out, f'x{n}_{j}.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
            for i, (x, y, d) in enumerate(sq):
                w.writerow([i, n, i, repr(x), repr(y), repr(math.radians(d))])
    print(f'n={n}: {len(found)} basins below {lim:.7f}' + (f', best {found[0][0]:.10f}' if found else ''))


if __name__ == '__main__':
    if sys.argv[1] == 'start':
        start(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        basins(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
