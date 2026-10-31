# Genome search over two-channel layouts (Evan Daniel's layout.py family) for one puzzle n.
# Each step: mutate the genome (rows of (w, theta_deg, setback) in the left and bottom channels), build it at the first
# side where the exact upright fill reaches n, squeeze with `packer relax --squeeze-pen` (layout.s_hyb), keep the best by
# simulated annealing on the squeezed side. Writes OUT.best.txt (packer format, degrees) and OUT.csv (competition
# format, radians) for the best layout found.
# Usage: python3 layout_run.py --n 299 --seed 1 --time 14000 --extra 2 --out out/299_1
# Needs EVAND_PACKER = <evand clone>/search/packer with target/release/packer built.
import argparse, csv, math, os, random, shutil, sys, tempfile, time

sys.path.insert(0, os.environ.get('EVAND_PACKER', '/content/evand/search/packer'))
import layout  # noqa: E402

# decoded board 176 (Fable 31 Oct, design2/chan.py G176); bottom angles converted to layout.py's transposed frame
G176_LEFT = [(3, 63.6, 0), (4, 64.3, 0), (5, 64.3, 0), (5, 64.3, 1), (4, 64.3, 2), (3, 64.3, 3), (2, 64.3, 4)]
G176_BOTTOM = [(4, 26.9, 0), (5, 26.9, 0), (6, 26.9, 0), (6, 26.9, 0), (5, 26.9, 1), (4, 26.9, 2), (3, 26.9, 3)]


def init_genome(extra, dw):
    """G176 grown for larger K: `extra` copies of the widest row inserted after it in each channel, widths + dw."""
    out = []
    for ch in (G176_LEFT, G176_BOTTOM):
        i = max(range(len(ch)), key=lambda j: ch[j][0])
        rows = ch[:i + 1] + [ch[i]] * extra + ch[i + 1:]
        out.append([(max(1, w + dw), t, k) for w, t, k in rows])
    return tuple(out)


def mutate(g, rng):
    """layout.mutate plus moves it lacks: widths above 6, a row inserted or removed anywhere."""
    r = rng.random()
    if r < 0.7:
        return layout.mutate(g, rng)
    left, bottom = [list(c) for c in g]
    ch = left if rng.random() < 0.5 else bottom
    if not ch:
        return (left, bottom)
    i = rng.randrange(len(ch))
    if r < 0.82:
        w, t, k = ch[i]
        ch[i] = (max(1, min(14, w + rng.choice((-1, 1)))), t, k)
    elif r < 0.92:
        ch.insert(i, ch[i])
    elif len(ch) > 1:
        ch.pop(i)
    return (left, bottom)


def to_csv(txt, path):
    L = [l.split() for l in open(txt) if l.strip()]
    n = int(L[0][0])
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
        for i, (x, y, d, *_) in enumerate(L[1:n + 1]):
            w.writerow([i, n, i, repr(float(x)), repr(float(y)), repr(math.radians(float(d)))])


def fmt(g):
    return '/'.join(' '.join(f'{w}@{t:.2f}+{k:g}' for w, t, k in ch) for ch in g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True); ap.add_argument('--k', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--time', type=float, default=600)
    ap.add_argument('--temp', type=float, default=0.005); ap.add_argument('--extra', type=int, default=0)
    ap.add_argument('--dw', type=int, default=0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    k = a.k or math.ceil(math.sqrt(a.n))          # a line of k upright squares would force side >= k
    rng = random.Random(a.seed)
    tmp = tempfile.mkdtemp()
    g = init_genome(a.extra, a.dw)
    # grow the genome until it reaches n somewhere below k + 0.8
    f, p = layout.s_hyb(g, a.n, k, tmp)
    tries = 0
    while f == float('inf') and tries < 200:
        tries += 1
        g = mutate(g, rng) if rng.random() < 0.5 else (g[0] + [g[0][-1]], g[1] + [g[1][-1]])
        f, p = layout.s_hyb(g, a.n, k, tmp)
    best = f
    if p:
        shutil.copy(p, f'{a.out}.best.txt')
    t0 = time.time(); it = 0
    print(f'n={a.n} k={k} init s={f:.6f} after {tries} grow steps  {fmt(g)}', flush=True)
    while time.time() - t0 < a.time:
        it += 1
        g2 = mutate(g, rng)
        f2, p2 = layout.s_hyb(g2, a.n, k, tmp)
        if f2 < best - 1e-7 and p2:
            best = f2
            shutil.copy(p2, f'{a.out}.best.txt')
            print(f'it={it} s={f2:.6f} t={time.time() - t0:.0f}s  {fmt(g2)}', flush=True)
        if f2 <= f or (f2 < float('inf') and rng.random() < math.exp(-(f2 - f) / a.temp)):
            g, f = g2, f2
        if it % 50 == 0:
            print(f'  it={it} cur={f:.5f} best={best:.6f} t={time.time() - t0:.0f}s', flush=True)
    print(f'done n={a.n} best s={best:.6f} its={it}', flush=True)
    if os.path.exists(f'{a.out}.best.txt'):
        to_csv(f'{a.out}.best.txt', f'{a.out}.csv')


if __name__ == '__main__':
    main()
