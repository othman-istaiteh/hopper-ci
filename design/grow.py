# Growth evaluator: build puzzle n from our packing of n - k by inserting ONE square at a time, each followed by an fq
# quench (side free, fq soft-to-rigid rounding), keeping a beam of the best states (by side) at every step; the best final states get
# packer hop --temp 1.0. The leaders' sides rise smoothly with the count inside a band, as if their packings were grown
# square by square; our earlier "insert" engine added all k squares at once and quenched once (1 hit in 114).
# Usage: python3 -u grow.py BEST.csv N OUT_DIR [--ks 1,2,3] [--beam 3] [--cands 8] [--qsec 60] [--hopmin 12] [--procs 4]
# Needs FQ_BIN and PACKER_BIN. Writes OUT_DIR/g<N>_<j>.csv (competition format, radians) for the final states.
import argparse, csv, math, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hop'))
import sq09_requench as R  # noqa: E402  (arr, to_fq, from_fq, side_arr)

FQ = os.environ.get('FQ_BIN', 'fq'); PACKER = os.environ.get('PACKER_BIN', 'packer')
# a plain quench (loosen 1.0 to 1.03) of an inserted state falls to the grid side at n ~ 300 (tested 3 Nov); fq's
# soft-to-rigid rounding (Evan's best relax for constructions) keeps it near the donor's basin
FQ_OPTS = os.environ.get('GROW_FQ_OPTS', '--round-r0 0.15 --round-compress --round-press --round-steps 5 --loosen 1.0').split()


def load(path):
    P = {}
    for r in csv.DictReader(open(path)):
        P.setdefault(int(r['n']), []).append(r)
    return P


def clearance_points(a, s, kmax=8, step=0.1):
    g = np.arange(0.5, s - 0.5 + 1e-9, step)
    X, Y = np.meshgrid(g, g); X = X.ravel(); Y = Y.ravel()
    c = np.minimum(np.minimum(X, Y), np.minimum(s - X, s - Y)) - 0.5
    for x, y, t in a:
        dx, dy = X - x, Y - y; ct, st = math.cos(t), math.sin(t)
        c = np.minimum(c, np.maximum(np.abs(dx * ct + dy * st), np.abs(-dx * st + dy * ct)) - 1.0)
    pts = []
    for i in np.argsort(-c)[:5000]:
        if all((X[i] - px) ** 2 + (Y[i] - py) ** 2 >= 1.0 for px, py, _ in pts):
            pts.append((float(X[i]), float(Y[i]), float(c[i])))
            if len(pts) >= kmax:
                break
    return pts


def quench(job):
    """(a, tag, seed, qsec) -> (side, a') or None"""
    a, tag, seed, qsec = job
    d = tempfile.mkdtemp(prefix='g_')
    inp, out = os.path.join(d, 'in.txt'), os.path.join(d, 'out.txt')
    R.to_fq(a, inp)
    try:
        subprocess.run([FQ, 'quench', '--in', inp, '--out', out, '--seed', str(seed), '--deadline', str(int(qsec))] + FQ_OPTS,
                       capture_output=True, text=True, timeout=qsec + 300)
        s, rows = R.from_fq(out)
        return s, R.arr(rows), tag
    except Exception:
        return None


def hop(job):
    a, seed, minutes = job
    d = tempfile.mkdtemp(prefix='h_')
    inp, out = os.path.join(d, 'in.txt'), os.path.join(d, 'out.txt')
    s0 = R.to_fq(a, inp)
    try:
        subprocess.run([PACKER, 'hop', '--in', inp, '--s', '%.17g' % (s0 - 1e-4), '--seed', str(seed), '--iters', '100000000',
                        '--out', out, '--quiet', '--temp', '1.0'], capture_output=True, text=True, timeout=minutes * 60)
    except subprocess.TimeoutExpired:
        pass
    if os.path.exists(out):
        s, rows = R.from_fq(out)
        return s, R.arr(rows)
    return R.side_arr(a), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('best'); ap.add_argument('n', type=int); ap.add_argument('out')
    ap.add_argument('--ks', default='1,2,3'); ap.add_argument('--beam', type=int, default=3)
    ap.add_argument('--cands', type=int, default=8); ap.add_argument('--qsec', type=float, default=60)
    ap.add_argument('--hopmin', type=float, default=12); ap.add_argument('--procs', type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    P = load(a.best); n = a.n
    ours = R.side_arr(R.arr(P[n]))
    print(f'n={n} ours {ours:.10f}', flush=True)
    finals = []
    with ProcessPoolExecutor(a.procs) as ex:
        for k in [int(x) for x in a.ks.split(',')]:
            if n - k not in P:
                continue
            don = R.arr(P[n - k]); beam = [(R.side_arr(don), don, f'd{n - k}')]
            print(f'  k={k}: donor {n - k} side {beam[0][0]:.6f}', flush=True)
            for step in range(k):
                jobs = []
                for s, st, tag in beam:
                    h = 0.5 * (np.abs(np.cos(st[:, 2])) + np.abs(np.sin(st[:, 2])))
                    st0 = st.copy(); st0[:, 0] -= (st[:, 0] - h).min(); st0[:, 1] -= (st[:, 1] - h).min()
                    pts = clearance_points(st0, s, kmax=max(1, a.cands // 2))
                    for c in range(a.cands):
                        x, y, _ = pts[c % len(pts)]
                        near = int(np.argmin(np.hypot(st0[:, 0] - x, st0[:, 1] - y)))
                        t = st0[near, 2] + (0 if (c // len(pts)) % 2 == 0 else math.pi / 4)
                        jobs.append((np.vstack([st0, [x, y, t]]), f'{tag}+{c}', 1000 * step + c, a.qsec))
                res = [r for r in ex.map(quench, jobs) if r and len(r[1]) == n - k + step + 1]
                res.sort(key=lambda r: r[0])
                nb, seen = [], []
                for s, st, tag in res:
                    if all(abs(s - q) > 1e-7 for q in seen):
                        nb.append((s, st, tag)); seen.append(s)
                    if len(nb) >= a.beam:
                        break
                if not nb:
                    break
                beam = nb
                print(f'    step {step + 1}: best {beam[0][0]:.6f} ({beam[0][2]}), beam {[round(b[0], 5) for b in beam]}', flush=True)
            if beam and len(beam[0][1]) == n:
                finals += beam
        finals.sort(key=lambda r: r[0])
        finals = finals[:a.procs]
        if not finals:
            print(f'done n={n}: no grown state', flush=True); return
        hopped = list(ex.map(hop, [(st, 7 + j, a.hopmin) for j, (s, st, tag) in enumerate(finals)]))
    os.makedirs(a.out, exist_ok=True)
    allc = sorted([(s, st) for s, st, _ in finals] + hopped, key=lambda r: r[0])
    for j, (s, st) in enumerate(allc[:6]):
        with open(os.path.join(a.out, f'g{n}_{j}.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
            for i, (x, y, t) in enumerate(st):
                w.writerow([i, n, i, repr(float(x)), repr(float(y)), repr(float(t))])
    best = allc[0][0]
    print(f'done n={n}: grown best {best:.10f} vs ours {ours:.10f} ({best - ours:+.6f}); after hop {min(h[0] for h in hopped):.10f}', flush=True)


if __name__ == '__main__':
    main()
