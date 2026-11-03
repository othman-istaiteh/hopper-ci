# SQ09 (own code): re-quench pass. For every puzzle, the best public / own candidates are re-converged with Evan Daniel's
# `fq` quench (MIT, github.com/evand/square-packing, search/packer/src/fq.rs: augmented-Lagrangian quench + LP polish,
# side free), at loosen 1.0 and 1.02 (his "screen1.0 / screen1.02" schedules). Original and quenched candidates are then
# merged per puzzle by sq08's rule: try candidates by side, keep the first one sq_metric certifies (contact failures
# repaired by a tiny uniform scaling). Finally the per-puzzle sides are printed in one compact line for gap analysis.
import os, sys, csv, math, json, glob, time, subprocess, tempfile
import numpy as np
from multiprocessing import Pool
from sq_metric import evaluate

GRID = lambda n: math.ceil(math.sqrt(n))
DELTAS = [1e-12, 1e-11, 1e-10, 1e-9]
FQ = os.environ.get('FQ_BIN', '/kaggle/working/evand/search/packer/target/release/fq')
WORK = os.environ.get('SQ_WORK', '/kaggle/working/rq')
SCHED = {'s100': ['--loosen', '1.0', '--pit', '8', '--flip-top', '0'],
         's102': ['--loosen', '1.02', '--pit', '8', '--flip-top', '0']}


def norm(rs):
    return [dict(square_id=str(int(r['square_id'])), x=repr(float(r['x'])), y=repr(float(r['y'])), angle=repr(float(r['angle']))) for r in rs]


def arr(rs):
    return np.array([[float(r['x']), float(r['y']), float(r['angle'])] for r in sorted(rs, key=lambda r: int(r['square_id']))])


def side_arr(a):
    h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    return max((a[:, 0] + h).max() - (a[:, 0] - h).min(), (a[:, 1] + h).max() - (a[:, 1] - h).min())


def side(rs):
    return side_arr(arr(rs))


def to_fq(a, path):
    """CSV rows (radians) -> fq text: 'n s' then 'x y deg' with the packing moved into [0, s]^2 (tiny slack)."""
    h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    x0 = (a[:, 0] - h).min(); y0 = (a[:, 1] - h).min()
    s = max((a[:, 0] + h).max() - x0, (a[:, 1] + h).max() - y0) * (1 + 1e-12)
    with open(path, 'w') as f:
        f.write('%d %.17g\n' % (len(a), s))
        for x, y, t in a:
            f.write('%.17g %.17g %.15g\n' % (x - x0, y - y0, math.degrees(t) % 90.0))
    return s


def from_fq(path):
    lines = [l for l in open(path).read().splitlines() if l.strip() and not l.startswith('#')]
    n, s = lines[0].split()[:2]; n = int(n)
    rows = []
    for i, l in enumerate(lines[1:1 + n]):
        x, y, d = l.split()[:3]
        rows.append(dict(square_id=str(i), x=repr(float(x)), y=repr(float(y)), angle=repr(float(math.radians(float(d))))))
    return float(s), rows


def quench_job(job):
    """job = (n, tag, rows) -> list of (approxL, tag', rows') for every schedule that produced a packing."""
    n, tag, rs = job
    out = []
    try:
        a = arr(rs)
        d = os.path.join(WORK, '%d_%s' % (n, tag.replace('/', '_')))
        os.makedirs(d, exist_ok=True)
        inp = os.path.join(d, 'in.txt'); to_fq(a, inp)
        for name, flags in SCHED.items():
            o = os.path.join(d, name + '.txt')
            try:
                r = subprocess.run([FQ, 'quench', '--in', inp, '--out', o, '--deadline', os.environ.get('FQ_DEADLINE', '90')] + flags,
                                   capture_output=True, text=True, timeout=240)
            except subprocess.TimeoutExpired:
                continue
            if os.path.exists(o):
                try:
                    s, rows = from_fq(o)
                    if len(rows) == n:
                        out.append((side(rows), tag + '+' + name, rows))
                except Exception:
                    pass
    except Exception as e:
        return n, [], repr(e)[:120]
    return n, out, ''


def certify(job):
    n, cands = job
    log = []
    for aL, tag, rs in sorted(cands, key=lambda c: c[0]):
        if aL >= GRID(n) - 1e-12:
            break
        rs = norm(rs)
        ok, L, why = evaluate(n, rs)
        if ok:
            return n, L, tag, rs, log
        log.append((tag, round(aL, 9), why))
        if why.startswith('undecidable') or why.startswith('overlap'):
            x = np.array([float(r['x']) for r in rs]); y = np.array([float(r['y']) for r in rs]); cx, cy = x.mean(), y.mean()
            for dd in DELTAS:
                r2 = [dict(square_id=r['square_id'], x=repr(float(cx + (xi - cx) * (1 + dd))), y=repr(float(cy + (yi - cy) * (1 + dd))), angle=r['angle'])
                      for r, xi, yi in zip(rs, x, y)]
                ok, L, why2 = evaluate(n, r2)
                if ok and L < GRID(n) - 1e-12:
                    log.append((tag, 'repaired d=%g' % dd))
                    return n, L, tag + '+rep', r2, log
    return n, None, 'grid', None, log


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.setdefault(int(r['n']), []).append(r)
            except Exception:
                pass
    return rows


def total(best):
    return sum((best[n][0] if n in best else GRID(n)) / math.sqrt(n) - 1 for n in range(1, 1001))


def main(out_path):
    t0 = time.time()
    files = sorted(set(glob.glob(os.environ.get('SQ_IN', '/kaggle/input') + '/**/submission.csv', recursive=True)))
    cands = {}
    for p in files:
        rows = load(p); tag = p.split('/')[-2]; k = 0
        for n, rs in rows.items():
            if 1 <= n <= 1000 and len(rs) == n:
                try:
                    cands.setdefault(n, []).append((side(rs), tag, rs)); k += 1
                except Exception:
                    pass
        print(f'{tag}: {k} puzzles', flush=True)
    maxc = int(os.environ.get('SQ_MAXC', '4'))
    for n in cands:
        seen = set(); keep = []
        for c in sorted(cands[n], key=lambda c: c[0]):
            key = round(c[0], 9)
            if key not in seen and c[0] < GRID(n) - 1e-9:
                seen.add(key); keep.append(c)
        cands[n] = keep[:maxc]
    cands = {n: v for n, v in cands.items() if v}
    nmin, nmax = int(os.environ.get('SQ_NMIN', '1')), int(os.environ.get('SQ_NMAX', '1000'))
    jobs = [(n, tag, rs) for n in sorted(cands, key=lambda n: -n) if nmin <= n <= nmax for (aL, tag, rs) in cands[n]]
    print(f'{len(jobs)} quench jobs over {len(cands)} puzzles; fq={FQ} exists={os.path.exists(FQ)}', flush=True)
    workers = int(os.environ.get('SQ_WORKERS', str(os.cpu_count() or 4)))
    nq = 0; errs = 0
    if os.path.exists(FQ) and os.environ.get('SQ_QUENCH', '1') == '1':
        with Pool(workers) as pool:
            for k, (n, out, err) in enumerate(pool.imap_unordered(quench_job, jobs, chunksize=1)):
                if err:
                    errs += 1
                    if errs <= 5: print('quench error', n, err, flush=True)
                for c in out:
                    cands[n].append(c); nq += 1
                if k % 200 == 0:
                    print(f'quenched {k}/{len(jobs)} jobs, {nq} outputs, {time.time() - t0:.0f}s', flush=True)
    print(f'quench stage done: {nq} outputs, {errs} errors, {time.time() - t0:.0f}s', flush=True)
    best = {}; fails = []
    with Pool(workers) as pool:
        cj = [(n, cands[n]) for n in sorted(cands, key=lambda n: -n)]
        for n, L, tag, rs, log in pool.imap_unordered(certify, cj, chunksize=1):
            if log: fails.append((n, log))
            if L is not None and L < best.get(n, (GRID(n),))[0] - 1e-13:
                best[n] = (L, tag, rs)
    print(f'certified {len(best)} puzzles, total {total(best):.10f}, {time.time() - t0:.0f}s', flush=True)
    # report: improvements of the quench over the best original candidate per puzzle
    gain = 0.0; imp = []
    for n in best:
        orig = min((c[0] for c in cands[n] if '+s1' not in c[1]), default=GRID(n))
        if best[n][0] < orig - 1e-9:
            imp.append((n, orig, best[n][0])); gain += (orig - best[n][0]) / math.sqrt(n)
    print(f'quench improved {len(imp)} puzzles, score gain vs best original candidates {gain:.6f}', flush=True)
    print('IMPROVED ' + ' '.join('%d:%.9f>%.9f' % t for t in sorted(imp)), flush=True)
    for n, log in sorted(fails)[:60]:
        print('rejected', n, log[:3], flush=True)
    src = {}
    for n in best:
        src[best[n][1].split('+')[0]] = src.get(best[n][1].split('+')[0], 0) + 1
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle']); rid = 0
        for n in range(1, 1001):
            if n in best:
                for r in sorted(best[n][2], key=lambda r: int(r['square_id'])):
                    w.writerow([rid, n, r['square_id'], r['x'], r['y'], r['angle']]); rid += 1
            else:
                k = GRID(n)
                for i in range(n):
                    w.writerow([rid, n, i, repr(0.5 + i % k), repr(0.5 + i // k), '0.0']); rid += 1
    print('SIDES ' + ' '.join('%d:%.10f' % (n, best[n][0]) for n in sorted(best)), flush=True)
    print(json.dumps(dict(certified_total=total(best), n_certified=len(best), sources=src, quenched=nq, sec=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
