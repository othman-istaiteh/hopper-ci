# LLM program search, evaluator (own code). Each program design/llm/<round>/<name>.py defines
#     build(n, base, board, rng) -> np.ndarray (n, 3) of x, y, theta (radians), centres of unit squares (overlaps allowed)
# where base = our certified best packing for n (same format) and board = the per-puzzle board side. For every program,
# panel puzzle and seed, the candidate is relaxed with fq quench + packer hop --temp 1.0 (sq10_hop engine quenchhop) and
# re-quenched; the best relaxed side is the fitness. Candidates below our side are certified with the metric copy.
# Output: results.json (per program, per n: seed side, relaxed side, ours, board), submission.csv (certified wins only),
# eval.log.  Usage: python3 llm_eval.py <programs_dir> <base_submission.csv> <out_dir>
import os, sys, json, glob, time, math, csv, importlib.util, traceback, subprocess
os.environ.setdefault('HOP_ENGINE', 'quenchhop')
import numpy as np
import sq09_requench as R
import sq10_hop as H
from multiprocessing import Pool

PANEL = [int(x) for x in os.environ.get('LLM_PANEL', '299,412,498,544').split(',')]
MIN = float(os.environ.get('LLM_MIN', '6'))
SEEDS = int(os.environ.get('LLM_SEEDS', '2'))
SHARD, NSHARD = int(os.environ.get('LLM_SHARD', '0')), int(os.environ.get('LLM_NSHARD', '1'))


def board():
    B = {}
    for line in open(os.environ.get('SQ_BOARD', 'board_29oct.txt')):
        if line.startswith('#') or not line.strip():
            continue
        n, L = line.split()[:2]; B[int(n)] = float(L)
    return B


def make_candidate(job):
    path, n, base, bd, seed = job
    try:
        spec = importlib.util.spec_from_file_location('prog', path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        a = np.asarray(m.build(n, base.copy(), bd, np.random.default_rng(seed)), dtype=float)
        if a.shape != (n, 3) or not np.isfinite(a).all():
            return None, 'bad shape %s' % (a.shape,)
        return a, str(getattr(m, 'LAST_PARAMS', ''))[:300]
    except Exception:
        return None, traceback.format_exc()[-300:]


def rows_of(a):
    return [dict(square_id=str(i), x=repr(float(x)), y=repr(float(y)), angle=repr(float(t))) for i, (x, y, t) in enumerate(a)]


def run(cmd, timeout):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or '')[-300:] + (r.stderr or '')[-200:]
    except subprocess.TimeoutExpired:
        return 'timeout'


def relax(job):
    """fq quench (loosen 1.03, 3 min) -> packer hop --temp 1.0 for the rest (its output exists only if it improved,
    and a hop timeout is normal) -> re-quench of the best state with loosen 1.0 and 1.02. Returns all candidates."""
    name, n, a, seed = job
    t = time.time(); out = []; err = ''
    d = os.path.join(H.WORK, '%d_%s_%d' % (n, name, seed)); os.makedirs(d, exist_ok=True)
    inp = os.path.join(d, 'in.txt'); R.to_fq(a, inp)
    q0 = os.path.join(d, 'q0.txt'); hp = os.path.join(d, 'hop.txt')
    qm = min(3.0, MIN / 2)
    log = run([R.FQ, 'quench', '--in', inp, '--out', q0, '--seed', str(seed), '--deadline', str(int(qm * 60)), '--loosen', '1.03'], qm * 60 + 300)
    if not os.path.exists(q0):
        return name, n, seed, out, 'no quench output: ' + log[-200:].replace('\n', ' '), time.time() - t
    srcs = [('q0', q0)]
    sq, _ = R.from_fq(q0)
    run([H.PACKER, 'hop', '--in', q0, '--s', '%.17g' % (sq - 1e-4), '--seed', str(seed), '--iters', '100000000',
         '--out', hp, '--quiet', '--temp', '1.0'], max(60, (MIN - qm) * 60))
    if os.path.exists(hp):
        srcs.append(('hop', hp))
    for tag, path in srcs:
        try:
            s, rows = R.from_fq(path)
            if len(rows) == n:
                out.append((R.side(rows), name + '+' + tag, rows))
        except Exception as e:
            err = repr(e)[:100]
    if out:
        b = min(out, key=lambda c: c[0]); bp = dict(srcs)[b[1].split('+')[-1]]
        for qn, flags in R.SCHED.items():
            o = os.path.join(d, 'rq' + qn + '.txt')
            run([R.FQ, 'quench', '--in', bp, '--out', o, '--deadline', os.environ.get('FQ_DEADLINE', '60')] + flags, 300)
            try:
                s2, rows2 = R.from_fq(o)
                if len(rows2) == n:
                    out.append((R.side(rows2), b[1] + '+' + qn, rows2))
            except Exception:
                pass
    return name, n, seed, out, err, time.time() - t


def main(pdir, base_csv, odir):
    os.makedirs(odir, exist_ok=True)
    B = board(); base = {n: R.arr(rs) for n, rs in R.load(base_csv).items() if n in PANEL}
    ours = {n: R.side_arr(base[n]) for n in PANEL}
    progs = sorted(p for p in glob.glob(os.path.join(pdir, '*.py')) if not os.path.basename(p).startswith('_'))[SHARD::NSHARD]
    print('programs', [os.path.basename(p) for p in progs], 'panel', PANEL, 'min', MIN, 'seeds', SEEDS, flush=True)
    res = {}; jobs = []; bjobs = []
    for p in progs:
        name = os.path.basename(p)[:-3]; res[name] = {}
        for n in PANEL:
            res[name][n] = dict(ours=ours[n], board=B.get(n), seed_side=[], relaxed=[], err='', params={}, by_seed={})
            for s in range(SEEDS):
                bjobs.append((name, p, n, 1000 * s + n))
    with Pool(os.cpu_count()) as pool:  # build in parallel (some programs take ~45 s)
        built = pool.map(make_candidate, [(p, n, base[n], B.get(n), sd) for name, p, n, sd in bjobs])
    for (name, p, n, sd), (a, err) in zip(bjobs, built):
        if a is None:
            res[name][n]['err'] = err; print(name, n, 'build error', err[-200:], flush=True); continue
        res[name][n]['seed_side'].append(R.side_arr(a)); res[name][n]['params'][sd + 7] = err
        print('%s n=%d seed=%d seed_side=%.6f params=%s' % (name, n, sd + 7, R.side_arr(a), err), flush=True)
        jobs.append((name, n, a, sd + 7))
    print(len(jobs), 'relax jobs', flush=True)
    wins = {}
    with Pool(os.cpu_count()) as pool:
        for name, n, seed, out, err, dt in pool.imap_unordered(relax, jobs):
            best = min((c[0] for c in out), default=None)
            res[name][n]['relaxed'].append(best); res[name][n]['by_seed'][seed] = best
            print('%s n=%d seed=%d relaxed=%s ours=%.6f board=%.6f (%.0fs) %s' % (name, n, seed, best and '%.6f' % best, ours[n], B.get(n, 0), dt, err), flush=True)
            if best is not None and best < ours[n] - 1e-9:
                k, L, tag, rs, log = R.certify((n, out))
                if L is not None and L < ours[n] - 1e-9 and (n not in wins or L < wins[n][0]):
                    wins[n] = (L, name, rs); print('CERTIFIED WIN', name, n, L, flush=True)
    for name in res:
        gaps = []
        for n, v in res[name].items():
            r = [x for x in v['relaxed'] if x is not None]
            gaps.append(min(r) - v['ours'] if r else 9.0)
        res[name]['fitness'] = float(np.mean(gaps))  # mean relaxed side minus ours over the panel (lower is better, < 0 = win)
        print('FITNESS %s %.6f %s' % (name, res[name]['fitness'], ' '.join('%.4f' % x for x in gaps)), flush=True)
    json.dump(res, open(os.path.join(odir, 'results_%d.json' % SHARD), 'w'), indent=1, default=float)
    with open(os.path.join(odir, 'submission.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle']); rid = 0
        for n, (L, name, rs) in sorted(wins.items()):
            for r in sorted(rs, key=lambda r: int(r['square_id'])):
                w.writerow([rid, n, r['square_id'], r['x'], r['y'], r['angle']]); rid += 1


if __name__ == '__main__':
    main(*sys.argv[1:4])
