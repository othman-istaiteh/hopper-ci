# A/B test of search engines on a few puzzles from our best packings (30 Oct): Evan Daniel's `packer hop` (our current
# engine) against his `fq quench --shrink --hops` (SQUISH-style iterative shrink-hopping, the engine he moved to).
# Env: SQ_IN (seed root), ST_N "865,742,587,648,911,498", ST_MIN minutes per run, ST_CFG one of hop / sh100 / sh102 /
# sh100b, ST_SEEDS seeds per puzzle, FQ_BIN, PACKER_BIN, SQ_WORKERS. Prints per puzzle the start side and the best
# certified side per config, and a JSON summary.
import os, sys, glob, json, time, subprocess
from multiprocessing.pool import ThreadPool
import sq09_requench as R

PACKER = os.environ.get('PACKER_BIN', '/kaggle/working/evand/search/packer/target/release/packer')
WORK = os.environ.get('SQ_WORK', '/tmp/shtest')
CFG = os.environ.get('ST_CFG', 'hop')
MIN = float(os.environ.get('ST_MIN', '15'))
SEEDS = int(os.environ.get('ST_SEEDS', '2'))
NS = [int(x) for x in os.environ.get('ST_N', '865,742,587,648,911,498').split(',') if x.strip()]
SH = {'sh100': ['--loosen', '1.0', '--hops', '20', '--hop-sigma', '0.02', '--hop-k', '3'],
      'sh102': ['--loosen', '1.02', '--hops', '20', '--hop-sigma', '0.02', '--hop-k', '3'],
      'sh100b': ['--loosen', '1.0', '--hops', '40', '--hop-sigma', '0.05', '--hop-k', '0'],
      'sh100d': ['--loosen', '1.0', '--hops', '20', '--hop-sigma', '0.02', '--hop-k', '3', '--d0', '2e-4']}


def run(job):
    n, rs, seed = job
    d = os.path.join(WORK, '%d_%d' % (n, seed)); os.makedirs(d, exist_ok=True)
    inp = os.path.join(d, 'in.txt'); s0 = R.to_fq(R.arr(rs), inp)
    out = []; t = time.time()
    try:
        if CFG == 'hop':
            hp = os.path.join(d, 'hop.txt')
            try:
                subprocess.run([PACKER, 'hop', '--in', inp, '--s', '%.17g' % s0, '--seed', str(seed), '--iters', '100000000', '--out', hp, '--quiet'],
                               capture_output=True, text=True, timeout=MIN * 60)
            except subprocess.TimeoutExpired:
                pass
            if os.path.exists(hp):
                s, rows = R.from_fq(hp)
                if len(rows) == n: out.append((R.side(rows), 'hop', rows))
                for qn, flags in R.SCHED.items():
                    o = os.path.join(d, 'hop' + qn + '.txt')
                    try:
                        subprocess.run([R.FQ, 'quench', '--in', hp, '--out', o, '--deadline', '120'] + flags, capture_output=True, text=True, timeout=300)
                        s2, rows2 = R.from_fq(o)
                        if len(rows2) == n: out.append((R.side(rows2), 'hop' + qn, rows2))
                    except Exception:
                        pass
        else:
            o = os.path.join(d, 'sh.txt')
            r = subprocess.run([R.FQ, 'quench', '--in', inp, '--out', o, '--shrink', '--seed', str(seed), '--deadline', str(int(MIN * 60)), '--pit', '60', '--flip-top', '8'] + SH[CFG],
                               capture_output=True, text=True, timeout=MIN * 60 + 600)
            if os.path.exists(o):
                s, rows = R.from_fq(o)
                if len(rows) == n: out.append((R.side(rows), CFG, rows))
            else:
                return n, seed, s0, [], 'no output: ' + (r.stderr or r.stdout)[-200:], time.time() - t
    except Exception as e:
        return n, seed, s0, out, repr(e)[:200], time.time() - t
    return n, seed, s0, out, '', time.time() - t


def main():
    files = sorted(set(glob.glob(os.environ.get('SQ_IN', '/kaggle/input') + '/**/submission.csv', recursive=True)))
    cands = {}
    for p in files:
        rows = R.load(p); tag = p.split('/')[-2]
        for n, rs in rows.items():
            if n in NS and len(rs) == n:
                cands.setdefault(n, []).append((R.side(rs), tag, rs))
    jobs = []
    for n in NS:
        if n not in cands: print('no seed for', n); continue
        best = min(cands[n], key=lambda c: c[0])
        for sd in range(1, SEEDS + 1): jobs.append((n, best[2], sd))
    print(f'{CFG}: {len(jobs)} runs x {MIN} min; packer={os.path.exists(PACKER)} fq={os.path.exists(R.FQ)}', flush=True)
    res = {}
    with ThreadPool(int(os.environ.get('SQ_WORKERS', '4'))) as pool:
        for n, seed, s0, out, err, dt in pool.imap_unordered(run, jobs):
            if err: print(f'n {n} seed {seed}: ERROR {err}', flush=True)
            L, tag, rows, log = R.certify((n, out)) [1:5] if out else (None, '', None, [])
            cert = L if L is not None else float('nan')
            approx = min((c[0] for c in out), default=float('nan'))
            print(f'n {n} seed {seed}: start {s0:.9f} approx {approx:.9f} certified {cert:.9f} ({tag}) delta {cert - s0:+.3e} in {dt:.0f}s', flush=True)
            r = res.setdefault(n, dict(start=s0, best=float('inf')))
            if L is not None: r['best'] = min(r['best'], L)
    tot = sum((r['best'] - r['start']) / n ** 0.5 for n, r in res.items() if r['best'] < r['start'])
    print(json.dumps(dict(cfg=CFG, minutes=MIN, seeds=SEEDS, score_gain=-tot, per_n={n: round(r['best'] - r['start'], 9) if r['best'] < 1e9 else None for n, r in res.items()})), flush=True)


if __name__ == '__main__':
    main()
