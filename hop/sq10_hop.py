# SQ10 (own code): many-core basin hopping + re-quench + certified merge, for the TPU VM (many CPU cores).
# Per puzzle, the best candidates from the inputs are hopped with Evan Daniel's `packer hop` (MIT) for HOP_MIN minutes
# each (one process per job, as many jobs in parallel as there are cores), every hop output is re-quenched with his `fq`
# (loosen 1.0 and 1.02), and the certified merge of sq09 keeps per puzzle the first candidate (by side) that the metric
# copy certifies. Target weighting: puzzles with the biggest gap to the per-puzzle board get more seeds.
import os, sys, csv, math, json, glob, time, subprocess, re
import numpy as np
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
# SQ_THREADS=1: threads instead of processes (the work is in subprocesses anyway); avoids the process pool on macOS,
# where the parent died silently after the first result in two runs (29 Oct)
PoolCls = ThreadPool if os.environ.get('SQ_THREADS') == '1' else Pool
import sq09_requench as R

PACKER = os.environ.get('PACKER_BIN', '/kaggle/working/evand/search/packer/target/release/packer')
WORK = os.environ.get('SQ_WORK', '/kaggle/working/hop')
HOP_ARGS = os.environ.get('HOP_ARGS', '').split()  # extra packer hop options, e.g. '--descend 3e-5 --amp 0.7' (defaults 1e-4, 1.0)
ENGINE = os.environ.get('HOP_ENGINE', 'hop')  # 'hop' = packer hop, 'shrink' = fq --shrink --hops (30 Oct test: 20x the gain per core-hour)
SHRINK_DEFAULT = ['--loosen', '1.02', '--hops', '20', '--hop-sigma', '0.02', '--hop-k', '3']
TARGET_FRAC = float(os.environ.get('HOP_TARGET_FRAC', '0'))  # 0 = classic hop; 0.3 .. 1.0 = start the walk that far towards the board side
TARGET = {}  # n -> board side, filled in main() from SQ_BOARD


def hop_job(job):
    """job = (n, tag, rows, seed, minutes) -> (n, [(approxL, tag', rows'), ...], err)"""
    n, tag, rs, seed, minutes = job
    out = []
    try:
        a = R.arr(rs)
        d = os.path.join(WORK, '%d_%s_%d' % (n, tag.replace('/', '_')[:40], seed))
        os.makedirs(d, exist_ok=True)
        inp = os.path.join(d, 'in.txt'); s0 = R.to_fq(a, inp)
        hp = os.path.join(d, 'hop.txt')
        try:
            if ENGINE == 'shrink':
                # fq shrink-hopping (SQUISH-style): grow to feasibility, shrink the side step by step, kick a few squares when a
                # step fails; deterministic per seed, converges in minutes. HOP_ARGS: fq options (--hops, --hop-sigma, --hop-k, --loosen, --d0).
                subprocess.run([R.FQ, 'quench', '--in', inp, '--out', hp, '--shrink', '--seed', str(seed), '--deadline', str(int(minutes * 60)),
                                '--pit', '60', '--flip-top', '8'] + (HOP_ARGS or SHRINK_DEFAULT), capture_output=True, text=True, timeout=minutes * 60 + 600)
            elif ENGINE == 'hhop':
                # the packer's second engine: hard-square basin hopping (loosen, event chains, squeeze, Metropolis on the side);
                # HOP_ARGS e.g. '--temp 1e-4 --chains 8 --chainlen 1.0 --gmax 0.02 --forbid-lines 1'
                subprocess.run([PACKER, 'hhop', '--in', inp, '--seed', str(seed), '--iters', '100000000', '--out', hp] + HOP_ARGS,
                               capture_output=True, text=True, timeout=minutes * 60)
            else:
                # target-side hop: with HOP_TARGET_FRAC f > 0 the walk starts at S = s0 - f (s0 - board side), i.e. on the penalty
                # landscape at (a fraction of) the leaders' density, where rows can break and the topology can change; the
                # output is written only when the walk reaches feasibility (default f = 0: start 1e-4 below our side)
                S = s0
                if TARGET_FRAC > 0 and TARGET.get(n, 9e9) < s0:
                    S = s0 - TARGET_FRAC * (s0 - TARGET[n])
                subprocess.run([PACKER, 'hop', '--in', inp, '--s', '%.17g' % S, '--seed', str(seed), '--iters', '100000000', '--out', hp, '--quiet'] + HOP_ARGS,
                               capture_output=True, text=True, timeout=minutes * 60)
        except subprocess.TimeoutExpired:
            pass
        srcs = [('hop', hp)] if os.path.exists(hp) else []
        for name, path in srcs:
            try:
                s, rows = R.from_fq(path)
                if len(rows) == n:
                    out.append((R.side(rows), tag + '+' + name, rows))
            except Exception:
                continue
            if os.path.exists(R.FQ):
                for qn, flags in R.SCHED.items():
                    o = os.path.join(d, name + qn + '.txt')
                    try:
                        subprocess.run([R.FQ, 'quench', '--in', path, '--out', o, '--deadline', os.environ.get('FQ_DEADLINE', '120')] + flags,
                                       capture_output=True, text=True, timeout=300)
                        s2, rows2 = R.from_fq(o)
                        if len(rows2) == n:
                            out.append((R.side(rows2), tag + '+' + name + '+' + qn, rows2))
                    except Exception:
                        continue
    except Exception as e:
        return n, [], repr(e)[:120]
    return n, out, ''


def load_board(path):
    B = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        n, L, o = line.split()[:3]; B[int(n)] = float(L)
    return B


def main(out_path):
    t0 = time.time()
    files = sorted(set(glob.glob(os.environ.get('SQ_IN', '/kaggle/input') + '/**/submission.csv', recursive=True)))
    cands = {}
    for p in files:
        rows = R.load(p); tag = p.split('/')[-2]; k = 0
        for n, rs in rows.items():
            if 1 <= n <= 1000 and len(rs) == n:
                try:
                    cands.setdefault(n, []).append((R.side(rs), tag, rs)); k += 1
                except Exception:
                    pass
        print(f'{tag}: {k} puzzles', flush=True)
    maxc = int(os.environ.get('SQ_MAXC', '2'))
    for n in cands:
        seen = set(); keep = []
        for c in sorted(cands[n], key=lambda c: c[0]):
            key = round(c[0], 9)
            if key not in seen and c[0] < R.GRID(n) - 1e-9:
                seen.add(key); keep.append(c)
        cands[n] = keep[:maxc]
    cands = {n: v for n, v in cands.items() if v}
    board = load_board(os.environ.get('SQ_BOARD', 'board_29oct.txt')) if os.path.exists(os.environ.get('SQ_BOARD', 'board_29oct.txt')) else {}
    TARGET.update(board)
    nmin, nmax = int(os.environ.get('SQ_NMIN', '1')), int(os.environ.get('SQ_NMAX', '1000'))
    minutes = float(os.environ.get('HOP_MIN', '20')); seeds = int(os.environ.get('HOP_SEEDS', '1')); seed0 = int(os.environ.get('HOP_SEED0', '100'))
    ntarget = int(os.environ.get('HOP_NTARGET', '1000'))
    # gap-weighted target list: biggest (best candidate - board) / sqrt(n) first
    gaps = sorted(((cands[n][0][0] - board.get(n, cands[n][0][0])) / math.sqrt(n), n) for n in cands if nmin <= n <= nmax)[::-1]
    skip = int(os.environ.get('HOP_SKIP', '0'))  # skip the first `skip` targets (slice the list between notebooks)
    targets = [n for g, n in gaps if g > 1e-9]
    nlist = [int(x) for x in os.environ.get('HOP_NLIST', '').replace(' ', '').split(',') if x]
    if nlist:  # explicit target list (e.g. the puzzles that keep yielding), in the given order, instead of the gap ranking
        targets = [n for n in nlist if n in cands]
    targets = targets[skip:skip + ntarget]
    print(f'targets {len(targets)} (of {len(cands)} non-grid puzzles), total gap to board {sum(g for g, n in gaps if g > 0):.5f}', flush=True)
    jobs = []
    for n in targets:
        for ci, (aL, tag, rs) in enumerate(cands[n]):
            for sd in range(seeds):
                jobs.append((n, tag, rs, seed0 + 10 * ci + sd, minutes))
    workers = int(os.environ.get('SQ_WORKERS', str(os.cpu_count() or 4)))
    budget = float(os.environ.get('HOP_BUDGET_H', '5'))  # wall-clock budget for the hop stage, in hours
    if jobs and len(jobs) * minutes / workers / 60 > budget:
        minutes = max(3.0, budget * 60 * workers / len(jobs))
        jobs = [(n, tag, rs, sd, minutes) for (n, tag, rs, sd, _) in jobs]
    print(f'{len(jobs)} hop jobs x {minutes:.1f} min on {workers} workers: about {len(jobs) * minutes / workers / 60:.1f} h; packer={os.path.exists(PACKER)} fq={os.path.exists(R.FQ)}', flush=True)
    nh = 0; errs = 0; imp = {}
    if os.path.exists(PACKER):
        with PoolCls(workers) as pool:
            for k, (n, out, err) in enumerate(pool.imap_unordered(hop_job, jobs, chunksize=1)):
                if err:
                    errs += 1
                    if errs <= 5: print('hop error', n, err, flush=True)
                for c in out:
                    cands[n].append(c); nh += 1
                    if c[0] < min(x[0] for x in cands[n][:maxc]) - 1e-9:
                        imp[n] = min(imp.get(n, 99.0), c[0])
                if k % 50 == 0:
                    print(f'hopped {k}/{len(jobs)} jobs, {nh} outputs, {len(imp)} puzzles improved (approx), {time.time() - t0:.0f}s', flush=True)
    print(f'hop stage done: {nh} outputs, {errs} errors, {len(imp)} puzzles improved (approx), {time.time() - t0:.0f}s', flush=True)
    best = {}; fails = []
    with PoolCls(workers) as pool:
        cj = [(n, cands[n]) for n in sorted(cands, key=lambda n: -n)]
        for n, L, tag, rs, log in pool.imap_unordered(R.certify, cj, chunksize=1):
            if log: fails.append((n, log))
            if L is not None and L < best.get(n, (R.GRID(n),))[0] - 1e-13:
                best[n] = (L, tag, rs)
    print(f'certified {len(best)} puzzles, total {R.total(best):.10f}, {time.time() - t0:.0f}s', flush=True)
    gain = 0.0; impl = []
    for n in best:
        orig = min((c[0] for c in cands[n][:maxc]), default=R.GRID(n))
        if best[n][0] < orig - 1e-9:
            impl.append((n, orig, best[n][0])); gain += (orig - best[n][0]) / math.sqrt(n)
    print(f'search improved {len(impl)} puzzles (certified), score gain vs best original candidates {gain:.6f}', flush=True)
    print('IMPROVED ' + ' '.join('%d:%.9f>%.9f' % t for t in sorted(impl)), flush=True)
    for n, log in sorted(fails)[:40]:
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
                k = R.GRID(n)
                for i in range(n):
                    w.writerow([rid, n, i, repr(0.5 + i % k), repr(0.5 + i // k), '0.0']); rid += 1
    print('SIDES ' + ' '.join('%d:%.10f' % (n, best[n][0]) for n in sorted(best)), flush=True)
    print(json.dumps(dict(certified_total=R.total(best), n_certified=len(best), sources=src, hopped=nh, sec=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
