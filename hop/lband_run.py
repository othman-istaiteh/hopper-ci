# L-band sweep driver (6 Nov 2026, own code) around 3jeonghun's L-band builder (Kaggle dataset
# az05192000gmailcom/squares-in-squares-l-band-packings: band_gen.py, lband.py, lband2.py, Apache 2.0; the band geometry
# and MILP assembly there are ported from lostdecimal27's notebook). His notebook found 19 puzzles ahead of the public
# frontier with a small phase grid and few targets (K >= 27 only); this driver sweeps more targets, phases, band
# widths and assembly sides on free CPUs, quenches the assemblies that reach n with fq, and keeps what beats our side.
#
#   LB_TOOLS=<dataset dir> FQ_BIN=<fq> SQ_IN=<seeds root> python3 lband_run.py OUT.csv --slice 0 --slices 8 [--kmin 27]
#        [--gapmin 0.004] [--phases 5] [--nmin 4 --nmax 8] [--df -0.003,0.004,0.010] [--tl 15] [--quench 60] [--budget-h 4.5]
# Output: competition-format rows of every quenched result below our side (plus grid rows for the other puzzles are NOT
# written; the publish job merges with our packings).
import argparse, csv, glob, math, os, sys, time, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sq09_requench as R

TOOLS = os.environ.get('LB_TOOLS', '')
sys.path.insert(0, TOOLS)
import band_gen as BG
import lband2 as LB2

FQ = os.environ.get('FQ_BIN', '')


def ours_best():
    files = sorted(set(glob.glob(os.environ.get('SQ_IN', 'seeds') + '/**/submission.csv', recursive=True)))
    best = {}
    for p in files:
        for n, rs in R.load(p).items():
            if len(rs) == n:
                try:
                    s = R.side(rs)
                except Exception:
                    continue
                if n not in best or s < best[n][0]:
                    best[n] = (s, rs)
    return best


def load_board(path):
    B = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        n, L = line.split()[:2]; B[int(n)] = float(L)
    return B


def one_target(args):
    n, s_ours, dfs, Ns, P, tl, qsec, work = args
    K = math.isqrt(n); t0 = time.time()
    os.makedirs(work, exist_ok=True)
    f0 = s_ours - K
    log = []; cands = []
    for df in dfs:
        f = f0 + df
        if not (0.0 < f < 1.0):
            continue
        for N in Ns:
            if N - 1 > K or BG.solve_theta(N, N - 1, f) is None:
                continue
            for i in range(P):
                for j in range(P):
                    pt, pr = i / P, j / P
                    try:
                        cnt, sol = LB2.assemble_split(K, f, N, pt, pr, 'right_rot', jm=3.0, time_limit=tl)
                    except Exception as e:
                        log.append(f'f={f:.4f} N={N} ph=({pt:.2f},{pr:.2f}) error {repr(e)[:80]}'); continue
                    if sol is not None and cnt >= n:
                        cands.append((f, -cnt, N, pt, pr, sol))
        if len(cands) >= 12:
            break   # enough assemblies at the smallest sides
    if not cands:
        return n, s_ours, None, 'no assembly reached n', time.time() - t0
    cands.sort(key=lambda c: (c[0], c[1]))
    best = None; tried = 0
    for f, negc, N, pt, pr, sol in cands[:6]:
        sol = np.asarray(sol, float)
        if len(sol) > n:
            # drop surplus uprights far from the band ends: keep the first n (fixed squares and MILP picks come first)
            sol = sol[:n]
        if not BG.valid(sol, None, tol=1e-7):
            continue
        tried += 1
        Lq, q = BG.quench(sol, FQ, qsec, work) if FQ else (None, None)
        L0 = BG.side_L(sol)
        log.append(f'f={f:.4f} N={N} ph=({pt:.2f},{pr:.2f}) count={-negc} L0={L0:.6f} Lq={Lq if Lq is None else round(Lq, 6)}')
        if Lq is not None and (best is None or Lq < best[0]):
            best = (Lq, q, f'N={N} f={f:.4f} ph=({pt:.2f},{pr:.2f}) count={-negc}')
    if best is None:
        return n, s_ours, None, 'no valid quench; ' + ' | '.join(log[-3:]), time.time() - t0
    return n, s_ours, best, ' | '.join(log[-4:]), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--slice', type=int, default=0); ap.add_argument('--slices', type=int, default=1)
    ap.add_argument('--kmin', type=int, default=27); ap.add_argument('--kmax', type=int, default=31)
    ap.add_argument('--gapmin', type=float, default=0.004)
    ap.add_argument('--phases', type=int, default=5)
    ap.add_argument('--nmin', type=int, default=4); ap.add_argument('--nmax', type=int, default=8)
    ap.add_argument('--df', default='-0.003,0.004,0.010')
    ap.add_argument('--tl', type=float, default=15.0)
    ap.add_argument('--quench', type=float, default=60.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--budget-h', type=float, default=4.5)
    ap.add_argument('--targets', default='', help='explicit comma list of n (overrides the gap ranking)')
    A = ap.parse_args()
    t0 = time.time()
    best = ours_best()
    board = load_board(os.environ.get('SQ_BOARD', os.path.join(HERE, 'board_29oct.txt')))
    if A.targets:
        targets = [int(v) for v in A.targets.split(',') if v]
    else:
        targets = [n for n in best if A.kmin <= math.isqrt(n) <= A.kmax and best[n][0] - board.get(n, best[n][0]) >= A.gapmin]
        targets.sort(key=lambda n: -(best[n][0] - board.get(n, best[n][0])) / math.sqrt(n))
    mine = targets[A.slice::A.slices]
    dfs = [float(v) for v in A.df.split(',')]
    Ns = list(range(A.nmin, A.nmax + 1))
    print(f'{len(targets)} targets, slice {A.slice}/{A.slices}: {mine}; dfs {dfs} N {Ns} phases {A.phases}x{A.phases} tl {A.tl}s quench {A.quench}s fq={FQ} exists={os.path.exists(FQ)}', flush=True)
    jobs = [(n, best[n][0], dfs, Ns, A.phases, A.tl, A.quench, os.path.join('lb_work', str(n))) for n in mine]
    wins = []
    with ProcessPoolExecutor(A.workers) as ex, open(A.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle']); rid = 0
        for n, s_ours, res, log, sec in ex.map(one_target, jobs):
            if res is None:
                print(f'n={n} ours={s_ours:.6f} board={board.get(n, 0):.6f} NONE {log} {sec:.0f}s', flush=True); continue
            Lq, q, tag = res
            d = s_ours - Lq
            print(f'n={n} ours={s_ours:.6f} board={board.get(n, 0):.6f} lband={Lq:.6f} diff={d:+.6f} {tag} | {log} {sec:.0f}s', flush=True)
            if d > 1e-7:
                wins.append((n, s_ours, Lq))
                for i, (x, y, t) in enumerate(q):
                    w.writerow([rid, n, i, repr(float(x)), repr(float(y)), repr(float(t) % (math.pi / 2))]); rid += 1
                f.flush()
            if time.time() - t0 > A.budget_h * 3600:
                print('budget reached', flush=True); break
    print('LBAND ' + json.dumps(dict(slice=A.slice, targets=len(mine), wins=[(n, round(a, 6), round(b, 6)) for n, a, b in wins],
                                   gain=round(sum((a - b) / math.sqrt(n) for n, a, b in wins), 6), sec=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
