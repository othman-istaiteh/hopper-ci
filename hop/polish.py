# POLISH (own code): deep polish pass over our non-grid packings. Our sq09 re-quench only ever used Evan Daniel's SCREEN
# schedule (`fq quench --pit 8 --flip-top 0`, MIT, github.com/evand/square-packing), so none of our packings has had his
# POLISH_DEEP schedule (search/packer/pk/engine.py: `--no-alm --loosen 1.0 --flip-top 64 --pit 600`): no augmented-
# Lagrangian phase (the packing stays in its basin), 600 SLP polish iterations instead of 300, 64 corner-corner branch
# flips tried per stall instead of 8; we add `--finish` (Newton finish on the stable load set) and `--deadline`.
# Optional second step: his search/exact/descend.py (MILP escape from false jams at corner-corner touches, then his slp2
# re-squeeze; needs only numpy, scipy with HiGHS and mpmath; no store), run on the polished packing when PO_DESCEND points
# at the script. Original, polished and descended candidates are then certified with sq09's rule (candidates tried by side,
# the first one the metric copy certifies is kept). The slice (PO_SKIP, PO_NTARGET) is taken over the puzzles ordered by
# gap to the per-puzzle board (ties and no board: n descending). Output: the certified merge of the slice (grid rows for
# every other puzzle), merged with the seeds afterwards by sq09_requench.py.
#
#   FQ_BIN=.../fq PO_DESCEND=.../search/exact/descend.py SQ_IN=seeds SQ_WORKERS=4 PO_SKIP=0 PO_NTARGET=28 python3 polish.py out.csv
#
# env: SQ_IN (glob root, seeds/**/submission.csv), SQ_WORK (scratch dir), SQ_BOARD (board file, default board_29oct.txt next to
# this script), SQ_WORKERS, SQ_NMIN, SQ_NMAX, PO_SKIP, PO_NTARGET, PO_DEADLINE (fq --deadline seconds, default 900; the
# subprocess is killed at deadline + 120), PO_PIT (600), PO_FLIP (64), PO_DESCEND (path of descend.py; unset or missing: step
# skipped), PO_DESC_TIMEOUT (600 s), PO_DESC_ROUNDS (20), PO_BUDGET_H (wall-clock budget in hours, default 5.0: puzzles that
# have not started by then are passed through unpolished).
import os, sys, csv, math, json, glob, time, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sq09_requench as R

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('SQ_WORK', os.path.join(os.getcwd(), 'po_work'))
DEADLINE = float(os.environ.get('PO_DEADLINE', '900'))
PIT = os.environ.get('PO_PIT', '600')
FLIP = os.environ.get('PO_FLIP', '64')
DESCEND = os.environ.get('PO_DESCEND', '')
DESC_TIMEOUT = float(os.environ.get('PO_DESC_TIMEOUT', '600'))
DESC_ROUNDS = os.environ.get('PO_DESC_ROUNDS', '20')
BUDGET_S = float(os.environ.get('PO_BUDGET_H', '5.0')) * 3600
DEEP = ['--no-alm', '--loosen', '1.0', '--pit', PIT, '--flip-top', FLIP, '--finish']
T0 = time.time()


def load_board(path):
    B = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        n, L, o = line.split()[:3]; B[int(n)] = float(L)
    return B


def read_out(path, n):
    """fq / descend output file -> (side, rows) if it holds exactly n squares, else None."""
    if not os.path.exists(path):
        return None
    try:
        s, rows = R.from_fq(path)
        if len(rows) == n:
            return R.side(rows), rows
    except Exception:
        pass
    return None


def polish_job(job):
    """job = (n, tag, rows) -> dict(n, orig, pol, desc, L, tag, rows, log, sec, err)."""
    n, tag, rs = job
    t0 = time.time()
    res = dict(n=n, orig=R.side(rs), pol=None, desc=None, L=None, tag='grid', rows=None, log=[], sec=0.0, err='', skipped=False)
    cands = [(res['orig'], tag, rs)]
    try:
        if t0 - T0 > BUDGET_S:
            res['skipped'] = True
        else:
            d = os.path.join(WORK, '%d_%s' % (n, tag.replace('/', '_')[:40]))
            os.makedirs(d, exist_ok=True)
            inp = os.path.join(d, 'in.txt'); R.to_fq(R.arr(rs), inp)
            pol = os.path.join(d, 'deep.txt'); dsc = os.path.join(d, 'desc.txt')
            for p in (pol, dsc):
                if os.path.exists(p):
                    os.remove(p)
            if os.path.exists(R.FQ):
                try:
                    subprocess.run([R.FQ, 'quench', '--in', inp, '--out', pol, '--deadline', '%g' % DEADLINE] + DEEP,
                                   capture_output=True, text=True, timeout=DEADLINE + 120)
                except subprocess.TimeoutExpired:
                    res['err'] = 'fq timeout'
                got = read_out(pol, n)
                if got:
                    res['pol'] = got[0]; cands.append((got[0], tag + '+deep', got[1]))
            # exact corner-corner descent on the polished packing (on the input when the polish produced nothing)
            if DESCEND and os.path.exists(DESCEND):
                src = pol if res['pol'] is not None else inp
                try:
                    r = subprocess.run([sys.executable, DESCEND, src, dsc, '--rounds', DESC_ROUNDS],
                                       capture_output=True, text=True, timeout=DESC_TIMEOUT)
                    res['log'] = [l for l in r.stdout.splitlines() if l.startswith('round')][-3:]
                except subprocess.TimeoutExpired:
                    res['err'] += ' descend timeout'
                got = read_out(dsc, n)
                if got:
                    res['desc'] = got[0]; cands.append((got[0], tag + ('+deep' if res['pol'] is not None else '') + '+desc', got[1]))
        _, L, ctag, crows, clog = R.certify((n, cands))
        res.update(L=L, tag=ctag, rows=crows)
        if clog:
            res['log'] += [str(x)[:100] for x in clog[:3]]
    except Exception as e:
        res['err'] += ' ' + repr(e)[:120]
    res['sec'] = time.time() - t0
    return res


def fmt(x):
    return '%.9f' % x if x is not None else '-'


def main(out_path):
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
    # SQ_MAXC=1 semantics: the single best candidate below the grid side per puzzle
    best_in = {}
    for n in cands:
        c = min(cands[n], key=lambda c: c[0])
        if c[0] < R.GRID(n) - 1e-9:
            best_in[n] = c
    bpath = os.environ.get('SQ_BOARD', os.path.join(HERE, 'board_29oct.txt'))
    board = load_board(bpath) if os.path.exists(bpath) else {}
    nmin, nmax = int(os.environ.get('SQ_NMIN', '1')), int(os.environ.get('SQ_NMAX', '1000'))
    # order: biggest (best candidate - board) / sqrt(n) first, then n descending (no board: n descending)
    order = sorted((n for n in best_in if nmin <= n <= nmax),
                   key=lambda n: (-(best_in[n][0] - board.get(n, best_in[n][0])) / math.sqrt(n), -n))
    skip, ntarget = int(os.environ.get('PO_SKIP', '0')), int(os.environ.get('PO_NTARGET', '1000'))
    targets = order[skip:skip + ntarget]
    workers = int(os.environ.get('SQ_WORKERS', str(os.cpu_count() or 4)))
    print(f'{len(best_in)} non-grid puzzles, board {len(board)} rows; slice {skip}..{skip + len(targets)}: {len(targets)} puzzles; '
          f'fq={R.FQ} exists={os.path.exists(R.FQ)}; descend={DESCEND or "-"} exists={bool(DESCEND) and os.path.exists(DESCEND)}; '
          f'deadline {DEADLINE:g}s, {workers} workers, about {len(targets) * (DEADLINE + 60) / workers / 3600:.1f} h', flush=True)
    jobs = [(n, best_in[n][1], best_in[n][2]) for n in targets]
    best = {}; npol = 0; ndesc = 0; nskip = 0; imp = []; gain = 0.0; fails = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(polish_job, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs)):
            r = fu.result(); n = r['n']
            if r['skipped']: nskip += 1
            if r['pol'] is not None: npol += 1
            if r['desc'] is not None: ndesc += 1
            if r['L'] is not None:
                best[n] = (r['L'], r['tag'], r['rows'])
                d = r['orig'] - r['L']
                if d > 1e-9:
                    imp.append((n, r['orig'], r['L'])); gain += d / math.sqrt(n)
                dtxt = '%+.3e' % d
            else:
                fails.append((n, r['log'])); dtxt = 'n/a'
            print(f'[{k + 1}/{len(jobs)}] n={n} orig={fmt(r["orig"])} pol={fmt(r["pol"])} desc={fmt(r["desc"])} '
                  f'cert={fmt(r["L"])} ({r["tag"]}) imp={dtxt} {r["sec"]:.0f}s'
                  + (' ' + r['err'].strip() if r['err'].strip() else '') + (' SKIPPED(budget)' if r['skipped'] else ''), flush=True)
            desc_won = r['desc'] is not None and r['desc'] < min(x for x in (r['orig'], r['pol']) if x is not None) - 1e-9
            if r['log'] and (desc_won or r['L'] is None):
                for l in r['log']:
                    print('    ', l, flush=True)
    print(f'polish stage done: {npol} polished, {ndesc} descended, {nskip} skipped (budget), {len(best)} certified, '
          f'{len(imp)} improved, {time.time() - T0:.0f}s', flush=True)
    print('IMPROVED ' + ' '.join('%d:%.9f>%.9f' % t for t in sorted(imp)), flush=True)
    for n, log in sorted(fails)[:40]:
        print('rejected', n, log[:3], flush=True)
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
    print(json.dumps(dict(puzzles=len(jobs), certified=len(best), polished=npol, descended=ndesc, skipped=nskip,
                          improved=len(imp), gain=round(gain, 9), sec=round(time.time() - T0))), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
