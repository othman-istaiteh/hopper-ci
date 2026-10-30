# explore_run.py (own glue code): one puzzle, one job. Runs Evan Daniel's breadth-first basin explorer
# (search/packer/explore.py, MIT, github.com/evand/square-packing: MAP-Elites archive of polished fq minima, moves
# kick/lkick/bigkick/crot/aswap/band/rowslide/mirror/cross2/ashallow, SCREEN quench with loosen 1.0/1.02/1.05, full
# polish gated to within --polish-gate of the best) from OUR best certified packing(s) of puzzle n, then harvests the
# archive, certifies the best polished minima below our start with the metric copy (sq_metric via sq09_requench) and
# writes a one-puzzle submission CSV plus a one-line JSON summary.
#
# Environment (all optional except EX_N):
#   EX_N            puzzle size n
#   EX_MINUTES      explorer wall budget (default 60)          EX_PROCS   worker processes (default 4)
#   EX_SEED         explorer seed (default 1)                  EX_XARGS   extra explore.py arguments (one string)
#   EX_PACKER       the author's search/packer directory (explore.py, target/release/fq, target/release/anneal);
#                   default <repo>/evand/search/packer
#   EX_OUT          run directory (default $SQ_WORK/ex<n>, SQ_WORK default <repo>/work)
#   EX_SMAX_ABOVE   stepping-stone ceiling above our best side (default 0.05; explore.py --smax)
#   EX_GATE         polish gate (default 2e-3; explore.py --polish-gate)
#   EX_NSTARTS      at most this many distinct start packings within EX_SMAX_ABOVE of the best (default 4)
#   EX_NCERT        certify at most this many archive minima below the start (default 6)
#   EX_DRAIN_MIN    grace minutes after EX_MINUTES before the explorer is terminated (default 20)
#   SQ_IN           seeds root, every **/submission.csv is a candidate source (sq09_requench loader)
#   FQ_BIN          fq binary for the start pre-polish (default EX_PACKER/target/release/fq)
# Usage: python3 explore_run.py OUT_CSV [SUMMARY_JSON]
#
# What explore.py expects and writes (read from its source, commit 0e76d9f): --starts are text files in its packing
# format ("n s" then "x y deg" rows, box [0, s]^2; sq09_requench.to_fq writes exactly that); each start is polished in
# place (fq --no-alm --loosen 1.0) and becomes archive entry 'start:<file>'. Under --out it writes manifest.json,
# archive.jsonl (one JSON line per distinct basin: i, s, path, kind, parent, unpolished ...), proposals.jsonl, state.json
# and the basin packings b00000.txt, b00001.txt, ... in the same text format. Entries flagged "unpolished" are gated
# stepping stones (screen side only) and are not harvested. Starts that are copies of the same packing (loosened or
# not) polish back into the same basin and are dropped as duplicates, so extra starts are only taken from DISTINCT
# candidates in the seeds; the explorer's own moves provide the kicked copies.
import os, sys, csv, json, glob, math, time, shutil, signal, select, subprocess

HOP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HOP)
sys.path.insert(0, HOP)
import sq09_requench as R


def env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, '') else default


def candidates(n, root):
    """Distinct non-grid candidates for puzzle n from every **/submission.csv under root: [(side, tag, rows)] by side."""
    out = []
    for p in sorted(set(glob.glob(root + '/**/submission.csv', recursive=True))):
        rows = R.load(p).get(n)
        tag = p.split('/')[-2]
        if rows and len(rows) == n:
            try:
                out.append((float(R.side(rows)), tag, rows))     # plain float: np.float64 repr breaks argv / JSON
            except Exception as e:
                print(f'skip {tag}: {e!r}'[:200], flush=True)
    seen, keep = set(), []
    for c in sorted(out, key=lambda c: c[0]):
        key = round(c[0], 9)
        if key not in seen and c[0] < R.GRID(n) - 1e-9:
            seen.add(key); keep.append(c)
    return keep


def fq_run(fq, inp, out, flags, timeout=600):
    """fq quench inp -> out; returns (feasible, side) from fq's JSON line, (False, None) on any failure."""
    try:
        r = subprocess.run([fq, 'quench', '--in', inp, '--out', out] + flags, capture_output=True, text=True, timeout=timeout)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        ok = r.returncode == 0 and os.path.exists(out) and d.get('min_gap', -1) >= 0 and d.get('min_wall', -1) >= 0
        return ok, float(d.get('s', float('nan')))
    except Exception as e:
        print(f'fq failed on {inp}: {e!r}'[:200], flush=True)
        return False, None


def prepare_start(fq, n, i, side, tag, rows, sdir):
    """Write our packing in the explorer's text format and pre-polish it with fq (polish only; ALM screen as the fallback
    for an input fq finds infeasible). Returns (path, polished side) or None if fq cannot make it feasible."""
    raw = os.path.join(sdir, f'raw{i}_{tag[:30]}.txt')
    R.to_fq(R.arr(rows), raw)
    if not os.path.exists(fq):
        print(f'no fq at {fq}: start {i} used unpolished', flush=True)
        return raw, side
    pol = os.path.join(sdir, f'start{i}_{tag[:30]}.txt')
    ok, s = fq_run(fq, raw, pol, ['--no-alm', '--loosen', '1.0'])
    if not ok:
        ok, s = fq_run(fq, raw, pol, ['--loosen', '1.0', '--pit', '8', '--flip-top', '0'])
        if not ok:
            print(f'start {i} ({tag}, side {side:.9f}) rejected by fq', flush=True)
            return None
        print(f'start {i} ({tag}) needed the ALM screen: {side:.9f} -> {s:.9f}', flush=True)
    else:
        print(f'start {i} ({tag}) polished: {side:.9f} -> {s:.9f}', flush=True)
    try:
        s2, rows2 = R.from_fq(pol)
        if len(rows2) != n:
            return None
    except Exception:
        return None
    return pol, s


def run_explorer(cmd, cwd, log_path, budget_s):
    """Run explore.py, streaming its output to log_path and stdout. After budget_s the process group gets SIGTERM
    (explore.py's manifest handler exits cleanly; archive.jsonl and the b*.txt files are flushed per entry), then
    SIGKILL. Returns the exit code (None if killed)."""
    t0 = time.time()
    with open(log_path, 'w') as log:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            while True:
                line = p.stdout.readline() if select.select([p.stdout], [], [], 10.0)[0] else ''
                if line:
                    log.write(line); log.flush()
                    if not line.startswith('  bandit'):
                        sys.stdout.write(line); sys.stdout.flush()
                elif p.poll() is not None:
                    break
                if time.time() - t0 > budget_s and p.poll() is None:
                    print(f'explorer over budget after {(time.time() - t0) / 60:.1f} min: SIGTERM', flush=True)
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        print('explorer still alive: SIGKILL', flush=True)
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait(timeout=60)
                    break
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    return p.returncode


def harvest(rundir):
    """archive.jsonl -> (all entries, polished entries by side)."""
    path = os.path.join(rundir, 'archive.jsonl')
    E = []
    if os.path.exists(path):
        for l in open(path):
            try:
                E.append(json.loads(l))
            except Exception:
                pass
    pol = sorted((e for e in E if not e.get('unpolished') and 's' in e and 'path' in e), key=lambda e: e['s'])
    return E, pol


def write_csv(path, n, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
        for rid, r in enumerate(sorted(rows, key=lambda r: int(r['square_id']))):
            w.writerow([rid, n, r['square_id'], r['x'], r['y'], r['angle']])


def main(out_csv, summary_path):
    t0 = time.time()
    n = env('EX_N', None, int)
    if n is None:
        raise SystemExit('EX_N (puzzle size) is required')
    minutes = env('EX_MINUTES', 60.0, float); procs = env('EX_PROCS', 4, int); seed = env('EX_SEED', 1, int)
    packer = env('EX_PACKER', os.path.join(REPO, 'evand', 'search', 'packer'))
    rundir = env('EX_OUT', os.path.join(env('SQ_WORK', os.path.join(REPO, 'work')), f'ex{n}'))
    smax_above = env('EX_SMAX_ABOVE', 0.05, float); gate = env('EX_GATE', 2e-3, float)
    nstarts = env('EX_NSTARTS', 4, int); ncert = env('EX_NCERT', 6, int); drain = env('EX_DRAIN_MIN', 20.0, float)
    xargs = env('EX_XARGS', '').split()
    fq = R.FQ if os.path.exists(R.FQ) else os.path.join(packer, 'target', 'release', 'fq')
    explore_py = os.path.join(packer, 'explore.py')
    os.makedirs(rundir, exist_ok=True); os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    summ = dict(n=n, minutes=minutes, procs=procs, seed=seed, start_side=None, best_found=None, certified_side=None,
                improved=False, n_minima=0, n_basins=0)

    def finish(code=0, **kw):
        summ.update(kw, wall_min=round((time.time() - t0) / 60, 2))
        with open(summary_path, 'w') as f:
            json.dump(summ, f)
        print('SUMMARY ' + json.dumps(summ), flush=True)
        raise SystemExit(code)

    if not os.path.exists(explore_py):
        finish(1, error=f'explore.py not found at {explore_py}')
    cands = candidates(n, env('SQ_IN', '/kaggle/input'))
    if not cands:
        finish(1, error='no candidate below the grid side in the seeds')
    best_side, best_tag = cands[0][0], cands[0][1]
    summ.update(start_side=best_side, start_tag=best_tag, n_candidates=len(cands))
    print(f'n {n}: {len(cands)} distinct candidates, best {best_side:.9f} ({best_tag}); fq={fq} exists={os.path.exists(fq)}', flush=True)
    sdir = os.path.join(rundir, 'starts'); os.makedirs(sdir, exist_ok=True)
    starts = []
    for i, (side, tag, rows) in enumerate(cands[:nstarts]):
        if side >= best_side + smax_above:
            break
        r = prepare_start(fq, n, i, side, tag, rows, sdir)
        if r:
            starts.append((r[0], r[1], tag, side))
    if not starts:
        finish(1, error='no start accepted by fq')
    summ['starts'] = [dict(tag=t, side=s0, polished=sp) for _, sp, t, s0 in starts]
    smax = best_side + smax_above
    cmd = [sys.executable, '-u', explore_py, '--n', str(n), '--starts', *[p for p, *_ in starts],
           '--minutes', repr(minutes), '--procs', str(procs), '--seed', str(seed), '--adapt', '--adapt-cells',
           '--polish-gate', repr(gate), '--smax', repr(smax), '--final-polish', '8', '--out', rundir] + xargs
    print('explore:', ' '.join(cmd), flush=True)
    rc = run_explorer(cmd, packer, os.path.join(rundir, 'log.txt'), 60 * (minutes + drain))
    summ['explore_rc'] = rc
    try:
        man = json.load(open(os.path.join(rundir, 'manifest.json')))
        summ['stop'] = man.get('summary', {}).get('stop', man.get('status'))
        summ['stats'] = man.get('summary', {}).get('stats')
        summ['cpu_s'] = man.get('cpu_s')
    except Exception:
        pass
    try:
        rep = subprocess.run([sys.executable, explore_py, '--report', rundir], cwd=packer, capture_output=True, text=True, timeout=600)
        open(os.path.join(rundir, 'report.txt'), 'w').write(rep.stdout + rep.stderr)
        print(rep.stdout, flush=True)
    except Exception as e:
        print(f'report failed: {e!r}'[:200], flush=True)
    E, pol = harvest(rundir)
    summ.update(n_basins=len(E), n_minima=len(pol))
    if not pol:
        finish(0 if rc == 0 else 1, error='empty archive')
    summ['best_found'] = pol[0]['s']
    summ['best_gap'] = pol[0]['s'] - best_side
    below = [e for e in pol if e['s'] < best_side - 1e-9]
    summ['n_below_start'] = len(below)
    print(f'archive: {len(E)} basins, {len(pol)} polished, best {pol[0]["s"]:.10f} ({pol[0]["s"] - best_side:+.3e} vs start), '
          f'{len(below)} below the start', flush=True)
    # keep the best few minima as text for later deep polish (small: a few files)
    for j, e in enumerate(pol[:3]):
        try:
            shutil.copy(e['path'], os.path.join(rundir, f'best{j}.txt'))
        except Exception:
            pass
    cc = []
    for e in below[:ncert]:
        try:
            s, rows = R.from_fq(e['path'])
            if len(rows) == n:
                cc.append((float(R.side(rows)), f'explore-b{e["i"]}', rows))
        except Exception as ex:
            print(f'cannot read {e.get("path")}: {ex!r}'[:200], flush=True)
    if not cc:
        finish(0)
    _, L, tag, rs, log = R.certify((n, cc))
    L = None if L is None else float(L)
    summ['cert_log'] = [list(map(str, x)) for x in log][:12]
    if L is not None and L < best_side - 1e-9:
        write_csv(out_csv, n, rs)
        summ.update(certified_side=L, certified_tag=tag, improved=True, gain=(best_side - L) / math.sqrt(n))
        print(f'IMPROVED {n}: {best_side:.9f} -> {L:.9f} ({tag}), score gain {(best_side - L) / math.sqrt(n):.3e}', flush=True)
    else:
        summ['certified_side'] = L
        print(f'no certified improvement for {n} ({len(cc)} tried)', flush=True)
    finish(0)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: explore_run.py OUT_CSV [SUMMARY_JSON]')
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), 'summary.json'))
