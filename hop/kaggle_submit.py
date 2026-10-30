# Submit a certified merge to the Kaggle competition from GitHub Actions, only when it beats the best public score of
# our previous submissions and the daily budget allows (Kaggle allows 10 per day; we keep MAX_PER_DAY for automation).
# Usage: python3 kaggle_submit.py merge.log out/submission.csv "message"
# Needs the kaggle CLI configured (KAGGLE_USERNAME / KAGGLE_KEY in the environment, stored as GitHub secrets by Othman).
import csv, datetime, io, json, os, re, subprocess, sys

COMP = 'squares-in-squares'
MAX_PER_DAY = int(os.environ.get('KAGGLE_MAX_PER_DAY', '8'))
MIN_GAIN = 1e-9


def sh(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def main():
    log, csv_path, msg = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'GitHub Actions merge'
    total = None
    for line in open(log, errors='replace'):
        m = re.search(r'"certified_total":\s*([0-9.]+)', line)
        if m: total = float(m.group(1))
    if total is None:
        print('no certified_total in the merge log: not submitting'); return
    rc, out, err = sh(['kaggle', 'competitions', 'submissions', '-c', COMP, '-v'])
    if rc != 0:
        print('submissions listing failed:', err[-300:]); return
    rows = list(csv.DictReader(io.StringIO(out)))
    today = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
    n_today = sum(1 for r in rows if (r.get('date') or '').startswith(today))
    scores = []
    for r in rows:
        try: scores.append(float(r.get('publicScore') or 'nan'))
        except ValueError: pass
    best = min((s for s in scores if s == s), default=float('inf'))
    pending = sum(1 for r in rows if not any(k in (r.get('status') or '').lower() for k in ('complete', 'error')))  # status looks like SubmissionStatus.COMPLETE
    print(json.dumps(dict(certified_total=total, best_public=best, submissions_today=n_today, pending=pending)))
    if total > best - MIN_GAIN:
        print('not better than the best submission: not submitting'); return
    if n_today + pending >= MAX_PER_DAY:
        print('daily automation budget used: not submitting'); return
    rc, out, err = sh(['kaggle', 'competitions', 'submit', '-c', COMP, '-f', csv_path, '-m', f'{msg} (certified {total:.10f})'])
    print('submit rc', rc, out[-300:], err[-300:])


if __name__ == '__main__':
    main()
