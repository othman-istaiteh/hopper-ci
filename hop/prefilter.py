# Shrink candidate slices before a certified merge: our best side per puzzle from OURS_ROOT/*/submission.csv, then every
# CAND_ROOT/*/submission.csv is rewritten in place with only the puzzles whose bounding side beats ours by > 1e-9 (empty
# slices are removed). Keeps the merge's memory small when dozens of public submissions are harvested.
# Usage: python3 prefilter.py OURS_ROOT CAND_ROOT
import csv, glob, os, shutil, sys
from update_soft import sides


def main():
    ours_root, cand_root = sys.argv[1:3]
    best = {}
    for p in glob.glob(os.path.join(ours_root, '*', 'submission.csv')):
        for n, s in sides(p).items():
            best[n] = min(best.get(n, 9e9), s)
    kept_total = 0
    for p in sorted(glob.glob(os.path.join(cand_root, '*', 'submission.csv'))):
        try:
            s = sides(p)
        except Exception as e:
            print(f'{p}: unreadable ({e!r:.60}), removed'); shutil.rmtree(os.path.dirname(p)); continue
        keep = {n for n in s if s[n] < best.get(n, 9e9) - 1e-9}
        if not keep:
            shutil.rmtree(os.path.dirname(p)); continue
        tmp = p + '.tmp'
        with open(p) as f, open(tmp, 'w', newline='') as g:
            r = csv.reader(f); w = csv.writer(g); w.writerow(next(r)); k = 0
            for row in r:
                if int(row[1]) in keep:
                    row[0] = k; w.writerow(row); k += 1
        os.replace(tmp, p)
        kept_total += len(keep)
        print(f'{os.path.basename(os.path.dirname(p))}: {len(keep)} puzzles below ours: {sorted(keep)[:20]}')
    print(f'prefilter: {kept_total} candidate puzzles kept')


if __name__ == '__main__':
    main()
