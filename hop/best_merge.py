# Uncertified merge: per puzzle the candidate with the smallest bounding side across ROOT/*/submission.csv -> OUT.csv.
# (Fast reference of "our current sides" for harvest comparisons; the certified merge is sq09_requench.py.)
# Usage: python3 best_merge.py ROOT OUT.csv
import csv, glob, os, sys
import sq09_requench as R


def main():
    root, out = sys.argv[1:3]
    best = {}
    for p in sorted(glob.glob(os.path.join(root, '*', 'submission.csv'))):
        for n, rs in R.load(p).items():
            if len(rs) != n:
                continue
            try:
                s = R.side(rs)
            except Exception:
                continue
            if n not in best or s < best[n][0] - 1e-12:
                best[n] = (s, rs)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle']); rid = 0
        for n in sorted(best):
            for r in sorted(best[n][1], key=lambda r: int(r['square_id'])):
                w.writerow([rid, n, r['square_id'], r['x'], r['y'], r['angle']]); rid += 1
    print(f'{len(best)} puzzles, total {sum(s / n ** 0.5 - 1 for n, (s, rs) in best.items()):.6f} (uncertified)')


if __name__ == '__main__':
    main()
