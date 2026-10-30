# Keep only the puzzles of a merged submission.csv that are below the best side found in a root of */submission.csv
# files (so that a worker commits a small file holding only its improvements).
# Usage: python3 improved_only.py SEEDS_ROOT merged.csv out.csv
import csv, glob, os, sys
from update_soft import sides


def main():
    root, merged, out = sys.argv[1:4]
    best = {}
    for p in glob.glob(os.path.join(root, '*', 'submission.csv')):
        for n, s in sides(p).items():
            best[n] = min(best.get(n, 9e9), s)
    new = sides(merged)
    keep = {n for n in new if n in best and best[n] - new[n] > 1e-9}
    with open(merged) as f, open(out, 'w', newline='') as g:
        r = csv.reader(f); w = csv.writer(g); w.writerow(next(r)); k = 0
        for row in r:
            if int(row[1]) in keep:
                row[0] = k; w.writerow(row); k += 1
    print(f'{len(keep)} improved puzzles kept: {sorted(keep)}')


if __name__ == '__main__':
    main()
