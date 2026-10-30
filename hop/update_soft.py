# Keep hop/soft_list.txt (the puzzles that respond to hopping) up to date after a merge: every puzzle whose certified
# side in the new merge is below the best of the seed files is a responder; new ones are appended (biggest gain first),
# known ones keep their place. Usage: python3 update_soft.py <seeds root with */submission.csv> <merged submission.csv> soft_list.txt
import csv, glob, math, os, sys
import numpy as np


def sides(path):
    S = {}
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            S.setdefault(int(row[1]), []).append((float(row[3]), float(row[4]), float(row[5])))
    out = {}
    for n, sq in S.items():
        a = np.array(sq); h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
        out[n] = max((a[:, 0] + h).max() - (a[:, 0] - h).min(), (a[:, 1] + h).max() - (a[:, 1] - h).min())
    return out


def main():
    root, merged, listfile = sys.argv[1:4]
    best = {}
    for p in glob.glob(os.path.join(root, '*', 'submission.csv')):
        tag = os.path.basename(os.path.dirname(p))
        if tag.startswith('hop-') or tag == 'kaggle_merge':  # the run's own slices are the new results, not the baseline
            continue
        for n, s in sides(p).items():
            best[n] = min(best.get(n, 9e9), s)
    new = sides(merged)
    gains = sorted(((best[n] - new[n]) / math.sqrt(n), n) for n in new if n in best and best[n] - new[n] > 1e-9)[::-1]
    known = [int(x) for x in open(listfile).read().replace(' ', '').split(',') if x] if os.path.exists(listfile) else []
    added = [n for g, n in gains if n not in known]
    out = known + added
    open(listfile, 'w').write(','.join(map(str, out)) + '\n')
    print(f'improved {len(gains)} puzzles (gain {sum(g for g, n in gains):.6f}): ' + ' '.join(f'{n}:{g * 1e5:.1f}' for g, n in gains[:25]))
    print(f'soft list: {len(known)} known + {len(added)} new = {len(out)}; new: {added}')


if __name__ == '__main__':
    main()
