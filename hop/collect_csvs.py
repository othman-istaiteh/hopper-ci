# Collect competition-format packings from a downloaded folder (read only, nothing is executed): every *.csv and
# *.csv.gz whose header has n, x, y, angle and square_id, also inside *.zip archives (read in memory, not extracted).
# Small per-puzzle files and big submissions are handled alike: each complete packing (exactly n rows for its n) goes
# into the first bucket that does not hold that n yet, and each bucket is written as OUT_ROOT/<PREFIX>_b<k>/submission.csv
# (header id,n,square_id,x,y,angle). prefilter.py then keeps only the puzzles that beat ours.
# Usage: python3 -I collect_csvs.py SRC_DIR OUT_ROOT PREFIX
import csv, gzip, io, os, sys, zipfile

MAX_BYTES = 400 * 1024 * 1024
NEED = ('n', 'x', 'y', 'angle')


def packings(text_lines):
    r = csv.reader(text_lines)
    try:
        head = [h.strip().lower() for h in next(r)]
    except StopIteration:
        return {}
    if 'square_id' not in head or any(k not in head for k in NEED):
        return {}
    ix = {k: head.index(k) for k in NEED}
    by_n = {}
    for row in r:
        try:
            n = int(float(row[ix['n']]))
            by_n.setdefault(n, []).append((row[ix['x']], row[ix['y']], row[ix['angle']]))
        except (ValueError, IndexError):
            continue
    return {n: rows for n, rows in by_n.items() if 1 <= n <= 1000 and len(rows) == n}


def sources(src):
    for root, _, files in os.walk(src):
        for name in files:
            p = os.path.join(root, name)
            low = name.lower()
            try:
                if os.path.getsize(p) > MAX_BYTES:
                    continue
                if low.endswith('.csv'):
                    with open(p, newline='', errors='replace') as f:
                        yield p, packings(f)
                elif low.endswith('.csv.gz'):
                    with gzip.open(p, 'rt', newline='', errors='replace') as f:
                        yield p, packings(f)
                elif low.endswith('.zip'):
                    with zipfile.ZipFile(p) as z:
                        for info in z.infolist():
                            m = info.filename.lower()
                            if info.file_size > MAX_BYTES or not (m.endswith('.csv') or m.endswith('.csv.gz')):
                                continue
                            raw = z.read(info)
                            if m.endswith('.gz'):
                                raw = gzip.decompress(raw)
                            yield p + ':' + info.filename, packings(io.StringIO(raw.decode('utf-8', 'replace')))
            except Exception as e:
                print(f'  skip {p}: {e!r:.80}')


def main():
    src, out_root, prefix = sys.argv[1:4]
    buckets = []
    nfiles = 0
    for _, packs in sources(src):
        if not packs:
            continue
        nfiles += 1
        for n, rows in packs.items():
            for b in buckets:
                if n not in b:
                    b[n] = rows
                    break
            else:
                buckets.append({n: rows})
    for k, b in enumerate(buckets, 1):
        d = os.path.join(out_root, f'{prefix}_b{k}')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'submission.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
            i = 0
            for n in sorted(b):
                for j, (x, y, a) in enumerate(b[n]):
                    w.writerow([i, n, j, x, y, a])
                    i += 1
    print(f'  {prefix}: {nfiles} files with packings, {len(buckets)} buckets, '
          f'{sum(len(b) for b in buckets)} packings')


if __name__ == '__main__':
    main()
