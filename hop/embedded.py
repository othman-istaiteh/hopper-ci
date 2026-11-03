# Packings typed into notebook code (no output file): pull the source of each harvested notebook and extract
# literals of the form  N: [[x, y, angle], ...]  (dict entries keyed by n, n triples). Writes two candidate slices per
# notebook, angles read as radians and as degrees; certification downstream keeps only the valid ones that beat ours.
# Usage: python3 -I embedded.py SRC_DIR OUT_DIR TAG       (SRC_DIR holds the pulled .ipynb / .py files; nothing is run)
import ast, csv, glob, json, math, os, re, sys

KEY = re.compile(r'["\']?(\d{1,4})["\']?\s*:\s*\[\s*\[')


def text_of(path):
    raw = open(path, encoding='utf-8', errors='replace').read()
    if path.endswith('.ipynb'):
        try:
            nb = json.loads(raw)
            return '\n'.join(''.join(c.get('source', '')) for c in nb.get('cells', []))
        except Exception:
            return raw
    return raw


def literal_at(t, i):
    """The bracketed list starting at t[i] == '[' (balanced), or None."""
    d = 0
    for j in range(i, min(len(t), i + 5_000_000)):
        if t[j] == '[':
            d += 1
        elif t[j] == ']':
            d -= 1
            if d == 0:
                return t[i:j + 1]
    return None


def extract(t):
    out = {}
    for m in KEY.finditer(t):
        n = int(m.group(1))
        if not 1 <= n <= 1000:
            continue
        lit = literal_at(t, m.start() + m.group(0).index('['))
        if not lit:
            continue
        try:
            v = ast.literal_eval(lit)
        except Exception:
            continue
        if len(v) == n and all(isinstance(r, (list, tuple)) and len(r) >= 3 for r in v):
            out[n] = [(float(r[0]), float(r[1]), float(r[2])) for r in v]
    return out


def main():
    src, out, tag = sys.argv[1:4]
    P = {}
    for p in glob.glob(os.path.join(src, '**', '*'), recursive=True):
        if p.endswith(('.ipynb', '.py', '.json', '.txt')) and os.path.getsize(p) < 200_000_000:
            P.update(extract(text_of(p)))
    if not P:
        return
    for unit, f in (('rad', 1.0), ('deg', math.pi / 180)):
        d = os.path.join(out, f'{tag}_{unit}'); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'submission.csv'), 'w', newline='') as g:
            w = csv.writer(g); w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle']); k = 0
            for n in sorted(P):
                for i, (x, y, a) in enumerate(P[n]):
                    w.writerow([k, n, i, repr(x), repr(y), repr(a * f)]); k += 1
    print(f'{tag}: embedded packings for n = {sorted(P)}')


if __name__ == '__main__':
    main()
