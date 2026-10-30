import json, gzip, base64, sys
m = open('sq_metric.py').read(); r9 = open('sq09_requench.py').read(); h = open('sq10_hop.py').read()
board = open('board_29oct.txt').read()
md = ("# SQ10: many-core basin hopping + re-quench + certified merge\n\n"
      "Inputs (public outputs, credited to their authors): evgendvorkin (Deep EDA & Analogies), avazaku (C++ Seam DP Certified Merge), "
      "busyaprime (Best Public Packings + FQ Quench), malyshevdanil (Living Atlas of Tight Packings; Neighbour Transfer + Squeeze), "
      "nilanjansaha123 (Certified Merge + Projection Quench + GPU Screen), alexchilton (Fault-line DP + Basin Hopping), "
      "plus our own merged / hopped files.\n\n"
      "Engines: Evan Daniel's `packer hop` (basin hopping with a descending target side) and `fq` (augmented-Lagrangian quench + LP "
      "polish), both MIT, [evand/square-packing](https://github.com/evand/square-packing), search/packer. Own code: target selection "
      "by gap to the per-puzzle leaderboard (Kaggle staff notebook by inversion), one hop process per job on every core, re-quench of "
      "each hop output, and a certified merge with `sq_metric.py` (re-implementation of the competition metric notebook, Apache 2.0).")
build = r'''import os, subprocess, time
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
t = time.time()
print('cpu_count', os.cpu_count(), flush=True)
sh('curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal')
sh('git clone -q --depth 1 --filter=blob:none --sparse https://github.com/evand/square-packing.git /kaggle/working/evand && '
   'cd /kaggle/working/evand && git sparse-checkout set --no-cone /search/packer/src/ /search/packer/Cargo.toml /search/packer/Cargo.lock')
r = sh('cd /kaggle/working/evand/search/packer && $HOME/.cargo/bin/cargo build --release --bin fq --bin packer')
print(r.stderr[-400:], f'built in {time.time() - t:.0f}s', flush=True)
for b in ('fq', 'packer'):
    print(b, 'present:', os.path.exists(f'/kaggle/working/evand/search/packer/target/release/{b}'), flush=True)
'''
run = '''import os, subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mpmath'], check=False)
env = dict(os.environ, SQ_WORKERS=str(os.cpu_count()), SQ_MAXC='%s', HOP_MIN='%s', HOP_SEEDS='%s', HOP_NTARGET='%s', HOP_BUDGET_H='%s', HOP_SKIP='%s', HOP_SEED0='%s', HOP_ARGS='%s', HOP_NLIST='%s', FQ_DEADLINE='120', SQ_BOARD='board_29oct.txt')
p = subprocess.Popen([sys.executable, '-u', 'sq10_hop.py', '/kaggle/working/submission.csv'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout: print(line, end='', flush=True)
print('exit', p.wait())
subprocess.run('rm -rf /kaggle/working/hop /kaggle/working/evand/search/packer/target/release/build /kaggle/working/evand/search/packer/target/release/deps', shell=True)'''
def cell(t, src):
    c = {'cell_type': t, 'metadata': {}, 'source': src}
    if t == 'code': c.update(execution_count=None, outputs=[])
    return c
maxc, hop_min, seeds, ntarget, budget, skip, seed0, hop_args, nlist = (sys.argv[1:10] + ['2', '20', '1', '1000', '5', '0', '100', '', ''][len(sys.argv) - 1:])[:9]
nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
      'cells': [cell('markdown', md), cell('code', build),
                cell('code', 'open("sq_metric.py", "w").write(' + repr(m) + ')'),
                cell('code', 'open("sq09_requench.py", "w").write(' + repr(r9) + ')'),
                cell('code', 'open("sq10_hop.py", "w").write(' + repr(h) + ')'),
                cell('code', 'open("board_29oct.txt", "w").write(' + repr(board) + ')'),
                cell('code', run % (maxc, hop_min, seeds, ntarget, budget, skip, seed0, hop_args, nlist))]}
s = json.dumps(nb)
b = base64.b64encode(gzip.compress(s.encode(), 9)).decode()
open('sq10.ipynb', 'w').write(s); open('sq10.b64', 'w').write(b)
print(len(s), len(b))
