# Build the SQ12 merge notebook: certified per-puzzle merge of the notebook inputs (public outputs, our hop notebooks, the
# Colab dataset) plus files fetched from our GitHub repo (results/*.csv.gz = Mac harvests, latest release asset =
# GitHub Actions hop run). No quench (fq not built). Usage: python3 mk_sq12.py [maxc] -> sq12.ipynb / sq12.b64
import json, gzip, base64, sys
m = open('sq_metric.py').read(); d = open('sq09_requench.py').read()
import base64
md = base64.b64decode('IyBTUTEyOiBjZXJ0aWZpZWQgcGVyLXB1enpsZSBtZXJnZSBvZiBwdWJsaWMgcGFja2luZ3MsIG91ciBob3AgY2FtcGFpZ25zIGFuZCBvdXIgR2l0SHViIHJlc3VsdHMKCklucHV0cyAocHVibGljIG91dHB1dHMsIGNyZWRpdGVkIHRvIHRoZWlyIGF1dGhvcnMpOiBldmdlbmR2b3JraW4gKERlZXAgRURBICYgQW5hbG9naWVzKSwgYXZhemFrdSAoQysrIFNlYW0gRFAgQ2VydGlmaWVkIE1lcmdlKSwgYnVzeWFwcmltZSAoQmVzdCBQdWJsaWMgUGFja2luZ3MgKyBGUSBRdWVuY2gpLCBtYWx5c2hldmRhbmlsIChMaXZpbmcgQXRsYXMgb2YgVGlnaHQgUGFja2luZ3M7IE5laWdoYm91ciBUcmFuc2ZlciArIFNxdWVlemUpLCBuaWxhbmphbnNhaGExMjMgKENlcnRpZmllZCBNZXJnZSArIFByb2plY3Rpb24gUXVlbmNoICsgR1BVIFNjcmVlbiksIGFsZXhjaGlsdG9uIChGYXVsdC1saW5lIERQICsgQmFzaW4gSG9wcGluZyksIHBsdXMgb3VyIG93biBiYXNpbi1ob3BwaW5nIGNhbXBhaWducyAoRXZhbiBEYW5pZWwncyBNSVQgYHBhY2tlciBob3BgICsgYGZxYCwgZ2l0aHViLmNvbS9ldmFuZC9zcXVhcmUtcGFja2luZykgcnVuIG9uIEthZ2dsZSBDUFUsIENvbGFiLCBhIE1hYyBhbmQgR2l0SHViIEFjdGlvbnMgKHJlcG8gZ2l0aHViLmNvbS9vdGhtYW4taXN0YWl0ZWgvc3F1YXJlcy1ob3AsIGZldGNoZWQgYmVsb3cpLgoKT3duIGNvZGU6IGBzcV9tZXRyaWMucHlgIHJlLWltcGxlbWVudHMgdGhlIGNoZWNrcyBvZiB0aGUgY29tcGV0aXRpb24gbWV0cmljIG5vdGVib29rIChBcGFjaGUgMi4wKTsgYHNxMDlfcmVxdWVuY2gucHlgIGdhdGhlcnMgdGhlIGNhbmRpZGF0ZXMgcGVyIHB1enpsZSBhbmQga2VlcHMgcGVyIHB1enpsZSB0aGUgZmlyc3QgY2FuZGlkYXRlIChieSBzaWRlKSB0aGF0IHRoZSBtZXRyaWMgY29weSBjZXJ0aWZpZXM7IGNvbnRhY3QgZmFpbHVyZXMgYXJlIHJlcGFpcmVkIGJ5IGEgdGlueSB1bmlmb3JtIHNjYWxpbmcgb2YgdGhlIGNlbnRyZXMu').decode()  # the notebook's markdown header (credits), kept encoded here
fetch = r'''import os, json, gzip, urllib.request, shutil
REPO = 'othman-istaiteh/hopper-ci'
os.makedirs('/kaggle/working/gh', exist_ok=True)
def get(url, timeout=120):
    req = urllib.request.Request(url, headers={'User-Agent': 'sq12-merge'})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()
try:
    names = [e['name'] for e in json.loads(get('https://api.github.com/repos/%s/contents/results' % REPO)) if e['name'].endswith('.csv.gz')]
except Exception as e:
    names = []; print('results listing failed:', repr(e)[:120])
for name in names:
    try:
        raw = get('https://raw.githubusercontent.com/%s/main/results/%s' % (REPO, name))
        d = '/kaggle/working/gh/' + name[:-7]; os.makedirs(d, exist_ok=True)
        open(d + '/submission.csv', 'wb').write(gzip.decompress(raw)); print('fetched', name, len(raw), flush=True)
    except Exception as e:
        print('fetch failed', name, repr(e)[:120], flush=True)
try:
    rels = json.loads(get('https://api.github.com/repos/%s/releases?per_page=8' % REPO))
except Exception as e:
    rels = []; print('release listing failed:', repr(e)[:120])
for rel in rels:
    for a in rel.get('assets', []):
        if a['name'] != 'submission.csv': continue
        try:
            raw = get(a['browser_download_url'], timeout=300)
            d = '/kaggle/working/gh/gh_' + rel['tag_name'].replace('-', '_'); os.makedirs(d, exist_ok=True)
            open(d + '/submission.csv', 'wb').write(raw); print('fetched release', rel['tag_name'], len(raw), flush=True)
        except Exception as e:
            print('release fetch failed', rel['tag_name'], repr(e)[:120], flush=True)
shutil.rmtree('/kaggle/working/inputs', ignore_errors=True); os.makedirs('/kaggle/working/inputs')
os.symlink('/kaggle/input', '/kaggle/working/inputs/kaggle'); os.symlink('/kaggle/working/gh', '/kaggle/working/inputs/gh')
print(sorted(os.listdir('/kaggle/working/gh')))
'''
run = '''import os, subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mpmath'], check=False)
env = dict(os.environ, SQ_WORKERS=str(os.cpu_count()), SQ_MAXC='%s', FQ_DEADLINE='90', SQ_QUENCH='0', SQ_IN='/kaggle/working/inputs')
p = subprocess.Popen([sys.executable, '-u', 'sq09_requench.py', '/kaggle/working/submission.csv'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout: print(line, end='', flush=True)
print('exit', p.wait())
subprocess.run('rm -rf /kaggle/working/rq /kaggle/working/inputs /kaggle/working/gh', shell=True)'''
def cell(t, src):
    c = {'cell_type': t, 'metadata': {}, 'source': src}
    if t == 'code': c.update(execution_count=None, outputs=[])
    return c
maxc = sys.argv[1] if len(sys.argv) > 1 else '3'
nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
      'cells': [cell('markdown', md), cell('code', 'open("sq_metric.py", "w").write(' + repr(m) + ')'),
                cell('code', 'open("sq09_requench.py", "w").write(' + repr(d) + ')'), cell('code', fetch), cell('code', run % maxc)]}
s = json.dumps(nb)
b = base64.b64encode(gzip.compress(s.encode(), 9)).decode()
open('sq12.ipynb', 'w').write(s); open('sq12.b64', 'w').write(b)
print(len(s), len(b))
