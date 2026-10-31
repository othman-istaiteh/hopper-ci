import os, sys, math, time, numpy as np
sys.path.insert(0, os.environ.get('EVAND_PACKER', '/content/evand/search/packer'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hop'))
import fillfast, sq09_requench as R
rows = R.load(os.environ.get('OURS_CSV', 'ours.csv'))
for n in (176, 299, 591):
    a = R.arr(rows[n]); h = 0.5*(np.abs(np.cos(a[:,2]))+np.abs(np.sin(a[:,2]))); a = a - [(a[:,0]-h).min(), (a[:,1]-h).min(), 0]
    s = R.side_arr(a)
    th = np.mod(a[:,2], math.pi/2); tilted = np.abs(np.mod(th+math.pi/4, math.pi/2)-math.pi/4) > 0.02
    T = [(x, y, math.degrees(t) % 90) for x, y, t in a[tilted]]
    t0 = time.time(); X = fillfast.fill(s * (1 + 1e-9), T, 'exact', time_limit=60)
    print(n, 'side %.5f tilted %d axis in packing %d exact fill %d (total %d) %.1fs' % (s, len(T), (~tilted).sum(), len(X) - len(T), len(X), time.time()-t0), fillfast.LAST.get('candidates'))
