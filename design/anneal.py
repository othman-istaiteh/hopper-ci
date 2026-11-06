# Perturb-and-shrink annealer for unit-square packings (6 Nov 2026, own code; JAX + scipy L-BFGS-B, CPU).
# The idea follows pjmathematician's forum answer (755628) and the AlphaSquare write-up: start from a rigid seed (band
# or any packing), treat the squares as ROUNDED squares (corner radius r, so they can slide past each other), minimise a
# smooth overlap penalty at a fixed box side S, shrink S while the penalty can be driven to zero, kick the packing
# (random perturbations, biggest kicks on the most crowded squares) when it cannot, and harden the squares (r -> 0) as
# the run goes on. The best feasible state at r = 0 is written in competition format; certification is done outside
# (hop/sq09_requench.py certify, or the publish job of a workflow).
#
# Penalty: for every pair, the separating-axis gap over the 4 face normals of the two inner squares (half-size 0.5 - r)
# minus 2r (rounded squares overlap when the inner squares are closer than 2r); walls likewise. E = sum of squared
# penetrations. L-BFGS-B over (x, y, theta) with jax gradients.
#
# Usage: python3 anneal.py IN.csv N OUT.csv [--minutes M] [--seed K] [--r0 0.15] [--ds 0.002] [--kicks 12]
#                         [--sigma 0.08] [--smin S] [--log LOG]
#   IN.csv: competition format rows of puzzle N (angles in radians); other puzzles in the file are ignored.
#   Prints one line per outer step and a final JSON line: {"n", "s0", "s_best", "sec", "feasible", "steps"}.
import argparse, csv, json, math, os, sys, time
import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=true')
import jax, jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update('jax_enable_x64', True)


def load(path, n):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r['n']) == n:
                rows.append((int(r['square_id']), float(r['x']), float(r['y']), float(r['angle'])))
    rows.sort()
    a = np.array([[x, y, t] for _, x, y, t in rows], dtype=float)
    if len(a) != n:
        raise SystemExit(f'{path}: {len(a)} rows for n={n}')
    return a


def bbox(a):
    h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    x0 = (a[:, 0] - h).min(); y0 = (a[:, 1] - h).min()
    return x0, y0, max((a[:, 0] + h).max() - x0, (a[:, 1] + h).max() - y0)


def make_energy(n):
    iu = jnp.triu_indices(n, 1)

    def energy(p, S, r):
        x, y, th = p[:n], p[n:2 * n], p[2 * n:]
        h = 0.5 - r
        c, s = jnp.cos(th), jnp.sin(th)
        dx = x[None, :] - x[:, None]; dy = y[None, :] - y[:, None]          # d = c_j - c_i
        dth = th[None, :] - th[:, None]
        ext = h * (jnp.abs(jnp.cos(dth)) + jnp.abs(jnp.sin(dth)))           # extent of j on an axis of i (and of i on an axis of j)
        # axes of i: u_i = (c_i, s_i), v_i = (-s_i, c_i); projections of d
        pu_i = dx * c[:, None] + dy * s[:, None]
        pv_i = -dx * s[:, None] + dy * c[:, None]
        # axes of j
        pu_j = dx * c[None, :] + dy * s[None, :]
        pv_j = -dx * s[None, :] + dy * c[None, :]
        gap = jnp.maximum(jnp.maximum(jnp.abs(pu_i), jnp.abs(pv_i)), jnp.maximum(jnp.abs(pu_j), jnp.abs(pv_j))) - h - ext
        pen = jnp.maximum(0.0, 2 * r - gap)[iu]
        e_pairs = jnp.sum(pen * pen)
        # walls: the rounded square must lie inside [0, S]^2
        ew = h * (jnp.abs(c) + jnp.abs(s)) + r
        w = jnp.maximum(0.0, ew - x) + jnp.maximum(0.0, x + ew - S) + jnp.maximum(0.0, ew - y) + jnp.maximum(0.0, y + ew - S)
        return e_pairs + jnp.sum(w * w)

    vg = jax.jit(jax.value_and_grad(energy))
    pair_pen = jax.jit(lambda p, S, r: _per_square(p, S, r, n))
    return vg, pair_pen


def _per_square(p, S, r, n):
    """penetration load per square (for the kicks): sum of pair penetrations touching it plus its wall penetration."""
    x, y, th = p[:n], p[n:2 * n], p[2 * n:]
    h = 0.5 - r
    c, s = jnp.cos(th), jnp.sin(th)
    dx = x[None, :] - x[:, None]; dy = y[None, :] - y[:, None]; dth = th[None, :] - th[:, None]
    ext = h * (jnp.abs(jnp.cos(dth)) + jnp.abs(jnp.sin(dth)))
    pu_i = dx * c[:, None] + dy * s[:, None]; pv_i = -dx * s[:, None] + dy * c[:, None]
    pu_j = dx * c[None, :] + dy * s[None, :]; pv_j = -dx * s[None, :] + dy * c[None, :]
    gap = jnp.maximum(jnp.maximum(jnp.abs(pu_i), jnp.abs(pv_i)), jnp.maximum(jnp.abs(pu_j), jnp.abs(pv_j))) - h - ext
    pen = jnp.maximum(0.0, 2 * r - gap) * (1.0 - jnp.eye(n))
    ew = h * (jnp.abs(c) + jnp.abs(s)) + r
    w = jnp.maximum(0.0, ew - x) + jnp.maximum(0.0, x + ew - S) + jnp.maximum(0.0, ew - y) + jnp.maximum(0.0, y + ew - S)
    return jnp.sum(pen, axis=1) + w


def descend(vg, p, S, r, maxiter=400, gtol=1e-12):
    f = lambda q: tuple(np.asarray(v, dtype=np.float64) for v in vg(jnp.asarray(q), S, r))
    res = minimize(f, p, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxfun=maxiter * 2, gtol=gtol, ftol=1e-16))
    return res.x, float(res.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('n', type=int); ap.add_argument('out')
    ap.add_argument('--minutes', type=float, default=20.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--r0', type=float, default=0.15, help='initial corner radius of the rounded squares')
    ap.add_argument('--ds', type=float, default=0.002, help='box shrink per accepted step (halved on failure down to ds_min before kicking)')
    ap.add_argument('--ds-min', type=float, default=2e-4)
    ap.add_argument('--kicks', type=int, default=12, help='kicks tried at a side before giving up on it')
    ap.add_argument('--sigma', type=float, default=0.08, help='kick size (positions; angles get sigma radians)')
    ap.add_argument('--smin', type=float, default=0.0, help='stop when a feasible rigid packing at or below this side is found')
    ap.add_argument('--tol', type=float, default=1e-14)
    ap.add_argument('--log', default='')
    A = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(A.seed)
    a = load(A.inp, A.n); n = A.n
    x0, y0, S0 = bbox(a)
    a[:, 0] -= x0; a[:, 1] -= y0
    p = np.concatenate([a[:, 0], a[:, 1], a[:, 2]])
    vg, loads = make_energy(n)
    logf = open(A.log, 'a') if A.log else None

    def say(*w):
        line = ' '.join(str(v) for v in w)
        print(line, flush=True)
        if logf: logf.write(line + '\n'); logf.flush()

    # stage 0: make the start feasible at its own side with rounded squares (an overlapping seed is allowed)
    r = A.r0; S = S0 * (1 + 1e-9)
    p, e = descend(vg, p, S, r, maxiter=2000)
    say(f'start n={n} S0={S0:.6f} r={r:.3f} E={e:.3e} {time.time() - t0:.0f}s')
    k_up = 0
    while e > A.tol and k_up < 40:                      # grow the box until the rounded seed is feasible
        S *= 1.002; p, e = descend(vg, p, S, r, maxiter=1000); k_up += 1
    if e > A.tol:
        say('seed cannot be made feasible even with a 8% larger box; giving up'); return
    say(f'feasible at S={S:.6f} (rounded r={r:.3f}) after {k_up} growth steps, {time.time() - t0:.0f}s')
    best_rigid = None                                   # (S, p) with r = 0 and E <= tol
    feas = (S, p.copy(), r)                             # last feasible (S, p) at radius r
    steps = 0; fails = 0; stalls = 0; cur_ds = A.ds
    deadline = t0 + A.minutes * 60
    # hardening plan: r goes from r0 to 0 over the run; at every stall we also harden a little
    while time.time() < deadline:
        steps += 1
        S_try = feas[0] - cur_ds
        p_try, e = descend(vg, feas[1], S_try, r)
        if e <= A.tol:
            feas = (S_try, p_try, r); fails = 0
            if cur_ds < A.ds: cur_ds = min(A.ds, cur_ds * 2)
            if r == 0.0 and (best_rigid is None or S_try < best_rigid[0]):
                best_rigid = (S_try, p_try.copy())
                if A.smin and S_try <= A.smin:
                    say(f'reached smin: S={S_try:.6f}'); break
            if steps % 10 == 0:
                say(f'step {steps} S={S_try:.6f} r={r:.3f} E={e:.1e} {time.time() - t0:.0f}s')
            continue
        # infeasible at S_try: first take smaller steps, then kick the last feasible state (biggest kicks where the
        # penetration load is largest)
        if cur_ds > A.ds_min:
            cur_ds = max(A.ds_min, cur_ds / 2); continue
        fails += 1
        if fails <= A.kicks:
            L = np.asarray(loads(jnp.asarray(p_try), S_try, r)); L = L / (L.max() + 1e-300)
            amp = A.sigma * (0.3 + L) * (0.5 + rng.random())
            q = feas[1].copy()
            q[:n] += rng.normal(0, 1, n) * amp; q[n:2 * n] += rng.normal(0, 1, n) * amp
            q[2 * n:] += rng.normal(0, 1, n) * amp * (0.5 if r > 0 else 0.3)
            q[:n] = np.clip(q[:n], 0.5, S_try - 0.5); q[n:2 * n] = np.clip(q[n:2 * n], 0.5, S_try - 0.5)
            q, e2 = descend(vg, q, S_try, r, maxiter=800)
            if e2 <= A.tol:
                feas = (S_try, q, r); fails = 0; cur_ds = A.ds
                if r == 0.0 and (best_rigid is None or S_try < best_rigid[0]):
                    best_rigid = (S_try, q.copy())
                say(f'kick ok at S={S_try:.6f} r={r:.3f} ({fails} fails) {time.time() - t0:.0f}s')
            continue
        # stalled at this radius: harden (r down), re-settle at the current feasible side
        stalls += 1; fails = 0; cur_ds = A.ds
        if r > 0.0:
            r_new = max(0.0, r - max(0.02, r * 0.35))
            q, e3 = descend(vg, feas[1], feas[0], r_new, maxiter=2000)
            k = 0; S_h = feas[0]
            while e3 > A.tol and k < 30:                # hardening needs a little room: grow the box slightly
                S_h += A.ds; q, e3 = descend(vg, q, S_h, r_new, maxiter=1000); k += 1
            if e3 > A.tol:
                say(f'hardening to r={r_new:.3f} failed at S={S_h:.6f}; stopping'); break
            r = r_new; feas = (S_h, q, r)
            if r == 0.0 and (best_rigid is None or S_h < best_rigid[0]):
                best_rigid = (S_h, q.copy())
            say(f'stall {stalls}: hardened to r={r:.3f}, feasible at S={S_h:.6f} {time.time() - t0:.0f}s')
        else:
            # rigid and stuck: bigger kicks from the best rigid state, a few times, then stop
            if stalls > 6:
                say('rigid and stuck after 6 stalls; stopping'); break
            A.sigma *= 1.5
            say(f'stall {stalls}: rigid, sigma -> {A.sigma:.3f} at S={feas[0]:.6f} {time.time() - t0:.0f}s')
    if best_rigid is None:
        # last resort: harden the final feasible state fully
        q, e = descend(vg, feas[1], feas[0], 0.0, maxiter=3000); S_h = feas[0]; k = 0
        while e > A.tol and k < 60:
            S_h += A.ds; q, e = descend(vg, q, S_h, 0.0, maxiter=1000); k += 1
        if e <= A.tol:
            best_rigid = (S_h, q)
    out = dict(n=n, s0=round(S0, 9), sec=round(time.time() - t0), steps=steps, feasible=best_rigid is not None)
    if best_rigid is not None:
        S_b, q = best_rigid
        out['s_best'] = round(float(S_b), 9)
        with open(A.out, 'w') as f:
            f.write('id,n,square_id,x,y,angle\n')
            for i in range(n):
                f.write('%d,%d,%d,%.15f,%.15f,%.15f\n' % (i, n, i, q[i], q[n + i], q[2 * n + i] % (math.pi / 2)))
    say('ANNEAL ' + json.dumps(out))


if __name__ == '__main__':
    main()
