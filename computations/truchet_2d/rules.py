"""Forbid checker 2x2 windows; preserve extensive entropy.
All outputs are local. Four-sublattice conditional heat-bath targets uniform
allowed configurations; finite-chain mixing is checked, not assumed proved.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.optimize import minimize
from computations.truchet_2d.frame import Frame2D, ROOT, dump, canonical, patterns
from computations.truchet_2d.periodic_search import hydro, affine_matrices
from computations.common.acoustic_atlas import first_root, restricted, sym_basis, volume_hessian

def forbidden(p):
    b = np.roll(p, -1, 0)
    c = np.roll(p, -1, 1)
    d = np.roll(b, -1, 1)
    return (p == d) & (b == c) & (p != b)

def randbelow(rng, n):
    n = int(n)
    if n <= 1:
        return 0
    bits = n.bit_length()
    words = (bits + 63) // 64
    mask = (1 << bits) - 1
    while True:
        val = 0
        for _ in range(words):
            val = val << 64 | int(rng.bit_generator.random_raw())
        val &= mask
        if val < n:
            return val

def row_gibbs(p, rng, fixed):
    """Exact uniform conditional row on a cycle using integer transfer counts.
    Row moves connect the entire no-checker finite-state space: a zero row
    can always replace a full row; hence every configuration connects to zero.
    """
    L, W = p.shape
    for i in rng.permutation(L):
        above = p[(i - 1) % L]
        below = p[(i + 1) % L]
        T = [[[1, 1], [1, 1]] for _ in range(W)]
        for j in range(W):
            for neigh in [above, below]:
                a = int(neigh[j])
                b = int(neigh[(j + 1) % W])
                if a != b:
                    T[j][b][a] = 0
            if fixed[i, j]:
                other = 1 - int(p[i, j])
                T[j][other] = [0, 0]
        suffixes = []
        counts = []
        for start in [0, 1]:
            dp = [[0, 0] for _ in range(W + 1)]
            dp[W][start] = 1
            for j in range(W - 1, -1, -1):
                dp[j] = [T[j][x][0] * dp[j + 1][0] + T[j][x][1] * dp[j + 1][1] for x in [0, 1]]
            suffixes.append(dp)
            counts.append(dp[0][start])
        assert sum(counts) > 0
        start = 0 if randbelow(rng, sum(counts)) < counts[0] else 1
        dp = suffixes[start]
        current = start
        p[i, 0] = start
        for j in range(W - 1):
            weights = [T[j][current][y] * dp[j + 1][y] for y in [0, 1]]
            current = 0 if randbelow(rng, sum(weights)) < weights[0] else 1
            p[i, j + 1] = current
    return p

def heatbath(p, rng, sweeps=200, fixed=None, record=False):
    p = p.copy()
    L = p.shape[0]
    assert L % 2 == 0 and p.shape[1] % 2 == 0
    if fixed is None:
        fixed = np.zeros(p.shape, bool)
    xx, yy = np.indices(p.shape)
    sublats = [(xx % 2 == a) & (yy % 2 == b) & ~fixed for a, b in itertools.product([0, 1], repeat=2)]
    hist = []
    accepted = 0
    proposed = 0
    for sweep in range(sweeps):
        for mask in sublats:
            bad = np.zeros(p.shape, bool)
            for dx, dy in itertools.product([-1, 1], repeat=2):
                u = np.roll(p, dx, 0)
                v = np.roll(p, dy, 1)
                w = np.roll(u, dy, 1)
                bad |= (u == p) & (v == p) & (w != p)
            prop = (rng.random(p.shape) < 0.5) & mask
            flip = prop & ~bad
            proposed += int(prop.sum())
            accepted += int(flip.sum())
            p[flip] ^= 1
        if (sweep + 1) % 20 == 0 or sweep == sweeps - 1:
            p = row_gibbs(p, rng, fixed)
        if record and sweep % 5 == 0:
            hist.append(float(p.mean()))
    assert not forbidden(p).any()
    return (p, dict(accepted=accepted, proposed=proposed, acceptance=accepted / max(1, proposed), density_trace=hist, sampler='four-sublattice heatbath plus exact conditional whole-row Gibbs'))

def scan(p, grid=4, refine=True):
    start = time.perf_counter()
    f = Frame2D(p)
    p0 = hydro(f)
    K, H = affine_matrices(f)
    HP = H.copy()
    HP[-4:, -4:] -= p0 * np.prod(p.shape) * volume_hessian(2)
    pd = first_root(*restricted(K, HP, sym_basis(2)))

    def acoustic(th):
        n = np.array([np.cos(th[0]), np.sin(th[0])])
        return first_root(*restricted(K, H, np.kron(np.eye(2), n[:, None])))
    aa = [(acoustic([th]), th) for th in np.arange(16) * np.pi / 16]
    le, th = min(aa)
    opt = minimize(acoustic, [th], method='Nelder-Mead', options={'maxiter': 60, 'xatol': 1e-05, 'fatol': 1e-10})
    le = min(le, float(opt.fun))
    rows = []
    for k in itertools.product(np.arange(grid) / grid, repeat=2):
        if tuple(k) > tuple(-np.array(k) % 1):
            continue
        rows.append((float(f.eigen(k, count=1)[0][0]), np.array(k)))
    rows.sort(key=lambda x: x[0])
    best = rows[0]
    if refine:
        for _, k in rows[:2]:
            if la.norm(k) < 1e-08:
                continue

            def obj(x):
                q = x - np.rint(x)
                if la.norm(q) < 0.01:
                    return 1000.0
                return float(f.eigen(q, count=1)[0][0]) / best[0]
            opt = minimize(obj, k, method='Nelder-Mead', options={'maxiter': 70, 'xatol': 0.0001, 'fatol': 1e-07})
            if opt.fun < 1:
                best = (float(opt.fun * best[0]), opt.x % 1)
    first = min(pd, le, best[0])
    gamma = float(f.eigen(count=1)[0][0])
    cls = 'PD-only' if pd < min(le, best[0]) * 0.997 else 'LE' if le < min(gamma, best[0]) * 0.997 else 'Gamma-optical' if la.norm(best[1]) < 1e-09 else 'finite-k'
    return dict(pattern=p.tolist(), shape=list(p.shape), allowed=not bool(forbidden(p).any()), pressure=float(first * p0), critical_load=float(first), PD=float(pd), LE=float(le), gamma=gamma, bloch_load=float(best[0]), k=best[1].tolist(), class_first=cls, pressure_per_factor=float(p0), seconds=time.perf_counter() - start)

def periodic():
    path = ROOT / 'truchet_2d/inputs/axial_periodic.json'
    out = dict(rule='no alternating checker in any periodic 2x2 window', runs=[], raw_covered=0)
    for i, (p, mult) in enumerate(patterns(3)):
        if forbidden(p).any():
            continue
        row = scan(p)
        row['orbit_size'] = mult
        out['runs'].append(row)
        out['raw_covered'] += mult
        dump(path, out)
        print('allowed_periodic', i, p.shape, row['pressure'], row['class_first'], flush=True)
    seen = {}
    raw = 0
    for bits in itertools.product([0, 1], repeat=16):
        p = np.array(bits, dtype=np.uint8).reshape(4, 4)
        if forbidden(p).any():
            continue
        raw += 1
        key = canonical(p)
        if key not in seen:
            seen[key] = p
    out['four_by_four_allowed_count'] = raw
    out['four_by_four_orbit_count'] = len(seen)
    dump(path, out)
    screen = []
    for i, p in enumerate(seen.values()):
        f = Frame2D(p)
        p0 = hydro(f)
        val = float(f.eigen(count=1)[0][0] * p0)
        screen.append(dict(pattern=p.tolist(), gamma_pressure=val))
        if (i + 1) % 50 == 0:
            print('4x4_Gamma_screen', i + 1, len(seen), min((x['gamma_pressure'] for x in screen)), flush=True)
    out['four_by_four_Gamma_screen'] = screen
    dump(path, out)
    for j, row in enumerate(sorted(screen, key=lambda x: x['gamma_pressure'])[:12]):
        r = scan(np.array(row['pattern']), grid=4)
        r['screen_rank'] = j + 1
        out['runs'].append(r)
        dump(path, out)
        print('4x4_refine', j + 1, r['pressure'], r['k'], flush=True)
    out['weakest_sampled_allowed'] = min(out['runs'], key=lambda x: x['pressure'])
    out['baseline_checker'] = scan(np.array([[0, 1], [1, 0]]))
    dump(path, out)

def stats(p, base, ensemble, L, sample):
    start = time.perf_counter()
    f = Frame2D(p)
    p0 = hydro(f)
    vals, v, res = f.eigen(count=2)
    st = f.mode_stats(v[:, 0])
    return (dict(L=L, sample=sample, ensemble=ensemble, pattern=p.tolist(), pressure=float(vals[0] * p0), ratio_baseline=float(vals[0] * p0 / base), participation_fraction=st['energy_participation_fraction'], energy_ipr=st['energy_ipr'], eigen_residual=res, equilibrium_residual=float(f.pre['residual']), seconds=time.perf_counter() - start, tile_energy=st['tile_energy'] if sample == 0 else None), f, v)

def samples(sizes=(4, 8, 16, 32, 64), count=12):
    per = json.loads((ROOT / 'truchet_2d/inputs/axial_periodic.json').read_text())
    base = per['baseline_checker']['pressure']
    path = ROOT / 'truchet_2d/inputs/axial_samples.json'
    out = json.loads(path.read_text()) if path.exists() else dict(base=base, runs=[])
    done = {(r['ensemble'], r['L'], r['sample']) for r in out['runs']}
    for L in sizes:
        rng = np.random.default_rng([2606, L])
        p = np.zeros((L, L), np.uint8)
        p, burn = heatbath(p, rng, 1200, record=True)
        for j in range(count):
            p, m = heatbath(p, rng, 150, record=True)
            for name, q in [('no_checker', p), ('IID', rng.integers(0, 2, (L, L), dtype=np.uint8))]:
                if (name, L, j) in done:
                    continue
                row, f, v = stats(q, base, name, L, j)
                row['mc'] = m if name == 'no_checker' else None
                if j == 0 and name == 'no_checker':
                    row['burnin'] = burn
                out['runs'].append(row)
                dump(path, out)
                print('sample2d', name, L, j, row['ratio_baseline'], row['participation_fraction'], flush=True)
                del f

def check():
    counts = []
    for L in [2, 3, 4]:
        hist = {}
        total = 0
        for bits in itertools.product([0, 1], repeat=L * L):
            p = np.array(bits, dtype=np.uint8).reshape(L, L)
            if not forbidden(p).any():
                total += 1
                ones = int(p.sum())
                hist[ones] = hist.get(ones, 0) + 1
        counts.append(dict(L=L, allowed=total, entropy_per_cell=float(np.log(total) / L ** 2), ones_histogram=hist))
    chains = []
    for start in ['zero', 'one', 'stripe']:
        p = np.zeros((16, 16), np.uint8) if start == 'zero' else np.ones((16, 16), np.uint8) if start == 'one' else np.indices((16, 16))[0] % 2
        p, r = heatbath(p, np.random.default_rng(600 + len(chains)), 4000, record=True)
        h = np.array(r['density_trace'])
        tail = h[len(h) // 2:]
        rho = float(np.corrcoef(tail[:-1], tail[1:])[0, 1])
        chains.append(dict(start=start, mean_density=float(tail.mean()), sd_density=float(tail.std()), lag5_sweeps_correlation=rho, **r))
    hist = np.zeros(17, int)
    p = np.zeros((4, 4), np.uint8)
    rng = np.random.default_rng(654)
    p, _ = heatbath(p, rng, 500)
    for j in range(10000):
        p, _ = heatbath(p, rng, 2)
        hist[int(p.sum())] += 1
    exact = counts[-1]
    prob = np.array([exact['ones_histogram'].get(i, 0) / exact['allowed'] for i in range(17)])
    out = dict(counts=counts, entropy_lower_bound=float(np.log(2) / 4), entropy_proof='Bits arbitrary only at even-even sites, all other sites zero; 2^(L^2/4) allowed configurations.', chains=chains, small4x4_MC=dict(samples=10000, ones_histogram=hist.tolist(), exact_probabilities=prob.tolist(), total_variation_ones_distribution=float(abs(hist / 10000 - prob).sum() / 2)), mixing_claim='Whole-row moves prove finite-volume irreducibility via all-zero rows. Finite-time and infinite-volume mixing rates are not proved.')
    dump(ROOT / 'truchet_2d/inputs/rule_patterns.json', out)

def patch_pressure(f, lo, hi):
    imposed = f.pre.get('w')
    ep = la.solve(f.C, np.array([-1.0, -1.0, 0.0]))
    ep /= -(ep[0] + ep[1]) / 2
    d = np.einsum('a,aei->ei', ep, f.affine) + f.pre['w'][f.dofs]
    ef = np.einsum('eij,ej->ei', f.Ke, d)
    vir = -np.sum(ef[:, 3:5] * f.delta, axis=1) / 2
    tile = np.bincount(f.parents // 10, weights=vir, minlength=int(np.prod(f.shape))).reshape(f.shape)
    mask = np.zeros(tuple(f.shape), bool)
    mask[lo:hi, lo:hi] = True
    return dict(s_P_far=float(tile[mask].mean() / tile[~mask].mean()), s_P_global=float(tile[mask].mean() / tile.mean()))

def patches():
    per = json.loads((ROOT / 'truchet_2d/inputs/axial_periodic.json').read_text())
    base = per['baseline_checker']['pressure']
    motif = np.array(per['weakest_sampled_allowed']['pattern'], np.uint8)
    path = ROOT / 'truchet_2d/inputs/axial_patches.json'
    out = dict(base_baseline=base, allowed_motif_reference=per['weakest_sampled_allowed'], runs=[])
    for L in [32, 64]:
        for R in [4, 8, 16, 24] + ([32, 48] if L == 64 else []):
            if R > L - 2:
                continue
            lo = (L - R) // 2
            hi = lo + R
            for name in ['allowed_domain', 'forbidden_checker_intervention']:
                p = np.zeros((L, L), np.uint8)
                inds = np.indices((R, R))
                p[lo:hi, lo:hi] = motif[inds[0] % motif.shape[0], inds[1] % motif.shape[1]] if name == 'allowed_domain' else inds.sum(axis=0) % 2
                fixed = np.zeros((L, L), bool)
                fixed[lo - 1:hi + 1, lo - 1:hi + 1] = True
                if name == 'allowed_domain':
                    p, mc = heatbath(p, np.random.default_rng(600 + L + R), 800, fixed=fixed)
                else:
                    q = p.copy()
                    q[lo:hi, lo:hi] = 0
                    q, mc = heatbath(q, np.random.default_rng(600 + L + R), 800, fixed=fixed)
                    q[lo:hi, lo:hi] = p[lo:hi, lo:hi]
                    p = q
                row, f, v = stats(p, base, name, L, 0)
                row['R'] = R
                row['inside_rule_support'] = not bool(forbidden(p).any())
                ene = np.array(f.mode_stats(v[:, 0])['tile_energy']).reshape(L, L)
                row['patch_energy_fraction'] = float(ene[lo:hi, lo:hi].sum())
                row.update(patch_pressure(f, lo, hi))
                ids = np.flatnonzero(np.repeat(np.all((f.pos > lo + 1e-09) & (f.pos < hi - 1e-09), axis=1), 3))
                K = f.K[ids, :][:, ids].tocsc()
                H = f.H[ids, :][:, ids].tocsc()
                fac = sla.splu(K)
                ev, vec = sla.eigsh(H, k=1, M=K, Minv=sla.LinearOperator(K.shape, matvec=fac.solve), which='LA', tol=1e-08)
                row['compact_ratio_baseline'] = float(-np.trace(f.pre['stress']) / 2 / ev[0] / base)
                out['runs'].append(row)
                dump(path, out)
                print('patch2d', name, L, R, row['ratio_baseline'], row['compact_ratio_baseline'], row['patch_energy_fraction'], flush=True)
                del f
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['periodic', 'samples', 'patches', 'check'])
    ap.add_argument('--sizes', nargs='+', type=int)
    ap.add_argument('--count', type=int, default=12)
    a = ap.parse_args()
    if a.action == 'periodic':
        periodic()
    elif a.action == 'samples':
        samples(a.sizes or [4, 8, 16, 32, 64], a.count)
    elif a.action == 'patches':
        patches()
    else:
        check()
