from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.trumof.mechanics.verify import *
from scipy.optimize import minimize

def compact(selected_R=None):
    out = []
    start = time.perf_counter()
    cpu = time.process_time()
    rr = next((r for r in CONVENTIONS_SEED['rows'] if r['name'] == 'trumof_1x1x1_1' and r['parameters'] == 'soft'))
    e = Engine(rr)
    a = APP[rr['name']]
    for reference in SUMMARY_SEED['compact_modes']:
        if selected_R is not None and reference['R_cells'] != selected_R:
            continue
        R = reference['R_cells']
        L = R + 2
        pos = lattice(L, RULE)[0]
        m = CG(pos * RULE['a'], np.eye(3) * L * RULE['a'], graph(np.tile(a['q'], L ** 3), RULE, L), e.m.kr, e.m.kt)
        support = np.all((pos >= 1) & (pos < 1 + R), axis=1)
        ids = (3 * np.flatnonzero(support)[:, None] + np.arange(3)).ravel()
        sp = Spectrum(m)
        B, AB, _ = sp.matrices(np.zeros(3))
        Bs = B[:, ids]
        As = AB[:, ids]

        def calc(p, full=False):
            xu, info = e.equilibrium(p, None)
            assert info['converged']
            xx = np.r_[np.tile(xu[:-6], L ** 3), xu[-6:]]
            sp.set_state(xx)
            H = (Bs.T @ sp.HB @ Bs + As.T @ sp.HA @ As).real.tocsr()
            ev, U = eigsh(H, k=2, which='SA', tol=1e-10, ncv=50, v0=np.random.default_rng(42).normal(size=len(ids)))
            ix = np.argsort(ev)
            val = float(ev[ix[0]])
            u = U[:, ix[0]]
            residual = float(np.linalg.norm(H @ u - val * u))
            assert residual < 1e-06
            return (val, u, xx, H, residual, ev[ix]) if full else val
        p0 = calc(0)
        pc = brentq(calc, reference['critical_pressure_GPa'] - 0.01, reference['critical_pressure_GPa'] + 0.01, xtol=2e-10)
        ev, u, xx, H, res, vals = calc(pc, True)
        full = np.zeros(3 * m.N)
        full[ids] = u
        nodal = np.sum(full.reshape(-1, 3) ** 2, axis=1)
        weight = np.prod(np.where((pos >= 1) & (pos <= 1 + R), np.sin(np.pi * np.clip((pos - 1) / R, 0, 1)), 0), axis=1)
        trial = np.zeros((m.N, 3))
        trial[:, 0] = np.where(m.cc == 0, 1.0, -1.0) * weight
        trial = trial.ravel()
        sp.set_state(np.zeros_like(xx))
        dw = B @ trial
        da = AB @ trial
        ray = float(np.real(np.vdot(dw, sp.HB @ dw) + np.vdot(da, sp.HA @ da)) / np.vdot(trial, trial))
        row = dict(reference)
        row.update(p0_min_eigen=p0, window_relative_translation_p0_Rayleigh=ray, critical_pressure_GPa=pc, bracket=[pc - 1e-06, pc + 1e-06], critical_eigen=ev, participation_fraction=float(sum(nodal) ** 2 / (len(nodal) * sum(nodal ** 2))))
        out.append(row)
        np.savez_compressed(OUT / 'states' / f'compact_R{R}.npz', u=full, positions=m.pos, pressure=pc, cell=m.A, support=support)
        print('compact', R, pc, 'two minima', vals, 'res', res, flush=True)
    if not out:
        raise ValueError('No matching compact domain')
    dump(OUT / ('compact_bound.json' if selected_R is None else f'compact_R{selected_R}.json'), dict(rows=out, seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu, participation_note='Individual degenerate eigenvector participation is basis-dependent; compact-mode participation depends on the selected degenerate eigenvector.'))

def extra(shard, name=None, kind='soft'):
    rows = [r for r in CONVENTIONS_SEED['rows'] if r['group'] == 'approximant' and r['L'] == 2]
    rows = [r for r in rows if r['name'] == name and r['parameters'] == kind] if name else rows[shard::2]
    if not rows:
        raise ValueError('No matching two-cell approximant')
    for row in rows:
        stem = row['name'] + '_' + row['parameters']
        dest = OUT / 'results' / f'{stem}.extra.json'
        if dest.exists():
            continue
        start = time.perf_counter()
        cpu = time.process_time()
        e = Engine(row)
        m = e.m
        st = np.load(OUT / 'states' / f'{stem}.npz')
        p = float(st['p_PD'])
        x = st['x_PD']
        e.cache.extend([(p, x.copy()), (float(st['p_published']), st['x_published'].copy())])

        def value(k):
            k = np.asarray(k)
            k = k - np.round(k)
            if np.linalg.norm(k) < 0.0001:
                k = np.array([0.0001, 0, 0]) if np.linalg.norm(k) < 1e-12 else k * 0.0001 / np.linalg.norm(k)
            H = m.bloch(x, k).toarray()
            ev = linalg.eigh(H, subset_by_index=[0, 0], eigvals_only=True)[0]
            return float(ev / sum(np.sin(np.pi * k) ** 2))
        ks = gridks(4)
        vals = np.array([value(k) for k in ks])
        starts = np.argsort(vals)[:3]
        best = (float(vals[starts[0]]), ks[starts[0]])
        for j in starts:
            opt = minimize(value, ks[j], method='L-BFGS-B', bounds=[(-0.5, 0.5)] * 3, options={'maxiter': 45, 'ftol': 1e-10, 'gtol': 1e-06, 'finite_diff_rel_step': 1e-05})
            if opt.fun < best[0]:
                best = (float(opt.fun), opt.x)
        ref = dict(name=stem, p_GPa=p, min_normalized_Bloch_eigen=best[0], k=best[1], normalization='lambda_min / sum(sin(pi*k)^2), avoids trivial acoustic origin', search='4-grid, 3 local refinements; numerical search, not interval certification')
        reference = next((z for z in SUMMARY_SEED['mechanics'] if z['name'] == row['name'] and z['parameters'] == row['parameters']))
        le0 = reference['source_Born_LE_GPa']
        le = None
        if le0 is not None:

            def fun(pp):
                return le_min(m.tangent(e.load(pp), pp)[0] + pp * SOURCE_DELTA)[0]
            delta = 0.001
            lo = le0 - delta
            hi = le0 + delta
            while fun(lo) * fun(hi) > 0:
                delta *= 2
                lo = le0 - delta
                hi = le0 + delta
                assert hi < p
            le = brentq(fun, lo, hi, xtol=2e-09)
        first = read(OUT / 'results' / f'{stem}.json')['objective']['first_pressure_GPa']
        vals_at_first = value(best[1]) if abs(first - p) < 1e-08 else e.sp.eigen(e.load(first), best[1])
        dump(dest, dict(bloch_refinement=ref, source_Born_LE_GPa=le, best_k_eigen_at_first=vals_at_first, seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu))
        print(stem, 'refined', best, 'at first', vals_at_first, flush=True)

def pcu():
    out = {}
    for kind in ['soft', 'hard']:
        par = PAR[kind]
        m = CG([[0, 0, 0]], np.eye(3) * 13.0, [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], par['C11_GPa'] * 13, par['C44_GPa'] * 13 ** 3 / 4)

        def shear(p, source=False):
            x, info = m.relax(p)
            assert info['converged']
            return m.tangent(x, p, source=source)[0][3, 3]
        root1 = brentq(lambda p: shear(p), 0.1, 5.0, xtol=1e-12)
        root2 = brentq(lambda p: shear(p, True), 0.1, 2.0, xtol=1e-12)
        out[kind] = dict(enthalpy_shear_root_GPa=root1, source_shear_root_GPa=root2, checks=[])
        for p in [0, 0.1, 0.3]:
            x, info = m.relax(p)
            F = m.configuration(x)[0]
            C = m.tangent(x, p)[0]
            B = C + p * SOURCE_DELTA
            out[kind]['checks'].append(dict(p=p, C44=C[3, 3], B44=B[3, 3], objective_formula_error=C[3, 3] - (par['C44_GPa'] / F[0, 0] ** 3 - p), published_formula_error=B[3, 3] - (par['C44_GPa'] / F[0, 0] ** 3 - 2.5 * p)))
    dump(OUT / 'pcu_check.json', out)
    print(out, flush=True)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['compact', 'pcu', 'extra'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--name')
    ap.add_argument('--kind', default='soft')
    ap.add_argument('--R', type=int)
    args = ap.parse_args()
    if args.mode == 'compact':
        compact(args.R)
    elif args.mode == 'pcu':
        pcu()
    else:
        extra(args.shard, args.name, args.kind)
