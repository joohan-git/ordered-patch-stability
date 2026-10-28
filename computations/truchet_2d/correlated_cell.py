"""Correlated-cell mechanics: sparse coupled internal/cell Gibbs buckling and long-chain checks."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.correlation_runtime import *
from computations.truchet_2d.ising_sampling import clusters, labels_open, squares
import argparse
D2 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

def build_coupled(f):
    volume = float(np.prod(f.shape))
    length = np.sqrt(volume)

    def blocks(em):
        cross = np.array([f.scatter(np.einsum('eij,ej->ei', em, d)) for d in f.affine]).T[f.free]
        macro = np.einsum('aei,eij,bej->ab', f.affine, em, f.affine)
        return (cross, macro)
    B, C = blocks(f.Ke)
    HB, HC = blocks(f.He)
    HC = HC - f.p0 * volume * D2
    Z = -f.factor.solve(B) / length
    S = (C + B.T @ Z * length) / length ** 2
    S = (S + S.T) / 2
    HZ = f.Hf @ Z
    HBnew = HZ + HB / length
    HS = HC / length ** 2 + Z.T @ HZ + (Z.T @ HB + HB.T @ Z) / length
    HS = (HS + HS.T) / 2
    n = f.Kf.shape[0]
    M = sp.block_diag((f.Kf, sp.csc_matrix(S)), format='csc')
    cf = la.cho_factor(S)

    def mv(v):
        return np.r_[f.Hf @ v[:n] + HBnew @ v[n:], HBnew.T @ v[:n] + HS @ v[n:]]

    def inv(v):
        return np.r_[f.factor.solve(v[:n]), la.cho_solve(cf, v[n:])]
    H = sla.LinearOperator(M.shape, matvec=mv, dtype=float)
    MI = sla.LinearOperator(M.shape, matvec=inv, dtype=float)
    return dict(M=M, H=H, MI=MI, B=B, C=C, HB=HB, HC=HC, Z=Z, S=S, HS=HS, HBnew=HBnew, n=n, length=length)

def solve_coupled(f, validate=False):
    b = build_coupled(f)
    M, H, MI, n = (b['M'], b['H'], b['MI'], b['n'])
    if M.shape[0] < 550:
        Hd = np.block([[f.Hf.toarray(), b['HBnew']], [b['HBnew'].T, b['HS']]])
        d = 1 / np.sqrt(M.diagonal())
        Md = M.toarray() * d[:, None] * d[None, :]
        ev, v = la.eigh(Hd * d[:, None] * d[None, :], Md, subset_by_index=[len(d) - 2, len(d) - 1])
        v = d[:, None] * v
    else:
        ev, v = sla.eigsh(H, k=2, M=M, Minv=MI, which='LA', tol=2e-09, maxiter=12000, v0=np.random.default_rng(932).normal(size=M.shape[0]))
    j = int(np.argmax(ev))
    load = float(1 / ev[j])
    z = v[:, j]
    w = z[:n]
    y = z[n:]
    eps = y / b['length']
    u = w + b['Z'] @ y
    kv = M @ z
    hv = H @ z
    res = la.norm(kv - load * hv) / (la.norm(kv) + abs(load) * la.norm(hv))
    ku = np.r_[f.Kf @ u + b['B'] @ eps, b['B'].T @ u + b['C'] @ eps]
    hu = np.r_[f.Hf @ u + b['HB'] @ eps, b['HB'].T @ u + b['HC'] @ eps]
    rawres = la.norm(ku - load * hu) / (la.norm(ku) + abs(load) * la.norm(hu))
    internal = float(w @ (f.Kf @ w))
    cell = float(y @ b['S'] @ y)
    full = np.zeros(f.ndof)
    full[f.free] = u
    ue = full[f.dofs] + np.einsum('a,aei->ei', eps, f.affine)
    en = np.einsum('ei,eij,ej->e', ue, f.Ke, ue)
    tile = np.bincount(f.parents // 10, weights=np.maximum(en, 0), minlength=int(np.prod(f.shape)))
    tile /= tile.sum()
    out = dict(load=load, pressure=load * f.p0, eigen_residual=float(res), physical_coordinate_residual=float(rawres), cell_elastic_participation=cell / (cell + internal), strain_kelvin=eps, energy_participation_fraction=float(1 / (np.prod(f.shape) * (tile @ tile))), participating_tiles=float(1 / (tile @ tile)), elastic_energy_consistency=float(abs(en.sum() - cell - internal) / (cell + internal)), cell_coordinate='epsilon=(exx,eyy,sqrt(2)*exy); reported strain vector has arbitrary eigenvector normalization')
    if validate:
        Kfull = np.block([[f.Kf.toarray(), b['B']], [b['B'].T, b['C']]])
        Hfull = np.block([[f.Hf.toarray(), b['HB']], [b['HB'].T, b['HC']]])
        d = 1 / np.sqrt(Kfull.diagonal())
        ef = la.eigh(Hfull * d[:, None] * d[None, :], Kfull * d[:, None] * d[None, :], subset_by_index=[len(d) - 1, len(d) - 1], eigvals_only=True)
        out['dense_direct_load'] = float(1 / ef[-1])
        out['dense_relative_error'] = abs(out['dense_direct_load'] / load - 1)
        q = np.random.default_rng(121).normal(size=len(d))
        phys = np.r_[q[:n] + b['Z'] @ q[n:], q[n:] / b['length']]
        out['congruence_K_relative_error'] = float(abs(q @ (M @ q) - phys @ Kfull @ phys) / abs(q @ (M @ q)))
        out['congruence_H_relative_error'] = float(abs(q @ (H @ q) - phys @ Hfull @ phys) / abs(q @ (H @ q)))
    return (out, tile)

def fixed_solve(f):
    vals, v, res = f.eigen(count=2)
    stats = f.mode_stats(v[:, 0])
    tile = np.array(stats.pop('tile_energy')).reshape(tuple(f.shape))
    return (dict(load=float(vals[0]), pressure=float(vals[0] * f.p0), second_pressure=float(vals[1] * f.p0), eigen_residual=res, preload_residual=f.pre['residual'], **stats), tile)

def overlaps(sig, tile):
    lab, big, size = clusters(sig)
    lab0 = labels_open(sig)
    ids, cts = np.unique(lab0, return_counts=True)
    reference = ids[np.argmax(cts)]
    ce = [float(tile[lab == b].sum()) for b in big]
    side, starts = squares(sig)
    L = len(sig)
    se = [float(tile[np.ix_((np.arange(side) + a) % L, (np.arange(side) + b) % L)].sum()) for a, b in starts]
    return dict(open_boundary_cluster=int(cts.max()), open_boundary_overlap=float(tile[lab0 == reference].sum()), largest_cluster=size, cluster_ties=len(big), mode_energy_largest_cluster=ce[0], cluster_energy_tie_range=[min(ce), max(ce)], largest_square_side=side, mode_energy_largest_square=se[0], square_energy_tie_range=[min(se), max(se)])

def evaluate(sig, p_cb, coupled=True, validate=False):
    L = len(sig)
    stagger = np.where(np.indices((L, L)).sum(0) % 2 == 0, 1, -1)
    f = Frame2D(((sig * stagger + 1) // 2).astype(int))
    f.hydro()
    fixed, te = fixed_solve(f)
    fixed['over_checkerboard'] = fixed['pressure'] / p_cb
    row = dict(fixed=fixed, overlap=overlaps(sig, te), dofs=f.ndof, pattern_sha256=phash(sig))
    if coupled:
        free, fte = solve_coupled(f, validate)
        free['over_checkerboard'] = free['pressure'] / p_cb
        row.update(free=free, free_over_fixed=free['pressure'] / fixed['pressure'], free_overlap=overlaps(sig, fte.reshape(L, L)))
        assert row['free_over_fixed'] <= 1 + 2e-06, (L, row['free_over_fixed'])
        assert free['eigen_residual'] < 2e-06, free
    assert fixed['eigen_residual'] < 2e-06, fixed
    del f
    gc.collect()
    return row

def aggregate(out):
    result = []
    for src in ('pressure convention', 'ising'):
        groups = {}
        for r in out['runs']:
            if src in r['sources']:
                groups.setdefault((r['L'], r['J']), []).append(r)
        for (L, J), z in sorted(groups.items()):
            result.append(dict(source=src, L=L, J=J, n=len(z), fixed_ratio=summary([r['fixed']['over_checkerboard'] for r in z]), free_ratio=summary([r['free']['over_checkerboard'] for r in z]), free_over_fixed=summary([r['free_over_fixed'] for r in z]), cell_participation=summary([r['free']['cell_elastic_participation'] for r in z]), fixed_PR=summary([r['fixed']['energy_participation_fraction'] for r in z]), free_PR=summary([r['free']['energy_participation_fraction'] for r in z]), open_boundary_cluster=summary([r['overlap']['open_boundary_cluster'] for r in z]), torus_cluster=summary([r['overlap']['largest_cluster'] for r in z]), open_boundary_overlap=summary([r['overlap']['open_boundary_overlap'] for r in z]), torus_overlap=summary([r['overlap']['mode_energy_largest_cluster'] for r in z])))
    out['reference_summary'] = result
    long = []
    for L in (64, 128):
        for J in (0.38, 0.42):
            for start in ('random', 'ordered'):
                for sweep in (3000, 30000):
                    z = [r for r in out.get('long_runs', []) if (r['L'], r['J'], r['start'], r['sweep']) == (L, J, start, sweep)]
                    if z:
                        long.append(dict(L=L, J=J, start=start, sweep=sweep, n=len(z), fixed_ratio=summary([r['fixed']['over_checkerboard'] for r in z]), PR=summary([r['fixed']['energy_participation_fraction'] for r in z]), overlap=summary([r['overlap']['mode_energy_largest_cluster'] for r in z])))
    out['long_summary'] = long

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    job = Job('truchet_2d/correlated_cell.json')
    out = job.out
    p_cb = read(DATA_ROOT / 'truchet_2d/planted_pressure.json')['baseline']['pressure']
    out['checkerboard_pressure'] = p_cb
    out['definitions'] = dict(loading='Initial-geometry full-wrench affine Gibbs buckling Q=K-tH+t*p0*V*D2, internal plus three symmetric cell strains', D2=D2, solver='Elastic congruence u=w-Kff^-1 B epsilon; full coupled generalized eigenproblem, no inversion at critical tangent', cell_participation='K-orthogonal relaxed homogeneous elastic energy / total elastic energy, between 0 and 1; not raw coordinate norm', scope='Periodic supercell Gamma internal and homogeneous cell strains; no full supercell Bloch search', cluster_boundary_comparison='Open-boundary and torus cluster fractions use the fixed-cell critical-mode elastic tile energies', long_endpoints='fixed-cell modes at 3000 and 30000 sweeps; arbitrary solver mode within any unresolved degeneracy')
    if not out.get('validation'):
        out['validation'] = []
        for L, kind in [(2, 'checkerboard'), (4, 'checkerboard'), (4, 'random')]:
            sig = np.ones((L, L), int) if kind == 'checkerboard' else np.random.default_rng(81).choice([-1, 1], (L, L))
            r = evaluate(sig, p_cb, True, True)
            r.update(L=L, kind=kind)
            out['validation'].append(r)
            assert r['free']['dense_relative_error'] < 1e-07
            if kind == 'checkerboard':
                assert abs(r['free']['over_checkerboard'] - 1) < 1e-06
        eps = np.array([0.3, -0.2, 0.7])
        E = np.array([[eps[0], eps[2] / np.sqrt(2)], [eps[2] / np.sqrt(2), eps[1]]])
        h = 0.001
        fd = (np.linalg.det(np.eye(2) + h * E) + np.linalg.det(np.eye(2) - h * E) - 2) / h ** 2
        out['volume_hessian_check'] = dict(analytic=float(eps @ D2 @ eps), finite_difference=float(fd), absolute_error=float(abs(fd - eps @ D2 @ eps)))
        assert out['volume_hessian_check']['absolute_error'] < 1e-08
        job.checkpoint()
        print('VALIDATION PASSED', [(r['kind'], r['L'], r['free_over_fixed']) for r in out['validation']], flush=True)
    if args.validate_only:
        return
    sampling = read(OUT / 'truchet_2d/ising_sampling.json')
    orig = sampling['references']
    orig.sort(key=lambda r: (0 if r['L'] == 64 and r['J'] in (0.38, 0.42, 0.7) else 1, r['L'], r['J'], r['seed']))
    done = {r['id'] for r in out['runs']}
    n = 0
    for o in orig:
        if o['id'] in done:
            continue
        if not job.limit_ok(reserve=180):
            return
        cpu = time.process_time()
        wall = time.perf_counter()
        print('start', o['id'], flush=True)
        sig = np.load(OUT / o['state_file'])['sigma']
        r = evaluate(sig, p_cb)
        r.update({k: o[k] for k in ('id', 'L', 'J', 'seed', 'sources', 'state_file')})
        r['source_verification'] = {}
        for source, reference in o['sources'].items():
            check = dict(threshold_ratio_error=r['fixed']['over_checkerboard'] - reference['ratio'])
            if source == 'pressure convention':
                check.update(open_boundary_cluster_error=r['overlap']['open_boundary_cluster'] - reference['largest_cluster'], open_boundary_overlap_error=r['overlap']['open_boundary_overlap'] - reference['mode_in_largest'])
            else:
                check.update(PR_error=r['fixed']['energy_participation_fraction'] - reference['participation_ratio'])
            r['source_verification'][source] = check
            assert abs(check['threshold_ratio_error']) < 1e-05, check
        r.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
        out['runs'].append(r)
        aggregate(out)
        job.checkpoint()
        print('done', len(out['runs']), 'free/fixed', r['free_over_fixed'], 'cell', r['free']['cell_elastic_participation'], 'cpu', r['cpu_seconds'], flush=True)
        n += 1
        if args.limit and n >= args.limit:
            return
    sampling = read(OUT / 'truchet_2d/ising_sampling.json')
    out.setdefault('long_runs', [])
    done = {(r['id'], r['sweep']) for r in out['long_runs']}
    for chain in sampling['runs']:
        for sweep in (30000, 3000):
            if (chain['id'], sweep) in done:
                continue
            if not job.limit_ok(reserve=180):
                return
            cpu = time.process_time()
            wall = time.perf_counter()
            if sweep == 3000 and chain['start'] == 'random':
                reference = next((r for r in out['runs'] if (r['L'], r['J'], r['seed']) == (chain['L'], chain['J'], chain['seed'])))
                r = {k: reference[k] for k in ('fixed', 'overlap', 'pattern_sha256', 'dofs')}
                r['reused_reference'] = reference['id']
            else:
                print('long start', chain['id'], sweep, flush=True)
                sig = np.load(OUT / chain['state_file'])[f'sigma_{sweep}']
                r = evaluate(sig, p_cb, coupled=False)
            r.update({k: chain[k] for k in ('id', 'L', 'J', 'seed', 'start', 'state_file')})
            r['sweep'] = sweep
            r.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
            out['long_runs'].append(r)
            aggregate(out)
            job.checkpoint()
            print('long done', len(out['long_runs']), r['fixed']['over_checkerboard'], 'cpu', r['cpu_seconds'], flush=True)
    if len(out['runs']) == 108 and len(out['long_runs']) == 96:
        job.finish()
    else:
        out['status'] = 'partial_pending_sampling_or_limit'
        job.checkpoint()
if __name__ == '__main__':
    main()
