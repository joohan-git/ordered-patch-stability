"""Compute variable-cell onset eigenvectors and their mechanical observables."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.check_common import *
import computations.inclusions.soft_frame as inclusions_soft_frame
from computations.common.sparse_frames import InclusionFrame
from computations.atlas.edge_center_atlas import volume_hessian
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import time, gc
inclusions_soft_frame.graph_data = lambda name: hydrate(graph(name))

def cell_export(name):
    g = graph(name)
    reference = raw(name, 0.02)
    at = ConsistentAtlas(frame(name, 0.02))
    pd = reference['PD_pressure_pV']
    le = reference['LE']
    E = np.asarray(pd['strain_mode'])
    acoustic = None
    tested = []
    if le['load'] is not None:
        n = np.asarray(le['direction'])
        P = np.kron(np.eye(3), n[:, None])
        Q = P.T @ at.tangent(le['load'], True) @ P
        ev, U = la.eigh(Q)
        a = U[:, 0]
        strain = (np.outer(a, n) + np.outer(n, a)) / 2
        strain /= la.norm(strain)
        acoustic = dict(direction_n=n, polarization_a=a, eigenvalues=ev, null_residual=float(la.norm(Q @ a) / la.norm(at.gibbs(0))), lambda_LE=le['load'] * at.p0, compatible_strain=strain, compatible_strain_PD_residual=float(la.norm(at.gibbs(pd['load']) @ (at.S.T @ strain.ravel())) / la.norm(at.gibbs(0))))
        if acoustic['compatible_strain_PD_residual'] < 1e-08:
            E = strain
    else:
        for n in [np.eye(3)[i] for i in range(3)] + [np.asarray(le['direction'])]:
            P = np.kron(np.eye(3), n[:, None])
            tested.append(dict(n=n, eigenvalues_at_PD=la.eigvalsh(P.T @ at.tangent(pd['load'], True) @ P)))
    out = dict(name=name, model=reference['model'], graph_sha256=g['graph_sha256'], units='mean parent strut length=1, E_s=1', positions=np.asarray(g['frac']) @ np.asarray(g['A']), fractional_positions=g['frac'], edges=g['edges'], cell=g['A'], strut_radii=0.02 * np.asarray(g['lengths']), r=0.02, class_first=reference['class_pressure_pV'], lambda_PD=reference['lambda_PD_pressure'], homogeneous_critical_strain=E, reference_atlas_PD_strain=pd['strain_mode'], PD_strain_norm='Frobenius norm 1', invariant_fraction=reference['invariant_fraction'], acoustic=acoustic, acoustic_status='LE null vector' if acoustic else 'No acoustic null before optical Gamma pole; PD strain is incompatible with a rank-one field.', acoustic_checks=tested)
    return out

def solve(f, count=6):
    BH = f.cross(f.He)
    MH = np.einsum('aei,eij,bej->ab', f.affine, f.He, f.affine)
    MH -= f.p0 * f.volume * (f.S.T @ volume_hessian(3) @ f.S)
    K = sp.bmat([[f.Kf, f.BK], [f.BK.T, f.MK]], format='csc')
    H = sp.bmat([[f.Hf, BH], [BH.T, MH]], format='csc')
    cf = la.cho_factor(f.Ctotal)

    def inverse_once(rhs):
        y = f.factor.solve(rhs[:-6])
        e = la.cho_solve(cf, rhs[-6:] - f.BK.T @ y)
        return np.r_[y - f.Z @ e, e]

    def inverse(rhs):
        v = inverse_once(rhs)
        return v + inverse_once(rhs - K @ v)
    inv = sla.LinearOperator(K.shape, matvec=inverse, dtype=float)
    ev, v = sla.eigsh(H, k=count, M=K, Minv=inv, which='LA', tol=2e-11, maxiter=12000, v0=np.random.default_rng(77).normal(size=K.shape[0]))
    order = np.argsort(ev)[::-1]
    vals = 1 / ev[order]
    v = v[:, order]
    residual = [float(la.norm(K @ v[:, j] - vals[j] * H @ v[:, j]) / (la.norm(K @ v[:, j]) + abs(vals[j]) * la.norm(H @ v[:, j]))) for j in range(count)]
    return (vals, v, residual, K, H)

def run(name, L=10):
    start = time.time()
    cell = cell_export(name)
    suffix = name if L == 10 else f'{name}_L{L}'
    print('MODE setup', name, flush=True)
    f = InclusionFrame(name, L, 0.34, rin=0.02, rout=0.03, div=2)
    f.preload('pressure')
    print('MODE solve', name, f.ndof, flush=True)
    vals, v, residual, K, H = solve(f)
    deg = np.flatnonzero(vals / vals[0] - 1 < 1e-07)
    selected = v[:, 0]
    selection = 'first eigenvector'
    coefficients = None
    if cell['acoustic'] is not None and len(deg) > 1:
        n = np.asarray(cell['acoustic']['direction_n'])
        a = np.asarray(cell['acoustic']['polarization_a'])
        x = ((f.posfrac - f.center + f.L / 2) % f.L - f.L / 2) @ f.A
        profile = x @ n * np.exp(-np.sum(x * x, axis=1) / (2 * (0.55 * f.R) ** 2))
        template = np.zeros((len(x), 6))
        template[:, :3] = profile[:, None] * a
        t = np.r_[template.ravel()[f.free], np.zeros(6)]
        coefficients = v[:, deg].T @ (K @ t)
        selected = v[:, deg] @ coefficients
        selected /= np.sqrt(selected @ (K @ selected))
        selection = 'K-orthogonal projection of localized acoustic shear template into degenerate first eigenspace'
    rayleigh = float(selected @ (K @ selected) / (selected @ (H @ selected)))
    eigres = float(la.norm(K @ selected - rayleigh * (H @ selected)) / (la.norm(K @ selected) + rayleigh * la.norm(H @ selected)))
    u = np.zeros(f.ndof)
    u[f.free] = selected[:-6]
    ep = selected[-6:]
    E = np.einsum('a,aij->ij', ep, f.basis)
    st = f.stats(u, ep)
    ue = u[f.dofs] + np.einsum('a,aei->ei', ep, f.affine)
    energy = np.maximum(np.einsum('ei,eij,ej->e', ue, f.Ke, ue), 0)
    energy /= energy.sum()
    ei = f.dofs[:, 0] // 6
    ej = f.dofs[:, 6] // 6
    node_energy = np.bincount(np.r_[ei, ej], weights=np.r_[energy / 2, energy / 2], minlength=len(f.pos))
    wrap = np.floor(f.posfrac / f.L).astype(int)
    frac = f.posfrac - wrap * f.L
    pos = frac @ f.A
    shifts = f.shifts + wrap[ej] - wrap[ei]
    periodic = u.reshape(-1, 6).copy()
    periodic[:, :3] -= periodic[:, :3].mean(axis=0)
    total = periodic.copy()
    total[:, :3] += (pos - f.center @ f.A) @ E.T
    norm = float(la.norm(total[:, :3], axis=1).max())
    periodic /= norm
    total /= norm
    E /= norm
    maxidx = np.unravel_index(np.argmax(abs(total[:, :3])), total[:, :3].shape)
    sign = 1 if total[maxidx] > 0 else -1
    total *= sign
    periodic *= sign
    E *= sign
    parent_i = f.dofs[::2, 0] // 6
    parent_j = f.dofs[1::2, 6] // 6
    delta = f.delta[::2] * 2
    du = u.reshape(-1, 6)[parent_j, :3] - u.reshape(-1, 6)[parent_i, :3] + delta @ np.einsum('a,aij->ij', ep, f.basis).T
    D = delta.reshape(f.ncells, f.m0, 3)
    U = du.reshape(f.ncells, f.m0, 3)
    gradients = np.einsum('cij,cjk->cik', np.linalg.pinv(D), U).transpose(0, 2, 1)
    sv = np.linalg.svd(gradients, compute_uv=False)
    norms = np.sum(sv ** 2, axis=1)
    rank_fraction = np.divide(sv[:, 0] ** 2, norms, out=np.zeros(len(norms)), where=norms > 0)
    tile = np.asarray(st['tile_energy'])
    localrank = float(tile @ rank_fraction)
    trans = total[:, :3]
    cov = np.einsum('n,ni,nj->ij', node_energy, trans, trans)
    ce, cv = la.eigh(cov)
    pol = cv[:, -1]
    shear_alignment = None
    if cell['acoustic'] is not None:
        a = np.asarray(cell['acoustic']['polarization_a'])
        n = np.asarray(cell['acoustic']['direction_n'])
        coeff = np.einsum('i,cij,j->c', a, gradients, n)
        shear_alignment = float(np.sum(tile * coeff ** 2) / np.sum(tile * norms))
    pin = float(-np.trace(f.pre['tile_stress'][f.patch_cells].mean(axis=0)) / 3)
    s = pin / f.p0
    pressure = rayleigh * f.p0
    reference = raw(name, 0.02)
    update = DEST / 'bloch' / f'atlas/check_{name}_0.02.json'
    mu = read(update)['micro'] if update.exists() else reference['micro']
    loc = min([x for x in [reference['LE']['load'], None if mu['near_excluded_origin'] else mu['load']] if x is not None]) * reference['p0']
    metadata = dict(name=name, model=reference['model'], graph_sha256=graph(name)['graph_sha256'], L=L, R_fraction=0.34, R=f.R, rin=0.02, rout=0.03, div=2, cell=f.A * f.L, unit_cell=f.A, inclusion_center=f.center @ f.A, units='Cartesian lengths in units of mean parent strut length; E_s=1; no Angstrom conversion', n_nodes=len(pos), n_edges=len(ei), reference_joint_nodes=f.n0 * f.ncells, geometry_note='Edges are the two finite elements per parent strut; virtual midpoint nodes are retained. Edge image shifts multiply rows of cell.', displacement_note='displacements are total normalized translations and rotations at stored positions. For an image endpoint add image_shift @ cell @ macro_strain.T to its translations; rotations repeat. periodic_displacements excludes macro strain.', normalization='max node translational Euclidean norm = 1; same multiplier applied to translations, rotations and macro_strain', energy_note='Positive elastic quadratic energy u_e^T K_e u_e, split equally between each element endpoints; normalized to sum=1. Not the zero net tangent energy at onset.', critical_load_factor=rayleigh, onset_pressure=pressure, pressure_per_factor=f.p0, s_P=s, onset_pressure_ratio_to_PD=pressure / reference['lambda_PD_pressure'], rho_PD=pressure * s / reference['lambda_PD_pressure'], rho_loc=pressure * s / loc, parent_lambda_PD=reference['lambda_PD_pressure'], parent_lambda_loc=loc, macro_strain=E, acoustic=cell['acoustic'], mode_statistics=st, rank_one_diagnostics=dict(energy_weighted_cell_gradient_rank1_fraction=localrank, acoustic_rank1_gradient_projection_fraction=shear_alignment, displacement_polarization_fraction=float(ce[-1] / ce.sum()), dominant_displacement_polarization=pol, interpretation='Cell-gradient fit is descriptive; finite spherical confinement adds transverse gradients and need not give a pure rank-one field.'), eigen_residual=eigres, first_six_loads=vals, first_six_residuals=residual, degenerate_indices=deg, mode_selection=selection, projection_coefficients=coefficients, preload_residual=f.pre['residual'], threadpools=POOLS, seconds=time.time() - start)
    metadata['mode_statistics'].pop('tile_energy')
    out = dict(**metadata, positions=np.round(pos, 10), edges=np.column_stack((ei, ej, shifts)), strut_radii=np.sqrt(f.areas / np.pi), inclusion_mask=f.distance(frac) <= f.R, edge_inclusion_mask=f.element_distance <= f.R, displacements=np.round(total, 12), periodic_displacements=np.round(periodic, 12), node_energy_fraction=node_energy)
    save(DEST / f'inclusions/mode_{suffix}.json', out)
    assert (DEST / f'inclusions/mode_{suffix}.json').stat().st_size < 30000000
    if not (DEST / f'inclusions/cell_{name}.json').exists():
        save(DEST / f'inclusions/cell_{name}.json', cell)
    save(DEST / f'inclusions/mode_metrics_{suffix}.json', metadata)
    np.savez_compressed(DEST / f'inclusions/mode_raw_{suffix}.npz', eigenvalues=vals, eigenvectors=v, selected=selected)
    for filename in [f'inclusions/mode_{suffix}.json', f'inclusions/cell_{name}.json']:
        if (ROOT / filename).exists():
            continue
        with (ROOT / filename).open('xb') as h:
            h.write((DEST / filename).read_bytes())
    print('MODE DONE', name, plain(metadata), flush=True)
    del f, K, H, v
    gc.collect()
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', choices=['pcu', 'cco'])
    ap.add_argument('--L', type=int, default=10)
    args = ap.parse_args()
    for name in [args.name] if args.name else ['pcu', 'cco']:
        suffix = name if args.L == 10 else f'{name}_L{args.L}'
        if not (ROOT / f'inclusions/mode_{suffix}.json').exists():
            run(name, args.L)
