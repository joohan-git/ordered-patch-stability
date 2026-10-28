"""First-instability cell data, including critical-load internal relaxation."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.selection_rule.mode_common import *
from computations.atlas.edge_center_atlas import volume_hessian
from fractions import Fraction
import itertools, time

def homogeneous(name):
    f = frame(name, 0.02)
    at = ConsistentAtlas(f)
    pd = at.pd(True)
    t = pd['load']
    cell = read(ROOT / f'inclusions/cell_{name}.json')
    E = np.asarray(cell['homogeneous_critical_strain'])
    e = at.S.T @ E.ravel()
    M = at.K6 - t * at.H6
    nf = at.nf
    scale = 1 / np.sqrt(np.diag(at.K6[:nf, :nf]))
    w = -scale * la.solve(M[:nf, :nf] * scale[:, None] * scale[None, :], M[:nf, nf:] @ e * scale, assume_a='pos')
    v = np.zeros(f.ndof)
    v[f.free] = w
    err = residual(at.K6, at.H6, np.r_[w, e], t)
    schur = float(la.norm(at.gibbs(t) @ e) / (la.norm(at.gibbs(0)) * la.norm(e)))
    assert max(err, schur) < 1e-09, (name, err, schur)
    first = atlas13(name)['lambda_PD_pressure']
    info = dict(mode_type='homogeneous_PD_with_critical_load_internal_relaxation', threshold=t * at.p0, critical_load_factor=t, pressure_per_factor=at.p0, atlas_first_threshold=first, threshold_relative_change_vs_atlas=t * at.p0 / first - 1, macro_strain_unit_frobenius=E, k=[0.0, 0.0, 0.0], actual_first_instability_k=[0.0, 0.0, 0.0], actual_first_instability_threshold=t * at.p0, commensuration_relative_threshold_change=0.0, mode_selection='Cell strain: a compatible xz shear in the degenerate pcu nullspace; the unique symmetry-preserving cco null strain.', internal_relaxation='w = -(K_ww - t H_ww)^(-1) (K_we - t H_we) e at the PD threshold; all translations and rotations retained.', eigen_residual=err, schur_null_residual=schur, unnormalized_internal_translation_max=float(la.norm(v.reshape(-1, 6)[:, :3], axis=1).max()), unnormalized_internal_rotation_max=float(la.norm(v.reshape(-1, 6)[:, 3:], axis=1).max()), acoustic=cell['acoustic'], invariant_fraction=cell['invariant_fraction'])
    return (f, v.reshape(-1, 6), E, info)

def nearest_commensurate(k, A):
    q = np.asarray(k) - np.rint(k)
    for denom in itertools.product(range(1, 5), repeat=3):
        p = np.rint(q * np.asarray(denom))
        if la.norm(q - p / denom) < 1e-06:
            return (p / denom, np.asarray(denom), True)
    choices = {Fraction(n, d) for d in range(1, 5) for n in range(-d // 2, d // 2 + 1)}
    candidates = []
    for x in itertools.product(choices, repeat=3):
        p = np.asarray([float(v) for v in x])
        delta = q - p
        delta -= np.rint(delta)
        physical = la.norm(la.solve(A, delta))
        dims = np.array([v.denominator for v in x])
        candidates.append((physical, int(np.prod(dims)), tuple(dims), p))
    _, _, dims, p = min(candidates, key=lambda x: x[:3])
    return (p, np.asarray(dims), False)

def bloch(name):
    candidates = read(ROOT / 'selection_rule/candidates.json')
    selected = next((x for x in candidates['candidates'] if x['name'] == name))
    assert selected['finite_k_first']
    f = frame(name, 0.02)
    f.prestress('stress')
    q = np.asarray(selected['k'])
    k, dims, exact = nearest_commensurate(q, f.A)
    vals, V = f.eigen(k, 8, True)
    t = float(vals[0])
    deg = np.flatnonzero(abs(vals / t - 1) < 1e-09)
    T = V.reshape(-1, 6, V.shape[1])[:, :3, :].reshape(-1, V.shape[1])[:, deg]
    ev, U = la.eigh(T.conj().T @ T)
    v = V[:, deg] @ U[:, -1]
    trans = v.reshape(-1, 6)[:, :3]
    index = np.argmax(abs(trans))
    phase = np.exp(-1j * np.angle(trans.ravel()[index]))
    v *= phase
    K = f.assemble([e['K'] for e in f.elements], k)
    H = -f.assemble(f.Gm, k)
    err = residual(K, H, v, t)
    assert err < 1e-09, err
    p0 = float(-np.trace(f.pre['stress']) / 3)
    reference = float(selected['scan']['micro']['load']) * p0
    info = dict(mode_type='real_part_of_finite_k_Bloch_mode', threshold=t * p0, critical_load_factor=t, pressure_per_factor=p0, atlas_first_threshold=atlas13(name)['lambda_micro_pressure'], threshold_relative_change_vs_atlas=t * p0 / atlas13(name)['lambda_micro_pressure'] - 1, k=k, actual_first_instability_k=q, actual_first_instability_threshold=reference, commensuration_relative_threshold_change=t * p0 / reference - 1, commensurate_exact_to_search_precision=bool(exact), commensuration_note='Smallest axis-aligned integer supercell for the rational k; exact rational snap within 1e-6 reciprocal units, otherwise nearest physical reciprocal vector among component denominators <=4.', gamma_optical_threshold=selected['scan']['micro']['gamma_optical'] * p0, mode_selection='Maximum Euclidean translational norm within the first K-orthonormal eigenspace; global phase makes its largest translation component real and positive; real part tiled over the commensurate cell.', first_eigenvalues=vals, first_eigenspace_indices=deg, eigen_residual=err, macro_strain_unit_frobenius=np.zeros((3, 3)), unit_complex_displacements_real=v.reshape(-1, 6).real, unit_complex_displacements_imag=v.reshape(-1, 6).imag)
    return (f, v.reshape(-1, 6), np.zeros((3, 3)), info, dims)

def tile(f, amplitude, E, info, dims):
    dims = np.asarray(dims, int)
    cells = list(np.ndindex(tuple(dims)))
    nn = len(f.frac)
    wrap = np.floor(f.frac + 1e-10).astype(int)
    frac = f.frac - wrap
    assert np.min(frac) > -1e-09 and np.max(frac) < 1 + 1e-09
    k = np.asarray(info['k'])
    assert np.max(abs(np.exp(2j * np.pi * k * dims) - 1)) < 1e-10
    basis = amplitude * np.exp(-2j * np.pi * (wrap @ k))[:, None]
    cell = np.diag(dims) @ f.A
    center = np.ones(3) @ cell / 2
    positions = []
    modes = []
    for R in cells:
        x = (frac + R) @ f.A
        u = (np.exp(2j * np.pi * np.dot(k, R)) * basis).real
        u[:, :3] += (x - center) @ E.T
        positions.append(x)
        modes.append(u)
    pos = np.concatenate(positions)
    u = np.concatenate(modes)
    mean = u[:, :3].mean(axis=0)
    u[:, :3] -= mean
    norm = float(la.norm(u[:, :3], axis=1).max())
    assert norm > 1e-12
    u /= norm
    macro = E / norm
    edges = []
    endpoint = []
    endpoint_modes = []
    energy = []
    radii = []
    owners = []
    elastic_force = []
    stress_force = []
    total_force = np.zeros_like(u)
    length_error = []
    for cellidx, R in enumerate(cells):
        R = np.asarray(R)
        for h, (e, g) in enumerate(zip(f.elements, f.Gm)):
            shift = e['shift'] + wrap[e['j']] - wrap[e['i']]
            target = R + shift
            image = np.floor_divide(target, dims)
            target = target - image * dims
            i = cellidx * nn + e['i']
            j = np.ravel_multi_index(tuple(target), tuple(dims)) * nn + e['j']
            a = pos[i]
            b = pos[j] + image @ cell
            ui = u[i].copy()
            uj = u[j].copy()
            uj[:3] += image @ cell @ macro.T
            d = np.r_[ui, uj]
            ek = e['K'] @ d
            eg = g @ d
            force = ek + info['critical_load_factor'] * eg
            total_force[i] += force[:6]
            total_force[j] += force[6:]
            elastic_force.append(ek)
            stress_force.append(eg)
            energy.append(float(0.5 * d @ ek))
            radii.append(float(np.sqrt(f.areas[h] / np.pi)))
            edges.append([int(i), int(j), *image.tolist()])
            endpoint.append([a, b])
            endpoint_modes.append([ui, uj])
            owners.append(h)
            length_error.append(abs(la.norm(b - a) / la.norm(e['b'] - e['a']) - 1))
    energy = np.asarray(energy)
    assert energy.min() > -1e-12 * energy.max()
    energy = np.maximum(energy, 0)
    energy_max = energy / energy.max()
    endpoint = np.asarray(endpoint)
    endpoint_modes = np.asarray(endpoint_modes)
    t = info['critical_load_factor']
    ek = np.asarray(elastic_force)
    eg = np.asarray(stress_force)
    d = endpoint_modes.reshape(-1, 12)
    bulk_tangent = float(np.sum(d * (ek + t * eg)))
    pressure_tangent = float(t * info['pressure_per_factor'] * abs(la.det(cell)) * (macro.ravel() @ volume_hessian(3) @ macro.ravel()))
    force_error = float(la.norm(total_force) / (la.norm(ek) + abs(t) * la.norm(eg)))
    energy_error = abs(bulk_tangent + pressure_tangent) / (2 * energy.sum())
    assert max(force_error, energy_error) < 1e-08, (f.name, force_error, energy_error)
    ab = endpoint[:, 1] - endpoint[:, 0]
    lengths = la.norm(ab, axis=1)
    n = ab / lengths[:, None]
    ui = endpoint_modes[:, 0, :3]
    uj = endpoint_modes[:, 1, :3]
    vi = np.sum(ui * n, axis=1)[:, None] * n
    vj = np.sum(uj * n, axis=1)[:, None] * n
    ti = np.cross(endpoint_modes[:, 0, 3:], n)
    tj = np.cross(endpoint_modes[:, 1, 3:], n)
    reference = []
    displacement = []
    samples = np.linspace(0, 1, 9)
    for s in samples:
        reference.append(endpoint[:, 0] + s * ab)
        transverse = (1 - 3 * s * s + 2 * s ** 3) * (ui - vi) + (3 * s * s - 2 * s ** 3) * (uj - vj)
        slopes = lengths[:, None] * ((s - 2 * s * s + s ** 3) * ti + (-s * s + s ** 3) * tj)
        displacement.append((1 - s) * vi + s * vj + transverse + slopes)
    reference = np.asarray(reference).transpose(1, 0, 2)
    displacement = np.asarray(displacement).transpose(1, 0, 2)
    assert la.norm(displacement[:, 0] - ui) < 1e-10 and la.norm(displacement[:, -1] - uj) < 1e-10
    row = atlas13(f.name)
    nonaffine = u[:, :3] - (pos - center) @ macro.T
    nonaffine -= nonaffine.mean(axis=0)
    return dict(net=f.name, **{'class': row['class_pressure_pV']}, class_label='PD-only' if row['class_pressure_pV'].startswith('PD-only') else row['class_pressure_pV'], **info, model=raw(f.name, 0.02)['model'], r_over_parent_strut_length=0.02, elements_per_parent_strut=2, graph_sha256=graph(f.name)['graph_sha256'], atlas_row=row, atlas_sha256=sha(ATLAS_PATH), supercell=dims, unit_cell=f.A, cell=cell, box_origin=[0.0, 0.0, 0.0], supercell_note='3x3x3 pcu and 2x2x2 cco for block views; finite-k net uses the smallest commensurate rectangular supercell.', units='Cartesian lengths in units of mean parent strut length; pressure/E_s with E_s=1; k in fractional reciprocal coordinates of unit_cell.', wavevector_cartesian=2 * np.pi * la.solve(f.A, k), n_nodes=len(pos), n_elements=len(edges), positions=pos, edges=edges, strut_radius=radii, translations=u[:, :3], rotations=u[:, 3:], nonaffine_translations=nonaffine if la.norm(E) > 0 else None, macro_strain=macro, element_mode_energy=energy_max, element_elastic_energy_fraction=energy / energy.sum(), node_unit_index=np.tile(np.arange(nn), len(cells)), node_cell_index=np.repeat(np.asarray(cells), nn, axis=0), node_is_element_midpoint=np.tile(np.arange(nn) >= len(f.data['frac']), len(cells)), element_unit_index=owners, edge_endpoint_positions=endpoint, edge_endpoint_translations=endpoint_modes[:, :, :3], edge_endpoint_rotations=endpoint_modes[:, :, 3:], hermite_sample_parameters=samples, element_centerline_positions=reference, element_centerline_translations=displacement, geometry_note='Each edge is one of two FE segments per parent strut. edges=[i,j,sx,sy,sz]; draw from positions[i] to positions[j]+[sx,sy,sz]@cell. Unwrapped endpoints and Hermite centerlines are provided explicitly to avoid long wraparound chords.', displacement_note='Translations include both affine strain and relaxed internal translations. At an image endpoint add (image@cell)@macro_strain.T; rotations repeat over the supercell. Finite-k field is the real part and has zero macro_strain. Hermite centerline samples also use the recovered rotations.', normalization='max Euclidean norm of stored node translations=1, independently for each net. Same multiplier applied to rotations and macro_strain. A global rigid translation is removed; illustrative amplitude and sign are arbitrary.', energy_note='Positive elastic bending/stretching/torsion energy 0.5*u_e^T*K_e*u_e of the real mode, normalized by its maximum across elements. This is not the vanishing signed Gibbs tangent energy at onset. Colors are normalized separately per net.', checks=dict(eigen_residual=info['eigen_residual'], supercell_nodal_force_residual=force_error, gibbs_tangent_energy_over_elastic_energy=energy_error, bulk_tangent_energy=bulk_tangent, pressure_volume_tangent_energy=pressure_tangent, elastic_quadratic_form=float(2 * energy.sum()), max_node_translation=float(la.norm(u[:, :3], axis=1).max()), max_element_mode_energy=float(energy_max.max()), min_element_mode_energy=float(energy_max.min()), element_energy_fraction_sum=float((energy / energy.sum()).sum()), max_edge_length_relative_error=max(length_error), max_nonaffine_translation=float(la.norm(nonaffine, axis=1).max()), max_rotation=float(la.norm(u[:, 3:], axis=1).max()), commensurate_phase_error=float(np.max(abs(np.exp(2j * np.pi * k * dims) - 1)))), threadpools=POOLS)

def main():
    for name, dims in [('pcu', [3, 3, 3]), ('cco', [2, 2, 2])]:
        target = output_dir() / f'selection_rule/cell_{name}.json'
        if target.exists():
            continue
        f, v, E, info = homogeneous(name)
        out = tile(f, v, E, info, dims)
        save(DEST / target.name, out)
        assert (DEST / target.name).stat().st_size < 10000000
        with target.open('xb') as h:
            h.write((DEST / target.name).read_bytes())
        print(name, plain(out['checks']), flush=True)
    name = read(ROOT / 'selection_rule/candidates.json')['selected']
    target = output_dir() / f'selection_rule/cell_{name}.json'
    if not target.exists():
        f, v, E, info, dims = bloch(name)
        out = tile(f, v, E, info, dims)
        save(DEST / target.name, out)
        assert (DEST / target.name).stat().st_size < 10000000
        with target.open('xb') as h:
            h.write((DEST / target.name).read_bytes())
        print(name, plain(out['checks']), flush=True)
if __name__ == '__main__':
    main()
