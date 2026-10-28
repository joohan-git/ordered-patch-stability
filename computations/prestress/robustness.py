"""Independent backgrounds, full-wrench transfer and compact-form errors."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.correlation_runtime import *
import argparse

def metrics(x, ref, s, denom):
    delta = x - s * ref
    return dict(max_common=float(np.max(np.abs(delta)) / denom), rms_common=float(np.sqrt(np.mean(delta ** 2)) / denom), relative_l2=float(la.norm(delta.ravel()) / max(la.norm(ref.ravel()), 1e-30)))

def run_case(L, shape, seed, base, p_cb):
    cpu = time.process_time()
    wall = time.perf_counter()
    shape = np.array(shape)
    rng = np.random.default_rng([20261027, 3, 1, L, seed])
    pat = rng.integers(0, 2, (L, L))
    background_hash = phash(pat)
    lo = (L - shape) // 2
    hi = lo + shape
    pat[lo[0]:hi[0], lo[1]:hi[1]] = np.indices(tuple(shape)).sum(0) % 2
    f = Frame2D(pat)
    f.hydro()
    coords = np.array(np.unravel_index(f.parents // 10, tuple(f.shape))).T
    rel = (coords - lo) % 2
    mapped = np.ravel_multi_index(tuple(rel.T), (2, 2)) * 10 + f.parents % 10
    inside = np.all((coords >= lo) & (coords < hi), axis=1)
    ax = f.pre['forces'] / f.p0
    refax = base.pre['forces'][mapped] / base.p0
    ef = f.pre['full_element_forces'] / f.p0
    refef = base.pre['full_element_forces'][mapped] / base.p0
    stress = np.einsum('ei,ej->ij', ef[inside, 3:5], f.delta[inside]) / np.prod(shape)
    stress = (stress + stress.T) / 2
    s = -np.trace(stress) / 2
    dev = stress + np.eye(2) * s
    devfrac = la.norm(dev, 'fro') / (np.sqrt(2) * abs(s))
    denom = np.max(abs(base.pre['forces'] / base.p0))
    ed = np.max(abs(base.pre['full_element_forces'] / base.p0))
    dh = f.He / f.p0 - s * base.He[mapped] / base.p0
    hd = np.max(abs(base.He / base.p0))
    n = f.delta / la.norm(f.delta, axis=1)[:, None]
    t = np.c_[-n[:, 1], n[:, 0]]

    def local_wrench(z):
        return np.c_[np.sum(z[:, :2] * n, axis=1), np.sum(z[:, :2] * t, axis=1), z[:, 2], np.sum(z[:, 3:5] * n, axis=1), np.sum(z[:, 3:5] * t, axis=1), z[:, 5]]
    lw = local_wrench(ef)
    rlw = local_wrench(refef)

    def region(mask):
        return dict(members=int(mask.sum()), axial=metrics(ax[mask], refax[mask], s, denom), full_wrench=metrics(ef[mask], refef[mask], s, ed), local_components={name: metrics(lw[mask][:, inds], rlw[mask][:, inds], s, ed) for name, inds in [('axial', [0, 3]), ('shear', [1, 4]), ('moment', [2, 5])]}, tangent_coefficient_max=float(np.max(abs(dh[mask])) / hd), tangent_coefficient_rms=float(np.sqrt(np.mean(dh[mask] ** 2)) / hd))
    clo = np.array([L // 2 - 2] * 2)
    chi = clo + 4
    fixed = np.all((coords >= clo) & (coords < chi), axis=1)
    ids = compact_ids(f, clo, chi)
    K4 = f.K[ids, :][:, ids].toarray()
    D4 = f.assemble(dh)[ids, :][:, ids].toarray()
    eig = la.eigvalsh((D4 + D4.T) / 2, K4, check_finite=False)
    operator = dict(dofs=len(ids), H_mismatch_eigen_min_per_pressure=float(eig[0]), H_mismatch_eigen_max_per_pressure=float(eig[-1]), Q_mismatch_eigen_min_per_pressure=float(-eig[-1]), Q_mismatch_eigen_max_per_pressure=float(-eig[0]), at_checkerboard_pressure=float(p_cb * max(abs(eig[0]), abs(eig[-1]))), definition='sup |v*(Qomega(p_cb)-QP(s_R*p_cb))v|/(v*Kv), fields strictly inside fixed central 4x4 window')
    w = max(2, int(np.ceil(min(shape) / 4)))
    grow = np.all((coords >= lo + w) & (coords < hi - w), axis=1)
    w2 = np.all((coords >= lo + 2) & (coords < hi - 2), axis=1)
    ids = compact_ids(f, lo, hi)
    K = f.K[ids, :][:, ids].tocsc()
    H = f.H[ids, :][:, ids].tocsc() / f.p0
    vals, vec, res = eigpair(K, H, 1)
    p = float(vals[0])
    v = np.zeros(f.ndof)
    v[ids] = vec[:, 0]
    ve = v[f.dofs]
    elastic = np.einsum('ei,eij,ej->', ve, f.Ke, ve)
    signed = -p * np.einsum('ei,eij,ej->e', ve, dh, ve) / elastic
    active = np.any(ve != 0, axis=1)
    row = dict(L=L, patch_shape=shape, seed=seed, seed_sequence=[20261027, 3, 1, L, seed], background_sha256=background_hash, pattern_sha256=phash(pat), patch_lo=lo, patch_hi=hi, applied_pressure_scale=f.p0, mean_patch_stress_per_applied_pressure=stress, s_R=float(s), deviatoric_fraction=float(devfrac), fixed_window=dict(side=4, **region(fixed), operator=operator), growing_interior=dict(width=w, shape=shape - 2 * w, **region(grow)), fixed_width_interior=dict(width=2, **region(w2)), mode=dict(compact_pressure=p, compact_over_checkerboard=p / p_cb, e_signed=float(signed.sum()), absolute_element_sum=float(abs(signed).sum()), equation_check=float((elastic - p * float(v @ (f.H @ v)) / f.p0) / elastic), eigen_residual=res, dofs=len(ids), geometry_matching_max=float(np.max(abs(f.Ke[active] - base.Ke[mapped[active]])))), preload_residual=f.pre['residual'], force_normalizer=float(denom), wrench_normalizer=float(ed), tangent_normalizer=float(hd), cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
    del f
    gc.collect()
    return row

def aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['L'], *r['patch_shape']), []).append(r)
    out = []
    for (L, a, b), z in sorted(groups.items()):
        stress = np.asarray([r['mean_patch_stress_per_applied_pressure'] for r in z])
        out.append(dict(L=L, patch_shape=[a, b], n=len(z), s_R=summary([r['s_R'] for r in z]), patch_stress=dict(mean=stress.mean(0), sd=stress.std(0, ddof=1) if len(z) > 1 else np.zeros((2, 2)), min=stress.min(0), max=stress.max(0)), deviatoric_fraction=summary([r['deviatoric_fraction'] for r in z]), fixed_axial=summary([r['fixed_window']['axial']['max_common'] for r in z]), fixed_wrench=summary([r['fixed_window']['full_wrench']['max_common'] for r in z]), fixed_local_components={name: {metric: summary([r['fixed_window']['local_components'][name][metric] for r in z]) for metric in ('max_common', 'rms_common', 'relative_l2')} for name in ('axial', 'shear', 'moment')}, fixed_tangent=summary([r['fixed_window']['tangent_coefficient_max'] for r in z]), fixed_tangent_rms=summary([r['fixed_window']['tangent_coefficient_rms'] for r in z]), fixed_operator=summary([r['fixed_window']['operator']['at_checkerboard_pressure'] for r in z]), growing_wrench=summary([r['growing_interior']['full_wrench']['max_common'] for r in z]), growing_axial=summary([r['growing_interior']['axial']['max_common'] for r in z]), growing_tangent=summary([r['growing_interior']['tangent_coefficient_max'] for r in z]), fixed_width_wrench=summary([r['fixed_width_interior']['full_wrench']['max_common'] for r in z]), signed_mode_correction=summary([r['mode']['e_signed'] for r in z])))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    job = Job('prestress/robustness.json')
    out = job.out
    for r in out['runs']:
        op = r['fixed_window']['operator']
        if 'eigen_min_per_pressure' in op:
            low = op.pop('eigen_min_per_pressure')
            high = op.pop('eigen_max_per_pressure')
            op.update(H_mismatch_eigen_min_per_pressure=low, H_mismatch_eigen_max_per_pressure=high, Q_mismatch_eigen_min_per_pressure=-high, Q_mismatch_eigen_max_per_pressure=-low)
    base = Frame2D(np.indices((2, 2)).sum(0) % 2)
    base.hydro()
    p_cb = float(base.eigen(count=1)[0][0] * base.p0)
    out['checkerboard_pressure'] = p_cb
    out['definitions'] = dict(background='iid binary, independent across (L,seed); same realization across R at fixed L/seed', s_R='mean owner-tile patch pressure divided by applied pressure; finite-patch diagnostic', full_wrench='two end forces and moments; local axial/shear/moment columns share the maximum reference full-wrench normalization', tangent='full geometric element tangent per applied pressure, normalized by max reference coefficient', growing_interior='exclude ceil(min(patch_shape)/4) owner-tile layers', limitation='Finite periodic boxes and finite patches; no proof of H2 or H2_infinity.')
    common_examples = [(L, (R, R), s) for L in (64, 96) for s in range(8) for R in (16, 32, 48)]
    common_examples += [(128, (48, 48), s) for s in range(8)] + [(96, (32, 16), s) for s in range(8)]
    done = {(r['L'], tuple(r['patch_shape']), r['seed']) for r in out['runs']}
    n = 0
    for L, shape, seed in common_examples:
        if (L, shape, seed) in done:
            continue
        if not job.limit_ok():
            return
        print('start', L, shape, seed, flush=True)
        row = run_case(L, shape, seed, base, p_cb)
        out['runs'].append(row)
        out['summary'] = aggregate(out['runs'])
        job.checkpoint()
        print('done', len(out['runs']), 'cpu', round(row['cpu_seconds'], 2), 'fixed', row['fixed_window']['full_wrench']['max_common'], flush=True)
        n += 1
        if args.limit and n >= args.limit:
            return
    out['summary'] = aggregate(out['runs'])
    job.finish()
if __name__ == '__main__':
    main()
