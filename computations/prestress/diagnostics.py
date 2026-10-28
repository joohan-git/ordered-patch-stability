"""Maximum / RMS / fixed-centre / edge profiles and signed mode corrections."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.diagnostics import *
import gc

def metrics(x, ref, s, denominator):
    delta = x - s * ref
    sls = float(np.sum(x * ref) / np.sum(ref * ref))
    return dict(s_used=s, max_common=float(np.max(np.abs(delta)) / denominator), rms_common=float(np.sqrt(np.mean(delta ** 2)) / denominator), relative_l2=float(la.norm(delta.ravel()) / la.norm(ref.ravel())), fit_s=sls, fit_max_common=float(np.max(np.abs(x - sls * ref)) / denominator), fit_relative_l2=float(la.norm((x - sls * ref).ravel()) / la.norm(ref.ravel())))

def run_case(d, L, R, motif, base):
    start = time.perf_counter()
    slots = 10 if d == 2 else 17
    nd = 3 if d == 2 else 6
    p = np.random.default_rng(441 + L).integers(0, 2, (L,) * d)
    lo = (L - R) // 2
    hi = lo + R
    grid = np.indices((R,) * d)
    p[(slice(lo, hi),) * d] = motif[tuple((grid[i] % motif.shape[i] for i in range(d)))]
    f = (Frame2D if d == 2 else Frame3D)(p)
    if d == 2:
        f.hydro()
    coords = np.array(np.unravel_index(f.parents // slots, tuple(f.shape))).T
    rel = (coords - lo) % np.array(motif.shape)
    mapped = np.ravel_multi_index(tuple(rel.T), motif.shape) * slots + f.parents % slots
    inside = np.all((coords >= lo) & (coords < hi), axis=1)
    ax = f.pre['forces'] / f.p0
    refax = base.pre['forces'][mapped] / base.p0
    ef = f.pre['full_element_forces'] / f.p0
    refef = base.pre['full_element_forces'][mapped] / base.p0
    vir = np.einsum('ei,ej->eij', ef[:, nd:nd + d], f.delta)
    stress = vir[inside].sum(0) / R ** d
    stress = (stress + stress.T) / 2
    s = float(-np.trace(stress) / d)
    smean = float(np.mean(ax[inside]) / np.mean(refax[inside]))
    denom = float(np.max(np.abs(base.pre['forces'] / base.p0)))
    ed = float(np.max(np.abs(base.pre['full_element_forces'] / base.p0)))

    def summarize(mask):
        if not mask.any():
            return dict(status='empty')
        return dict(status='ok', members=int(mask.sum()), axial=metrics(ax[mask], refax[mask], s, denom), axial_s1=metrics(ax[mask], refax[mask], 1.0, denom), axial_smean=metrics(ax[mask], refax[mask], smean, denom), full_wrench=metrics(ef[mask], refef[mask], s, ed))
    cores = []
    for w in (1, 2, 4):
        mask = np.all((coords >= lo + w) & (coords < hi - w), axis=1)
        cores.append(dict(w=w, core_side=max(0, R - 2 * w), **summarize(mask)))
    csize = 4 if d == 2 else 2
    clo = L // 2 - csize // 2
    fixed = np.all((coords >= clo) & (coords < clo + csize), axis=1)
    distance = np.min(np.minimum(coords - lo, hi - 1 - coords), axis=1)
    profile = [dict(distance_owner_tiles=b, **summarize(inside & (distance == b))) for b in range((R + 1) // 2)]
    keep = np.all((f.pos > lo + 1e-09) & (f.pos < hi - 1e-09), axis=1)
    ids = (nd * np.flatnonzero(keep)[:, None] + np.arange(nd)).ravel()
    K = f.K[ids, :][:, ids].tocsc()
    H = f.H[ids, :][:, ids].tocsc() / f.p0
    vals, vec, res = eigpair(K, H, 1)
    pressure = float(vals[0])
    v = np.zeros(f.ndof)
    v[ids] = vec[:, 0]
    ve = v[f.dofs]
    en = np.einsum('ei,eij,ej->e', ve, f.Ke, ve)
    elastic = float(en.sum())
    active = np.any(ve != 0, axis=1)
    geometry_error = float(np.max(np.abs(f.Ke[active] - base.Ke[mapped[active]])))
    dh = f.He / f.p0 - s * base.He[mapped] / base.p0
    signed = -pressure * np.einsum('ei,eij,ej->e', ve, dh, ve) / elastic
    e = float(signed.sum())
    href = float(np.einsum('ei,eij,ej->', ve, base.He[mapped] / base.p0, ve))
    Hp = f.assemble(base.He[mapped] / base.p0)[ids, :][:, ids].tocsc()
    refvals, _, refres = eigpair(K, Hp, 1)
    axial_e = pressure * np.einsum('e,ei,eij,ej->', ax - s * refax, ve, f.Ge, ve) / elastic
    max_bound = pressure * sum((la.norm(dh[j], 2) * float(ve[j] @ ve[j]) for j in np.flatnonzero(active))) / elastic
    owner_map = np.zeros(L ** d)
    counts = np.zeros(L ** d)
    np.maximum.at(owner_map, f.parents // slots, np.abs(ax - s * refax) / denom)
    maps = dict(tile_max_common=owner_map.reshape((L,) * d), pattern=p)
    if d == 2:
        np.savez_compressed(OUT / 'prestress' / f'h2_map_d{d}_L{L}_R{R}.npz', **maps)
    row = dict(d=d, L=L, R=R, seed=441 + L, pressure_scale=f.p0, s_pressure=s, s_mean_axial=smean, mean_patch_stress=stress, force_normalizer=denom, wrench_normalizer=ed, cores=cores, fixed_centre=dict(side=csize, lo=clo, **summarize(fixed)), profile=profile, mode=dict(pressure=pressure, reference_compact_pressure=float(refvals[0]), reference_rayleigh_same_mode=elastic / href, e_signed=e, e_axial_only=float(axial_e), absolute_element_sum=float(np.abs(signed).sum()), operator_norm_bound=max_bound, equation_check=float((elastic - pressure * float(v @ (f.H @ v)) / f.p0) / elastic), reference_Q_at_s_pressure_over_K=float(1 - pressure * s * href / elastic), geometry_matching_max=geometry_error, eigen_residual=res, reference_residual=refres, dofs=len(ids), active_members=int(active.sum()), active_force_metrics=summarize(active)), equilibrium_residual=f.pre['residual'], seconds=time.perf_counter() - start)
    del f
    gc.collect()
    return row

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    computed = 0
    name = 'prestress/diagnostics.json'
    path = OUT / name
    out = json.loads(path.read_text()) if path.exists() else dict(metadata=metadata(), runs=[])
    out['definitions'] = dict(s='mean pressure of all owner tiles in planted patch / applied pressure; finite-R estimate, not proven limit', norm='max and rms share max_e |n_P| denominator; relative_l2 matches reference RMS convention', core='complete owner tiles [lo+w,hi-w)^d as upstream prestress/convergence.py', mode='strict physical interior of whole patch; Q=K-p H/p0; e=(Qomega-QP(s*p))/K uses full end wrenches', fixed='fixed absolute central 4x4 tiles (2D), 2x2x2 cells (3D); geometry phase mapped to planted motif', limitations='one nested random background per L, periodic finite box, no infinite-volume convergence proof')
    for d in (2, 3):
        motif = np.indices((2, 2)).sum(0) % 2 if d == 2 else np.array(json.loads((DATA_ROOT / 'frame_3d/periodic.json').read_text())['weakest']['pattern'])
        base = (Frame2D if d == 2 else Frame3D)(motif)
        if d == 2:
            base.hydro()
        common_examples = [(L, R) for L in (64, 96) for R in (8, 16, 24, 32, 40, 48)] if d == 2 else [(10, R) for R in range(4, 9)]
        for L, R in common_examples:
            if any(((r['d'], r['L'], r['R']) == (d, L, R) for r in out['runs'])):
                continue
            print('start', d, L, R, flush=True)
            row = run_case(d, L, R, motif, base)
            out['runs'].append(row)
            save(name, out)
            print('done', d, L, R, 'e', row['mode']['e_signed'], 'seconds', row['seconds'], flush=True)
            computed += 1
            if args.limit and computed >= args.limit:
                return
    save(name, out)
if __name__ == '__main__':
    main()
