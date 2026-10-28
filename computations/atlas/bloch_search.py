"""8-cubed grid plus continuous Bloch refinement, unchanged element model."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.check_common import *
from scipy.optimize import minimize
from computations.common.bloch_frame import representatives
import time
EXCLUDE = 0.025 / 4
CENSOR = 0.027 / 4

def scan(f, reference):
    start = time.time()
    f.prestress('stress')
    from computations.common.fast_bloch import install
    assembly_check = install(f)
    points = representatives(f, 8)
    grid = sorted([(float(f.eigen(k, 1)[0]), np.asarray(k)) for k in points], key=lambda x: x[0])
    scale = grid[0][0]
    trials = [(v, k, 'grid') for v, k in grid]
    seeds = [np.asarray(reference['micro']['k'])]
    seeds += [k for v, k in grid if la.norm(k - np.rint(k)) > 1e-09][:6]
    n = np.asarray(reference['LE']['direction'])
    k = f.A @ n
    k = k / la.norm(k) * 0.012
    seeds.append(k)
    opts = []

    def objective(k):
        q = np.asarray(k) - np.rint(k)
        if la.norm(q) < EXCLUDE:
            return 1000.0 + (EXCLUDE - la.norm(q)) * 1000.0
        return float(f.eigen(q, 1)[0]) / scale
    for seed in seeds:
        if la.norm(seed - np.rint(seed)) < EXCLUDE:
            continue
        initial = float(f.eigen(seed, 1)[0])
        trials.append((initial, seed, 'seed'))
        opt = minimize(objective, seed, method='Nelder-Mead', options={'maxiter': 220, 'xatol': 5e-07, 'fatol': 2e-10})
        q = opt.x - np.rint(opt.x)
        val = float(f.eigen(q, 1)[0])
        opts.append(dict(seed=seed, k=q, load=val, nit=opt.nit, success=bool(opt.success), origin_distance=float(la.norm(q))))
        if la.norm(q) >= EXCLUDE:
            trials.append((val, q, 'refined'))
    best = min(trials, key=lambda x: x[0])
    q = best[1] - np.rint(best[1])
    near = 0 < la.norm(q) < CENSOR
    mu = dict(load=best[0], k=q.tolist(), grid=8, points=len(grid), near_excluded_origin=bool(near), gamma_optical=float(f.eigen([0, 0, 0], 1)[0]))
    computed_class_name = classify(reference['PD_pressure_pV'], reference['LE'], mu)
    return dict(name=reference['name'], r=reference['r_over_each_parent_length'], baseline_class=reference['class_pressure_pV'], refined_class=computed_class_name, baseline_micro=reference['micro'], micro=mu, lambda_mu=None if near else best[0] * reference['p0'], sampled_micro_upper_pressure=best[0] * reference['p0'], lambda_PD=reference['lambda_PD_pressure'], lambda_LE=reference['lambda_LE_pressure'], micro_relative_change=best[0] / reference['micro']['load'] - 1, minimum_root_over_PD=None if reference['PD_pressure_pV']['load'] is None else best[0] / reference['PD_pressure_pV']['load'], grid_samples=[dict(k=k, load=v) for v, k in grid], refinements=opts, preload_pressure_relative_change=float(-np.trace(f.pre['stress']) / 3 / reference['p0'] - 1), graph_sha256=graph(reference['name'])['graph_sha256'], cached_assembly_root_relative_errors=assembly_check, seconds=time.time() - start)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--chunks', type=int, default=1)
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--nets', nargs='+')
    args = ap.parse_args()
    rows = atlas_rows()
    pd = sorted({x['name'] for x in rows if x['class_pressure_pV'].startswith('PD-only')})
    eligible = sorted(set.intersection(*[{x['name'] for x in rows if x['r_over_each_parent_length'] == r and x['class_pressure_pV'] == 'PD+LE'} for r in RADII]))
    controls = sorted(np.random.default_rng(120026).choice(eligible, 40, replace=False).tolist())
    if args.nets:
        unknown = set(args.nets) - set(pd + controls)
        if unknown:
            ap.error('Not in the frozen check cohort: ' + ', '.join(sorted(unknown)))
        pd = [n for n in pd if n in args.nets]
        controls = [n for n in controls if n in args.nets]
    jobs = [(n, r, 'PD-only union' if n in pd else 'PD+LE control') for n in pd + controls for r in RADII]
    results = []
    start = time.time()
    for idx, (name, r, group) in enumerate(jobs):
        if not args.aggregate and idx % args.chunks != args.chunk:
            continue
        target = DEST / 'bloch' / f'atlas/check_{name}_{r:.2f}.json'
        if target.exists():
            result = read(target)
        else:
            assert not args.aggregate, str(target)
            result = scan(frame(name, r), raw(name, r))
            result['group'] = group
            save(target, result)
        if args.aggregate:
            checked = DEST / 'threshold_checks' / f'atlas/check_{name}_{r:.2f}.json'
            if checked.exists():
                diagnostic = read(checked)
                if diagnostic['relative_LE_change'] < 0:
                    result = dict(result, grid_class_before_acoustic_check=result['refined_class'], refined_class=diagnostic['refined_class'], refined_LE=diagnostic['refined_LE'], lambda_LE=diagnostic['refined_lambda_LE'], acoustic_check=diagnostic)
        results.append(result)
        print('bloch', idx + 1, len(jobs), name, r, result['baseline_class'], result['refined_class'], round(result['seconds'], 2), flush=True)
    if args.chunks > 1 and (not args.aggregate):
        return
    out = dict(model=raw('cco', 0.02)['model'], grid=8, full_grid_points=512, exclude_radius=EXCLUDE, censor_radius=CENSOR, wavevector_units='fractional reciprocal coordinates, Euclidean distance after nearest-integer wrapping', pdonly_union=pd, control_seed=120026, control_eligible=eligible, controls=controls, n_rows=len(results), rows=results, method='All symmetry/time-reversal representatives of 8x8x8 grid including optical Gamma; generalized first positive instability root, followed by Nelder-Mead from six lowest non-Gamma grid points, baseline minimum and small-k LE direction. Baseline and grid seeds are included. Same raw-root classifier and quarter-sized near-Gamma censor. Near-Gamma minima >0.1% below baseline LE trigger exact acoustic-limit checks; the lower refined LE determines the acoustic threshold.', class_changes=[x for x in results if x['baseline_class'] != x['refined_class']], acoustic_material_changes=[dict(name=x['name'], r=x['r'], relative_LE_change=x['acoustic_check']['relative_LE_change'], baseline_lambda_LE=x['acoustic_check']['baseline_lambda_LE'], refined_lambda_LE=x['lambda_LE']) for x in results if 'acoustic_check' in x and x['acoustic_check']['relative_LE_change'] < -1e-06], summary={str(r): dict(pdonly_baseline=sum((x['r'] == r and x['baseline_class'].startswith('PD-only') for x in results)), pdonly_stay=sum((x['r'] == r and x['baseline_class'].startswith('PD-only') and x['refined_class'].startswith('PD-only') for x in results)), control_stay=sum((x['r'] == r and x['group'] == 'PD+LE control' and (x['refined_class'] == 'PD+LE') for x in results))) for r in RADII}, certification='Numerical grid/local convergence check, not interval-certified global minimization.', threadpools=POOLS, compute_seconds_sum=sum((x['seconds'] for x in results)), aggregation_seconds=time.time() - start)
    publish('atlas/bloch_checks.json', out)
    print('DONE', out['summary'], flush=True)
if __name__ == '__main__':
    main()
