"""Resolve a lower near-Gamma root against the exact acoustic limit."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.check_common import *
from scipy.optimize import minimize

def check(name, r):
    scan = read(DEST / 'bloch' / f'atlas/check_{name}_{r:.2f}.json')
    reference = raw(name, r)
    f = frame(name, r)
    at = ConsistentAtlas(f)
    k = np.asarray(scan['micro']['k'])
    seeds = [np.asarray(reference['LE']['direction'])]
    if la.norm(k) > 1e-09:
        n = la.solve(f.A, k)
        seeds.append(n / la.norm(n))
    pd = np.asarray(reference['PD_pressure_pV']['strain_mode'])
    pe, pv = la.eigh(pd)
    if pe[0] < 0 < pe[2]:
        for sign in [-1, 1]:
            n = np.sqrt(pe[2]) * pv[:, 2] + sign * np.sqrt(-pe[0]) * pv[:, 0]
            seeds.append(n / la.norm(n))
    seeded = sorted([(at.acoustic(n), n) for n in seeds], key=lambda x: x[0])
    initial, n = seeded[0]
    load = initial
    success = True
    for scale, seed in seeded[:2]:
        opt = minimize(lambda v: at.acoustic(v) / scale, seed, method='Nelder-Mead', options={'maxiter': 350, 'xatol': 1e-08, 'fatol': 1e-11})
        candidate = at.acoustic(opt.x)
        if candidate < load:
            load = candidate
            n = opt.x / la.norm(opt.x)
            success = bool(opt.success)
    censored = load >= at.gamma * (1 - 2e-06)
    eval_load = load * (1 - 1e-05) if censored else load
    P = np.kron(np.eye(3), n[:, None])
    Q = P.T @ at.tangent(eval_load, True) @ P
    ev, U = la.eigh(Q)
    le = dict(load=None if censored else float(load), bound=float(min(load, at.gamma)), censored_by_gamma=bool(censored), direction=n.tolist(), acoustic_eigenvalues_at_bound=ev.tolist(), polarization=U[:, 0].tolist(), evaluation_load=eval_load)
    scales = [1, 0.5, 0.25, 0.125]
    limit_k = f.A @ n
    limit_k *= 0.012 / la.norm(limit_k)
    samples = [dict(scale=h, load=float(f.eigen(limit_k * h, 1)[0]), ratio_to_acoustic=float(f.eigen(limit_k * h, 1)[0] / load)) for h in scales]
    row = dict(name=name, r=r, baseline_LE=reference['LE'], refined_LE=le, baseline_lambda_LE=reference['lambda_LE_pressure'], refined_lambda_LE=None if censored else load * at.p0, acoustic_initial=initial, relative_LE_change=load / reference['LE']['bound'] - 1, raw_grid_class=scan['refined_class'], refined_class=classify(reference['PD_pressure_pV'], le, scan['micro']), PD_strain_eigenvalues=pe, lambda_PD=reference['lambda_PD_pressure'], small_k_samples=samples, small_k_direction_seed=limit_k, acoustic_null_relative_residual=None if censored else float(la.norm(Q @ U[:, 0]) / la.norm(at.gibbs(0))), optimizer_success=success, seed_loads=[float(x[0]) for x in seeded])
    save(DEST / 'threshold_checks' / f'atlas/check_{name}_{r:.2f}.json', row)
    print('acoustic', name, r, 'relative_LE_change', row['relative_LE_change'], 'class', row['refined_class'], flush=True)
    return row

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name')
    ap.add_argument('--r', type=float, default=0.02)
    args = ap.parse_args()
    if args.name:
        check(args.name, args.r)
        return
    results = []
    for path in sorted((DEST / 'bloch').glob('atlas/check_*.json')):
        row = read(path)
        reference = raw(row['name'], row['r'])
        trigger = row['micro']['near_excluded_origin'] and row['micro']['load'] < reference['LE']['bound'] * (1 - 0.001)
        if not trigger and row['group'] != 'PD-only union':
            continue
        target = DEST / 'threshold_checks' / path.name
        results.append(read(target) if target.exists() else check(row['name'], row['r']))
    publish('atlas/threshold_checks.json', dict(rows=results, method='Exact full-gradient Schur acoustic limits. Check lower near-Gamma minima (>0.1% below baseline LE bound). Check the complete PD-only union using the baseline LE direction, refined Bloch direction and both approximately compatible PD-strain eigendirection seeds, followed by refinement of the two lowest seeds. The same beam tangent is used for every direction.'))
if __name__ == '__main__':
    main()
