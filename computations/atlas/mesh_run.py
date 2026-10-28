"""baseline: fixed 138-net cohort, independently recomputed 2/4/8 meshes."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.mesh_common import *
from computations.atlas.mesh_solver import setup, install_sparse
from computations.atlas.mesh_condense import install as install_condensed
from computations.common.bloch_frame import representatives
from scipy.optimize import minimize
import time, traceback, argparse

def scan12(f, reference):
    points = representatives(f, 8)
    grid = sorted([(float(f.eigen(k, 1)[0]), np.asarray(k)) for k in points], key=lambda x: x[0])
    scale = grid[0][0]
    trials = [(v, k, 'grid') for v, k in grid]
    seeds = [np.asarray(reference['micro']['k'])] + [k for v, k in grid if la.norm(k - np.rint(k)) > 1e-09][:6]
    n = np.asarray(reference['LE']['direction'])
    k = f.A @ n
    seeds.append(k / la.norm(k) * 0.012)
    opts = []

    def objective(k):
        q = np.asarray(k) - np.rint(k)
        if la.norm(q) < 0.00625:
            return 1000.0 + (0.00625 - la.norm(q)) * 1000.0
        return float(f.eigen(q, 1)[0]) / scale
    for seed in seeds:
        if la.norm(seed - np.rint(seed)) < 0.00625:
            continue
        trials.append((float(f.eigen(seed, 1)[0]), seed, 'seed'))
        opt = minimize(objective, seed, method='Nelder-Mead', options={'maxiter': 220, 'xatol': 5e-07, 'fatol': 2e-10})
        q = opt.x - np.rint(opt.x)
        val = float(f.eigen(q, 1)[0])
        opts.append(dict(seed=seed, k=q, load=val, nit=opt.nit, success=bool(opt.success), origin_distance=float(la.norm(q))))
        if la.norm(q) >= 0.00625:
            trials.append((val, q, 'refined'))
    val, k, source = min(trials, key=lambda x: x[0])
    q = k - np.rint(k)
    micro = dict(load=val, k=q, grid=8, points=len(grid), near_excluded_origin=bool(0 < la.norm(q) < 0.00675), gamma_optical=float(f.eigen([0, 0, 0], 1)[0]), source=source)
    return (micro, dict(grid_samples=[dict(k=k, load=v) for v, k in grid], refinements=opts))

def acoustic12(at, pd, micro, reference, standard):
    seeds = [np.asarray(reference['LE']['direction'])]
    k = np.asarray(micro['k'])
    if la.norm(k) > 1e-09:
        n = la.solve(at.f.A, k)
        seeds.append(n / la.norm(n))
    ev, U = la.eigh(np.asarray(pd['strain_mode']))
    if ev[0] < 0 < ev[2]:
        for sign in [-1, 1]:
            n = np.sqrt(ev[2]) * U[:, 2] + sign * np.sqrt(-ev[0]) * U[:, 0]
            seeds.append(n / la.norm(n))
    seeded = sorted([(at.acoustic(n), n) for n in seeds], key=lambda x: x[0])
    load = standard['bound']
    n = np.asarray(standard['direction'])
    opts = []
    for scale, seed in seeded[:2]:
        if scale < load:
            load = scale
            n = seed
        opt = minimize(lambda v: at.acoustic(v) / scale, seed, method='Nelder-Mead', options={'maxiter': 350, 'xatol': 1e-08, 'fatol': 1e-11})
        val = at.acoustic(opt.x)
        opts.append(dict(load=val, success=bool(opt.success), nit=opt.nit, direction=opt.x / la.norm(opt.x)))
        if val < load:
            load = val
            n = opt.x / la.norm(opt.x)
    censored = load >= at.gamma * (1 - 2e-06)
    return dict(load=None if censored else float(load), bound=float(min(load, at.gamma)), censored_by_gamma=bool(censored), direction=n, standard=standard, extra_seed_loads=[x[0] for x in seeded], refinements=opts)

def run(name, div, group):
    start = time.perf_counter()
    baseline_row = raw(name, 0.02)
    f = newframe(name, div)
    setup(f)
    f.prestress('stress')
    preload = f.prestress
    f.prestress = lambda protocol='stress': f.pre if protocol == 'stress' else preload(protocol)
    dense = f.eigen
    sparse_info = install_sparse(f)
    sparse = f.eigen
    solver = install_condensed(f)
    checks = []
    for k in [[0, 0, 0], [0.125, 0.25, 0.375], [0.007, 0.002, 0.001]]:
        reference = float((dense if f.ndof <= 350 else sparse)(k, 1)[0])
        calculated = float(f.eigen(k, 1)[0])
        error = calculated / reference - 1
        assert abs(error) < 1e-07, (name, div, k, error)
        checks.append(dict(k=k, reference=reference, condensed=calculated, relative_error=error, reference_solver='dense' if f.ndof <= 350 else 'sparse'))
    at = ConsistentAtlas(f)
    pd = at.pd(True)
    reference_pd = at.pd
    at.pd = lambda pressure=False: pd if pressure else reference_pd(False)
    le = at.le()
    if group == 'PD-only':
        micro, detail = scan12(f, baseline_row)
        le = acoustic12(at, pd, micro, baseline_row, le)
    else:
        micro = at.bloch(4)
        detail = {}
    final_reference = float(sparse(micro['k'], 1)[0])
    final_error = final_reference / micro['load'] - 1
    assert abs(final_error) < 1e-07, (name, div, 'final', final_error)
    force = np.zeros(f.ndof)
    for e, ff in zip(f.elements, f.pre['full_element_forces']):
        np.add.at(force, e['di'], ff[:6])
        np.add.at(force, e['dj'], ff[6:])
    return dict(name=name, r=0.02, div=div, group=group, ndof=f.ndof, graph_sha256=graph(name)['graph_sha256'], p0=float(at.p0), PD=pd, LE=le, micro=micro, class_pressure_pV=classify(pd, le, micro), lambda_PD=None if pd['load'] is None else pd['load'] * at.p0, lambda_LE=None if le['load'] is None else le['load'] * at.p0, lambda_mu=None if micro['near_excluded_origin'] else micro['load'] * at.p0, lambda_mu_sampled_upper=micro['load'] * at.p0, lambda_PD_bound=pd['bound'] * at.p0, lambda_LE_bound=le['bound'] * at.p0, force_normalized_residual=float(la.norm(force) / la.norm(f.pre['full_element_forces'])), solver=solver, sparse_validation=sparse_info, validation=checks, final_bloch_sparse_relative_error=final_error, bloch_detail=detail, seconds=time.perf_counter() - start, threadpools=POOLS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--chunks', type=int, default=1)
    ap.add_argument('--name')
    ap.add_argument('--div', type=int)
    args = ap.parse_args()
    c = cohort()
    names = c['pdonly'] + c['controls']
    for idx, name in enumerate(names):
        if args.name and name != args.name:
            continue
        if not args.name and idx % args.chunks != args.chunk:
            continue
        for div in [2, 4, 8]:
            if args.div and div != args.div:
                continue
            target = DEST / 'results' / f'atlas/mesh_{name}_d{div}.json'
            if target.exists():
                continue
            try:
                row = run(name, div, 'PD-only' if name in c['pdonly'] else 'control')
                save(target, row)
                print('DONE', idx + 1, len(names), name, div, row['class_pressure_pV'], round(row['seconds'], 2), flush=True)
            except Exception:
                text = traceback.format_exc()
                print('ERROR', name, div, text, flush=True)
                target = DEST / 'errors' / f'atlas/mesh_{name}_d{div}_{time.time_ns()}.json'
                save(target, dict(name=name, div=div, error=text))
if __name__ == '__main__':
    main()
