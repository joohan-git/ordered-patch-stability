"""Localizable crystal thresholds under the measured inclusion stress tensor.

Uses the full Kirchhoff-Hermite initial-stress tangent, full-gradient acoustic
Schur complement and Bloch scan. No isotropic Gibbs/symmetric-strain completion
is used for anisotropic prestress. Only time reversal reduces the grid.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys, time, json, argparse, itertools, hashlib
from pathlib import Path
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--grid', type=int, default=8)
ap.add_argument('--cpu-limit', type=float, default=3500)
ap.add_argument('--nets', nargs='+', choices=['cco', 'vne'])
ap.add_argument('--inputs', type=Path, help='Defaults to freshly prepared output/inclusions/tensor/inputs.json, otherwise bundled inputs.json')
args = ap.parse_args()
sys.path.insert(0, str(HERE.parent))
from computations.common.runtime import output_dir
if args.grid < 2:
    ap.error('--grid must be >= 2')
INPUT = args.inputs or (output_dir() / 'inclusions/tensor/inputs.json' if (output_dir() / 'inclusions/tensor/inputs.json').exists() else HERE / 'inputs.json')
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from computations.atlas.bootstrap import bootstrap
from computations.atlas.run_atlas import hydrate
from computations.validation.beam_validation import ConsistentFrame
from computations.common.initial_stress import geometric_full
from computations.atlas.edge_center_atlas import TangentAtlas, sym_basis
from computations.common.fast_bloch import install
CPU0 = time.process_time()
T0 = time.perf_counter()

def limit():
    if time.process_time() - CPU0 > args.cpu_limit:
        raise RuntimeError('Tensor computation CPU limit reached')

def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)

def save(path, z):
    path.write_text(json.dumps(z, indent=2, default=plain, allow_nan=False), encoding='utf-8')

def frame_stress(graph, sigma):
    f = ConsistentFrame(graph['name'], 0.02, 2, hydrate(graph))
    S = sym_basis(3)
    basis = S.T.reshape(6, 3, 3)
    C = np.array([[np.sum(b * f.relaxed(a)['stress']) for a in basis] for b in basis])
    ep = la.solve(C, S.T @ sigma.ravel(), assume_a='pos')
    strain = sum((x * b for x, b in zip(ep, basis)))
    scale = 1 / (-np.trace(strain) / 3)
    strain *= scale
    f.pre = f.relaxed(strain)
    f.pre['strain'] = strain
    gm = []
    for e in f.elements:
        d = np.r_[strain @ e['a'], np.zeros(3), strain @ e['b'], np.zeros(3)] + np.r_[f.pre['w'][e['di']], f.pre['w'][e['dj']]]
        gm.append(geometric_full(e['a'], e['b'], e['K'] @ d))
    f.Gm = gm
    f.affine_matrices()
    error = float(la.norm(f.pre['stress'] / scale - sigma) / la.norm(sigma))
    assert error < 1e-10, error
    a = TangentAtlas.__new__(TangentAtlas)
    a.f = f
    a.K = f.affineK
    a.H = f.affineH
    a.Hpressure = a.H.copy()
    a.objective = False
    a.vol = abs(la.det(f.A))
    a.p0 = scale
    a.nf = len(a.K) - 9
    a.gamma = float(f.eigen([0, 0, 0], 1)[0])
    a.S = S
    a.cache = {}
    return (f, a, scale, error)

def acoustic_check(f, a, scale, le):
    n = np.array(le['direction'])
    P = np.kron(np.eye(3), n[:, None])
    t = 0.5 * le['bound']
    Q = P.T @ a.tangent(t) @ P
    trans = np.zeros((f.ndof, 3))
    trans.reshape(-1, 6, 3)[:, :3, :] = np.eye(3)
    internal = la.null_space(trans.T)
    out = []
    for eps in [0.01, 0.005, 0.0025]:
        k = eps * f.A @ n / (2 * np.pi)
        phase = np.repeat(np.exp(1j * eps * (f.frac @ f.A @ n)), 6)
        B = phase[:, None] * np.column_stack((internal, trans))
        H = f.assemble([e['K'] for e in f.elements], k) + t * f.assemble(f.Gm, k)
        D = B.conj().T @ H @ B
        q = (D[-3:, -3:] - D[-3:, :-3] @ la.solve(D[:-3, :-3], D[:-3, -3:], assume_a='her')) / (eps ** 2 * a.vol)
        out.append({'wave_number': eps, 'relative_error': float(la.norm(q - Q) / la.norm(Q))})
    assert out[-1]['relative_error'] < 0.002, out
    return out

def scan(graph, sigma, grid):
    start = time.perf_counter()
    cpu = time.process_time()
    f, a, scale, error = frame_stress(graph, sigma)
    le = a.le()
    limit()
    check = acoustic_check(f, a, scale, le)
    assembly = install(f)
    points = []
    for t in itertools.product(range(grid), repeat=3):
        inverse = tuple(-np.array(t) % grid)
        if t <= inverse:
            points.append(np.array(t, float) / grid)
    rows = []
    for i, k in enumerate(points):
        val = float(f.eigen(k, 1)[0])
        rows.append((val, k))
        limit()
    rows.sort(key=lambda z: z[0])
    best = rows[0]
    normal = best[0]
    seeds = [k for val, k in rows if la.norm(k - np.rint(k)) > 1e-08][:6]
    small = f.A @ np.array(le['direction'])
    seeds.append(small / la.norm(small) * 0.012)
    opts = []
    exclusion = 0.00625

    def objective(k):
        limit()
        q = np.asarray(k) - np.rint(k)
        if la.norm(q) < exclusion:
            return 1000.0 + (exclusion - la.norm(q)) * 1000.0
        return float(f.eigen(q, 1)[0]) / normal
    for k in seeds:
        opt = minimize(objective, k, method='Nelder-Mead', options={'maxiter': 220, 'xatol': 5e-07, 'fatol': 2e-10})
        q = opt.x - np.rint(opt.x)
        v = float(f.eigen(q, 1)[0])
        opts.append({'seed': k, 'k': q, 'load': v, 'success': bool(opt.success), 'nit': opt.nit})
        if la.norm(q) >= exclusion and v < best[0]:
            best = (v, q)
    loc = min(le['bound'], best[0]) * scale
    return {'lambda_loc_applied_scale': loc, 'lambda_LE_bound': le['bound'] * scale, 'LE': le, 'lambda_Bloch_min': best[0] * scale, 'k_min': best[1], 'gamma_optical': a.gamma * scale, 'pressure_per_factor': scale, 'stress_reconstruction_relative_error': error, 'preload_equilibrium_residual': f.pre['relative_residual'], 'grid': grid, 'grid_points_time_reversal': len(points), 'grid_samples': [{'k': k, 'load': v, 'applied_pressure': v * scale} for v, k in rows], 'refinements': opts, 'bloch_acoustic_checks': check, 'cached_assembly_relative_errors': assembly, 'seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu}
inputs = json.loads(INPUT.read_text(encoding='utf-8'))
out = {'cases': [], 'model': 'Kirchhoff-Hermite exact-strain initial tangent v1', 'loading': 'sigma(p_applied)=p_applied * Sigma; tension positive', 'acoustic': 'full-gradient relaxed material tangent; no hydrostatic Gibbs projection', 'certification': f'{args.grid}-cubed time-reversal grid plus local refinements and acoustic limit; numerical search, not a global spectral certificate', 'selected_nets': args.nets}
try:
    for c in inputs['cases']:
        if args.nets and c['name'] not in args.nets:
            continue
        limit()
        graph = bootstrap([c['name']])[0]
        assert np.allclose(graph['A'], c['A'], atol=1e-12, rtol=1e-12), 'Stress coordinate frame differs'
        sigma = np.array(c['patch_stress_per_reference_factor']) / c['applied_pressure_per_reference_factor']
        s = -np.trace(sigma) / 3
        dev = la.norm(sigma + s * np.eye(3)) / (np.sqrt(3) * abs(s))
        assert abs(s / c['s_P'] - 1) < 1e-12
        print('START', c['name'], 'deviatoric fraction', dev, flush=True)
        hydro = scan(graph, -np.eye(3), 4)
        tensor = scan(graph, sigma, args.grid)
        res = {**c, 'Sigma_per_applied_pressure': sigma, 'deviatoric_fraction': float(dev), 'hydro_control': hydro, 'tensor': tensor, 'hydro_loc_relative_difference_from_final_atlas': hydro['lambda_loc_applied_scale'] / c['bulk_hydro_loc'] - 1, 'tensor_rho_loc': c['onset_pressure'] / tensor['lambda_loc_applied_scale'], 'scalar_rho_loc_recomputed': c['onset_pressure'] * s / hydro['lambda_loc_applied_scale']}
        out['cases'].append(res)
        out['cpu_seconds'] = time.process_time() - CPU0
        save(output_dir() / 'inclusions/tensor/results.json', out)
        print('DONE', c['name'], 'rho_tensor', res['tensor_rho_loc'], 'rho_scalar', res['scalar_rho_loc_recomputed'], 'cpu', out['cpu_seconds'], flush=True)
    out['status'] = 'complete'
except Exception as e:
    import traceback
    out['status'] = 'failed'
    out['error'] = str(e)
    out['traceback'] = traceback.format_exc()
    raise
finally:
    out['cpu_seconds'] = time.process_time() - CPU0
    out['seconds'] = time.perf_counter() - T0
    save(output_dir() / 'inclusions/tensor/results.json', out)
