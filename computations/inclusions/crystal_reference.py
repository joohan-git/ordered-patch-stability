"""Axial-prestress crystal thresholds on edge-centre RCSR graphs."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json, time, argparse, hashlib
from pathlib import Path
import numpy as np
import scipy.linalg as la
import gemmi
from computations.common.bloch_frame import LD, NETS
from computations.common.acoustic_atlas import SlenderFrame, TangentAtlas, sym_basis, volume_hessian, classify
ROOT = _DATA_ROOT
BENCHMARK = _DATA_ROOT
NAMES = ['utp', 'ncd', 'ncj', 'cco', 'vne', 'oof', 'bnn', 'acs', 'qtz', 'srs', 'pcu', 'hex']

def dump(path, obj):

    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=convert, allow_nan=False)
    path = Path(path)
    temporary = path.with_suffix('.writing')
    with open(temporary, 'w', encoding='utf-8') as f:
        f.write(text)
    for attempt in range(16):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 15:
                raise
            time.sleep(0.05 * (attempt + 1))

def graph_data(name):
    g = LD['load_ec'](name, NETS)
    if g is None or not g['ok']:
        raise ValueError('EDGE_CENTER degree validation failed')
    if not g['net'].get('ecs'):
        raise ValueError('No EDGE_CENTER: fallback deliberately disallowed')
    lengths = np.array([la.norm((g['frac'][j] + sh - g['frac'][i]) @ g['A']) for i, j, sh in g['edges']])
    mean = lengths.mean()
    A = g['A'] / mean
    sg = gemmi.find_spacegroup_by_name(g['net']['group'] + (g['setting'] or ''))
    return dict(name=name, net=g['net'], sg=sg, frac=g['frac'], cn=g['cn'], edges=g['edges'], A=A, trials=[dict(EDGE_CENTER=True, CN_validated=True)], lengths=(lengths / mean).tolist())

def benchmark_rows():
    return {r['name']: r for p in sorted(BENCHMARK.glob('atlas_chunk*.json')) for r in json.loads(p.read_text(encoding='utf-8'))['runs'] if r['r_over_each_parent_length'] == 0.02}

def reference(name, slender=0.02, div=2, grid=4, protocol='pressure'):
    start = time.perf_counter()
    d = graph_data(name)
    f = SlenderFrame(name, slender, div, d)
    f.prestress('stress' if protocol == 'pressure' else 'strain')
    f.affine_matrices()
    a = TangentAtlas.__new__(TangentAtlas)
    a.f = f
    a.K = f.affineK
    a.H = f.affineH
    a.vol = abs(la.det(f.A))
    a.p0 = -np.trace(f.pre['stress']) / 3
    a.nf = len(a.K) - 9
    a.gamma = float(f.eigen([0, 0, 0], 1)[0])
    a.S = sym_basis(3)
    a.Hpressure = a.H.copy()
    a.Hpressure[-9:, -9:] -= a.p0 * a.vol * volume_hessian(3)
    a.cache = {}
    pd = a.pd(protocol == 'pressure')
    le = a.le()
    mu = a.bloch(grid)
    diagnostic_class = classify(pd, le, mu)
    cls = diagnostic_class
    if protocol == 'strain':
        cls = 'LE' if le['load'] is not None and le['load'] <= mu['load'] * 1.003 else 'micro'
    scale = a.p0 if protocol == 'pressure' else 1.0
    pdval = None if pd['load'] is None else float(pd['load'] * scale)
    leval = None if le['load'] is None else float(le['load'] * scale)
    micro = None if mu['near_excluded_origin'] else float(mu['load'] * scale)
    loc = min(le['bound'], mu['load']) * scale
    first = min(pd['bound'], le['bound'], mu['load']) * scale if protocol == 'pressure' else loc
    values, v = f.eigen([0, 0, 0], count=min(12, len(f.free) - 1), vectors=True)
    vg = v[f.free, 0]
    B = a.K[a.nf:, :a.nf] - a.gamma * a.H[a.nf:, :a.nf]
    coupling = float(la.norm(B @ vg) / max(la.norm(a.K[a.nf:, :a.nf] @ vg) + abs(a.gamma) * la.norm(a.H[a.nf:, :a.nf] @ vg), 1e-30))
    V = la.orth(v[f.free][:, values / values[0] < 1 + 2e-06])
    bsv = la.svdvals(B @ V) / max(la.norm(B), 1e-30)
    decoupled = bool(len(bsv) < V.shape[1] or bsv[-1] < 1e-08)
    gamma_absorbed = bool(not decoupled and pd['load'] is not None and (pd['load'] < a.gamma * (1 - 2e-06)))
    raw_gamma_pole = None
    if la.norm(np.asarray(mu['k']) - np.rint(mu['k'])) < 1e-08 and gamma_absorbed:
        raw_gamma_pole = micro
        micro = None
    refvir = []
    for e in f.elements:
        disp = np.r_[f.pre['strain'] @ e['a'], np.zeros(3), f.pre['strain'] @ e['b'], np.zeros(3)]
        disp += np.r_[f.pre['w'][e['di']], f.pre['w'][e['dj']]]
        force = e['K'] @ disp
        delta = e['b'] - e['a']
        vv = np.outer(force[6:9], delta)
        refvir.append((vv + vv.T) / 2)
    out = dict(name=name, protocol=protocol, slender=slender, div=div, grid=grid, nodes=len(d['frac']), parent_edges=len(d['edges']), dof=f.ndof, cell_volume=a.vol, A=d['A'], lengths=d['lengths'], setting=d['sg'].xhm(), graph='EDGE_CENTER; no nearest-neighbour fallback', class_first=cls, macro_diagnostic_class=diagnostic_class, PD=pd, LE=le, micro=mu, lambda_PD=pdval, lambda_LE=leval, lambda_mu=micro, lambda_loc=float(loc), lambda_first=float(first), lambda_gamma=float(a.gamma * scale), lambda_mu_sampled_upper=float(mu['load'] * scale), pressure_per_factor=float(a.p0), preload_strain=f.pre['strain'], preload_stress=f.pre['stress'], forces_per_factor=f.pre['forces'], element_virials_per_factor=np.array(refvir), gamma_coupling_relative=coupling, gamma_coupling_singular_values_normalized=bsv, gamma_multiplicity=int(V.shape[1]), gamma_has_decoupled_branch=decoupled, gamma_absorbed=gamma_absorbed, lambda_mu_raw_absorbed_gamma=raw_gamma_pole, equilibrium_residual=f.pre['force_scaled_residual'], seconds=time.perf_counter() - start)
    return out

def run(names):
    benchmark = benchmark_rows()
    for name in names:
        path = ROOT / f'inclusions/axial_ref_{name}.json'
        out = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(name=name, benchmark_class=benchmark.get(name, {}).get('class_pressure_pV'), runs=[])
        done = {(r['protocol'], r['slender'], r['div']) for r in out['runs']}
        for slender, protocol in [(0.02, 'pressure'), (0.02, 'strain'), (0.03, 'pressure'), (0.03, 'strain')]:
            if (protocol, slender, 2) in done:
                continue
            row = reference(name, slender, 2, 4, protocol)
            out['runs'].append(row)
            dump(path, out)
            print(name, slender, protocol, row['class_first'], row['lambda_PD'], row['lambda_LE'], row['lambda_mu'], round(row['seconds'], 2), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*', default=NAMES)
    a = ap.parse_args()
    run(a.names)
