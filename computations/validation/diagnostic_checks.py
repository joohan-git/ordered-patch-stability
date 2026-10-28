"""Check diagnostic results, regenerating missing mechanical maps and Ising states."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.diagnostics import *
from computations.truchet_2d.ising_scan import periodic_clusters, largest_squares
from collections import Counter
import ast, argparse
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--limit', type=int, help='Check this many H2 and Ising cases; regenerate missing states')
args = ap.parse_args()

def read(name):
    path = OUT / name
    if not path.is_file():
        path = _DATA_ROOT / name
    return json.loads(path.read_text(encoding='utf-8'))

def moments(a):
    a = np.asarray(a, dtype=float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0, minimum=float(a.min()), maximum=float(a.max()))
h2 = read('prestress/diagnostics.json')
ising = read('truchet_2d/ising.json')
windows = read('prestress/windows.json')
math = read('validation/ward_identities.json')
check = dict(source_hashes_match={}, counts={}, h2=[], ising=[], checks={}, files={})
for name in ('common/diagnostics.py', 'prestress/diagnostics.py', 'truchet_2d/ising_scan.py', 'prestress/window_feasibility.py', 'validation/ward_identities.py'):
    p = _DATA_ROOT / name
    ast.parse(p.read_text(encoding='utf-8'))
    check['files'][name] = dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for label, d in [('h2', h2), ('ising', ising), ('windows', windows)]:
    check['source_hashes_match'][label] = all((hashlib.sha256((DATA_ROOT / n).read_bytes()).hexdigest() == h for n, h in d['metadata']['source_sha256'].items()))
    assert check['source_hashes_match'][label]
assert h2['runs'] and ising['runs'] and len(windows['runs']) == 32
keys = [(r['d'], r['L'], r['R']) for r in h2['runs']]
assert len(keys) == len(set(keys))
if args.limit:
    h2['runs'] = h2['runs'][:args.limit]
    ising['runs'] = ising['runs'][:args.limit]
map_errors = []
for r in h2['runs']:
    row = dict(d=r['d'], L=r['L'], R=r['R'], s_pressure=r['s_pressure'], s_mean_axial=r['s_mean_axial'], cores={}, fixed_max_pct=100 * r['fixed_centre']['axial']['max_common'], fixed_rms_pct=100 * r['fixed_centre']['axial']['rms_common'], signed_pct=100 * r['mode']['e_signed'], axial_only_pct=100 * r['mode']['e_axial_only'], absolute_element_sum_pct=100 * r['mode']['absolute_element_sum'], operator_bound_pct=100 * r['mode']['operator_norm_bound'], profile=[dict(distance=p['distance_owner_tiles'], max_pct=100 * p['axial']['max_common'], rms_pct=100 * p['axial']['rms_common']) for p in r['profile']])
    assert abs(r['mode']['geometry_matching_max']) < 1e-12
    assert abs(r['mode']['e_signed'] + r['mode']['reference_Q_at_s_pressure_over_K']) < 1e-07
    for c in r['cores']:
        if c['status'] == 'empty':
            row['cores'][str(c['w'])] = None
            continue
        row['cores'][str(c['w'])] = dict(max_pct=100 * c['axial']['max_common'], rms_pct=100 * c['axial']['rms_common'], relative_l2_pct=100 * c['axial']['relative_l2'])
    if r['d'] == 2:
        path = OUT / 'prestress' / f"h2_map_d2_L{r['L']}_R{r['R']}.npz"
        if not path.exists():
            from computations.prestress.diagnostics import run_case
            motif = np.indices((2, 2)).sum(0) % 2
            base = Frame2D(motif)
            base.hydro()
            print('COMPUTE H2 map', r['L'], r['R'], flush=True)
            generated = run_case(2, r['L'], r['R'], motif, base)
            assert abs(generated['s_pressure'] - r['s_pressure']) < 1e-7
        z = np.load(path)
        lo = (r['L'] - r['R']) // 2
        for c in r['cores']:
            if c['status'] == 'empty':
                continue
            a = lo + c['w']
            b = lo + r['R'] - c['w']
            map_errors.append(abs(z['tile_max_common'][a:b, a:b].max() - c['axial']['max_common']))
    check['h2'].append(row)
check['checks']['h2_map_max_discrepancy'] = float(max(map_errors))
assert max(map_errors) < 1e-7
for key in ('equilibrium_residual',):
    check['checks']['h2_max_' + key] = max((r[key] for r in h2['runs']))
for key in ('eigen_residual', 'reference_residual', 'equation_check'):
    check['checks']['h2_max_' + key] = max((abs(r['mode'][key]) for r in h2['runs']))
keys = [(r['L'], r['J'], r['sample']) for r in ising['runs']]
assert len(keys) == len(set(keys))
errors = {k: [] for k in ['energy_sum', 'pr', 'cluster_energy', 'square_energy', 'pressure_ratio']}
for r in ising['runs']:
    from computations.truchet_2d.ising_scan import ensure_state
    z = np.load(ensure_state(r))
    sig = z['sigma']
    te = z['tile_energy']
    assert sig.shape == te.shape == (r['L'], r['L']) and set(np.unique(sig)) <= set([-1, 1])
    assert hashlib.sha256(sig.astype(np.int8).tobytes()).hexdigest() == r['pattern_sha256']
    assert te.min() > -1e-10
    lab, big, bigsize = periodic_clusters(sig)
    side, starts = largest_squares(sig)
    assert bigsize == r['largest_cluster'] and side == r['largest_square_side']
    assert len(big) == r['cluster_ties'] and len(starts) == r['square_ties']
    a, b = starts[0]
    L = r['L']
    errors['energy_sum'].append(abs(te.sum() - 1))
    errors['pr'].append(abs(1 / (te.size * np.sum(te ** 2)) - r['participation_ratio']))
    errors['cluster_energy'].append(abs(te[lab == big[0]].sum() - r['mode_energy_largest_cluster']))
    errors['square_energy'].append(abs(te[np.ix_((np.arange(side) + a) % L, (np.arange(side) + b) % L)].sum() - r['mode_energy_largest_square']))
    errors['pressure_ratio'].append(abs(r['pressure'] / ising['baseline_pressure'] - r['ratio']))
for k, v in errors.items():
    check['checks']['ising_max_' + k + '_discrepancy'] = float(max(v))
assert max(max(v) for v in errors.values()) < 1e-5
for key in ('eigen_residual', 'preload_residual'):
    check['checks']['ising_max_' + key] = max((r[key] for r in ising['runs']))
for L, J in sorted(set(((r['L'], r['J']) for r in ising['runs']))):
    a = [r for r in ising['runs'] if (r['L'], r['J']) == (L, J)]
    row = dict(L=L, J=J, n=len(a), samples=[r['sample'] for r in a])
    for key in ('ratio', 'largest_cluster', 'cluster_fraction', 'largest_square_side', 'participation_ratio', 'participating_tiles', 'mode_energy_largest_cluster', 'mode_energy_largest_square', 'seconds'):
        row[key] = moments([r[key] for r in a])
    row['correlations'] = {}
    for key in ('largest_cluster', 'largest_square_side', 'participation_ratio', 'mode_energy_largest_cluster'):
        row['correlations']['ratio_vs_' + key] = float(np.corrcoef([r['ratio'] for r in a], [r[key] for r in a])[0, 1]) if len(a) > 1 and np.std([r[key] for r in a]) > 0 else None
    check['ising'].append(row)
reference = json.loads((DATA_ROOT / 'truchet_2d/correlated.json').read_text(encoding='utf-8'))
same = []
for r in ising['runs']:
    if r['L'] != 64:
        continue
    match = [x for x in reference['runs'] if (x['J'], x['sample']) == (r['J'], r['sample'])]
    if match:
        same.append(abs(r['ratio'] - match[0]['ratio']))
check['checks']['ising_reference_ratio_matches'] = len(same)
check['checks']['ising_reference_max_ratio_discrepancy'] = max(same) if same else None
check['checks']['windows_minor_eigenvalue_discrepancy'] = max((abs(la.eigvalsh(np.array(r['principal_minor']))[0] - r['principal_minor_eigenvalue']) for r in windows['runs']))
from scipy.optimize import curve_fit
c3 = json.loads((DATA_ROOT / 'frame_3d/compact_bound.json').read_text(encoding='utf-8'))
rr = np.array([r['R'] for r in c3['runs']], float)
yy = np.array([r['pressure_normalized_excess'] for r in c3['runs']])
fits = {}
for lower in (3, 4, 5, 6):
    m = rr >= lower
    coef, cov = curve_fit(lambda x, a, g: a * x ** (-g), rr[m], yy[m], p0=[10, 2])
    pred = coef[0] * rr[m] ** (-coef[1])
    fits[str(lower)] = dict(exponent=coef[1], conditional_fit_se=np.sqrt(cov[1, 1]), relative_rms=np.sqrt(np.mean(((pred - yy[m]) / yy[m]) ** 2)))
fp = c3['fit_R2_R3']  # compact fit: a = alpha_2, b = alpha_3
m = rr >= 4
pred = fp['a'] / rr[m] ** 2 + fp['b'] / rr[m] ** 3
fits['R2_R3'] = dict(**fp, max_relative_error=float(np.max(abs((pred - yy[m]) / yy[m]))), observed_R=rr[m], relative_errors=(pred - yy[m]) / yy[m])
save('fit_sensitivity.json', fits)
check['counts'] = dict(h2=len(h2['runs']), ising=len(ising['runs']), windows=len(windows['runs']), math_acoustic=len(math['acoustic_anisotropic']), math_optical=len(math['flat_optical']))
save('validation/diagnostic_checks.json', check)
print(json.dumps({k: v for k, v in check.items() if k not in ('h2', 'ising', 'files')}, indent=2))
print('H2: d L R; core max w1,w2,w4; core RMS w2; fixed max/RMS; signed; abs sum; norm bound (all percent)')
for r in check['h2']:
    vals = [r['cores'][str(w)]['max_pct'] if r['cores'][str(w)] else None for w in (1, 2, 4)]
    print(r['d'], r['L'], r['R'], vals, r['cores']['2']['rms_pct'] if r['cores']['2'] else None, r['fixed_max_pct'], r['fixed_rms_pct'], r['signed_pct'], r['absolute_element_sum_pct'], r['operator_bound_pct'])
print('Ising summaries')
for r in check['ising']:
    print(json.dumps(r))
