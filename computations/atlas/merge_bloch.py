"""Merge the baseline check into the baseline atlas (no recomputation here).
- class, lambda_LE and lambda_mu from the 8^3 Bloch check of every net that is PD-only at either radius plus 40 PD+LE controls
  (computations/atlas/bloch_checks.json);
- force-normalized preload residual for all rows (computations/prestress/atlas_preload.json).
Writes computations/atlas/bloch_rows.json and atlas/bloch_summary.json (same schema as baseline, plus the residual)."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import json, os, collections, argparse
import numpy as np
HERE = str(_DATA_ROOT)
DATA_ROOT = HERE
D = HERE
from computations.common.runtime import output_dir
ap = argparse.ArgumentParser()
ap.add_argument('--archived', action='store_true', help='Merge only the bundled full-cohort results')
args = ap.parse_args()

def J(*parts):
    candidate = output_dir() / parts[-1]
    path = candidate if not args.archived and candidate.is_file() else os.path.join(*parts)
    return json.load(open(path, encoding='utf-8'))
rows = J(D, 'atlas/baseline_rows.json')
if len(rows) != 1752:
    raise ValueError('Final atlas merge requires all 1752 baseline rows; use --archived for bundled full-cohort results')
bl = {(r['name'], r['r']): r for r in J(DATA_ROOT, 'atlas/bloch_checks.json')['rows']}
pre = {(r['name'], r['r']): r for r in J(DATA_ROOT, 'prestress/atlas_preload.json')['rows']}
changed = []
for r in rows:
    key = (r['name'], r['r_over_each_parent_length'])
    p = pre[key]
    r['preload_force_normalized_residual'] = p['force_normalized_residual']
    r['preload_abs_residual'] = p['abs_residual']
    if key in bl:
        b = bl[key]
        assert abs(b['lambda_PD'] / r['lambda_PD_pressure'] - 1) < 1e-09, key
        new = dict(class_pressure_pV=b['refined_class'], lambda_LE_pressure=b['lambda_LE'], lambda_micro_pressure=b['lambda_mu'])
        if any((r[k] != v for k, v in new.items())):
            changed.append((key, {k: (r[k], v) for k, v in new.items() if r[k] != v}))
        r.update(new)
        r['bloch_check'] = '8^3 grid, Gamma censor/4 (baseline)'
summ = dict(n_rows=len(rows), source='baseline atlas + baseline check (computations/atlas/bloch_checks.json, prestress/atlas_preload.json)', radii={})
for source0 in (0.02, 0.05):
    rs = [r for r in rows if r['r_over_each_parent_length'] == source0]
    cls = collections.Counter((r['class_pressure_pV'] for r in rs))
    bysys = collections.defaultdict(collections.Counter)
    for r in rs:
        bysys[r['crystal_system']][r['class_pressure_pV']] += 1
    pdo = [r for r in rs if r['class_pressure_pV'] == 'PD-only-nonvolumetric']
    win = []
    for r in pdo:
        c = [x for x in (r['lambda_LE_pressure'], r['lambda_micro_pressure']) if x]
        if c:
            win.append(dict(name=r['name'], window=min(c) / r['lambda_PD_pressure'] - 1))
    w = np.array([x['window'] for x in win])
    hb = [r for r in rs if r['lambda_PD_pressure'] and abs(r['invariant_fraction']) < 0.5]
    gaps = np.array([r['lambda_LE_pressure'] / r['lambda_PD_pressure'] - 1 for r in hb if r['lambda_LE_pressure']])
    f = np.array([abs(r['invariant_fraction']) for r in rs if r['lambda_PD_pressure']])
    res = np.array([r['preload_force_normalized_residual'] for r in rs])
    summ['radii'][str(source0)] = dict(n=len(rs), classes=dict(cls), by_crystal_system={k: dict(v) for k, v in bysys.items()}, PD_only_n=len(pdo), PD_only_invariant_n=int(sum((abs(r['invariant_fraction']) > 0.99 for r in pdo))), protection_window=dict(n=len(w), median=float(np.median(w)), q25=float(np.percentile(w, 25)), q75=float(np.percentile(w, 75)), min=float(w.min()), max=float(w.max()), max_net=win[int(np.argmax(w))]['name'], nets=win), selection_rule_test=dict(n_breaking_defined=int(len(gaps)), max_gap=float(gaps.max()), n_gap_gt_1e8=int((gaps > 1e-08).sum()), n_inv_between_1e9_and_099=int(((f > 1e-09) & (f < 0.99)).sum())), preload_force_normalized_residual=dict(max=float(res.max()), p99=float(np.percentile(res, 99))))
by = collections.defaultdict(dict)
for r in rows:
    by[r['name']][r['r_over_each_parent_length']] = r['class_pressure_pV']
summ['n_nets'] = len(by)
summ['class_same_both_radii'] = sum((1 for v in by.values() if len(v) == 2 and v[0.02] == v[0.05]))
summ['changes_from_baseline'] = [dict(name=k[0], r=k[1], changes={a: list(b) for a, b in c.items()}) for k, c in changed]
json.dump(rows, open(output_dir() / 'atlas/bloch_rows.json', 'w', encoding='utf-8'))
json.dump(summ, open(output_dir() / 'atlas/bloch_summary.json', 'w', encoding='utf-8'), indent=1)
for source0, v in summ['radii'].items():
    print(source0, v['classes'], 'PD-only inv', v['PD_only_invariant_n'], '/', v['PD_only_n'], 'window n %d median %.4f q25 %.4f q75 %.4f max %.3f (%s)' % (v['protection_window']['n'], v['protection_window']['median'], v['protection_window']['q25'], v['protection_window']['q75'], v['protection_window']['max'], v['protection_window']['max_net']), v['selection_rule_test'], v['preload_force_normalized_residual'])
print('class same', summ['class_same_both_radii'], 'changed rows', len(changed))
