"""baseline: merge the dense loss-of-ellipticity search (atlas/dense_search_<r>.jsonl) into the baseline atlas.
Rule, applied identically to the atlas rows and to the tolerance study: lambda_LE is replaced by the dense value when the dense value is
lower, or when the production search found none and the dense value precedes the Bloch threshold lambda_mu (otherwise a missing
lambda_LE stays missing, i.e. beyond an earlier finite-wavevector or Gamma-optical instability). Classes use edge_center_atlas.classify (tolerance 0.3%).
Writes atlas/atlas_rows.json, atlas/summary.json and atlas/tolerance_sensitivity.json under K08_OUTPUT (same schema as the baseline files)."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import json, os, sys, collections
import numpy as np
from pathlib import Path
HERE = _DATA_ROOT
D = HERE
DATA_ROOT = HERE
from computations.common.runtime import output_dir
import argparse
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--nets', nargs='+', help='Explicit small subset; omitted means the full atlas')
args = ap.parse_args()
OUT = output_dir()
sys.path.insert(0, str(DATA_ROOT))
dense = {}
for source0 in ('0.02', '0.05'):
    for path in (HERE / f'atlas/dense_search_{source0}.jsonl', OUT / f'atlas/dense_search_{source0}.jsonl'):
        if not path.exists():
            continue
        for l in path.read_text(encoding='utf-8').splitlines():
            x = json.loads(l)
            dense[x['name'], x['r']] = x['LE_dense_pressure']

def new_le(reference, new, mu):
    """reference, new, mu on the same scale (pressure or load); None = not found / censored."""
    if new is None:
        return reference
    if reference is not None:
        return new if new < reference * (1 - 1e-09) else reference
    return new if mu is not None and new < mu else None

def cls(pd, le, mu, pd_volumetric, tol=0.003):
    """edge_center_atlas.classify on pressures."""
    lv = le if le is not None else np.inf
    pv = pd if pd is not None else np.inf
    mv = mu if mu is not None else np.inf
    if pv < min(lv, mv) * (1 - tol):
        return 'PD-only-volumetric' if pd_volumetric else 'PD-only-nonvolumetric'
    if lv <= mv * (1 + tol) and lv <= pv * (1 + tol):
        return 'LE' if pv > lv * (1 + tol) else 'PD+LE'
    if mv <= pv * (1 + tol) and mv <= lv * (1 + tol):
        return 'micro' if pv > mv * (1 + tol) else 'PD+micro'
    return 'near-tie'
rows = json.load(open(D / 'atlas/bloch_rows.json', encoding='utf-8'))
changed = []
if args.nets:
    rows = [r for r in rows if r['name'] in args.nets]
    if not rows:
        raise ValueError('No atlas nets selected')
for r in rows:
    key = (r['name'], r['r_over_each_parent_length'])
    if not r['lambda_PD_pressure'] or key not in dense:
        continue
    le = new_le(r['lambda_LE_pressure'], dense[key], r['lambda_micro_pressure'])
    if le == r['lambda_LE_pressure']:
        continue
    c = cls(r['lambda_PD_pressure'], le, r['lambda_micro_pressure'], r['class_pressure_pV'] == 'PD-only-volumetric')
    if c == 'near-tie':
        c = r['class_pressure_pV']
    changed.append(dict(name=key[0], r=key[1], lambda_LE=[r['lambda_LE_pressure'], le], cls=[r['class_pressure_pV'], c]))
    r['lambda_LE_pressure'] = le
    r['class_pressure_pV'] = c
    r['le_search'] = '3,000 directions + 10 refinements (baseline)'
summ = dict(n_rows=len(rows), source='baseline atlas + baseline dense loss-of-ellipticity search (computations/atlas/dense_search_*.jsonl; regenerated output rows take precedence)', radii={}, selected_nets=args.nets, scope='explicit subset' if args.nets else 'full atlas')
for source0 in (0.02, 0.05):
    rs = [r for r in rows if r['r_over_each_parent_length'] == source0]
    cc = collections.Counter((r['class_pressure_pV'] for r in rs))
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
    summ['radii'][str(source0)] = dict(n=len(rs), classes=dict(cc), by_crystal_system={k: dict(v) for k, v in bysys.items()}, PD_only_n=len(pdo), PD_only_invariant_n=int(sum((abs(r['invariant_fraction']) > 0.99 for r in pdo))), protection_window=dict(n=len(w), median=float(np.median(w)) if len(w) else None, q25=float(np.percentile(w, 25)) if len(w) else None, q75=float(np.percentile(w, 75)) if len(w) else None, min=float(w.min()) if len(w) else None, max=float(w.max()) if len(w) else None, max_net=win[int(np.argmax(w))]['name'] if len(w) else None, nets=win), selection_rule_test=dict(n_breaking_defined=int(len(gaps)), max_gap=float(gaps.max()) if len(gaps) else None, n_gap_gt_1e8=int((gaps > 1e-08).sum()), n_inv_between_1e9_and_099=int(((f > 1e-09) & (f < 0.99)).sum())), preload_force_normalized_residual=dict(max=float(res.max()), p99=float(np.percentile(res, 99))))
by = collections.defaultdict(dict)
for r in rows:
    by[r['name']][r['r_over_each_parent_length']] = r['class_pressure_pV']
summ['n_nets'] = len(by)
summ['class_same_both_radii'] = sum((1 for v in by.values() if len(v) == 2 and v[0.02] == v[0.05]))
summ['changes_from_bloch'] = changed
json.dump(rows, open(OUT / 'atlas/atlas_rows.json', 'w', encoding='utf-8'))
json.dump(summ, open(OUT / 'atlas/summary.json', 'w', encoding='utf-8'), indent=1)
for source0, v in summ['radii'].items():
    print(source0, v['classes'], 'PD-only inv', v['PD_only_invariant_n'], '/', v['PD_only_n'], 'window median %s q25 %s q75 %s max %s (%s)' % (v['protection_window']['median'], v['protection_window']['q25'], v['protection_window']['q75'], v['protection_window']['max'], v['protection_window']['max_net']), v['selection_rule_test'])
print('class same', summ['class_same_both_radii'], 'changed rows', len(changed))
for c in changed:
    if c['cls'][0] != c['cls'][1]:
        print('  class change', c)
import computations.atlas.tolerance as S
_raw = S.raw

def raw18(name, r):
    x = _raw(name, r)
    return x
orig_build = S.build

def build18():
    out = None
    _orig_atlas = S.atlas_rows
    S.atlas_rows = lambda: json.load(open(D / 'atlas/baseline_rows.json', encoding='utf-8'))
    atlas = S.atlas_rows()
    rows_ = []
    if args.nets:
        atlas = [r for r in atlas if r['name'] in args.nets]
    b = S.read((S.DEST if (S.DEST / 'atlas/bloch_checks.json').exists() else HERE) / 'atlas/bloch_checks.json')
    updates = {(x['name'], x['r']): x for x in b['rows']}
    for a in atlas:
        x = S.raw(a['name'], a['r_over_each_parent_length'])
        k = (x['name'], x['r_over_each_parent_length'])
        if k in updates:
            x = dict(x, micro=updates[k]['micro'], LE=updates[k].get('refined_LE', x['LE']))
        pd = x['PD_pressure_pV']['load']
        if pd is not None and k in dense:
            p0 = x['p0']
            mu = x['micro']['load']
            le = new_le(x['LE']['load'], dense[k] / p0, mu)
            if le != x['LE']['load']:
                x = dict(x, LE=dict(x['LE'], load=le))
        rows_.append(x)
    return rows_
rows18 = build18()
out = dict(formula='baseline formula with the baseline dense loss-of-ellipticity search', radii={}, selected_nets=args.nets, scope='explicit subset' if args.nets else 'full atlas')
for source0 in S.RADII:
    subset = [x for x in rows18 if x['r_over_each_parent_length'] == source0]
    table = []
    for tol in S.TOLS:
        classes = {x['name']: S.classify(x['PD_pressure_pV'], x['LE'], x['micro'], tol) for x in subset}
        table.append(dict(tolerance=tol, classes=dict(collections.Counter(classes.values())), pdonly_invariant_count=sum((cl.startswith('PD-only') and next((x['invariant_fraction'] for x in subset if x['name'] == n)) > 0.99 for n, cl in classes.items())), windows={k: v for k, v in S.windows(subset, tol).items() if k != 'rows'}))
    out['radii'][str(source0)] = dict(n=len(subset), tolerances=table)
json.dump(out, open(OUT / 'atlas/tolerance_sensitivity.json', 'w', encoding='utf-8'), indent=1)
for source0, v in out['radii'].items():
    print('tol', source0, [(t['tolerance'], sum((c for k, c in t['classes'].items() if k.startswith('PD-only'))), round(t['windows']['p50'], 4) if 'p50' in t['windows'] else None) for t in v['tolerances']])
