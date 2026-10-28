"""Summarize the baseline atlas (symmetry-exact graphs, consistent beam initial-stress tangent).
Reads per-net JSON from K08_CACHE/atlas/results and writes computations/atlas/baseline_summary.json + compact rows."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import json, glob, os, collections, tempfile
import numpy as np
from computations.common.runtime import cache_dir, output_dir
SRC = str(cache_dir('atlas') / 'results')
OUT = str(output_dir())
rows = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(os.path.join(SRC, '*.json')))]
summ = dict(n_rows=len(rows), radii={})
compact = []
for r in rows:
    compact.append({k: r.get(k) for k in ('name', 'r_over_each_parent_length', 'space_group', 'crystal_system', 'point_group_order', 'class_pressure_pV', 'invariant_fraction', 'lambda_PD_pressure', 'lambda_LE_pressure', 'lambda_micro_pressure')} | {'preload_equilibrium': r['check']['preload_equilibrium'], 'ward': r['check']['ward'], 'micro_near_origin': r['micro'].get('near_excluded_origin')})
for source0 in (0.02, 0.05):
    rs = [r for r in rows if r['r_over_each_parent_length'] == source0]
    cls = collections.Counter((r['class_pressure_pV'] for r in rs))
    bysys = collections.defaultdict(collections.Counter)
    for r in rs:
        bysys[r['crystal_system']][r['class_pressure_pV']] += 1
    inv = collections.defaultdict(list)
    for r in rs:
        inv[r['class_pressure_pV']].append(r['invariant_fraction'])
    pdo = [r for r in rs if r['class_pressure_pV'] == 'PD-only-nonvolumetric']
    win = []
    for r in pdo:
        c = [x for x in (r['lambda_LE_pressure'], r['lambda_micro_pressure']) if x]
        if r['lambda_PD_pressure'] and c:
            win.append(dict(name=r['name'], window=min(c) / r['lambda_PD_pressure'] - 1))
    w = np.array([x['window'] for x in win])
    eq = np.array([r['check']['preload_equilibrium'] for r in rs])
    summ['radii'][str(source0)] = dict(n=len(rs), classes=dict(cls), by_crystal_system={k: dict(v) for k, v in bysys.items()}, invariant_fraction_gt_0p99={k: float(np.mean(np.array(v) > 0.99)) for k, v in inv.items()}, PD_only_n=len(pdo), PD_only_invariant_n=int(sum((r['invariant_fraction'] > 0.99 for r in pdo))), PD_only_noninvariant=[r['name'] for r in pdo if r['invariant_fraction'] <= 0.99], protection_window=dict(n=len(w), median=float(np.median(w)), q25=float(np.percentile(w, 25)), q75=float(np.percentile(w, 75)), min=float(w.min()), max=float(w.max()), nets=win), max_ward=float(max((r['check']['ward'] for r in rs))), preload_equilibrium=dict(median=float(np.median(eq)), q99=float(np.percentile(eq, 99)), max=float(eq.max()), n_gt_1em6=int((eq > 1e-06).sum()), worst=sorted([(r['name'], r['check']['preload_equilibrium'], r['class_pressure_pV']) for r in rs], key=lambda x: -x[1])[:12]))
by = collections.defaultdict(dict)
for r in rows:
    by[r['name']][r['r_over_each_parent_length']] = r['class_pressure_pV']
summ['n_nets'] = len(by)
summ['class_same_both_radii'] = sum((1 for v in by.values() if len(v) == 2 and v[0.02] == v[0.05]))
json.dump(summ, open(os.path.join(OUT, 'atlas/baseline_summary.json'), 'w'), indent=1)
json.dump(compact, open(os.path.join(OUT, 'atlas/baseline_rows.json'), 'w'))
print(json.dumps({k: v for k, v in summ.items() if k != 'radii'}))
for source0, v in summ['radii'].items():
    print(source0, v['classes'], 'PDonly inv', v['PD_only_invariant_n'], '/', v['PD_only_n'], v['PD_only_noninvariant'], 'window med %.4f q25 %.4f q75 %.4f max %.3f' % (v['protection_window']['median'], v['protection_window']['q25'], v['protection_window']['q75'], v['protection_window']['max']), 'eq', v['preload_equilibrium']['median'], v['preload_equilibrium']['max'], v['preload_equilibrium']['n_gt_1em6'])
    print('   worst eq:', v['preload_equilibrium']['worst'][:8])
