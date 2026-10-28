"""Exact baseline classification formula and direct selection-rule statistic."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.check_common import *
from collections import Counter
TOLS = [1e-08, 1e-06, 0.0001, 0.001, 0.003, 0.01, 0.03]

def windows(rows, tol):
    vals = []
    unavailable = []
    for x in rows:
        if not classify(x['PD_pressure_pV'], x['LE'], x['micro'], tol).startswith('PD-only'):
            continue
        roots = [v for v in [x['LE']['load'], None if x['micro']['near_excluded_origin'] else x['micro']['load']] if v is not None]
        if roots:
            vals.append(dict(name=x['name'], window=min(roots) / x['PD_pressure_pV']['load'] - 1))
        else:
            unavailable.append(x['name'])
    return dict(**stats([x['window'] for x in vals]), q25=float(np.quantile([x['window'] for x in vals], 0.25)) if vals else None, q75=float(np.quantile([x['window'] for x in vals], 0.75)) if vals else None, unavailable=unavailable, rows=vals)

def build(require_complete=True, names=None):
    atlas = atlas_rows()
    rows = []
    reference = []
    if names:
        atlas = [x for x in atlas if x['name'] in names]
    if require_complete:
        b = read(DEST / 'atlas/bloch_checks.json')
        updates = {(x['name'], x['r']): x for x in b['rows']}
    else:
        updates = {}
    for a in atlas:
        x = raw(a['name'], a['r_over_each_parent_length'])
        reference.append(x)
        assert classify(x['PD_pressure_pV'], x['LE'], x['micro']) == a['class_pressure_pV']
        if (x['name'], x['r_over_each_parent_length']) in updates:
            update = updates[x['name'], x['r_over_each_parent_length']]
            x = dict(x, micro=update['micro'], LE=update.get('refined_LE', x['LE']))
        rows.append(x)
    per_radius = {}
    selection = {}
    for r in RADII:
        subset = [x for x in rows if x['r_over_each_parent_length'] == r]
        before = [x for x in reference if x['r_over_each_parent_length'] == r]
        table = []
        for tol in TOLS:
            classes = {x['name']: classify(x['PD_pressure_pV'], x['LE'], x['micro'], tol) for x in subset}
            table.append(dict(tolerance=tol, classes=dict(Counter(classes.values())), pdonly_invariant_count=sum((cl.startswith('PD-only') and next((x['invariant_fraction'] for x in subset if x['name'] == n)) > 0.99 for n, cl in classes.items())), windows=windows(subset, tol), per_net_class=classes))
        per_radius[str(r)] = dict(n=len(subset), tolerances=table, baseline_windows=windows(before, 0.003))
        records = []
        for x in subset:
            if x['invariant_fraction'] >= 0.5:
                continue
            pd = x['PD_pressure_pV']['load']
            le = x['LE']['load']
            records.append(dict(name=x['name'], r=r, invariant_fraction=x['invariant_fraction'], class_first=classify(x['PD_pressure_pV'], x['LE'], x['micro']), lambda_PD=None if pd is None else pd * x['p0'], lambda_LE=None if le is None else le * x['p0'], relative_gap=None if pd is None or le is None else le / pd - 1, status='defined' if pd is not None and le is not None else 'PD or LE censored by optical Gamma; no gap inferred'))
        valid = [x for x in records if x['relative_gap'] is not None]
        pdle = [x for x in valid if x['class_first'] == 'PD+LE']
        selection[str(r)] = dict(all_f_inv_below_half=len(records), defined=len(valid), censored=len(records) - len(valid), distribution=stats([x['relative_gap'] for x in valid]), above_1e8=[x for x in valid if x['relative_gap'] > 1e-08], count_above_1e8=sum((x['relative_gap'] > 1e-08 for x in valid)), pd_plus_le_subset=dict(distribution=stats([x['relative_gap'] for x in pdle]), count_above_1e8=sum((x['relative_gap'] > 1e-08 for x in pdle))), rows=records)
    lookup = {(x['name'], x['r_over_each_parent_length']): x for x in rows}
    pairs = []
    for name in sorted({x['name'] for x in rows}):
        pair = [classify(lookup[name, r]['PD_pressure_pV'], lookup[name, r]['LE'], lookup[name, r]['micro']) for r in RADII]
        pairs.append(dict(name=name, classes=pair, same=pair[0] == pair[1]))
    return dict(model=rows[0]['model'], n_rows=len(rows), refined_thresholds_applied=require_complete, formula='Exact edge_center_atlas.classify order: PD-only if PD < min(LE,mu)*(1-tol); then LE <= mu*(1+tol) and LE <= PD*(1+tol); then micro. Censored PD/LE loads are +infinity; raw Bloch root retained in classifier even when near-Gamma censored for window reporting.', rounding_note='A gap > tol is not identical to PD < LE*(1-tol); boundary counts can differ. No rounding before classification.', radii=per_radius, selection_rule=selection, class_stability=dict(same=sum((x['same'] for x in pairs)), total=len(pairs), changed=[x for x in pairs if not x['same']]), scope=('Explicit net subset; counts are in n_rows. ' if names else 'All 1752 rows use the baseline PD thresholds; 238 rows use the refined Bloch thresholds. ') + 'LE uses the lower near-Gamma minimum when an exact acoustic-limit check confirms that direction. Selection-rule gaps require both uncensored thresholds; optical-censored pseudo-strains are listed but not counted as tested failing strains.')
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-only', action='store_true')
    ap.add_argument('--nets', nargs='+')
    args = ap.parse_args()
    out = build(not args.reference_only, args.nets)
    if args.reference_only:
        print({r: {'counts': [(x['tolerance'], x['classes']) for x in v['tolerances']], 'selection': {k: z for k, z in out['selection_rule'][r].items() if k != 'rows'}} for r, v in out['radii'].items()})
    else:
        publish('atlas/bloch_tolerance.json', out)
        print('DONE', {r: out['selection_rule'][r]['distribution'] for r in out['radii']})
