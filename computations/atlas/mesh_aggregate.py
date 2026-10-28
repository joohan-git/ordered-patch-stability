"""Aggregate immutable row checkpoints, with explicit censored denominators."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.mesh_common import *
from collections import Counter
import argparse
THRESHOLDS = ['lambda_PD', 'lambda_LE', 'lambda_mu']

def rel(a, b):
    return None if a is None or b is None else float(b / a - 1)

def summarize(rows):
    thresholds = {}
    for key in THRESHOLDS:
        valid = [(x['name'], x['relative_change_2_to_8'][key]) for x in rows if x['relative_change_2_to_8'][key] is not None]
        vals = [abs(v) for n, v in valid]
        st = stats(vals)
        if valid:
            n, v = max(valid, key=lambda x: abs(x[1]))
            st.update(max_net=n, max_signed_change=v)
        st['undefined_pair_count'] = len(rows) - len(valid)
        st['censor_state_changes'] = [x['name'] for x in rows if (x['meshes']['2'][key] is None) != (x['meshes']['8'][key] is None)]
        thresholds[key] = st
    changed = [x['name'] for x in rows if x['classes']['2'] != x['classes']['8']]
    anychanged = [x['name'] for x in rows if len(set(x['classes'].values())) > 1]
    exits = [x['name'] for x in rows if x['group'] == 'PD-only' and any((not y.startswith('PD-only') for y in x['classes'].values()))]
    entries = [x['name'] for x in rows if x['group'] == 'control' and any((y.startswith('PD-only') for y in x['classes'].values()))]
    return dict(n_nets=len(rows), thresholds=thresholds, class_counts={str(d): dict(Counter((x['classes'][str(d)] for x in rows))) for d in [2, 4, 8]}, class_changes_2_to_8_count=len(changed), class_changes_2_to_8=changed, class_changes_any_mesh_count=len(anychanged), class_changes_any_mesh=anychanged, baseline_pdonly_exits_count=len(exits), baseline_pdonly_exits=exits, baseline_other_entries_count=len(entries), baseline_other_entries=entries)

def collect():
    c = cohort()
    rows = []
    missing = []
    for n in c['pdonly'] + c['controls']:
        mesh = {}
        for d in [2, 4, 8]:
            p = DEST / 'results' / f'atlas/mesh_{n}_d{d}.json'
            if p.exists():
                mesh[str(d)] = read(p)
            else:
                missing.append((n, d))
        if len(mesh) != 3:
            continue
        rows.append(dict(name=n, group='PD-only' if n in c['pdonly'] else 'control', baseline_class=c['baseline_classes'][n], classes={d: x['class_pressure_pV'] for d, x in mesh.items()}, meshes=mesh, relative_change_definition='lambda(8)/lambda(2)-1; null unless both thresholds are uncensored', relative_change_2_to_8={k: rel(mesh['2'][k], mesh['8'][k]) for k in THRESHOLDS}, relative_change_4_to_8={k: rel(mesh['4'][k], mesh['8'][k]) for k in THRESHOLDS}, diagnostic_bound_relative_change_2_to_8={k: rel(mesh['2'][k], mesh['8'][k]) for k in ['lambda_PD_bound', 'lambda_LE_bound', 'lambda_mu_sampled_upper']}))
    return (c, rows, missing)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--publish', action='store_true')
    args = ap.parse_args()
    c, rows, missing = collect()
    summary = summarize(rows)
    print(json.dumps(dict(complete=len(rows), missing=len(missing), summary=summary), indent=2), flush=True)
    if not args.publish:
        return
    assert not missing, missing
    flat = [y for x in rows for y in x['meshes'].values()]
    revised = {x['name']: x for x in read(ROOT / 'atlas/bloch_checks.json')['rows'] if x['r'] == 0.02}
    comparisons = []
    for x in rows:
        reference = raw(x['name'], 0.02)
        update = revised.get(x['name'])
        base = dict(lambda_PD=reference['lambda_PD_pressure'], lambda_LE=reference['lambda_LE_pressure'], lambda_mu=reference['lambda_micro_pressure'])
        if update:
            base.update(lambda_PD=update['lambda_PD'], lambda_LE=update['lambda_LE'], lambda_mu=update['lambda_mu'])
        comparisons.append(dict(name=x['name'], baseline_class=x['baseline_class'], recomputed_d2_class=x['classes']['2'], baseline_thresholds=base, recomputed_d2_threshold_relative_change={k: rel(base[k], x['meshes']['2'][k]) for k in THRESHOLDS}))
    files = ['common/initial_stress.py', 'validation/beam_validation.py', 'common/bloch_frame.py', 'atlas/edge_center_atlas.py', 'common/atlas_support.py', 'atlas/run_atlas.py', 'atlas/check_common.py', 'atlas/bloch_tolerance.json', 'atlas/bloch_checks.json', 'atlas/mesh_common.py', 'atlas/mesh_solver.py', 'atlas/mesh_condense.py', 'atlas/mesh_run.py', 'atlas/mesh_aggregate.py']
    out = dict(model=raw('cco', 0.02)['model'], cohort=c, n_nets=138, n_mesh_runs=414, rows=rows, summary=summary, group_summaries={g: summarize([x for x in rows if x['group'] == g]) for g in ['PD-only', 'control']}, baseline_baseline_comparison=comparisons, baseline_class_changes=[x for x in comparisons if x['baseline_class'] != x['recomputed_d2_class']], verification=dict(max_condensed_reference_root_relative_error=max((abs(v['relative_error']) for x in flat for v in x['validation'])), max_final_bloch_sparse_relative_error=max((abs(x['final_bloch_sparse_relative_error']) for x in flat)), max_sparse_scaled_eigen_residual=max((x['sparse_validation']['max_scaled_residual'] for x in flat)), max_force_normalized_preload_residual=max((x['force_normalized_residual'] for x in flat)), threadpools=POOLS, workers=2), protocol=dict(pdonly='All 8^3 symmetry/time-reversal representatives including optical Gamma; six lowest non-Gamma grid seeds, fixed baseline Bloch and small-k LE seeds; Nelder-Mead maxiter 220; exclusion .00625, censor .00675; additional acoustic-direction checks', controls='All 4^3 representatives; two lowest non-Gamma grid seeds; Nelder-Mead maxiter 90; exclusion .025, censor .027; baseline LE search', arithmetic='Same full Kirchhoff-Hermite FE matrices. Exact load-dependent spectral elimination of strut-interior DOFs for Bloch first roots; dense full internal spectral Schur for PD and LE. Full sparse eigenproblem independently checks three fixed wavevectors and the final minimizing wavevector at every net/mesh.', threshold_units='lambda = p/E_s = initial hydrostatic p0 times proportional load factor; E_s=1', censor='PD and LE null when at optical Gamma within 2e-6; micro null in the specified near-Gamma censor region. Raw bounds and sampled micro upper bounds are separate diagnostics, never substituted into threshold statistics.', limitations='Grid plus local refinement is a numerical search, not an interval-certified global minimum. Only r/l=.02 and this fixed 138-net cohort are checked.'), source_sha256={s: sha(ROOT / s) for s in files}, compute_seconds_sum=sum((x['seconds'] for x in flat)))
    publish('atlas/mesh.json', out)
if __name__ == '__main__':
    main()
