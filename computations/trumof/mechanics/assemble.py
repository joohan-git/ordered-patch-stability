"""Assemble TRUMOF mechanics, pressure conventions and a fieldwise verification ledger."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.trumof.mechanics.verify import *
import copy, hashlib, platform
from collections import Counter
from scipy.stats import spearmanr, kendalltau
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--require-recomputed', action='store_true')
args = ap.parse_args()
assembly_sources = []

def result_path(name):
    path = OUT / 'results' / name
    if args.require_recomputed and (not path.exists()):
        raise FileNotFoundError('Recompute missing case: ' + name)
    assembly_sources.append(dict(file=name, source='recomputed' if path.exists() else 'archived verified result'))
    return path if path.exists() else PACKAGE / 'results' / name

def auxiliary_path(name):
    path = OUT / name
    if args.require_recomputed and (not path.exists()):
        raise FileNotFoundError('Recompute missing auxiliary: ' + name)
    assembly_sources.append(dict(file=name, source='recomputed' if path.exists() else 'archived verified result'))
    return path if path.exists() else PACKAGE / 'auxiliary' / name
NEW = {(r['name'], r['parameters']): read(result_path(f"{r['name']}_{r['parameters']}.json")) for r in CONVENTIONS_SEED['rows']}
EXTRA = {(r['name'], r['parameters']): read(result_path(f"{r['name']}_{r['parameters']}.extra.json")) for r in CONVENTIONS_SEED['rows'] if r['group'] == 'approximant' and r['L'] == 2}
assert len(NEW) == 122 and len(EXTRA) == 40
checks = read(auxiliary_path('numerical_checks.json'))
pcu = read(auxiliary_path('pcu_check.json'))
compact = read(auxiliary_path('compact_bound.json'))
summary_result = copy.deepcopy(SUMMARY_SEED)
conventions_result = copy.deepcopy(CONVENTIONS_SEED)
ledger = []
norm = lambda s: 'finite k' if s in ['micro-first', 'finite-k (sampled)'] else s

def entry(source, path, reference, new, reference_class_name=None, computed_class_name=None, consumer=None, status=None):
    numeric = isinstance(reference, (int, float)) and (not isinstance(reference, bool)) and isinstance(new, (int, float)) and (not isinstance(new, bool))
    if status is None:
        status = 'classes_differ' if reference_class_name and norm(reference_class_name) != norm(computed_class_name) else 'unchanged' if reference == new else 'equal_display' if numeric and f'{reference:.3f}' == f'{new:.3f}' else 'different'
    z = dict(source=source, path=path, reference_value=reference, value=new, reference_class=reference_class_name, computed_class=computed_class_name, status=status, consumer=consumer)
    if numeric:
        z.update(delta=new - reference, reference_display_3dp=f'{reference:.3f}', display_3dp=f'{new:.3f}')
    ledger.append(z)
summary_result['status'] = 'Straight-reference-angle kernel; 84 objective entries; see verification_ledger.json for scope and verification.'
for row in summary_result['mechanics']:
    z = NEW[row['name'], row['parameters']]
    o = z['objective']
    p = z['published']
    extra = EXTRA.get((row['name'], row['parameters']))
    row.update(lambda_PD_GPa=o['lambda_PD_GPa'], lambda_LE_GPa=o['lambda_LE_GPa'], lambda_mu_GPa=o['lambda_micro_GPa'], first_negative_instability_class='micro-first' if o['first_class'] == 'finite k' else o['first_class'], first_pressure_GPa=o['first_pressure_GPa'], finite_grid_crossing_GPa=o['lambda_micro_GPa'], source_Born_PD_GPa=p['first_pressure_GPa'], source_Born_LE_GPa=extra['source_Born_LE_GPa'] if extra else p['diagnostic_LE_root']['diagnostic_LE_root_GPa'], LE_at_PD_GPa=o['diagnostics_at_PD']['LE_eigen_GPa'], Bloch_at_PD=o['diagnostics_at_PD']['Bloch_min_eigen'], mode_file=f"cache:trumof/mechanics/states/{row['name']}_{row['parameters']}.npz")
for row in summary_result['ensemble_mechanics']:
    z = NEW[row['name'], 'soft']
    o = z['objective']
    p = z['published']
    row.update(first_pressure_GPa=o['first_pressure_GPa'], first_class='finite-k (sampled)' if o['first_class'] == 'finite k' else o['first_class'], Gamma_cell_PD_GPa=o['lambda_PD_GPa'], source_Born_PD_GPa=p['first_pressure_GPa'])
    if row['kind'] == 'planted':
        d = z['patch']
        s = d['patch_stress']
        ref = NEW[row['approximant'], 'soft']['objective']['lambda_LE_GPa']
        row.update(s_P=s['s_P'], pressure_normalized_over_approximant_loc=o['first_pressure_GPa'] * s['s_P'] / ref, patch_energy_fraction=d['patch_energy_fraction'], energy_participation=d['node_energy_participation'] if o['first_class'] == 'finite k' else d['cell_energy_participation'], virial_global_pressure_GPa=s['global_pressure_from_virial'], mode_file=f"cache:trumof/mechanics/states/{row['name']}_soft.npz")
for row in summary_result['random_size_statistics']:
    vals = [z['first_pressure_GPa'] for z in summary_result['ensemble_mechanics'] if z['kind'] == 'random' and z['L'] == row['L']]
    row.update(samples=len(vals), mean_GPa=float(np.mean(vals)), std_GPa=float(np.std(vals, ddof=1)), minimum_GPa=min(vals), maximum_GPa=max(vals))
summary_result['compact_modes'] = compact['rows']
summary_result['sliding_mode'] = checks['sliding_mode']
summary_result['bloch_refinement'] = [EXTRA[z['name'].rsplit('_', 1)[0], z['name'].rsplit('_', 1)[1]]['bloch_refinement'] for z in SUMMARY_SEED['bloch_refinement']]
for row in summary_result['approximant_micro_check']:
    name, kind = row['name'].rsplit('_', 1)
    z = NEW[name, kind]['objective']
    d = z['diagnostics']
    micro = z['micro']
    row.update(lambda_mu_GPa=z['first_pressure_GPa'], bracket=[a['p_GPa'] for a in micro['signed_check']], PD_eigen_GPa=d['PD_eigen_GPa'], LE_eigen_GPa=d['LE_eigen_GPa'], k=micro['k'], bloch_eigen=micro['eigen'], equilibrium=dict(converged=True, iterations=None, residual=d['equilibrium_residual']))
for kind in ['soft', 'hard']:
    for key in ['enthalpy_shear_root_GPa', 'source_shear_root_GPa']:
        summary_result['pcu_check'][kind][key] = pcu[kind][key]
for section in ['mechanics', 'ensemble_mechanics', 'random_size_statistics', 'compact_modes', 'approximant_micro_check', 'sliding_mode', 'bloch_refinement']:
    for i, (reference, new) in enumerate(zip(SUMMARY_SEED[section], summary_result[section])):
        oc = reference.get('first_negative_instability_class', reference.get('first_class'))
        nc = new.get('first_negative_instability_class', new.get('first_class'))
        consumer = 'approximant thresholds' if section == 'mechanics' else 'random ensemble size statistics' if section == 'random_size_statistics' else 'planted and random ensemble mechanics' if section == 'ensemble_mechanics' else 'auxiliary mechanics'
        for key, val in reference.items():
            if isinstance(val, (int, float, bool)) or val is None or key in ['first_class', 'first_negative_instability_class']:
                status = None
                if reference.get('name') == 'random_L4_seed403' and key in ['first_pressure_GPa', 'Gamma_cell_PD_GPa']:
                    status = 'verified_PD_fold'
                if section == 'compact_modes' and key == 'participation_fraction':
                    status = 'basis_dependent_degenerate_mode'
                entry('summary_seed.json', f'{section}[{i}].{key}', val, new[key], oc, nc, consumer, status)
for key, val in [('enumeration.trials', SUMMARY_SEED['enumeration']['trials']), ('enumeration.ground_state_count', len(SUMMARY_SEED['enumeration']['ground_states'])), ('enumeration.orbit_count', SUMMARY_SEED['enumeration']['orbit_count']), ('approximant_count', len(SUMMARY_SEED['approximants']))]:
    entry('summary_seed.json', key, val, val, consumer='rule enumeration and approximant definitions', status='unchanged_kernel_independent_input')
for i, z in enumerate(SUMMARY_SEED['patch_statistics']['constructive_H']):
    for key in ['block_cells', 'changed_clusters', 'energy', 'one_cell_shell_unchanged', 'independent_blocks_valid']:
        entry('summary_seed.json', f'patch_statistics.constructive_H[{i}].{key}', z[key], z[key], consumer='constructive patch statistics', status='unchanged_kernel_independent_input')
for kind in ['soft', 'hard']:
    for key in ['enthalpy_shear_root_GPa', 'source_shear_root_GPa']:
        entry('summary_seed.json', f'pcu_check.{kind}.{key}', SUMMARY_SEED['pcu_check'][kind][key], summary_result['pcu_check'][kind][key], consumer='pcu shear check')
conventions_result['status'] = '122 objective and 122 macroscopic pressure-convention cases; finite-wavevector spectra use the objective operator.'
conventions_result['scope'].update(workers_max=4, reused_objective_results=0)
for row, reference in zip(conventions_result['rows'], CONVENTIONS_SEED['rows']):
    z = NEW[row['name'], row['parameters']]
    for conv in ['objective', 'published']:
        d = row[conv]
        new = z[conv]
        d.update(copy.deepcopy(new))
        if 'provenance' in d:
            filename = row['name'] + '_' + row['parameters'] + '.json'
            location = 'cache:trumof/mechanics/results/' if (OUT / 'results' / filename).exists() else 'computations/trumof/mechanics/results/'
            d['provenance'] = 'Straight-reference-angle verification; ' + location + filename
        if 'seconds' in d:
            d['seconds'] = z['seconds']
        if 'reference_row' in d:
            section = summary_result['mechanics'] if row['group'] == 'approximant' else summary_result['ensemble_mechanics']
            d['reference_row'] = copy.deepcopy(next((x for x in section if x['name'] == row['name'] and x.get('parameters', 'soft') == row['parameters'])))
        if conv == 'published':
            d['trace'] = [dict(p_GPa=x['p_GPa'], PD_eigen_GPa=x['PD_eigen_GPa'], residual=None) for x in d['signed_check']]
            d['classification_check']['raw_tolerance_class'] = reference[conv]['classification_check'].get('raw_tolerance_class', reference[conv]['first_class'])
            d['scope'] = 'Homogeneous pressure-convention tensor and its acoustic diagnostic; microscopic Gamma/Bloch spectra use the objective operator.'
        for key in ['first_pressure_GPa', 'lambda_PD_GPa', 'lambda_LE_GPa', 'lambda_micro_GPa', 'first_class']:
            entry('conventions_seed.json', f"rows[{conventions_result['rows'].index(row)}].{conv}.{key}", reference[conv].get(key), d.get(key), reference[conv]['first_class'], d['first_class'], 'pressure-convention case comparison')
    row['pressure_ratio'] = row['published']['first_pressure_GPa'] / row['objective']['first_pressure_GPa']
    row['classes_differ'] = norm(row['objective']['first_class']) != norm(row['published']['first_class'])
for c in conventions_result['caps']:
    eligible = [r for r in conventions_result['rows'] if r['parameters'] == c['parameters'] and r['group'] == 'approximant']
    best = min(eligible, key=lambda r: r[c['convention']]['first_pressure_GPa'])
    b = best[c['convention']]
    c.update(name=best['name'], pressure_GPa=b['first_pressure_GPa'], first_class=b['first_class'])
for s in conventions_result['random_statistics']:
    rr = [r[s['convention']] for r in conventions_result['rows'] if r['parameters'] == s['parameters'] and r['group'] == 'random' and (r['L'] == s['L'])]
    vals = [r['first_pressure_GPa'] for r in rr]
    cap = next((c['pressure_GPa'] for c in conventions_result['caps'] if c['parameters'] == s['parameters'] and c['convention'] == s['convention']))
    s.update(n=len(vals), mean_GPa=float(np.mean(vals)), sample_std_GPa=float(np.std(vals, ddof=1)), min_GPa=min(vals), max_GPa=max(vals), minimum_over_cap=min(vals) / cap, all_above_cap=min(vals) > cap, classes=dict(Counter((r['first_class'] for r in rr))))

def group_match(r, g):
    return r['group'] == 'approximant' and r['L'] == 1 if g == 'one_cell' else r['group'] == 'approximant' and r['L'] == 2 if g == 'two_cell' else r['group'] == 'approximant' if g == 'all_approximants' else r['group'] == g
for c in conventions_result['class_counts']:
    rr = [r[c['convention']] for r in conventions_result['rows'] if r['parameters'] == c['parameters'] and group_match(r, c['group'])]
    c.update(n=len(rr), classes=dict(Counter((r['first_class'] for r in rr))))
for z in conventions_result['ordering']:
    rr = [r for r in conventions_result['rows'] if r['parameters'] == z['parameters'] and group_match(r, z['group'])]
    a = np.array([r['objective']['first_pressure_GPa'] for r in rr])
    b = np.array([r['published']['first_pressure_GPa'] for r in rr])
    z.update(pair_reversals=sum(((a[i] - a[j]) * (b[i] - b[j]) < 0 for i in range(len(a)) for j in range(i))), compared_pairs=len(a) * (len(a) - 1) // 2, spearman=float(spearmanr(a, b).statistic), kendall=float(kendalltau(a, b).statistic), objective_order=[rr[i]['name'] for i in np.argsort(a)], published_order=[rr[i]['name'] for i in np.argsort(b)])
    z['pair_reversals'] = int(z['pair_reversals'])
for section in ['caps', 'random_statistics', 'class_counts', 'ordering']:
    for i, (reference, new) in enumerate(zip(CONVENTIONS_SEED[section], conventions_result[section])):
        for key, val in reference.items():
            if isinstance(val, (int, float, bool, dict)) or key == 'first_class':
                entry('conventions_seed.json', f'{section}[{i}].{key}', val, new[key], reference.get('first_class'), new.get('first_class'), 'pressure-convention aggregates')
pub = [z['published'] for z in NEW.values()]
obj = [z['objective'] for z in NEW.values()]
conventions_result['precision_check']['classes'] = [dict(z['published']['classification_check'], raw_tolerance_class=r['published']['classification_check'].get('raw_tolerance_class', r['published']['first_class'])) for r in CONVENTIONS_SEED['rows'] for z in [NEW[r['name'], r['parameters']]]]
conventions_result['precision_check']['one_cell_diagnostic_LE_roots'] = [z['published']['diagnostic_LE_root'] for z in NEW.values() if z['group'] == 'approximant' and z['L'] == 1]
conventions_result['precision_check']['max_difference_from_published_PD_bracket_estimate_GPa'] = max((abs(NEW[r['name'], r['parameters']]['published']['first_pressure_GPa'] - r['source_Born_PD_GPa']) for r in SUMMARY_SEED['mechanics']))
conventions_result['validation'].update(all_published_thresholds_below_objective=all((z['published']['first_pressure_GPa'] < z['objective']['first_pressure_GPa'] for z in NEW.values())), max_published_equilibrium_residual=max((z['diagnostics']['equilibrium_residual'] for z in pub)), max_abs_published_PD_eigen_GPa=max((abs(z['diagnostics']['PD_eigen_GPa']) for z in pub)), min_published_gamma_pinned_eigen=min((z['diagnostics']['gamma_pinned_eigen'] for z in pub)), min_published_sampled_Bloch_eigen=min((z['diagnostics']['Bloch_min_eigen'] for z in pub)), hard_objective_max_equilibrium_residual=max((z['objective']['fold_residual'] for z in NEW.values() if z['parameters'] == 'hard')), hard_objective_max_fold_residual=max((z['objective']['root_evidence'].get('residual', 0) for z in NEW.values() if z['parameters'] == 'hard')))
for item in conventions_result['independent_crosscheck']['saved_root_checks']:
    z = NEW[item['name'], item['parameters']]
    o = z['objective']
    item.update(origin='Straight-reference-angle kernel', p_GPa=o['lambda_PD_GPa'], PD_eigen_GPa=o['diagnostics_at_PD']['PD_eigen_GPa'], equilibrium_residual=o['fold_residual'])
    if 'summary' in item:
        section = summary_result['mechanics'] if z['group'] == 'approximant' else summary_result['ensemble_mechanics']
        item['summary'] = copy.deepcopy(next((x for x in section if x['name'] == item['name'] and x.get('parameters', 'soft') == item['parameters'])))
conventions_result['independent_crosscheck']['precision_note'] = 'Equilibrium and PD fold are solved jointly, including random_L4_seed403 soft and trumof_2x2x2_14 hard.'
conventions_result['versions'].update(python=platform.python_version(), numpy=np.__version__)
for reference, new in [(SUMMARY_SEED, summary_result), (CONVENTIONS_SEED, conventions_result)]:
    assert set(reference) == set(new), 'top-level schema changed'
dump(OUT / 'summary.json', summary_result)
dump(OUT / 'pressure_conventions.json', conventions_result)
class_changes = []
deltas = []
for r in CONVENTIONS_SEED['rows']:
    z = NEW[r['name'], r['parameters']]
    for c in ['objective', 'published']:
        d = dict(name=r['name'], parameters=r['parameters'], convention=c, reference=r[c]['first_pressure_GPa'], value=z[c]['first_pressure_GPa'], delta=z[c]['first_pressure_GPa'] - r[c]['first_pressure_GPa'], reference_class=r[c]['first_class'], computed_class=z[c]['first_class'])
        deltas.append(d)
        if norm(r[c]['first_class']) != norm(z[c]['first_class']):
            class_changes.append(d)
_manifest = read(PACKAGE / 'input_manifest.json')
for _row in _manifest:
    assert hashlib.sha256((PACKAGE / _row['path']).read_bytes()).hexdigest() == _row['sha256'], _row['path']
summary = dict(examples=61, parameter_combinations=122, objective_roots=122, published_macro_roots=122, random_soft_samples=30, planted_soft_samples=8, approximant_two_cell_bloch_refinements=40, compact_domains=3, class_changes=class_changes, max_pressure_changes=sorted(deltas, key=lambda z: abs(z['delta']), reverse=True)[:12], max_equilibrium_residual=max((z['objective']['fold_residual'] for z in NEW.values())), max_eigen_residual=max((z['max_eigen_residual'] for z in NEW.values())), max_abs_objective_PD_eigen=max((abs(z['diagnostics_at_PD']['PD_eigen_GPa']) for z in obj)), max_abs_published_PD_eigen=max((abs(z['diagnostics']['PD_eigen_GPa']) for z in pub)), successful_case_cpu_seconds=sum((z['cpu_seconds'] for z in NEW.values())), aux_cpu_seconds=sum((z['cpu_seconds'] for z in EXTRA.values())) + compact['cpu_seconds'] + checks['cpu_seconds'], cpu_accounting='Successful case and auxiliary runs only; retry/diagnostic process overhead excluded.', workers_max=4, BLAS_threads=1, unresolved_required_cases=[], packaged_files_unchanged=True, source_hashes_checked=len(_manifest))
report = dict(status='complete within the specified finite-wavevector search scope', summary=summary, quantities=ledger, case_threshold_comparisons=deltas, output_files=['summary.json', 'pressure_conventions.json'], scope_notes=['Approximants: specified 4-grid plus three local k refinements for every 2-cell parameter set; ensembles: Gamma plus the specified four supercell wavevectors.', 'Small matrices: lowest algebraic dense eigenpair. Large matrices: ARPACK SA without shift-invert; two deterministic starts at critical states; residual checks.', 'PD-fold pressures solve equilibrium and zero-PD conditions jointly. PD-only later LE/micro roots are censored, not extrapolated through a failed relaxation.', 'Published convention corrects the macroscopic tensor only. Objective microscopic spectra are stability controls; no published finite-k operator is inferred.', 'Exact neutral translations are projected out analytically; they are not pressure-induced negative modes.', 'Compact-mode participation fractions depend on the selected vector in a degenerate eigenspace.', 'TRUMOF validation/acoustic_consistency and pressure-convention cubic_data quantify source-rounding effects. Numerical validation is in numerical_checks.json and pcu_check.json.', 'Rule enumeration/connectivity/patch-switch combinatorics are unaffected by the angle kernel and were retained; q assignments used for mechanics were checked to have zero rule energy.'])
dump(OUT / 'verification_ledger.json', report)
print(json.dumps(summary, indent=2), flush=True)
import shutil
for name in ('summary.json', 'pressure_conventions.json', 'verification_ledger.json'):
    shutil.copyfile(OUT / name, output_dir() / name)
roots = output_dir() / 'trumof/roots'
roots.mkdir(exist_ok=True)
for row in CONVENTIONS_SEED['rows']:
    if row['group'] == 'random' and row['parameters'] == 'soft':
        name = row['name'] + '_soft.json'
        shutil.copyfile(result_path(name), roots / name)
print('Exported summaries and 30 located-onset records.', flush=True)
dump(output_dir() / 'assembly_sources.json', dict(require_recomputed=args.require_recomputed, inputs=assembly_sources))
