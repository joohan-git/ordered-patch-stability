"""Independent algebra and immutable-source checks, plus complete-row check."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.mesh_common import *
from computations.atlas.mesh_solver import affine_fast, setup
from computations.atlas.mesh_aggregate import collect, summarize, rel, THRESHOLDS

def verify_sources():
    manifest = read(ROOT.parent / 'SHA256SUMS.json')
    names = ['common/initial_stress.py', 'validation/beam_validation.py', 'atlas/edge_center_atlas.py', 'common/bloch_frame.py']
    frozen = {name: manifest['computations/' + name] for name in names}
    for name, digest in frozen.items():
        assert sha(ROOT / name) == digest, name
    return frozen


def validate_only():
    checks = dict(source_sha256=verify_sources(), affine_assembly_checks=[])
    for name, div in [('cco', 2), ('pcu', 8)]:
        frame = newframe(name, div)
        frame.prestress('stress')
        frame.affine_matrices()
        elastic, stress = frame.affineK.copy(), frame.affineH.copy()
        affine_fast(frame)
        errors = [float(la.norm(frame.affineK - elastic) / la.norm(elastic)), float(la.norm(frame.affineH - stress) / la.norm(stress))]
        assert max(errors) < 1e-12
        checks['affine_assembly_checks'].append(dict(name=name, div=div, relative_errors=errors))
    publish('atlas/mesh_source_check.json', checks)
    print('COMPUTE mesh affine checks passed', flush=True)
    return checks


def verify(out):
    c, rows, missing = collect()
    assert not missing
    assert len(rows) == 138 and len({x['name'] for x in rows}) == 138
    assert len(c['pdonly']) == 78 and len(c['controls']) == 60
    assert not set(c['pdonly']) & set(c['controls'])
    assert out['summary'] == summarize(rows)
    frozen = verify_sources()
    flat = [z for x in rows for z in x['meshes'].values()]
    for x in rows:
        for div, z in x['meshes'].items():
            assert z['div'] == int(div) and z['r'] == 0.02
            assert z['graph_sha256'] == graph(x['name'])['graph_sha256']
            assert z['class_pressure_pV'] == classify(z['PD'], z['LE'], z['micro'], 0.003)
            assert z['micro']['grid'] == (8 if x['group'] == 'PD-only' else 4)
            assert z['lambda_PD'] == (None if z['PD']['load'] is None else z['p0'] * z['PD']['load'])
            assert z['lambda_LE'] == (None if z['LE']['load'] is None else z['p0'] * z['LE']['load'])
            assert z['lambda_mu'] == (None if z['micro']['near_excluded_origin'] else z['p0'] * z['micro']['load'])
            assert all((t['num_threads'] == 1 for t in z['threadpools']))
            assert z['force_normalized_residual'] < 1e-08
            if x['group'] == 'PD-only':
                grid = z['bloch_detail']['grid_samples']
                assert len(grid) == z['micro']['points']
                assert z['micro']['load'] <= min((g['load'] for g in grid)) * (1 + 1e-10)
        for key in THRESHOLDS:
            assert x['relative_change_2_to_8'][key] == rel(x['meshes']['2'][key], x['meshes']['8'][key])
    algebra = []
    for name, div in [('cco', 2), ('pcu', 8)]:
        f = newframe(name, div)
        f.prestress('stress')
        f.affine_matrices()
        K = f.affineK.copy()
        H = f.affineH.copy()
        affine_fast(f)
        ek = float(la.norm(f.affineK - K) / la.norm(K))
        eh = float(la.norm(f.affineH - H) / la.norm(H))
        assert max(ek, eh) < 1e-12
        algebra.append(dict(name=name, div=div, affine_elastic_matrix_relative_error=ek, affine_initial_stress_matrix_relative_error=eh))
    benchmark = [read(p) for p in sorted(DEST.glob('atlas/mesh_solver_*_d*.json'))]
    failed_attempt_records = [read(p) for p in (DEST / 'errors').glob('*.json')]
    assert all(((DEST / 'results' / f"atlas/mesh_{e['name']}_d{e['div']}.json").exists() for e in failed_attempt_records))
    full = [dict(name=z['name'], div=z['div'], dense_validation=z['solver'].get('dense_validation', [])) for z in flat if z['solver']['method'].startswith('Full sparse')]
    dense_checks = read(DEST / 'atlas/mesh_dense_retry_check.json')
    for check in full:
        z = next((v for v in flat if v['name'] == check['name'] and v['div'] == check['div']))
        comparison = next((v for v in dense_checks['rows'] if v['name'] == z['name'] and v['div'] == z['div']))
        assert comparison['k'] == z['micro']['k'] and comparison['sparse'] == z['micro']['load']
        assert abs(comparison['relative_error']) < 1e-06 and z['lambda_mu'] is None
        assert classify(z['PD'], z['LE'], dict(z['micro'], load=comparison['dense'])) == z['class_pressure_pV']
        check['final_wavevector_dense_check'] = comparison
    refinements = [t for z in flat for t in z['bloch_detail'].get('refinements', [])]
    checks = dict(all_checks_pass=True, n_nets=len(rows), n_mesh_runs=len(flat), immutable_model_sha256=frozen, affine_assembly_checks=algebra, full_dense_sparse_benchmarks=benchmark, full_sparse_rows=full, failed_attempts=len(failed_attempt_records), all_failed_attempts_resolved=True, near_gamma_full_dense_sparse_max_relative_difference=dense_checks['max_relative_difference'], near_gamma_validation_note='Eight full-sparse retry minima remain Gamma-censored. Dense/sparse agreement is quantified at a separate 1e-6 relative tolerance because of near-Gamma conditioning; all discrepancies are below this tolerance, the threshold statistics exclude these values, and substituting the dense roots preserves every class.', local_refinements=len(refinements), local_refinements_at_iteration_cap=sum((not z['success'] for z in refinements)), max_preload_pressure_change_2_to_8=max((abs(x['meshes']['8']['p0'] / x['meshes']['2']['p0'] - 1) for x in rows)), source_sha256={s: sha(ROOT / s) for s in ['atlas/mesh_retry.py', 'atlas/mesh_verify.py', 'atlas/mesh_dense_check.py']})
    return checks

if __name__ == '__main__':
    validate_only()
