"""Empty-cache production checks; RCSR_CGD is required (no silent skip).

Checks graph bootstrap, both null and numeric
threshold checks, Pc constructor/onset, and no archive-side writes.
"""
from pathlib import Path
import argparse, json, os, subprocess, sys, time, uuid

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True, type=Path)
    a = p.parse_args()
    cache = a.cache.resolve()
    if cache.exists() and any(cache.iterdir()):
        p.error('--cache must be empty or absent')
    cache.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    c = root / 'computations'
    cgd = Path(os.environ.get('RCSR_CGD', str(c / 'rcsr3d_web.cgd'))).resolve()
    if not cgd.is_file():
        p.error('Set RCSR_CGD to rcsr3d_web.cgd')
    os.environ.update(K08_CACHE=str(cache), K08_OUTPUT=str(cache / 'outputs'), PYTHONDONTWRITEBYTECODE='1', RCSR_CGD=str(cgd))
    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(c))
    start = time.perf_counter()
    scripts = sorted(root.rglob('*.py'))
    for f in scripts:
        compile(f.read_text(encoding='utf-8-sig'), f.relative_to(root).as_posix(), 'exec')
    print('1. compile:', len(scripts), 'scripts, 0 failures', flush=True)
    import numpy as np
    from computations.common.sparse_frames import Frame2D
    f = Frame2D(np.array([[0, 1], [1, 0]]))
    p0 = f.hydro()
    vals, vec, res = f.eigen(count=1)
    ref = json.loads((c / 'truchet_2d/planted_pressure.json').read_text(encoding='utf-8'))['baseline']['pressure']
    pressure = float(vals[0] * p0)
    error = abs(pressure / ref - 1)
    assert error < 1e-08
    print(f'2. checkerboard: {pressure:.15e}; relative error {error:.3e}', flush=True)
    assert not (cache / 'atlas/graphs').exists()
    subprocess.run([sys.executable, '-B', str(c / 'atlas/run_atlas.py'), '--nets', 'apo', 'cco', '--workers', '1'], cwd=cache, check=True)
    rows = {(r['name'], r['r_over_each_parent_length']): r for r in json.loads((c / 'atlas/baseline_rows.json').read_text(encoding='utf-8'))}
    from computations.atlas.bootstrap import bootstrap
    graphs = bootstrap(['apo', 'cco'])
    worst = 0.0
    count = 0
    for graph in graphs:
        for radius in (0.02, 0.05):
            new = json.loads((cache / 'atlas/results' / f"{graph['name']}_{radius:.2f}.json").read_text(encoding='utf-8'))
            reference = rows[graph['name'], radius]
            assert new['graph_sha256'] == graph['graph_sha256']
            assert new['class_pressure_pV'] == reference['class_pressure_pV']
            for key in ('lambda_PD_pressure', 'lambda_LE_pressure', 'lambda_micro_pressure'):
                assert (new[key] is None) == (reference[key] is None), (graph['name'], key)
                if reference[key] is not None:
                    worst = max(worst, abs(new[key] / reference[key] - 1))
                    count += 1
    assert worst < 1e-09
    print(f'3. atlas: 2 graph identities, 4 classes, {count} thresholds; max relative error {worst:.3e}', flush=True)
    subprocess.run([sys.executable, '-B', str(c / 'trumof/mechanics/verify.py'), '--name', 'trumof_1x1x1_7', '--kind', 'soft'], cwd=cache, check=True)
    result = json.loads((cache / 'trumof/mechanics/results/trumof_1x1x1_7_soft.json').read_text(encoding='utf-8'))
    pc = result['objective']['first_pressure_GPa']
    assert abs(pc - 0.746024948) < 1e-08 and result['status'] == 'verified'
    from computations.trumof.mechanics.model.model import CG
    package = c / 'trumof/mechanics'
    reference = json.loads((package / 'inputs/continuation/summary_seed.json').read_text(encoding='utf-8'))
    app = next((z for z in reference['approximants'] if z['name'] == 'trumof_1x1x1_7'))
    par = json.loads((package / 'inputs/continuation/pcu_parameters.json').read_text(encoding='utf-8'))['soft']
    general = CG(np.array(app['positions']) * reference['rule']['a'], app['cell'], app['edges'], kr=par['C11_GPa'] * 13.0, kt=par['C44_GPa'] * 13.0 ** 3 / 4.0)
    assert np.any(general.th == np.pi)
    assert not np.any((general.th > np.pi - 1e-06) & (general.th < np.pi))
    state = np.load(cache / 'trumof/mechanics/states/trumof_1x1x1_7_soft.npz')
    tangent = general.tangent(state['x_PD'], float(state['p_PD']))[0]
    assert abs(np.linalg.eigvalsh(tangent)[0]) < 1e-07
    print(f"4. Pc soft: {pc:.12f} GPa; {result['objective']['first_class']}", flush=True)
    report = dict(status='passed', compiled_scripts=len(scripts), checkerboard_pressure=pressure, checkerboard_relative_error=error, atlas_graphs={g['name']: g['graph_sha256'] for g in graphs}, atlas_max_relative_error=worst, Pc_soft_GPa=pc, seconds=time.perf_counter() - start)
    (cache / 'smoke_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"ALL CHECKS PASSED ({report['seconds']:.2f} s)", flush=True)
if __name__ == '__main__':
    main()
