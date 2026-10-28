"""Parameterized single cases using the production solvers.

Each case computes its result in the selected cache and exports JSON.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, json, time
from computations.common.runtime import ROOT, output_dir, cache_dir

def plain(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, Path):
        return x.name
    raise TypeError(type(x).__name__)

def calculate(a):
    import numpy as np
    if a.case == 'atlas-check':
        from computations.atlas.check_common import frame, raw
        from computations.atlas.bloch_search import scan
        f = frame(a.name, a.radius)
        reference = raw(a.name, a.radius)
        z = scan(f, reference)
        z['force_normalized_preload_residual'] = float(f.pre['absolute_residual'] / np.linalg.norm(f.pre['full_element_forces']))
        from computations.atlas.check_common import ConsistentAtlas
        from computations.atlas.mesh_run import acoustic12
        at = ConsistentAtlas(f)
        pd = at.pd(True)
        le = acoustic12(at, pd, z['micro'], reference, at.le())
        z['refined_LE'] = le
        return z
    if a.case == 'mesh':
        from computations.atlas.mesh_run import run
        return run(a.name, a.div, 'PD-only' if a.name == 'cco' else 'control')
    if a.case == 'beam':
        import computations.validation.beam_validation as validation_beam_validation
        return validation_beam_validation.validate()
    if a.case in ('truchet', 'rule', 'compact3d', 'h2', 'ising'):
        import computations.common.cohorts as latest
        from computations.common.sparse_frames import Frame2D, Frame3D
        base = latest.baseline2d()
        motif = np.array([[0, 1], [1, 0]])
        if a.case == 'rule':
            from computations.truchet_2d.rules import scan
            return scan(np.tile(motif, (2, 2)), grid=4, refine=True)
        if a.case == 'h2':
            if a.R < 6 or a.L < a.R:
                raise ValueError('H2 fixed-width interior requires 6 <= R <= L; use --L 16 --R 8')
            from computations.prestress.robustness import run_case
            f = Frame2D(motif)
            f.hydro()
            return run_case(a.L, [a.R, a.R], a.seed, f, base['pressure'])
        if a.case == 'ising':
            from computations.truchet_2d.ising_sampling import sample, observables
            from computations.truchet_2d.correlated_cell import evaluate
            from computations.truchet_2d.compact_domains import largest_square, largest_disc
            sig, trace, ends = sample(a.J, a.L, a.seed, sweeps=a.sweeps)
            return dict(L=a.L, J=a.J, sweeps=a.sweeps, statistics=observables(sig, a.J), mechanics=evaluate(sig, base['pressure'], coupled=True, validate=True), largest_square=max(largest_square(sig == 1), largest_square(sig == -1)), largest_disc=max(largest_disc(sig == 1), largest_disc(sig == -1)))
        d = 3 if a.case == 'compact3d' else a.dimension
        pat = np.random.default_rng(441 + a.L).integers(0, 2, (a.L,) * d)
        lo = (a.L - a.R) // 2
        hi = lo + a.R
        if d == 3:
            b = json.loads((ROOT / 'frame_3d/periodic.json').read_text(encoding='utf-8'))['weakest']
            motif = np.array(b['pattern'])
        if a.R:
            ids = np.indices((a.R,) * d)
            pat[(slice(lo, hi),) * d] = motif[tuple((ids[i] % motif.shape[i] for i in range(d)))]
        if d == 2:
            from computations.truchet_2d.planted_domains import compact2d
            from computations.truchet_2d.rules import patch_pressure
            f = Frame2D(pat)
            p0 = f.hydro()
            vals, u, res = f.eigen(count=2)
            q, cres = compact2d(f, lo, hi)
            return dict(L=a.L, R=a.R, pressure=float(vals[0] * p0), ratio=float(vals[0] * p0 / base['pressure']), compact_ratio=float(q * p0 / base['pressure']), residual=res, compact_residual=cres, stress=patch_pressure(f, lo, hi))
        f = Frame3D(pat, zfactor=b['zfactor'])
        sup = f.compact([lo] * 3, [hi] * 3)
        pressure = -np.trace(f.pre['tile_stress'], axis1=1, axis2=2).reshape((a.L,) * 3) / 3
        s = float(pressure[(slice(lo, hi),) * 3].mean() / f.p0)
        z = dict(L=a.L, R=a.R, compact=sup, s_P=s, pressure_normalized_compact_ratio=sup['pressure'] / b['critical_pressure'] * s)
        if a.case == 'truchet':
            from computations.frame_3d.frame import sample
            z['full_sample'] = sample(pat, b)[0]
        return z
    if a.case == 'windows':
        import computations.prestress.window_feasibility as w
        w.main()
        return json.loads((w.OUT / 'prestress/windows.json').read_text(encoding='utf-8'))
    if a.case == 'cowley':
        import computations.selection_rule.cowley as w
        w.ROOT = output_dir()
        w.main()
        return json.loads((w.ROOT / 'selection_rule/cowley.json').read_text(encoding='utf-8'))
    if a.case == 'inclusion':
        from computations.common.sparse_frames import InclusionFrame
        f = InclusionFrame(a.name, a.L, 0.34, div=2)
        f.preload('pressure')
        vals, u, ep, res = f.variable(count=2)
        stress = f.pre['tile_stress'][f.patch_cells].mean(axis=0)
        return dict(name=a.name, L=a.L, R_fraction=0.34, pressure=float(vals[0] * f.p0), mean_interior_stress_tensor=stress, s_P=float(-np.trace(stress) / (3 * f.p0)), eigen_residual=res, mode=f.stats(u[:, 0], ep[:, 0]))
    if a.case == 'modes':
        from computations.inclusions.modes import run
        return run(a.name, L=a.L)
    if a.case == 'trumof-inputs':
        from computations.trumof.mechanics.model.rule import reconstruct, RAW, enumerate_rule
        from computations.trumof.extract import extract
        r = reconstruct()
        reference = os.getcwd()
        os.chdir(output_dir())
        try:
            enum = enumerate_rule(r)
        finally:
            os.chdir(reference)
        z = extract(RAW / 'science.ade5239_data_s6/1x1x1/trumof_1x1x1_7.cif', 1, r)
        z['file'] = 'Meekel2023:DataS6/1x1x1/trumof_1x1x1_7.cif'
        return dict(rule=r, enumeration=enum, Pc=z)
    if a.case == 'trumof-correlated':
        import computations.trumof.correlated as w
        r = json.loads((ROOT / 'trumof/rule.json').read_text(encoding='utf-8'))
        ap = json.loads((ROOT / 'trumof/approximants.json').read_text(encoding='utf-8'))
        qpc = next((z['q'] for z in ap if z['name'] == 'trumof_1x1x1_7'))
        z = w.mc_field(a.L, r, qpc, 1.8, a.seed, sweeps=a.sweeps)
        if z['energy'] != 0:
            raise RuntimeError('Annealing did not satisfy the local rule; increase --sweeps')
        z['name'] = f'correlated_L{a.L}_{a.seed}'
        m = w.sample_model(z, r)
        onset = w.onset(m, z['name'])
        return dict(sample=z, onset=onset)
    raise ValueError(a.case)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', choices=['atlas-check', 'mesh', 'beam', 'truchet', 'rule', 'compact3d', 'h2', 'ising', 'windows', 'cowley', 'inclusion', 'modes', 'trumof-inputs', 'trumof-correlated'])
    p.add_argument('--name', default='cco')
    p.add_argument('--radius', type=float, default=0.02)
    p.add_argument('--div', type=int, default=2)
    p.add_argument('--L', type=int, default=8)
    p.add_argument('--R', type=int, default=4)
    p.add_argument('--dimension', type=int, choices=[2, 3], default=2)
    p.add_argument('--J', type=float, default=0.4)
    p.add_argument('--sweeps', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    start = time.perf_counter()
    z = calculate(a)
    dest = output_dir() / ('case_' + a.case + '.json')
    dest.write_text(json.dumps(dict(parameters=vars(a), seconds=time.perf_counter() - start, result=z), indent=2, default=plain, allow_nan=False), encoding='utf-8')
    print('SAVED', dest, 'seconds', round(time.perf_counter() - start, 3), flush=True)
if __name__ == '__main__':
    main()
