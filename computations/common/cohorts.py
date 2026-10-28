"""Compute the periodic, random, rule and inclusion beam cohorts."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import argparse, json, time, gc, shutil
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.optimize import curve_fit
from computations.common.sparse_frames import Frame2D, Frame3D, InclusionFrame
from computations.common.initial_stress import MODEL
import computations.truchet_2d.frame as truchet_2d_frame, computations.frame_3d.frame as frame_3d_frame, computations.truchet_2d.rules as truchet_2d_rules
from computations.validation.beam_validation import ConsistentFrame, ConsistentAtlas, graph_data
from computations.atlas.edge_center_atlas import classify
ROOT = _DATA_ROOT
from computations.common.runtime import cache_dir, output_dir
WORK = cache_dir('truchet')
truchet_2d_frame.Frame2D = Frame2D
frame_3d_frame.Frame3D = Frame3D
truchet_2d_rules.Frame2D = Frame2D
truchet_2d_rules.hydro = lambda f: f.hydro()

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def dump(name, obj):
    obj['model'] = MODEL
    tmp = WORK / (name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(WORK / name)

def resume(name, default):
    for path in [WORK / name]:
        if path.exists():
            try:
                out = json.loads(path.read_text(encoding='utf-8'))
            except ValueError:
                continue
            if out.get('model') != MODEL:
                raise ValueError('Model mismatch: ' + name)
            return out
    return default()

def baseline2d():
    f = Frame2D(np.array([[0, 1], [1, 0]]))
    strain = float(f.eigen(count=1)[0][0])
    p0 = f.hydro()
    pressure = float(f.eigen(count=1)[0][0] * p0)
    return dict(strain=strain, pressure=pressure)

def compact2d(f, lo, hi):
    ids = np.flatnonzero(np.repeat(np.all((f.pos > lo + 1e-09) & (f.pos < hi - 1e-09), axis=1), 3))
    K = f.K[ids, :][:, ids].tocsc()
    H = f.H[ids, :][:, ids].tocsc()
    fac = sla.splu(K)
    ev, vec = sla.eigsh(H, k=1, M=K, Minv=sla.LinearOperator(K.shape, matvec=fac.solve), which='LA', tol=1e-09, v0=np.random.default_rng(932).normal(size=len(ids)))
    val = float(1 / ev[0])
    kv = K @ vec[:, 0]
    hv = H @ vec[:, 0]
    return (val, float(la.norm(kv - val * hv) / (la.norm(kv) + val * la.norm(hv))))

def periodic2d(limit=None):
    computed = 0
    reference = read('truchet_2d/inputs/periodic_patterns.json')
    out = resume('truchet_2d/periodic.json', lambda: dict(baseline=baseline2d(), runs=[]))
    for j, r in enumerate(reference['runs']):
        if any((x['pattern'] == r['pattern'] for x in out['runs'])):
            continue
        q = truchet_2d_frame.scan_periodic(np.array(r['pattern']), grid=reference['grid'])
        out['runs'].append(dict(pattern=q['pattern'], critical=q['critical'], k=q['k_supercell'], baseline_critical=r['critical'], ratio_to_baseline=q['critical'] / r['critical']))
        dump('truchet_2d/periodic.json', out)
        print('COMPUTE periodic2d', j + 1, len(reference['runs']), q['critical'], flush=True)
        computed += 1
        if limit and computed >= limit:
            return
    out['weakest'] = min(out['runs'], key=lambda r: r['critical'])
    dump('truchet_2d/periodic.json', out)

def samples2d():
    out = resume('truchet_2d/random_samples.json', lambda: dict(baseline=baseline2d(), runs=[]))
    base = out['baseline']
    for j in range(20):
        if any((x['sample'] == j for x in out['runs'])):
            continue
        start = time.perf_counter()
        p = np.random.default_rng(np.random.SeedSequence([20261025, 32, j])).integers(0, 2, (32, 32))
        f = Frame2D(p)
        vals, v, res = f.eigen(count=2)
        out['runs'].append(dict(sample=j, load=float(vals[0]), ratio=float(vals[0] / base['strain']), residual=res, seconds=time.perf_counter() - start))
        dump('truchet_2d/random_samples.json', out)
        print('samples2d', j, out['runs'][-1], flush=True)
        del f
        gc.collect()
    out['mean_ratio'] = float(np.mean([r['ratio'] for r in out['runs']]))
    dump('truchet_2d/random_samples.json', out)

def patches2d():
    out = resume('truchet_2d/planted_pressure.json', lambda: dict(baseline=baseline2d(), runs=[]))
    base = out['baseline']
    for L, R in [(64, 16), (64, 24), (64, 32), (64, 40), (64, 48), (96, 64)]:
        if any((x['L'] == L and x['R'] == R for x in out['runs'])):
            continue
        start = time.perf_counter()
        p = np.random.default_rng(441 + L).integers(0, 2, (L, L))
        lo = (L - R) // 2
        hi = lo + R
        p[lo:hi, lo:hi] = np.indices((R, R)).sum(axis=0) % 2
        f = Frame2D(p)
        row = dict(L=L, R=R)
        if R in [48, 64]:
            q, res = compact2d(f, lo, hi)
            row.update(strain_compact_ratio=q / base['strain'], strain_compact_residual=res)
            if R == 48:
                vals, v, res = f.eigen(count=2)
                energy = np.array(f.mode_stats(v[:, 0])['tile_energy']).reshape(L, L)
                row.update(strain_ratio=float(vals[0] / base['strain']), strain_energy_fraction=float(energy[lo:hi, lo:hi].sum()), strain_residual=res)
        p0 = f.hydro()
        q, res = compact2d(f, lo, hi)
        row.update(pressure_compact_ratio=q * p0 / base['pressure'], pressure_compact_residual=res, **truchet_2d_rules.patch_pressure(f, lo, hi))
        row['pressure_normalized_compact_excess'] = row['pressure_compact_ratio'] * row['s_P_global'] - 1
        if R == 48:
            vals, v, res = f.eigen(count=2)
            energy = np.array(f.mode_stats(v[:, 0])['tile_energy']).reshape(L, L)
            row.update(pressure_ratio=float(vals[0] * p0 / base['pressure']), pressure_energy_fraction=float(energy[lo:hi, lo:hi].sum()), pressure_residual=res)
        row['seconds'] = time.perf_counter() - start
        out['runs'].append(row)
        dump('truchet_2d/planted_pressure.json', out)
        print('patch2d', row, flush=True)
        del f
        gc.collect()
    R = np.array([r['R'] for r in out['runs']], float)
    y = np.array([r['pressure_normalized_compact_excess'] for r in out['runs']])
    coef, _ = curve_fit(lambda x, a, alpha: a * x ** (-alpha), R, y, p0=[15, 2])
    out['fit'] = dict(power_prefactor=float(coef[0]), power_exponent=float(coef[1]), R=R.tolist(), excess=y.tolist())
    dump('truchet_2d/planted_pressure.json', out)

def periodic3d():
    reference = read('frame_3d/inputs/axial_periodic.json')
    out = resume('frame_3d/periodic.json', lambda: dict(runs=[]))
    for j, r in enumerate(reference['runs']):
        if any((x['pattern'] == r['pattern'] and x['zfactor'] == r['zfactor'] and (x['div'] == r['div']) and (x['grid'] == r['grid']) for x in out['runs'])):
            continue
        q = frame_3d_frame.scan(np.array(r['pattern']), zfactor=r['zfactor'], div=r['div'], grid=r['grid'])
        q['baseline_critical_pressure'] = r['critical_pressure']
        out['runs'].append(q)
        dump('frame_3d/periodic.json', out)
        print('periodic3d', j + 1, len(reference['runs']), q['critical_pressure'], flush=True)
    out['weakest'] = min(out['runs'], key=lambda r: r['critical_pressure'])
    dump('frame_3d/periodic.json', out)

def samples3d():
    reference = read('frame_3d/inputs/axial_periodic.json')['weakest']
    base = frame_3d_frame.scan(np.array(reference['pattern']), zfactor=reference['zfactor'], div=reference['div'], grid=reference['grid'])
    out = dict(base=base, runs=[], patches=[])
    for j in range(10):
        p = np.random.default_rng([20261026, 6, 10, j]).integers(0, 2, (10, 10, 10))
        row, f, st = frame_3d_frame.sample(p, base)
        row.pop('pattern')
        row['sample'] = j
        out['runs'].append(row)
        dump('frame_3d/samples.json', out)
        print('sample3d', j, row['ratio'], row['seconds'], flush=True)
        del f
        gc.collect()
    out['mean_ratio'] = float(np.mean([r['ratio'] for r in out['runs']]))
    motif = np.array(base['pattern'])
    for R in [0, 8]:
        L = 10
        p = np.random.default_rng(441 + L).integers(0, 2, (L, L, L))
        lo = (L - R) // 2
        hi = lo + R
        if R:
            ids = np.indices((R, R, R))
            p[lo:hi, lo:hi, lo:hi] = motif[tuple((ids[d] % motif.shape[d] for d in range(3)))]
        row, f, st = frame_3d_frame.sample(p, base)
        row.pop('pattern')
        row['R'] = R
        if R:
            en = np.array(st['tile_energy']).reshape(L, L, L)
            row['patch_energy_fraction'] = float(en[lo:hi, lo:hi, lo:hi].sum())
        out['patches'].append(row)
        dump('frame_3d/samples.json', out)
        print('patch3d', R, row, flush=True)
        del f
        gc.collect()

def inclusions(limit=None):
    computed = 0
    reference = read('inclusions/inputs/preload_cohort.json')
    names = ['utp', 'ncd', 'ncj', 'cco', 'vne', 'oof']
    rows = [r for r in reference['rows'] if r['name'] in names and r['protocol'] == 'pressure']
    out = resume('inclusions/soft_inclusions.json', lambda: dict(references={}, runs=[]))
    for name in names:
        if name in out['references']:
            continue
        f = ConsistentFrame(name, 0.02, 2, graph_data(name))
        at = ConsistentAtlas(f)
        pd, le, mu = (at.pd(True), at.le(), at.bloch(4))
        loc = min([x for x in [le['load'], None if mu['near_excluded_origin'] else mu['load']] if x is not None]) * at.p0
        out['references'][name] = dict(PD=pd['load'] * at.p0 if pd['load'] else None, loc=loc, class_first=classify(pd, le, mu), LE=le, micro=mu)
    atlas = read('atlas/atlas_rows.json')
    for name, ref in out['references'].items():
        row = next(r for r in atlas if r['name'] == name and r['r_over_each_parent_length'] == 0.02)
        ref['final_atlas'] = dict(lambda_loc=min(v for v in [row['lambda_LE_pressure'], row['lambda_micro_pressure']] if v is not None),
            lambda_PD=row['lambda_PD_pressure'], class_first=row['class_pressure_pV'], radius=0.02,
            source='atlas/atlas_rows.json', rho_loc_formula='pressure * s_P / final_atlas.lambda_loc')
    out['normalization'] = 'runs.rho_loc uses references[name].loc from the cohort search. For the final atlas normalization, use pressure * s_P / references[name].final_atlas.lambda_loc.'
    dump('inclusions/soft_inclusions.json', out)
    for oldrow in rows:
        if any((x['name'] == oldrow['name'] and x['L'] == oldrow['L'] and (x['R_fraction'] == oldrow['R_fraction']) for x in out['runs'])):
            continue
        start = time.perf_counter()
        name = oldrow['name']
        f = InclusionFrame(name, oldrow['L'], oldrow['R_fraction'], div=2)
        f.preload('pressure')
        vals, u, ep, res = f.variable(count=2)
        st = f.stats(u[:, 0], ep[:, 0])
        ref = out['references'][name]
        pin = float(-np.trace(f.pre['tile_stress'][f.patch_cells].mean(axis=0)) / 3)
        p = float(vals[0] * f.p0)
        s = pin / f.p0
        row = dict(name=name, L=f.L, R_fraction=f.R_fraction, pressure=p, s_P=s, rho_PD=p * s / ref['PD'] if ref['PD'] else None, rho_loc=p * s / ref['loc'], patch_energy=st['patch_energy_fraction'], core_energy=st['core_energy_fraction'], eigen_residual=res, baseline_rho_PD=oldrow['rho_PD'], baseline_rho_loc=oldrow['rho_loc'], seconds=time.perf_counter() - start)
        out['runs'].append(row)
        dump('inclusions/soft_inclusions.json', out)
        print('COMPUTE inclusion', len(out['runs']), len(rows), row, flush=True)
        computed += 1
        if limit and computed >= limit:
            return
        del f
        gc.collect()

def rules(limit=None):
    computed = 0
    out = resume('truchet_2d/rules.json', lambda: dict(baseline=baseline2d(), periodic=[], samples=[]))
    base = out['baseline']
    pats = {}
    for name in ['truchet_2d/inputs/axial_periodic.json', 'truchet_2d/inputs/full_four_patterns.json']:
        for r in read(name)['runs']:
            p = np.array(r['pattern'])
            pats[p.shape, p.tobytes()] = p
    for j, p in enumerate(pats.values()):
        if j < len(out['periodic']):
            continue
        row = truchet_2d_rules.scan(p)
        out['periodic'].append(row)
        dump('truchet_2d/rules.json', out)
        print('COMPUTE rules_periodic', j + 1, len(pats), row['pressure'] / base['pressure'], flush=True)
        computed += 1
        if limit and computed >= limit:
            return
    strong = read('truchet_2d/inputs/strong_rule_patterns.json')
    candidates = strong['periodic_candidates']
    out.setdefault('strong_periodic', [])
    for r in candidates:
        if any((x['pattern'] == r['pattern'] for x in out['strong_periodic'])):
            continue
        out['strong_periodic'].append(truchet_2d_rules.scan(np.array(r['pattern'])))
        dump('truchet_2d/rules.json', out)
    for r in read('truchet_2d/inputs/axial_samples.json')['runs']:
        if r['L'] != 64:
            continue
        if any((x['ensemble'] == r['ensemble'] and x['sample'] == r['sample'] for x in out['samples'])):
            continue
        p = np.array(r['pattern'])
        f = Frame2D(p)
        p0 = f.hydro()
        vals, _, res = f.eigen(count=2)
        row = dict(ensemble=r['ensemble'], sample=r['sample'], ratio=float(vals[0] * p0 / base['pressure']), residual=res)
        out['samples'].append(row)
        dump('truchet_2d/rules.json', out)
        print('rules_sample', row, flush=True)
        del f
        gc.collect()
    dump('truchet_2d/rules.json', out)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['periodic2d', 'samples2d', 'patches2d', 'periodic3d', 'samples3d', 'inclusions', 'rules'])
    ap.add_argument('--limit', type=int, help='Maximum newly solved periodic2d, rules or inclusion cases')
    args = ap.parse_args()
    if args.limit and args.action not in ('periodic2d', 'rules', 'inclusions'):
        ap.error('--limit is supported for periodic2d, rules and inclusions')
    globals()[args.action](**({'limit': args.limit} if args.limit else {}))
    names = dict(periodic2d='truchet_2d/periodic.json', periodic3d='frame_3d/periodic.json', samples2d='truchet_2d/random_samples.json', samples3d='frame_3d/samples.json', patches2d='truchet_2d/planted_pressure.json', rules='truchet_2d/rules.json', inclusions='inclusions/soft_inclusions.json')
    name = names[args.action]
    for attempt in range(20):
        try:
            shutil.copyfile(WORK / name, output_dir() / name)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)
