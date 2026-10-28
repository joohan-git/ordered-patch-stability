"""Recompute the Fig. 2 scans (baseline examples and seeds) with the consistent baseline beam tangent.
2D: planted checkerboard domains, full-frame and compact-support thresholds at fixed pressure (planted-domain cohort examples).
3D: random samples L=2..10 and planted domains L=4..10 (frame.samples / patches examples).
Checkpoints go to K08_CACHE/truchet_domains; final JSON is copied next to this script."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, gc, shutil, tempfile, argparse
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
HERE = _DATA_ROOT
DATA_ROOT = HERE
sys.path.insert(0, str(DATA_ROOT))
from computations.common.sparse_frames import Frame2D, Frame3D
from computations.common.initial_stress import MODEL
import computations.truchet_2d.frame as truchet_2d_frame, computations.frame_3d.frame as frame_3d_frame, computations.truchet_2d.rules as truchet_2d_rules
truchet_2d_frame.Frame2D = Frame2D
frame_3d_frame.Frame3D = Frame3D
truchet_2d_rules.Frame2D = Frame2D
truchet_2d_rules.hydro = lambda f: f.hydro()
from computations.common.runtime import cache_dir
WORK = cache_dir('truchet_domains')
WORK.mkdir(exist_ok=True)

def load(name, default):
    p = WORK / name
    if p.exists():
        out = json.loads(p.read_text(encoding='utf-8'))
        if out.get('model') == MODEL:
            return out
    return default

def dump(name, obj):
    obj['model'] = MODEL
    tmp = WORK / (name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False), encoding='utf-8')
    tmp.replace(WORK / name)

def compact2d(f, lo, hi):
    ids = np.flatnonzero(np.repeat(np.all((f.pos > lo + 1e-09) & (f.pos < hi - 1e-09), axis=1), 3))
    K = f.K[ids, :][:, ids].tocsc()
    H = f.H[ids, :][:, ids].tocsc()
    fac = sla.splu(K)
    ev, vec = sla.eigsh(H, k=1, M=K, Minv=sla.LinearOperator(K.shape, matvec=fac.solve), which='LA', tol=1e-09, v0=np.random.default_rng(932).normal(size=len(ids)))
    q = float(1 / ev[0])
    kv = K @ vec[:, 0]
    hv = H @ vec[:, 0]
    return (q, float(la.norm(kv - q * hv) / (la.norm(kv) + q * la.norm(hv))))

def patches2d():
    base = json.loads((DATA_ROOT / 'truchet_2d/planted_pressure.json').read_text(encoding='utf-8'))['baseline']['pressure']
    name = 'truchet_2d/planted_domains.json'
    out = load(name, dict(baseline_pressure=base, runs=[]))
    done = {(r['L'], r['R']) for r in out['runs']}
    common_examples = [(32, r) for r in [2, 3, 4, 6, 8, 12, 16, 24]] + [(64, r) for r in [16, 24, 32, 40, 48]] + [(96, 64)]
    for L, R in common_examples:
        if (L, R) in done:
            continue
        t = time.perf_counter()
        p = np.random.default_rng(441 + L).integers(0, 2, (L, L))
        lo = (L - R) // 2
        hi = lo + R
        p[lo:hi, lo:hi] = np.indices((R, R)).sum(axis=0) % 2
        f = Frame2D(p)
        p0 = f.hydro()
        vals, v, res = f.eigen(count=2)
        energy = np.array(f.mode_stats(v[:, 0])['tile_energy']).reshape(L, L)
        q, cres = compact2d(f, lo, hi)
        row = dict(L=L, R=R, pressure=float(vals[0] * p0), ratio=float(vals[0] * p0 / base), compact_ratio=float(q * p0 / base), energy_fraction=float(energy[lo:hi, lo:hi].sum()), eigen_residual=res, compact_residual=cres, **truchet_2d_rules.patch_pressure(f, lo, hi))
        row['stress_normalized_compact_excess'] = row['compact_ratio'] * row['s_P_global'] - 1
        row['seconds'] = time.perf_counter() - t
        out['runs'].append(row)
        dump(name, out)
        print('2d', L, R, round(row['ratio'], 5), round(row['compact_ratio'], 5), round(row['seconds'], 1), flush=True)
        del f
        gc.collect()

def base3d():
    return json.loads((DATA_ROOT / 'frame_3d/periodic.json').read_text(encoding='utf-8'))['weakest']

def samples3d(sizes=(2, 4, 6, 8, 10), count=10):
    base = base3d()
    name = 'frame_3d/extended_samples.json'
    out = load(name, dict(base_critical_pressure=base['critical_pressure'], runs=[]))
    done = {(r['L'], r['sample']) for r in out['runs']}
    for L in sizes:
        for j in range(count):
            if (L, j) in done:
                continue
            p = np.random.default_rng([20261026, 6, L, j]).integers(0, 2, (L, L, L))
            row, f, st = frame_3d_frame.sample(p, base)
            row.pop('pattern')
            row['sample'] = j
            out['runs'].append(row)
            dump(name, out)
            print('3d sample', L, j, round(row['ratio'], 5), round(row['seconds'], 1), flush=True)
            del f
            gc.collect()

def patches3d(sizes=(4, 6, 8, 10)):
    base = base3d()
    motif = np.array(base['pattern'])
    name = 'frame_3d/planted_domains.json'
    out = load(name, dict(base_critical_pressure=base['critical_pressure'], runs=[]))
    done = {(r['L'], r['R']) for r in out['runs']}
    for L in sizes:
        for R in [0] + list(range(2, L - 1)):
            if (L, R) in done:
                continue
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
                row['patch_energy_fraction'] = float(np.array(st['tile_energy']).reshape(L, L, L)[lo:hi, lo:hi, lo:hi].sum())
            out['runs'].append(row)
            dump(name, out)
            print('3d patch', L, R, round(row['ratio'], 5), round(row['seconds'], 1), flush=True)
            del f
            gc.collect()
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('parts', nargs='+', choices=['patches2d', 'samples3d', 'patches3d', 'copy'])
    for part in ap.parse_args().parts:
        if part == 'copy':
            for n in ('truchet_2d/planted_domains.json', 'frame_3d/extended_samples.json', 'frame_3d/planted_domains.json'):
                if (WORK / n).exists():
                    shutil.copyfile(WORK / n, HERE / n)
        else:
            globals()[part]()
