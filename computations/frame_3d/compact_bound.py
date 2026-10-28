"""R^-2 excess in three dimensions: compact-support bound of a planted R^3 domain of the weakest periodic motif in one
14^3 random background (seed 441+14), baseline tangent, fixed hydrostatic pressure. Pressure-normalized excess = compact ratio x s_P - 1,
s_P = mean pressure inside the domain / applied pressure. Output frame_3d/compact_bound.json (checkpointed)."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
HERE = _DATA_ROOT
DATA_ROOT = HERE
sys.path.insert(0, str(DATA_ROOT))
from computations.common.sparse_frames import Frame3D
import computations.frame_3d.frame as frame_3d_frame
frame_3d_frame.Frame3D = Frame3D
OUT = HERE / 'frame_3d/compact_bound.json'
L = 14
RS = [2, 3, 4, 5, 6, 7, 8, 10]
base = json.loads((DATA_ROOT / 'frame_3d/periodic.json').read_text(encoding='utf-8'))['weakest']
motif = np.array(base['pattern'])
out = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else dict(L=L, base_critical_pressure=base['critical_pressure'], runs=[])
for R in RS:
    if any((r['R'] == R for r in out['runs'])):
        continue
    t0 = time.perf_counter()
    p = np.random.default_rng(441 + L).integers(0, 2, (L, L, L))
    lo = (L - R) // 2
    hi = lo + R
    ids = np.indices((R, R, R))
    p[lo:hi, lo:hi, lo:hi] = motif[tuple((ids[d] % motif.shape[d] for d in range(3)))]
    f = Frame3D(p, zfactor=base['zfactor'])
    sup = f.compact([lo] * 3, [hi] * 3)
    mask = np.zeros((L, L, L), bool)
    mask[lo:hi, lo:hi, lo:hi] = True
    pressure = -np.trace(f.pre['tile_stress'], axis1=1, axis2=2).reshape(L, L, L) / 3
    sP = float(pressure[mask].mean() / f.p0)
    ratio = sup['pressure'] / base['critical_pressure']
    row = dict(R=R, compact_ratio=ratio, s_P=sP, pressure_normalized_excess=ratio * sP - 1, residual=sup['residual'], dofs=sup['dofs'], seconds=time.perf_counter() - t0)
    out['runs'].append(row)
    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(R, round(ratio, 5), round(sP, 5), round(row['pressure_normalized_excess'], 5), round(row['seconds'], 1), flush=True)
R = np.array([r['R'] for r in out['runs']], float)
y = np.array([r['pressure_normalized_excess'] for r in out['runs']])
k = R >= 3
coef, cov = curve_fit(lambda x, a, g: a * x ** (-g), R[k], y[k], p0=[1, 2])
out['fit'] = dict(R_min=3, prefactor=float(coef[0]), exponent=float(coef[1]), exponent_se=float(np.sqrt(cov[1, 1])))
OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
print('fit', out['fit'])
