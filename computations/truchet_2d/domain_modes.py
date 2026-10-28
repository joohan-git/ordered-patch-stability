"""Compute critical eigenvectors, energies and residuals for random and planted Truchet domains.
2D: 32x32 background (seed 441+32), no domain and a central 16x16 checkerboard domain, fixed hydrostatic pressure.
3D: 10^3 background (seed 441+10), no domain and a central 8^3 domain of the weakest periodic motif.
Per sample: element end points (tile/cube units, periodic images unwrapped), element mode energy (normalized), node translations
of the critical mode, patch mask, threshold ratio. Output truchet_2d/modes_<case>.json next to this script."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np
HERE = _DATA_ROOT
DATA_ROOT = HERE
sys.path.insert(0, str(DATA_ROOT))
from computations.common.sparse_frames import Frame2D, Frame3D
import computations.truchet_2d.frame as truchet_2d_frame, computations.frame_3d.frame as frame_3d_frame, computations.truchet_2d.rules as truchet_2d_rules
truchet_2d_frame.Frame2D = Frame2D
frame_3d_frame.Frame3D = Frame3D
truchet_2d_rules.Frame2D = Frame2D

def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, allow_nan=False), encoding='utf-8')
    print(name, {k: v for k, v in obj.items() if not isinstance(v, list)})

def case2d(R, base_p):
    L = 32
    p = np.random.default_rng(441 + L).integers(0, 2, (L, L))
    lo = (L - R) // 2
    hi = lo + R
    if R:
        p[lo:hi, lo:hi] = np.indices((R, R)).sum(axis=0) % 2
    f = Frame2D(p)
    p0 = f.hydro()
    vals, v, res = f.eigen(count=2)
    v = v[:, 0].real
    u = v.reshape(f.n, 3)
    ue = v[f.dofs]
    ene = np.maximum(np.einsum('ei,eij,ej->e', ue, f.Ke, ue), 0)
    ene /= ene.max()
    I = f.dofs[:, 0] // 3
    J = f.dofs[:, 3] // 3
    P = f.pos[I]
    Q = f.pos[J] + f.shifts * L
    st = f.mode_stats(v)
    tile = np.array(st['tile_energy']).reshape(L, L)
    mask = np.zeros((L, L), bool)
    if R:
        mask[lo:hi, lo:hi] = True
    return dict(dim=2, L=L, R=R, ratio=float(vals[0] * p0 / base_p), P=np.round(P, 5).tolist(), Q=np.round(Q, 5).tolist(), energy=np.round(ene, 5).tolist(), uP=np.round(u[I, :2], 6).tolist(), uQ=np.round(u[J, :2], 6).tolist(), tile_energy=np.round(tile, 6).tolist(), patch=mask.astype(int).tolist(), energy_in_patch=float(tile[mask].sum()) if R else None, participation=float(st['energy_participation_fraction']), eigen_residual=float(res))

def case3d(R, base):
    L = 10
    motif = np.array(base['pattern'])
    p = np.random.default_rng(441 + L).integers(0, 2, (L, L, L))
    lo = (L - R) // 2
    hi = lo + R
    if R:
        ids = np.indices((R, R, R))
        p[lo:hi, lo:hi, lo:hi] = motif[tuple((ids[d] % motif.shape[d] for d in range(3)))]
    f = Frame3D(p, zfactor=base['zfactor'])
    vals, v, res = f.eigen(count=2)
    v = v[:, 0].real
    u = v.reshape(f.n, 6)
    ue = v[f.dofs]
    ene = np.maximum(np.einsum('ei,eij,ej->e', ue, f.Ke, ue), 0)
    ene /= ene.max()
    I = f.dofs[:, 0] // 6
    J = f.dofs[:, 6] // 6
    P = f.pos[I]
    Q = f.pos[J] + f.shifts * L
    st = f.stats(v)
    tile = np.array(st['tile_energy']).reshape(L, L, L)
    mask = np.zeros((L, L, L), bool)
    if R:
        mask[lo:hi, lo:hi, lo:hi] = True
    return dict(dim=3, L=L, R=R, ratio=float(vals[0] * f.p0 / base['critical_pressure']), P=np.round(P, 5).tolist(), Q=np.round(Q, 5).tolist(), energy=np.round(ene, 5).tolist(), uP=np.round(u[I, :3], 6).tolist(), uQ=np.round(u[J, :3], 6).tolist(), owner=f.owners.tolist(), patch=mask.astype(int).ravel().tolist(), energy_in_patch=float(tile[mask].sum()) if R else None, participation=float(st['participation_fraction']), eigen_residual=float(res))
if __name__ == '__main__':
    which = sys.argv[1:] or ['2d', '3d']
    if '2d' in which:
        bp = json.loads((DATA_ROOT / 'truchet_2d/planted_pressure.json').read_text(encoding='utf-8'))['baseline']['pressure']
        for R in (0, 16):
            dump(f'truchet_2d/modes_2d_R{R}.json', case2d(R, bp))
    if '3d' in which:
        base = json.loads((DATA_ROOT / 'frame_3d/periodic.json').read_text(encoding='utf-8'))['weakest']
        for R in (0, 8):
            dump(f'truchet_2d/modes_3d_R{R}.json', case3d(R, base))
