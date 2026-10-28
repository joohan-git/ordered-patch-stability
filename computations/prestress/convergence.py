"""Reproduce Bloch planted backgrounds and check scalar prestress convergence.

One sequential worker; all BLAS pools one thread. No source mutations/import
bytecode. Units: forces and stress divided by each sample's applied pressure.
Core comprises all complete owner tiles [lo+w,hi-w)^d, retaining each tile's
assigned member set. Reference members match motif-relative tile and edge slot.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys
sys.dont_write_bytecode = True
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import json, time, gc, hashlib, argparse
OUT = _DATA_ROOT
DATA_ROOT = _DATA_ROOT
sys.path.insert(0, str(DATA_ROOT))
from computations.common.runtime import cache_dir
OUT = cache_dir('prestress_convergence')
import numpy as np
import scipy
from scipy import linalg as la
from computations.common.sparse_frames import Frame2D, Frame3D

def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))

def save(out):
    (OUT / 'prestress/convergence.json').write_text(json.dumps(out, indent=2, default=plain, allow_nan=False), encoding='utf-8')

def fit_scalar(f, ref):
    f, ref = (f.ravel(), ref.ravel())
    s = float(f @ ref / (ref @ ref))
    return {'s': s, 'relative_deviation': float(la.norm(f - s * ref) / la.norm(ref))}

def stress_info(stress):
    d = len(stress)
    hydro = np.trace(stress) / d
    dev = stress - hydro * np.eye(d)
    return {'mean_stress': stress, 'mean_pressure': float(-hydro), 'deviatoric_over_hydrostatic': float(la.norm(dev) / (np.sqrt(d) * abs(hydro)))}

def fields(f, d):
    ef = f.pre['full_element_forces'] / f.p0
    j = 3 if d == 2 else 6
    vir = np.einsum('ei,ej->eij', ef[:, j:j + d], f.delta)
    vir = (vir + vir.transpose(0, 2, 1)) / 2
    return (f.pre['forces'] / f.p0, ef, vir)

def summarize(f, reference, L, R, d, motif):
    ax, ef, vir = fields(f, d)
    pax, pef, pvir = reference
    slots = 10 if d == 2 else 17
    owners = f.parents // slots
    coords = np.array(np.unravel_index(owners, tuple(f.shape))).T
    lo = (L - R) // 2
    hi = lo + R
    rel = (coords - lo) % np.array(motif.shape)
    mapped = np.ravel_multi_index(tuple(rel.T), motif.shape) * slots + f.parents % slots
    rows = []
    for w in (1, 2, 4):
        if R <= 2 * w:
            rows.append({'w': w, 'core_side': 0, 'status': 'empty core: R <= 2w'})
            continue
        mask = np.all((coords >= lo + w) & (coords < hi - w), axis=1)
        ntiles = (R - 2 * w) ** d
        rmap = mapped[mask]
        stress = vir[mask].sum(axis=0) / ntiles
        pstress = pvir[rmap].sum(axis=0) / ntiles
        rows.append({'w': w, 'core_side': R - 2 * w, 'tiles': ntiles, 'members': int(mask.sum()), 'status': 'ok', 'axial': fit_scalar(ax[mask], pax[rmap]), 'full_end_wrench_tile_length_units': fit_scalar(ef[mask], pef[rmap]), **stress_info(stress), 'periodic_core_stress': stress_info(pstress)})
    return {'dimension': d, 'L': L, 'R': R, 'seed': 441 + L, 'p0': f.p0, 'global_stress': f.pre['stress'] / f.p0, 'equilibrium_residual': f.pre['residual'], 'cores': rows}

def update_fits(out):
    fits = []
    for d, L in sorted({(r['dimension'], r['L']) for r in out['runs']}):
        for w in (1, 2, 4):
            points = []
            for row in out['runs']:
                if (row['dimension'], row['L']) != (d, L):
                    continue
                c = next((c for c in row['cores'] if c['w'] == w))
                if c['status'] == 'ok':
                    points.append((row['R'], c['axial']['relative_deviation'], c['deviatoric_over_hydrostatic']))
            if len(points) < 3:
                continue
            points.sort()
            a = np.array(points)
            for col, metric in [(1, 'axial_relative_deviation'), (2, 'stress_anisotropy')]:
                slope, intercept = np.polyfit(np.log(a[:, 0]), np.log(a[:, col]), 1)
                pred = intercept + slope * np.log(a[:, 0])
                y = np.log(a[:, col])
                fits.append({'dimension': d, 'L': L, 'w': w, 'metric': metric, 'n': len(a), 'R': a[:, 0], 'form': 'A * R^(-alpha); descriptive log-OLS only', 'A': float(np.exp(intercept)), 'alpha': float(-slope), 'log_R_squared': float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))})
    out['fits'] = fits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dimension', type=int, choices=[2, 3])
    args = ap.parse_args()
    path = OUT / 'prestress/convergence.json'
    out = json.loads(path.read_text()) if path.exists() else {'runs': []}
    out['metadata'] = {'workers': 1, 'numpy': np.__version__, 'scipy': scipy.__version__, 'thread_environment': {k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}, 'loading': 'hydrostatic stress, p=1 by force / p0; radius=.01, div=1, common/sparse adapters', 'reference': 'periodic checkerboard 2x2; common/3d_periodic weakest motif; same-pressure member mapping', 'core': 'all owner tiles [lo+w,hi-w)^d; edge slots include assigned boundary members', 'force_metric': 'min_s ||f-s*f_P||_2 / ||f_P||_2; f=axial forces, companion complete end-wrench metric', 'anisotropy': '||dev(mean sigma)||_F / ||(tr(mean sigma)/d) I||_F; full symmetrized end-force virial', 'fit_limitations': 'one deterministic background per L; nested correlated patches; fixed L/R and shrinking-matrix bias; no asymptotic-rate proof', 'source_sha256': {n: hashlib.sha256((DATA_ROOT / n).read_bytes()).hexdigest() for n in ['common/sparse_frames.py', 'common/initial_stress.py', 'truchet_2d/frame.py', 'frame_3d/frame.py']}}
    for d in [args.dimension] if args.dimension else [2, 3]:
        if d == 2:
            motif = np.indices((2, 2)).sum(axis=0) % 2
            base = Frame2D(motif)
            base.hydro()
            common_examples = [(L, R) for L in (32, 64, 96) for R in (8, 16, 24, 32, 48) if R < L]
        else:
            motif = np.array(json.loads((DATA_ROOT / 'frame_3d/periodic.json').read_text())['weakest']['pattern'])
            base = Frame3D(motif)
            common_examples = [(10, R) for R in range(4, 9)]
        reference = fields(base, d)
        del base
        gc.collect()
        out.setdefault('periodic_motifs', {})[str(d)] = motif.tolist()
        for L, R in common_examples:
            if any(((r['dimension'], r['L'], r['R']) == (d, L, R) for r in out['runs'])):
                continue
            start = time.perf_counter()
            print(f'start d={d} L={L} R={R}', flush=True)
            pattern = np.random.default_rng(441 + L).integers(0, 2, (L,) * d)
            lo = (L - R) // 2
            ids = np.indices((R,) * d)
            pattern[(slice(lo, lo + R),) * d] = motif[tuple((ids[i] % motif.shape[i] for i in range(d)))]
            f = (Frame2D if d == 2 else Frame3D)(pattern)
            if d == 2:
                f.hydro()
            row = summarize(f, reference, L, R, d, motif)
            row['seconds'] = time.perf_counter() - start
            row['pattern_sha256'] = hashlib.sha256(pattern.astype(np.uint8).tobytes()).hexdigest()
            out['runs'].append(row)
            update_fits(out)
            save(out)
            print(f'done {d} {L} {R}: ' + str([(c['w'], c.get('axial')) for c in row['cores']]), flush=True)
            del f
            gc.collect()
    save(out)
if __name__ == '__main__':
    main()
