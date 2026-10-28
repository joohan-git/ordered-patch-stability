"""Correlated disorder brings the cap within reach: 2D decorated Truchet samples whose tile states carry Ising correlations
that favour the checkerboard (the weakest allowed crystal). Staggered spin sigma = (2s-1)(-1)^(x+y); sigma-uniform regions are
checkerboard patches (either phase). Coupling J on nearest neighbours of sigma, zero field, heat-bath sampling with checkerboard
updates; J = 0 is the independent-tile ensemble. Every configuration is still an allowed sample of the same tile rule.
Per sample: threshold at fixed hydrostatic pressure relative to the checkerboard (baseline tangent), correlation length of sigma,
largest sigma-cluster size, and the fraction of critical-mode energy inside that cluster.
Output truchet_2d/correlated.json (checkpointed after every sample)."""
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
from scipy import ndimage
HERE = _DATA_ROOT
DATA_ROOT = HERE
sys.path.insert(0, str(DATA_ROOT))
from computations.common.sparse_frames import Frame2D
import computations.truchet_2d.frame as truchet_2d_frame
truchet_2d_frame.Frame2D = Frame2D
from computations.common.runtime import cache_dir, output_dir
OUT = cache_dir('correlations') / 'truchet_2d/correlated.json'
L, NS, SWEEPS = (64, 8, 3000)
JS = [0.0, 0.15, 0.25, 0.32, 0.38, 0.42, 0.46, 0.55, 0.7]

def sample_sigma(J, rng):
    s = rng.choice([-1, 1], size=(L, L))
    mask = (np.indices((L, L)).sum(0) % 2).astype(bool)
    for _ in range(SWEEPS):
        for m in (mask, ~mask):
            h = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
            p = 1 / (1 + np.exp(-2 * J * h))
            new = np.where(rng.random((L, L)) < p, 1, -1)
            s = np.where(m, new, s)
    return s

def corr_length(s):
    f = np.fft.fft2(s - s.mean())
    c = np.real(np.fft.ifft2(f * np.conj(f))) / s.size
    c /= c[0, 0] if c[0, 0] else 1
    r = np.arange(1, L // 2)
    cr = np.array([(c[d, 0] + c[0, d]) / 2 for d in r])
    pos = cr > np.exp(-1)
    return float(r[np.argmin(pos)] if (~pos).any() else L / 2)

def run(limit=None):
    computed = 0
    base = json.loads((DATA_ROOT / 'truchet_2d/planted_pressure.json').read_text(encoding='utf-8'))['baseline']['pressure']
    out = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else dict(L=L, sweeps=SWEEPS, couplings=JS, baseline_pressure=base, runs=[])
    done = {(r['J'], r['sample']) for r in out['runs']}
    stagger = np.where(np.indices((L, L)).sum(0) % 2 == 0, 1, -1)
    for J in JS:
        for j in range(NS):
            if (J, j) in done:
                continue
            t0 = time.perf_counter()
            rng = np.random.default_rng([20261026, 14, int(round(J * 1000)), j])
            sig = sample_sigma(J, rng)
            tiles = ((sig * stagger + 1) // 2).astype(int)
            f = Frame2D(tiles)
            p0 = f.hydro()
            vals, v, res = f.eigen(count=2)
            tile_e = np.array(f.mode_stats(v[:, 0])['tile_energy']).reshape(L, L)
            lab = np.zeros((L, L), int)
            sizes = []
            for sign in (1, -1):
                l, n = ndimage.label(sig == sign)
                l[l > 0] += lab.max()
                lab = np.where(l > 0, l, lab)
            ids, counts = np.unique(lab[lab > 0], return_counts=True)
            big = ids[np.argmax(counts)]
            row = dict(J=J, sample=j, ratio=float(vals[0] * p0 / base), xi=corr_length(sig), largest_cluster=int(counts.max()), mode_in_largest=float(tile_e[lab == big].sum()), staggered_m=float(abs(sig.mean())), eigen_residual=float(res), seconds=time.perf_counter() - t0)
            out['runs'].append(row)
            OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
            (output_dir() / 'truchet_2d/correlated.json').write_bytes(OUT.read_bytes())
            print('COMPUTE correlated', J, j, flush=True)
            computed += 1
            if limit and computed >= limit:
                return
            print(J, j, round(row['ratio'], 4), row['xi'], row['largest_cluster'], round(row['mode_in_largest'], 3), round(row['seconds'], 1), flush=True)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    run(ap.parse_args().limit)
