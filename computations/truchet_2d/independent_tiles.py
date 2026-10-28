"""Independent-tile (J = 0) size scan of the two-dimensional decorated Truchet frame: does the mean first-instability
pressure fall with sample size, and does the critical mode sit in the largest checkerboard square?
Same frame, preload and fixed-cell fluctuation routine as truchet_2d/ising_scan.py; tiles drawn independently."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import sys, json, time, hashlib, gc
from pathlib import Path
HERE = _DATA_ROOT
from computations.common.diagnostics import *
from computations.truchet_2d.ising_scan import periodic_clusters, largest_squares
from computations.common.runtime import output_dir
OUTF = output_dir() / 'truchet_2d/independent_tiles.json'

def main(sizes=(32, 64, 96, 128, 192), n=8):
    base = json.loads((DATA_ROOT / 'truchet_2d/planted_pressure.json').read_text())['baseline']['pressure']
    out = json.loads(OUTF.read_text()) if OUTF.exists() else dict(baseline_pressure=base, runs=[], protocol='independent tiles, fixed-cell fluctuations after hydrostatic preload (Ising mechanical routine), seed [20261027, 0, L, sample]')
    for L in sizes:
        stagger = np.where(np.indices((L, L)).sum(0) % 2 == 0, 1, -1)
        for j in range(n):
            if any(((r['L'], r['sample']) == (L, j) for r in out['runs'])):
                continue
            t0 = time.perf_counter()
            tiles = np.random.default_rng([20261027, 0, L, j]).integers(0, 2, (L, L))
            sig = (2 * tiles - 1) * stagger
            lab, big, bigsize = periodic_clusters(sig)
            side, starts = largest_squares(sig)
            f = Frame2D(tiles.astype(int))
            p0 = f.hydro()
            vals, v, res = f.eigen(count=2)
            st_ = f.mode_stats(v[:, 0])
            te = np.array(st_['tile_energy']).reshape((L, L))
            se = [float(te[np.ix_((np.arange(side) + a) % L, (np.arange(side) + b) % L)].sum()) for a, b in starts]
            out['runs'].append(dict(L=L, sample=j, ratio=float(vals[0] * p0 / base), largest_square_side=int(side), square_ties=len(starts), mode_energy_largest_square_max=max(se), participation_ratio=st_['energy_participation_fraction'], eigen_residual=res, seconds=time.perf_counter() - t0, pattern_sha256=hashlib.sha256(tiles.astype(np.int8).tobytes()).hexdigest()))
            OUTF.write_text(json.dumps(out, indent=1), encoding='utf-8')
            print('done', L, j, round(out['runs'][-1]['ratio'], 5), 'sq', side, 'sec', round(out['runs'][-1]['seconds'], 1), flush=True)
            del f
            gc.collect()
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', type=int, nargs='+', default=[32, 64, 96, 128, 192])
    ap.add_argument('--samples', type=int, default=8)
    args = ap.parse_args()
    if args.samples < 1 or any((n < 2 for n in args.sizes)):
        ap.error('Use sizes >= 2 and samples >= 1')
    main(args.sizes, args.samples)
