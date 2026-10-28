"""compact domain: does one law govern planted and grown domains?

Regenerates the staggered Ising configurations of truchet_2d/correlated.py (same seeds, same sampler) and measures, for each
sample, the largest compact checkerboard domain: the side of the largest sigma-uniform square and the diameter of the
largest sigma-uniform disc (periodic boundaries). The threshold excess (ratio - 1) of pressure convention is then compared with the
planted-domain law of beam tangent/Bloch (excess vs planted square side R). Output truchet_2d/compact_domains.json.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
import computations.truchet_2d.correlated as c
HERE = _DATA_ROOT
OUT = HERE / 'truchet_2d/compact_domains.json'

def largest_square(mask):
    """Largest all-True square in a periodic boolean array (side length)."""
    L = mask.shape[0]
    big = np.tile(mask, (2, 2))
    dp = np.zeros(big.shape, int)
    best = 0
    for i in range(big.shape[0]):
        for j in range(big.shape[1]):
            if big[i, j]:
                dp[i, j] = 1 if i == 0 or j == 0 else 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
                best = max(best, dp[i, j])
    return min(best, L)

def largest_disc(mask):
    L = mask.shape[0]
    big = np.tile(mask, (3, 3))
    d = ndimage.distance_transform_edt(big)[L:2 * L, L:2 * L]
    return float(2 * d.max())

def main():
    src = json.loads((HERE / 'truchet_2d/correlated.json').read_text(encoding='utf-8'))
    rows = []
    for r in src['runs']:
        rng = np.random.default_rng([20261026, 14, int(round(r['J'] * 1000)), r['sample']])
        sig = c.sample_sigma(r['J'], rng)
        sq = max(largest_square(sig == 1), largest_square(sig == -1))
        dc = max(largest_disc(sig == 1), largest_disc(sig == -1))
        rows.append(dict(J=r['J'], sample=r['sample'], excess=r['ratio'] - 1, largest_square=int(sq), largest_disc=dc, largest_cluster=r['largest_cluster'], check_m=float(abs(sig.mean())), m=r['staggered_m']))
        print(r['J'], r['sample'], round(r['ratio'] - 1, 4), sq, round(dc, 1), round(abs(sig.mean()), 3), round(r['staggered_m'], 3), flush=True)
    OUT.write_text(json.dumps(dict(rows=rows), indent=1), encoding='utf-8')
if __name__ == '__main__':
    main()
