"""Size means and weighted excess = alpha_0 + alpha_1/ln(N), using the bundled cohorts. Output keys a and b denote alpha_0 and alpha_1."""
from pathlib import Path
import sys
DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA.parent))
import json
import numpy as np
from computations.common.runtime import output_dir


def moments(rows):
    ratios = np.array([r['ratio'] for r in rows], dtype=float)
    n = len(ratios)
    mean = float(ratios.mean())
    sd = float(ratios.std(ddof=1)) if n > 1 else 0.0
    size = rows[0]['L']
    return dict(L=size, N=size**2, samples=n, mean_ratio=mean, sample_sd=sd,
                sem=sd / np.sqrt(n), excess_times_lnN=(mean - 1) * np.log(size**2))


def calculate():
    ising = json.loads((DATA / 'truchet_2d/ising.json').read_text())['runs']
    independent = json.loads((DATA / 'truchet_2d/independent_tiles.json').read_text())['runs']
    groups, fits = [], []
    for coupling in sorted({r['J'] for r in ising}):
        rows = [dict(J=coupling, **moments([r for r in ising if r['J'] == coupling and r['L'] == size]))
                for size in sorted({r['L'] for r in ising if r['J'] == coupling})]
        groups.extend(rows)
        design = np.array([[1.0, 1 / np.log(r['N'])] for r in rows])
        excess = np.array([r['mean_ratio'] - 1 for r in rows])
        sem = np.array([r['sem'] for r in rows])
        assert len(rows) > 2 and np.all(sem > 0)
        weighted = design / sem[:, None]
        coefficients = np.linalg.lstsq(weighted, excess / sem, rcond=None)[0]
        covariance = np.linalg.inv(weighted.T @ weighted)
        fits.append(dict(J=coupling, a=float(coefficients[0]), b=float(coefficients[1]),
                         a_se=float(np.sqrt(covariance[0, 0])), b_se=float(np.sqrt(covariance[1, 1])),
                         covariance=covariance.tolist(),
                         chi_squared=float(np.sum(((excess - design @ coefficients) / sem)**2)),
                         degrees_of_freedom=len(rows)-2))
    iid = [moments([r for r in independent if r['L'] == size]) for size in sorted({r['L'] for r in independent})]
    return dict(sources=['truchet_2d/ising.json', 'truchet_2d/independent_tiles.json'],
                method='Fit mean_ratio - 1 = alpha_0 + alpha_1/ln(N), N=L^2; JSON keys a,b denote alpha_0,alpha_1 and a_se,b_se their standard errors. Weights=1/SEM^2; SEM uses sample SD (ddof=1). Covariance=(X^T W X)^-1, without residual rescaling.',
                ising=groups, independent_tiles=iid, weighted_fits=fits)


if __name__ == '__main__':
    result = calculate()
    (output_dir() / 'truchet_2d/size_statistics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
