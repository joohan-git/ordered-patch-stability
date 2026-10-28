"""Quantify full dense/sparse rounding at the near-Gamma retry minima."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.mesh_common import *
from computations.atlas.mesh_solver import setup
rows = []
for path in sorted((DEST / 'results').glob('*.json')):
    z = read(path)
    if not z['solver']['method'].startswith('Full sparse'):
        continue
    f = newframe(z['name'], z['div'])
    setup(f)
    f.prestress('stress')
    value = float(f.eigen(z['micro']['k'], 1)[0])
    row = dict(name=z['name'], div=z['div'], k=z['micro']['k'], dense=value, sparse=z['micro']['load'], relative_error=value / z['micro']['load'] - 1, lambda_mu_is_censored=z['lambda_mu'] is None, sparse_residual=z['solver']['max_scaled_residual'])
    rows.append(row)
    print(row, flush=True)
save(DEST / 'atlas/mesh_dense_retry_check.json', dict(rows=rows, max_relative_difference=max((abs(x['relative_error']) for x in rows))))
