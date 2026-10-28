"""Retry failed checkpoints using the entire sparse generalized eigenproblem."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.mesh_common import *
import computations.atlas.mesh_run as runner
from computations.atlas.mesh_solver import install_sparse

def install_full(f):
    info = install_sparse(f)
    checks = []
    for k in [[0, 0, 0], [0.125, 0.25, 0.375], [0.007, 0.002, 0.001]]:
        reference = float(type(f).eigen(f, k, 1)[0])
        actual = float(f.eigen(k, 1)[0])
        checks.append(dict(k=k, dense=reference, sparse=actual, relative_error=actual / reference - 1))
    assert max((abs(x['relative_error']) for x in checks)) < 1e-07, checks
    info.update(method='Full sparse generalized eigenproblem; all DOFs retained; deterministic ARPACK start; independent full dense validation', dense_validation=checks)
    return info
runner.install_condensed = install_full
if __name__ == '__main__':
    runner.main()
