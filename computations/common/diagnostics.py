"""Read-only reuse of upstream solvers; every output stays in this directory."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys
sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
OUT = _DATA_ROOT
DATA_ROOT = _DATA_ROOT
sys.path.insert(0, str(DATA_ROOT))
from computations.common.runtime import cache_dir, output_dir
OUT = cache_dir('diagnostics')
import json, hashlib, time
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from computations.common.sparse_frames import Frame2D, Frame3D
from computations.frame_3d.frame import eigpair

def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))

def save(name, data):
    path = OUT / name
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Checkpoint must stay within the cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(data, indent=2, default=plain, allow_nan=False), encoding='utf-8')
    for attempt in range(30):
        try:
            os.replace(tmp, path)
            export = output_dir() / name
            export.parent.mkdir(parents=True, exist_ok=True)
            export.write_bytes(path.read_bytes())
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.1 * (attempt + 1))

def metadata():
    return dict(numpy=np.__version__, scipy=scipy.__version__, workers_per_script=1, blas_threads=1, loading='hydrostatic preload; fixed periodic cell fluctuation eigenproblem', source_sha256={n: hashlib.sha256((DATA_ROOT / n).read_bytes()).hexdigest() for n in ('common/sparse_frames.py', 'common/initial_stress.py', 'truchet_2d/frame.py', 'frame_3d/frame.py')})
