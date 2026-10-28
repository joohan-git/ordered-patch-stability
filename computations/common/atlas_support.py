"""Offline baseline paths, strict UTF-8 JSON, and reproducible input identity."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
import sys
sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
ROOT = _DATA_ROOT
from computations.common.runtime import cache_dir, output_dir
PROJECT = ROOT
DEST = cache_dir('atlas')
RADII = (0.02, 0.05)
CLASSES = ('PD-only-nonvolumetric', 'PD-only-volumetric', 'PD+LE', 'LE', 'micro', 'PD+micro', 'near-tie')

from computations.common.runtime import read, dump, sha

def publish(name, obj):
    dump(DEST / name, obj)
    shutil.copyfile(DEST / name, output_dir() / name)

def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()

def baseline_rows():
    return [r for k in range(2) for r in read(ROOT / f'common/atlas_chunk{k}.json')['runs']]

def result_path(name, radius):
    return DEST / 'results' / f'{name}_{radius:.2f}.json'

def storage():
    DEST.mkdir(parents=True, exist_ok=True)
    return dict(output=str(DEST), max_worker_processes=2, blas_threads=1, encoding='UTF-8')
