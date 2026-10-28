"""baseline immutable inputs and no-clobber outputs."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.atlas_support import *
source11 = DEST
DEST = cache_dir('atlas_check')
DEST.mkdir(parents=True, exist_ok=True)
from computations.atlas.run_atlas import hydrate, POOLS
from computations.validation.beam_validation import ConsistentFrame, ConsistentAtlas
import numpy as np
import scipy.linalg as la

def read(path):
    path = Path(path)
    candidate = output_dir() / path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    if candidate.is_file():
        path = candidate
    return json.loads(path.read_text(encoding='utf-8'))

def plain(x):
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x

def save(path, obj):
    dump(Path(path), plain(obj))

def publish(name, obj):
    save(DEST / name, obj)
    (output_dir() / name).write_bytes((DEST / name).read_bytes())

def atlas_rows():
    return read(ROOT / 'atlas/baseline_rows.json')

def raw(name, r):
    path = source11 / 'results' / f'{name}_{r:.2f}.json'
    if not path.exists():
        from computations.atlas.run_atlas import run_one
        dump(path, run_one(graph(name), r))
    return read(path)

def graph(name):
    from computations.atlas.bootstrap import bootstrap
    return bootstrap([name])[0]

def frame(name, r):
    return ConsistentFrame(name, r, 2, hydrate(graph(name)))

def classify(pd, le, mu, tol=0.003):
    pv = pd['load'] if pd['load'] is not None else np.inf
    lv = le['load'] if le['load'] is not None else np.inf
    mv = mu['load']
    if pv < min(lv, mv) * (1 - tol):
        return 'PD-only-volumetric' if pd['volumetric_fraction'] > 0.8 else 'PD-only-nonvolumetric'
    if lv <= mv * (1 + tol) and lv <= pv * (1 + tol):
        return 'LE' if pv > lv * (1 + tol) else 'PD+LE'
    if mv <= pv * (1 + tol) and mv <= lv * (1 + tol):
        return 'micro' if pv > mv * (1 + tol) else 'PD+micro'
    return 'near-tie'

def stats(values):
    a = np.asarray(values, float)
    return dict(n=len(a), min=float(a.min()), p50=float(np.median(a)), p99=float(np.quantile(a, 0.99)), max=float(a.max())) if len(a) else dict(n=0)
