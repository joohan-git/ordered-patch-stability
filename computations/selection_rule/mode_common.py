"""baseline uses immutable baseline atlas graphs/model and the published Bloch classes."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.check_common import *
source12 = DEST
DEST = cache_dir('modes')
DEST.mkdir(parents=True, exist_ok=True)
ATLAS_PATH = ROOT / 'atlas/bloch_rows.json'

def read(path):
    path = Path(path)
    candidate = output_dir() / path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    if candidate.is_file():
        path = candidate
    return json.loads(path.read_text(encoding='utf-8'))

def atlas13(name):
    return next((x for x in read(ATLAS_PATH) if x['name'] == name and x['r_over_each_parent_length'] == 0.02))

def publish(name, obj):
    save(DEST / name, obj)
    with (output_dir() / name).open('wb') as h:
        h.write((DEST / name).read_bytes())

def residual(K, H, u, t):
    return float(la.norm(K @ u - t * (H @ u)) / (la.norm(K @ u) + abs(t) * la.norm(H @ u)))
