"""Portable locations shared by production computations.

Inputs are immutable packaged files. K08_CACHE contains generated checkpoints;
K08_OUTPUT contains exported result JSON (defaults to K08_CACHE/outputs).
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
import tempfile
from pathlib import Path
ROOT = _DATA_ROOT

def cache_root():
    return Path(os.environ.get('K08_CACHE', str(ROOT.parent / 'work'))).expanduser().resolve()

def cache_dir(name):
    path = cache_root() / name
    path.mkdir(parents=True, exist_ok=True)
    for topic in ('common', 'atlas', 'truchet_2d', 'frame_3d', 'inclusions', 'inclusions/tensor', 'prestress', 'selection_rule', 'validation', 'trumof', 'trumof/initialization', 'truchet_2d/inputs', 'frame_3d/inputs'):
        (path / topic).mkdir(parents=True, exist_ok=True)
    return path

def output_dir():
    path = Path(os.environ.get('K08_OUTPUT', str(cache_root() / 'outputs'))).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    for topic in ('common', 'atlas', 'truchet_2d', 'frame_3d', 'inclusions', 'inclusions/tensor', 'prestress', 'selection_rule', 'validation', 'trumof', 'trumof/initialization', 'truchet_2d/inputs', 'frame_3d/inputs'):
        (path / topic).mkdir(parents=True, exist_ok=True)
    return path


def read(path):
    import json
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, obj):
    """Atomically write a JSON checkpoint, tolerating transient file locks."""
    import json, time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.1)


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def external_cgd():
    return Path(os.environ.get('RCSR_CGD', str(ROOT / 'rcsr3d_web.cgd'))).expanduser().resolve()
