"""Read-only upstream imports and local-only output/limit helpers for frame mechanics."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys
sys.dont_write_bytecode = True
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
OUT = _DATA_ROOT
DATA_ROOT = _DATA_ROOT
import computations.common.diagnostics as dc
from computations.common.runtime import cache_dir, output_dir
OUT = cache_dir('correlations')
from computations.common.diagnostics import Frame2D, eigpair, np, la, sp, sla
import json, time, hashlib, gc
RESULTS = ('prestress/robustness.json', 'truchet_2d/ising_sampling.json', 'truchet_2d/correlated_cell.json')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))

def save(name, value):
    path = OUT / name
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Checkpoint must stay within the cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(value, indent=2, default=plain, allow_nan=False), encoding='utf-8')
    for i in range(30):
        try:
            os.replace(tmp, path)
            export = output_dir() / name
            export.parent.mkdir(parents=True, exist_ok=True)
            export.write_bytes(path.read_bytes())
            return
        except PermissionError:
            time.sleep(0.1)
    raise RuntimeError('Could not checkpoint ' + str(path))

def metadata():
    d = dc.metadata()
    d['loading'] = 'See each block: affine hydrostatic full-wrench frame tangent.'
    d.update(date='2026-10-27', output_directory='cache:correlations/', no_upstream_writes=True)
    return d

def summary(values):
    a = np.asarray(values, float)
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0, min=float(a.min()), max=float(a.max()))

def phash(sig):
    return hashlib.sha256(np.asarray(sig, dtype=np.int8).tobytes()).hexdigest()

class Job:

    def __init__(self, name):
        self.name = name
        self.wall = time.perf_counter()
        self.out = read(OUT / name) if (OUT / name).exists() else dict(metadata=metadata(), runs=[], status='running')
        self.prev_cpu = self.out.get('timing', {}).get('cpu_seconds', 0.0)
        self.prev_wall = self.out.get('timing', {}).get('wall_seconds', 0.0)
        self.out['status'] = 'running'

    def checkpoint(self):
        self.out['timing'] = dict(cpu_seconds=self.prev_cpu + time.process_time(), wall_seconds=self.prev_wall + time.perf_counter() - self.wall)
        save(self.name, self.out)

    def limit_ok(self, reserve=120):
        used = self.prev_cpu + time.process_time()
        for name in RESULTS:
            if name != self.name and (OUT / name).exists():
                used += read(OUT / name).get('timing', {}).get('cpu_seconds', 0.0)
        if used > 7200 - reserve:
            self.out['status'] = 'stopped_at_shared_cpu_limit'
            self.checkpoint()
            return False
        return True

    def finish(self):
        self.out['status'] = 'complete'
        self.checkpoint()

def compact_ids(f, lo, hi):
    keep = np.all((f.pos > np.asarray(lo) + 1e-09) & (f.pos < np.asarray(hi) - 1e-09), axis=1)
    return (3 * np.flatnonzero(keep)[:, None] + np.arange(3)).ravel()
