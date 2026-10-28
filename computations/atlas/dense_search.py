"""Dense loss-of-ellipticity search for atlas nets with homogeneous thresholds.

The atlas search uses 26 symmetry axes and 38 quasi-uniform directions, with
local refinement from the best two. This search samples 3,000 Fibonacci
directions and refines the ten lowest with Nelder-Mead. Both use
validation/beam_validation.ConsistentAtlas on the frozen edge-centre graphs.
Output: atlas/dense_search_<r>.jsonl, with cache checkpoints for resuming.
Usage: python atlas/dense_search.py <r/l> (0.02 or 0.05).
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys, json, time, tempfile
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
HERE = _DATA_ROOT
DATA_ROOT = HERE
sys.path.insert(0, str(DATA_ROOT))
import numpy as np, gemmi
from scipy.optimize import minimize
from computations.validation.beam_validation import ConsistentFrame, ConsistentAtlas
from computations.common.runtime import cache_dir, output_dir
from computations.atlas.bootstrap import bootstrap
ROOT = HERE
GRAPHS = cache_dir('atlas') / 'graphs'
N_DIR, N_REFINE = (3000, 10)

def hydrate(g):
    return dict(name=g['name'], frac=np.asarray(g['frac']), cn=np.asarray(g['cn']), edges=[(i, j, np.asarray(sh, int)) for i, j, sh in g['edges']], A=np.asarray(g['A']), sg=gemmi.find_spacegroup_by_name(g['space_group']), lengths=g['lengths'], trials=[dict(loader='load_sym', graph_sha256=g['graph_sha256'])])

def fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    ang = np.pi * (3 - np.sqrt(5)) * np.arange(n)
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(ang), s * np.sin(ang), z], 1)

def main(rad, names=None):
    rows = json.load(open(ROOT / 'atlas/bloch_rows.json', encoding='utf-8'))
    todo = [r for r in rows if r['r_over_each_parent_length'] == rad and r['lambda_PD_pressure']]
    if names:
        todo = [r for r in todo if r['name'] in names]
        if not todo:
            raise ValueError('No eligible atlas nets selected')
    bootstrap([r['name'] for r in todo])
    out = output_dir() / f'atlas/dense_search_{rad}.jsonl'
    done = {json.loads(l)['name'] for l in open(out, encoding='utf-8')} if out.exists() else set()
    dirs = fibonacci(N_DIR)
    with open(out, 'a', encoding='utf-8') as fo:
        for r in todo:
            if r['name'] in done:
                continue
            t = time.time()
            g = json.load(open(GRAPHS / f"{r['name']}.json", encoding='utf-8'))
            at = ConsistentAtlas(ConsistentFrame(r['name'], rad, 2, hydrate(g)))
            vals = sorted(((at.acoustic(d), i) for i, d in enumerate(dirs)))
            best, bdir = (vals[0][0], dirs[vals[0][1]])
            for v, i in vals[:N_REFINE]:
                o = minimize(lambda x: at.acoustic(x) / vals[0][0], dirs[i], method='Nelder-Mead', options=dict(maxiter=400, xatol=1e-08, fatol=1e-11))
                val = at.acoustic(o.x)
                if val < best:
                    best, bdir = (val, o.x / np.linalg.norm(o.x))
            fo.write(json.dumps(dict(name=r['name'], r=rad, LE_dense_pressure=float(best * at.p0), direction=[float(x) for x in bdir], LE_production_pressure=r['lambda_LE_pressure'], seconds=round(time.time() - t, 2))) + '\n')
            fo.flush()
            print(r['name'], rad, round(time.time() - t, 1), flush=True)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('radius', type=float, choices=[0.02, 0.05])
    ap.add_argument('--nets', nargs='+')
    args = ap.parse_args()
    main(args.radius, args.nets)
