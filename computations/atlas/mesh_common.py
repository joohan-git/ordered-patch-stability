"""Immutable baseline inputs; all running output stays in TEMP."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.check_common import *
source12 = DEST
DEST = cache_dir('atlas_mesh')
DEST.mkdir(parents=True, exist_ok=True)

def publish(name, obj):
    save(DEST / name, obj)
    with (output_dir() / name).open('wb') as h:
        h.write((DEST / name).read_bytes())

def cohort():
    source = json.loads((ROOT / 'atlas/bloch_tolerance.json').read_text(encoding='utf-8'))
    c = next((x for x in source['radii']['0.02']['tolerances'] if x['tolerance'] == 0.003))['per_net_class']
    pd = sorted((n for n, cl in c.items() if cl.startswith('PD-only')))
    other = sorted(set(c) - set(pd))
    controls = sorted(np.random.default_rng(260126).choice(other, 60, replace=False).tolist())
    assert len(pd) == 78 and len(other) == 798
    return dict(seed=260126, rng='numpy.random.default_rng / PCG64; sorted eligible names', radius=0.02, divisions=[2, 4, 8], tie_tolerance=0.003, pdonly=pd, controls=controls, eligible_controls=other, baseline_classes={n: c[n] for n in pd + controls}, source_sha256=sha(ROOT / 'atlas/bloch_tolerance.json'))

def newframe(name, div):
    return ConsistentFrame(name, 0.02, div, hydrate(graph(name)))
if __name__ == '__main__':
    c = cohort()
    p = DEST / 'atlas/mesh_cohort.json'
    if not p.exists():
        save(p, c)
    sizes = []
    for n in c['pdonly'] + c['controls']:
        g = graph(n)
        sizes.append((6 * (len(g['frac']) + 7 * len(g['edges'])), n, len(g['frac']), len(g['edges'])))
    print('COHORT', len(c['pdonly']), len(c['controls']), c['controls'])
    print('div8 ndof median/max', np.median([x[0] for x in sizes]), sorted(sizes)[-10:])
