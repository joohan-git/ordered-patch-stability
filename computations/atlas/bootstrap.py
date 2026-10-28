"""Rebuild the frozen RCSR graphs and verify their published SHA-256 identities.

Geometry, normalization and hash serialization are exactly those of atlas/baseline_prepare.
Only accepted catalogue entries are bootstrapped; rejected graphs remain in the
archived catalogue. Graph reconstruction uses the catalogue and the external RCSR dataset.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import argparse
import hashlib
import json
import time
import numpy as np
import scipy.linalg as la
import gemmi
from computations.common.runtime import ROOT, cache_dir, external_cgd
from computations.common.rcsr_loader import read_cgd, load_sym

def identity(graph):
    return hashlib.sha256(json.dumps(graph, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()

def build_graph(name, nets):
    g = load_sym(name, nets)
    if g is None or not g['ok']:
        raise ValueError(f'RCSR graph failed degree validation: {name}')
    sg = gemmi.find_spacegroup_by_name(g['net']['group'] + (g['setting'] or ''))
    lengths = np.array([la.norm((g['frac'][j] + sh - g['frac'][i]) @ g['A']) for i, j, sh in g['edges']])
    graph = dict(name=name, frac=g['frac'].tolist(), cn=g['cn'].tolist(), edges=[[int(i), int(j), list(map(int, sh))] for i, j, sh in g['edges']], A=(g['A'] / lengths.mean()).tolist(), lengths=(lengths / lengths.mean()).tolist(), space_group=sg.xhm(), crystal_system=sg.crystal_system_str(), setting=g['setting'], nodes=len(g['frac']), edge_count=len(g['edges']), mean_parent_length=float(lengths.mean()))
    graph['graph_sha256'] = identity(graph)
    return graph

def bootstrap(names=None):
    catalog = json.loads((ROOT / 'atlas/catalog.json').read_text(encoding='utf-8'))
    expected = {x['name']: x for x in catalog['graphs']}
    names = sorted(expected if names is None else set(names))
    unknown = set(names) - set(expected)
    if unknown:
        raise ValueError(f'Nets absent from frozen catalogue: {sorted(unknown)}')
    dest = cache_dir('atlas') / 'graphs'
    dest.mkdir(exist_ok=True)
    nets = None
    rows = []
    for name in names:
        path = dest / f'{name}.json'
        graph = None
        if path.exists():
            candidate = json.loads(path.read_text(encoding='utf-8'))
            content = {k: v for k, v in candidate.items() if k != 'graph_sha256'}
            if identity(content) == candidate.get('graph_sha256') == expected[name]['graph_sha256']:
                graph = candidate
        if graph is None:
            if nets is None:
                if not external_cgd().is_file():
                    raise FileNotFoundError('Set RCSR_CGD to the documented rcsr3d_web.cgd input.')
                nets = read_cgd(str(external_cgd()))
            graph = build_graph(name, nets)
            if graph['graph_sha256'] != expected[name]['graph_sha256']:
                raise ValueError(f"{name}: frozen graph identity differs; use the documented RCSR input and package versions. Expected {expected[name]['graph_sha256']}, obtained {graph['graph_sha256']}")
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(graph, indent=2, allow_nan=False), encoding='utf-8')
            tmp.replace(path)
        rows.append(graph)
    return rows
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nets', nargs='+')
    args = ap.parse_args()
    start = time.perf_counter()
    rows = bootstrap(args.nets)
    print(json.dumps({'graphs_verified': len(rows), 'seconds': time.perf_counter() - start, 'identities': {x['name']: x['graph_sha256'] for x in rows}}, indent=2))
