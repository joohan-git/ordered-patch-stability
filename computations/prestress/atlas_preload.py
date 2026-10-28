"""All 1752 hydrostatic preloads, with force normalization."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.check_common import *
import time

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+')
    args = ap.parse_args()
    rows = []
    start = time.time()
    for idx, reference in enumerate(atlas_rows()):
        if args.nets and reference['name'] not in args.nets:
            continue
        name = reference['name']
        r = reference['r_over_each_parent_length']
        target = DEST / 'preload' / f'atlas/check_{name}_{r:.2f}.json'
        if target.exists():
            rows.append(read(target))
            continue
        f = frame(name, r)
        f.prestress('stress')
        ef = f.pre['full_element_forces']
        nodal = np.zeros(f.ndof)
        for e, force in zip(f.elements, ef):
            np.add.at(nodal, e['di'], force[:6])
            np.add.at(nodal, e['dj'], force[6:])
        absres = float(f.pre['absolute_residual'])
        norm = float(la.norm(ef))
        row = dict(name=name, r=r, abs_residual=absres, force_normalized_residual=absres / norm, full_element_force_norm=norm, baseline_relative_residual=reference['preload_equilibrium'], recomputed_relative_residual=float(f.pre['relative_residual']), scattered_force_residual=float(la.norm(nodal)), scattered_force_normalized_residual=float(la.norm(nodal) / norm), hydrostatic_stress_relative_error=float(la.norm(f.pre['stress'] - np.eye(3) * np.trace(f.pre['stress']) / 3) / la.norm(f.pre['stress'])), graph_sha256=graph(name)['graph_sha256'], requires_resolve=absres / norm > 1e-08)
        save(target, row)
        rows.append(row)
        if (idx + 1) % 50 == 0:
            print('preload', idx + 1, len(atlas_rows()), round(time.time() - start, 1), flush=True)
    bad = [x for x in rows if x['requires_resolve']]
    assert not bad, 'Resolve required; do not publish until addressed: ' + str(bad)
    out = dict(model=raw('cco', 0.02)['model'], n_rows=len(rows), rows=rows, definition='Euclidean norm(K0 w + f_affine), including pinned translation equations, divided by Frobenius norm of all 12-component element end forces; hydrostatic relaxed preload of mean strain -1.', absolute_units='E_s=1, mean parent length=1; generalized nodal force/torque Euclidean norm, as in baseline.', summary={key: stats([x[key] for x in rows]) for key in ['abs_residual', 'force_normalized_residual', 'scattered_force_normalized_residual', 'hydrostatic_stress_relative_error']}, by_radius={str(r): {key: stats([x[key] for x in rows if x['r'] == r]) for key in ['abs_residual', 'force_normalized_residual']} for r in RADII}, above_1e8=bad, resolved_rows=[], class_or_threshold_changes=[], threadpools=POOLS, seconds=time.time() - start)
    publish('prestress/atlas_preload.json', out)
    print('DONE', out['summary'], flush=True)
if __name__ == '__main__':
    main()
