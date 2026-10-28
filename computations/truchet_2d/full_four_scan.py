"""four-cell: complete the 4x4 periodic catalogue of the iid 2D ensemble.

axial model scanned the 158 orbits of 4x4 patterns without a 2x2 checkerboard block.
Here every remaining orbit (patterns that contain a checkerboard block) is
scanned with the same Bloch/PD/LE routine (grid=4, no refinement); the 20
weakest are then re-scanned with k refinement. Output: truchet_2d/full_four_scan.json.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys, json, itertools, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
DATA_ROOT = str(_DATA_ROOT)
sys.path.insert(0, DATA_ROOT)
import numpy as np
from computations.truchet_2d.frame import canonical
from computations.truchet_2d.rules import scan, forbidden
OUT = os.path.join(str(_DATA_ROOT), 'truchet_2d/full_four_scan.json')

def dump(obj):
    tmp = OUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1)
    for attempt in range(20):
        try:
            os.replace(tmp, OUT)
            return
        except PermissionError:
            time.sleep(0.5)
    os.replace(tmp, OUT)

def main():
    out = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else dict(runs=[], refined=[])
    seen, mult = ({}, {})
    n_checker = 0
    for bits in itertools.product([0, 1], repeat=16):
        p = np.array(bits, dtype=np.uint8).reshape(4, 4)
        if not forbidden(p).any():
            continue
        n_checker += 1
        key = canonical(p)
        if key not in seen:
            seen[key] = p
            mult[key] = 0
        mult[key] += 1
    out['checker_pattern_count'] = n_checker
    out['checker_orbit_count'] = len(seen)
    done = {json.dumps(r['pattern']) for r in out['runs']}
    t0 = time.time()
    for i, (key, p) in enumerate(seen.items()):
        if json.dumps(p.tolist()) in done:
            continue
        r = scan(p, grid=4, refine=False)
        r['orbit_size'] = mult[key]
        out['runs'].append(r)
        if (i + 1) % 10 == 0:
            dump(out)
            print('checker4', i + 1, len(seen), round(time.time() - t0), min((x['pressure'] for x in out['runs'])), flush=True)
    dump(out)
    ref_done = {json.dumps(r['pattern']) for r in out['refined']}
    for j, row in enumerate(sorted(out['runs'], key=lambda x: x['pressure'])[:20]):
        if json.dumps(row['pattern']) in ref_done:
            continue
        r = scan(np.array(row['pattern'], dtype=np.uint8), grid=4, refine=True)
        r['rank_unrefined'] = j + 1
        out['refined'].append(r)
        dump(out)
        print('refine', j + 1, r['pressure'], r['class_first'], flush=True)
    checker = scan(np.array([[0, 1], [1, 0]], dtype=np.uint8), grid=4, refine=True)
    out['checker_reference'] = checker
    allp = [r['pressure'] for r in out['runs']] + [r['pressure'] for r in out['refined']]
    out['min_over_cap'] = min(allp) / checker['pressure']
    out['weakest'] = min(out['runs'] + out['refined'], key=lambda r: r['pressure'])
    dump(out)
    print('done', n_checker, len(seen), out['min_over_cap'], flush=True)
if __name__ == '__main__':
    main()
