"""blocked-tiling: fixed-pressure thresholds of the periodic tilings (periods up to 3x3) that the no-checker rule blocks,
so that all tilings of the three rule catalogues can be ordered on one scale (Supplementary tiling gallery).
Same routine as rules.scan (grid 4, local wavevector refinement). Output truchet_2d/blocked_pressure.json."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys, json
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
DATA_ROOT = str(_DATA_ROOT)
sys.path.insert(0, DATA_ROOT)
import numpy as np
from computations.truchet_2d.rules import scan, forbidden
PE = json.load(open(os.path.join(DATA_ROOT, 'truchet_2d/periodic.json'), encoding='utf-8'))
out = []
for r in PE['runs']:
    p = np.array(r['pattern'], dtype=np.uint8)
    if not forbidden(p).any():
        continue
    s = scan(p, grid=4, refine=True)
    out.append(dict(pattern=r['pattern'], pressure=s['pressure'], class_first=s['class_first'], k=s['k'], strain_critical=r['critical']))
    print(r['pattern'], s['pressure'], s['class_first'], flush=True)
json.dump(dict(runs=out), open(os.path.join(str(_DATA_ROOT), 'truchet_2d/blocked_pressure.json'), 'w', encoding='utf-8'), indent=1)
print('done', len(out))
