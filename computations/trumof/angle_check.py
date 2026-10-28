"""TRUMOF onset sensitivity to the straight-reference-angle tolerance.

Angles within 1e-6 of pi use the linear-angle energy branch. This includes
arccos rounding deviations of approximately 1.5e-8. Recompute the onset and
Bloch minimum for the specified samples and compare their recorded pressures.
Output: trumof/angle_check.json.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys, json, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
HERE = str(_DATA_ROOT)
DATA_ROOT = HERE
sys.path.insert(0, DATA_ROOT)
import computations.trumof.correlated as correlated
os.chdir(correlated.WORK)
import numpy as np
import computations.trumof.ensemble_mechanics as em
from computations.common.runtime import output_dir
OUT = str(output_dir() / 'trumof/angle_check.json')

def angle_regularized_model(z, r):
    m = em.sample_model(z, r)
    snap = m.th > np.pi - 1e-06
    m.th = np.where(snap, np.pi, m.th)
    return (m, int(snap.sum()))

def main():
    r = json.load(open('trumof/rule.json', encoding='utf-8'))
    approx = json.load(open('trumof/approximants.json', encoding='utf-8'))
    ens = json.load(open(os.path.join(DATA_ROOT, 'trumof/random_configurations.json'), encoding='utf-8'))
    rec = {(x['L'], x['seed']): x['stable_Gamma_cell_lower_GPa'] for x in json.load(open(os.path.join(DATA_ROOT, 'trumof/inputs/ensemble_continuation.json'), encoding='utf-8')) if x['kind'] == 'random'}
    common_examples = [('pc_tiled_L2', dict(L=2, q=np.tile([a for a in approx if a['name'] == 'trumof_1x1x1_7'][0]['q'], 8).tolist()), None)]
    for L in (2, 3, 4):
        for z in [z for z in ens if z['L'] == L][:3]:
            common_examples.append(('random_L%d_seed%d' % (L, z['seed']), z, rec.get((L, z['seed']))))
    out = []
    for name, z, reference in common_examples:
        m, n = angle_regularized_model(z, r)
        t0 = time.time()
        res = em.onset(m, 'angle_regularized_' + name)
        row = dict(name=name, snapped_angles=n, recorded_GPa=reference, angle_regularized_GPa=res['stable_Gamma_cell_lower_GPa'], status=res['status'], bloch_min=res['bloch_at_lower_min'], seconds=time.time() - t0)
        out.append(row)
        json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
        print(row, flush=True)
if __name__ == '__main__':
    main()
