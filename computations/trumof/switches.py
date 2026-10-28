from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.trumof.monte_carlo import *

def run():
    r = json.load(open('trumof/rule.json'))
    a = json.load(open('trumof/approximants.json'))
    out = []
    for L in [4, 6, 8]:
        pos, inc, occ, lig = lattice(L, r)
        frozen = np.any((pos < 1) | (pos >= L - 1), axis=1)
        for z in [a[0], a[1], a[6]]:
            base = np.tile(z['q'], L ** 3)
            best = None
            for trial in range(5):
                v = mc(L, r, 700000 + 1000 * L + int(z['name'].split('_')[-1]) * 10 + trial, frozen=frozen, initial=base, sweeps=12000, Tmax=0.7, Tmin=0.08)
                if best is None or v['changed_from_initial'] > best['changed_from_initial']:
                    best = v
                if v['changed_from_initial'] > 0:
                    break
            best.update(approximant=z['name'], frozen_mask=frozen, boundary_cells=1, reference_q=z['q'])
            out.append(best)
            save('trumof/initialization/compact_switches.json', out)
            print(L, z['name'], best['changed_from_initial'], best['zero_visits'], flush=True)
if __name__ == '__main__':
    run()
