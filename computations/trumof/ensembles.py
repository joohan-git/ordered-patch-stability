from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.trumof.monte_carlo import *

def run():
    r = json.load(open('trumof/rule.json', encoding='utf-8'))
    approx = json.load(open('trumof/approximants.json', encoding='utf-8'))
    existing = json.load(open('trumof/annealed_configurations.json', encoding='utf-8'))
    out = []
    for L in [2, 3, 4, 6, 8]:
        candidates = [z for z in existing if z['L'] == L and z['energy'] == 0]
        seed = 1000 + L * 100
        while len(candidates) < 10:
            z = mc(L, r, seed, sweeps=7000)
            seed += 1
            if z['energy'] == 0:
                candidates.append(z)
            print('random', L, len(candidates), z['energy'], flush=True)
            if seed >= 1000 + L * 100 + 60:
                break
        out.extend(candidates[:10])
        save('trumof/random_configurations.json', out)
    patches = []
    for L in [4, 6, 8]:
        pos, inc, occ, lig = lattice(L, r)
        for rep in [approx[0], approx[1], approx[6]]:
            base = np.tile(rep['q'], L ** 3)
            for R in sorted(set([1, 2, L // 2, L - 2])):
                corner = (L - R) // 2
                frozen = np.all((pos >= corner) & (pos < corner + R), axis=1)
                z = mc(L, r, 90000 + 100 * L + 10 * R + int(rep['name'].split('_')[-1]), frozen=frozen, initial=base, sweeps=9000)
                z.update(approximant=rep['name'], R_cells=R, frozen_mask=frozen, reference_q=rep['q'])
                if z['energy'] != 0:
                    for trial in range(4):
                        retry = mc(L, r, z['seed'] + trial + 1, frozen=frozen, initial=base, sweeps=13000)
                        if retry['energy'] == 0:
                            z.update(retry)
                            break
                patches.append(z)
                save('trumof/planted_configurations.json', patches)
                print('patch', L, rep['name'], R, z['energy'], z['changed_from_initial'], flush=True)
if __name__ == '__main__':
    run()
