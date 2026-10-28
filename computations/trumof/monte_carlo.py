from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.trumof.mechanics.model.rule import *
import time

def mc(L, r, seed=1, frozen=None, initial=None, sweeps=2500, Tmax=1.4, Tmin=0.06):
    rng = np.random.default_rng(seed)
    pos, inc, occ, lig = lattice(L, r)
    states = np.array(r['states'], np.int8)
    N = len(pos)
    q = rng.integers(24, size=N) if initial is None else np.array(initial).copy()
    frozen = np.zeros(N, bool) if frozen is None else np.array(frozen, bool)
    counts = np.zeros(len(occ), np.int16)
    np.add.at(counts, inc.ravel(), states[q].ravel())
    groups = [np.flatnonzero((np.arange(N) % 4 == b) & ~frozen) for b in range(4)]
    assert all((len(np.unique(inc[g])) == len(g) * 12 for g in groups))
    start = time.perf_counter()
    lowest = int(np.sum(counts % 2))
    best = q.copy()
    zeros = 0
    for sweep in range(sweeps):
        T = Tmax * (Tmin / Tmax) ** (sweep / max(sweeps - 1, 1))
        for g in groups:
            if len(g) == 0:
                continue
            proposal = rng.integers(24, size=len(g))
            delta = states[proposal] - states[q[g]]
            curr = counts[inc[g]]
            de = ((curr + delta) % 2 - curr % 2).sum(1)
            acc = (de <= 0) | (rng.random(len(g)) < np.exp(np.minimum(0, -de / T)))
            ids = g[acc]
            counts[inc[ids]] += delta[acc]
            q[ids] = proposal[acc]
        E = int(np.sum(counts % 2))
        if E < lowest:
            lowest = E
            best = q.copy()
        if E == 0:
            best = q.copy()
            zeros += 1
        if E == 0 and T < 0.12:
            break
    return dict(L=L, seed=seed, q=best, energy=lowest, sweeps=sweep + 1, seconds=time.perf_counter() - start, zero_visits=zeros, frozen_nodes=int(sum(frozen)), changed_from_initial=None if initial is None else int(sum(best != initial)))
if __name__ == '__main__':
    r = json.load(open('trumof/rule.json', encoding='utf-8'))
    out = []
    for L in [2, 3, 4]:
        for seed in range(10):
            z = mc(L, r, seed + 100 * L, sweeps=5000)
            out.append(z)
            save('trumof/annealed_configurations.json', out)
            print(L, seed, z['energy'], z['sweeps'], round(z['seconds'], 2), flush=True)
