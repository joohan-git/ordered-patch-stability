"""correlated: correlated growth inside the TRUMOF-1 Truchet rule.

Rule-satisfying samples are drawn by the TRUMOF annealing Monte Carlo with an extra
ordering field h that favours the Pc approximant (trumof_1x1x1_7, the weakest
approximant and the cap of the rule): each cluster whose orientation differs from
the tiled Pc assignment costs h. The rule term is annealed to zero violations;
h is held fixed, so it plays the role of a growth bias toward the lower-energy
ordered configuration (Meekel 2023 Science, Fig. 3A). For each sample the TRUMOF
onset routine (soft parameters, objective -p tangent) gives the threshold; we
record the Pc fraction, the largest connected Pc cluster and the fraction of the
onset-mode amplitude inside it. The launcher prepares the configurable cache and copies required JSON inputs.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys, json, time, shutil
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
HERE = str(_DATA_ROOT)
DATA_ROOT = HERE
from computations.common.runtime import cache_dir, output_dir
WORK = str(cache_dir('trumof_correlated'))
os.makedirs(WORK, exist_ok=True)
for fn in ('trumof/rule.json', 'trumof/pcu_parameters.json', 'trumof/approximants.json'):
    if not os.path.exists(os.path.join(WORK, fn)):
        shutil.copyfile(os.path.join(DATA_ROOT, fn), os.path.join(WORK, fn))
sys.path.insert(0, DATA_ROOT)
os.chdir(WORK)
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from computations.trumof.mechanics.model.rule import lattice, graph
from computations.trumof.ensemble_mechanics import sample_model as _sample_model, onset

def sample_model(z, r):
    m = _sample_model(z, r)
    m.th = np.where(m.th > np.pi - 1e-06, np.pi, m.th)
    return m
OUT = str(output_dir() / 'trumof/correlated.json')
L = 6
H = [0.0, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
TARGETS = {'Pc': 'trumof_1x1x1_7', 'P21': 'trumof_1x1x1_2'}
NSAMP = 5

def dump(obj):
    tmp = OUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1)
    for _ in range(20):
        try:
            os.replace(tmp, OUT)
            return
        except PermissionError:
            time.sleep(0.5)
    os.replace(tmp, OUT)

def mc_field(L, r, qpc, h, seed, sweeps=9000, Tmax=1.4, Tmin=0.06):
    """TRUMOF annealing for the parity rule, plus a fixed field h on Pc mismatches."""
    rng = np.random.default_rng(seed)
    pos, inc, occ, lig = lattice(L, r)
    states = np.array(r['states'], np.int8)
    N = len(pos)
    q = rng.integers(24, size=N)
    target = np.tile(qpc, L ** 3)
    counts = np.zeros(len(occ), np.int16)
    np.add.at(counts, inc.ravel(), states[q].ravel())
    groups = [np.flatnonzero(np.arange(N) % 4 == b) for b in range(4)]
    best = None
    for sweep in range(sweeps):
        T = Tmax * (Tmin / Tmax) ** (sweep / max(sweeps - 1, 1))
        for g in groups:
            prop = rng.integers(24, size=len(g))
            delta = states[prop] - states[q[g]]
            curr = counts[inc[g]]
            de = ((curr + delta) % 2 - curr % 2).sum(1)
            dm = (prop != target[g]).astype(float) - (q[g] != target[g]).astype(float)
            arg = -de / T - h * dm
            acc = rng.random(len(g)) < np.exp(np.minimum(0, arg))
            ids = g[acc]
            counts[inc[ids]] += delta[acc]
            q[ids] = prop[acc]
        E = int(np.sum(counts % 2))
        if E == 0 and T < 0.12:
            best = q.copy()
            break
    return dict(L=L, seed=seed, h=h, q=(best if best is not None else q).tolist(), energy=int(np.sum(counts % 2)), sweeps=sweep + 1, pc_fraction=float(np.mean(q == target)))

def pc_clusters(L, r, q, qpc):
    pos, inc, occ, lig = lattice(L, r)
    match = np.array(q) == np.tile(qpc, L ** 3)
    rows, cols = ([], [])
    for t in occ:
        ids = [i for i, a in t]
        for a in ids:
            for b in ids:
                if a != b and match[a] and match[b]:
                    rows.append(a)
                    cols.append(b)
    n = len(pos)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    nc, lab = connected_components(A, directed=False)
    lab = np.where(match, lab, -1)
    sizes = np.bincount(lab[lab >= 0]) if match.any() else np.array([0])
    big = int(np.argmax(sizes)) if match.any() else -1
    return (match, lab == big, int(sizes.max()) if match.any() else 0)

def main():
    r = json.load(open('trumof/rule.json', encoding='utf-8'))
    approx = json.load(open('trumof/approximants.json', encoding='utf-8'))
    out = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else dict(L=L, runs=[], tiled={})
    done = {(x['target'], x['h'], x['sample']) for x in out['runs']}
    for tg, nm in TARGETS.items():
        qt = [a for a in approx if a['name'] == nm][0]['q']
        if tg not in out['tiled']:
            res = onset(sample_model(dict(L=L, q=np.tile(qt, L ** 3).tolist()), r), '%s_tiled_L%d' % (tg, L))
            out['tiled'][tg] = dict(threshold_GPa=res['stable_Gamma_cell_lower_GPa'], status=res['status'], bloch_min=res['bloch_at_lower_min'])
            dump(out)
            print(tg, 'tiled', res['stable_Gamma_cell_lower_GPa'], flush=True)
        for h in H:
            if tg == 'P21' and h == 0.0:
                continue
            for s in range(NSAMP):
                if (tg, h, s) in done:
                    continue
                seed = 16000 + (0 if tg == 'Pc' else 500000) + int(h * 100) * 10 + s
                z = mc_field(L, r, qt, h, seed)
                tries = 0
                while z['energy'] != 0 and tries < 6:
                    seed += 1000
                    tries += 1
                    z = mc_field(L, r, qt, h, seed, sweeps=14000)
                row = dict(target=tg, h=h, sample=s, seed=z['seed'], rule_energy=z['energy'])
                if z['energy'] == 0:
                    match, big, nbig = pc_clusters(L, r, z['q'], qt)
                    name = '%s_h%.1f_s%d' % (tg, h, s)
                    res = onset(sample_model(dict(L=L, q=z['q']), r), name)
                    u = np.load('trumof/initialization/ensemble_state_' + name + '.npz', allow_pickle=True)['u']
                    frac = None
                    if u is not None and np.size(u) > 1:
                        nodal = np.sum(np.abs(np.asarray(u).reshape(-1, 3)) ** 2, axis=1)
                        frac = float(nodal[big].sum() / nodal.sum())
                    row.update(order_fraction=float(match.mean()), largest_cluster=nbig, mode_in_largest=frac, threshold_GPa=res['stable_Gamma_cell_lower_GPa'], status=res['status'], bloch_min=res['bloch_at_lower_min'], participation=res.get('participation'), seconds=res['seconds'])
                out['runs'].append(row)
                dump(out)
                print(tg, 'h', h, 's', s, row.get('order_fraction'), row.get('largest_cluster'), row.get('threshold_GPa'), row.get('mode_in_largest'), flush=True)
    print('done', flush=True)
if __name__ == '__main__':
    main()
