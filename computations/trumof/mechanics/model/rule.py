from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json, itertools
from pathlib import Path
import numpy as np
import gemmi
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
BASE = Path(__import__('os').environ.get('MEEKEL_ROOT', str(_DATA_ROOT / 'input')))
RAW = BASE / 'data_Meekel2023_Science/raw'

def save(name, x):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(x, f, indent=2, default=lambda z: z.tolist() if hasattr(z, 'tolist') else int(z))

def atoms(path):
    b = gemmi.cif.read_file(str(path)).sole_block()
    st = gemmi.make_small_structure_from_block(b)
    return (st, np.array([[s.fract.x, s.fract.y, s.fract.z] for s in st.sites]), [s.label for s in st.sites], [s.element.name for s in st.sites])

def reconstruct():
    st, _, _, _ = atoms(RAW / 'science.ade5239_data_s5/trumof_100K.cif')
    ops = list(gemmi.find_spacegroup_by_number(216).operations())

    def orbit(label):
        q = next((s for s in st.sites if s.label == label))
        return np.unique(np.round([np.mod(op.apply_to_xyz([q.fract.x, q.fract.y, q.fract.z]), 1) for op in ops], 7), axis=0)
    nodes = orbit('O1')
    carb = orbit('C1')
    oxy = orbit('O2')
    car = carb - np.round(carb)
    car = car[np.linalg.norm(car, axis=1) < 0.3]
    o = oxy - np.round(oxy)
    o = o[np.linalg.norm(o, axis=1) < 0.3]
    opairs = [o[np.argsort(np.linalg.norm(o - c, axis=1))[:2]] for c in car]
    assert len(car) == 12 and len(o) == 24
    groups = {}
    for i, c in enumerate(car):
        ax = int(np.argmax(abs(c)))
        groups.setdefault((ax, int(np.sign(c[ax]))), []).append(i)
    assert len(groups) == 6 and all((len(g) == 2 for g in groups.values()))
    decs = []
    scores = []
    for choices in itertools.product(*groups.values()):
        selected = np.array([opairs[i] for i in choices])
        ds = []
        for i, j in itertools.combinations(range(6), 2):
            ds.extend(cdist(selected[i], selected[j]).ravel())
        dec = np.zeros(12, dtype=int)
        dec[list(choices)] = 1
        decs.append(dec)
        scores.append(sum(1 / np.array(ds)))
    states = np.array(decs)[np.isclose(scores, min(scores), rtol=1e-10, atol=1e-07)]
    print('local minimum-repulsion degeneracy', len(states), 'repulsion minimum', min(scores))
    ringatoms = np.r_[orbit('C2A'), orbit('C3')]
    delta = ringatoms[None, :, :] - ringatoms[:, None, :]
    delta -= np.round(delta)
    distances = np.linalg.norm(delta, axis=2) * st.cell.a
    adj = (distances < 1.6) & (distances > 0.5)
    nc, labels = connected_components(csr_matrix(adj))
    assert nc == 16
    centres = []
    for label in range(nc):
        group = np.flatnonzero(labels == label)
        assert len(group) == 6
        unwrap = {int(group[0]): ringatoms[group[0]]}
        todo = [int(group[0])]
        while todo:
            i = todo.pop()
            for j in group:
                if adj[i, j] and int(j) not in unwrap:
                    unwrap[int(j)] = unwrap[i] + delta[i, j]
                    todo.append(int(j))
        centres.append(np.mod(np.mean(list(unwrap.values()), axis=0), 1))
    centres = np.array(centres)
    arms = []
    expanded = np.array([p + sh for sh in itertools.product([-1, 0, 1], repeat=3) for p in nodes])
    ideal_centres = []
    for c in centres:
        nearest = expanded[np.argsort(np.linalg.norm(expanded - c, axis=1))[:3]]
        assert np.max(np.linalg.norm(nearest[:, None] - nearest[None, :], axis=2)) < 0.708
        ideal_centres.append(np.mod(nearest.mean(axis=0), 1))
    ideal_centres = np.array(ideal_centres)
    for c in car:
        dv = ringatoms - c
        dv -= np.round(dv)
        ring = labels[np.argmin(np.linalg.norm(dv, axis=1))]
        v = ideal_centres[ring] - np.round(ideal_centres[ring])
        arms.append(v)
    arms = np.array(arms)
    print('centroid grid residual', np.max(abs(arms - np.round(arms * 6) / 6)))
    centroid_residual = float(np.max(abs(arms - np.round(arms * 6) / 6)))
    assert centroid_residual < 0.002
    arms = np.round(arms * 6) / 6
    return dict(a=st.cell.a, nodes=nodes, carboxylate_vectors=car, oxygen_pairs=opairs, arms=arms, ligand_centres_from_C2A_C3=centres, ligand_triplet_centroids=ideal_centres, ligand_sites=nc, ligand_centroid_sixth_grid_max_residual_fractional=centroid_residual, states=states, local_selection='minimum sum inverse inter-carboxylate O-O distances; same 24 for powers 1,2,6,12', local_repulsion_min=min(scores), symmetry_ops=[o.triplet() for o in ops])

def lattice(L, r):
    basis = np.array(r['nodes'])
    arms = np.array(r['arms'])
    cells = np.array(list(itertools.product(range(L), repeat=3)))
    pos = np.array([c + b for c in cells for b in basis])
    ligand_keys = {}
    inc = np.empty((len(pos), 12), int)
    occ = []
    for i, p in enumerate(pos):
        for a, d in enumerate(arms):
            key = tuple(np.rint(np.mod((p + d) * 6, 6 * L)).astype(int) % (6 * L))
            if key not in ligand_keys:
                ligand_keys[key] = len(ligand_keys)
                occ.append([])
            t = ligand_keys[key]
            inc[i, a] = t
            occ[t].append((i, a))
    assert all((len(t) == 3 for t in occ))
    return (pos, inc, np.array(occ), np.array(list(ligand_keys)) / 6)

def energy(q, r, L=1):
    pos, inc, occ, lig = lattice(L, r)
    s = np.array(r['states'])[list(q)]
    counts = s[occ[:, :, 0], occ[:, :, 1]].sum(1)
    return int(np.sum(counts % 2))

def graph(q, r, L=1):
    pos, inc, occ, lig = lattice(L, r)
    s = np.array(r['states'])[list(q)]
    arms = np.array(r['arms'])
    edges = []
    for t, ports in enumerate(occ):
        active = [(i, a) for i, a in ports if s[i, a]]
        assert len(active) in [0, 2]
        if not active:
            continue
        (i, a), (j, b) = active
        d = arms[a] - arms[b]
        n = np.rint((pos[i] + d - pos[j]) / L).astype(int)
        assert np.linalg.norm(pos[j] + L * n - pos[i] - d) < 1e-08
        edges.append([i, j, *n])
    return np.array(edges)

def enumerate_rule(r):
    pos, inc, occ, lig = lattice(1, r)
    states = np.array(r['states'])
    good = []
    for rest in itertools.product(range(len(states)), repeat=3):
        q = (0,) + rest
        counts = states[list(q)][occ[:, :, 0], occ[:, :, 1]].sum(1)
        if not np.any(counts % 2):
            good.append(q)
    print('ground states', good)
    car = np.array(r['carboxylate_vectors'])

    def decorated(q):
        return np.array([np.mod(p + car[a], 1) for i, p in enumerate(pos) for a in np.flatnonzero(states[q[i]])])
    keys = []
    for q in good:
        d = decorated(q)
        images = []
        for op in gemmi.find_spacegroup_by_number(216).operations():
            z = np.round(np.mod([op.apply_to_xyz(x) for x in d], 1), 6)
            z[z >= 1] = 0
            images.append(tuple(sorted(map(tuple, z))))
        keys.append(min(images))
    unique = list(dict.fromkeys(keys))
    result = dict(trials=len(states) ** 3, ground_states=good, symmetry_classes=[unique.index(k) for k in keys], orbit_count=len(unique), ground_graphs=[graph(q, r).tolist() for q in good])
    save('trumof/enumeration.json', result)
    return result
if __name__ == '__main__':
    r = reconstruct()
    save('trumof/rule.json', r)
    enumerate_rule(r)
