from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.trumof.mechanics.model.rule import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import spglib

def extract(path, L, r):
    st, f, labels, els = atoms(path)
    A = np.array(st.cell.orth.mat).T
    sy = np.array(els)
    df = f[None, :, :] - f[:, None, :]
    df -= np.round(df)
    dist = np.linalg.norm(df @ A, axis=2)
    oi = np.flatnonzero((sy == 'O') & ((dist[:, sy == 'Zn'] < 2.4).sum(1) == 4))
    assert len(oi) == 4 * L ** 3
    phase = np.mean(np.exp(4j * np.pi * L * f[oi]), axis=0)
    shift = np.angle(phase) / (4 * np.pi)
    rounded = np.mod(np.round(2 * (L * f[oi] - shift)) / 2, L)
    pos, inc, occ, lig = lattice(L, r)
    idx = [int(np.argmin(np.linalg.norm(np.mod(pos - x + L / 2, L) - L / 2, axis=1))) for x in rounded]
    assert len(set(idx)) == len(pos)
    node_by_o = dict(zip(oi, idx))
    ordered_oi = np.array([oi[idx.index(i)] for i in range(len(pos))])
    ci = np.flatnonzero(sy == 'C')
    adj = (dist[np.ix_(ci, ci)] < 1.8) & (dist[np.ix_(ci, ci)] > 0.5)
    nc, comp = connected_components(csr_matrix(adj))
    assert nc == 12 * L ** 3
    decorations = np.zeros((len(pos), 12), int)
    measured_edges = []
    carbon_errors = []
    car = np.array(r['carboxylate_vectors'])

    def canon(i, j, n):
        z = (int(i), int(j), *map(int, n))
        zz = (int(j), int(i), *map(int, -np.array(n)))
        return min(z, zz)
    for k in range(nc):
        cs = ci[comp == k]
        assert len(cs) == 8
        terminal = [c for c in cs if np.sum((dist[c] < 1.6) & (sy == 'O')) == 2]
        assert len(terminal) == 2
        unwrap = {int(cs[0]): f[cs[0]].copy()}
        todo = [int(cs[0])]
        while todo:
            x = todo.pop()
            for y in cs:
                if 0.5 < dist[x, y] < 1.8 and int(y) not in unwrap:
                    unwrap[int(y)] = unwrap[x] + df[x, y]
                    todo.append(int(y))
        assert len(unwrap) == 8
        ends = []
        for c in terminal:
            o = oi[np.argmin(dist[c, oi])]
            i = node_by_o[o]
            dv = f[c] - f[o]
            dv -= np.round(dv)
            v = dv * L
            ring = np.mean([unwrap[x] for x in cs if x not in terminal], axis=0) * L - shift
            lv = ring - pos[i]
            lv -= L * np.round(lv / L)
            a = int(np.argmin(np.linalg.norm(np.array(r['arms']) - lv, axis=1)))
            decorations[i, a] += 1
            carbon_errors.append(np.linalg.norm(car[a] - v) * r['a'])
            u = unwrap[c] + df[c, o]
            image = np.rint((u * L - shift - pos[i]) / L).astype(int)
            ends.append((i, image))
        (i, ni), (j, nj) = ends
        measured_edges.append(canon(i, j, nj - ni))
    states = np.array(r['states'])
    qs = []
    mismatch = []
    for i, d in enumerate(decorations):
        err = np.sum(abs(states - d), axis=1)
        qs.append(int(np.argmin(err)))
        mismatch.append(int(min(err)))
    assert max(mismatch) == 0, (path.name, mismatch)
    assert energy(qs, r, L) == 0
    edges = graph(qs, r, L)
    rule_edges = sorted((canon(e[0], e[1], e[2:]) for e in edges))
    assert rule_edges == sorted(measured_edges), path.name
    dec = np.array([np.mod((pos[i] + car[a]) / L, 1) for i in range(len(pos)) for a in np.flatnonzero(decorations[i])])
    sym = spglib.get_symmetry_dataset((np.eye(3) * L * r['a'], dec, np.ones(len(dec), int)), symprec=0.001)
    K = 4
    points = np.array(list(itertools.product(range(K), repeat=3)))
    nn = len(pos)
    rows = []
    cols = []
    for c in points:
        u = np.ravel_multi_index(c, (K, K, K))
        for i, j, *n in edges:
            v = np.ravel_multi_index((c + np.array(n)) % K, (K, K, K))
            rows.extend([u * nn + i, v * nn + j])
            cols.extend([v * nn + j, u * nn + i])
    ncomp, cc = connected_components(csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(K ** 3 * nn,) * 2))
    return dict(file=str(path), name=path.stem, L=L, nodes=len(pos), edges=edges, q=qs, rule_energy=0, local_rule_mismatches=max(mismatch), extracted_edges_match_rule=True, ligand_count=nc, carboxylate_mapping_max_error_A=max(carbon_errors), cluster_offset_conventional=shift, ideal_spacegroup=sym.international, ideal_spacegroup_number=int(sym.number), components_4repeat=int(ncomp), component_sizes=np.bincount(cc), positions=pos, cell=np.eye(3) * L * r['a'])
if __name__ == '__main__':
    r = json.load(open('trumof/rule.json', encoding='utf-8'))
    out = []
    for L in [1, 2]:
        for k in range(1, 11 if L == 1 else 21):
            path = RAW / f'science.ade5239_data_s6/{L}x{L}x{L}/trumof_{L}x{L}x{L}_{k}.cif'
            v = extract(path, L, r)
            out.append(v)
            print(v['name'], v['ideal_spacegroup'], v['components_4repeat'], v['carboxylate_mapping_max_error_A'], flush=True)
            save('trumof/approximants.json', out)
