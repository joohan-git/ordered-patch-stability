from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.trumof.mechanics.model.model import *
from computations.trumof.mechanics.model.bloch_scan import bloch_min
import time

def sample_model(z, r):
    L = z['L']
    pos, _, _, _ = lattice(L, r)
    par = params('soft')
    return CG(pos * r['a'], np.eye(3) * L * r['a'], graph(z['q'], r, L), par['kr'], par['kt'])

def metrics(m, x, p):
    C, K, _, _ = m.tangent(x, p)
    kv = float(eigsh(K[m.free][:, m.free], k=1, which='SA', tol=1e-07, return_eigenvectors=False)[0])
    return dict(PD=float(linalg.eigvalsh(C)[0]), gamma_pinned=kv, source_PD=float(linalg.eigvalsh(C + p * SOURCE_DELTA)[0]))

def pressure_patch(m, x, p, mask):
    F, u, w, aw, E, g, h, ag, ah = m.local(x)
    V = m.V0 * np.linalg.det(F)
    radial = np.sum(g * w, axis=1)
    angular = np.sum(ag * aw, axis=1)
    node = np.zeros(m.N)
    np.add.at(node, m.e[:, 0], radial / 2)
    np.add.at(node, m.e[:, 1], radial / 2)
    np.add.at(node, m.ai, angular)
    pp = -sum(node[mask]) / (3 * V * sum(mask) / m.N)
    return dict(s_P=pp / p, patch_pressure_GPa=pp, global_pressure_from_virial=-sum(node) / (3 * V), convention='radial energy virial half to each endpoint, angular virial to centre; current volume per node')

def onset(m, name, patch=None):
    start = time.perf_counter()
    x = np.zeros(3 * m.N + 6)
    lo = 0.0
    ds = 0.1
    rows = []
    source = None
    status = 'max_pressure'
    hi = None
    for iteration in range(120):
        p = lo + ds
        xx, inf = m.relax(p, x, maxiter=45)
        if not inf['converged'] or max(abs(xx[-6:] - x[-6:])) > 0.035 or max(abs(xx[:-6] - x[:-6])) > 0.7:
            ds /= 2
            if ds < 0.0002:
                status = 'continuation_censored'
                hi = p
                break
            continue
        met = metrics(m, xx, p)
        rows.append(dict(p_GPa=p, **met, residual=inf['residual']))
        if source is None and met['source_PD'] <= 0:
            sl, sh = (lo, p)
            xs = x.copy()
            for it in range(10):
                sm = (sl + sh) / 2
                xm, check = m.relax(sm, xs)
                if not check['converged']:
                    break
                Cm, _, _, _ = m.tangent(xm, sm)
                if linalg.eigvalsh(Cm + sm * SOURCE_DELTA)[0] <= 0:
                    sh = sm
                else:
                    sl = sm
                    xs = xm
            source = (sl + sh) / 2
        if min(met['PD'], met['gamma_pinned']) < -1e-07:
            hi = p
            status = 'signed_Gamma_or_cell_crossing'
            for it in range(11):
                pm = (lo + hi) / 2
                xm, check = m.relax(pm, x)
                if not check['converged']:
                    hi = pm
                    continue
                mt = metrics(m, xm, pm)
                if min(mt['PD'], mt['gamma_pinned']) < 0:
                    hi = pm
                else:
                    lo = pm
                    x = xm
            break
        lo = p
        x = xx
        if lo >= 3.5:
            break
        if iteration % 5 == 0:
            print(name, round(lo, 4), round(met['PD'], 5), round(met['gamma_pinned'], 5), flush=True)
    final = metrics(m, x, lo)
    ks = [np.array([0.5, 0, 0]), np.array([0.5, 0.5, 0]), np.array([0.5, 0.5, 0.5]), np.array([0.25, 0.25, 0.25])]
    be, bk, bv = bloch_min(m, x, ks)
    out = dict(name=name, nodes=m.N, status=status, stable_Gamma_cell_lower_GPa=lo, upper_GPa=hi, source_PD_GPa=source, at_lower=final, bloch_at_lower_min=be, bloch_at_lower_k=bk, first_instability_scope='Gamma/cell threshold with four supercell Bloch checks at lower endpoint; not complete BZ', rows=rows, seconds=time.perf_counter() - start)
    if patch is not None:
        out['patch_stress'] = pressure_patch(m, x, lo, np.array(patch, bool))
    if bv is not None:
        nodal = np.sum(abs(bv.reshape(-1, 3)) ** 2, axis=1)
        out['participation'] = float(sum(nodal) ** 2 / (m.N * sum(nodal ** 2)))
        if patch is not None:
            out['patch_displacement_fraction'] = float(sum(nodal[np.array(patch, bool)]) / sum(nodal))
    np.savez_compressed('trumof/initialization/ensemble_state_' + name + '.npz', x=x, p=lo, u=bv, k=bk, positions=m.pos, cell=m.A)
    save('trumof/initialization/ensemble_scan_' + name + '.json', out)
    return out
if __name__ == '__main__':
    r = json.load(open('trumof/rule.json'))
    out = []
    samples = json.load(open('trumof/random_configurations.json'))
    for z in samples:
        if z['L'] > 4:
            continue
        name = f"random_L{z['L']}_seed{z['seed']}"
        fn = Path('trumof/initialization/ensemble_scan_' + name + '.json')
        row = json.load(open(fn)) if fn.exists() else onset(sample_model(z, r), name)
        row.update(L=z['L'], seed=z['seed'], kind='random')
        out.append(row)
        save('trumof/initialization/ensemble_mechanics.json', out)
    patches = json.load(open('trumof/planted_configurations.json'))
    for z in patches:
        if z['L'] > 6 or z['changed_from_initial'] == 0:
            continue
        if z['L'] == 6 and (z['approximant'] != 'trumof_1x1x1_1' or z['R_cells'] not in [2, 4]):
            continue
        name = f"planted_L{z['L']}_R{z['R_cells']}_P{z['approximant'].split('_')[-1]}"
        fn = Path('trumof/initialization/ensemble_scan_' + name + '.json')
        row = json.load(open(fn)) if fn.exists() else onset(sample_model(z, r), name, patch=z['frozen_mask'])
        row.update(L=z['L'], R=z['R_cells'], approximant=z['approximant'], kind='planted')
        out.append(row)
        save('trumof/initialization/ensemble_mechanics.json', out)
