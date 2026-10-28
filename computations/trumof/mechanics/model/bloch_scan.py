from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.trumof.mechanics.model.model import *
import time

def gridks(n=4):
    pts = np.arange(n) / n
    out = []
    for q in itertools.product(pts, repeat=3):
        q = np.array(q)
        q[q > 0.5] -= 1
        if not np.any(q):
            continue
        first = q[np.flatnonzero(q)[0]]
        if first < 0:
            continue
        out.append(q)
    return out

def bloch_min(m, x, ks):
    best = (float('inf'), None, None)
    if not hasattr(m, 'null_cache'):
        m.null_cache = {}
    for k in ks:
        K = m.bloch(x, k)
        if m.N <= 128:
            key = tuple(k)
            if key not in m.null_cache:
                ee = m.e
                phase = np.exp(2j * np.pi * (ee[:, 2:] @ k))
                B = sparse.coo_matrix((np.c_[-np.ones(len(ee)), phase].ravel(), (np.repeat(np.arange(len(ee)), 2), ee[:, :2].ravel())), shape=(len(ee), m.N)).toarray()
                ev, U = linalg.eigh(B.conj().T @ B)
                m.null_cache[key] = np.kron(U[:, ev > 1e-08], np.eye(3)) if min(ev) < 1e-08 else None
            Q = m.null_cache[key]
            dense = K.toarray()
            if Q is not None:
                val, vec = linalg.eigh(Q.conj().T @ dense @ Q, subset_by_index=[0, 0])
                vec = Q @ vec
            else:
                val, vec = linalg.eigh(dense, subset_by_index=[0, 0])
        else:
            val, vec = eigsh(K, k=6, sigma=-1e-07, which='LM', tol=2e-07)
            order = np.argsort(val)
            val = val[order]
            vec = vec[:, order]
        if val[0] < best[0]:
            best = (float(val[0]), np.array(k), vec[:, 0])
    return best

def scalar_summary(m, x, p, do_bloch=True):
    C, K, c, D = m.tangent(x, p)
    src = C + p * SOURCE_DELTA
    pd, pm = linalg.eigh(C, subset_by_index=[0, 0])
    sp = linalg.eigvalsh(src)[0]
    le, direction = le_min(C)
    sle, sdirection = le_min(src)
    if m.N <= 128:
        tr = np.zeros((3 * m.N, 3 * m.ncc))
        for i, cid in enumerate(m.cc):
            tr[3 * i:3 * i + 3, 3 * cid:3 * cid + 3] = np.eye(3)
        Q = linalg.null_space(tr.T)
        ev = linalg.eigvalsh(Q.T @ K.toarray() @ Q)
        gamma = float(ev[0]) if len(ev) else None
    else:
        gamma = float(eigsh(K[m.free][:, m.free], k=1, which='SA', return_eigenvectors=False)[0])
    micro = (None, None, None)
    if do_bloch:
        micro = bloch_min(m, x, gridks(4))
    return (dict(p_GPa=p, PD_eigen_GPa=float(pd[0]), PD_mode=pm[:, 0], LE_eigen_GPa=le, LE_direction=direction, source_PD_eigen_GPa=float(sp), source_LE_eigen_GPa=sle, gamma_nontranslation_eigen=gamma, bloch_min_eigen=micro[0], bloch_k=micro[1]), micro[2])

def run(m, name, kind, maxp=None, step=None, bloch=True):
    start = time.perf_counter()
    step = (0.075 if kind == 'soft' else 0.18) if step is None else step
    maxp = (3 if kind == 'soft' else 7) if maxp is None else maxp
    x = np.zeros(3 * m.N + 6)
    p = 0.0
    rows = []
    states = []
    roots = {}
    status = 'max_pressure'
    lastrow = None
    mode = None
    ds = step
    while p <= maxp + 1e-10:
        if p == 0:
            xx = x
            info = dict(converged=True, residual=0, iterations=0)
        else:
            xx, info = m.relax(p, x, maxiter=50)
        if not info['converged']:
            ds /= 2
            if ds < 0.0002:
                status = 'equilibrium_branch_endpoint'
                break
            p = rows[-1]['p_GPa'] + ds
            continue
        if rows and (np.max(abs(xx[-6:] - x[-6:])) > 0.025 or np.max(abs(xx[:-6] - x[:-6])) > 0.6):
            ds /= 2
            if ds < 0.0002:
                status = 'branch_tracking_failed'
                break
            p = rows[-1]['p_GPa'] + ds
            continue
        row, vec = scalar_summary(m, xx, p, bloch)
        row['equilibrium'] = info
        rows.append(row)
        states.append(xx.copy())
        print(name, kind, round(p, 5), round(row['PD_eigen_GPa'], 5), round(row['LE_eigen_GPa'], 5), flush=True)
        if lastrow is not None:
            for key in ['PD_eigen_GPa', 'LE_eigen_GPa', 'source_PD_eigen_GPa', 'source_LE_eigen_GPa', 'gamma_nontranslation_eigen', 'bloch_min_eigen']:
                v = row[key]
                pv = lastrow[key]
                if v is not None and pv is not None and (pv > 1e-08) and (v < -1e-08) and (key not in roots):
                    lo = lastrow['p_GPa']
                    hi = p
                    xl = states[-2].copy()
                    xh = xx.copy()
                    for it in range(13):
                        mid = (lo + hi) / 2
                        xm, inf = m.relax(mid, (xl + xh) / 2, maxiter=50)
                        if not inf['converged']:
                            break
                        if key == 'bloch_min_eigen':
                            value, km, vm = bloch_min(m, xm, gridks(4))
                        elif key == 'gamma_nontranslation_eigen':
                            _, Km, _, _ = m.tangent(xm, mid)
                            value = linalg.eigvalsh(Km.toarray())[3 * m.ncc]
                        else:
                            Cm, _, _, _ = m.tangent(xm, mid)
                            Cm += mid * SOURCE_DELTA if key.startswith('source') else 0
                            value = le_min(Cm)[0] if 'LE_' in key else linalg.eigvalsh(Cm)[0]
                        if value <= 0:
                            hi = mid
                            xh = xm
                        else:
                            lo = mid
                            xl = xm
                    roots[key] = dict(lower_GPa=lo, upper_GPa=hi, estimate_GPa=(lo + hi) / 2)
        physical = [v['estimate_GPa'] for k, v in roots.items() if not k.startswith('source')]
        if physical and p > min(physical) + 2 * step:
            status = 'continued_past_first_instability'
            x = xx
            mode = vec
            break
        if len(rows) > 1 and row['PD_eigen_GPa'] < rows[-2]['PD_eigen_GPa'] and (row['PD_eigen_GPa'] < 0.08):
            ds = min(ds, step / 4)
        lastrow = row
        x = xx
        mode = vec
        p += ds
    stem = name + '_' + kind
    np.savez_compressed('trumof/initialization/path_' + stem + '.npz', p=np.array([z['p_GPa'] for z in rows]), x=np.array(states))
    final = rows[-1]
    if status == 'equilibrium_branch_endpoint' and abs(final['PD_eigen_GPa']) < 0.005:
        roots['PD_endpoint'] = dict(lower_GPa=final['p_GPa'], upper_GPa=p, estimate_GPa=(final['p_GPa'] + p) / 2, note='stationary branch endpoint bracket, not a signed Hessian crossing')
    physical = [(v['estimate_GPa'], k) for k, v in roots.items() if not k.startswith('source')]
    first = min(physical) if physical else (None, None)
    pc = first[0]
    near = [k for p0, k in physical if pc is not None and abs(p0 - pc) < max(0.002, step / 20)]
    cls = 'unresolved'
    if 'PD_endpoint' in near:
        cls = 'PD-only endpoint candidate' if final['LE_eigen_GPa'] > 0.01 and (final['bloch_min_eigen'] is None or final['bloch_min_eigen'] > 0.0001) else 'endpoint mixed/unresolved'
    elif near:
        cls = '+'.join(('PD' if k == 'PD_eigen_GPa' else 'LE' if k == 'LE_eigen_GPa' else 'finite-k' if k == 'bloch_min_eigen' else 'Gamma' for k in near))
    if mode is not None:
        np.savez_compressed('trumof/initialization/mode_' + stem + '.npz', p=final['p_GPa'], k=final['bloch_k'], u=mode, positions=m.pos, cell=m.A)
    out = dict(name=name, parameters=kind, nodes=m.N, components_gamma=m.ncc, nontrivial_exact_translation_zero_modes=3 * (m.ncc - 1), status=status, roots=roots, first_pressure_GPa=pc, first_class=cls, rows=rows, seconds=time.perf_counter() - start, bloch_grid=4 if bloch else None, threshold_convention='objective Hessian of Eq.1 plus pV; source diagnostics separately')
    save('trumof/initialization/scan_' + stem + '.json', out)
    return out
if __name__ == '__main__':
    import sys
    a = json.load(open('trumof/approximants.json'))
    out = []
    selected = [a[0], a[1], a[6]] if '--small' in sys.argv else [a[0], a[1], a[6]] + a[10:]
    for kind in ['soft', 'hard']:
        for z in selected:
            filename = 'trumof/initialization/scan_' + z['name'] + '_' + kind + '.json'
            if Path(filename).exists() and '--rerun' not in sys.argv:
                v = json.load(open(filename, encoding='utf-8'))
            else:
                v = run(model_approximant(z, kind), z['name'], kind)
            out.append(v)
            save('trumof/initialization/mechanics.json', out)
