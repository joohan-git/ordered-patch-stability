"""Exact load-dependent elimination of strut-interior DOFs (no reduction)."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.mesh_common import *
from scipy.optimize import brentq
import time

def install(f):
    div = f.subdivisions
    ne = len(f.data['edges'])
    nj = len(f.data['frac'])
    nd = 6 * nj
    kk = []
    hh = []
    b0 = []
    b1 = []
    mus = []
    for p in range(ne):
        n = 6 * (div + 1)
        K = np.zeros((n, n))
        H = K.copy()
        for h in range(div):
            e = f.elements[p * div + h]
            idx = np.arange(h * 6, h * 6 + 12)
            K[np.ix_(idx, idx)] += e['K']
            H[np.ix_(idx, idx)] -= f.Gm[p * div + h]
        end = np.r_[np.arange(6), np.arange(n - 6, n)]
        mid = np.arange(6, n - 6)
        mu, V = la.eigh(H[np.ix_(mid, mid)], K[np.ix_(mid, mid)], check_finite=False)
        kk.append(K[np.ix_(end, end)])
        hh.append(H[np.ix_(end, end)])
        b0.append(V.T @ K[np.ix_(mid, end)])
        b1.append(V.T @ H[np.ix_(mid, end)])
        mus.append(mu)
    K = np.asarray(kk)
    H = np.asarray(hh)
    B0 = np.asarray(b0)
    B1 = np.asarray(b1)
    mu = np.asarray(mus)
    pole = 1 / np.max(mu)
    upper = pole * (1 - 2e-10)
    row = []
    col = []
    delta = []
    diag = np.zeros(nd)
    for p, (i, j, sh) in enumerate(f.data['edges']):
        ids = np.r_[6 * i + np.arange(6), 6 * j + np.arange(6)]
        rr, cc = np.meshgrid(ids, ids, indexing='ij')
        row.extend(rr.ravel())
        col.extend(cc.ravel())
        ss = np.zeros((12, 3))
        ss[6:] = sh
        delta.extend((ss[None, :, :] - ss[:, None, :]).reshape(-1, 3))
        np.add.at(diag, ids, K[p].diagonal())
    row = np.asarray(row)
    col = np.asarray(col)
    delta = np.asarray(delta)
    scale = 1 / np.sqrt(diag)
    scaling = scale[row] * scale[col]
    info = dict(method='Exact spectral Schur complement of every interior FE DOF; first zero below the first clamped-interior pole', clamped_interior_pole=float(pole), calls=0, evaluations=0, seconds=0.0, pole_returns=0)

    def eigen(k, count=1, vectors=False):
        assert count == 1 and (not vectors)
        start = time.perf_counter()
        k = np.asarray(k)
        phase = np.exp(2j * np.pi * (delta @ k)) * scaling
        gamma = la.norm(k - np.rint(k)) < 1e-10

        def value(t):
            b = B0 - t * B1
            s = K - t * H - np.matmul(b.transpose(0, 2, 1), b / (1 - t * mu)[:, :, None])
            a = np.zeros((nd, nd), complex)
            np.add.at(a, (row, col), s.ravel() * phase)
            if gamma:
                a = a[3:, 3:]
            a = (a + a.conj().T) / 2
            info['evaluations'] += 1
            return la.eigvalsh(a, subset_by_index=(0, 0), check_finite=False, driver='evr')[0]
        if value(upper) >= 0:
            root = pole
            info['pole_returns'] += 1
        else:
            root = pole * brentq(lambda x: value(pole * x), 0, upper / pole, xtol=2e-13, rtol=5e-13)
        info['calls'] += 1
        info['seconds'] += time.perf_counter() - start
        return np.array([root])
    f.eigen = eigen
    return info
if __name__ == '__main__':
    from computations.atlas.mesh_solver import setup, install_sparse
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cco')
    ap.add_argument('--div', type=int, default=8)
    a = ap.parse_args()
    f = newframe(a.name, a.div)
    setup(f)
    f.prestress('stress')
    dense = f.eigen
    sparse_stats = install_sparse(f)
    sparse = f.eigen
    stat = install(f)
    rows = []
    for k in [[0, 0, 0], [0.125, 0.25, 0.375], [0.007, 0.002, 0.001], [0.25, 0.25, 0], [0.5, 0, 0]]:
        t = time.perf_counter()
        d = float(sparse(k, 1)[0])
        dt = time.perf_counter() - t
        t = time.perf_counter()
        s = float(f.eigen(k, 1)[0])
        st = time.perf_counter() - t
        row = dict(k=k, sparse=d, condensed=s, relative_error=s / d - 1, sparse_seconds=dt, condensed_seconds=st)
        rows.append(row)
        print(row, flush=True)
    save(DEST / f'atlas/mesh_condense_{a.name}_d{a.div}.json', dict(name=a.name, div=a.div, rows=rows, stats=stat, sparse_stats=sparse_stats, threadpools=POOLS))
