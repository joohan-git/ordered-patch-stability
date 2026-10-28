"""Algebra-only acceleration of the unchanged full FE matrices."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.mesh_common import *
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, splu, LinearOperator
import time

def affine_fast(f):
    nf = len(f.free)
    out = []
    for matrices, internal in [([e['K'] for e in f.elements], f.K0), ([-g for g in f.Gm], -f.assemble(f.Gm, [0, 0, 0]).real)]:
        coupling = np.zeros((f.ndof, 9))
        macro = np.zeros((9, 9))
        for e, m in zip(f.elements, matrices):
            b = np.zeros((12, 9))
            for i in range(3):
                b[i, 3 * i:3 * i + 3] = e['a']
                b[6 + i, 3 * i:3 * i + 3] = e['b']
            c = m @ b
            np.add.at(coupling, e['di'], c[:6])
            np.add.at(coupling, e['dj'], c[6:])
            macro += b.T @ c
        a = np.zeros((nf + 9, nf + 9))
        a[:nf, :nf] = internal[np.ix_(f.free, f.free)]
        a[:nf, nf:] = coupling[f.free]
        a[nf:, :nf] = coupling[f.free].T
        a[nf:, nf:] = macro
        out.append(a)
    f.affineK, f.affineH = out

def install_sparse(f):
    row = []
    col = []
    shift = []
    kval = []
    hval = []
    for e, g in zip(f.elements, f.Gm):
        ids = np.r_[e['di'], e['dj']]
        ss = np.zeros((12, 3))
        ss[6:] = e['shift']
        delta = ss[None, :, :] - ss[:, None, :]
        rr, cc = np.meshgrid(ids, ids, indexing='ij')
        row.extend(rr.ravel())
        col.extend(cc.ravel())
        shift.extend(delta.reshape(-1, 3))
        kval.extend(e['K'].ravel())
        hval.extend((-g).ravel())
    row = np.asarray(row)
    col = np.asarray(col)
    shift = np.asarray(shift)
    kval = np.asarray(kval)
    hval = np.asarray(hval)
    diagonal = 1 / np.sqrt(f.K0.diagonal())
    scales = diagonal[row] * diagonal[col]
    kval *= scales
    hval *= scales
    stats = dict(calls=0, max_scaled_residual=0.0, seconds=0.0)

    def eigen(k, count=1, vectors=False):
        assert count == 1 and (not vectors)
        start = time.perf_counter()
        k = np.asarray(k)
        gamma = la.norm(k - np.rint(k)) < 1e-10
        phase = np.exp(2j * np.pi * (shift @ k))
        K = coo_matrix((kval * phase, (row, col)), shape=(f.ndof, f.ndof)).tocsc()
        H = coo_matrix((hval * phase, (row, col)), shape=K.shape).tocsc()
        if gamma:
            K = K[3:, 3:]
            H = H[3:, 3:]
        factor = splu(K, permc_spec='COLAMD')
        inv = LinearOperator(K.shape, matvec=factor.solve, dtype=complex)
        v0 = np.sin(np.arange(K.shape[0]) * 0.731 + 1) + 1j * np.cos(np.arange(K.shape[0]) * 0.219 + 0.2)
        vals, vec = eigsh(H, k=1, M=K, Minv=inv, which='LA', tol=2e-11, maxiter=1500, ncv=min(32, K.shape[0] - 1), v0=v0)
        mu = float(vals[0])
        v = vec[:, 0]
        hv = H @ v
        kv = K @ v
        residual = float(la.norm(hv - mu * kv) / (la.norm(hv) + abs(mu) * la.norm(kv)))
        assert residual < 1e-06, (f.name, k, residual)
        stats['calls'] += 1
        stats['max_scaled_residual'] = max(stats['max_scaled_residual'], residual)
        stats['seconds'] += time.perf_counter() - start
        return np.array([1 / mu])
    f.eigen = eigen
    return stats

def setup(f):
    f.affine_matrices = lambda: affine_fast(f)
    relaxed = f.relaxed
    cache = {}

    def solve(strain):
        key = np.asarray(strain).tobytes()
        if key not in cache:
            cache[key] = relaxed(strain)
        return dict(cache[key])
    f.relaxed = solve
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cco')
    ap.add_argument('--div', type=int, default=8)
    a = ap.parse_args()
    start = time.perf_counter()
    f = newframe(a.name, a.div)
    setup(f)
    f.prestress('stress')
    print('frame', f.ndof, time.perf_counter() - start, flush=True)
    dense = f.eigen
    stat = install_sparse(f)
    rows = []
    for k in [[0, 0, 0], [0.125, 0.25, 0.375], [0.007, 0.002, 0.001], [0.25, 0.25, 0], [0.5, 0, 0]]:
        t = time.perf_counter()
        d = float(dense(k, 1)[0])
        dt = time.perf_counter() - t
        t = time.perf_counter()
        s = float(f.eigen(k, 1)[0])
        st = time.perf_counter() - t
        row = dict(k=k, dense=d, sparse=s, relative_error=s / d - 1, dense_seconds=dt, sparse_seconds=st)
        rows.append(row)
        print(row, flush=True)
    save(DEST / f'atlas/mesh_solver_{a.name}_d{a.div}.json', dict(name=a.name, div=a.div, rows=rows, stats=stat, threadpools=POOLS))
