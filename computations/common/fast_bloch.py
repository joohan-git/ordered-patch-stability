"""Identical dense Bloch matrices with cached phase blocks."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.check_common import *

def install(f):
    nd = f.ndof
    shifts = sorted({tuple(e['shift']) for e in f.elements})
    blocks = []
    for matrices in [[e['K'] for e in f.elements], [-g for g in f.Gm]]:
        diag = np.zeros((nd, nd))
        cross = np.zeros((len(shifts), nd, nd))
        for e, m in zip(f.elements, matrices):
            i, j = (e['di'], e['dj'])
            s = shifts.index(tuple(e['shift']))
            diag[np.ix_(i, i)] += m[:6, :6]
            diag[np.ix_(j, j)] += m[6:, 6:]
            cross[s][np.ix_(i, j)] += m[:6, 6:]
        blocks.append((diag, cross))
    shifts = np.asarray(shifts)

    def eigen(k, count=1, vectors=False):
        assert not vectors
        k = np.asarray(k)
        phase = np.exp(2j * np.pi * (shifts @ k))
        mats = []
        free = f.free if la.norm(k - np.rint(k)) < 1e-10 else np.arange(nd)
        for diag, cross in blocks:
            m = np.tensordot(phase, cross, axes=1)
            m = m + m.conj().T + diag
            mats.append(m[np.ix_(free, free)])
        K, H = mats
        scale = 1 / np.sqrt(K.diagonal().real)
        ev = la.eigh(H * scale[:, None] * scale[None, :], K * scale[:, None] * scale[None, :], subset_by_index=(len(free) - count, len(free) - 1), driver='gvx', check_finite=False, eigvals_only=True)[::-1]
        return 1 / ev
    errors = []
    for k in [[0, 0, 0], [0.125, 0.25, 0.375], [0.007, 0.002, 0.001]]:
        a = f.eigen(k, 1)[0]
        b = eigen(k, 1)[0]
        errors.append(abs(b / a - 1))
    assert max(errors) < 1e-07, errors
    f.eigen = eigen
    return errors
