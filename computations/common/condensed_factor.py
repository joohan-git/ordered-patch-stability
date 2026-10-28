"""Exact block elimination for K inverse; H and all eigenvector DOFs retained."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

class CondensedFactor:

    def __init__(self, K, njoint, block):
        self.njoint = njoint
        self.block = block
        if block == 0 or njoint == K.shape[0]:
            self.direct = sla.splu(K, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
            self.factor = self.direct
            self.L = self.factor.L
            self.U = self.factor.U
            self.internal_nnz = 0
            return
        self.direct = None
        self.B = K[:njoint, njoint:].tocsr()
        D = K[njoint:, njoint:].tocoo()
        nblock = (K.shape[0] - njoint) // block
        assert np.all(D.row // block == D.col // block), 'internal-node K must be parent-block diagonal'
        blocks = np.zeros((nblock, block, block), dtype=K.dtype)
        blocks[D.row // block, D.row % block, D.col % block] = D.data
        self.Dinv = np.linalg.inv(blocks)
        self.internal_nnz = self.Dinv.size
        rows = (np.arange(nblock)[:, None, None] * block + np.arange(block)[None, :, None] + np.zeros((1, 1, block), int)).ravel()
        cols = (np.arange(nblock)[:, None, None] * block + np.arange(block)[None, None, :] + np.zeros((1, block, 1), int)).ravel()
        Di = sp.coo_matrix((self.Dinv.ravel(), (rows, cols)), shape=D.shape).tocsr()
        Di.eliminate_zeros()
        S = (K[:njoint, :njoint] - self.B @ Di @ self.B.T.conj()).tocsc()
        S.eliminate_zeros()
        self.factor = sla.splu(S, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
        self.L = self.factor.L
        self.U = self.factor.U

    def apply_Dinv(self, x):
        if x.ndim == 1:
            return np.einsum('bij,bj->bi', self.Dinv, x.reshape(-1, self.block)).ravel()
        return np.einsum('bij,bjk->bik', self.Dinv, x.reshape(-1, self.block, x.shape[1])).reshape(-1, x.shape[1])

    def solve(self, rhs):
        if self.direct is not None:
            return self.direct.solve(rhs)
        a = rhs[:self.njoint]
        b = rhs[self.njoint:]
        x = self.factor.solve(a - self.B @ self.apply_Dinv(b))
        y = self.apply_Dinv(b - self.B.T.conj() @ x)
        return np.concatenate([x, y], axis=0)
