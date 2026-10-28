"""Objective node-spring/angular-spring Eq.1 model. Energy unit GPa A^3.

Phonon Hessians use the objective potential. The Source Data Born convention
is evaluated separately as a homogeneous pressure-convention tensor.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from computations.trumof.mechanics.model.rule import *
from scipy import sparse, linalg
from scipy.sparse.linalg import spsolve, eigsh
from scipy.optimize import minimize, brentq
S = np.zeros((6, 3, 3))
S[0, 0, 0] = S[1, 1, 1] = S[2, 2, 2] = 1
S[3, 1, 2] = S[3, 2, 1] = S[4, 0, 2] = S[4, 2, 0] = S[5, 0, 1] = S[5, 1, 0] = 0.5
TRACE = np.array([1, 1, 1, 0, 0, 0])
VOLH = np.outer(TRACE, TRACE) - np.einsum('aij,bji->ab', S, S)
SOURCE_DELTA = np.outer(TRACE, TRACE) - 3 * np.einsum('aij,bji->ab', S, S)

def angle_terms(w, theta0, kt):
    u = w[:, :3]
    v = w[:, 3:]
    a = np.linalg.norm(u, axis=1)
    b = np.linalg.norm(v, axis=1)
    c = np.clip(np.sum(u * v, axis=1) / (a * b), -1, 1)
    theta = np.arccos(c)
    dt = theta - theta0
    f = 0.5 * kt * dt ** 2
    fp = np.zeros(len(c))
    fpp = fp.copy()
    linear = (theta0 > np.pi - 1e-08) & (c < -0.9999)
    z = c[linear] + 1
    f[linear] = kt * (z + z * z / 6 + 2 * z ** 3 / 45)
    fp[linear] = kt * (1 + z / 3 + 2 * z * z / 15)
    fpp[linear] = kt * (1 / 3 + 4 * z / 15)
    non = ~linear
    sn = np.sqrt(np.maximum(1 - c[non] ** 2, 1e-24))
    fp[non] = -kt * dt[non] / sn
    fpp[non] = kt / sn ** 2 - kt * dt[non] * c[non] / sn ** 3
    gu = v / (a * b)[:, None] - c[:, None] * u / a[:, None] ** 2
    gv = u / (a * b)[:, None] - c[:, None] * v / b[:, None] ** 2
    gc = np.concatenate([gu, gv], axis=1)
    uu = np.einsum('ni,nj->nij', u, u)
    vv = np.einsum('ni,nj->nij', v, v)
    uv = np.einsum('ni,nj->nij', u, v)
    I = np.eye(3)[None]
    huu = -(uv + uv.transpose(0, 2, 1)) / (a ** 3 * b)[:, None, None] - c[:, None, None] * I / a[:, None, None] ** 2 + 3 * c[:, None, None] * uu / a[:, None, None] ** 4
    hvv = -(uv + uv.transpose(0, 2, 1)) / (a * b ** 3)[:, None, None] - c[:, None, None] * I / b[:, None, None] ** 2 + 3 * c[:, None, None] * vv / b[:, None, None] ** 4
    huv = I / (a * b)[:, None, None] - vv / (a * b ** 3)[:, None, None] - uu / (a ** 3 * b)[:, None, None] + c[:, None, None] * uv / (a * a * b * b)[:, None, None]
    hc = np.concatenate([np.concatenate([huu, huv], axis=2), np.concatenate([huv.transpose(0, 2, 1), hvv], axis=2)], axis=1)
    return (f, fp[:, None] * gc, fpp[:, None, None] * np.einsum('ni,nj->nij', gc, gc) + fp[:, None, None] * hc)

class CG:

    def __init__(self, pos, A, edges, kr, kt):
        self.pos = np.array(pos, float)
        self.A = np.array(A, float)
        self.e = np.array(edges, int)
        self.N = len(pos)
        self.kr = kr
        self.kt = kt
        self.V0 = abs(np.linalg.det(A))
        i, j = self.e[:, :2].T
        self.d = self.pos[j] + self.e[:, 2:] @ self.A - self.pos[i]
        self.re = np.linalg.norm(self.d, axis=1)
        self.B = sparse.coo_matrix((np.tile([-1.0, 1.0], len(i) * 3), (np.repeat(np.arange(3 * len(i)), 2), np.stack([3 * i[:, None] + np.arange(3), 3 * j[:, None] + np.arange(3)], axis=2).ravel())), shape=(3 * len(i), 3 * self.N)).tocsr()
        neigh = [[] for _ in range(self.N)]
        for a, (i, j, *n) in enumerate(self.e):
            neigh[i].append((j, np.array(n), self.d[a]))
            neigh[j].append((i, -np.array(n), -self.d[a]))
        self.ai = []
        self.aj = []
        self.ak = []
        self.ajshift = []
        self.akshift = []
        d1 = []
        d2 = []
        for i, ns in enumerate(neigh):
            for (j, sj, u), (k, sk, v) in itertools.combinations(ns, 2):
                self.ai.append(i)
                self.aj.append(j)
                self.ak.append(k)
                self.ajshift.append(sj)
                self.akshift.append(sk)
                d1.append(u)
                d2.append(v)
        for n in ['ai', 'aj', 'ak', 'ajshift', 'akshift']:
            setattr(self, n, np.array(getattr(self, n), int))
        self.ad = np.concatenate([d1, d2], axis=1)
        self.th = np.arccos(np.clip(np.sum(self.ad[:, :3] * self.ad[:, 3:], axis=1) / (np.linalg.norm(self.ad[:, :3], axis=1) * np.linalg.norm(self.ad[:, 3:], axis=1)), -1, 1))
        self.th = np.where(self.th > np.pi - 1e-06, np.pi, self.th)
        ids = np.concatenate([np.stack([3 * self.ai[:, None] + np.arange(3), 3 * self.aj[:, None] + np.arange(3)], axis=2), np.stack([3 * self.ai[:, None] + np.arange(3), 3 * self.ak[:, None] + np.arange(3)], axis=2)], axis=1).ravel()
        self.AB = sparse.coo_matrix((np.tile([-1.0, 1.0], len(self.ai) * 6), (np.repeat(np.arange(6 * len(self.ai)), 2), ids)), shape=(6 * len(self.ai), 3 * self.N)).tocsr()
        adj = sparse.coo_matrix((np.ones(2 * len(self.e)), (np.r_[self.e[:, 0], self.e[:, 1]], np.r_[self.e[:, 1], self.e[:, 0]])), shape=(self.N, self.N))
        ncc, cc = sparse.csgraph.connected_components(adj)
        self.cc = cc
        self.ncc = ncc
        fixed = np.array([np.flatnonzero(cc == c)[0] for c in range(ncc)])
        self.free = np.setdiff1d(np.arange(3 * self.N), np.concatenate([3 * fixed[:, None] + np.arange(3)]).ravel())
        self.M0 = np.einsum('aij,nj->nia', S, self.d).reshape(-1, 6)
        self.AM0 = np.concatenate([np.einsum('aij,nj->nia', S, self.ad[:, :3]), np.einsum('aij,nj->nia', S, self.ad[:, 3:])], axis=1).reshape(-1, 6)

    def configuration(self, x):
        F = np.eye(3) + np.einsum('a,aij->ij', x[-6:], S)
        u = x[:-6].reshape(-1, 3)
        w = self.d @ F + (self.B @ u.ravel()).reshape(-1, 3)
        aw = np.concatenate([self.ad[:, :3] @ F, self.ad[:, 3:] @ F], axis=1) + (self.AB @ u.ravel()).reshape(-1, 6)
        return (F, u, w, aw)

    def local(self, x):
        F, u, w, aw = self.configuration(x)
        length = np.linalg.norm(w, axis=1)
        n = w / length[:, None]
        dr = length - self.re
        E = 0.5 * self.kr * np.dot(dr, dr)
        g = self.kr * dr[:, None] * n
        h = self.kr * (np.einsum('ni,nj->nij', n, n) + (dr / length)[:, None, None] * (np.eye(3) - np.einsum('ni,nj->nij', n, n)))
        ae, ag, ah = angle_terms(aw, self.th, self.kt)
        return (F, u, w, aw, E + sum(ae), g, h, ag, ah)

    def objective(self, x, p, need_h=False):
        F, u, w, aw, E, g, h, ag, ah = self.local(x)
        det = np.linalg.det(F)
        if det <= 0:
            return (1e+100, np.zeros_like(x))
        inv = np.linalg.inv(F)
        V = self.V0 * det
        gf = np.einsum('ij,aji->a', inv, S)
        grad = np.r_[self.B.T @ g.ravel() + self.AB.T @ ag.ravel(), self.M0.T @ g.ravel() + self.AM0.T @ ag.ravel() + p * V * gf]
        if not need_h:
            return (E + p * V, grad)
        HB = self.blockdiag(h)
        HA = self.blockdiag(ah)
        J = sparse.hstack([self.B, sparse.csr_matrix(self.M0)], format='csr')
        AJ = sparse.hstack([self.AB, sparse.csr_matrix(self.AM0)], format='csr')
        H = (J.T @ HB @ J + AJ.T @ HA @ AJ).tocsc()
        vH = p * V * (np.outer(gf, gf) - np.einsum('ij,ajk,kl,bli->ab', inv, S, inv, S))
        correction = sparse.coo_matrix((vH.ravel(), (np.repeat(np.arange(3 * self.N, 3 * self.N + 6), 6), np.tile(np.arange(3 * self.N, 3 * self.N + 6), 6))), shape=H.shape)
        return (E + p * V, grad, H + correction)

    @staticmethod
    def blockdiag(h):
        n, k, _ = h.shape
        rows = (np.arange(n)[:, None, None] * k + np.arange(k)[None, :, None] + np.zeros((1, 1, k), int)).ravel()
        cols = (np.arange(n)[:, None, None] * k + np.arange(k)[None, None, :] + np.zeros((1, k, 1), int)).ravel()
        return sparse.coo_matrix((h.ravel(), (rows, cols)), shape=(n * k, n * k)).tocsr()

    def relax(self, p, x=None, maxiter=35):
        if x is None:
            x = np.zeros(3 * self.N + 6)
        else:
            x = x.copy()
        free = np.r_[self.free, np.arange(3 * self.N, 3 * self.N + 6)]
        success = False
        for it in range(maxiter):
            E, g, H = self.objective(x, p, True)
            err = max(np.max(abs(g[:3 * self.N])) / max(self.kr, 1), np.max(abs(g[-6:])) / self.V0)
            if err < 2e-09:
                success = True
                break
            dx = np.zeros_like(x)
            dx[free] = spsolve(H[free][:, free], -g[free])
            factor = min(1, 0.06 / max(np.max(abs(dx[-6:])), 0.06), 0.5 / max(np.max(abs(dx[:-6])), 0.5))
            trial = x + factor * dx
            if np.linalg.det(np.eye(3) + np.einsum('a,aij->ij', trial[-6:], S)) <= 0.02:
                break
            x = trial
        return (x, dict(converged=success, iterations=it + 1, residual=err))

    def tangent(self, x, p, source=False):
        F, u, w, aw, E, g, h, ag, ah = self.local(x)
        V = self.V0 * np.linalg.det(F)
        HB = self.blockdiag(h)
        HA = self.blockdiag(ah)
        M = np.einsum('aij,nj->nia', S, w).reshape(-1, 6)
        AM = np.concatenate([np.einsum('aij,nj->nia', S, aw[:, :3]), np.einsum('aij,nj->nia', S, aw[:, 3:])], axis=1).reshape(-1, 6)
        K = (self.B.T @ HB @ self.B + self.AB.T @ HA @ self.AB).tocsc()
        C = self.B.T @ HB @ M + self.AB.T @ HA @ AM
        D = M.T @ HB @ M + AM.T @ HA @ AM + p * V * VOLH
        Kf = K[self.free][:, self.free]
        Cf = C[self.free]
        sol = spsolve(Kf, Cf) if len(self.free) else np.zeros((0, 6))
        relaxed = (D - Cf.T @ sol) / V
        if source:
            relaxed += p * SOURCE_DELTA
        return ((relaxed + relaxed.T) / 2, K, C, D)

    def bloch(self, x, k):
        F, u, w, aw, E, g, h, ag, ah = self.local(x)
        k = np.array(k)

        def incidence(center, end, sh):
            phase = np.exp(2j * np.pi * (sh @ k))
            N = len(center)
            ids = np.stack([3 * center[:, None] + np.arange(3), 3 * end[:, None] + np.arange(3)], axis=2).ravel()
            vals = np.stack([-np.ones((N, 3)), np.repeat(phase[:, None], 3, axis=1)], axis=2).ravel()
            return sparse.coo_matrix((vals, (np.repeat(np.arange(3 * N), 2), ids)), shape=(3 * N, 3 * self.N)).tocsr()
        B = incidence(self.e[:, 0], self.e[:, 1], self.e[:, 2:])
        first_angle_incidence = incidence(self.ai, self.aj, self.ajshift)
        second_angle_incidence = incidence(self.ai, self.ak, self.akshift)
        AB = sparse.vstack([first_angle_incidence, second_angle_incidence], format='csr')[np.arange(6 * len(self.ai)).reshape(2, -1, 3).transpose(1, 0, 2).ravel()]
        return (B.conj().T @ self.blockdiag(h) @ B + AB.conj().T @ self.blockdiag(ah) @ AB).tocsr()

def params(kind='soft', re_mof=13.0):
    z = json.load(open(_DATA_ROOT / 'trumof/pcu_parameters.json'))[kind]
    return dict(kr=z['C11_GPa'] * re_mof, kt=z['C44_GPa'] * re_mof ** 3 / 4, re_mof_A=re_mof, kr_eV_A2=z['C11_GPa'] * re_mof / 160.2176634, kt_eV_rad2=z['C44_GPa'] * re_mof ** 3 / 4 / 160.2176634)

def model_approximant(z, kind='soft'):
    par = params(kind)
    return CG(np.array(z['positions']) * json.load(open(_DATA_ROOT / 'trumof/rule.json'))['a'], z['cell'], z['edges'], par['kr'], par['kt'])

def acoustic(C, n):
    T = np.array([[n[0], 0, 0], [0, n[1], 0], [0, 0, n[2]], [0, n[2], n[1]], [n[2], 0, n[0]], [n[1], n[0], 0]])
    return T.T @ C @ T

def le_min(C, refine=True):
    dirs = np.array([v for v in itertools.product([-1, 0, 1], repeat=3) if any(v)], float)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    j = np.arange(96)
    zz = 1 - 2 * (j + 0.5) / 96
    phi = j * np.pi * (3 - np.sqrt(5))
    rr = np.sqrt(1 - zz * zz)
    dirs = np.r_[dirs, np.c_[rr * np.cos(phi), rr * np.sin(phi), zz]]
    vals = np.array([linalg.eigvalsh(acoustic(C, n))[0] for n in dirs])
    idx = []
    for ii in np.argsort(vals):
        if all((abs(dirs[ii] @ dirs[jj]) < 0.97 for jj in idx)):
            idx.append(ii)
        if len(idx) >= 4:
            break
    best = (vals[idx[0]], dirs[idx[0]])
    if refine:
        for i in idx:
            opt = minimize(lambda n: linalg.eigvalsh(acoustic(C, n / np.linalg.norm(n)))[0], dirs[i], method='BFGS', options={'gtol': 1e-08})
            if opt.fun < best[0]:
                best = (opt.fun, opt.x / np.linalg.norm(opt.x))
    return (float(best[0]), best[1])
if __name__ == '__main__':
    out = {}
    for kind in ['soft', 'hard']:
        par = params(kind)
        source0 = par['re_mof_A']
        m = CG([[0, 0, 0]], np.eye(3) * source0, [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], par['kr'], par['kt'])
        vals = []
        for p in [0, 0.1, 0.3, 1.2]:
            x, info = m.relax(p)
            C, _, _, _ = m.tangent(x, p)
            B, _, _, _ = m.tangent(x, p, source=True)
            vals.append(dict(p=p, relax=info, physical_C44=C[3, 3], source_B44=B[3, 3], C=C, B=B))
        out[kind] = dict(parameters=par, points=vals)
        print(kind, [(v['p'], v['physical_C44'], v['source_B44']) for v in vals])
    save('trumof/initialization/cg_pcu.json', out)
