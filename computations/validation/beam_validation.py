"""Verify beam tangents, rigid-rotation identities and frame stability in JSON."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import json, sys
import numpy as np
import scipy.linalg as la
ROOT = _DATA_ROOT
from computations.common.initial_stress import geometric_full, basic_jet, apply_frame_prestress, axes, MODEL
from computations.common.bloch_frame import geometric_matrix, beam_matrix
from computations.atlas.edge_center_atlas import SlenderFrame, TangentAtlas, graph_data, volume_hessian
from scipy.spatial.transform import Rotation

def exact_basic(q, L):
    """Independent exact Kirchhoff strains, used only for verification."""
    d = np.array([L, 0.0, 0.0]) + q[6:9] - q[:3]
    length = la.norm(d)
    Ra = Rotation.from_rotvec(q[3:6]).as_matrix()
    Rb = Rotation.from_rotvec(q[9:12]).as_matrix()

    def align(s):
        v = np.cross([1.0, 0.0, 0.0], s)
        W = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + W + W @ W / (1 + s[0])
    end = align(Ra.T @ Rb[:, 0]).T @ Ra.T @ Rb
    twist = np.arctan2(end[2, 1], end[1, 1])
    points, weights = np.polynomial.legendre.leggauss(6)
    result = np.zeros(6)
    for t, weight in zip((points + 1) / 2, weights / 2):
        slope = 6 * t * (1 - t) * d / L + (3 * t * t - 4 * t + 1) * length / L * Ra[:, 0] + (3 * t * t - 2 * t) * length / L * Rb[:, 0]
        second = ((6 - 12 * t) * d / L + (6 * t - 4) * length / L * Ra[:, 0] + (6 * t - 2) * length / L * Rb[:, 0]) / L
        stretch = la.norm(slope)
        tangent = slope / stretch
        tangent_prime = (second - tangent * np.dot(tangent, second)) / stretch
        s = Ra.T @ tangent
        sp = Ra.T @ tangent_prime
        R = Ra @ align(s) @ Rotation.from_rotvec([t * twist, 0.0, 0.0]).as_matrix()
        kappa = R.T @ np.cross(tangent, tangent_prime)
        kappa[0] = twist / L + (s[2] * sp[1] - s[1] * sp[2]) / (1 + s[0])
        result += weight * L * np.r_[stretch - 1, kappa[0], -(1 - t) * kappa[1:], t * kappa[1:]]
    return result

def column_and_portal():
    results = []
    radius = 0.03
    EI = np.pi * radius ** 4 / 4
    for div in [1, 2, 4, 8, 16]:
        K = np.zeros((6 * (div + 1),) * 2)
        G = K.copy()
        for i in range(div):
            a = np.array([i / div, 0.0, 0.0])
            b = np.array([(i + 1) / div, 0.0, 0.0])
            ix = np.arange(6 * i, 6 * i + 12)
            K[np.ix_(ix, ix)] += beam_matrix(a, b, radius)
            f = np.zeros(12)
            f[0] = -1.0
            f[6] = 1.0
            G[np.ix_(ix, ix)] += geometric_full(a, b, f)
        free = [6 * i + c for i in range(div + 1) for c in [1, 5] if c == 5 or 0 < i < div]
        val = la.eigvalsh(K[np.ix_(free, free)], G[np.ix_(free, free)])[0]
        free2 = [6 * i + c for i in range(1, div + 1) for c in [1, 5] if i < div or c == 1]
        sway = la.eigvalsh(K[np.ix_(free2, free2)], G[np.ix_(free2, free2)])[0]
        results.append(dict(div=div, euler=val, portal_per_column=sway, exact=np.pi ** 2 * EI, euler_relative=val / (np.pi ** 2 * EI) - 1, portal_relative=sway / (np.pi ** 2 * EI) - 1))
    return results

def balanced_selfstress():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.2], [0.2, 1.1, 0.1], [0.1, 0.2, 1.2]])
    edges = [(i, j) for i in range(4) for j in range(i)]
    C = np.zeros((24, 6 * len(edges)))
    mappings = []
    for h, (i, j) in enumerate(edges):
        L, R = axes(pts[i], pts[j])
        T = np.kron(np.eye(4), R)
        B = ((basic_jet(np.eye(12), L) - basic_jet(-np.eye(12), L)) / 2).T @ T
        ix = np.r_[6 * i + np.arange(6), 6 * j + np.arange(6)]
        C[np.ix_(ix, np.arange(6 * h, 6 * h + 6))] += B.T
        mappings.append((ix, B))
    null = la.null_space(C)
    rng = np.random.default_rng(1013)
    rows = []
    Rigid = np.zeros((24, 3))
    for h, omega in enumerate(np.eye(3)):
        Rigid.reshape(4, 6, 3)[:, :3, h] = np.cross(omega, pts)
        Rigid.reshape(4, 6, 3)[:, 3:, h] = omega
    for k in range(10):
        s = null @ rng.normal(size=null.shape[1])
        G = np.zeros((24, 24))
        for h, ((i, j), (ix, B)) in enumerate(zip(edges, mappings)):
            force = B.T @ s[6 * h:6 * h + 6]
            G[np.ix_(ix, ix)] += geometric_full(pts[i], pts[j], force)
        rows.append(dict(balance=float(la.norm(C @ s) / (la.norm(C) * la.norm(s))), ward=float(la.norm(G @ Rigid) / (la.norm(G) * la.norm(Rigid))), rotation_energy=float(la.norm(Rigid.T @ G @ Rigid) / (la.norm(G) * la.norm(Rigid) ** 2))))
    return rows

def loaded_portal():
    """Two unit columns and a unit girder, all r=.03, bases clamped.

    Vertical forces -EI at both top joints, horizontal +.2 EI at right top.
    A separate centered finite-difference Hessian of exact_basic validates
    the moment/shear terms, beyond the axial analytical sway benchmark.
    """
    rows = []
    radius = 0.03
    EI = np.pi * radius ** 4 / 4
    for div in [1, 2, 4, 8, 16]:
        pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        edges = []
        for i, j in [(0, 2), (1, 3), (2, 3)]:
            ids = [i]
            for h in range(1, div):
                ids.append(len(pts))
                pts.append((np.array(pts[i]) + (np.array(pts[j]) - pts[i]) * h / div).tolist())
            ids.append(j)
            edges += list(zip(ids[:-1], ids[1:]))
        pts = np.array(pts)
        K = np.zeros((6 * len(pts),) * 2)
        elements = []
        for i, j in edges:
            ix = np.r_[6 * i + np.arange(6), 6 * j + np.arange(6)]
            ke = beam_matrix(pts[i], pts[j], radius)
            K[np.ix_(ix, ix)] += ke
            elements.append((i, j, ix, ke))
        free = np.array([6 * i + c for i in range(2, len(pts)) for c in [0, 1, 5]])
        load = np.zeros(len(K))
        load[6 * 2 + 1] = load[6 * 3 + 1] = -EI
        load[6 * 3] = 0.2 * EI
        u = np.zeros(len(K))
        u[free] = la.solve(K[np.ix_(free, free)], load[free], assume_a='pos')
        G = K * 0
        Gfd = K * 0
        for i, j, ix, ke in elements:
            force = ke @ u[ix]
            ge = geometric_full(pts[i], pts[j], force)
            G[np.ix_(ix, ix)] += ge
            if div == 1:
                L, R = axes(pts[i], pts[j])
                T = np.kron(np.eye(4), R)
                f = T @ force
                s = np.r_[f[6], f[9], f[4:6], f[10:12]]
                eps = 0.0002
                eye = np.eye(12) * eps
                local = np.zeros((12, 12))
                for a in range(12):
                    local[a, a] = s @ (exact_basic(eye[a], L) + exact_basic(-eye[a], L)) / eps ** 2
                    for b in range(a):
                        local[a, b] = local[b, a] = s @ (exact_basic(eye[a] + eye[b], L) - exact_basic(eye[a] - eye[b], L) - exact_basic(-eye[a] + eye[b], L) + exact_basic(-eye[a] - eye[b], L)) / (4 * eps ** 2)
                Gfd[np.ix_(ix, ix)] += T.T @ local @ T

        def root(g):
            return float(1 / la.eigvalsh(-g[np.ix_(free, free)], K[np.ix_(free, free)])[-1])
        row = dict(div=div, critical_load_factor=root(G), reference_force=EI)
        if div == 1:
            row.update(finite_difference_factor=root(Gfd), hessian_relative_error=float(la.norm(G - Gfd) / la.norm(G)))
        rows.append(row)
    return rows

class ConsistentFrame(SlenderFrame):
    prestress = apply_frame_prestress

class ConsistentAtlas(TangentAtlas):

    def __init__(self, f):
        super().__init__(f, objective=False)
        self.objective = True

def validate():
    out = {'model': MODEL, 'elements': [], 'nets': [], 'benchmarks': column_and_portal(), 'selfstress': balanced_selfstress(), 'loaded_portal': loaded_portal()}
    rng = np.random.default_rng(1031)
    for h in range(8):
        a = rng.normal(size=3)
        b = a + rng.normal(size=3)
        n = (b - a) / la.norm(b - a)
        f = np.r_[-n, np.zeros(3), n, np.zeros(3)]
        full = geometric_full(a, b, f)
        axial = geometric_matrix(a, b)
        force = beam_matrix(a, b, 0.03) @ rng.normal(size=12)
        g = geometric_full(a, b, force)
        swap = np.r_[np.arange(6, 12), np.arange(6)]
        rev = geometric_full(b, a, force[swap])[np.ix_(swap, swap)]
        L, R = axes(a, b)
        T = np.kron(np.eye(4), R)
        B = ((basic_jet(np.eye(12), L) - basic_jet(-np.eye(12), L)) / 2).T @ T
        EI = np.pi * 0.03 ** 4 / 4
        Cb = np.diag([np.pi * 0.03 ** 2 / L, EI / (1.3 * L), 4 * EI / L, 4 * EI / L, 4 * EI / L, 4 * EI / L])
        Cb[2, 4] = Cb[4, 2] = Cb[3, 5] = Cb[5, 3] = 2 * EI / L
        elastic = beam_matrix(a, b, 0.03)
        q = rng.normal(size=12) * 0.1
        O = Rotation.from_rotvec(rng.normal(size=3) * 0.7).as_matrix()
        shift = rng.normal(size=3)
        transformed = q.copy()
        for offset, position in [(0, np.zeros(3)), (6, np.array([L, 0.0, 0.0]))]:
            transformed[offset:offset + 3] = O @ (position + q[offset:offset + 3]) + shift - position
            transformed[offset + 3:offset + 6] = Rotation.from_matrix(O @ Rotation.from_rotvec(q[offset + 3:offset + 6]).as_matrix()).as_rotvec()
        out['elements'].append({'axial_relative_error': float(la.norm(full - axial) / la.norm(axial)), 'symmetry_error': float(la.norm(g - g.T) / la.norm(g)), 'end_reversal_error': float(la.norm(g - rev) / la.norm(g)), 'linear_stiffness_error': float(la.norm(B.T @ Cb @ B - elastic) / la.norm(elastic)), 'finite_objectivity_error': float(la.norm(exact_basic(q, L) - exact_basic(transformed, L)))})
    out['finite_rotation_jet'] = []
    direction = rng.normal(size=12)
    for eps in [0.01, 0.001, 0.0001]:
        q = eps * direction
        out['finite_rotation_jet'].append(dict(eps=eps, error=float(la.norm(exact_basic(q, 1.3) - basic_jet(q, 1.3)[0]))))
    for name in ['pcu', 'utp', 'cco', 'cyg', 'tfk', 'srs']:
        f = ConsistentFrame(name, 0.02, 2, data=graph_data(name))
        at = ConsistentAtlas(f)
        probe = 0.4 * min(at.pd(True)['bound'], at.gamma)
        M = at.K - probe * at.Hpressure
        resid = la.norm(M @ at.R) / (la.norm(M) * la.norm(at.R))
        Gresid = la.norm(at.Hpressure @ at.R) / (la.norm(at.Hpressure) * la.norm(at.R))
        nf = at.nf
        raw = (M[nf:, nf:] - M[nf:, :nf] @ la.solve(M[:nf, :nf], M[:nf, nf:], assume_a='sym')) / at.vol
        comp = at.tangent(probe, True)
        row = dict(name=name, load=probe, ward=float(resid), initial_stress_ward=float(Gresid), schur_relative_error=float(la.norm(raw - comp) / la.norm(raw)), rotation_energy=float(la.norm(at.R.T @ M @ at.R) / (la.norm(M) * la.norm(at.R) ** 2)))
        n = np.array([0.31, 0.57, 0.76])
        n /= la.norm(n)
        P = np.kron(np.eye(3), n[:, None])
        Q = P.T @ at.tangent(probe) @ P
        leading = np.zeros((nf + 3, nf + 3))
        for e, g in zip(f.elements, f.Gm):
            B = np.zeros((12, nf + 3))
            for offset, dofs in [(0, e['di']), (6, e['dj'])]:
                for h, dof in enumerate(dofs):
                    if dof >= 3:
                        B[offset + h, dof - 3] += 1
            B[6:9, -3:] = np.eye(3) * np.dot(n, e['b'] - e['a'])
            leading += B.T @ (e['K'] + probe * g) @ B
        q0 = (leading[-3:, -3:] - leading[-3:, :-3] @ la.solve(leading[:-3, :-3], leading[:-3, -3:], assume_a='sym')) / at.vol
        row['analytic_bloch_coefficient_error'] = float(la.norm(q0 - Q) / la.norm(Q))
        trans = np.zeros((f.ndof, 3))
        trans.reshape(-1, 6, 3)[:, :3, :] = np.eye(3)
        internal = la.null_space(trans.T)
        row['bloch'] = []
        for eps in [0.02, 0.01, 0.005, 0.0025]:
            k = eps * f.A @ n / (2 * np.pi)
            phase = np.repeat(np.exp(1j * eps * (f.frac @ f.A @ n)), 6)
            B = phase[:, None] * np.column_stack((internal, trans))
            H = f.assemble([e['K'] for e in f.elements], k) + probe * f.assemble(f.Gm, k)
            D = B.conj().T @ H @ B
            q = (D[-3:, -3:] - D[-3:, :-3] @ la.solve(D[:-3, :-3], D[:-3, -3:], assume_a='her')) / (eps ** 2 * at.vol)
            row['bloch'].append(dict(wave_number=eps, relative_error=float(la.norm(q - Q) / la.norm(Q))))
        out['nets'].append(row)
        print(name, row, flush=True)
    return out
if __name__ == '__main__':
    out = validate()
    (ROOT / 'validation/beam_validation.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
