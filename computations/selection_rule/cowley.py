"""Independent constructive tests of Cowley Table I, engineering Voigt input.

Independent of other verification code. Kelvin eigenvectors, explicit
rank-one factorization, and Cartesian acoustic contractions are cross-checked.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh, norm
from scipy.optimize import brentq, minimize_scalar
ROOT = _DATA_ROOT
PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
S = np.zeros((9, 6))
for k, (i, j) in enumerate(PAIRS):
    S[3 * i + j, k] = 1 if i == j else 1 / np.sqrt(2)
    S[3 * j + i, k] = S[3 * i + j, k]
D = np.diag([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])

def rz(t):
    c, s = (np.cos(t), np.sin(t))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def invariance(C, rots):
    errs = []
    for R in rots:
        T = np.column_stack([S.T @ (R @ S[:, i].reshape(3, 3) @ R.T).ravel() for i in range(6)])
        errs.append(norm(T.T @ C @ T - C) / norm(C))
    return float(max(errs))

def compatible_null(C):
    ev, U = eigh(C)
    Z = U[:, abs(ev) < 1e-09]
    if Z.shape[1] == 1:
        v = Z[:, 0]
        angle = None
    else:

        def vtheta(t):
            return Z[:, 0] * np.cos(t) + Z[:, 1] * np.sin(t)

        def det(t):
            return np.linalg.det((S @ vtheta(t)).reshape(3, 3))
        ts = np.linspace(0, 2 * np.pi, 721)
        roots = []
        for a, b in zip(ts[:-1], ts[1:]):
            if det(a) * det(b) < 0:
                roots.append(brentq(det, a, b, xtol=1e-14))
        angle = roots[0]
        v = vtheta(angle)
    E = (S @ v).reshape(3, 3)
    eig, P = eigh(E)
    positive = eig[-1]
    negative = -eig[0]
    n = np.sqrt(positive) * P[:, -1] + np.sqrt(negative) * P[:, 0]
    a = np.sqrt(positive) * P[:, -1] - np.sqrt(negative) * P[:, 0]
    a *= norm(n)
    n /= norm(n)
    return (ev, Z, E, a, n, angle)

def acoustic(C, n):
    A = (S @ C @ S.T).reshape(3, 3, 3, 3)
    return np.einsum('ijkl,j,l->ik', A, n, n)

def case(label, Cv, rots, condition, coupling):
    C = D @ Cv @ D
    ev, Z, E, a, n, angle = compatible_null(C)
    Q = acoustic(C, n)
    B = np.zeros((6, 3))
    for j in range(3):
        F = np.outer(np.eye(3)[j], n)
        e = (F + F.T) / 2
        B[:, j] = [e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]]
    sides = []
    for delta in (-0.0001, 0, 0.0001):
        V = Cv.copy()
        for i, j in coupling:
            V[i, j] *= 1 + delta
            V[j, i] = V[i, j]
        K = D @ V @ D
        sides.append(dict(coupling_fraction_delta=delta, elastic_min=float(eigh(K, eigvals_only=True)[0]), acoustic_min_at_boundary_n=float(eigh(acoustic(K, n), eigvals_only=True)[0])))
    out = dict(case=label, engineering_voigt_order=['xx', 'yy', 'zz', 'yz', 'xz', 'xy'], C_voigt=Cv.tolist(), C_kelvin=C.tolist(), condition=condition, elastic_eigenvalues=ev.tolist(), nullity=Z.shape[1], null_basis_kelvin=Z.tolist(), symmetry_error=invariance(C, rots), E=E.tolist(), a=a.tolist(), n=n.tolist(), strain_trace=float(np.trace(E)), strain_determinant=float(np.linalg.det(E)), rank_one_reconstruction_error=float(norm(E - (np.outer(a, n) + np.outer(n, a)) / 2)), acoustic_matrix=Q.tolist(), acoustic_eigenvalues=eigh(Q, eigvals_only=True).tolist(), acoustic_null_vector_residual=float(norm(Q @ a) / (norm(C) * norm(a))), contraction_crosscheck=float(norm(Q - B.T @ Cv @ B) / norm(C)), det_root_angle=angle, boundary_crossing=sides, cowley_table_type='0')
    assert ev[0] > -1e-10 and out['symmetry_error'] < 1e-12
    assert out['rank_one_reconstruction_error'] < 1e-12
    assert out['acoustic_null_vector_residual'] < 1e-12
    assert sides[0]['elastic_min'] > 0 and sides[-1]['acoustic_min_at_boundary_n'] < 0
    return out

def main():
    common_examples = []
    V = np.array([[10, 4, 2, 0, 0, 3], [4, 10, 2, 0, 0, -3], [2, 2, 12, 0, 0, 0], [0, 0, 0, 4, 0, 0], [0, 0, 0, 0, 4, 0], [3, -3, 0, 0, 0, 3]], float)
    common_examples.append(case('tetragonal 4/-4/4/m: 2B', V, [rz(np.pi / 2), -rz(np.pi / 2), np.diag([1, 1, -1])], dict(expression='(c11-c12)c66-2c16^2', value=0), [(0, 5), (1, 5)]))
    for label, c11, c12, c44, c14, c15, rots in [('trigonal 32: 2E', 10, 4, 3, 3, 0, [rz(2 * np.pi / 3), np.diag([1, -1, -1])]), ('trigonal 3: 2E', 12, 2, 5, 3, 4, [rz(2 * np.pi / 3)])]:
        V = np.array([[c11, c12, 2, c14, c15, 0], [c12, c11, 2, -c14, -c15, 0], [2, 2, 12, 0, 0, 0], [c14, -c14, 0, c44, 0, -c15], [c15, -c15, 0, 0, c44, c14], [0, 0, 0, -c15, c14, (c11 - c12) / 2]], float)
        common_examples.append(case(label, V, rots, dict(expression='(c11-c12)c44-2c14^2-2c15^2', value=(c11 - c12) * c44 - 2 * c14 ** 2 - 2 * c15 ** 2), [(0, 3), (0, 4), (1, 3), (1, 4), (3, 5), (4, 5)]))
    V = np.array([[10, 1, 2, 0, 1, 0], [1, 11, 1, 0, 2, 0], [2, 1, 12, 0, 1, 0], [0, 0, 0, 4, 0, 6], [1, 2, 1, 0, 8, 0], [0, 0, 0, 6, 0, 9]], float)
    common_examples.append(case('monoclinic 2/m, 2, m: 2B (unique y)', V, [np.diag([-1, 1, -1]), np.diag([1, -1, 1])], dict(expression='c44*c66-c46^2', value=0), [(3, 5)]))
    rng = np.random.default_rng(19764877)
    sweep = []
    for group in ('tetragonal', 'trigonal32', 'trigonal3', 'monoclinic'):
        for index in range(24):
            d = float(rng.uniform(2, 8))
            b = float(rng.uniform(2, 8))
            total = float(rng.uniform(10, 20))
            c11 = (total + d) / 2
            c12 = (total - d) / 2
            if group == 'tetragonal':
                h = np.sqrt(d * b / 2)
                V = np.array([[c11, c12, 1, 0, 0, h], [c12, c11, 1, 0, 0, -h], [1, 1, 12, 0, 0, 0], [0, 0, 0, 4, 0, 0], [0, 0, 0, 0, 4, 0], [h, -h, 0, 0, 0, b]])
                rots = [rz(np.pi / 2)]
                coupling = [(0, 5), (1, 5)]
            elif group.startswith('trigonal'):
                theta = 0 if group == 'trigonal32' else float(rng.uniform(0.1, 6.1))
                h = np.sqrt(d * b / 2) * np.cos(theta)
                j = np.sqrt(d * b / 2) * np.sin(theta)
                V = np.array([[c11, c12, 1, h, j, 0], [c12, c11, 1, -h, -j, 0], [1, 1, 12, 0, 0, 0], [h, -h, 0, b, 0, -j], [j, -j, 0, 0, b, h], [0, 0, 0, -j, h, d / 2]])
                rots = [rz(2 * np.pi / 3)]
                coupling = [(0, 3), (0, 4), (1, 3), (1, 4), (3, 5), (4, 5)]
            else:
                h = np.sqrt(d * b)
                V = np.array([[10, 1, 2, 0, 1, 0], [1, 11, 1, 0, 2, 0], [2, 1, 12, 0, 1, 0], [0, 0, 0, d, 0, h], [1, 2, 1, 0, 8, 0], [0, 0, 0, h, 0, b]], float)
                rots = [np.diag([-1, 1, -1])]
                coupling = [(3, 5)]
            z = case(group, V, rots, dict(expression='random boundary', value=0), coupling)
            sweep.append(dict(group=group, index=index, C_voigt=z['C_voigt'], n=z['n'], a=z['a'], nullity=z['nullity'], acoustic_min=z['acoustic_eigenvalues'][0], acoustic_null_vector_residual=z['acoustic_null_vector_residual'], symmetry_error=z['symmetry_error']))
    controls = []
    for label, E in [('cubic invariant A1', np.eye(3) / np.sqrt(3)), ('uniaxial invariant mixed A', np.diag([1, 1, -2]) / np.sqrt(6))]:
        v = S.T @ E.ravel()
        C = np.eye(6) - np.outer(v, v)

        def q(t):
            return float(eigh(acoustic(C, np.array([np.sin(t), 0, np.cos(t)])), eigvals_only=True)[0])
        opt = minimize_scalar(q, bounds=(0, np.pi / 2), method='bounded', options={'xatol': 1e-14})
        controls.append(dict(case=label, C_kelvin=C.tolist(), E=E.tolist(), elastic_min=float(eigh(C, eigvals_only=True)[0]), acoustic_min=min(q(0), q(np.pi / 2), opt.fun), axisymmetry_reduces_global_search_to_theta=True))
    quotes = [dict(printed_page=4879, pdf_page=3, section='II.C', text='The only examples which have not been discussed in Sec. II B are those which occur when the elastic constant matrix has two or more irreducible representations of the same symmetry.'), dict(printed_page=4879, pdf_page=3, section='II.C', text='For example, the 2A1 representations in the hexagonal, or trigonal classes. In these cases, there is in general no acoustic wave whose velocity is zero when the elastic constant stability condition given in Table I is equal to zero.'), dict(printed_page=4879, pdf_page=3, section='II.C', text='In general, this is not the case and there is no requirement that an acoustic mode must have zero velocity when a stability condition is violated.')]
    out = dict(source='Cowley, Phys. Rev. B 13, 4877 (1976)', table=dict(printed_page=4880, pdf_page=4, verification='Table I, acoustic-waves column: type 0 for the listed repeated representations', rows=[dict(case=r['case'], acoustic_waves='0') for r in common_examples]), quotes=quotes, quote_normalization='Whitespace and line-wrap hyphens normalized; A1 subscript rendered inline.', examples=common_examples, controls=controls, random_sweep=sweep, conclusion='The listed repeated nontrivial representations have compatible zero-acoustic directions. Table I lists type 0; the examples quantify the acoustic eigenvalues and compatibility residuals. The crystal system 4/-4/4/m is tetragonal.')
    (ROOT / 'selection_rule/cowley.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    for r in common_examples:
        print(r['case'], r['acoustic_eigenvalues'][0], r['symmetry_error'], r['n'])
    print('controls', [(r['case'], r['acoustic_min']) for r in controls])
if __name__ == '__main__':
    main()
