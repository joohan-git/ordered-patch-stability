"""Conservative geometrically exact Kirchhoff--Hermite beam initial tangent.

The Hermite centreline uses nodal positions and endpoint derivatives
(current chord length / reference length) exp(theta_i) e_x. Its unit tangent
is the first director: there is no shear deformation. The transverse frame
is transported by the shortest rotation in end 1's frame; the residual end
twist is linear in material arclength. Strains are |r'|-1 and vee(R.T R').
Integrating N*epsilon + M(s).kappa gives the full initial-stress Hessian.
N,T are constant and M_perp(s)=-(1-s/L)m1+(s/L)m2. Shears follow from moment
equilibrium. Exponential rotations retain the symmetric semi-tangential
moment terms. Three Gauss points integrate the second-order jet exactly.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from functools import lru_cache
import numpy as np
MODEL = 'Kirchhoff-Hermite exact-strain initial tangent v1'

def axes(a, b):
    d = np.asarray(b) - np.asarray(a)
    L = np.linalg.norm(d)
    x = d / L
    ref = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(ref, x)
    y /= np.linalg.norm(y)
    return (L, np.array([x, y, np.cross(x, y)]))

def basic_jet(q, L):
    """Integrated strains conjugate to N,T,m1y,m1z,m2y,m2z, degree <=2.

    theta1=(linear twist, e_x cross w), w=linear transverse centreline slope.
    theta2_perp=e_x cross source2'_perp-v_x theta1_perp+theta1_x w/2.
    kappa2=theta2' - theta1 cross theta1'/2. Integration by parts removes
    theta2' because theta2 vanishes at the nodal rotations. This retains
    both the shear (m1+m2)/L and nonuniform bending moment contributions.
    """
    q = np.atleast_2d(q)
    v = (q[:, 6:9] - q[:, :3]) / L
    a, b = (q[:, 3:6], q[:, 9:12])
    e = np.array([1.0, 0.0, 0.0])
    vp = v.copy()
    vp[:, 0] = 0.0
    c = np.cross(e, vp)
    out = np.column_stack((L * v[:, 0], b[:, 0] - a[:, 0], (a - c)[:, 1:], (b - c)[:, 1:]))
    wa, wb = (np.cross(a, e), np.cross(b, e))
    points, weights = np.polynomial.legendre.leggauss(3)
    for t, weight in zip((points + 1) / 2, weights / 2):
        h = 6 * t * (1 - t)
        g1 = 3 * t * t - 4 * t + 1
        g2 = 3 * t * t - 2 * t
        hp = 6 - 12 * t
        g1p = 6 * t - 4
        g2p = 6 * t - 2
        w = h * vp + g1 * wa + g2 * wb
        wp = hp * vp + g1p * wa + g2p * wb
        theta = np.cross(e, w)
        theta[:, 0] = (1 - t) * a[:, 0] + t * b[:, 0]
        prime = np.cross(e, wp)
        prime[:, 0] = b[:, 0] - a[:, 0]
        source2 = g1 * (v[:, :1] * wa + 0.5 * a[:, :1] * (a - a[:, :1] * e))
        source2 += g2 * (v[:, :1] * wb + 0.5 * b[:, :1] * (b - b[:, :1] * e))
        theta2 = np.cross(e, source2) - v[:, :1] * np.cross(e, w) + 0.5 * theta[:, :1] * w
        cross = np.cross(theta, prime)
        out[:, 0] += weight * L / 2 * np.sum(w * w, axis=1)
        out[:, 1] -= weight / 2 * cross[:, 0]
        out[:, 2:4] += weight * (-theta2 + 0.5 * (1 - t) * cross)[:, 1:]
        out[:, 4:6] += weight * (-theta2 - 0.5 * t * cross)[:, 1:]
    return out

@lru_cache(maxsize=4096)
def local_hessians(L):
    eye = np.eye(12)
    plus, minus = (basic_jet(eye, L), basic_jet(-eye, L))
    diag = plus + minus
    h = np.zeros((6, 12, 12))
    h[:, np.arange(12), np.arange(12)] = diag.T
    pairs = [(i, j) for i in range(12) for j in range(i)]
    q = np.array([eye[i] + eye[j] for i, j in pairs])
    cross = (basic_jet(q, L) + basic_jet(-q, L) - np.array([diag[i] + diag[j] for i, j in pairs])) / 2
    for row, (i, j) in zip(cross, pairs):
        h[:, i, j] = h[:, j, i] = row
    return h

def geometric_full(a, b, nodal_force):
    """Hessian of integrated initial stress times exact Kirchhoff strain.

    nodal_force is the complete 12-vector K_e u_e in global axes. Shears
    enter through equilibrated end moments; no axial-only projection occurs.
    Positive N means tension. Major symmetry follows from a scalar potential.
    """
    L, R = axes(a, b)
    T = np.kron(np.eye(4), R)
    f = T @ np.asarray(nodal_force)
    stress = np.r_[f[6], f[9], f[4:6], f[10:12]]
    g = np.einsum('i,ijk->jk', stress, local_hessians(float(L)))
    return T.T @ g @ T

def apply_frame_prestress(frame, protocol='strain'):
    """Reuse its linear equilibrium, then replace every element tangent."""
    from computations.common.bloch_frame import Frame
    pre = Frame.prestress(frame, protocol)
    gm = []
    full = []
    for e in frame.elements:
        d = np.r_[pre['strain'] @ e['a'], np.zeros(3), pre['strain'] @ e['b'], np.zeros(3)]
        d += np.r_[pre['w'][e['di']], pre['w'][e['dj']]]
        force = e['K'] @ d
        full.append(force)
        gm.append(geometric_full(e['a'], e['b'], force))
    frame.Gm = gm
    pre['full_element_forces'] = np.asarray(full)
    return pre
