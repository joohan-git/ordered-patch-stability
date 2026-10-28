"""2-D frame search, compact-patch probes and an exact S2 counterexample.

The negative-bulk inclusion control is a continuum quadratic-energy model,
not falsely presented as a decorated-Truchet beam construction.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import argparse, itertools, time, json
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize
from computations.truchet_2d.frame import Frame2D, ROOT, dump
from computations.common.acoustic_atlas import sym_basis, volume_hessian, restricted, first_root

def hydro(f):
    ep = la.solve(f.C, np.array([-1.0, -1.0, 0.0]))
    ep /= -(ep[0] + ep[1]) / 2
    pre = f.relaxed(np.einsum('a,aei->ei', ep, f.affine))
    f.pre = pre
    f.He = -pre['forces'][:, None, None] * f.Ge
    f.H = f.assemble(f.He)
    f.Hf = f.H[f.free, :][:, f.free].tocsc()
    return -np.trace(pre['stress']) / 2

def affine_matrices(f):
    affine = []
    for E in np.eye(4).reshape(4, 2, 2):
        d = np.zeros((len(f.Ke), 6))
        d[:, 3:5] = f.delta @ E.T
        affine.append(d)
    affine = np.array(affine)
    out = []
    for em, M in [(f.Ke, f.Kf), (f.He, f.Hf)]:
        cross = np.array([f.scatter(np.einsum('eij,ej->ei', em, d)) for d in affine]).T[f.free]
        macro = np.einsum('aei,eij,bej->ab', affine, em, affine)
        out.append(np.block([[M.toarray(), cross], [cross.T, macro]]))
    return out

def analyze(p, stub=0.2, factors=(1, 1, 1), radius=0.005):
    f = Frame2D(p, radius=radius, stub=stub, radius_factors=factors)
    p0 = hydro(f)
    K, H = affine_matrices(f)
    HP = H.copy()
    HP[-4:, -4:] -= p0 * np.prod(p.shape) * volume_hessian(2)
    S = sym_basis(2)
    kp, hp = restricted(K, HP, S)
    pd, v = first_root(kp, hp, True)
    E = (S @ v[-3:]).reshape(2, 2)
    volfrac = float(np.trace(E) ** 2 / (2 * np.sum(E * E)))
    gd = f.eigen(count=1)[0][0]
    angles = np.arange(32) * np.pi / 32
    le = np.inf
    direction = None
    for th in angles:
        n = np.array([np.cos(th), np.sin(th)])
        k, h = restricted(K, H, np.kron(np.eye(2), n[:, None]))
        val = first_root(k, h)
        if val < le:
            le = val
            direction = th
    opt = minimize(lambda x: first_root(*restricted(K, H, np.kron(np.eye(2), np.array([np.cos(x[0]), np.sin(x[0])])[:, None]))), [direction], method='Nelder-Mead', options={'maxiter': 60, 'xatol': 1e-05, 'fatol': 1e-10})
    le = min(le, float(opt.fun))
    best = (np.inf, None)
    for k in itertools.product([0, 0.25, 0.5, 0.75], repeat=2):
        val = f.eigen(k, count=1)[0][0]
        if val < best[0]:
            best = (float(val), list(k))
    return dict(stub=stub, factors=list(factors), radius=radius, pattern=p.tolist(), PD=float(pd), LE=float(le), micro=best[0], k=best[1], gamma=float(gd), volumetric_fraction=volfrac, PD_only=bool(pd < min(le, best[0]) * 0.99), pressure_per_factor=float(p0), pressure_PD=float(pd * p0), pressure_localizable=float(min(le, best[0]) * p0))

def search():
    patterns = [np.zeros((2, 2), int), np.array([[0, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
    out = dict(runs=[], families=[])
    for stub, skeleton, stubs in itertools.product([0.08, 0.15, 0.2, 0.3, 0.4], [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0], [0.5, 1.0, 2.0]):
        rows = []
        for p in patterns:
            r = analyze(p, stub, (skeleton, stubs, 1))
            rows.append(r)
            out['runs'].append(r)
        lowest = min(rows, key=lambda r: min(r['PD'], r['LE'], r['micro']) * r['pressure_per_factor'])
        family = dict(stub=stub, factors=[skeleton, stubs, 1], weakest=lowest, volumetric_PD_first=bool(lowest['PD_only'] and lowest['volumetric_fraction'] > 0.8))
        out['families'].append(family)
        dump(ROOT / 'truchet_2d/inputs/search_truchet_search.json', out)
        print('family', len(out['families']), stub, skeleton, stubs, lowest['PD_only'], lowest['volumetric_fraction'], flush=True)

def radial_control():
    """Exact radial trial and independent 1-D radial finite-element solve.
    W=K/2 (div u)^2 + mu |dev sym grad u|^2 in two spatial dimensions.
    K_in=1-lambda, mu_in=1. Matrix K_out=1, mu_out adjustable >0.
    Both phases are strongly elliptic for lambda<2. PD fails at lambda=1.
    """
    out = dict(rows=[], analytic=dict(lambda_PD=1.0, lambda_LE=2.0, infinite_radial_threshold='1 + mu_matrix', finite_radial_threshold='1 + (mu_matrix + K_matrix*(R/B)^2)/(1-(R/B)^2)'))
    for mu in [0.1, 1.0, 3.0]:
        for R in [2.0, 4.0, 8.0, 16.0, 32.0]:
            B = 4 * R
            nr = 80
            no = 240
            nodes = np.r_[np.linspace(0, R, nr + 1), np.linspace(R, B, no + 1)[1:]]
            ne = len(nodes) - 1
            K = np.zeros((ne + 1, ne + 1))
            H = K.copy()
            q, w = np.polynomial.legendre.leggauss(5)
            for e, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
                rr = (a + b) / 2 + (b - a) / 2 * q
                phi = np.column_stack([(b - rr) / (b - a), (rr - a) / (b - a)])
                deriv = np.tile(np.array([-1, 1]) / (b - a), (len(q), 1))
                hoop = phi / rr[:, None]
                div = deriv + hoop
                dev = deriv - hoop
                inside = b <= R + 1e-09
                shear = 1.0 if inside else mu
                km = np.einsum('q,qi,qj->ij', w * rr * np.pi * (b - a) / 2, div, div)
                km += shear * np.einsum('q,qi,qj->ij', w * rr * np.pi * (b - a) / 2, dev, dev)
                hm = np.einsum('q,qi,qj->ij', w * rr * np.pi * (b - a) / 2, div, div) if inside else np.zeros((2, 2))
                ix = np.ix_([e, e + 1], [e, e + 1])
                K[ix] += km
                H[ix] += hm
            val, v = first_root(K[1:-1, 1:-1], H[1:-1, 1:-1], True)
            dilation = 1 + (mu + (R / B) ** 2) / (1 - (R / B) ** 2)
            exact = min(2.0, dilation)
            out['rows'].append(dict(mu_matrix=mu, R=R, outer_radius=B, critical_radial_FE=val, exact_first=exact, uniform_dilation_trial_threshold=dilation, relative_error=val / exact - 1, PD_only_negative_mode=bool(val < 2 - 1e-06)))
            dump(ROOT / 'truchet_2d/inputs/search_radial_control.json', out)
            print('radial', mu, R, val, exact, flush=True)
    out['counterexample'] = dict(lambda_test=1.5, K_inside=-0.5, mu_inside=1.0, K_matrix=1.0, mu_matrix=0.1, outer_to_patch_radius=4.0, normalized_trial_energy=-0.5 + (0.1 + 1 / 16) / (1 - 1 / 16), acoustic_eigenvalues_inside=[1.0, 0.5], acoustic_eigenvalues_matrix=[0.1, 1.1])
    dump(ROOT / 'truchet_2d/inputs/search_radial_control.json', out)

def patch_b(stub=0.2, factors=(1, 1, 1), radius=0.01):
    """The periodic checkerboard optical branch, in common hydrostatic pressure units."""
    base = Frame2D(np.array([[0, 1], [1, 0]]), radius=radius, stub=stub, radius_factors=factors)
    pb = hydro(base)
    crit = base.eigen(count=2)[0][0] * pb
    out = dict(periodic_pressure=crit, rows=[])
    for L, R in [(32, 0), (32, 8), (32, 16), (32, 24), (64, 16), (64, 32), (64, 48)]:
        p = np.random.default_rng(441 + L).integers(0, 2, (L, L))
        lo = (L - R) // 2
        hi = lo + R
        if R:
            p[lo:hi, lo:hi] = np.indices((R, R)).sum(axis=0) % 2
        f = Frame2D(p, radius=radius, stub=stub, radius_factors=factors)
        press = hydro(f)
        vals, v, res = f.eigen(count=3)
        stats = f.mode_stats(v[:, 0])
        energy = np.array(stats['tile_energy']).reshape(L, L)
        row = dict(L=L, R=R, pressure_critical=float(vals[0] * press), ratio=float(vals[0] * press / crit), energy_fraction_inside=float(energy[lo:hi, lo:hi].sum()) if R else None, IPR=stats['energy_ipr'], eigen_residual=res)
        out['rows'].append(row)
        dump(ROOT / 'truchet_2d/inputs/search_patch_b.json', out)
        print('patch-b', row, flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['search', 'radial', 'patch'])
    a = ap.parse_args()
    if a.action == 'search':
        search()
    elif a.action == 'radial':
        radial_control()
    else:
        patch_b()
