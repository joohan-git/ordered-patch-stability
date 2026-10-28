"""Independent exact-strain energy FD checks of the baseline atlas tangent.

The production model is a fixed-geometry initial-stress pencil, not nonlinear
continuation. A local nonlinear energy with exactly that initial tangent is
constructed from exact_basic (finite-rotation Kirchhoff strains), not from G.
For each element E(q)=1/2 b(q).Cb.b(q)+t*s0.b(q). The reference has initial
stress t*s0; the elastic moduli and geometry are fixed as in the atlas.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys
sys.dont_write_bytecode = True
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import json, hashlib
OUT = _DATA_ROOT
DATA_ROOT = _DATA_ROOT
sys.path.insert(0, str(DATA_ROOT))
from computations.common.runtime import cache_dir
OUT = cache_dir('prestress_convergence')
import numpy as np
import scipy.linalg as la
import gemmi
from computations.validation.beam_validation import ConsistentFrame, ConsistentAtlas, exact_basic, axes
from computations.common.acoustic_atlas import volume_hessian

def data(name):
    cell = json.loads((DATA_ROOT / f'inclusions/cell_{name}.json').read_text())
    return dict(A=np.array(cell['cell']), frac=np.array(cell['fractional_positions']), edges=[(i, j, np.array(sh)) for i, j, sh in cell['edges']], sg=gemmi.find_spacegroup_by_name('P 1'), name=name, net={'name': name})

def main():
    results = {'model': 'Kirchhoff-Hermite exact-strain initial tangent v1', 'energy': 'sum_e [0.5 b_e(q)^T Cb_e b_e(q) + t s0_e^T b_e(q)] + sign*p*V*(det(F)-1)', 'scope': 'local potential verifying the exact initial tangent; no nonlinear preload continuation', 'step_sizes': [0.001, 0.0003, 0.0001, 3e-05], 'runs': []}
    for name in ('pcu', 'cco'):
        f = ConsistentFrame(name, 0.02, 2, data=data(name))
        at = ConsistentAtlas(f)
        pd = at.pd(True)
        t = 0.4 * min(pd['bound'], at.gamma)
        p = t * at.p0
        V = at.vol
        nf = at.nf
        MG = at.K - t * at.Hpressure
        MF = at.K - t * at.H

        def cond(M):
            return (M[-9:, -9:] - M[-9:, :nf] @ la.solve(M[:nf, :nf], M[:nf, -9:], assume_a='sym')) / V
        AG, AF = (cond(MG), cond(MF))
        elements = []
        elastic_errors = []
        for e, force, A in zip(f.elements, f.pre['full_element_forces'], f.areas):
            L, R = axes(e['a'], e['b'])
            T = np.kron(np.eye(4), R)
            local_force = T @ force
            s0 = np.r_[local_force[6], local_force[9], local_force[4:6], local_force[10:12]]
            I = A * A / (4 * np.pi)
            Cb = np.diag([A / L, I / (1.3 * L), 4 * I / L, 4 * I / L, 4 * I / L, 4 * I / L])
            Cb[2, 4] = Cb[4, 2] = Cb[3, 5] = Cb[5, 3] = 2 * I / L
            B = np.zeros((12, nf + 9))
            for off, dofs in [(0, e['di']), (6, e['dj'])]:
                for j, dof in enumerate(dofs):
                    if dof >= 3:
                        B[off + j, dof - 3] += 1
            for i in range(3):
                B[i, nf + 3 * i:nf + 3 * i + 3] = e['a']
                B[i + 6, nf + 3 * i:nf + 3 * i + 3] = e['b']
            elements.append((T @ B, L, s0, Cb))
        directions = {'dilation': np.eye(3) / np.sqrt(3), 'deviator': np.diag([1.0, -1.0, 0.0]) / np.sqrt(2), 'symmetric_shear': np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / np.sqrt(2), 'critical_PD_strain': np.array(pd['strain_mode']), 'rank_one_shear': np.array([[0, 1.0, 0], [0, 0, 0], [0, 0, 0]])}
        rows = []
        for label, H in directions.items():
            hvec = H.ravel()
            w = -la.solve(MG[:nf, :nf], MG[:nf, nf:] @ hvec, assume_a='sym')
            z = np.r_[w, hvec]
            qdir = [B @ z for B, L, s0, Cb in elements]

            def energy(step, sign):
                en = 0.0
                for q, (_, L, s0, Cb) in zip(qdir, elements):
                    b = exact_basic(step * q, L)
                    en += 0.5 * b @ Cb @ b + t * s0 @ b
                return en + sign * p * V * (la.det(np.eye(3) + step * H) - 1)
            expected = float(hvec @ AG @ hvec)
            expectedF = float(hvec @ AF @ hvec)
            sweep = []
            for eps in results['step_sizes']:
                plus = (energy(eps, 1) + energy(-eps, 1)) / (eps * eps * V)
                minus = (energy(eps, -1) + energy(-eps, -1)) / (eps * eps * V)
                helm = (energy(eps, 0) + energy(-eps, 0)) / (eps * eps * V)
                sweep.append({'step': eps, 'FD_E_plus_pV': plus, 'FD_E_minus_pV': minus, 'FD_E': helm, 'relative_error_plus': abs(plus - expected) / abs(expected), 'relative_error_minus': abs(minus - expected) / abs(expected), 'relative_error_Helmholtz': abs(helm - expectedF) / max(abs(expectedF), 1e-30), 'first_derivative_plus': (energy(eps, 1) - energy(-eps, 1)) / (2 * eps * V)})
            rows.append({'direction': label, 'H': H.tolist(), 'assembled_Gibbs': expected, 'assembled_Helmholtz': expectedF, 'D_det': float(np.trace(H) ** 2 - np.trace(H @ H)), 'sweep': sweep})
        acoustic = []
        for a, n in [(np.array([1.0, 0, 0]), np.array([0.0, 1, 0])), (np.array([1.0, 0, 0]), np.array([1.0, 0, 0])), (np.array([0.2, 0.7, 0.5]), np.array([0.3, 0.1, 0.8]))]:
            a = a / la.norm(a)
            n = n / la.norm(n)
            H = np.outer(a, n)
            e = (H + H.T) / 2
            ac = float(H.ravel() @ AF @ H.ravel())
            symF = float(e.ravel() @ AF @ e.ravel())
            cross2 = float(la.norm(np.cross(a, n)) ** 2)
            pressure_adjusted = symF - p * cross2 / 2
            green = symF + p * np.sum(e * e)
            green_correct = green - p * np.sum(H * H)
            acoustic.append({'a': a.tolist(), 'n': n.tolist(), 'acoustic': ac, 'sym_gradient_Helmholtz': symF, 'p_cross_squared_over_2': p * cross2 / 2, 'formula_relative_error': abs(ac - pressure_adjusted) / max(abs(ac), 1e-30), 'Green_material_Hessian': green, 'Green_formula_relative_error': abs(ac - green_correct) / max(abs(ac), 1e-30), 'Gibbs_rank_one_relative_error': abs(ac - float(e.ravel() @ AG @ e.ravel())) / max(abs(ac), 1e-30)})
        helm_pd = at.pd(False)
        le = at.le()
        row = {'name': name, 'nodes': len(f.data['frac']), 'elements': len(f.elements), 'r_over_l': 0.02, 'div': 2, 't': t, 'p': p, 'p0': at.p0, 'preload_stress': f.pre['stress'].tolist(), 'PD_pressure': pd['bound'] * at.p0, 'PD_Helmholtz_sym_gradient_pressure': helm_pd['bound'] * at.p0, 'PD_Helmholtz_censored': helm_pd['censored_by_gamma'], 'LE_bound_pressure': le['bound'] * at.p0, 'LE_censored': le['censored_by_gamma'], 'raw_schur_vs_code_relative': float(la.norm(AG - at.tangent(t, True)) / la.norm(AG)), 'directions': rows, 'acoustic_relations': acoustic}
        results['runs'].append(row)
        print(name, 'p=', p, 'raw Schur error', row['raw_schur_vs_code_relative'], flush=True)
        for r in rows:
            print(r['direction'], [(x['step'], x['relative_error_plus'], x['relative_error_minus']) for x in r['sweep']], flush=True)
    results['source_sha256'] = {n: hashlib.sha256((DATA_ROOT / n).read_bytes()).hexdigest() for n in ['common/initial_stress.py', 'validation/beam_validation.py', 'inclusions/cell_pcu.json', 'inclusions/cell_cco.json']}
    (OUT / 'sign_check.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
if __name__ == '__main__':
    main()
