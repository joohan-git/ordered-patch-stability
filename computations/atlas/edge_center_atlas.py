"""Edge-centre frame model with PD, acoustic and Bloch stability operators.

SlenderFrame uses axial prestress in the reference geometry, with each strut's
radius/reference length equal to slenderness. TangentAtlas supports constant
nominal dead traction and hydrostatic +p V loading. With objective=True it
completes the homogeneous Gibbs block to enforce rigid-rotation invariance;
objective=False uses the supplied full-gradient matrices. The microscopic
Bloch operator always comes from the supplied frame, so an axial frame with
homogeneous completion can have a different acoustic limit.

The paper's RCSR atlas, mesh studies and selection-rule results use this
module through validation/beam_validation.ConsistentFrame and ConsistentAtlas.
Those classes supply the complete beam end-force/end-moment tangent and
evaluate the hydrostatic Gibbs acoustic operator. The tensor inclusion bound
uses the full-gradient variant with the measured nonhydrostatic stress.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, brentq
from computations.common.bloch_frame import Frame, NETS, LD, beam_matrix, representatives
from computations.truchet_2d.frame import dump, ROOT
import gemmi

def sym_basis(d):
    out = []
    for i in range(d):
        a = np.zeros((d, d))
        a[i, i] = 1
        out.append(a.ravel())
    for i in range(d):
        for j in range(i + 1, d):
            a = np.zeros((d, d))
            a[i, j] = a[j, i] = 1 / np.sqrt(2)
            out.append(a.ravel())
    return np.array(out).T

def volume_hessian(d):
    eye = np.eye(d).ravel()
    out = np.outer(eye, eye)
    for i in range(d):
        for j in range(d):
            out[d * i + j, d * j + i] -= 1
    return out

class SlenderFrame(Frame):

    def __init__(self, name, slender=0.02, div=2, data=None):
        super().__init__(name, slender, div, data=data)
        self.areas = []
        for e in self.elements:
            length = la.norm(e['b'] - e['a']) * div
            r = slender * length
            e['K'] = beam_matrix(e['a'], e['b'], r)
            self.areas.append(np.pi * r * r)
        self.areas = np.array(self.areas)
        self.K0 = self.assemble([e['K'] for e in self.elements], np.zeros(3)).real
        self.kept = self.K0[np.ix_(self.free, self.free)]
        self.scale = 1 / np.sqrt(np.diag(self.kept))
        self.factor = la.cho_factor(self.kept * self.scale[:, None] * self.scale[None, :])

    def relaxed(self, strain):
        force = np.zeros(self.ndof)
        imposed = []
        for e in self.elements:
            d = np.r_[strain @ e['a'], np.zeros(3), strain @ e['b'], np.zeros(3)]
            f = e['K'] @ d
            np.add.at(force, e['di'], f[:6])
            np.add.at(force, e['dj'], f[6:])
            imposed.append(d)
        w = np.zeros(self.ndof)
        w[self.free] = -self.scale * la.cho_solve(self.factor, force[self.free] * self.scale)
        forces = []
        stress = np.zeros((3, 3))
        energy = 0.0
        moments = 0.0
        for e, d, A in zip(self.elements, imposed, self.areas):
            d = d + np.r_[w[e['di']], w[e['dj']]]
            f = e['K'] @ d
            delta = e['b'] - e['a']
            length = la.norm(delta)
            forces.append(A / length * np.dot(delta / length, d[6:9] - d[:3]))
            stress += 0.5 * (np.outer(f[6:9], delta) + np.outer(delta, f[6:9]))
            energy += 0.5 * d @ f
            moments += sum((f[j] ** 2 for j in [3, 4, 5, 9, 10, 11]))
        stress /= abs(la.det(self.A))
        res = la.norm(self.K0 @ w + force)
        return dict(w=w, forces=np.array(forces), stress=stress, energy=energy, relative_residual=res / max(la.norm(force), 1e-30), force_scaled_residual=res / max(abs(np.array(forces)).max(), 1e-30), absolute_residual=res, max_nonaffine=float(abs(w).max()), moment_norm=float(np.sqrt(moments)))

def restricted(K, H, P):
    nf = len(K) - P.shape[0]
    B = np.zeros((len(K), nf + P.shape[1]))
    B[:nf, :nf] = np.eye(nf)
    B[nf:, nf:] = P
    return (B.T @ K @ B, B.T @ H @ B)

def first_root(K, H, vector=False):
    scale = 1 / np.sqrt(np.diag(K))
    K = K * scale[:, None] * scale[None, :]
    H = H * scale[:, None] * scale[None, :]
    ev, v = la.eigh(H, K, subset_by_index=[len(K) - 1, len(K) - 1], check_finite=False)
    val = float(1 / ev[-1]) if ev[-1] > 1e-13 else np.inf
    return (val, scale * v[:, -1]) if vector else val

class TangentAtlas:
    """Macroscopic Gibbs completion or full-gradient material-Hessian operator.

    objective=True enforces rigid-rotation invariance in the homogeneous block.
    objective=False takes that block directly from Frame.affineK/affineH.
    acoustic() evaluates the selected macroscopic model; Frame.eigen supplies
    the microscopic Bloch spectrum independently.
    """

    def __init__(self, f, objective=True):
        self.f = f
        f.prestress('stress')
        f.affine_matrices()
        self.K = f.affineK
        self.H = f.affineH
        self.vol = abs(la.det(f.A))
        self.p0 = -np.trace(f.pre['stress']) / 3
        self.nf = len(self.K) - 9
        self.gamma = float(f.eigen([0, 0, 0], 1)[0])
        self.S = sym_basis(3)
        self.Hpressure = self.H.copy()
        self.Hpressure[-9:, -9:] -= self.p0 * self.vol * volume_hessian(3)
        self.rawK = self.K.copy()
        self.rawH = self.H.copy()
        self.rawHpressure = self.Hpressure.copy()
        self.K6, self.H6 = restricted(self.K, self.Hpressure, self.S)
        self.objective = objective
        self.W = np.zeros((9, 3))
        for h, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            self.W[3 * i + j, h] = 1 / np.sqrt(2)
            self.W[3 * j + i, h] = -1 / np.sqrt(2)
        U = np.zeros((f.ndof, 3))
        for h in range(3):
            O = self.W[:, h].reshape(3, 3)
            U.reshape(-1, 6, 3)[:, 3:, h] = np.array([O[2, 1], O[0, 2], O[1, 0]])
        self.U = U[f.free]
        self.R = np.vstack((self.U, self.W))
        if objective:
            T = np.zeros((self.nf + 6, self.nf + 9))
            T[:self.nf, :self.nf] = np.eye(self.nf)
            T[:self.nf, self.nf:] = -self.U @ self.W.T
            T[self.nf:, self.nf:] = self.S.T
            self.K = T.T @ self.K6 @ T
            self.Hpressure = T.T @ self.H6 @ T
            self.H = self.Hpressure.copy()
            self.H[-9:, -9:] += self.p0 * self.vol * volume_hessian(3)
        self.mu, self.V = la.eigh(self.H6[:self.nf, :self.nf], self.K6[:self.nf, :self.nf])
        self.B0 = self.V.T @ self.K6[:self.nf, self.nf:]
        self.B1 = self.V.T @ self.H6[:self.nf, self.nf:]
        self.cache = {}

    def tangent(self, load, pressure=False):
        key = (float(load), pressure)
        if key not in self.cache:
            if self.objective:
                A = self.S @ self.gibbs(load) @ self.S.T
                if not pressure:
                    A = A - load * self.p0 * volume_hessian(3)
            else:
                M = self.K - load * (self.Hpressure if pressure else self.H)
                nf = self.nf
                A = (M[nf:, nf:] - M[nf:, :nf] @ la.solve(M[:nf, :nf], M[:nf, nf:], assume_a='sym')) / self.vol
            self.cache[key] = (A + A.T) / 2
        return self.cache[key]

    def gibbs(self, load):
        B = self.B0 - load * self.B1
        C = self.K6[-6:, -6:] - load * self.H6[-6:, -6:] - B.T @ (B / (1 - load * self.mu)[:, None])
        return (C + C.T) / (2 * self.vol)

    def pd(self, pressure=False):
        K, H = restricted(self.K, self.Hpressure if pressure else self.H, self.S)
        root, v = first_root(K, H, True)
        E = (self.S @ v[-6:]).reshape(3, 3)
        frac = float(np.trace(E) ** 2 / (3 * np.sum(E * E))) if la.norm(E) > 1e-15 else 0.0
        censored = root >= self.gamma * (1 - 2e-06)
        return dict(load=None if censored else root, bound=float(min(root, self.gamma)), censored_by_gamma=censored, volumetric_fraction=frac, strain_mode=(E / max(la.norm(E), 1e-30)).tolist())

    def acoustic(self, n):
        n = np.asarray(n)
        n = n / la.norm(n)
        P = np.kron(np.eye(3), n[:, None])
        if self.objective:
            P6 = self.S.T @ P

            def smallest(t):
                return la.eigvalsh(P6.T @ self.gibbs(t) @ P6, check_finite=False)[0]
            upper = self.gamma * (1 - 1e-09)
            if smallest(upper) >= 0:
                return self.gamma
            return brentq(smallest, 0, upper, xtol=max(self.gamma * 1e-13, 1e-16), rtol=2e-13)
        K, H = restricted(self.K, self.H, P)
        return first_root(K, H)

    def le(self):
        dirs = [np.array(t, float) / la.norm(t) for t in itertools.product((-1, 0, 1), repeat=3) if any(t)]
        count = 38
        for i in range(count):
            z = 1 - 2 * (i + 0.5) / count
            angle = i * np.pi * (3 - np.sqrt(5))
            dirs.append(np.array([np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z]))
        rows = sorted([(self.acoustic(n), n) for n in dirs], key=lambda x: x[0])
        best = rows[0]
        if self.objective:
            pd = self.pd(True)
            if not pd['censored_by_gamma']:
                E = np.array(pd['strain_mode'])
                ev, U = la.eigh(E)
                if abs(ev[1]) < 1e-08 and ev[0] < 0 < ev[2]:
                    n = np.sqrt(ev[2]) * U[:, 2] + np.sqrt(-ev[0]) * U[:, 0]
                    n /= la.norm(n)
                    rows.append((self.acoustic(n), n))
                    rows.sort(key=lambda x: x[0])
                    best = rows[0]
        for _, n in rows[:2]:
            opt = minimize(lambda v: self.acoustic(v) / best[0], n, method='Nelder-Mead', options={'maxiter': 110, 'xatol': 1e-05, 'fatol': 1e-08})
            val = self.acoustic(opt.x)
            if val < best[0]:
                best = (val, opt.x / la.norm(opt.x))
        root, n = best
        censored = root >= self.gamma * (1 - 2e-06)
        eval_load = min(root, self.gamma) * (0.999 if censored else 1 - 1e-05)
        A = self.tangent(eval_load)
        P = np.kron(np.eye(3), n[:, None])
        eig = la.eigvalsh(P.T @ A @ P)
        null_error = la.norm(P.T @ volume_hessian(3) @ P)
        return dict(load=None if censored else float(root), bound=float(min(root, self.gamma)), censored_by_gamma=censored, direction=n.tolist(), acoustic_eigenvalues_near_bound=eig.tolist(), pressure_null_lagrangian_error=float(null_error))

    def bloch(self, grid=4):
        f = self.f
        points = representatives(f, grid)
        rows = []
        for k in points:
            rows.append((float(f.eigen(k, 1)[0]), np.array(k)))
        rows.sort(key=lambda t: t[0])
        best = rows[0]

        def objective(k):
            q = np.asarray(k) - np.rint(k)
            if la.norm(q) < 0.025:
                return 1000.0
            return float(f.eigen(q, 1)[0]) / rows[0][0]
        for _, k in [r for r in rows if la.norm(r[1]) > 1e-08][:2]:
            opt = minimize(objective, k, method='Nelder-Mead', options={'maxiter': 90, 'xatol': 2e-05, 'fatol': 2e-08})
            val = opt.fun * rows[0][0]
            if val < best[0]:
                best = (val, opt.x % 1)
        return dict(load=float(best[0]), k=best[1].tolist(), grid=grid, points=len(points), near_excluded_origin=bool(la.norm(best[1] - np.rint(best[1])) < 0.027 and la.norm(best[1]) > 1e-08), gamma_optical=self.gamma)

def classify(pd, le, mu):
    lv = le['load'] if le['load'] is not None else np.inf
    pv = pd['load'] if pd['load'] is not None else np.inf
    mv = mu['load']
    smallest = min(pv, lv, mv)
    tol = 0.003
    if pv < min(lv, mv) * (1 - tol):
        return 'PD-only-volumetric' if pd['volumetric_fraction'] > 0.8 else 'PD-only-nonvolumetric'
    if lv <= mv * (1 + tol) and lv <= pv * (1 + tol):
        return 'LE' if pv > lv * (1 + tol) else 'PD+LE'
    if mv <= pv * (1 + tol) and mv <= lv * (1 + tol):
        return 'micro' if pv > mv * (1 + tol) else 'PD+micro'
    return 'near-tie'

def graph_data(name):
    """Production loader: edges from RCSR EDGE_CENTER orbits (rcsr_loader.load_ec); no nearest-neighbour fallback."""
    g = LD['load_ec'](name, NETS)
    if g is None or not g['ok']:
        raise ValueError('CN mismatch')
    if len(g['frac']) > 12:
        raise ValueError('cell exceeds 12 nodes')
    lengths = np.array([la.norm((g['frac'][j] + np.array(sh) - g['frac'][i]) @ g['A']) for i, j, sh in g['edges']])
    mean = lengths.mean()
    sg = gemmi.find_spacegroup_by_name(g['net']['group'] + (g['setting'] or ''))
    return dict(name=name, net=g['net'], sg=sg, frac=g['frac'], cn=g['cn'], edges=g['edges'], A=g['A'] / mean, trials=[dict(edge_centre_loader=True)], lengths=(lengths / mean).tolist())

def select(limit=40):
    preferred = ['pcu', 'dia', 'srs', 'nbo', 'sod', 'bor', 'fcu', 'utp', 'acs', 'ths']
    candidates = preferred + [n['name'] for n in NETS if '-' not in n['name'] and n['name'] not in preferred]
    result = []
    rejected = []
    for name in candidates:
        net = next((n for n in NETS if n['name'] == name))
        ex = LD['expand'](net)
        if ex is None:
            continue
        if len(ex[0]) > 12:
            ex = LD['expand'](net, ':2')
            if ex is None or len(ex[0]) > 12:
                continue
        try:
            d = graph_data(name)
            f = SlenderFrame(name, 0.02, 1, d)
            f.prestress('stress')
            result.append(dict(name=name, nodes=len(d['frac']), edges=len(d['edges']), cn=d['cn'].tolist(), lengths=d['lengths'], setting=d['sg'].xhm()))
            print('selected', len(result), name, len(d['frac']), flush=True)
        except Exception as exc:
            rejected.append(dict(name=name, reason=str(exc)))
        if len(result) >= limit:
            break
    dump(ROOT / 'ec_catalog.json', dict(selected=result, rejected=rejected, selection='ten specified nets, then CGD order; no hyphen derivatives; <=12 conventional nodes, CN validated, K positive after translation gauge'))
    return [r['name'] for r in result]

def run(name, radius, div=2, grid=4):
    start = time.perf_counter()
    d = graph_data(name)
    f = SlenderFrame(name, radius, div, d)
    atlas = TangentAtlas(f)
    pd = atlas.pd(False)
    pressurepd = atlas.pd(True)
    le = atlas.le()
    micro = atlas.bloch(grid)
    first = min(pd['bound'], le['bound'], micro['load'])
    probe = first * 0.8
    A = atlas.tangent(probe)
    AP = atlas.tangent(probe, True)
    S = sym_basis(3)
    C = S.T @ AP @ S
    skew = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        M = np.zeros((3, 3))
        M[i, j] = 1 / np.sqrt(2)
        M[j, i] = -1 / np.sqrt(2)
        skew.append(M.ravel())
    W = np.array(skew).T
    return dict(name=name, r_over_each_parent_length=radius, div=div, nodes=len(d['frac']), edges=len(d['edges']), cn=d['cn'].tolist(), setting=d['sg'].xhm(), pressure_per_unit_factor=float(atlas.p0), preload_strain=f.pre['strain'].tolist(), preload_stress=f.pre['stress'].tolist(), equilibrium_residual=f.pre['force_scaled_residual'], PD_dead_nominal=pd, PD_pressure_pV=pressurepd, LE=le, micro=micro, class_dead_nominal=classify(pd, le, micro), class_pressure_pV=classify(pressurepd, le, micro), lambda_PD_pressure=None if pressurepd['load'] is None else pressurepd['load'] * atlas.p0, lambda_PD_dead=None if pd['load'] is None else pd['load'] * atlas.p0, lambda_LE_pressure=None if le['load'] is None else le['load'] * atlas.p0, lambda_micro_pressure=None if micro['near_excluded_origin'] else micro['load'] * atlas.p0, sampled_micro_upper_pressure=micro['load'] * atlas.p0, tangent_probe_factor=probe, full_gradient_tangent=A.tolist(), pressure_symmetric_tangent=C.tolist(), pressure_objectivity_residual=float(la.norm(AP @ W) / max(la.norm(AP), 1e-30)), seconds=time.perf_counter() - start)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--select', type=int, default=0)
    ap.add_argument('--nets', nargs='+')
    ap.add_argument('--div', type=int, default=2)
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--radii', nargs='+', type=float, default=[0.02, 0.05])
    ap.add_argument('--out', default='ec_atlas.json')
    a = ap.parse_args()
    if a.select:
        select(a.select)
    else:
        names = a.nets or [r['name'] for r in json.loads((ROOT / 'ec_catalog.json').read_text())['selected']]
        out = dict(model='objective macro Gibbs completion; unchanged axial-prestress Bloch operator; macro/micro long-wave mismatch retained', runs=[], errors=[])
        for name in names:
            for radius in a.radii:
                try:
                    r = run(name, radius, a.div, a.grid)
                    out['runs'].append(r)
                    print(name, radius, r['class_dead_nominal'], r['class_pressure_pV'], r['PD_pressure_pV']['bound'], r['LE']['bound'], r['micro']['load'], round(r['seconds'], 2), flush=True)
                except Exception as e:
                    out['errors'].append(dict(name=name, radius=radius, error=repr(e)))
                    print('ERROR', name, repr(e), flush=True)
                dump(ROOT / a.out, out)
