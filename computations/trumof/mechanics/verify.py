"""TRUMOF mechanics. Four independent single-thread shards, resumable files.
Bundled inputs are read from this package; outputs go to the cache (K08_CACHE).
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os, sys
sys.dont_write_bytecode = True
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import json, time, traceback, itertools
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE.parent))
from computations.common.runtime import cache_dir, output_dir
OUT = cache_dir('trumof/mechanics')
for sub in ('results', 'states'):
    (OUT / sub).mkdir(exist_ok=True)
sys.path.insert(0, str(PACKAGE / 'model'))
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import eigsh, spsolve, ArpackNoConvergence
from scipy.optimize import root, brentq
from computations.trumof.mechanics.model.model import CG, S, SOURCE_DELTA, le_min
from computations.trumof.mechanics.model.rule import lattice, graph, energy
from computations.trumof.mechanics.model.bloch_scan import gridks
continuation_inputs = PACKAGE / 'inputs/continuation'
convention_inputs = PACKAGE / 'inputs/conventions'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def dump(p, z):
    p = Path(p)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(z, indent=2, allow_nan=False, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)), encoding='utf-8')
    os.replace(tmp, p)
SUMMARY_SEED = read(continuation_inputs / 'summary_seed.json')
CONVENTIONS_SEED = read(convention_inputs / 'conventions_seed.json')
RULE = SUMMARY_SEED['rule']
PAR = read(continuation_inputs / 'pcu_parameters.json')
APP = {z['name']: z for z in SUMMARY_SEED['approximants']}
ENS = read(continuation_inputs / 'random_configurations.json')
PAT = read(continuation_inputs / 'planted_configurations.json')

def case_data(row):
    if row['group'] == 'approximant':
        return APP[row['name']]
    if row['group'] == 'random':
        return next((z for z in ENS if z['L'] == row['L'] and z['seed'] == row['seed']))
    return next((z for z in PAT if z['L'] == row['L'] and z['R_cells'] == row['R'] and (z['approximant'] == row['planted_approximant'])))

def model(row):
    z = case_data(row)
    p = PAR[row['parameters']]
    kw = dict(kr=p['C11_GPa'] * 13.0, kt=p['C44_GPa'] * 13.0 ** 3 / 4.0)
    if row['group'] == 'approximant':
        m = CG(np.array(z['positions']) * RULE['a'], z['cell'], z['edges'], **kw)
    else:
        assert energy(z['q'], RULE, z['L']) == 0
        m = CG(lattice(z['L'], RULE)[0] * RULE['a'], np.eye(3) * z['L'] * RULE['a'], graph(z['q'], RULE, z['L']), **kw)
    return m

def initial_PD(row):
    name = row['name']
    kind = row['parameters']
    if name == 'random_L4_seed403' and kind == 'soft':
        seed = PACKAGE / 'inputs/random403_seed.npz'
        if seed.exists():
            st = np.load(seed)
            return (float(st['p']), st['x'].copy())
    if row['group'] != 'approximant' and kind == 'hard':
        st = np.load(convention_inputs / f'objective_{name}_{kind}.npz')
        return (float(st['p_PD']), st['x_PD'].copy())
    stem = name + '_' + kind if row['group'] == 'approximant' else name
    path = continuation_inputs / f'critical_state_{stem}.npz'
    if path.exists():
        st = np.load(path)
        return (float(st['p']), st['x'].copy())
    if row['group'] != 'approximant':
        st = np.load(continuation_inputs / f'ensemble_state_{name}.npz')
        return (float(st['p']), st['x'].copy())
    st = np.load(continuation_inputs / f'path_{stem}.npz')
    i = int(np.argmin(abs(st['p'] - row['objective']['lambda_PD_GPa'])))
    return (float(st['p'][i]), st['x'][i].copy())

class Spectrum:

    def __init__(self, m):
        self.m = m
        self.cache = {}
        self.last = None
        self.max_residual = 0.0
        self.calls = 0

    def set_state(self, x):
        if self.last is not None and np.array_equal(x, self.last):
            return
        self.last = x.copy()
        loc = self.m.local(x)
        self.HB = self.m.blockdiag(loc[6])
        self.HA = self.m.blockdiag(loc[8])

    def matrices(self, k):
        key = tuple(np.asarray(k, float))
        m = self.m
        if key not in self.cache:

            def incidence(center, end, sh):
                phase = np.exp(2j * np.pi * (sh @ np.asarray(k)))
                n = len(center)
                ids = np.stack([3 * center[:, None] + np.arange(3), 3 * end[:, None] + np.arange(3)], axis=2).ravel()
                vals = np.stack([-np.ones((n, 3)), np.repeat(phase[:, None], 3, axis=1)], axis=2).ravel()
                return sparse.coo_matrix((vals, (np.repeat(np.arange(3 * n), 2), ids)), shape=(3 * n, 3 * m.N)).tocsr()
            B = incidence(m.e[:, 0], m.e[:, 1], m.e[:, 2:])
            first_angle_incidence = incidence(m.ai, m.aj, m.ajshift)
            second_angle_incidence = incidence(m.ai, m.ak, m.akshift)
            AB = sparse.vstack([first_angle_incidence, second_angle_incidence], format='csr')[np.arange(6 * len(m.ai)).reshape(2, -1, 3).transpose(1, 0, 2).ravel()]
            Q = None
            if m.N <= 108:
                phase = np.exp(2j * np.pi * (m.e[:, 2:] @ np.asarray(k)))
                A = sparse.coo_matrix((np.c_[-np.ones(len(m.e)), phase].ravel(), (np.repeat(np.arange(len(m.e)), 2), m.e[:, :2].ravel())), shape=(len(m.e), m.N)).toarray()
                v, U = linalg.eigh(A.conj().T @ A)
                if v[0] < 1e-09:
                    Q = np.kron(U[:, v > 1e-09], np.eye(3))
            self.cache[key] = (B, AB, Q)
        return self.cache[key]

    def eigen(self, x, k, vector=False, check=False):
        self.set_state(x)
        m = self.m
        B, AB, Q = self.matrices(k)
        H = (B.conj().T @ self.HB @ B + AB.conj().T @ self.HA @ AB).tocsr()
        isgamma = np.linalg.norm(k) == 0
        pinned = isgamma and m.N > 108
        if pinned:
            Hwork = H[m.free][:, m.free]
        elif Q is not None:
            Hwork = Q.conj().T @ H @ Q
        else:
            Hwork = H
        self.calls += 1
        if Hwork.shape[0] <= 324:
            A = Hwork.toarray() if sparse.issparse(Hwork) else Hwork
            ev, U = linalg.eigh(A, subset_by_index=[0, 0])
            val = float(ev[0])
            vec = U[:, 0]
            residual = float(np.linalg.norm(A @ vec - val * vec) / max(1.0, abs(val)))
        else:
            vals = []
            for seed in [421, 913] if check else [421]:
                rng = np.random.default_rng(seed)
                v0 = rng.normal(size=Hwork.shape[0]) + 1j * rng.normal(size=Hwork.shape[0])
                success = False
                for ncv in [40, 90]:
                    try:
                        ev, U = eigsh(Hwork, k=1, which='SA', tol=1e-09, maxiter=20000, ncv=min(ncv, Hwork.shape[0] - 1), v0=v0)
                        val = float(ev[0])
                        vec = U[:, 0]
                        residual = float(np.linalg.norm(Hwork @ vec - val * vec) / max(1.0, abs(val)))
                        if residual < 2e-06:
                            success = True
                            break
                    except ArpackNoConvergence:
                        pass
                assert success, ('SA failed', m.N, k)
                vals.append((val, vec, residual))
            if check:
                assert abs(vals[0][0] - vals[1][0]) < 2e-05 * max(1.0, abs(vals[0][0])), ('SA starts disagree', vals[0][0], vals[1][0])
            val, vec, residual = min(vals, key=lambda z: z[0])
        self.max_residual = max(self.max_residual, residual)
        if vector:
            if pinned:
                full = np.zeros(3 * m.N, complex)
                full[m.free] = vec
                vec = full
            elif Q is not None:
                vec = Q @ vec
            return (val, vec, residual)
        return val

def classify(C):
    ev, U = linalg.eigh(C)
    strain = np.einsum('a,aij->ij', U[:, 0], S)
    eig = linalg.eigvalsh(strain)
    defect = float(abs(eig[1]) / np.linalg.norm(eig))
    le, n = le_min(C)
    cls = 'PD+LE' if defect < 1e-07 and eig[0] * eig[2] <= 0 else 'PD-only'
    return (cls, dict(PD_eigen_GPa=float(ev[0]), second_PD_eigen_GPa=float(ev[1]), principal_null_strains=eig, relative_middle_principal_strain=defect, LE_eigen_GPa=le, LE_direction=n, PD_strain_mode=U[:, 0]))

class Engine:

    def __init__(self, row):
        self.row = row
        self.m = model(row)
        self.sp = Spectrum(self.m)
        self.cache = [(0.0, np.zeros(3 * self.m.N + 6))]
        self.trace = []

    def equilibrium(self, p, initial):
        if self.row['name'] == 'trumof_1x1x1_1':
            m = self.m
            A = m.kr * np.sum(m.re ** 2)
            scale = 2 * A / (A + np.sqrt(A * A + 12 * p * m.V0 * A))
            x = np.zeros(3 * m.N + 6)
            x[-6:-3] = scale - 1
            err = self.residual(x, p)
            return (x, dict(converged=err < 1e-08, residual=err))
        return self.m.relax(float(p), initial, maxiter=100)

    def load(self, p, initial=None):
        for q, x in self.cache:
            if abs(q - p) < 1e-13:
                return x.copy()
        if initial is None:
            initial = min(self.cache, key=lambda z: abs(z[0] - p))[1]
        x, info = self.equilibrium(float(p), initial)
        if not info['converged']:
            lower = [(q, y) for q, y in self.cache if q < p - 1e-10]
            q0, y0 = max(lower, key=lambda z: z[0])
            for subdivisions in [16, 64]:
                x = y0.copy()
                for qq in np.linspace(q0, p, subdivisions + 1)[1:]:
                    x, info = self.equilibrium(float(qq), x)
                    if not info['converged']:
                        break
                if info['converged']:
                    break
        assert info['converged'], ('relax', self.row['name'], p, info)
        self.cache.append((float(p), x.copy()))
        return x

    def residual(self, x, p):
        g = self.m.objective(x, p)[1]
        return float(max(np.max(abs(g[:-6])) / self.m.kr, np.max(abs(g[-6:])) / self.m.V0))

    def pd_root(self):
        m = self.m
        p0, x0 = initial_PD(self.row)
        if self.row['group'] == 'approximant' and self.row['L'] == 1:
            cache = [(p0, x0)]

            def fun(p):
                xx, info = self.equilibrium(p, min(cache, key=lambda z: abs(z[0] - p))[1])
                assert info['converged'], ('unit relaxation', p, info)
                cache.append((p, xx))
                return float(linalg.eigvalsh(m.tangent(xx, p)[0])[0])
            delta = max(0.0002, p0 * 0.0001)
            lo = p0 - delta
            hi = p0 + delta
            while fun(lo) * fun(hi) > 0:
                delta *= 2
                lo = p0 - delta
                hi = p0 + delta
                assert delta < 0.1
            pc = brentq(fun, lo, hi, xtol=2e-11)
            fun(pc)
            xc = cache[-1][1]
            method = 'signed PD crossing'
            evidence = dict(bracket=[lo, hi], signs=[fun(lo), fun(hi)])
        else:
            cache = x0.copy()
            calls = 0

            def calc(t, full=False):
                nonlocal cache, calls
                x = cache.copy()
                x[-6:] = t[:6]
                p = float(t[6])
                for _ in range(40):
                    _, g, H = m.objective(x, p, True)
                    if np.max(abs(g[m.free])) < 5e-09:
                        break
                    x[m.free] += spsolve(H[m.free][:, m.free], -g[m.free])
                g = m.objective(x, p)[1]
                C = m.tangent(x, p)[0]
                cache = x.copy()
                calls += 1
                res = np.r_[g[-6:] / m.V0, linalg.eigvalsh(C)[0]]
                return (res, x) if full else res
            opt = root(calc, np.r_[x0[-6:], p0], method='hybr', options={'xtol': 1e-10, 'eps': 1e-10, 'maxfev': 240})
            res, xc = calc(opt.x, True)
            pc = float(opt.x[6])
            assert np.max(abs(res)) < 2e-07 and abs(pc - p0) < 0.1, ('PD fold', self.row['name'], res, opt.message)
            method = 'augmented equilibrium/zero PD fold'
            evidence = dict(residual=float(np.max(abs(res))), calls=calls, message=str(opt.message))
        self.cache.append((pc, xc.copy()))
        assert self.residual(xc, pc) < 2e-07
        return (pc, xc, dict(method=method, **evidence))

    def diagnostics(self, x, p, ks, published=False, check=False):
        C, K, _, _ = self.m.tangent(x, p)
        if published:
            C = C + p * SOURCE_DELTA
        cls, d = classify(C)
        checks = [dict(k=k, eigen=self.sp.eigen(x, k, check=check)) for k in ks]
        best = min(checks, key=lambda r: r['eigen'])
        gamma = self.sp.eigen(x, np.zeros(3), check=check)
        d.update(gamma_pinned_eigen=gamma, Bloch_min_eigen=best['eigen'], Bloch_k=best['k'], Bloch_checks=checks, equilibrium_residual=self.residual(x, p))
        return (cls, d)

    def published(self, ks):
        p0 = self.row['published']['first_pressure_GPa']
        delta = max(0.0002, p0 * 0.0001)

        def fun(p):
            xx = self.load(p)
            return float(linalg.eigvalsh(self.m.tangent(xx, p)[0] + p * SOURCE_DELTA)[0])
        lo = p0 - delta
        hi = p0 + delta
        while fun(lo) * fun(hi) > 0:
            delta *= 2
            lo = p0 - delta
            hi = p0 + delta
            assert delta < 0.1
        pc = brentq(fun, lo, hi, xtol=2e-10)
        xc = self.load(pc)
        cls, diag = self.diagnostics(xc, pc, ks, published=True)
        signed = [dict(p_GPa=pc + d, PD_eigen_GPa=fun(pc + d)) for d in [-1e-05, 1e-05]]
        assert signed[0]['PD_eigen_GPa'] > 0 and signed[1]['PD_eigen_GPa'] < 0
        assert diag['Bloch_min_eigen'] > -1e-06 and diag['gamma_pinned_eigen'] > -1e-06
        out = dict(first_pressure_GPa=pc, first_class=cls, lambda_PD_GPa=pc, lambda_LE_GPa=pc if cls == 'PD+LE' else None, lambda_micro_GPa=None, diagnostics=diag, signed_check=signed, bracket_GPa=[lo, hi], classification_check={k: diag[k] for k in ['PD_eigen_GPa', 'second_PD_eigen_GPa', 'principal_null_strains', 'relative_middle_principal_strain', 'LE_eigen_GPa']})
        out['classification_check'].update(name=self.row['name'], parameters=self.row['parameters'], first_class=cls)
        if self.row['group'] == 'approximant' and self.row['L'] == 1:
            if cls == 'PD+LE':
                pr = pc
            else:

                def lefun(p):
                    return le_min(self.m.tangent(self.load(p), p)[0] + p * SOURCE_DELTA)[0]
                hi = pc * 1.03
                while lefun(hi) > 0:
                    hi += pc * 0.03
                    assert hi < self.row['objective']['lambda_PD_GPa']
                pr = brentq(lefun, pc, hi, xtol=2e-09)
            xx = self.load(pr)
            cc = self.m.tangent(xx, pr)[0]
            out['diagnostic_LE_root'] = dict(name=self.row['name'], parameters=self.row['parameters'], diagnostic_LE_root_GPa=pr, PD_at_LE_GPa=float(linalg.eigvalsh(cc + pr * SOURCE_DELTA)[0]), objective_PD_at_LE_GPa=float(linalg.eigvalsh(cc)[0]), scope='Objective equilibrium continued beyond the pressure-convention PD threshold; the pressure-convention state is unstable.')
        return (out, xc)

def material_mode(m, x, p, k, u=None, strain=None, mask=None):
    F, uu, w, aw, E, g, h, ag, ah = m.local(x)
    if strain is not None:
        _, K, c, _ = m.tangent(x, p)
        u = np.zeros(3 * m.N)
        u[m.free] = -spsolve(K[m.free][:, m.free], c[m.free] @ strain)
        dw = (m.B @ u).reshape(-1, 3) + np.einsum('a,aij,nj->ni', strain, S, w)
        da = (m.AB @ u).reshape(-1, 6) + np.c_[np.einsum('a,aij,nj->ni', strain, S, aw[:, :3]), np.einsum('a,aij,nj->ni', strain, S, aw[:, 3:])]
    else:
        uv = u.reshape(-1, 3)
        dw = np.exp(2j * np.pi * (m.e[:, 2:] @ k))[:, None] * uv[m.e[:, 1]] - uv[m.e[:, 0]]
        da = np.c_[np.exp(2j * np.pi * (m.ajshift @ k))[:, None] * uv[m.aj] - uv[m.ai], np.exp(2j * np.pi * (m.akshift @ k))[:, None] * uv[m.ak] - uv[m.ai]]
    er = 0.5 * m.kr * abs(np.sum(w * dw, axis=1)) ** 2 / np.sum(w * w, axis=1)
    a = aw[:, :3]
    b = aw[:, 3:]
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    ua = a / an[:, None]
    ub = b / bn[:, None]
    c = np.einsum('ij,ij->i', ua, ub)
    s = np.linalg.norm(np.cross(ua, ub), axis=1)
    linear = s < 1e-12
    gc = np.c_[(ub - c[:, None] * ua) / an[:, None], (ua - c[:, None] * ub) / bn[:, None]]
    ea = np.zeros(len(a))
    non = ~linear
    ea[non] = 0.5 * m.kt * abs(np.einsum('ni,ni->n', gc[non], da[non]) / s[non]) ** 2
    if linear.any():
        Pa = np.eye(3)[None] - np.einsum('ni,nj->nij', ua[linear], ua[linear])
        Pb = np.eye(3)[None] - np.einsum('ni,nj->nij', ub[linear], ub[linear])
        dv = np.einsum('nij,nj->ni', Pa, da[linear, :3]) / an[linear, None] - np.sign(c[linear, None]) * np.einsum('nij,nj->ni', Pb, da[linear, 3:]) / bn[linear, None]
        ea[linear] = 0.5 * m.kt * np.sum(abs(dv) ** 2, axis=1)
    ne = np.zeros(m.N)
    np.add.at(ne, m.e[:, 0], er / 2)
    np.add.at(ne, m.e[:, 1], er / 2)
    np.add.at(ne, m.ai, ea)
    V = m.V0 * np.linalg.det(F)
    rad = np.sum(g * w, axis=1)
    ang = np.sum(ag * aw, axis=1)
    vir = np.zeros(m.N)
    np.add.at(vir, m.e[:, 0], rad / 2)
    np.add.at(vir, m.e[:, 1], rad / 2)
    np.add.at(vir, m.ai, ang)
    pp = -sum(vir[mask]) / (3 * V * sum(mask) / m.N)
    cell = ne.reshape(-1, 4).sum(1)
    d = dict(patch_energy_fraction=float(sum(ne[mask]) / sum(ne)), node_energy_participation=float(sum(ne) ** 2 / (m.N * sum(ne ** 2))), cell_energy_participation=float(sum(cell) ** 2 / (len(cell) * sum(cell ** 2))), material_energy_total=float(sum(ne)), patch_stress=dict(s_P=float(pp / p), patch_pressure_GPa=float(pp), global_pressure_from_virial=float(-sum(vir) / (3 * V)), convention='Current-volume virial, radial half to endpoints, angle term to centre'), energy_convention='Positive material quadratic energy at current geometry; exact linear-angle limit; PD includes macro strain and relaxed fluctuation.')
    return (d, ne, u)

def verify(row):
    start = time.perf_counter()
    cpu = time.process_time()
    e = Engine(row)
    m = e.m
    ks = gridks(4) if row['group'] == 'approximant' else [np.array(k, float) for k in [[0.5, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]]
    pub, xpub = e.published(ks)
    pd, xpd, evidence = e.pd_root()
    pdcls, pdiag = e.diagnostics(xpd, pd, ks, check=True)
    path = []
    allks = [np.zeros(3)] + ks
    for fraction in [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]:
        p = pd * fraction
        x = xpd if fraction == 1 else e.load(p)
        vals = [e.sp.eigen(x, k) for k in allks]
        C = m.tangent(x, p)[0]
        path.append(dict(p_GPa=p, PD_eigen_GPa=float(linalg.eigvalsh(C)[0]), eigenvalues=vals, equilibrium_residual=e.residual(x, p)))
    assert min(path[0]['eigenvalues']) > -1e-07, ('unstable zero load', row['name'])
    candidates = []
    for j, k in enumerate(allks):
        for prev, nxt in zip(path[:-1], path[1:]):
            if nxt['eigenvalues'][j] < -1e-07:
                lo = prev['p_GPa']
                hi = nxt['p_GPa']
                assert prev['eigenvalues'][j] > 0, ('missed lower negative', row['name'], k)

                def fun(p):
                    return e.sp.eigen(e.load(p), k)
                pc = brentq(fun, lo, hi, xtol=2e-09)
                candidates.append(dict(pressure_GPa=pc, k=k, bracket_GPa=[lo, hi]))
                break
    micro = min(candidates, key=lambda z: z['pressure_GPa']) if candidates else None
    if micro and micro['pressure_GPa'] < pd - 1e-07:
        first = micro['pressure_GPa']
        xf = e.load(first)
        k = np.array(micro['k'])
        cls = 'finite k' if np.linalg.norm(k) > 0 else 'Gamma optical'
        val, u, res = e.sp.eigen(xf, k, vector=True, check=True)
        _, diag = e.diagnostics(xf, first, ks)
        strain = None
        signs = [dict(p_GPa=first + d, eigen=e.sp.eigen(e.load(first + d), k)) for d in [-1e-06, 1e-06]]
        assert signs[0]['eigen'] > 0 and signs[1]['eigen'] < 0
        micro.update(eigen=val, eigen_residual=res, signed_check=signs)
    else:
        first = pd
        xf = xpd
        k = np.zeros(3)
        cls = pdcls
        diag = pdiag
        u = None
        strain = np.array(diag['PD_strain_mode'])
        micro = None
    objective = dict(first_pressure_GPa=first, first_class=cls, lambda_PD_GPa=pd, lambda_LE_GPa=pd if pdcls == 'PD+LE' else None, lambda_micro_GPa=first if micro else None, micro=micro, diagnostics=diag, diagnostics_at_PD=pdiag, fold_residual=e.residual(xpd, pd), root_evidence=evidence, trace=path, scope='Straight-reference-angle kernel; smallest algebraic eigenvalue at every declared wavevector; finite grids, not a global BZ certificate.')
    patch = None
    ne = None
    if row['group'] == 'planted':
        mask = np.array(case_data(row)['frozen_mask'], bool)
        patch, ne, u = material_mode(m, xf, first, k, u, strain, mask)
    stem = row['name'] + '_' + row['parameters']
    states = dict(x=xf, p=first, x_PD=xpd, p_PD=pd, x_published=xpub, p_published=pub['first_pressure_GPa'], k=k)
    if u is not None:
        states['u'] = u
    if strain is not None:
        states['strain_mode'] = strain
    if ne is not None:
        states['energy_per_node'] = ne
    np.savez_compressed(OUT / 'states' / f'{stem}.npz', **states)
    out = dict(name=row['name'], parameters=row['parameters'], group=row['group'], L=row['L'], nodes=m.N, objective=objective, published=pub, patch=patch, reference_angle_snapping='theta0>pi-1e-6 snapped to pi in constructor', spectral_search='dense lowest algebraic eigensolve for <=324 DOF, otherwise ARPACK SA, residual checks and two starts at critical states', eigen_calls=e.sp.calls, max_eigen_residual=e.sp.max_residual, seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu, status='verified')
    dump(OUT / 'results' / f'{stem}.json', out)
    print(stem, f'obj={first:.9f}', cls, f"pub={pub['first_pressure_GPa']:.9f}", pub['first_class'], f"{out['seconds']:.1f}s", flush=True)
    return out

def jobs():
    return sorted(CONVENTIONS_SEED['rows'], key=lambda r: (0 if r['group'] == 'approximant' and r['L'] == 1 else 1 if r['group'] == 'random' else 2 if r['group'] == 'approximant' else 3, r['L'], r['name'], r['parameters']))
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int)
    ap.add_argument('--name')
    ap.add_argument('--kind', default='soft')
    ap.add_argument('--retry', action='store_true')
    args = ap.parse_args()
    rows = jobs()
    if args.name:
        rows = [r for r in rows if r['name'] == args.name and r['parameters'] == args.kind]
    elif args.shard is not None:
        rows = [r for i, r in enumerate(rows) if i % 4 == args.shard]
    if not rows:
        ap.error('No matching production case')
    failures = []
    for row in rows:
        stem = row['name'] + '_' + row['parameters']
        dest = OUT / 'results' / f'{stem}.json'
        err = OUT / 'results' / f'{stem}.error.json'
        if dest.exists():
            continue
        if err.exists() and (not args.retry):
            failures.append(stem)
            continue
        try:
            verify(row)
        except Exception as exc:
            dump(err, dict(name=stem, error=str(exc), traceback=traceback.format_exc(), status='failed'))
            print(stem, 'ERROR', repr(exc), flush=True)
            failures.append(stem)
    if failures:
        raise SystemExit('Failed cases: ' + ', '.join(failures))
