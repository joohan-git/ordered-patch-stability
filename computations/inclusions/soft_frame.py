"""Net-agnostic full-six-DOF graded RCSR inclusion test, offline.
EDGE_CENTER graphs only; initial-geometry axial-force beam-column tangent.
Constant pressure +pV is distinct from fixed nominal dead traction.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, json, time, gc
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from computations.inclusions.crystal_reference import ROOT, graph_data, dump
from computations.common.bloch_frame import beam_matrix, geometric_matrix
from computations.common.acoustic_atlas import sym_basis, volume_hessian
from computations.frame_3d.frame import eigpair
from computations.common.condensed_factor import CondensedFactor

def process_memory():
    try:
        import psutil
        m = psutil.Process().memory_info()
        return dict(rss_bytes=m.rss, peak_working_set_bytes=getattr(m, 'peak_wset', m.rss))
    except Exception:
        return {}

class InclusionFrame:

    def __init__(self, name, L=6, R_fraction=0.22, rin=0.02, rout=0.03, div=2, uniform=None):
        start = time.perf_counter()
        self.name = name
        self.L = L
        self.div = div
        self.rin = rin
        self.rout = rout
        self.data = d = graph_data(name)
        self.A = d['A']
        self.cell_volume = abs(la.det(self.A))
        self.ncells = L ** 3
        self.volume = self.ncells * self.cell_volume
        self.n0 = len(d['frac'])
        self.m0 = len(d['edges'])
        heights = 1 / la.norm(la.inv(self.A), axis=0)
        self.R = R_fraction * L * heights.min()
        self.R_fraction = R_fraction
        self.heights = heights
        self.center = np.full(3, L / 2)
        cells = np.array(list(np.ndindex(L, L, L)))
        self.cells = cells
        self.center_convention = 'box center'
        if np.min(self.distance(cells + 0.5)) > 0.22 * L * heights.min():
            self.center += 0.5
            self.center_convention = 'central-cell center (anisotropic-cell fallback)'
        posfrac = (cells[:, None, :] + d['frac'][None, :, :]).reshape(-1, 3)
        pi = np.tile(np.array([e[0] for e in d['edges']]), self.ncells) + np.repeat(np.arange(self.ncells) * self.n0, self.m0)
        dest = cells[:, None, :] + np.array([e[2] for e in d['edges']])[None, :, :]
        target = np.ravel_multi_index(tuple(np.moveaxis(dest % L, -1, 0)), (L, L, L)).ravel()
        pj = target * self.n0 + np.tile(np.array([e[1] for e in d['edges']]), self.ncells)
        pshift = (dest // L).reshape(-1, 3)
        parentdelta = np.array([(d['frac'][j] + sh - d['frac'][i]) @ self.A for i, j, sh in d['edges']])
        df0 = parentdelta @ la.inv(self.A)
        pdf = np.tile(df0, (self.ncells, 1))
        pd = np.tile(parentdelta, (self.ncells, 1))
        midpoint = posfrac[pi] + pdf / 2
        self.parent_distance = self.distance(midpoint)
        sl = np.where(self.parent_distance <= self.R, rin, rout)
        if uniform is not None:
            sl[:] = uniform
        self.parent_slender = sl
        self.parent_lengths = la.norm(pd, axis=1)
        radius = sl * self.parent_lengths
        np0 = len(pi)
        ids = [pi]
        if div > 1:
            virtual = np.arange(len(posfrac), len(posfrac) + np0 * (div - 1)).reshape(np0, div - 1)
            extra = (posfrac[pi, None, :] + pdf[:, None, :] * np.arange(1, div)[None, :, None] / div).reshape(-1, 3)
            posfrac = np.vstack([posfrac, extra])
            ids.extend([virtual[:, h] for h in range(div - 1)])
        ids.append(pj)
        ids = np.array(ids).T
        ei = ids[:, :-1].ravel()
        ej = ids[:, 1:].ravel()
        self.posfrac = posfrac
        self.pos = posfrac @ self.A
        self.ndof = 6 * len(posfrac)
        self.free = np.arange(3, self.ndof)
        self.dofs = np.hstack([6 * ei[:, None] + np.arange(6), 6 * ej[:, None] + np.arange(6)])
        self.shifts = np.zeros((np0, div, 3), int)
        self.shifts[:, -1] = pshift
        self.shifts = self.shifts.reshape(-1, 3)
        self.delta = np.repeat(pd / div, div, axis=0)
        self.parents = np.repeat(np.arange(np0), div)
        self.owners = np.repeat(np.repeat(np.arange(self.ncells), self.m0), div)
        self.element_type = np.tile(np.arange(self.m0 * div), self.ncells)
        self.element_distance = self.parent_distance[self.parents]
        self.areas = np.repeat(np.pi * radius ** 2, div)
        self.rows = np.repeat(self.dofs, 12, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 12)).ravel()
        cache = {}
        kms = []
        gms = []
        for j in range(np0):
            typ = j % self.m0
            key = (typ, float(sl[j]))
            if key not in cache:
                cache[key] = (beam_matrix(np.zeros(3), pd[j] / div, radius[j]), geometric_matrix(np.zeros(3), pd[j] / div))
            km, gm = cache[key]
            kms.append(km)
            gms.append(gm)
        self.Ke = np.repeat(kms, div, axis=0)
        self.Ge = np.repeat(gms, div, axis=0)
        self.K = self.assemble(self.Ke)
        self.Kf = self.K[self.free, :][:, self.free].tocsc()
        self.assembly_seconds = time.perf_counter() - start
        tf = time.perf_counter()
        self.factor_method = os.environ.get('K08_INCLUSION_FACTOR', 'condensed')
        self.factor = CondensedFactor(self.Kf, 6 * self.n0 * self.ncells - 3, 6 * (div - 1)) if self.factor_method == 'condensed' else sla.splu(self.Kf, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
        self.factor_seconds = time.perf_counter() - tf
        self.factor_nnz = self.factor.L.nnz + self.factor.U.nnz
        self.S = sym_basis(3)
        self.basis = self.S.T.reshape(6, 3, 3)
        self.affine = np.zeros((6, len(ei), 12))
        self.affine[:, :, 6:9] = np.einsum('aij,ej->aei', self.basis, self.delta)
        self.BK = self.cross(self.Ke)
        self.Z = self.factor.solve(self.BK)
        self.MK = np.einsum('aei,eij,bej->ab', self.affine, self.Ke, self.affine)
        self.Ctotal = self.MK - self.BK.T @ self.Z
        self.Ctotal = (self.Ctotal + self.Ctotal.T) / 2
        self.C = self.Ctotal / self.volume
        self.cell_distance = self.distance(cells + 0.5)
        self.patch_cells = self.cell_distance <= self.R
        if not self.patch_cells.any():
            raise ValueError('no patch cells')
        self.far_cells = self.cell_distance > self.R + np.max(d['lengths'])
        if not self.far_cells.any():
            self.far_cells = ~self.patch_cells
        self.setup_seconds = time.perf_counter() - start
        self.setup_memory = process_memory()

    def distance(self, frac):
        df = (np.asarray(frac) - self.center + self.L / 2) % self.L - self.L / 2
        return la.norm(df @ self.A, axis=-1)

    def assemble(self, em, k=None):
        if k is not None and la.norm(k) > 0:
            em = em.astype(complex).copy()
            ph = np.exp(2j * np.pi * (self.shifts @ np.asarray(k)))
            em[:, :6, 6:] *= ph[:, None, None]
            em[:, 6:, :6] *= ph.conj()[:, None, None]
        M = sp.coo_matrix((em.ravel(), (self.rows, self.cols)), shape=(self.ndof, self.ndof)).tocsr()
        M.eliminate_zeros()
        return M

    def scatter(self, ef):
        out = np.zeros(self.ndof)
        np.add.at(out, self.dofs.ravel(), ef.ravel())
        return out

    def cross(self, em):
        return np.array([self.scatter(np.einsum('eij,ej->ei', em, d)) for d in self.affine]).T[self.free]

    def preload(self, protocol):
        start = time.perf_counter()
        ep = np.array([-1.0, -1.0, -1.0, 0, 0, 0])
        if protocol == 'pressure':
            ep = la.solve(self.C, ep, assume_a='pos')
            ep /= -ep[:3].sum() / 3
        imposed = np.einsum('a,aei->ei', ep, self.affine)
        rhs = self.scatter(np.einsum('eij,ej->ei', self.Ke, imposed))
        w = np.zeros(self.ndof)
        w[self.free] = -self.factor.solve(rhs[self.free])
        actual = imposed + w[self.dofs]
        ef = np.einsum('eij,ej->ei', self.Ke, actual)
        length = la.norm(self.delta, axis=1)
        normal = self.delta / length[:, None]
        forces = self.areas / length * np.sum(normal * (actual[:, 6:9] - actual[:, :3]), axis=1)
        virial = np.einsum('ei,ej->eij', ef[:, 6:9], self.delta)
        virial = (virial + virial.transpose(0, 2, 1)) / 2
        tiles = np.zeros((self.ncells, 3, 3))
        np.add.at(tiles, self.owners, virial)
        tiles /= self.cell_volume
        strain = np.einsum('a,aij->ij', ep, self.basis)
        self.pre = dict(strain=strain, stress=tiles.mean(axis=0), tile_stress=tiles, forces=forces, w=w, actual=actual, residual=float(la.norm(self.K @ w + rhs) / max(la.norm(rhs), abs(forces).max(), 1e-30)), seconds=time.perf_counter() - start)
        self.p0 = float(-np.trace(self.pre['stress']) / 3)
        self.protocol = protocol
        self.He = -forces[:, None, None] * self.Ge
        self.H = self.assemble(self.He)
        self.Hf = self.H[self.free, :][:, self.free].tocsc()
        return self.pre

    def gamma(self, count=2):
        vals, v, res = eigpair(self.Kf, self.Hf, count, self.factor)
        full = np.zeros((self.ndof, len(vals)))
        full[self.free] = v
        return (vals, full, res)

    def variable(self, count=2):
        BH = self.cross(self.He)
        MH = np.einsum('aei,eij,bej->ab', self.affine, self.He, self.affine)
        MH -= self.p0 * self.volume * (self.S.T @ volume_hessian(3) @ self.S)
        K = sp.bmat([[self.Kf, self.BK], [self.BK.T, self.MK]], format='csc')
        H = sp.bmat([[self.Hf, BH], [BH.T, MH]], format='csc')
        cf = la.cho_factor(self.Ctotal)

        def inverse(rhs):
            y = self.factor.solve(rhs[:-6])
            e = la.cho_solve(cf, rhs[-6:] - self.BK.T @ y)
            return np.r_[y - self.Z @ e, e]
        inv = sla.LinearOperator(K.shape, matvec=inverse, dtype=float)
        ev, v = sla.eigsh(H, k=count, M=K, Minv=inv, which='LA', tol=1e-08, maxiter=8000, v0=np.random.default_rng(77).normal(size=K.shape[0]))
        order = np.argsort(ev)[::-1]
        ev = ev[order]
        v = v[:, order]
        vals = 1 / ev
        res = max((float(la.norm(K @ v[:, j] - vals[j] * (H @ v[:, j])) / (la.norm(K @ v[:, j]) + abs(vals[j]) * la.norm(H @ v[:, j]))) for j in range(count)))
        u = np.zeros((self.ndof, count))
        u[self.free] = v[:-6]
        return (vals, u, v[-6:], res)

    def bloch(self, k):
        K = self.assemble(self.Ke, k).tocsc()
        H = self.assemble(self.He, k).tocsc()
        fac = CondensedFactor(K, 6 * self.n0 * self.ncells, 6 * (self.div - 1)) if self.factor_method == 'condensed' else None
        vals, v, res = eigpair(K, H, 1, fac)
        return (vals, v, res)

    def stats(self, u, eps=None, k=None):
        ue = u[self.dofs].astype(complex)
        if eps is not None:
            ue += np.einsum('a,aei->ei', eps, self.affine)
        if k is not None:
            ue[:, 6:] *= np.exp(2j * np.pi * (self.shifts @ k))[:, None]
        en = np.einsum('ei,eij,ej->e', ue.conj(), self.Ke, ue).real
        en = np.maximum(en, 0)
        en /= en.sum()
        tile = np.bincount(self.owners, weights=en, minlength=self.ncells)
        ipr = float(tile @ tile)
        core = self.element_distance < max(0.5 * self.R, self.R - np.max(self.data['lengths']))
        soft = self.element_distance <= self.R
        boundary = ~core & (self.element_distance <= self.R + np.max(self.data['lengths']))
        macro = 0.0
        if eps is not None:
            total = float(np.einsum('ei,eij,ej', ue.conj(), self.Ke, ue).real)
            macro = float(eps @ self.Ctotal @ eps / total)
        patchfrac = float(en[soft].sum())
        cf = float(en[core].sum())
        bf = float(en[boundary].sum())
        location = 'patch-interior' if cf > 0.5 else 'patch/boundary' if patchfrac > 0.5 or bf > 0.5 else 'extended/matrix'
        return dict(patch_energy_fraction=patchfrac, patch_cell_energy_fraction=float(tile[self.patch_cells].sum()), core_energy_fraction=cf, boundary_energy_fraction=bf, matrix_far_energy_fraction=float(en[~(core | boundary)].sum()), participation_fraction=float(1 / (self.ncells * ipr)), energy_ipr=ipr, macro_elastic_fraction=macro, location=location, tile_energy=tile.tolist())

    def concentration(self, ref):
        sigma = self.pre['tile_stress'][self.patch_cells].mean(axis=0)
        far = self.pre['tile_stress'][self.far_cells].mean(axis=0)
        pin = -np.trace(sigma) / 3
        pout = -np.trace(far) / 3
        scale = self.p0 if self.protocol == 'pressure' else 1.0
        refscale = ref['pressure_per_factor'] if self.protocol == 'pressure' else 1.0
        nr = np.asarray(ref['forces_per_factor'])[self.element_type] / refscale
        nc = self.pre['forces'] / scale
        core = self.element_distance < max(0.5 * self.R, self.R - np.max(self.data['lengths']))
        fitmask = core.copy()
        fit_region = 'core'
        if not fitmask.any() or nr[fitmask] @ nr[fitmask] < 1e-30:
            fitmask = self.element_distance <= self.R
            fit_region = 'whole-patch fallback'
        if not fitmask.any():
            fitmask = np.ones(len(nr), bool)
            fit_region = 'whole-box control fallback'
        fit = float(nc[fitmask] @ nr[fitmask] / (nr[fitmask] @ nr[fitmask]))
        error = float(la.norm(nc[fitmask] - fit * nr[fitmask]) / max(la.norm(nc[fitmask]), 1e-30))
        actual = self.pre['actual']
        du = actual[:, 6:9] - actual[:, :3]
        mask = self.element_distance <= self.R
        F = la.lstsq(self.delta[mask], du[mask])[0].T
        return dict(s_P_global=float(pin / self.p0), s_P_far=float(pin / pout), patch_pressure_per_factor=float(pin), far_pressure_per_factor=float(pout), patch_stress=sigma, far_stress=far, patch_stress_deviatoric_ratio=float(la.norm(sigma + pin * np.eye(3)) / max(np.sqrt(3) * abs(pin), 1e-30)), core_force_scale=fit, core_force_shape_residual=error, core_element_count=int(core.sum()), force_fit_region=fit_region, strain_amplification_fit=float(-np.trace(F) / 3), patch_average_gradient_fit=F)

    def compact(self):
        r = self.R - np.max(self.data['lengths'])
        ids = (6 * np.flatnonzero(self.distance(self.posfrac) < r)[:, None] + np.arange(6)).ravel()
        if r <= 0 or len(ids) < 12:
            return None
        vals, v, res = eigpair(self.K[ids, :][:, ids].tocsc(), self.H[ids, :][:, ids].tocsc(), 1)
        return dict(load=float(vals[0]), pressure=float(vals[0] * self.p0), support_radius=float(r), dofs=len(ids), residual=res)

def get_ref(name, protocol, slender=0.02):
    data = json.loads((ROOT / f'inclusions/axial_ref_{name}.json').read_text(encoding='utf-8'))
    return next((r for r in data['runs'] if r['protocol'] == protocol and r['slender'] == slender and (r['div'] == 2)))

def experiment(name, L, R_fraction, rin=0.02, rout=0.03, div=2, bloch=False, uniform=None):
    start = time.perf_counter()
    f = InclusionFrame(name, L, R_fraction, rin, rout, div, uniform)
    out = dict(name=name, L=L, R_fraction=R_fraction, R=f.R, rin=rin, rout=rout, div=div, uniform=uniform, graph='EDGE_CENTER', center_fractional=f.center, center_convention=f.center_convention, A=f.A, cell_heights=f.heights, cell_volume=f.cell_volume, volume=f.volume, conventional_nodes=f.n0, parent_edges_per_cell=f.m0, ndof=f.ndof, elements=len(f.Ke), patch_cells=int(f.patch_cells.sum()), patch_cell_volume_fraction=float(f.patch_cells.mean()), soft_parent_fraction=float(np.mean(f.parent_slender == rin)), assembly_seconds=f.assembly_seconds, factor_method=f.factor_method, factor_seconds=f.factor_seconds, factor_nnz=f.factor_nnz, setup_seconds=f.setup_seconds, setup_memory=f.setup_memory, protocols=[])
    for protocol in ['pressure', 'strain']:
        ts = time.perf_counter()
        ref = get_ref(name, protocol, rin)
        matrix = get_ref(name, protocol, rout)
        f.preload(protocol)
        con = f.concentration(ref)
        vg, u, resg = f.gamma(2)
        stats_fixed = f.stats(u[:, 0])
        first = float(vg[0])
        ucrit = u[:, 0]
        ec = None
        res = resg
        tests = []
        if protocol == 'pressure':
            vv, uv, ev, resv = f.variable(2)
            first = float(vv[0])
            ucrit = uv[:, 0]
            ec = ev[:, 0]
            res = resv
            tests.append(dict(kind='variable-cell', load=first, second_load=float(vv[1]), residual=resv))
        bc = []
        if bloch:
            for k in [np.array([0.5, 0, 0]), np.array([0, 0.5, 0]), np.array([0, 0, 0.5]), np.array([0.5, 0.5, 0.5])]:
                val, vec, er = f.bloch(k)
                bc.append(dict(k=k.tolist(), load=float(val[0]), residual=er))
        critical = f.stats(ucrit, ec)
        support = f.compact()
        unit = f.p0 if protocol == 'pressure' else 1.0
        ratioPD = first * unit * con['s_P_global'] / ref['lambda_PD'] if protocol == 'pressure' and ref['lambda_PD'] else None
        scale = con['s_P_global'] if protocol == 'pressure' else con['core_force_scale']
        row = dict(protocol=protocol, critical_load=first, lambda_c=first * unit, pressure_at_onset=first * f.p0, pressure_per_factor=f.p0, preload_strain=f.pre['strain'], preload_stress=f.pre['stress'], fixed_gamma_load=float(vg[0]), fixed_gamma_lambda=float(vg[0] * unit), variable_tests=tests, critical_mode=critical, fixed_gamma_mode=stats_fixed, concentration=con, compact=support, ratio_to_scaled_PD=ratioPD, ratio_to_scaled_loc=float(first * unit * scale / ref['lambda_loc']), ratio_to_matrix_first=float(first * unit / matrix['lambda_first']), reference_class=ref['class_first'], eigen_residual=res, gamma_residual=resg, preload_residual=f.pre['residual'], bloch_checks=bc, solve_seconds=time.perf_counter() - ts, memory=process_memory())
        out['protocols'].append(row)
        print('inclusion', name, L, R_fraction, protocol, row['lambda_c'], row['ratio_to_scaled_PD'], row['ratio_to_scaled_loc'], critical['patch_energy_fraction'], con['s_P_global'], con['core_force_shape_residual'], round(row['solve_seconds'], 2), flush=True)
    out['total_seconds'] = time.perf_counter() - start
    return out

def run(args):
    for name in args.nets:
        for rf in args.radii:
            label = f'{rf:.3f}'.replace('.', 'p')
            suffix = f'_uniform{args.uniform:g}' if args.uniform else ''
            path = ROOT / f'inclusions/axial_inc_{name}_L{args.L}_R{label}_d{args.div}{suffix}.json'
            if path.exists() and (not args.force):
                print('exists', path.name, flush=True)
                continue
            result = experiment(name, args.L, rf, args.rin, args.rout, args.div, args.bloch, args.uniform)
            dump(path, result)
            gc.collect()
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', required=True)
    ap.add_argument('--L', type=int, default=6)
    ap.add_argument('--radii', type=float, nargs='+', default=[0.22, 0.34])
    ap.add_argument('--rin', type=float, default=0.02)
    ap.add_argument('--rout', type=float, default=0.03)
    ap.add_argument('--div', type=int, default=2)
    ap.add_argument('--bloch', action='store_true')
    ap.add_argument('--uniform', type=float)
    ap.add_argument('--force', action='store_true')
    run(ap.parse_args())
