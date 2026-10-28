"""baseline: genuinely connected 3D rigid-beam decorated cubic ensemble.
Binary Truchet pairing in xy at each z, all seven site types joined in z.
All six translations/rotations are retained; not decoupled planar sheets.
Initial-geometry axial-prestress model, conservative +pV macro interpretation.
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
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize
from computations.common.bloch_frame import beam_matrix, geometric_matrix
from computations.truchet_2d.frame import ROOT, dump
from computations.common.acoustic_atlas import sym_basis, volume_hessian, restricted, first_root

def geometry(p, stub=0.2):
    shape = np.array(p.shape)
    offsets = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 1 - stub, 0], [1 - stub, 0.5, 0], [0.5, stub, 0], [stub, 0.5, 0]])
    cells = np.array(list(np.ndindex(*p.shape)))
    pos = (cells[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    edges = []

    def idx(c, h):
        return int(np.ravel_multi_index(tuple(c % shape), tuple(shape)) * 7 + h)
    for c in cells:
        own = int(np.ravel_multi_index(tuple(c), tuple(shape)))

        def put(h, d, j, category):
            dest = c + np.array(d)
            edges.append((idx(c, h), idx(dest, j), dest // shape, offsets[j] + d - offsets[h], own, category))
        put(0, [0, 0, 0], 1, 0)
        put(1, [1, 0, 0], 0, 0)
        put(0, [0, 0, 0], 2, 0)
        put(2, [0, 1, 0], 0, 0)
        put(3, [0, 1, 0], 1, 1)
        put(4, [1, 0, 0], 2, 1)
        put(5, [0, 0, 0], 1, 1)
        put(6, [0, 0, 0], 2, 1)
        for a, b in [(3, 4), (5, 6)] if p[tuple(c)] == 0 else [(3, 6), (5, 4)]:
            put(a, [0, 0, 0], b, 2)
        for h in range(7):
            put(h, [0, 0, 1], h, 3)
    return (pos, edges)

def eigpair(K, H, count=2, factor=None):
    count = min(count, K.shape[0] - 1)
    if K.shape[0] < 450:
        scale = 1 / np.sqrt(K.diagonal().real)
        ev, v = la.eigh(H.toarray() * scale[:, None] * scale[None, :], K.toarray() * scale[:, None] * scale[None, :], subset_by_index=[len(scale) - count, len(scale) - 1], check_finite=False)
        v = scale[:, None] * v
    else:
        fac = factor if factor is not None else sla.splu(K, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
        inv = sla.LinearOperator(K.shape, matvec=fac.solve, dtype=K.dtype)
        ev, v = sla.eigsh(H, k=count, M=K, Minv=inv, which='LA', tol=2e-08, v0=np.random.default_rng(932).normal(size=K.shape[0]), maxiter=8000)
    order = np.argsort(ev)[::-1]
    ev = ev[order]
    v = v[:, order]
    good = ev > 1e-12
    vals = 1 / ev[good]
    v = v[:, good]
    residual = []
    for j in range(len(vals)):
        kv = K @ v[:, j]
        hv = H @ v[:, j]
        residual.append(float(la.norm(kv - vals[j] * hv) / (la.norm(kv) + abs(vals[j]) * la.norm(hv))))
    return (vals, v, max(residual, default=0.0))

class Frame3D:

    def __init__(self, p, radius=0.01, zfactor=1.0, div=1, stub=0.2):
        self.pattern = np.asarray(p, dtype=np.uint8)
        self.shape = np.array(p.shape)
        self.volume = int(np.prod(self.shape))
        self.radius = radius
        self.zfactor = zfactor
        self.div = div
        self.stub = stub
        pos, edges = geometry(self.pattern, stub)
        self.reference_pos = pos.copy()
        self.edges = edges
        nodes = pos.tolist()
        elements = []
        cache = {}
        for parent, (i, j, sh, delta, own, cat) in enumerate(edges):
            ids = [i]
            for h in range(1, div):
                ids.append(len(nodes))
                nodes.append((pos[i] + delta * h / div).tolist())
            ids.append(j)
            rr = radius * (zfactor if cat == 3 else 1.0)
            dd = np.array(delta) / div
            key = tuple(np.round(dd, 10)) + (rr,)
            if key not in cache:
                cache[key] = (beam_matrix(np.zeros(3), dd, rr), geometric_matrix(np.zeros(3), dd))
            km, gm = cache[key]
            for h in range(div):
                elements.append((ids[h], ids[h + 1], sh if h == div - 1 else np.zeros(3, int), dd, km, gm, parent, own, np.pi * rr * rr))
        self.pos = np.array(nodes)
        self.n = len(nodes)
        self.ndof = 6 * self.n
        self.dofs = np.array([np.r_[6 * e[0] + np.arange(6), 6 * e[1] + np.arange(6)] for e in elements])
        self.shifts = np.array([e[2] for e in elements])
        self.delta = np.array([e[3] for e in elements])
        self.Ke = np.array([e[4] for e in elements])
        self.Ge = np.array([e[5] for e in elements])
        self.parents = np.array([e[6] for e in elements])
        self.owners = np.array([e[7] for e in elements])
        self.areas = np.array([e[8] for e in elements])
        self.rows = np.repeat(self.dofs, 12, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 12)).ravel()
        self.K = self.assemble(self.Ke)
        self.free = np.arange(3, self.ndof)
        self.Kf = self.K[self.free, :][:, self.free].tocsc()
        self.factor = sla.splu(self.Kf, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
        self.basis = sym_basis(3).T.reshape(6, 3, 3)
        self.affine = np.zeros((6, len(elements), 12))
        self.affine[:, :, 6:9] = np.einsum('aij,ej->aei', self.basis, self.delta)
        f = np.array([self.scatter(np.einsum('eij,ej->ei', self.Ke, d)) for d in self.affine]).T
        self.C = (np.einsum('aei,eij,bej->ab', self.affine, self.Ke, self.affine) - f[self.free].T @ self.factor.solve(f[self.free])) / self.volume
        ep = la.solve(self.C, np.array([-1.0, -1.0, -1.0, 0, 0, 0]))
        ep /= -ep[:3].sum() / 3
        self.pre = self.relaxed(np.einsum('a,aei->ei', ep, self.affine))
        self.pre['strain'] = np.einsum('a,aij->ij', ep, self.basis)
        self.p0 = float(-np.trace(self.pre['stress']) / 3)
        self.He = -self.pre['forces'][:, None, None] * self.Ge
        self.H = self.assemble(self.He)
        self.Hf = self.H[self.free, :][:, self.free].tocsc()

    def assemble(self, em, k=None):
        if k is not None and la.norm(k) > 0:
            em = em.astype(complex).copy()
            ph = np.exp(2j * np.pi * (self.shifts @ k))
            em[:, :6, 6:] *= ph[:, None, None]
            em[:, 6:, :6] *= ph.conj()[:, None, None]
        out = sp.coo_matrix((em.ravel(), (self.rows, self.cols)), shape=(self.ndof, self.ndof)).tocsr()
        out.eliminate_zeros()
        return out

    def scatter(self, f):
        out = np.zeros(self.ndof)
        np.add.at(out, self.dofs.ravel(), f.ravel())
        return out

    def relaxed(self, d):
        f = self.scatter(np.einsum('eij,ej->ei', self.Ke, d))
        w = np.zeros(self.ndof)
        w[self.free] = -self.factor.solve(f[self.free])
        actual = d + w[self.dofs]
        ef = np.einsum('eij,ej->ei', self.Ke, actual)
        length = la.norm(self.delta, axis=1)
        n = self.delta / length[:, None]
        forces = self.areas / length * np.sum(n * (actual[:, 6:9] - actual[:, :3]), axis=1)
        virial = np.einsum('ei,ej->eij', ef[:, 6:9], self.delta)
        virial = (virial + virial.transpose(0, 2, 1)) / 2
        tile = np.zeros((self.volume, 3, 3))
        np.add.at(tile, self.owners, virial)
        res = la.norm(self.K @ w + f) / max(abs(forces).max(), 1e-30)
        return dict(w=w, forces=forces, stress=tile.sum(axis=0) / self.volume, tile_stress=tile, residual=float(res), max_nonaffine=float(abs(w).max()))

    def eigen(self, k=(0, 0, 0), count=2):
        k = np.asarray(k, float)
        gamma = la.norm(k - np.rint(k)) < 1e-12
        free = self.free if gamma else np.arange(self.ndof)
        K = self.Kf if gamma else self.assemble(self.Ke, k).tocsc()
        H = self.Hf if gamma else self.assemble(self.He, k).tocsc()
        vals, v, res = eigpair(K, H, count, self.factor if gamma else None)
        full = np.zeros((self.ndof, len(vals)), dtype=v.dtype)
        full[free] = v
        return (vals, full, res)

    def stats(self, v, k=(0, 0, 0)):
        ue = v[self.dofs].astype(complex)
        ue[:, 6:] *= np.exp(2j * np.pi * (self.shifts @ np.asarray(k)))[:, None]
        ene = np.einsum('ei,eij,ej->e', ue.conj(), self.Ke, ue).real
        tile = np.bincount(self.owners, weights=ene, minlength=self.volume)
        tile = np.maximum(tile, 0)
        tile /= tile.sum()
        ipr = float(tile @ tile)
        return dict(energy_ipr=ipr, participating_cells=1 / ipr, participation_fraction=1 / (self.volume * ipr), tile_energy=tile.tolist())

    def affine_matrices(self):
        aff = np.zeros((9, len(self.Ke), 12))
        aff[:, :, 6:9] = np.einsum('aij,ej->aei', np.eye(9).reshape(9, 3, 3), self.delta)
        out = []
        for em, M in [(self.Ke, self.Kf), (self.He, self.Hf)]:
            cross = np.array([self.scatter(np.einsum('eij,ej->ei', em, d)) for d in aff]).T[self.free]
            macro = np.einsum('aei,eij,bej->ab', aff, em, aff)
            out.append(np.block([[M.toarray(), cross], [cross.T, macro]]))
        return out

    def thresholds(self):
        K, H = self.affine_matrices()
        HP = H.copy()
        HP[-9:, -9:] -= self.p0 * self.volume * volume_hessian(3)
        S = sym_basis(3)
        pd, v = first_root(*restricted(K, HP, S), vector=True)
        E = (S @ v[-6:]).reshape(3, 3)
        gamma = float(self.eigen(count=1)[0][0])
        dirs = [np.array(x, float) / la.norm(x) for x in itertools.product([-1, 0, 1], repeat=3) if any(x)]

        def acoustic(n):
            n = n / la.norm(n)
            return first_root(*restricted(K, H, np.kron(np.eye(3), n[:, None])))
        le, n = min([(acoustic(n), n) for n in dirs], key=lambda x: x[0])
        opt = minimize(lambda x: acoustic(x) / le, n, method='Nelder-Mead', options={'maxiter': 80, 'xatol': 1e-05, 'fatol': 1e-08})
        if opt.fun < 1:
            le = float(opt.fun * le)
            n = opt.x / la.norm(opt.x)
        nf = len(self.free)
        probe = min(pd, le, gamma) * 0.99
        M = K - probe * HP
        C = (M[nf:, nf:] - M[nf:, :nf] @ la.solve(M[:nf, :nf], M[:nf, nf:], assume_a='sym')) / self.volume
        gv = self.eigen(count=1)[1][self.free, 0]
        coupling = la.norm((K[nf:, :nf] - gamma * H[nf:, :nf]) @ gv)
        return dict(PD=float(pd), LE=float(le), gamma=gamma, PD_censored=bool(pd >= gamma * (1 - 2e-06)), LE_censored=bool(le >= gamma * (1 - 2e-06)), LE_direction=n.tolist(), PD_volumetric_fraction=float(np.trace(E) ** 2 / (3 * np.sum(E * E))) if la.norm(E) > 1e-12 else None, relaxed_pressure_eigenvalues_at_099_first=la.eigvalsh(S.T @ C @ S).tolist(), gamma_strain_coupling_norm=float(coupling))

    def compact(self, lo, hi):
        keep = np.all((self.pos > np.asarray(lo) + 1e-09) & (self.pos < np.asarray(hi) - 1e-09), axis=1)
        ids = (6 * np.flatnonzero(keep)[:, None] + np.arange(6)).ravel()
        if len(ids) < 2:
            return None
        vals, v, res = eigpair(self.K[ids, :][:, ids].tocsc(), self.H[ids, :][:, ids].tocsc(), 1)
        return dict(pressure=float(vals[0] * self.p0), load=float(vals[0]), residual=res, dofs=len(ids))

def reduce_period(p):
    p = p.copy()
    for axis in range(3):
        if p.shape[axis] == 2 and np.array_equal(np.take(p, 0, axis=axis), np.take(p, 1, axis=axis)):
            p = np.take(p, [0], axis=axis)
    return p

def canonical_cube(p):
    keys = []
    for r in range(4):
        a = np.rot90(p, r, axes=(0, 1)) ^ r % 2
        for refl in [0, 1]:
            b = np.flip(a, 0) ^ 1 if refl else a
            for rz in [0, 1]:
                c = np.flip(b, 2) if rz else b
                for shift in itertools.product([0, 1], repeat=3):
                    keys.append(np.roll(c, shift, axis=(0, 1, 2)).tobytes())
    return min(keys)

def scan(p, zfactor=1.0, div=1, grid=4, refine=False):
    start = time.perf_counter()
    f = Frame3D(p, zfactor=zfactor, div=div)
    points = list(itertools.product(np.arange(grid) / grid, repeat=3))
    rows = []
    for k in points:
        k = np.array(k)
        q = tuple(np.round(-k % 1, 10))
        if tuple(k) > q:
            continue
        rows.append((float(f.eigen(k, 1)[0][0]), k))
    rows.sort(key=lambda x: x[0])
    best = rows[0]
    if refine:
        for _, k in rows[:2]:
            if la.norm(k) < 1e-08:
                continue

            def obj(x):
                q = x - np.rint(x)
                if la.norm(q) < 0.01:
                    return 1000.0
                return float(f.eigen(q, 1)[0][0]) / best[0]
            opt = minimize(obj, k, method='Nelder-Mead', options={'maxiter': 90, 'xatol': 0.0001, 'fatol': 1e-07})
            if opt.fun < 1:
                best = (float(opt.fun * best[0]), opt.x % 1)
    th = f.thresholds()
    first = min(best[0], th['PD'], th['LE'])
    if not th['PD_censored'] and th['PD'] < min(th['LE'], best[0]) * 0.997:
        cls = 'PD-only'
    elif not th['LE_censored'] and th['LE'] <= best[0] * 1.003:
        cls = 'LE'
    else:
        cls = 'Gamma-optical' if la.norm(best[1] - np.rint(best[1])) < 1e-08 else 'finite-k'
    vals, v, res = f.eigen(best[1], 2)
    st = f.stats(v[:, 0], best[1])
    return dict(pattern=p.tolist(), shape=list(p.shape), zfactor=zfactor, radius=f.radius, div=div, grid=grid, critical_pressure=float(first * f.p0), bloch_pressure=float(best[0] * f.p0), k=best[1].tolist(), class_first=cls, thresholds=th, pressure_per_factor=f.p0, preload_strain=f.pre['strain'].tolist(), preload_stress=f.pre['stress'].tolist(), equilibrium_residual=f.pre['residual'], eigen_residual=res, seconds=time.perf_counter() - start, participation_fraction=st['participation_fraction'])

def local_star_check():
    sig = []
    for p in [np.zeros((2, 2, 2), int), np.indices((2, 2, 2)).sum(axis=0) % 2, np.random.default_rng(6).integers(0, 2, (2, 2, 2))]:
        pos, edges = geometry(p)
        stars = [[] for _ in pos]
        for i, j, sh, d, own, cat in edges:
            stars[i].append(d)
            stars[j].append(-d)
        counts = {}
        cache = {}
        for rays in stars:
            v = np.rint(np.array(rays) * 10).astype(int)
            g = v @ v.T
            raw = tuple(g.ravel())
            if raw not in cache:
                cache[raw] = min((tuple(g[np.ix_(q, q)].ravel()) for q in itertools.permutations(range(len(v)))))
            key = str(cache[raw])
            counts[key] = counts.get(key, 0) + 1
        sig.append(counts)
    return dict(identical_star_multisets=sig[0] == sig[1] == sig[2], signatures=sig, entropy_per_cube=float(np.log(2)), nodes_per_cube=7, edges_per_cube=17, full_3D_dofs_per_cube=42)

def periodic(zfactor=1.0):
    orbits = {}
    for bits in itertools.product([0, 1], repeat=8):
        p = np.array(bits, dtype=np.uint8).reshape(2, 2, 2)
        key = canonical_cube(p)
        if key not in orbits:
            orbits[key] = [p, 0]
        orbits[key][1] += 1
    out = dict(model='3D extruded decorated Truchet, full six-DOF rigid beams, IID cube states', zfactor=zfactor, raw_patterns=256, orbit_count=len(orbits), local_star_check=local_star_check(), runs=[])
    for i, (p, mult) in enumerate(orbits.values()):
        row = scan(reduce_period(p), zfactor=zfactor)
        row['orbit_size'] = mult
        out['runs'].append(row)
        dump(ROOT / 'frame_3d/inputs/axial_periodic.json', out)
        print('periodic', i + 1, len(orbits), row['shape'], row['critical_pressure'], row['class_first'], round(row['seconds'], 2), flush=True)
    out['weakest'] = min(out['runs'], key=lambda r: r['critical_pressure'])
    dump(ROOT / 'frame_3d/inputs/axial_periodic.json', out)

def sample(p, base, save_mode=False):
    start = time.perf_counter()
    f = Frame3D(p, zfactor=base['zfactor'])
    vals, v, res = f.eigen(count=2)
    stats = f.stats(v[:, 0])
    row = dict(L=p.shape[0], pattern=p.tolist(), critical_pressure=float(vals[0] * f.p0), ratio=float(vals[0] * f.p0 / base['critical_pressure']), critical_load=float(vals[0]), pressure_per_factor=f.p0, energy_ipr=stats['energy_ipr'], participating_cells=stats['participating_cells'], participation_fraction=stats['participation_fraction'], eigen_residual=res, equilibrium_residual=f.pre['residual'], seconds=time.perf_counter() - start, ndof=f.ndof)
    if save_mode:
        row['tile_energy'] = stats['tile_energy']
    return (row, f, stats)

def samples(sizes=(2, 4, 6, 8), count=10):
    per = json.loads((ROOT / 'frame_3d/inputs/axial_periodic.json').read_text())
    base = per['weakest']
    path = ROOT / 'frame_3d/inputs/axial_samples.json'
    out = json.loads(path.read_text()) if path.exists() else dict(base=base, runs=[])
    done = {(r['L'], r['sample']) for r in out['runs']}
    for L in sizes:
        for j in range(count):
            if (L, j) in done:
                continue
            p = np.random.default_rng([20261026, 6, L, j]).integers(0, 2, (L, L, L))
            row, f, st = sample(p, base, save_mode=j == 0)
            row['sample'] = j
            out['runs'].append(row)
            dump(path, out)
            print('sample3d', L, j, row['ratio'], row['participation_fraction'], round(row['seconds'], 2), flush=True)
            del f

def patches(sizes=(6, 8, 10)):
    per = json.loads((ROOT / 'frame_3d/inputs/axial_periodic.json').read_text())
    base = per['weakest']
    motif = np.array(base['pattern'])
    pure = Frame3D(motif, zfactor=base['zfactor'])
    ref = pure.pre['forces'].reshape(*motif.shape, 17) / pure.p0
    path = ROOT / 'frame_3d/inputs/axial_patches.json'
    out = json.loads(path.read_text()) if path.exists() else dict(base=base, runs=[])
    done = {(r['L'], r['R']) for r in out['runs']}
    for L in sizes:
        for R in [0] + list(range(2, L - 1)):
            if (L, R) in done:
                continue
            p = np.random.default_rng(441 + L).integers(0, 2, (L, L, L))
            lo = (L - R) // 2
            hi = lo + R
            if R:
                inds = np.indices((R, R, R))
                p[lo:hi, lo:hi, lo:hi] = motif[tuple((inds[d] % motif.shape[d] for d in range(3)))]
            row, f, st = sample(p, base, save_mode=True)
            row['R'] = R
            mask = np.zeros((L, L, L), bool)
            mask[lo:hi, lo:hi, lo:hi] = True
            pressure = -np.trace(f.pre['tile_stress'], axis1=1, axis2=2).reshape(L, L, L) / 3
            if R:
                pin = float(pressure[mask].mean())
                pout = float(pressure[~mask].mean())
                row.update(patch_energy_fraction=float(np.array(st['tile_energy']).reshape(L, L, L)[mask].sum()), patch_mean_pressure_per_factor=pin, matrix_mean_pressure_per_factor=pout, s_P_far=pin / pout, s_P_global=pin / f.p0, patch_mean_stress=f.pre['tile_stress'].reshape(L, L, L, 3, 3)[mask].mean(axis=0).tolist())
                support = f.compact([lo] * 3, [hi] * 3)
                if support:
                    support['ratio'] = support['pressure'] / base['critical_pressure']
                row['compact'] = support
                a = []
                b = []
                forces = f.pre['forces'].reshape(L, L, L, 17) / f.p0
                margin = max(0, R // 4)
                for c in itertools.product(range(lo + margin, hi - margin), repeat=3):
                    a.extend(forces[c])
                    b.extend(ref[tuple(((c[d] - lo) % motif.shape[d] for d in range(3)))])
                a = np.array(a)
                b = np.array(b)
                beta = float(a @ b / (b @ b))
                row['core_force_scale'] = beta
                row['core_force_shape_residual'] = float(la.norm(a - beta * b) / la.norm(a))
            out['runs'].append(row)
            dump(path, out)
            print('patch3d', L, R, row['ratio'], row.get('s_P_far'), row.get('compact', {}), flush=True)
            del f
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['benchmark', 'periodic', 'samples', 'patches'])
    ap.add_argument('--zfactor', type=float, default=1.0)
    ap.add_argument('--sizes', nargs='+', type=int)
    ap.add_argument('--count', type=int, default=10)
    a = ap.parse_args()
    if a.action == 'benchmark':
        out = []
        for z in [0.5, 1.0, 2.0]:
            for tag, p in [('uniform', np.zeros((1, 1, 1), int)), ('checkerxy', np.indices((2, 2, 1))[:2].sum(axis=0) % 2), ('checkerxyz', np.indices((2, 2, 2)).sum(axis=0) % 2)]:
                r = scan(p, zfactor=z, grid=2)
                r['tag'] = tag
                out.append(r)
                dump(ROOT / 'frame_3d/inputs/axial_benchmark.json', out)
                print('benchmark', z, tag, r['critical_pressure'], r['class_first'], r['k'], flush=True)
    elif a.action == 'periodic':
        periodic(a.zfactor)
    elif a.action == 'samples':
        samples(a.sizes or [2, 4, 6, 8], a.count)
    else:
        patches(a.sizes or [6, 8, 10])
