"""Offline baseline probe. Sparse 2-D adaptation of bloch_frame beam machinery.

The suggested one-diagonal/two-endpoint constraint has ONLY stripe states.
The main ensemble instead uses independent, congruent two-chord Truchet tiles
inside a fixed square skeleton. Fixed radial port stubs make every local node
star congruent (allowing reflection) across all configurations.
All lengths and E=1 are nondimensional. Axial-force beam-column tangent only.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize
from computations.common.bloch_frame import beam_matrix, geometric_matrix, column_check
ROOT = _DATA_ROOT
PLANAR = [0, 1, 5, 6, 7, 11]

def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix('.writing')
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    for attempt in range(12):
        try:
            tmp.write_text(payload, encoding='utf-8')
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == 11:
                raise
            time.sleep(0.1 * (attempt + 1))

def geometry(pattern, kind='truchet', stub=0.2):
    p = np.asarray(pattern, dtype=int)
    nx, ny = p.shape
    offsets = np.array([[0, 0], [0.5, 0], [0, 0.5], [0.5, 1 - stub], [1 - stub, 0.5], [0.5, stub], [stub, 0.5]]) if kind == 'truchet' else np.array([[0.0, 0.0]])
    ncell = len(offsets)
    pos = np.array([np.array([x, y]) + o for x in range(nx) for y in range(ny) for o in offsets])
    edges = []

    def idx(x, y, h):
        return (x % nx * ny + y % ny) * ncell + h

    def put(x, y, h, xx, yy, hh):
        i, j = (idx(x, y, h), idx(xx, yy, hh))
        shift = np.array([xx // nx - x // nx, yy // ny - y // ny])
        delta = np.array([xx - x, yy - y]) + offsets[hh] - offsets[h]
        edges.append((i, j, shift, delta))
    for x in range(nx):
        for y in range(ny):
            if kind == 'truchet':
                put(x, y, 0, x, y, 1)
                put(x, y, 1, x + 1, y, 0)
                put(x, y, 0, x, y, 2)
                put(x, y, 2, x, y + 1, 0)
                put(x, y, 3, x, y + 1, 1)
                put(x, y, 4, x + 1, y, 2)
                put(x, y, 5, x, y, 1)
                put(x, y, 6, x, y, 2)
                for a, b in [(3, 4), (5, 6)] if p[x, y] == 0 else [(3, 6), (5, 4)]:
                    put(x, y, a, x, y, b)
            else:
                put(x, y, 0, x + 1, y, 0)
                put(x, y, 0, x, y + 1, 0)
                if p[x, y] == 0:
                    put(x, y, 0, x + 1, y + 1, 0)
                else:
                    put(x + 1, y, 0, x, y + 1, 0)
    return (pos, edges)

class Frame2D:

    def __init__(self, pattern, radius=0.01, div=1, kind='truchet', stub=0.2, radius_factors=None):
        self.pattern = np.asarray(pattern, dtype=int)
        self.shape = np.array(self.pattern.shape)
        self.radius = radius
        self.div = div
        self.kind = kind
        pos, edges = geometry(self.pattern, kind, stub=stub)
        radius_factors = np.ones(3) if radius_factors is None else np.asarray(radius_factors)
        self.reference_pos = pos.copy()
        self.edges = edges
        self.reference_n = len(pos)
        nodes = pos.tolist()
        elements = []
        cache = {}
        for parent, (i, j, shift, delta) in enumerate(edges):
            ids = [i]
            for h in range(1, div):
                ids.append(len(nodes))
                nodes.append((pos[i] + h / div * delta).tolist())
            ids.append(j)
            category = (0 if parent % 10 < 4 else 1 if parent % 10 < 8 else 2) if kind == 'truchet' else 0
            eradius = radius * radius_factors[category]
            d = delta / div
            key = tuple(np.round(d, 10)) + (float(eradius),)
            if key not in cache:
                a = np.zeros(3)
                b = np.r_[d, 0.0]
                cache[key] = (beam_matrix(a, b, eradius)[np.ix_(PLANAR, PLANAR)], geometric_matrix(a, b)[np.ix_(PLANAR, PLANAR)])
            km, gm = cache[key]
            for h in range(div):
                elements.append((ids[h], ids[h + 1], shift if h == div - 1 else np.zeros(2, int), d, km, gm, parent))
        self.pos = np.array(nodes)
        self.n = len(nodes)
        self.ndof = 3 * self.n
        self.dofs = np.array([np.r_[3 * i + np.arange(3), 3 * j + np.arange(3)] for i, j, *_ in elements])
        self.shifts = np.array([e[2] for e in elements])
        self.delta = np.array([e[3] for e in elements])
        self.Ke = np.array([e[4] for e in elements])
        self.Ge = np.array([e[5] for e in elements])
        self.parents = np.array([e[6] for e in elements])
        category = np.where(self.parents % 10 < 4, 0, np.where(self.parents % 10 < 8, 1, 2)) if kind == 'truchet' else np.zeros(len(elements), int)
        self.element_areas = np.pi * (radius * radius_factors[category]) ** 2
        self.rows = np.repeat(self.dofs, 6, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 6)).ravel()
        self.K = self.assemble(self.Ke)
        self.free = np.arange(2, self.ndof)
        self.Kf = self.K[self.free, :][:, self.free].tocsc()
        self.factor = sla.splu(self.Kf)
        self.affine = []
        for E in [np.array([[1.0, 0], [0, 0]]), np.array([[0, 0], [0, 1.0]]), np.array([[0, 1], [1, 0.0]]) / np.sqrt(2)]:
            d = np.zeros((len(elements), 6))
            d[:, 3:5] = self.delta @ E.T
            self.affine.append(d)
        self.affine = np.array(self.affine)
        self.pre = self.relaxed(-self.affine[0] - self.affine[1])
        self.He = -self.pre['forces'][:, None, None] * self.Ge
        self.H = self.assemble(self.He)
        self.Hf = self.H[self.free, :][:, self.free].tocsc()
        self.C = self.macro_matrix(self.Ke, self.factor)

    def assemble(self, em, k=None):
        if k is not None and la.norm(k) > 0:
            em = em.astype(complex).copy()
            ph = np.exp(2j * np.pi * (self.shifts @ np.asarray(k)))
            em[:, :3, 3:] *= ph[:, None, None]
            em[:, 3:, :3] *= ph.conj()[:, None, None]
        return sp.coo_matrix((em.ravel(), (self.rows, self.cols)), shape=(self.ndof, self.ndof)).tocsr()

    def scatter(self, f):
        out = np.zeros(self.ndof)
        np.add.at(out, self.dofs.ravel(), f.ravel())
        return out

    def relaxed(self, imposed):
        f = self.scatter(np.einsum('eij,ej->ei', self.Ke, imposed))
        w = np.zeros(self.ndof)
        w[self.free] = -self.factor.solve(f[self.free])
        d = imposed + w[self.dofs]
        n = self.delta / la.norm(self.delta, axis=1)[:, None]
        forces = self.element_areas / la.norm(self.delta, axis=1) * np.sum(n * (d[:, 3:5] - d[:, :2]), axis=1)
        ef = np.einsum('eij,ej->ei', self.Ke, d)
        stress = np.einsum('ei,ej->ij', ef[:, 3:5], self.delta) / np.prod(self.shape)
        residual = la.norm(self.K @ w + f) / max(abs(forces).max(), 1e-30)
        return dict(forces=forces, stress=(stress + stress.T) / 2, residual=residual, max_nonaffine=float(abs(w).max()), energy=float(np.sum(d * ef) / 2), w=w)

    def macro_matrix(self, em, factor=None):
        M = self.assemble(em)
        if factor is None:
            factor = sla.splu(M[self.free, :][:, self.free].tocsc())
        f = np.array([self.scatter(np.einsum('eij,ej->ei', em, d)) for d in self.affine]).T
        C = np.einsum('aei,eij,bej->ab', self.affine, em, self.affine)
        C -= f[self.free].T @ factor.solve(f[self.free])
        return C / np.prod(self.shape)

    def eigen(self, k=(0.0, 0.0), count=16, vectors=True):
        gamma = la.norm(k) < 1e-13
        free = self.free if gamma else np.arange(self.ndof)
        K = self.Kf if gamma else self.assemble(self.Ke, k).tocsc()
        H = self.Hf if gamma else self.assemble(self.He, k).tocsc()
        count = min(count, len(free) - 1)
        if len(free) < 550:
            scale = 1 / np.sqrt(K.diagonal().real)
            ev, v = la.eigh(H.toarray() * scale[:, None] * scale[None, :], K.toarray() * scale[:, None] * scale[None, :], subset_by_index=(len(free) - count, len(free) - 1), check_finite=False)
            v = scale[:, None] * v
        else:
            fac = self.factor if gamma else sla.splu(K)
            inv = sla.LinearOperator(K.shape, matvec=fac.solve, dtype=K.dtype)
            ev, v = sla.eigsh(H, k=count, M=K, Minv=inv, which='LA', tol=2e-09, v0=np.random.default_rng(932).normal(size=len(free)), maxiter=12000)
        order = np.argsort(ev)[::-1]
        ev = ev[order]
        v = v[:, order]
        positive = ev > 1e-12
        ev = ev[positive]
        v = v[:, positive]
        vals = 1 / ev
        residual = []
        for j in range(min(3, len(vals))):
            kv = K @ v[:, j]
            hv = H @ v[:, j]
            residual.append(float(la.norm(kv - vals[j] * hv) / (la.norm(kv) + abs(vals[j]) * la.norm(hv))))
        full = np.zeros((self.ndof, len(vals)), dtype=v.dtype)
        full[free] = v
        if gamma:
            full = full.reshape(self.n, 3, -1)
            full[:, :2] -= full[:, :2].mean(axis=0)
            full = full.reshape(self.ndof, -1)
        return (vals, full, max(residual, default=0.0))

    def inertia_below(self, load):
        """Count K-load H negative directions by symmetric unpivoted sparse LDL.
        SuperLU's symmetric row/column permutation and U signs supply inertia;
        reject any factorization requiring different row and column permutations.
        """
        if load <= 0:
            return 0
        A = (self.Kf - load * self.Hf).tocsc()
        fac = sla.splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
        if not np.array_equal(fac.perm_r, fac.perm_c):
            raise RuntimeError('Inertia requires matching symmetric permutations')
        return int(np.sum(fac.U.diagonal() < 0))

    def mode_stats(self, v, k=(0.0, 0.0)):
        u = v.reshape(self.n, 3).copy()
        if la.norm(k) < 1e-13:
            u[:, :2] -= u[:, :2].mean(axis=0)
        weights = np.sum(abs(u[:, :2]) ** 2, axis=1)
        ue = v[self.dofs].astype(complex, copy=True)
        ue[:, 3:] *= np.exp(2j * np.pi * (self.shifts @ np.asarray(k)))[:, None]
        ene = np.einsum('ei,eij,ej->e', ue.conj(), self.Ke, ue).real
        tile = np.bincount(self.parents // (10 if self.kind == 'truchet' else 3), weights=ene, minlength=int(np.prod(self.shape)))
        tile = np.maximum(tile, 0)
        tile /= tile.sum()
        w = weights[:self.reference_n]
        w /= w.sum()
        ipr = float(tile @ tile)
        return dict(energy_ipr=ipr, participating_tiles=1 / ipr, localization_length=np.sqrt(1 / ipr), energy_participation_fraction=1 / (ipr * np.prod(self.shape)), displacement_ipr=float(w @ w), tile_energy=tile.tolist())

def canonical(p):
    """Translation + D4 orbits. Tile pairing swaps under quarter rotation/reflection."""
    p = np.asarray(p, dtype=np.uint8)
    candidates = []
    for r in range(4):
        a = np.rot90(p, r) ^ r % 2
        for mirror in [False, True]:
            b = np.flip(a, 0) ^ 1 if mirror else a
            if b.shape != p.shape:
                continue
            for x in range(p.shape[0]):
                for y in range(p.shape[1]):
                    candidates.append(np.roll(np.roll(b, x, 0), y, 1).tobytes())
    return min(candidates)

def patterns(maxside=3):
    result = []
    for nx in range(1, maxside + 1):
        for ny in range(nx, maxside + 1):
            seen = {}
            for bits in itertools.product((0, 1), repeat=nx * ny):
                p = np.array(bits, dtype=np.uint8).reshape(nx, ny)
                key = canonical(p)
                if key not in seen:
                    seen[key] = [p, 0]
                seen[key][1] += 1
            for p, num in seen.values():
                result.append((p, num))
    return result

def scan_periodic(p, radius=0.01, div=1, grid=6, refine=True):
    start = time.perf_counter()
    f = Frame2D(p, radius, div)
    points = [np.array([x, y], float) / grid for x in range(grid) for y in range(grid)]
    points += [0.001 * np.array([np.cos(t), np.sin(t)]) for t in np.arange(16) * np.pi / 16]
    rows = []

    def evaluate(k):
        val, _, res = f.eigen(k, 4)
        return float(val[0])
    for k in points:
        rows.append((evaluate(k), k))
    rows.sort(key=lambda x: x[0])
    best = rows[0]
    if refine:
        for _, k in rows[:3]:
            if la.norm(k) < 0.01:
                continue
            opt = minimize(evaluate, k, method='Nelder-Mead', options={'xatol': 2e-06, 'fatol': 1e-10, 'maxiter': 100})
            if opt.fun < best[0]:
                best = (float(opt.fun), opt.x)
    vals, v, res = f.eigen(best[1], min(20, f.ndof - 3))
    stats = f.mode_stats(v[:, 0], best[1])
    return dict(pattern=p.tolist(), critical=float(vals[0]), k_supercell=(np.asarray(best[1]) % 1).tolist(), k_lattice=(np.asarray(best[1]) / f.shape % 1).tolist(), eigenvalues=vals.tolist(), mode_real=v[:, 0].real.tolist(), mode_imag=v[:, 0].imag.tolist(), mode_stats=stats, C=f.C.tolist(), prestress={k: v.tolist() if hasattr(v, 'tolist') else v for k, v in f.pre.items() if k != 'w'}, residual=res, seconds=time.perf_counter() - start, grid_min=float(rows[0][0]), grid_values=[r[0] for r in rows], ndof=f.ndof)

def sample_record(p, threshold, radius=0.01, div=1, count=28, full=False):
    start = time.perf_counter()
    f = Frame2D(p, radius, div)
    vals, v, res = f.eigen(count=count)
    stats = f.mode_stats(v[:, 0])
    load_fractions = [0, 0.5, 0.9, 0.99]
    macro = []
    for frac in load_fractions:
        load = frac * threshold
        if load >= vals[0] * (1 - 1e-08):
            macro.append(None)
        else:
            C = f.C if frac == 0 else f.macro_matrix(f.Ke - load * f.He)
            macro.append(dict(C=C.tolist(), bulk_dilation_modulus=float(np.array([1.0, 1.0, 0]) @ C @ np.array([1.0, 1.0, 0]) / 4), lowest_C_eigenvalue=float(la.eigvalsh(C)[0])))
    out = dict(shape=list(p.shape), pattern=p.tolist(), critical=float(vals[0]), eigenvalues=vals.tolist(), mode_stats=stats, below_periodic=f.inertia_below(threshold * (1 - 1e-07)), within_1pct_periodic=f.inertia_below(threshold * 1.01), within_5pct_periodic=f.inertia_below(threshold * 1.05), prestress_residual=f.pre['residual'], eigen_residual=res, prestress_force_range=[float(f.pre['forces'].min()), float(f.pre['forces'].max())], pressure_per_unit_strain=float(-np.trace(f.pre['stress']) / 2), mean_C=f.C.tolist(), macro_load_fractions=load_fractions, macro=macro, ndof=f.ndof, seconds=time.perf_counter() - start)
    if full:
        out.update(mode=v[:, 0].real.tolist(), positions=f.pos.tolist())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['benchmark', 'periodic', 'samples', 'verify'])
    ap.add_argument('--div', type=int, default=1)
    ap.add_argument('--radius', type=float, default=0.01)
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--sizes', nargs='+', type=int, default=[4, 8, 16, 32])
    ap.add_argument('--maxside', type=int, default=3)
    ap.add_argument('--grid', type=int, default=6)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    path = ROOT / (args.out or f'{args.action}.json')
    if args.action == 'benchmark':
        out = []
        for p in [np.zeros((1, 1), int), np.array([[0, 1]]), np.array([[0, 1], [1, 0]]), np.array([[0, 0], [0, 1]])]:
            r = scan_periodic(p, args.radius, args.div, args.grid)
            out.append(r)
            dump(path, out)
            print(p.tolist(), r['critical'], r['k_supercell'], r['mode_stats']['energy_participation_fraction'], r['seconds'], flush=True)
    elif args.action == 'periodic':
        out = dict(radius=args.radius, div=args.div, grid=args.grid, maxside=args.maxside, runs=[])
        pats = patterns(args.maxside)
        for j, (p, num) in enumerate(pats):
            r = scan_periodic(p, args.radius, args.div, args.grid)
            r['orbit_size'] = num
            out['runs'].append(r)
            dump(path, out)
            print(j + 1, len(pats), p.shape, r['critical'], r['k_supercell'], round(r['seconds'], 2), flush=True)
    elif args.action == 'samples':
        per = json.loads((ROOT / 'truchet_2d/inputs/periodic_patterns.json').read_text())
        threshold = min((r['critical'] for r in per['runs']))
        out = dict(seed=20261025, radius=args.radius, div=args.div, periodic_threshold=threshold, ensemble='IID uniform binary decorated Truchet tiles, exact independent draws', runs=[])
        for L in args.sizes:
            for j in range(args.samples):
                rng = np.random.default_rng(np.random.SeedSequence([20261025, L, j]))
                p = rng.integers(0, 2, (L, L))
                r = sample_record(p, threshold, args.radius, args.div, full=j == 0)
                r['sample'] = j
                out['runs'].append(r)
                dump(path, out)
                print(L, j, r['critical'], r['below_periodic'], r['mode_stats']['participating_tiles'], round(r['seconds'], 2), flush=True)
    elif args.action == 'verify':
        out = dict(column=column_check(), mesh=[])
        for div in [1, 2, 4, 8]:
            for p in [np.zeros((1, 1), int), np.array([[0, 1], [1, 0]])]:
                r = scan_periodic(p, args.radius, div, 4)
                r['div'] = div
                out['mesh'].append(r)
                dump(path, out)
                print('mesh', div, p.tolist(), r['critical'], r['k_supercell'], flush=True)
        rng = np.random.default_rng(31)
        p = rng.integers(0, 2, (4, 4))
        out['random_mesh'] = []
        for div in [1, 2, 4]:
            f = Frame2D(p, args.radius, div)
            vals, _, res = f.eigen(count=12)
            out['random_mesh'].append(dict(div=div, eigenvalues=vals.tolist(), residual=res))
            dump(path, out)
            print('random_mesh', div, vals[0], flush=True)
        f = Frame2D(np.array([[0, 1], [1, 0]]), args.radius, 1)
        K, H = (f.Kf.toarray(), f.Hf.toarray())
        ev = la.eigvalsh(H, K)
        out['inertia'] = [dict(load=float(t), dense=int(np.sum(1 - t * ev < 0)), sparse=f.inertia_below(t)) for t in np.linspace(0.001, 0.2, 11)]
        dump(path, out)
if __name__ == '__main__':
    main()
