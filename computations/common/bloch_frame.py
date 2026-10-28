"""Periodic Euler--Bernoulli frame buckling. All writes stay beside this file.

E=1, nu=.3, circular uniform struts, mean reference edge length = 1.
Bloch convention: d(i,R)=exp(2 pi i k.R) d(i,0), fractional reciprocal k.
Beam-column geometric stiffness includes axial prestress only. It is NOT a
finite-deformation/postbuckling calculation or a certified continuum bound.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import gemmi
from computations.common.beam_element import beam_matrix
ROOT = _DATA_ROOT
LOADER = ROOT / 'common/rcsr_loader.py'
LD = {'__file__': str(LOADER)}
exec(compile(LOADER.read_text(encoding='utf-8'), str(LOADER), 'exec'), LD)
NETS = LD['read_cgd']() if Path(os.environ.get('RCSR_CGD', str(ROOT / 'rcsr3d_web.cgd'))).is_file() else []

def expanded(net, sg):
    pts, cns = ([], [])
    for cn, xyz in net['nodes']:
        clusters = []
        for op in sg.operations():
            p = np.array(op.apply_to_xyz(list(xyz))) % 1
            for cl in clusters:
                if la.norm((p - cl[0] + 0.5) % 1 - 0.5) < 0.0003:
                    cl.append(cl[0] + (p - cl[0] + 0.5) % 1 - 0.5)
                    break
            else:
                clusters.append([p])
        for cl in clusters:
            p = np.mean(cl, axis=0) % 1
            if not any((la.norm((p - q + 0.5) % 1 - 0.5) < 0.0003 for q in pts)):
                pts.append(p)
                cns.append(cn)
    return (np.array(pts), np.array(cns))

def load_net(name):
    net = next((n for n in NETS if n['name'] == name))
    default = gemmi.find_spacegroup_by_name(net['group'])
    choices = [default] + [s for s in gemmi.spacegroup_table() if s.number == default.number and s.xhm() != default.xhm()]
    trials = []
    for sg in choices:
        pts, cns = expanded(net, sg)
        edges, deg, A = LD['periodic_edges'](net, pts, cns)
        trials.append(dict(setting=sg.xhm(), nodes=len(pts), degrees=deg.tolist()))
        if np.array_equal(cns, deg):
            lengths = [la.norm((pts[j] + sh - pts[i]) @ A) for i, j, sh in edges]
            A = A / np.mean(lengths)
            return dict(name=name, net=net, sg=sg, frac=pts, cn=cns, edges=edges, A=A, trials=trials, lengths=(np.array(lengths) / np.mean(lengths)).tolist())
    raise ValueError(f'{name}: no coordination-consistent space group setting: {trials}')

def geometric_matrix(a, b):
    """Geometric stiffness per positive (tensile) axial force."""
    d = b - a
    L = la.norm(d)
    e = d / L
    ref = np.array([0.0, 0.0, 1.0]) if abs(e[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(ref, e)
    y /= la.norm(y)
    R = np.array([e, y, np.cross(e, y)])
    T = np.kron(np.eye(4), R)
    g = np.zeros((12, 12))
    block = np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L], [-36, -3 * L, 36, -3 * L], [3 * L, -L * L, -3 * L, 4 * L * L]]) / (30 * L)
    g[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = block
    sign = np.array([1, -1, 1, -1])
    g[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = block * sign[:, None] * sign[None, :]
    return T.T @ g @ T

class Frame:

    def __init__(self, name, radius=0.03, subdivisions=2, data=None):
        self.data = load_net(name) if data is None else data
        self.name = name
        self.A = self.data['A']
        self.radius = radius
        self.subdivisions = subdivisions
        frac = self.data['frac'].tolist()
        elements = []
        for edge_id, (i, j, sh) in enumerate(self.data['edges']):
            a = self.data['frac'][i]
            b = self.data['frac'][j] + sh
            ids = [i]
            for h in range(1, subdivisions):
                ids.append(len(frac))
                frac.append((a + (b - a) * h / subdivisions).tolist())
            ids.append(j)
            for h in range(subdivisions):
                shift = np.array(sh, dtype=int) if h == subdivisions - 1 else np.zeros(3, dtype=int)
                x = (a + (b - a) * h / subdivisions) @ self.A
                y = (a + (b - a) * (h + 1) / subdivisions) @ self.A
                elements.append(dict(i=ids[h], j=ids[h + 1], shift=shift, a=x, b=y, K=beam_matrix(x, y, radius), G=geometric_matrix(x, y), parent=edge_id))
        self.frac = np.array(frac)
        self.elements = elements
        self.ndof = 6 * len(frac)
        self.free = np.arange(3, self.ndof)
        for e in elements:
            e['di'] = 6 * e['i'] + np.arange(6)
            e['dj'] = 6 * e['j'] + np.arange(6)
        self.K0 = self.assemble([e['K'] for e in elements], np.zeros(3)).real
        self.kept = self.K0[np.ix_(self.free, self.free)]
        self.scale = 1 / np.sqrt(np.diag(self.kept))
        self.factor = la.cho_factor(self.kept * self.scale[:, None] * self.scale[None, :])

    def assemble(self, matrices, k):
        out = np.zeros((self.ndof, self.ndof), complex)
        for e, m in zip(self.elements, matrices):
            di, dj = (e['di'], e['dj'])
            phase = np.exp(2j * np.pi * np.dot(k, e['shift']))
            out[np.ix_(di, di)] += m[:6, :6]
            out[np.ix_(dj, dj)] += m[6:, 6:]
            out[np.ix_(di, dj)] += phase * m[:6, 6:]
            out[np.ix_(dj, di)] += phase.conjugate() * m[6:, :6]
        return (out + out.conj().T) / 2

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
        absolute_residual = la.norm(self.K0 @ w + force)
        residual = absolute_residual / max(la.norm(force), 1e-15)
        forces = []
        energy = 0.0
        stress = np.zeros((3, 3))
        bending = 0.0
        for e, d in zip(self.elements, imposed):
            d = d + np.r_[w[e['di']], w[e['dj']]]
            f = e['K'] @ d
            delta = e['b'] - e['a']
            L = la.norm(delta)
            n = delta / L
            axial = np.pi * self.radius ** 2 / L * np.dot(n, d[6:9] - d[:3])
            forces.append(axial)
            energy += 0.5 * np.dot(d, f)
            stress += 0.5 * (np.outer(f[6:9], delta) + np.outer(delta, f[6:9]))
            bending += np.dot(f[3:6], f[3:6]) + np.dot(f[9:12], f[9:12])
        stress /= abs(la.det(self.A))
        return dict(w=w, forces=np.array(forces), stress=stress, energy=energy, relative_residual=residual, max_nonaffine=float(np.max(np.abs(w))), moment_norm=float(np.sqrt(bending)), absolute_residual=float(absolute_residual), force_scaled_residual=float(absolute_residual / max(abs(np.array(forces)).max(), 1e-30)))

    def prestress(self, protocol='strain'):
        strain = -np.eye(3)
        if protocol == 'stress':
            bases = []
            for i in range(3):
                x = np.zeros((3, 3))
                x[i, i] = 1
                bases.append(x)
            for i, j in [(1, 2), (0, 2), (0, 1)]:
                x = np.zeros((3, 3))
                x[i, j] = x[j, i] = 1 / np.sqrt(2)
                bases.append(x)
            C = np.array([[np.sum(b * self.relaxed(a)['stress']) for a in bases] for b in bases])
            ep = la.solve(C, np.array([-1.0, -1.0, -1.0, 0, 0, 0]))
            strain = sum((c * b for c, b in zip(ep, bases)))
            strain /= -np.trace(strain) / 3
        self.pre = self.relaxed(strain)
        self.pre['strain'] = strain
        self.Gm = [N * e['G'] for N, e in zip(self.pre['forces'], self.elements)]
        return self.pre

    def eigen(self, k, count=8, vectors=False):
        K = self.assemble([e['K'] for e in self.elements], k)
        H = -self.assemble(self.Gm, k)
        gamma = la.norm(np.asarray(k) - np.rint(k)) < 1e-10
        free = self.free if gamma else np.arange(self.ndof)
        K = K[np.ix_(free, free)]
        H = H[np.ix_(free, free)]
        scale = 1 / np.sqrt(K.diagonal().real)
        K = K * scale[:, None] * scale[None, :]
        H = H * scale[:, None] * scale[None, :]
        kw = dict(subset_by_index=(max(0, len(free) - count), len(free) - 1), driver='gvx', check_finite=False)
        if vectors:
            vals, vec = la.eigh(H, K, **kw)
            vals, vec = (vals[::-1], vec[:, ::-1])
            v = np.zeros((self.ndof, len(vals)), complex)
            v[free] = scale[:, None] * vec
            if gamma:
                v = v.reshape(-1, 6, len(vals))
                v[:, :3] -= v[:, :3].mean(axis=0)
                v = v.reshape(self.ndof, -1)
            return (1 / vals, v)
        vals = la.eigh(H, K, eigvals_only=True, **kw)[::-1]
        return 1 / vals

    def operations(self):
        ops = []
        for op in self.data['sg'].operations():
            R = np.array(op.rot, dtype=float) / op.DEN
            t = np.array(op.tran, dtype=float) / op.DEN
            Q = self.A.T @ R @ la.inv(self.A.T)
            ops.append((R, t, Q))
        return ops

    def affine_matrices(self):
        """Gamma fluctuations plus nine macroscopic displacement-gradient dofs.

        Restricting F=a tensor n gives the exact k->0 Rayleigh problem of this
        discretized, prestressed beam model without small-k cancellation.
        Gamma optical eigenmodes remain in this space and must be distinguished.
        """
        size = len(self.free) + 9
        K = np.zeros((size, size))
        H = K.copy()
        for e, g in zip(self.elements, self.Gm):
            B = np.zeros((12, size))
            for local, global_dofs in [(0, e['di']), (6, e['dj'])]:
                for j, d in enumerate(global_dofs):
                    if d >= 3:
                        B[local + j, d - 3] += 1
            for i in range(3):
                B[i, len(self.free) + 3 * i:len(self.free) + 3 * i + 3] = e['a']
                B[i + 6, len(self.free) + 3 * i:len(self.free) + 3 * i + 3] = e['b']
            K += B.T @ e['K'] @ B
            H -= B.T @ g @ B
        self.affineK, self.affineH = (K, H)

    def acoustic(self, n, vectors=False):
        if not hasattr(self, 'affineK'):
            self.affine_matrices()
        n = np.asarray(n, dtype=float)
        n /= la.norm(n)
        P = np.kron(np.eye(3), n[:, None])
        matrices = []
        for M in [self.affineK, self.affineH]:
            b = M[:-9, -9:] @ P
            matrices.append(np.block([[M[:-9, :-9], b], [b.T, P.T @ M[-9:, -9:] @ P]]))
        K, H = matrices
        scale = 1 / np.sqrt(np.diag(K))
        kw = dict(subset_by_index=(len(K) - 4, len(K) - 1), driver='gvx', check_finite=False)
        if vectors:
            ev, v = la.eigh(H * scale[:, None] * scale[None, :], K * scale[:, None] * scale[None, :], **kw)
            return (1 / ev[::-1], scale[:, None] * v[:, ::-1])
        ev = la.eigh(H * scale[:, None] * scale[None, :], K * scale[:, None] * scale[None, :], eigvals_only=True, **kw)
        return 1 / ev[::-1]

def orbit_key(k, operations):
    orbit = []
    for R, t, Q in operations:
        q = la.solve(R.T, k) % 1
        orbit.append(tuple(np.round(q % 1, 8)))
        orbit.append(tuple(np.round(-q % 1, 8)))
    return min(orbit)

def representatives(frame, grid):
    ops = frame.operations()
    reps = {}
    for p in itertools.product(range(grid), repeat=3):
        k = np.array(p, dtype=float) / grid
        key = orbit_key(k, ops)
        if key not in reps:
            reps[key] = k
    return list(reps.values())

def scan(name, radius, subdivisions, grid=4, protocol='strain', acoustic=True):
    start = time.perf_counter()
    f = Frame(name, radius, subdivisions)
    p = f.prestress(protocol)
    points = representatives(f, grid)
    if acoustic:
        small = {}
        for d in itertools.product((-1, 0, 1), repeat=3):
            if any(d):
                k = np.array(d, dtype=float) * 0.001
                small[orbit_key(k, f.operations())] = k
        points += list(small.values())
    rows = []
    for k in points:
        vals = f.eigen(k, count=8)
        rows.append(dict(k=k.tolist(), critical=float(vals[0]), eigenvalues=vals[:8].tolist(), multiplicity=int(np.sum(abs(vals / vals[0] - 1) < 2e-05))))
    rows.sort(key=lambda x: x['critical'])
    best = rows[0]
    selected = [r for r in rows if abs(r['critical'] / best['critical'] - 1) < 2e-05]
    result = dict(name=name, radius_over_mean_edge=radius, subdivisions=subdivisions, grid=grid, protocol=protocol, space_group=f.data['sg'].xhm(), source_group=f.data['net']['group'], reference_nodes=len(f.data['frac']), reference_edges=len(f.data['edges']), nodes=len(f.frac), dof=f.ndof, edge_lengths=f.data['lengths'], loader_trials=f.data['trials'], A=f.A.tolist(), pre=dict(strain=p['strain'].tolist(), stress=p['stress'].tolist(), axial_min=float(p['forces'].min()), axial_max=float(p['forces'].max()), max_nonaffine=p['max_nonaffine'], moment_norm=p['moment_norm'], equilibrium_residual=p['relative_residual']), critical=best, degenerate_representatives=selected, pressure_over_E=float(-np.trace(p['stress']) / 3 * best['critical']), sampled_points=len(points), spectrum=rows, seconds=time.perf_counter() - start)
    return (result, f)

def column_check():
    output = []
    for div in [1, 2, 4, 8, 16]:
        K = np.zeros((6 * (div + 1), 6 * (div + 1)))
        G = K.copy()
        for i in range(div):
            a = np.array([i / div, 0.0, 0.0])
            b = np.array([(i + 1) / div, 0.0, 0.0])
            dof = np.arange(6 * i, 6 * i + 12)
            K[np.ix_(dof, dof)] += beam_matrix(a, b, 0.03)
            G[np.ix_(dof, dof)] += geometric_matrix(a, b)
        free = [6 * i + c for i in range(div + 1) for c in [1, 5] if c == 5 or 0 < i < div]
        vals = la.eigvalsh(K[np.ix_(free, free)], G[np.ix_(free, free)])
        exact = np.pi ** 2 * (np.pi * 0.03 ** 4 / 4)
        output.append(dict(subdivisions=div, critical=vals[0], exact=exact, relative_error=vals[0] / exact - 1))
    return output

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', default=['pcu', 'dia', 'srs', 'nbo', 'sod', 'bor', 'fcu', 'utp'])
    ap.add_argument('--radii', nargs='+', type=float, default=[0.02, 0.05])
    ap.add_argument('--subdivisions', type=int, default=2)
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--protocol', default='strain')
    ap.add_argument('--out', default='validation/bloch_results.json')
    args = ap.parse_args()
    result = dict(column_check=column_check(), runs=[])
    for name in args.nets:
        for radius in args.radii:
            r, f = scan(name, radius, args.subdivisions, args.grid, args.protocol)
            result['runs'].append(r)
            dump(ROOT / args.out, result)
            print(name, radius, 'dof', r['dof'], 'critical', r['critical']['critical'], 'k', r['critical']['k'], 'mult', r['critical']['multiplicity'], 'seconds', round(r['seconds'], 2), flush=True)
