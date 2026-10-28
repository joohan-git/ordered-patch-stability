"""Finite 3D Euler--Bernoulli rigid-joint frame tests; numpy/scipy only.

All coordinates, material constants and dimensions are nondimensional.
Circular beams: E=1, nu=0.3. Intersections are joints only at shared endpoints.
The square specimen is clipped in its loading frame, not rotated after cutting.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

def orientation(direction):
    n = np.array(direction, dtype=float)
    n /= np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(ref, n)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.array([t1, t2, n])

def source_edges(kind, lower, upper, h=0.27):
    edges = []

    def put(a, b):
        edges.append((np.asarray(a, dtype=float), np.asarray(b, dtype=float)))
    ranges = [range(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper)]
    if kind in ('polar', 'centro'):
        h = 0.5 if kind == 'centro' else h
        for cell in itertools.product(*ranges):
            c = np.array(cell, dtype=float)
            apex = c + [0.5, 0.5, h]
            for corner in itertools.product((0, 1), repeat=3):
                put(apex, c + corner)
            for axis in np.eye(3):
                put(c, c + axis)
    elif kind in ('chiral', 'racemic'):
        r = 0.28
        ring = np.array([[r, 0, 0], [0, r, 0], [-r, 0, 0], [0, -r, 0]])
        for cell in itertools.product(*ranges):
            c = np.array(cell, dtype=float)
            hand = 1 if kind == 'chiral' else (-1) ** (cell[0] + cell[1])
            for j in range(4):
                put(c + ring[j], c + ring[(j + 1) % 4])
                put(c + ring[j], c + ring[(j + hand) % 4] + [0, 0, 1])
            put(c + ring[0], c + [1, 0, 0] + ring[2])
            put(c + ring[1], c + [0, 1, 0] + ring[3])
    elif kind == 'td':
        fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / 4
        same = []
        for zero_axis in range(3):
            axes = [i for i in range(3) if i != zero_axis]
            for signs in itertools.product((-1, 1), repeat=2):
                v = np.zeros(3)
                v[axes] = np.array(signs) / 2
                if v[np.flatnonzero(v)[0]] > 0:
                    same.append(v)
        for cell in itertools.product(*ranges):
            for offset in fcc:
                a = np.array(cell) + offset
                for d in tetra:
                    put(a, a + d)
                for d in same:
                    put(a, a + d)
    else:
        raise ValueError(kind)
    return np.asarray(edges)

def specimen(kind, size, direction, aspect=2.0, h=0.27, mirror=False, offset=None, units=(1.0, 1.0, 1.0), shape='square'):
    q = orientation(direction)
    half = np.array([size / 2, size / 2, aspect * size / 2]) * np.array(units)
    shift = np.zeros(3) if offset is None else np.array(offset, dtype=float)
    corners = np.array(list(itertools.product((-1, 1), repeat=3))) * half
    crystal_corners = corners @ q + shift
    lower = np.floor(crystal_corners.min(axis=0)).astype(int) - 1
    upper = np.ceil(crystal_corners.max(axis=0)).astype(int) + 1
    edge = (source_edges(kind, lower, upper, h) - shift) @ q.T
    a, b = (edge[:, 0], edge[:, 1])
    v = b - a
    enter, leave = (np.zeros(len(a)), np.ones(len(a)))
    good = np.ones(len(a), dtype=bool)
    for j in range(3):
        parallel = np.abs(v[:, j]) < 1e-12
        good &= ~(parallel & ((a[:, j] < -half[j] - 1e-10) | (a[:, j] > half[j] + 1e-10)))
        div = np.where(parallel, 1.0, v[:, j])
        t1, t2 = ((-half[j] - a[:, j]) / div, (half[j] - a[:, j]) / div)
        enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(t1, t2)))
        leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(t1, t2)))
    if shape == 'cylinder':
        if abs(half[0] - half[1]) > 1e-10:
            raise ValueError('Cylinder needs equal transverse dimensions')
        qa = np.sum(v[:, :2] ** 2, axis=1)
        qb = 2 * np.sum(a[:, :2] * v[:, :2], axis=1)
        qc = np.sum(a[:, :2] ** 2, axis=1) - half[0] ** 2
        parallel = qa < 1e-20
        discr = qb * qb - 4 * qa * qc
        good &= ~(parallel & (qc > 1e-10) | ~parallel & (discr < 0))
        denom = np.where(parallel, 1.0, 2 * qa)
        root = np.sqrt(np.maximum(discr, 0))
        enter = np.maximum(enter, np.where(parallel, -np.inf, (-qb - root) / denom))
        leave = np.minimum(leave, np.where(parallel, np.inf, (-qb + root) / denom))
    good &= (leave - enter) * np.linalg.norm(v, axis=1) > 1e-06
    a, b, v, enter, leave = (a[good], b[good], v[good], enter[good], leave[good])
    clipped = np.stack([a + enter[:, None] * v, a + leave[:, None] * v], axis=1)
    if mirror:
        clipped[:, :, 0] *= -1
    nodes, index = np.unique(np.round(clipped.reshape(-1, 3), 9), axis=0, return_inverse=True)
    edges = np.unique(np.sort(index.reshape(-1, 2), axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    graph = coo_matrix((np.ones(2 * len(edges)), (edges.ravel(), edges[:, ::-1].ravel())), shape=(len(nodes), len(nodes))).tocsr()
    nc, labels = connected_components(graph, directed=False)
    if nc > 1:
        sizes = np.bincount(labels)
        spanning = []
        for lab in range(nc):
            z = nodes[labels == lab, 2]
            if z.min() < -half[2] + 1e-07 and z.max() > half[2] - 1e-07:
                spanning.append(lab)
        if not spanning:
            raise RuntimeError('No spanning component')
        if len(spanning) > 1:
            raise RuntimeError('Multiple spanning components require explicit plate assembly')
        lab = max(spanning, key=lambda i: sizes[i])
        keep = labels == lab
        remap = np.full(len(nodes), -1, dtype=int)
        remap[keep] = np.arange(keep.sum())
        edges = remap[edges[keep[edges].all(axis=1)]]
        nodes = nodes[keep]
    return (nodes, edges, half, q, nc)

def beam_matrix(a, b, radius=0.045, theory='euler'):
    tangent = b - a
    length = np.linalg.norm(tangent)
    tangent /= length
    ref = np.array([0.0, 0.0, 1.0]) if abs(tangent[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = np.cross(ref, tangent)
    e2 /= np.linalg.norm(e2)
    r = np.array([tangent, e2, np.cross(tangent, e2)])
    transform = np.kron(np.eye(4), r)
    area = np.pi * radius ** 2
    inertia = np.pi * radius ** 4 / 4
    shear = 1 / (2 * 1.3)
    k = np.zeros((12, 12))
    k[np.ix_([0, 6], [0, 6])] = area / length * np.array([[1, -1], [-1, 1]])
    k[np.ix_([3, 9], [3, 9])] = shear * 2 * inertia / length * np.array([[1, -1], [-1, 1]])
    l = length
    shear_factor = 6 * (1 + 0.3) / (7 + 6 * 0.3)
    phi = 12 * inertia / (shear_factor * shear * area * l * l) if theory == 'timoshenko' else 0.0
    kb = inertia / (l ** 3 * (1 + phi)) * np.array([[12, 6 * l, -12, 6 * l], [6 * l, (4 + phi) * l * l, -6 * l, (2 - phi) * l * l], [-12, -6 * l, 12, -6 * l], [6 * l, (2 - phi) * l * l, -6 * l, (4 + phi) * l * l]])
    k[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = kb
    signs = np.array([1, -1, 1, -1])
    k[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = kb * signs[:, None] * signs[None, :]
    return transform.T @ k @ transform

def assemble(nodes, edges, radius, theory='euler'):
    rows, cols, vals = ([], [], [])
    for i, j in edges:
        dof = np.r_[6 * i + np.arange(6), 6 * j + np.arange(6)]
        k = beam_matrix(nodes[i], nodes[j], radius, theory)
        rows.append(np.repeat(dof, 12))
        cols.append(np.tile(dof, 12))
        vals.append(k.ravel())
    n = 6 * len(nodes)
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()

def solve_case(kind, size, direction=(0, 0, 1), aspect=2.0, h=0.27, mirror=False, radius=0.045, offset=None, units=(1.0, 1.0, 1.0), shape='square', theory='euler', condensed=False):
    started = time.perf_counter()
    nodes, edges, half, q, nc = specimen(kind, size, direction, aspect, h, mirror, offset, units, shape)
    top = np.flatnonzero(abs(nodes[:, 2] - half[2]) < 2e-08)
    bottom = np.flatnonzero(abs(nodes[:, 2] + half[2]) < 2e-08)
    if len(top) < 3 or len(bottom) < 3:
        raise RuntimeError('Insufficient plate nodes')
    k = assemble(nodes, edges, radius, theory)
    fixed_nodes = np.zeros(len(nodes), dtype=bool)
    fixed_nodes[top] = fixed_nodes[bottom] = True
    free = (6 * np.flatnonzero(~fixed_nodes)[:, None] + np.arange(6)).ravel()
    tr, tc, tv = (free.tolist(), list(range(len(free))), np.ones(len(free)).tolist())
    for i in top:
        for comp, value in [(0, -nodes[i, 1]), (1, nodes[i, 0]), (5, 1.0)]:
            if value:
                tr.append(6 * i + comp)
                tc.append(len(free))
                tv.append(value)
    transform = coo_matrix((tv, (tr, tc)), shape=(k.shape[0], len(free) + 1)).tocsr()
    imposed = np.zeros(k.shape[0])
    strain = -0.001
    delta = strain * 2 * half[2]
    imposed[6 * top + 2] = delta
    reduced = (transform.T @ k @ transform).tocsc()
    rhs = -transform.T @ (k @ imposed)
    diagonal = np.sqrt(reduced.diagonal())
    from scipy.sparse import diags
    invd = diags(1 / diagonal)
    solution = spsolve(invd @ reduced @ invd, rhs / diagonal) / diagonal
    displacement = transform @ solution + imposed
    reaction = k @ displacement
    theta = solution[-1]
    torque = np.sum(reaction[6 * top + 5] + nodes[top, 0] * reaction[6 * top + 1] - nodes[top, 1] * reaction[6 * top])
    force = np.sum(reaction[6 * top + 2])
    relative_residual = np.linalg.norm(reduced @ solution - rhs) / (np.linalg.norm(rhs) + 1e-300)
    if not np.isfinite(relative_residual) or relative_residual > 1e-07:
        raise RuntimeError(f'Unacceptable linear residual {relative_residual}')
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    result = {'kind': kind, 'N': size, 'M': aspect * size, 'direction': list(direction), 'h': h, 'mirrored_specimen_x': mirror, 'radius': radius, 'offset': offset, 'units': list(units), 'shape': shape, 'specimen_dimensions': (2 * half).tolist(), 'theory': theory, 'nodes': len(nodes), 'beams': len(edges), 'top_nodes': len(top), 'bottom_nodes': len(bottom), 'components_before_filter': int(nc), 'minimum_beam_length': float(lengths.min()), 'theta_per_strain': float(theta / strain), 'theta': float(theta), 'strain': strain, 'axial_reaction': float(force), 'torque_residual': float(torque), 'relative_linear_residual': float(relative_residual), 'energy': float(displacement @ (k @ displacement) / 2), 'wall_seconds': time.perf_counter() - started}
    if condensed:
        from scipy.sparse.linalg import splu
        interior = k[free, :][:, free].tocsc()
        endpoint = np.column_stack([imposed / delta, transform[:, -1].toarray().ravel()])
        coupling = k[free, :] @ endpoint
        factor = splu(interior)
        plate = endpoint.T @ (k @ endpoint) - coupling.T @ factor.solve(coupling)
        result['plate_stiffness_delta_theta'] = plate.tolist()
        result['plate_reciprocity_error'] = float(np.linalg.norm(plate - plate.T) / np.linalg.norm(plate))
        result['dimensionless_plate_coupling'] = float(plate[0, 1] / np.sqrt(plate[0, 0] * plate[1, 1]))
        result['theta_per_strain_from_condensed_energy'] = float(-2 * half[2] * plate[0, 1] / plate[1, 1])
    return result

def element_check():
    a, b = (np.array([0.1, -0.2, 0.3]), np.array([0.7, 0.4, 1.5]))
    k = beam_matrix(a, b)
    errors = []
    for omega in np.eye(3):
        d = np.r_[np.cross(omega, a), omega, np.cross(omega, b), omega]
        errors.append(np.linalg.norm(k @ d) / (np.linalg.norm(k) * np.linalg.norm(d)))
    for translation in np.eye(3):
        d = np.r_[translation, [0, 0, 0], translation, [0, 0, 0]]
        errors.append(np.linalg.norm(k @ d) / (np.linalg.norm(k) * np.linalg.norm(d)))
    assert max(errors) < 1e-12
    return max(errors)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='benchmark', choices=['benchmark', 'main', 'extra'])
    args = parser.parse_args()
    result = {'element_rigid_motion_relative_residual': element_check(), 'cases': []}
    common_examples = []
    if args.suite == 'benchmark':
        common_examples = [dict(kind='chiral', size=2), dict(kind='polar', size=2, direction=(2, 1, 0)), dict(kind='centro', size=2, direction=(2, 1, 0)), dict(kind='td', size=2, direction=(1, 2, 3))]
    elif args.suite == 'main':
        for size in [2, 3, 4]:
            common_examples.extend([dict(kind='chiral', size=size), dict(kind='chiral', size=size, mirror=True), dict(kind='polar', size=size, direction=(2, 1, 0)), dict(kind='polar', size=size, direction=(2, -1, 0)), dict(kind='polar', size=size, direction=(2, 1, 0), mirror=True), dict(kind='centro', size=size, direction=(2, 1, 0)), dict(kind='td', size=size, direction=(1, 2, 3)), dict(kind='td', size=size, direction=(1, 2, 3), mirror=True)])
    else:
        common_examples = [dict(kind='polar', size=n, direction=d) for n in [2, 3, 4] for d in [(1, 0, 0), (1, 1, 0)]]
        common_examples += [dict(kind='td', size=n, direction=(1, 1, 1)) for n in [2, 3, 4]]
        common_examples += [dict(kind='polar', size=n, direction=(2, 1, 0), h=0.73) for n in [2, 3, 4]]
    output = Path(__file__).with_name('validation/beam_' + args.suite + '.json')
    for case in common_examples:
        value = solve_case(**case)
        result['cases'].append(value)
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(value), flush=True)
