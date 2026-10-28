"""RCSR 3D net -> periodic graph (nodes in the unit cell + edges with lattice shifts).
Adapted from the RCSR graph parser (read_cgd/expand/edges), generalised to non-orthogonal cells.
Edges: for every node, its CN nearest neighbours (RCSR maximum-symmetry embedding); the edge set is symmetrised."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import numpy as np, gemmi
CGD = __import__('os').environ.get('RCSR_CGD', 'rcsr3d_web.cgd')
if not __import__('os').path.exists(CGD):
    pass

def read_cgd(path=CGD):
    out = []
    cur = None
    for line in open(path, encoding='utf-8', errors='replace'):
        s = line.split()
        if not s:
            continue
        if s[0] == 'CRYSTAL':
            cur = {'nodes': []}
        elif s[0] == 'NAME':
            cur['name'] = s[1]
        elif s[0] == 'GROUP':
            cur['group'] = s[1]
        elif s[0] == 'CELL':
            cur['cell'] = list(map(float, s[1:7]))
        elif s[0] == 'NODE':
            cur['nodes'].append((int(s[2]), np.array(list(map(float, s[3:6])))))
        elif s[0] == 'EDGE_CENTER':
            cur.setdefault('ecs', []).append(np.array(list(map(float, s[3:6]))))
        elif s[0] == 'END' and cur is not None:
            out.append(cur)
            cur = None
    return out

def cell_matrix(cell):
    """Rows = lattice vectors a, b, c in Cartesian coordinates (gemmi convention)."""
    uc = gemmi.UnitCell(*cell)
    M = np.array(uc.orth.mat.tolist())
    return M.T

def expand(net, setting=None, tol=0.002):
    """Orbit expansion. setting=':2' selects origin choice 2 (RCSR/Systre convention for groups with two origins).
    Near-duplicate images (rounded 1/3, 2/3 coordinates) are merged within tol."""
    sg = gemmi.find_spacegroup_by_name(net['group'] + (setting or ''))
    if sg is None:
        return None
    pts, cns = ([], [])
    for cn, xyz in net['nodes']:
        for op in sg.operations():
            p = np.array(op.apply_to_xyz(list(xyz))) % 1.0
            if not any((np.linalg.norm((p - q + 0.5) % 1 - 0.5) < tol for q in pts)):
                pts.append(p)
                cns.append(cn)
    return (np.array(pts), np.array(cns))

def periodic_edges(net, pts, cns, tol=0.001):
    """Return list of (i, j, shift) with i<=j convention removed duplicates; frac(j)+shift is the partner of i."""
    A = cell_matrix(net['cell'])
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')).reshape(3, -1).T
    E = set()
    for i, p in enumerate(pts):
        cand = []
        for j, q in enumerate(pts):
            d = (q[None, :] + shifts - p[None, :]) @ A
            r = np.linalg.norm(d, axis=1)
            for k in range(len(shifts)):
                if r[k] > 1e-06:
                    cand.append((r[k], j, tuple(shifts[k])))
        cand.sort()
        source0 = cand[cns[i] - 1][0]
        for r, j, sh in cand:
            if r > source0 + tol:
                break
            a = (i, j, sh)
            b = (j, i, tuple(-np.array(sh)))
            E.add(min(a, b))
    E = sorted(E)
    deg = np.zeros(len(pts), int)
    for i, j, sh in E:
        deg[i] += 1
        deg[j] += 1
    return (E, deg, A)

def load(name, nets=None):
    """Try the default setting, then origin choice 2; keep the first whose edge degrees match the CGD coordination."""
    nets = nets or read_cgd()
    net = next((n for n in nets if n['name'] == name))
    best = None
    for setting in (None, ':2'):
        e = expand(net, setting)
        if e is None:
            continue
        pts, cns = e
        E, deg, A = periodic_edges(net, pts, cns)
        G = dict(net=net, frac=pts, cn=cns, edges=E, deg=deg, A=A, ok=bool(np.all(deg == cns)), setting=setting)
        if G['ok']:
            return G
        best = best or G
    return best

def edge_center_edges(net, pts, cns, setting=None, tol=0.002, rmax_factor=2.5):
    """Edges from the CGD EDGE_CENTER orbits: a node pair (i, j+shift) is an edge iff its midpoint lies on an
    expanded edge-centre position. Robust where nearest-neighbour shells are ambiguous."""
    sg = gemmi.find_spacegroup_by_name(net['group'] + (setting or ''))
    ec = []
    for xyz in net.get('ecs', []):
        for op in sg.operations():
            p = np.array(op.apply_to_xyz(list(xyz))) % 1.0
            if not any((np.linalg.norm((p - q + 0.5) % 1 - 0.5) < tol for q in ec)):
                ec.append(p)
    ec = np.array(ec)
    A = cell_matrix(net['cell'])
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')).reshape(3, -1).T
    E = set()
    imgs = np.array([(k, *pts[k] + sh) for k in range(len(pts)) for sh in shifts])
    for c in ec:
        d = np.linalg.norm((imgs[:, 1:] - c) @ A, axis=1)
        order = np.argsort(d)
        if len(order) < 2 or abs(d[order[1]] - d[order[0]]) > 0.001 * max(d[order[0]], 1e-09):
            continue
        (i, *pi), (j, *pj) = (imgs[order[0]], imgs[order[1]])
        i, j = (int(i), int(j))
        cell_i = np.floor(np.array(pi) + 1e-09)
        cell_j = np.floor(np.array(pj) + 1e-09)
        sh = tuple((cell_j - cell_i).astype(int))
        a = (i, j, sh)
        b = (j, i, tuple(-np.array(sh)))
        E.add(min(a, b))
    E = sorted(E)
    deg = np.zeros(len(pts), int)
    for i, j, sh in E:
        deg[i] += 1
        deg[j] += 1
    return (E, deg, A)

def load_ec(name, nets=None):
    """Like load(), but edges from EDGE_CENTER orbits (falls back to nearest neighbours if no edge centres)."""
    nets = nets or read_cgd()
    net = next((n for n in nets if n['name'] == name))
    best = None
    for setting in (None, ':2'):
        e = expand(net, setting)
        if e is None:
            continue
        pts, cns = e
        E, deg, A = edge_center_edges(net, pts, cns, setting) if net.get('ecs') else periodic_edges(net, pts, cns)
        G = dict(net=net, frac=pts, cn=cns, edges=E, deg=deg, A=A, ok=bool(np.all(deg == cns)), setting=setting)
        if G['ok']:
            return G
        best = best or G
    return best

def _ops(sg):
    """Space-group operations as (R, t) with fractional R (3x3) and t (3,)."""
    out = []
    for op in sg.operations():
        R = np.array(op.rot, float) / op.DEN
        t = np.array(op.tran, float) / op.DEN
        out.append((R, t))
    return out

def load_sym(name, nets=None, tol=0.002):
    """Symmetry-exact loader: node coordinates snapped to their site-symmetry average, and edges generated as
    full space-group orbits of one representative edge per EDGE_CENTER (nearest two vertices of the edge centre).
    The edge set is invariant under every operation by construction."""
    nets = nets or read_cgd()
    net = next((n for n in nets if n['name'] == name))
    best = None
    for setting in (None, ':2'):
        sg = gemmi.find_spacegroup_by_name(net['group'] + (setting or ''))
        if sg is None:
            continue
        ops = _ops(sg)
        pts, cns = ([], [])
        for cn, xyz in net['nodes']:
            imgs = [R @ xyz + t for R, t in ops]
            stab = [(g % 1.0 + 0.5) % 1 - 0.5 for g in imgs]
            close = [g for g in imgs if np.linalg.norm((g - xyz + 0.5) % 1 - 0.5) < tol]
            x0 = xyz + np.mean([(g - xyz + 0.5) % 1 - 0.5 for g in close], axis=0)
            for R, t in ops:
                p = (R @ x0 + t) % 1.0
                if not any((np.linalg.norm((p - q + 0.5) % 1 - 0.5) < tol for q in pts)):
                    pts.append(p)
                    cns.append(cn)
        pts = np.array(pts)
        cns = np.array(cns)
        A = cell_matrix(net['cell'])
        shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')).reshape(3, -1).T
        imgs = np.array([(k, *pts[k] + sh) for k in range(len(pts)) for sh in shifts])

        def locate(x):
            f = x % 1.0
            d = np.linalg.norm((pts - f + 0.5) % 1 - 0.5, axis=1)
            k = int(np.argmin(d))
            if d[k] > 10 * tol:
                return None
            return (k, np.round(x - pts[k]).astype(int))
        E = set()
        ok_rep = True
        for c in net.get('ecs', []):
            P = imgs[:, 1:]
            best_pair = None
            for u in range(len(P)):
                v_target = 2 * c - P[u]
                dv = np.linalg.norm((P - v_target) @ A, axis=1)
                v = int(np.argmin(dv))
                if dv[v] < 0.001 and v != u:
                    L = np.linalg.norm((P[v] - P[u]) @ A)
                    if best_pair is None or L < best_pair[0] - 1e-09:
                        best_pair = (L, P[u], P[v])
            if best_pair is None:
                ok_rep = False
                continue
            a, b = (best_pair[1], best_pair[2])
            for R, t in ops:
                la = locate(R @ a + t)
                lb = locate(R @ b + t)
                if la is None or lb is None:
                    ok_rep = False
                    continue
                (ii, si), (jj, sj) = (la, lb)
                sh = tuple((sj - si).astype(int))
                if ii == jj and (not any(sh)):
                    continue
                e1 = (ii, jj, sh)
                e2 = (jj, ii, tuple(-np.array(sh)))
                E.add(min(e1, e2))
        E = sorted(E)
        deg = np.zeros(len(pts), int)
        for i, j, sh in E:
            deg[i] += 1
            deg[j] += 1
        G = dict(net=net, frac=pts, cn=cns, edges=E, deg=deg, A=A, ok=bool(np.all(deg == cns)) and ok_rep, setting=setting)
        if G['ok']:
            return G
        best = best or G
    return best
