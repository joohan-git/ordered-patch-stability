"""Ising sampler and seeds, periodic clusters/squares, energy localization."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.diagnostics import *
from scipy import ndimage
import argparse, gc

def sample_sigma(J, L, rng, sweeps=3000):
    s = rng.choice([-1, 1], size=(L, L))
    mask = (np.indices((L, L)).sum(0) % 2).astype(bool)
    for _ in range(sweeps):
        for m in (mask, ~mask):
            h = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
            p = 1 / (1 + np.exp(-2 * J * h))
            new = np.where(rng.random((L, L)) < p, 1, -1)
            s = np.where(m, new, s)
    return s

def periodic_clusters(sig):
    L = len(sig)
    lab = np.zeros((L, L), int)
    for sign in (1, -1):
        part, n = ndimage.label(sig == sign)
        part[part > 0] += lab.max()
        lab = np.where(part > 0, part, lab)
    parent = np.arange(lab.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(L):
        for a, b, match in ((lab[0, i], lab[-1, i], sig[0, i] == sig[-1, i]), (lab[i, 0], lab[i, -1], sig[i, 0] == sig[i, -1])):
            if match:
                parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(len(parent))])
    lab = roots[lab]
    ids, counts = np.unique(lab, return_counts=True)
    big = ids[counts == counts.max()]
    return (lab, big, int(counts.max()))

def largest_squares(sig):
    L = len(sig)
    tiled = np.tile(sig, (2, 2))
    ii = np.pad(tiled, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def sums(r):
        return ii[r:r + L, r:r + L] - ii[:L, r:r + L] - ii[r:r + L, :L] + ii[:L, :L]
    lo, hi = (1, L)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if np.any(np.abs(sums(mid)) == mid * mid):
            lo = mid
        else:
            hi = mid - 1
    return (lo, np.argwhere(np.abs(sums(lo)) == lo * lo))

def state_path(L, J, sample):
    path = OUT / 'truchet_2d' / 'states' / f'ising_L{L}_J{int(J * 100)}_s{sample}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_state(row):
    """Regenerate a missing endpoint and its mechanical energy from the recorded seed."""
    L, J, seed = row['L'], row['J'], row['sample']
    path = state_path(L, J, seed)
    if not path.exists():
        print('COMPUTE Ising state', L, J, seed, flush=True)
        sig = sample_sigma(J, L, np.random.default_rng([20261026, 14, int(round(J * 1000)), seed]))
        assert hashlib.sha256(sig.astype(np.int8).tobytes()).hexdigest() == row['pattern_sha256']
        stagger = np.where(np.indices((L, L)).sum(0) % 2 == 0, 1, -1)
        frame = Frame2D(((sig * stagger + 1) // 2).astype(int))
        frame.hydro()
        vals, vectors, residual = frame.eigen(count=2)
        assert abs(vals[0] * frame.p0 / row['pressure'] - 1) < 1e-5
        energy = np.array(frame.mode_stats(vectors[:, 0])['tile_energy']).reshape(L, L)
        np.savez_compressed(path, sigma=sig, tile_energy=energy)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', nargs='+', type=int, default=[32, 64, 96, 128])
    ap.add_argument('--limit', type=int, help='Maximum newly solved samples')
    args = ap.parse_args()
    computed = 0
    name = 'truchet_2d/ising.json'
    path = OUT / name
    base = json.loads((DATA_ROOT / 'truchet_2d/planted_pressure.json').read_text())['baseline']['pressure']
    out = json.loads(path.read_text()) if path.exists() else dict(metadata=metadata(), runs=[], baseline_pressure=base)
    out['sampling'] = dict(sweeps=3000, samples_per_J_L=6, Jc=float(np.log(1 + np.sqrt(2)) / 2), seed='[20261026,14,round(1000*J),sample], exactly pressure convention; seed reused across L', algorithm='checkerboard heat bath, random initial state, one draw after 3000 full sweeps', cluster='nearest-neighbour same sigma, torus seam unions included', square='largest axis-aligned uniform sigma square on torus, side <= L', ties='first lexicographic square / smallest root cluster; also min/max energy over size ties', mode='elastic-energy tile weights; PR=1/(N sum(tile_energy**2)); fixed-cell fluctuations', equilibrium='same sampler as pressure convention; 3000-sweep equilibration not independently certified')
    for L in args.sizes:
        stagger = np.where(np.indices((L, L)).sum(0) % 2 == 0, 1, -1)
        for J in (0.38, 0.42):
            for j in range(6):
                if any(((r['L'], r['J'], r['sample']) == (L, J, j) for r in out['runs'])):
                    ensure_state(next(r for r in out['runs'] if (r['L'], r['J'], r['sample']) == (L, J, j)))
                    continue
                start = time.perf_counter()
                print('start', L, J, j, flush=True)
                sig = sample_sigma(J, L, np.random.default_rng([20261026, 14, int(round(J * 1000)), j]))
                lab, big, bigsize = periodic_clusters(sig)
                side, starts = largest_squares(sig)
                f = Frame2D(((sig * stagger + 1) // 2).astype(int))
                p0 = f.hydro()
                vals, v, res = f.eigen(count=2)
                stats = f.mode_stats(v[:, 0])
                te = np.array(stats['tile_energy']).reshape((L, L))
                ce = [float(te[lab == b].sum()) for b in big]
                se = [float(te[np.ix_((np.arange(side) + a) % L, (np.arange(side) + b) % L)].sum()) for a, b in starts]
                row = dict(L=L, J=J, sample=j, ratio=float(vals[0] * p0 / base), pressure=float(vals[0] * p0), largest_cluster=bigsize, cluster_fraction=bigsize / L ** 2, cluster_ties=len(big), largest_square_side=side, square_start=starts[0], square_ties=len(starts), participation_ratio=stats['energy_participation_fraction'], participating_tiles=stats['participating_tiles'], mode_energy_largest_cluster=ce[0], cluster_energy_tie_range=[min(ce), max(ce)], mode_energy_largest_square=se[0], square_energy_tie_range=[min(se), max(se)], staggered_m=float(abs(sig.mean())), eigen_residual=res, preload_residual=f.pre['residual'], second_ratio=float(vals[1] * p0 / base), seconds=time.perf_counter() - start, pattern_sha256=hashlib.sha256(sig.astype(np.int8).tobytes()).hexdigest())
                np.savez_compressed(state_path(L, J, j), sigma=sig, tile_energy=te)
                out['runs'].append(row)
                save(name, out)
                print('done', L, J, j, round(row['ratio'], 6), 'PR', round(row['participation_ratio'], 4), 'sec', round(row['seconds'], 1), flush=True)
                del f
                gc.collect()
                computed += 1
                if args.limit and computed >= args.limit:
                    return
    save(name, out)
if __name__ == '__main__':
    main()
