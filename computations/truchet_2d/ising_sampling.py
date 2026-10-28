"""Correlated sampling: reproducible references, two starts, long chains, toroidal geometry."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.correlation_runtime import *
from scipy import ndimage, special
import argparse
STATE = OUT / 'states'

def labels_open(sig):
    lab = np.zeros(sig.shape, int)
    for sign in (1, -1):
        part, n = ndimage.label(sig == sign)
        part[part > 0] += lab.max()
        lab = np.where(part > 0, part, lab)
    return lab

def clusters(sig):
    lab = labels_open(sig)
    parent = np.arange(lab.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    L = len(sig)
    for i in range(L):
        for a, b, same in ((lab[0, i], lab[-1, i], sig[0, i] == sig[-1, i]), (lab[i, 0], lab[i, -1], sig[i, 0] == sig[i, -1])):
            if same:
                parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(len(parent))])
    lab = roots[lab]
    ids, counts = np.unique(lab, return_counts=True)
    big = ids[counts == counts.max()]
    return (lab, big, int(counts.max()))

def squares(sig):
    L = len(sig)
    ii = np.pad(np.tile(sig, (2, 2)), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def sums(r):
        return ii[r:r + L, r:r + L] - ii[:L, r:r + L] - ii[r:r + L, :L] + ii[:L, :L]
    lo, hi = (1, L)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if np.any(abs(sums(mid)) == mid * mid):
            lo = mid
        else:
            hi = mid - 1
    return (lo, np.argwhere(abs(sums(lo)) == lo * lo))

def axis_e_fold_length(sig):
    L = len(sig)
    f = np.fft.fft2(sig - sig.mean())
    c = np.real(np.fft.ifft2(f * np.conj(f))) / sig.size
    c /= c[0, 0] if c[0, 0] else 1
    r = np.arange(1, L // 2)
    cr = np.array([(c[d, 0] + c[0, d]) / 2 for d in r])
    pos = cr > np.exp(-1)
    return float(r[np.argmin(pos)] if (~pos).any() else L / 2)

def basic(sig):
    nn = float(np.mean(sig * (np.roll(sig, 1, 0) + np.roll(sig, 1, 1))) / 2)
    return (nn, float(sig.mean()))

def exact_nn(J):
    if J == 0:
        return 0.0
    k = 2 * np.sinh(2 * J) / np.cosh(2 * J) ** 2
    return float(0.5 / np.tanh(2 * J) * (1 + 2 / np.pi * (2 * np.tanh(2 * J) ** 2 - 1) * special.ellipk(k * k)))

def observables(sig, J):
    nn, m = basic(sig)
    lab, big, size = clusters(sig)
    side, starts = squares(sig)
    openlab = labels_open(sig)
    ids, cts = np.unique(openlab, return_counts=True)
    return dict(nn_correlation=nn, energy_per_site=-2 * nn, exact_infinite_nn=exact_nn(J), nn_minus_exact=nn - exact_nn(J), staggered_m=abs(m), signed_m=m, largest_cluster=size, cluster_fraction=size / sig.size, cluster_ties=len(big), largest_square_side=side, square_start=starts[0], square_ties=len(starts), open_boundary_cluster=int(cts.max()), axis_e_fold_length=axis_e_fold_length(sig), pattern_sha256=phash(sig))

def sample(J, L, seed, start='random', sweeps=3000, record=False):
    rng = np.random.default_rng([20261026, 14, int(round(J * 1000)), seed])
    sig = rng.choice([-1, 1], size=(L, L)) if start == 'random' else np.ones((L, L), dtype=int)
    mask = (np.indices((L, L)).sum(0) % 2).astype(bool)
    trace = []
    ends = {}
    for step in range(1, sweeps + 1):
        for m in (mask, ~mask):
            h = np.roll(sig, 1, 0) + np.roll(sig, -1, 0) + np.roll(sig, 1, 1) + np.roll(sig, -1, 1)
            p = 1 / (1 + np.exp(-2 * J * h))
            new = np.where(rng.random((L, L)) < p, 1, -1)
            sig = np.where(m, new, sig)
        if record and step % 10 == 0:
            nn, mag = basic(sig)
            trace.append((step, nn, mag, abs(mag)))
        if record and step in (3000, 30000):
            ends[step] = sig.copy()
    return (sig, np.asarray(trace), ends)

def autocorrelation(x, spacing=10):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if float(x @ x) == 0:
        return dict(tau_integrated_sweeps=0.0, constant=True)
    nfft = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(fft * np.conj(fft), nfft)[:n] / np.arange(n, 0, -1)
    ac /= ac[0]
    cutoff = 1
    while cutoff < min(n // 5, 501) and ac[cutoff] > 0:
        cutoff += 1
    tau = 0.5 + float(ac[1:cutoff].sum())
    return dict(tau_integrated_sweeps=spacing * tau, cutoff_sweeps=spacing * cutoff, effective_sample_size=n / (2 * tau), lag_sweeps=[spacing * i for i in (1, 2, 5, 10, 20, 50, 100)], rho=[float(ac[i]) for i in (1, 2, 5, 10, 20, 50, 100)], method='first-positive-lobe sum, samples every 10 sweeps; short-time correlations unresolved')

def trace_stats(trace):
    blocks = []
    for end in range(1000, 30001, 1000):
        x = trace[(trace[:, 0] > end - 1000) & (trace[:, 0] <= end)]
        blocks.append(dict(end_sweep=end, nn_mean=float(x[:, 1].mean()), signed_m_mean=float(x[:, 2].mean()), abs_m_mean=float(x[:, 3].mean())))
    tail = trace[trace[:, 0] > 3000]
    return dict(blocks=blocks, post_3000=dict(nn_mean=float(tail[:, 1].mean()), abs_m_mean=float(tail[:, 3].mean()), autocorrelation={key: autocorrelation(tail[:, col]) for key, col in [('nn', 1), ('signed_m', 2), ('abs_m', 3)]}))

def state_path(name):
    return STATE / (name + '.npz')

def references(job, limit=None):
    computed = 0
    out = job.out
    out.setdefault('references', [])
    done = {r['id'] for r in out['references']}
    ising = read(_DATA_ROOT / 'truchet_2d/ising.json')['runs']
    convention_inputs = read(DATA_ROOT / 'truchet_2d/correlated.json')['runs']
    combined = {}
    for source, rows in [('pressure convention', convention_inputs), ('ising', ising)]:
        for r in rows:
            key = (r.get('L', 64), r['J'], r['sample'])
            combined.setdefault(key, {})[source] = r
    for (L, J, seed), sources in sorted(combined.items()):
        ident = f'reference_L{L}_J{round(J * 1000)}_s{seed}'
        if ident in done and state_path(ident).exists():
            continue
        if not job.limit_ok():
            return False
        cpu = time.process_time()
        if 'ising' in sources:
            from computations.truchet_2d.ising_scan import ensure_state
            reference = ensure_state(sources['ising'])
            a = np.load(reference)
            sig = a['sigma']
            te = a['tile_energy']
        else:
            sig, _, _ = sample(J, L, seed)
            te = None
        obs = observables(sig, J)
        checks = {}
        for source, r in sources.items():
            checks[source] = dict(staggered_m_error=obs['staggered_m'] - r['staggered_m'])
            if source == 'pressure convention':
                checks[source].update(open_cluster_error=obs['open_boundary_cluster'] - r['largest_cluster'], xi_error=obs['axis_e_fold_length'] - r['xi'])
            else:
                checks[source].update(pattern_hash_match=obs['pattern_sha256'] == r['pattern_sha256'], cluster_error=obs['largest_cluster'] - r['largest_cluster'], square_error=obs['largest_square_side'] - r['largest_square_side'])
        assert all((abs(c['staggered_m_error']) < 1e-14 for c in checks.values()))
        assert all((c.get('pattern_hash_match', True) and c.get('open_cluster_error', 0) == 0 and (c.get('xi_error', 0) == 0) and (c.get('cluster_error', 0) == 0) and (c.get('square_error', 0) == 0) for c in checks.values()))
        args = dict(sigma=sig.astype(np.int8))
        if te is not None:
            args['tile_energy'] = te
        np.savez_compressed(state_path(ident), **args)
        out['references'] = [r for r in out['references'] if r['id'] != ident]
        out['references'].append(dict(id=ident, L=L, J=J, seed=seed, sources=sources, observables=obs, verification=checks, state_file=str(state_path(ident).relative_to(OUT)), cpu_seconds=time.process_time() - cpu))
        job.checkpoint()
        computed += 1
        print('COMPUTE endpoint', ident, flush=True)
        if limit and computed >= limit:
            return False
    return True

def summarize_chains(rows):
    result = []
    for L in (64, 128):
        for J in (0.38, 0.42):
            for start in ('random', 'ordered'):
                z = [r for r in rows if (r['L'], r['J'], r['start']) == (L, J, start)]
                if not z:
                    continue
                for end in ('3000', '30000'):
                    vals = [r['endpoints'][end] for r in z]
                    result.append(dict(L=L, J=J, start=start, sweep=int(end), **{key: summary([a[key] for a in vals]) for key in ('nn_correlation', 'nn_minus_exact', 'staggered_m', 'largest_cluster', 'cluster_fraction', 'largest_square_side')}))
    return result

def augment_reference_traces(job):
    """Autocorrelation cannot be recovered from an endpoint: regenerate its chain."""
    out = job.out
    for r in out['references']:
        if 'short_chain_diagnostics' in r:
            continue
        if not job.limit_ok():
            return
        cpu = time.process_time()
        sig, trace, _ = sample(r['J'], r['L'], r['seed'], sweeps=3000, record=True)
        assert phash(sig) == r['observables']['pattern_sha256']
        tail = trace[trace[:, 0] > 1000]
        blocks = []
        for end in (1000, 2000, 3000):
            z = trace[(trace[:, 0] > end - 1000) & (trace[:, 0] <= end)]
            blocks.append(dict(end_sweep=end, nn_mean=float(z[:, 1].mean()), signed_m_mean=float(z[:, 2].mean()), abs_m_mean=float(z[:, 3].mean())))
        r['short_chain_diagnostics'] = dict(blocks=blocks, window='sweeps 1010-3000; short diagnostic, not an equilibration certificate', autocorrelation={key: autocorrelation(tail[:, col]) for key, col in [('nn', 1), ('signed_m', 2), ('abs_m', 3)]}, endpoint_hash_reverified=True, cpu_seconds=time.process_time() - cpu)
        job.checkpoint()
    for r in out['runs']:
        trace = np.load(OUT / r['state_file'])['trace']
        tail = trace[(trace[:, 0] > 1000) & (trace[:, 0] <= 3000)]
        r['early_autocorrelation_1000_3000'] = {key: autocorrelation(tail[:, col]) for key, col in [('nn', 1), ('signed_m', 2), ('abs_m', 3)]}
    groups = {}
    for r in out['references']:
        for source in r['sources']:
            groups.setdefault((source, r['L'], r['J']), []).append(r)
    out['reference_summary'] = []
    for (source, L, J), z in sorted(groups.items()):
        out['reference_summary'].append(dict(source=source, L=L, J=J, n=len(z), **{key: summary([r['observables'][key] for r in z]) for key in ('nn_correlation', 'nn_minus_exact', 'energy_per_site', 'staggered_m', 'largest_cluster', 'cluster_fraction', 'largest_square_side')}, autocorrelation={key: summary([r['short_chain_diagnostics']['autocorrelation'][key]['tau_integrated_sweeps'] for r in z]) for key in ('nn', 'signed_m', 'abs_m')}))
    out['definitions']['reference_autocorrelation'] = 'All reference chains deterministically regenerated and hash-matched; autocorrelation from sweeps 1010-3000, not from a lone snapshot.'
    job.finish()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    ap.add_argument('--augment-reference-traces', action='store_true')
    args = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    job = Job('truchet_2d/ising_sampling.json')
    out = job.out
    if args.augment_reference_traces:
        augment_reference_traces(job)
        return
    out['definitions'] = dict(algorithm='checkerboard heat bath, one sweep updates both sublattices', seeds='[20261026,14,round(1000*J),seed]; random starts reproduce ising/pressure convention; ordered starts are all +1', energy='energy per site in units of exchange = -2 nearest-neighbour correlation', exact='Onsager infinite square lattice, diagnostic only; finite-box corrections allowed', geometry='nearest-neighbour clusters with both torus seams joined; periodic axis-aligned uniform squares', trace='every 10 sweeps; 1000-sweep blocks; autocorrelation after sweep 3000')
    if not references(job, args.limit):
        return
    done = {r['id'] for r in out['runs']}
    n = 0
    for L in (64, 128):
        for J in (0.38, 0.42):
            for start in ('random', 'ordered'):
                for seed in range(6):
                    ident = f'chain_L{L}_J{round(J * 1000)}_{start}_s{seed}'
                    if ident in done and state_path(ident).exists():
                        continue
                    if not job.limit_ok():
                        return
                    cpu = time.process_time()
                    wall = time.perf_counter()
                    print('start', ident, flush=True)
                    sig, trace, ends = sample(J, L, seed, start, 30000, True)
                    endpoints = {str(k): observables(v, J) for k, v in ends.items()}
                    reference_match = None
                    if start == 'random':
                        orig = next((r for r in out['references'] if (r['L'], r['J'], r['seed']) == (L, J, seed)))
                        reference_match = endpoints['3000']['pattern_sha256'] == orig['observables']['pattern_sha256']
                        assert reference_match
                    np.savez_compressed(state_path(ident), sigma_3000=ends[3000].astype(np.int8), sigma_30000=ends[30000].astype(np.int8), trace=trace)
                    out['runs'].append(dict(id=ident, L=L, J=J, start=start, seed=seed, endpoints=endpoints, reference_3000_hash_match=reference_match, **trace_stats(trace), state_file=str(state_path(ident).relative_to(OUT)), cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall))
                    out['summary'] = summarize_chains(out['runs'])
                    job.checkpoint()
                    print('done', len(out['runs']), round(time.process_time() - cpu, 2), flush=True)
                    n += 1
                    if args.limit and n >= args.limit:
                        return
    job.finish()
if __name__ == '__main__':
    main()
