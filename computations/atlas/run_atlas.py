"""Two offline workers, BLAS=1, atomic per-net/radius checkpoints in TEMP."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.atlas_support import *
import argparse
import gc
import subprocess
import time
import traceback
import numpy as np
import scipy.linalg as la
import gemmi
from computations.validation.beam_validation import ConsistentFrame, ConsistentAtlas
from computations.common.initial_stress import MODEL
from computations.atlas.edge_center_atlas import classify

def threadpools():
    """Query the loaded OpenBLAS DLLs directly; no optional package needed."""
    import ctypes
    import psutil
    pools = []
    for entry in psutil.Process().memory_maps():
        if 'openblas' not in entry.path.lower():
            continue
        lib = ctypes.CDLL(entry.path)
        for symbol in ('scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads', 'openblas_get_num_threads64_', 'openblas_get_num_threads'):
            try:
                fun = getattr(lib, symbol)
            except AttributeError:
                continue
            fun.restype = ctypes.c_int
            pools.append(dict(library=Path(entry.path).name, symbol=symbol, num_threads=fun()))
            break
        else:
            raise RuntimeError('Cannot verify OpenBLAS thread count: ' + entry.path)
    assert pools and all((p['num_threads'] == 1 for p in pools)), pools
    return pools
POOLS = threadpools()

def hydrate(graph):
    return dict(name=graph['name'], frac=np.asarray(graph['frac']), cn=np.asarray(graph['cn']), edges=[(i, j, np.asarray(sh, int)) for i, j, sh in graph['edges']], A=np.asarray(graph['A']), sg=gemmi.find_spacegroup_by_name(graph['space_group']), lengths=graph['lengths'], trials=[dict(loader='load_sym', graph_sha256=graph['graph_sha256'])])

def valid(path, graph, radius):
    try:
        r = read(path)
        return r['model'] == MODEL and r['graph_sha256'] == graph['graph_sha256'] and (r['name'] == graph['name']) and (r['r_over_each_parent_length'] == radius) and all((k in r for k in ['PD_pressure_pV', 'LE', 'micro', 'check', 'invariant_fraction']))
    except (OSError, ValueError, KeyError):
        return False

def run_one(graph, radius):
    start = time.perf_counter()
    f = ConsistentFrame(graph['name'], radius, 2, hydrate(graph))
    at = ConsistentAtlas(f)
    pd, le, mu = (at.pd(True), at.le(), at.bloch(4))
    E = np.asarray(pd['strain_mode'])
    qs = []
    for _, _, q in f.operations():
        if not any((np.allclose(q, x, atol=1e-07) for x in qs)):
            qs.append(q)
    inv = float(np.sum(E * sum((q @ E @ q.T for q in qs)) / len(qs)) / np.sum(E * E))
    probe = 0.4 * min(pd['bound'], at.gamma)
    M = at.K - probe * at.Hpressure
    nf = at.nf
    direct = (M[nf:, nf:] - M[nf:, :nf] @ la.solve(M[:nf, :nf], M[:nf, nf:], assume_a='sym')) / at.vol
    macro = at.tangent(probe, True)
    C = at.gibbs(0)
    stiffness_sym = max((float(la.norm((at.S.T @ np.kron(q, q) @ at.S).T @ C @ (at.S.T @ np.kron(q, q) @ at.S) - C) / la.norm(C)) for q in qs))
    prestrain = f.pre['strain']
    check = dict(ward=float(la.norm(M @ at.R) / (la.norm(M) * la.norm(at.R))), elastic_ward=float(la.norm(at.K @ at.R) / (la.norm(at.K) * la.norm(at.R))), initial_stress_ward=float(la.norm(at.Hpressure @ at.R) / (la.norm(at.Hpressure) * la.norm(at.R))), rotation_energy=float(la.norm(at.R.T @ M @ at.R) / (la.norm(M) * la.norm(at.R) ** 2)), schur=float(la.norm(direct - macro) / la.norm(direct)), zero_load_stiffness_symmetry=stiffness_sym, prestrain_symmetry=max((float(la.norm(q @ prestrain @ q.T - prestrain) / la.norm(prestrain)) for q in qs)), preload_equilibrium=float(f.pre['relative_residual']))
    return dict(model=MODEL, loader='rcsr_loader.load_sym (frozen common/rcsr_loader.py)', graph_sha256=graph['graph_sha256'], name=f.name, r_over_each_parent_length=radius, div=2, nodes=graph['nodes'], edges=graph['edge_count'], space_group=graph['space_group'], crystal_system=graph['crystal_system'], point_group_order=len(qs), p0=at.p0, PD_pressure_pV=pd, LE=le, micro=mu, class_pressure_pV=classify(pd, le, mu), invariant_fraction=inv, lambda_PD_pressure=None if pd['load'] is None else pd['load'] * at.p0, lambda_LE_pressure=None if le['load'] is None else le['load'] * at.p0, lambda_micro_pressure=None if mu['near_excluded_origin'] else mu['load'] * at.p0, sampled_micro_upper_pressure=mu['load'] * at.p0, check_probe=probe, check=check, threadpools=POOLS, seconds=time.perf_counter() - start)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--chunks', type=int, default=1)
    ap.add_argument('--nets', nargs='+')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    storage()
    if not 1 <= args.workers <= 2:
        ap.error('At most two worker processes')
    if args.workers == 2:
        children = []
        started = time.time()
        for k in range(2):
            log = (DEST / f'worker{k}.log').open('w', encoding='utf-8')
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--workers', '1', '--chunk', str(k), '--chunks', '2']
            if args.resume:
                cmd += ['--resume']
            if args.nets:
                cmd += ['--nets', *args.nets]
            p = subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            children.append((p, log))
        dump(DEST / 'job.json', dict(pids=[p.pid for p, _ in children], started=started, **storage()))
        print('START', DEST, [p.pid for p, _ in children], flush=True)
        codes = [p.wait() for p, _ in children]
        for _, log in children:
            log.close()
        dump(DEST / 'completion.json', dict(exit_codes=codes, wall_seconds=time.time() - started))
        print('DONE', codes, flush=True)
        if any(codes):
            raise SystemExit(1)
        return
    from computations.atlas.bootstrap import bootstrap
    cohort = sorted({r['name'] for r in read(ROOT / 'atlas/baseline_rows.json')})
    selected = args.nets or cohort
    if set(selected) - set(cohort):
        ap.error('Select a net in the final 876-net atlas; lcw and qcx are catalogue-only exclusions')
    bootstrap(selected)
    graphs = read(ROOT / 'atlas/catalog.json')['graphs']
    graphs = [g for g in graphs if g['name'] in selected]
    graphs = graphs[args.chunk::args.chunks]
    errors = []
    done = 0
    for meta in graphs:
        graph = read(DEST / 'graphs' / f"{meta['name']}.json")
        for radius in RADII:
            target = result_path(graph['name'], radius)
            try:
                if args.resume and valid(target, graph, radius):
                    done += 1
                    continue
                dump(DEST / f'active{args.chunk}.json', dict(name=graph['name'], radius=radius, started=time.time()))
                row = run_one(graph, radius)
                dump(target, row)
                done += 1
                print(done, '/', 2 * len(graphs), graph['name'], radius, row['class_pressure_pV'], round(row['seconds'], 2), flush=True)
                del row
                gc.collect()
            except Exception:
                error = dict(name=graph['name'], radius=radius, traceback=traceback.format_exc())
                errors.append(error)
                dump(DEST / f'errors{args.chunk}.json', errors)
                print('ERROR', error, flush=True)
    dump(DEST / f'errors{args.chunk}.json', errors)
    if errors:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
