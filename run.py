"""Run released computations in an isolated, writable numerical workspace.

Native scripts retain their production arguments. The source archive stays
read-only; current-directory outputs live under the selected cache.
"""
from pathlib import Path
import argparse, os, shutil, subprocess, sys, uuid

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True, type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('script', help='release-relative script inside computations/')
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    src = Path(__file__).resolve().parent / 'computations'
    target = (src / a.script).resolve()
    if not target.is_relative_to(src) or not target.is_file() or target.suffix != '.py':
        p.error('Unknown released computation script')
    cache = a.cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    out = (a.output or cache / 'outputs').expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = cache / 'runs' / uuid.uuid4().hex[:12] / 'computations'
    shutil.copytree(src, work, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(src.parent / 'SHA256SUMS.json', work.parent / 'SHA256SUMS.json')
    # Continuation drivers can generate new states from the bundled initial guesses.
    states = work / 'trumof' / 'initialization'
    states.mkdir(parents=True, exist_ok=True)
    for f in (work / 'trumof/mechanics/inputs/continuation').glob('*.npz'):
        shutil.copy2(f, states / f.name)
    env = dict(os.environ, K08_CACHE=str(cache), K08_OUTPUT=str(out), PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    env.pop('PYTHONPATH', None)
    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[k] = '1'
    print('WORKSPACE', work, flush=True)
    print('OUTPUT', out, flush=True)
    args = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    result = subprocess.run([sys.executable, '-B', str(work / a.script), *args], cwd=work, env=env)
    raise SystemExit(result.returncode)
if __name__ == '__main__':
    main()
