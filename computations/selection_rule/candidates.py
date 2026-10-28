"""Check whether the familiar candidate nets really fail at finite k."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.selection_rule.mode_common import *
from computations.atlas.bloch_search import scan

def main():
    rows = []
    for name in ['dia', 'srs', 'acs']:
        p = DEST / f'selection_rule/scan_{name}.json'
        if p.exists():
            x = read(p)
        else:
            x = scan(frame(name, 0.02), raw(name, 0.02))
            save(p, x)
        m = x['micro']
        gamma = m['gamma_optical']
        q = np.asarray(m['k'])
        row = dict(name=name, atlas=atlas13(name), scan=x, finite_k_first=bool(la.norm(q) > 1e-07 and (not m['near_excluded_origin']) and (m['load'] < gamma * (1 - 1e-05))), gamma_over_minimum=gamma / m['load'], k=q)
        rows.append(row)
        print(name, 'k', q, 'gamma/min', row['gamma_over_minimum'], 'finite-k-first', row['finite_k_first'], flush=True)
    chosen = next((x['name'] for x in rows[:2] if x['finite_k_first']), None)
    if chosen is None:
        assert rows[2]['finite_k_first']
        chosen = 'acs'
    out = dict(selected=chosen, candidates=rows, other_fallbacks=[dict(name=n, atlas=atlas13(n), reason='PD+LE first, not finite-wavelength-first') for n in ['nbo', 'ths']], model=raw('cco', 0.02)['model'], atlas_sha256=sha(ATLAS_PATH), threadpools=POOLS)
    publish('selection_rule/candidates.json', out)
if __name__ == '__main__':
    main()
