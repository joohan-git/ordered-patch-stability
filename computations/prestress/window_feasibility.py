"""S4 feasibility: verify IMS algebra and test one allowed no-checker crystal.

A universal certificate must include this allowed example.
Uses <= 1 process, one BLAS thread.
"""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.diagnostics import *

def windows(Q, x):
    D = sp.diags(x)
    D2 = sp.diags(x * x)
    localized = D @ Q @ D
    double = D2 @ Q - 2 * localized + Q @ D2
    return (localized - 0.5 * double, localized + 0.5 * double)

def main():
    start = time.perf_counter()
    out = dict(metadata=metadata(), runs=[])
    Q = sp.csr_matrix(np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]))
    x = np.array([1.0, 0.6, 0.0])
    y = np.sqrt(1 - x * x)
    minus1, plus1 = windows(Q, x)
    minus2, plus2 = windows(Q, y)
    out['algebra'] = dict(Q=Q.toarray(), chi1=x, chi2=y, minus_commutator_sum=(minus1 + minus2).toarray(), plus_commutator_sum=(plus1 + plus2).toarray(), minus_commutator_relative_error=float(sla.norm(minus1 + minus2 - Q) / sla.norm(Q)), plus_commutator_relative_error=float(sla.norm(plus1 + plus2 - Q) / sla.norm(Q)))
    L = 24
    f = Frame2D(np.zeros((L, L), int))
    f.hydro()
    base = json.loads((DATA_ROOT / 'truchet_2d/planted_pressure.json').read_text())['baseline']['pressure']
    for R in (2, 4, 6, 8):
        dist = np.max(np.abs(f.pos - L / 2), axis=1)
        chi_node = np.where(dist < R, np.cos(np.pi * np.minimum(dist / R, 1) / 2) ** 2, 0.0)
        chi = np.repeat(chi_node, 3)
        for ratio in (0.0, 0.5, 1.0, 1.224):
            Q = f.K - ratio * base / f.p0 * f.H
            for label, W in zip(('minus_commutator', 'plus_commutator'), windows(Q, chi)):
                W = W.tocsr()
                W.eliminate_zeros()
                ids = np.flatnonzero(np.asarray(abs(W).sum(axis=1)).ravel() > 1e-30)
                S = W[ids, :][:, ids].tocsc()
                scale = 1 / np.sqrt(f.K.diagonal()[ids])
                D = sp.diags(scale)
                S = D @ S @ D
                S = S.tocsr()
                co = S.tocoo()
                diagonal = S.diagonal()
                good = (diagonal[co.row] == 0) & (co.row != co.col) & (np.abs(co.data) > 1e-30)
                candidates = np.flatnonzero(good)
                assert len(candidates) > 0
                q = candidates[np.argmax(np.abs(co.data[candidates]))]
                pair = np.array([co.row[q], co.col[q]])
                minor = S[pair, :][:, pair].toarray()
                ev, vec = la.eigh(minor)
                trial = np.zeros(len(ids))
                trial[pair] = vec[:, 0]
                rayleigh = float(trial @ (S @ trial))
                assert rayleigh < 0
                out['runs'].append(dict(window_radius=R, pressure_ratio=ratio, operator=label, negative_trial_rayleigh_scaled=rayleigh, principal_minor=minor, principal_minor_eigenvalue=float(ev[0]), dofs=len(ids)))
    out['conclusion'] = 'The minus-commutator sum does not equal Q. The plus-commutator windows sum to Q but are indefinite at zero load for an allowed uniform crystal; these windows provide no positive stability certificate.'
    out['candidate_comparison'] = 'No certified lambda* exists for this construction, so a numerical gap to 1.224 is undefined (not a lower bound of zero).'
    out['seconds'] = time.perf_counter() - start
    save('prestress/windows.json', out)
    print(json.dumps(out, default=plain, indent=2), flush=True)
if __name__ == '__main__':
    main()
