"""Verify all 414 numerical checkpoints, then publish JSON."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.atlas.mesh_common import *
import computations.atlas.mesh_aggregate as agg
from computations.atlas.mesh_verify import verify

def finish(name, out):
    import scipy, sys
    checks = verify(out)
    out['verification']['independent_checks'] = checks
    out['verification']['max_independent_bloch_root_relative_difference'] = max(out['verification']['max_final_bloch_sparse_relative_error'], checks['near_gamma_full_dense_sparse_max_relative_difference'])
    out['environment'] = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__)
    out['protocol']['arithmetic'] += ' Rows using the full sparse generalized eigenproblem are validated against full dense roots and listed in verification.independent_checks.full_sparse_rows.'
    out['source_sha256'].update({s: sha(ROOT / s) for s in ['atlas/mesh_retry.py', 'atlas/mesh_verify.py', 'atlas/mesh_finalize.py', 'atlas/mesh_dense_check.py']})
    out['summary_4_to_8'] = {k: stats([abs(x['relative_change_4_to_8'][k]) for x in out['rows'] if x['relative_change_4_to_8'][k] is not None]) for k in agg.THRESHOLDS}
    out['baseline_pdonly_baseline_error'] = {k: stats([abs(x['recomputed_d2_threshold_relative_change'][k]) for x in out['baseline_baseline_comparison'] if x['name'] in out['cohort']['pdonly'] and x['recomputed_d2_threshold_relative_change'][k] is not None]) for k in agg.THRESHOLDS}
    publish(name, out)
    publish('atlas/mesh_verification.json', checks)
if __name__ == '__main__':
    import sys
    if '--validate-only' in sys.argv:
        from computations.atlas.mesh_verify import validate_only
        validate_only()
        raise SystemExit(0)
    sys.argv = [sys.argv[0], '--publish']
    agg.publish = finish
    agg.main()
