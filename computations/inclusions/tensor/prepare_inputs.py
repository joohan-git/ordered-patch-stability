from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from pathlib import Path
import json, hashlib
HERE = Path(__file__).resolve().parent
ROOT = _DATA_ROOT
import sys
sys.path.insert(0, str(ROOT))
from computations.common.runtime import output_dir
SOURCE = HERE / 'source_inputs'
rd = lambda p: json.loads(p.read_text(encoding='utf-8'))
inc = rd(_DATA_ROOT / 'inclusions/soft_inclusions.json')
atlas = rd(_DATA_ROOT / 'atlas/bloch_rows.json')
out = {'source_note': 'Earlier inclusion preload uses the same elastic geometry, stiffness and hydrostatic preload as the production beam tangent run; the later change affects the instability tangent only. Stress-factor identity is verified below.', 'cases': []}
for name in ['cco', 'vne']:
    row = max((x for x in inc['runs'] if x['name'] == name), key=lambda x: (x['L'], x['R_fraction']))
    src = SOURCE / f"inc_{name}_L{row['L']}_R{row['R_fraction']:.3f}_d2.json"
    src = src.with_name(src.name.replace(f"R{row['R_fraction']:.3f}", f"R{row['R_fraction']:.3f}".replace('.', 'p')))
    reference = rd(src)
    pre = next((x for x in reference['protocols'] if x['protocol'] == 'pressure'))
    con = pre['concentration']
    assert abs(con['s_P_global'] / row['s_P'] - 1) < 1e-12
    a = next((x for x in atlas if x['name'] == name and x['r_over_each_parent_length'] == 0.02))
    out['cases'].append({'name': name, 'L': row['L'], 'R_fraction': row['R_fraction'], 'onset_pressure': row['pressure'], 's_P': row['s_P'], 'scalar_rho_loc': row['rho_loc'], 'scalar_rho_PD': row['rho_PD'], 'patch_stress_per_reference_factor': con['patch_stress'], 'applied_pressure_per_reference_factor': pre['pressure_per_factor'], 'preload_residual': pre['preload_residual'], 'source_deviatoric_fraction': con['patch_stress_deviatoric_ratio'], 'A': reference['A'], 'atlas_graph_hash': a.get('graph_sha256'), 'bulk_hydro_loc': min((x for x in [a['lambda_LE_pressure'], a['lambda_micro_pressure']] if x is not None)), 'source': src.name, 'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest()})
(output_dir() / 'inclusions/tensor/inputs.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
