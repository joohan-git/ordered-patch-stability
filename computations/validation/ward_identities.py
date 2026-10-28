"""Small deterministic counterexamples to the spectral/rate/window claims."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.common.diagnostics import *
out = dict(acoustic_anisotropic=[], flat_optical=[])
for R in (8, 16, 32, 64, 128, 256):
    T = 2 * np.eye(R) - np.eye(R, k=1) - np.eye(R, k=-1)
    B = np.zeros((R, R))
    B[0, 0] = B[-1, -1] = 1.0
    T2 = T @ T + B
    cy = 4 * np.sin(np.pi / (2 * (R + 1))) ** 2
    delta = float(la.eigvalsh(T2 + cy * np.eye(R), T, subset_by_index=(0, 0))[0])
    out['acoustic_anisotropic'].append(dict(R=R, delta=delta, R_delta=R * delta, lower_bound=2 * np.sqrt(cy), ratio_to_Rminus2=delta * R ** 2))
    delta_flat = float(la.eigvalsh(T2, subset_by_index=(0, 0))[0])
    out['flat_optical'].append(dict(R=R, delta=delta_flat, R4_delta=delta_flat * R ** 4))
out['definitions'] = dict(acoustic='Phi(k;lambda)=rho_y+(1-lambda)*rho_x+rho_x^2; rho_i=4 sin^2(k_i/2), lambda_LE=1. Along x, up to positive factors: -delta*kx^2+kx^4+...; square-patch excess is Theta(1/R).', optical='Phi(k;lambda)=1-lambda+rho_x^2; isolated minimum has M=0 but finite compact-support excess >0, decaying as R^-4.', interpretation='finite-range bounded quadratic-form counterexamples; not claimed to be the decorated beam frame')
save('validation/ward_identities.json', out)
print(json.dumps(out, indent=2))
