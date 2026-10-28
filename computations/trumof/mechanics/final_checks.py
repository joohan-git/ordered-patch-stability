from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_DATA_ROOT.parent))
from computations.trumof.mechanics.verify import *
from computations.trumof.mechanics.model.model import angle_terms
import hashlib
start = time.process_time()
out = {}
sliding = []
for z in SUMMARY_SEED['sliding_mode']:
    rr = next((r for r in CONVENTIONS_SEED['rows'] if r['name'] == 'trumof_1x1x1_1' and r['parameters'] == z['parameters']))
    e = Engine(rr)
    m = e.m
    p = z['p_GPa']
    x = e.load(p)
    k = np.array(z['k'])
    ee = m.e
    phase = np.exp(2j * np.pi * (ee[:, 2:] @ k))
    B = sparse.coo_matrix((np.c_[-np.ones(len(ee)), phase].ravel(), (np.repeat(np.arange(len(ee)), 2), ee[:, :2].ravel())), shape=(len(ee), m.N)).toarray()
    Q = np.kron(linalg.null_space(B), np.eye(3))
    H = m.bloch(x, k).toarray()
    new = dict(z)
    new.update(exact_kernel_dimension=Q.shape[1], kernel_residual=float(np.linalg.norm(H @ Q)), kernel_energy_eigenvalues=linalg.eigvalsh(Q.conj().T @ H @ Q), minimum_full_eigen=float(linalg.eigvalsh(H)[0]), nearby=[])
    for direction in z['nearby']:
        d = np.array(direction['direction'])
        ev = linalg.eigvalsh(m.bloch(x, k + 0.001 * d).toarray())[0]
        new['nearby'].append(dict(direction=d, minimum_eigen=float(ev)))
    assert Q.shape[1] == 3 and new['kernel_residual'] < 1e-08
    sliding.append(new)
out['sliding_mode'] = sliding
rr = next((r for r in CONVENTIONS_SEED['rows'] if r['name'] == 'trumof_1x1x1_7' and r['parameters'] == 'soft'))
e = Engine(rr)
unit = e.m
L = 2
a = APP[rr['name']]
pos = lattice(L, RULE)[0] * RULE['a']
m = CG(pos, np.eye(3) * L * RULE['a'], graph(np.tile(a['q'], L ** 3), RULE, L), unit.kr, unit.kt)
p = 0.01
xu = e.load(p)
xx = np.r_[np.tile(xu[:-6], L ** 3), xu[-6:]]
fixedth = m.th.copy()
m.th = np.arccos(np.clip(np.einsum('ni,ni->n', m.ad[:, :3], m.ad[:, 3:]) / (np.linalg.norm(m.ad[:, :3], axis=1) * np.linalg.norm(m.ad[:, 3:], axis=1)), -1, 1))
oldth = m.th.copy()
K = m.tangent(xx, p)[1]
oldmin = float(linalg.eigvalsh(K[m.free][:, m.free].toarray())[0])
m.th = fixedth
K = m.tangent(xx, p)[1]
newmin = float(linalg.eigvalsh(K[m.free][:, m.free].toarray())[0])
vals = linalg.eigvalsh(K.toarray())
folded = np.sort(np.concatenate([linalg.eigvalsh(unit.bloch(xu, np.array(k)).toarray()) for k in itertools.product([0.0, 0.5], repeat=3)]))
out['Pc_L2_folding'] = dict(p_GPa=p, baseline_pinned_min_eigen=oldmin, pinned_min_eigen=newmin, folding_max_abs=float(max(abs(vals - folded))), folding_relative=float(max(abs(vals - folded)) / max(abs(folded))))
assert newmin > 0 and out['Pc_L2_folding']['folding_relative'] < 1e-12
idx = np.flatnonzero((fixedth == np.pi) & (oldth <= np.pi - 1e-08))[0]
w = m.ad[idx:idx + 1] * 0.999
v = np.array([[0.1, 0.2, -0.3, -0.2, 0.1, 0.15]])
v /= np.linalg.norm(v)
f, g, H = angle_terms(w, np.array([np.pi]), m.kt)
pred = float(v[0] @ H[0] @ v[0])
fd = []
for h in [0.001, 0.0003, 0.0001]:
    value = float((angle_terms(w + h * v, np.array([np.pi]), m.kt)[0][0] + angle_terms(w - h * v, np.array([np.pi]), m.kt)[0][0] - 2 * f[0]) / h ** 2)
    fd.append(dict(step=h, predicted=pred, finite_difference=value, relative_error=abs(pred - value) / abs(pred)))
out['angle_derivative'] = fd
assert min((z['relative_error'] for z in fd)) < 1e-06
out['unsnapped_angle_at_equilibrium_root'] = []
for name, kind in [('random_L4_seed403', 'soft'), ('trumof_2x2x2_14', 'hard')]:
    r = next((r for r in CONVENTIONS_SEED['rows'] if r['name'] == name and r['parameters'] == kind))
    m = model(r)
    state = OUT / 'states' / f'{name}_{kind}.npz'
    if not state.exists():
        state = PACKAGE / 'inputs/final_checks' / f'{name}_{kind}.npz'
    st = np.load(state)
    p = float(st['p_PD'])
    x = st['x_PD']
    new = linalg.eigvalsh(m.tangent(x, p)[0])[0]
    m.th = np.arccos(np.clip(np.einsum('ni,ni->n', m.ad[:, :3], m.ad[:, 3:]) / (np.linalg.norm(m.ad[:, :3], axis=1) * np.linalg.norm(m.ad[:, 3:], axis=1)), -1, 1))
    reference = linalg.eigvalsh(m.tangent(x, p)[0])[0]
    out['unsnapped_angle_at_equilibrium_root'].append(dict(name=name, parameters=kind, p_GPa=p, snapped_angle_PD_eigen=float(new), unsnapped_angle_PD_eigen=float(reference)))
out['cpu_seconds'] = time.process_time() - start
dump(OUT / 'numerical_checks.json', out)
manifest = read(PACKAGE / 'input_manifest.json')
for row in manifest:
    assert hashlib.sha256((PACKAGE / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
print('Numerical checks and packaged input identities passed.', len(manifest), flush=True)
