"""Consistent element adapters for the saved 2D, 3D and inclusion solvers."""
from pathlib import Path as _Path
import sys as _sys
_DATA_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_DATA_ROOT.parent))
import numpy as np
import scipy.linalg as la
from computations.common.initial_stress import geometric_full
from computations.truchet_2d.frame import Frame2D as AxialFrame2D, PLANAR
from computations.frame_3d.frame import Frame3D as AxialFrame3D
from computations.inclusions.soft_frame import InclusionFrame as AxialInclusionFrame

def batch_geo(delta, ef):
    planar = ef.shape[1] == 6
    indices = np.asarray(PLANAR) if planar else np.arange(12)
    ds = np.column_stack((delta, np.zeros(len(delta)))) if planar else delta
    unique, inverse = np.unique(np.round(ds, 11), axis=0, return_inverse=True)
    result = np.zeros((len(ef), len(indices), len(indices)))
    for h, d in enumerate(unique):
        select = np.flatnonzero(inverse == h)
        d = ds[select[0]]
        ops = []
        for i in indices:
            force = np.zeros(12)
            force[i] = 1.0
            ops.append(geometric_full(np.zeros(3), d, force)[np.ix_(indices, indices)])
        result[select] = np.einsum('ea,aij->eij', ef[select], np.asarray(ops))
    return result

def replace_geo(f, actual):
    ef = np.einsum('eij,ej->ei', f.Ke, actual)
    f.pre['full_element_forces'] = ef
    f.He = -batch_geo(f.delta, ef)
    f.H = f.assemble(f.He)
    f.Hf = f.H[f.free, :][:, f.free].tocsc()

class Frame2D(AxialFrame2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replace_geo(self, -self.affine[0] - self.affine[1] + self.pre['w'][self.dofs])

    def hydro(self):
        ep = la.solve(self.C, np.array([-1.0, -1.0, 0.0]))
        ep /= -(ep[0] + ep[1]) / 2
        imposed = np.einsum('a,aei->ei', ep, self.affine)
        self.pre = self.relaxed(imposed)
        replace_geo(self, imposed + self.pre['w'][self.dofs])
        self.p0 = float(-np.trace(self.pre['stress']) / 2)
        return self.p0

class Frame3D(AxialFrame3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        imposed = np.zeros_like(self.pre['w'][self.dofs])
        imposed[:, 6:9] = self.delta @ self.pre['strain'].T
        replace_geo(self, imposed + self.pre['w'][self.dofs])

class InclusionFrame(AxialInclusionFrame):

    def preload(self, protocol):
        super().preload(protocol)
        replace_geo(self, self.pre['actual'])
        return self.pre
