"""Low frequency time domain (transient) eddy current solver interface for oersted"""

from .solver import DEFAULT_SETTINGS

from .mesh import Mesh
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from ._oersted import transient_solve as _transient_solve

class TransientResults:
    time: NDArray[float64]
    j: NDArray[float64]
    a: NDArray[float64]
    b: NDArray[float64]

    def __init__(self, time, j, a, b):
        self.time = time 
        self.j = j 
        self.a = a 
        self.b = b


def transient_solve(
    mesh: Mesh,
    rho: float,
    dt: float,
    tmax: float,
    a_ext: NDArray[float64],
    b_ext: NDArray[float64],
    settings=DEFAULT_SETTINGS,
) -> TransientResults:
    """ Solve a low-frequency (time-domain) eddy current problem
    """

    assert dt > 0.0 and tmax > 0.0 and rho > 0.0 
    assert b_ext.ndim == 3 and a_ext.ndim == 3 
    assert b_ext.shape[2] == 3 and a_ext.shape == b_ext.shape 

    (time, j, a, b) = _transient_solve(mesh.nodes, mesh.connectivity, rho, dt, tmax, a_ext, b_ext)

    return TransientResults(time, j, a, b) 
