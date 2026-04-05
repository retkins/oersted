"""Operations for magnetic materials"""

import numpy as np
from numpy.typing import NDArray
from numpy import float64

from .mesh import Mesh
from .materials import Material
from ._oersted import h_mag_tet4_direct, magnetization_tet4


def h_demag_tet4(
    mesh: Mesh,
    material: Material,
    m_field: NDArray[float64],
    targets: NDArray[float64] | None = None,
    nthreads_requested: int = 0,
) -> NDArray[float64]:
    """Compute the demagnetization field H(M) on mesh element centroids given the current M-field

    Args:
        nodes: (Nn, 3) nodal coordinates per element
        element_connectivity: (Ne, 4) indices of each node per element;
            these are indices of the array `nodes`, not of the solver's node numbers
        material: linear or nonlinear magnetic maaterial properties
        m_field: (Ne,3) current M-field at each element centroid
        max_iterations: number of solver iterations before exit
        tol: maximum amount of change per individual component of M at each element

    Returns:
        (Ne 3): demagnetization field H(M) at each node
    """

    # Check that the M field is calculated at the element centroids
    try:
        assert mesh.connectivity.shape[0] == m_field.shape[0]
    except AssertionError:
        print("Error. The M-field should be calculated at element centroids.")

    if targets is None:
        targets = mesh.centroids

    return h_mag_tet4_direct(
        np.ascontiguousarray(mesh.nodes),
        np.ascontiguousarray(mesh.connectivity),
        np.ascontiguousarray(m_field),
        np.ascontiguousarray(targets),
        nthreads_requested,
    )


def demag_tet4(
    mesh: Mesh,
    material: Material,
    h_external: NDArray[float64],
    max_iterations: int = 50,
    tol: float = 1.0,
    nthreads_requested: int = 0,
    octree: bool = False,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Compute magnetization field M and the total H field at element centroids, given a background field

    Uses simple fixed-point iteration and therefore only converges for low-permeable materials.

    Args:
        nodes: (Nn, 3) nodal coordinates
        element_connectivity: (Ne, 4) indices of each node per element;
            these are indices of the array `nodes`, not of the solver's node numbers
        material: linear or nonlinear magnetic maaterial properties
        h_external: (Ne,3) external field at each element centroid
        max_iterations: number of solver iterations before exit
        tol: maximum amount of change per individual component of M at each element

    Returns:
        (M, Htotal): each (Ne, 3), magnetization field M(Htotal) and total H field at element
            centroids. These can be summed to give B = mu0 * (Htotal + M)
    """
    return magnetization_tet4(mesh.nodes, mesh.connectivity, material.chi(1.0), h_external, tol, max_iterations, nthreads_requested)
