"""Operations for magnetic materials"""

from numpy.typing import NDArray
from numpy import float64

from .mesh import Mesh
from .materials import Material
from ._oersted import magnetization_tet4


def demag_tet4(
    mesh: Mesh,
    material: Material,
    h_external: NDArray[float64],
    max_iterations: int = 50,
    tol: float = 1.0,
    theta: float = 0.5,
    leaf_threshold: int = 0,
    nthreads_requested: int = 0,
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
    return magnetization_tet4(
        mesh.nodes, mesh.connectivity, material.chi(1.0), h_external, tol, max_iterations, theta, leaf_threshold, nthreads_requested
    )
