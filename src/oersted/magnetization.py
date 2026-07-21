"""Operations for magnetic materials"""

from oersted import SolverSettings

from numpy.typing import NDArray
from numpy import float64, ascontiguousarray

from .mesh import Mesh
from .materials import Material
from ._oersted import magnetization_solve


def demag_solve(
    mesh: Mesh,
    material: Material,
    h_external: NDArray[float64],
    settings: SolverSettings,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Compute magnetization field M and the total H field at element centroids,
        given a background field

    Uses simple fixed-point iteration and therefore only converges for
        low-permeable materials.

    Args:
        mesh: finite element mesh on which to evaluate the demagnetizing field
        material: linear or nonlinear magnetic maaterial properties
        h_external: (A/m) an (Ne,3) array of external field at each element centroid
        solver: solution parameters for the problem, including iteration method

    Returns:
        (M, Htotal): each (Ne, 3), magnetization field M(Htotal) and total H field
        at element centroids. These can be summed to give B = mu0 * (Htotal + M).
    """

    element_integration = settings.integration == "element"
    use_octree = settings.method == "octree"

    return magnetization_solve(
        ascontiguousarray(mesh.nodes),
        ascontiguousarray(mesh.connectivity),
        ascontiguousarray(mesh.centroids),
        material.chi(1.0),
        ascontiguousarray(h_external),
        element_integration,
        settings.n_threads,
        settings.atol,
        settings.max_iterations,
        settings.under_relaxation_factor,
        use_octree,
        settings.theta,
        settings.expansion_order(),
        settings.max_leaf_size,
        settings.batch_size,
    )
