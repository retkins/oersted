"""Operations for magnetic materials"""

from oersted import DirectSolver, OctreeSolver

from numpy.typing import NDArray
from numpy import float64, uint32, ascontiguousarray

from .mesh import Mesh
from .materials import Material
from ._oersted import magnetization_tet4


def demag_solve(
    mesh: Mesh,
    material: Material,
    h_external: NDArray[float64],
    solver: DirectSolver | OctreeSolver,
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

    theta: float
    leaf_threshold: uint32

    if isinstance(solver, DirectSolver):
        theta = 0.5
        leaf_threshold = uint32(0)
    else:
        theta = solver.theta
        leaf_threshold = solver.leaf_threshold

    return magnetization_tet4(
        ascontiguousarray(mesh.nodes),
        ascontiguousarray(mesh.connectivity),
        material.chi(1.0),
        ascontiguousarray(h_external),
        solver.tol,
        solver.max_iterations,
        theta,
        leaf_threshold,
        solver.alpha,
        solver.n_threads,
        solver.edge,
    )
