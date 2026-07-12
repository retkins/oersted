"""Magnetic field calculation routines"""

from numpy import float64, uint32, ascontiguousarray
from numpy.typing import NDArray

# Create bindings for calculation engine written in Rust
from ._oersted import (
    h_current_point_direct,
    h_current_point_octree,
    a_current,
    h_current_tet4_direct,
    h_current_tet4_octree,
    h_mag_tet4,
    h_mag_point,
    h_current_octree,
)

from .mesh import Mesh, CentroidMesh
from .solver import DirectSolver, OctreeSolver, OctreeSolver2Zone, DEFAULT_SOLVER
from .constants import MU0

# For typing; currently unused
Nx3Array = NDArray[float64]


def a_field(
    source: Mesh,
    j_density: NDArray[float64],
    targets: NDArray[float64],
    solver: DirectSolver = DEFAULT_SOLVER,
) -> NDArray[float64]:
    """Compute the magnetic vector potential at a collection of target points"""

    # TODO: make these variable based on selected solver
    exact_integration: bool = True
    n_threads: uint32 = uint32(solver.n_threads)
    use_octree: bool = False
    theta: float = 0.0

    return a_current(
        source.nodes,
        source.connectivity,
        j_density,
        targets,
        exact_integration,
        n_threads,
        use_octree,
        theta,
    )


def b_field(
    source: Mesh | CentroidMesh,
    j_density: NDArray[float64],
    targets: NDArray[float64],
    solver: DirectSolver | OctreeSolver | OctreeSolver2Zone | None = None,
) -> NDArray[float64]:
    """Compute the magnetic flux density at a collection of target points using the
    specific source mesh and solver options, assuming the target points are in free
    space

    Args:
        source: mesh to use as the field source
        j_density: (A/m^2) (N,3) array of current density vectors at each of the
            element centroids
        targets: (m) (N,3) array of target point positions in 3D space
        solver: selects the solution settings

    Returns:
        (T) (N,3) array of magnetic flux density (B) vectors at each target position
    """
    return MU0 * h_field(source, j_density, targets, solver)


def h_field(
    source: Mesh | CentroidMesh,
    j_density: NDArray[float64],
    targets: NDArray[float64],
    solver: DirectSolver | OctreeSolver | OctreeSolver2Zone | None = None,
    edge: bool = False,
) -> NDArray[float64]:
    """Compute the magnetic field strength at a collection of target points using
    a current-carrying source mesh.

    Args:
        source: mesh to use as the field source
        j_density: (A/m^2) (N,3) array of current density vectors at each of the
            element centroids
        targets: (m) (N,3) array of target point positions in 3D space
        solver: selects the solution settings

    Returns:
        (T) (N,3) array of magnetic field strength (H) vectors at each target position
    """

    if solver is None:
        solver = DirectSolver()

    j_density: NDArray[float64] = ascontiguousarray(j_density, dtype=float64)
    tgt_pts: NDArray[float64] = ascontiguousarray(targets, dtype=float64)

    if isinstance(source, CentroidMesh):
        src_pts = ascontiguousarray(source.centroids, dtype=float64)
        src_vol = ascontiguousarray(source.volumes, dtype=float64)

        if isinstance(solver, DirectSolver):
            return h_current_point_direct(
                src_pts, src_vol, j_density, tgt_pts, solver.n_threads
            )

        elif isinstance(solver, OctreeSolver2Zone):
            return h_current_point_octree(
                src_pts,
                src_vol,
                j_density,
                tgt_pts,
                solver.theta,
                solver.leaf_threshold,
                solver.n_threads,
            )

        else:
            raise TypeError(
                f"Unsupported source/solver combination: {type(source)}, {type(solver)}"
            )

    elif isinstance(source, Mesh):
        src_nodes = ascontiguousarray(source.nodes, dtype=float64)
        src_connectivity = ascontiguousarray(source.connectivity, dtype=uint32)
        if isinstance(solver, DirectSolver):
            return h_current_tet4_direct(
                src_nodes, src_connectivity, j_density, tgt_pts, solver.n_threads, edge
            )

        elif isinstance(solver, OctreeSolver2Zone):
            return h_current_tet4_octree(
                src_nodes,
                src_connectivity,
                j_density,
                tgt_pts,
                solver.theta,
                solver.leaf_threshold,
                solver.n_threads,
            )

        elif isinstance(solver, OctreeSolver):
            return h_current_octree(
                src_nodes,
                src_connectivity,
                tgt_pts,
                j_density,
                uint32(solver.leaf_threshold),
                solver.alpha,
                solver.theta,
                solver.n_threads,
            )

        else:
            raise TypeError(
                f"Unsupported source/solver combination: {type(source)}, {type(solver)}"
            )

    else:
        raise TypeError(
            f"Unsupported source/solver combination: {type(source)}, {type(solver)}"
        )


def h_mag(
    source: Mesh | CentroidMesh,
    m_field: NDArray[float64],
    targets: NDArray[float64],
    solver: DirectSolver | OctreeSolver | OctreeSolver2Zone | None = None,
) -> NDArray[float64]:
    """Compute the magnetic field strength using a magnetized mesh as the source

    Args:
        source: mesh to use as the field source
        m_field: (A/m) (N,3) array of magnetization field vectors at each of the
            element centroids
        targets: (m) (N,3) array of target point positions in 3D space
        solver: selects the solution settings

    Returns:
        (T) (N,3) array of magnetic field strength (H) vectors at each target position
    """

    if solver is None:
        solver = DirectSolver()

    m_field: NDArray[float64] = ascontiguousarray(m_field, dtype=float64)
    targets: NDArray[float64] = ascontiguousarray(targets, dtype=float64)

    if isinstance(source, CentroidMesh):
        src_centroids = ascontiguousarray(source.centroids, dtype=float64)
        src_volumes = ascontiguousarray(source.volumes, dtype=float64)

        assert source.centroids.shape[0] == m_field.shape[0]

        if isinstance(solver, DirectSolver):
            theta = 0.0
            leaf_threshold: uint32 = uint32(0)
            use_octree = False

            return h_mag_point(
                src_centroids,
                src_volumes,
                m_field,
                targets,
                theta,
                leaf_threshold,
                solver.n_threads,
                use_octree,
            )

        elif isinstance(solver, OctreeSolver2Zone):
            use_octree = True
            return h_mag_point(
                src_centroids,
                src_volumes,
                m_field,
                targets,
                solver.theta,
                solver.leaf_threshold,
                solver.n_threads,
                use_octree,
            )
        else:
            raise TypeError(
                f"Unsupported source/solver combination: {type(source)}, {type(solver)}"
            )

    elif isinstance(source, Mesh):
        src_nodes = ascontiguousarray(source.nodes, dtype=float64)
        src_connectivity = ascontiguousarray(source.connectivity, dtype=uint32)

        assert src_connectivity.shape[0] == m_field.shape[0]
        if isinstance(solver, DirectSolver):
            theta = 0.0
            leaf_threshold: uint32 = uint32(0)
            use_octree = False
            return h_mag_tet4(
                src_nodes,
                src_connectivity,
                m_field,
                targets,
                theta,
                leaf_threshold,
                solver.n_threads,
                use_octree,
                solver.edge,
            )

        elif isinstance(solver, OctreeSolver2Zone):
            use_octree = True
            return h_mag_tet4(
                src_nodes,
                src_connectivity,
                m_field,
                targets,
                solver.theta,
                solver.leaf_threshold,
                solver.n_threads,
                use_octree,
                solver.edge,
            )

        else:
            raise TypeError(
                f"Unsupported source/solver combination: {type(source)}, {type(solver)}"
            )

    else:
        raise TypeError(
            f"Unsupported source/solver combination: {type(source)}, {type(solver)}"
        )
