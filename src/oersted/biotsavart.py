"""Magnetic field calculation routines"""

from numpy import float64, uint32, ascontiguousarray, zeros, hstack, newaxis, pi
from numpy.typing import NDArray

# Create bindings for calculation engine written in Rust
from ._oersted import (
    b_current_point_direct,
    b_current_point_octree,
    _hfield_dipole,
    _hfield_tetrahedrons,
    _hfield_tetrahedrons_direct,
    _hfield_dipole_tetrahedrons,
)

from .mesh import Mesh, CentroidMesh
from .solver import DirectSolver, OctreeSolver
from .constants import MU0

# For typing; currently unused
Nx3Array = NDArray[float64]


def b_field(
    source: Mesh | CentroidMesh, j_density: NDArray[float64], targets: NDArray[float64], solver: DirectSolver | OctreeSolver | None = None
) -> NDArray[float64]:
    """Compute the magnetic flux density at a collection of target points using the specific source mesh and
    solver options, assuming the target points are in free space
    """

    if solver is None:
        solver = DirectSolver()

    j_density: NDArray[float64] = ascontiguousarray(j_density, dtype=float64)
    tgt_pts: NDArray[float64] = ascontiguousarray(targets, dtype=float64)

    if isinstance(source, CentroidMesh):
        src_pts = ascontiguousarray(source.centroids, dtype=float64)
        src_vol = ascontiguousarray(source.volumes, dtype=float64)

        if isinstance(solver, DirectSolver):
            return b_current_point_direct(src_pts, src_vol, j_density, tgt_pts, solver.n_threads)

        elif isinstance(solver, OctreeSolver):
            # TODO: update octree solver to output B field instead of H field
            return MU0 * b_current_point_octree(src_pts, src_vol, j_density, tgt_pts, solver.theta, solver.leaf_threshold, solver.n_threads)

        else:
            raise TypeError(f"Unsupported source/solver combination: {type(source)}, {type(solver)}")

    elif isinstance(source, Mesh):
        src_nodes = ascontiguousarray(source.nodes, dtype=float64)
        src_connectivity = ascontiguousarray(source.connectivity, dtype=uint32)
        if isinstance(solver, DirectSolver):
            return bfield_tetrahedrons_direct(src_nodes, src_connectivity, j_density, tgt_pts, solver.n_threads)

        # elif isinstance(solver, OctreeSolver):
        #     src_vol = ascontiguousarray(source.volumes, dtype=float64)
        #     return bfield_tetrahedrons(
        #         src_nodes, src_connectivity, src_vol, j_density, tgt_pts, solver.theta, solver.leaf_threshold, solver.n_threads
        #     )

        else:
            raise TypeError(f"Unsupported source/solver combination: {type(source)}, {type(solver)}")

    else:
        raise TypeError(f"Unsupported source/solver combination: {type(source)}, {type(solver)}")


def bfield_tetrahedrons(
    nodes: NDArray[float64],
    centroids: NDArray[float64],
    vol: NDArray[float64],
    jdensity: NDArray[float64],
    targets: NDArray[float64],
    theta: float = 0.5,
    leaf_threshold: int = 1,
    nthreads: int = 0,
) -> NDArray[float64]:
    """Compute the magnetic flux density at a set of target points
        using a tetrahedral finite element mesh as a near-field source,
        and a point approximation for the far-field.

    Args:
        nodes: (12*N,) nodal coordinates of each element
        vol: (N,) volume of each element
        jdensity: (N,3) current density vector assumed constant over each element
        targets: (M,3) target point locations in 3d space
        theta: angle-opening criteria for barnes-hut

    Returns:
        (N,3) magnetic flux density at each target point

    """

    ntargets = targets.shape[0]
    hx = zeros(ntargets)
    hy = zeros(ntargets)
    hz = zeros(ntargets)

    _hfield_tetrahedrons(
        ascontiguousarray(nodes[:]),
        ascontiguousarray(centroids.ravel()),
        ascontiguousarray(vol[:]),
        ascontiguousarray(jdensity.ravel()),
        ascontiguousarray(targets[:, 0]),
        ascontiguousarray(targets[:, 1]),
        ascontiguousarray(targets[:, 2]),
        ascontiguousarray(hx[:]),
        ascontiguousarray(hy[:]),
        ascontiguousarray(hz[:]),
        theta,
        leaf_threshold,
        nthreads,
    )

    return (4 * pi * 10**-7) * hstack((hx[:, newaxis], hy[:, newaxis], hz[:, newaxis]))


def bfield_tetrahedrons_direct(
    nodes: NDArray[float64],
    connectivity: NDArray[uint32],
    jdensity: NDArray[float64],
    targets: NDArray[float64],
    nthreads: int = 0,
) -> NDArray[float64]:
    """Compute the magnetic flux density at a set of target points
        using a tetrahedral finite element mesh as a source. This
        function performs the analytic (exact) integral for every
        element in the mesh.

    Args:
        nodes: (N,3) nodal coordinates of each element
        connectivity: (N,4) indices into `nodes` for each element
        jdensity: (N,3) current density vector assumed constant over each element
        targets: (M,3) target point locations in 3d space

    Returns:
        (N,3) magnetic flux density at each target point

    """

    ntargets = targets.shape[0]
    hx = zeros(ntargets)
    hy = zeros(ntargets)
    hz = zeros(ntargets)

    _hfield_tetrahedrons_direct(
        ascontiguousarray(nodes.flatten()),
        ascontiguousarray(connectivity.flatten()),
        ascontiguousarray(jdensity.flatten()),
        ascontiguousarray(targets[:, 0]),
        ascontiguousarray(targets[:, 1]),
        ascontiguousarray(targets[:, 2]),
        ascontiguousarray(hx[:]),
        ascontiguousarray(hy[:]),
        ascontiguousarray(hz[:]),
        nthreads,
    )

    return (4 * pi * 10**-7) * hstack((hx[:, newaxis], hy[:, newaxis], hz[:, newaxis]))


def hfield_dipole(
    centroids: NDArray[float64],
    vol: NDArray[float64],
    moments: NDArray[float64],
    targets: NDArray[float64],
    theta: float = 0.5,
    leaf_threshold: int = 16,
    nthreads: int = 0,
) -> NDArray[float64]:
    """Compute the magnetic H-field generated by a collection of magnetized
        finite elements at `targets` using the Barnes-Hut (octree) Biot-Savart method.

    Args:
        centroids: [m] Nx3 array, location of source element centroids in 3D (cartesian) space
        vol: [m^3] N-length array, volume of each source element
        jdensity: [A/m^2] Nx3 array, current density vectors for each source element
        targets: [m] Nx3 array, describes location of target points at which to calculate magnetic fields
        theta: angle-opening parameter for Barnes-Hut; smaller values correspond to lower error
            at the expense of longer runtimes. Suggested values in the range of [0.1, 0.5]
        leaf_threshold: how many individual sources to aggregate per leaf
        nthreads: number of threads to use for the calculation (default: all available)

    Returns:
        [A/m] Nx3 array, magnetic field intensity [H-field] at each specified target point

    """

    ntargets = targets.shape[0]
    hx = zeros(ntargets)
    hy = zeros(ntargets)
    hz = zeros(ntargets)

    _hfield_dipole(
        ascontiguousarray(centroids[:, 0]),
        ascontiguousarray(centroids[:, 1]),
        ascontiguousarray(centroids[:, 2]),
        ascontiguousarray(vol[:]),
        ascontiguousarray(moments[:, 0]),
        ascontiguousarray(moments[:, 1]),
        ascontiguousarray(moments[:, 2]),
        ascontiguousarray(targets[:, 0]),
        ascontiguousarray(targets[:, 1]),
        ascontiguousarray(targets[:, 2]),
        ascontiguousarray(hx[:]),
        ascontiguousarray(hy[:]),
        ascontiguousarray(hz[:]),
        theta,
        leaf_threshold,
        nthreads,
    )

    return hstack((hx[:, newaxis], hy[:, newaxis], hz[:, newaxis]))


def hfield_dipole_tetrahedrons(
    nodes: NDArray[float64],
    centroids: NDArray[float64],
    vol: NDArray[float64],
    moments: NDArray[float64],
    targets: NDArray[float64],
    theta: float = 0.5,
    leaf_threshold=1,
    nthreads: int = 0,
) -> NDArray[float64]:
    """Compute the magnetic field intensity at a set of target points
        using a tetrahedral finite element mesh as a near-field dipole source,
        and a point approximation for the far-field.

    Args:
        nodes: (12*N,) nodal coordinates of each element
        vol: (N,) volume of each element
        moments: (N,3) current density vector assumed constant over each element
        targets: (M,3) target point locations in 3d space
        theta: angle-opening criteria for barnes-hut

    Returns:
        (N,3) magnetic flux density at each target point

    """

    ntargets = targets.shape[0]
    hx = zeros(ntargets)
    hy = zeros(ntargets)
    hz = zeros(ntargets)

    _hfield_dipole_tetrahedrons(
        ascontiguousarray(nodes[:]),
        ascontiguousarray(centroids.ravel()),
        ascontiguousarray(vol[:]),
        ascontiguousarray(moments.ravel()),
        ascontiguousarray(targets[:, 0]),
        ascontiguousarray(targets[:, 1]),
        ascontiguousarray(targets[:, 2]),
        ascontiguousarray(hx[:]),
        ascontiguousarray(hy[:]),
        ascontiguousarray(hz[:]),
        theta,
        leaf_threshold,
        nthreads,
    )

    return (4 * pi * 10**-7) * hstack((hx[:, newaxis], hy[:, newaxis], hz[:, newaxis]))
