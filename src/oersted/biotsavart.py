"""Magnetic field calculation routines"""

from numpy import float64, uint32, ascontiguousarray
from numpy.typing import NDArray

# Create bindings for calculation engine written in Rust
from ._oersted import calculate_fields

from .mesh import Mesh
from .solver import SolverSettings, DEFAULT_SETTINGS
from .constants import MU0
import warnings
from enum import Enum


class _RequestedField(Enum):
    AFIELD = 0
    HFIELD = 1


class _SourceVectorType(Enum):
    CURRENT_DENSITY = 0
    MAGNETIZATION = 1


def _check_inputs(
    src_mesh: Mesh,
    targets: NDArray[float64],
    jdensity: NDArray[float64] | None,
    magnetization: NDArray[float64] | None,
):
    """Ensure that the input arrays are the right shape, make then contiguous for
    passing to Rust, and raise an exception if jdensity/magnetization are not
    exclusively passed
    """

    src_vectors: NDArray[float64]
    source_vector_type: _SourceVectorType

    if jdensity is None and magnetization is None:
        raise ValueError("No source jdensity or magnetization provided")
    if jdensity is not None and magnetization is not None:
        raise ValueError(
            "Ambiguous request: both source jdensity and magnetization provided"
        )
    if jdensity is not None:
        if jdensity.shape[0] != src_mesh.num_elems:
            raise ValueError(
                f"Array for jdensity has shape {jdensity.shape} but "
                f"mesh centroids array has shape ({src_mesh.num_elems}, 3)"
            )

        src_vectors = ascontiguousarray(jdensity, dtype=float64)
        source_vector_type = _SourceVectorType.CURRENT_DENSITY

    if magnetization is not None:
        if magnetization.shape[0] != src_mesh.num_elems:
            raise ValueError(
                f"Array for magnetization has shape "
                f"{magnetization.shape} but mesh centroids array has shape"
                f"({src_mesh.num_elems}, 3)"
            )

        src_vectors = ascontiguousarray(magnetization, dtype=float64)
        source_vector_type = _SourceVectorType.MAGNETIZATION
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"Target array must be N x 3, received {targets.shape}")

    return (
        ascontiguousarray(src_mesh.nodes, dtype=float64),
        ascontiguousarray(src_mesh.connectivity, dtype=uint32),
        ascontiguousarray(targets, dtype=float64),
        src_vectors,
        source_vector_type,
    )


def _solver_args(settings: SolverSettings) -> tuple[bool, bool]:
    element_integration: bool = settings.integration == "element"
    use_octree: bool = settings.method == "octree"

    return (element_integration, use_octree)


def _evaluate_fields(
    src_mesh: Mesh,
    targets: NDArray[float64],
    requested_field: _RequestedField,
    *,  # Force remaining variables to be passed by name
    jdensity: NDArray[float64] | None = None,
    magnetization: NDArray[float64] | None = None,
    settings: SolverSettings = DEFAULT_SETTINGS,
) -> NDArray[float64]:

    src_nodes, src_connectivity, targets, src_vectors, source_vector_type = (
        _check_inputs(src_mesh, targets, jdensity, magnetization)
    )
    element_integration, use_octree = _solver_args(settings)

    return calculate_fields(
        src_nodes,
        src_connectivity,
        src_vectors,
        source_vector_type.value,
        requested_field.value,
        targets,
        element_integration,
        settings.n_threads,
        use_octree,
        settings.theta,
        settings.expansion_order(),
        settings.max_leaf_size,
    )


def a_field(
    src_mesh: Mesh,
    targets: NDArray[float64],
    *,  # Force remaining variables to be passed by name
    jdensity: NDArray[float64] | None = None,
    magnetization: NDArray[float64] | None = None,
    settings: SolverSettings = DEFAULT_SETTINGS,
) -> NDArray[float64]:
    """Compute the magnetic vector potential (A field) at a collection of
        target points

    Args:
        src_mesh: mesh to use as the field source
        targets: (m) (N,3) array of target point positions in 3D space
        jdensity: (A/m^2) (N,3) array of current density vectors at each of the
            source element centroids
        magnetization: (A/m) (N,3) array of magnetization vectors at each of the
            source element centroids
        settings: selects the solution settings

    Returns:
        (T-m) (N,3) array of magnetic vector potential (A) vectors at each
            target position
    """

    return _evaluate_fields(
        src_mesh,
        targets,
        _RequestedField.AFIELD,
        jdensity=jdensity,
        magnetization=magnetization,
        settings=settings,
    )


def h_field(
    src_mesh: Mesh,
    targets: NDArray[float64],
    *,  # Force remaining variables to be passed by name
    jdensity: NDArray[float64] | None = None,
    magnetization: NDArray[float64] | None = None,
    settings: SolverSettings = DEFAULT_SETTINGS,
) -> NDArray[float64]:
    """Compute the magnetic field strength (H field) at a collection of target points

    Args:
        src_mesh: mesh to use as the field source
        targets: (m) (N,3) array of target point positions in 3D space
        jdensity: (A/m^2) (N,3) array of current density vectors at each of the
            source element centroids
        magnetization: (A/m) (N,3) array of magnetization vectors at each of the
            source element centroids
        settings: selects the solution settings

    Returns:
        (A/m) (N,3) array of magnetic field strength (H) vectors at each target position
    """

    return _evaluate_fields(
        src_mesh,
        targets,
        _RequestedField.HFIELD,
        jdensity=jdensity,
        magnetization=magnetization,
        settings=settings,
    )


def b_field(
    src_mesh: Mesh,
    targets: NDArray[float64],
    *,  # Force remaining variables to be passed by name
    jdensity: NDArray[float64] | None = None,
    magnetization: NDArray[float64] | None = None,
    settings: SolverSettings = DEFAULT_SETTINGS,
) -> NDArray[float64]:
    """Compute the magnetic flux density (B field) at a collection of target points,
    assuming the target points are in free space

    Args:
        src_mesh: mesh to use as the field source
        targets: (m) (N,3) array of target point positions in 3D space
        jdensity: (A/m^2) (N,3) array of current density vectors at each of the
            source element centroids
        magnetization: (A/m) (N,3) array of magnetization vectors at each of the
            source element centroids
        settings: selects the solution settings

    Returns:
        (T) (N,3) array of magnetic flux density (B) vectors at each target position
    """

    if magnetization is not None:
        warnings.warn(
            "Computing magnetic flux density using a magnetized mesh as the source.\n"
            "This calculation is only valid in free space. Ensure that target points "
            "are in free space and not within a magnetized mesh.",
            UserWarning,
            stacklevel=2,
        )

    return MU0 * h_field(
        src_mesh,
        targets,
        jdensity=jdensity,
        magnetization=magnetization,
        settings=settings,
    )
