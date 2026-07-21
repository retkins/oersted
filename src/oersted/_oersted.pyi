from __future__ import annotations

from typing import TypeAlias

from numpy import float64, uint32
from numpy.typing import NDArray

Float64Array: TypeAlias = NDArray[float64]
U32Array: TypeAlias = NDArray[uint32]
Vector3: TypeAlias = tuple[float, float, float]

def calculate_fields(
    src_nodes: Float64Array,
    src_connectivity: U32Array,
    src_vectors: Float64Array,
    src_vector_type: int,
    requested_field: int,
    targets: Float64Array,
    element_integration: bool,
    n_threads_requested: int,
    use_octree: bool,
    theta: float,
    multipole_order: int,
    max_leaf_size: int,
    batch_size: int,
) -> Float64Array: ...
def magnetization_solve(
    nodes: Float64Array,
    connectivity: U32Array,
    centroids: Float64Array,
    chi: float,
    h_ext: Float64Array,
    element_integration: bool,
    n_threads_requested: int,
    atol: float,
    max_iterations: int,
    under_relaxation_factor: float,
    use_octree: bool,
    theta: float,
    multipole_order: int,
    max_leaf_size: int,
    batch_size: int,
) -> tuple[Float64Array, Float64Array]: ...
def mesh_volumes(nodes: Float64Array, connectivity: U32Array) -> Float64Array: ...
def mesh_centroids(nodes: Float64Array, connectivity: U32Array) -> Float64Array: ...
def mesh_surface_faces(connectivity: U32Array) -> U32Array: ...
def mesh_surface_face_properties(
    nodes: Float64Array, faces: U32Array
) -> tuple[Float64Array, Float64Array, Float64Array]: ...
def mesh_surface_forces(
    face_areas: Float64Array, face_normals: Float64Array, b_field: Float64Array
) -> Float64Array: ...
def _mesh_surface_tets(
    nodes: Float64Array, faces: U32Array, centroids: Float64Array, normals: Float64Array
) -> tuple[Float64Array, U32Array]: ...
def mesh_kelvin_force_density(
    nodes: Float64Array,
    connectivity: U32Array,
    m_field_centroids: Float64Array,
    h_field_nodes: Float64Array,
) -> Float64Array: ...

# Interaction list octree functions
def interaction_lists(
    nodes: Float64Array,
    connectivity: U32Array,
    targets: Float64Array,
    leaf_threshold: uint32,
    alpha: float,
    theta: float,
) -> tuple[U32Array, U32Array, U32Array]: ...
def h_current_octree(
    nodes: Float64Array,
    connectivity: U32Array,
    targets: Float64Array,
    jdensity: Float64Array,
    leaf_threshold: uint32,
    alpha: float,
    theta: float,
    n_threads_requested: uint32,
) -> Float64Array: ...
def atan2(
    yvals: Float64Array,
    xvals: Float64Array,
) -> Float64Array: ...
