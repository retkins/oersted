from __future__ import annotations

from typing import TypeAlias

from numpy import float64, uint32
from numpy.typing import NDArray

Float64Array: TypeAlias = NDArray[float64]
U32Array: TypeAlias = NDArray[uint32]
Vector3: TypeAlias = tuple[float, float, float]

def b_current_point_direct(
    src_pts: Float64Array,
    src_vol: Float64Array,
    src_jdensity: Float64Array,
    tgt_pts: Float64Array,
    nthreads_requested: int,
) -> Float64Array: ...
def h_current_point_octree(
    src_pts: Float64Array,
    src_vol: Float64Array,
    src_jdensity: Float64Array,
    tgt_pts: Float64Array,
    theta: float,
    leaf_threshold: int,
    nthreads_requested: int,
) -> Float64Array: ...
def h_current_tet4_direct(
    nodes: Float64Array,
    connectivity: U32Array,
    jdensity: Float64Array,
    tgt_pts: Float64Array,
    nthreads_requested: int,
) -> Float64Array: ...
def h_current_tet4_octree(
    nodes: Float64Array,
    connectivity: U32Array,
    jdensity: Float64Array,
    tgt_pts: Float64Array,
    theta: float,
    leaf_threshold: int,
    nthreads_requested: int,
) -> Float64Array: ...
def _hfield_dipole_tetrahedrons(
    nodes_flat: Float64Array,
    centroids_flat: Float64Array,
    vol: Float64Array,
    jdensity_flat: Float64Array,
    x: Float64Array,
    y: Float64Array,
    z: Float64Array,
    bx: Float64Array,
    by: Float64Array,
    bz: Float64Array,
    theta: float,
    leaf_threshold: int,
    nthreads_requested: int,
) -> None: ...
def _hfield_dipole(
    centx: Float64Array,
    centy: Float64Array,
    centz: Float64Array,
    vol: Float64Array,
    mx: Float64Array,
    my: Float64Array,
    mz: Float64Array,
    x: Float64Array,
    y: Float64Array,
    z: Float64Array,
    hx: Float64Array,
    hy: Float64Array,
    hz: Float64Array,
    theta: float,
    leaf_threshold: int,
    nthreads_requested: int,
) -> None: ...
def h_mag_tet4_direct(
    nodes: Float64Array,
    connectivity: U32Array,
    mvectors: Float64Array,
    targets: Float64Array,
    nthreads_requested: int,
) -> Float64Array: ...
def mesh_volumes(nodes: Float64Array, connectivity: U32Array) -> Float64Array: ...
def mesh_centroids(nodes: Float64Array, connectivity: U32Array) -> Float64Array: ...
def _mesh_surface_faces(connectivity: U32Array) -> U32Array: ...
def _mesh_surface_face_properties(nodes: Float64Array, faces: U32Array) -> tuple[Float64Array, Float64Array, Float64Array]: ...
def _mesh_surface_forces(face_areas: Float64Array, face_normals: Float64Array, b_field: Float64Array) -> Float64Array: ...
def _mesh_surface_tets(nodes: Float64Array, faces: U32Array, centroids: Float64Array, normals: Float64Array) -> tuple[Float64Array, U32Array]: ...
