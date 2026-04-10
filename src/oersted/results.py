"""Post-processing routines and results containers"""

from .mesh import SurfaceMesh, Mesh
from numpy.typing import NDArray
from numpy import float64, newaxis, ascontiguousarray
from ._oersted import mesh_surface_forces, mesh_kelvin_force_density


def maxwell_forces(mesh: SurfaceMesh, b_field: NDArray[float64]) -> NDArray[float64]:
    """Compute the maxwell stress tensor and determine the force vector acting on each
    surface face centroid. Returns an (N,3) array of the force vector

    Note: this form of the Maxwell stress tensor is numerically unstable, especially in the presence
    of large background fields. Perform a mesh convergence study to identify an appropriate mesh.
    """

    return mesh_surface_forces(mesh.areas, mesh.normals, b_field)


def kelvin_forces(mesh: Mesh, m_field_centroids: NDArray[float64], b_field_nodes: NDArray[float64]) -> NDArray[float64]:
    """Compute the Kelvin forces acting on a magnetized mesh.

    Note: the h_field must be calculated at the element nodes, while the magnetization field must be known at the centroids.
    """

    # Check that the inputs are defined properly
    assert mesh.centroids.shape == m_field_centroids.shape
    assert mesh.nodes.shape == b_field_nodes.shape

    return kelvin_force_density(mesh, m_field_centroids, b_field_nodes) * mesh.volumes[:, newaxis]


def kelvin_force_density(mesh: Mesh, m_field_centroids: NDArray[float64], b_field_nodes: NDArray[float64]) -> NDArray[float64]:
    """Compute the Kelvin force density acting on a magnetized mesh."""

    # Check that the inputs are defined properly
    assert mesh.centroids.shape == m_field_centroids.shape
    assert mesh.nodes.shape == b_field_nodes.shape

    return mesh_kelvin_force_density(
        ascontiguousarray(mesh.nodes), ascontiguousarray(mesh.connectivity), ascontiguousarray(m_field_centroids), ascontiguousarray(b_field_nodes)
    )
