"""Post-processing routines and results containers"""

from .mesh import SurfaceMesh, Mesh
from numpy.typing import NDArray
from numpy import float64, newaxis, ascontiguousarray, cross, sum
from ._oersted import mesh_surface_forces, mesh_kelvin_force_density


def maxwell_forces(mesh: SurfaceMesh, b_field: NDArray[float64]) -> NDArray[float64]:
    """Compute the maxwell stress tensor and determine the force vector acting on each
    surface face centroid. Returns an (N,3) array of the force vector

    !!! warning
        This form of the Maxwell stress tensor is numerically unstable, especially in
        the presence of large background fields. Perform a mesh convergence study to
        identify an appropriate mesh.

    Args:
        mesh: a surface mesh on which to integrate the Maxwell stress tensor
        b_field: (T) magnetic flux density evaluated at each of the centroids of the
            surface mesh elements

    Returns:
        (N) an (N,3) array of the force acting on each of the surface mesh centroids
    """

    assert mesh.centroids.shape == b_field.shape

    return mesh_surface_forces(mesh.areas, mesh.normals, b_field)


def kelvin_forces(
    mesh: Mesh, m_field_centroids: NDArray[float64], b_field_nodes: NDArray[float64]
) -> NDArray[float64]:
    """Compute the Kelvin forces acting on a magnetized mesh.

    !!! note
        The h_field must be calculated at the element nodes, while the magnetization
        field must be known at the centroids.

    Args:
        mesh: the mesh on which to evaluate the forces
        m_field_centroids: (A/m) the magnetization field (M field), evaluated at the
            centroids of the elements in the mesh
        b_field_nodes: (T) the magnetic flux density, evaluated at the nodes within
            the mesh

    Returns:
        (N) an (N,3) array of the force vector acting on each element centroid
    """

    # Check that the inputs are defined properly
    assert mesh.centroids.shape == m_field_centroids.shape
    assert mesh.nodes.shape == b_field_nodes.shape

    return (
        kelvin_force_density(mesh, m_field_centroids, b_field_nodes)
        * mesh.volumes[:, newaxis]
    )


def kelvin_force_density(
    mesh: Mesh, m_field_centroids: NDArray[float64], b_field_nodes: NDArray[float64]
) -> NDArray[float64]:
    """Compute the Kelvin force density acting on a magnetized mesh.

    !!! note
        The h_field must be calculated at the element nodes, while the magnetization
        field must be known at the centroids.

    Args:
        mesh: the mesh on which to evaluate the forces
        m_field_centroids: (A/m) the magnetization field (M field), evaluated at the
            centroids of the elements in the mesh
        b_field_nodes: (T) the magnetic flux density, evaluated at the nodes within
            the mesh

    Returns:
        (N/m^3) an (N,3) array of the force density vector acting on each element
            centroid
    """

    # Check that the inputs are defined properly
    assert mesh.centroids.shape == m_field_centroids.shape
    assert mesh.nodes.shape == b_field_nodes.shape

    return mesh_kelvin_force_density(
        ascontiguousarray(mesh.nodes),
        ascontiguousarray(mesh.connectivity),
        ascontiguousarray(m_field_centroids),
        ascontiguousarray(b_field_nodes),
    )


def lorentz_forces(
    mesh: Mesh,
    j_density: NDArray[float64],
    b_field: NDArray[float64],
    total: bool = False,
) -> NDArray[float64]:
    """Compute the Lorentz forces acting on a mesh

    Args:
        mesh: the finite element mesh on which the fields and forces are calculated
        j_density: (A/m^2) an (N,3) array containing the current density vector
            at every element
        b_field: (T) an (N,3) array containing the magnetic flux density vector
            at every element
        total: if true, return only the total force (default: false)

    Returns:
        (N) an (N,3) array of the force vectors acting on the mesh at every element
    """

    assert j_density.shape[0] == mesh.num_elems

    forces: NDArray[float64] = (
        lorentz_force_density(j_density, b_field) * mesh.volumes[:, newaxis]
    )

    if total:
        return sum(forces, axis=0)

    else:
        return forces


def lorentz_force_density(
    j_density: NDArray[float64], b_field: NDArray[float64]
) -> NDArray[float64]:
    """Compute the Lorentz forces acting on a mesh

    Args:
        j_density: (A/m^2) an (N,3) array containing the current density vector
            at every element
        b_field: (T) an (N,3) array containing the magnetic flux density vector
            at every element

    Returns:
        (N/m^3) an (N,3) array of the force density vectors acting on the mesh
            at every element
    """

    assert j_density.shape == b_field.shape
    assert j_density.shape[1] == 3

    # This might introduce a copy, though shouldn't be a big hit to the user's
    # observed performance
    jxb: NDArray[float64] = cross(j_density, b_field).astype(float64)

    return jxb
