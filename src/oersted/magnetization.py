"""Operations for magnetic materials"""

import numpy as np
from numpy.typing import NDArray
from numpy import float64, uint32

from .mesh import Mesh
from .materials import Material
from ._oersted import h_mag_tet4_direct, _hfield_dipole_tetrahedrons


def h_demag_tet4(
    mesh: Mesh,
    material: Material,
    m_field: NDArray[float64],
    targets: NDArray[float64] | None = None,
    nthreads_requested: int = 0,
) -> NDArray[float64]:
    """Compute the demagnetization field H(M) on mesh element centroids given the current M-field

    Args:
        nodes: (Nn, 3) nodal coordinates per element
        element_connectivity: (Ne, 4) indices of each node per element;
            these are indices of the array `nodes`, not of the solver's node numbers
        material: linear or nonlinear magnetic maaterial properties
        m_field: (Ne,3) current M-field at each element centroid
        max_iterations: number of solver iterations before exit
        tol: maximum amount of change per individual component of M at each element

    Returns:
        (Ne 3): demagnetization field H(M) at each node
    """

    # Check that the M field is calculated at the element centroids
    try:
        assert mesh.connectivity.shape[0] == m_field.shape[0]
    except AssertionError:
        print("Error. The M-field should be calculated at element centroids.")

    if targets is None:
        targets = mesh.centroids

    return h_mag_tet4_direct(
        np.ascontiguousarray(mesh.nodes),
        np.ascontiguousarray(mesh.connectivity),
        np.ascontiguousarray(m_field),
        np.ascontiguousarray(targets),
        nthreads_requested,
    )


def h_demag_tet4_octree(
    nodes: NDArray[float64],
    element_connectivity: NDArray[uint32],
    material: Material,
    m_field: NDArray[float64],
    centroids: NDArray[float64],
    vol: NDArray[float64],
    nthreads_requested: int = 0,
    theta: float = 0.5,
    leaf_threshold: int = 16,
) -> NDArray[float64]:
    """Compute the demagnetization field H(M) on mesh element centroids given the current M-field

    Args:
        nodes: (Nn, 3) nodal coordinates per element
        element_connectivity: (Ne, 4) indices of each node per element;
            these are indices of the array `nodes`, not of the solver's node numbers
        material: linear or nonlinear magnetic maaterial properties
        m_field: (Ne,3) current M-field at each element centroid
        max_iterations: number of solver iterations before exit
        tol: maximum amount of change per individual component of M at each element

    Returns:
        (Ne 3): demagnetization field H(M) at each node
    """

    # Check that the M field is calculated at the element centroids
    try:
        assert element_connectivity.shape[0] == m_field.shape[0]
    except AssertionError:
        print("Error. The M-field should be calculated at element centroids.")

    n_elements: int = element_connectivity.shape[0]
    hx = np.zeros(n_elements)
    hy = np.zeros(n_elements)
    hz = np.zeros(n_elements)

    _hfield_dipole_tetrahedrons(
        np.ascontiguousarray(nodes.flatten()),
        np.ascontiguousarray(centroids.flatten()),
        np.ascontiguousarray(vol),
        np.ascontiguousarray(m_field.flatten(), dtype=float64),
        np.ascontiguousarray(m_field[:, 0]),
        np.ascontiguousarray(m_field[:, 1]),
        np.ascontiguousarray(m_field[:, 2]),
        np.ascontiguousarray(hx),
        np.ascontiguousarray(hy),
        np.ascontiguousarray(hz),
        theta,
        leaf_threshold,
        nthreads_requested,
    )

    return np.hstack((hx[:, np.newaxis], hy[:, np.newaxis], hz[:, np.newaxis]))


def demag_tet4(
    mesh: Mesh,
    material: Material,
    h_external: NDArray[float64],
    max_iterations: int = 50,
    tol: float = 1.0,
    nthreads_requested: int = 0,
    octree: bool = False,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Compute magnetization field M and the total H field at element centroids

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

    # Check that the external field is calculated at the nodes
    try:
        assert mesh.num_elems == h_external.shape[0]
    except AssertionError:
        print("Error. The external should be calculated at mesh nodes.")

    centroids = mesh.centroids
    vol = mesh.volumes

    # We need the magnetization curve; sometimes users may have a B-H curve
    h_values, m_values = material.to_mh_curve()

    n_elements: int = mesh.num_elems

    m_field = np.zeros((n_elements, 3), dtype=float64)
    h_hat = np.zeros((n_elements, 3))
    h_total = np.zeros((n_elements, 3))

    for i in range(max_iterations):
        # Get the demag and total H field at the element centroids
        if octree:
            h_demag = h_demag_tet4_octree(mesh.nodes, mesh.connectivity, material, m_field, centroids, vol, nthreads_requested=nthreads_requested)

        else:
            h_demag = h_demag_tet4(mesh, material, m_field, nthreads_requested=nthreads_requested)

        h_demag = h_demag_tet4(mesh, material, m_field, nthreads_requested=nthreads_requested)
        h_total = h_demag + h_external

        # We consider isotropic materials for the B-H curve iteration
        h_magnitude = np.linalg.norm(h_total, axis=1)
        m_magnitude = np.interp(h_magnitude, h_values, m_values)
        mask = h_magnitude > 1e-8
        h_hat.fill(0.0)
        h_hat[mask] = h_total[mask, :] / h_magnitude[mask, np.newaxis]
        m_field_new = h_hat * m_magnitude[:, np.newaxis]
        max_err: float = np.max(np.abs(m_field_new - m_field))
        if max_err < tol:
            break
        else:
            print(f"Iteration: {i} | max err = {max_err:.3e} | continuing")
        m_field = m_field_new

    return (m_field, h_total)
