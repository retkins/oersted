"""Operations for magnetic materials"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from numpy import float64, uint32

from .mesh import Mesh
from .constants import MU0
from .materials import Material, LinearMaterial
from ._oersted import _h_demag_tet4, _hfield_dipole_tetrahedrons


def mag_force(centroids: NDArray[float64], vol: NDArray[float64], material: LinearMaterial, h_ext: Callable) -> NDArray[float64]:
    """Compute the force acting on the elements of a mesh with a magnetic material
        under the influence of an external magnetic field.

    Limitations:
    1. This does not (yet) consider the demagnetizing field
    2. Materials are limited to linear (constant mu_r)

    Args:
        centroids: (N,3) centroid of each target element in 3d space
        vol: (N,) volume of each target element
        material: linear magnetic material
        h_ext(centroids): function to evaluate the external field

    Returns:
        (N,3) force vector on each element
    """

    n: int = vol.shape[0]

    # keep the finite difference calculation within the element
    # perhaps make this constant across the mesh
    # delta = 0.5 * (3.0/(4.0*np.pi))*vol**(1.0/3.0)
    delta = 1e-4

    # grad_h[i,a,b] = dH_a/d_b
    grad_h = np.zeros((n, 3, 3))

    # second order finite differencing
    for axis in range(3):
        targets_plus = centroids.copy()
        targets_minus = centroids.copy()
        targets_plus[:, axis] += delta
        targets_minus[:, axis] -= delta

        # in the future, this should be:
        # h = h_ext + h_demag
        h_plus = h_ext(targets_plus)
        h_minus = h_ext(targets_minus)
        grad_h[:, :, axis] = (h_plus - h_minus) / (2.0 * delta)

    # in absence of demag field, assume moments are proportional to mu_r
    moments = material.chi(1) * vol[:, np.newaxis] * h_ext(centroids)

    # Force on each element
    forces = np.zeros((n, 3))
    for a in range(3):
        for b in range(3):
            forces[:, a] += MU0 * moments[:, b] * grad_h[:, b, a]

    return forces


def h_demag_tet4(
    src_nodes: NDArray[float64],
    src_connectivity: NDArray[uint32],
    material: Material,
    m_field: NDArray[float64],
    tgt_nodes: NDArray[float64] | None = None,
    tgt_connectivity: NDArray[uint32] | None = None,
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
        assert src_connectivity.shape[0] == m_field.shape[0]
    except AssertionError:
        print("Error. The M-field should be calculated at element centroids.")

    if tgt_connectivity is None:
        tgt_connectivity = src_connectivity

    if tgt_nodes is None:
        tgt_nodes = src_nodes
    n_elements: int = tgt_connectivity.shape[0]
    hx = np.zeros(n_elements)
    hy = np.zeros(n_elements)
    hz = np.zeros(n_elements)

    _h_demag_tet4(
        np.ascontiguousarray(src_nodes.flatten()),
        np.ascontiguousarray(src_connectivity.flatten().astype(np.uint32)),
        np.ascontiguousarray(tgt_nodes.flatten()),
        np.ascontiguousarray(tgt_connectivity.flatten().astype(np.uint32)),
        np.ascontiguousarray(m_field[:, 0]),
        np.ascontiguousarray(m_field[:, 1], dtype=float64),
        np.ascontiguousarray(m_field[:, 2]),
        np.ascontiguousarray(hx),
        np.ascontiguousarray(hy),
        np.ascontiguousarray(hz),
        nthreads_requested,
    )

    return np.hstack((hx[:, np.newaxis], hy[:, np.newaxis], hz[:, np.newaxis]))


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
            h_demag = h_demag_tet4(mesh.nodes, mesh.connectivity, material, m_field, nthreads_requested=nthreads_requested)
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
