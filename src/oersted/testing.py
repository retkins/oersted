"""Utility functions for tests"""

import numpy as np
from numpy import float64, uint32
from numpy.typing import NDArray
from .mesh import Mesh
from .constants import MU0
from pathlib import Path

RTOL: float = 1e-8


def verr(
    measurement: NDArray[float64], baseline: NDArray[float64], eps: float = 1e-6
) -> NDArray[float64]:
    """Compute simple relative vector error, with the denominator bounded by `eps` to
    prevent divide by zero issues

    Returns:
        element-wise error at each output point
    """

    return np.linalg.norm(measurement - baseline, axis=1) / (
        np.linalg.norm(baseline, axis=1) + eps
    )


def mean_verr(
    measurement: NDArray[float64], baseline: NDArray[float64], eps: float = 1e-6
) -> float:
    """Compute mean relative vector error, with the denominator bounded by `eps` to
    prevent divide by zero issues
    """

    return float(np.mean(verr(measurement, baseline, eps)))


def max_verr(
    measurement: NDArray[float64], baseline: NDArray[float64], eps: float = 1e-6
) -> float:
    """Compute max relative vector error, with the denominator bounded by `eps` to
    prevent divide by zero issues
    """

    return float(np.max(verr(measurement, baseline, eps)))


def mean_squared_error(
    baseline: NDArray[float64], measurement: NDArray[float64]
) -> float:
    """Compute the mean squared error of `measurement` measured against `baseline`

    Error is computed according to this reference:
    https://en.wikipedia.org/wiki/Mean_squared_error#Predictor

    This is an absolute, not a relative error measurement.

    Args:
    ---
        baseline: N-length array of 'ground-truth' values
        measurement: N-length array of values to which determine error (deviation)
            from baseline

    """

    assert baseline.shape == measurement.shape
    assert len(baseline.shape) == 1
    n: int = baseline.shape[0]
    assert n > 0

    return (1 / n) * np.sum((baseline - measurement) ** 2)


def mean_relative_error(
    baseline: NDArray[float64], measurement: NDArray[float64]
) -> float:
    """Compute the mean *relative* error of `measurement` against `baseline`"""

    assert baseline.shape == measurement.shape
    assert len(baseline.shape) == 1
    n: int = baseline.shape[0]
    assert n > 0

    relative_diff = (measurement - baseline) / baseline
    return (1 / n) * np.sum(np.abs(relative_diff))


def mean_absolute_error(
    baseline: NDArray[float64], measurement: NDArray[float64]
) -> float:
    """Compute the mean absolute error of `measurement` against `baseline`"""

    assert baseline.shape == measurement.shape
    assert len(baseline.shape) == 1
    n: int = baseline.shape[0]
    assert n > 0

    return (1 / n) * np.sum(baseline - measurement)


def smape(baseline: NDArray[float64], measurement: NDArray[float64]) -> float:
    """Compute the symmetric mean absolute percentage error of `measurement`
        against `baseline`

    Args:
        baseline: an (N,) array of data values to compute error against
        measurement: an (N,) array of data values of which to compute error

    Returns:
        error associated with this comparison

    !!! reference
        <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>
    """

    assert baseline.shape == measurement.shape
    assert len(baseline.shape) == 1
    n: int = baseline.shape[0]
    assert n > 0

    numerator = np.abs(measurement - baseline)
    denominator = np.abs(measurement) + np.abs(baseline)
    if denominator.any() < RTOL:
        raise ValueError("Both measurement and baseline are near zero.")

    return (2 / n) * np.sum(numerator / denominator)


def make_helmholtz(
    filename: str | Path, size: float, jmag: None | float = None, scale: float = 1e-3
) -> tuple[Mesh, NDArray[float64]]:
    """Make the helmholtz coil test problem

    The geometry of the problem is defined as:

    * Two circular coils of radius R=0.2m
    * Distance between the coils is d=0.2m
    * Coils are aligned with the z-axis and symmetric about the xy plane
    * Currents in the currents are flowing in the same direction

    Args:
        filename: STEP file containing the geometry of a single loop in the Helmholtz
            pair
        size: (m) mesh size
        jmag: (A/m^2) magnitude of current density in each loop
        scale: (mm/m) scale factor for meshing

    Returns:
        (mesh, current density) the volumetric mesh and current density vectors of the
            Helmholtz coil pair

    """

    ring_mesh: Mesh = Mesh.from_step(filename, size, 1e3, scale)

    # The current mesh is centered on the xy plane and is only one circular ring
    # Split the single ring into two rings and assign current densities to the elements
    if jmag is None:
        jmag: float = 100.0e3 / (0.02 * 0.02)
    nodes_upper: NDArray[float64] = ring_mesh.nodes.copy()
    nodes_upper[:, 2] += 0.1  # shift upper coil up
    nodes_lower: NDArray[float64] = nodes_upper.copy()
    nodes_lower[:, 2] -= 0.2  # flip to lower side
    nodes = np.vstack((nodes_upper, nodes_lower))
    connectivity_upper: NDArray[uint32] = ring_mesh.connectivity.copy()
    connectivity_lower: NDArray[uint32] = ring_mesh.connectivity.copy() + uint32(
        ring_mesh.num_nodes
    )

    helmholtz_mesh = Mesh(nodes, np.vstack((connectivity_upper, connectivity_lower)))

    jdensity = np.zeros((helmholtz_mesh.num_elems, 3))
    phi = np.atan2(helmholtz_mesh.centroids[:, 1], helmholtz_mesh.centroids[:, 0])
    jdensity[:, 0] = -jmag * np.sin(phi)
    jdensity[:, 1] = jmag * np.cos(phi)

    return (helmholtz_mesh, jdensity)


def make_ring(
    mesh_size: float = 15e-3, jmag: float = 1e8
) -> tuple[Mesh, NDArray[float64]]:
    """Make a mesh of a current-carrying ring for testing"""

    mesh = Mesh.from_step("tests/data/ring.stp", mesh_size)
    phi = np.atan2(mesh.centroids[:, 1], mesh.centroids[:, 0])
    jdensity = np.zeros_like(mesh.centroids)
    jdensity[:, 0] = -jmag * np.sin(phi)
    jdensity[:, 1] = jmag * np.cos(phi)

    return mesh, jdensity


def bz_finite_length_solenoid(
    jmag: float, length: float, r: float, dr: float, z: float
) -> float:
    """Compute the magnetic field on the axis of a finite-length solenoid

    This function assumes that the solenoid `dr` dimension is small relative to the
    radius and thickness, and does not correct for finite radial thickness.

    Args:
        jmag: (A/m2) magnitude of current density in the solenoid
        length: (m) length of the solenoid
        r: (m) representative radius of the solenoid
        dr: (m) thickness of the solenoid cross section
        z: (m) position along the axis of the solenoid at which the field should
            be calculated

    Returns:
        (T) axial magnetic field

    Reference:
        <https://en.wikipedia.org/wiki/Solenoid#Finite_continuous_solenoid>
        (with modifications for current density and finite thickness)
    """

    a: float = 0.5 * MU0 * jmag * dr
    b: float = (z + 0.5 * length) / np.sqrt(r**2 + (z + 0.5 * length) ** 2)
    c: float = (z - 0.5 * length) / np.sqrt(r**2 + (z - 0.5 * length) ** 2)

    return a * (b - c)


def bz_loop_axis(current: float, radius: float, z: float) -> float:
    """Compute the vertical field Bz at the center of a current-carrying loop

    Args:
        current: (A) total electrical current in the loop
        radius: (m) centerline radius of the loop
        z: (m) height of the target point along the loop axis

    Returns:
        (T) magnetic flux density, oriented along the loop axis
    """
    return MU0 * current * (radius**2) / (2.0 * (z**2 + radius**2) ** 1.5)


def dbzdz_loop_axis(current: float, radius: float, z: float) -> float:
    """Compute the vertical field gradient dBz/dz at the center of a
    current-carrying loop

    Args:
        current: (A) total electrical current in the loop
        radius: (m) centerline radius of the loop
        z: (m) height of the target point along the loop axis

    Returns:
        (T/m) magnetic flux density gradient, oriented along the loop axis
    """
    return -1.5 * MU0 * current * (radius**2) * z / (z**2 + radius**2) ** 2.5


def curl(
    fx: NDArray[float64],
    fy: NDArray[float64],
    fz: NDArray[float64],
    spacing: tuple[float, float, float],
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Compute the curl of a vector-value function in 3D space on a uniform grid

    Args:
        fx: shape (nx, ny, nz) value of function in x-direction at all points in 3d grid
        fy: shape (nx, ny, nz) value of function in y-direction at all points in 3d grid
        fz: shape (nx, ny, nz) value of function in z-direction at all points in 3d grid
        spacing: (dx, dy, dz) step size in each direction

    Returns:
        curl_x, curl_y, curl_z: each shape (nx, ny, nz)
    """

    (dx, dy, dz) = spacing

    dfz_dy = np.gradient(fz, dy, axis=1)
    dfy_dz = np.gradient(fy, dz, axis=2)

    dfx_dz = np.gradient(fx, dz, axis=2)
    dfz_dx = np.gradient(fz, dx, axis=0)

    dfy_dx = np.gradient(fy, dx, axis=0)
    dfx_dy = np.gradient(fx, dy, axis=1)

    curl_x = dfz_dy - dfy_dz
    curl_y = dfx_dz - dfz_dx
    curl_z = dfy_dx - dfx_dy

    return curl_x, curl_y, curl_z


def uniform_3d_grid(
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    zrange: tuple[float, float],
    n: tuple[int, int, int],
) -> tuple[NDArray[float64], tuple[float, float, float]]:
    """Create a uniform 3D grid

    Args
        xrange: (xmin, xmax)
        yrange: (ymin, ymax)
        zrange: (zmin, zmax)
        n: (nx, ny, nz)

    Returns:
        pts: shape (nx*ny*nz,3), locations of points in 3d space
        (dx, dy, dz): length increment along each axis
    """

    (xmin, xmax) = xrange
    (ymin, ymax) = yrange
    (zmin, zmax) = zrange
    (nx, ny, nz) = n

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]

    return pts, (dx, dy, dz)
