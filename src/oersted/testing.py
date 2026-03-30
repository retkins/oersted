"""Utility functions for tests"""

from pathlib import Path
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from .mesh import mesh_step


def mean_squared_error(baseline: NDArray[float64], measurement: NDArray[float64]) -> float:
    """Compute the mean squared error of `measurement` measured against `baseline`

    Error is computed according to this reference:
    https://en.wikipedia.org/wiki/Mean_squared_error#Predictor

    This is an absolute, not a relative error measurement.

    Args:
    ---
        baseline: N-length array of 'ground-truth' values
        measurement: N-length array of values to which determine error (deviation) from baseline

    """

    assert baseline.shape == measurement.shape
    assert len(baseline.shape) == 1
    n: int = baseline.shape[0]
    assert n > 0

    return (1 / n) * np.sum((baseline - measurement) ** 2)


def mean_relative_error(baseline: NDArray[float64], measurement: NDArray[float64]) -> float:
    """Compute the mean *relative* error of `measurement` against `baseline`"""

    assert baseline.shape == measurement.shape
    assert len(baseline.shape) == 1
    n: int = baseline.shape[0]
    assert n > 0

    relative_diff = (measurement - baseline) / baseline
    return (1 / n) * np.sum(np.abs(relative_diff))


def mean_absolute_error(baseline: NDArray[float64], measurement: NDArray[float64]) -> float:
    """Compute the mean absolute error of `measurement` against `baseline`"""

    assert baseline.shape == measurement.shape
    assert len(baseline.shape) == 1
    n: int = baseline.shape[0]
    assert n > 0

    return (1 / n) * np.sum(baseline - measurement)


def smape(baseline: NDArray[float64], measurement: NDArray[float64]) -> float:
    """Compute the symmetric mean absolute percentage error of `measurement` against `baseline`

    SMAPE is defined here:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """

    assert baseline.shape == measurement.shape
    assert len(baseline.shape) == 1
    n: int = baseline.shape[0]
    assert n > 0

    numerator = np.abs(measurement - baseline)
    denominator = np.abs(measurement) + np.abs(baseline)
    return (2 / n) * np.sum(numerator / denominator)


def make_helmholtz(size, jmag: None | float = None, scale=1.0) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Make the helmholtz coil test problem"""

    datafile: str = "ring"
    package_root: Path = Path("__file__").parent.parent.absolute()  # tests is 2 levels up
    mesh = mesh_step(str(package_root / f"tests/data/{datafile}.stp"), str(package_root / f"tests/data/{datafile}_mesh.csv"), size, size, scale)

    nsources = mesh.num_elems  # Targets are now the source centroids for self fields

    # The current mesh is centered on the xy plane and is only one circular ring
    # We need to split the single ring into two rings and assign current densities to the elements
    if jmag is None:
        jmag: float = 100.0e3 / (0.02 * 0.02)
    centroids_upper = mesh.centroids
    centroids_upper[:, 2] += 0.1  # shift upper coil up
    centroids_lower = centroids_upper.copy()
    centroids_lower[:, 2] -= 0.2  # flip to lower side
    centroids = np.vstack((centroids_upper, centroids_lower))
    vol = np.hstack((mesh.volumes, mesh.volumes))
    nsources = vol.shape[0]
    jdensity = np.zeros((nsources, 3))
    phi = np.atan2(centroids[:, 1], centroids[:, 0])
    jdensity[:, 0] = -jmag * np.sin(phi)
    jdensity[:, 1] = jmag * np.cos(phi)

    return (centroids, vol, jdensity)
