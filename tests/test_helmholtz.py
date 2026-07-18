"""Compute centerline and self magnetic fields in a Helmholtz coil

Each coil has the following parameters:
- Centerline radius `r = 0.2m`, separated by a vertical (z) distance of 0.2m
- Square cross-section with h=0.02m
- Total current 100 kA-turn (each) in +Y (cylindrical coords)

We basically do two tests in one:
1. Compute the magnitude of the vertical (z-axis) field
    along the coil axis. This tests the accuracy of the fields themselves.
2. Compute the self-fields and total force induced in each coil. Each
    coil has current in the same direction, so each will experience a
    net attractive force (z) but total zero in x and y. We also time this one
    for speed tests/benchmarking.

Note: there's some sort of units mismatch with gmsh
"""

from oersted import SolverSettings, make_helmholtz

import numpy as np
import oersted
import pathlib

step_file: pathlib.Path = pathlib.Path(__file__).parent / "../tests/data/ring.stp"

# Runtime parameters
theta: float = 0.1
mesh_size: float = 0.033  # ~10M interactions; set to 33 for 1e6 interactions
ntargets_axis: int = 100  # Along the axis
nthreads = 0
max_leaf_size = 16
near_field_ratio = 10
axis_halfdistance = 0.01
batch_size = 1
MAX_ERR = 1e-2

direct_element = SolverSettings(
    method="direct", integration="element", n_threads=nthreads
)
all_settings = [
    direct_element,
    SolverSettings(
        method="octree",
        integration="element",
        theta=theta,
        max_leaf_size=max_leaf_size,
        near_field_ratio=near_field_ratio,
        n_threads=nthreads,
        batch_size=batch_size,
    ),
    SolverSettings(method="direct", integration="point", n_threads=nthreads),
    SolverSettings(
        method="octree",
        integration="point",
        theta=theta,
        max_leaf_size=max_leaf_size,
        near_field_ratio=near_field_ratio,
        n_threads=nthreads,
        batch_size=batch_size,
    ),
]

mesh, jdensity = make_helmholtz(str(step_file), mesh_size)

# Setup the targets for the axis accuracy test
targets_axis = np.zeros((ntargets_axis, 3))
targets_axis[:, 2] = np.linspace(-axis_halfdistance, axis_halfdistance, ntargets_axis)


def test_on_axis():
    """Test that the field on the axis is the same for all four solver methods"""

    b_direct = oersted.b_field(
        mesh, targets_axis, jdensity=jdensity, settings=direct_element
    )
    for settings in all_settings:
        b = oersted.b_field(mesh, targets_axis, jdensity=jdensity, settings=settings)
        assert oersted.max_verr(b, b_direct) < MAX_ERR


def test_centroid():
    # Check that the field at the center matches analytical
    current: float = 100e3
    bz_analytical = (0.8**1.5) * oersted.MU0 * current / 0.2
    target_center = np.array([[0.0, 0.0, 0.0]])

    for settings in all_settings:
        b = oersted.b_field(mesh, target_center, jdensity=jdensity, settings=settings)
        assert (b[0, 2] - bz_analytical) / bz_analytical < MAX_ERR


if __name__ == "__main__":
    test_on_axis()
    test_centroid()
    print("helmholtz test passed")
