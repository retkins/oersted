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

from oersted import CentroidMesh, Mesh, SolverSettings, OctreeSettings
from oersted.testing import make_helmholtz, smape

import numpy as np
import oersted
import pathlib

step_file: pathlib.Path = pathlib.Path(__file__).parent / "../tests/data/ring.stp"

# Runtime parameters
theta: float = 0.5
mesh_size: float = 0.010  # ~10M interactions; set to 33 for 1e6 interactions
ntargets_axis: int = 100  # Along the axis
nthreads = 0
max_leaf_size = 16
near_field_ratio = 10
axis_halfdistance = 0.01

def setup_test():

    direct_tet4 = SolverSettings(method="direct", integration="element", n_threads=nthreads)
    octree_tet4 = SolverSettings(method="octree", integration="element",
        octree=OctreeSettings(theta=theta,max_leaf_size=max_leaf_size, near_field_ratio=near_field_ratio), n_threads=nthreads
    )
    direct_point = SolverSettings(method="direct", integration="point", n_threads=nthreads)
    octree_point = SolverSettings(method="octree", integration="point",
        octree=OctreeSettings(theta=theta,max_leaf_size=max_leaf_size, near_field_ratio=near_field_ratio), n_threads=nthreads
    )

    #
    # Generate a mesh from a STEP file
    #
    # mesh: Mesh = oersted.mesh_step(f"tests/data/{datafile}", min_size, max_size)
    mesh, jdensity = make_helmholtz(str(step_file), mesh_size)

    # Setup the targets for the axis accuracy test
    targets_axis = np.zeros((ntargets_axis, 3))
    targets_axis[:, 2] = np.linspace(
        -axis_halfdistance, axis_halfdistance, ntargets_axis
    )

    return (
        mesh,
        jdensity,
        targets_axis,
        direct_tet4,
        direct_point,
        octree_tet4,
        octree_point
    )


def rel_field_on_axis(
    mesh: Mesh,
    jdensity,
    targets_axis,
    direct_tet4,
    direct_point,
    octree_tet4,
    octree_point, 
    verbose: bool = True
):
    """Test that the field on the axis is the same for all four solver methods"""

    b_tet4_direct = oersted.b_field(
        mesh, targets_axis, jdensity=jdensity, settings=direct_tet4
    )
    b_tet4_octree = oersted.b_field(
        mesh, targets_axis, jdensity=jdensity, settings=octree_tet4
    )
    b_point_direct = oersted.b_field(
        mesh, targets_axis, jdensity=jdensity, settings=direct_point
    )
    b_point_octree = oersted.b_field(
        mesh, targets_axis, jdensity=jdensity, settings=octree_point
    )

    # Use the tet4 direct field as the comparison
    # Field should only be in the z-axis
    err_tet4_octree = smape(b_tet4_direct[:, 2], b_tet4_octree[:, 2])
    err_point_direct = smape(b_tet4_direct[:, 2], b_point_direct[:, 2])
    err_point_octree = smape(b_tet4_direct[:, 2], b_point_octree[:, 2])

    assert err_tet4_octree < 1e-2
    assert err_point_direct < 1e-2
    assert err_point_octree < 1e-2

    # Check that the x/y fields are near zero
    assert np.max(np.linalg.norm(b_tet4_direct[:, 0:2])) < 1e-3
    assert np.max(np.linalg.norm(b_tet4_octree[:, 0:2])) < 1e-3
    assert np.max(np.linalg.norm(b_point_direct[:, 0:2])) < 1e-3
    assert np.max(np.linalg.norm(b_point_octree[:, 0:2])) < 1e-3

    # Check that the field at the center matches analytical
    current: float = 100e3
    bz_analytical = (0.8**1.5) * oersted.MU0 * current / 0.2
    target_center = np.array([[0.0, 0.0, 0.0]])
    bz_tet4_direct = oersted.b_field(
        mesh, target_center, jdensity=jdensity, settings=direct_tet4
    )[0, 2]
    bz_tet4_octree = oersted.b_field(
        mesh, target_center, jdensity=jdensity, settings=octree_tet4
    )[0, 2]
    bz_point_direct = oersted.b_field(
        mesh, target_center, jdensity=jdensity, settings=direct_point
    )[0, 2]
    bz_point_octree = oersted.b_field(
        mesh, target_center, jdensity=jdensity, settings=octree_point
    )[0, 2]

    if verbose:
        print("On axis fields:")
        print(f"\tbz_analytical:   {bz_analytical:.6f}")
        print(f"\tbz_tet4_direct:  {bz_tet4_direct:.6f}")
        print(f"\tbz_tet4_octree:  {bz_tet4_octree:.6f}")
        print(f"\tbz_point_direct: {bz_point_direct:.6f}")
        print(f"\tbz_point_octree: {bz_point_octree:.6f}")
    assert np.abs(bz_analytical - bz_tet4_direct) / bz_analytical < 1e-3
    assert np.abs(bz_analytical - bz_tet4_octree) / bz_analytical < 1e-2
    assert np.abs(bz_analytical - bz_point_direct) / bz_analytical < 1e-3
    assert np.abs(bz_analytical - bz_point_octree) / bz_analytical < 1e-2


def test_helmholtz(verbose: bool = False):
    mesh, jdensity, targets_axis, direct_tet4, direct_point, octree_tet4, octree_point= (
        setup_test()
    )
    rel_field_on_axis(
        mesh, jdensity, targets_axis, direct_tet4, direct_point, octree_tet4, octree_point
    )


if __name__ == "__main__":
    test_helmholtz(verbose=True)
