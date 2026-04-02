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

from oersted import CentroidMesh, Mesh, DirectSolver, OctreeSolver
from oersted.testing import make_helmholtz, smape

import numpy as np
import oersted


def setup_test():

    # Runtime parameters
    theta: float = 0.1
    mesh_size: float = 10.0  # ~10M interactions; set to 33 for 1e6 interactions
    ntargets_axis: int = 100  # Along the axis
    nthreads = 0
    leaf_threshold = 16
    axis_halfdistance = 0.01

    direct_solver = DirectSolver(n_threads=nthreads)
    octree_solver = OctreeSolver(theta=theta, leaf_threshold=leaf_threshold, n_threads=nthreads)

    #
    # Generate a mesh from a STEP file
    #
    # mesh: Mesh = oersted.mesh_step(f"tests/data/{datafile}", min_size, max_size)
    mesh, jdensity = make_helmholtz(mesh_size)

    # Setup the targets for the axis accuracy test
    targets_axis = np.zeros((ntargets_axis, 3))
    targets_axis[:, 2] = np.linspace(-axis_halfdistance, axis_halfdistance, ntargets_axis)

    return mesh, jdensity, targets_axis, direct_solver, octree_solver


def rel_field_on_axis(mesh: Mesh, jdensity, targets_axis, direct_solver, octree_solver):
    """Test that the field on the axis is the same for all four solver methods"""

    centroid_mesh: CentroidMesh = mesh.to_centroid_mesh()

    b_tet4_direct = oersted.b_field(mesh, jdensity, targets_axis, solver=direct_solver)
    b_tet4_octree = oersted.b_field(mesh, jdensity, targets_axis, solver=octree_solver)
    b_point_direct = oersted.b_field(centroid_mesh, jdensity, targets_axis, solver=direct_solver)
    b_point_octree = oersted.b_field(centroid_mesh, jdensity, targets_axis, solver=octree_solver)

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
    bz_tet4_direct = oersted.b_field(mesh, jdensity, target_center, solver=direct_solver)[0, 2]
    bz_tet4_octree = oersted.b_field(mesh, jdensity, target_center, solver=octree_solver)[0, 2]
    bz_point_direct = oersted.b_field(centroid_mesh, jdensity, target_center, solver=direct_solver)[0, 2]
    bz_point_octree = oersted.b_field(centroid_mesh, jdensity, target_center, solver=octree_solver)[0, 2]

    assert np.abs(bz_analytical - bz_tet4_direct) / bz_analytical < 1e-3
    assert np.abs(bz_analytical - bz_tet4_octree) / bz_analytical < 1e-3
    assert np.abs(bz_analytical - bz_point_direct) / bz_analytical < 1e-3
    assert np.abs(bz_analytical - bz_point_octree) / bz_analytical < 1e-3


def test_helmholtz():
    mesh, jdensity, targets_axis, direct_solver, octree_solver = setup_test()
    rel_field_on_axis(mesh, jdensity, targets_axis, direct_solver, octree_solver)


if __name__ == "__main__":
    test_helmholtz()
