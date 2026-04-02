"""Compute centerline and self magnetic fields in a solenoid
centered at the origin with main axis in Z.

Solenoid has parameters:
- Inner radius = 0.025m
- Outer radius = 0.050m (thk = 0.025m)
- Length = 0.25m
- Current density = 100 MA/m^2

Note: there's some sort of units mismatch with gmsh
"""

import oersted
from oersted import Mesh, DirectSolver, OctreeSolver
import numpy as np


def check_solenoid():
    mesh_size: float = 15.0
    jmag: float = 1e8
    theta: float = 0.1
    nthreads: int = 0
    ntargets_axis: int = 100
    direct_solver = DirectSolver(n_threads=nthreads)
    octree_solver = OctreeSolver(n_threads=nthreads, leaf_threshold=16, theta=theta)

    # load mesh
    mesh: Mesh = oersted.mesh_step("tests/data/solenoid.stp", mesh_size, mesh_size)
    n: int = mesh.num_elems

    # assign current density
    jdensity = np.zeros((n, 3))
    phi = np.atan2(mesh.centroids[:, 1], mesh.centroids[:, 0])
    jdensity[:, 0] = -jmag * np.sin(phi)
    jdensity[:, 1] = jmag * np.cos(phi)

    # ---
    # Solution at center of solenoid
    # ---

    targets_axis = np.zeros((ntargets_axis, 3))
    targets_axis[:, 2] = np.linspace(-0.125, 0.125, ntargets_axis)
    # targets_axis[:,0] = np.linspace(0, 0.10, ntargets_axis)

    bdirect_pt_axis = oersted.b_field(mesh.to_centroid_mesh(), jdensity, targets_axis, solver=direct_solver)
    boctree_pt_axis = oersted.b_field(mesh.to_centroid_mesh(), jdensity, targets_axis, solver=octree_solver)
    bdirect_tet_axis = oersted.b_field(mesh, jdensity, targets_axis, solver=direct_solver)
    boctree_tet_axis = oersted.b_field(mesh, jdensity, targets_axis, solver=octree_solver)

    # Errors along axis
    bmag_direct_pt_axis = np.linalg.norm(bdirect_pt_axis, axis=1)
    bmag_direct_tet_axis = np.linalg.norm(bdirect_tet_axis, axis=1)
    bmag_octree_pt_axis = np.linalg.norm(boctree_pt_axis, axis=1)
    bmag_octree_tet_axis = np.linalg.norm(boctree_tet_axis, axis=1)

    err_direct_pt_axis = oersted.testing.smape(bmag_direct_tet_axis, bmag_direct_pt_axis)
    err_octree_pt_axis = oersted.testing.smape(bmag_direct_tet_axis, bmag_octree_pt_axis)
    err_octree_tet_axis = oersted.testing.smape(bmag_direct_tet_axis, bmag_octree_tet_axis)

    assert err_direct_pt_axis < 1e-2
    assert err_octree_pt_axis < 1e-2
    assert err_octree_tet_axis < 1e-2

    #
    # Solve for self-fields
    #

    targets = mesh.centroids

    bdirect_pt = oersted.b_field(mesh.to_centroid_mesh(), jdensity, targets, solver=direct_solver)
    bdirect_tet = oersted.b_field(mesh, jdensity, targets, solver=direct_solver)
    boctree_pt = oersted.b_field(mesh.to_centroid_mesh(), jdensity, targets, solver=octree_solver)
    boctree_tet = oersted.b_field(mesh, jdensity, targets, solver=octree_solver)

    # Errors on mesh
    bmag_direct_pt = np.linalg.norm(bdirect_pt, axis=1)
    bmag_direct_tet = np.linalg.norm(bdirect_tet, axis=1)
    bmag_octree_pt = np.linalg.norm(boctree_pt, axis=1)
    bmag_octree_tet = np.linalg.norm(boctree_tet, axis=1)

    err_mesh_pt_octree = oersted.testing.smape(bmag_direct_tet, bmag_octree_pt)
    err_mesh_pt_direct = oersted.testing.smape(bmag_direct_tet, bmag_direct_pt)
    err_mesh_tet_octree = oersted.testing.smape(bmag_direct_tet, bmag_octree_tet)

    assert err_mesh_pt_octree < 1e-1  # pt method known to be inaccurate inside the mesh
    assert err_mesh_pt_direct < 1e-1
    assert err_mesh_tet_octree < 1e-2


def test_solenoid():
    check_solenoid()


if __name__ == "__main__":
    test_solenoid()
