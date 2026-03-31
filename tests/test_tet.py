"""Test magnetic field calculations on a tetrahedral mesh of a solenoid using all four
methods:
1. Point sources at centroids, direct integration
2. Point sources at centroids, Barnes Hut
3. Tetrahedral integral, direct integration
4. Tetrahedral integral, Barnes Hut

TODO: rewrite tet/octree function to accept nodal connectivity
"""

import oersted
from oersted import Mesh, DirectSolver, OctreeSolver
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

min_size: float = 2.5
max_size: float = 10.0
jmag: float = 1e8
theta_pt: float = 0.5
theta_tet: float = 0.5
nthreads: int = 0
ntargets_axis: int = 100

# load mesh
# nodes, centroids, vol = oersted.mesh.mesh_step_tets("tests/data/solenoid.stp", min_size, max_size)
mesh: Mesh = oersted.mesh_step("tests/data/solenoid.stp", "", min_size, max_size)
n: int = mesh.num_elems

print("Oersted: Solenoid Test for tetrahedron sources in octree \n---")
print(f"\tn = {n}")
print(f"\ttheta (pt) =  {theta_pt:.3f}")
print(f"\ttheta (tet) = {theta_tet:.3f}")
print(f"\tNthreads = {nthreads}")
print("\tErrors reported relative to direct tet source summation\n")

# assign current density
jdensity = np.zeros((n, 3))
phi = np.atan2(mesh.centroids[:, 1], mesh.centroids[:, 0])
jdensity[:, 0] = -jmag * np.sin(phi)
jdensity[:, 1] = jmag * np.cos(phi)
mesh._j_density = jdensity

# ---
# Solution at center of solenoid
# ---

targets_axis = np.zeros((ntargets_axis, 3))
targets_axis[:, 2] = np.linspace(-0.125, 0.125, ntargets_axis)
# targets_axis[:,0] = np.linspace(0, 0.10, ntargets_axis)

bdirect_pt_axis = oersted.b_field(mesh, mesh.j_density, targets_axis)
boctree_pt_axis = oersted.b_field(mesh, jdensity, targets_axis, solver=OctreeSolver(n_threads=nthreads, theta=theta_pt))
bdirect_tet_axis = oersted.b_field(mesh, jdensity, targets_axis, solver=DirectSolver(n_threads=nthreads))
# boctree_tet_axis = oersted.bfield_tetrahedrons(nodes, centroids, vol, jdensity, targets_axis, theta=theta_tet, nthreads=nthreads)


# Errors along axis
print("Bfield at solenoid center: ")
i: int = ntargets_axis // 2
print(f"\tDirect (pt) solution:  ({bdirect_pt_axis[i, 0]:.6f}, {bdirect_pt_axis[i, 1]:.6f}, {bdirect_pt_axis[i, 2]:.6f})")
print(f"\tDirect (tet) solution: ({bdirect_tet_axis[i, 0]:.6f}, {bdirect_tet_axis[i, 1]:.6f}, {bdirect_tet_axis[i, 2]:.6f})")
print(f"\tOctree (pt) solution:  ({boctree_pt_axis[i, 0]:.6f}, {boctree_pt_axis[i, 1]:.6f}, {boctree_pt_axis[i, 2]:.6f})")
# print(f"\tOctree (tet) solution: ({boctree_tet_axis[i, 0]:.6f}, {boctree_tet_axis[i, 1]:.6f}, {boctree_tet_axis[i, 2]:.6f})")
# err = (boctree_tet_axis[i, 2] - bdirect_tet_axis[i, 2]) / bdirect_tet_axis[i, 2]
# print(f"Error at center (tet octree vs tet direct): {100 * err:.3f} %")

bmag_direct_pt_axis = np.linalg.norm(bdirect_pt_axis, axis=1)
bmag_direct_tet_axis = np.linalg.norm(bdirect_tet_axis, axis=1)
bmag_octree_pt_axis = np.linalg.norm(boctree_pt_axis, axis=1)
# bmag_octree_tet_axis = np.linalg.norm(boctree_tet_axis, axis=1)
err_direct_pt_axis = oersted.testing.smape(bmag_direct_tet_axis, bmag_direct_pt_axis)
err_octree_pt_axis = oersted.testing.smape(bmag_direct_tet_axis, bmag_octree_pt_axis)
# err_octree_tet_axis = oersted.testing.smape(bmag_direct_tet_axis, bmag_octree_tet_axis)
print("Mean error along solenoid axis (|z| < radius): ")
print(f"\tdirect (pt):  {err_direct_pt_axis * 100:.2}%")
print(f"\toctree (pt):  {err_octree_pt_axis * 100:.2}%")
# print(f"\toctree (tet): {err_octree_tet_axis * 100:.2}%")

#
# Solve for self-fields
#

targets = mesh.centroids
ntargets = targets.shape[0]
nsources = n

start = perf_counter()
bdirect_pt = boctree = oersted.b_field(mesh, mesh.j_density, targets, solver=DirectSolver(n_threads=nthreads))
end = perf_counter()
direct_pt_elapsed = end - start

start = perf_counter()
bdirect_tet = oersted.b_field(mesh, mesh.j_density, targets, solver=DirectSolver(n_threads=nthreads))
end = perf_counter()
direct_tet_elapsed = end - start

start = perf_counter()
boctree_pt = oersted.b_field(mesh, jdensity, targets_axis, solver=OctreeSolver(n_threads=nthreads, theta=theta_pt))
end = perf_counter()
octree_pt_elapsed = end - start

start = perf_counter()
# boctree_tet = oersted.bfield_tetrahedrons(nodes, centroids, vol, jdensity, targets, theta=theta_tet, nthreads=nthreads)
end = perf_counter()
octree_tet_elapsed = end - start

# Errors on mesh
bmag_direct_pt = np.linalg.norm(bdirect_pt, axis=1)
bmag_direct_tet = np.linalg.norm(bdirect_tet, axis=1)
bmag_octree_pt = np.linalg.norm(boctree_pt, axis=1)
# bmag_octree_tet = np.linalg.norm(boctree_tet, axis=1)
err_mesh_pt = oersted.testing.smape(bmag_direct_tet, bmag_octree_pt)
err_mesh_pt_direct = oersted.testing.smape(bmag_direct_tet, bmag_direct_pt)
# err_mesh_tet = oersted.testing.smape(bmag_direct_tet, bmag_octree_tet)
print(f"Mean fields error within the mesh (pt/direct sources):  {err_mesh_pt_direct * 100:.2}%")
print(f"Mean fields error within the mesh (pt/octree sources):  {err_mesh_pt * 100:.2}%")
# print(f"Mean fields error within the mesh (tet/octree sources): {err_mesh_tet * 100:.2}%")

print("Times: ")
print(f"\tDirect (pt) solution time:  {direct_pt_elapsed * 1e3:.3f} ms")
print(f"\tDirect (tet) solution time: {direct_tet_elapsed * 1e3:.3f} ms")
print(f"\t\tSpeedup: {direct_pt_elapsed / direct_tet_elapsed:.2f}x")
print(f"\tOctree (pt) solution time:  {octree_pt_elapsed * 1e3:.3f} ms")
print(f"\t\tSpeedup: {direct_pt_elapsed / octree_pt_elapsed:.2f}x")
print(f"\tOctree (tet) solution time: {octree_tet_elapsed * 1e3:.3f} ms")
print(f"\t\tSpeedup: {direct_pt_elapsed / octree_tet_elapsed:.2f}x")


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(targets_axis[:, 2], bdirect_pt_axis[:, 2], "k", label="Direct (pt)")
ax.plot(targets_axis[:, 2], bdirect_tet_axis[:, 2], "b--", label="Direct (tet)")

ax.plot(targets_axis[:, 2], boctree_pt_axis[:, 2], "r.", label="Octree (pt)")
# ax.plot(targets_axis[:, 2], boctree_tet_axis[:, 2], "g^", label="Octree (tet)")

ax.set_xlabel("Distance from Solenoid Center (Z-axis) [m]")
ax.set_ylabel("Field Along Solenoid Axis (Bz) [T]")
ax.set_title("Oersted - Solenoid Test - Pt vs Tet Sources")
plt.legend()
plt.savefig("tests/fig/solenoid_tet_test.svg")
# print(boctree_tet_axis)
