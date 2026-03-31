"""Compute centerline and self magnetic fields in a solenoid
centered at the origin with main axis in Z.

Solenoid has parameters:
- Inner radius = 0.025m
- Outer radius = 0.050m (thk = 0.025m)
- Length = 0.25m
- Current density = 100 MA/m^2

Note: there's some sort of units mismatch with gmsh
"""

import numpy as np
import matplotlib.pyplot as plt
import oersted
from oersted import Mesh, DirectSolver
from time import perf_counter

#
# Runtime parameters
#

datafile: str = "solenoid"
remesh: bool = True
theta: float = 0.1
mesh_size: float = 15  # ~10M interactions; set to 33 for 1e6 interactions
ntargets_axis: int = 25  # Along the axis
nthreads: int = 1

#
# Generate a mesh from a STEP file
#

mesh: Mesh = oersted.mesh.mesh_step(f"tests/data/{datafile}.stp", f"tests/data/{datafile}_mesh.csv", mesh_size, mesh_size)


#
# Setup sources from the mesh
#

nsources = mesh.num_elems

# We need to assign current densities to the elements
jmag: float = 1e8

jdensity = np.zeros((nsources, 3))
phi = np.atan2(mesh.centroids[:, 1], mesh.centroids[:, 0])
jdensity[:, 0] = -jmag * np.sin(phi)
jdensity[:, 1] = jmag * np.cos(phi)

#
# Setup targets for axis accuracy test and solve for fields
#

targets_axis = np.zeros((ntargets_axis, 3))
targets_axis[:, 2] = np.linspace(-0.125, 0.125, ntargets_axis)

bdirect_axis = oersted.b_field(mesh, jdensity, targets_axis)
boctree_axis = oersted.bfield_octree(mesh.centroids, mesh.volumes, jdensity, targets_axis, nthreads=nthreads, theta=theta)

#
# Solve for self-fields
#

targets = mesh.centroids
ntargets = targets.shape[0]

start = perf_counter()
bdirect = boctree = oersted.b_field(mesh, jdensity, targets, solver=DirectSolver(n_threads=nthreads))
end = perf_counter()
direct_elapsed = end - start

start = perf_counter()
boctree = oersted.bfield_octree(mesh.centroids, mesh.volumes, jdensity, targets, nthreads=nthreads, theta=theta)
end = perf_counter()
octree_elapsed = end - start

print("oersted: Solenoid Test\n---")
print(f"theta = {theta:.3}")
print(f"Problem size: {nsources} x {ntargets} ({nsources * ntargets:.3e} interactions)")

# Errors along axis
print("Bfield at solenoid center: ")
i: int = ntargets_axis // 2
print(f"\tDirect solution: ({bdirect_axis[i, 0]:.6f}, {bdirect_axis[i, 1]:.6f}, {bdirect_axis[i, 2]:.6f})")
print(f"\tOctree solution: ({boctree_axis[i, 0]:.6f}, {boctree_axis[i, 1]:.6f}, {boctree_axis[i, 2]:.6f})")
err = (bdirect_axis[i, 2] - boctree_axis[i, 2]) / bdirect_axis[i, 2]
print(f"Error at center: {100 * err:.3f} %")

bmag_direct_axis = np.linalg.norm(bdirect_axis, axis=1)
bmag_octree_axis = np.linalg.norm(boctree_axis, axis=1)
err_axis = oersted.testing.smape(bmag_direct_axis, bmag_octree_axis)
print(f"Mean error along solenoid axis (|z| < radius): {err_axis * 100:.2}%")

# Errors on mesh
bmag_direct = np.linalg.norm(bdirect, axis=1)
bmag_octree = np.linalg.norm(boctree, axis=1)
err_mesh = oersted.testing.smape(bmag_direct, bmag_octree)
print(f"Mean fields error within the mesh: {err_mesh * 100:.2}%")

print("Times: ")
print(f"\tDirect solution time: {direct_elapsed * 1e3:.3f} ms")
print(f"\tOctree solution time: {octree_elapsed * 1e3:.3f} ms")
print(f"\tSpeedup: {direct_elapsed / octree_elapsed:.2f}x")


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(targets[:, 2], bdirect[:, 2], label="Direct")
ax.plot(targets[:, 2], boctree[:, 2], "rs", label="Octree")
ax.set_xlabel("Distance from Solenoid Center (Z-axis) [m]")
ax.set_ylabel("Field Along Solenoid Axis (Bz) [T]")
ax.set_title("oersted - Solenoid Test")
plt.savefig("tests/fig/solenoid_test.svg")


def test_solenoid():
    assert err_mesh < 1e-2
    assert err_axis < 1e-2
