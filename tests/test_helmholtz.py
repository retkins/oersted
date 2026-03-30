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

import numpy as np
import matplotlib.pyplot as plt
import oersted
from time import perf_counter


# Runtime parameters
datafile: str = "ring"
remesh: bool = True
theta: float = 0.1
mesh_size: float = 5  # ~10M interactions; set to 33 for 1e6 interactions
ntargets_axis: int = 100  # Along the axis
nthreads = 0
leaf_threshold = 1
axis_halfdistance = 0.3

#
# Generate a mesh from a STEP file
#
min_size: float = mesh_size
max_size: float = mesh_size
mesh = oersted.mesh_step(f"tests/data/{datafile}.stp", f"tests/data/{datafile}_mesh.csv", min_size, max_size)

nsources = mesh.num_elems  # Targets are now the source centroids for self fields

# The current mesh is centered on the xy plane and is only one circular ring
# We need to split the single ring into twoand assign current densities to the elements
jmag: float = 100.0e3 / (0.02 * 0.02)  # 100 A each
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


# Setup the targets for the axis accuracy test
targets_axis = np.zeros((ntargets_axis, 3))
targets_axis[:, 2] = np.linspace(-axis_halfdistance, axis_halfdistance, ntargets_axis)

bdirect_axis = oersted.bfield_direct(centroids, vol, jdensity, targets_axis)
boctree_axis = oersted.bfield_octree(centroids, vol, jdensity, targets_axis, nthreads=nthreads, theta=theta, leaf_threshold=leaf_threshold)

# Targets are now the source centroids for self fields
targets = centroids
ntargets = nsources

print("oersted: Helmholtz Coil Test\n---")
print(f"theta = {theta:.3}")
print(f"Problem size: {nsources} x {ntargets} ({nsources * ntargets:.3e} interactions)")
print(f"Using nthreads = {nthreads}")
print("")

# Compute magnetic fields
start = perf_counter()
bdirect = oersted.bfield_direct(centroids, vol, jdensity, targets, nthreads=nthreads)
end = perf_counter()
direct_elapsed = end - start

start = perf_counter()
boctree = oersted.bfield_octree(centroids, vol, jdensity, targets, nthreads=nthreads, theta=theta, leaf_threshold=leaf_threshold)
end = perf_counter()
octree_elapsed = end - start


def total_force_on_coil(jdensity, bfield, vol):
    fdensity = np.cross(jdensity[:], bfield)
    return np.sum(fdensity * vol[:, np.newaxis], axis=0)


# Compute total force on each coil
j_upper, j_lower = np.split(jdensity, 2, axis=0)
vol_upper, vol_lower = np.split(vol, 2, axis=0)
b_direct_upper, b_direct_lower = np.split(bdirect, 2, axis=0)
b_octree_upper, b_octree_lower = np.split(boctree, 2, axis=0)
fdirect_upper = total_force_on_coil(j_upper, b_direct_upper, vol_upper)
fdirect_lower = total_force_on_coil(j_lower, b_direct_lower, vol_lower)
foctree_upper = total_force_on_coil(j_upper, b_octree_upper, vol_upper)
foctree_lower = total_force_on_coil(j_lower, b_octree_lower, vol_lower)
# print("Total force on each coil: (direct/octree, upper/lower):")
# for f in [fdirect_upper, fdirect_lower, foctree_upper, foctree_lower]:
#     print(f)

f_upper_error = np.linalg.norm(foctree_upper - fdirect_upper) / np.linalg.norm(fdirect_upper)
f_lower_error = np.linalg.norm(foctree_lower - fdirect_lower) / np.linalg.norm(fdirect_lower)
print("Total force errors:")
print(f"\tLower: {f_lower_error * 100:.2f}%")
print(f"\tUpper: {f_upper_error * 100:.2f}%")

print("Bfield at Helmholtz coil center: ")
i: int = ntargets_axis // 2
print(f"Direct solution: ({bdirect_axis[i, 0]:.6f}, {bdirect_axis[i, 1]:.6f}, {bdirect_axis[i, 2]:.6f})")
print(f"Octree solution: ({boctree_axis[i, 0]:.6f}, {boctree_axis[i, 1]:.6f}, {boctree_axis[i, 2]:.6f})")
err = (bdirect_axis[i, 2] - boctree_axis[i, 2]) / bdirect_axis[i, 2]
print(f"Error at center: {100 * err:.3f} %")

bmag_direct_axis = np.linalg.norm(bdirect_axis, axis=1)
bmag_octree_axis = np.linalg.norm(boctree_axis, axis=1)
err_axis = oersted.testing.smape(bmag_direct_axis, bmag_octree_axis)
print(f"Mean error along coil axis (|z| < radius): {err_axis * 100:.2f}%")

bmag_direct = np.linalg.norm(bdirect, axis=1)
bmag_octree = np.linalg.norm(boctree, axis=1)

err_mesh = oersted.testing.smape(bmag_direct, bmag_octree)
print(f"Mean fields error within the mesh: {err_mesh * 100:.2f}%")

print("Times: ")
print(f"Direct solution time: {direct_elapsed * 1e3:.3f} ms")
print(f"Octree solution time: {octree_elapsed * 1e3:.3f} ms")
print(f"Speedup: {direct_elapsed / octree_elapsed:.2f}x")

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(targets_axis[:, 2], bdirect_axis[:, 2], label="Direct")
ax.plot(targets_axis[:, 2], boctree_axis[:, 2], "rs", label="Octree")
ax.set_xlabel("Distance Along Coil Centerr (Z-axis) [m]")
ax.set_ylabel("Field Along Axis (Bz) [T]")
ax.set_title("oersted - Helmholtz Coil Test")
fig.savefig("tests/fig/helmholtz_test.svg")

axis_error = 2 * np.abs(bdirect_axis - boctree_axis) / (np.abs(bdirect_axis) + np.abs(boctree_axis))
fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.plot(targets_axis[:, 2], axis_error)
ax2.set_xlabel("z position")
ax2.set_ylabel("SMAPE")
ax2.legend("lowerright")
fig2.savefig("tests/fig/error.svg")

fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.plot((bmag_octree - bmag_direct) / bmag_direct)
fig3.savefig("tests/fig/error_mesh.svg")


def test_helmholtz():
    assert err_mesh < 1e-2
    assert err_axis < 1e-2
