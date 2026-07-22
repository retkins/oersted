"""Octree speed tests"""

import oersted
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

mesh_sizes = np.linspace(1.0e-3, 8e-3, 10)
thetas = np.linspace(0.1, 0.5, 1)
thetas = [0.5]
n_iterations = 5

meshes = []
jdensities = []
for mesh_size in mesh_sizes:
    mesh, jdensity = oersted.make_helmholtz("tests/data/ring.stp", mesh_size)
    meshes.append(mesh)
    jdensities.append(jdensity)


def bench(theta):

    sizes = []
    times = []

    for i, mesh_size in enumerate(mesh_sizes):
        print(f"theta = {theta:.1f}, mesh size = {1e3 * mesh_size:.0f} mm")
        mesh = meshes[i]
        jdensity = jdensities[i]
        start = perf_counter()
        for _ in range(n_iterations):
            _ = oersted.b_field(
                mesh,
                mesh.centroids,
                jdensity=jdensity,
                settings=oersted.SolverSettings(
                    theta=theta, method="octree", integration="element"
                ),
            )
        elapsed = (perf_counter() - start) / n_iterations
        sizes.append(mesh.num_elems)
        times.append(elapsed)
    return sizes, times


fig, ax = plt.subplots()
for theta in thetas:
    sizes, times = bench(theta)
    ax.plot(sizes, times, label=f"theta = {theta:.1f}")
ax.set_xlabel("Problem Size, N Elements")
ax.set_ylabel("Solve Time (s)")
ax.set_title(
    "oersted Barnes Hut Solver Benchmarks\nRyzen 9 9950X (16 cores) + 64GB DDR5"
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
fig.savefig("benchmarks/figs/octree_times.svg")
