"""Use the Helmholz coil problem to measure the accuracy and speed of the (new) 3-zone
    octree method

Error metric: total force acting on the upper coil is < 0.5% error from the
direct solution
"""

import oersted
import numpy as np
from oersted import DirectSolver, OctreeSolver, OctreeSolver2Zone
from numpy import uint32, abs
from time import perf_counter
import matplotlib.pyplot as plt

# Problem parameters
thetas = np.linspace(0.05, 0.5, 10)
# alphas = [1.0, 2.0, 3.0, 5.0, 10.0]
alphas: list[float] = [2.0]
leaf_threshold: uint32 = uint32(1)
mesh_size: float = 10.0  # (m)
jmag: float = 1e8  # (A/m^2)
MAX_ERR_PCT: float = 5e-3
iterations: int = 5

# Make the helmholz coil problem
radius: float = 0.2  # (m)
total_current: float = 1e4  # (A/m^2)

mesh1 = oersted.mesh_step("tests/data/ring.stp", mesh_size, mesh_size)
mesh1._nodes[:, 2] += 0.01

mesh2 = oersted.mesh_step("tests/data/ring.stp", mesh_size, mesh_size)
mesh2._nodes[:, 2] -= 0.01

mesh = mesh1.append(mesh2)
print(f"Number of elements: {mesh.num_elems}")

# Assign current densities to each mesh
jmag: float = total_current / (0.02 * 0.02)
jdensity = np.zeros((mesh1.num_elems, 3))
phi = np.atan2(mesh1.centroids[:, 1], mesh1.centroids[:, 0])
jdensity[:, 0] = -jmag * np.sin(phi)
jdensity[:, 1] = jmag * np.cos(phi)
jdensity_total = np.vstack((jdensity, jdensity))

# Direct solution
start = perf_counter()
b_upper = oersted.b_field(mesh, jdensity_total, mesh1.centroids, solver=DirectSolver())
direct_elapsed = perf_counter() - start
f_upper = oersted.lorentz_forces(mesh1, jdensity, b_upper, total=True)

print(f"Direct solution\ntime = {direct_elapsed:.3f} sec\nTotal_force = {f_upper}")


def run():
    for alpha in alphas:
        times_lists: list[float] = []
        times_tree: list[float] = []
        speedup_lists: list[float] = []
        speedup_tree: list[float] = []

        for theta in thetas:
            print(f"\ntheta = {theta:.3f}, alpha = {alpha:.3f}")

            print("Octree Lists solution: ")
            start = perf_counter()
            for _ in range(iterations):
                b_lists = oersted.b_field(
                    mesh,
                    jdensity_total,
                    mesh1.centroids,
                    solver=OctreeSolver(theta=theta, alpha=alpha),
                )

            elapsed = (perf_counter() - start) / float(iterations)
            times_lists.append(elapsed)
            speedup = direct_elapsed / elapsed
            speedup_lists.append(speedup)
            f = oersted.lorentz_forces(mesh1, jdensity, b_lists, total=True)
            f_err = (f[2] - f_upper[2]) / f_upper[2]
            assert abs(f_err) < MAX_ERR_PCT
            print(f"Elapsed: {elapsed:.3f} sec, speedup = {speedup:.3f}x")
            print(f"Total force = {f}")
            print(f"Fz error: {f_err * 100.0:.3f}%")

            print("Octree solution: (old)")
            start = perf_counter()
            for _ in range(iterations):
                b_tree = oersted.b_field(
                    mesh,
                    jdensity_total,
                    mesh1.centroids,
                    solver=OctreeSolver2Zone(theta, leaf_threshold=int(leaf_threshold)),
                )
            elapsed = (perf_counter() - start) / float(iterations)
            times_tree.append(elapsed)
            speedup = direct_elapsed / elapsed
            speedup_tree.append(speedup)
            f = oersted.lorentz_forces(mesh1, jdensity, b_tree, total=True)
            f_err = (f[2] - f_upper[2]) / f_upper[2]
            assert abs(f_err) < MAX_ERR_PCT
            print(f"Elapsed: {elapsed:.3f} sec, speedup = {speedup:.3f}x")
            print(f"Total force = {f}")
            print(f"Fz error: {f_err * 100.0:.3f}%")

        fig, ax = plt.subplots()
        ax.plot(thetas, times_lists, label="Octree Lists (3-zone)")
        ax.plot(thetas, times_tree, label="Octree (2-zone)")
        ax.plot(
            [min(thetas), max(thetas)],
            [direct_elapsed, direct_elapsed],
            label="Direct Solution",
        )
        ax.set_xlabel("Barnes Hut Angle Opening Criteria (theta)")
        ax.set_ylabel("Evaluation Time [s]")
        ax.set_title(f"Octree Testing - Helmholtz Coil\nalpha = {alpha:.3f}")
        ax.legend()
        fig.savefig(f"tests/fig/octree_test_alpha={alpha:.3f}.svg")

        fig, ax = plt.subplots()
        ax.plot(thetas, speedup_lists, label="Octree Lists (3-zone)")
        ax.plot(thetas, speedup_tree, label="Octree (2-zone)")
        ax.set_xlabel("Barnes Hut Angle Opening Criteria (theta)")
        ax.set_ylabel("Speedup vs Direct Solution")
        ax.set_title(f"Octree Testing - Helmholtz Coil\nalpha = {alpha:.3f}")
        ax.legend()
        fig.savefig(f"tests/fig/octree_speedup_alpha={alpha:.3f}.svg")


def test_octree():
    run()


if __name__ == "__main__":
    run()
