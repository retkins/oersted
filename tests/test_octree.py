"""Use the Helmholz coil problem to measure the accuracy and speed of the (new) 3-zone
    octree method

Error metric: total force acting on the upper coil is < 0.5% error from the
direct solution
"""

import oersted
import numpy as np
from oersted import SolverSettings
from numpy import abs
from time import perf_counter
import matplotlib.pyplot as plt

# Problem parameters
thetas = np.linspace(0.05, 0.5, 10)
max_leaf_size = 16
mesh_size: float = 15.0  # (m)
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
b_upper = oersted.b_field(
    mesh,
    mesh1.centroids,
    jdensity=jdensity_total,
    settings=SolverSettings(method="direct"),
)
direct_elapsed = perf_counter() - start
f_upper = oersted.lorentz_forces(mesh1, jdensity, b_upper, total=True)

print(f"Direct solution\ntime = {direct_elapsed:.3f} sec\nTotal_force = {f_upper}")


def run():

    times_tree: list[float] = []
    speedup_tree: list[float] = []
    for theta in thetas:
        print(f"\ntheta = {theta:.3f}")

        print("Octree solution: ")
        start = perf_counter()
        for _ in range(iterations):
            b_lists = oersted.b_field(
                mesh,
                mesh1.centroids,
                jdensity=jdensity_total,
                settings=SolverSettings(
                    theta=theta,
                    method="octree",
                    max_leaf_size=max_leaf_size,
                ),
            )

        elapsed = (perf_counter() - start) / float(iterations)
        times_tree.append(elapsed)
        speedup = direct_elapsed / elapsed
        speedup_tree.append(speedup)
        f = oersted.lorentz_forces(mesh1, jdensity, b_lists, total=True)
        f_err = (f[2] - f_upper[2]) / f_upper[2]
        assert abs(f_err) < MAX_ERR_PCT
        print(f"Elapsed: {elapsed:.3f} sec, speedup = {speedup:.3f}x")
        print(f"Total force = {f}")
        print(f"Fz error: {f_err * 100.0:.3f}%")

    fig, ax = plt.subplots()
    ax.plot(thetas, times_tree, label="Octree")
    ax.plot(
        [min(thetas), max(thetas)],
        [direct_elapsed, direct_elapsed],
        label="Direct Solution",
    )
    ax.set_xlabel("Barnes Hut Angle Opening Criteria (theta)")
    ax.set_ylabel("Evaluation Time [s]")
    ax.set_title("Octree Testing - Helmholtz Coil")
    ax.legend()
    fig.savefig("tests/fig/octree_test_alpha.svg")

    fig, ax = plt.subplots()
    ax.plot(thetas, speedup_tree, label="Octree")
    ax.set_xlabel("Barnes Hut Angle Opening Criteria (theta)")
    ax.set_ylabel("Speedup vs Direct Solution")
    ax.set_title("Octree Testing - Helmholtz Coil")
    ax.legend()
    fig.savefig("tests/fig/octree_speedup.svg")


def check_large_model():
    """Test that large models don't run out of memory due to overly large interaction
    list allocations
    Note: this function basically tests gmsh, so its not run as part of CI
    """
    mesh_size = 1e-3
    mesh, jdensity = oersted.make_ring(mesh_size)

    print(f"Size: {mesh.num_elems}")
    settings = oersted.SolverSettings(method="octree", theta=0.5)
    start = perf_counter()
    _ = oersted.b_field(mesh, mesh.centroids, jdensity=jdensity, settings=settings)
    end = perf_counter()
    print(
        f"Solved {mesh.num_elems} element self-fields problem in {end - start:.3f} sec"
    )


def check_j_accuracy():

    theta = 0.5
    direct = SolverSettings(method="direct", integration="element")
    all_settings = [
        SolverSettings(
            method="octree",
            integration="element",
            theta=theta,
            multipole_order="monopole",
        ),
        SolverSettings(
            method="octree",
            integration="element",
            theta=theta,
            multipole_order="dipole",
        ),
    ]

    mesh_size = 10e-3
    mesh, jdensity = oersted.make_ring(mesh_size=mesh_size)
    # Make test off origin to catch symmetry issues
    shift = 10.0
    mesh = oersted.Mesh(mesh.nodes - shift, mesh.connectivity)

    targets = mesh.centroids
    b_direct = oersted.b_field(mesh, targets, jdensity=jdensity, settings=direct)
    a_direct = oersted.a_field(mesh, targets, jdensity=jdensity, settings=direct)

    for settings in all_settings:
        err_tol = 5e-2 if settings.integration == "octree" else 12e-2
        print(
            f"method = {settings.method}, integration = {settings.integration}, expansion = {settings.multipole_order}"
        )
        print("bfield")
        b = oersted.b_field(mesh, targets, jdensity=jdensity, settings=settings)
        err = oersted.mean_verr(b, b_direct)
        print(f"Mean err: {100.0 * err:.3f} %")
        assert err < err_tol

        print("afield")
        a = oersted.a_field(mesh, targets, jdensity=jdensity, settings=settings)
        err = oersted.mean_verr(a, a_direct)
        print(f"Mean err: {100.0 * err:.3f} %")
        assert err < err_tol


def test_octree():
    run()
    check_j_accuracy()


if __name__ == "__main__":
    run()
    check_j_accuracy()
    # check_large_model()
