import oersted
from oersted import SolverSettings
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
import os
import platform

min_theta = 0.1
max_theta = 0.5
n_theta = 5
n_meshes = 10
min_mesh_size = 4e-3
max_mesh_size = 15e-3
max_leaf_size = 16
batch_size = 1
err_threshold = 1.0  # %
err_target = 0.1  # %
n_iterations = 5
mesh_size = 8e-3
thetas = np.linspace(min_theta, max_theta, n_theta)

mesh, jdensity = oersted.make_helmholtz("tests/data/ring.stp", mesh_size)
# mesh = oersted.Mesh.from_step("tests/data/solenoid.stp", mesh_size)
targets = mesh.centroids
# mesh.plot()

platform_info = (
    f"Platform: {platform.system()}/{platform.machine()}/{os.cpu_count()} cores"
)


def bench_error(axs):

    start = perf_counter()
    b_direct = oersted.b_field(mesh, mesh.centroids, jdensity=jdensity)
    time_direct = perf_counter() - start

    max_err = []
    mean_err = []
    speedup = []
    timing = []
    for theta in thetas:
        start = perf_counter()
        for _ in range(n_iterations):
            b = oersted.b_field(
                mesh,
                targets,
                jdensity=jdensity,
                settings=SolverSettings(
                    method="octree",
                    integration="element",
                    theta=theta,
                    max_leaf_size=max_leaf_size,
                    batch_size=batch_size,
                ),
            )
        elapsed = (perf_counter() - start) / n_iterations
        timing.append(elapsed)
        speedup.append(time_direct / elapsed)

        max_err.append(100 * oersted.max_verr(b, b_direct))
        mean_err.append(100 * oersted.mean_verr(b, b_direct))

    plot_error(axs[0, 0], max_err, mean_err)
    plot_timing(axs[0, 1], timing)
    plot_speedup(axs[1, 0], speedup)


def bench_scaling(axs, mesh_sizes):
    problem_size = []
    timing = []

    # speedup[theta][mesh_size]
    all_speedup = []
    for mesh_size in mesh_sizes:
        speedup = []
        mesh, jdensity = oersted.make_helmholtz("tests/data/ring.stp", mesh_size)
        problem_size.append(mesh.centroids.shape[0])
        start = perf_counter()
        _ = oersted.b_field(mesh, mesh.centroids, jdensity=jdensity)
        direct_elapsed = perf_counter() - start

        for theta in thetas:
            start = perf_counter()
            for _ in range(n_iterations):
                _ = oersted.b_field(
                    mesh,
                    mesh.centroids,
                    jdensity=jdensity,
                    settings=SolverSettings(
                        method="octree",
                        integration="element",
                        theta=theta,
                        max_leaf_size=max_leaf_size,
                        batch_size=batch_size,
                    ),
                )
            elapsed = (perf_counter() - start) / n_iterations
            timing.append(elapsed)
            speedup.append(direct_elapsed / elapsed)
        all_speedup.append(speedup)

    all_speedup = np.array(all_speedup)
    plot_scaling(axs[1, 1], problem_size, all_speedup.T, theta)


def plot_error(ax, max_err, mean_err):
    # fig, ax = plt.subplots()
    ax.plot(thetas, max_err, label="Max")
    ax.plot(thetas, mean_err, label="Mean")
    ax.plot(
        [thetas[0], thetas[-1]],
        [err_threshold, err_threshold],
        "r--",
        label="Threshold",
    )
    ax.plot([thetas[0], thetas[-1]], [err_target, err_target], "g--", label="Target")
    ax.set_xlabel("theta")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Relative Vector Error")
    ax.legend()
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2e1)
    # fig.tight_layout()
    from matplotlib.ticker import ScalarFormatter

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())


def plot_timing(ax, timing):
    # fig, ax = plt.subplots()
    ax.plot(thetas, timing)
    ax.set_xlabel("theta")
    ax.set_ylabel("Solve Time (s)")
    ax.set_title(f"Solve Time, n = {targets.shape[0]} elements")


def plot_speedup(ax, speedup):
    # fig, ax = plt.subplots()
    ax.plot(thetas, speedup)
    ax.set_xlabel("theta")
    ax.set_ylabel("Speedup ")
    ax.set_title(f"Speedup, n = {targets.shape[0]} elements")


def plot_scaling(ax, problem_size, speedup, theta):
    for i, theta in enumerate(thetas):
        ax.plot(problem_size, speedup[i], label=f"theta = {theta:.1f}")
    ax.set_xlabel("Problem Size (N elements)")
    ax.set_ylabel("Speedup")
    ax.set_title("Scaling")
    ax.legend()


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2, layout="constrained", figsize=(8, 6))
    fig.suptitle("oersted Barnes-Hut Solver Benchmarks")
    bench_error(axs)
    bench_scaling(axs, np.linspace(min_mesh_size, max_mesh_size, n_meshes))
    fig.savefig("benchmarks/figs/octree_benchmarks.svg")
