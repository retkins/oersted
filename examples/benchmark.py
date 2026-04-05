"""Use the helmholtz coil problem as a benchmarking example"""

import oersted
from oersted import OctreeSolver
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def main(nbenches: int = 1, theta: float = 0.5, mesh_size_max: float = 33.0, mesh_size_min: float = 33.0):
    mesh_sizes = np.linspace(mesh_size_max, mesh_size_min, nbenches)
    direct_times = []
    direct_interactions = []
    est_direct_times = []
    est_direct_interactions = []
    i_est = 0
    octree_times = np.zeros(nbenches)
    interactions = np.zeros(nbenches)

    for i, mesh_size in enumerate(mesh_sizes):
        mesh, jdensity = oersted.testing.make_helmholtz("tests/data/ring.stp", mesh_size)
        n = mesh.num_elems
        interactions[i] = n * n

        if n < 50000:
            start = perf_counter()
            _ = oersted.b_field(mesh.to_centroid_mesh(), jdensity, mesh.centroids)
            end = perf_counter()
            direct_times.append(end - start)
            est_direct_times = [end - start]
            direct_interactions.append(n * n)
            est_direct_interactions = [n * n]
            i_est = i
        else:
            m = (direct_times[i_est] - direct_times[0]) / (interactions[i_est] - interactions[0])
            b = direct_times[i_est] - m * interactions[i_est]
            est_direct_interactions.append(n * n)
            est_direct_times.append(m * n * n + b)

        start = perf_counter()
        _ = oersted.b_field(mesh.to_centroid_mesh(), jdensity, mesh.centroids, OctreeSolver(theta=theta))
        end = perf_counter()
        octree_times[i] = end - start

    all_direct_times = direct_times + est_direct_times[1:]
    print("Sources/Targets | Interactions | Direct Time | Octree Time | Speedup ")
    for i in range(0, nbenches):
        n = np.sqrt(interactions[i])
        speedup = all_direct_times[i] / octree_times[i]

        print(f"{n:.0f} | {interactions[i]:.3e} | {all_direct_times[i]:.3f}" + f"| {octree_times[i]:.3f} | {speedup:.3f}x")

    # Plot solution times
    fig, ax = plt.subplots()
    ax.plot(direct_interactions, direct_times, "r", label="direct")
    ax.plot(interactions, octree_times, "k", label="octree")
    ax.plot(est_direct_interactions, est_direct_times, "r--", label="direct trend")
    ax.set_xlabel("Interactions ($N^2$)")
    ax.set_ylabel("Solution time [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Oersted Benchmarks: Helmholtz Coil Problem\n$\\theta={theta:.2}$")
    ax.legend()
    fig.savefig("tests/fig/benchmarks.svg")

    # Plot interactions per second
    direct_throughput = [i / (t * 1e9) for (i, t) in zip(direct_interactions, direct_times, strict=True)]
    est_direct_throughput = [i / (t * 1e9) for (i, t) in zip(est_direct_interactions, est_direct_times, strict=True)]
    octree_throughput = [i / (t * 1e9) for (i, t) in zip(interactions, octree_times, strict=True)]

    fig, ax = plt.subplots()
    ax.plot(direct_interactions, direct_throughput, "r", label="direct")
    ax.plot(est_direct_interactions, est_direct_throughput, "r--", label="direct (trend)")
    ax.plot(interactions, octree_throughput, "k", label="octree")
    ax.set_xlabel("Interactions ($N^2$)")
    ax.set_ylabel("Throughput ($1e9$ interactions/sec)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Oersted Benchmarks: Helmholtz Coil Problem\n$\\theta={theta:.2}$")
    ax.legend()
    fig.savefig("tests/fig/benchmarks_throughput.svg")


if __name__ == "__main__":
    main(nbenches=10, theta=0.5, mesh_size_max=33.0, mesh_size_min=1.5)
