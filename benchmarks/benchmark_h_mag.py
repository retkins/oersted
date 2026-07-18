"""Use the magnetized sphere example as a benchmark for magnetization calc"""

import oersted
from oersted import SolverSettings
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

theta = 0.5
max_leaf_size = 16
batch_size = 1
mesh_sizes = [20e-3, 15e-3, 10e-3, 8e-3, 5e-3]
mesh_sizes = np.linspace(4e-3, 20e-3, 10)

all_settings = [
    SolverSettings(method="direct", integration="point"),
    SolverSettings(method="direct", integration="element"),
    SolverSettings(
        method="octree",
        integration="point",
        theta=theta,
        max_leaf_size=max_leaf_size,
        batch_size=batch_size,
    ),
    SolverSettings(
        method="octree",
        integration="element",
        theta=theta,
        max_leaf_size=max_leaf_size,
        batch_size=batch_size,
    ),
]

results = {}

for settings in all_settings:
    timings = []
    interactions = []
    throughputs = []

    for mesh_size in mesh_sizes:
        mesh = oersted.Mesh.from_step("tests/data/sphere.stp", mesh_size)
        start = perf_counter()
        m_field = np.random.random(mesh.centroids.shape)
        start = perf_counter()
        print(f"running with {settings.method} + {settings.integration} + {mesh_size}")
        oersted.h_field(mesh, mesh.centroids, magnetization=m_field, settings=settings)
        elapsed = perf_counter() - start
        problem_size = mesh.num_elems**2
        throughput = problem_size / elapsed
        timings.append(elapsed)
        interactions.append(problem_size)
        throughputs.append(throughput)
    results[settings.method + "-" + settings.integration] = {
        "timings": timings,
        "interactions": interactions,
        "throughputs": throughputs,
    }

fig, ax = plt.subplots()
for key in results:
    result = results[key]
    ax.plot(result["interactions"], result["throughputs"], label=key)

ax.legend()
ax.set_xlabel("Interactions")
ax.set_ylabel("Throughput(interactions/s)")
ax.set_title("H_mag Benchmarks")
ax.set_xscale("log")
ax.set_yscale("log")

fig.savefig("benchmarks/figs/h_mag_benchmarks.svg")

# Print benchmarking results to command line as well
print("Benchmarking Results - Magnetization Sources")
for key in results:
    print(f"\nMethod: {key}\n---")
    print("Interactions | Throughput (int./s)")
    interactions = results[key]["interactions"]
    throughput = results[key]["throughputs"]
    for i in range(0, len(result["interactions"])):
        print(f"{interactions[i]} | {throughput[i]:.3e}")
