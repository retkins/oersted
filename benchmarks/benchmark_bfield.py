"""Use the magnetized sphere example as a benchmark for magnetization calc"""

import oersted
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

theta = 0.5
leaf_threshold = 1

methods = ["point", "tet4"]
solvers = ["direct", "octree"]
mesh_sizes = [20e-3, 15e-3, 10e-3, 8e-3, 5e-3]

results = {}

for solver in solvers:
    for method in methods:
        timings = []
        interactions = []
        throughputs = []

        for mesh_size in mesh_sizes:
            mesh = oersted.Mesh.from_step("tests/data/sphere.stp", mesh_size)
            start = perf_counter()
            if method == "point":
                mesh = mesh.to_centroid_mesh()

            use_solver = oersted.DirectSolver() if solver == "direct" else oersted.OctreeSolver(theta=theta, leaf_threshold=leaf_threshold)
            jdensity = np.random.random(mesh.centroids.shape)
            start = perf_counter()
            print(f"running with {solver} + {method} + {mesh_size}")
            oersted.b_field(mesh, jdensity, mesh.centroids, solver=use_solver)
            elapsed = perf_counter() - start
            problem_size = mesh.num_elems**2
            throughput = problem_size / elapsed
            timings.append(elapsed)
            interactions.append(problem_size)
            throughputs.append(throughput)
        results[solver + "-" + method] = {"timings": timings, "interactions": interactions, "throughputs": throughputs}

fig, ax = plt.subplots()
for key in results:
    result = results[key]
    ax.plot(result["interactions"], result["throughputs"], label=key)

ax.legend()
ax.set_xlabel("Interactions")
ax.set_ylabel("Throughput(interactions/s)")
ax.set_title("Bfield Benchmarks")
ax.set_xscale("log")
ax.set_yscale("log")

fig.savefig("benchmarks/figs/b_field_benchmarks.svg")
