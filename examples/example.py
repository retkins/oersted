from oersted.testing import make_helmholtz
from oersted import b_field, OctreeSolver
from time import perf_counter

size = 0.015
theta = 0.5
nthreads = 0
mesh, jdensity = make_helmholtz("tests/data/ring.stp", size)

solver = OctreeSolver(theta=theta, n_threads=nthreads)


print("Oersted Example - Helmholtz Problem")
n = mesh.num_elems
print(f"n = {n:.3e} ({n * n:.3e} interactions)")


start = perf_counter()
b = b_field(mesh.to_centroid_mesh(), jdensity, mesh.centroids, solver=solver)
end = perf_counter()
elapsed = end - start

print(f"Elapsed time: {elapsed:.3f} s")
