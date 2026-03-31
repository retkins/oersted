from oersted.testing import make_helmholtz
from oersted import CentroidMesh, b_field, OctreeSolver
from time import perf_counter

size = 15.0
theta = 0.5
nthreads = 0
centroids, vol, jdensity = make_helmholtz(size)
mesh = CentroidMesh(centroids, vol)

solver = OctreeSolver(theta=theta, n_threads=nthreads)


print("Oersted Example - Helmholtz Problem")
n = centroids.shape[0]
print(f"n = {n:.3e} ({n * n:.3e} interactions)")


start = perf_counter()
b = b_field(mesh, jdensity, centroids, solver=solver)
end = perf_counter()
elapsed = end - start

print(f"Elapsed time: {elapsed:.3f} s")
