import numpy as np
import oersted
from oersted import Mesh
import time

# Test parameters
infile: str = "tests/data/sphere.stp"
outfile: str = "tests/data/sphere.data"
min_size: float = 10.0  # mm
max_size: float = 10.0  # mm
b_ext_mag: float = 1.0  # T
mu_r: float = 1.5
solver = oersted.DirectSolver()

# Mesh the sphere
mesh: Mesh = oersted.mesh.mesh_step(infile, outfile, min_size, max_size)

# Create material properties and calculate uniform background fiel
mat = oersted.materials.LinearMaterial(mu_r)
h_external = np.zeros((mesh.num_elems, 3))
h_ext_mag: float = b_ext_mag / oersted.MU0
h_external[:, 2] = b_ext_mag / oersted.MU0

# Compute demag parameters: magnetization and internal H field
start = time.perf_counter()
# M, Htotal = oersted.magnetization.demag_tet4(mesh, mat, h_external, nthreads_requested=solver.n_threads)
elapsed = time.perf_counter() - start

mesh._m_field = None

# compute external field at mesh face centroids
b_ext = np.zeros((mesh.surface_face_centroids.shape[0], 3))
b_ext[:, 2] = b_ext_mag
forces = mesh.surface_forces(b_ext, mat, solver)
print(np.sum(forces, axis=0))
centers = mesh.surface_face_centroids
normals = mesh.surface_face_normals
dots = np.sum(centers * normals, axis=1)
print(f"Inward normals: {np.sum(dots < 0)} / {len(dots)}")
