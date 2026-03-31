import numpy as np
import oersted
from oersted import Mesh
import time
import pytest

# Test parameters
infile: str = "tests/data/sphere.stp"
outfile: str = "tests/data/sphere.data"
min_size: float = 5.0  # mm
max_size: float = 15.0  # mm
b_ext_mag: float = 1.0  # T
mu_r: float = 1.5
solver = oersted.DirectSolver()
mat = oersted.materials.LinearMaterial(mu_r)

# Mesh the sphere
mesh: Mesh = oersted.mesh.mesh_step(infile, outfile, min_size, max_size)
print(f"Number of elements: {mesh.num_elems}")


@pytest.mark.xfail(reason="Magnetization calculation on mesh surface known to be incorrect", strict=False)
def test_magnetization_forces():
    """Test maxwell forces for a magnetized component under an external field"""

    # Calculate uniform background field (force should be near-zero)

    h_external = np.zeros((mesh.num_elems, 3))
    h_ext_mag: float = b_ext_mag / oersted.MU0
    h_external[:, 2] = h_ext_mag

    # Compute demag parameters: magnetization and internal H field
    start = time.perf_counter()
    M, Htotal = oersted.magnetization.demag_tet4(mesh, mat, h_external, nthreads_requested=solver.n_threads)
    elapsed = time.perf_counter() - start
    print(f"Calculation time elapsed: {elapsed:.3f} sec")

    mesh._m_field = M

    # Compute external field at mesh face centroids
    b_ext = np.zeros(mesh.surface_face_centroids.shape)
    b_ext[:, 2] = b_ext_mag
    forces = mesh.surface_forces(b_ext, mat, solver)
    total_force = np.sum(forces, axis=0)
    print(np.sum(forces, axis=0))

    face_force_mags = np.linalg.norm(forces, axis=1)
    print(f"Avg face force: {np.mean(face_force_mags)}")
    print(f"Max face force: {np.max(face_force_mags)}")
    print(f"Net force: {np.linalg.norm(total_force)}")

    assert np.abs(np.max(total_force)) < 1.0  # small value like 1 N


def test_lorentz_forces():
    """Use the maxwell stress tensor to compute the lorentz force acting on a mesh"""

    # Make the helmholz coil problem
    radius = 0.2
    total_current = 1e4
    mesh_size: float = 10.0
    mesh1 = oersted.mesh_step("tests/data/ring.stp", "", mesh_size, mesh_size)
    mesh1._nodes[:, 2] += 0.01

    mesh2 = oersted.mesh_step("tests/data/ring.stp", "", mesh_size, mesh_size)
    mesh2._nodes[:, 2] -= 0.01
    print(f"Number of elements: {mesh1.num_elems}")

    # Assign current densities to each mesh
    jmag: float = total_current / (0.02 * 0.02)
    jdensity = np.zeros((mesh1.num_elems, 3))
    phi = np.atan2(mesh1.centroids[:, 1], mesh1.centroids[:, 0])
    jdensity[:, 0] = -jmag * np.sin(phi)
    jdensity[:, 1] = jmag * np.cos(phi)

    solver = oersted.DirectSolver()

    # Compute the analytical solution by checking that the vertical force is approximately
    # equal to Fz = -2pi * R * Itotal * Br
    bavg = oersted.b_field(mesh1, jdensity, np.array([[radius, 0.0, -0.01]]))
    fz_expected = -float(2 * np.pi * radius * total_current * bavg[0, 0])
    print(f"fz expected: {fz_expected:.3f} N")

    # Compute the field at the lower coil's surface elements using both coils
    bext = oersted.b_field(mesh1, jdensity, mesh2.surface_face_centroids, solver=solver)
    bext += oersted.b_field(mesh2, jdensity, mesh2.surface_face_centroids, solver=solver)

    forces = mesh2.surface_forces(bext, mat, solver)
    total_force = np.sum(forces, axis=0)
    print(total_force)
    assert np.abs((fz_expected - total_force[2]) / fz_expected) < 1e-2

    # Check that the other components are small, like less than 1.0 N
    assert np.abs(total_force[0]) < 1.0
    assert np.abs(total_force[1]) < 1.0


def main():
    test_magnetization_forces()
    test_lorentz_forces()


if __name__ == "__main__":
    main()
