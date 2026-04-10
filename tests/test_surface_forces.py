import numpy as np
import oersted
from oersted import Mesh

# Test parameters
mesh_size: float = 15.0  # mm
b_ext_mag: float = 5.0  # T
mu_r: float = 1.5
solver = oersted.DirectSolver()
mat = oersted.materials.LinearMaterial(mu_r)

def calculate_magnetization_forces_on_sphere(b_ext_mag: float, b_ext_grad: float):
    """Test maxwell forces for a magnetized component under an external field

    Assume that the gradient in the field is in the z-direction"""

    # Mesh the sphere
    mesh: Mesh = oersted.mesh.mesh_step("tests/data/sphere.stp", mesh_size, mesh_size)
    mesh.nodes[:, 2] -= 1.0

    # Calculate background field
    h_external = np.zeros((mesh.num_elems, 3))
    h_external[:, 2] = (1.0 / oersted.MU0) * (b_ext_mag + b_ext_grad * mesh.centroids[:, 2])

    # Compute demag parameters: magnetization and internal H field
    M, Htotal = oersted.magnetization.demag_tet4(mesh, mat, h_external, solver)

    # Compute external field at mesh nodes
    h_ext_nodes = np.zeros((mesh.num_nodes, 3))
    h_ext_nodes[:, 2] = (1.0 / oersted.MU0) * (b_ext_mag + b_ext_grad * mesh.nodes[:, 2])
    forces = oersted.kelvin_forces(mesh, M, h_ext_nodes)

    # Compute analytical result: f = (m*grad)H
    m = np.average(M, axis=0)[2] * np.sum(mesh.volumes)
    analytical_force = m * b_ext_grad / oersted.MU0

    return np.sum(forces, axis=0), analytical_force


def test_lorentz_forces():
    """Use the maxwell stress tensor to compute the lorentz force acting on a mesh"""

    # Make the helmholz coil problem
    radius = 0.2
    total_current = 1e4
    mesh_size: float = 15.0
    mesh1 = oersted.mesh_step("tests/data/ring.stp", mesh_size, mesh_size)
    mesh1._nodes[:, 2] += 0.01

    mesh2 = oersted.mesh_step("tests/data/ring.stp", mesh_size, mesh_size)
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
    bext = oersted.b_field(mesh1, jdensity, mesh2.surface.centroids, solver=solver)
    bext += oersted.b_field(mesh2, jdensity, mesh2.surface.centroids, solver=solver)

    forces = oersted.maxwell_forces(mesh2.surface, bext)
    total_force = np.sum(forces, axis=0)
    error = np.abs((fz_expected - total_force[2]) / fz_expected)
    print(f"total force, lower: {total_force}")

    print(f"error: {100 * error:.2f} %")
    assert error < 1e-2

    # Check that the other components are small, like less than 1.0 N
    assert np.abs(total_force[0]) < 1.0
    assert np.abs(total_force[1]) < 1.0

    # Compute the field at the lower coil's surface elements using both coils
    bext = oersted.b_field(mesh1, jdensity, mesh1.surface.centroids, solver=solver)
    bext += oersted.b_field(mesh2, jdensity, mesh1.surface.centroids, solver=solver)

    forces = oersted.maxwell_forces(mesh1.surface, bext)
    total_force = np.sum(forces, axis=0)
    error = np.abs((-fz_expected - total_force[2]) / fz_expected)
    print(f"total force, upper: {total_force}")

    print(f"error: {100 * error:.2f} %")
    assert error < 1e-2

    # Check that the other components are small, like less than 1.0 N
    assert np.abs(total_force[0]) < 1.0
    assert np.abs(total_force[1]) < 1.0


def main():
    test_lorentz_forces()

    # Magnetized sphere in uniform background field
    b_ext_mag = 5.0
    b_ext_grad = 0.0
    total_force, analytical_force = calculate_magnetization_forces_on_sphere(b_ext_mag, b_ext_grad)
    print(f"total:      {total_force}")
    print(f"analytical: {analytical_force}")
    assert np.abs(total_force[0]) < 1e-8
    assert np.abs(total_force[1]) < 1e-8
    assert np.abs(total_force[2] - analytical_force) < 1e-8  # expected value is zero

    # Magnetized sphere in linear background field gradient
    b_ext_mag = 5.0
    b_ext_grad = 1.0
    total_force, analytical_force = calculate_magnetization_forces_on_sphere(b_ext_mag, b_ext_grad)
    print(f"total:      {total_force}")
    print(f"analytical: {analytical_force}")
    assert np.abs(total_force[0]) < 1e-8
    assert np.abs(total_force[1]) < 1e-8
    assert np.abs((total_force[2] - analytical_force) / analytical_force) < 1e-4


if __name__ == "__main__":
    main()
