"""Compute the total force acting on a magnetized sphere when acted on by a
loop magnet. Run a mesh convergence study; compare against analytical.

Loop magnet: R=1.0m, square cross-section with h=0.1m, total current: I=10e6 A
Sphere: R=0.05m, z=0.2m
"""

import oersted
import numpy as np
from numpy import float64
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pathlib

# ---
# Problem parameters
# ---

solver = oersted.OctreeSolver(theta=0.25, leaf_threshold=1, alpha=0.9)
solver = oersted.DirectSolver(alpha=0.9)
current: float = 10e6  # (A)
loop_radius: float = 1.0  # (m)
sphere_radius: float = 0.05  # (m)
sphere_z: float = 0.2  # (m) height of the sphere above the xy plane
mu_r: float = 1.5
loop_mesh_size: float = 50e-3  # (m)
sphere_mesh_sizes = [30e-3, 20e-3, 15e-3, 10e-3, 8e-3]

material = oersted.LinearMaterial(mu_r)


def make_loop_magnet(current: float, mesh_size: float):
    # Mesh the magnet and compute current density at each element centroid

    step_file = pathlib.Path(__file__).parent.parent / "tests/data/loop_magnet.stp"
    loop_mesh = oersted.Mesh.from_step(str(step_file), mesh_size)
    print(f"avg z: {np.average(loop_mesh.nodes, axis=0)}")
    print(f"min z: {np.min(loop_mesh.nodes, axis=0)}")
    print(f"max z: {np.max(loop_mesh.nodes, axis=0)}")

    area: float = 0.01  # (m^2)
    j_density: NDArray[float64] = np.zeros(loop_mesh.centroids.shape)
    jmag: float = current / area
    phi: NDArray[float64] = np.atan2(
        loop_mesh.centroids[:, 1], loop_mesh.centroids[:, 0]
    )
    j_density[:, 0] = -jmag * np.sin(phi)
    j_density[:, 1] = jmag * np.cos(phi)

    # oersted.plot_mesh(
    #     loop_mesh,
    #     filename="docs/figs/example_magnetized_sphere_j_density.svg",
    #     centroids = loop_mesh.centroids,
    #     vectors=j_density,
    #     # scalars=np.linalg.norm(j_density, axis=1),
    #     vector_scale = 5.0e-2,
    #     transparency=True
    # )

    return loop_mesh, j_density


def mesh_convergence_study(
    mesh_sizes: list[float], loop_mesh: oersted.Mesh, j_density: NDArray[float64]
):

    # Parameters for the mesh convergence study
    fx = []
    fy = []
    fz = []

    for mesh_size in mesh_sizes:
        # The sphere starts at (0, 0, 0)
        mesh = oersted.Mesh.from_step("tests/data/sphere.stp", mesh_size)
        mesh.nodes[:, 2] += sphere_z

        # Plot both the sphere and loop together
        # mesh.append(loop_mesh).plot("docs/figs/example_magnetized_sphere_meshes.svg")

        # Compute the external field acting on the sphere
        bext = oersted.b_field(loop_mesh, j_density, mesh.centroids, solver=solver)

        # Solve for the magnetization of the sphere and demagnetization field
        M, H = oersted.demag_solve(mesh, material, bext / oersted.MU0, solver)

        # Compute the background field on the nodes for the kelvin force evaluation
        b_field_nodes = oersted.b_field(loop_mesh, j_density, mesh.nodes, solver=solver)

        forces = oersted.kelvin_forces(mesh, M, b_field_nodes)
        force = np.sum(forces, axis=0)

        fx.append(force[0])
        fy.append(force[1])
        fz.append(force[2])
        print(f"Fz on sphere: {force[2]}")

    return fx, fy, fz


def analytical_force() -> float:
    # Analytical force on the sphere is F = (m*grad)B
    # Sphere is on the axis; background field is only in Z-direction
    # The magnetization of the sphere is in the same direction as the background
    #   field, so we can simplify this to Fz = m_z * dBz/dz by symmetry

    # First, find m
    b_centroid: float = oersted.testing.bz_loop_axis(current, loop_radius, sphere_z)
    chi: float = 3.0 * (mu_r - 1.0) / (mu_r + 2.0)  # analytical for sphere
    M: float = (1.0 / oersted.MU0) * b_centroid * chi
    volume: float = (4.0 * np.pi / 3.0) * sphere_radius**3
    m: float = M * volume
    print(f"Magnetic field at sphere center: {b_centroid:.3f} T")
    print(f"chi = {chi:.3f}")
    print(f"Magnetization field, M: {M:.3f} A/m")
    print(f"Magnetic moment, m = {m:.3f} A m^2")

    # Then, compute the field gradient
    dbdz = oersted.testing.dbzdz_loop_axis(current, loop_radius, sphere_z)
    print(f"dBz/dz at sphere center: {dbdz:.3f} T/m")
    return m * dbdz


def plot_result(mesh_sizes, fx, fy, fz, fz_analytical):

    # Comvert to mm for plotting
    mesh_sizes = [1e3 * size for size in mesh_sizes]
    fig, ax = plt.subplots()
    ax.plot(mesh_sizes, fx, label="fx")
    ax.plot(mesh_sizes, fy, label="fy")
    ax.plot(mesh_sizes, fz, label="|fz|")
    ax.plot(
        [mesh_sizes[0], mesh_sizes[-1]],
        [fz_analytical, fz_analytical],
        "r--",
        label="|fz| (analytical)",
    )
    ax.set_xlabel("Mesh Size (mm)")
    ax.set_ylabel("Total Force (N)")
    ax.set_title("Mesh Convergence - Magnetized Sphere & Coil")
    ax.legend()
    fig.savefig("docs/figs/example_magnetized_sphere_mesh_convergence.svg")


def plot_fields_on_axis(loop_mesh: oersted.Mesh, j_density: NDArray[float64]):

    n_pts: int = 100
    z_max: float = 3.0
    axis_pts = np.zeros((n_pts, 3))
    axis_pts[:, 2] = np.linspace(0.0, z_max, n_pts)
    bz_axis = oersted.b_field(loop_mesh, j_density, axis_pts, solver=solver)

    n_pts_analytical: int = 20
    axis_pts_analytical = np.linspace(0, z_max, n_pts_analytical)
    bz_analytical = np.zeros((n_pts_analytical,))
    for i in range(n_pts_analytical):
        bz_analytical[i] = oersted.testing.bz_loop_axis(
            current, loop_radius, axis_pts_analytical[i]
        )

    fig, ax = plt.subplots()
    ax.plot(axis_pts[:, 2], bz_axis[:, 2], "k", label="oersted")
    ax.plot(axis_pts_analytical, bz_analytical, "rs", label="analytical")
    ax.set_xlabel("Distance Along Z-Axis (m)")
    ax.set_ylabel("Magnetic Flux Density, B (T)")
    ax.set_title("Magnetic Field Along the Loop Axis")
    ax.legend()
    fig.savefig("docs/figs/example_magnetized_sphere_field_on_axis.svg")


def main():
    # Create the loop
    loop_mesh, j_density = make_loop_magnet(current, loop_mesh_size)

    # Compute and plot the fields on axis
    plot_fields_on_axis(loop_mesh, j_density)

    # Run the mesh convergence study
    fx, fy, fz = mesh_convergence_study(sphere_mesh_sizes, loop_mesh, j_density)

    # Compute analytical value
    fz_analytical = analytical_force()
    print(f"Analytical force acting on the sphere: {fz_analytical:.3f} N")

    assert np.abs(fx[-1]) < 1.0
    assert np.abs(fy[-1]) < 1.0
    assert np.abs((fz[-1] - fz_analytical) / fz_analytical) < 5e-2

    plot_result(sphere_mesh_sizes, fx, fy, np.abs(fz), np.abs(fz_analytical))


if __name__ == "__main__":
    main()
