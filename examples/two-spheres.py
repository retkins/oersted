"""Compute the interaction force between two magnetized spheres in a uniform
background field."""

import oersted
from oersted import Mesh, MU0
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mesh_size: float = 0.012  # (m)
mu_r: float = 2.5
b_external_magnitude: float = 5.0  # (T)
radius: float = 0.05  # (m)
distance: float = 0.3  # (m)

# Create two spheres, each of 0.1m diameter separated by a CL-CL distance of 0.3m
# The CAD file is for a sphere centered at (0.0, 0.0, 0.0)

upper_sphere: Mesh = Mesh.from_step("tests/data/sphere.stp", mesh_size)
upper_sphere.nodes[:, 2] += distance / 2.0
lower_sphere: Mesh = Mesh.from_step("tests/data/sphere.stp", mesh_size)
lower_sphere.nodes[:, 2] -= distance / 2.0

# Combine the meshes together so that we can compute field interactions
combined_mesh = lower_sphere.append(upper_sphere)
# combined_mesh.plot("docs/figs/example_two_spheres_mesh.svg")

# Compute a constant external field
h_external = np.zeros(combined_mesh.centroids.shape)
h_external[:, 2] = b_external_magnitude / oersted.MU0

# Assign a material
material = oersted.LinearMaterial(mu_r)

# Create a solver
# solver = oersted.OctreeSolver(leaf_threshold=1, alpha=0.9, tol=1e0)
# solver = oersted.DirectSolver(alpha=0.5, tol=1e0)
settings = oersted.SolverSettings(under_relaxation_factor=0.5, atol=1.0)

# Demagnetization solve on both spheres
M, _ = oersted.demag_solve(combined_mesh, material, h_external, settings=settings)

# Select only the magnetization field in the appropriate sphere
M_lower = M[: lower_sphere.num_elems]
M_upper = M[lower_sphere.num_elems :]

# Compute the total field acting on the nodes of both spheres,
# using only the other sphere as a source
h_field_nodes_upper = oersted.h_mag(
    lower_sphere, M_lower, upper_sphere.nodes, solver=solver
)
h_field_nodes_lower = oersted.h_mag(
    upper_sphere, M_upper, lower_sphere.nodes, solver=solver
)
h_field_nodes_upper[:, 2] += b_external_magnitude / MU0
h_field_nodes_lower[:, 2] += b_external_magnitude / MU0

# Compute the forces acting on each sphere
forces_upper = oersted.kelvin_forces(upper_sphere, M_upper, MU0 * h_field_nodes_upper)
forces_lower = oersted.kelvin_forces(lower_sphere, M_lower, MU0 * h_field_nodes_lower)

# Sum the forces and output
force_upper = np.sum(forces_upper, axis=0)
force_lower = np.sum(forces_lower, axis=0)

print(
    f"Force on upper sphere: \
        ({force_upper[0]:.1f},{force_upper[1]:.1f},{force_upper[2]:.1f}) N "
)
print(
    f"Force on lower sphere: \
        ({force_lower[0]:.1f},{force_lower[1]:.1f},{force_lower[2]:.1f}) N"
)

# Analytical solution; see: <https://en.wikipedia.org/wiki/Force_between_magnets#Magnetic_dipole-dipole_interaction>
N: float = 1.0 / 3.0  # demag factor for sphere
H_ext: float = b_external_magnitude / oersted.MU0
M_analytical = 3 * (mu_r - 1) / (mu_r + 2) * H_ext
H_analytical: float = H_ext - N * M_analytical
B_analytical = oersted.MU0 * (H_ext + (1.0 - N) * M_analytical)
m = (4.0 / 3.0) * np.pi * (radius**3) * M_analytical
F_analytical = 3 * MU0 * m * m / (2 * np.pi * distance**4)
print(f"Analytical M field: {M_analytical:.3f} A/m")
print(f"Analytical magnetic moment: {m:.3f} A-m^2")
print(f"Analytical force: {F_analytical}")

# Compute the fields along a line going through the spheres
n = 1000
targets = np.zeros((n, 3))
targets[:, 2] = np.linspace(-0.2, 0.2, n)
htargets = oersted.h_mag(combined_mesh, M, targets, solver)
fig, ax = plt.subplots()
ax.plot(targets[:, 2], htargets[:, 2])
ax.set_xlabel("Distance Along Z-Axis (m)")
ax.set_ylabel("H-Field (A/m)")
ax.set_title("Two Spheres Example: H-Field Along Z-Axis")
fig.savefig("docs/figs/two-spheres-hfield.svg")


# For testing
def main():
    # We get better results with a finer mesh, but 5% is good for testing
    assert np.abs(force_lower[2] - F_analytical) / F_analytical < 0.05
    assert np.abs(force_lower[0]) < 1.0
    assert np.abs(force_lower[1]) < 1.0

    # Make sure they're equal and opposite
    assert np.linalg.norm(force_lower + force_upper) < 1.0


if __name__ == "__main__":
    main()
