"""Create a mesh of a solenoid, calculate the fields around it, and display the
results."""

import oersted
from oersted import Mesh
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

mesh_size = 0.010
mesh: Mesh = oersted.Mesh.from_step("tests/data/solenoid-short.step", mesh_size)
mesh.plot("docs/figs/solenoid-short-mesh.svg")

# Make current density vectors
j_density_magnitude = 1e8  # (A/m2)
j_density = np.zeros((mesh.num_elems, 3))
phi = np.atan2(mesh.centroids[:, 1], mesh.centroids[:, 0])
j_density[:, 0] = -j_density_magnitude * np.sin(phi)
j_density[:, 1] = j_density_magnitude * np.cos(phi)

# Make the target points for evaluation
x = np.linspace(-0.1, 0.1, 100)
z = np.linspace(-0.1, 0.1, 100)
X, Z = np.meshgrid(x, z)
n = len(X.flatten())
targets = np.vstack((X.flatten(), np.full((n,), 0.0), Z.flatten())).T

# Solve for self-fields on the solenoid
start = perf_counter()
solver = oersted.OctreeSolver()
b = oersted.b_field(mesh, j_density, mesh.centroids, solver=solver)
elapsed = perf_counter() - start
print(f"Elapsed time: {elapsed:.3f} sec")
print(f"\t({mesh.num_elems**2 / elapsed:.2e} interactions/sec)")

# Plot the solenoid with contour+vector plots of the field
bmag = np.linalg.norm(b, axis=1)
oersted.plot_mesh(
    mesh,
    filename="docs/figs/solenoid-fields-3d.svg",
    scalars=bmag,
    centroids=mesh.centroids,
    vectors=b,
)

# Solve for background fields on a cut-plane (XZ)
start = perf_counter()
b = oersted.b_field(mesh, j_density, targets, solver=oersted.OctreeSolver())
elapsed = perf_counter() - start
print(f"Elapsed time: {elapsed:.3f} sec")
print(f"\t({mesh.num_elems**2 / elapsed:.2e} interactions/sec)")


# Plot the fields on a cut plane
def plot():
    bmag = np.linalg.norm(b, axis=1)
    bx = b[:, 0].reshape(X.shape)
    bz = b[:, 2].reshape(Z.shape)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.streamplot(X, Z, bx, bz, color="black", linewidth=0.5)
    im = ax.imshow(
        bmag.reshape(X.shape),
        origin="lower",
        interpolation="bicubic",
        norm="log",
        extent=(x.min(), x.max(), z.min(), z.max()),
    )
    fig.colorbar(im, label="$|\\vec{B}|$ [T]", ax=ax)
    # Plot the solenoid cross-section
    ax.plot(
        [0.025, 0.050, 0.050, 0.0250, 0.025], [0.025, 0.025, -0.025, -0.025, 0.025], "k"
    )
    ax.plot(
        [-0.025, -0.050, -0.050, -0.0250, -0.025],
        [0.025, 0.025, -0.025, -0.025, 0.025],
        "k",
    )
    ax.set_xlim(-0.10, 0.10)
    ax.set_ylim(-0.10, 0.10)
    ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10])
    ax.set_yticks([-0.10, -0.05, 0.0, 0.05, 0.10])
    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Z Coordinate (m)")
    ax.set_title("Finite Length Solenoid\nMagnetic Flux Density (T)")
    fig.tight_layout()
    fig.savefig("docs/figs/solenoid-fields.svg")


# For testing
def main():
    # Placeholder for now
    plot()
    assert True


if __name__ == "__main__":
    main()
