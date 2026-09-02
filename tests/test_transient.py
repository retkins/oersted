"""Compute the torque acting on a torus with a ring coil inside of it, when the
ring coil suddenly loses current and the resultant eddy currents in the torus interact
with a cross field.
"""

import oersted
import numpy as np
import matplotlib.pyplot as plt


def test_transient():
    cross_field = np.array([1.0, 0.0, 0.0])  # (T)

    # Make a torus
    rho = 1e-7
    major_radius = 0.10
    minor_radius = 0.02
    wall_thickness = 0.01
    mesh_size = 0.012
    torus = oersted.testing.make_torus(
        major_radius, minor_radius, wall_thickness, mesh_size
    )
    print(f"Mesh size: {torus.num_elems} elements")

    # Make a ring coil
    ring_current = 1e6
    b = 0.005
    h = 0.005
    ring_jmag = ring_current / (b * h)
    ring, ring_jdensity = oersted.testing.make_ring(
        major_radius, 0.0, b, h, b, jmag=ring_jmag
    )

    # mesh = torus.append(ring)
    # oersted.mesh.plot_mesh(mesh, transparency=True, filename="docs/figs/torus_mesh.svg")

    # Time properties
    nt = 50
    t_end = 0.015  # (s)
    t_ramp = 0.010  # (s)
    time = np.linspace(0, t_end, nt)
    scale = np.maximum(1.0 - time / t_ramp, 0.0)

    # Compute external fields acting on the torus (from the ring coil) over time
    a_ext = np.zeros((nt, torus.num_elems, 3))
    b_ext = np.zeros((nt, torus.num_elems, 3))

    # A and B fields at maximum coil current
    a_unit = oersted.a_field(ring, torus.centroids, jdensity=ring_jdensity)
    b_unit = oersted.b_field(ring, torus.centroids, jdensity=ring_jdensity)

    # Scale the fields to produce a simple linear decay over time
    a_ext = scale[:, None, None] * a_unit
    b_ext = scale[:, None, None] * b_unit

    fig, ax = plt.subplots()
    ax.plot(time, scale * ring_current)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Total Current in Driving Coil")
    fig.savefig("docs/figs/driving-coil-current.svg")

    # Solve
    results = oersted.transient_solve(torus, rho, nt, t_end, a_ext, b_ext)

    # Compute results

    # oersted.mesh.plot_mesh(
    #     torus,
    #     "docs/figs/torus-currents.svg",
    #     centroids=torus.centroids,
    #     vectors=results.j[25, :, :],
    #     vector_scale=mesh_size / 2,
    #     transparency=True,
    # )

    # Torque about centroid of torus, which is the origin
    f = np.cross(results.j, cross_field) * torus.volumes[None, :, None]
    torque = np.sum(np.cross(torus.centroids[None, :, :], f), axis=1)

    # Compute semi-analytic solution by estimating mutual inductance and L/R properties
    # of the torus, using a thin-walled approximation good for R/a <= 5.0
    phi = np.atan2(torus.centroids[:, 1], torus.centroids[:, 0])
    a_phi = -a_unit[:, 0] * np.sin(phi) + a_unit[:, 1] * np.cos(phi)
    a_phi_avg = np.sum(a_phi * torus.volumes) / np.sum(torus.volumes)
    M = a_phi_avg * 2.0 * np.pi * major_radius / ring_current
    L_tw = (
        oersted.MU0 * major_radius * (np.log(8.0 * major_radius / minor_radius) - 2.0)
    )
    R_tw = (
        rho
        * (2.0 * np.pi * major_radius)
        / (2.0 * np.pi * minor_radius * wall_thickness)
    )
    tau = L_tw / R_tw
    emf = M * ring_current / t_ramp
    I_ss = emf / R_tw
    current = np.where(
        time <= t_ramp,
        I_ss * (1.0 - np.exp(-time / tau)),
        I_ss * (1.0 - np.exp(-t_ramp / tau)) * np.exp(-(time - t_ramp) / tau),
    )
    m_z = current * np.pi * major_radius**2
    torque_analytic = m_z * cross_field[0]

    avg_error = np.average((torque[:, 1] - torque_analytic) / (torque_analytic + 1e-6))
    print(f"Avg error: {avg_error:.3f}")

    fig, ax = plt.subplots()
    ax.plot(time, torque[:, 1], label="oersted")
    ax.plot(time, torque_analytic, label="analytic")
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque on Torus (N*m)")
    ax.set_title("Torque on Torus During Transient Coil Event")
    fig.savefig("docs/figs/torus_torque.svg")

    assert avg_error < 0.01


if __name__ == "__main__":
    test_transient()
