"""Compute centerline and self magnetic fields in a solenoid
centered at the origin with main axis in Z.

Solenoid has parameters:
- Inner radius = 0.025m
- Outer radius = 0.050m (thk = 0.025m)
- Length = 0.25m
- Current density = 100 MA/m^2

Note: there's some sort of units mismatch with gmsh
"""

import oersted
from oersted import Mesh, SolverSettings
from oersted.testing import bz_finite_length_solenoid
import numpy as np

# ---
# Solenoid properties
# ---
ri: float = 0.025  # (m)
ro: float = 0.050  # (m)
dr: float = ro - ri
length: float = 0.250  # (m)
jmag: float = 1e8  # (A/m2)

# ---
# Solver settings
# ---
max_leaf_size = 16
batch_size = 1
theta = 0.5
direct_element = SolverSettings(method="direct", integration="element")
all_settings = [
    direct_element,
    SolverSettings(method="direct", integration="point"),
    SolverSettings(
        method="octree",
        integration="element",
        max_leaf_size=max_leaf_size,
        theta=theta,
    ),
    SolverSettings(
        method="octree",
        integration="point",
        max_leaf_size=max_leaf_size,
        theta=theta,
    ),
]

# ---
# Test settings
# ---
mesh_size: float = 12.0
ntargets_axis: int = 100
MAX_ERR = 1e-2

# load mesh
mesh: Mesh = oersted.mesh_step("tests/data/solenoid.stp", mesh_size, mesh_size)
n: int = mesh.num_elems

# assign current density
jdensity = np.zeros((n, 3))
phi = np.atan2(mesh.centroids[:, 1], mesh.centroids[:, 0])
jdensity[:, 0] = -jmag * np.sin(phi)
jdensity[:, 1] = jmag * np.cos(phi)


def test_on_axis():
    targets_axis = np.zeros((ntargets_axis, 3))
    targets_axis[:, 2] = np.linspace(-0.125, 0.125, ntargets_axis)

    b_direct = oersted.b_field(
        mesh, targets_axis, jdensity=jdensity, settings=direct_element
    )

    print("On-axis solenoid test results:")
    for settings in all_settings:
        b = oersted.b_field(mesh, targets_axis, jdensity=jdensity, settings=settings)
        err = oersted.max_verr(b, b_direct)
        print(
            f"Method = {settings.method}, integration = {settings.integration}, \
            err = {err}"
        )
        assert err < MAX_ERR


def test_centroid():
    # bz_analytical: float = oersted.MU0 * jmag * dr
    r_avg = 0.5 * (ro + ri)
    target = np.array([[0.0, 0.0, 0.0]])
    bz_analytical: float = bz_finite_length_solenoid(jmag, length, r_avg, dr, 0.0)

    for settings in all_settings:
        bz = oersted.b_field(mesh, target, jdensity=jdensity, settings=settings)[0, 2]

        assert (bz - bz_analytical) / bz_analytical < MAX_ERR


def test_self_fields():
    targets = mesh.centroids
    b_direct = oersted.b_field(
        mesh, targets, jdensity=jdensity, settings=direct_element
    )

    print("Self-fields solenoid test results:")
    for settings in all_settings:
        b = oersted.b_field(mesh, targets, jdensity=jdensity, settings=settings)

        # point method inaccurate inside mesh, but needs to be bounded for testing
        if settings.integration == "point":
            errtol = 15e-2
            err = oersted.mean_verr(b, b_direct)

        # Element integration should be much stricter
        else:
            max_err = oersted.max_verr(b, b_direct)
            assert max_err < 10e-2
            errtol = MAX_ERR
            err = oersted.mean_verr(b, b_direct)

        print(
            f"Method = {settings.method}, integration = {settings.integration}, \
            err = {err}"
        )
        assert err < errtol


if __name__ == "__main__":
    test_on_axis()
    test_centroid()
    test_self_fields()
