"""Compare a-field computations against b' = curl(a)

Use a helmholtz coil to generate fields and evaluate b vs b' near the center, away from
the source elements
"""

# TODO: enable testing for all source kernels

import numpy as np
import oersted
import pathlib

# Test parameters
theta = 0.07
MAX_ERR = 1e-3

# Target grid parameters
n_grid = 10
xrange = (0.0, 1.5e-3)
yrange = (0.0, 1.5e-3)
zrange = (-2e-3, 2e-3)

mesh_size = 10e-3  # (m)
jmag = 1e8
step_file: pathlib.Path = pathlib.Path(__file__).parent / "../tests/data/ring.stp"
mesh, jdensity = oersted.make_helmholtz(str(step_file), mesh_size, jmag=jmag)

pts, spacing = oersted.uniform_3d_grid(xrange, yrange, zrange, (n_grid, n_grid, n_grid))

n_src = mesh.num_elems
n_tgt = pts.shape[0]
n_int = n_src * n_tgt


def verr(y, x):
    return np.linalg.norm(y - x, axis=1) / (np.linalg.norm(x, axis=1) + 1e-6)


def run_test_a_field(settings):
    b_expected = oersted.b_field(
        mesh, pts, jdensity=jdensity, settings=oersted.SolverSettings(method="direct")
    )

    a = oersted.a_field(mesh, pts, jdensity=jdensity, settings=settings)

    ax = a[:, 0].reshape((n_grid, n_grid, n_grid))
    ay = a[:, 1].reshape((n_grid, n_grid, n_grid))
    az = a[:, 2].reshape((n_grid, n_grid, n_grid))

    bx, by, bz = oersted.curl(ax, ay, az, spacing)
    _bx = bx.reshape((n_grid * n_grid * n_grid,))
    _by = by.reshape((n_grid * n_grid * n_grid,))
    _bz = bz.reshape((n_grid * n_grid * n_grid,))
    b = np.column_stack((_bx, _by, _bz))

    err = np.abs(b - b_expected)
    max_err = np.max(err)
    mean_err = np.mean(err)
    print(f"Mean error:    {mean_err}")
    print(f"Max error:     {max_err}")

    # Compute error allowable relative to the average field
    # Allow some variation here due to the first-order approximation of the
    # curl computation
    allowed_err = MAX_ERR * np.mean(np.linalg.norm(b_expected, axis=1))
    print(f"Allowed error: {allowed_err}")
    assert max_err < allowed_err


def test_a_field():
    all_settings = [
        oersted.SolverSettings(method="direct"),
        oersted.SolverSettings(method="octree", theta=theta, batch_size=1),
    ]
    for settings in all_settings:
        print(settings)
        run_test_a_field(settings)
        print("Test passed")


def test_bh_vs_direct():
    theta = 1.0
    batch_size = 1
    n = 10000
    targets = np.zeros((n, 3))
    targets = np.random.rand(n, 3) * 100
    # print(targets)
    # targets = mesh.centroids

    a_direct = oersted.a_field(mesh, targets, jdensity=jdensity)
    a_bh = oersted.a_field(
        mesh,
        targets,
        jdensity=jdensity,
        settings=oersted.SolverSettings(
            method="octree", theta=theta, batch_size=batch_size
        ),
    )

    err = verr(a_bh, a_direct)
    print("a-field")
    print(f"max bh err vs direct: {np.max(err)}")
    print(f"avg bh err vs direct: {np.mean(err)}")

    b_direct = oersted.b_field(mesh, targets, jdensity=jdensity)
    b_bh = oersted.b_field(
        mesh,
        targets,
        jdensity=jdensity,
        settings=oersted.SolverSettings(
            method="octree", theta=theta, batch_size=batch_size
        ),
    )

    err = verr(b_bh, b_direct)
    print("b-field")
    print(f"max bh err vs direct: {np.max(err)}")
    print(f"avg bh err vs direct: {np.mean(err)}")


if __name__ == "__main__":
    test_a_field()
    test_bh_vs_direct()
