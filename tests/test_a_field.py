"""Compare a-field computations against b' = curl(a)

Use a helmholtz coil to generate fields and evaluate b vs b' near the center, away from
the source elements
"""

# TODO: enable testing for all source kernels

import numpy as np
import oersted
import pathlib

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


def test_a_field():
    b_expected = oersted.b_field(mesh, jdensity, pts)

    a = oersted.a_field(mesh, jdensity, pts)

    ax = a[:, 0].reshape((n_grid, n_grid, n_grid))
    ay = a[:, 1].reshape((n_grid, n_grid, n_grid))
    az = a[:, 2].reshape((n_grid, n_grid, n_grid))

    bx, by, bz = oersted.curl(ax, ay, az, spacing)
    _bx = bx.reshape((n_grid * n_grid * n_grid,))
    _by = by.reshape((n_grid * n_grid * n_grid,))
    _bz = bz.reshape((n_grid * n_grid * n_grid,))
    b = np.column_stack((_bx, _by, _bz))

    max_err = np.max(np.abs(b - b_expected))

    # Compute error allowable relative to the average field
    # Allow some variation here due to the first-order approximation of the
    # curl computation
    assert max_err < 1e-4 * np.mean(np.linalg.norm(b_expected, axis=1))
