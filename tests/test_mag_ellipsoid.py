"""Compute the magnetization of a very flat oblate ellipsoid
per equations 2.21 and 2.22 of the following reference:
<http://www.cmap.polytechnique.fr/~alouges/coursm2/Osborn.pdf>

The problem has the following parameters:
* Uniform background magnetic field of `Bz = 2 T`
* Material with relative permeability `mu_r = 1.5`
* Ellipsoid with semi-axes (a,b,c) = (1.0, 1.0, 0.1)

"""

import oersted
import numpy as np


def test_mag_ellipsoid(min_size: float = 0.15, max_size: float = 0.15):
    # Mesh the part
    nodes, connectivity = oersted.mesh.mesh_step("tests/data/ellipsoid.stp", "tests/data/ellipsoid.msh", min_size, max_size)

    # Parameters
    a: float = 1.0
    # b: float = 1.0
    c: float = 0.1
    m: float = a / c
    demag_tensor = np.array(
        [(np.pi / (4.0 * m)) * (1.0 - 4.0 / (np.pi * m)), (np.pi / (4.0 * m)) * (1.0 - 4.0 / (np.pi * m)), 1.0 - np.pi / (2.0 * m) + 2.0 / (m**2)]
    )
    b_ext: float = 2.0
    h_ext: float = b_ext / oersted.MU0
    mu_r: float = 1.5
    chi: float = mu_r - 1.0

    # Finite element solution
    mat = oersted.materials.LinearMaterial(mu_r)
    h_external = np.zeros((connectivity.shape[0], 3))
    h_external[:, 2] = h_ext
    M, Htotal = oersted.magnetization.demag_tet4(nodes, connectivity, mat, h_external, octree=False)
    Btotal = oersted.MU0 * (Htotal + M)
    Bavg = np.average(Btotal, axis=0)
    print(f"avg B (element): {Bavg}")

    # Analytical solution
    Manalytical = chi * h_ext / (1.0 + chi * demag_tensor[2])
    Hanalytical = h_ext - demag_tensor[2] * Manalytical
    Banalytical = oersted.MU0 * (Hanalytical + Manalytical)
    print(f"Banalytical: {Banalytical:.3f}")

    err = (Banalytical - Bavg[2]) / Banalytical
    print(f"Error: {100 * err:.3f} %")

    # Coarse error tolerance; a better mesh takes too long to run in the standard
    # test suite
    assert np.abs(err) < 4e-2


if __name__ == "__main__":
    test_mag_ellipsoid()
