import numpy as np
import oersted
from oersted import Mesh, DirectSolver, OctreeSolver2Zone
import time

# Test parameters
infile: str = "tests/data/sphere.stp"
mesh_size: float = 15e-3  # (m)
b_ext_mag: float = 1.0  # (T)
mu_r: float = 1.5
solver = oersted.DirectSolver()
solvers = [oersted.DirectSolver(), oersted.OctreeSolver2Zone()]


def check_mag_sphere(mesh: Mesh, solver: DirectSolver | OctreeSolver2Zone):

    # Create material properties and calculate uniform background field
    mat = oersted.materials.LinearMaterial(mu_r)
    h_external = np.zeros((mesh.num_elems, 3))
    h_ext_mag: float = b_ext_mag / oersted.MU0
    h_external[:, 2] = h_ext_mag

    # Compute demag parameters: magnetization and internal H field
    start = time.perf_counter()
    M, Htotal = oersted.demag_solve(mesh, mat, h_external, solver)
    elapsed = time.perf_counter() - start

    # Postprocessing
    h_z_mean = np.average(Htotal[:, 2])
    Mnorm = np.linalg.norm(M, axis=1)
    Hnorm = np.linalg.norm(Htotal, axis=1)
    Btotal = oersted.MU0 * (M + Htotal)
    Mavg = np.linalg.norm(np.average(M, axis=0))
    Bavg = np.linalg.norm(np.average(Btotal, axis=0))

    # Analytical solution
    N: float = 1.0 / 3.0  # demag factor for sphere
    H_ext: float = b_ext_mag / oersted.MU0
    M_analytical = 3 * (mu_r - 1) / (mu_r + 2) * H_ext
    H_analytical: float = H_ext - N * M_analytical
    B_analytical = oersted.MU0 * (H_ext + (1.0 - N) * M_analytical)

    # Compute errors
    M_error = (M_analytical - Mavg) / M_analytical
    B_error = (B_analytical - Bavg) / B_analytical

    print("\n\nDemag tet test - magnetized sphere\n---\n")
    print(f"{mesh.num_elems} elements")
    print(f"Background field: {b_ext_mag:.3f} T")
    print(f"Relative permeability: {mu_r:.3f}")
    print("")
    print("Analytical solution:")
    print(f"\tM = {M_analytical:.3e} A/m")
    print(f"\tB = {B_analytical:.3f} T")
    print(f"Total solver time: {elapsed:.3f} sec")
    print(
        f"Avg/min/max M: \
            {np.average(Mnorm)} / {np.min(Mnorm):.3e} / {np.max(Mnorm):.3e}"
    )
    print(
        f"Avg/min/max H: \
            {np.average(Hnorm)} / {np.min(Hnorm):.3e} / {np.max(Hnorm):.3e}"
    )
    print(f"Bavg = {Bavg:.3f} T")
    print(f"M error: {100 * M_error:.2f}%")
    print(f"B error: {100 * B_error:.2f}%")

    err = float(np.abs(h_z_mean - H_analytical) / H_analytical)
    print(f"Error = {err * 100:.3f} %")
    assert err < 0.01
    assert M_error < 0.01
    assert B_error < 0.01


def test_magnetized_sphere():

    for solver in solvers:
        mesh: Mesh = oersted.Mesh.from_step(infile, mesh_size)
        check_mag_sphere(mesh, solver)


if __name__ == "__main__":
    test_magnetized_sphere()
    print("Test passed!")
