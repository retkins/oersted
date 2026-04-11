from oersted.testing import make_helmholtz

import oersted
from oersted import OctreeSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def main(
    nthetas: int = 1,
    size: float = 0.015,
    theta_min: float = 0.5,
    theta_max: float = 0.5,
):
    theta_vals = np.linspace(theta_min, theta_max, nthetas)
    errs = np.zeros(nthetas)
    mesh, jdensity = make_helmholtz("tests/data/ring.stp", size)
    bdirect = oersted.b_field(mesh, jdensity, mesh.centroids)
    for i, theta in enumerate(theta_vals):
        boctree = oersted.b_field(
            mesh, jdensity, mesh.centroids, solver=OctreeSolver(theta=theta)
        )
        bmag_direct = np.linalg.norm(bdirect, axis=1)
        bmag_octree = np.linalg.norm(boctree, axis=1)
        # errs[i] = oersted.mean_relative_error(bmag_direct, bmag_octree)
        errs[i] = oersted.testing.smape(bmag_direct, bmag_octree)

    print(errs)
    fig, ax = plt.subplots()
    ax.plot(theta_vals, errs)
    ax.set_xlabel("Theta (B-H Acceptance Criteria)")
    ax.set_ylabel("B-Field Error Over Mesh")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.set_title(
        "Oersted Benchmarks: Helmholtz Coil Problem\nAcceptance Criteria vs. Error"
    )
    fig.tight_layout()
    fig.savefig("tests/fig/benchmarks_error.svg")


if __name__ == "__main__":
    nthetas = 10
    size = 5.0
    theta_min: float = 0.01
    theta_max: float = 0.5
    main(nthetas=10, size=5.0, theta_min=0.01, theta_max=0.5)
