"""Python bindings for oersted"""

from .materials import MU0, FreeSpace, LinearMaterial, NonlinearMaterial, BHCurve
from .solver import Solver, DirectSolver, OctreeSolver

from .biotsavart import b_field, h_field, h_mag

from .mesh import Mesh, CentroidMesh, mesh_step, plot_mesh
from .testing import make_helmholtz, smape
from .magnetization import demag_solve
from .results import (
    kelvin_force_density,
    kelvin_forces,
    maxwell_forces,
    lorentz_forces,
    lorentz_force_density,
)


__all__ = [
    "MU0",
    "FreeSpace",
    "LinearMaterial",
    "NonlinearMaterial",
    "BHCurve",
    "Mesh",
    "CentroidMesh",
    "mesh_step",
    "plot_mesh",
    "b_field",
    "h_field",
    "h_mag",
    "bfield_tetrahedrons",
    "bfield_tetrahedrons_direct",
    "hfield_dipole",
    "hfield_dipole_tetrahedrons",
    "Solver",
    "DirectSolver",
    "OctreeSolver",
    "demag_solve",
    "maxwell_forces",
    "kelvin_forces",
    "kelvin_force_density",
    "lorentz_forces",
    "lorentz_force_density",
    "make_helmholtz",
    "smape",
]
