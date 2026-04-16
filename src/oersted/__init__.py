"""Python bindings for oersted"""

from .materials import MU0, FreeSpace, LinearMaterial, NonlinearMaterial, BHCurve
from .solver import Solver, DirectSolver, OctreeSolver

from .biotsavart import b_field, h_field, h_mag

from .mesh import Mesh, CentroidMesh, mesh_step, plot_mesh
from .testing import (
    make_helmholtz,
    smape,
    bz_finite_length_solenoid,
    bz_loop_axis,
    dbzdz_loop_axis,
)
from .magnetization import demag_solve
from .results import (
    kelvin_force_density,
    kelvin_forces,
    maxwell_forces,
    lorentz_forces,
    lorentz_force_density,
)


__all__ = [
    # Constants and material properties
    "MU0",
    "FreeSpace",
    "LinearMaterial",
    "NonlinearMaterial",
    "BHCurve",
    # Meshing
    "Mesh",
    "CentroidMesh",
    "mesh_step",
    "plot_mesh",
    # Fields calculations
    "b_field",
    "h_field",
    "h_mag",
    "demag_solve",
    # Solver settings
    "Solver",
    "DirectSolver",
    "OctreeSolver",
    # Results
    "maxwell_forces",
    "kelvin_forces",
    "kelvin_force_density",
    "lorentz_forces",
    "lorentz_force_density",
    # Testing
    "make_helmholtz",
    "smape",
    "bz_loop_axis",
    "dbzdz_loop_axis",
    "bz_finite_length_solenoid",
]
