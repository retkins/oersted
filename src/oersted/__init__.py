"""Python bindings for oersted"""

from .materials import MU0, FreeSpace, LinearMaterial, NonlinearMaterial, BHCurve
from .solver import SolverSettings

from .biotsavart import a_field, b_field, h_field

from .mesh import Mesh, CentroidMesh, mesh_step, plot_mesh
from .testing import (
    make_helmholtz,
    make_ring,
    verr,
    mean_verr,
    max_verr,
    smape,
    bz_finite_length_solenoid,
    bz_loop_axis,
    dbzdz_loop_axis,
    uniform_3d_grid,
    curl,
)
from .magnetization import demag_solve
from .results import (
    kelvin_force_density,
    kelvin_forces,
    maxwell_forces,
    lorentz_forces,
    lorentz_force_density,
)

from ._oersted import atan2

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
    "a_field",
    "b_field",
    "h_field",
    "demag_solve",
    # Solver settings
    "SolverSettings",
    "OctreeSettings",
    "IterationSettings",
    # Results
    "maxwell_forces",
    "kelvin_forces",
    "kelvin_force_density",
    "lorentz_forces",
    "lorentz_force_density",
    # Testing
    "make_helmholtz",
    "make_ring",
    "verr",
    "mean_verr",
    "max_verr",
    "smape",
    "bz_loop_axis",
    "dbzdz_loop_axis",
    "bz_finite_length_solenoid",
    "uniform_3d_grid",
    "curl",
    # Math
    "atan2",
]
