"""Python bindings for oersted"""

from .materials import MU0, FreeSpace, LinearMaterial, NonlinearMaterial, BHCurve
from .solver import Solver, DirectSolver, OctreeSolver

from .biotsavart import b_field, h_field, h_mag

from .mesh import Mesh, CentroidMesh, mesh_step, plot_mesh, surface_forces
from . import testing
from . import magnetization


__all__ = [
    "MU0",
    "FreeSpace",
    "LinearMaterial",
    "NonlinearMaterial",
    "BHCurve",
    "testing",
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
    "magnetization",
    "surface_forces",
]
