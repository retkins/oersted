"""Python bindings for oersted"""

from .materials import MU0, FreeSpace, LinearMaterial, NonlinearMaterial, BHCurve
from .solver import Solver, DirectSolver, OctreeSolver

from .biotsavart import (
    b_field,
    h_field,
    hfield_dipole,
    hfield_dipole_tetrahedrons,
)

from .mesh import Mesh, CentroidMesh, mesh_step
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
    "b_field",
    "h_field",
    "bfield_tetrahedrons",
    "bfield_tetrahedrons_direct",
    "hfield_dipole",
    "hfield_dipole_tetrahedrons",
    "Solver",
    "DirectSolver",
    "OctreeSolver",
    "magnetization"
]
