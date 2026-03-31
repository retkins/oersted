"""Python bindings for oersted"""

from .materials import MU0, FreeSpace, LinearMaterial, NonlinearMaterial, BHCurve
from .solver import Solver, DirectSolver, OctreeSolver

from .biotsavart import (
    b_field,
    bfield_tetrahedrons,
    bfield_tetrahedrons_direct,
    hfield_dipole,
    hfield_dipole_tetrahedrons,
)

from .mesh import Mesh, CentroidMesh, mesh_step
from .magnetization import mag_force

from . import testing


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
    "bfield_tetrahedrons",
    "bfield_tetrahedrons_direct",
    "hfield_dipole",
    "hfield_dipole_tetrahedrons",
    "mag_force",
    "Solver",
    "DirectSolver",
    "OctreeSolver",
]
