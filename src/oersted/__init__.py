"""Python bindings for oersted"""

from .materials import MU0, FreeSpace, LinearMaterial, NonlinearMaterial, BHCurve
from .solver import Solver, DirectSolver, OctreeSolver

from .biotsavart import (
    bfield_direct,
    bfield_octree,
    bfield_dualtree,
    bfield_hexahedron,
    bfield_tetrahedrons,
    bfield_tetrahedrons_direct,
    hfield_dipole,
    hfield_dipole_tetrahedrons,
)

from .mesh import Mesh, mesh_step
from .magnetization import mag_force

from . import testing


__all__ = [
    "MU0",
    "FreeSpace",
    "LinearMaterial",
    "NonlinearMaterial",
    "BHCurve",
    "bfield_direct",
    "bfield_octree",
    "testing",
    "Mesh",
    "mesh_step",
    "bfield_dualtree",
    "bfield_hexahedron",
    "bfield_tetrahedrons",
    "bfield_tetrahedrons_direct",
    "hfield_dipole",
    "hfield_dipole_tetrahedrons",
    "mag_force",
    "Solver",
    "DirectSolver",
    "OctreeSolver",
]
