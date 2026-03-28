"""Mesh generation and processing routines"""

import oersted

import numpy as np
from numpy.typing import NDArray
from numpy import float64, uint32
from ._oersted import (
    _mesh_centroids,
    _mesh_volumes,
    _mesh_surface_faces,
    # _mesh_surface_face_normals,
    # _mesh_surface_face_areas,
    # _maxwell_stress_tensor,
)

from .biotsavart import Solver


class CentroidMesh:
    """A finite element mesh represented solely by the centroidal values of the elements

    This is used in the `point source` calculations. It is an approximation, but extremely
    fast and accurate for far field or force calculations.
    """

    # Topology data
    _centroids: NDArray[float64]
    _volumes: NDArray[float64]

    # Source and result data
    _j_density: NDArray[float64] | None
    _m_field: NDArray[float64] | None
    _h_field: NDArray[float64] | None

    def __init__(self, centroids: NDArray[float64], volumes: NDArray[float64], j_density: NDArray[float64] | None = None):
        self._centroids = centroids
        self._volumes = volumes
        self._j_density = j_density

    @property
    def num_elems(self):
        return self._centroids.shape[0]

    @property
    def centroids(self):
        return self._centroids

    @property
    def volumes(self):
        return self._volumes

    @property
    def j_density(self):
        if self._j_density is None:
            self._j_density = np.zeros((self.num_elems, 3))

        return self._j_density


class Mesh:
    """A continuous finite element mesh made of tet4 elements"""

    # Basic mesh topology data
    _nodes: NDArray[float64]
    _connectivity: NDArray[uint32]
    _edges: NDArray[uint32] | None
    _faces: NDArray[uint32] | None
    _centroids: NDArray[float64] | None
    _volumes: NDArray[float64] | None
    _face_centroids: NDArray[float64] | None
    _surface_faces: NDArray[uint32] | None
    _surface_face_centroids: NDArray[float64] | None
    _surface_face_normals: NDArray[float64] | None
    _surface_face_areas: NDArray[float64] | None

    # Results information is stored at the element centroids
    _j_density: NDArray[float64] | None
    _h_field: NDArray[float64] | None
    _m_field: NDArray[float64] | None

    def __init__(self, nodes: NDArray[float64], connectivity: NDArray[uint32], j_density: NDArray[float64] | None = None):
        self._nodes = nodes
        self._connectivity = connectivity
        self._j_density = j_density
        self._edges = None
        self._faces = None
        self._centroids = None
        self._volumes = None
        self._face_centroids = None
        self._surface_faces = None

    @property
    def nodes(self) -> NDArray[float64]:
        """Returns an (N,3) array of nodal coordinates in the mesh"""
        return self._nodes

    @property
    def connectivity(self) -> NDArray[uint32]:
        """Returns an (N,4) array of the node numbers associated with each element

        Node numbers are indices into the self._nodes array
        """
        return self._connectivity

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the model"""
        return self._nodes.shape[0]

    @property
    def num_elems(self) -> int:
        """Returns the number of elements in the model"""
        return self._connectivity.shape[0]

    @property
    def edges(self):  # -> NDArray[uint32]:
        """Returns an (N,2) array of edges in the model

        Each value in the array is a node number associated with that edge.
        The first node is the start node, the second is the end node. This
        provides directionality for the edge.
        """
        if self._edges is None:
            # self._edges = _mesh_edges(self.nodes, self.connectivity)
            pass
        return self._edges

    @property
    def faces(self):
        """Returns an (N,3) array of nodes associated with each element face
        in the model

        Nodes are ordered such that the right hand rule forms the face normal.
        """
        if self._faces is None:
            # self._faces = mesh_faces(self.nodes, self.connectivity)
            pass
        return self._faces

    @property
    def centroids(self) -> NDArray[float64]:
        """Returns an (N,3) array of all element centroids in the mesh"""
        if self._centroids is None:
            cx = np.zeros((self.num_elems,))
            cy = np.zeros((self.num_elems,))
            cz = np.zeros((self.num_elems,))
            _mesh_centroids(
                np.ascontiguousarray(self.nodes.flatten()),
                np.ascontiguousarray(self.connectivity.flatten()),
                np.ascontiguousarray(cx[:]),
                np.ascontiguousarray(cy[:]),
                np.ascontiguousarray(cz[:]),
            )
            self._centroids = np.hstack((cx[:, np.newaxis], cy[:, np.newaxis], cz[:, np.newaxis]))
        return self._centroids

    @property
    def volumes(self) -> NDArray[float64]:
        """Return an (N,) array of the volume of each element in the mesh"""
        if self._volumes is None:
            self._volumes: NDArray[float64] = np.zeros((self.num_elems,))
            _mesh_volumes(
                np.ascontiguousarray(self.nodes.flatten()), np.ascontiguousarray(self.connectivity.flatten()), np.ascontiguousarray(self._volumes)
            )
        return self._volumes

    @property
    def face_centroids(self):
        if self._face_centroids is None:
            # self._face_centroids = mesh_face_centroids(self.nodes, self.connectivity)
            pass
        return self._face_centroids

    @property
    def surface_faces(self):
        if self._surface_faces is None:
            self._surface_faces = _mesh_surface_faces(self.connectivity)
            pass
        return self._surface_faces

    @property
    def surface_face_centroids(self):
        if self._surface_face_centroids is None:
            # self._surface_face_centroids = mesh_surface_face_centroids(self.nodes, self.surface_faces)
            pass
        return self._surface_face_centroids

    @property
    def surface_face_normals(self):
        """Returns an (N,3) array of the normal vectors associated with each surface face in the model"""
        if self._surface_face_normals is None:
            # self._surface_face_normals = _mesh_surface_face_normals(self.nodes, self.surface_faces)
            pass

        return self._surface_face_normals

    @property
    def surface_face_areas(self):
        """Returns an (N,) array of the area of each surface face"""
        if self._surface_face_areas is None:
            # self._surface_face_areas = _mesh_surface_face_areas(self.nodes, self.surface_faces)
            pass
        return self._surface_face_areas

    @property
    def h_field(self) -> NDArray[float64]:
        """Return an (N,3) array of the magnetic field strength vector at each element centroid"""
        if self._h_field is None:
            raise Exception("Error - h field has not been calculated for this mesh.")

        return self._h_field

    @property
    def m_field(self) -> NDArray[float64]:
        """Return an (N,3) array of the magnetization vector at each element centroid"""
        if self._m_field is None:
            self._m_field = np.zeros((self.num_elems, 3))

        return self._m_field

    @property
    def b_field(self) -> NDArray[float64]:
        """Return an (N,3) array of the magnetic flux density at each element centroid"""
        if self._h_field is None and self._m_field is None:
            raise Exception("Error - results have not been calculated for this mesh.")

        return oersted.MU0 * (self.m_field + self.h_field)

    @property
    def j_density(self) -> NDArray[float64]:
        """Returns an (N,3) array of the current density at each element centroid

        This function will return an empty array if current densities have not been
        provided at object initiation.
        """
        if self._j_density is None:
            return np.zeros((self.num_elems, 3))

        else:
            return self._j_density

    def surface_forces(self, solver: Solver):  # -> NDArray[float64]:
        """Compute the maxwell stress tensor and determine the force vector acting on each
        surface face centroid. Returns an (N,3) array of the force vector
        """

        # return _maxwell_stress_tensor(self.surface_face_centroids, self.surface_face_normals, self.surface_face_areas, self.j_density, self.m_field)
        pass


def plot_mesh(x, y, z):
    """Make a scatter plot of element centroids"""

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x, y, z)
        plt.show()
    except ImportError:
        print("Error - matplotlib is not installed. Could not plot mesh.")


def mesh_step(infile: str, outfile: str, min_size: float, max_size: float, scale=1e-3) -> Mesh:
    """Mesh a step file using gmsh"""

    mshfile = infile.split(".")[0] + ".msh"
    nodes: NDArray[float64]
    connectivity: NDArray[uint32]

    try:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # suppress output
        gmsh.model.occ.importShapes(infile)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
        gmsh.model.mesh.generate(3)  # mesh 3d elements
        gmsh.write(mshfile)

        print(f"Wrote gmsh mesh to `{mshfile}")

        # Get all nodes: returns (tags, coords, parametricCoords)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()

        # coords is flat [x0,y0,z0,x1,y1,z1,...], reshape to (Nn, 3)
        nodes = np.array(coords).reshape(-1, 3) * scale

        # Build compact renumbering: gmsh tags can be sparse/non-sequential
        tag_to_compact = {tag: i for i, tag in enumerate(node_tags)}

        # Get tet elements (type 4 = 4-node tetrahedra)
        elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(4)

        # elem_node_tags is flat [n0,n1,n2,n3, n0,n1,n2,n3, ...], reshape to (Ne, 4)
        raw_connectivity = np.array(elem_node_tags).reshape(-1, 4)

        # Renumber to compact 0-based indices
        connectivity = np.array([[tag_to_compact[tag] for tag in elem] for elem in raw_connectivity], dtype=np.uint32)
        gmsh.finalize()

        return Mesh(nodes, connectivity)

    except ImportError:
        raise RuntimeError(f"Error - gmsh is not installed. Could not mesh file `{infile}`") from None


def mesh_step_tets(
    step_file: str, min_size: float, max_size: float, scale: float = 1e-3
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Mesh a step file with gmsh and return tet element data. This is meant to
    be used with the tet element source functionality.

    Args
    ---
    step_file: Path to STEP file
    min_size, `max_size`: Mesh element size bounds (in model units, usually mm)

    Returns
    ---
    (`nodes`, `centroids`, `volume`)
    nodes: N*12-length flat array of nodal coordinates for each tet, row-major:
        [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3, ...]
    centroids: Nx3 array of the centroids of each element
    volume: N-length array of volume of each element
    """

    import gmsh

    # Setup gmsh and generate elements
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # suppress output
    gmsh.model.add("model")
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
    gmsh.model.mesh.generate(3)

    # Get all node coordinates: node_tags is 1-indexed
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # node_coords is flat [x1,y1,z1, x2,y2,z2, ...]
    # Build a lookup from tag -> coordinates
    all_coords = node_coords.reshape(-1, 3)
    # node_tags might not be contiguous, so use a dict
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

    # Get tet elements (type 4 = linear tet with 4 nodes)
    tet_type = 4
    tet_tags, tet_node_tags = gmsh.model.mesh.getElementsByType(tet_type)

    n_tets = len(tet_tags)
    tet_connectivity = tet_node_tags.reshape(n_tets, 4)  # each row: 4 node tags

    # Build the 12*N flat node coordinate array
    nodes = np.zeros((n_tets, 4, 3))
    for i in range(n_tets):
        for j in range(4):
            idx = tag_to_idx[int(tet_connectivity[i, j])]
            nodes[i, j, :] = all_coords[idx]

    nodes *= scale

    # Centroids: average of 4 nodes (TODO: should this be done differently?)
    centroids = nodes.mean(axis=1)

    # Volumes: V = |det([v1-v0, v2-v0, v3-v0])| / 6 like below, but now vectorized
    v0 = nodes[:, 0, :]
    v1 = nodes[:, 1, :]
    v2 = nodes[:, 2, :]
    v3 = nodes[:, 3, :]

    d1 = v1 - v0
    d2 = v2 - v0
    d3 = v3 - v0

    cross = np.cross(d1, d2)
    det = np.sum(cross * d3, axis=1)
    volumes = np.abs(det) / 6.0

    # Flatten nodes to row-major 12*N
    nodes_flat = nodes.reshape(-1)  # [x0,y0,z0,x1,y1,z1,...] per tet

    gmsh.finalize()

    return nodes_flat, centroids, volumes


def tet_volume(p0, p1, p2, p3):
    """Calculate volume of a tetrahedron given 4 vertex coordinates:
    volume = |det(p1-p0, p2-p0, p3-p0)| / 6

    Each coordinate is a 3-length numpy array
    TODO: turn this into numpy operations for efficiency
    """

    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    return abs(np.dot(v1, np.cross(v2, v3))) / 6.0


def process_elements(infile: str, outfile: str, scale: float = 1e-3):
    """Convert gmsh .msh file into format readable by `oersted`
    and calculate the volume of each element
    """

    try:
        import gmsh

        gmsh.initialize()
        gmsh.open(infile)
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(3)

        with open(outfile, "w") as f:
            f.write("x,y,z,volume\n")

            for elem_type, elem_tags, elem_nodes in zip(element_types, element_tags, node_tags, strict=True):
                elem_name, dim, order, num_nodes, local_coords, num_primary_nodes = gmsh.model.mesh.getElementProperties(elem_type)

                # Reshape node tags to have one row per element
                elem_nodes = elem_nodes.reshape(-1, num_nodes)

                print(f"Processing {len(elem_tags)} {elem_name} elements...")

                # Process each element
                for nodes in elem_nodes:
                    # Get coordinates of all nodes in this element
                    coords = []
                    for node in nodes:
                        coord = gmsh.model.mesh.getNode(node)[0]
                        coords.append(coord)
                    coords = np.array(coords) * scale

                    # Calculate centroid (average of node coordinates)
                    # TODO: is this the right way to calculate centroid?
                    centroid = np.mean(coords, axis=0)

                    # Calculate volume based on element type
                    if "Tetrahedron" in elem_name:
                        volume = tet_volume(coords[0], coords[1], coords[2], coords[3])

                    else:
                        print(f"Warning: Unknown element type {elem_name}, approximating volume")
                        volume = 0.0

                    f.write(f"{centroid[0]:.6f},{centroid[1]:.6f},{centroid[2]:.6f},{volume:.10e}\n")

        print(f"Element data written to: {outfile}")
        gmsh.finalize()

    except ImportError:
        print(f"Error - gmsh is not installed. Could not process elements in file `{infile}`")
