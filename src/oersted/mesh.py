"""Mesh generation and processing routines"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from numpy import float64, uint32, ascontiguousarray
from ._oersted import (
    mesh_centroids,
    mesh_volumes,
    mesh_surface_faces,
    mesh_surface_face_properties,
)


class CentroidMesh:
    """A finite element mesh represented solely by the centroidal values of the elements

    This is used in the `point source` calculations. It is an approximation, but
    extremely fast and accurate for far field or force calculations.
    """

    # Topology data
    _centroids: NDArray[float64]
    _volumes: NDArray[float64]

    def __init__(self, centroids: NDArray[float64], volumes: NDArray[float64]):
        self._centroids = centroids
        self._volumes = volumes

    @property
    def num_elems(self) -> int:
        """Return the number of elements in the mesh"""
        return self._centroids.shape[0]

    @property
    def centroids(self) -> NDArray[float64]:
        """Return an (N,3) array of the centroid position of each element in the mesh"""
        return self._centroids

    @property
    def volumes(self) -> NDArray[float64]:
        """Return an (N,) array of the volume of each element"""
        return self._volumes


class SurfaceMesh:
    """The surface (triangle) mesh of a 3D volumetric mesh consisting solely of
    4-node tetrahedral elements. This is primarily meant to be used for surface
    force calculations.
    """

    # Basic topology data
    _nodes: NDArray[float64]
    _faces: NDArray[uint32]

    # Information requested by the user
    _centroids: NDArray[float64] | None
    _normals: NDArray[float64] | None
    _areas: NDArray[float64] | None

    def __init__(self, nodes: NDArray[float64], faces: NDArray[uint32]):
        assert nodes.shape[1] == 3
        assert faces.shape[1] == 3
        self._nodes = nodes
        self._faces = faces

        self._centroids = None
        self._normals = None
        self._areas = None

    @property
    def num_faces(self) -> int:
        """Return the number of faces in the surface mesh"""
        return self._faces.shape[0]

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the surface mesh

        Note: this is all of the nodes in the volumetric mesh!
        """
        return self._nodes.shape[0]

    @property
    def nodes(self) -> NDArray[float64]:
        """Return an (N,3) array of the x,y,z nodal coordinates in the surface mesh

        Note: these are all of the nodes in the volumetric mesh!"""
        return self._nodes

    @property
    def faces(self) -> NDArray[uint32]:
        """Return an (N,3) array of the face connectivity"""
        return self._faces

    def _properties(self):
        # Compute the properties of the surface mesh
        self._areas, self._centroids, self._normals = mesh_surface_face_properties(
            ascontiguousarray(self.nodes), ascontiguousarray(self.faces)
        )

    @property
    def areas(self):
        """Return an (N,) array of the area of each surface face"""
        if self._areas is None:
            self._properties()
        return self._areas

    @property
    def centroids(self):
        """Return an (N,3) array of the x,y,z coordinates of each face centroid"""
        if self._centroids is None:
            self._properties()
        return self._centroids

    @property
    def normals(self):
        """Return an (N,3) array of the unit normal vectors on each face"""
        if self._normals is None:
            self._properties()

        return self._normals


class Mesh:
    """A continuous finite element mesh made of tet4 elements"""

    # Basic mesh topology data
    _nodes: NDArray[float64]
    _connectivity: NDArray[uint32]

    # Information that is compute on-demand for the user
    _edges: NDArray[uint32] | None
    _faces: NDArray[uint32] | None
    _centroids: NDArray[float64] | None
    _volumes: NDArray[float64] | None

    # Surface mesh data
    _surface: SurfaceMesh | None

    def __init__(self, nodes: NDArray[float64], connectivity: NDArray[uint32]):
        assert len(nodes.shape) == 2
        assert nodes.shape[1] == 3
        assert len(connectivity.shape) == 2
        assert connectivity.shape[1] == 4
        self._nodes = nodes
        self._connectivity = connectivity

        self._edges = None
        self._faces = None
        self._centroids = None
        self._volumes = None
        self._surface = None

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

    def to_centroid_mesh(self) -> CentroidMesh:
        """Create a centroid mesh from a tet4 mesh"""
        return CentroidMesh(self.centroids, self.volumes)

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
        raise NotImplementedError
        # return self._edges

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
            self._centroids = mesh_centroids(
                np.ascontiguousarray(self.nodes),
                np.ascontiguousarray(self.connectivity),
            )

        return self._centroids

    @property
    def volumes(self) -> NDArray[float64]:
        """Return an (N,) array of the volume of each element in the mesh"""
        if self._volumes is None:
            self._volumes = mesh_volumes(
                np.ascontiguousarray(self.nodes),
                np.ascontiguousarray(self.connectivity),
            )
        return self._volumes

    @property
    def surface(self) -> SurfaceMesh:
        """Return the surface mesh associated with the volumetric mesh"""
        if self._surface is None:
            faces = mesh_surface_faces(ascontiguousarray(self.connectivity))
            self._surface = SurfaceMesh(self.nodes.copy(), faces)

        return self._surface

    def plot(self, filename: str | None = None):
        """Convenience function for plotting just the mesh itself"""
        plot_mesh(self, filename)

    @classmethod
    def from_step(
        cls,
        filename: str,
        mesh_size: float,
        mesh_size_scale: float = 1e3,
        part_size_scale: float = 1e-3,
    ) -> Mesh:
        """Create a Mesh from a step file

        !!! note
            `oersted` needs mesh dimensions in meters, but gmsh typically works in mm.
            This function scales the input mesh size *up* to be in mm, and scales the
            gmsh resultant *down* to be in m. Adjust these parameters if the mesh isn't
            working properly.

        Args:
            filename: STEP file to mesh
            mesh_size: (m) nominal element size to use for the mesh
            mesh_size_scale: (mm/m) adjust if the model units are in mm and not m
            part_size_scale: (m/mm) adjust if the mesh units are in mm and not m

        Returns:
            volumetric tet4 mesh of the STEP file
        """
        return mesh_step(
            filename,
            mesh_size * mesh_size_scale,
            mesh_size * mesh_size_scale,
            part_size_scale,
        )

    def append(self, mesh: Mesh) -> Mesh:
        """Convenience function for appending two meshes together."""

        nodes = np.vstack((self.nodes, mesh.nodes))
        connectivity = np.vstack(
            (self.connectivity, mesh.connectivity + uint32(self.num_nodes))
        )
        return Mesh(nodes, connectivity)


def plot_mesh(
    mesh: Mesh,
    filename: str | None = None,
    scalars: NDArray[float64] | None = None,
    centroids: NDArray[float64] | None = None,
    vectors: NDArray[float64] | None = None,
    vector_scale: float | None = None,
    transparency: bool = False,
):
    """Make a 3D plot of the mesh

    !!! note
        This function requires `pyvista`: `pip install pyvista`.

    Args:
        mesh: the tet4 mesh to plot
        filename: if this argument is passed, save to file only
        scalars: (Ne,) array of scalar values to color the mesh, defined at element
            centroids
        centroids: (N,3) array of vector positions for plotting vector values on the
            mesh
        vectors: (N,3) array of vector magnitudes for plotting vector values on the
            mesh; must be same length as `centroids`
        vector_scale: adjust for setting vector length
        transparency: set to `True` to plot a wireframe of the mesh
    """

    try:
        import pyvista as pv

        cells = np.hstack(
            [np.full((mesh.num_elems, 1), 4, dtype=np.int64), mesh.connectivity]
        )
        celltypes = np.full(mesh.num_elems, pv.CellType.TETRA)
        pv_mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, mesh.nodes)

        pl = pv.Plotter(off_screen=filename is not None)
        if transparency:
            pl.add_mesh(pv_mesh, style="wireframe", color="black", line_width=0.5)
        else:
            pl.add_mesh(
                pv_mesh,
                scalars=scalars,
                show_edges=True,
                line_width=0.5,
                scalar_bar_args={"title": "magnitude", "vertical": True},
            )

        factor = 1.0
        if vector_scale is not None:
            factor = vector_scale

        if centroids is not None and vectors is not None:
            assert centroids.shape == vectors.shape
            arrow_mesh = pv.PolyData(centroids)
            arrow_mesh["vectors"] = vectors
            arrow_mesh["magnitude"] = np.linalg.norm(vectors, axis=1)
            arrows = arrow_mesh.glyph(orient="vectors", scale=False, factor=factor)
            pl.add_mesh(
                arrows, scalars="magnitude", cmap="viridis", show_scalar_bar=False
            )

        if filename is not None:
            pl.save_graphic(filename)
        else:
            pl.show()

    except ImportError:
        print("`pyvista` is not installed.")
        print("Please install before continuing: `pip install pyvista")


def mesh_step(
    infile: str, max_size: float, min_size: float = 0.0, scale: float = 1e-3
) -> Mesh:
    """Mesh a step file using gmsh

    !!! note
        This requires `gmsh` to be installed: `pip install gmsh`

    Args:
        infile: path to the STEP file to mesh
        max_size: (m) maximum allowable element size
        min_size: (m) minimum allowable element size
        scale: (mm/m) adjust if the part or mesh is scaled incorrectly

    Returns:
        a tet4 (volumetric) mesh of the component
    """

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
        connectivity = np.array(
            [[tag_to_compact[tag] for tag in elem] for elem in raw_connectivity],
            dtype=np.uint32,
        )
        gmsh.finalize()

        mesh_out: Mesh = Mesh(nodes, connectivity)
        print(f"Nodes: {mesh_out.num_nodes}, Elems: {mesh_out.num_elems}\n")

        return mesh_out

    except ImportError:
        raise RuntimeError(
            f"Error - gmsh is not installed. Could not mesh file `{infile}`"
        ) from None
