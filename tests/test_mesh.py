"""Test mesh-related functionality (Python and Rust)"""

import oersted
import numpy as np
import gmsh

# Mesh a sphere in gmsh, which is used by subsequent tests
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)  # suppress output

# Create geometry in gmsh this time
gmsh.model.occ.addSphere(0, 0, 0, 1.0)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25)
gmsh.model.mesh.generate(3)

# Get surface triangles from gmsh; element type 2 = 3-node triangle
surface_element_types, surface_element_tags, surface_node_tags = gmsh.model.mesh.getElements(dim=2)
gmsh_surface_faces = np.array(surface_node_tags[0], dtype=np.uint32).reshape(-1, 3)

# Volume mesh for oersted
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
nodes = np.array(node_coords).reshape(-1, 3)
tet_types, tet_tags, tet_node_tags = gmsh.model.mesh.getElements(dim=3)
connectivity = np.array(tet_node_tags[0], dtype=np.uint32).reshape(-1, 4)
# gmsh node tags are 1-indexed, convert to 0-indexed
connectivity -= 1
gmsh_surface_faces -= 1

gmsh.finalize()

mesh = oersted.Mesh(nodes, connectivity)


def test_volume():
    """Test the volume calculation on a meshed sphere

    Sphere has radius = 50mm
    Volume of a sphere is: 4/3 pi r^3

    TODO: resolve discrepency with user-defined and gmsh units
    """

    diff_allowable: float = 1e-4
    mesh_size: float = 5  # mm, gmsh units
    radius = 0.050  # m, user units
    volume_expected: float = (4.0 / 3.0) * np.pi * radius**3
    mesh = oersted.mesh_step("tests/data/sphere.stp", mesh_size, mesh_size)
    volume_mesh: float = float(np.sum(mesh.volumes))
    assert np.abs(volume_mesh - volume_expected) < diff_allowable


def test_surface_faces():
    """Test the identification of surface elements/faces on a meshed sphere"""

    # Check that the number of surface faces is the same between gmsh and oersted
    assert gmsh_surface_faces.shape == mesh.surface_faces.shape


def test_face_normals():
    """Compute face normals using gmsh and check that they match the oersted calculation"""

    # Compute normals from gmsh mesh
    normals_gmsh = np.zeros(gmsh_surface_faces.shape)
    for i, face in enumerate(gmsh_surface_faces):
        n0 = nodes[face[0]]
        n1 = nodes[face[1]]
        n2 = nodes[face[2]]
        e0 = n1 - n0
        e1 = n2 - n0
        n = np.cross(e0, e1)
        normals_gmsh[i] = n / np.linalg.norm(n)

    # Build lookup from sorted face nodes to gmsh normal
    gmsh_lookup = {}
    for i, face in enumerate(gmsh_surface_faces):
        key = tuple(sorted(face))
        gmsh_lookup[key] = normals_gmsh[i]

    # Compare
    for i, face in enumerate(mesh.surface_faces):
        key = tuple(sorted(face))
        oersted_normal = mesh.surface_face_normals[i]
        gmsh_normal = gmsh_lookup[key]
        dot = np.dot(oersted_normal, gmsh_normal)
        if dot < 0:
            raise AssertionError("gmsh and oersted do not compute the same surface normals")


def test_surface_face_areas_and_centroids():
    """Test that gmsh and oersted compute the same area and centroid for all surface faces

    These are tested together because the data is very similar
    """

    gmsh_areas = np.zeros((gmsh_surface_faces.shape[0],))
    gmsh_centroids = np.zeros((gmsh_surface_faces).shape)

    for i, face in enumerate(gmsh_surface_faces):
        n0 = nodes[face[0]]
        n1 = nodes[face[1]]
        n2 = nodes[face[2]]
        gmsh_centroids[i] = (n0 + n1 + n2) / 3.0
        gmsh_areas[i] = 0.5 * np.linalg.norm(np.cross(n1 - n0, n2 - n0))

    oersted_lookup = {}
    for i, face in enumerate(mesh.surface_faces):
        key = tuple(sorted(face))
        oersted_lookup[key] = (mesh.surface_face_areas[i], mesh.surface_face_centroids[i])

    for i, face in enumerate(gmsh_surface_faces):
        key = tuple(sorted(face))
        gmsh_centroid = gmsh_centroids[i]
        gmsh_area = gmsh_areas[i]
        oersted_area, oersted_centroid = oersted_lookup[key]

        if not np.allclose(oersted_centroid, gmsh_centroid):
            raise AssertionError(f"Centroid mismatch: {oersted_centroid} vs {gmsh_centroid}")
        if not np.isclose(oersted_area, gmsh_area):
            raise AssertionError(f"Area mismatch: {oersted_area} vs {gmsh_area}")


def main():
    test_volume()
    test_surface_faces()
    test_face_normals()
    test_surface_face_areas_and_centroids()


if __name__ == "__main__":
    main()
    print("All tests passed!")
