//! Finite element mesh operations

use crate::vec3::Vec3;
use std::collections::HashMap;

/// Return the coordinates of a node with number `idx`
///
/// `nodes` is a flat array representing an Nx3 table of nodal coordinates
#[inline]
pub fn node_coords(nodes: &[f64], idx: usize) -> Vec3 {
    Vec3([nodes[3 * idx], nodes[3 * idx + 1], nodes[3 * idx + 2]])
}

/// Compute the centroids of the elements in a mesh containing
/// 4-node linear tetrahedral elements by taking the mean position
/// of all 4 nodes.
pub fn centroids(nodes: &[f64], connectivity: &[u32], x: &mut [f64], y: &mut [f64], z: &mut [f64]) {
    let n_elements: usize = connectivity.len() / 4;
    assert_eq!(n_elements, x.len());
    assert_eq!(n_elements, y.len());
    assert_eq!(n_elements, z.len());

    for (i, elem) in connectivity.chunks_exact(4).enumerate() {
        // Node numbers
        let [n0, n1, n2, n3] = [
            elem[0] as usize,
            elem[1] as usize,
            elem[2] as usize,
            elem[3] as usize,
        ];

        let centroid = (node_coords(nodes, n0)
            + node_coords(nodes, n1)
            + node_coords(nodes, n2)
            + node_coords(nodes, n3))
            * 0.25;
        x[i] = centroid[0];
        y[i] = centroid[1];
        z[i] = centroid[2];
    }
}

/// Compute the volumes of the elements in a mesh containing
/// 4-node linear tetrahedral elements:
///
/// volume = (1/6) * |a x (b - c))|
///
/// # Reference:
/// <https://en.wikipedia.org/wiki/Tetrahedron#Other_approaches>
pub fn volumes(nodes: &[f64], connectivity: &[u32], vol: &mut [f64]) {
    let n_elements: usize = connectivity.len() / 4;
    assert_eq!(n_elements, vol.len());

    for (i, elem) in connectivity.chunks_exact(4).enumerate() {
        // Node numbers
        let [n0, n1, n2, n3] = [
            elem[0] as usize,
            elem[1] as usize,
            elem[2] as usize,
            elem[3] as usize,
        ];

        // Vertices
        let v0: Vec3 = node_coords(nodes, n0);
        let v1: Vec3 = node_coords(nodes, n1);
        let v2: Vec3 = node_coords(nodes, n2);
        let v3: Vec3 = node_coords(nodes, n3);

        // Edge vectors
        let a = v1 - v0;
        let b = v2 - v0;
        let c = v3 - v0;

        vol[i] = (1.0 / 6.0) * (a.dot(&b.cross(&c))).abs();
    }
}

/// Determine which faces in a mesh are on the surface
///
/// Returns a flat vector of the indices for each node in the surface faces
pub fn surface_faces(connectivity: &[u32]) -> Vec<u32> {
    let n_elements: usize = connectivity.len() / 4;

    // Waste a bit of memory by allocating more memory than we need...
    let mut faces_out: Vec<u32> = Vec::with_capacity(n_elements);
    let mut face_map: HashMap<[u32; 3], [u32; 3]> = HashMap::with_capacity(n_elements);

    for elem in connectivity.chunks_exact(4) {
        // Refer the gmsh docs for node numbering scheme:
        // <https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering>
        let faces = [
            [elem[0], elem[3], elem[2]],
            [elem[0], elem[2], elem[1]],
            [elem[0], elem[1], elem[3]],
            [elem[1], elem[2], elem[3]],
        ];

        for face in faces {
            let mut key = face;

            // Internal faces will have opposite ordering, so these need to be sorted
            key.sort();
            if let std::collections::hash_map::Entry::Vacant(e) = face_map.entry(key) {
                e.insert(face);
            } else {
                face_map.remove(&key);
            }
        }
    }

    for face in face_map.values() {
        faces_out.push(face[0]);
        faces_out.push(face[1]);
        faces_out.push(face[2]);
    }

    faces_out
}

/// Compute the area and normal vector of each of the surface faces on a tetrahedral mesh
pub fn surface_face_properties(nodes: &[f64], surface_faces: &[u32]) -> (Vec<f64>, Vec<f64>) {
    let n_faces = surface_faces.len() / 3;

    let mut areas: Vec<f64> = vec![0.0; n_faces];
    let mut normals: Vec<f64> = vec![0.0; 3 * n_faces];
    for (i, face) in surface_faces.chunks_exact(3).enumerate() {
        let n0 = node_coords(nodes, face[0] as usize);
        let n1 = node_coords(nodes, face[1] as usize);
        let n2 = node_coords(nodes, face[2] as usize);
        let e0 = n2 - n1;
        let e1 = n0 - n1;
        let e0_cross_e1 = e0.cross(&e1);
        let norm_e0_cross_e1 = e0_cross_e1.mag();
        let normal = e0_cross_e1 * (1.0 / norm_e0_cross_e1);

        // Accumulate

        // area = 1/2 (e0 x e1)
        areas[i] = 0.5 * norm_e0_cross_e1;
        normals[i * 3] = normal[0];
        normals[i * 3 + 1] = normal[1];
        normals[i * 3 + 2] = normal[2];
    }

    (areas, normals)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use the example here:
    // <https://en.wikipedia.org/wiki/Regular_tetrahedron#Cartesian_coordinates>
    #[test]
    fn test_centroids() {
        let nodes = [
            -1.0,
            0.0,
            -1.0 / (2.0f64).sqrt(),
            1.0,
            0.0,
            -1.0 / (2.0f64).sqrt(),
            0.0,
            -1.0,
            1.0 / (2.0f64).sqrt(),
            0.0,
            1.0,
            1.0 / (2.0f64).sqrt(),
        ];
        let connectivity: [u32; 4] = [0, 1, 2, 3];
        let mut x = [1.0];
        let mut y = [1.0];
        let mut z = [1.0];
        centroids(&nodes, &connectivity, &mut x, &mut y, &mut z);

        // Centroid should be at origin
        assert!(x[0].abs() < 1e-8);
        assert!(y[0].abs() < 1e-8);
        assert!(z[0].abs() < 1e-8);
    }
}
