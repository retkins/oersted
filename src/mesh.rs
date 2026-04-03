//! Finite element mesh operations

use crate::MU0;
use crate::math::mag3;
use crate::types::{Mat3, Vec3};
use std::collections::HashMap;

const INV_MU0: f64 = 1.0 / MU0;

/// Return the coordinates of a node with number `idx`
///
/// `nodes` is a flat array representing an Nx3 table of nodal coordinates
#[inline]
pub fn node_coords(nodes: &[f64], idx: usize) -> Vec3 {
    Vec3([nodes[3 * idx], nodes[3 * idx + 1], nodes[3 * idx + 2]])
}

/// Compute the centroids of the elements in a mesh containing
/// 4-node linear tetrahedral elements by taking the mean position
/// of all 4 nodes
///
/// # Arguments
///
/// * `nodes`: (m) (x,y,z) node coordinates of each node in the mesh
/// * `connectivity`: indices into `nodes` of the four nodes in each element
/// * `out`: (m) output buffer, allocated by the caller
pub fn centroids(nodes: &[Vec3], connectivity: &[[u32; 4]], out: &mut [Vec3]) {
    assert_eq!(connectivity.len(), out.len());

    for (i, elem) in connectivity.iter().enumerate() {
        // Node numbers
        let [n0, n1, n2, n3] = [
            elem[0] as usize,
            elem[1] as usize,
            elem[2] as usize,
            elem[3] as usize,
        ];

        out[i] = (nodes[n0] + nodes[n1] + nodes[n2] + nodes[n3]) * 0.25;
    }
}

/// Compute the volumes of the elements in a mesh containing
/// 4-node linear tetrahedral elements:
///
/// $$ volume = (1/6) \cdot |\vec{a} \times (\vec{b} - \vec{c}))| $$
///
/// # Arguments
///
/// * `nodes`: (m) (x,y,z) node coordinates of each node in the mesh
/// * `connectivity`: indices into `nodes` of the four nodes in each element
/// * `out`: (m^3) output buffer, allocated by the caller
///
/// # Notes
/// This function is tested via a part meshed in Python.
///
/// # Reference
/// <https://en.wikipedia.org/wiki/Tetrahedron#Other_approaches>
pub fn volumes(nodes: &[Vec3], connectivity: &[[u32; 4]], out: &mut [f64]) {
    assert_eq!(connectivity.len(), out.len());

    for (i, elem) in connectivity.iter().enumerate() {
        // Node numbers
        let [n0, n1, n2, n3] = [
            elem[0] as usize,
            elem[1] as usize,
            elem[2] as usize,
            elem[3] as usize,
        ];

        // Vertices
        let [v0, v1, v2, v3] = [nodes[n0], nodes[n1], nodes[n2], nodes[n3]];

        // Edge vectors
        let a = v1 - v0;
        let b = v2 - v0;
        let c = v3 - v0;

        out[i] = (1.0 / 6.0) * (a.dot(&b.cross(&c))).abs();
    }
}

/// Determine which faces in a mesh are on the surface
///
/// # Arguments
/// * `connectivity`: flat array of indices into `nodes` of the four nodes in each element; row-major order
///
/// # Returns
/// a flat vector of the indices for each node in the surface faces (3 nodes per face)
///
/// # Notes
/// * This function is tested via a part meshed in Python.
/// * This function cannot know the number of surface elements before it is called and
///   therefore must allocate memory instead of writing to a fixed-size preallocated buffer.
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

/// Compute the area, centroid, and normal vector of each of the surface faces on a tetrahedral mesh
///
/// # Arguments
/// * `nodes`: (m) flat array of nodal coordinates (row-major)
/// * `surface_faces`: flat array of indices into `nodes` that define an element's surface face;
///   row-major, 3 nodes per face
///
/// # Returns
/// (areas, centroids, normal_vectors): (m^2, m) `centroids` and `normal_vectors` are flat
/// arrays (row-major)
///
/// # Notes
/// * This function is tested via the Python API.
pub fn surface_face_properties(
    nodes: &[f64],
    surface_faces: &[u32],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_faces = surface_faces.len() / 3;

    let mut areas: Vec<f64> = vec![0.0; n_faces];
    let mut centroids: Vec<f64> = vec![0.0; 3 * n_faces];
    let mut normals: Vec<f64> = vec![0.0; 3 * n_faces];
    for (i, face) in surface_faces.chunks_exact(3).enumerate() {
        let n0 = node_coords(nodes, face[0] as usize);
        let n1 = node_coords(nodes, face[1] as usize);
        let n2 = node_coords(nodes, face[2] as usize);
        let centroid = (n0 + n1 + n2) * (1.0 / 3.0);
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
        centroids[i * 3] = centroid[0];
        centroids[i * 3 + 1] = centroid[1];
        centroids[i * 3 + 2] = centroid[2];
    }

    (areas, centroids, normals)
}

/// Compute the Maxwell stress tensor on a series of triangular surface faces
///
/// # Reference:
/// "Theory and applications of the Maxwell stress tensor", Field Precision LLC
/// <https://www.fieldp.com/tutorials/stresstensor.pdf>
///
pub fn maxwell_stress_tensor(b_field: &[f64]) -> Vec<Mat3> {
    let n_faces: usize = b_field.len() / 3;
    let mut stress_tensor: Vec<Mat3> = vec![Mat3::default(); n_faces];
    let mut row: Vec3 = Vec3::default();

    for (i, tensor) in stress_tensor.iter_mut().enumerate() {
        let bx: f64 = b_field[i * 3];
        let by: f64 = b_field[i * 3 + 1];
        let bz: f64 = b_field[i * 3 + 2];
        let b: f64 = mag3(bx, by, bz);
        let b2_over_2: f64 = 0.5 * b * b;

        // Update tensor
        row[0] = bx * bx - b2_over_2;
        row[1] = bx * by;
        row[2] = bx * bz;
        tensor[0] = row;
        row[0] = by * bx;
        row[1] = by * by - b2_over_2;
        row[2] = by * bz;
        tensor[1] = row;
        row[0] = bz * bx;
        row[1] = bz * by;
        row[2] = bz * bz - b2_over_2;
        tensor[2] = row;
        *tensor *= INV_MU0;
    }

    stress_tensor
}

/// Compute the surface forces acting on a surface mesh using the Maxwell stress
/// tensor
pub fn surface_forces(
    face_areas: &[f64],
    face_normals: &[f64],
    stress_tensor: &[Mat3],
) -> Vec<Vec3> {
    let n_faces: usize = face_areas.len();
    assert_eq!(n_faces, face_areas.len());
    assert_eq!(n_faces, face_normals.len() / 3);
    assert_eq!(n_faces, stress_tensor.len());

    let mut forces = vec![Vec3::default(); n_faces];

    for (i, tensor) in stress_tensor.iter().enumerate() {
        let normal = Vec3([
            face_normals[i * 3],
            face_normals[i * 3 + 1],
            face_normals[i * 3 + 2],
        ]);
        forces[i] = tensor.mul_vec(&normal) * face_areas[i];
    }

    forces
}

/// Create surface tetrahedrons for calculating the gradient at the surface of a mesh
///
/// Returns (nodes, connectivity)
pub fn surface_tets(
    nodes: &[f64],
    faces: &[u32],
    centroids: &[f64],
    normals: &[f64],
) -> (Vec<f64>, Vec<u32>) {
    let n_faces = faces.len() / 3;
    let mut nodes_out: Vec<f64> = Vec::with_capacity(n_faces * 12); // 4 nodes x 3 coords per face
    let mut connectivity_out: Vec<u32> = Vec::with_capacity(n_faces * 4);

    for (i, face) in faces.chunks_exact(3).enumerate() {
        let first_node: Vec3 = node_coords(nodes, face[0] as usize);
        let centroid: Vec3 = node_coords(centroids, i); // still works for non-node things
        let normal: Vec3 = node_coords(normals, i);

        // Vector from centroid to 1st node in the surface triangle
        let v0: Vec3 = first_node - centroid;
        let v0_mag: f64 = v0.mag();
        let delta: f64 = 0.01 * v0_mag; // edge length is 2*delta
        let u0: Vec3 = v0 * (1.0 / v0_mag); // normal vector for two nodes outside the surface
        let u1: Vec3 = normal.cross(&u0); // normal vector for two nodes inside the surface

        // Now that we have our vectors, create nodes associated with them
        // using the definition of a regular tetrahedron centered at the origin
        // <https://en.wikipedia.org/wiki/Regular_tetrahedron#Cartesian_coordinates>
        let n0 = centroid + (u0 * delta) + (normal * delta * (1.0 / 2.0f64.sqrt()));
        let n1 = centroid + (u0 * (-delta)) + (normal * delta * (1.0 / 2.0f64.sqrt()));
        let n2 = centroid + (u1 * delta) + (normal * delta * (-1.0 / 2.0f64.sqrt()));
        let n3 = centroid + (u1 * (-delta)) + (normal * delta * (-1.0 / 2.0f64.sqrt()));

        // Accumulate
        nodes_out.push(n0[0]);
        nodes_out.push(n0[1]);
        nodes_out.push(n0[2]);
        nodes_out.push(n1[0]);
        nodes_out.push(n1[1]);
        nodes_out.push(n1[2]);
        nodes_out.push(n2[0]);
        nodes_out.push(n2[1]);
        nodes_out.push(n2[2]);
        nodes_out.push(n3[0]);
        nodes_out.push(n3[1]);
        nodes_out.push(n3[2]);
        connectivity_out.push((i * 4) as u32);
        connectivity_out.push((i * 4 + 1) as u32);
        connectivity_out.push((i * 4 + 2) as u32);
        connectivity_out.push((i * 4 + 3) as u32);
    }

    (nodes_out, connectivity_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use the example here:
    // <https://en.wikipedia.org/wiki/Regular_tetrahedron#Cartesian_coordinates>
    #[test]
    fn test_centroids() {
        let nodes: [Vec3; 4] = [
            Vec3([-1.0, 0.0, -1.0 / (2.0f64).sqrt()]),
            Vec3([1.0, 0.0, -1.0 / (2.0f64).sqrt()]),
            Vec3([0.0, -1.0, 1.0 / (2.0f64).sqrt()]),
            Vec3([0.0, 1.0, 1.0 / (2.0f64).sqrt()]),
        ];
        let connectivity: [[u32; 4]; 1] = [[0, 1, 2, 3]];
        // Set buffer to non-zero value
        let mut centroid = [Vec3::default()];
        centroids(&nodes, &connectivity, &mut centroid);

        // Centroid should be at origin
        for i in 0..3 {
            assert!(centroid[0][i].abs() < 1e-8);
        }
    }
}
