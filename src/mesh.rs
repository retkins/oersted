//! Finite element mesh operations

use crate::vec3::Vec3;

/// Return the coordinates of a node with number `idx`
///
/// `nodes` is a flat array representing an Nx3 table of nodal coordinates
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
