//! Internals used by multiple transient solvers
#![allow(unused)]

use faer::{diag::Diag, matrix_free::LinOp, sparse::SparseColMat};

use crate::{
    krylov::LinearOperator, math::min_and_max, mesh::Mesh, octree::OctreeSettings, types::Vec3,
};
use std::f64::consts::PI;

pub type Triplets = Vec<(usize, usize, f64)>;

pub enum TransientSolver {
    DenseDirect,
    DenseIterative,
    BH,
}

pub struct CyclicOptions {
    pub n_sectors: usize,
    pub low_angle: f64,
    pub high_angle: f64,
    pub atol_distance: f64,
    pub atol_angle: f64,
}

pub struct TransientOptions {
    pub solver: TransientSolver,
    pub verbose: bool,
    pub rtol: f64,
    pub cyclic: Option<CyclicOptions>,
    pub octree_settings: OctreeSettings,
}

/// Assemble the resistance diagonal matrix R
///
/// This matrix has length `n_elems`, each of which are rho*vol[e]
///
pub fn assemble_r(mesh: &Mesh, rho_values: &[f64], rho_indices: &[usize]) -> Diag<f64> {
    assert_eq!(rho_values.len(), rho_indices.len());
    let mut r = Diag::zeros(mesh.n_elems());
    let mut j = 0usize;
    let mut rho = rho_values[j];
    for i in 0..mesh.n_elems() {
        if i >= j {
            j = i;
        }

        r[i] = rho_values[j] * mesh.volumes[i];
    }
    r
}

/// Find nodes to pin using a union-find over all of the nodes:
/// https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Finding_set_representatives
/// This finds the lowest numbered node in each 'island' body, which is the
/// node that is chosen to have the gauge reduced (equal to zero, "set to ground")
///
/// Returns: indices in the global mesh that should be guage-pinned
pub fn find_pin_nodes(connectivity: &[[u32; 4]], n_nodes: usize, verbose: bool) -> Vec<usize> {
    let mut pinned: Vec<usize> = Vec::new();

    // Tarjan and Van Leeuwen path-halving algorithm:
    // https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Finding_set_representatives
    fn find(mut i: usize, parent: &mut [usize]) -> usize {
        while i != parent[i] {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    let mut parent: Vec<usize> = (0..n_nodes).collect();
    for elem in connectivity {
        let root0 = find(elem[0] as usize, &mut parent);
        for &n in &elem[1..] {
            let root = find(n as usize, &mut parent);
            if root != root0 {
                parent[root] = root0;
            }
        }
    }

    // Return only the nodes that are the parent of other nodes on that island
    for i in 0..n_nodes {
        if find(i, &mut parent) == i {
            pinned.push(i);
        }
    }

    if verbose {
        println!(
            "Number of nodes chosen for gauge pinning on island bodies: {}",
            pinned.len()
        );
        println!(
            "If this does not match the number of individual disconnected bodies in your model, check the input data carefully."
        );
    }

    pinned
}

/// Find all nodes on cyclic cut boundary faces to be included in tie constraints
/// Returns: node renumbering to use when forming the G matrix,
/// and the count of reduced nodes (number of phi DOF).
///
/// This function shouldn't be used for cyclic faces that enter the second or
/// third cartesien quadrants (phi > pi or phi < pi), because atan2() doesn't
/// correctly compute the angle for these nodes.
///
/// To use: node_map[original index] -> reduced index
///
pub fn find_cyclic_nodes(
    mesh: &Mesh,
    n_sectors: usize,
    (theta_low, theta_high): (f64, f64),
    (tol_distance, tol_angle): (f64, f64),
    verbose: bool,
) -> (Vec<usize>, usize) {
    let theta_total = theta_high - theta_low;
    assert!((theta_total * n_sectors as f64 - 2.0 * PI).abs() < tol_angle);

    // Theta coordinate (r-theta-z space) of each node location; check that
    // the node coordinates fall within the angular extent specified
    let r: Vec<f64> = mesh.nodes.iter().map(|p| p[1].hypot(p[0])).collect();
    let theta: Vec<f64> = mesh.nodes.iter().map(|p| p[1].atan2(p[0])).collect();
    let (min_theta, max_theta) = min_and_max(&theta).unwrap();
    assert!(min_theta >= theta_low - tol_angle);
    assert!(max_theta <= theta_high + tol_angle);

    // Compute the nodes on each cut boundary face. Nodes must be some distance away
    // from the cyclic axis to be included in the cut boundary set
    let low: Vec<usize> = (0..mesh.nodes.len())
        .filter(|&i| (theta[i] - theta_low).abs() < tol_angle && r[i] > tol_distance)
        .collect();
    let high: Vec<usize> = (0..mesh.nodes.len())
        .filter(|&i| (theta[i] - theta_high).abs() < tol_angle && r[i] > tol_distance)
        .collect();
    assert_eq!(
        low.len(),
        high.len(),
        "Cut cyclic boundary faces do not have matching node counts."
    );
    let n_pairs = low.len();

    // Compute matching pairs by sorting the low and high node sets by rz coordinates
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n_pairs);
    let high_sorted = sort_nodes(mesh, &high);
    let low_sorted = sort_nodes(mesh, &low);
    for i in 0..n_pairs {
        let (hr, hz) = rz(&mesh.nodes[high_sorted[i]]);
        let (lr, lz) = rz(&mesh.nodes[low_sorted[i]]);
        assert!((hr - lr).abs() < tol_distance && (hz - lz).abs() < tol_distance);
        pairs.push((low_sorted[i], high_sorted[i]));
    }

    // Determine which nodes are 'folded' into the other
    let mut folded = vec![false; mesh.n_nodes()];
    for (_, h) in &pairs {
        folded[*h] = true;
    }

    // Nodes now need to be renumbered
    let mut node_map = vec![usize::MAX; mesh.n_nodes()];
    let mut n_reduced = 0usize;
    for i in 0..mesh.n_nodes() {
        if !folded[i] {
            node_map[i] = n_reduced;
            n_reduced += 1;
        }
    }
    for (l, h) in pairs {
        node_map[h] = node_map[l];
    }

    if verbose {
        println!(
            "Computed an angular extent of {:.3} degrees each for {} total sectors.",
            theta_total * 180.0 / (2.0 * PI),
            n_sectors
        );
        println!("{} matching cyclic node pairs found.", n_pairs);
    }

    (node_map, n_reduced)
}

// Compute rz coordinates on a point
fn rz(p: &Vec3) -> (f64, f64) {
    (p[0].hypot(p[1]), p[2])
}

// Sort node sets by RZ coordinates
fn sort_nodes(mesh: &Mesh, node_set: &[usize]) -> Vec<usize> {
    let mut sorted = node_set.to_vec();
    sorted.sort_by(|&a, &b| {
        let (ra, za) = rz(&mesh.nodes[a]);
        let (rb, zb) = rz(&mesh.nodes[b]);
        za.partial_cmp(&zb)
            .unwrap()
            .then(ra.partial_cmp(&rb).unwrap())
    });
    sorted
}

// Assemble the constraint-gradient matrix G
//
// This matrix is 3*num_elems x num_nodes. The first num_elems rows are for the
// x-dof, second num_elems (second third) rows are for y-dof, etc.
//
// To save on memory, G is saved as COO triplets (sparse) and never formed into its own
// array. Instead, it is scattered into the KKT system directly.
pub fn assemble_g(
    mesh: &Mesh,
    node_map: &[usize],
    n_reduced: usize,
    grounded: &[usize],
) -> Triplets {
    let mut triplets: Triplets = Vec::with_capacity(12 * mesh.n_elems());
    let mut pinned = vec![false; n_reduced];
    for &g in grounded {
        pinned[g] = true;
    }

    for e in 0..mesh.n_elems() {
        let vg_e: [Vec3; 4] = mesh.hat_gradients(e);

        for ni in 0..4usize {
            let n: usize = mesh.connectivity[e][ni] as usize;
            let col = node_map[n];
            if pinned[col] {
                // Skip pinned columns
                continue;
            }
            for k in 0..3usize {
                triplets.push((mesh.n_elems() * k + e, col, vg_e[ni][k]));
            }
        }
    }
    triplets
}

/// Assemble the momentum diagonal, diag(A) = R + M[e,e]/dt
///
/// Computing M[e,e] can't be a simple loop over self-fields, because for cyclic
/// calculations, there are n_sectors-1 additional terms to consider.
pub fn assemble_d(mesh: &Mesh, r: &Diag<f64>, dt: f64, m: &impl LinearOperator) -> Vec<f64> {
    let inv_dt: f64 = 1.0 / dt;
    let n_el = mesh.n_elems();
    let mut d: Vec<f64> = vec![0.0; 3 * mesh.n_elems()];
    m.diagonal(&mut d);

    for c in 0..3 {
        for i in 0..d.len() {
            let k = c * n_el + i;
            d[k] = r[i] + d[k] * inv_dt;
        }
    }

    debug_assert!(d.iter().all(|&v| v > 0.0 && v.is_finite()));
    d
}

/// Assemble the nodal graph Laplacian, `L = G^T D^-1 G`
// pub fn assemble_l(g_coo: &Triplets, d: &Diag<f64>, n_el: usize, n_reduced: usize) {
//     // G size `3*n_el x n_reduced`
//     // L is square and size `n_reduced`
//     let g_size = 3*n_el + n_reduced;
//     let g = SparseColMat::try_new_from_triplets(
//         g_size, g_size, g_coo).unwrap();
//     let mut d_inv = d.clone();
//     for i in 0..d_inv.nrows() {
//         d_inv[i] = 1.0 / d_inv[i];
//     }

//     let l = g.transpose() * d_inv * g;
// }

#[cfg(test)]
mod tests {
    use std::assert_eq;

    /// Simple test of the pinned-node finding algorithm. Two 2d meshes of
    /// triangles, one with 3 elements and the other with two. The pinned nodes
    /// should be 0 (lowest on first body) and 5 (the lowest on the second body).
    /// Nodes are numbered from 0, so these are indices of the nodes themselves.
    #[test]
    fn test() {
        let connectivity = [[0, 1, 2], [1, 3, 2], [3, 4, 2], [5, 7, 6], [5, 8, 7]];
        let n_nodes = 9usize;

        fn find(mut i: usize, parent: &mut [usize]) -> usize {
            println!("Finding {} in {:?}", i, parent);
            while i != parent[i] {
                println!("i = {} was not equal to parent[i] = {}", i, parent[i]);
                parent[i] = parent[parent[i]];
                println!(
                    "parent[parent[i]] = {}, now in parent[i]",
                    parent[parent[i]]
                );
                i = parent[i];
                println!("i = parent[i] = {}", i);
            }
            i
        }

        let mut parent: Vec<usize> = (0..n_nodes).collect();
        for elem in &connectivity {
            let root0 = find(elem[0] as usize, &mut parent);
            for &n in &elem[1..] {
                let root = find(n as usize, &mut parent);
                if root != root0 {
                    parent[root] = root0;
                }
            }
        }

        println!("{:?}", parent);

        for (i, &p) in parent.iter().enumerate() {
            if i < 5 {
                assert_eq!(p, 0usize);
            } else {
                assert_eq!(p, 5usize);
            }
        }
    }
}
