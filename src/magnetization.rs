//! Magnetization calculations for oersted
//!
//! These differ from the standard direct/octree calculations in that they:
//! * Require a material defined for the targets
//! * In the present form, require a gradient calculation at the targets, which is simplified
//! by only considering meshes as the targets
//! * Require iteration and therefore benefit from non-trivial solver techniques

use crate::{
    math::gradient,
    sources::h_mag_tet4,
    types::{Mat3, Vec3},
};

pub enum Solver {
    PointDirect(u32),           // num threads
    PointOctree(u32, f64, u32), // num threads, theta, leaf_threshold
    Tet4Direct(u32),
    Tet4Octree(u32, f64, u32),
}

/// Compute the magnetization of a finite element mesh, using a background field generated
/// by sources that are assumed to be independent of the magnetized mesh.
///
/// Currently, this function is only defined for linear magnetic materials
///
/// # Arguments
///
/// # Returns
/// (H_total, M): total H and M fields acting on each element
///     H_total = H_external - H_demag
/// B = mu0 * (H_total + M)
pub fn magnetization(
    nodes: &[Vec3],
    connectivity: &[[u32; 4]],
    centroids: &[Vec3],
    chi: f64,
    hext: (&[f64], &[f64], &[f64]),
    solver: Solver,
    tol: f64,
    max_iterations: u32,
) -> (
    (Vec<f64>, Vec<f64>, Vec<f64>),
    (Vec<f64>, Vec<f64>, Vec<f64>),
) {
    let n_elem: usize = connectivity.len();

    // Initialize memory for results
    let [mut hx, mut hy, mut hz, mut mx, mut my, mut mz] = [
        vec![0.0; n_elem],
        vec![0.0; n_elem],
        vec![0.0; n_elem],
        vec![0.0; n_elem],
        vec![0.0; n_elem],
        vec![0.0; n_elem],
    ];

    // Intermediate result: unit vector for direction of the magnetization field
    let [mut mhatx, mut mhaty, mut mhatz] =
        [vec![0.0; n_elem], vec![0.0; n_elem], vec![0.0; n_elem]];
    let j_invt: Vec<Mat3> = gradient::jmatrices(nodes, connectivity);

    for _ in 0..max_iterations {
        // Dispatch over solver method to compute the current iteration of the demag field
        match solver {
            Solver::Tet4Direct(n_threads) => {
                for i in 0..n_elem {
                    let elem = connectivity[i];
                    let elem_nodes = [
                        nodes[elem[0] as usize],
                        nodes[elem[1] as usize],
                        nodes[elem[2] as usize],
                        nodes[elem[3] as usize],
                    ];
                    // hmag_tet4(&elem_nodes, &Vec3([mx[i], my[i], mz[i]]), &[j_invt[i]], &elem_nodes, &elem, f, h);
                }
            }
            _ => {}
        }
    }

    ((hx, hy, hz), (mx, my, mz))
}
