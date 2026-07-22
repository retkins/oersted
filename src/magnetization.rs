//! Magnetization calculations for oersted
//!
//! These differ from the standard direct/octree calculations in that they:
//! * Require a material defined for the targets
//! * In the present form, require a gradient calculation at the targets, which is simplified
//!   by only considering meshes as the targets
//! * Require iteration and therefore benefit from non-trivial solver techniques

use crate::{
    biotsavart::{IntegrationMethod, RequestedField, SourceVectors, calculate_fields},
    octree::{Octree, OctreeSettings, Source},
    types::Vec3,
};

/// Compute the magnetization of a finite element mesh, using a background field generated
/// by sources that are assumed to be independent of the magnetized mesh.
///
/// Currently, this function is only defined for linear magnetic materials
///
/// # Arguments
/// * `nodes`: (m) x,y,z coordinates of each node in the mesh
/// * `connectivity`: indices into `nodes` representing each node in the element
/// * `centroids`: (m) x,y,z coordinates of the centroid of each element in the mesh
/// * `chi`: magnetic susceptibility of the material
/// * `hext`: (A/m) external magnetic field acting on the mesh
/// * `solver`: select point/tet4 direct/octree integration
/// * `tol`: (A/m) convergence criteria
/// * `max_iterations`: number of iterations before exiting
/// * `alpha`: under-relaxation factor (smaller for more stability, larger for faster convergence)
///
/// # Returns
/// (H_total, M): total H and M fields acting on each element
///     H_total = H_external - H_demag
///     B = mu0 * (H_total + M)
pub fn magnetization_solve(
    nodes: &[Vec3],
    connectivity: &[[u32; 4]],
    centroids: (&[f64], &[f64], &[f64]),
    chi: f64,
    h_ext: (&[f64], &[f64], &[f64]),
    m_out: &mut [Vec3],
    h_total_out: (&mut [f64], &mut [f64], &mut [f64]),
    method: IntegrationMethod,
    n_threads_requested: u32,
    atol: f64,
    max_iterations: u32,
    under_relaxation_factor: f64,
    octree_settings: Option<OctreeSettings>,
    verbose: bool,
) {
    let n_centroids: usize = centroids.0.len();

    // Initialize memory for results
    let (hx, hy, hz) = h_total_out;
    let mvectors = m_out;

    // initial guess: note that starting an initial guess of zeros causes the
    // octree solver to calculate extremely slowly
    for (i, mvector) in mvectors.iter_mut().enumerate() {
        *mvector = Vec3([h_ext.0[i], h_ext.1[i], h_ext.2[i]]) * chi;
    }

    let mut octree =
        octree_settings.map(|s| Octree::new(nodes, connectivity, None, Some(mvectors), s));

    for it in 0..max_iterations {
        // Dispatch over solver method to compute the current iteration of the demag field
        if let Some(oc) = &mut octree {
            oc.update_magnetization(mvectors);
            oc.compute_fields(
                centroids,
                (hx, hy, hz),
                RequestedField::HField,
                Source::Magnetization,
            );
        } else {
            calculate_fields(
                nodes,
                connectivity,
                SourceVectors::Magnetization(mvectors),
                RequestedField::HField,
                centroids,
                (hx, hy, hz),
                method,
                n_threads_requested,
            );
        }

        let mut max_change: f64 = 0.0;
        for i in 0..n_centroids {
            let mx_new: f64 = chi * (hx[i] + h_ext.0[i]);
            let my_new: f64 = chi * (hy[i] + h_ext.1[i]);
            let mz_new: f64 = chi * (hz[i] + h_ext.2[i]);

            let mx_change: f64 = (mvectors[i][0] - mx_new).abs();
            let my_change: f64 = (mvectors[i][1] - my_new).abs();
            let mz_change: f64 = (mvectors[i][2] - mz_new).abs();

            max_change = max_change.max(mx_change).max(my_change).max(mz_change);

            // Use under-relaxation to improve convergence for higher mu_r materials
            let alpha: f64 = under_relaxation_factor;
            mvectors[i][0] = alpha * mx_new + (1.0 - alpha) * mvectors[i][0];
            mvectors[i][1] = alpha * my_new + (1.0 - alpha) * mvectors[i][1];
            mvectors[i][2] = alpha * mz_new + (1.0 - alpha) * mvectors[i][2];
        }

        if verbose {
            println!("Iteration: {}; max change: {:.3e}", it, max_change);
        }

        if max_change <= atol {
            break;
        } else if it < max_iterations - 1 {
            // zero the results vector between calls
            // Do not zero on the last iteration
            hx.fill(0.0);
            hy.fill(0.0);
            hz.fill(0.0);
        }
    }

    // Return h = h_demag + h_ext
    for i in 0..n_centroids {
        hx[i] += h_ext.0[i];
        hy[i] += h_ext.1[i];
        hz[i] += h_ext.2[i];
    }
}
