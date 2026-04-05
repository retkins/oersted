//! Magnetization calculations for oersted
//!
//! These differ from the standard direct/octree calculations in that they:
//! * Require a material defined for the targets
//! * In the present form, require a gradient calculation at the targets, which is simplified
//! by only considering meshes as the targets
//! * Require iteration and therefore benefit from non-trivial solver techniques

use crate::{biotsavart_parallel::h_mag_tet4_direct_parallel, types::Vec3};

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
    centroids: (&[f64], &[f64], &[f64]),
    chi: f64,
    hext: (&[f64], &[f64], &[f64]),
    solver: Solver,
    tol: f64,
    max_iterations: u32,
) -> ((Vec<f64>, Vec<f64>, Vec<f64>), Vec<Vec3>) {
    let n_elem: usize = connectivity.len();
    let n_centroids: usize = centroids.0.len();

    // Initialize memory for results
    let [mut hx, mut hy, mut hz] = [vec![0.0; n_elem], vec![0.0; n_elem], vec![0.0; n_elem]];

    // // Intermediate result: unit vector for direction of the magnetization field
    // let [mut mhatx, mut mhaty, mut mhatz] =
    //     [vec![0.0; n_elem], vec![0.0; n_elem], vec![0.0; n_elem]];
    // let [mut hhatx, mut hhaty, mut hhatz] =
    //     [vec![0.0; n_elem], vec![0.0; n_elem], vec![0.0; n_elem]];

    let mut mvectors = vec![Vec3::default(); n_centroids];
    for it in 0..max_iterations {
        // Dispatch over solver method to compute the current iteration of the demag field
        match solver {
            Solver::Tet4Direct(n_threads) => {
                h_mag_tet4_direct_parallel(
                    nodes,
                    connectivity,
                    &mvectors,
                    centroids,
                    (&mut hx, &mut hy, &mut hz),
                    n_threads,
                )
                .unwrap();
            }
            _ => {}
        }

        let mut max_change = 0.0;
        for i in 0..n_centroids {
            let mx_new: f64 = chi * (hx[i] + hext.0[i]);
            let my_new: f64 = chi * (hy[i] + hext.1[i]);
            let mz_new: f64 = chi * (hz[i] + hext.2[i]);

            let mx_change = (mvectors[i][0] - mx_new).abs();
            let my_change = (mvectors[i][1] - my_new).abs();
            let mz_change = (mvectors[i][2] - mz_new).abs();

            if mx_change > max_change {
                max_change = mx_change;
            }
            if my_change > max_change {
                max_change = my_change;
            }
            if mz_change > max_change {
                max_change = mz_change;
            }
            mvectors[i][0] = mx_new;
            mvectors[i][1] = my_new;
            mvectors[i][2] = mz_new;
        }

        println!("Iteration: {}; max change: {}", it, max_change);

        if max_change <= tol {
            break;
        } else {
            // zero the results vector between calls
            hx.fill(0.0);
            hy.fill(0.0);
            hz.fill(0.0);
        }
    }

    // Return h = h_demag + h_ext
    for i in 0..n_centroids {
        hx[i] += hext.0[i];
        hy[i] += hext.1[i];
        hz[i] += hext.2[i];
    }

    ((hx, hy, hz), mvectors)
}
