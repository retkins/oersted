#![allow(non_snake_case)]

use crate::MU0_4PI;
use crate::mesh::node_coords;
use crate::vec3::Vec3;

/// Compute the magnetic field at target points (x, y, z) using a direct (O(N^2)) Biot-Savart summation
///
/// This is the 'scalar' version of this function; i.e. it uses only one thread.
///
/// # Arguments
/// - `centx`, `centy`, `centz`: (m) locations of source element centroids in 3D space
/// - `vol`:                     (m^3) volume of each source element
/// - `jx`, `jy`, `jz`:          (A/m^2) current density vector of each source element
/// - `x`, `y`, `z`:             (m) location of each target point
/// - `bx`, `by`, `bz`:          (T) magnetic flux density at each target point
pub fn bfield_direct(
    centx: &[f64],
    centy: &[f64],
    centz: &[f64],
    vol: &[f64],
    jx: &[f64],
    jy: &[f64],
    jz: &[f64],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    bx: &mut [f64],
    by: &mut [f64],
    bz: &mut [f64],
) -> Result<(), ()> {
    // TODO: length checks on input arrays

    // Outer loop over source elements
    for i in 0..centx.len() {
        // Hoist invariants out of the loop explicitly
        let centxi: f64 = centx[i];
        let centyi: f64 = centy[i];
        let centzi: f64 = centz[i];
        let vol_mu0_4pi: f64 = vol[i] * MU0_4PI;
        let jxi = jx[i];
        let jyi = jy[i];
        let jzi = jz[i];

        // Inner loop over target points
        for (((xj, yj), zj), ((bxj, byj), bzj)) in x
            .iter()
            .zip(y.iter())
            .zip(z.iter())
            .zip(bx.iter_mut().zip(by.iter_mut()).zip(bz.iter_mut()))
        {
            // Vector from the element centroid to the target point: r'
            let rx: f64 = xj - centxi;
            let ry: f64 = yj - centyi;
            let rz: f64 = zj - centzi;
            let r: f64 = (rx * rx + ry * ry + rz * rz).sqrt();

            // J x r'
            let jxrpx: f64 = jyi * rz - jzi * ry;
            let jxrpy: f64 = jzi * rx - jxi * rz;
            let jxrpz: f64 = jxi * ry - jyi * rx;

            // Null out the singularity around the element centroid
            // This avoids `jmp` instructions and enables auto-vectorization of the inner loop
            // Hard-coded a tolerance of 0.1mm (TODO: update)
            let mask = if r > 1e-4 { 1.0 } else { 0.0 };
            let r3 = r * r * r + (1.0 - mask);
            let constant = vol_mu0_4pi * mask / r3;

            // Accumulation
            *bxj += constant * jxrpx;
            *byj += constant * jxrpy;
            *bzj += constant * jxrpz;
        }
    }
    Ok(())
}

// Older version of the above that uses explicit indices for the inner loop
#[allow(unused)]
fn bfield_direct_old(
    centx: &[f64],
    centy: &[f64],
    centz: &[f64],
    vol: &[f64],
    jx: &[f64],
    jy: &[f64],
    jz: &[f64],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    bx: &mut [f64],
    by: &mut [f64],
    bz: &mut [f64],
) {
    let m: usize = centx.len();
    let n: usize = x.len();

    for i in 0..m {
        let centxi: f64 = centx[i];
        let centyi: f64 = centy[i];
        let centzi: f64 = centz[i];
        let vol_mu0_4pi: f64 = vol[i] * MU0_4PI;
        let jxi = jx[i];
        let jyi = jy[i];
        let jzi = jz[i];

        for j in 0..n {
            let rx: f64 = x[j] - centxi;
            let ry: f64 = y[j] - centyi;
            let rz: f64 = z[j] - centzi;
            let r: f64 = (rx * rx + ry * ry + rz * rz).sqrt();

            let jxrpx: f64 = jyi * rz - jzi * ry;
            let jxrpy: f64 = jzi * rx - jxi * rz;
            let jxrpz: f64 = jxi * ry - jyi * rx;

            let mask = if r > 1e-4 { 1.0 } else { 0.0 };
            let r3 = r * r * r + (1.0 - mask); // avoid div by zero
            let constant = vol_mu0_4pi * mask / r3;
            bx[j] += constant * jxrpx;
            by[j] += constant * jxrpy;
            bz[j] += constant * jxrpz;
        }
    }
}

use crate::sources::tet4::h_field_tet4;

/// Compute the magnetic field using the direct tetrahedral integration method
///
///
pub fn hfield_direct_tet(
    nodes: &[f64],
    connectivity: &[u32],
    jdensity_flat: &[f64],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    hx: &mut [f64],
    hy: &mut [f64],
    hz: &mut [f64],
) -> Result<(), ()> {
    let n_targets = x.len();
    let mut f = vec![Vec3([0.0; 3]); n_targets];

    // TODO: length checks
    for (i, elem) in connectivity.chunks_exact(4).enumerate() {
        let nodes = [
            node_coords(nodes, elem[0] as usize),
            node_coords(nodes, elem[1] as usize),
            node_coords(nodes, elem[2] as usize),
            node_coords(nodes, elem[3] as usize),
        ];

        let jdensity = Vec3([
            jdensity_flat[3 * i],
            jdensity_flat[3 * i + 1],
            jdensity_flat[3 * i + 2],
        ]);
        h_field_tet4(&nodes, &jdensity, (x, y, z), &mut f, (hx, hy, hz));
        f.fill(Vec3([0.0; 3]));
    }

    Ok(())
}

use crate::math::gradient::jmatrices;
use crate::sources::tet4::hmag_tet4;

pub fn hmag_direct_tet(
    source_nodes: (&[f64], &[f64], &[f64]),
    source_element_connectivity: &[[u32; 4]],
    source_mvectors: &[Vec3],
    target_nodes: (&[f64], &[f64], &[f64]),
    target_element_connectivity: &[[u32; 4]],
    hx: &mut [f64],
    hy: &mut [f64],
    hz: &mut [f64],
) -> Result<(), ()> {
    let n_sources = source_element_connectivity.len();
    let n_target_nodes = target_nodes.0.len();

    // Compute j_invt at each target element
    let mut node_vecs = vec![Vec3::default(); n_target_nodes];
    for i in 0..n_target_nodes {
        node_vecs[i] = Vec3([target_nodes.0[i], target_nodes.1[i], target_nodes.2[i]]);
    }
    let j_invt = jmatrices(&node_vecs, target_element_connectivity);

    let mut f = vec![Vec3::default(); n_target_nodes];

    for i in 0..n_sources {
        let n_idx = source_element_connectivity[i];
        let _source_nodes = [
            Vec3([
                source_nodes.0[n_idx[0] as usize],
                source_nodes.1[n_idx[0] as usize],
                source_nodes.2[n_idx[0] as usize],
            ]),
            Vec3([
                source_nodes.0[n_idx[1] as usize],
                source_nodes.1[n_idx[1] as usize],
                source_nodes.2[n_idx[1] as usize],
            ]),
            Vec3([
                source_nodes.0[n_idx[2] as usize],
                source_nodes.1[n_idx[2] as usize],
                source_nodes.2[n_idx[2] as usize],
            ]),
            Vec3([
                source_nodes.0[n_idx[3] as usize],
                source_nodes.1[n_idx[3] as usize],
                source_nodes.2[n_idx[3] as usize],
            ]),
        ];
        hmag_tet4(
            &_source_nodes,
            &source_mvectors[i],
            &j_invt,
            target_nodes,
            target_element_connectivity,
            &mut f,
            (hx, hy, hz),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_direct() {}
}
