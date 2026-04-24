//! Magnetic field calculations

#![allow(non_snake_case)]

use crate::{
    math::mag3,
    mesh::node_coords,
    sources::{h_current_tet4, h_current_tet4_edge, h_mag_tet4, h_mag_tet4_edge, h_point_dipole},
    types::Vec3,
};
use std::f64::consts::PI;

const ONE_OVER_4PI: f64 = 1.0 / (4.0 * PI);

/// Compute the magnetic field strength at target points (x, y, z) using a direct (O(N^2)) Biot-Savart summation
///
/// This is the 'scalar' version of this function; i.e. it uses only one thread.
///
/// # Arguments
/// - `centx`, `centy`, `centz`: (m) locations of source element centroids in 3D space
/// - `vol`:                     (m^3) volume of each source element
/// - `jx`, `jy`, `jz`:          (A/m^2) current density vector of each source element
/// - `x`, `y`, `z`:             (m) location of each target point
/// - `out`:          (T) magnetic field strength at each target point
pub fn h_current_point_direct(
    src_pts: (&[f64], &[f64], &[f64]),
    src_vol: &[f64],
    src_jdensity: (&[f64], &[f64], &[f64]),
    tgt_pts: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), ()> {
    // Unpack
    let (centx, centy, centz) = src_pts;
    let (jx, jy, jz) = src_jdensity;
    let (x, y, z) = tgt_pts;
    let (hx, hy, hz) = out;
    // TODO: length checks on input arrays

    // Outer loop over source elements
    for i in 0..centx.len() {
        // Hoist invariants out of the loop explicitly
        let centxi: f64 = centx[i];
        let centyi: f64 = centy[i];
        let centzi: f64 = centz[i];
        let vol_over_4pi: f64 = src_vol[i] * ONE_OVER_4PI;
        let radius = ((3.0 / (4.0 * PI)) * src_vol[i]).powf(1.0 / 3.0);
        let R3 = radius * radius * radius;
        let jxi = jx[i];
        let jyi = jy[i];
        let jzi = jz[i];

        // Inner loop over target points
        for (((xj, yj), zj), ((hxj, hyj), hzj)) in x
            .iter()
            .zip(y.iter())
            .zip(z.iter())
            .zip(hx.iter_mut().zip(hy.iter_mut()).zip(hz.iter_mut()))
        {
            // Vector from the element centroid to the target point: r'
            let rx: f64 = xj - centxi;
            let ry: f64 = yj - centyi;
            let rz: f64 = zj - centzi;
            let r: f64 = mag3(rx, ry, rz);

            // J x r'
            let jxrpx: f64 = jyi.mul_add(rz, -jzi * ry);
            let jxrpy: f64 = jzi.mul_add(rx, -jxi * rz);
            let jxrpz: f64 = jxi.mul_add(ry, -jyi * rx);

            // Null out the singularity around the element centroid
            // This avoids `jmp` instructions and enables auto-vectorization of the inner loop
            let r3 = r * r * r;
            let constant = vol_over_4pi / r3.max(R3);

            // Accumulation
            *hxj = constant.mul_add(jxrpx, *hxj);
            *hyj = constant.mul_add(jxrpy, *hyj);
            *hzj = constant.mul_add(jxrpz, *hzj);
        }
    }
    Ok(())
}

/// Compute the magnetic field strength (H-field) at a set of target points, using a simplified dipole model
/// of a magnetized finite element mesh as the source
///
pub fn h_mag_point_direct(
    centroids: &[Vec3],
    volumes: &[f64],
    mvectors: &[Vec3],
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), ()> {
    let (hx, hy, hz) = out;

    for i in 0..centroids.len() {
        let radius = (volumes[i] * 3.0 / (4.0 * PI)).cbrt();
        let mvector: Vec3 = mvectors[i] * volumes[i];

        for j in 0..targets.0.len() {
            let target = Vec3([targets.0[j], targets.1[j], targets.2[j]]);
            let h = h_point_dipole(&centroids[i], &mvector, radius, &target);
            hx[j] += h[0];
            hy[j] += h[1];
            hz[j] += h[2];
        }
    }

    Ok(())
}

/// Compute the magnetic field using the direct tetrahedral integration method
///
///
pub fn h_current_tet4_direct(
    nodes: &[f64],
    connectivity: &[u32],
    jdensity_flat: &[f64],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    hx: &mut [f64],
    hy: &mut [f64],
    hz: &mut [f64],
    edge: bool,
) -> Result<(), ()> {
    let n_targets = x.len();

    let mut f: Vec<Vec3> = vec![Vec3([0.0; 3]); n_targets];

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
        if edge {
            h_current_tet4_edge(&nodes, &jdensity, (x, y, z), &mut f, (hx, hy, hz));
        } else {
            h_current_tet4(&nodes, &jdensity, (x, y, z), (hx, hy, hz));
        }

        if edge {
            f.fill(Vec3([0.0; 3]));
        }
    }

    Ok(())
}

/// Compute the H field due to the magnetization of a source mesh at a collection of target points
pub fn h_mag_tet4_direct(
    nodes: &[Vec3],
    connectivity: &[[u32; 4]],
    mvectors: &[Vec3],
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
    edge: bool,
) -> Result<(), ()> {
    let n_targets = targets.0.len();

    let mut wx: Vec<Vec3> = vec![Vec3::default(); n_targets];
    let mut wy: Vec<Vec3> = vec![Vec3::default(); n_targets];
    let mut wz: Vec<Vec3> = vec![Vec3::default(); n_targets];

    for (i, elem) in connectivity.iter().enumerate() {
        let elem_nodes = [
            nodes[elem[0] as usize],
            nodes[elem[1] as usize],
            nodes[elem[2] as usize],
            nodes[elem[3] as usize],
        ];

        if edge {
            h_mag_tet4_edge(
                &elem_nodes,
                &mvectors[i],
                targets,
                (&mut wx, &mut wy, &mut wz),
                (out.0, out.1, out.2),
            );
        } else {
            h_mag_tet4(&elem_nodes, &mvectors[i], targets, (out.0, out.1, out.2));
        }
    }

    Ok(())
}
