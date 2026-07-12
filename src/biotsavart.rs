//! Magnetic field calculations

#![allow(non_snake_case)]

use crate::{
    get_nthreads,
    math::mag3,
    mesh::node_coords,
    sources::{
        a_current_point, a_current_tet4, h_current_tet4, h_current_tet4_edge, h_mag_tet4,
        h_mag_tet4_edge, h_point_dipole,
    },
    types::Vec3,
};
use std::{f64::consts::PI, thread};

const ONE_OVER_4PI: f64 = 1.0 / (4.0 * PI);

/// Compute the magnetic vector potential at a collection of target points, using a
/// collection of "point" sources with associated constant current density vectors as
/// the source
///
/// # Args
/// * `src_centroids`: (m) x,y,z coordinates of the centroid locations of the sources
/// * `src_volumes`: (m^3) volume associated with each source
/// * `src_jdensity`: (A/m^2) current density vector associated with each source
/// * `targets`: (m) x,y,z coordinates of the target points
/// * `out`: (T*m) magnetic vector potential at the target points
/// * `n_threads_requested`: set equal to 0 to use all available cpu cores
///
/// # Accuracy
/// This function does not handle singularities at the source points, and has reduced
/// accuracy near the sources. Therefore, it is only suitable for use as a far-field
/// approximation. The approximation is quite good for targets 2-3 source lengths away.
///
/// # Solver
/// This performs a "direct" O(N^2) integration of the effect of every source at every
/// target point.
pub fn a_current_point_direct(
    src_centroids: &[Vec3],
    src_volumes: &[f64],
    src_jdensity: &[Vec3],
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
    n_threads_requested: u32,
) {
    let n_threads: usize = get_nthreads(n_threads_requested);
    let n_targets: usize = targets.0.len();
    let n_src: usize = src_centroids.len();
    let chunk_size: usize = n_targets.div_ceil(n_threads);

    let (x, y, z) = targets;
    let (ax, ay, az) = out;

    let chunks = x
        .chunks(chunk_size)
        .zip(y.chunks(chunk_size))
        .zip(z.chunks(chunk_size))
        .zip(ax.chunks_mut(chunk_size))
        .zip(ay.chunks_mut(chunk_size))
        .zip(az.chunks_mut(chunk_size));

    thread::scope(|s| {
        for (((((xc, yc), zc), axc), ayc), azc) in chunks {
            s.spawn(move || {
                for i in 0..n_src {
                    let jmoment: Vec3 = src_jdensity[i] * src_volumes[i];
                    // Compute the effect of an individual source on all targets
                    a_current_point(&src_centroids[i], &jmoment, (xc, yc, zc), (axc, ayc, azc));
                }
            });
        }
    });
}

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

/// Compute the magnetic vector potential at a collection of target points, using a
/// collection of 4-node tetrahedral elements with constant current density vectors
/// as the sources
///
/// This function is parallelized over target points
///
/// # Args
/// * `src_nodes`: (m) x,y,z coordinates of all nodes in the source mesh
/// * `src_connectivity`: indices into `src_nodes` defining the nodes associated with
///   each element
/// * `src_jdensity`: (A/m^2) current density vector associated with the centroid of
///   each element
/// * `targets`: (m) x,y,z coordinates of target points
/// * `out`: (T*m) magnetic vector potential at each target point
/// * `n_threads_requested`: set equal to 0 to use all available parallelism, or specify
///   a number of cpu cores to use
///
/// # Note
/// This function accumulates into `out`. Responsibility for properly zeroing this
/// parameter is the caller's.
pub fn a_current_tet4_direct(
    src_nodes: &[Vec3],
    src_connectivity: &[[u32; 4]],
    src_jdensity: &[Vec3],
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
    n_threads_requested: u32,
) {
    let n_threads: usize = get_nthreads(n_threads_requested);
    let n_targets: usize = targets.0.len();
    let chunk_size: usize = n_targets.div_ceil(n_threads);

    let (x, y, z) = targets;
    let (ax, ay, az) = out;

    let chunks = x
        .chunks(chunk_size)
        .zip(y.chunks(chunk_size))
        .zip(z.chunks(chunk_size))
        .zip(ax.chunks_mut(chunk_size))
        .zip(ay.chunks_mut(chunk_size))
        .zip(az.chunks_mut(chunk_size));

    thread::scope(|s| {
        for (((((xc, yc), zc), axc), ayc), azc) in chunks {
            s.spawn(move || {
                for (i, elem) in src_connectivity.iter().enumerate() {
                    let elem_nodes = [
                        src_nodes[elem[0] as usize],
                        src_nodes[elem[1] as usize],
                        src_nodes[elem[2] as usize],
                        src_nodes[elem[3] as usize],
                    ];

                    // Compute the effect of an individual element on all targets
                    a_current_tet4(&elem_nodes, &src_jdensity[i], (xc, yc, zc), (axc, ayc, azc));
                }
            });
        }
    });
}
