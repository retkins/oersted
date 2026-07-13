//! Magnetic field calculations

use crate::{
    biotsavart::SourceVectors::{CurrentDensity, Magnetization},
    get_nthreads, par_chunks,
    sources::{
        a_current_point, a_current_tet4, a_mag_point, a_mag_tet4, h_current_point, h_current_tet4,
        h_mag_point, h_mag_tet4,
    },
    types::Vec3,
};
use std::thread;

#[derive(Clone, Copy)]
pub enum IntegrationMethod {
    Element,
    Point,
}

pub enum SourceVectors<'a> {
    CurrentDensity(&'a [Vec3]),
    Magnetization(&'a [Vec3]),
}

#[derive(Clone, Copy)]
pub enum RequestedField {
    AField,
    HField,
}

// An evaluation kernel, which is the same over all combinations of requested output
// field, source vector type, and integration method
type Kernel = fn(&[Vec3; 4], &Vec3, (&[f64], &[f64], &[f64]), (&mut [f64], &mut [f64], &mut [f64]));

/// Compute the requested output field at target points, using a 4-node tetrahedral
/// mesh as the source and the provided solution settings
///
/// This function is parallelized over target points. Responsibility for zeroing the
/// `out` arrays is on the caller.
pub fn calculate_fields(
    src_nodes: &[Vec3],
    src_connectivity: &[[u32; 4]],
    src_vectors: SourceVectors,
    requested_field: RequestedField,
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
    method: IntegrationMethod,
    n_threads_requested: u32,
) {
    let kernel: Kernel = match (requested_field, &src_vectors, method) {
        (RequestedField::AField, SourceVectors::CurrentDensity(_), IntegrationMethod::Element) => {
            a_current_tet4
        }
        (RequestedField::AField, SourceVectors::CurrentDensity(_), IntegrationMethod::Point) => {
            a_current_point
        }
        (RequestedField::AField, SourceVectors::Magnetization(_), IntegrationMethod::Element) => {
            a_mag_tet4
        }
        (RequestedField::AField, SourceVectors::Magnetization(_), IntegrationMethod::Point) => {
            a_mag_point
        }
        (RequestedField::HField, SourceVectors::CurrentDensity(_), IntegrationMethod::Element) => {
            h_current_tet4
        }
        (RequestedField::HField, SourceVectors::CurrentDensity(_), IntegrationMethod::Point) => {
            h_current_point
        }
        (RequestedField::HField, SourceVectors::Magnetization(_), IntegrationMethod::Element) => {
            h_mag_tet4
        }
        (RequestedField::HField, SourceVectors::Magnetization(_), IntegrationMethod::Point) => {
            h_mag_point
        }
    };

    let vectors = match src_vectors {
        CurrentDensity(j) => j,
        Magnetization(m) => m,
    };

    let n_threads: usize = get_nthreads(n_threads_requested);
    let n_targets: usize = targets.0.len();
    let chunk_size: usize = n_targets.div_ceil(n_threads);

    let (x, y, z) = targets;
    let (fx, fy, fz) = out;

    let chunks = par_chunks(x, y, z, fx, fy, fz, chunk_size);

    thread::scope(|s| {
        for (xc, yc, zc, fxc, fyc, fzc) in chunks {
            s.spawn(move || {
                for (i, elem) in src_connectivity.iter().enumerate() {
                    let elem_nodes = [
                        src_nodes[elem[0] as usize],
                        src_nodes[elem[1] as usize],
                        src_nodes[elem[2] as usize],
                        src_nodes[elem[3] as usize],
                    ];

                    // Compute the effect of an individual element on all targets
                    kernel(&elem_nodes, &vectors[i], (xc, yc, zc), (fxc, fyc, fzc));
                }
            });
        }
    });
}

/// Compute the magnetic vector potential at a collection of target points
///
/// # Args
/// * `src_nodes`: (m) x,y,z coordinates of all nodes in the source mesh
/// * `src_connectivity`: indices into `src_nodes` defining the nodes associated with
///   each element
/// * `src_vectors`: source vector, corresponding to the excitation at each element
/// * `targets`: (m) x,y,z coordinates of target points
/// * `out`: (T*m) magnetic vector potential at each target point
/// * `method`: choose either full element integration or point approximation
/// * `n_threads_requested`: set equal to 0 to use all available parallelism, or specify
///   a number of cpu cores to use
///
/// # Solver
/// This performs a "direct" O(N^2) integration of the effect of every source at every
/// target point. This function is parallelized over target points
///
/// # Note
/// This function accumulates into `out`. Responsibility for properly zeroing this
/// parameter is the caller's.
pub fn a_field(
    src_nodes: &[Vec3],
    src_connectivity: &[[u32; 4]],
    src_vectors: SourceVectors,
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
    method: IntegrationMethod,
    n_threads_requested: u32,
) {
    calculate_fields(
        src_nodes,
        src_connectivity,
        src_vectors,
        RequestedField::AField,
        targets,
        out,
        method,
        n_threads_requested,
    );
}

/// Compute the magnetic field strength at a collection of target points
///
/// # Args
/// * `src_nodes`: (m) x,y,z coordinates of all nodes in the source mesh
/// * `src_connectivity`: indices into `src_nodes` defining the nodes associated with
///   each element
/// * `src_vectors`: source vector, corresponding to the excitation at each element
/// * `targets`: (m) x,y,z coordinates of target points
/// * `out`: (T/m) magnetic field strength at each target point
/// * `method`: choose either full element integration or point approximation
/// * `n_threads_requested`: set equal to 0 to use all available parallelism, or specify
///   a number of cpu cores to use
///
/// # Solver
/// This performs a "direct" O(N^2) integration of the effect of every source at every
/// target point. This function is parallelized over target points
///
/// # Note
/// This function accumulates into `out`. Responsibility for properly zeroing this
/// parameter is the caller's.
pub fn h_field(
    src_nodes: &[Vec3],
    src_connectivity: &[[u32; 4]],
    src_vectors: SourceVectors,
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
    method: IntegrationMethod,
    n_threads_requested: u32,
) {
    calculate_fields(
        src_nodes,
        src_connectivity,
        src_vectors,
        RequestedField::HField,
        targets,
        out,
        method,
        n_threads_requested,
    );
}
