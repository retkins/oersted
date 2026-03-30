use rayon::prelude::*;
use std::num::NonZeroUsize;
use std::thread::available_parallelism;

use crate::biotsavart::{bfield_direct, hfield_direct_tet, hmag_direct_tet};
use crate::vec3::Vec3;

pub fn get_nthreads(nthreads_requested: u32) -> usize {
    let nthreads_available: usize = available_parallelism().unwrap_or(NonZeroUsize::MIN).get();
    let nthreads: usize =
        if nthreads_requested as usize > nthreads_available || nthreads_requested == 0 {
            nthreads_available
        } else {
            nthreads_requested as usize
        };
    nthreads
}

/// Calculate magnetic flux density using direct biot-savart law integration
///
/// This version of the function uses a user-specified number of threads
///
/// # Arguments
/// - `centx`, `centy`, `centz`: (m) locations of source element centroids in 3D space
/// - `vol`:                     (m^3) volume of each source element
/// - `jx`, `jy`, `jz`:          (A/m^2) current density vector of each source element
/// - `x`, `y`, `z`:             (m) location of each target point
/// - `bx`, `by`, `bz`:          (T) magnetic flux density at each target point
/// - `nthreads_requested`:      how many OS threads the calculation should run on
pub fn bfield_direct_parallel(
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
    nthreads_requested: u32,
) -> Result<(), ()> {
    // TODO: length checks
    let n: usize = x.len();
    let nthreads: usize = get_nthreads(nthreads_requested);
    let chunk_size: usize = (n / nthreads).max(1);

    // chunk the inputs
    let _x = x.par_chunks(chunk_size);
    let _y = y.par_chunks(chunk_size);
    let _z = z.par_chunks(chunk_size);
    let _bx = bx.par_chunks_mut(chunk_size);
    let _by = by.par_chunks_mut(chunk_size);
    let _bz = bz.par_chunks_mut(chunk_size);

    (_x, _y, _z, _bx, _by, _bz)
        .into_par_iter()
        .try_for_each(|(_x, _y, _z, _bx, _by, _bz)| {
            bfield_direct(
                centx, centy, centz, vol, jx, jy, jz, _x, _y, _z, _bx, _by, _bz,
            )
        })?;

    Ok(())
}

pub fn hfield_direct_tet_parallel(
    nodes: &[f64],
    connectivity: &[u32],
    jdensity: &[f64],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    hx: &mut [f64],
    hy: &mut [f64],
    hz: &mut [f64],
    nthreads_requested: u32,
) -> Result<(), ()> {
    // TODO: length checks
    let n: usize = x.len();
    let nthreads: usize = get_nthreads(nthreads_requested);
    let chunk_size: usize = (n / nthreads).max(1);

    // chunk the inputs
    let _x = x.par_chunks(chunk_size);
    let _y = y.par_chunks(chunk_size);
    let _z = z.par_chunks(chunk_size);
    let _hx = hx.par_chunks_mut(chunk_size);
    let _hy = hy.par_chunks_mut(chunk_size);
    let _hz = hz.par_chunks_mut(chunk_size);

    (_x, _y, _z, _hx, _hy, _hz)
        .into_par_iter()
        .try_for_each(|(_x, _y, _z, _hx, _hy, _hz)| {
            hfield_direct_tet(nodes, connectivity, jdensity, _x, _y, _z, _hx, _hy, _hz)
        })?;

    Ok(())
}

pub fn hmag_direct_tet_parallel(
    source_nodes: (&[f64], &[f64], &[f64]),
    source_element_connectivity: &[[u32; 4]],
    source_mvectors: &[Vec3],
    target_nodes: (&[f64], &[f64], &[f64]),
    target_element_connectivity: &[[u32; 4]],
    hx: &mut [f64],
    hy: &mut [f64],
    hz: &mut [f64],
    nthreads_requested: u32,
) -> Result<(), ()> {
    // TODO: length checks
    let n: usize = target_nodes.0.len();
    let nthreads: usize = get_nthreads(nthreads_requested);
    let chunk_size: usize = (n / nthreads).max(1);

    // chunk the inputs over target elements
    let _tec = target_element_connectivity.par_chunks(chunk_size);
    let _hx = hx.par_chunks_mut(chunk_size);
    let _hy = hy.par_chunks_mut(chunk_size);
    let _hz = hz.par_chunks_mut(chunk_size);

    (_tec, _hx, _hy, _hz)
        .into_par_iter()
        .try_for_each(|(_tec, _hx, _hy, _hz)| {
            hmag_direct_tet(
                source_nodes,
                source_element_connectivity,
                source_mvectors,
                target_nodes,
                &_tec,
                _hx,
                _hy,
                _hz,
            )
        })?;
    Ok(())
}
