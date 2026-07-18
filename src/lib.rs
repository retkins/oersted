#![cfg_attr(docsrs, feature(doc_cfg))]
//! Lightning-fast magnetic field calculations using octrees and the Barnes-Hut algorithm  
//!
//! This is the Rust API documentation.
//! Main documentation, including a theory manual and the Python API,
//! is hosted [here](https://retkins.github.io/oersted).

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::result_unit_err)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::type_complexity)]

use std::{f64::consts::PI, num::NonZeroUsize, thread::available_parallelism};

/// Biot-Savart integration constant:  
/// $$\frac{\mu_0}{4\pi} [H/m]$$
pub const MU0_4PI: f64 = 1e-7;

/// Magnetic permeability of free space:  
/// $$\mu_0 = 4\pi \cdot 10^{-7} H/m$$
pub const MU0: f64 = 4.0 * PI * MU0_4PI;

pub const INV_4PI: f64 = 1.0 / (4.0 * PI);

pub mod analytical;
// pub mod archive;
pub mod biotsavart;
pub mod errors;
pub mod io;
pub mod magnetization;
pub mod math;
pub mod mesh;
pub mod morton;
pub mod octree;
pub mod sources;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

// Utility functions used across the library

/// Determine the number of cpu threads to use
///
/// If `nthreads_requested` is 0, then use all available threads. Otherwise, use the
/// specified value, but no more than is available.
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

/// Chunk targets for parallel evaluation
pub fn par_chunks<'a>(
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
    fx: &'a mut [f64],
    fy: &'a mut [f64],
    fz: &'a mut [f64],
    chunk_size: usize,
) -> impl Iterator<
    Item = (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
    ),
> {
    let _ = check_lengths!(x, y, z, fx, fy, fz);
    x.chunks(chunk_size)
        .zip(y.chunks(chunk_size))
        .zip(z.chunks(chunk_size))
        .zip(fx.chunks_mut(chunk_size))
        .zip(fy.chunks_mut(chunk_size))
        .zip(fz.chunks_mut(chunk_size))
        .map(|(((((xc, yc), zc), fxc), fyc), fzc)| (xc, yc, zc, fxc, fyc, fzc))
}

/// Check the lengths on an arbitrary number of vectors
#[macro_export]
macro_rules! check_lengths {
    ($first:expr $(, $rest:expr)*) => {{
        let n = $first.len();
        $(
            assert_eq!(
                $rest.len(), n,
                // Simplified error message required to enable LLVM to do bounds checks
                concat!("length of `", stringify!($rest), "` does not match")
            );
        )*
        n
    }};
}

/// Check the lengths on an arbitrary number of optional vectors with a provided
/// common length
#[macro_export]
macro_rules! check_optional_lengths {
    ($common_length:expr $(, $rest:expr)*) => {{
        let n = $common_length;
        $(
            if let Some(r) = $rest.as_ref() {
                assert_eq!(
                    r.len(),
                    n,
                    // Simplified error message required to enable LLVM to do bounds checks
                    concat!("length of `", stringify!($rest), "` does not match")
                );
            }
        )*
    }};
}
