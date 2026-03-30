//! Lightning-fast magnetic field calculations using octrees and the Barnes-Hut algorithm  
//!
//! This is the Rust API documentation.
//! Main documentation, including a theory manual and the Python API,
//! is hosted [here](https://retkins.github.io/oersted).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::result_unit_err)]
#![allow(clippy::len_without_is_empty)]

use std::f64::consts::PI;

/// Biot-Savart integration constant:  
/// $$\frac{\mu_0}{4\pi} [H/m]$$
pub const MU0_4PI: f64 = 1e-7;

/// Magnetic permeability of free space:  
/// $$\mu_0 = 4\pi \cdot 10^{-7} H/m$$
pub const MU0: f64 = 4.0 * PI * MU0_4PI;

/// Expressions for various Biot-Savart sources
pub mod sources;

/// Magnetic field calculations
pub mod biotsavart;

/// Analytical expressions for the magnetic field under highly specific conditions
pub mod analytical;

/// Basic read/write file operations
pub mod io;

/// Error types for `oersted`
pub mod errors;

/// Low-level math and array operations
pub mod math;

pub mod mat3;
/// Define a 3-length vector type for math operations
pub mod vec3;

pub mod mesh;

/// Compute Morton codes and related functions
///
/// Notes:
/// 1. Grids, etc. are based on a *cube* in 3d space
pub mod morton;

/// Single-tree Barnes-Hut octree methods
///
/// Used to calculate the effect of M sources on N target points
/// using the Biot-Savart law for magnetic fields
pub mod octree;

/// Python bindings
#[cfg(feature = "python")]
pub mod python;

/// Parallel processing
#[cfg(feature = "parallel")]
pub mod biotsavart_parallel;

/// Older versions of the methods; keep around for testing
pub mod archive;
