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

pub mod analytical;
pub mod archive;
pub mod biotsavart;
pub mod errors;
pub mod io;
pub mod magnetization;
pub mod mat3;
pub mod math;
pub mod mesh;
pub mod morton;
pub mod octree;
pub mod sources;
pub mod types;
pub mod vec3;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "parallel")]
pub mod biotsavart_parallel;
