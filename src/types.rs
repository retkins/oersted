//! Custom data types used throughout the library

use crate::vec3::Vec3;
use std::slice::{from_raw_parts, from_raw_parts_mut};

/// Convert a flat array representing an (N,3) matrix of f64's in row-major form
///
/// This function does not require a data copy.
///
/// # Unsafe
/// This function has an unsafe call, which is proven safe at runtime by checking that
/// the length of the input data is correct. If the length of input data is not correct,
/// this function will panic.
pub fn to_vec3s(flat: &[f64]) -> &[Vec3] {
    // We must determine that the input array has N 3-length vectors within it; otherwise
    // we will have a memory error
    assert_eq!(flat.len() % 3, 0);

    let n_vecs: usize = flat.len() / 3;

    unsafe { from_raw_parts(flat.as_ptr() as *const Vec3, n_vecs) }
}

/// Convert a flat array representing an (N,3) matrix of f64's in row-major form
///
/// This function does not require a data copy.
///
/// # Unsafe
/// This function has an unsafe call, which is proven safe at runtime by checking that
/// the length of the input data is correct. If the length of input data is not correct,
/// this function will panic.
pub fn to_vec3s_mut(flat: &mut [f64]) -> &mut [Vec3] {
    // We must determine that the input array has N 3-length vectors within it; otherwise
    // we will have a memory error
    assert_eq!(flat.len() % 3, 0);

    let n_vecs: usize = flat.len() / 3;

    unsafe { from_raw_parts_mut(flat.as_mut_ptr() as *mut Vec3, n_vecs) }
}

/// Convert a flat array representing an (N,4) matrix of u32's in row-major form
///
/// This function does not require a data copy.
///
/// # Unsafe
/// This function has an unsafe call, which is proven safe at runtime by checking that
/// the length of the input data is correct. If the length of input data is not correct,
/// this function will panic.
pub fn to_u32x4s(flat: &[u32]) -> &[[u32; 4]] {
    // We must determine that the input array has N 4-length vectors within it; otherwise
    // we will have a memory error
    assert_eq!(flat.len() % 4, 0);

    let n_vecs: usize = flat.len() / 4;

    unsafe { from_raw_parts(flat.as_ptr() as *const [u32; 4], n_vecs) }
}
