//! Low-level math operations

pub mod atan2;
pub mod gradient;
pub mod ln;

pub use atan2::{atan, atan_approx, atan2};
pub use gradient::gradient;
pub use ln::ln;
use std::f64::consts::PI;

/// In place cross-product  
/// $$ \vec{a} \times \vec{b} = \vec{c} $$
#[inline(always)]
pub fn cross(a: &[f64; 3], b: &[f64; 3], c: &mut [f64; 3]) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

/// Custom min/max finding function  
///
/// # Returns:
/// `(min, max)`: the maximum and minimum values in the slice
pub fn min_and_max(arr: &[f64]) -> Option<(f64, f64)> {

    if arr.is_empty() {
        return None;
    }

    let mut min: f64 = arr[0];
    let mut max: f64 = arr[0];

    for a in arr.iter() {
        if *a < min {
            min = *a;
        }
        if *a > max {
            max = *a;
        }
    }
    Some((min, max))
}

/// Sort a slice using pre-computed indices, in place
///
/// # Arguments:
/// - `data`:       slice to be sorted
/// - `scratch`:    pre-allocated buffer
/// - `indices`:    pre-computed index for each value in `data`
pub fn sort_by_indices<T>(data: &mut [T], scratch: &mut [T], indices: &[usize])
where
    T: Copy,
{
    scratch.copy_from_slice(data);
    for (i, &idx) in indices.iter().enumerate() {
        data[i] = scratch[idx];
    }
}

/// Take the vector magnitude (norm) of a slice
#[inline]
pub fn mag(slice: &[f64]) -> f64 {
    let mut sumsq: f64 = 0.0;
    for &value in slice.iter() {
        sumsq += value * value;
    }
    sumsq.sqrt()
}

/// Magnitude of a 3-length vector given separate coordinates
///
/// This uses explicitly fused multiply-add instructions
#[inline]
pub fn mag3(x: f64, y: f64, z: f64) -> f64 {
    x.mul_add(x, y.mul_add(y, z * z)).sqrt()
}

/// Compute the dot product between two 3-length vectors
pub fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the unit vector from a to b
///
/// \vec{u_ab} = \frac{\vec{b} - \vec{a}}{|\vec{b} - \vec{a}|}
#[inline(always)]
pub fn unit_vector(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    let cx = b[0] - a[0];
    let cy = b[1] - a[1];
    let cz = b[2] - a[2];
    let cmag_inv = 1.0 / mag3(cx, cy, cz);

    [cx * cmag_inv, cy * cmag_inv, cz * cmag_inv]
}

/// Compute the distance between two points in 3D space
///
/// # Arguments
/// - `a`: start vector
/// - `b`: end vector
///
/// # Returns
/// magnitude of the distance between the two points
#[inline(always)]
pub fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    mag(&vec_distance(a, b))
}

/// Distance vector from a to b
#[inline(always)]
pub fn vec_distance(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    let mut vec_dist: [f64; 3] = [0.0; 3];
    for i in 0..3 {
        vec_dist[i] = b[i] - a[i];
    }
    vec_dist
}

/// Running average
///
/// # Arguments
/// * `old_quantity`: the number of values that factor into the current average `vector`
/// * `vector`: represents the current average of a number of values
/// * `new_number`: number of new values that are being added to the average
/// * `new_vector`: the average of a number of new values being added to the current average
///
/// # Modifies
/// `vector` with the updated average value of each of its components
///
/// # Returns
/// new number of values that form the average
pub fn running_average(
    old_quantity: u32,
    vector: &mut [f64],
    new_number: u32,
    new_vector: &[f64],
) -> u32 {
    let n: usize = vector.len();
    assert_eq!(n, new_vector.len());

    let new_quantity: u32 = old_quantity + new_number;

    // Compute the new average of each of the components of the vector
    for i in 0..n {
        vector[i] = (old_quantity as f64 * vector[i] + new_number as f64 * new_vector[i])
            / new_quantity as f64;
    }

    new_quantity
}

/// Compute the radius of a sphere given its volume
#[inline]
pub fn sphere_radius(volume: f64) -> f64 {
    ((3.0 / (4.0 * PI)) * volume).cbrt()
}
