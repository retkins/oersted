//! Analytical expressions for the magnetic field under highly specific conditions

#![allow(non_snake_case)]

use crate::MU0;

/// Compute the magnetic field along the axis of a current-carrying loop
///
/// # Arguments:
/// - `z`:      (m) target location on the axis (distance from midplane)
/// - `I`:      (A) total current in the loop
/// - `R`:      (m) radius of the loop
///
/// # Returns
/// (T) magnetic flux density (in z-direction)
pub fn bfield_loop_axis(z: f64, I: f64, R: f64) -> f64 {
    let R2: f64 = R * R;
    MU0 * I * R2 / (2.0 * (z * z + R2).powf(1.5))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let z = 0.2;
        let i = 10e6;
        let r = 1.0;
        let b = bfield_loop_axis(z, i, r);
        println!("Bfield on axis: {:.8} T", b);
        assert!((b - 5.92420201).abs() < 1e-6);
    }
}
