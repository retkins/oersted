use crate::{
    math::{atan, ln, mag3, unit_vector},
    types::Vec3,
};

/// Construct a local coordinate system on a polyhedron edge
#[inline]
pub fn edge_csys(node1: &Vec3, node2: &Vec3, node3: &Vec3) -> (Vec3, Vec3, Vec3) {
    let xp_hat = Vec3(unit_vector(node2.to_slice(), node1.to_slice()));
    let yp_hat_temp = unit_vector(node3.to_slice(), node1.to_slice());
    let mut zp_hat = xp_hat.cross(&Vec3(yp_hat_temp));
    let norm_inv = 1.0 / zp_hat.mag();
    zp_hat *= norm_inv;
    let yp_hat = xp_hat.cross(&zp_hat);
    (xp_hat, yp_hat, zp_hat)
}

// Transform a global xyz position into a local coordinate system
#[inline]
pub fn transform(g: &Vec3, xhat: &Vec3, yhat: &Vec3, zhat: &Vec3) -> Vec3 {
    Vec3([xhat.dot(g), yhat.dot(g), zhat.dot(g)])
}

/// Compute the definite edge integral of a finite element given the coordinates
/// of the target point in the edge reference frame.
///
/// Uses inlined/vectorizable approximate functions for `ln()` and `atan2()`.
/// `mag3()` is also inlined and uses fma instructions.
///
/// # Arguments
/// * `x`, `y`, `z`: (m) coordinates of the target point in the edge reference frame
///
/// # Returns
/// Integral along the polyhedron edge
#[inline]
pub fn edge_integral(x: f64, y: f64, z: f64) -> f64 {
    let r = mag3(x, y, z);
    let x_plus_r = x + r;
    let ln_x_plus_r = if x_plus_r.abs() > 1e-8 {
        ln(x_plus_r)
    } else {
        0.0
    };
    let y_inv = 1.0 / y;
    let za = z.abs();
    let za_over_y = if y.abs() > 1e-12 * r { za * y_inv } else { 0.0 };
    let atan_1 = if y.abs() > 1e-12 * r && r > 1e-8 {
        atan(x * za_over_y / r)
    } else {
        0.0
    };
    let atan_2 = if y.abs() > 1e-12 * r {
        atan(x * y_inv)
    } else {
        0.0
    };

    ln_x_plus_r + za_over_y * (atan_1 - atan_2)
}

/// Compute the gradient of the edge integral
///
/// This is useful for magnetization sources
/// 
/// # Arguments
/// * `x`, `y`, `z`: (m) coordinates of the target point in the local edge csys
/// 
/// # Returns 
/// Gradient of the edge integral: [de/dx, de/dy, de/dz]
pub fn edge_integral_gradient(x: f64, y: f64, z: f64) -> Vec3 {
    let r2: f64 = x.mul_add(x, y.mul_add(y, z * z));
    let r: f64 = r2.sqrt();
    let atan_x_over_y: f64 = if y.abs() > 1e-8 { atan(x / y) } else { 0.0 };
    let atan_xz_over_yr: f64 = if y.abs() > 1e-8 && r > 1e-8 {
        atan(x * z / (y * r))
    } else {
        0.0
    };

    let de_dx: f64 = if y.abs() > 1e-8 && r.abs() > 1e-8 {
        (r2 - z * r) / ((x * x + y * y) * r)
    } else {
        0.0
    };
    let de_dy: f64 = if y.abs() > 1e-8 {
        y / ((x + r) * r)
            + z * (x / (y * y * (x * x / (y * y) + 1.0))
                - (x * z / (r2 * r) + x * z / (y * y * r)) / (x * x * z * z / (y * y * (r2)) + 1.0))
                / y
            + z * (atan_x_over_y - atan_xz_over_yr) / (y * y)
    } else {
        0.0
    };
    let de_dz: f64 = if y.abs() > 1e-8 {
        (-y * y * atan_x_over_y + y * y * atan_xz_over_yr + y * z - z * z * atan_x_over_y
            + z * z * atan_xz_over_yr)
            / (y * (y * y + z * z))
    } else {
        0.0
    };

    Vec3([de_dx, de_dy, de_dz])
}
