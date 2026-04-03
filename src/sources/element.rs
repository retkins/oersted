use crate::{
    math::{atan, ln, mag3, unit_vector},
    types::Vec3,
};

// Compute the indefinite edge integral of a finite element given the coordinates
// of the target point in the edge reference frame.
// Uses inlined/vectorizable approximate functions for `ln()` and `atan2()`.
// `mag3()` is also inlined and uses fma instructions.
#[inline(always)]
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

#[inline(always)]
pub fn edge_csys(node1: &Vec3, node2: &Vec3, node3: &Vec3) -> (Vec3, Vec3, Vec3) {
    let xp_hat = Vec3(unit_vector(node2.to_slice(), node1.to_slice()));
    let yp_hat_temp = unit_vector(node3.to_slice(), node1.to_slice());
    let mut zp_hat = xp_hat.cross(&Vec3(yp_hat_temp));
    let norm_inv = 1.0 / zp_hat.mag();
    zp_hat *= norm_inv;
    let yp_hat = xp_hat.cross(&zp_hat);
    (xp_hat, yp_hat, zp_hat)
}

// transform a global xyz position into a local coordinate system
#[inline(always)]
pub fn transform(g: &Vec3, xhat: &Vec3, yhat: &Vec3, zhat: &Vec3) -> Vec3 {
    Vec3([xhat.dot(g), yhat.dot(g), zhat.dot(g)])
}
