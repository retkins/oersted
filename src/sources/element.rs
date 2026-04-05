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
/// This is useful for magnetization sources. See the documentation for derivation.
///
/// # Arguments
/// * `x`, `y`, `z`: (m) coordinates of the target point in the local edge csys
///
/// # Returns
/// Gradient of the edge integral: [de/dx, de/dy, de/dz]
#[inline]
pub fn edge_integral_gradient(x: f64, y: f64, z: f64) -> Vec3 {
    let za = z.abs();
    let sign_z = z.signum();

    let r2 = x.mul_add(x, y.mul_add(y, z * z));
    let r = r2.sqrt();

    let atan_x_over_y = if y.abs() > 1e-8 { atan(x / y) } else { 0.0 };
    let atan_xza_over_yr = if y.abs() > 1e-8 && r > 1e-8 {
        atan(x * za / (y * r))
    } else {
        0.0
    };

    let de_dx = if y.abs() > 1e-8 && r > 1e-8 {
        (r - za) / (x * x + y * y) // note: za, not z
    } else {
        0.0
    };

    // Compute de_dy and de_dz using za everywhere z appeared
    // then multiply de_dz by sign_z at the end
    let de_dy = if y.abs() > 1e-8 {
        let x2 = x * x;
        let y2 = y * y;
        let za2 = za * za;
        y / ((x + r) * r)
            + za * (x / (y2 * (x2 / y2 + 1.0))
                - (x * za / (r2 * r) + x * za / (y2 * r)) / (x2 * za2 / (y2 * r2) + 1.0))
                / y
            + za * (atan_x_over_y - atan_xza_over_yr) / y2
    } else {
        0.0
    };

    let de_dz = if y.abs() > 1e-8 {
        let y2 = y * y;
        let za2 = za * za;
        sign_z
            * ((-y2 * atan_x_over_y + y2 * atan_xza_over_yr + y * za - za2 * atan_x_over_y
                + za2 * atan_xza_over_yr)
                / (y * (y2 + za2)))
    } else {
        0.0
    };

    Vec3([de_dx, de_dy, de_dz])
}

#[cfg(test)]
mod tests {

    use super::*;

    // Test that the edge gradient function compares well to the numerical gradient
    // using the existing scalar function
    #[test]
    fn test_edge_gradient() {
        let eps: f64 = 1e-6; // perturb the output location slightly to calculate gradient numerically
        let x: f64 = 2.0;
        let y: f64 = 2.05;
        let z: f64 = -2.05;
        let de_dx: f64 =
            (edge_integral(x + eps, y, z) - edge_integral(x - eps, y, z)) / (2.0 * eps);
        let de_dy: f64 =
            (edge_integral(x, y + eps, z) - edge_integral(x, y - eps, z)) / (2.0 * eps);
        let de_dz: f64 =
            (edge_integral(x, y, z + eps) - edge_integral(x, y, z - eps)) / (2.0 * eps);

        let de_analytical = edge_integral_gradient(x, y, z);

        assert!((de_dx - de_analytical[0]).abs() < 1e-4);
        assert!((de_dy - de_analytical[1]).abs() < 1e-4);
        assert!((de_dz - de_analytical[2]).abs() < 1e-4);
    }
}
