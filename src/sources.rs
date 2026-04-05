//! Expressions for various Biot-Savart sources

use crate::math::cross;

pub mod element;
pub use element::{edge_csys, edge_integral, transform};
pub mod point;
pub use point::{h_point, h_point_dipole};
pub mod tet4;
pub use tet4::{h_field_tet4, h_mag_tet4};
pub mod hex8;
pub use hex8::bfield_hexahedron;

/// Monopole approximation of the effect of a group of source points
/// Approximates the group of sources as a single 'super source'
///
/// Returns:
///     (monopole centroid, current density moment)
///
pub fn monopole(
    centroids: (&[f64], &[f64], &[f64]),
    volumes: &[f64],
    jdensity: (&[f64], &[f64], &[f64]),
) -> ((f64, f64, f64), (f64, f64, f64)) {
    // TODO: length-check guards
    let n: usize = volumes.len();

    let mut mx = vec![0.0; n];
    let mut my = vec![0.0; n];
    let mut mz = vec![0.0; n];
    let mut mnet = vec![0.0; n];
    let (x, y, z) = centroids;
    let (jx, jy, jz) = jdensity;

    // Compute current density moments for each of the individual source points
    for i in 0..n {
        mx[i] = volumes[i] * jx[i];
        my[i] = volumes[i] * jy[i];
        mz[i] = volumes[i] * jz[i];
        mnet[i] = (mx[i].powi(2) + my[i].powi(2) + mz[i].powi(2)).sqrt();
    }

    // Compute moment-weighted centroid location
    let mx_total: f64 = mx.iter().sum();
    let my_total: f64 = my.iter().sum();
    let mz_total: f64 = mz.iter().sum();

    let mnet_total: f64 = (mx_total * mx_total + my_total * my_total + mz_total * mz_total).sqrt();
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for i in 0..n {
        cx += x[i] * mnet[i];
        cy += y[i] * mnet[i];
        cz += z[i] * mnet[i];
    }
    cx /= mnet_total;
    cy /= mnet_total;
    cz /= mnet_total;

    ((cx, cy, cz), (mx_total, my_total, mz_total))
}

/// Computes the magnetic dipole of a collection of source points in 3D space
///
/// $$ m = 1/2 \cdot \int\int\int{r' \times j(r') dV} $$
pub fn dipole(
    centroids: (&[f64], &[f64], &[f64]),
    volumes: &[f64],
    jdensity: (&[f64], &[f64], &[f64]),
) -> (f64, f64, f64) {
    let n: usize = centroids.0.len();
    // The monopole calculation provides the centroid of the source point distribution
    let ((cx, cy, cz), _) = monopole(centroids, volumes, jdensity);

    let mut moment: (f64, f64, f64) = (0.0, 0.0, 0.0);
    let mut rxj = [0.0; 3];

    // Compute the position vector
    for i in 0..n {
        let rpx: f64 = centroids.0[i] - cx;
        let rpy: f64 = centroids.1[i] - cy;
        let rpz: f64 = centroids.2[i] - cz;
        let jx: f64 = jdensity.0[i];
        let jy: f64 = jdensity.1[i];
        let jz: f64 = jdensity.2[i];
        cross(&[rpx, rpy, rpz], &[jx, jy, jz], &mut rxj);
        moment.0 += rxj[0];
        moment.1 += rxj[1];
        moment.2 += rxj[2];
    }
    moment.0 *= 0.5;
    moment.1 *= 0.5;
    moment.2 *= 0.5;

    moment
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_monopole() {
        // if the sources are in the same position, this should be a
        // simple sum
        let n: usize = 5;
        let cx = vec![1.0; n];
        let cy = vec![2.0; n];
        let cz = vec![3.0; n];
        let volumes = vec![1e-3; n];
        let jx = vec![1e8; n];
        let jy = vec![0.0; n];
        let jz = vec![0.0; n];
        let ((x, y, z), (mx, my, mz)) = monopole((&cx, &cy, &cz), &volumes, (&jx, &jy, &jz));

        assert_eq!(x, cx[0]);
        assert_eq!(y, cy[0]);
        assert_eq!(z, cz[0]);
        let mx_expected: f64 = jx.iter().zip(volumes).map(|(_jx, _v)| _jx * _v).sum();
        assert_eq!(mx_expected, mx);
        assert_eq!(0.0, my);
        assert_eq!(0.0, mz);
    }

    #[test]
    fn test_dipole() {
        let cx = [1.0, 0.0, 2.0, -1.0];
        let cy = [-1.0, 2.0, 0.0, 1.0];
        let cz = [0.0, 0.0, 0.0, 0.0];
        let jx = [0.0, 0.0, 0.0, 0.0];
        let jy = [1e7, 1e7, 1e7, 1e7];
        let jz = [0.0, 0.0, 0.0, 0.0];

        let volumes = [1e-4, 1e-4, 1e-4, 1e-4];
        let (_, mm) = monopole((&cx, &cy, &cz), &volumes, (&jx, &jy, &jz));
        let dm = dipole((&cx, &cy, &cz), &volumes, (&jx, &jy, &jz));

        println!("Monopole = ({:.6}, {:.6}, {:.6})", mm.0, mm.1, mm.2);
        println!("Dipole   = ({:.6}, {:.6}, {:.6})", dm.0, dm.1, dm.2);
    }
}
