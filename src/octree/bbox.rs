use crate::{math::min_and_max, types::Vec3};

/// Determines the location and extent of a collection of source points
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    xc: f64,
    yc: f64,
    zc: f64,
    pub side_length: f64,
    xbounds: (f64, f64),
    ybounds: (f64, f64),
    zbounds: (f64, f64),
}

// Get the maximum side length of a bounding box cube that encloses all x,y,z bounds
fn side_length_from_bounds(xbounds: (f64, f64), ybounds: (f64, f64), zbounds: (f64, f64)) -> f64 {
    let xrange: f64 = xbounds.1 - xbounds.0;
    let yrange: f64 = ybounds.1 - ybounds.0;
    let zrange: f64 = zbounds.1 - zbounds.0;

    let xymax = match xrange > yrange {
        true => xrange,
        false => yrange,
    };

    match xymax > zrange {
        true => xymax,
        false => zrange,
    }
}

impl BoundingBox {
    pub fn from_centroids(centroids: (&[f64], &[f64], &[f64])) -> Option<Self> {
        // TODO: length check

        let xbounds: Option<(f64, f64)> = min_and_max(centroids.0);
        let ybounds: Option<(f64, f64)> = min_and_max(centroids.1);
        let zbounds: Option<(f64, f64)> = min_and_max(centroids.2);

        if xbounds.is_none() | ybounds.is_none() | zbounds.is_none() {
            return None;
        }

        let (xb, yb, zb) = (xbounds.unwrap(), ybounds.unwrap(), zbounds.unwrap());
        let side_length = side_length_from_bounds(xb, yb, zb);
        let xc: f64 = xb.0 + 0.5 * side_length;
        let yc: f64 = yb.0 + 0.5 * side_length;
        let zc: f64 = zb.0 + 0.5 * side_length;

        Some(Self {
            xc,
            yc,
            zc,
            side_length,
            xbounds: xb,
            ybounds: yb,
            zbounds: zb,
        })
    }

    /// TODO: fix this so there's no data copy
    pub fn from_centroids_vec(centroids: &[Vec3]) -> Self {
        let n: usize = centroids.len();
        let mut x: Vec<f64> = vec![0.0; n];
        let mut y: Vec<f64> = vec![0.0; n];
        let mut z: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            x[i] = centroids[i][0];
            y[i] = centroids[i][1];
            z[i] = centroids[i][2];
        }

        Self::from_centroids((&x, &y, &z)).unwrap()
    }

    pub fn min_corner(&self) -> (f64, f64, f64) {
        (self.xbounds.0, self.ybounds.0, self.zbounds.0)
    }

    pub fn centroid(&self) -> Vec3 {
        Vec3([self.xc, self.yc, self.zc])
    }
}
