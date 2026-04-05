use crate::{
    math::sort_by_indices,
    morton,
    octree::{BoundingBox, CurrentSources, DipoleSources, HFieldSolver, Sources},
    sources::{h_point, h_point_dipole},
    types::Vec3,
};
use std::f64::consts::PI;

pub struct PointSources {
    pub xg: Vec<f64>,
    pub yg: Vec<f64>,
    pub zg: Vec<f64>,
    pub r: Vec<f64>,
    pub vjx: Vec<f64>,
    pub vjy: Vec<f64>,
    pub vjz: Vec<f64>,
    bbox: BoundingBox,
}

impl PointSources {
    pub fn new(
        centroids: (&[f64], &[f64], &[f64]),
        vol: &[f64],
        jdensity: (&[f64], &[f64], &[f64]),
    ) -> Self {
        let (x, y, z) = centroids;
        let (jx, jy, jz) = jdensity;
        let n = x.len();

        let xg: Vec<f64> = x.to_vec();
        let yg: Vec<f64> = y.to_vec();
        let zg: Vec<f64> = z.to_vec();
        let mut r = vec![0.0; n];
        let mut vjx = vec![0.0; n];
        let mut vjy = vec![0.0; n];
        let mut vjz = vec![0.0; n];

        // compute radius of each point and current density moment
        for i in 0..n {
            r[i] = ((3.0 / (4.0 * PI)) * vol[i]).powf(1.0 / 3.0);
            vjx[i] = vol[i] * jx[i];
            vjy[i] = vol[i] * jy[i];
            vjz[i] = vol[i] * jz[i];
        }
        let bbox = BoundingBox::from_centroids((&xg, &yg, &zg)).unwrap();

        Self {
            xg,
            yg,
            zg,
            r,
            vjx,
            vjy,
            vjz,
            bbox,
        }
    }

    pub fn new_dipole(
        x: &[f64],
        y: &[f64],
        z: &[f64],
        vol: &[f64],
        mx: &[f64],
        my: &[f64],
        mz: &[f64],
    ) -> Self {
        let n = x.len();
        let xg: Vec<f64> = x.to_vec();
        let yg: Vec<f64> = y.to_vec();
        let zg: Vec<f64> = z.to_vec();
        let mut r = vec![0.0; n];
        let vjx = mx.to_vec();
        let vjy = my.to_vec();
        let vjz = mz.to_vec();

        // compute radius of each point and current density moment
        for i in 0..n {
            r[i] = ((3.0 / (4.0 * PI)) * vol[i]).powf(1.0 / 3.0);
        }
        let bbox = BoundingBox::from_centroids((&xg, &yg, &zg)).unwrap();

        Self {
            xg,
            yg,
            zg,
            r,
            vjx,
            vjy,
            vjz,
            bbox,
        }
    }
}

impl Sources for PointSources {
    fn len(&self) -> usize {
        self.xg.len()
    }

    fn centroid(&self, i: usize) -> Vec3 {
        Vec3([self.xg[i], self.yg[i], self.zg[i]])
    }

    fn moment(&self, i: usize) -> Vec3 {
        Vec3([self.vjx[i], self.vjy[i], self.vjz[i]])
    }

    fn sort(&mut self, indices: &[usize]) {
        let mut scratch = vec![0.0; self.len()];
        sort_by_indices(&mut self.xg, &mut scratch, indices);
        sort_by_indices(&mut self.yg, &mut scratch, indices);
        sort_by_indices(&mut self.zg, &mut scratch, indices);
        sort_by_indices(&mut self.r, &mut scratch, indices);
        sort_by_indices(&mut self.vjx, &mut scratch, indices);
        sort_by_indices(&mut self.vjy, &mut scratch, indices);
        sort_by_indices(&mut self.vjz, &mut scratch, indices);
    }

    fn bbox(&self) -> &BoundingBox {
        &self.bbox
    }

    fn encode(&mut self, max_depth: u8) -> (&BoundingBox, Vec<u64>) {
        let n = self.len();
        let mut codes: Vec<u64> = Vec::with_capacity(n);
        let bbox = self.bbox();
        let scale: f64 = morton::calculate_scale_factor(max_depth as u32);
        let min_corner: (f64, f64, f64) = bbox.min_corner();

        for i in 0..n {
            let pt: (f64, f64, f64) = (self.xg[i], self.yg[i], self.zg[i]);
            codes.push(morton::encode(pt, scale, bbox.side_length, min_corner));
        }

        (bbox, codes)
    }
}

impl HFieldSolver for CurrentSources<PointSources> {
    fn h_field_branch(&self, centroid: &Vec3, vj: &Vec3, target: &Vec3) -> Vec3 {
        let radius = 0.0;
        h_point(centroid, vj, radius, target)
    }

    fn h_field_leaf(&self, start: usize, end: usize, target: &Vec3) -> Vec3 {
        let mut h = Vec3([0.0; 3]);
        for i in start..end {
            let centroid = Vec3::from_slice_tuple((&self.0.xg, &self.0.yg, &self.0.zg), i);
            let vj = Vec3::from_slice_tuple((&self.0.vjx, &self.0.vjy, &self.0.vjz), i);
            h += h_point(&centroid, &vj, self.0.r[i], target);
        }
        h
    }
}

impl HFieldSolver for DipoleSources<PointSources> {
    fn h_field_branch(&self, centroid: &Vec3, moment: &Vec3, target: &Vec3) -> Vec3 {
        let radius = 0.0; // far-field
        h_point_dipole(centroid, moment, radius, target)
    }

    fn h_field_leaf(&self, start: usize, end: usize, target: &Vec3) -> Vec3 {
        let mut h = Vec3([0.0; 3]);
        for i in start..end {
            let centroid = Vec3::from_slice_tuple((&self.0.xg, &self.0.yg, &self.0.zg), i);
            let moment = Vec3::from_slice_tuple((&self.0.vjx, &self.0.vjy, &self.0.vjz), i);
            h += h_point_dipole(&centroid, &moment, self.0.r[i], target);
        }
        h
    }
}
