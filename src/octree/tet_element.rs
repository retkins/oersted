use crate::{
    mat3::Mat3,
    math::{gradient::jmatrices, sort_by_indices},
    morton,
    octree::BoundingBox,
    octree::{CurrentSources, DipoleSources, HFieldSolver, Sources},
    sources::{h_field_tet4, h_point, h_point_dipole, hmag_tet4},
    vec3::Vec3,
};

pub struct TetSources {
    pub nodes: Vec<[Vec3; 4]>, // of the element, not the tree
    pub centroids: Vec<Vec3>,
    pub volumes: Vec<f64>,
    pub jdensity: Vec<Vec3>,
    pub bbox: BoundingBox,
}

impl TetSources {
    /// Constructor
    pub fn new(
        nodes_flat: &[f64],
        centroids_flat: &[f64],
        vol: &[f64],
        jdensity_flat: &[f64],
    ) -> Self {
        // TODO: length check on inputs
        let n: usize = vol.len();

        let mut nodes: Vec<[Vec3; 4]> = vec![[Vec3([0.0; 3]); 4]; n];
        let mut centroids: Vec<Vec3> = vec![Vec3([0.0; 3]); n];
        let volumes: Vec<f64> = vol.to_vec();
        let mut jdensity: Vec<Vec3> = vec![Vec3([0.0; 3]); n];

        for i in 0..n {
            let node_slice = &nodes_flat[12 * i..12 * i + 12];
            for j in 0..4 {
                for k in 0..3 {
                    nodes[i][j][k] = node_slice[3 * j + k];
                }
            }
            let j = 3 * i;
            centroids[i] = Vec3([
                centroids_flat[j],
                centroids_flat[j + 1],
                centroids_flat[j + 2],
            ]);
            jdensity[i] = Vec3([jdensity_flat[j], jdensity_flat[j + 1], jdensity_flat[j + 2]]);
        }

        let bbox = BoundingBox::from_centroids_vec(&centroids);
        Self {
            nodes,
            centroids,
            volumes,
            jdensity,
            bbox,
        }
    }
}

impl Sources for TetSources {
    fn len(&self) -> usize {
        self.volumes.len()
    }

    fn centroid(&self, i: usize) -> Vec3 {
        self.centroids[i]
    }

    fn moment(&self, i: usize) -> Vec3 {
        self.jdensity[i] * self.volumes[i]
    }

    fn sort(&mut self, indices: &[usize]) {
        let n = self.len();
        let mut scratch_nodes = vec![[Vec3([0.0; 3]); 4]; n];
        sort_by_indices(&mut self.nodes, &mut scratch_nodes, indices);

        let mut scratch_vecs = vec![Vec3([0.0; 3]); n];
        sort_by_indices(&mut self.centroids, &mut scratch_vecs, indices);
        sort_by_indices(&mut self.jdensity, &mut scratch_vecs, indices);

        let mut scratch_vol = vec![0.0; n];
        sort_by_indices(&mut self.volumes, &mut scratch_vol, indices);
    }

    fn bbox(&self) -> &crate::octree::BoundingBox {
        &self.bbox
    }

    fn encode(&mut self, max_depth: u8) -> (&BoundingBox, Vec<u64>) {
        let n = self.len();
        let mut codes: Vec<u64> = Vec::with_capacity(n);
        let bbox = self.bbox();
        let scale: f64 = morton::calculate_scale_factor(max_depth as u32);
        let min_corner: (f64, f64, f64) = bbox.min_corner();

        for i in 0..n {
            let pt: (f64, f64, f64) = (
                self.centroids[i][0],
                self.centroids[i][1],
                self.centroids[i][2],
            );
            codes.push(morton::encode(pt, scale, bbox.side_length, min_corner));
        }

        (bbox, codes)
    }
}

impl HFieldSolver for CurrentSources<TetSources> {
    fn h_field_branch(&self, centroid: &Vec3, vj: &Vec3, target: &Vec3) -> Vec3 {
        let radius = 0.0;
        h_point(centroid, vj, radius, target)
    }

    fn h_field_leaf(&self, start: usize, end: usize, target: &Vec3) -> Vec3 {
        let mut hx = [0.0];
        let mut hy = [0.0];
        let mut hz = [0.0];
        let mut f = vec![Vec3([0.0; 3]); 1];
        for i in start..end {
            h_field_tet4(
                &self.0.nodes[i],
                &self.0.jdensity[i],
                (&[target[0]], &[target[1]], &[target[2]]),
                &mut f,
                (&mut hx, &mut hy, &mut hz),
            );
            f.fill(Vec3([0.0; 3]));
        }
        Vec3([hx[0], hy[0], hz[0]])
    }
}

impl HFieldSolver for DipoleSources<TetSources> {
    fn h_field_branch(&self, centroid: &Vec3, moment: &Vec3, target: &Vec3) -> Vec3 {
        h_point_dipole(centroid, moment, 0.0, target)
    }

    fn h_field_leaf(&self, start: usize, end: usize, target: &Vec3) -> Vec3 {
        let mut hx = [0.0];
        let mut hy = [0.0];
        let mut hz = [0.0];
        let mut f = vec![Vec3([0.0; 3]); 1];
        let elements: [[u32; 4]; 1] = [[0u32, 1u32, 2u32, 3u32]];
        for i in start..end {
            let j_invt: Vec<crate::mat3::Mat3> = jmatrices(&self.0.nodes[start], &elements);
            hmag_tet4(
                &self.0.nodes[i],
                &self.0.jdensity[i], // TODO: make this its own value?
                &j_invt,
                (&[target[0]], &[target[1]], &[target[2]]),
                &elements,
                &mut f,
                (&mut hx, &mut hy, &mut hz),
            );
            f.fill(Vec3([0.0; 3]));
        }
        Vec3([hx[0], hy[0], hz[0]])
    }
}
