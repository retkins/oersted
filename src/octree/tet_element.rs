use crate::{
    math::{gradient::jmatrices, sort_by_indices},
    morton,
    octree::BoundingBox,
    octree::{CurrentSources, DipoleSources, HFieldSolver, Sources},
    sources::{
        h_current_tet4, h_current_tet4_edge, h_mag_tet4, h_mag_tet4_edge, h_point, h_point_dipole,
    },
    types::{Mat3, Vec3},
};

pub struct TetSources {
    pub nodes: Vec<Vec3>, // of the element, not the tree
    pub connectivity: Vec<[u32; 4]>,
    pub centroids: Vec<Vec3>,
    pub volumes: Vec<f64>,
    pub source_vectors: Vec<Vec3>,
    pub bbox: BoundingBox,
}

impl TetSources {
    /// Constructor
    pub fn new(
        nodes: &[Vec3],
        connectivity: &[[u32; 4]],
        centroids: &[Vec3],
        volumes: &[f64],
        source_vectors: &[Vec3],
    ) -> Self {
        let bbox = BoundingBox::from_centroids_vec(centroids);
        Self {
            nodes: nodes.to_vec(),
            connectivity: connectivity.to_vec(),
            centroids: centroids.to_vec(),
            volumes: volumes.to_vec(),
            source_vectors: source_vectors.to_vec(),
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
        self.source_vectors[i] * self.volumes[i]
    }

    fn sort(&mut self, indices: &[usize]) {
        let n = self.len();
        let mut scratch_vecs = vec![Vec3([0.0; 3]); n];
        sort_by_indices(&mut self.centroids, &mut scratch_vecs, indices);
        sort_by_indices(&mut self.source_vectors, &mut scratch_vecs, indices);

        let mut scratch_conn: Vec<[u32; 4]> = vec![[0u32; 4]; n];
        sort_by_indices(&mut self.connectivity, &mut scratch_conn, indices);

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
            let elem = self.0.connectivity[i];
            let nodes = [
                self.0.nodes[elem[0] as usize],
                self.0.nodes[elem[1] as usize],
                self.0.nodes[elem[2] as usize],
                self.0.nodes[elem[3] as usize],
            ];
            h_current_tet4_edge(
                &nodes,
                &self.0.source_vectors[i],
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
        // Output vectors and workspace
        let (mut hx, mut hy, mut hz) = ([0.0], [0.0], [0.0]);
        let (mut wx, mut wy, mut wz) = ([Vec3::default()], [Vec3::default()], [Vec3::default()]);

        for i in start..end {
            let elem = self.0.connectivity[i];
            let nodes = [
                self.0.nodes[elem[0] as usize],
                self.0.nodes[elem[1] as usize],
                self.0.nodes[elem[2] as usize],
                self.0.nodes[elem[3] as usize],
            ];
            h_mag_tet4_edge(
                &nodes,
                &self.0.source_vectors[i],
                (&[target[0]], &[target[1]], &[target[2]]),
                (&mut wx, &mut wy, &mut wz),
                (&mut hx, &mut hy, &mut hz),
            );
        }
        Vec3([hx[0], hy[0], hz[0]])
    }
}
