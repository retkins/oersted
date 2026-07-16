use crate::{
    check_lengths, check_optional_lengths,
    math::{sort_by_indices, sphere_radius},
    mesh,
    octree::bbox::BoundingBox,
    octree::node::encode,
    types::Vec3,
};

/// Information about the individual sources the octree represents
///
/// All of the data must be owned by this object to allow for in-place sorting
#[derive(Debug)]
pub struct Sources {
    // Mesh data that is morton sorted
    pub elem_connectivity: Vec<[u32; 4]>,
    pub elem_centroids: Vec<Vec3>,
    pub elem_volumes: Vec<f64>,
    pub elem_radii: Vec<f64>, // TODO: perhaps remove?
    pub elem_extents: Vec<f64>,

    // Mesh nodes are not morton sorted, as connectivity stores indices into this array
    pub elem_nodes: Vec<Vec3>,

    // Physics data, also morton sorted
    pub jdensity: Option<Vec<Vec3>>,
    pub mvectors: Option<Vec<Vec3>>,
    pub sorted_to_unsorted: Vec<usize>,
}

impl Sources {
    fn new(
        elem_connectivity: Vec<[u32; 4]>,
        elem_centroids: Vec<Vec3>,
        elem_volumes: Vec<f64>,
        elem_radii: Vec<f64>,
        elem_extents: Vec<f64>,
        elem_nodes: Vec<Vec3>,
        jdensity: Option<Vec<Vec3>>,
        mvectors: Option<Vec<Vec3>>,
        sorted_to_unsorted: Vec<usize>,
    ) -> Self {
        // Defensive length checks
        let n_sources: usize = check_lengths!(elem_connectivity, elem_centroids, elem_volumes);
        check_optional_lengths!(n_sources, &jdensity, &mvectors);

        Self {
            elem_connectivity,
            elem_centroids,
            elem_volumes,
            elem_radii,
            elem_extents,
            elem_nodes,
            jdensity,
            mvectors,
            sorted_to_unsorted,
        }
    }

    pub fn len(&self) -> usize {
        self.elem_connectivity.len()
    }

    /// Source vector data may be updated after tree construction and therefore has
    /// its own update function
    pub fn update_jdensity(&mut self, jdensity: Option<&[Vec3]>) {
        self.jdensity = sort_source_vectors(jdensity, &self.sorted_to_unsorted);
    }

    /// Source vector data may be updated after tree construction and therefore has
    /// its own update function
    pub fn update_mvectors(&mut self, mvectors: Option<&[Vec3]>) {
        self.mvectors = sort_source_vectors(mvectors, &self.sorted_to_unsorted);
    }

    // Return the node coordinates associated with a source element
    pub fn elem_node_coords(&self, idx: usize) -> [Vec3; 4] {
        let elem: [u32; 4] = self.elem_connectivity[idx];
        [
            self.elem_nodes[elem[0] as usize],
            self.elem_nodes[elem[1] as usize],
            self.elem_nodes[elem[2] as usize],
            self.elem_nodes[elem[3] as usize],
        ]
    }
}

fn sort_source_vectors(vectors: Option<&[Vec3]>, indices: &[usize]) -> Option<Vec<Vec3>> {
    match vectors {
        Some(v) => {
            let mut vectors_sorted: Vec<Vec3> = v.to_vec();
            let mut scratch: Vec<Vec3> = vec![Vec3::default(); v.len()];
            sort_by_indices(&mut vectors_sorted, &mut scratch, &indices);
            Some(vectors_sorted)
        }
        None => None,
    }
}

// Sort and organize the tree source data
pub fn sort_sources(
    nodes: &[Vec3],
    connectivity: &[[u32; 4]],
    jdensity: Option<&[Vec3]>,
    mvectors: Option<&[Vec3]>,
    max_depth: u8,
) -> (Vec<u64>, BoundingBox, Sources) {
    let n_sources: usize = connectivity.len();

    // Compute bounding box and morton codes
    let mut centroids: Vec<Vec3> = vec![Vec3::default(); n_sources];
    mesh::centroids(nodes, connectivity, &mut centroids);
    let bbox: BoundingBox = BoundingBox::from_centroids_vec(&centroids);
    let mut codes: Vec<u64> = encode(&centroids, &bbox, max_depth);

    // Sort the morton codes and the source data
    let mut sorted_to_unsorted: Vec<usize> = (0..n_sources).collect();
    sorted_to_unsorted.sort_by(|&i, &j| codes[i].cmp(&codes[j]));

    let mut scratch_vec3: Vec<Vec3> = vec![Vec3::default(); n_sources];

    let mut scratch_connectivity: Vec<[u32; 4]> = vec![[0; 4]; n_sources];
    let mut connectivity_sorted: Vec<[u32; 4]> = connectivity.to_vec();
    sort_by_indices(
        &mut connectivity_sorted,
        &mut scratch_connectivity,
        &sorted_to_unsorted,
    );

    // Sort centroids and compute effective radii and volumes of the source elements
    // Volumes are naturally already sorted at this stage
    sort_by_indices(&mut centroids, &mut scratch_vec3, &sorted_to_unsorted);
    let mut volumes: Vec<f64> = vec![0.0; n_sources];
    mesh::volumes(nodes, &connectivity_sorted, &mut volumes);
    for &v in &volumes {
        assert!(v > 0.0);
    }
    let mut radii: Vec<f64> = vec![0.0; n_sources];
    for (i, v) in volumes.iter().enumerate() {
        radii[i] = sphere_radius(*v);
    }
    let mut elem_extents: Vec<f64> = vec![0.0; n_sources];
    for i in 0..n_sources {
        let c = centroids[i]; 
        let elem_nodes = connectivity_sorted[i];
        for n in elem_nodes {
            let d = (nodes[n as usize] - c).mag();
            elem_extents[i] = elem_extents[i].max(d);
        }
    }

    // Source data may or may not be available at construction time
    let jdensity_sorted: Option<Vec<Vec3>> = sort_source_vectors(jdensity, &sorted_to_unsorted);
    let mvectors_sorted: Option<Vec<Vec3>> = sort_source_vectors(mvectors, &sorted_to_unsorted);

    // Finally, sort the morton codes
    let mut scratch_codes: Vec<u64> = vec![0; n_sources];
    sort_by_indices(&mut codes, &mut scratch_codes, &sorted_to_unsorted);

    let sources = Sources {
        elem_connectivity: connectivity_sorted,
        elem_centroids: centroids,
        elem_volumes: volumes,
        elem_radii: radii,
        elem_extents: elem_extents,
        elem_nodes: nodes.to_vec(), // Perhaps we can avoid a copy and just keep a reference?
        jdensity: jdensity_sorted,
        mvectors: mvectors_sorted,
        sorted_to_unsorted,
    };

    (codes, bbox, sources)
}
