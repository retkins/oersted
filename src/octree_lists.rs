//! Interaction-list implementation of the Barnes-Hut algorithm
//! 
//! Notable new features:
//! 1. Completely removes recursion (replaced with stack-based traversal)
//! 2. Separation of concerns between the octree (geometry) and the physics kernels
//! 3. Splits the interactions into near (tet4), mid (point) and far-field (node) 
//! interactions

use crate::{
    types::Vec3,
    morton,
    octree::bbox::{BoundingBox},
    mesh,
    math::sort_by_indices
};


pub struct Octree {

    // Raw source data 
    raw_nodes: Vec<Vec3>, 
    raw_connectivity: Vec<[u32; 4]>,
    source_vectors: Vec<Vec3>,

    // Computed data, which is sorted by morton code
    codes: Vec<u64>,
    idx_sorted: Vec<usize>, // map from unsorted to sorted
    nodes: Vec<Vec3>,
    connectivity: Vec<[u32;4]>,
    centroids: Vec<Vec3>,
    volumes: Vec<f64>,

    // Data about the octree itself
    bbox: BoundingBox,

    // Data for physics evaluations
    jdensity: Option<Vec<Vec3>>,
    mvectors: Option<Vec<Vec3>>
}

impl Octree {

    fn new(nodes: &[Vec3], connectivity: &[[u32;4]], source_vectors: &[Vec3]) -> Self {
        let max_depth: u8 = 21;
        let n_sources: usize = connectivity.len();

        // Compute centroids and volumes of the source elements
        let mut centroids: Vec<Vec3> = vec![Vec3::default(); n_sources]; 
        mesh::centroids(nodes, connectivity, &mut centroids);
        let mut volumes: Vec<f64> = vec![0.0; n_sources];
        mesh::volumes(nodes, connectivity, &mut volumes);

        // Compute bounding box and morton codes
        let bbox: BoundingBox = BoundingBox::from_centroids_vec(&centroids);
        let codes: Vec<u64> = encode(&centroids, &bbox, max_depth);

        // Sort the morton codes and the source data 
        let mut unsorted_to_sorted: Vec<usize> = (0..n_sources).collect();
        unsorted_to_sorted.sort_by(|&i, &j| codes[i].cmp(&codes[j]));

        let mut scratch_nodes: Vec<Vec3> = vec![Vec3::default(); nodes.len()];
        let mut nodes_sorted: Vec<Vec3> = nodes.to_vec();
        sort_by_indices(&mut nodes_sorted, &mut scratch_nodes, &unsorted_to_sorted);

        let mut scratch_connectivity: Vec<[u32;4]> = vec![[0; 4]; n_sources];
        let mut connectivity_sorted: Vec<[u32; 4]> = connectivity.to_vec();
        sort_by_indices(&mut connectivity_sorted, &mut scratch_connectivity, &unsorted_to_sorted);

        Octree {
            raw_nodes: nodes.to_vec(),
            raw_connectivity: connectivity.to_vec(),
            source_vectors: source_vectors.to_vec(),
            codes: codes,
            idx_sorted: unsorted_to_sorted,
            nodes: nodes_sorted,
            connectivity: connectivity_sorted,
            centroids: centroids,
            volumes: volumes,
            bbox: bbox,
            jdensity: None,
            mvectors: None
        }
    }

    // Create a new Octree with elemental current density vectors as the source
    pub fn new_with_currents(nodes: &[Vec3], connectivity: &[[u32;4]], jdensity: &[Vec3]) -> Self {

        let mut tree = Self::new(nodes, connectivity, jdensity);
        let mut jdensity_sorted: Vec<Vec3> = jdensity.to_vec(); 
        let mut scratch = vec![Vec3::default(); jdensity.len()];
        sort_by_indices(&mut jdensity_sorted, &mut scratch, &tree.idx_sorted);
        tree.jdensity = Some(jdensity_sorted);

        tree
    }

}

// Return the morton code of each source in the octree
fn encode(centroids: &[Vec3], bbox: &BoundingBox, max_depth: u8) -> Vec<u64> {
    let n: usize = centroids.len();
    let mut codes: Vec<u64> = Vec::with_capacity(n);
    let scale: f64 = morton::calculate_scale_factor(max_depth as u32);
    let min_corner: (f64, f64, f64) = bbox.min_corner();

    // Rewrite this to directly use Vec3
    for i in 0..n {
        let pt: (f64, f64, f64) = (
            centroids[i][0],
            centroids[i][1],
            centroids[i][2],
        );
        codes.push(morton::encode(pt, scale, bbox.side_length, min_corner));
    }

    codes
}