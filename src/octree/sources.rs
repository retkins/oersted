use crate::{
    check_lengths, check_optional_lengths,
    math::{sort_by_indices, sphere_radius},
    mesh,
    octree::bbox::BoundingBox,
    octree::node::encode,
    sources::point,
    types::Vec3,
};

use std::f64::consts::PI;

/// Information about the individual sources the octree represents
///
/// All of the data must be owned by this object to allow for in-place sorting
#[derive(Debug)]
pub struct Sources {
    // Mesh data that is morton sorted
    pub elem_connectivity: Vec<[u32; 4]>,
    pub elem_centroids: Vec<Vec3>,
    pub elem_volumes: Vec<f64>,
    pub elem_radii: Vec<f64>,

    // Mesh nodes are not morton sorted
    pub elem_nodes: Vec<Vec3>,

    // Physics data, also morton sorted
    pub jdensity: Option<Vec<Vec3>>,
    pub mvectors: Option<Vec<Vec3>>,
    pub unsorted_to_sorted: Vec<usize>,
}

impl Sources {
    pub fn new(
        elem_connectivity: Vec<[u32; 4]>,
        elem_centroids: Vec<Vec3>,
        elem_volumes: Vec<f64>,
        elem_radii: Vec<f64>,
        elem_nodes: Vec<Vec3>,
        jdensity: Option<Vec<Vec3>>,
        mvectors: Option<Vec<Vec3>>,
        unsorted_to_sorted: Vec<usize>,
    ) -> Self {
        // Defensive length checks
        let n_sources: usize = check_lengths!(elem_connectivity, elem_centroids, elem_volumes);
        check_optional_lengths!(n_sources, &jdensity, &mvectors);

        Self {
            elem_connectivity,
            elem_centroids,
            elem_volumes,
            elem_radii,
            elem_nodes,
            jdensity,
            mvectors,
            unsorted_to_sorted,
        }
    }

    pub fn len(&self) -> usize {
        self.elem_connectivity.len()
    }

    /// Source vector data may be updated after tree construction and therefore has
    /// its own update function
    pub fn update_jdensity(&mut self, jdensity: Option<&[Vec3]>) {
        self.jdensity = sort_source_vectors(jdensity, &self.unsorted_to_sorted);
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
    let mut unsorted_to_sorted: Vec<usize> = (0..n_sources).collect();
    unsorted_to_sorted.sort_by(|&i, &j| codes[i].cmp(&codes[j]));

    let mut scratch_vec3: Vec<Vec3> = vec![Vec3::default(); n_sources];

    let mut scratch_connectivity: Vec<[u32; 4]> = vec![[0; 4]; n_sources];
    let mut connectivity_sorted: Vec<[u32; 4]> = connectivity.to_vec();
    sort_by_indices(
        &mut connectivity_sorted,
        &mut scratch_connectivity,
        &unsorted_to_sorted,
    );

    // Sort centroids and compute effective radii and volumes of the source elements
    // Volumes are naturally already sorted at this stage
    sort_by_indices(&mut centroids, &mut scratch_vec3, &unsorted_to_sorted);
    let mut volumes: Vec<f64> = vec![0.0; n_sources];
    mesh::volumes(nodes, &connectivity_sorted, &mut volumes);
    for &v in &volumes {
        assert!(v > 0.0);
    }
    let mut radii: Vec<f64> = vec![0.0; n_sources];
    for (i, v) in volumes.iter().enumerate() {
        radii[i] = sphere_radius(*v);
    }

    // Source data may or may not be available at construction time
    let jdensity_sorted: Option<Vec<Vec3>> = sort_source_vectors(jdensity, &unsorted_to_sorted);
    let mvectors_sorted: Option<Vec<Vec3>> = sort_source_vectors(mvectors, &unsorted_to_sorted);

    // Finally, sort the morton codes
    let mut scratch_codes: Vec<u64> = vec![0; n_sources];
    sort_by_indices(&mut codes, &mut scratch_codes, &unsorted_to_sorted);

    let sources = Sources {
        elem_connectivity: connectivity_sorted,
        elem_centroids: centroids,
        elem_volumes: volumes,
        elem_radii: radii,
        elem_nodes: nodes.to_vec(), // Perhaps we can avoid a copy and just keep a reference?
        jdensity: jdensity_sorted,
        mvectors: mvectors_sorted,
        unsorted_to_sorted,
    };

    (codes, bbox, sources)
}
