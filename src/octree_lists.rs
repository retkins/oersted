//! Interaction-list implementation of the Barnes-Hut algorithm
//!
//! Notable new features:
//! 1. Completely removes recursion (replaced with stack-based traversal)
//! 2. Separation of concerns between the octree (geometry) and the physics kernels
//! 3. Splits the interactions into near (tet4), mid (point) and far-field (node)
//!    interactions
//!
#![allow(unused)]

use crate::{
    check_lengths, check_optional_lengths,
    math::sort_by_indices,
    mesh, morton,
    octree::{bbox::BoundingBox, size_at_level},
    sources::{h_current_point, h_current_tet4},
    types::Vec3,
};

use std::f64::consts::PI;
use std::time::Instant;

// Sentinel value to identify nodes who have not been assigned in the tree yet
const INVALID_NODE: u32 = u32::MAX;

/// Return the morton code prefix at a given level of the tree
#[inline(always)]
pub fn get_prefix(code: u64, max_level: u8, level: u8) -> u64 {
    let shift: u64 = 3u64 * (max_level - level) as u64;
    let prefix: u64 = code >> shift;
    prefix
}

// Get the end index of a range that has the same parent node at the current level
// Returns the index that has the changed prefix, so an open range [start_index, end_index)
// The range is bounded by [start, end]
pub fn get_range_in_same_node(
    codes: &[u64],
    level: u8,
    max_depth: u8,
    start: usize,
    end: usize,
) -> usize {
    let shift = 3 * (max_depth - level) as u64;
    let target_prefix = codes[start] >> shift;

    // Binary search for first index with different prefix
    let remainder = &codes[start..end];
    let offset = remainder.partition_point(|&c| c >> shift == target_prefix);
    start + offset
}

// Topology information for traversing the octree
#[derive(Debug)]
struct Topology {
    // Each vector has length Nnodes
    children: Vec<[u32; 8]>, // provides indices into these arrrays (tree connectivity)
    centroids: Vec<Vec3>,
    volumes: Vec<f64>,
    sizes: Vec<f64>,
    source_range: Vec<(u32, u32)>,
    is_leaf: Vec<bool>,
    max_depth: u8,
}

impl Topology {
    fn new(
        children: Vec<[u32; 8]>,
        centroids: Vec<Vec3>,
        volumes: Vec<f64>,
        sizes: Vec<f64>,
        source_range: Vec<(u32, u32)>,
        is_leaf: Vec<bool>,
        max_depth: u8,
    ) -> Self {
        check_lengths!(children, centroids, volumes, sizes, source_range, is_leaf);

        Self {
            children,
            centroids,
            volumes,
            sizes,
            source_range,
            is_leaf,
            max_depth,
        }
    }

    // Return the number of sources in the Topology object
    fn len(&self) -> usize {
        // Lengths of all vectors were checked at build, so we can choose any
        // of the member data
        self.children.len()
    }
}

// Information about the individual sources the octree represents
#[derive(Debug)]
struct Sources {
    // Mesh data that is morton sorted
    elem_connectivity: Vec<[u32; 4]>,
    elem_centroids: Vec<Vec3>,
    elem_volumes: Vec<f64>,
    elem_radii: Vec<f64>,

    // Mesh nodes are not morton sorted
    elem_nodes: Vec<Vec3>,

    // Physics data, also morton sorted
    jdensity: Option<Vec<Vec3>>,
    mvectors: Option<Vec<Vec3>>,
}

impl Sources {
    fn new(
        elem_connectivity: Vec<[u32; 4]>,
        elem_centroids: Vec<Vec3>,
        elem_volumes: Vec<f64>,
        elem_radii: Vec<f64>,
        elem_nodes: Vec<Vec3>,
        jdensity: Option<Vec<Vec3>>,
        mvectors: Option<Vec<Vec3>>,
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
        }
    }

    fn len(&self) -> usize {
        self.elem_connectivity.len()
    }
}

#[derive(Debug)]
pub struct InteractionList {
    pub source_indices: Vec<u32>,
    pub target_indices: Vec<u32>,
}

impl InteractionList {
    fn new(capacity: usize) -> Self {
        Self {
            source_indices: Vec::with_capacity(capacity),
            target_indices: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, source: u32, target: u32) {
        self.source_indices.push(source);
        self.target_indices.push(target);
    }

    pub fn len(&self) -> usize {
        self.source_indices.len()
    }

    fn sort_by_sources(&mut self) {
        let mut scratch: Vec<u32> = vec![0; self.len()];
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.sort_by(|&i, &j| self.source_indices[i].cmp(&self.source_indices[j]));
        sort_by_indices(&mut self.target_indices, &mut scratch, &indices);
        sort_by_indices(&mut self.source_indices, &mut scratch, &indices);
    }
}

/// An octree constructed from Nsources tet4 elements
#[derive(Debug)]
pub struct Octree {
    // Morton codes (length = Nsources)
    codes: Vec<u64>,
    idx_sorted: Vec<usize>, // map from unsorted to sorted

    // Data about the octree itself (node-level data)
    bbox: BoundingBox,
    topology: Topology,
    j_moments: Option<Vec<Vec3>>,
    m_moments: Option<Vec<Vec3>>,

    // Physics and geometry data about the sources in the octree
    sources: Sources,
}

impl Octree {
    /// Create a new octree
    pub fn new(
        nodes: &[Vec3],
        connectivity: &[[u32; 4]],
        jdensity: Option<&[Vec3]>,
        mvectors: Option<&[Vec3]>,
        leaf_threshold: u32,
    ) -> Self {
        let max_depth: u8 = 21;

        let sort_timer = Instant::now();
        let (codes, idx_sorted, bbox, sources) =
            Self::sort_sources(nodes, connectivity, jdensity, mvectors, max_depth);
        println!(
            "Source sort time: {:.3} sec",
            sort_timer.elapsed().as_secs_f64()
        );

        let top_timer = Instant::now();
        let topology = Self::build_topology(&sources, &codes, &bbox, max_depth, leaf_threshold);
        println!(
            "Topology time: {:.3} sec",
            top_timer.elapsed().as_secs_f64()
        );

        let mut octree: Octree = Octree {
            codes,
            idx_sorted,
            bbox,
            topology,
            j_moments: None,
            m_moments: None,
            sources,
        };

        if let Some(j) = jdensity {
            octree.update_jdensity_moments(j);
        }

        if let Some(m) = mvectors {
            octree.update_mvector_moments(m);
        }

        octree
    }

    // Sort and organize the tree source data
    fn sort_sources(
        nodes: &[Vec3],
        connectivity: &[[u32; 4]],
        jdensity: Option<&[Vec3]>,
        mvectors: Option<&[Vec3]>,
        max_depth: u8,
    ) -> (Vec<u64>, Vec<usize>, BoundingBox, Sources) {
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
            radii[i] = (v * 3.0 / (4.0 * PI)).powf(1.0 / 3.0);
        }

        // Source data may or may not be available at construction time
        let jdensity_sorted: Option<Vec<Vec3>> = match jdensity {
            Some(j) => {
                let mut _jdensity_sorted: Vec<Vec3> = j.to_vec();
                sort_by_indices(
                    &mut _jdensity_sorted,
                    &mut scratch_vec3,
                    &unsorted_to_sorted,
                );
                Some(_jdensity_sorted)
            }
            None => None,
        };
        let mvectors_sorted: Option<Vec<Vec3>> = match mvectors {
            Some(m) => {
                let mut _mvectors_sorted: Vec<Vec3> = m.to_vec();
                sort_by_indices(
                    &mut _mvectors_sorted,
                    &mut scratch_vec3,
                    &unsorted_to_sorted,
                );
                Some(_mvectors_sorted)
            }
            None => None,
        };

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
        };

        (codes, unsorted_to_sorted, bbox, sources)
    }

    // Build the internal structure of the tree, using a top down approach
    fn build_topology(
        sources: &Sources,
        codes: &[u64],
        bbox: &BoundingBox,
        max_depth: u8,
        leaf_threshold: u32,
    ) -> Topology {
        let n_sources: usize = codes.len();
        // TODO: develop a simple way to estimate number of nodes in the tree
        // This will avoid reallocations, but is a minor optimization as Vec is
        // generally very efficient with reallocations
        let n_nodes_estimate: usize = n_sources / leaf_threshold as usize;

        // Together, these form the same information as a Vec<Node>
        // They are stored separately for cache efficiency when traversing the tree
        let mut levels: Vec<u8> = Vec::with_capacity(n_nodes_estimate);
        let mut children: Vec<[u32; 8]> = Vec::with_capacity(n_nodes_estimate);
        let mut centroids: Vec<Vec3> = Vec::with_capacity(n_nodes_estimate);
        let mut volumes: Vec<f64> = Vec::with_capacity(n_nodes_estimate);
        let mut sizes: Vec<f64> = Vec::with_capacity(n_nodes_estimate);
        let mut source_range: Vec<(u32, u32)> = Vec::with_capacity(n_nodes_estimate);
        let mut is_leaf: Vec<bool> = Vec::with_capacity(n_nodes_estimate);

        // Start at root node
        levels.push(0); // TODO: might be unnecessary
        children.push([INVALID_NODE; 8]); // Updated later 
        // Centroids and volumes are computed in bottom-up pass
        centroids.push(Vec3::default());
        volumes.push(0.0);
        sizes.push(size_at_level(bbox.side_length, 0));
        source_range.push((0, n_sources as u32));
        is_leaf.push(false);

        // Nodes that need children built at the next level
        let mut current_level_nodes: Vec<usize> = vec![0];

        for level in 0..max_depth {
            let mut next_level_nodes: Vec<usize> = Vec::new();

            for &idx_parent in &current_level_nodes {
                let (range_start, range_end) = source_range[idx_parent];
                let mut child_slot: u32 = 0;
                let mut cursor: usize = range_start as usize;

                while cursor < range_end as usize {
                    let child_end = get_range_in_same_node(
                        codes,
                        level + 1,
                        max_depth,
                        cursor,
                        range_end as usize,
                    );

                    // Child index is the current length of the nodes arrays
                    let idx_child = levels.len();

                    // Add a child node
                    levels.push(level + 1);
                    children.push([INVALID_NODE; 8]);
                    centroids.push(Vec3::default()); // Compute later 
                    volumes.push(0.0);
                    sizes.push(size_at_level(bbox.side_length, level + 1));
                    source_range.push((cursor as u32, child_end as u32));

                    let child_is_leaf: bool = {
                        let range_size = child_end - cursor;
                        range_size <= leaf_threshold as usize || (level + 1) >= max_depth
                    };
                    is_leaf.push(child_is_leaf);

                    // Update tree connectivity for this node
                    debug_assert!(child_slot < 8, "Error! Node has more than 8 children!");
                    children[idx_parent][child_slot as usize] = idx_child as u32;
                    child_slot += 1;

                    if !child_is_leaf {
                        next_level_nodes.push(idx_child);
                    }

                    cursor = child_end;
                }
            }

            current_level_nodes = next_level_nodes;
            if current_level_nodes.is_empty() {
                break;
            }
        }

        let mut topology: Topology = Topology::new(
            children,
            centroids,
            volumes,
            sizes,
            source_range,
            is_leaf,
            max_depth,
        );

        Self::update_centroids(sources, &mut topology);

        topology
    }

    // Update the nodal centroid positions in the tree topology object
    // This is done by considering the volume-weighted location of all source
    // elements in the tree
    fn update_centroids(sources: &Sources, topology: &mut Topology) {
        // Nodes at the end are generally leaves, so we start there
        for i in (0..topology.len()).rev() {
            let mut centroid: Vec3 = Vec3::default();
            let mut total_volume: f64 = 0.0;

            if topology.is_leaf[i] {
                // Sum from the source elements
                let source_range = topology.source_range[i];
                for j in source_range.0..source_range.1 {
                    let c: Vec3 = sources.elem_centroids[j as usize];
                    let v: f64 = sources.elem_volumes[j as usize];
                    centroid += c * v;
                    total_volume += v;
                }
            } else {
                // Sum from the child nodes
                for child in topology.children[i] {
                    if child == INVALID_NODE {
                        continue;
                    }
                    let c: Vec3 = topology.centroids[child as usize];
                    let v: f64 = topology.volumes[child as usize];
                    centroid += c * v;
                    total_volume += v;
                }
            }
            topology.volumes[i] = total_volume;
            topology.centroids[i] = centroid / total_volume;
        }
    }

    // Update source moments in the tree
    // Note: source vectors must be sorted prior to calling this function!
    fn update_moments(
        source_vectors: &[Vec3],
        source_volumes: &[f64],
        topology: &Topology,
        node_moments: &mut [Vec3],
    ) {
        // Nodes at the end are generally leaves, so we start there
        for i in (0..topology.len()).rev() {
            if topology.is_leaf[i] {
                // Sum from the source elements
                let source_range = topology.source_range[i];
                for j in source_range.0..source_range.1 {
                    node_moments[i] += source_vectors[j as usize] * source_volumes[j as usize];
                }
            } else {
                // Sum from the child nodes
                for child in topology.children[i] {
                    if child == INVALID_NODE {
                        continue;
                    }
                    node_moments[i] += node_moments[child as usize]
                }
            }
        }
    }

    /// Update jdensity moments in the tree using unsorted values
    pub fn update_jdensity_moments(&mut self, jdensity: &[Vec3]) {
        // Reset the j_moments array on the octree
        self.j_moments = Some(vec![Vec3::default(); self.topology.len()]);

        let mut jdensity_sorted: Vec<Vec3> = jdensity.to_vec();
        let mut jdensity_scratch: Vec<Vec3> = vec![Vec3::default(); jdensity.len()];
        sort_by_indices(
            &mut jdensity_sorted,
            &mut jdensity_scratch,
            &self.idx_sorted,
        );

        Self::update_moments(
            &jdensity_sorted,
            &self.sources.elem_volumes,
            &self.topology,
            self.j_moments.as_mut().unwrap(),
        );

        self.sources.jdensity = Some(jdensity_sorted);
    }

    /// Update magnetization moments in the tree
    pub fn update_mvector_moments(&mut self, mvectors: &[Vec3]) {
        // Reset the mvectors array on the tree
        self.m_moments = Some(vec![Vec3::default(); self.topology.len()]);

        // The user will be assigning based on the original array order, so we need to
        // re-sort
        let mut mvectors_sorted: Vec<Vec3> = mvectors.to_vec();
        let mut mvectors_scratch: Vec<Vec3> = vec![Vec3::default(); mvectors.len()];
        sort_by_indices(
            &mut mvectors_sorted,
            &mut mvectors_scratch,
            &self.idx_sorted,
        );

        Self::update_moments(
            &mvectors_sorted,
            &self.sources.elem_volumes,
            &self.topology,
            self.m_moments.as_mut().unwrap(),
        );

        self.sources.mvectors = Some(mvectors_sorted);
    }

    /// Traverse the octree and generate interaction lists
    ///
    /// Arguments:
    /// * `targets`: (m) (x,y,z) coordinates at which to evaluate the octree
    /// * `alpha`: multiplier on element radius, at which to determine near/mid field
    /// * `theta`: Barnes-Hut angle-opening criteria
    ///
    /// Returns:
    /// (near_list, mid_list, far_list)
    pub fn traverse(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        alpha: f64,
        theta: f64,
    ) -> (InteractionList, InteractionList, InteractionList) {
        let n_targets: usize = check_lengths!(targets.0, targets.1, targets.2);

        let initial_list_capacity: usize = 1_000_000;
        let mut near: InteractionList = InteractionList::new(initial_list_capacity);
        let mut mid: InteractionList = InteractionList::new(initial_list_capacity);
        let mut far: InteractionList = InteractionList::new(initial_list_capacity);

        let mut stack: Vec<u32> = Vec::with_capacity(128);
        let theta2 = theta * theta;

        for i in 0..n_targets {
            let target: Vec3 = Vec3([targets.0[i], targets.1[i], targets.2[i]]);

            // Clear the stack for each target and start at root
            stack.clear();
            stack.push(0);

            while let Some(idx_node) = stack.pop() {
                // let d: f64 = target.distance(&self.topology.centroids[idx_node as usize]);
                let dx: f64 = target[0] - self.topology.centroids[idx_node as usize][0];
                let dy: f64 = target[1] - self.topology.centroids[idx_node as usize][1];
                let dz: f64 = target[2] - self.topology.centroids[idx_node as usize][2];
                let d2: f64 = dx * dx + dy * dy + dz * dz;

                if self.topology.is_leaf[idx_node as usize] {
                    let source_range = self.topology.source_range[idx_node as usize];
                    for idx_source in source_range.0..source_range.1 {
                        // if d > alpha * self.sources.elem_radii[idx_source as usize] {
                        let r: f64 = alpha * self.sources.elem_radii[idx_source as usize];
                        if d2 > r * r {
                            mid.push(idx_source, i as u32);
                        } else {
                            near.push(idx_source, i as u32);
                        }
                    }
                } else {
                    // Avoid division by zero if the node is very close
                    // The BH check is:
                    // ACCEPT if theta > size / distance
                    // OPEN if theta < size / distance
                    let size = self.topology.sizes[idx_node as usize];
                    if size * size < theta2 * d2 {
                        // Accept node
                        far.push(idx_node, i as u32);
                    } else {
                        for &child in &self.topology.children[idx_node as usize] {
                            if child == INVALID_NODE {
                                continue;
                            }
                            stack.push(child);
                        }
                    }
                }
            }
        }

        near.sort_by_sources();
        mid.sort_by_sources();
        far.sort_by_sources();

        // TODO: source output lists for easy aggregation by source (target in inner loop)
        (near, mid, far)
    }

    /// Compute the magnetic field strength at target points using the interaction lists
    pub fn h_current(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        alpha: f64,
        theta: f64,
        mut out: (&mut [f64], &mut [f64], &mut [f64]),
    ) {
        let (near, mid, far) = self.traverse(targets, alpha, theta);

        // Near interactions first
        evaluate_source_batch(
            &near,
            targets,
            &mut out,
            |idx_source, txb, tyb, tzb, hxb, hyb, hzb| {
                let elem: [u32; 4] = self.sources.elem_connectivity[idx_source];
                let nodes: [Vec3; 4] = [
                    self.sources.elem_nodes[elem[0] as usize],
                    self.sources.elem_nodes[elem[1] as usize],
                    self.sources.elem_nodes[elem[2] as usize],
                    self.sources.elem_nodes[elem[3] as usize],
                ];

                h_current_tet4(
                    &nodes,
                    &self.sources.jdensity.as_ref().unwrap()[idx_source],
                    (txb, tyb, tzb),
                    (hxb, hyb, hzb),
                );
            },
        );

        // Mid field
        evaluate_source_batch(
            &mid,
            targets,
            &mut out,
            |idx_source, txb, tyb, tzb, hxb, hyb, hzb| {
                let centroid: Vec3 = self.sources.elem_centroids[idx_source];
                let vj: Vec3 = self.sources.jdensity.as_ref().unwrap()[idx_source]
                    * self.sources.elem_volumes[idx_source];
                h_current_point(&centroid, &vj, (txb, tyb, tzb), (hxb, hyb, hzb));
            },
        );

        // Far field
        evaluate_source_batch(
            &far,
            targets,
            &mut out,
            |idx_source, txb, tyb, tzb, hxb, hyb, hzb| {
                let centroid: Vec3 = self.topology.centroids[idx_source];
                let vj: Vec3 = self.j_moments.as_ref().unwrap()[idx_source];
                h_current_point(&centroid, &vj, (txb, tyb, tzb), (hxb, hyb, hzb));
            },
        );
    }

    #[cfg(feature = "parallel")]
    pub fn h_current_parallel(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        alpha: f64,
        theta: f64,
        out: (&mut [f64], &mut [f64], &mut [f64]),
        n_threads_requested: u32,
    ) {
        // Unpack
        let (x, y, z) = targets;
        let (hx, hy, hz) = out;

        // TODO: length checks
        let n_tgt: usize = check_lengths!(x, y, z, hx, hy, hz);

        use rayon::prelude::*;
        let nthreads: usize = crate::biotsavart_parallel::get_nthreads(n_threads_requested);
        let chunk_size: usize = (n_tgt / nthreads).max(1);

        // chunk the inputs
        let _x = x.par_chunks(chunk_size);
        let _y = y.par_chunks(chunk_size);
        let _z = z.par_chunks(chunk_size);
        let _hx = hx.par_chunks_mut(chunk_size);
        let _hy = hy.par_chunks_mut(chunk_size);
        let _hz = hz.par_chunks_mut(chunk_size);

        (_x, _y, _z, _hx, _hy, _hz)
            .into_par_iter()
            .for_each(|(_x, _y, _z, _hx, _hy, _hz)| {
                self.h_current((_x, _y, _z), alpha, theta, (_hx, _hy, _hz))
            });
    }
}

// Compute the effect of a source at a batch of target points
fn evaluate_source_batch<F>(
    list: &InteractionList,
    targets: (&[f64], &[f64], &[f64]),
    out: &mut (&mut [f64], &mut [f64], &mut [f64]),
    mut eval: F,
) where
    F: FnMut(usize, &[f64], &[f64], &[f64], &mut [f64], &mut [f64], &mut [f64]),
{
    let (hx, hy, hz) = out;
    let (tx, ty, tz) = targets;

    let batch_capacity: usize = 1000;
    let mut tx_batch: Vec<f64> = Vec::with_capacity(batch_capacity);
    let mut ty_batch: Vec<f64> = Vec::with_capacity(batch_capacity);
    let mut tz_batch: Vec<f64> = Vec::with_capacity(batch_capacity);
    let mut hx_batch: Vec<f64> = Vec::with_capacity(batch_capacity);
    let mut hy_batch: Vec<f64> = Vec::with_capacity(batch_capacity);
    let mut hz_batch: Vec<f64> = Vec::with_capacity(batch_capacity);

    let mut start: usize = 0;
    while start < list.len() {
        let idx_source: usize = list.source_indices[start] as usize;
        let remainder: &[u32] = &list.source_indices[start..];
        let run_length: usize = remainder.partition_point(|&s| s as usize <= idx_source);
        let end = start + run_length;

        let n_batch = end - start;
        tx_batch.clear();
        ty_batch.clear();
        tz_batch.clear();
        hx_batch.clear();
        hy_batch.clear();
        hz_batch.clear();
        tx_batch.reserve(n_batch);
        ty_batch.reserve(n_batch);
        tz_batch.reserve(n_batch);
        hx_batch.resize(n_batch, 0.0);
        hy_batch.resize(n_batch, 0.0);
        hz_batch.resize(n_batch, 0.0);

        // Copy to batch
        for i in start..end {
            let idx_target = list.target_indices[i] as usize;
            tx_batch.push(tx[idx_target]);
            ty_batch.push(ty[idx_target]);
            tz_batch.push(tz[idx_target]);
        }

        // Run computations
        eval(
            idx_source,
            &tx_batch,
            &ty_batch,
            &tz_batch,
            &mut hx_batch,
            &mut hy_batch,
            &mut hz_batch,
        );

        // Copy back
        for (i, idx_batch) in (start..end).zip(0..n_batch) {
            let idx_target = list.target_indices[i] as usize;
            hx[idx_target] += hx_batch[idx_batch];
            hy[idx_target] += hy_batch[idx_batch];
            hz[idx_target] += hz_batch[idx_batch];
        }

        start = end;
    }
}

// Return the morton code of each source in the octree
fn encode(centroids: &[Vec3], bbox: &BoundingBox, max_depth: u8) -> Vec<u64> {
    let n: usize = centroids.len();
    let mut codes: Vec<u64> = Vec::with_capacity(n);
    let scale: f64 = morton::calculate_scale_factor(max_depth as u32);
    let min_corner: (f64, f64, f64) = bbox.min_corner();

    // TODO: Rewrite this to directly use Vec3
    for i in 0..n {
        let pt: (f64, f64, f64) = (centroids[i][0], centroids[i][1], centroids[i][2]);
        codes.push(morton::encode(pt, scale, bbox.side_length, min_corner));
    }

    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test to check the construction on a single-level octree
    #[test]
    fn test_construction_single_level() {
        let n_sources: usize = 8;

        // We need these for construction but not to do anything
        let mut connectivity: Vec<[u32; 4]> = vec![[0; 4]; n_sources];
        let mut idx: u32 = 0;
        for source in 0..n_sources {
            for i in 0..4 {
                connectivity[source][i] = idx;
                idx += 1;
            }
        }

        let mut corners: Vec<Vec3> = vec![Vec3::default(); n_sources];
        corners[0] = Vec3([1.0, 1.0, -1.0]);
        corners[1] = Vec3([-1.0, 1.0, -1.0]);
        corners[2] = Vec3([-1.0, -1.0, -1.0]);
        corners[3] = Vec3([1.0, -1.0, -1.0]);
        corners[4] = Vec3([1.0, 1.0, 1.0]);
        corners[5] = Vec3([-1.0, 1.0, 1.0]);
        corners[6] = Vec3([-1.0, -1.0, 1.0]);
        corners[7] = Vec3([1.0, -1.0, 1.0]);

        let mut elem_nodes: Vec<Vec3> = Vec::new();
        let scale: f64 = 0.01;
        for c in corners {
            elem_nodes.push(c + Vec3([1.0, 0.0, -1.0 / (2.0f64).sqrt()]) * scale);
            elem_nodes.push(c + Vec3([-1.0, 0.0, -1.0 / (2.0f64).sqrt()]) * scale);
            elem_nodes.push(c + Vec3([0.0, -1.0, 1.0 / (2.0f64).sqrt()]) * scale);
            elem_nodes.push(c + Vec3([0.0, 1.0, 1.0 / (2.0f64).sqrt()]) * scale);
        }

        let mut tree = Octree::new(&elem_nodes, *&&connectivity, None, None, 16);
        println!("{:?}", tree.topology);

        let mut targets: (Vec<f64>, Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..elem_nodes.len() {
            targets.0.push(elem_nodes[i][0]);
            targets.1.push(elem_nodes[i][1]);
            targets.2.push(elem_nodes[i][2]);
        }

        targets.0.push(100.0);
        targets.1.push(0.0);
        targets.2.push(0.0);

        let alpha = 2.5;
        let theta = 0.5;
        let (near, mid, far) = tree.traverse((&targets.0, &targets.1, &targets.2), alpha, theta);
        println!("Near: {:?}", near);
        println!("Mid: {:?}", mid);
        println!("Far: {:?}", far);

        let jdensity: Vec<Vec3> = vec![Vec3([1.0e7, 0.0, 0.0]); 8];
        tree.update_jdensity_moments(&jdensity);

        // Each leaf should have moment = vol*J
        let m_exp = jdensity[0][0] * tree.sources.elem_volumes[0];
        println!("Expected moment: {}", m_exp);
        for i in 1..=8 {
            // skip root
            let m = tree.j_moments.as_ref().unwrap()[i];
            let diff = (m[0] - m_exp).abs();
            println!("diff: {}", diff);
            assert!(diff < 1e-12);
            assert!(m[1].abs() < 1e-12);
            assert!(m[2].abs() < 1e-12);
        }

        // Root should be sum of all 8 leaves
        let root_m = tree.j_moments.as_ref().unwrap()[0];
        assert!((root_m[0] - 8.0 * m_exp).abs() < 1e-12);

        // TODO: extra checks...
    }
}
