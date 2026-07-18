//! Interaction-list implementation of the Barnes-Hut algorithm
//!
//! Notable new features:
//! 1. Completely removes recursion (replaced with stack-based traversal)
//! 2. Separation of concerns between the octree (geometry) and the physics kernels
//! 3. Splits the interactions into near (tet4), mid (point) and far-field (node)
//!    interactions
//!

use crate::{
    biotsavart::{
        IntegrationMethod, Kernel, RequestedField,
        SourceVectors::{self, CurrentDensity, Magnetization},
    },
    check_lengths, get_nthreads,
    math::sort_by_indices,
    par_chunks,
    types::Vec3,
};

pub(self) mod bbox;
use bbox::BoundingBox;
pub(self) mod node;
use node::{INVALID_NODE, get_range_in_same_node, size_at_level};
mod topology;
use topology::{Topology, build_topology};
mod sources;
use sources::{Sources, sort_sources};
pub mod aggregation;
use aggregation::TreeMoments;
mod evaluation;
use evaluation::{sort_targets, unsort_fields};
mod kernels;
use kernels::{FarKernel, select_far_kernel, select_near_kernel};

use std::thread;

#[derive(Clone, Copy)]
pub struct OctreeSettings {
    pub theta: f64,
    pub near_field_ratio: f64,
    pub max_leaf_size: u32,
}

#[derive(Clone, Copy)]
pub enum Source {
    CurrentDensity,
    Magnetization,
}

/// An octree constructed from Nsources tet4 elements
#[derive(Debug)]
pub struct Octree {
    // Data about the octree itself (node-level data)
    topology: Topology,

    // Physics and geometry data about the sources in the octree
    sources: Sources,
    j_moments: Option<TreeMoments>,
    m_moments: Option<TreeMoments>,
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

        let (codes, bbox, sources) =
            sort_sources(nodes, connectivity, jdensity, mvectors, max_depth);

        let topology: Topology = build_topology(&codes, &bbox, max_depth, leaf_threshold);

        let mut octree: Octree = Octree {
            topology,
            j_moments: None,
            m_moments: None,
            sources,
        };

        if let Some(j) = jdensity {
            octree.update_jdensity(j);
        }

        if let Some(m) = mvectors {
            octree.update_magnetization(m);
        }

        octree
    }

    // Update the stored current density source information in the tree
    pub fn update_jdensity(&mut self, jdensity: &[Vec3]) {
        self.update_source_vectors(CurrentDensity(jdensity));
    }

    // Update the stored magnetiation source information in the tree
    pub fn update_magnetization(&mut self, magnetization: &[Vec3]) {
        self.update_source_vectors(Magnetization(magnetization));
    }

    fn update_source_vectors(&mut self, vectors: SourceVectors) {
        let mut vectors_sorted: Vec<Vec3> = match vectors {
            SourceVectors::CurrentDensity(v) => v.to_vec(),
            SourceVectors::Magnetization(v) => v.to_vec(),
        };
        let mut scratch: Vec<Vec3> = vec![Vec3::default(); vectors_sorted.len()];
        sort_by_indices(
            &mut vectors_sorted,
            &mut scratch,
            &self.sources.sorted_to_unsorted,
        );

        let mut tree_moments = TreeMoments::new(self.topology.len());
        tree_moments.update(
            &vectors_sorted,
            &self.sources.elem_volumes,
            &self.sources.elem_centroids,
            &self.sources.elem_extents,
            &self.topology,
            true,
        );

        match vectors {
            SourceVectors::CurrentDensity(j) => {
                self.j_moments = Some(tree_moments);
                self.sources.update_jdensity(Some(j));
            }
            SourceVectors::Magnetization(m) => {
                self.m_moments = Some(tree_moments);
                self.sources.update_mvectors(Some(m));
            }
        }
    }

    fn traverse_tree(
        &self,
        target_centroid: &Vec3,
        target_radius: f64,
        targets: (&[f64], &[f64], &[f64]),
        out: (&mut [f64], &mut [f64], &mut [f64]),
        theta: f64,
        near_kernel: Kernel,
        far_kernel: FarKernel,
        tree_moments: &TreeMoments,
        source_vectors: &[Vec3],
    ) {
        // Create a stack of tree source nodes and start at root
        let mut stack: Vec<u32> = Vec::with_capacity(128);
        stack.push(0);

        // Pop a node off of the stack and evaluate what to do
        while let Some(_ni) = stack.pop() {
            let ni = _ni as usize;

            // Node centroid, distance from node to target, and size of node
            // let c: Vec3 = self.topology.centroids[ni];
            let c = tree_moments.centers[ni];
            let d: f64 = (*target_centroid - c).mag();

            // Barnes-Hut acceptance test
            if tree_moments.bmax[ni] < theta * (d - target_radius) {
                // Branch node accepted
                far_kernel(
                    &c,
                    &tree_moments.monopole[ni],
                    &tree_moments.dipole[ni],
                    (targets.0, targets.1, targets.2),
                    (out.0, out.1, out.2),
                )
            } else {
                // BH-test failed; open the leaf
                if self.topology.is_leaf[ni] {
                    // Evaluate leaves directly, using all source elements in the leaf
                    let (start, end) = self.topology.source_range[ni];
                    for e in start as usize..end as usize {
                        near_kernel(
                            &self.sources.elem_node_coords(e),
                            &source_vectors[e],
                            (targets.0, targets.1, targets.2),
                            (out.0, out.1, out.2),
                        )
                    }
                } else {
                    // Add the child nodes to the stack
                    for child in self.topology.children[ni] {
                        if child == INVALID_NODE {
                            continue;
                        } else {
                            stack.push(child)
                        }
                    }
                }
            }
        }
    }

    /// Compute requested fields at the target location and accumulate into `out`
    ///
    /// WARNING: this function currently only computes H-fields from current-carrying
    /// meshes, as the kernels are being completed
    fn compute_fields_scalar(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        out: (&mut [f64], &mut [f64], &mut [f64]),
        field: RequestedField,
        source: Source,
        near_field_method: IntegrationMethod,
        theta: f64,
        batch_size: usize,
    ) {
        // Copy and sort targets; store index mapping for output arrays
        let ((x, y, z), indices) = sort_targets((targets.0, targets.1, targets.2));
        let n_targets = x.len();

        let (mut fx, mut fy, mut fz) = (
            vec![0.0; n_targets],
            vec![0.0; n_targets],
            vec![0.0; n_targets],
        );

        // Select near and far field kernels (requested field, source type, integration method)
        let near_kernel: Kernel = select_near_kernel(field, source, near_field_method);
        let far_kernel: FarKernel = select_far_kernel(field, source);

        let (tree_moments, source_vectors) = match source {
            // TODO: return an error if the moments are not available
            Source::CurrentDensity => (
                self.j_moments.as_ref().unwrap(),
                self.sources.jdensity.as_ref().unwrap(),
            ),
            Source::Magnetization => (
                self.m_moments.as_ref().unwrap(),
                self.sources.mvectors.as_ref().unwrap(),
            ),
        };

        // Batch targets
        for (xb, yb, zb, fxb, fyb, fzb) in
            par_chunks(&x, &y, &z, &mut fx, &mut fy, &mut fz, batch_size)
        {
            let (target_centroid, target_radius) = evaluation::target_bounds((&xb, &yb, &zb));

            // Traverse the tree
            // 1. Compute distance from nearest target to node centroid
            // 2. Decide to accept node or to open
            //  If accept -> compute field using far-field kernel
            //  If open and leaf -> compute field using direct kernel
            self.traverse_tree(
                &target_centroid,
                target_radius,
                (xb, yb, zb),
                (fxb, fyb, fzb),
                theta,
                near_kernel,
                far_kernel,
                tree_moments,
                source_vectors,
            );
        }
        // Sort results back to output indices
        unsort_fields((&fx, &fy, &fz), (out.0, out.1, out.2), &indices);
    }

    // Parallel version of the above
    pub fn compute_fields(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        out: (&mut [f64], &mut [f64], &mut [f64]),
        field: RequestedField,
        source: Source,
        near_field_method: IntegrationMethod,
        theta: f64,
        batch_size: usize,
        n_threads_requested: u32,
    ) {
        let (x, y, z) = targets;
        let (fx, fy, fz) = out;
        let n_targets = check_lengths!(x, y, z, fx, fy, fz);
        let n_threads = get_nthreads(n_threads_requested);
        let chunk_size = n_targets.div_ceil(n_threads);
        let chunks = par_chunks(x, y, z, fx, fy, fz, chunk_size);

        thread::scope(|s| {
            for (xc, yc, zc, fxc, fyc, fzc) in chunks {
                s.spawn(move || {
                    self.compute_fields_scalar(
                        (xc, yc, zc),
                        (fxc, fyc, fzc),
                        field,
                        source,
                        near_field_method,
                        theta,
                        batch_size,
                    );
                });
            }
        })
    }
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

        // let alpha = 2.5;
        // let theta = 0.5;
        // let (near, mid, far) = tree.traverse((&targets.0, &targets.1, &targets.2), alpha, theta);
        // println!("Near: {:?}", near);
        // println!("Mid: {:?}", mid);
        // println!("Far: {:?}", far);

        let jdensity: Vec<Vec3> = vec![Vec3([1.0e7, 0.0, 0.0]); 8];
        tree.update_jdensity(&jdensity);

        // Each leaf should have moment = vol*J
        let m_exp = jdensity[0][0] * tree.sources.elem_volumes[0];
        println!("Expected moment: {}", m_exp);
        for i in 1..=8 {
            // skip root
            let m = tree.j_moments.as_ref().unwrap().monopole[i];
            let diff = (m[0] - m_exp).abs();
            println!("diff: {}", diff);
            assert!(diff < 1e-12);
            assert!(m[1].abs() < 1e-12);
            assert!(m[2].abs() < 1e-12);
        }

        // Root should be sum of all 8 leaves
        let root_m = tree.j_moments.as_ref().unwrap().monopole[0];
        assert!((root_m[0] - 8.0 * m_exp).abs() < 1e-12);

        // TODO: extra checks...
    }
}
