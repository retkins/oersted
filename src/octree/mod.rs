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
        IntegrationMethod, RequestedField,
        SourceVectors::{self, CurrentDensity, Magnetization},
    },
    check_lengths,
    math::sort_by_indices,
    sources::{h_current_point, h_current_tet4},
    types::{Mat3, Vec3},
};

pub(self) mod bbox;
use bbox::BoundingBox;
pub(self) mod node;
use node::{INVALID_NODE, get_range_in_same_node, size_at_level};
mod topology;
use topology::{Topology, build_topology};
mod sources;
use sources::{Sources, sort_sources};
mod aggregation;
use aggregation::TreeMoments;
mod evaluation;
use evaluation::{sort_targets, unsort_fields};

use std::thread;

#[derive(Clone, Copy)]
pub struct OctreeSettings {
    pub theta: f64,
    pub near_field_ratio: f64,
    pub max_leaf_size: u32,
}

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

        let topology: Topology = build_topology(&sources, &codes, &bbox, max_depth, leaf_threshold);

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
            &self.sources.unsorted_to_sorted,
        );

        let mut tree_moments = TreeMoments::new(self.topology.len());
        tree_moments.update_moments(
            &vectors_sorted,
            &self.sources.elem_volumes,
            &self.sources.elem_centroids,
            &self.topology,
        );

        match vectors {
            SourceVectors::CurrentDensity(_) => {
                self.j_moments = Some(tree_moments);
            }
            SourceVectors::Magnetization(_) => {
                self.m_moments = Some(tree_moments);
            }
        }
    }

    /// Compute requested fields at the target location and accumulate into `out`
    ///
    /// WARNING: this function currently only computes H-fields from current-carrying
    /// meshes, as the kernels are being completed
    pub fn compute_fields(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        out: (&mut [f64], &mut [f64], &mut [f64]),
        field: RequestedField,
        source: Source,
        near_field_method: IntegrationMethod,
        theta: f64,
        batch_size: usize,
        n_threads: usize,
    ) {
        // Copy and sort targets; store index mapping for output arrays
        let ((x, y, z), indices) = sort_targets((targets.0, targets.1, targets.2));

        // Select near and far field kernels (requested field, source type, integration method)

        // Batch targets
        for (xb, yb, zb) in x
            .chunks(batch_size)
            .zip(y.chunks(batch_size))
            .zip(z.chunks(batch_size))
            .map(|((a, b), c)| (a, b, c))
        {
            let (target_centroid, target_radius) = evaluation::target_bounds((&xb, &yb, &zb));
        }

        // Traverse the tree
        // 1. Compute distance from nearest target to node centroid
        // 2. Decide to accept node or to open
        //  If accept -> compute field using far-field kernel
        //  If open and leaf -> compute field using direct kernel

        // Sort results back to output indices
        unsort_fields((out.0, out.1, out.2), &indices);
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
                        // TODO: attempted fix for leaf_threshold issue (still not working...)
                        let c: Vec3 = self.sources.elem_centroids[idx_source as usize];
                        let e: Vec3 = target - c;
                        let de2: f64 = e.0[0].powi(2) + e.0[1].powi(2) + e.0[2].powi(2);
                        // if d > alpha * self.sources.elem_radii[idx_source as usize] {
                        let r: f64 = alpha * self.sources.elem_radii[idx_source as usize];
                        if de2 > r * r {
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
                let volume: f64 = self.sources.elem_volumes[idx_source];
                let jdensity: Vec3 = self.sources.jdensity.as_ref().unwrap()[idx_source];
                // let vj: Vec3 = self.sources.jdensity.as_ref().unwrap()[idx_source] * volume;
                // let radius: f64 = 0.0;

                let elem = self.sources.elem_connectivity[idx_source];
                let nodes = [
                    self.sources.elem_nodes[elem[0] as usize],
                    self.sources.elem_nodes[elem[1] as usize],
                    self.sources.elem_nodes[elem[2] as usize],
                    self.sources.elem_nodes[elem[3] as usize],
                ];

                h_current_point(&nodes, &jdensity, (txb, tyb, tzb), (hxb, hyb, hzb));
            },
        );

        // Far field
        evaluate_source_batch(
            &far,
            targets,
            &mut out,
            |idx_source, txb, tyb, tzb, hxb, hyb, hzb| {
                let centroid: Vec3 = self.sources.elem_centroids[idx_source];
                let volume: f64 = self.sources.elem_volumes[idx_source];
                let jdensity: Vec3 = self.sources.jdensity.as_ref().unwrap()[idx_source];
                // let vj: Vec3 = self.sources.jdensity.as_ref().unwrap()[idx_source] * volume;
                // let radius: f64 = 0.0;

                let elem = self.sources.elem_connectivity[idx_source];
                let nodes = [
                    self.sources.elem_nodes[elem[0] as usize],
                    self.sources.elem_nodes[elem[1] as usize],
                    self.sources.elem_nodes[elem[2] as usize],
                    self.sources.elem_nodes[elem[3] as usize],
                ];
            },
        );
    }

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

        let nthreads: usize = crate::get_nthreads(n_threads_requested);
        let chunk_size: usize = (n_tgt / nthreads).max(1);

        let chunks = crate::par_chunks(x, y, z, hx, hy, hz, chunk_size);

        thread::scope(|s| {
            for (xc, yc, zc, hxc, hyc, hzc) in chunks {
                s.spawn(move || {
                    self.h_current((xc, yc, zc), alpha, theta, (hxc, hyc, hzc));
                });
            }
        });
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
