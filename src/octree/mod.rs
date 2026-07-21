//! Interaction-list implementation of the Barnes-Hut algorithm

use crate::{
    biotsavart::{
        IntegrationMethod, Kernel, RequestedField,
        SourceVectors::{self, CurrentDensity, Magnetization},
    },
    check_lengths, get_nthreads,
    math::sort_by_indices,
    octree::kernels::MultipoleExpansion,
    par_chunks,
    types::Vec3,
};

mod bbox;
use bbox::BoundingBox;
mod node;
use node::{INVALID_NODE, get_range_in_same_node, size_at_level};
mod topology;
use topology::{Topology, build_topology};
mod sources;
use sources::{Sources, sort_sources};
pub mod aggregation;
use aggregation::TreeMoments;
mod evaluation;
use evaluation::{sort_targets, traverse_tree, unsort_fields};
mod kernels;
pub use kernels::MultipoleOrder;
use kernels::{FarKernel, select_far_kernel, select_near_kernel};

use std::thread;

/// Define properties for octree construction and evaluation
#[derive(Clone, Copy, Debug)]
pub struct OctreeSettings {
    pub theta: f64,
    pub max_leaf_size: u32,
    pub multipole_order: MultipoleOrder,
}

impl Default for OctreeSettings {
    fn default() -> Self {
        Self {
            theta: 0.5,
            max_leaf_size: 16,
            multipole_order: MultipoleOrder::Dipole,
        }
    }
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
    settings: OctreeSettings,
}

impl Octree {
    /// Create a new octree
    pub fn new(
        nodes: &[Vec3],
        connectivity: &[[u32; 4]],
        jdensity: Option<&[Vec3]>,
        mvectors: Option<&[Vec3]>,
        settings: OctreeSettings,
    ) -> Self {
        let max_depth: u8 = 21;

        let (codes, bbox, sources) =
            sort_sources(nodes, connectivity, jdensity, mvectors, max_depth);

        let topology: Topology = build_topology(&codes, &bbox, max_depth, settings.max_leaf_size);

        let mut octree: Octree = Octree {
            topology,
            j_moments: None,
            m_moments: None,
            sources,
            settings,
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

    /// Compute requested fields at the target location and accumulate into `out`
    fn compute_fields_scalar(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        out: (&mut [f64], &mut [f64], &mut [f64]),
        field: RequestedField,
        source: Source,
        near_field_method: IntegrationMethod,
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
        let far_kernel: FarKernel = select_far_kernel(field, source, self.settings.multipole_order);

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

        let (near, far) = traverse_tree(
            &self.topology,
            &self.sources,
            tree_moments,
            (&x, &y, &z),
            self.settings.theta,
        );
        let buf_len = near.buf_len.max(far.buf_len);
        let (mut xb, mut yb, mut zb) = (vec![0.0; buf_len], vec![0.0; buf_len], vec![0.0; buf_len]);
        let (mut fxb, mut fyb, mut fzb) =
            (vec![0.0; buf_len], vec![0.0; buf_len], vec![0.0; buf_len]);

        // Process near sources first
        for e in 0..self.sources.len() {
            let start = near.offsets[e] as usize;
            let end = near.offsets[e + 1] as usize;
            if start == end {
                continue;
            } // No targets for this source 
            let nodes = &self.sources.elem_node_coords(e);

            // Fill target buffers
            let mut i = 0usize;
            for &ti in &near.data[start..end] {
                let ti = ti as usize;
                xb[i] = x[ti];
                yb[i] = y[ti];
                zb[i] = z[ti];
                i += 1;
            }
            let len = end - start;
            fxb[..len].fill(0.0);
            fyb[..len].fill(0.0);
            fzb[..len].fill(0.0);
            near_kernel(
                nodes,
                &source_vectors[e],
                (&xb[..len], &yb[..len], &zb[..len]),
                (&mut fxb[..len], &mut fyb[..len], &mut fzb[..len]),
            );

            // Clear and return buffers
            i = 0;
            for &ti in &near.data[start..end] {
                let ti = ti as usize;
                fx[ti] += fxb[i];
                fy[ti] += fyb[i];
                fz[ti] += fzb[i];
                i += 1;
            }
        }

        // Process far sources
        for ni in 0..self.topology.len() {
            let start = far.offsets[ni] as usize;
            let end = far.offsets[ni + 1] as usize;
            if start == end {
                continue;
            } // No targets for this source 

            // Fill target buffers
            let mut i = 0usize;
            for &ti in &far.data[start..end] {
                let ti = ti as usize;
                xb[i] = x[ti];
                yb[i] = y[ti];
                zb[i] = z[ti];
                i += 1;
            }
            let len = end - start;
            fxb[..len].fill(0.0);
            fyb[..len].fill(0.0);
            fzb[..len].fill(0.0);

            let expansion = MultipoleExpansion {
                c: &tree_moments.centers[ni],
                p: &tree_moments.monopole[ni],
                D: &tree_moments.dipole[ni],
            };

            far_kernel(
                expansion,
                (&xb[..len], &yb[..len], &zb[..len]),
                (&mut fxb[..len], &mut fyb[..len], &mut fzb[..len]),
            );

            // Clear and return buffers
            i = 0;
            for &ti in &far.data[start..end] {
                let ti = ti as usize;
                fx[ti] += fxb[i];
                fy[ti] += fyb[i];
                fz[ti] += fzb[i];
                i += 1;
            }
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
        n_threads_requested: u32,
        batch_size: usize,
    ) {
        let (x, y, z) = targets;
        let (fx, fy, fz) = out;
        let n_targets: usize = check_lengths!(x, y, z, fx, fy, fz);
        let n_threads: usize = get_nthreads(n_threads_requested);
        let chunk_size: usize = n_targets.div_ceil(n_threads);
        let chunks = par_chunks(x, y, z, fx, fy, fz, chunk_size);

        thread::scope(|s| {
            for (xc, yc, zc, fxc, fyc, fzc) in chunks {
                s.spawn(move || {
                    let batches = par_chunks(xc, yc, zc, fxc, fyc, fzc, batch_size);
                    for (xb, yb, zb, fxb, fyb, fzb) in batches {
                        self.compute_fields_scalar(
                            (xb, yb, zb),
                            (fxb, fyb, fzb),
                            field,
                            source,
                            near_field_method,
                        );
                    }
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

        let mut tree = Octree::new(
            &elem_nodes,
            *&&connectivity,
            None,
            None,
            OctreeSettings::default(),
        );
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
