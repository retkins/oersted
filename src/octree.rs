//! Octree acceleration for magnetic fields calculations

#![allow(unused)]

pub mod bbox;
pub use bbox::BoundingBox;
pub mod point;
pub mod tet_element;

use crate::{
    math::{distance, mag, sort_by_indices},
    types::Vec3,
};

/// Return the size of an octree node given the side length of the root
/// node and the level in the tree
pub fn size_at_level(side_length: f64, level: u8) -> f64 {
    side_length / (2f64.powi(level as i32))
}

/// Return the morton code prefix at a given level of the tree
#[inline(always)]
pub fn get_prefix(code: u64, max_level: u8, level: u8) -> u64 {
    let shift: u64 = 3u64 * (max_level - level) as u64;
    let prefix: u64 = code >> shift;
    prefix
}

// Get the end index of a range that has the same parent node at the current level
// Returns the index that has the changed prefix, so an open range [start_index, end_index)
pub fn get_range_in_same_node(
    codes: &[u64],
    level: u8,
    max_depth: u8,
    start_index: usize,
) -> usize {
    let n: usize = codes.len();
    let current_prefix: u64 = codes[start_index] >> (3 * (max_depth - level));

    for i in start_index..n {
        let prefix: u64 = get_prefix(codes[i], max_depth, level);
        if prefix != current_prefix {
            return i;
        }
    }

    n
}

pub trait Sources {
    fn len(&self) -> usize;
    fn bbox(&self) -> &BoundingBox;
    fn centroid(&self, i: usize) -> Vec3;
    fn moment(&self, i: usize) -> Vec3; // vj for current sources, dipole for dipole sources
    fn sort(&mut self, indices: &[usize]);
    fn encode(&mut self, max_depth: u8) -> (&BoundingBox, Vec<u64>);
}

pub struct DipoleSources<S>(pub S); // for magnetic materials

impl<S: Sources> Sources for DipoleSources<S> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn bbox(&self) -> &BoundingBox {
        self.0.bbox()
    }
    fn centroid(&self, i: usize) -> Vec3 {
        self.0.centroid(i)
    }
    fn moment(&self, i: usize) -> Vec3 {
        self.0.moment(i)
    }
    fn sort(&mut self, indices: &[usize]) {
        self.0.sort(indices)
    }
    fn encode(&mut self, max_depth: u8) -> (&BoundingBox, Vec<u64>) {
        self.0.encode(max_depth)
    }
}

pub struct CurrentSources<S>(pub S); // for current-carrying conductors

impl<S: Sources> Sources for CurrentSources<S> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn bbox(&self) -> &BoundingBox {
        self.0.bbox()
    }
    fn centroid(&self, i: usize) -> Vec3 {
        self.0.centroid(i)
    }
    fn moment(&self, i: usize) -> Vec3 {
        self.0.moment(i)
    }
    fn sort(&mut self, indices: &[usize]) {
        self.0.sort(indices)
    }
    fn encode(&mut self, max_depth: u8) -> (&BoundingBox, Vec<u64>) {
        self.0.encode(max_depth)
    }
}

pub trait HFieldSolver {
    fn h_field_branch(&self, centroid: &Vec3, moment: &Vec3, target: &Vec3) -> Vec3;
    fn h_field_leaf(&self, start: usize, end: usize, target: &Vec3) -> Vec3;
}

pub trait GradHFieldSolver {
    fn gradh_field_branch(&self, centroid: &Vec3, moment: &Vec3, target: &Vec3) -> [Vec3; 3];
    fn gradh_field_leaf(&self, start: usize, end: usize, target: &Vec3) -> [Vec3; 3];
}

pub trait AFieldSolver {
    fn a_field_branch(&self, centroid: &Vec3, moment: &Vec3, target: &Vec3) -> Vec3;
    fn a_field_leaf(&self, start: usize, end: usize, target: &Vec3) -> Vec3;
}

/// Previous node definition
pub enum Node {
    Branch {
        level: u8,
        size: f64, // or maybe f32?
        children: [u32; 8],
        centroid: Vec3, // of the cluster of sources it represents
        moment: Vec3,
    },
    Leaf {
        level: u8, // maybe unecessary but harmless
        source_range: (u32, u32),
        centroid: Vec3,
    },
}

// Copied from the original implementation
fn update_centroid(
    parent_centroid: &mut Vec3,
    parent_mag: f64,
    child_centroid: &Vec3,
    child_mag: f64,
) {
    let total_mag: f64 = parent_mag + child_mag;
    for i in 0..3_usize {
        parent_centroid[i] =
            (parent_centroid[i] * parent_mag + child_centroid[i] * child_mag) / total_mag;
    }
}

/// Recursively add nodes to the octree
///
/// Returns: the index of the node that it created
///
/// This function was copied from the previous version and needs some cleanup
/// TODO:
/// - add guard on level (prevent infinite recursion if there's an error in the inputs)
fn add_node<S: Sources>(
    sources: &S,
    codes: &Vec<u64>,
    nodes: &mut Vec<Node>,
    max_depth: u8,
    leaf_threshold: u32,
    start: usize,
    end: usize,
    level: u8,
) -> u32 {
    // Recursion guard
    if level > 21 {
        eprintln!(
            "DEEP: level={}, start={}, end={}, n={}",
            level,
            start,
            end,
            end - start
        );
        panic!();
    }

    let current_index: usize;

    // Create a leaf if the number of sources in the range is small enough (dead-end for recursion)
    if end - start <= leaf_threshold as usize {
        let mut centroid: Vec3 = Vec3([0.0; 3]);
        for i in start..end {
            centroid += sources.centroid(i);
        }

        let scale = 1.0 / ((end - start) as f64);
        centroid *= scale;
        nodes.push(
            // Do not descend level
            Node::Leaf {
                level,
                source_range: (start as u32, end as u32),
                centroid,
            },
        );
        current_index = nodes.len() - 1;
    }
    // Otherwise, create an branch node
    else {
        // Size of the node (cube side length) for BH acceptance criteria
        let size: f64 = size_at_level(sources.bbox().side_length, level);

        // Initialize the branch node first, then recursive calls later fill it
        nodes.push(Node::Branch {
            level,
            size,
            children: [0; 8],
            centroid: Vec3([0.0; 3]),
            moment: Vec3([0.0; 3]),
        });

        current_index = nodes.len() - 1; // index of the node just created

        // Now recurse to create its children
        let mut child_indices: [u32; 8] = [0; 8];
        let mut n_children: usize = 0;
        let mut cursor = start;

        while cursor < end {
            // Descend to next level here, hence `level+1`
            // index to sources array
            let child_end = get_range_in_same_node(codes, level + 1, max_depth, cursor);

            // index in nodes array
            let child_idx = add_node(
                sources,
                codes,
                nodes,
                max_depth,
                leaf_threshold,
                cursor,
                child_end,
                level + 1,
            );
            child_indices[n_children] = child_idx;
            n_children += 1;
            cursor = child_end; // index in sources array
        }

        // Update the previously-initialized node with indices to its children
        // and the volume current density product
        let mut parent_centroid: Vec3 = Vec3([0.0; 3]);
        let mut parent_moment: Vec3 = Vec3([0.0; 3]);

        for idx in child_indices {
            if idx < 1 {
                // Skip empty nodes
                break;
            }

            let mut parent_mag: f64 = parent_moment.mag();

            match nodes[idx as usize] {
                Node::Branch {
                    level: _,
                    size: _,
                    children: _,
                    centroid,
                    moment,
                } => {
                    update_centroid(&mut parent_centroid, parent_mag, &centroid, moment.mag());
                    parent_moment += moment;
                }
                Node::Leaf {
                    level: _,
                    source_range,
                    centroid,
                } => {
                    for _i in source_range.0..source_range.1 {
                        let i = _i as usize;
                        let moment = sources.moment(i);

                        update_centroid(
                            &mut parent_centroid,
                            parent_mag,
                            &sources.centroid(i),
                            moment.mag(),
                        );
                        parent_moment += moment;
                        parent_mag = parent_moment.mag();
                    }
                }
            }
        }

        if let Node::Branch {
            children,
            centroid,
            moment,
            ..
        } = &mut nodes[current_index]
        {
            *children = child_indices;
            *centroid = parent_centroid;
            *moment = parent_moment;
        }
    }

    current_index as u32
}

pub struct Octree<S> {
    pub nodes: Vec<Node>,
    pub codes: Vec<u64>,
    pub bbox: BoundingBox,
    pub sources: S,
}

impl<S: Sources> Octree<S> {
    pub fn build_from_sources(mut s: S, max_depth: u8, leaf_threshold: u32) -> Self {
        let (bbox, mut codes) = s.encode(max_depth);
        let bbox = *bbox;

        // sort the sources by morton code
        let mut indices: Vec<usize> = (0..codes.len()).collect();
        indices.sort_by(|&i, &j| codes[i].cmp(&codes[j]));
        s.sort(&indices);

        // now sort the codes themselves
        let mut scratch_codes: Vec<u64> = vec![0; codes.len()];
        sort_by_indices(&mut codes, &mut scratch_codes, &indices);

        // make nodes
        let mut nodes: Vec<Node> = Vec::with_capacity(s.len());
        let start = 0;
        let end = s.len();
        let level: u8 = 0;
        add_node(
            &s,
            &codes,
            &mut nodes,
            max_depth,
            leaf_threshold,
            start,
            end,
            level,
        );

        Self {
            nodes,
            codes,
            bbox,
            sources: s,
        }
    }
}

impl<S: HFieldSolver + std::marker::Sync> Octree<S> {
    // Recursively traverse the tree to compute the h-field at a single target from all nodes
    fn h_traverse(&self, idx: u32, target: &Vec3, theta: f64) -> Vec3 {
        let mut h = Vec3([0.0; 3]);

        match &self.nodes[idx as usize] {
            Node::Branch {
                level,
                size,
                children,
                centroid,
                moment,
            } => {
                let d = (*target - *centroid).mag();
                if d * theta > *size {
                    // Node accepted
                    h = self.sources.h_field_branch(centroid, moment, target);
                } else {
                    // Node too close, recurse into children
                    for &child in children {
                        if child > 0 {
                            let h_child = self.h_traverse(child, target, theta);
                            h[0] += h_child[0];
                            h[1] += h_child[1];
                            h[2] += h_child[2];
                        }
                    }
                }
            }
            Node::Leaf {
                level,
                source_range,
                centroid,
            } => {
                let (i, j) = (source_range.0 as usize, source_range.1 as usize);
                h = self.sources.h_field_leaf(i, j, target);
            }
        }
        h
    }

    // Compute the h-field at the target locations
    pub fn h_field(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        h: (&mut [f64], &mut [f64], &mut [f64]),
        theta: f64,
    ) -> Result<(), ()> {
        let n = targets.0.len();

        for i in 0..n {
            let target = Vec3::from_slice_tuple(targets, i);
            let idx = 0;
            let _h = self.h_traverse(idx, &target, theta);
            h.0[i] += _h[0];
            h.1[i] += _h[1];
            h.2[i] += _h[2];
        }
        Ok(())
    }

    #[cfg(feature = "parallel")]
    pub fn h_field_parallel(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        h: (&mut [f64], &mut [f64], &mut [f64]),
        theta: f64,
        nthreads_requested: u32,
    ) -> Result<(), ()> {
        let n: usize = targets.0.len();
        let nthreads: usize = crate::biotsavart_parallel::get_nthreads(nthreads_requested);
        let chunk_size: usize = (n / nthreads).max(1);

        // chunk the inputs
        use rayon::prelude::*;
        let _x = targets.0.par_chunks(chunk_size);
        let _y = targets.1.par_chunks(chunk_size);
        let _z = targets.2.par_chunks(chunk_size);
        let _hx = h.0.par_chunks_mut(chunk_size);
        let _hy = h.1.par_chunks_mut(chunk_size);
        let _bz = h.2.par_chunks_mut(chunk_size);

        (_x, _y, _z, _hx, _hy, _bz).into_par_iter().try_for_each(
            |(_x, _y, _z, _hx, _hy, _hz)| self.h_field((_x, _y, _z), (_hx, _hy, _hz), theta),
        )?;

        Ok(())
    }
}

impl<S: GradHFieldSolver> Octree<S> {
    // need to figure out proper layout for gradh, should it be 9 mutable slices? or something like:
    // gradh: &mut [[Vec3; 3]]
    // perhaps it needs its own type?
    fn gradh_field(
        &self,
        targets: (&[f64], &[f64], &[f64]),
        gradh: (&mut [f64], &mut [f64], &mut [f64]),
    ) {
    }
}
