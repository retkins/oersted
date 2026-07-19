//! Methods and data structures for using the octree to evaluate fields problems

use crate::{
    math::sort_by_indices,
    octree::{
        BoundingBox, Sources, Topology, TreeMoments,
        node::{INVALID_NODE, encode_cols},
    },
    types::Vec3,
};

pub struct InteractionList {
    pub data: Vec<u32>,
    pub offsets: Vec<u32>,
    pub buf_len: usize,
}

// Sort targets into morton order
//
// TODO: make this a 'plan' so that for iterative solves, this doesn't have to
// happen every time
pub fn sort_targets(
    targets: (&[f64], &[f64], &[f64]),
) -> ((Vec<f64>, Vec<f64>, Vec<f64>), Vec<usize>) {
    let n = targets.0.len();
    let bbox = BoundingBox::from_centroids((targets.0, targets.1, targets.2)).unwrap();
    let max_depth: u8 = 21;
    let codes = encode_cols(targets, &bbox, max_depth);

    let (mut x, mut y, mut z) = (targets.0.to_vec(), targets.1.to_vec(), targets.2.to_vec());

    // Sort the morton codes and the source data
    let mut unsorted_to_sorted: Vec<usize> = (0..n).collect();
    unsorted_to_sorted.sort_by(|&i, &j| codes[i].cmp(&codes[j]));
    let mut scratch = vec![0.0; n];
    sort_by_indices(&mut x, &mut scratch, &unsorted_to_sorted);
    sort_by_indices(&mut y, &mut scratch, &unsorted_to_sorted);
    sort_by_indices(&mut z, &mut scratch, &unsorted_to_sorted);
    ((x, y, z), unsorted_to_sorted)
}

/// Return calculated fields values into the order the caller expects them
/// Accumulates `fields` back into `out`
pub fn unsort_fields(
    fields: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
    indices: &[usize],
) {
    let n = fields.0.len();
    for sorted_i in 0..n {
        let original_i: usize = indices[sorted_i];
        out.0[original_i] += fields.0[sorted_i];
        out.1[original_i] += fields.1[sorted_i];
        out.2[original_i] += fields.2[sorted_i];
    }
}

// Traverse the tree and return (near, far) interaction lists
pub fn traverse_tree(
    topology: &Topology,
    sources: &Sources,
    tree_moments: &TreeMoments,
    targets: (&[f64], &[f64], &[f64]),
    theta: f64,
) -> (InteractionList, InteractionList) {
    let (x, y, z) = targets;
    let n_targets = x.len();
    let n_sources = sources.len();
    let n_nodes = topology.len();

    let mut near_counts = vec![0u32; n_sources];
    let mut far_counts = vec![0u32; n_nodes];

    // Tree source nodes
    let mut stack: Vec<u32> = Vec::with_capacity(128);

    // Count first
    for ti in 0..n_targets {
        stack.push(0); // start at root 

        while let Some(ni) = stack.pop() {
            let ni = ni as usize;

            let c = tree_moments.centers[ni];
            let t = Vec3([x[ti], y[ti], z[ti]]);
            let d = (t - c).mag();

            if tree_moments.bmax[ni] < theta * d {
                // Branch node accepted
                far_counts[ni] += 1;
            } else if topology.is_leaf[ni] {
                let (start, end) = topology.source_range[ni];
                for e in start as usize..end as usize {
                    near_counts[e] += 1;
                }
            } else {
                // Add the child nodes to the stack
                for child in topology.children[ni] {
                    if child == INVALID_NODE {
                        continue;
                    } else {
                        stack.push(child)
                    }
                }
            }
        }
        stack.clear();
    }

    // Create offsets
    let mut near_offsets = vec![0u32; n_sources + 1];
    let mut far_offsets = vec![0u32; n_nodes + 1];

    let mut running = 0u32;
    for e in 0..n_sources {
        near_offsets[e] = running;
        running += near_counts[e];
    }
    near_offsets[n_sources] = running;
    let mut near_data = vec![0u32; running as usize];

    running = 0u32;
    for ni in 0..n_nodes {
        far_offsets[ni] = running;
        running += far_counts[ni];
    }
    far_offsets[n_nodes] = running;
    let mut far_data = vec![0u32; running as usize];

    // Fill interaction list data by re-traversal
    let mut near_cursor = near_offsets.clone();
    let mut far_cursor = far_offsets.clone();

    stack.clear();
    for ti in 0..n_targets {
        stack.push(0u32);
        while let Some(ni) = stack.pop() {
            let ni = ni as usize;

            let c = tree_moments.centers[ni];
            let t = Vec3([x[ti], y[ti], z[ti]]);
            let d = (t - c).mag();

            if tree_moments.bmax[ni] < theta * d {
                // Branch node accepted
                far_data[far_cursor[ni] as usize] = ti as u32;
                far_cursor[ni] += 1;
            } else if topology.is_leaf[ni] {
                let (start, end) = topology.source_range[ni];
                for e in start as usize..end as usize {
                    near_data[near_cursor[e] as usize] = ti as u32;
                    near_cursor[e] += 1;
                }
            } else {
                // Add the child nodes to the stack
                for child in topology.children[ni] {
                    if child == INVALID_NODE {
                        continue;
                    } else {
                        stack.push(child)
                    }
                }
            }
        }
        stack.clear();
    }

    (
        InteractionList {
            data: near_data,
            offsets: near_offsets,
            buf_len: *near_counts.iter().max().unwrap() as usize,
        },
        InteractionList {
            data: far_data,
            offsets: far_offsets,
            buf_len: *far_counts.iter().max().unwrap() as usize,
        },
    )
}
