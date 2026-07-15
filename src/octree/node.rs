//! Functions for working with individual nodes in the tree

use crate::{morton, octree::bbox::BoundingBox, types::Vec3};

// Sentinel value to identify nodes who have not been assigned in the tree yet
pub const INVALID_NODE: u32 = u32::MAX;

/// Return the size of an octree node given the side length of the root
/// node and the level in the tree
pub fn size_at_level(side_length: f64, level: u8) -> f64 {
    side_length / (2f64.powi(level as i32))
}

/// Return the morton code prefix at a given level of the tree
#[inline]
#[allow(unused)]
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

// Return the morton code of each source in the octree
pub fn encode(centroids: &[Vec3], bbox: &BoundingBox, max_depth: u8) -> Vec<u64> {
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

pub fn encode_cols(
    centroids: (&[f64], &[f64], &[f64]),
    bbox: &BoundingBox,
    max_depth: u8,
) -> Vec<u64> {
    let (x, y, z) = centroids;
    let n = x.len();

    let mut codes: Vec<u64> = Vec::with_capacity(n);
    let scale: f64 = morton::calculate_scale_factor(max_depth as u32);
    let min_corner: (f64, f64, f64) = bbox.min_corner();

    for i in 0..n {
        codes.push(morton::encode(
            (x[i], y[i], z[i]),
            scale,
            bbox.side_length,
            min_corner,
        ));
    }
    codes
}
