//! Methods and data structures for using the octree to evaluate fields problems

use crate::{
    math::sort_by_indices,
    octree::{BoundingBox, node::encode_cols},
    types::Vec3,
};

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

/// Compute the centroid and bounding radius of a collection of target points
pub fn target_bounds(targets: (&[f64], &[f64], &[f64])) -> (Vec3, f64) {
    let mut centroid: Vec3 = Vec3::default();
    let mut bounding_radius: f64 = 0.0;
    let (x, y, z) = targets;
    let n: usize = x.len();

    for i in 0..n {
        centroid[0] += x[i];
        centroid[1] += y[i];
        centroid[2] += z[i];
    }

    centroid /= n as f64;
    for i in 0..n {
        let target = Vec3([x[i], y[i], z[i]]);
        bounding_radius = bounding_radius.max((target - centroid).mag());
    }

    (centroid, bounding_radius)
}
