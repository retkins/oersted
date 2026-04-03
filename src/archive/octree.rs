#![allow(unused)]
use crate::{
    math::{mag, min_and_max, sort_by_indices},
    morton,
    types::Vec3,
};

/// Determines the location and extent of a collection of source points
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    xc: f64,
    yc: f64,
    zc: f64,
    pub side_length: f64,
    xbounds: (f64, f64),
    ybounds: (f64, f64),
    zbounds: (f64, f64),
}

// Get the maximum side length of a bounding box cube that encloses all x,y,z bounds
fn side_length_from_bounds(xbounds: (f64, f64), ybounds: (f64, f64), zbounds: (f64, f64)) -> f64 {
    let xrange: f64 = xbounds.1 - xbounds.0;
    let yrange: f64 = ybounds.1 - ybounds.0;
    let zrange: f64 = zbounds.1 - zbounds.0;

    let xymax = match xrange > yrange {
        true => xrange,
        false => yrange,
    };

    match xymax > zrange {
        true => xymax,
        false => zrange,
    }
}

impl BoundingBox {
    pub fn from_centroids(centroids: (&[f64], &[f64], &[f64])) -> Option<Self> {
        // TODO: length check

        let xbounds: Option<(f64, f64)> = min_and_max(centroids.0);
        let ybounds: Option<(f64, f64)> = min_and_max(centroids.1);
        let zbounds: Option<(f64, f64)> = min_and_max(centroids.2);

        if xbounds.is_none() | ybounds.is_none() | zbounds.is_none() {
            return None;
        }

        let (xb, yb, zb) = (xbounds.unwrap(), ybounds.unwrap(), zbounds.unwrap());
        let side_length = side_length_from_bounds(xb, yb, zb);
        let xc: f64 = xb.0 + 0.5 * side_length;
        let yc: f64 = yb.0 + 0.5 * side_length;
        let zc: f64 = zb.0 + 0.5 * side_length;

        Some(Self {
            xc,
            yc,
            zc,
            side_length,
            xbounds: xb,
            ybounds: yb,
            zbounds: zb,
        })
    }

    /// TODO: fix this so there's no data copy
    pub fn from_centroids_vec(centroids: &[Vec3]) -> Self {
        let n: usize = centroids.len();
        let mut x: Vec<f64> = vec![0.0; n];
        let mut y: Vec<f64> = vec![0.0; n];
        let mut z: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            x[i] = centroids[i][0];
            y[i] = centroids[i][1];
            z[i] = centroids[i][2];
        }

        Self::from_centroids((&x, &y, &z)).unwrap()
    }

    pub fn min_corner(&self) -> (f64, f64, f64) {
        (self.xbounds.0, self.ybounds.0, self.zbounds.0)
    }
}

/// A collection of sources that can be sorted by morton codes
pub struct Sources {
    pub codes: Vec<u64>, // Morton code for each source point
    pub xg: Vec<f64>,    // Geometric location of the source point
    pub yg: Vec<f64>,
    pub zg: Vec<f64>,
    pub vjx: Vec<f64>, // Current density moment product
    pub vjy: Vec<f64>,
    pub vjz: Vec<f64>,
    pub bbox: Option<BoundingBox>,
}

impl Sources {
    /// Basic constructor; zeroed with initial size
    pub fn new(n: usize) -> Self {
        Self {
            codes: Vec::with_capacity(n),
            xg: Vec::with_capacity(n),
            yg: Vec::with_capacity(n),
            zg: Vec::with_capacity(n),
            vjx: Vec::with_capacity(n),
            vjy: Vec::with_capacity(n),
            vjz: Vec::with_capacity(n),
            bbox: None,
        }
    }

    /// Add source points (elements) to the octree
    ///
    /// Note that this involves a data copy. We're going to be sorting the arrays
    /// and need to own the memory. (?)
    pub fn from_source_points(
        centroids: (&[f64], &[f64], &[f64]),
        volumes: &[f64],
        jdensity: (&[f64], &[f64], &[f64]),
        max_depth: u8,
    ) -> Option<Self> {
        // TODO: length check on all vectors
        let n: usize = centroids.0.len();

        let _bbox: Option<BoundingBox> = BoundingBox::from_centroids(centroids);
        _bbox?;
        let bbox = _bbox.unwrap();

        let (x, y, z) = centroids;
        let (jx, jy, jz) = jdensity;
        let mut codes: Vec<u64> = Vec::with_capacity(n);
        let mut xg = x.to_vec();
        let mut yg = y.to_vec();
        let mut zg = z.to_vec();
        let mut vjx: Vec<f64> = Vec::with_capacity(n);
        let mut vjy: Vec<f64> = Vec::with_capacity(n);
        let mut vjz: Vec<f64> = Vec::with_capacity(n);

        let scale: f64 = morton::calculate_scale_factor(max_depth as u32);
        let min_corner: (f64, f64, f64) = bbox.min_corner();

        for i in 0..n {
            let pt: (f64, f64, f64) = (x[i], y[i], z[i]);
            codes.push(morton::encode(pt, scale, bbox.side_length, min_corner));
        }
        for i in 0..n {
            let vol: f64 = volumes[i];
            vjx.push(vol * jx[i]);
            vjy.push(vol * jy[i]);
            vjz.push(vol * jz[i]);
        }

        Some(Sources {
            codes,
            xg,
            yg,
            zg,
            vjx,
            vjy,
            vjz,
            bbox: Some(bbox),
        })
    }

    /// Sort the octree by morton order
    pub fn morton_sort(&mut self) {
        // Get the indices to sort by from the codes
        let mut indices: Vec<usize> = (0..self.codes.len()).collect();
        indices.sort_by(|&i, &j| self.codes[i].cmp(&self.codes[j]));

        // Now sort all of the other data arrays
        let mut scratch = vec![0.0; self.codes.len()];

        sort_by_indices(&mut self.xg, &mut scratch, &indices);
        sort_by_indices(&mut self.yg, &mut scratch, &indices);
        sort_by_indices(&mut self.zg, &mut scratch, &indices);
        sort_by_indices(&mut self.vjx, &mut scratch, &indices);
        sort_by_indices(&mut self.vjy, &mut scratch, &indices);
        sort_by_indices(&mut self.vjz, &mut scratch, &indices);

        // Now sort the codes themselves
        let mut scratch_codes: Vec<u64> = vec![0; self.codes.len()];
        sort_by_indices(&mut self.codes, &mut scratch_codes, &indices);
    }

    /// Get the centroid of a particular Source
    pub fn centroid(&self, idx: u32) -> [f64; 3] {
        let i = idx as usize;
        [self.xg[i], self.yg[i], self.zg[i]]
    }

    pub fn vj(&self, idx: u32) -> [f64; 3] {
        let i = idx as usize;
        [self.vjx[i], self.vjy[i], self.vjz[i]]
    }
}

pub enum SourceNode {
    Branch {
        level: u8,
        size: f64, // or maybe f32?
        children: [u32; 8],
        centroid: [f64; 3], // of the cluster of sources it represents
        vj: [f64; 3],
    },
    Leaf {
        level: u8, // maybe unecessary but harmless
        source_range: (u32, u32),
        centroid: [f64; 3],
    },
}

pub struct SourceOctree {
    max_depth: u8,
    // we use sources to build the octree and then do not need it again
    pub sources: Sources,
    pub nodes: Vec<SourceNode>,
    leaf_threshold: u32,
}

impl SourceOctree {
    /// Construct a new octree using source point definitions
    pub fn from_source_points(
        centroids: (&[f64], &[f64], &[f64]),
        volumes: &[f64],
        jdensity: (&[f64], &[f64], &[f64]),
        max_depth: u8,
        leaf_threshold: u32,
    ) -> Option<Self> {
        // TODO: length check

        let n_sources: usize = centroids.0.len();
        let mut sources = Sources::from_source_points(centroids, volumes, jdensity, max_depth)?;
        sources.morton_sort();

        // Memory usage will be significantly more than n_sources, as
        // every source point has a leaf node, but this is a good place to start
        let mut nodes: Vec<SourceNode> = Vec::with_capacity(n_sources);

        build_tree(&sources, &mut nodes, max_depth, leaf_threshold);

        Some(Self {
            max_depth,
            sources,
            nodes,
            leaf_threshold,
        })
    }
}

/// Build the octree by making recursive calls to `add_node()`
fn build_tree(sources: &Sources, nodes: &mut Vec<SourceNode>, max_depth: u8, leaf_threshold: u32) {
    // Recursively build the tree, starting at the beginning and at the root node
    let start: usize = 0;
    let end: usize = sources.codes.len();
    let level: u8 = 0;
    let _ = add_node(sources, nodes, max_depth, leaf_threshold, start, end, level);
}

/// Recursively add nodes to the octree
///
/// Returns: the index of the node that it created
fn add_node(
    sources: &Sources,
    nodes: &mut Vec<SourceNode>,
    max_depth: u8,
    leaf_threshold: u32,
    start: usize,
    end: usize,
    level: u8,
) -> u32 {
    let current_index: usize;

    // Create a leaf if the number of sources in the range is small enough (dead-end for recursion)
    if end - start <= leaf_threshold as usize {
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for i in start..end {
            cx += sources.xg[i];
            cy += sources.yg[i];
            cz += sources.zg[i];
        }
        cx /= (end - start) as f64;
        cy /= (end - start) as f64;
        cz /= (end - start) as f64;
        nodes.push(
            // Do not descend level
            SourceNode::Leaf {
                level,
                source_range: (start as u32, end as u32),
                centroid: [cx, cy, cz],
            },
        );
        current_index = nodes.len() - 1;
    }
    // Otherwise, create an branch node
    else {
        // Size of the node (cube side length) for BH acceptance criteria
        let size: f64 = size_at_level(sources.bbox.as_ref().unwrap().side_length, level);

        // Initialize the branch node first, then recursive calls later fill it
        nodes.push(SourceNode::Branch {
            level,
            size,
            children: [0; 8],
            centroid: [0.0; 3],
            vj: [0.0; 3],
        });

        current_index = nodes.len() - 1; // index of the node just created

        // Now recurse to create its children
        let mut child_indices: [u32; 8] = [0; 8];
        let mut n_children: usize = 0;
        let mut cursor = start;

        while cursor < end {
            // We descend to next level here, hence `level+1`
            // index to sources array
            let child_end = get_range_in_same_node(&sources.codes, level + 1, max_depth, cursor);

            // index in nodes array
            let child_idx = add_node(
                sources,
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
        let mut parent_centroid: [f64; 3] = [0.0; 3];
        let mut parent_vj: [f64; 3] = [0.0; 3];

        for idx in child_indices {
            if idx < 1 {
                // Skip empty nodes
                break;
            }

            let mut parent_mag: f64 = mag(&parent_vj);

            match nodes[idx as usize] {
                SourceNode::Branch {
                    level: _,
                    size: _,
                    children: _,
                    centroid,
                    vj,
                } => {
                    update_centroid(&mut parent_centroid, parent_mag, &centroid, mag(&vj));
                    for k in 0..3_usize {
                        parent_vj[k] += vj[k];
                    }
                }
                SourceNode::Leaf {
                    level: _,
                    source_range,
                    centroid,
                } => {
                    for _i in source_range.0..source_range.1 {
                        let i = _i as usize;
                        let centroid = [sources.xg[i], sources.yg[i], sources.zg[i]];
                        let vj = [sources.vjx[i], sources.vjy[i], sources.vjz[i]];
                        update_centroid(&mut parent_centroid, parent_mag, &centroid, mag(&vj));
                        for k in 0..3_usize {
                            parent_vj[k] += vj[k];
                        }
                        parent_mag = mag(&parent_vj);
                    }
                }
            }
        }

        if let SourceNode::Branch {
            children,
            centroid,
            vj,
            ..
        } = &mut nodes[current_index]
        {
            *children = child_indices;
            *centroid = parent_centroid;
            *vj = parent_vj;
        }
    }

    current_index as u32
}

/// Update the parent centroid by weighting each of the axes
pub fn update_centroid(
    parent_centroid: &mut [f64; 3],
    parent_mag: f64,
    child_centroid: &[f64; 3],
    child_mag: f64,
) {
    let total_mag: f64 = parent_mag + child_mag;
    for i in 0..3_usize {
        parent_centroid[i] =
            (parent_centroid[i] * parent_mag + child_centroid[i] * child_mag) / total_mag;
    }
}

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

pub fn size_at_level(side_length: f64, level: u8) -> f64 {
    side_length / (2f64.powi(level as i32))
}

#[cfg(test)]
mod tests {

    use crate::sources;

    use super::*;

    #[test]
    fn test_octree_build_1() {
        // test along the diagonal
        let n = 5;
        let x = vec![-0.2, -0.1, 0.0, 0.1, 0.2];
        let y = vec![-0.2, -0.1, 0.0, 0.1, 0.2];
        let z = vec![-0.2, -0.1, 0.0, 0.1, 0.2];
        let volumes = vec![1e-4; n];
        let jx = vec![1e8; n];
        let jy = vec![-1e8; n];
        let jz = vec![2e8; n];
        let max_depth: u8 = 21;

        let mut tree =
            Sources::from_source_points((&x, &y, &z), &volumes, (&jx, &jy, &jz), max_depth)
                .unwrap();

        println!("Bounding box: {:?}", tree.bbox);
        println!("Max depth: {}", 21);
        println!("Morton codes:");
        for (i, code) in tree.codes.iter().enumerate() {
            println!("Point: ({:.2}, {:.2}, {:.2})", x[i], y[i], z[i]);
            println!("\tCode (decimal): {}", code);
            println!("\tCode (binary):  {:064b}", code);
            println!("\tCode (hex):     0x{:016x}", code);
        }
        tree.morton_sort();
        println!("After sorting: ");
        for (i, code) in tree.codes.iter().enumerate() {
            println!("Point: ({:.2}, {:.2}, {:.2})", x[i], y[i], z[i]);
            println!("\tCode (decimal): {}", code);
            println!("\tCode (binary):  {:064b}", code);
            println!("\tCode (hex):     0x{:016x}", code);
        }
    }

    #[test]
    fn test_octree_build_2() {
        // keep xy=0, increase z
        let n = 5;
        let x = vec![0.0; n];
        let y = vec![0.0; n];
        let z = vec![-0.2, -0.1, 0.0, 0.1, 0.2];
        let volumes = vec![1e-4; n];
        let jx = vec![1e8; n];
        let jy = vec![-1e8; n];
        let jz = vec![2e8; n];
        let max_depth: u8 = 21;

        let mut tree =
            Sources::from_source_points((&x, &y, &z), &volumes, (&jx, &jy, &jz), max_depth)
                .unwrap();

        println!("Bounding box: {:?}", tree.bbox);
        println!("Max depth: {}", 21);
        println!("Morton codes:");
        for (i, code) in tree.codes.iter().enumerate() {
            println!("Point: ({:.2}, {:.2}, {:.2})", x[i], y[i], z[i]);
            println!("\tCode (decimal): {}", code);
            println!("\tCode (binary):  {:064b}", code);
            println!("\tCode (hex):     0x{:016x}", code);
        }

        tree.morton_sort();
        println!("After sorting: ");
        for (i, code) in tree.codes.iter().enumerate() {
            println!("Point: ({:.2}, {:.2}, {:.2})", x[i], y[i], z[i]);
            println!("\tCode (decimal): {}", code);
            println!("\tCode (binary):  {:064b}", code);
            println!("\tCode (hex):     0x{:016x}", code);
        }
    }

    // Test the octree build with 7 source points: 5 within the domain, 2 on the extreme corners.
    #[test]
    fn test_tree_build() {
        let n: usize = 7;
        let x = vec![0.0, 0.4, 0.9, 0.6, 0.6, 0.1, 1.0];
        let y = vec![0.0, 0.2, 0.4, 0.6, 0.9, 0.7, 1.0];
        let z = vec![0.0, 0.6, 0.6, 0.6, 0.6, 0.6, 1.0];
        let volumes = vec![1e-4; n];
        let jx = vec![1e8; n];
        let jy = vec![2e7; n];
        let jz = vec![3e6; n];

        let max_depth: u8 = 21;
        let leaf_threshold: u32 = 1;
        let tree = SourceOctree::from_source_points(
            (&x, &y, &z),
            &volumes,
            (&jx, &jy, &jz),
            max_depth,
            leaf_threshold,
        )
        .unwrap();

        for node in &tree.nodes {
            match node {
                SourceNode::Branch {
                    level,
                    size,
                    children,
                    centroid,
                    vj,
                } => {
                    println!(
                        "BRANCH on level: {}, size: {}, children: {:?}, centroid: {:?}, vj: {:?}",
                        level, size, children, centroid, vj
                    );
                }
                SourceNode::Leaf {
                    level,
                    source_range,
                    centroid,
                } => {
                    let i = source_range.0 as usize;
                    let (x, y, z) = (tree.sources.xg[i], tree.sources.yg[i], tree.sources.zg[i]);
                    println!(
                        "LEAF on level: {}, source range: ({:?}); loc: ({:.1}, {:.1}, {:.1})",
                        level, source_range, x, y, z
                    );
                }
            }
        }

        let (m_centroid, m_vj) = sources::monopole((&x, &y, &z), &volumes, (&jx, &jy, &jz));
        println!(
            "Monopole approximation: \n\tCentroids: {:?}, vj: {:?}",
            m_centroid, m_vj
        );

        let root: &SourceNode = &tree.nodes[0];
        if let SourceNode::Branch { centroid, vj, .. } = root {
            assert_eq!((centroid[0], centroid[1], centroid[2]), m_centroid);
            // assert_eq!(centroid[1], m_centroid.1);
            // assert_eq!(centroid[2], m_centroid.2);
            assert_eq!((vj[0], vj[1], vj[2]), m_vj)
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_prefix_find() {
        let n: usize = 7;
        let x = vec![0.0, 0.4, 0.9, 0.6, 0.6, 0.1, 1.0];
        let y = vec![0.0, 0.2, 0.4, 0.6, 0.9, 0.7, 1.0];
        let z = vec![0.0, 0.6, 0.6, 0.6, 0.6, 0.6, 1.0];
        let volumes = vec![1e-4; n];
        let jx = vec![1e8; n];
        let jy = vec![2e7; n];
        let jz = vec![3e6; n];

        let max_depth: u8 = 21;
        let leaf_threshold: u32 = 1;
        let tree = SourceOctree::from_source_points(
            (&x, &y, &z),
            &volumes,
            (&jx, &jy, &jz),
            max_depth,
            leaf_threshold,
        )
        .unwrap();

        // tree.sources.morton_sort();
        let mut idx = 0;
        let level: u8 = 1;
        let mut i = 0;
        while idx < tree.sources.codes.len() {
            let end_idx = get_range_in_same_node(&tree.sources.codes, level, max_depth, idx);
            let count = end_idx - idx;
            let prefix: u64 = get_prefix(tree.sources.codes[idx], max_depth, level);
            println!(
                "Octant {}: prefix: {:b}, num child nodes: {}",
                i, prefix, count
            );
            i += 1;
            idx = end_idx;
        }
    }
}
