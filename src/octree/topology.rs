use crate::{
    check_lengths,
    octree::{BoundingBox, INVALID_NODE, Sources, get_range_in_same_node, size_at_level},
    types::Vec3,
};

// Topology information for traversing the octree
#[derive(Debug)]
pub struct Topology {
    // Each vector has length Nnodes
    pub children: Vec<[u32; 8]>, // provides indices into these arrrays (tree connectivity)
    pub centroids: Vec<Vec3>,
    pub volumes: Vec<f64>,
    pub sizes: Vec<f64>,
    pub source_range: Vec<(u32, u32)>,
    pub is_leaf: Vec<bool>,
    pub max_depth: u8,
}

impl Topology {
    pub fn new(
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
    pub fn len(&self) -> usize {
        // Lengths of all vectors were checked at build, so we can choose any
        // of the member data
        self.children.len()
    }
}

// Build the internal structure of the tree, using a top down approach
pub fn build_topology(
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
                let child_end =
                    get_range_in_same_node(codes, level + 1, max_depth, cursor, range_end as usize);

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

    Topology::new(
        children,
        centroids,
        volumes,
        sizes,
        source_range,
        is_leaf,
        max_depth,
    )
}
