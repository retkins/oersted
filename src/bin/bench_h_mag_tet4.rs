use oersted::biotsavart::{IntegrationMethod, SourceVectors, h_field};
use oersted::io::{CsvData, read_csv};
use oersted::types::Vec3;

// Run an example benchmark without Python
// This is a test for only speed, not accuracy
// The benchmark is the loop magnet STEP file meshed with 40mm elements
//
// Build with:
//  cargo build --release --features=parallel --bin bench_h_mag_tet4
// Run with:
//  ./target/release/bench_h_mag_tet4
// or using your favorite profiler
fn main() {
    let node_data: CsvData = read_csv("tests/data/bench.nodes", ',', 0).unwrap();
    let conn_data: CsvData = read_csv("tests/data/bench.connectivity", ',', 0).unwrap();

    let n_elem = conn_data.nrows();
    let n_nodes = node_data.nrows();

    let mut connectivity: Vec<[u32; 4]> = vec![[0u32; 4]; n_elem];
    let mut nodes: Vec<Vec3> = vec![Vec3::default(); n_nodes];

    for (i, elem) in conn_data.data().chunks_exact(4).enumerate() {
        for (j, e) in elem.iter().enumerate() {
            connectivity[i][j] = *e as u32;
        }
    }

    for (i, node) in node_data.data().chunks_exact(3).enumerate() {
        nodes[i] = Vec3([node[0], node[1], node[2]]);
    }

    // Targets are the nodes on the body
    let mut targets = (vec![0.0; n_nodes], vec![0.0; n_nodes], vec![0.0; n_nodes]);
    for (i, node) in nodes.iter().enumerate() {
        targets.0[i] = node[0];
        targets.1[i] = node[1];
        targets.2[i] = node[2];
    }

    // Zero the Mvectors for now (hopefully this doesn't impact branch prediction...)
    let mvectors = vec![Vec3::default(); n_elem];
    let mut out = (vec![0.0; n_elem], vec![0.0; n_elem], vec![0.0; n_elem]);

    let start = std::time::Instant::now();
    let n_iter = 5;
    for _ in 0..n_iter {
        h_field(
            &nodes,
            &connectivity,
            SourceVectors::CurrentDensity(&mvectors),
            (&targets.0, &targets.1, &targets.2),
            (&mut out.0, &mut out.1, &mut out.2),
            IntegrationMethod::Element,
            0,
        );
    }
    let interactions = n_elem * targets.0.len();
    let elapsed = start.elapsed();
    let throughput = (n_iter as f64) * (interactions as f64) / elapsed.as_secs_f64();
    println!(
        "\nRunning h_mag_tet4() benchmark for {} iterations...\n",
        n_iter
    );
    println!("Sources: {}\nTargets: {}", n_elem, targets.0.len());
    println!("Interactions: {:.3e}", interactions);
    println!("Elapsed: {:?}", elapsed);
    println!("Interactions/sec: {:.3e}", throughput);
}
