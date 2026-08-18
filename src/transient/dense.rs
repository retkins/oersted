//! Basic dense transient solver
//!
//! This solve forms a fully dense interaction matrix and therefore should only be used
//! for relatively small systems (<10k elements)
//!

use faer::{diag::Diag, mat::Mat, sparse::SparseColMat};
use ndarray::{Array1, Array3};

use crate::{
    biotsavart::{IntegrationMethod, Kernel, SourceVectors, a_field},
    mesh::Mesh,
    types::{Vec3, vec3_to_3vec},
};

/// Solve a transient problem
pub fn solve(
    mesh: &Mesh,
    rho: f64,
    nt: usize,
    tmax: f64,
    a_ext: &Array3<f64>,
    b_ext: &Array3<f64>,
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    let n_elem: usize = mesh.connectivity.len();

    // Allocate memory for the time steps and the results data
    // a and b are the TOTAL value at element centroids, including the external
    // sources. Overwriting the external source arrays would save memory, but it
    // may not be what the caller wants to do.
    let mut time: Array1<f64> = Array1::zeros(nt);
    let mut j: Array3<f64> = Array3::zeros((nt, n_elem, 3));
    let mut a: Array3<f64> = Array3::zeros((nt, n_elem, 3));
    let mut b: Array3<f64> = Array3::zeros((nt, n_elem, 3));

    //

    // Assembly
    let r = assemble_r(rho, mesh);
    let g = assemble_g(mesh);
    let (m, asym_m) = assemble_m(mesh);

    (time, j, a, b)
}

// Assemble the constraint-gradient matrix G
//
// This matrix is 3*num_elems x num_nodes. The first num_elems rows are for the
// x-dof, second num_elems (second third) rows are for y-dof, etc.
//
// To save on memory, G is saved as COO triplets (sparse) and never formed into its own
// array. Instead, it is scattered into the KKT system directly.
fn assemble_g(mesh: &Mesh) -> Vec<(usize, usize, f64)> {
    // let mut g = Mat::<f64>::zeros(3*mesh.n_elems(), mesh.n_nodes());
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(12 * mesh.n_elems());

    for e in 0..mesh.n_elems() {
        let vg_e: [Vec3; 4] = mesh.hat_gradients(e);

        for ni in 0..4usize {
            let n: usize = mesh.connectivity[e][ni] as usize;
            for k in 0..3usize {
                triplets.push((mesh.n_elems() * k + e, n, vg_e[ni][k]));
            }
        }
    }
    triplets
}

// Assemble the resistance diagonal matrix R
//
// This matrix has length `n_elems`, each of which are rho*vol[e]
fn assemble_r(rho: f64, mesh: &Mesh) -> Diag<f64> {
    let mut r = Diag::zeros(mesh.n_elems());
    for i in 0..mesh.n_elems() {
        r[i] = rho * mesh.volumes[i];
    }
    r
}

// Assemble the inductance matrix M
//
// This matrix is fully dense, as each element couples to every other element
// Returns the symmetrized matrix M and the asymmetry value of the unsymmetrized M
fn assemble_m(mesh: &Mesh) -> (Mat<f64>, f64) {
    let n_el: usize = mesh.n_elems();
    let mut m = Mat::<f64>::zeros(n_el, n_el);

    // Allocate memory for the a-field vectors (unfortunately only ax is used), and
    // the target positions
    let (mut ax, mut ay, mut az) = (vec![0.0; n_el], vec![0.0; n_el], vec![0.0; n_el]);
    let (tx, ty, tz) = vec3_to_3vec(mesh.centroids());

    // For every element, compute the effect of that element at every other element,
    // using unity current density in x-direction. Only the value of ax is then used
    // in the matrix M
    for e in 0..mesh.n_elems() {
        let elem_nodes = mesh.elem_nodes(e);

        a_field(
            &elem_nodes,
            &[[0u32, 1u32, 2u32, 3u32]],
            SourceVectors::CurrentDensity(&[Vec3([1.0, 0.0, 0.0])]),
            (&tx, &ty, &tz),
            (&mut ax, &mut ay, &mut az),
            IntegrationMethod::Element,
            0,
        );
        for r in 0..n_el {
            m[(r, e)] = ax[r] * mesh.volumes()[r];
        }

        // Zero buffer to prevent accumulation between `a_field` calls
        ax.fill(0.0);
        ay.fill(0.0);
        az.fill(0.0);
    }

    // Compute asymmetry properties of M
    let asym: f64 = (&m - &m.transpose()).norm_l2() / m.norm_l2();

    (0.5 * (&m + &m.transpose()), asym)
}

// Ground nodes by identifying separated bodies and returning one node index per body
//
// TODO: this version assumes the mesh represents a single body and therefore just 
// grounds the first node
fn ground_nodes(mesh: &Mesh) -> Vec<usize> {
    let grounded = vec![0;1];
    grounded
}
