//! Basic dense transient solver
//!
//! This solve forms a fully dense interaction matrix and therefore should only be used
//! for relatively small systems (<10k elements)
//!
#![allow(unused)]

use faer::{
    Col, Scale,
    diag::Diag,
    linalg::solvers::{PartialPivLu, Solve},
    mat::Mat,
    sparse::SparseColMat,
};
use ndarray::{Array1, Array3};

use crate::{
    biotsavart::{IntegrationMethod, Kernel, SourceVectors, a_field},
    mesh::Mesh,
    types::{Vec3, vec3_to_3vec},
};

type Triplets = Vec<(usize, usize, f64)>;

/// Solve a transient problem
pub fn solve(
    mesh: &Mesh,
    rho: f64,
    nt: usize,
    tmax: f64,
    a_ext: &Array3<f64>,
    b_ext: &Array3<f64>,
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    let n_elem: usize = mesh.n_elems();
    let size = 3 * n_elem + mesh.n_nodes();
    let vols = mesh.volumes();

    let dt: f64 = tmax / (nt - 1) as f64;

    // Allocate memory for the time steps and the results data
    // a and b are the TOTAL value at element centroids, including the external
    // sources. Overwriting the external source arrays would save memory, but it
    // may not be what the caller wants to do.
    let mut time: Array1<f64> = Array1::zeros(nt);
    let mut j: Array3<f64> = Array3::zeros((nt, n_elem, 3));
    let mut a: Array3<f64> = Array3::zeros((nt, n_elem, 3));
    let mut b: Array3<f64> = Array3::zeros((nt, n_elem, 3));

    // Assembly
    println!("Assembling matrices");
    let r = assemble_r(rho, mesh);
    let g: Triplets = assemble_g(mesh);
    let (m, asym_m) = assemble_m(mesh);
    let grounded: Vec<usize> = ground_nodes(mesh);
    let k = assemble_kkt(mesh, &m, &g, &r, dt, &grounded);

    // Factorize the KKT system
    println!("Factorizing KKT system");
    let lu: PartialPivLu<f64> = k.partial_piv_lu();

    // Buffers reused at every step: rhs, J^k, (M/dt)*J^k
    // These are stored component-major: i.e. all x's, all y's, then all z's
    let mut rhs = Col::<f64>::zeros(size);
    let mut j_prev = Mat::<f64>::zeros(n_elem, 3);
    let mut mj = Mat::<f64>::zeros(n_elem, 3);

    // Initial conditions at time = 0.0
    for e in 0..n_elem {
        for c in 0..3 {
            a[[0, e, c]] = a_ext[[0, e, c]];
            b[[0, e, c]] = b_ext[[0, e, c]];
        }
    }

    for t in 1..nt {
        time[t] = t as f64 * dt;
        println!("Solving timestep {} of {}", t, nt);

        // Compute the rhs of the system
        // Momentum block: (M/dt)*J^k - V_e * da_ext/dt
        mj = m.as_ref() * j_prev.as_ref();

        for c in 0..3 {
            for e in 0..n_elem {
                let dadt = (a_ext[[t, e, c]] - a_ext[[t - 1, e, c]]) / dt;
                rhs[c * n_elem + e] = mj[(e, c)] / dt - vols[e] * dadt;
            }
        }

        // Solve the system
        let x = lu.solve(&rhs);

        // Store J from this time step
        for c in 0..3 {
            for e in 0..n_elem {
                let j_kp1 = x[c * n_elem + e];
                j[[t, e, c]] = j_kp1;
                j_prev[(e, c)] = j_kp1;
            }
        }

        // TODO: compute self a/b fields and store
    }

    (time, j, a, b)
}

// Assemble the constraint-gradient matrix G
//
// This matrix is 3*num_elems x num_nodes. The first num_elems rows are for the
// x-dof, second num_elems (second third) rows are for y-dof, etc.
//
// To save on memory, G is saved as COO triplets (sparse) and never formed into its own
// array. Instead, it is scattered into the KKT system directly.
fn assemble_g(mesh: &Mesh) -> Triplets {
    // let mut g = Mat::<f64>::zeros(3*mesh.n_elems(), mesh.n_nodes());
    let mut triplets: Triplets = Vec::with_capacity(12 * mesh.n_elems());

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
    let asym: f64 = (&m - m.transpose()).norm_l2() / m.norm_l2();

    (0.5 * (&m + m.transpose()), asym)
}

// Ground nodes by identifying separated bodies and returning one node index per body
//
// TODO: this version assumes the mesh represents a single body and therefore just
// grounds the first node
fn ground_nodes(mesh: &Mesh) -> Vec<usize> {
    let grounded: Vec<usize> = vec![0; 1];
    grounded
}

// Assemble the KKT system as a dense square matrix
fn assemble_kkt(
    mesh: &Mesh,
    m: &Mat<f64>,
    g: &Triplets,
    r: &Diag<f64>,
    dt: f64,
    grounded: &[usize],
) -> Mat<f64> {
    let n_el: usize = mesh.n_elems();
    let size: usize = 3 * n_el + mesh.n_nodes();
    let mut k = Mat::<f64>::zeros(size, size);

    // Upper left: R + M/dt, but in 3x blocks since R and M are scalar
    let mut a = Scale(1.0 / dt) * m.as_ref();
    for e in 0..n_el {
        a[(e, e)] += r[e];
    }

    // Insert R + M/dt
    for comp in 0..3 {
        let start = comp * mesh.n_elems();
        let stride = mesh.n_elems();
        k.submatrix_mut(start, start, stride, stride).copy_from(&a);
    }

    // Insert G and G^T
    for &(row, col, val) in g.iter() {
        k[(row, col + 3 * n_el)] = val;
        k[(col + 3 * n_el, row)] = val;
    }

    // Pin grounded nodes
    for &p in grounded {
        let i = p + 3 * n_el;
        for j in 0..size {
            k[(i, j)] = 0.0;
            k[(j, i)] = 0.0;
        }
        k[(i, i)] = 1.0;
    }

    k
}
