//! Transient solver using Barnes-Hut for inductance matrix acceleration
//!
//! This is a prototype feature that may have trouble with preconditioning until one is
//! implemented.
#![allow(unused, dead_code)]

use crate::{math::min_and_max, mesh::Mesh, octree::Octree};
use faer::{diag::Diag, sparse::SparseColMat};
use ndarray::{Array1, Array3};

struct KktSystem {
    r: Diag<f64>,
    g: SparseColMat<usize, f64>,
    octree: Octree,
    dt: f64,
}

impl KktSystem {
    // Returns the rhs of the KKT system
    fn matvec(&self, j_kp1: &[f64], phi_kp1: &[f64], out: &mut [f64]) {

        // top = r + bh(J^{k+1]})/dt + g * phi^{k+1}
        // bot = g^T * J^{k+1}
        // rhs = stack(top, bot)

        // bh(J) -> 3*Nel (Jx, Jy, Jz)
        // g -> (3*Nel, Nr)
        // Nr -> reduced number of nodes (after gauge pinning removes some DOF)
    }
}

pub fn solve(
    mesh: &Mesh,
    rho: &[f64],
    rho_indices: &[usize],
    nt: usize,
    tmax: f64,
    a_ext: &Array3<f64>,
    b_ext: &Array3<f64>,
    cyclic: bool,
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

    (time, j, a, b)
}