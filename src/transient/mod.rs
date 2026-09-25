//! Transient (time-domain) eddy-current solver for oersted
//!
//! This solver assumes the following:
//! 1. The problem is discretized into a mesh consisting of 4-node tetrahedral elements.
//! 2. The DOF are phi (defined piecewise linear (P1 basis) on the nodes, and
//!    current density (J, A/m^2) defined piecewise constant (P0 basis) on the elements.
//! 3. phi is a lagrange multiplier that enforced div J = 0 (eliminates the cohomology problem).

mod common;
mod dense;
mod bh; 

use crate::{
    mesh::Mesh, 
};
use ndarray::{Array1, Array3};

pub enum TransientSolver {
    Dense, 
    BH
}

pub fn solve(
    mesh: &Mesh, 
    rho: f64, 
    nt: usize, 
    tmax: f64, 
    a_ext: &Array3<f64>, 
    b_ext: &Array3<f64>, 
    solver: TransientSolver
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>)  {
    match solver {
        TransientSolver::Dense => {
            dense::solve(mesh, rho, nt, tmax, a_ext, b_ext)
        }, 
        TransientSolver::BH => {
            dense::solve(mesh, rho, nt, tmax, a_ext, b_ext)
        }, 
    }
}

