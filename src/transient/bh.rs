//! Transient solver using Barnes-Hut for inductance matrix acceleration
//! 
//! This is a prototype feature that may have trouble with preconditioning until one is
//! implemented.

use crate::octree::Octree;
use faer::{
    diag::Diag,
    sparse::SparseColMat
};

struct KktSystem {
    r: Diag<f64>, 
    g: SparseColMat<usize, f64>, 
    octree: Octree, 
    dt: f64
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