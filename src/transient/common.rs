//! Internals used by multiple transient solvers

use faer::{
    diag::Diag
};

use crate::{
    mesh::Mesh
};

/// Assemble the resistance diagonal matrix R
///
/// This matrix has length `n_elems`, each of which are rho*vol[e]
pub fn assemble_r(rho: f64, mesh: &Mesh) -> Diag<f64> {
    let mut r = Diag::zeros(mesh.n_elems());
    for i in 0..mesh.n_elems() {
        r[i] = rho * mesh.volumes[i];
    }
    r
}