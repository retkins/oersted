//! Basic dense transient solver 
//! 
//! This solve forms a fully dense interaction matrix and therefore should only be used
//! for relatively small systems (<10k elements)
//! 

use ndarray::{Array1,Array3};
use faer::{mat::{Mat}, diag::{Diag}};

use crate::{
    mesh::{Mesh},
    types::{Vec3}
};

/// Solve a transient problem
pub fn solve(
    mesh: &Mesh, 
    rho: f64, 
    nt: usize, 
    tmax: f64, 
    a_ext: &Array3<f64>,
    b_ext: &Array3<f64>
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>){

    let n_elem: usize = mesh.connectivity.len();

    // Allocate memory for the time steps and the results data
    // a and b are the TOTAL value at element centroids, including the external
    // sources. Overwriting the external source arrays would save memory, but it
    // may not be what the caller wants to do.
    let mut time: Array1<f64> = Array1::zeros(nt);
    let mut j: Array3<f64> = Array3::zeros((nt, n_elem,3));
    let mut a: Array3<f64> = Array3::zeros((nt, n_elem,3));
    let mut b: Array3<f64> = Array3::zeros((nt, n_elem,3));

    // 

    // Assembly 
    let r = assemble_r(rho, mesh);
    let g = assemble_g(&mesh);


    (time, j, a, b)

}

// Assemble the constraint-gradient matrix G
//
// This matrix is 3*num_elems x num_nodes. The first num_elems rows are for the
// x-dof, second num_elems (second third) rows are for y-dof, etc.
fn assemble_g(mesh: &Mesh) -> Mat<f64> {

    let mut g = Mat::<f64>::zeros(3*mesh.n_elems(), mesh.n_nodes()); 

    for e in 0..mesh.n_elems() {
        let vg_e: [Vec3; 4] = mesh.hat_gradients(e);

        for ni in 0..4usize {
            for k in 0..3usize {
                let n: usize = mesh.connectivity[e][ni] as usize;
                g[(mesh.n_elems()*k + e, n)] = vg_e[ni][k];
            }
        }
        
    }
    g
}

// Assemble the resistance diagonal matrix R 
//
// This matrix has length `n_elems`, each of which are rho*vol[e]
fn assemble_r(rho: f64, mesh: &Mesh) -> Diag<f64> {
    let mut r = Diag::zeros(mesh.n_elems());
    for i in 0..mesh.n_elems() {
        r[i] = rho*mesh.volumes[i];
    }
    r
}