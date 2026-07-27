//! Basic dense transient solver 
//! 
//! This solve forms a fully dense interaction matrix and therefore should only be used
//! for relatively small systems (<10k elements)
//! 

use ndarray::{Array1,Array3};
use faer::mat::Mat;

use crate::mesh::{Mesh, volumes};

/// Solve a transient problem
pub fn solve(
    mesh: Mesh, 
    rho: f64, 
    nt: usize, 
    tmax: f64, 
    a_ext: &Array3<f64>,
    b_ext: &Array3<f64>
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>){

    let n_elem: usize = mesh.connectivity.len();

    let mut time: Array1<f64> = Array1::zeros(nt);
    let mut j: Array3<f64> = Array3::zeros((nt, n_elem,3));
    let mut a: Array3<f64> = Array3::zeros((nt, n_elem,3));
    let mut b: Array3<f64> = Array3::zeros((nt, n_elem,3));

    // Assembly 
    let g = assemble_g(&mesh);


    (time, j, a, b)

}

// Assembly the constraint-gradient matrix G
fn assemble_g(mesh: &Mesh) -> Mat<f64> {

    let mut g = Mat::with_capacity(3*mesh.n_elems(), mesh.n_nodes()); 

    let mut vols: Vec<f64> = vec![0.0; mesh.n_elems()];
    volumes(&mesh.nodes,&mesh.connectivity, &mut vols);


    g
}