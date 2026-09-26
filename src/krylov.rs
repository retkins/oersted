//! Krylov methods for solving linear systems of the form `A x = b` by iteratively
//! computing `x` 


use crate::{
    math::{mag, dot, min_and_max}
};

/// Computes `A*x`
pub type MatrixFreeEvaluator = fn(&[f64], &mut[f64]); 

pub enum KrylovResult {
    Converged(usize), 
    DidNotConverge(usize), 
    ZeroUnknowns, 
    ZeroRhs
}

pub struct CgSolver {
    pub max_iterations: usize, 
    pub rtol: f64, 
    pub unknowns: usize,
    pub n_threads: usize
}

// Subtract one vector from another, `a - b = out`
fn vsub(a: &[f64], b: &[f64], out: &mut[f64]) {
    assert!(a.len() == b.len() && b.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

// BLAS-1: y = ax + y
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    assert!(x.len() == y.len());
    for i in 0..y.len() {
        y[i] = y[i] + a * x[i];
    }
}

// BLAS-1: y = ay + x
fn aypx(a: f64, y: &mut [f64], x: &[f64]) {
    assert!(x.len() == y.len());
    for i in 0..y.len() {
        y[i] = a * y[i] + x[i];
    }
}

pub trait KrylovSolver {
    fn solve(&mut self, f: MatrixFreeEvaluator, x0: &[f64], rhs: &[f64], out: &mut [f64]) -> KrylovResult;
}

impl KrylovSolver for CgSolver {
    // TODO: make multithreaded, move workspace into struct so that it can be 
    // reused across calls
    fn solve(&mut self, f: MatrixFreeEvaluator, x0: &[f64], rhs: &[f64], out: &mut [f64]) -> KrylovResult {

        if self.unknowns == 0 {
            return KrylovResult::ZeroUnknowns;
        }

        // Compute error tolerance metric
        let atol: f64 = self.rtol * mag(rhs);
        if atol == 0.0 {
            out.fill(0.0);
            return KrylovResult::ZeroRhs;
        }

        // Allocate workspace
        let mut r = vec![0.0; self.unknowns];
        let mut ax = vec![0.0; self.unknowns];
        let mut ap = vec![0.0; self.unknowns];
        out.copy_from_slice(x0);

        // Compute initial residual, `r0 = b - A*x0`
        f(out, &mut ax);
        vsub(rhs, &ax, &mut r);

        // Initial search direction 
        let mut p: Vec<f64> = r.clone();

        // Early return (lucky guess!)
        if mag(&r) <= atol {
            return KrylovResult::Converged(0usize);
        }
        
        for i in 0..self.max_iterations {   

            // Compute A*p for the current iteration
            f(&p, &mut ap); 

            // Compute r_k^2 and alpha_k 
            let den: f64 = dot(&p, &ap);
            let rk_2: f64 = dot(&r, &r);
            let alpha: f64 = rk_2/den;
            // TODO: divide by zero guard

            // Update x and r, compute norm of r_k+1 and beta
            axpy(alpha, &p, out);
            axpy(-alpha, &ap, &mut r);
            
            // Check for convergence 
            let (min, max) = min_and_max(&r).unwrap();
            if min.abs() < atol && max.abs() < atol {
                return KrylovResult::Converged(i+1);
            }

            // Update search direction: `p_k+1 = r_k+1 + beta*p_k`
            let rkp1_norm: f64 = dot(&r, &r); 
            let beta: f64 = rkp1_norm / rk_2;
            aypx(beta, &mut p, &r);

        }

        KrylovResult::DidNotConverge(self.max_iterations)

    }
}