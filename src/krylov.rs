//! Krylov methods for solving linear systems of the form `A x = b` by iteratively
//! computing `x` 


use std::{debug_assert, fmt::Display};

use crate::{
    check_lengths, math::{dot, mag}
};

/// Computes `A*x = b`
pub trait LinearOperator {
    fn apply(&self, x: &[f64], out: &mut [f64]);
    fn len(&self) -> usize;
}

pub trait Preconditioner {
    /// Compute `out = M^-1 x`
    fn apply(&self, x: &[f64], out: &mut [f64]);
    fn len(&self) -> usize;
}


pub enum KrylovResult {
    Converged(usize, f64), 
    DidNotConverge(usize, f64), 
    Breakdown(usize),
    ZeroUnknowns, 
    ZeroRhs, 

}

impl Display for KrylovResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KrylovResult::Converged(i, residual) => write!(f, "Converged in {} iterations with residual {:.6e}", i, residual), 
            KrylovResult::DidNotConverge(i, residual) => write!(f, "Did not converge after {} iterations with residual {:.6e}", i, residual),
            KrylovResult::ZeroUnknowns => write!(f, "Unknowns were length zero"), 
            KrylovResult::ZeroRhs => write!(f, "RHS was zero, no solution needed"), 
            KrylovResult::Breakdown(i) => write!(f, "Breakdown in SPD at iteration {}", i)
        }
    }
}



pub struct CgSolver {
    pub max_iterations: usize, 
    pub rtol: f64, 
}

/// Add one vector to another, `a + b = out`
fn vadd(a: &[f64], b: &[f64], out: &mut[f64]) {
    assert!(a.len() == b.len() && b.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Subtract one vector from another, `a - b = out`
fn vsub(a: &[f64], b: &[f64], out: &mut[f64]) {
    assert!(a.len() == b.len() && b.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

/// BLAS-1: `y = ax + y`
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    assert!(x.len() == y.len());
    for i in 0..y.len() {
        y[i] = y[i] + a * x[i];
    }
}

/// BLAS-1: `y = ay + x`
fn aypx(a: f64, y: &mut [f64], x: &[f64]) {
    assert!(x.len() == y.len());
    for i in 0..y.len() {
        y[i] = a * y[i] + x[i];
    }
}

pub trait KrylovSolver {
    /// Solve `Ax = b`, using a preconditioner `M` and starting with initial guess `x0`
    fn solve(&self, a: impl LinearOperator, m: impl Preconditioner, x0: &[f64], b: &[f64], x: &mut [f64]) -> KrylovResult;
}

impl KrylovSolver for CgSolver {
    // TODO: make multithreaded, move workspace into struct so that it can be 
    // reused across calls
    fn solve(&self, a: impl LinearOperator, m: impl Preconditioner, x0: &[f64], b: &[f64], x: &mut [f64]) -> KrylovResult {

        let n: usize = check_lengths!(a, m, x0, b, x);

        if n == 0 {
            return KrylovResult::ZeroUnknowns;
        }

        // Compute error tolerance metric
        let bmag: f64 = mag(b);
        let atol: f64 = self.rtol * bmag;
        if bmag == 0.0 {
            x.fill(0.0);
            return KrylovResult::ZeroRhs;
        }

        // Allocate workspace
        let mut r: Vec<f64> = vec![0.0; n];
        let mut z: Vec<f64> = vec![0.0; n];
        let mut ap: Vec<f64> = vec![0.0; n];

        // Start with x = x0
        x.copy_from_slice(x0);

        // Compute initial residual `r0 = b - A*x0` and `z = M^-1 r`
        a.apply(x0, &mut r);
        for i in 0..r.len() {
            r[i] = b[i] - r[i];
        }
        m.apply(&r, &mut z);

        // Initial search direction 
        let mut p: Vec<f64> = z.clone();

        // Early return (lucky guess!)
        let rmag = mag(&r);
        if rmag <= atol {
            return KrylovResult::Converged(0usize, rmag);
        }

        let mut rz = dot(&r, &z);

        for i in 0..self.max_iterations {   
        
            // Compute A*p for the current iteration
            a.apply(&p, &mut ap); 

            // Compute r_k^2 and alpha_k 
            let den: f64 = dot(&p, &ap);
            let alpha: f64 = rz/den;
            if !(den > 0.0) {

            }

            // Update x and r
            axpy(alpha, &p, x);
            axpy(-alpha, &ap, &mut r);
            
            // Check for convergence 
            let rmag = mag(&r);
            if rmag < atol {
                return KrylovResult::Converged(i+1, rmag);
            }

            // Update z, `z = M^-1 r`
            m.apply(&r, &mut z);

            // Update search direction: `p_k+1 = r_k+1 + beta*p_k`
            let rz_kp1: f64 = dot(&r, &z); 
            let beta: f64 = rz_kp1 / rz;
            aypx(beta, &mut p, &z);
            rz = rz_kp1;

        }
        let rmag = mag(&r);
        KrylovResult::DidNotConverge(self.max_iterations, rmag)

    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use faer::{Mat, diag::Diag};
    use rand;

    pub struct ClosureOperator<F> {
        pub n: usize,
        pub f: F
    }

    impl <F: Fn(&[f64], &mut [f64])> LinearOperator for ClosureOperator<F> {
        fn apply(&self, x: &[f64], out: &mut [f64]) {
            (self.f)(x, out)
        }
        fn len(&self) -> usize {
            self.n
        }
    }

    pub struct ClosurePreconditioner<F> {
        pub n: usize,
        pub f: F, 
    }

    impl <F: Fn(& [f64], &mut [f64])> Preconditioner for ClosurePreconditioner<F> {
        fn apply(&self, x: &[f64], out: &mut [f64]) {
            (self.f)(x, out)
        }
        fn len(&self) -> usize {self.n}
    }


    // Make a symmetric positive definite matrix, A = Bt * B + I
    fn make_spd(n: usize) -> Mat<f64> {
        let m = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>() - 0.5);
        let mt = m.transpose().to_owned();
        let mut d = Diag::<f64>::ones(n);
        for i in 0..n {
            d[i] = rand::random::<f64>();
        }
        d.as_ref()*(mt * m + Mat::<f64>::identity(n, n))*d.as_ref()
    }

    // Make a random vector 
    fn make_random_vector(n: usize) -> Vec<f64> {
        let mut v = vec![0.0; n]; 
        for i in 0..v.len() as usize {
            v[i] = rand::random();
        }
        v
    }

    #[test]
    fn test_cg() {
        let n = 200; 
        let rtol = 1e-8;
        let a = make_spd(n);
        let x = make_random_vector(n);
        let b = a.as_ref() * Mat::from_fn(n, 1, |i, _| x[i]);
        let mut bv = vec![0.0; n]; 
        for i in 0..bv.len() as usize {
            bv[i] = b[(i,0)];
        }

        let operator = ClosureOperator{n: n, f: |xin: &[f64], bout: &mut [f64]| {
            let b = a.as_ref() * Mat::<f64>::from_fn(n, 1, |i, _| xin[i]);
            for i in 0..bout.len() as usize {
                bout[i] = b[(i,0)];
            }
        }};

        let preconditioner = ClosurePreconditioner{n: n, f: |xin: &[f64], out: &mut [f64]| {
            for i in 0..out.len() {
                out[i] = xin[i] / a[(i,i)];
            }
        }};

        let cg = CgSolver {max_iterations: 1000, rtol: rtol};
        let mut out = vec![0.0; n]; 
        let x0 = vec![0.0; n];

        use std::time::Instant; 
        let start = Instant::now();
        let result = cg.solve(operator, preconditioner, &x0, &bv, &mut out);
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        
        println!("{}", result);
        println!("Solved {}-size matrix in {:.3} ms", n, elapsed);

        for i in 0..n {
            assert!((out[i] - x[i]).abs() < rtol*10.0);
        }
    }
}