//! Krylov methods for solving linear systems of the form `A x = b` by iteratively
//! computing `x`

use std::fmt::Display;

use crate::{
    check_lengths,
    math::{axpy, aypx, dot, mag},
};

pub trait LinearOperator {
    /// Computes `A*x = b`
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
            KrylovResult::Converged(i, residual) => write!(
                f,
                "Converged in {} iterations with residual {:.6e}",
                i, residual
            ),
            KrylovResult::DidNotConverge(i, residual) => write!(
                f,
                "Did not converge after {} iterations with residual {:.6e}",
                i, residual
            ),
            KrylovResult::ZeroUnknowns => write!(f, "Unknowns were length zero"),
            KrylovResult::ZeroRhs => write!(f, "RHS was zero, no solution needed"),
            KrylovResult::Breakdown(i) => write!(f, "Breakdown in SPD at iteration {}", i),
        }
    }
}

pub struct Workspace {
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        Self {
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
        }
    }
    pub fn reset(&mut self, n: usize) {
        for v in [&mut self.r, &mut self.z, &mut self.p, &mut self.ap] {
            v.resize(n, 0.0);
        }
    }
}

pub trait KrylovSolver {
    /// Solve `Ax = b`, using a preconditioner `M` and starting with initial guess `x0`
    fn solve(
        &self,
        ws: &mut Workspace,
        a: &impl LinearOperator,
        m: &impl Preconditioner,
        x0: &[f64],
        b: &[f64],
        x: &mut [f64],
    ) -> KrylovResult;
}

/// Preconditioned conjugate gradient solver
pub struct PcgSolver {
    pub max_iterations: usize,
    pub rtol: f64,
}

impl KrylovSolver for PcgSolver {
    fn solve(
        &self,
        ws: &mut Workspace,
        a: &impl LinearOperator,
        m: &impl Preconditioner,
        x0: &[f64],
        b: &[f64],
        x: &mut [f64],
    ) -> KrylovResult {
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
        ws.reset(n);

        // Start with x = x0
        x.copy_from_slice(x0);

        // Compute initial residual `r0 = b - A*x0` and `z = M^-1 r`
        a.apply(x0, &mut ws.r);
        for i in 0..ws.r.len() {
            ws.r[i] = b[i] - ws.r[i];
        }
        m.apply(&ws.r, &mut ws.z);

        // Initial search direction
        ws.p.copy_from_slice(&ws.z);

        // Early return (lucky guess!)
        let rmag = mag(&ws.r);
        if rmag <= atol {
            return KrylovResult::Converged(0usize, rmag);
        }

        let mut rz = dot(&ws.r, &ws.z);

        for i in 0..self.max_iterations {
            // Compute A*p for the current iteration
            a.apply(&ws.p, &mut ws.ap);

            // Compute r_k^2 and alpha_k
            let den: f64 = dot(&ws.p, &ws.ap);

            if !(den > 0.0) {
                return KrylovResult::Breakdown(i + 1);
            }
            let alpha: f64 = rz / den;

            // Update x and r
            axpy(alpha, &ws.p, x);
            axpy(-alpha, &ws.ap, &mut ws.r);

            // Check for convergence
            let rmag = mag(&ws.r);
            if rmag < atol {
                return KrylovResult::Converged(i + 1, rmag);
            }

            // Update z, `z = M^-1 r`
            m.apply(&ws.r, &mut ws.z);

            // Update search direction: `p_k+1 = r_k+1 + beta*p_k`
            let rz_kp1: f64 = dot(&ws.r, &ws.z);
            let beta: f64 = rz_kp1 / rz;
            aypx(beta, &mut ws.p, &ws.z);
            rz = rz_kp1;
        }
        let rmag = mag(&ws.r);
        KrylovResult::DidNotConverge(self.max_iterations, rmag)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::math::vsub;
    use faer::{Accum, Mat, MatMut, MatRef, Par, diag::Diag, linalg::matmul::matmul};
    use rand;
    use std::time::Instant;

    struct MatrixOperator {
        a: Mat<f64>,
    }

    impl LinearOperator for MatrixOperator {
        fn apply(&self, x: &[f64], out: &mut [f64]) {
            let n = x.len();
            matmul(
                MatMut::from_column_major_slice_mut(out, n, 1),
                Accum::Replace,
                self.a.as_ref(),
                MatRef::from_column_major_slice(&x, n, 1),
                1.0,
                Par::Seq,
            )
        }
        fn len(&self) -> usize {
            self.a.nrows()
        }
    }

    struct NoPreconditioner {
        n: usize,
    }

    impl Preconditioner for NoPreconditioner {
        fn apply(&self, x: &[f64], out: &mut [f64]) {
            for i in 0..x.len() {
                out[i] = x[i];
            }
        }
        fn len(&self) -> usize {
            self.n
        }
    }

    struct JacobiPreconditioner<'a> {
        a: MatRef<'a, f64>,
    }

    impl Preconditioner for JacobiPreconditioner<'_> {
        fn apply(&self, x: &[f64], out: &mut [f64]) {
            for i in 0..x.len() {
                out[i] = x[i] / self.a[(i, i)];
            }
        }
        fn len(&self) -> usize {
            self.a.nrows()
        }
    }

    // Make a symmetric positive definite matrix, A = Bt * B + I
    #[allow(unused)]
    fn make_spd(n: usize) -> Mat<f64> {
        let m = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>() - 0.5);
        let mt = m.transpose().to_owned();
        let mut d = Diag::<f64>::ones(n);
        for i in 0..n {
            d[i] = rand::random::<f64>();
        }
        d.as_ref() * (mt * m + Mat::<f64>::identity(n, n)) * d.as_ref()
    }

    // Make an 'inductance' matrix, which is SPD by construction
    fn make_ind(n: usize) -> Mat<f64> {
        let s: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(((i * 61 % 41) as f64) / 20.0 - 1.0))
            .collect();
        Mat::<f64>::from_fn(n, n, |i, j| {
            let d = (i as f64 - j as f64).abs();
            let k = if i == j { 2.0 } else { 1.0 / (1.0 + d) };
            s[i] * s[j] * k
        })
    }

    // Make a random vector
    fn make_random_vector(n: usize) -> Vec<f64> {
        let mut v = vec![0.0; n];
        for i in 0..v.len() as usize {
            v[i] = rand::random();
        }
        v
    }

    fn count_iterations(
        cg: &PcgSolver,
        ws: &mut Workspace,
        op: &impl LinearOperator,
        m: &impl Preconditioner,
        b: &[f64],
        x: &mut [f64], 
        xref: &[f64],
    ) -> usize {
        let n: usize = op.len();
        let it = match cg.solve(ws, op, m, &mut vec![0.0; n], b, x) {
            KrylovResult::Converged(i, _) => i,
            other => panic!("{}", other),
        };

        let mut err = vec![0.0; n];
        vsub(&x, xref, &mut err);
        let r_err = mag(&err) / mag(xref);
        assert!(r_err < 1e-6, "Relative solution error: {:.6e}", r_err);
        it
    }

    // Compute |Ax - b| < rtol*|b|
    fn check_residual(rtol: f64, op: &impl LinearOperator, x: &[f64], b: &[f64]) {
        let n = op.len();
        let mut ax = vec![0.0; n];
        op.apply(&x, &mut ax);

        // res = Ax - b
        let mut res = vec![0.0; n];
        vsub(b, &ax, &mut res);
        let rmag = mag(&res);

        // |res| < rtol * |b|
        assert!(rmag <= rtol * mag(b), "Residual {:.3e}", rmag);
    }

    #[test]
    fn test_pcg() {
        let n = 1000;
        let rtol = 1e-10;
        let a = make_ind(n);
        let xref = make_random_vector(n);
        let b = a.as_ref() * MatRef::from_column_major_slice(&xref, n, 1);
        let bv = b.col_as_slice(0).to_vec();

        let cg = PcgSolver {
            max_iterations: 10000,
            rtol: rtol,
        };

        let mut ws = Workspace::new(n);

        let op = MatrixOperator { a: a.clone() };

        let m_none = NoPreconditioner { n };
        let m_jacobi = JacobiPreconditioner { a: a.as_ref() };

        let mut x = vec![0.0; n];

        let start = Instant::now();
        let it_none: usize = count_iterations(&cg, &mut ws, &op, &m_none, &bv, &mut x, &xref);
        let elapsed: f64 = start.elapsed().as_secs_f64() * 1e3;
        println!("Solved {}-size matrix in {:.3} ms", n, elapsed);
        println!("Using no preconditioner: {} iterations", it_none);

        check_residual(cg.rtol, &op, &x, &bv);
        x.fill(0.0);

        let start = Instant::now();
        let it_jacobi: usize = count_iterations(&cg, &mut ws, &op, &m_jacobi, &bv, &mut x, &xref);
        let elapsed: f64 = start.elapsed().as_secs_f64() * 1e3;
        println!("Solved {}-size matrix in {:.3} ms", n, elapsed);
        println!("Using jacobi preconditioner: {} iterations", it_jacobi);

        check_residual(cg.rtol, &op, &x, &bv);

        let m_iter_ratio: f64 = it_none as f64 / it_jacobi as f64;
        assert!(m_iter_ratio > 2.0, "No preconditioner / Jacobi preconditioner iterations: {:.1}", m_iter_ratio);
    }
}
