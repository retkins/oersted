//! Krylov methods for solving linear systems of the form `A x = b` by iteratively
//! computing `x`

use std::fmt::Display;

use crate::{
    check_lengths,
    krylov::KrylovResult::{Converged, DidNotConverge},
    math::{axpy, aypx, dot, mag},
};

use faer::{
    Accum, Col, Mat, MatMut, MatRef, Par, linalg::matmul::matmul, linalg::solvers::SolveLstsq,
};

pub trait LinearOperator {
    /// Computes `A*x = b`
    fn apply(&self, x: &[f64], out: &mut [f64]);
    /// Computes the diagonal of `A`
    fn diagonal(&self, out: &mut [f64]);
    fn len(&self) -> usize;
}

pub struct MatrixOperator<'a> {
    pub a: MatRef<'a, f64>,
}

impl LinearOperator for MatrixOperator<'_> {
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        let n: usize = check_lengths!(self, x, out);
        matmul(
            MatMut::from_column_major_slice_mut(out, n, 1),
            Accum::Replace,
            self.a.as_ref(),
            MatRef::from_column_major_slice(x, n, 1),
            1.0,
            Par::Seq,
        )
    }
    fn diagonal(&self, out: &mut [f64]) {
        let n: usize = check_lengths!(self, out);
        for i in 0..n {
            out[i] = self.a[(i, i)];
        }
    }
    fn len(&self) -> usize {
        self.a.nrows()
    }
}

pub trait Preconditioner {
    /// Compute `out = M^-1 x`
    fn apply(&self, x: &[f64], out: &mut [f64]);
    fn len(&self) -> usize;
}

pub struct JacobiPreconditioner<'a> {
    pub a: MatRef<'a, f64>,
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

pub struct NoPreconditioner {
    pub n: usize,
}

impl Preconditioner for NoPreconditioner {
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        out.copy_from_slice(x);
    }
    fn len(&self) -> usize {
        self.n
    }
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

#[allow(non_snake_case)]
pub struct Workspace {
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    H: Option<Mat<f64>>,
    V: Option<Vec<Vec<f64>>>,
    w: Option<Vec<f64>>,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        Self {
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
            H: None,
            V: None,
            w: None,
        }
    }
    pub fn reset(&mut self, n: usize, max_iterations: usize) {
        for v in [&mut self.r, &mut self.z, &mut self.p, &mut self.ap] {
            v.resize(n, 0.0);
        }

        if self.H.is_some() {
            if self.H.as_ref().unwrap().ncols() != max_iterations {
                self.H = Some(Mat::zeros(max_iterations + 1, max_iterations));
            } else {
                self.H.as_mut().unwrap().fill(0.0);
            }
        }
        if self.V.is_some() {
            if self.V.as_ref().unwrap().len() != max_iterations + 1 {
                self.V = Some(vec![vec![0.0; n]; max_iterations + 1]);
            } else {
                for v in self.V.as_mut().unwrap() {
                    v.resize(n, 0.0);
                }
            }
        }
        if let Some(v) = &mut self.w {
            v.resize(n, 0.0);
        }
    }

    fn setup_gmres(&mut self, n: usize, max_iterations: usize) {
        if self.H.is_none() {
            self.H = Some(Mat::zeros(max_iterations + 1, max_iterations));
        }
        if self.V.is_none() {
            self.V = Some(vec![vec![0.0; n]; max_iterations + 1]);
        }
        if self.w.is_none() {
            self.w = Some(vec![0.0; n]);
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
        ws.reset(n, self.max_iterations);

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

            if den < 0.0 {
                return KrylovResult::Breakdown(i + 1);
            }
            let alpha: f64 = rz / den;

            // Update x and r
            axpy(alpha, &ws.p, x);
            axpy(-alpha, &ws.ap, &mut ws.r);

            // Check for convergence
            let rmag = mag(&ws.r);
            if rmag < atol {
                println!("Converged in {} iterations.", i + 1);
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
        println!("Did not converge");
        KrylovResult::DidNotConverge(self.max_iterations, rmag)
    }
}

/// Generalized Minimal Residual Method, with restarts
pub struct GmresSolver {
    pub max_iterations: usize,
    pub rtol: f64,
    pub restarts: usize,
}

impl KrylovSolver for GmresSolver {
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
        if bmag == 0.0 {
            x.fill(0.0);
            return KrylovResult::ZeroRhs;
        }

        // Allocate workspace for this solve
        ws.reset(n, self.max_iterations);
        ws.setup_gmres(n, self.max_iterations); // We can now safely unwrap() H/V/w

        // Start with x = x0
        x.copy_from_slice(x0);

        // Compute initial residual `r = M(b - A x)`
        a.apply(x0, &mut ws.ap);
        for i in 0..ws.r.len() {
            ws.ap[i] = b[i] - ws.ap[i];
        }
        m.apply(&ws.ap, &mut ws.r);

        let beta: f64 = mag(&ws.r);
        let atol: f64 = self.rtol * beta;
        if beta < self.rtol {
            // Solution has already converged; x = x0
            x.copy_from_slice(x0);
        }

        let v = ws.V.as_mut().unwrap();
        let h = ws.H.as_mut().unwrap();
        let w = ws.w.as_mut().unwrap();
        for i in 0..ws.r.len() {
            v[0][i] = ws.r[i] / beta;
        }

        let mut j_done = 1usize;
        let mut res = 0.0;
        for j in 0..self.max_iterations {
            // Compute next Krylov vector
            a.apply(&v[j], &mut ws.ap);
            m.apply(&ws.ap, w);

            // Gram-Schmidt orthogonalization
            for i in 0..(j + 1) {
                h[(i, j)] = dot(w, &v[i]);
                for k in 0..w.len() {
                    w[k] -= h[(i, j)] * v[i][k];
                }
            }
            h[(j + 1, j)] = mag(w);
            if h[(j + 1, j)] < 1e-14 * beta {
                j_done += 1;
                // break; // Should complete here, but just add another vector to basis for now
            }

            // Add new vector to basis
            for k in 0..w.len() {
                v[j + 1][k] = w[k] / h[(j + 1, j)];
            }

            // Find best approximation in the basis V
            let mut e1 = Col::<f64>::zeros(j_done + 1);
            e1[0] = beta;
            // Only use the vector bases that have already been completed
            let hs = h.submatrix(0, 0, j_done + 1, j_done);
            let y = hs.qr().solve_lstsq(&e1);

            // Estimate the current residual
            res = (hs * &y - &e1).norm_l2();
            if res < atol {
                for i in 0..x.len() {
                    let mut acc: f64 = 0.0;
                    for k in 0..y.nrows() {
                        acc += v[k][i] * y[k];
                    }
                    x[i] = x0[i] + acc;
                }
                println!("Converged in {} iterations.", j_done);
                return Converged(j_done, res);
            }

            j_done += 1;
        }

        println!("Did not converge with residual {:.6e}", res);
        DidNotConverge(self.max_iterations, 1e0)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::math::vsub;
    use faer::{Mat, MatRef, diag::Diag};
    use rand;
    use std::time::Instant;

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

        let op = MatrixOperator { a: a.as_ref() };

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
        assert!(
            m_iter_ratio > 2.0,
            "No preconditioner / Jacobi preconditioner iterations: {:.1}",
            m_iter_ratio
        );
    }
}
