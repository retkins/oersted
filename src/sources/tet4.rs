use crate::{
    MU0_4PI, check_lengths,
    math::{atan2, ln},
    types::Vec3,
};

use std::f64::consts::PI;
const INV_4PI: f64 = 1.0 / (4.0 * PI);
const MU0_8PI: f64 = 0.5 * MU0_4PI;

// Refer to the gmsh or ansys documentation for node numbering on a tet element:
// <https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering>
// <https://ansyshelp.ansys.com/public/account/secured?returnurl=//////////Views/Secured/corp/v242/en/ans_elem/Hlp_E_SOLID236.html>
const NODE_WINDING: [(usize, usize, usize); 4] = [(0, 2, 1), (0, 3, 2), (1, 2, 3), (0, 1, 3)];

// Return the node indices associated with a given triangular face, using the proper
// winding order: the perimeter of the face is traversed according to the right hand
// rule, with the normal facing away from the center of the element.
fn face_nodes(f: usize) -> (usize, usize, usize) {
    NODE_WINDING[f]
}

// Per-face data used for solid angle - based calculations
#[derive(Clone, Copy, Default)]
struct Face {
    // Vertices of the face
    a: Vec3,
    b: Vec3,
    c: Vec3,

    // Unit normal vector to the face
    n_hat: Vec3,

    // Unit vector along the edge
    e_hat: [Vec3; 3], // AB, BC, CA

    // Unit vector in the plane of the face (pointing to interior)
    t_hat: [Vec3; 3],
}

// Compute the unit vectors associated with a face given vertice A, B, C
// Vertices are given in counter-clockwise direction according to the right hand
// rule, to produce an outward facing normal vector
fn precompute_face(a: &Vec3, b: &Vec3, c: &Vec3) -> Face {
    let mut face: Face = Face {
        a: *a,
        b: *b,
        c: *c,
        ..Default::default()
    };

    // Face unit normal
    face.n_hat = (*b - *a).cross(&(*c - *a));
    face.n_hat /= face.n_hat.mag();

    for (e, (p, q)) in [(a, b), (b, c), (c, a)].into_iter().enumerate() {
        // Unit vector along edge
        let mut e_hat = *q - *p;
        e_hat /= e_hat.mag();

        // Unit vector in plane, perpendicular to edge and normal
        let mut t_hat = face.n_hat.cross(&e_hat);
        t_hat /= t_hat.mag();

        face.e_hat[e] = e_hat;
        face.t_hat[e] = t_hat;
    }
    face
}

// Compute the Van Osterom and Strackee solid angle
//
// A, B, C are the vertices of a triangle in 3D space
// r is the position vector of another pt in 3D space at which to evaluate the
// solid angle
//
// # Reference
// A. Van Oosterom and J. Strackee, "The Solid Angle of a Plane Triangle,"
// in IEEE Transactions on Biomedical Engineering,
// vol. BME-30, no. 2, pp. 125-126, Feb. 1983,
// doi: <10.1109/TBME.1983.325207>.
#[allow(non_snake_case)]
#[inline]
fn solid_angle(A: &Vec3, B: &Vec3, C: &Vec3, r: &Vec3) -> f64 {
    let a: Vec3 = *A - *r;
    let b: Vec3 = *B - *r;
    let c: Vec3 = *C - *r;
    let a_mag: f64 = a.mag();
    let b_mag: f64 = b.mag();
    let c_mag: f64 = c.mag();

    let num = a.dot(&b.cross(&c));
    let den1 = a_mag * b_mag * c_mag;
    let den2 = a.dot(&b) * c_mag;
    let den3 = a.dot(&c) * b_mag;
    let den4 = b.dot(&c) * a_mag;
    let den = den1 + den2 + den3 + den4;

    2.0 * atan2(num, den)
}

// Edge potential as defined in eq. 18 in Fabbri (2008).
//
// This function is NOT regularized and therefore is singular on the edge
fn edge_potential(r1: &Vec3, r2: &Vec3, r: &Vec3) -> f64 {
    // Distance from r1 to r, r2 to r, and r1 to r2
    let d1: f64 = (*r1 - *r).mag();
    let d2: f64 = (*r2 - *r).mag();
    let e: f64 = (*r2 - *r1).mag();
    let s: f64 = d1 + d2;

    // Edge potential in stable form
    // 2.0 * ln(s + e) - ln(2.0 * (d1 + d2 + (*r1 - *r).dot(&(*r2 - *r))));

    // Edge potential without regularization
    ln((s + e) / (s - e))
}

// Charge potential of the face as defined in eq. 17 in Fabbri (2008)
#[inline(always)]
fn charge_potential(face: &Face, r: &Vec3) -> f64 {
    // Normal distance from the plane of the face to the target point
    let d: f64 = face.n_hat.dot(&(*r - face.a));

    // Solid angle subtended by the triangular face
    let omega: f64 = solid_angle(&face.a, &face.b, &face.c, r);

    // Manually unroll the summation over edges so that this function can be
    // vectorized when inlined by the compiler
    let mut result: f64 = d * omega;

    // Use the scalar triple product definition to rearrange eq 17
    // This version uses the in-plane unit vector on the edge, which
    // saves a cross-product every iteration
    result += face.t_hat[0].dot(&(*r - face.a)) * edge_potential(&face.a, &face.b, r);
    result += face.t_hat[1].dot(&(*r - face.b)) * edge_potential(&face.b, &face.c, r);
    result += face.t_hat[2].dot(&(*r - face.c)) * edge_potential(&face.c, &face.a, r);

    result
}

// Charge potential gradient from Byzov 2022 (equation 5)
#[inline(always)]
fn charge_potential_gradient(face: &Face, r: &Vec3) -> Vec3 {
    let omega: f64 = solid_angle(&face.a, &face.b, &face.c, r);

    let mut result: Vec3 = face.n_hat * omega;

    // Vi(a) = 2 atanh(e / S)
    // e = vector from start node to end node on edge
    // S = sum of the magnitudes of the vectors from a to each node
    let v0: f64 = edge_potential(&face.a, &face.b, r);
    let v1: f64 = edge_potential(&face.b, &face.c, r);
    let v2: f64 = edge_potential(&face.c, &face.a, r);

    // Edge sum
    let mut edge_sum: Vec3 = face.e_hat[0] * v0;
    edge_sum += face.e_hat[1] * v1;
    edge_sum += face.e_hat[2] * v2;

    result += face.n_hat.cross(&edge_sum);

    result
}

/// Compute magnetic vector potential generated by a 4-node tetrahedral finite element
/// with constant current density defined at its centroid
///
/// This function operates on n-number of target points and
/// is vectorized for efficient computation. This version uses the solid angle
/// formulation (see reference).
///
/// # Arguments
/// - `nodes`: (m), coordinates of each corner node in 3D space
/// - `jdensity`: (A/m2), current density vector, assumed constant throughout the element
/// - `targets`: (m), target locations in 3d space
/// * `out`: (T-m) pre-allocated results accumulation for a-field (ax, ay, az)
///
/// # Note
/// Accumulates into `out`. Caller must zero `out` before the first call.
///
/// # Reference
/// M. Fabbri, "Magnetic Flux Density and Vector Potential of Uniform Polyhedral
/// Sources," in IEEE Transactions on Magnetics, vol. 44, no. 1, pp. 32-36, Jan. 2008,
/// doi: <https://ieeexplore.ieee.org/document/4407584>.
pub fn a_current_tet4(
    nodes: &[Vec3; 4],
    jdensity: &Vec3,
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    let (x, y, z) = targets;
    let (ax, ay, az) = out;
    let n_targets = x.len();
    assert_eq!(n_targets, y.len());
    assert_eq!(n_targets, z.len());
    assert_eq!(n_targets, ax.len());
    assert_eq!(n_targets, ay.len());
    assert_eq!(n_targets, az.len());

    let prefactor: Vec3 = *jdensity * MU0_8PI;

    for f in 0..4 {
        let (na, nb, nc) = face_nodes(f);

        let a: Vec3 = nodes[na];
        let b: Vec3 = nodes[nb];
        let c: Vec3 = nodes[nc];

        let face: Face = precompute_face(&a, &b, &c);

        // At each target point, for each face, compute the contribution from
        // Fabbri (2008) eq. 5
        for i in 0..n_targets {
            let r: Vec3 = Vec3([x[i], y[i], z[i]]);
            let rf: Vec3 = face.a;
            let psi: f64 = charge_potential(&face, &r);
            let a_contribution: Vec3 = prefactor * (rf - r).dot(&face.n_hat) * psi;
            ax[i] += a_contribution[0];
            ay[i] += a_contribution[1];
            az[i] += a_contribution[2];
        }
    }
}

/// Compute magnetic field intensity generated by a 4-node tetrahedral finite element
///
/// This function operates on n-number of target points and
/// is vectorized for efficient computation. This version uses the solid angle
/// formulation (see reference).
///
/// # Arguments
/// * `nodes`: (m), coordinates of each corner node in 3D space
/// * `jdensity`: (A/m2), current density vector, assumed constant throughout the element
/// * `targets`: (m), target locations in 3d space
/// * `out`: (A/m) pre-allocated results accumulation for h-field (hx, hy, hz)
///
/// # Note
/// Accumulates into `out`. Caller must zero `out` before the first call.
///
/// # Reference
/// M. Fabbri, "Magnetic Flux Density and Vector Potential of Uniform Polyhedral
/// Sources," in IEEE Transactions on Magnetics, vol. 44, no. 1, pp. 32-36, Jan. 2008,
/// doi: <https://ieeexplore.ieee.org/document/4407584>.
pub fn h_current_tet4(
    nodes: &[Vec3; 4],
    jdensity: &Vec3,
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    let (x, y, z) = targets;
    let (hx, hy, hz) = out;
    let n_targets: usize = check_lengths!(x, y, z, hx, hy, hz);

    for f in 0..4 {
        let (na, nb, nc) = face_nodes(f);

        let a: Vec3 = nodes[na];
        let b: Vec3 = nodes[nb];
        let c: Vec3 = nodes[nc];

        let face = precompute_face(&a, &b, &c);
        let prefactor = jdensity.cross(&face.n_hat) * INV_4PI;

        for i in 0..n_targets {
            let r = Vec3([x[i], y[i], z[i]]);
            let psi = charge_potential(&face, &r);
            hx[i] += prefactor[0] * psi;
            hy[i] += prefactor[1] * psi;
            hz[i] += prefactor[2] * psi;
        }
    }
}

pub fn a_mag_tet4(
    nodes: &[Vec3; 4], // source nodes
    mvector: &Vec3,    // source element
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    // From Fabbri eq 11, A-field from magnetization is the same as B-field from
    // current density
    let (x, y, z) = targets;
    let (ax, ay, az) = out;
    let n_targets: usize = check_lengths!(x, y, z, ax, ay, az);

    for f in 0..4 {
        let (na, nb, nc) = face_nodes(f);

        let a: Vec3 = nodes[na];
        let b: Vec3 = nodes[nb];
        let c: Vec3 = nodes[nc];

        let face: Face = precompute_face(&a, &b, &c);
        // Note the difference in prefactor term to compute A-field like B-field
        // (not like H-field)
        let prefactor: Vec3 = mvector.cross(&face.n_hat) * MU0_4PI;

        for i in 0..n_targets {
            let r: Vec3 = Vec3([x[i], y[i], z[i]]);
            let psi: f64 = charge_potential(&face, &r);
            ax[i] += prefactor[0] * psi;
            ay[i] += prefactor[1] * psi;
            az[i] += prefactor[2] * psi;
        }
    }
}

/// Compute the magnetic field intensity generated by a uniformly magnetized
/// tetrahedral element.
///
/// This function uses the solid-angle formulation, which is faster and more stable
/// than the edge-integral formulation.
///
/// # Arguments
/// * `nodes`: (m) coordinates of the corner nodes of the source element
/// * `mvector`: (A/m) magnetization vector, assumed constant over the element
/// * `targets`: (m) x,y,z coordinates of each target point
/// * `out`: (A/m) pre-allocated results accumulation for h-field (hx, hy, hz)
///
/// # Note
/// Accumulates into `out`. Caller must zero `out` before the first call.
///  
/// # Reference  
/// Byzov, D., Martyshko, P., & Chernoskutov, A. (2022).
/// Computationally Effective Modeling of Self-Demagnetization and Magnetic Field
/// for Bodies of Arbitrary Shape Using Polyhedron Discretization.
/// Mathematics, 10(10), 1656.
/// <https://doi.org/10.3390/math10101656>
pub fn h_mag_tet4(
    nodes: &[Vec3; 4], // source nodes
    mvector: &Vec3,    // source element
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    let (x, y, z) = targets;
    let (hx, hy, hz) = out;
    let n_targets: usize = x.len();

    for f in 0..4 {
        let (na, nb, nc) = face_nodes(f);

        let a: Vec3 = nodes[na];
        let b: Vec3 = nodes[nb];
        let c: Vec3 = nodes[nc];
        let face: Face = precompute_face(&a, &b, &c);
        let sigma: f64 = mvector.dot(&face.n_hat);

        for i in 0..n_targets {
            let r: Vec3 = Vec3([x[i], y[i], z[i]]);
            let grad_psi: Vec3 = charge_potential_gradient(&face, &r);
            let contribution: Vec3 = grad_psi * (-INV_4PI * sigma);
            hx[i] += contribution[0];
            hy[i] += contribution[1];
            hz[i] += contribution[2];
        }
    }
}
