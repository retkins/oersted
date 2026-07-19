#![allow(unused, non_snake_case)]

use crate::{
    INV_4PI, MU0_4PI,
    biotsavart::{IntegrationMethod, Kernel, RequestedField, SourceVectors, select_kernel},
    check_lengths,
    octree::Source,
    types::{Mat3, Vec3},
};

#[derive(Clone, Copy, Debug)]
pub enum MultipoleOrder {
    Monopole,
    Dipole,
    Octupole,
}

impl MultipoleOrder {
    pub fn from_int(order: u32) -> Self {
        match order {
            1 => Self::Dipole,
            2 => Self::Octupole,
            _ => Self::Monopole,
        }
    }
}

// Node centroid, monopole, dipole, targets, out
pub type FarKernel =
    fn(&Vec3, &Vec3, &Mat3, (&[f64], &[f64], &[f64]), (&mut [f64], &mut [f64], &mut [f64]));

pub fn select_near_kernel(
    field: RequestedField,
    source: Source,
    method: IntegrationMethod,
) -> Kernel {
    // TODO: make this not require a vec internally
    let src_vectors = match source {
        Source::CurrentDensity => SourceVectors::CurrentDensity(&[Vec3::default()]),
        Source::Magnetization => SourceVectors::Magnetization(&[Vec3::default()]),
    };
    select_kernel(field, &src_vectors, method)
}

pub fn select_far_kernel(field: RequestedField, source: Source, _: MultipoleOrder) -> FarKernel {
    match (field, source) {
        (RequestedField::AField, Source::CurrentDensity) => a_current_node,
        (RequestedField::HField, Source::CurrentDensity) => h_current_node,
        (RequestedField::AField, Source::Magnetization) => a_mag_node,
        (RequestedField::HField, Source::Magnetization) => h_mag_node,
    }
}

/// Compute the antisymmetric part of the dipole tensor
fn m_from_dipole(dipole: &Mat3) -> Vec3 {
    // m[k] = 0.5 * (M[j,i] - M[i,j])
    Vec3([
        0.5 * (dipole[2][1] - dipole[1][2]),
        0.5 * (dipole[0][2] - dipole[2][0]),
        0.5 * (dipole[1][0] - dipole[0][1]),
    ])
}

pub fn a_current_node(
    centroid: &Vec3,
    monopole: &Vec3,
    dipole: &Mat3,
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    let (x, y, z) = targets;
    let (ax, ay, az) = out;
    let n_targets = check_lengths!(x, y, z, ax, ay, az);
    let p = *monopole;
    let D = *dipole;

    for i in 0..n_targets {
        let target = Vec3([x[i], y[i], z[i]]);
        let rp = target - *centroid;

        let r2 = rp.dot(&rp);
        let inv_r = 1.0 / r2.sqrt();
        let rhat = rp * inv_r;
        let inv_r2 = inv_r * inv_r;

        let a = (p * inv_r + D.mul_vec(&rhat) * inv_r2) * MU0_4PI;
        ax[i] += a[0];
        ay[i] += a[1];
        az[i] += a[2];
    }
}

pub fn h_current_node(
    centroid: &Vec3,
    monopole: &Vec3,
    dipole: &Mat3,
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    let (x, y, z) = targets;
    let (hx, hy, hz) = out;
    let n_targets = check_lengths!(x, y, z, hx, hy, hz);

    for i in 0..n_targets {
        let target = Vec3([x[i], y[i], z[i]]);
        let rp = target - *centroid;

        let r2 = rp.dot(&rp);
        let inv_r = 1.0 / r2.sqrt();
        let rhat = rp * inv_r;
        let inv_r2 = inv_r * inv_r;
        let inv_r3 = inv_r * inv_r2;

        let m2 = m_from_dipole(dipole) * 2.0;

        // Monopole contribution: h += 1/(4*pi) * jv / |rhat|^2
        let h_monopole: Vec3 = monopole.cross(&rhat) * (INV_4PI * inv_r2);

        // Dipole contribution: h += 1/(4pi * |r|^3) * (2m - 3rhat x (M rhat))
        let h_dipole: Vec3 = (m2 - rhat.cross(&dipole.mul_vec(&rhat)) * 3.0) * (INV_4PI * inv_r3);
        let h = h_monopole + h_dipole;
        hx[i] += h[0];
        hy[i] += h[1];
        hz[i] += h[2];
    }
}

pub fn a_mag_node(
    centroid: &Vec3,
    monopole: &Vec3,
    dipole: &Mat3,
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    panic!("Not yet implemented");
}

pub fn h_mag_node(
    centroid: &Vec3,
    monopole: &Vec3,
    dipole: &Mat3,
    targets: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    panic!("Not yet implemented");
}
