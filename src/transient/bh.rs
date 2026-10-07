//! Transient solver using Barnes-Hut for inductance matrix acceleration
//!
//! This solver uses the preconditioned conjugate gradient method (PCG) on the
//! projected null space of G^T, a standard technique for solving saddle point
//! problems such as this. In essence, what this enables is:
//! 1. A SPD system, making PCG valid
//! 2. The usage of the most efficient solver (PCG)
//! 3. A smaller system (3*n_elems instead of 3*n_elems + n_nodes)
//!
//! While costing:
//! 1. A sparse Cholesky solve
//! 2. Significantly more complex setup and preconditioning

use crate::{
    biotsavart::{IntegrationMethod, RequestedField, SourceVectors, a_field},
    check_lengths,
    krylov::LinearOperator,
    mesh::Mesh,
    octree::{Octree, OctreeSettings, Source},
    transient::common::{
        TransientOptions, assemble_d, assemble_g, assemble_r, find_cyclic_nodes, find_pin_nodes,
    },
    types::{Vec3, vec3_to_3vec},
};
use faer::{diag::Diag, sparse::SparseColMat};
use ndarray::{Array1, Array3};
use std::cell::RefCell;
use std::f64::consts::PI;

struct KktSystem {
    r: Diag<f64>,
    g: SparseColMat<usize, f64>,
    octree: Octree,
    dt: f64,
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

/// Rotate a position vector around the z-axis by a specified angle
fn rotate(v: &Vec3, angle: f64) -> Vec3 {
    let r = v[0].hypot(v[1]);
    let phi = v[1].atan2(v[0]);
    Vec3([r * (phi + angle).cos(), r * (phi + angle).sin(), v[2]])
}

/// Increment an element's node counts by a specified value
fn increment_element(elem: &[u32; 4], n: usize) -> [u32; 4] {
    [
        elem[0] + n as u32,
        elem[1] + n as u32,
        elem[2] + n as u32,
        elem[3] + n as u32,
    ]
}

/// A Barnes-Hut (octree) operator for calculating portions of the dynamic system,
/// including the self-inductance diagonal (D) and the momentum block (A)
struct BhOperator<'a> {
    octree: RefCell<Octree>,
    n_sectors: usize,
    sector_angle: f64,
    sector_mesh: &'a Mesh,
    n_nodes_in_sector: usize,
    n_elems_in_sector: usize,
    jdensity: RefCell<Vec<Vec3>>,

    update_centers: RefCell<bool>,
    r: Diag<f64>,
    diag_a: Vec<f64>,
    dt: f64,
    targets: (Vec<f64>, Vec<f64>, Vec<f64>),
}

impl<'a> BhOperator<'a> {
    fn new(
        mesh: &'a Mesh,
        n_sectors: usize,
        settings: &OctreeSettings,
        r: &Diag<f64>,
        dt: f64,
    ) -> Self {
        let n_elems_in_sector = mesh.n_elems();
        let n_nodes_in_sector = mesh.n_nodes();
        let sector_angle = 2.0 * PI / n_sectors as f64;

        // Unit x-direction jdensity
        let mut jdensity: Vec<Vec3> = vec![Vec3::default(); n_sectors * mesh.n_elems()];
        for i in 0..jdensity.len() {
            jdensity[i][1] = 1.0;
        }

        let (nodes, connectivity) = if n_sectors > 1 {
            let mut nodes = vec![Vec3::default(); n_sectors * mesh.n_nodes()];
            let mut connectivity = vec![[0u32; 4]; n_sectors * mesh.n_elems()];

            // Copy target sector
            nodes[0..n_nodes_in_sector].copy_from_slice(&mesh.nodes);
            connectivity[0..n_elems_in_sector].copy_from_slice(&mesh.connectivity);

            // All other sectors require a rotation and a node renumbering
            for s in 1..n_sectors {
                for i in 0..n_nodes_in_sector {
                    nodes[s * n_nodes_in_sector + i] = rotate(&nodes[i], sector_angle * s as f64);
                }
                for j in 0..n_elems_in_sector {
                    connectivity[s * n_elems_in_sector + j] =
                        increment_element(&connectivity[j], n_nodes_in_sector);
                }
            }

            (nodes, connectivity)
        } else {
            (mesh.nodes.clone(), mesh.connectivity.clone())
        };

        // Targets in the explicitly modelled sector
        let targets = vec3_to_3vec(&mesh.centroids);

        Self {
            octree: RefCell::new(Octree::new(
                &nodes,
                &connectivity,
                Some(&jdensity),
                None,
                *settings,
            )),
            n_sectors,
            sector_angle: sector_angle,
            sector_mesh: mesh,
            n_nodes_in_sector,
            n_elems_in_sector,
            jdensity: RefCell::new(jdensity),
            update_centers: RefCell::new(false),
            r: r.clone(),
            diag_a: Self::build_diagonal(
                mesh,
                (&targets.0, &targets.1, &targets.2),
                r,
                dt,
                n_sectors,
                sector_angle,
            ),
            dt,
            targets,
        }
    }

    /// Build the momentum diagonal
    /// D = diag(A) -> 3*Ne tall
    /// d[c*Ne+e] = M[e,e]/dt + R[e] (for non cyclic models)
    /// TODO: parallelize this function
    fn build_diagonal(
        mesh: &Mesh,
        targets: (&[f64], &[f64], &[f64]),
        r: &Diag<f64>,
        dt: f64,
        n_sectors: usize,
        sector_angle: f64,
    ) -> Vec<f64> {
        let n_el: usize = mesh.n_elems();
        let mut d: Vec<f64> = vec![0.0; 3 * n_el];

        // Allocate temporary workspace and set jdensity vectors to unit value
        let mut temp_mesh: Mesh = mesh.clone();

        for e in 0..n_el {
            let c = mesh.centroids[e];

            for k in 0..3 {
                let mut acc = 0.0;
                for s in 0..n_sectors {
                    let current_angle = s as f64 * sector_angle;
                    if s > 0 {
                        for e in 0..mesh.n_nodes() {
                            temp_mesh.nodes[e] = rotate(&mesh.nodes[e], current_angle);
                        }
                    }

                    let mut junit = Vec3([0.0, 0.0, 0.0]);
                    junit[k] = 1.0;
                    let j_src = rotate(&junit, current_angle);

                    let connectivity = [temp_mesh.connectivity[e]];

                    let (mut ax, mut ay, mut az) = ([0.0], [0.0], [0.0]);
                    a_field(
                        &temp_mesh.nodes,
                        &connectivity,
                        SourceVectors::CurrentDensity(&[j_src]),
                        (
                            &targets.0[e..e + 1],
                            &targets.1[e..e + 1],
                            &targets.2[e..e + 1],
                        ),
                        (&mut ax, &mut ay, &mut az),
                        IntegrationMethod::Element,
                        1,
                    );
                    acc += [ax[0], ay[0], az[0]][k];
                }
                d[k * n_el + e] = r[e] + acc * mesh.volumes[e] / dt;
            }
        }

        d
    }
}

impl LinearOperator for BhOperator<'_> {
    // Compute A*J = (R + M/dt)*J
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        // `x` is a column of all x's, then all y's
        let (jx, jyz) = x.split_at(self.n_elems_in_sector);
        let (jy, jz) = jyz.split_at(self.n_elems_in_sector);
        let n = check_lengths!(jx, jy, jz);

        // Rotate current density vectors to image sectors
        for i in 0..n {
            self.jdensity.borrow_mut()[i] = Vec3([jx[i], jy[i], jz[i]]);
            for s in 1..self.n_sectors {
                let k = s * self.n_elems_in_sector + i;
                self.jdensity.borrow_mut()[k] =
                    rotate(&Vec3([jx[i], jy[i], jz[i]]), self.sector_angle * s as f64);
            }
        }

        self.octree
            .borrow_mut()
            .update_jdensity(&self.jdensity.borrow(), self.update_centers.take());
        let (out_x, out_yz) = out.split_at_mut(self.n_elems_in_sector);
        let (out_y, out_z) = out_yz.split_at_mut(self.n_elems_in_sector);

        // Compute M*J
        self.octree.borrow_mut().compute_fields(
            (&self.targets.0, &self.targets.1, &self.targets.2),
            (out_x, out_y, out_z),
            RequestedField::AField,
            Source::CurrentDensity,
        );

        let inv_dt = 1.0 / self.dt;
        for c in 0..3 {
            for e in 0..self.n_elems_in_sector {
                let i = c * self.n_elems_in_sector + e;
                out[i] = out[i] * inv_dt + self.r[e] * x[i];
            }
        }
    }

    /// Computing the diagonal requires evaluating the self field of one element and all
    /// of its images around all of the sectors
    fn diagonal(&self, out: &mut [f64]) {
        out.copy_from_slice(&self.diag_a);
    }

    fn len(&self) -> usize {
        self.sector_mesh.n_elems()
    }
}

pub fn solve(
    mesh: &Mesh,
    (rho_values, rho_indices): (&[f64], &[usize]),
    nt: usize,
    tmax: f64,
    a_ext: &Array3<f64>,
    b_ext: &Array3<f64>,
    options: TransientOptions,
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    let n_elem: usize = mesh.n_elems();
    let n_sectors = match &options.cyclic {
        Some(c) => c.n_sectors,
        None => 0usize,
    };

    let dt: f64 = tmax / (nt - 1) as f64;

    // Allocate memory for the time steps and the results data
    // a and b are the TOTAL value at element centroids, including the external
    // sources. Overwriting the external source arrays would save memory, but it
    // may not be what the caller wants to do.
    let mut time: Array1<f64> = Array1::zeros(nt);
    let mut j: Array3<f64> = Array3::zeros((nt, n_elem, 3));
    let mut a: Array3<f64> = Array3::zeros((nt, n_elem, 3));
    let mut b: Array3<f64> = Array3::zeros((nt, n_elem, 3));

    // Compute cyclic ties if requested
    // If cyclic boundary conditions were requested, compute the matching node sets and
    // the reduced node DOF count. If not, just use the total number of nodes and the
    // original node indices for the map.
    let (node_map, n_reduced) = match &options.cyclic {
        Some(c) => find_cyclic_nodes(
            mesh,
            c.n_sectors,
            (c.low_angle, c.high_angle),
            (c.atol_distance, c.atol_angle),
            options.verbose,
        ),
        None => ((0..mesh.n_nodes()).collect(), mesh.n_nodes()),
    };

    // Gauge pinning on the reduced node set
    let grounded: Vec<usize> = if options.cyclic.is_some() {
        let tied: Vec<[u32; 4]> = mesh
            .connectivity
            .iter()
            .map(|e| {
                [
                    node_map[e[0] as usize] as u32,
                    node_map[e[1] as usize] as u32,
                    node_map[e[2] as usize] as u32,
                    node_map[e[3] as usize] as u32,
                ]
            })
            .collect();
        find_pin_nodes(&tied, n_reduced, options.verbose)
    } else {
        // No reduced node set, just gauge pin on the original node set
        find_pin_nodes(&mesh.connectivity, mesh.n_nodes(), options.verbose)
    };

    // Create the Barnes-Hut linear operator

    let n_dof = 3 * n_elem + n_reduced;
    let g: Vec<(usize, usize, f64)> = assemble_g(mesh, &node_map, n_reduced, &grounded);
    let r = assemble_r(mesh, rho_values, rho_indices);
    let m = BhOperator::new(mesh, n_sectors, &options.octree_settings, &r, dt);
    let d = assemble_d(mesh, &r, dt, &m);
    // let l = assemble_l(&g, &d, n_elem, n_reduced);

    (time, j, a, b)
}
