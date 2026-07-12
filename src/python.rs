//! Python bindings for oersted

#![allow(unused)]

use numpy::{
    Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

use crate::{
    biotsavart, biotsavart_parallel, check_lengths, magnetization,
    math::gradient,
    mesh,
    octree::{CurrentSources, DipoleSources, HFieldSolver, Octree, point, tet_element},
    octree_lists,
    types::{Vec3, to_u32x4s, to_vec3s, to_vec3s_mut},
};

type BoundPyArray1f64<'py> = Bound<'py, PyArray1<f64>>;
type BoundPyArray2f64<'py> = Bound<'py, PyArray2<f64>>;

// ---
// Helpers
// ---

// Transpose a row-major (N,3) Python array to column vectors for SIMD operations in Rust
fn pyarray_to_3cols<T: Element + Copy>(arr: PyReadonlyArray2<T>) -> (Vec<T>, Vec<T>, Vec<T>) {
    let n = arr.shape()[0];
    let mut x: Vec<T> = Vec::with_capacity(n);
    let mut y: Vec<T> = Vec::with_capacity(n);
    let mut z: Vec<T> = Vec::with_capacity(n);
    for row in arr.as_array().rows() {
        x.push(row[0]);
        y.push(row[1]);
        z.push(row[2]);
    }
    (x, y, z)
}

// Copy a row-major (N,3) Python array to a vector of Vec3 for octree operations in Rust
fn pyarray_to_vec3(arr: PyReadonlyArray2<f64>) -> Vec<Vec3> {
    let n = arr.shape()[0];
    let mut out: Vec<Vec3> = Vec::with_capacity(n);

    for row in arr.as_array().rows() {
        out.push(Vec3([row[0], row[1], row[2]]));
    }
    out
}

// Transpose 3 column vectors into a row-major (N,3) PyArray2
fn cols_to_pyarray(py: Python, cols: (Vec<f64>, Vec<f64>, Vec<f64>)) -> Bound<PyArray2<f64>> {
    let (x, y, z) = cols;
    let n = x.len();
    assert_eq!(n, y.len());
    assert_eq!(n, z.len());
    let mut out: Vec<f64> = Vec::with_capacity(n * 3);
    for i in 0..n {
        out.push(x[i]);
        out.push(y[i]);
        out.push(z[i]);
    }

    // Reshape is guaranteed to succeed because the memory was allocated as n*3 above
    PyArray1::from_vec(py, out).reshape([n, 3]).unwrap()
}

// Allocate memory for a results buffer filled with zeros
fn col_buffer(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    (vec![0.0; n], vec![0.0; n], vec![0.0; n])
}

// ---
// Current Density Source Functions
// ---

#[pyfunction]
fn h_current_point_direct<'py>(
    py: Python<'py>,
    src_pts: PyReadonlyArray2<f64>,
    src_vol: PyReadonlyArray1<f64>,
    src_jdensity: PyReadonlyArray2<f64>,
    tgt_pts: PyReadonlyArray2<f64>,
    nthreads_requested: u32,
) -> PyResult<BoundPyArray2f64<'py>> {
    // Transpose input data and allocate output arrays
    let n = tgt_pts.shape()[0];
    let (centx, centy, centz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(src_pts);
    let _src_vol: &[f64] = src_vol.as_slice()?;
    let (jx, jy, jz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(src_jdensity);
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(tgt_pts);
    let (mut bx, mut by, mut bz): (Vec<f64>, Vec<f64>, Vec<f64>) = col_buffer(n);

    biotsavart_parallel::h_current_point_direct_parallel(
        (&centx, &centy, &centz),
        _src_vol,
        (&jx, &jy, &jz),
        (&x, &y, &z),
        (&mut bx, &mut by, &mut bz),
        nthreads_requested,
    );

    Ok(cols_to_pyarray(py, (bx, by, bz)))
}

#[pyfunction]
fn h_current_point_octree<'py>(
    py: Python<'py>,
    src_pts: PyReadonlyArray2<f64>,
    src_vol: PyReadonlyArray1<f64>,
    src_jdensity: PyReadonlyArray2<f64>,
    tgt_pts: PyReadonlyArray2<f64>,
    theta: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
) -> PyResult<BoundPyArray2f64<'py>> {
    // Transpose input data and allocate output arrays
    let n = tgt_pts.shape()[0];
    let (centx, centy, centz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(src_pts);
    let _src_vol: &[f64] = src_vol.as_slice()?;
    let (jx, jy, jz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(src_jdensity);
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(tgt_pts);
    let (mut bx, mut by, mut bz): (Vec<f64>, Vec<f64>, Vec<f64>) = col_buffer(n);

    let mut sources: CurrentSources<point::PointSources> = CurrentSources(
        point::PointSources::new((&centx, &centy, &centz), _src_vol, (&jx, &jy, &jz)),
    );
    let max_depth: u8 = 21;
    let tree: Octree<CurrentSources<point::PointSources>> =
        Octree::build_from_sources(sources, max_depth, leaf_threshold);

    tree.h_field_parallel(
        (&x, &y, &z),
        (&mut bx, &mut by, &mut bz),
        theta,
        nthreads_requested,
    );

    Ok(cols_to_pyarray(py, (bx, by, bz)))
}

#[pyfunction]
fn a_current<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    jdensity: PyReadonlyArray2<f64>,
    targets: PyReadonlyArray2<f64>,
    exact_integration: bool,
    n_threads_requested: u32,
    use_octree: bool,
    theta: f64,
) -> PyResult<BoundPyArray2f64<'py>> {
    let _nodes = to_vec3s(nodes.as_slice()?);
    let _connectivity = to_u32x4s(connectivity.as_slice()?);
    let _jdensity = to_vec3s(jdensity.as_slice()?);
    let (x, y, z) = pyarray_to_3cols(targets);
    let n_tgts = x.len();
    let (mut ax, mut ay, mut az) = col_buffer(n_tgts);

    if use_octree {
        return Err(PyErr::new::<PyNotImplementedError, _>(
            "Octree is not yet available for a-field solves",
        ));
    } else {
        if exact_integration {
            biotsavart::a_current_tet4_direct(
                &_nodes,
                &_connectivity,
                &_jdensity,
                (&x, &y, &z),
                (&mut ax, &mut ay, &mut az),
                n_threads_requested,
            );
        } else {
            let n_src = _connectivity.len();
            let mut src_centroids = vec![Vec3::default(); n_src];
            mesh::centroids(_nodes, _connectivity, &mut src_centroids);
            let mut src_volumes = vec![0.0; n_src];
            mesh::volumes(_nodes, _connectivity, &mut src_volumes);
            biotsavart::a_current_point_direct(
                &src_centroids,
                &src_volumes,
                &_jdensity,
                (&x, &y, &z),
                (&mut ax, &mut ay, &mut az),
                n_threads_requested,
            )
        }
    }

    Ok(cols_to_pyarray(py, (ax, ay, az)))
}

#[pyfunction]
pub fn h_current_tet4_direct<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    jdensity: PyReadonlyArray2<f64>,
    tgts: PyReadonlyArray2<f64>,
    nthreads_requested: u32,
    edge: bool,
) -> PyResult<BoundPyArray2f64<'py>> {
    // Transpose input data and allocate output arrays
    let n_tgts: usize = tgts.shape()[0];
    let (x, y, z) = pyarray_to_3cols(tgts);
    let (mut hx, mut hy, mut hz) = col_buffer(n_tgts);

    biotsavart_parallel::h_field_tet4_direct_parallel(
        nodes.as_slice()?,
        connectivity.as_slice()?,
        jdensity.as_slice()?,
        (&x, &y, &z),
        (&mut hx, &mut hy, &mut hz),
        nthreads_requested,
        edge,
    );

    Ok(cols_to_pyarray(py, (hx, hy, hz)))
}

#[pyfunction]
fn h_current_tet4_octree<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    jdensity: PyReadonlyArray2<f64>,
    tgt_pts: PyReadonlyArray2<f64>,
    theta: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
) -> PyResult<BoundPyArray2f64<'py>> {
    let n_srcs: usize = connectivity.shape()[0];
    let n_tgts: usize = tgt_pts.shape()[0];
    let max_depth: u8 = 21;
    let mut centroids: Vec<Vec3> = vec![Vec3::default(); n_srcs];
    let mut volumes: Vec<f64> = vec![0.0; n_srcs];
    let _nodes = to_vec3s(nodes.as_slice()?);
    let _connectivity = to_u32x4s(connectivity.as_slice()?);

    let (x, y, z) = pyarray_to_3cols(tgt_pts);
    let (mut hx, mut hy, mut hz) = col_buffer(n_tgts);

    mesh::centroids(_nodes, _connectivity, &mut centroids);
    mesh::volumes(_nodes, _connectivity, &mut volumes);

    let mut sources: CurrentSources<tet_element::TetSources> =
        CurrentSources(tet_element::TetSources::new(
            _nodes,
            _connectivity,
            &centroids,
            &volumes,
            to_vec3s(jdensity.as_slice()?),
        ));

    let tree: Octree<CurrentSources<tet_element::TetSources>> =
        Octree::build_from_sources(sources, max_depth, leaf_threshold);

    tree.h_field_parallel(
        (&x, &y, &z),
        (&mut hx, &mut hy, &mut hz),
        theta,
        nthreads_requested,
    );
    Ok(cols_to_pyarray(py, (hx, hy, hz)))
}

#[pyfunction]
fn h_mag_point<'py>(
    py: Python<'py>,
    centroids: PyReadonlyArray2<f64>,
    volumes: PyReadonlyArray1<f64>,
    mvectors: PyReadonlyArray2<f64>,
    targets: PyReadonlyArray2<f64>,
    theta: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
    use_octree: bool,
) -> PyResult<BoundPyArray2f64<'py>> {
    let n_targets = targets.shape()[0];
    let mut out = col_buffer(n_targets);

    if use_octree {
        let (centx, centy, centz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(centroids);
        let vol: &[f64] = volumes.as_slice()?;
        let (mx, my, mz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(mvectors);
        let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(targets);

        let mut sources: DipoleSources<point::PointSources> = DipoleSources(
            point::PointSources::new((&centx, &centy, &centz), vol, (&mx, &my, &mz)),
        );
        let max_depth: u8 = 21;
        let tree: Octree<DipoleSources<point::PointSources>> =
            Octree::build_from_sources(sources, max_depth, leaf_threshold);

        tree.h_field_parallel(
            (&x, &y, &z),
            (&mut out.0, &mut out.1, &mut out.2),
            theta,
            nthreads_requested,
        );
    } else {
        let _centroids = pyarray_to_vec3(centroids);
        let _volumes = volumes.as_slice()?;
        let _mvectors = pyarray_to_vec3(mvectors);
        let _targets = pyarray_to_3cols(targets);
        biotsavart_parallel::h_mag_point_direct_parallel(
            &_centroids,
            &_volumes,
            &_mvectors,
            (&_targets.0, &_targets.1, &_targets.2),
            (&mut out.0, &mut out.1, &mut out.2),
            nthreads_requested,
        );
    }

    Ok(cols_to_pyarray(py, out))
}

#[pyfunction]
fn h_mag_tet4<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    mvectors: PyReadonlyArray2<f64>,
    targets: PyReadonlyArray2<f64>,
    theta: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
    use_octree: bool,
    edge: bool,
) -> PyResult<BoundPyArray2f64<'py>> {
    let _nodes = to_vec3s(nodes.as_slice()?);
    let _connectivity = to_u32x4s(connectivity.as_slice()?);
    let _mvectors = to_vec3s(mvectors.as_slice()?);
    let (x, y, z) = pyarray_to_3cols(targets);
    let n_tgts = x.len();
    let (mut hx, mut hy, mut hz) = col_buffer(n_tgts);

    if use_octree {
        let n_sources: usize = connectivity.shape()[0];
        let mut centroids: Vec<Vec3> = vec![Vec3::default(); n_sources];
        let mut volumes: Vec<f64> = vec![0.0; n_sources];
        mesh::centroids(_nodes, _connectivity, &mut centroids);
        mesh::volumes(_nodes, _connectivity, &mut volumes);
        let sources: DipoleSources<tet_element::TetSources> = DipoleSources(
            tet_element::TetSources::new(_nodes, _connectivity, &centroids, &volumes, _mvectors),
        );
        let max_depth: u8 = 21;
        let tree: Octree<DipoleSources<tet_element::TetSources>> =
            Octree::build_from_sources(sources, max_depth, leaf_threshold);

        // Evaluate
        tree.h_field((&x, &y, &z), (&mut hx, &mut hy, &mut hz), theta);
    } else {
        // User requested a direct integration solve
        biotsavart_parallel::h_mag_tet4_direct_parallel(
            _nodes,
            _connectivity,
            _mvectors,
            (&x, &y, &z),
            (&mut hx, &mut hy, &mut hz),
            nthreads_requested,
            edge,
        );
    }

    Ok(cols_to_pyarray(py, (hx, hy, hz)))
}

#[pyfunction]
fn magnetization_tet4<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    chi: f64,
    hext: PyReadonlyArray2<f64>,
    tol: f64,
    max_iterations: u32,
    theta: f64,
    leaf_threshold: u32,
    alpha: f64,
    nthreads_requested: u32,
    edge: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let n_centroids = connectivity.shape()[0];

    let _nodes: &[Vec3] = to_vec3s(nodes.as_slice()?);
    let _connectivity: &[[u32; 4]] = to_u32x4s(connectivity.as_slice()?);
    let (hextx, hexty, hextz) = pyarray_to_3cols(hext);
    let mut centroids_flat: Vec<f64> = vec![0.0; n_centroids * 3];
    mesh::centroids(_nodes, _connectivity, to_vec3s_mut(&mut centroids_flat));

    // TODO: write simple function to do this
    let (mut cx, mut cy, mut cz) = (
        vec![0.0; n_centroids],
        vec![0.0; n_centroids],
        vec![0.0; n_centroids],
    );
    for i in 0..n_centroids {
        cx[i] = centroids_flat[i * 3];
        cy[i] = centroids_flat[i * 3 + 1];
        cz[i] = centroids_flat[i * 3 + 2];
    }

    let solver = if leaf_threshold < 1 {
        magnetization::Solver::Tet4Direct(nthreads_requested)
    } else {
        magnetization::Solver::Tet4Octree(nthreads_requested, theta, leaf_threshold)
    };

    let (htotal, mfield) = magnetization::magnetization(
        _nodes,
        _connectivity,
        (&cx, &cy, &cz),
        chi,
        (&hextx, &hexty, &hextz),
        solver,
        tol,
        max_iterations,
        alpha,
        edge,
    );

    let (mut mx, mut my, mut mz) = (
        vec![0.0; n_centroids],
        vec![0.0; n_centroids],
        vec![0.0; n_centroids],
    );

    for i in 0..n_centroids {
        mx[i] = mfield[i][0];
        my[i] = mfield[i][1];
        mz[i] = mfield[i][2];
    }

    Ok((
        cols_to_pyarray(py, (mx, my, mz)),
        cols_to_pyarray(py, htotal),
    ))
}

// ---
// Interaction list octree functions
// ---

#[pyfunction]
fn interaction_lists<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    targets: PyReadonlyArray2<f64>,
    leaf_threshold: u32,
    alpha: f64,
    theta: f64,
) -> PyResult<(
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray2<u32>>,
)> {
    let _nodes = pyarray_to_vec3(nodes);
    let _connectivity = to_u32x4s(connectivity.as_slice()?);
    let _targets = pyarray_to_3cols(targets);

    let tree = octree_lists::Octree::new(&_nodes, &_connectivity, None, None, leaf_threshold);

    use std::time::Instant;
    let now = Instant::now();
    let (mut near, mut mid, mut far) =
        tree.traverse((&_targets.0, &_targets.1, &_targets.2), alpha, theta);
    println!("Traversal time: {:.3} sec", now.elapsed().as_secs_f64());

    let near_len = near.len();
    let mid_len = mid.len();
    let far_len = far.len();

    near.source_indices.extend(&near.target_indices);
    mid.source_indices.extend(&mid.target_indices);
    far.source_indices.extend(&far.target_indices);

    let arr1 = PyArray1::from_vec(py, near.source_indices)
        .reshape([2, near_len])?
        .transpose()?;
    let arr2 = PyArray1::from_vec(py, mid.source_indices)
        .reshape([2, mid_len])?
        .transpose()?;
    let arr3 = PyArray1::from_vec(py, far.source_indices)
        .reshape([2, far_len])?
        .transpose()?;

    Ok((arr1, arr2, arr3))
}

#[pyfunction]
fn h_current_octree<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    targets: PyReadonlyArray2<f64>,
    jdensity: PyReadonlyArray2<f64>,
    leaf_threshold: u32,
    alpha: f64,
    theta: f64,
    n_threads_requested: u32,
) -> PyResult<(Bound<'py, PyArray2<f64>>)> {
    let n_targets = targets.shape()[0];
    let _nodes = to_vec3s(nodes.as_slice()?);
    let _connectivity = to_u32x4s(connectivity.as_slice()?);
    let _targets = pyarray_to_3cols(targets);
    let _jdensity = to_vec3s(jdensity.as_slice()?);

    let octree: octree_lists::Octree = octree_lists::Octree::new(
        &_nodes,
        &_connectivity,
        Some(&_jdensity),
        None,
        leaf_threshold,
    );

    let (mut hx, mut hy, mut hz) = col_buffer(n_targets);

    octree.h_current_parallel(
        (&_targets.0, &_targets.1, &_targets.2),
        alpha,
        theta,
        (&mut hx, &mut hy, &mut hz),
        n_threads_requested,
    );

    Ok(cols_to_pyarray(py, (hx, hy, hz)))
}

// ---
// Mesh Operations
// ---

#[pyfunction]
fn mesh_centroids<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n_centroids: usize = connectivity.shape()[0];
    let mut out: Vec<f64> = vec![0.0; n_centroids * 3];

    mesh::centroids(
        to_vec3s(nodes.as_slice()?),
        to_u32x4s(connectivity.as_slice()?),
        to_vec3s_mut(&mut out),
    );

    PyArray1::from_vec(py, out).reshape([n_centroids, 3])
}

#[pyfunction]
fn mesh_volumes<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n_elements = connectivity.shape()[0];

    let mut out: Vec<f64> = vec![0.0; n_elements];
    mesh::volumes(
        to_vec3s(nodes.as_slice()?),
        to_u32x4s(connectivity.as_slice()?),
        &mut out,
    );
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
fn mesh_surface_faces<'py>(
    py: Python<'py>,
    connectivity: PyReadonlyArray2<u32>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let surface_faces = mesh::surface_faces(connectivity.as_slice()?);
    let n = surface_faces.len() / 3;
    let result = PyArray1::from_vec(py, surface_faces);
    result.reshape([n, 3])
}

#[pyfunction]
fn mesh_surface_face_properties<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    faces: PyReadonlyArray2<u32>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let n_faces = faces.shape()[0];
    let (areas, centroids, normals) =
        mesh::surface_face_properties(nodes.as_slice()?, faces.as_slice()?);

    let area_out = PyArray1::from_vec(py, areas);
    let centroids_out = PyArray1::from_vec(py, centroids);
    let normals_out = PyArray1::from_vec(py, normals);
    Ok((
        area_out,
        centroids_out.reshape([n_faces, 3])?,
        normals_out.reshape([n_faces, 3])?,
    ))
}

#[pyfunction]
fn mesh_surface_forces<'py>(
    py: Python<'py>,
    face_areas: PyReadonlyArray1<f64>,
    face_normals: PyReadonlyArray2<f64>,
    b_field: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let stress_tensor = mesh::maxwell_stress_tensor(b_field.as_slice()?);
    let forces = mesh::surface_forces(
        face_areas.as_slice()?,
        face_normals.as_slice()?,
        &stress_tensor,
    );
    let n_faces = stress_tensor.len();
    let mut forces_out = vec![0.0; n_faces * 3];
    for i in 0..n_faces {
        forces_out[i * 3] = forces[i][0];
        forces_out[i * 3 + 1] = forces[i][1];
        forces_out[i * 3 + 2] = forces[i][2];
    }

    PyArray1::from_vec(py, forces_out).reshape([forces.len(), 3])
}

#[pyfunction]
fn mesh_kelvin_force_density<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    m_field_centroids: PyReadonlyArray2<f64>,
    h_field_nodes: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n_elements = connectivity.shape()[0];
    let _nodes = pyarray_to_vec3(nodes);
    let _elements = to_u32x4s(connectivity.as_slice()?);
    let m_field = pyarray_to_vec3(m_field_centroids);
    let jinvt = gradient::jmatrices(&_nodes, &_elements);

    let (hx, hy, hz) = pyarray_to_3cols(h_field_nodes);
    let (mut grad_hx, mut grad_hy, mut grad_hz) = (
        vec![Vec3::default(); n_elements],
        vec![Vec3::default(); n_elements],
        vec![Vec3::default(); n_elements],
    );

    let gvectors_x = gradient::gvectors(&_elements, &hx);
    let gvectors_y = gradient::gvectors(&_elements, &hy);
    let gvectors_z = gradient::gvectors(&_elements, &hz);

    for i in 0..n_elements {
        grad_hx[i] = gradient(&jinvt[i], &gvectors_x[i]);
        grad_hy[i] = gradient(&jinvt[i], &gvectors_y[i]);
        grad_hz[i] = gradient(&jinvt[i], &gvectors_z[i]);
    }

    let (mut fx, mut fy, mut fz) = col_buffer(n_elements);

    for i in 0..n_elements {
        fx[i] = m_field[i].dot(&grad_hx[i]);
        fy[i] = m_field[i].dot(&grad_hy[i]);
        fz[i] = m_field[i].dot(&grad_hz[i]);
    }

    Ok(cols_to_pyarray(py, (fx, fy, fz)))
}

#[pyfunction]
fn _mesh_surface_tets<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<f64>,
    faces: PyReadonlyArray1<u32>,
    centroids: PyReadonlyArray1<f64>,
    normals: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u32>>)> {
    let (nodes_out, connectivity_out) = mesh::surface_tets(
        nodes.as_slice()?,
        faces.as_slice()?,
        centroids.as_slice()?,
        normals.as_slice()?,
    );
    let n_faces = connectivity_out.len() / 4;
    Ok((
        PyArray1::from_vec(py, nodes_out),        //.reshape([n_faces, 3])?,
        PyArray1::from_vec(py, connectivity_out), //.reshape([n_faces, 3])?,
    ))
}

#[pyfunction]
fn atan2<'py>(
    py: Python<'py>,
    yvals: PyReadonlyArray1<f64>,
    xvals: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>)> {
    let _yvals = yvals.as_slice()?;
    let _xvals = xvals.as_slice()?;
    check_lengths!(_yvals, _xvals);
    let mut result = vec![0.0; _yvals.len()];

    for (i, (&y, &x)) in _yvals.iter().zip(_xvals.iter()).enumerate() {
        result[i] = crate::math::atan2(y, x);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pymodule]
fn _oersted<'py>(_py: Python, m: Bound<'py, PyModule>) -> PyResult<()> {
    // Field calculations
    m.add_function(wrap_pyfunction!(h_current_point_direct, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_current_point_octree, m.clone())?)?;
    m.add_function(wrap_pyfunction!(a_current, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_current_tet4_direct, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_current_tet4_octree, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_mag_point, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_mag_tet4, m.clone())?)?;
    m.add_function(wrap_pyfunction!(magnetization_tet4, m.clone())?)?;

    // Mesh functions
    m.add_function(wrap_pyfunction!(mesh_centroids, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_volumes, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_surface_faces, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_surface_face_properties, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_surface_forces, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_tets, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_kelvin_force_density, m.clone())?)?;

    // Interaction lists
    m.add_function(wrap_pyfunction!(interaction_lists, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_current_octree, m.clone())?)?;

    // Math
    m.add_function(wrap_pyfunction!(atan2, m.clone())?)?;

    Ok(())
}
