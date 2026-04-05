//! Python bindings for oersted

#![allow(unused)]

use numpy::{
    Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use crate::{
    biotsavart_parallel, mesh,
    octree::{CurrentSources, DipoleSources, HFieldSolver, Octree, point, tet_element},
    types::{Vec3, to_u32x4s, to_vec3s, to_vec3s_mut},
};

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

#[pyfunction]
fn b_current_point_direct<'py>(
    py: Python<'py>,
    src_pts: PyReadonlyArray2<f64>,
    src_vol: PyReadonlyArray1<f64>,
    src_jdensity: PyReadonlyArray2<f64>,
    tgt_pts: PyReadonlyArray2<f64>,
    nthreads_requested: u32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Transpose input data and allocate output arrays
    let n = tgt_pts.shape()[0];
    let (centx, centy, centz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(src_pts);
    let _src_vol: &[f64] = src_vol.as_slice()?;
    let (jx, jy, jz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(src_jdensity);
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(tgt_pts);
    let (mut bx, mut by, mut bz): (Vec<f64>, Vec<f64>, Vec<f64>) = col_buffer(n);

    biotsavart_parallel::bfield_direct_parallel(
        (&centx, &centy, &centz),
        &_src_vol,
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Transpose input data and allocate output arrays
    let n = tgt_pts.shape()[0];
    let (centx, centy, centz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(src_pts);
    let _src_vol: &[f64] = src_vol.as_slice()?;
    let (jx, jy, jz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(src_jdensity);
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_3cols(tgt_pts);
    let (mut bx, mut by, mut bz): (Vec<f64>, Vec<f64>, Vec<f64>) = col_buffer(n);

    let mut sources: CurrentSources<point::PointSources> = CurrentSources(
        point::PointSources::new((&centx, &centy, &centz), &_src_vol, (&jx, &jy, &jz)),
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
pub fn h_current_tet4_direct<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    jdensity: PyReadonlyArray2<f64>,
    tgts: PyReadonlyArray2<f64>,
    nthreads_requested: u32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Transpose input data and allocate output arrays
    let n_tgts: usize = tgts.shape()[0];
    let (x, y, z) = pyarray_to_3cols(tgts);
    let (mut hx, mut hy, mut hz) = col_buffer(n_tgts);

    biotsavart_parallel::hfield_direct_tet_parallel(
        nodes.as_slice()?,
        connectivity.as_slice()?,
        jdensity.as_slice()?,
        (&x, &y, &z),
        (&mut hx, &mut hy, &mut hz),
        nthreads_requested,
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
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
fn _hfield_dipole_tetrahedrons<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    magnetization: PyReadonlyArray2<f64>,
    tgt_pts: PyReadonlyArray2<f64>,
    theta: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n_srcs = connectivity.shape()[0];
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

    let mut sources: DipoleSources<tet_element::TetSources> =
        DipoleSources(tet_element::TetSources::new(
            _nodes,
            _connectivity,
            &centroids,
            &volumes,
            to_vec3s(magnetization.as_slice()?),
        ));
    let max_depth: u8 = 21;

    let tree: Octree<DipoleSources<tet_element::TetSources>> =
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
fn _hfield_dipole(
    centx: PyReadonlyArray1<f64>,
    centy: PyReadonlyArray1<f64>,
    centz: PyReadonlyArray1<f64>,
    vol: PyReadonlyArray1<f64>,
    mx: PyReadonlyArray1<f64>,
    my: PyReadonlyArray1<f64>,
    mz: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    mut hx: PyReadwriteArray1<f64>,
    mut hy: PyReadwriteArray1<f64>,
    mut hz: PyReadwriteArray1<f64>,
    theta: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
) -> PyResult<()> {
    use crate::octree::{DipoleSources, Octree, point::PointSources};
    let sources = DipoleSources(PointSources::new_dipole(
        centx.as_slice()?,
        centy.as_slice()?,
        centz.as_slice()?,
        vol.as_slice()?,
        mx.as_slice()?,
        my.as_slice()?,
        mz.as_slice()?,
    ));
    let max_depth: u8 = 21;
    let tree = Octree::build_from_sources(sources, max_depth, leaf_threshold);

    // Evaluate
    tree.h_field(
        (x.as_slice()?, y.as_slice()?, z.as_slice()?),
        (hx.as_slice_mut()?, hy.as_slice_mut()?, hz.as_slice_mut()?),
        theta,
    );

    Ok(())
}

#[pyfunction]
fn h_mag_tet4_direct<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    mvectors: PyReadonlyArray2<f64>,
    targets: PyReadonlyArray2<f64>,
    nthreads_requested: u32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _nodes = to_vec3s(nodes.as_slice()?);
    let _connectivity = to_u32x4s(connectivity.as_slice()?);
    let _mvectors = to_vec3s(mvectors.as_slice()?);
    let (x, y, z) = pyarray_to_3cols(targets);
    let n_tgts = x.len();
    let (mut hx, mut hy, mut hz) = col_buffer(n_tgts);

    biotsavart_parallel::h_mag_tet4_direct_parallel(
        &_nodes,
        &_connectivity,
        &_mvectors,
        (&x, &y, &z),
        (&mut hx, &mut hy, &mut hz),
        nthreads_requested,
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
fn _mesh_surface_faces<'py>(
    py: Python<'py>,
    connectivity: PyReadonlyArray1<u32>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let surface_faces = mesh::surface_faces(&connectivity.as_slice()?);
    let n = surface_faces.len() / 3;
    let result = PyArray1::from_vec(py, surface_faces);
    result.reshape([n, 3])
}

#[pyfunction]
fn _mesh_surface_face_properties<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<f64>,
    faces: PyReadonlyArray1<u32>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let n_faces = faces.as_slice()?.len() / (3 as usize);
    let (areas, centroids, normals) =
        mesh::surface_face_properties(&nodes.as_slice()?, faces.as_slice()?);

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
fn _mesh_surface_forces<'py>(
    py: Python<'py>,
    face_areas: PyReadonlyArray1<f64>,
    face_normals: PyReadonlyArray1<f64>,
    b_field: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let stress_tensor = mesh::maxwell_stress_tensor(&b_field.as_slice()?);
    let forces = mesh::surface_forces(
        &face_areas.as_slice()?,
        &face_normals.as_slice()?,
        &stress_tensor,
    );
    let n_faces = stress_tensor.len();
    let mut forces_out = vec![0.0; n_faces * 3];
    for i in 0..n_faces {
        forces_out[i * 3] = forces[i][0];
        forces_out[i * 3 + 1] = forces[i][1];
        forces_out[i * 3 + 2] = forces[i][2];
    }

    Ok(PyArray1::from_vec(py, forces_out).reshape([forces.len(), 3])?)
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
        &nodes.as_slice()?,
        &faces.as_slice()?,
        &centroids.as_slice()?,
        &normals.as_slice()?,
    );
    let n_faces = connectivity_out.len() / 4;
    Ok((
        PyArray1::from_vec(py, nodes_out),        //.reshape([n_faces, 3])?,
        PyArray1::from_vec(py, connectivity_out), //.reshape([n_faces, 3])?,
    ))
}

#[pymodule]
fn _oersted<'py>(_py: Python, m: Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(b_current_point_direct, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_current_point_octree, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_current_tet4_direct, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_current_tet4_octree, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_hfield_dipole, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_hfield_dipole_tetrahedrons, m.clone())?)?;
    m.add_function(wrap_pyfunction!(h_mag_tet4_direct, m.clone())?)?;

    // Mesh functions
    m.add_function(wrap_pyfunction!(mesh_centroids, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_volumes, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_faces, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_face_properties, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_forces, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_tets, m.clone())?)?;

    Ok(())
}
