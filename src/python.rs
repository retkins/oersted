//! Python bindings for oersted

#![allow(unused)]

use numpy::{
    Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

use crate::{
    biotsavart::{self, IntegrationMethod, RequestedField, SourceVectors},
    check_lengths, magnetization,
    math::gradient,
    mesh,
    octree::{self, OctreeSettings},
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

// Transpose 3 column vectors into a row-major (N,3) PyArray2
fn vec3s_to_pyarray(py: Python, cols: Vec<Vec3>) -> Bound<PyArray2<f64>> {
    let n = cols.len();
    let mut out: Vec<f64> = Vec::with_capacity(n * 3);
    for i in 0..n {
        out.push(cols[i][0]);
        out.push(cols[i][1]);
        out.push(cols[i][2]);
    }

    // Reshape is guaranteed to succeed because the memory was allocated as n*3 above
    PyArray1::from_vec(py, out).reshape([n, 3]).unwrap()
}

// Allocate memory for a results buffer filled with zeros
fn col_buffer(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    (vec![0.0; n], vec![0.0; n], vec![0.0; n])
}

// Create a centroid mesh from a tet4 mesh
fn to_centroid_mesh(src_nodes: &[Vec3], src_connectivity: &[[u32; 4]]) -> (Vec<Vec3>, Vec<f64>) {
    let n_src: usize = src_connectivity.len();
    let mut src_centroids: Vec<Vec3> = vec![Vec3::default(); n_src];
    let mut src_volumes: Vec<f64> = vec![0.0; n_src];

    mesh::centroids(src_nodes, src_connectivity, &mut src_centroids);
    mesh::volumes(src_nodes, src_connectivity, &mut src_volumes);

    (src_centroids, src_volumes)
}

#[pyfunction]
fn calculate_fields<'py>(
    py: Python<'py>,
    src_nodes: PyReadonlyArray2<f64>,
    src_connectivity: PyReadonlyArray2<u32>,
    src_vectors: PyReadonlyArray2<f64>,
    src_vector_type: u32,
    requested_field: u32,
    targets: PyReadonlyArray2<f64>,
    element_integration: bool,
    n_threads_requested: u32,
    use_octree: bool,
    theta: f64,
    multipole_order: u32,
    max_leaf_size: u32,
    batch_size: u32,
) -> PyResult<BoundPyArray2f64<'py>> {
    let _src_nodes: &[Vec3] = to_vec3s(src_nodes.as_slice()?);
    let _src_connectivity: &[[u32; 4]] = to_u32x4s(src_connectivity.as_slice()?);
    let mut source;
    let _src_vectors = if src_vector_type == 0 {
        source = octree::Source::CurrentDensity;
        SourceVectors::CurrentDensity(to_vec3s(src_vectors.as_slice()?))
    } else {
        source = octree::Source::Magnetization;
        SourceVectors::Magnetization(to_vec3s(src_vectors.as_slice()?))
    };
    let (x, y, z) = pyarray_to_3cols(targets);

    let n_targets: usize = x.len();
    let (mut fx, mut fy, mut fz) = col_buffer(n_targets);

    let fields = if requested_field == 0 {
        RequestedField::AField
    } else {
        RequestedField::HField
    };

    let method = if element_integration {
        IntegrationMethod::Element
    } else {
        IntegrationMethod::Point
    };

    if use_octree {
        let jdensity = if src_vector_type == 0 {
            Some(to_vec3s(src_vectors.as_slice()?))
        } else {
            panic!("Octree not available yet for magnetization solves.")
        };

        let order = octree::MultipoleOrder::from_int(multipole_order);
        let settings = octree::OctreeSettings {
            theta,
            max_leaf_size,
            multipole_order: order,
            near_field_method: method,
            n_threads_requested,
            batch_size: batch_size as usize,
        };

        let octree: octree::Octree =
            octree::Octree::new(&_src_nodes, &_src_connectivity, jdensity, None, settings);

        octree.compute_fields((&x, &y, &z), (&mut fx, &mut fy, &mut fz), fields, source);
    } else {
        biotsavart::calculate_fields(
            _src_nodes,
            _src_connectivity,
            _src_vectors,
            fields,
            (&x, &y, &z),
            (&mut fx, &mut fy, &mut fz),
            method,
            n_threads_requested,
        );
    }

    Ok(cols_to_pyarray(py, (fx, fy, fz)))
}

#[pyfunction]
fn magnetization_solve<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    connectivity: PyReadonlyArray2<u32>,
    centroids: PyReadonlyArray2<f64>,
    chi: f64,
    h_ext: PyReadonlyArray2<f64>,
    element_integration: bool,
    n_threads_requested: u32,
    atol: f64,
    max_iterations: u32,
    under_relaxation_factor: f64,
    use_octree: bool,
    theta: f64,
    multipole_order: u32,
    max_leaf_size: u32,
    batch_size: u32,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let n_centroids = connectivity.shape()[0];

    let _nodes: &[Vec3] = to_vec3s(nodes.as_slice()?);
    let _connectivity: &[[u32; 4]] = to_u32x4s(connectivity.as_slice()?);
    let (h_extx, h_exty, h_extz) = pyarray_to_3cols(h_ext);
    let (cx, cy, cz) = pyarray_to_3cols(centroids);

    let method = if element_integration {
        IntegrationMethod::Element
    } else {
        IntegrationMethod::Point
    };

    let octree_settings = if use_octree {
        Some(octree::OctreeSettings {
            theta,
            max_leaf_size,
            multipole_order: octree::MultipoleOrder::from_int(multipole_order),
            near_field_method: method,
            n_threads_requested,
            batch_size: batch_size as usize,
        })
    } else {
        None
    };

    let mut m_out = vec![Vec3::default(); n_centroids];
    let (mut hx, mut hy, mut hz) = col_buffer(n_centroids);

    magnetization::magnetization_solve(
        _nodes,
        _connectivity,
        (&cx, &cy, &cz),
        chi,
        (&h_extx, &h_exty, &h_extz),
        &mut m_out,
        (&mut hx, &mut hy, &mut hz),
        method,
        n_threads_requested,
        atol,
        max_iterations,
        under_relaxation_factor,
        octree_settings,
    );

    Ok((
        vec3s_to_pyarray(py, m_out),
        cols_to_pyarray(py, (hx, hy, hz)),
    ))
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
    m.add_function(wrap_pyfunction!(calculate_fields, m.clone())?)?;
    m.add_function(wrap_pyfunction!(magnetization_solve, m.clone())?)?;

    // Mesh functions
    m.add_function(wrap_pyfunction!(mesh_centroids, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_volumes, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_surface_faces, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_surface_face_properties, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_surface_forces, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_tets, m.clone())?)?;
    m.add_function(wrap_pyfunction!(mesh_kelvin_force_density, m.clone())?)?;

    // Math
    m.add_function(wrap_pyfunction!(atan2, m.clone())?)?;

    Ok(())
}
