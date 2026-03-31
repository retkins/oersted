#![allow(unused)]

use std::cmp::max;

use numpy::datetime::units::Years;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use crate::biotsavart;
#[cfg(feature = "parallel")]
use crate::biotsavart_parallel;

use crate::mesh;
use crate::octree::{CurrentSources, HFieldSolver, Octree, point};
use crate::vec3::Vec3;

// ---
// Helpers
// ---

// Transpose a row-major (N,3) PyArray2 to column vectors for SIMD operations in Rust
fn pyarray_to_cols(arr: PyReadonlyArray2<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = arr.shape()[0];
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    let mut z: Vec<f64> = Vec::with_capacity(n);
    for row in arr.as_array().rows() {
        x.push(row[0]);
        y.push(row[1]);
        z.push(row[2]);
    }
    (x, y, z)
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
    let (centx, centy, centz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_cols(src_pts);
    let _src_vol: &[f64] = src_vol.as_slice()?;
    let (jx, jy, jz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_cols(src_jdensity);
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_cols(tgt_pts);
    let (mut bx, mut by, mut bz): (Vec<f64>, Vec<f64>, Vec<f64>) = col_buffer(n);

    if nthreads_requested != 1 {
        biotsavart_parallel::bfield_direct_parallel(
            (&centx, &centy, &centz),
            &_src_vol,
            (&jx, &jy, &jz),
            (&x, &y, &z),
            (&mut bx, &mut by, &mut bz),
            nthreads_requested,
        );
    } else {
        biotsavart::bfield_direct(
            (&centx, &centy, &centz),
            &_src_vol,
            (&jx, &jy, &jz),
            (&x, &y, &z),
            (&mut bx, &mut by, &mut bz),
        );
    }

    Ok(cols_to_pyarray(py, (bx, by, bz)))
}

#[pyfunction]
fn b_current_point_octree<'py>(
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
    let (centx, centy, centz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_cols(src_pts);
    let _src_vol: &[f64] = src_vol.as_slice()?;
    let (jx, jy, jz): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_cols(src_jdensity);
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = pyarray_to_cols(tgt_pts);
    let (mut bx, mut by, mut bz): (Vec<f64>, Vec<f64>, Vec<f64>) = col_buffer(n);

    let mut sources: CurrentSources<point::PointSources> = CurrentSources(
        point::PointSources::new((&centx, &centy, &centz), &_src_vol, (&jx, &jy, &jz)),
    );
    let max_depth: u8 = 21;
    let tree: Octree<CurrentSources<point::PointSources>> =
        Octree::build_from_sources(sources, max_depth, leaf_threshold);

    #[cfg(feature = "parallel")]
    tree.h_field_parallel(
        (&x, &y, &z),
        (&mut bx, &mut by, &mut bz),
        theta,
        nthreads_requested,
    );

    Ok(cols_to_pyarray(py, (bx, by, bz)))
}

#[pyfunction]
fn _bfield_dualtree(
    centx: PyReadonlyArray1<f64>,
    centy: PyReadonlyArray1<f64>,
    centz: PyReadonlyArray1<f64>,
    vol: PyReadonlyArray1<f64>,
    jx: PyReadonlyArray1<f64>,
    jy: PyReadonlyArray1<f64>,
    jz: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    mut bx: PyReadwriteArray1<f64>,
    mut by: PyReadwriteArray1<f64>,
    mut bz: PyReadwriteArray1<f64>,
    theta_source: f64,
    theta_target: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
) -> PyResult<()> {
    use crate::archive::dualtree;
    dualtree::bfield_dualtree(
        centx.as_slice()?,
        centy.as_slice()?,
        centz.as_slice()?,
        vol.as_slice()?,
        jx.as_slice()?,
        jy.as_slice()?,
        jz.as_slice()?,
        x.as_slice()?,
        y.as_slice()?,
        z.as_slice()?,
        bx.as_slice_mut()?,
        by.as_slice_mut()?,
        bz.as_slice_mut()?,
        theta_source,
        theta_target,
        leaf_threshold,
    );
    Ok(())
}

#[pyfunction]
fn _bfield_hexahedron(
    nx: PyReadonlyArray1<f64>,
    ny: PyReadonlyArray1<f64>,
    nz: PyReadonlyArray1<f64>,
    jdensity: PyReadonlyArray1<f64>,
    target: PyReadonlyArray1<f64>,
) -> PyResult<([f64; 3])> {
    let b = crate::sources::hex8::bfield_hexahedron(
        nx.as_slice()?,
        ny.as_slice()?,
        nz.as_slice()?,
        jdensity.as_slice()?,
        target.as_slice()?,
    );

    Ok(b)
}

#[pyfunction]
fn _hfield_tetrahedrons(
    nodes_flat: PyReadonlyArray1<f64>,
    centroids_flat: PyReadonlyArray1<f64>,
    vol: PyReadonlyArray1<f64>,
    jdensity_flat: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    mut bx: PyReadwriteArray1<f64>,
    mut by: PyReadwriteArray1<f64>,
    mut bz: PyReadwriteArray1<f64>,
    theta: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
) -> PyResult<()> {
    use crate::octree::{CurrentSources, HFieldSolver, Octree, tet_element};
    let mut sources: CurrentSources<tet_element::TetSources> =
        CurrentSources(tet_element::TetSources::new(
            nodes_flat.as_slice()?,
            centroids_flat.as_slice()?,
            vol.as_slice()?,
            jdensity_flat.as_slice()?,
        ));
    let max_depth: u8 = 21;

    let tree: Octree<CurrentSources<tet_element::TetSources>> =
        Octree::build_from_sources(sources, max_depth, leaf_threshold);

    tree.h_field_parallel(
        (x.as_slice()?, y.as_slice()?, z.as_slice()?),
        (bx.as_slice_mut()?, by.as_slice_mut()?, bz.as_slice_mut()?),
        theta,
        nthreads_requested,
    );
    Ok(())
}

#[pyfunction]
fn _hfield_dipole_tetrahedrons(
    nodes_flat: PyReadonlyArray1<f64>,
    centroids_flat: PyReadonlyArray1<f64>,
    vol: PyReadonlyArray1<f64>,
    jdensity_flat: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    mut bx: PyReadwriteArray1<f64>,
    mut by: PyReadwriteArray1<f64>,
    mut bz: PyReadwriteArray1<f64>,
    theta: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
) -> PyResult<()> {
    use crate::octree::{DipoleSources, HFieldSolver, Octree, tet_element};
    let mut sources: DipoleSources<tet_element::TetSources> =
        DipoleSources(tet_element::TetSources::new(
            nodes_flat.as_slice()?,
            centroids_flat.as_slice()?,
            vol.as_slice()?,
            jdensity_flat.as_slice()?,
        ));
    let max_depth: u8 = 21;

    let tree: Octree<DipoleSources<tet_element::TetSources>> =
        Octree::build_from_sources(sources, max_depth, leaf_threshold);

    tree.h_field_parallel(
        (x.as_slice()?, y.as_slice()?, z.as_slice()?),
        (bx.as_slice_mut()?, by.as_slice_mut()?, bz.as_slice_mut()?),
        theta,
        nthreads_requested,
    );
    Ok(())
}

#[pyfunction]
pub fn _hfield_tetrahedrons_direct(
    nodes: PyReadonlyArray1<f64>,
    connectivity: PyReadonlyArray1<u32>,
    jdensity: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    mut hx: PyReadwriteArray1<f64>,
    mut hy: PyReadwriteArray1<f64>,
    mut hz: PyReadwriteArray1<f64>,
    nthreads_requested: u32,
) -> PyResult<()> {
    use crate::biotsavart_parallel;
    biotsavart_parallel::hfield_direct_tet_parallel(
        nodes.as_slice()?,
        connectivity.as_slice()?,
        jdensity.as_slice()?,
        x.as_slice()?,
        y.as_slice()?,
        z.as_slice()?,
        hx.as_slice_mut()?,
        hy.as_slice_mut()?,
        hz.as_slice_mut()?,
        nthreads_requested,
    );

    Ok(())
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
fn _h_demag_tet4(
    src_nodes_in: PyReadonlyArray1<f64>,
    src_connectivity_in: PyReadonlyArray1<u32>,
    tgt_nodes_in: PyReadonlyArray1<f64>,
    tgt_connectivity_in: PyReadonlyArray1<u32>,
    mx: PyReadonlyArray1<f64>,
    my: PyReadonlyArray1<f64>,
    mz: PyReadonlyArray1<f64>,
    mut hx: PyReadwriteArray1<f64>,
    mut hy: PyReadwriteArray1<f64>,
    mut hz: PyReadwriteArray1<f64>,
    nthreads_requested: u32,
) -> PyResult<()> {
    let src_nodes_raw = src_nodes_in.as_slice()?;
    let src_conn_raw = src_connectivity_in.as_slice()?;
    let tgt_nodes_raw = tgt_nodes_in.as_slice()?;
    let tgt_conn_raw = tgt_connectivity_in.as_slice()?;
    let mx_slice = mx.as_slice()?;
    let my_slice = my.as_slice()?;
    let mz_slice = mz.as_slice()?;
    let hx_slice = hx.as_slice_mut()?;
    let hy_slice = hy.as_slice_mut()?;
    let hz_slice = hz.as_slice_mut()?;

    // Reshape flat nodes into Vec<Vec3>: [x0,y0,z0,x1,y1,z1,...] -> [Vec3, Vec3, ...]
    let n_src_nodes = src_nodes_raw.len() / 3;
    let n_tgt_nodes = tgt_nodes_raw.len() / 3;
    let mut src_nodes: Vec<Vec3> = Vec::with_capacity(n_src_nodes);
    let mut tgt_nodes: Vec<Vec3> = Vec::with_capacity(n_tgt_nodes);
    for i in 0..n_src_nodes {
        src_nodes.push(Vec3([
            src_nodes_raw[3 * i],
            src_nodes_raw[3 * i + 1],
            src_nodes_raw[3 * i + 2],
        ]));
    }

    for i in 0..n_tgt_nodes {
        tgt_nodes.push(Vec3([
            tgt_nodes_raw[3 * i],
            tgt_nodes_raw[3 * i + 1],
            tgt_nodes_raw[3 * i + 2],
        ]));
    }

    // Reshape flat connectivity into &[[u32; 4]]: [n0,n1,n2,n3,...] -> [[u32;4], ...]
    let n_src_elements = src_conn_raw.len() / 4;
    let mut src_elements: Vec<[u32; 4]> = vec![[0; 4]; n_src_elements];
    let n_tgt_elements = tgt_conn_raw.len() / 4;
    let mut tgt_elements: Vec<[u32; 4]> = vec![[0; 4]; n_tgt_elements];
    for i in 0..n_src_elements {
        for j in 0..4 {
            src_elements[i][j] = src_conn_raw[i * 4 + j];
        }
    }
    for i in 0..n_tgt_elements {
        for j in 0..4 {
            tgt_elements[i][j] = tgt_conn_raw[i * 4 + j];
        }
    }

    // Build M vectors per element
    let mut mvectors: Vec<Vec3> = Vec::with_capacity(n_src_elements);
    for i in 0..n_src_elements {
        mvectors.push(Vec3([mx_slice[i], my_slice[i], mz_slice[i]]));
    }

    // Build SoA node arrays for target nodes
    let mut src_nx: Vec<f64> = Vec::with_capacity(n_src_nodes);
    let mut src_ny: Vec<f64> = Vec::with_capacity(n_src_nodes);
    let mut src_nz: Vec<f64> = Vec::with_capacity(n_src_nodes);
    for i in 0..n_src_nodes {
        src_nx.push(src_nodes_raw[3 * i]);
        src_ny.push(src_nodes_raw[3 * i + 1]);
        src_nz.push(src_nodes_raw[3 * i + 2]);
    }
    let mut tgt_nx: Vec<f64> = Vec::with_capacity(n_tgt_nodes);
    let mut tgt_ny: Vec<f64> = Vec::with_capacity(n_tgt_nodes);
    let mut tgt_nz: Vec<f64> = Vec::with_capacity(n_tgt_nodes);
    for i in 0..n_tgt_nodes {
        tgt_nx.push(tgt_nodes_raw[3 * i]);
        tgt_ny.push(tgt_nodes_raw[3 * i + 1]);
        tgt_nz.push(tgt_nodes_raw[3 * i + 2]);
    }

    // Source and target are the same mesh
    biotsavart_parallel::hmag_direct_tet_parallel(
        (&src_nx, &src_ny, &src_nz),
        &src_elements,
        &mvectors,
        (&tgt_nx, &tgt_ny, &tgt_nz),
        &tgt_elements,
        hx_slice,
        hy_slice,
        hz_slice,
        nthreads_requested,
    );

    Ok(())
}

// ---
// Mesh Operations
// ---

#[pyfunction]
fn _mesh_centroids(
    nodes_flat: PyReadonlyArray1<f64>,
    connectivity_flat: PyReadonlyArray1<u32>,
    mut x: PyReadwriteArray1<f64>,
    mut y: PyReadwriteArray1<f64>,
    mut z: PyReadwriteArray1<f64>,
) -> PyResult<()> {
    mesh::centroids(
        nodes_flat.as_slice()?,
        connectivity_flat.as_slice()?,
        x.as_slice_mut()?,
        y.as_slice_mut()?,
        z.as_slice_mut()?,
    );
    Ok(())
}

#[pyfunction]
fn _mesh_volumes(
    nodes: PyReadonlyArray1<f64>,
    connectivity: PyReadonlyArray1<u32>,
    mut vol: PyReadwriteArray1<f64>,
) -> PyResult<()> {
    mesh::volumes(
        nodes.as_slice()?,
        connectivity.as_slice()?,
        vol.as_slice_mut()?,
    );
    Ok(())
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
    m.add_function(wrap_pyfunction!(b_current_point_octree, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_bfield_dualtree, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_bfield_hexahedron, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_hfield_tetrahedrons, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_hfield_dipole, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_hfield_tetrahedrons_direct, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_hfield_dipole_tetrahedrons, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_h_demag_tet4, m.clone())?)?;

    // Mesh functions
    m.add_function(wrap_pyfunction!(_mesh_centroids, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_volumes, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_faces, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_face_properties, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_forces, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_mesh_surface_tets, m.clone())?)?;

    Ok(())
}
