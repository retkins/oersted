#![allow(unused)]

use std::cmp::max;

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;

use crate::biotsavart;
use crate::biotsavart::hmag_direct_tet;
use crate::biotsavart_parallel::hmag_direct_tet_parallel;
use crate::mesh;
use crate::sources::bfield_hexahedron;
use crate::vec3::Vec3;

#[pyfunction]
fn _bfield_direct(
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
    nthreads_requested: u32,
) -> PyResult<()> {
    #[cfg(feature = "parallel")]
    if nthreads_requested != 1 {
        use crate::biotsavart_parallel;
        biotsavart_parallel::bfield_direct_parallel(
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
            nthreads_requested,
        );
        return Ok(());
    }

    biotsavart::bfield_direct(
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
    );
    Ok(())
}

#[pyfunction]
fn _bfield_octree(
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
    theta: f64,
    leaf_threshold: u32,
    nthreads_requested: u32,
) -> PyResult<()> {
    // #[cfg(feature = "parallel")]
    // if nthreads_requested != 1 {
    //     use crate::biotsavart_parallel;
    //     biotsavart_parallel::bfield_octree_parallel(
    //         centx.as_slice()?,
    //         centy.as_slice()?,
    //         centz.as_slice()?,
    //         vol.as_slice()?,
    //         jx.as_slice()?,
    //         jy.as_slice()?,
    //         jz.as_slice()?,
    //         x.as_slice()?,
    //         y.as_slice()?,
    //         z.as_slice()?,
    //         bx.as_slice_mut()?,
    //         by.as_slice_mut()?,
    //         bz.as_slice_mut()?,
    //         theta, leaf_threshold,
    //         nthreads_requested
    //     );
    //     return Ok(());
    // }

    use crate::octree::{CurrentSources, HFieldSolver, Octree, point};
    let mut sources: CurrentSources<point::PointSources> =
        CurrentSources(point::PointSources::new(
            centx.as_slice()?,
            centy.as_slice()?,
            centz.as_slice()?,
            vol.as_slice()?,
            jx.as_slice()?,
            jy.as_slice()?,
            jz.as_slice()?,
        ));
    let max_depth: u8 = 21;
    let tree: Octree<CurrentSources<point::PointSources>> =
        Octree::build_from_sources(sources, max_depth, leaf_threshold);

    #[cfg(feature = "parallel")]
    tree.h_field_parallel(
        (x.as_slice()?, y.as_slice()?, z.as_slice()?),
        (bx.as_slice_mut()?, by.as_slice_mut()?, bz.as_slice_mut()?),
        theta,
        nthreads_requested,
    );

    // original version:
    //
    // biotsavart::bfield_octree(
    //     centx.as_slice()?,
    //     centy.as_slice()?,
    //     centz.as_slice()?,
    //     vol.as_slice()?,
    //     jx.as_slice()?,
    //     jy.as_slice()?,
    //     jz.as_slice()?,
    //     x.as_slice()?,
    //     y.as_slice()?,
    //     z.as_slice()?,
    //     bx.as_slice_mut()?,
    //     by.as_slice_mut()?,
    //     bz.as_slice_mut()?,
    //     theta, leaf_threshold
    // );
    Ok(())
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
) -> PyResult<(f64, f64, f64)> {
    let b = bfield_hexahedron(
        nx.as_slice()?,
        ny.as_slice()?,
        nz.as_slice()?,
        jdensity.as_slice()?,
        target.as_slice()?,
    );

    Ok((b[0], b[1], b[2]))
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
    nodes_flat: PyReadonlyArray1<f64>,
    centroids_flat: PyReadonlyArray1<f64>,
    vol: PyReadonlyArray1<f64>,
    jdensity_flat: PyReadonlyArray1<f64>,
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
        nodes_flat.as_slice()?,
        vol.as_slice()?,
        jdensity_flat.as_slice()?,
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
    nodes_flat: PyReadonlyArray1<f64>,
    element_connectivity_flat: PyReadonlyArray1<u32>,
    mx: PyReadonlyArray1<f64>,
    my: PyReadonlyArray1<f64>,
    mz: PyReadonlyArray1<f64>,
    mut hx: PyReadwriteArray1<f64>,
    mut hy: PyReadwriteArray1<f64>,
    mut hz: PyReadwriteArray1<f64>,
    nthreads_requested: u32,
) -> PyResult<()> {
    let nodes_raw = nodes_flat.as_slice()?;
    let conn_raw = element_connectivity_flat.as_slice()?;
    let mx_slice = mx.as_slice()?;
    let my_slice = my.as_slice()?;
    let mz_slice = mz.as_slice()?;
    let hx_slice = hx.as_slice_mut()?;
    let hy_slice = hy.as_slice_mut()?;
    let hz_slice = hz.as_slice_mut()?;

    // Reshape flat nodes into Vec<Vec3>: [x0,y0,z0,x1,y1,z1,...] -> [Vec3, Vec3, ...]
    let n_nodes = nodes_raw.len() / 3;
    let mut nodes: Vec<Vec3> = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        nodes.push(Vec3([
            nodes_raw[3 * i],
            nodes_raw[3 * i + 1],
            nodes_raw[3 * i + 2],
        ]));
    }

    // Reshape flat connectivity into &[[u32; 4]]: [n0,n1,n2,n3,...] -> [[u32;4], ...]
    let n_elements = conn_raw.len() / 4;
    let mut elements: Vec<[u32; 4]> = vec![[0; 4]; n_elements];
    for i in 0..n_elements {
        for j in 0..4 {
            elements[i][j] = conn_raw[i * 4 + j];
        }
    }

    // Build M vectors per element
    let mut mvectors: Vec<Vec3> = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        mvectors.push(Vec3([mx_slice[i], my_slice[i], mz_slice[i]]));
    }

    // Build SoA node arrays for target nodes
    let mut nx: Vec<f64> = Vec::with_capacity(n_nodes);
    let mut ny: Vec<f64> = Vec::with_capacity(n_nodes);
    let mut nz: Vec<f64> = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        nx.push(nodes_raw[3 * i]);
        ny.push(nodes_raw[3 * i + 1]);
        nz.push(nodes_raw[3 * i + 2]);
    }

    // Source and target are the same mesh
    hmag_direct_tet_parallel(
        (&nx, &ny, &nz),
        &elements,
        &mvectors,
        (&nx, &ny, &nz),
        &elements,
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

#[pymodule]
fn _oersted<'py>(_py: Python, m: Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_bfield_direct, m.clone())?)?;
    m.add_function(wrap_pyfunction!(_bfield_octree, m.clone())?)?;
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

    Ok(())
}
