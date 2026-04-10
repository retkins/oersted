# Changelog

## 0.1.0 
* Added changelog 
* Ran `rustfmt` and `clippy` and fixed all issues 
* Using `std::f64::consts::{LN_2, PI}` instead of approximate values (2 places fixed)
* Python project version is now dynamic to match rust project
* Added placeholder for `tests/fig/` to avoid errors in running tests
* Added `_thor.pyi` for typing functions in Rust dll
* Added `.git-blame-ignore-revs` file
* Added github actions workflows for testing python and rust, updated deploy_docs to use uv
* Converted project to src-layout per <https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>
* Fix `ruff` and `ty` suggestions; set `ruff` line-length to 150 for now
* Update test file path read/write to use pathlib
* Remove interactive test plots, add py.typed marker, use x86-64-v3 reference cpu
* Open python version, rename `test_utils` so it doesn't get picked up by pytest
* Set magnetized sphere test to expected fail, fix clippy/ty lints
* Fix ruff format checks
* Changed png to svg output for plots 
* Moved examples to `/examples` and added a `test_examples.py`
* Reduced some of the test mesh parameters to run tests faster 
* Derive PartialEq for Vec3 and Mat3, bump numpy and pyo3 crate versions, set abi3-py310 for bindings
* Added magnetization calc for tetrahedral elements and linear magnetic materials + test
* Updated magnetization test with more robust analytical solution
* Updated step mesh function to output node coordinates and element connectivity matrix
* Renamed project to `oersted`
* Add github actions workflows for python and rust releases
* Updated docs to use .svg, updated figures
* Updated benchmarks to use different parameters if run by pytest vs by the user
* Update docs homepage and readme

## v0.2.0
* Added ellipsoid test (flat disc)
* Updated mesh handling interface to have its own class
* Updated octree methods to operate on magnetized tet4's
* Added Mesh object, methods for calculating volume and centroid of tet4 elements
* Update magnetization functions and related tests to use Mesh object
* Reduce default error tolerance of demag calc to 1.0 without loss of Bfield precision to 6 decimal places
* Add function for computing which are surface faces on a tetrahedral mesh
* Add Solver and CentroidMesh classes
* Move Solver into its own file
* Add function for calculating surface face areas and normals
* Add Maxwell stress tensor and surface force calculation functions
* Add function for creating tetradhedrons at surface faces for magnetization calc
* Added surface force test
* rustdoc now refers to the katex header file
* Updated interface to direct tet4 integration method to use nodes+connectivity
* Small updates to function docstrings (Rust), fixed cargo doc warnings
* Fixed bug in `surface_face_properties` causing incorrect face centroid/normal calculations (bad return order)
* Added test for lorentz force calculations using maxwell stress tensor (passes)
* Added test for mesh-related functionality
* Remove maxwell force calculation on magnetized meshes (not currently working)

* Added helper functions in `python.rs` for transposing memory at the boundary
* Updated the interfaces for calling the flux density (B field) calculation for point sources using direct summation
* Fixed bug in parallel direct point solver and multiple bugs in solenoid test
* Updated interface for B field calculation on point sources with octree
* Removed bindings for dual-tree point and hexahedron source evaluations
* Updates to cargo.toml: bump version to 0.2.0, make python feature depend on parallel feature
* Updated bindings for direct tet4 solver
* Updated internal interfaces for Rust functions: now using Vec3 in centroids/volumes/octree functions
* Added `h_field()` function
* Mesh is now stateless: does not store jdensity or field results
* Updated and simplified helmholtz coil and solenoid tests; added analytical comparisons
* Cleanup duplicated functions
* Move `Vec3` and Mat3 into `types`
* Updated magnetized tetrahedral calcs to use analytical gradient and significantly cleaned up related functions
* Added `SurfaceMesh` class
* Moved magnetization calculation completely to Rust
* Updated magnetization calculation to use octree methods

* Added a "getting started" tutorial and set docs theme to `material`
* Added 3d plotting capabilities
* Updated `Solver` classes to handle iterative solver parameters
* Updated fixed point solver to use under-relaxation for better convergence
* Added cube and updated sphere geometries for testing
* Added benchmarks
* Added back point dipole sources (direct and octree)
* Added Kelvin force calculation (more stable than Maxwell forces currently)