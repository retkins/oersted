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