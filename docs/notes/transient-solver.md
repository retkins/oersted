# Design doc for the transient solver

This is a living document for designing the transient solver, and will be rolled into 
the theory/reference manual when this feature is complete.

## Background

The transient solver is meant to be used for low-frequency calculations of eddy currents
in conducting materials: the response of the conducting material due to changing external field sources.

### Assumptions
* Magnetoquasistatics: radiation effects are negligible; the size of the part is much
smaller than the time scale of the problem
* The changing field sources are not affected by the response of the conducting materials
    * External a-field at all time steps for all elements is an input to the problem
* Constant time steps
    * Dramatically simplifies the solution process

### Goals
* Both nonmagnetic and magnetic materials should be available
    * Isotropic
    * Magnetic: simple linear or BH-curve
    * BH curve should be smooth to facilitate convergence
    * Defer magnetic materials to a later release 
* Supporting moving materials 
    * Rigid body, translation only assumptions 
    * Example: magnet falling through a copper pipe
    * Defer to later release
* Solve time (performance budget) on 9950x:
    * Threshold: 100K element mesh, 100 timesteps, 10sec/timestep (~20min total)
    * Target: 1M element mesh, 100 timesteps, 1min/timestep (< ~2 hrs)
* Avoid the cohomology problem (identifying holes in the conductor) by choosing an appropriate DOF
    * Most similar VIM codes solve the cohomology problem. Why? That seems painful. [TODO: research this]

### Limitations

* This will be an approximate solver and every problem might require mesh refinement
    * DOF chosen are likely going to be weakly preventing current from leaving the surface of the part, for example
    * To make solve times tractable, using the Barnes Hut or a future Hmatrix solver will be necessary; it simply won't be feasible to do analytic volume integration over large-scale problems, multiple times at every timestep

## Mathematical Derivation

Basic idea:
* Changing magnetic fields drive currents to form in conducting materials
* Those currents have their own magnetic fields, which influence the formation of currents elsewhere in the material
* All elements interact mutually with all other elements 

### Physical Equations

### Discretization

### Numerical Methods

### Data Structures

The solver will form a mutual inductance matrix (called `M` or `L`), which is dense. This can be accelerated via a Barnes Hut solve at every iteration, which will make the matrix slightly unsymmetric and impair numerical solutions (via GMRES or similar). Another method would be to form an H-matrix ("Hierarchical") via ACA (adaptive cross approximation), which is more expensive to build but faster to evaluate than BH, and can be made to guarantee symmetry, to solve with MINRES.

## References
[Intuitive Guide to Maxwell's Equations](https://github.com/photonlines/Intuitive-Guide-to-Maxwells-Equations/blob/master/PDF/An%20Intuitive%20Guide%20to%20Maxwell's%20Equations.pdf)