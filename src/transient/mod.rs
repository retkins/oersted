//! Transient (time-domain) eddy-current solver for oersted 
//! 
//! This solver assumes the following: 
//! 1. The problem is discretized into a mesh consisting of 4-node tetrahedral elements.
//! 2. The DOF are phi (defined piecewise linear (P1 basis) on the nodes, and 
//! current density (J, A/m^2) defined piecewise constant (P0 basis) on the elements.
//! 3. phi is a lagrange multiplier that enforced div J = 0 (eliminates the cohomology problem).


mod dense;

pub use dense::solve;