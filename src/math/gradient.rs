use crate::types::{Mat3, Vec3};

/// Compute the `J^-T` matrix for every element in a tetrahedral finite element mesh
pub fn jmatrices(nodes: &[Vec3], elements: &[[u32; 4]]) -> Vec<Mat3> {
    let n_elements = elements.len();

    // Allocate memory for the result
    let mut jmatrices: Vec<Mat3> = vec![Mat3::default(); n_elements];

    // Construct each jmatrix and gvector
    for i in 0..n_elements {
        // Row-major matrix; subtract each column
        // TODO: make elegant function as part of the Mat3 trait
        let n0: Vec3 = nodes[elements[i][0] as usize];
        let n1: Vec3 = nodes[elements[i][1] as usize] - n0;
        let n2: Vec3 = nodes[elements[i][2] as usize] - n0;
        let n3: Vec3 = nodes[elements[i][3] as usize] - n0;
        jmatrices[i] = Mat3::from_cols(&n1, &n2, &n3).inverse_transpose();
    }

    jmatrices
}

/// Compute the `g` vector for each element in a tetrahedral finite element mesh
///
/// `fvalues` is defined at each node
/// `elements` contains the index value of each node in the element
pub fn gvectors(elements: &[[u32; 4]], fvalues: &[f64]) -> Vec<Vec3> {
    let n_elements: usize = elements.len();
    let mut gvectors: Vec<Vec3> = vec![Vec3::default(); n_elements];

    for i in 0..n_elements {
        gvectors[i] = Vec3([
            fvalues[elements[i][1] as usize] - fvalues[elements[i][0] as usize],
            fvalues[elements[i][2] as usize] - fvalues[elements[i][0] as usize],
            fvalues[elements[i][3] as usize] - fvalues[elements[i][0] as usize],
        ])
    }

    gvectors
}

/// Compute the gradient of a scalar-valued function using the `J^-T` matrix and `g` vector
#[inline]
pub fn gradient(j_invt: &Mat3, g: &Vec3) -> Vec3 {
    j_invt.mul_vec(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f_ex(xyz: &Vec3) -> f64 {
        xyz[0] + 2.0 * xyz[1] + 3.0 * xyz[2]
    }

    #[test]
    fn test_gradient() {
        let elements: [[u32; 4]; 1] = [[0, 1, 2, 3]];
        let nodes = [
            Vec3([-1.0, 0.0, -1.0 / (2.0 as f64).sqrt()]),
            Vec3([1.0, 0.0, -1.0 / (2.0 as f64).sqrt()]),
            Vec3([0.0, -1.0, 1.0 / (2.0 as f64).sqrt()]),
            Vec3([0.0, 1.0, 1.0 / (2.0 as f64).sqrt()]),
        ];
        let fvalues = [
            f_ex(&nodes[0]),
            f_ex(&nodes[1]),
            f_ex(&nodes[2]),
            f_ex(&nodes[3]),
        ];

        let j_invt = jmatrices(&nodes, &elements);
        let g = gvectors(&elements, &fvalues);

        let grad_expected = Vec3([1.0, 2.0, 3.0]);

        let grad = gradient(&j_invt[0], &g[0]);
        let err = grad - grad_expected;
        assert!(err[0].abs() < 1e-8);
        assert!(err[1].abs() < 1e-8);
        assert!(err[2].abs() < 1e-8);
        println!("Gradient test passed!");
    }
}
