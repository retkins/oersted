//! A 3x3 row-major matrix for common linear algebra operations

use crate::vec3::Vec3;

/// A row-major 3x3 matrix
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct Mat3(pub [Vec3; 3]);

impl Mat3 {
    /// Construct a matrix using vectors that represent the rows
    pub fn from_rows(row0: &Vec3, row1: &Vec3, row2: &Vec3) -> Self {
        Mat3([*row0, *row1, *row2])
    }

    /// Construct a matrix using vectors that represent the columns
    ///
    /// The matrix is stored as row-major
    pub fn from_cols(col0: &Vec3, col1: &Vec3, col2: &Vec3) -> Self {
        let row0 = Vec3([col0[0], col1[0], col2[0]]);
        let row1 = Vec3([col0[1], col1[1], col2[1]]);
        let row2 = Vec3([col0[2], col1[2], col2[2]]);
        Self::from_rows(&row0, &row1, &row2)
    }

    /// Multiply the matrix by a vector, returning a new vector
    pub fn mul_vec(&self, v: &Vec3) -> Vec3 {
        Vec3([self.0[0].dot(v), self.0[1].dot(v), self.0[2].dot(v)])
    }

    /// Return a column of the matrix
    pub fn col(&self, i: usize) -> Vec3 {
        Vec3([self[0][i], self[1][i], self[2][i]])
    }

    /// Compute the determinant of the matrix:
    ///
    /// ```
    /// | a b c |
    /// | d e f | = aei + bfg + cdh - ceg - bdi - afh
    /// | g h i |
    /// ```
    ///
    /// Reference:
    /// [https://en.wikipedia.org/wiki/Determinant](https://en.wikipedia.org/wiki/Determinant)
    pub fn det(&self) -> f64 {
        let aei: f64 = self[0][0] * self[1][1] * self[2][2];
        let bfg = self[0][1] * self[1][2] * self[2][0];
        let cdh = self[0][2] * self[1][0] * self[2][1];
        let ceg = self[0][2] * self[1][1] * self[2][0];
        let bdi = self[0][1] * self[1][0] * self[2][2];
        let afh = self[0][0] * self[1][2] * self[2][1];
        aei + bfg + cdh - ceg - bdi - afh
    }

    /// Compute the inverse of the matrix, if it exists
    ///
    /// Use the method described here:
    /// [https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_x_3_matrices](https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_x_3_matrices)
    ///
    pub fn inverse(&self) -> Option<Mat3> {
        let det = self.det();
        if det.abs() > 1e-8 {
            let c0 = self.col(0);
            let c1 = self.col(1);
            let c2 = self.col(2);

            let mut m: Mat3 = Self::from_rows(&c1.cross(&c2), &c2.cross(&c0), &c0.cross(&c1));
            m *= 1.0 / det;
            Some(m)
        } else {
            None
        }
    }

    /// Compute the transpose of the matrix
    pub fn transpose(&self) -> Mat3 {
        Self::from_cols(&self[0], &self[1], &self[2])
    }

    /// Compute the transpose of the inverse of the matrix
    ///
    /// This will be needed for gradient calculations on the finite element mesh
    pub fn inverse_transpose(&self) -> Mat3 {
        let det = self.det();
        debug_assert!(det.abs() > 1e-8);
        let c0 = self.col(0);
        let c1 = self.col(1);
        let c2 = self.col(2);

        let mut m: Mat3 = Self::from_cols(&c1.cross(&c2), &c2.cross(&c0), &c0.cross(&c1));
        m *= 1.0 / det;
        m
    }
}

impl std::ops::Index<usize> for Mat3 {
    type Output = Vec3;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Mat3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::ops::MulAssign<f64> for Mat3 {
    fn mul_assign(&mut self, rhs: f64) {
        self[0] *= rhs;
        self[1] *= rhs;
        self[2] *= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det() {
        let a = Mat3::from_rows(
            &Vec3([2.0, -3.0, 1.0]),
            &Vec3([2.0, 0.0, -1.0]),
            &Vec3([1.0, 4.0, 5.0]),
        );

        assert_eq!(a.det(), 49.0);
    }

    #[test]
    fn test_inverse() {
        let a = Mat3::from_rows(
            &Vec3([3.0, 0.0, 2.0]),
            &Vec3([2.0, 1.0, 0.0]),
            &Vec3([1.0, 4.0, 2.0]),
        );

        let expected = Mat3::from_rows(
            &Vec3([0.1, 0.4, -0.1]),
            &Vec3([-0.2, 0.2, 0.2]),
            &Vec3([0.35, -0.6, 0.15]),
        );

        assert_eq!(a.det(), 20.0);

        match a.inverse() {
            Some(m) => {
                assert_eq!(m.det(), expected.det());
            }
            None => assert!(false),
        }
    }
}
