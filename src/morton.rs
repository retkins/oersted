//! Morton codes for the octree  
//!
//! Note: Morton codes are calculated by considering a cube in 3D space.

#![allow(non_snake_case)]

/// Determine the scale factor based on the maximum quantized depth of the tree
///
/// For typical double-precision problems, L=21
pub fn calculate_scale_factor(L: u32) -> f64 {
    // TODO: should this be `1u << L` or `(1u << L) - 1`
    let factor = 1u32 << L;
    factor as f64
}

/// Normalize a points location in a grid to the range [0, 1] (inclusive)
///
/// # Args:  
/// * `point`: (x,y,z) coordinates of the point to be normalized  
/// * `side_length`: length of the grid size  
/// * `min_corner`: (xmin, ymin, zmin) of the grid  
pub fn normalize(
    point: (f64, f64, f64),
    side_length: f64,
    min_corner: (f64, f64, f64),
) -> (f64, f64, f64) {
    let (xmin, ymin, zmin) = min_corner;
    let inv_side_length: f64 = 1.0 / side_length;
    (
        (point.0 - xmin) * inv_side_length,
        (point.1 - ymin) * inv_side_length,
        (point.2 - zmin) * inv_side_length,
    )
}

/// Quantize a normalized point's location into an integer
pub fn quantize(normalized_pt: (f64, f64, f64), scale: f64) -> (u32, u32, u32) {
    let max_cell: u32 = scale as u32 - 1;
    (
        ((scale * normalized_pt.0).floor() as u32).min(max_cell),
        ((scale * normalized_pt.1).floor() as u32).min(max_cell),
        ((scale * normalized_pt.2).floor() as u32).min(max_cell),
    )
}

/// Interleave the bits of three integers into a Morton code
///
/// # Reference:
/// <https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/>
pub fn interleave(quantized_pt: (u32, u32, u32)) -> u64 {
    let mut code: u64 = 0;
    let (x, y, z) = quantized_pt;

    for i in 0..21_u64 {
        code |= ((x as u64 & (1u64 << i)) << (2u64 * i))
            | ((y as u64 & (1u64 << i)) << (2u64 * i + 1u64))
            | ((z as u64 & (1u64 << i)) << (2u64 * i + 2u64));
    }
    code
}

/// Encode a position into a Morton u64
pub fn encode(
    point: (f64, f64, f64),
    scale: f64,
    side_length: f64,
    min_corner: (f64, f64, f64),
) -> u64 {
    let normalized_pt: (f64, f64, f64) = normalize(point, side_length, min_corner);
    let quantized_pt = quantize(normalized_pt, scale);
    let code: u64 = interleave(quantized_pt);
    code
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_interleave() {
        // from the above reference
        let pt: (u32, u32, u32) = (5, 9, 1);
        let code_expected: u64 = 1095;
        let code: u64 = interleave(pt);

        assert_eq!(code, code_expected);
    }

    #[test]
    fn test_encode() {
        let L: u32 = 21;
        let scale = calculate_scale_factor(L);
        let min_corner: (f64, f64, f64) = (-0.5, -0.5, -0.5);
        let side_length: f64 = 1.0;
        let point: (f64, f64, f64) = (0.75, 0.75, 0.75);
        let code: u64 = encode(point, scale, side_length, min_corner);
        println!("Code = {}", code);
        assert_eq!(9223372036854775807, code);
    }
}
