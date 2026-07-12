use std::f64::consts::{FRAC_PI_2, PI};

/// Computes atan(y/x) on the range [-pi,pi] using a fast, approximate approach
///
/// This function is designed to be auto-vectorized when inlined inside other functions
#[inline(always)]
pub fn atan2(y: f64, x: f64) -> f64 {
    let ya: f64 = y.abs();
    let xa: f64 = x.abs();

    // Always divide smaller by larger to keep ratio in [0, 1]
    // If swap, then the angle is computed against the y-axis instead of the x-axis
    let swap: bool = ya > xa;
    let num: f64 = if swap { xa } else { ya };
    let den: f64 = if swap { ya } else { xa };

    let ratio: f64 = num / (den + 1e-16); // shield div/0
    let mut atan: f64 = atan_approx(ratio);

    // Exchange with the complementary angle if needed
    atan = if swap { FRAC_PI_2 - atan } else { atan };

    // Correct by quadrant if needed:
    // First quadrant (x >= 0, y >= 0): atan = +(atan)
    // Second quadrant (x < 0, y >= 0): PI - atan = +(PI - atan)
    // Third quadrant (x < 0, y < 0): -PI + atan = -(PI - atan)
    // Fourth quadrant (x > 0, y < 0): -atan = -(atan)
    // Once x has been tested, then the sign change is determined from the sign of y

    let ret_val = if x < 0.0 { PI - atan } else { atan };
    ret_val.copysign(y)
}

// Use magic numbers described in the following reference to approximate atan(v)
// via an 11th-order polynomial with 6 (odd power) terms:
// https://blasingame.engr.tamu.edu/z_zCourse_Archive/P620_18C/P620_zReference/PDF_Txt_Hst_Apr_Cmp_(1955).pdf
// This is accurate in the range [-1, 1]
#[inline(always)]
pub fn atan_approx(v: f64) -> f64 {
    let a1: f64 = 0.99997726;
    let a3: f64 = -0.33262347;
    let a5: f64 = 0.19354346;
    let a7: f64 = -0.11643287;
    let a9: f64 = 0.05265332;
    let a11: f64 = -0.01172120;

    let v2: f64 = v * v;

    // v * (a1 + v2 * (a3 + v2 * (a5 + v2 * (a7 + v2 * (a9 + v2 * a11)))))
    v * v2.mul_add(
        v2.mul_add(v2.mul_add(v2.mul_add(v2.mul_add(a11, a9), a7), a5), a3),
        a1,
    )
}

/// Compute atan(x), which is needed for the finite element edge integrals
#[inline(always)]
pub fn atan(v: f64) -> f64 {
    let av = v.abs();
    let swap = av > 1.0;
    let t = if swap { 1.0 / av } else { av };
    let p = atan_approx(t);
    let p = if swap { FRAC_PI_2 - p } else { p };
    if v < 0.0 { -p } else { p }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::time::Instant;

    #[test]
    fn test_atan2_fast() {
        let step: f64 = 1e-3;
        let n: usize = 100_000_000;
        let x: f64 = 1.0;
        let vmin = -500.0;
        // let vmax = 500.0;
        let mut y: Vec<f64> = vec![0.0; n];
        y[0] = vmin;
        for i in 1..n {
            y[i] = y[i - 1] + step;
        }
        let mut ay = vec![0.0; n];
        let mut ayp = vec![0.0; n];

        let start = Instant::now();
        for i in 0..n {
            ay[i] = y[i].atan2(x);
        }
        let elapsed_slow = start.elapsed().as_secs_f64();

        let start2 = Instant::now();
        for i in 0..n {
            ayp[i] = atan2(y[i], x);
        }
        let elapsed_fast = start2.elapsed().as_secs_f64();

        let mut max_err = 0.0f64;
        let mut worst_x = 0.0f64;
        let mut worst_y = 0.0f64;
        let mut worst_i = 0;
        for i in 0..n {
            let err = (ay[i] - ayp[i]).abs();
            if err > max_err {
                max_err = err;
                worst_x = x;
                worst_y = y[i];
                worst_i = i;
            }
        }
        println!(
            "max error: {:.2e} at x={}, y={}, ay={}, ayp={}",
            max_err, worst_x, worst_y, ay[worst_i], ayp[worst_i]
        );
        println!("slow time: {} sec", elapsed_slow);
        println!("fast time: {} sec", elapsed_fast);
        println!("speedup: {}", elapsed_slow / elapsed_fast);
        println!("test complete.");
        assert!(max_err < 2e-6);
    }
}
