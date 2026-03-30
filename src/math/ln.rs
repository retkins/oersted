const ONE_THIRD: f64 = 1.0 / 3.0;
const ONE_FIFTH: f64 = 1.0 / 5.0;
const ONE_SEVENTH: f64 = 1.0 / 7.0;
const ONE_NINTH: f64 = 1.0 / 9.0;
const ONE_ELEVENTH: f64 = 1.0 / 11.0;
const ONE_THIRTEENTH: f64 = 1.0 / 13.0;
const ONE_FIFTEENTH: f64 = 1.0 / 15.0;
use std::f64::consts::LN_2;

/// Fast, approximate natural logarithm of `x` with absolute error < 1e-8
///
/// Uses the algorithm described in the following reference to compute the
/// value of ln1p(x) = ln(1+x) = 2*atanh(x / (2 + x)):
/// <https://en.wikipedia.org/wiki/Natural_logarithm#High_precision>
///
/// atanh(x) itself is computed using a 15th-order polynomial expansion
/// described here:
/// <https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#Series_expansions>
///
/// The value of `x` is first reduced in range using these identities:
/// `x = m * 2^e`
/// `ln(x) = ln(m) + e * ln(2)`
///
/// This is carefully written to be branchless and therefore able to be
/// included in a series of calculations which are auto-vectorized by the
/// compiler.
#[inline(always)]
pub fn ln(x: f64) -> f64 {
    // from `x = m * 2^e`, get `m` as a double in [1,2) and
    // `e` as a double
    let bits: u64 = x.to_bits();

    // get `m` and push it into the [1,2) range
    let m_bits: u64 = (bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
    let m: f64 = f64::from_bits(m_bits);

    // get `e` by shifting mantissa out, masking the exponent bits,
    // and adjusting with offset described here:
    // <https://en.wikipedia.org/wiki/Double-precision_floating-point_format#Exponent_encoding>
    let e: f64 = (((bits >> 52) & 0x7FF) as i64 - 1023) as f64;

    // to compute `ln(m)`, we need to adjust `m`
    // `lnp1(z) = ln(1+z)`, so `lnp1(v-1) = ln(v)`
    // `v` is in [0, 1/3]
    // this has further benefit of improving accuracy of the series expansion
    // by reducing the range
    let v: f64 = (m - 1.0) / (m + 1.0);
    let v2: f64 = v * v;

    // Series expansion of atanh()
    let poly = v2.mul_add(
        v2.mul_add(
            v2.mul_add(
                v2.mul_add(
                    v2.mul_add(
                        v2.mul_add(v2.mul_add(v2 * ONE_FIFTEENTH, ONE_THIRTEENTH), ONE_ELEVENTH),
                        ONE_NINTH,
                    ),
                    ONE_SEVENTH,
                ),
                ONE_FIFTH,
            ),
            ONE_THIRD,
        ),
        1.0,
    );
    // ln(x) = ln(m) + e*ln(2)
    (2.0 * v * poly) + (e * LN_2)
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::time::Instant;

    #[test]
    fn test_ln() {
        let step: f64 = 1e-5;
        let n: usize = 100_000_000;
        let mut x: Vec<f64> = vec![0.0; n];
        x[0] = step;
        for i in 1..n {
            x[i] = x[i - 1] + step;
        }
        let mut y = vec![0.0; n];
        let mut yp = vec![0.0; n];

        let start = Instant::now();
        for i in 0..n {
            y[i] = x[i].ln();
        }
        let elapsed_slow = start.elapsed().as_secs_f64();

        let start2 = Instant::now();
        for i in 0..n {
            yp[i] = ln(x[i]);
        }
        let elapsed_fast = start2.elapsed().as_secs_f64();

        for i in 0..n {
            let err = (y[i] - yp[i]).abs();
            if err >= 1e-8 {
                println!("ERROR: x = {}, err = {}", x[i], err);
            }
        }

        let mut max_err = 0.0f64;
        let mut worst_i = 0;
        for i in 0..n {
            let err = (y[i] - yp[i]).abs();
            if err > max_err {
                max_err = err;
                worst_i = i;
            }
        }
        println!(
            "max error: {:.2e} at y={}, yp={}",
            max_err, y[worst_i], yp[worst_i]
        );
        println!("slow time: {} sec", elapsed_slow);
        println!("fast time: {} sec", elapsed_fast);
        println!("speedup: {}", elapsed_slow / elapsed_fast);
        println!("test complete.");
        assert!(max_err < 1e-8);
    }
}
