//! Radius estimation from the pairwise distance distribution.
//!
//! The paper describes sampling pairs; the reference enumerates every pair
//! inside each partition. Sampling 1e6 pairs gives 0.4% CV on the threshold
//! (proposal section 2), so it is the default, with exact enumeration kept for
//! validation and small inputs.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::utils::metric::Prepared;
use crate::utils::traits::MilkFloat;

/// Estimate the radius as a percentile of pairwise distances.
///
/// Falls back to exact enumeration when the number of distinct pairs does not
/// exceed `n_pairs`.
///
/// ### Params
///
/// * `prep` - Prepared data
/// * `n` - Number of objects, at least 2
/// * `percentile` - Percentile in percent, `[0, 100]`
/// * `n_pairs` - Pairs to sample with replacement; `None` enumerates all pairs
/// * `seed` - Seed for pair sampling
///
/// ### Returns
///
/// The radius as a distance (not a score).
pub(crate) fn estimate_threshold<T: MilkFloat>(
    prep: &Prepared<T>,
    n: usize,
    percentile: f64,
    n_pairs: Option<usize>,
    seed: u64,
) -> T {
    let total = n * (n - 1) / 2;
    let mut dists: Vec<T> = match n_pairs {
        Some(k) if k < total => {
            let mut rng = StdRng::seed_from_u64(seed);
            let pairs: Vec<(u32, u32)> = (0..k)
                .map(|_| {
                    let i = rng.random_range(0..n);
                    let mut j = rng.random_range(0..n - 1);
                    if j >= i {
                        j += 1;
                    }
                    (i as u32, j as u32)
                })
                .collect();
            pairs
                .par_iter()
                .map(|&(i, j)| {
                    prep.score_to_dist(prep.score(prep.row(i as usize), prep.row(j as usize)))
                })
                .collect()
        }
        _ => (0..n - 1)
            .into_par_iter()
            .flat_map_iter(|i| {
                let a = prep.row(i);
                (i + 1..n).map(move |j| prep.score_to_dist(prep.score(a, prep.row(j))))
            })
            .collect(),
    };
    T::from_f64(percentile_type7(&mut dists, percentile)).expect("f64 to float cannot fail")
}

/// Percentile by linear interpolation between order statistics.
///
/// Matches R `quantile(type = 7)` and StatsBase `percentile`, which is what the
/// reference uses. Partially reorders `values`.
///
/// ### Params
///
/// * `values` - Non-empty values, reordered in place
/// * `percentile` - Percentile in percent, `[0, 100]`
///
/// ### Returns
///
/// The interpolated percentile in f64.
pub(crate) fn percentile_type7<T: MilkFloat>(values: &mut [T], percentile: f64) -> f64 {
    let cmp = |a: &T, b: &T| a.partial_cmp(b).unwrap_or(Ordering::Equal);
    let h = (values.len() - 1) as f64 * percentile / 100.0;
    let lo = h.floor() as usize;
    let (_, lo_val, upper) = values.select_nth_unstable_by(lo, cmp);
    let lo_val = lo_val.to_f64().unwrap_or(f64::NAN);
    match upper.iter().min_by(|a, b| cmp(a, b)) {
        Some(hi) => lo_val + (h - lo as f64) * (hi.to_f64().unwrap_or(f64::NAN) - lo_val),
        None => lo_val,
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::metric::MilkDist;
    use approx::assert_relative_eq;

    #[test]
    fn test_percentile_type7_matches_r() {
        // R: quantile(c(3.1, 0.2, 7.5, 1.1, 4.4, 9.9, 2.6), c(0.001, 0.01, 0.1, 0.5, 0.9, 1), type = 7)
        let x = [3.1f64, 0.2, 7.5, 1.1, 4.4, 9.9, 2.6];
        let expected = [0.2054, 0.254, 0.74, 3.1, 8.46, 9.9];
        for (p, e) in [0.1, 1.0, 10.0, 50.0, 90.0, 100.0].iter().zip(expected) {
            let mut v = x.to_vec();
            assert_relative_eq!(percentile_type7(&mut v, *p), e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_sampled_threshold_close_to_exact() {
        let n = 400;
        let dim = 8;
        let mut rng = StdRng::seed_from_u64(1);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.random::<f64>()).collect();
        let prep = Prepared::new(&data, dim, MilkDist::Euclidean);
        let exact = estimate_threshold(&prep, n, 10.0, None, 0);
        let sampled = estimate_threshold(&prep, n, 10.0, Some(50_000), 0);
        assert_relative_eq!(exact, sampled, max_relative = 0.02);
        // Fewer distinct pairs than requested: exact path, any seed.
        assert_eq!(
            exact,
            estimate_threshold(&prep, n, 10.0, Some(10_000_000), 7)
        );
    }

    #[test]
    fn test_threshold_same_seed_reproducible() {
        let n = 300;
        let dim = 4;
        let mut rng = StdRng::seed_from_u64(2);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.random::<f32>()).collect();
        let prep = Prepared::new(&data, dim, MilkDist::Cosine);
        let a = estimate_threshold(&prep, n, 1.0, Some(1000), 11);
        let b = estimate_threshold(&prep, n, 1.0, Some(1000), 11);
        assert_eq!(a, b);
    }
}
