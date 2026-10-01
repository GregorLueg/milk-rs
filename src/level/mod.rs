//! One MILK level: radius, sweep, representative refinement, re-sweep.

pub mod medoid;
pub mod sweep;
pub mod threshold;

use std::time::Instant;

use crate::errors::MilkErrors;
use crate::level::medoid::refine_representatives;
use crate::level::sweep::{GEMM_MIN_REPS, SWEEP_BLOCK, sweep};
use crate::level::threshold::estimate_threshold;
use crate::utils::metric::{MilkDist, Prepared};
use crate::utils::traits::MilkFloat;

/// Default number of sampled pairs for the radius. 1e6 gives 0.4% CV and
/// negligible bias at 10k x 50 (proposal section 2).
const DEFAULT_N_PAIRS: usize = 1_000_000;

/// Default seed for pair sampling.
const DEFAULT_SEED: u64 = 42;

////////////
// Params //
////////////

/// Parameters for one MILK level
#[derive(Clone, Copy, Debug)]
pub struct LevelParams {
    /// Pairs sampled to estimate the radius; `None` enumerates all pairs.
    /// Default 1e6.
    pub n_pairs: Option<usize>,
    /// Seed for pair sampling. Default 42.
    pub seed: u64,
}

impl LevelParams {
    /// Create new level parameters.
    ///
    /// ### Params
    ///
    /// * `n_pairs` - Pairs sampled for the radius; `None` for all pairs
    /// * `seed` - Seed for pair sampling
    ///
    /// ### Returns
    ///
    /// The parameters.
    pub fn new(n_pairs: Option<usize>, seed: u64) -> Self {
        Self { n_pairs, seed }
    }
}

impl Default for LevelParams {
    fn default() -> Self {
        Self {
            n_pairs: Some(DEFAULT_N_PAIRS),
            seed: DEFAULT_SEED,
        }
    }
}

/// Result of one MILK level
#[derive(Clone, Debug)]
pub struct LevelResult<T> {
    /// Radius used, as a distance
    pub threshold: T,
    /// Object index of each group's representative, ascending; group `g` is
    /// the group whose representative is `reps[g]`
    pub reps: Vec<u32>,
    /// Group of each object; the parent-pointer array for this level
    pub assignment: Vec<u32>,
    /// Number of groups after the first sweep, before refinement
    pub n_groups_pass1: usize,
}

///////////
// Level //
///////////

/// Run one MILK level over a dense matrix.
///
/// Estimates the radius, sweeps in input order, moves each representative to
/// the member nearest its group centroid, then sweeps the remaining objects
/// again with those representatives pinned. Objects outside every pinned
/// radius still open new groups in the second sweep.
///
/// ### Params
///
/// * `data` - Flat row-major data, `n x dim`
/// * `n` - Number of objects
/// * `dim` - Dimensionality
/// * `metric` - Distance metric
/// * `percentile` - Radius percentile in percent, `[0, 100]` (reference
///   default 1.0)
/// * `params` - Optional level parameters, defaults otherwise
/// * `verbose` - Print stage timings
///
/// ### Returns
///
/// The radius, representatives and assignment.
///
/// ### References
///
/// Kiyota, Lee, Yao and Yachie, bioRxiv, 2026
pub fn milk_level<T: MilkFloat>(
    data: &[T],
    n: usize,
    dim: usize,
    metric: MilkDist,
    percentile: f64,
    params: Option<LevelParams>,
    verbose: bool,
) -> Result<LevelResult<T>, MilkErrors> {
    let params = params.unwrap_or_default();
    if n == 0 || dim == 0 {
        return Err(MilkErrors::EmptyInput);
    }
    if data.len() != n * dim {
        return Err(MilkErrors::ShapeMismatch {
            len: data.len(),
            n,
            dim,
        });
    }
    if !(0.0..=100.0).contains(&percentile) {
        return Err(MilkErrors::InvalidParam {
            param: "percentile",
            reason: format!("{percentile} is outside [0, 100]"),
        });
    }
    if n == 1 {
        return Ok(LevelResult {
            threshold: T::zero(),
            reps: vec![0],
            assignment: vec![0],
            n_groups_pass1: 1,
        });
    }

    let start = Instant::now();
    let prep = Prepared::new(data, dim, metric);
    let t_prep = start.elapsed();

    let start = Instant::now();
    let threshold = estimate_threshold(&prep, n, percentile, params.n_pairs, params.seed);
    let tau = prep.radius(threshold);
    let t_threshold = start.elapsed();

    let start = Instant::now();
    let first = sweep(&prep, n, &tau, &[], SWEEP_BLOCK, GEMM_MIN_REPS);
    let t_pass1 = start.elapsed();

    let start = Instant::now();
    let medoids = refine_representatives(data, &prep, &first.assignment, &first.reps);
    let t_medoid = start.elapsed();

    let start = Instant::now();
    let second = sweep(&prep, n, &tau, &medoids, SWEEP_BLOCK, GEMM_MIN_REPS);
    let t_pass2 = start.elapsed();

    if verbose {
        println!(
            "prep {:.3}s | threshold {:.3}s | pass 1 {:.3}s ({} groups) | medoid {:.3}s | pass 2 {:.3}s ({} groups)",
            t_prep.as_secs_f64(),
            t_threshold.as_secs_f64(),
            t_pass1.as_secs_f64(),
            first.reps.len(),
            t_medoid.as_secs_f64(),
            t_pass2.as_secs_f64(),
            second.reps.len(),
        );
    }

    // Number groups by their representative's input position, so the next
    // level sweeps in input order. Creation order is not input order (medoids
    // move, new reps append) and measurably changes how fast the tree
    // compresses.
    let mut reps = second.reps;
    let mut rank = vec![0u32; reps.len()];
    let mut by_pos: Vec<u32> = (0..reps.len() as u32).collect();
    by_pos.sort_unstable_by_key(|&g| reps[g as usize]);
    for (new, &old) in by_pos.iter().enumerate() {
        rank[old as usize] = new as u32;
    }
    reps.sort_unstable();
    let assignment = second
        .assignment
        .iter()
        .map(|&g| rank[g as usize])
        .collect();

    Ok(LevelResult {
        threshold,
        reps,
        assignment,
        n_groups_pass1: first.reps.len(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_milk_level_trivial_inputs() {
        let one = milk_level(&[1.0f64, 2.0], 1, 2, MilkDist::Euclidean, 1.0, None, false).unwrap();
        assert_eq!(one.reps, vec![0]);

        let two = milk_level(
            &[0.0f64, 0.0, 3.0, 4.0],
            2,
            2,
            MilkDist::Euclidean,
            1.0,
            None,
            false,
        )
        .unwrap();
        // One pair: the radius is that distance, so the second object joins the first.
        assert_eq!(two.threshold, 5.0);
        assert_eq!(two.reps.len(), 1);
        assert_eq!(two.assignment, vec![0, 0]);
    }

    #[test]
    fn test_milk_level_rejects_bad_input() {
        assert!(matches!(
            milk_level::<f64>(&[], 0, 2, MilkDist::Euclidean, 1.0, None, false),
            Err(MilkErrors::EmptyInput)
        ));
        assert!(matches!(
            milk_level(&[1.0f64; 5], 2, 3, MilkDist::Euclidean, 1.0, None, false),
            Err(MilkErrors::ShapeMismatch { .. })
        ));
        assert!(matches!(
            milk_level(&[1.0f64; 6], 2, 3, MilkDist::Euclidean, 101.0, None, false),
            Err(MilkErrors::InvalidParam { .. })
        ));
    }

    #[test]
    fn test_milk_level_groups_cover_and_reps_self_assigned() {
        let (n, dim) = (1000, 5);
        let mut rng = StdRng::seed_from_u64(9);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.random::<f32>()).collect();
        let res = milk_level(&data, n, dim, MilkDist::Euclidean, 1.0, None, false).unwrap();
        assert!(res.reps.len() > 1 && res.reps.len() < n);
        assert!(res.reps.windows(2).all(|w| w[0] < w[1]));
        for (g, &r) in res.reps.iter().enumerate() {
            assert_eq!(res.assignment[r as usize], g as u32);
        }
        assert!(
            res.assignment
                .iter()
                .all(|&g| (g as usize) < res.reps.len())
        );

        let again = milk_level(&data, n, dim, MilkDist::Euclidean, 1.0, None, false).unwrap();
        assert_eq!(res.assignment, again.assignment);
    }
}
