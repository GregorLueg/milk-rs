//! Greedy radius-cover sweep.
//!
//! Each object, in input order, joins the nearest representative within the
//! radius or becomes a new representative. The serial definition is batched
//! exactly in two phases per block: phase one scores the block in parallel
//! against the representatives that existed at block start; phase two walks
//! the block serially against only the representatives created inside it.
//! Strict `<` everywhere means ties go to the earliest representative, as in
//! the serial sweep, so the result is independent of block size and thread
//! count.

use rayon::prelude::*;

use crate::utils::metric::Prepared;
use crate::utils::traits::MilkFloat;

/// Sentinel for "no representative within the radius".
const NO_REP: u32 = u32::MAX;

/// Objects per sweep block. Phase two costs roughly `block x new reps in block`
/// serial score evaluations, phase one is parallel; 256 keeps phase two small
/// against phase one once G is in the thousands. Untuned.
pub(crate) const SWEEP_BLOCK: usize = 256;

/// Minimum objects per rayon task in phase one, so small G does not drown in
/// scheduling overhead. Untuned.
const SWEEP_MIN_PAR_LEN: usize = 16;

/// Outcome of one sweep.
pub(crate) struct SweepResult {
    /// Object index of each representative, in creation order. Initial
    /// representatives come first.
    pub reps: Vec<u32>,
    /// Group (position in `reps`) of each object
    pub assignment: Vec<u32>,
}

/// Run the greedy sweep.
///
/// ### Params
///
/// * `prep` - Prepared data
/// * `n` - Number of objects
/// * `tau` - Radius in score units
/// * `initial` - Pinned representatives, in group order; skipped by the sweep
/// * `block` - Objects per block, at least 1
///
/// ### Returns
///
/// Representatives and the object to group assignment.
pub(crate) fn sweep<T: MilkFloat>(
    prep: &Prepared<T>,
    n: usize,
    tau: T,
    initial: &[u32],
    block: usize,
) -> SweepResult {
    let dim = prep.dim();
    let mut assignment = vec![NO_REP; n];
    let mut reps: Vec<u32> = Vec::with_capacity(initial.len().max(n / 4));
    // Contiguous copy of the representative rows so the scan streams them.
    let mut rep_data: Vec<T> = Vec::with_capacity(reps.capacity() * dim);
    for &r in initial {
        assignment[r as usize] = reps.len() as u32;
        reps.push(r);
        rep_data.extend_from_slice(prep.row(r as usize));
    }

    let order: Vec<u32> = (0..n as u32)
        .filter(|&i| assignment[i as usize] == NO_REP)
        .collect();
    let mut phase_one: Vec<(u32, T)> = Vec::with_capacity(block);

    for chunk in order.chunks(block) {
        let g0 = reps.len();
        phase_one.clear();
        chunk
            .par_iter()
            .with_min_len(SWEEP_MIN_PAR_LEN)
            .map(|&i| {
                nearest(
                    prep,
                    prep.row(i as usize),
                    &rep_data,
                    0,
                    g0,
                    tau,
                    (NO_REP, T::infinity()),
                )
            })
            .collect_into_vec(&mut phase_one);

        for (&i, &best) in chunk.iter().zip(&phase_one) {
            let q = prep.row(i as usize);
            let (g, _) = nearest(prep, q, &rep_data, g0, reps.len(), tau, best);
            if g == NO_REP {
                assignment[i as usize] = reps.len() as u32;
                reps.push(i);
                rep_data.extend_from_slice(q);
            } else {
                assignment[i as usize] = g;
            }
        }
    }

    SweepResult { reps, assignment }
}

/// Nearest representative within the radius over a range of representatives.
///
/// ### Params
///
/// * `prep` - Prepared data (for the metric)
/// * `q` - Prepared query row
/// * `rep_data` - Contiguous representative rows
/// * `start` - First representative to scan
/// * `end` - One past the last representative to scan
/// * `tau` - Radius in score units
/// * `best` - Incumbent `(group, score)`; kept on ties
///
/// ### Returns
///
/// Updated `(group, score)`, group `NO_REP` if none within the radius.
#[inline]
fn nearest<T: MilkFloat>(
    prep: &Prepared<T>,
    q: &[T],
    rep_data: &[T],
    start: usize,
    end: usize,
    tau: T,
    best: (u32, T),
) -> (u32, T) {
    let dim = prep.dim();
    let row = |g: usize| &rep_data[g * dim..(g + 1) * dim];
    let (mut bg, mut bs) = best;
    let mut g = start;
    while g + 4 <= end {
        let s = prep.score_batch_4(q, [row(g), row(g + 1), row(g + 2), row(g + 3)]);
        for (k, &sk) in s.iter().enumerate() {
            if sk <= tau && sk < bs {
                bg = (g + k) as u32;
                bs = sk;
            }
        }
        g += 4;
    }
    for g in g..end {
        let s = prep.score(q, row(g));
        if s <= tau && s < bs {
            bg = g as u32;
            bs = s;
        }
    }
    (bg, bs)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::threshold::estimate_threshold;
    use crate::utils::metric::MilkDist;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Textbook serial sweep, the definition the batched version must match.
    #[allow(clippy::needless_range_loop)]
    fn naive_sweep(prep: &Prepared<f64>, n: usize, tau: f64, initial: &[u32]) -> SweepResult {
        let mut assignment = vec![NO_REP; n];
        let mut reps: Vec<u32> = Vec::new();
        for &r in initial {
            assignment[r as usize] = reps.len() as u32;
            reps.push(r);
        }
        for i in 0..n {
            if assignment[i] != NO_REP {
                continue;
            }
            let (mut bg, mut bs) = (NO_REP, f64::INFINITY);
            for (g, &r) in reps.iter().enumerate() {
                let s = prep.score(prep.row(i), prep.row(r as usize));
                if s <= tau && s < bs {
                    bg = g as u32;
                    bs = s;
                }
            }
            if bg == NO_REP {
                assignment[i] = reps.len() as u32;
                reps.push(i as u32);
            } else {
                assignment[i] = bg;
            }
        }
        SweepResult { reps, assignment }
    }

    fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.random::<f64>()).collect()
    }

    #[test]
    fn test_sweep_block_matches_naive_all_metrics() {
        let (n, dim) = (700, 6);
        let data = random_data(n, dim, 3);
        for metric in [MilkDist::Euclidean, MilkDist::Cosine, MilkDist::Correlation] {
            let prep = Prepared::new(&data, dim, metric);
            let tau = prep.dist_to_score(estimate_threshold(&prep, n, 2.0, None, 0));
            let truth = naive_sweep(&prep, n, tau, &[]);
            assert!(
                truth.reps.len() > 10 && truth.reps.len() < n,
                "{metric:?}: {} reps",
                truth.reps.len()
            );
            for block in [1, 3, 64, SWEEP_BLOCK, 10 * n] {
                let got = sweep(&prep, n, tau, &[], block);
                assert_eq!(got.reps, truth.reps, "{metric:?} block {block}");
                assert_eq!(got.assignment, truth.assignment, "{metric:?} block {block}");
            }
        }
    }

    #[test]
    fn test_sweep_pinned_reps_match_naive() {
        let (n, dim) = (500, 4);
        let data = random_data(n, dim, 5);
        let prep = Prepared::new(&data, dim, MilkDist::Euclidean);
        let tau = prep.dist_to_score(0.4);
        let initial = [17u32, 3, 250, 499];
        let truth = naive_sweep(&prep, n, tau, &initial);
        let got = sweep(&prep, n, tau, &initial, 32);
        assert_eq!(&got.reps[..4], &initial);
        assert_eq!(got.reps, truth.reps);
        assert_eq!(got.assignment, truth.assignment);
    }

    #[test]
    fn test_sweep_identical_rows_single_group() {
        let data = vec![1.5f64; 50 * 3];
        let prep = Prepared::new(&data, 3, MilkDist::Euclidean);
        let got = sweep(&prep, 50, 0.0, &[], 8);
        assert_eq!(got.reps, vec![0]);
        assert!(got.assignment.iter().all(|&g| g == 0));
    }
}
