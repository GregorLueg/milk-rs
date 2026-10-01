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
//!
//! Once enough representatives exist, phase one becomes GEMM: each rayon task
//! takes a few block rows, multiplies them against tiles of the
//! representatives into a cache-resident buffer and scans it straight away.
//! The GEMM is only a prefilter; the few pairs that could beat the incumbent
//! are rescored exactly, so the result stays identical to the scalar scan.
//! Each task calls GEMM sequentially; under `accelerate` that is cblas, which
//! on Apple silicon runs on the AMX units and was 1.9x faster here than faer's
//! NEON kernel.

use ann_search_rs::utils::gemm::gemm;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;

use crate::utils::metric::{Prepared, Radius};
use crate::utils::traits::MilkFloat;

/// Sentinel for "no representative within the radius".
const NO_REP: u32 = u32::MAX;

/// Objects per sweep block. Phase two costs roughly `block x new reps in block`
/// serial score evaluations; larger blocks amortise the per-block fork-join.
/// Swept 256 to 16384 at 200k x 50: 256 was 2x slower than 1024 to 4096,
/// 16384 lost in pass one to phase two.
pub(crate) const SWEEP_BLOCK: usize = 2048;

/// Minimum objects per rayon task in the scalar phase one, so small G does not
/// drown in scheduling overhead. Untuned.
const SWEEP_MIN_PAR_LEN: usize = 16;

/// Representatives at block start from which phase one switches to GEMM.
/// Below this the GEMM call overhead outweighs the scalar scan. Untuned.
pub(crate) const GEMM_MIN_REPS: usize = 256;

/// Block rows per rayon task in the GEMM phase one. 32 to 256 were within
/// 15% at 200k x 50 on an M1 Max; 32 was the slowest.
const GEMM_ROWS: usize = 128;

/// Representatives per GEMM tile. With `GEMM_ROWS` this bounds the per-task
/// output buffer, `128 x 1024` f32 is 512 KB. 512 to 2048 were within 10%.
const GEMM_TILE: usize = 1024;

/// Rounding slack of the GEMM prefilter, in units of `(dim + 1) * eps *
/// (|x|^2 + max |r|^2)`. Approximate and exact scores differ by at most about
/// three `k * eps * (|x|^2 + |r|^2)` terms for a length-`k` dot product and
/// the two norms; the factor leaves room on top.
const GEMM_ERR_FACTOR: f64 = 6.0;

/// Outcome of one sweep.
pub(crate) struct SweepResult {
    /// Object index of each representative, in creation order. Initial
    /// representatives come first.
    pub reps: Vec<u32>,
    /// Group (position in `reps`) of each object
    pub assignment: Vec<u32>,
}

/// Representative rows in both layouts the sweep needs.
struct RepStore<T> {
    /// Rows, `G x dim`, for the scalar scan and exact rescoring
    data: Vec<T>,
    /// Rows augmented with the GEMM column, `G x (dim + 1)`
    aug: Vec<T>,
    /// Largest squared norm stored so far
    max_norm: T,
    /// Prefilter slack per unit of `|x|^2 + |r|^2`
    rel_err: T,
}

impl<T: MilkFloat> RepStore<T> {
    /// Empty store.
    ///
    /// ### Params
    ///
    /// * `prep` - Prepared data, for the dimensionality
    /// * `capacity` - Expected number of representatives
    ///
    /// ### Returns
    ///
    /// The store with the prefilter slack resolved.
    fn new(prep: &Prepared<T>, capacity: usize) -> Self {
        let factor = T::from_f64(GEMM_ERR_FACTOR * (prep.dim() + 1) as f64)
            .expect("f64 to float cannot fail");
        Self {
            data: Vec::with_capacity(capacity * prep.dim()),
            aug: Vec::with_capacity(capacity * (prep.dim() + 1)),
            max_norm: T::zero(),
            rel_err: factor * T::epsilon(),
        }
    }

    /// Append a representative row.
    ///
    /// ### Params
    ///
    /// * `prep` - Prepared data
    /// * `row` - Prepared row of the new representative
    fn push(&mut self, prep: &Prepared<T>, row: &[T]) {
        let rn = prep.sq_norm(row);
        self.max_norm = self.max_norm.max(rn);
        self.data.extend_from_slice(row);
        self.aug.extend_from_slice(row);
        self.aug.push(prep.gemm_rep_column(rn));
    }
}

/// Run the greedy sweep.
///
/// ### Params
///
/// * `prep` - Prepared data
/// * `n` - Number of objects
/// * `tau` - Radius
/// * `initial` - Pinned representatives, in group order; skipped by the sweep
/// * `block` - Objects per block, at least 1
/// * `gemm_min_reps` - Use the GEMM phase one once at least this many
///   representatives exist at block start
///
/// ### Returns
///
/// Representatives and the object to group assignment.
pub(crate) fn sweep<T: MilkFloat>(
    prep: &Prepared<T>,
    n: usize,
    tau: &Radius<T>,
    initial: &[u32],
    block: usize,
    gemm_min_reps: usize,
) -> SweepResult {
    let mut assignment = vec![NO_REP; n];
    let mut reps: Vec<u32> = Vec::with_capacity(initial.len().max(n / 4));
    let mut store = RepStore::new(prep, reps.capacity());
    for &r in initial {
        assignment[r as usize] = reps.len() as u32;
        reps.push(r);
        store.push(prep, prep.row(r as usize));
    }

    let order: Vec<u32> = (0..n as u32)
        .filter(|&i| assignment[i as usize] == NO_REP)
        .collect();
    let mut phase_one: Vec<(u32, T)> = Vec::with_capacity(block);

    for chunk in order.chunks(block) {
        let g0 = reps.len();
        phase_one.clear();
        if g0 >= gemm_min_reps.max(1) {
            phase_one_gemm(prep, chunk, &store, g0, tau, &mut phase_one);
        } else {
            chunk
                .par_iter()
                .with_min_len(SWEEP_MIN_PAR_LEN)
                .map(|&i| {
                    nearest(
                        prep,
                        prep.row(i as usize),
                        &store.data,
                        0,
                        g0,
                        tau,
                        (NO_REP, T::infinity()),
                    )
                })
                .collect_into_vec(&mut phase_one);
        }

        for (&i, &best) in chunk.iter().zip(&phase_one) {
            let q = prep.row(i as usize);
            let (g, _) = nearest(prep, q, &store.data, g0, reps.len(), tau, best);
            if g == NO_REP {
                assignment[i as usize] = reps.len() as u32;
                reps.push(i);
                store.push(prep, q);
            } else {
                assignment[i as usize] = g;
            }
        }
    }

    SweepResult { reps, assignment }
}

/// Phase one through GEMM: nearest representative within the radius for each
/// block object, over the first `g0` representatives.
///
/// Rayon owns the threads; each task runs a sequential GEMM of its rows
/// against one tile at a time. Within a task, tiles are visited in ascending
/// representative order and each row keeps its incumbent across tiles, so ties
/// resolve exactly as in the scalar scan.
///
/// ### Params
///
/// * `prep` - Prepared data
/// * `chunk` - Object indices of the block
/// * `store` - Representative rows
/// * `g0` - Number of representatives to scan
/// * `tau` - Radius
/// * `out` - Receives `(group, score)` per block object, `NO_REP` if none
fn phase_one_gemm<T: MilkFloat>(
    prep: &Prepared<T>,
    chunk: &[u32],
    store: &RepStore<T>,
    g0: usize,
    tau: &Radius<T>,
    out: &mut Vec<(u32, T)>,
) {
    let dim = prep.dim();
    let da = dim + 1;
    let rel_err = store.rel_err;
    let radius_cap = prep.radius_cap(tau);
    out.clear();
    out.resize(chunk.len(), (NO_REP, T::infinity()));

    out.par_chunks_mut(GEMM_ROWS)
        .zip(chunk.par_chunks(GEMM_ROWS))
        .for_each_init(
            || (Vec::new(), Vec::new(), Vec::new()),
            |(xa, norms, dots): &mut (Vec<T>, Vec<(T, T)>, Vec<T>), (best, rows)| {
                let r = rows.len();
                xa.clear();
                norms.clear();
                for &i in rows {
                    let row = prep.row(i as usize);
                    let xn = prep.sq_norm(row);
                    xa.extend_from_slice(row);
                    xa.push(T::one());
                    norms.push((xn, rel_err * (xn + store.max_norm)));
                }
                let lhs = MatRef::from_row_major_slice(xa.as_slice(), r, da);

                for t0 in (0..g0).step_by(GEMM_TILE) {
                    let t1 = (t0 + GEMM_TILE).min(g0);
                    let gt = t1 - t0;
                    dots.resize(r * gt, T::zero());
                    let rhs = MatRef::from_row_major_slice(&store.aug[t0 * da..t1 * da], gt, da);
                    gemm(
                        MatMut::from_row_major_slice_mut(&mut dots[..r * gt], r, gt),
                        Accum::Replace,
                        lhs,
                        rhs.transpose(),
                        T::one(),
                        Par::Seq,
                    );

                    for (k, drow) in dots[..r * gt].chunks_exact(gt).enumerate() {
                        let (xn, err) = norms[k];
                        // Rescore only pairs that could beat the incumbent (or
                        // reach the radius while there is none); the threshold
                        // tightens as the incumbent improves.
                        let threshold = |b: T| prep.gemm_threshold(xn, b.min(radius_cap), err);
                        let mut c = threshold(best[k].1);
                        let mut q: Option<&[T]> = None;
                        for (c0, ch) in drow.chunks(SCAN_CHUNK).enumerate() {
                            if !chunk_reaches(ch, c) {
                                continue;
                            }
                            let q = *q.get_or_insert_with(|| prep.row(rows[k] as usize));
                            for (j, &v) in ch.iter().enumerate() {
                                if v < c {
                                    continue;
                                }
                                let g = t0 + c0 * SCAN_CHUNK + j;
                                let s = prep.score(q, &store.data[g * dim..(g + 1) * dim]);
                                if s < best[k].1 && prep.within(s, tau) {
                                    best[k] = (g as u32, s);
                                    c = threshold(s);
                                }
                            }
                        }
                    }
                }
            },
        );
}

/// Width of the vectorised prefilter scan.
const SCAN_CHUNK: usize = 16;

/// Whether any value in a chunk reaches the bound. Written as a max-fold so it
/// vectorises.
///
/// ### Params
///
/// * `ch` - Prefilter values
/// * `c` - Bound
///
/// ### Returns
///
/// `true` if some value is `>= c`.
#[inline(always)]
fn chunk_reaches<T: MilkFloat>(ch: &[T], c: T) -> bool {
    ch.iter().fold(false, |a, &v| a | (v >= c))
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
/// * `tau` - Radius
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
    tau: &Radius<T>,
    best: (u32, T),
) -> (u32, T) {
    let dim = prep.dim();
    let row = |g: usize| &rep_data[g * dim..(g + 1) * dim];
    let (mut bg, mut bs) = best;
    let mut g = start;
    while g + 4 <= end {
        let s = prep.score_batch_4(q, [row(g), row(g + 1), row(g + 2), row(g + 3)]);
        for (k, &sk) in s.iter().enumerate() {
            if sk < bs && prep.within(sk, tau) {
                bg = (g + k) as u32;
                bs = sk;
            }
        }
        g += 4;
    }
    for g in g..end {
        let s = prep.score(q, row(g));
        if s < bs && prep.within(s, tau) {
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
                if prep.score_to_dist(s) <= tau && s < bs {
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
            let tau = estimate_threshold(&prep, n, 2.0, None, 0);
            let truth = naive_sweep(&prep, n, tau, &[]);
            assert!(
                truth.reps.len() > 10 && truth.reps.len() < n,
                "{metric:?}: {} reps",
                truth.reps.len()
            );
            for block in [1, 3, 64, SWEEP_BLOCK, 10 * n] {
                for gemm_from in [0, usize::MAX] {
                    let got = sweep(&prep, n, &prep.radius(tau), &[], block, gemm_from);
                    assert_eq!(
                        got.reps, truth.reps,
                        "{metric:?} block {block} gemm {gemm_from}"
                    );
                    assert_eq!(
                        got.assignment, truth.assignment,
                        "{metric:?} block {block} gemm {gemm_from}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_sweep_gemm_multi_tile_matches_naive() {
        let (n, dim) = (6000, 8);
        let data = random_data(n, dim, 11);
        for metric in [MilkDist::Euclidean, MilkDist::Cosine] {
            let prep = Prepared::new(&data, dim, metric);
            let tau = estimate_threshold(&prep, n, 0.05, None, 0);
            let truth = naive_sweep(&prep, n, tau, &[]);
            assert!(truth.reps.len() > 2 * GEMM_TILE, "{} reps", truth.reps.len());
            let got = sweep(&prep, n, &prep.radius(tau), &[], 3 * GEMM_ROWS + 5, 0);
            assert_eq!(got.reps, truth.reps, "{metric:?}");
            assert_eq!(got.assignment, truth.assignment, "{metric:?}");
            let pinned = sweep(&prep, n, &prep.radius(tau), &truth.reps[..500], 1000, 0);
            let pinned_truth = naive_sweep(&prep, n, tau, &truth.reps[..500]);
            assert_eq!(pinned.assignment, pinned_truth.assignment, "{metric:?} pinned");
        }
    }

    #[test]
    fn test_sweep_pinned_reps_match_naive() {
        let (n, dim) = (500, 4);
        let data = random_data(n, dim, 5);
        let prep = Prepared::new(&data, dim, MilkDist::Euclidean);
        let tau = 0.4;
        let initial = [17u32, 3, 250, 499];
        let truth = naive_sweep(&prep, n, tau, &initial);
        let got = sweep(&prep, n, &prep.radius(tau), &initial, 32, 0);
        assert_eq!(&got.reps[..4], &initial);
        assert_eq!(got.reps, truth.reps);
        assert_eq!(got.assignment, truth.assignment);
    }

    #[test]
    fn test_sweep_identical_rows_single_group() {
        let data = vec![1.5f64; 50 * 3];
        let prep = Prepared::new(&data, 3, MilkDist::Euclidean);
        let got = sweep(&prep, 50, &prep.radius(0.0), &[], 8, 0);
        assert_eq!(got.reps, vec![0]);
        assert!(got.assignment.iter().all(|&g| g == 0));
    }
}
