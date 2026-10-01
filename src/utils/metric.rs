//! Distance metrics.
//!
//! The sweep compares "scores" rather than distances, chosen so that every
//! metric is a single SIMD kernel per pair: squared Euclidean for Euclidean,
//! `1 - dot` on unit rows for cosine, and `1 - dot` on centred unit rows for
//! correlation. Cosine and correlation therefore transform a working copy of
//! the data once per level; Euclidean borrows the input.

use rayon::prelude::*;
use std::borrow::Cow;

use crate::errors::MilkErrors;
use crate::utils::traits::MilkFloat;

////////////
// Metric //
////////////

/// Distance metrics supported by MILK
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MilkDist {
    /// Euclidean distance
    #[default]
    Euclidean,
    /// Cosine distance, `1 - cos(a, b)`
    Cosine,
    /// Correlation distance, cosine distance of the row-centred vectors
    Correlation,
}

/// Parse a metric name.
///
/// ### Params
///
/// * `s` - One of `"euclidean"`, `"cosine"` or `"correlation"`
///
/// ### Returns
///
/// The metric, or `MilkErrors::DistanceNotSupported` for anything else
/// (including the reference's `manhattan`, `hamming` and `jaccard`).
pub fn parse_milk_dist(s: &str) -> Result<MilkDist, MilkErrors> {
    match s.to_lowercase().as_str() {
        "euclidean" => Ok(MilkDist::Euclidean),
        "cosine" => Ok(MilkDist::Cosine),
        "correlation" => Ok(MilkDist::Correlation),
        _ => Err(MilkErrors::DistanceNotSupported(s.to_string())),
    }
}

//////////////
// Prepared //
//////////////

/// Row-major data in the form the score kernels expect.
pub(crate) struct Prepared<'a, T: Clone> {
    /// Flat row-major data, transformed for cosine and correlation
    data: Cow<'a, [T]>,
    /// Dimensionality of each row
    dim: usize,
    /// Metric the scores implement
    metric: MilkDist,
}

impl<'a, T: MilkFloat> Prepared<'a, T> {
    /// Prepare data for scoring.
    ///
    /// ### Params
    ///
    /// * `data` - Flat row-major data, `n x dim`
    /// * `dim` - Dimensionality
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// Borrowed data for Euclidean, a transformed copy otherwise.
    pub(crate) fn new(data: &'a [T], dim: usize, metric: MilkDist) -> Self {
        let data = match metric {
            MilkDist::Euclidean => Cow::Borrowed(data),
            MilkDist::Cosine | MilkDist::Correlation => {
                let mut owned = data.to_vec();
                owned
                    .par_chunks_exact_mut(dim)
                    .for_each(|row| transform_row(row, metric));
                Cow::Owned(owned)
            }
        };
        Self { data, dim, metric }
    }

    /// Prepared row `i`.
    ///
    /// ### Params
    ///
    /// * `i` - Row index
    ///
    /// ### Returns
    ///
    /// Slice of length `dim`.
    #[inline(always)]
    pub(crate) fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Dimensionality.
    ///
    /// ### Returns
    ///
    /// Length of each row.
    #[inline(always)]
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// Metric the scores implement.
    ///
    /// ### Returns
    ///
    /// The metric.
    #[inline(always)]
    pub(crate) fn metric(&self) -> MilkDist {
        self.metric
    }

    /// Score between two prepared rows.
    ///
    /// ### Params
    ///
    /// * `a` - Prepared row
    /// * `b` - Prepared row
    ///
    /// ### Returns
    ///
    /// Squared Euclidean distance, or `1 - dot` for cosine and correlation.
    #[inline(always)]
    pub(crate) fn score(&self, a: &[T], b: &[T]) -> T {
        match self.metric {
            MilkDist::Euclidean => T::euclidean_simd(a, b),
            MilkDist::Cosine | MilkDist::Correlation => T::one() - T::dot_simd(a, b),
        }
    }

    /// Scores of one row against four rows.
    ///
    /// ### Params
    ///
    /// * `q` - Prepared query row
    /// * `y` - Four prepared rows
    ///
    /// ### Returns
    ///
    /// The four scores in input order.
    #[inline(always)]
    pub(crate) fn score_batch_4(&self, q: &[T], y: [&[T]; 4]) -> [T; 4] {
        match self.metric {
            MilkDist::Euclidean => T::euclidean_simd_batch_4(q, y),
            MilkDist::Cosine | MilkDist::Correlation => {
                T::dot_simd_batch_4(q, y).map(|d| T::one() - d)
            }
        }
    }

    /// Convert a distance into score units.
    ///
    /// ### Params
    ///
    /// * `d` - Distance
    ///
    /// ### Returns
    ///
    /// `d^2` for Euclidean, `d` otherwise.
    #[inline(always)]
    pub(crate) fn dist_to_score(&self, d: T) -> T {
        match self.metric {
            MilkDist::Euclidean => d * d,
            MilkDist::Cosine | MilkDist::Correlation => d,
        }
    }

    /// Convert a score into a distance.
    ///
    /// ### Params
    ///
    /// * `s` - Score
    ///
    /// ### Returns
    ///
    /// `sqrt(max(s, 0))` for Euclidean, `s` otherwise.
    #[inline(always)]
    pub(crate) fn score_to_dist(&self, s: T) -> T {
        match self.metric {
            MilkDist::Euclidean => s.max(T::zero()).sqrt(),
            MilkDist::Cosine | MilkDist::Correlation => s,
        }
    }
}

/// Transform a raw row in place into the form the metric's score expects.
///
/// Mean and norm accumulate in f64. A zero-norm row stays all zeros, so its
/// cosine distance to anything is 1 rather than the reference's NaN.
///
/// ### Params
///
/// * `row` - Raw row, overwritten
/// * `metric` - Distance metric; Euclidean leaves the row untouched
pub(crate) fn transform_row<T: MilkFloat>(row: &mut [T], metric: MilkDist) {
    if metric == MilkDist::Euclidean {
        return;
    }
    if metric == MilkDist::Correlation {
        let mean = row.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum::<f64>() / row.len() as f64;
        let mean = T::from_f64(mean).expect("f64 to float cannot fail");
        row.iter_mut().for_each(|x| *x = *x - mean);
    }
    let norm = row
        .iter()
        .map(|x| {
            let v = x.to_f64().unwrap_or(0.0);
            v * v
        })
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let inv = T::from_f64(1.0 / norm).expect("f64 to float cannot fail");
        row.iter_mut().for_each(|x| *x = *x * inv);
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parse_milk_dist_rejects_manhattan() {
        assert_eq!(parse_milk_dist("Cosine").unwrap(), MilkDist::Cosine);
        assert!(matches!(
            parse_milk_dist("manhattan"),
            Err(MilkErrors::DistanceNotSupported(_))
        ));
    }

    #[test]
    fn test_scores_match_definitions() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [2.0f64, 0.0, 1.0, 5.0];
        let data: Vec<f64> = a.iter().chain(b.iter()).copied().collect();

        let eu = Prepared::new(&data, 4, MilkDist::Euclidean);
        assert_relative_eq!(
            eu.score_to_dist(eu.score(eu.row(0), eu.row(1))),
            10f64.sqrt()
        );

        let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cos = Prepared::new(&data, 4, MilkDist::Cosine);
        assert_relative_eq!(
            cos.score(cos.row(0), cos.row(1)),
            1.0 - dot / (na * nb),
            epsilon = 1e-12
        );

        let ma = a.iter().sum::<f64>() / 4.0;
        let mb = b.iter().sum::<f64>() / 4.0;
        let ac: Vec<f64> = a.iter().map(|x| x - ma).collect();
        let bc: Vec<f64> = b.iter().map(|x| x - mb).collect();
        let r = ac.iter().zip(&bc).map(|(x, y)| x * y).sum::<f64>()
            / (ac.iter().map(|x| x * x).sum::<f64>().sqrt()
                * bc.iter().map(|x| x * x).sum::<f64>().sqrt());
        let cor = Prepared::new(&data, 4, MilkDist::Correlation);
        assert_relative_eq!(cor.score(cor.row(0), cor.row(1)), 1.0 - r, epsilon = 1e-12);
    }
}
