//! Representative refinement.
//!
//! Each group's representative moves to the member nearest the group centroid.
//! This is the reference's "medoid": `O(|g|)` rather than the true medoid's
//! `O(|g|^2)`, and not metric-consistent for cosine and correlation, where the
//! centroid is the mean of the raw vectors. Kept as the reference has it.

use rayon::prelude::*;

use crate::utils::metric::{Prepared, transform_row};
use crate::utils::traits::MilkFloat;

/// Move every representative to the member nearest its group centroid.
///
/// Centroids accumulate in f64 and are unweighted by any leaf counts below
/// the level. Ties go to the lowest object index.
///
/// ### Params
///
/// * `data` - Raw flat row-major data, `n x dim`
/// * `prep` - The same data prepared for scoring
/// * `assignment` - Group of each object
/// * `reps` - Current representative of each group
///
/// ### Returns
///
/// The refined representative (object index) of each group, in group order.
pub(crate) fn refine_representatives<T: MilkFloat>(
    data: &[T],
    prep: &Prepared<T>,
    assignment: &[u32],
    reps: &[u32],
) -> Vec<u32> {
    let dim = prep.dim();
    let n_groups = reps.len();

    // CSR groups; members end up in ascending object order.
    let mut ptr = vec![0usize; n_groups + 1];
    for &g in assignment {
        ptr[g as usize + 1] += 1;
    }
    for g in 0..n_groups {
        ptr[g + 1] += ptr[g];
    }
    let mut cursor = ptr.clone();
    let mut members = vec![0u32; assignment.len()];
    for (i, &g) in assignment.iter().enumerate() {
        members[cursor[g as usize]] = i as u32;
        cursor[g as usize] += 1;
    }

    (0..n_groups)
        .into_par_iter()
        .map_init(
            || (vec![0f64; dim], vec![T::zero(); dim]),
            |(acc, centroid), g| {
                let group = &members[ptr[g]..ptr[g + 1]];
                if group.len() == 1 {
                    return reps[g];
                }
                acc.iter_mut().for_each(|a| *a = 0.0);
                for &m in group {
                    let row = &data[m as usize * dim..(m as usize + 1) * dim];
                    for (a, x) in acc.iter_mut().zip(row) {
                        *a += x.to_f64().unwrap_or(0.0);
                    }
                }
                let inv = 1.0 / group.len() as f64;
                for (c, a) in centroid.iter_mut().zip(acc.iter()) {
                    *c = T::from_f64(*a * inv).expect("f64 to float cannot fail");
                }
                transform_row(centroid, prep.metric());

                let mut best = (group[0], T::infinity());
                for &m in group {
                    let s = prep.score(centroid, prep.row(m as usize));
                    if s < best.1 {
                        best = (m, s);
                    }
                }
                best.0
            },
        )
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::metric::MilkDist;

    #[test]
    fn test_refine_picks_member_nearest_centroid() {
        // Group 0: 1-D points 0, 1, 2, 10 -> centroid 3.25, nearest member 2.
        // Group 1: a singleton keeps its representative.
        let data = vec![0.0f64, 1.0, 2.0, 10.0, 50.0];
        let prep = Prepared::new(&data, 1, MilkDist::Euclidean);
        let assignment = [0u32, 0, 0, 0, 1];
        let refined = refine_representatives(&data, &prep, &assignment, &[0, 4]);
        assert_eq!(refined, vec![2, 4]);
    }

    #[test]
    fn test_refine_cosine_uses_raw_centroid() {
        // Raw centroid of (1,0), (10,10), (0,1) is (11/3, 11/3), direction (1,1);
        // the member pointing that way is (10,10) although it is not the
        // Euclidean nearest.
        let data = vec![1.0f64, 0.0, 10.0, 10.0, 0.0, 1.0];
        let prep = Prepared::new(&data, 2, MilkDist::Cosine);
        let refined = refine_representatives(&data, &prep, &[0, 0, 0], &[0]);
        assert_eq!(refined, vec![1]);
    }
}
