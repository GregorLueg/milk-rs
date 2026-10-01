//! The MILK recursion and the tree it builds.
//!
//! Each level's representatives are real cells, so recursing only needs their
//! rows gathered into a smaller matrix. The tree is stored as one parent array
//! per level, `O(N)` in total. Node ids for the flat views put the leaves at
//! `0..n_leaves` and each level's groups in a contiguous block after the one
//! below, so every parent id is larger than its children's. Single-child
//! groups are kept as they are; collapsing them is left to the consumer.

use crate::errors::MilkErrors;
use crate::level::{LevelParams, milk_level};
use crate::utils::metric::MilkDist;
use crate::utils::traits::MilkFloat;

/// Parent id of a root in [`MilkTree::parent_array`].
pub const NO_NODE: u32 = u32::MAX;

///////////
// Types //
///////////

/// One level of the tree
#[derive(Clone, Debug)]
pub struct MilkLevel<T> {
    /// Radius used at this level, as a distance
    pub threshold: T,
    /// Input cell index of each group's representative
    pub reps: Vec<u32>,
    /// For each object of the level below (cells for the first level), the
    /// group it belongs to at this level
    pub parent: Vec<u32>,
}

/// Tree from a full MILK run
#[derive(Clone, Debug)]
pub struct MilkTree<T> {
    /// Number of input cells, the leaves
    pub n_leaves: usize,
    /// Levels from the finest (just above the leaves) to the coarsest
    pub levels: Vec<MilkLevel<T>>,
}

/// Per-node attributes, the reference's `vertices.csv` minus the optional
/// `spread`, `specificity` and `resolution` columns
#[derive(Clone, Debug)]
pub struct MilkVertices<T> {
    /// Input cell index represented by the node; the cell itself for a leaf
    pub representative: Vec<u32>,
    /// Number of leaves below the node
    pub group_size: Vec<u32>,
    /// 0 for leaves, `l + 1` for groups of level `l`
    pub iteration: Vec<u32>,
    /// Radius of the node's level, zero for leaves
    pub threshold: Vec<T>,
}

impl<T: MilkFloat> MilkTree<T> {
    /// Total number of nodes, leaves included.
    ///
    /// ### Returns
    ///
    /// `n_leaves` plus the group count of every level.
    pub fn n_nodes(&self) -> usize {
        self.n_leaves + self.levels.iter().map(|l| l.reps.len()).sum::<usize>()
    }

    /// Parent of every node as one flat array.
    ///
    /// ### Returns
    ///
    /// Parent id per node, [`NO_NODE`] for the groups of the top level. There
    /// is one root unless the run stopped early through `sample_size`.
    pub fn parent_array(&self) -> Vec<u32> {
        let mut parent = Vec::with_capacity(self.n_nodes());
        let mut offset = self.n_leaves;
        for level in &self.levels {
            parent.extend(level.parent.iter().map(|&g| (offset + g as usize) as u32));
            offset += level.reps.len();
        }
        parent.resize(self.n_nodes(), NO_NODE);
        parent
    }

    /// Per-node attributes, indexed like [`MilkTree::parent_array`].
    ///
    /// ### Returns
    ///
    /// Representative cell, leaf count, iteration and radius of every node.
    pub fn vertices(&self) -> MilkVertices<T> {
        let n_nodes = self.n_nodes();
        let mut v = MilkVertices {
            representative: (0..self.n_leaves as u32).collect(),
            group_size: vec![1; self.n_leaves],
            iteration: vec![0; self.n_leaves],
            threshold: vec![T::zero(); self.n_leaves],
        };
        v.representative.reserve(n_nodes - self.n_leaves);

        let mut below = 0;
        for (l, level) in self.levels.iter().enumerate() {
            let start = v.group_size.len();
            let mut sizes = vec![0u32; level.reps.len()];
            for (child, &g) in level.parent.iter().enumerate() {
                sizes[g as usize] += v.group_size[below + child];
            }
            v.representative.extend_from_slice(&level.reps);
            v.group_size.extend(sizes);
            v.iteration
                .extend(std::iter::repeat_n(l as u32 + 1, level.reps.len()));
            v.threshold
                .extend(std::iter::repeat_n(level.threshold, level.reps.len()));
            below = start;
        }
        v
    }
}

///////////////
// Recursion //
///////////////

/// Run MILK to completion.
///
/// Repeats [`milk_level`] on each level's representatives, recomputing the
/// radius every level, until at most `sample_size` objects remain.
///
/// ### Params
///
/// * `data` - Flat row-major data, `n x dim`
/// * `n` - Number of cells
/// * `dim` - Dimensionality
/// * `metric` - Distance metric
/// * `percentile` - Radius percentile in percent, `[0, 100]` (reference
///   default 1.0)
/// * `sample_size` - Stop once this many objects or fewer remain, at least 1
///   (reference default 1, a single root)
/// * `params` - Optional level parameters; the sampling seed is offset by the
///   level index
/// * `verbose` - Print one line per level
///
/// ### Returns
///
/// The tree, or `MilkErrors::NoProgress` if a level fails to merge anything.
///
/// ### References
///
/// Kiyota, Lee, Yao and Yachie, bioRxiv, 2026
#[allow(clippy::too_many_arguments)]
pub fn milk_tree<T: MilkFloat>(
    data: &[T],
    n: usize,
    dim: usize,
    metric: MilkDist,
    percentile: f64,
    sample_size: usize,
    params: Option<LevelParams>,
    verbose: bool,
) -> Result<MilkTree<T>, MilkErrors> {
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
    if sample_size == 0 {
        return Err(MilkErrors::InvalidParam {
            param: "sample_size",
            reason: "must be at least 1".to_string(),
        });
    }

    let mut levels = Vec::new();
    // Input cell index of each object at the current level.
    let mut cells: Vec<u32> = (0..n as u32).collect();
    let mut owned: Option<Vec<T>> = None;

    while cells.len() > sample_size {
        let rows = owned.as_deref().unwrap_or(data);
        let level_params = LevelParams::new(
            params.n_pairs,
            params.seed.wrapping_add(levels.len() as u64),
        );
        let res = milk_level(
            rows,
            cells.len(),
            dim,
            metric,
            percentile,
            Some(level_params),
            verbose,
        )?;
        if res.reps.len() == cells.len() {
            return Err(MilkErrors::NoProgress {
                level: levels.len(),
                n: cells.len(),
            });
        }
        if verbose {
            println!(
                "level {}: {} -> {} objects, tau {:?}",
                levels.len(),
                cells.len(),
                res.reps.len(),
                res.threshold
            );
        }

        let mut next = Vec::with_capacity(res.reps.len() * dim);
        for &r in &res.reps {
            next.extend_from_slice(&rows[r as usize * dim..(r as usize + 1) * dim]);
        }
        let reps: Vec<u32> = res.reps.iter().map(|&r| cells[r as usize]).collect();
        cells.clone_from(&reps);
        owned = Some(next);
        levels.push(MilkLevel {
            threshold: res.threshold,
            reps,
            parent: res.assignment,
        });
    }

    Ok(MilkTree {
        n_leaves: n,
        levels,
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
    use rand_distr::{Distribution, Normal};

    fn mixture(n: usize, dim: usize, k: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, 1.0).unwrap();
        let centres: Vec<f64> = (0..k * dim).map(|_| 6.0 * rng.random::<f64>()).collect();
        (0..n)
            .flat_map(|_| {
                let c = rng.random_range(0..k);
                (0..dim)
                    .map(|j| centres[c * dim + j] + noise.sample(&mut rng))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn test_milk_tree_single_root_valid_parents() {
        let (n, dim) = (800, 6);
        let data = mixture(n, dim, 5, 1);
        let tree = milk_tree(&data, n, dim, MilkDist::Euclidean, 1.0, 1, None, false).unwrap();
        let parent = tree.parent_array();
        assert_eq!(parent.len(), tree.n_nodes());
        assert_eq!(parent.iter().filter(|&&p| p == NO_NODE).count(), 1);
        for (i, &p) in parent.iter().enumerate() {
            assert!(p == NO_NODE || p as usize > i);
        }
        let v = tree.vertices();
        assert_eq!(*v.group_size.last().unwrap() as usize, n);
        assert_eq!(v.iteration.last().copied(), Some(tree.levels.len() as u32));
    }

    #[test]
    fn test_milk_tree_representative_is_a_descendant_leaf() {
        let (n, dim) = (600, 4);
        let data = mixture(n, dim, 4, 2);
        let tree = milk_tree(&data, n, dim, MilkDist::Cosine, 1.0, 1, None, false).unwrap();
        let parent = tree.parent_array();
        let v = tree.vertices();
        for node in n..tree.n_nodes() {
            let mut x = v.representative[node];
            while x != node as u32 {
                x = parent[x as usize];
                assert_ne!(
                    x, NO_NODE,
                    "node {node} does not contain its representative"
                );
            }
        }
    }

    #[test]
    fn test_milk_tree_sample_size_stops_early() {
        let (n, dim) = (500, 4);
        let data = mixture(n, dim, 3, 3);
        let tree = milk_tree(&data, n, dim, MilkDist::Euclidean, 1.0, 50, None, false).unwrap();
        let top = tree.levels.last().unwrap().reps.len();
        assert!(top <= 50);
        assert!(tree.levels.len() == 1 || tree.levels[tree.levels.len() - 2].reps.len() > 50);
        let roots = tree
            .parent_array()
            .iter()
            .filter(|&&p| p == NO_NODE)
            .count();
        assert_eq!(roots, top);
    }

    #[test]
    fn test_milk_tree_trivial_and_reproducible() {
        let one = milk_tree(
            &[1.0f64, 2.0],
            1,
            2,
            MilkDist::Euclidean,
            1.0,
            1,
            None,
            false,
        )
        .unwrap();
        assert!(one.levels.is_empty());
        assert_eq!(one.parent_array(), vec![NO_NODE]);

        let (n, dim) = (400, 3);
        let data = mixture(n, dim, 3, 4);
        let a = milk_tree(&data, n, dim, MilkDist::Correlation, 1.0, 1, None, false).unwrap();
        let b = milk_tree(&data, n, dim, MilkDist::Correlation, 1.0, 1, None, false).unwrap();
        assert_eq!(a.parent_array(), b.parent_array());
    }
}
