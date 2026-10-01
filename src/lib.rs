//! MILK - recursive greedy radius-cover clustering into a tree
//!
//! Rust port of MILK (Kiyota, Lee, Yao and Yachie, bioRxiv 2026). Each level
//! picks a radius as a low percentile of the pairwise distance distribution,
//! covers the objects greedily with representatives, moves each representative
//! to the group member nearest the group centroid and re-sweeps. Recursing on
//! the representatives yields the tree.
//!
//! Deliberate divergences from the Julia reference: the sweep runs in true
//! input order (the reference walks a `Dict` in hash order, so its result
//! depends on the cell id strings), and the radius is estimated from sampled
//! pairs rather than all pairs. Results are therefore not bit-identical to the
//! reference. Original code: [milk](https://github.com/yachielab/milk)

#![warn(missing_docs)]

pub mod errors;
pub mod level;
pub mod prelude;
pub mod utils;

/// Crate version, so a dependent can report the numerics version it built
/// against.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
