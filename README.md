[![CI](https://github.com/GregorLueg/milk-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/milk-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/milk-rs.svg)](https://crates.io/crates/milk-rs)
[![docs.rs](https://img.shields.io/docsrs/milk-rs)](https://docs.rs/milk-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# milk-rs

Recursive greedy radius-cover clustering into a tree. A Rust port of MILK:
Kiyota, Lee, Yao and Yachie, *Global tree encoding of atlas-scale single-cell
genomics*, bioRxiv 2026,
[doi 10.64898/2026.08.31.747971](https://doi.org/10.64898/2026.08.31.747971).
Original Julia code: [yachielab/milk](https://github.com/yachielab/milk).

## What it does

Not an embedding. No coordinates, no loss, no kNN graph. Each level:

1. Pick a radius `tau`, a low percentile of the pairwise distances.
2. Sweep the objects in order. Each one joins the nearest representative within
   `tau`, or becomes a new one.
3. Move every representative to the group member nearest the group centroid.
4. Sweep again with those representatives pinned.

Then recurse on the representatives until one is left. You get every
resolution in one pass, with no resolution parameter to sweep, and the
representatives are real cells, so labels transfer without imputation.

It's in the CD-HIT / UCLUST family of greedy incremental clustering, lifted from
sequences to vectors and made recursive. Compare it to that family, not to UMAP.

## Usage

```rust
use milk_rs::prelude::*;

// data is row-major [cell][feature], e.g. 50 PCs
let tree = milk_tree::<f32>(&data, n_cells, n_dims, MilkDist::Euclidean, 1.0, 1, None, false)?;

let parent = tree.parent_array(); // one entry per node, NO_NODE for the root
let v = tree.vertices();          // representative cell, leaf count, iteration, radius
```

Leaves are nodes `0..n_cells` in input order. Each level's groups follow in one
contiguous block, so every parent id is larger than its children's.
Single-child nodes are kept: the tree is deep and thin by construction, and what
to collapse is the consumer's call.

`percentile` is in percent, as in the reference (`-t`, default 1.0).
`sample_size` stops the recursion once that many objects or fewer remain; 1 gives
a single root. `LevelParams` sets how many pairs are sampled for the radius
(1e6 by default, `None` for all of them) and the seed. Only need one level?
`milk_level` returns its radius, representatives and assignment.

Metrics: Euclidean, cosine and correlation. The reference also offers
Manhattan, Hamming and Jaccard; those return `DistanceNotSupported`.

## Where it departs from the reference

All deliberate, all measured in [comparison](docs/COMPARISON.md).

- **The sweep runs in input order.** The paper says MILK keeps input order, and
  its advice on ordering cells by metadata relies on that. The Julia code sweeps
  a `Dict` in hash order, so the result depends on the cell id strings: rename
  the cells and the fine levels change. Here every level sweeps in input order.
- **The radius comes from sampled pairs**, as the paper describes. The code
  enumerates every pair in each partition. With 1e6 pairs the group count moves
  by under 1%.
- **No partitions yet.** The reference splits inputs above 10k cells into
  partitions and merges them, because it materialises an `n x n` distance
  matrix per partition. Nothing here is quadratic in memory, so one sweep covers
  all cells and the partition boundary heuristics go away. The catch: level 0
  costs `n x G` distances, which grows quickly past a few million cells.
- **`spread` and `specificity` aren't computed.** In the reference they were
  87% of the distance comparisons at 30k cells, for two diagnostic columns.

## Performance

Whole tree, ten-core M1 Max, 2026-10-02:

| data | cells | threads | seconds |
|---|---|---|---|
| Parse scRNA-seq, 50 PCs | 500,000 | 10 | 1.80 |
| Parse scRNA-seq, 50 PCs | 500,000 | 1 | 5.84 |
| Gaussian mixture, 50 dims | 1,000,000 | 10 | 4.36 |

The sweep is a GEMM prefilter followed by exact rescoring, so results are
identical to the plain greedy definition at any thread count. Under the default
`accelerate` feature, macOS runs the GEMM through Apple Accelerate, which is
about 2x faster than faer on Apple silicon. Elsewhere the feature does nothing
and faer does the work.

No GPU path. The sweep runs at Accelerate's measured ceiling for this shape,
about 1.4 TFLOP/s, and the tuned exhaustive GPU kernel in `ann-search-rs` reaches
about a third of that here. A discrete GPU might change the answer; that hasn't
been measured.

[Comparison](docs/COMPARISON.md) has the numbers against the Julia reference.

## Licence

MIT. See `LICENSE`. MILK itself is MIT (Yachielab 2025); its notice is in
`LICENSE-THIRD-PARTY`.

## Citing

Cite the paper. This is an independent port and claims none of the science.
