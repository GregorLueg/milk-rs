# Against the Julia reference

Same inputs to both, timed on one machine: ten-core M1 Max, 64 GB, Julia
1.12.7, reference at `yachielab/milk` `03bc7d1` (v1.0.0). milk-rs at 0.1.0,
2026-10-02. Julia numbers include its JIT; the single-level runs warm up first.

Data:

- **Synthetic**: 10,000 x 50, twelve Gaussian clusters (centres N(0, 16),
  unit noise), random cluster order. Ground truth known.
- **Real**: 47,805 x 50, the PCA of a spatial transcriptomics dataset, as
  shipped in its `.h5ad`.
- **Real, large**: 500,000 x 50, PCA (2,000 HVGs) of the `parse_500k` subset of
  a Parse scRNA-seq set. Rust only; see below.

## One level

The single-partition path: radius from all pairs, first sweep, medoid step,
second sweep. Synthetic data, one thread each.

| metric | percentile | Julia | Rust, all pairs | Rust, 1e6 sampled pairs |
|---|---|---|---|---|
| euclidean | 1.0 | 6.08 s | 0.52 s | 0.042 s |
| euclidean | 0.1 | 6.54 s | 0.54 s | 0.065 s |
| cosine | 1.0 | 6.13 s | 0.50 s | 0.056 s |
| cosine | 0.1 | 7.09 s | 0.52 s | 0.075 s |
| correlation | 1.0 | 11.04 s | 0.50 s | 0.058 s |
| correlation | 0.1 | 11.77 s | 0.53 s | 0.076 s |

With all pairs, Rust's time is mostly enumerating the 5e7 pairs for the
radius. Sample them, as the paper describes, and a level is 80 to 170 times
faster. On ten threads the sampled column is 0.019 to 0.048 s.

The radius from all pairs matches Julia to at least six significant digits in
all six rows (8.796314 against 8.796314220 for Euclidean at 1.0). The sampled
radius sits 0.02 to 0.3% off, which moves the group count by under 1%.

### Group counts

Bit-identical groups aren't possible: the reference sweeps a `Dict` in hash
order, so its result depends on the cell id strings. So both sides were run over
several orders. Julia with the ids renamed eight ways, Rust with the rows
shuffled thirty ways, Euclidean, all-pairs radius:

| percentile | Julia, 8 id renamings | Rust, 30 shuffles |
|---|---|---|
| 1.0 | 1,847 to 1,905 | 1,822 to 1,930 (mean 1,892) |
| 0.1 | 5,311 to 5,343 | 5,272 to 5,376 (mean 5,317) |

The spread is the same size on both sides, and the unrenamed Julia count lands
inside the Rust range for every metric and percentile above.

## Whole tree, synthetic

| | Julia | Rust, 1 thread |
|---|---|---|
| wall time | 21.6 s | 0.079 s |
| peak RSS | 4.2 GB | 43 MB |
| levels | 30 | 28 |
| ARI against the 12 clusters, at the level with 12 groups | 1.000 | 1.000 |

Both recover the planted clusters exactly, and the level with 13 groups scores
1.000 on both too. The finest level sits near zero on both (0.057 and 0.066), as
it should at a 1st-percentile radius.

Every level sweeps in input order here. An earlier build passed each level's
representatives on in creation order instead, and that compressed faster and
gave a visibly different tree (26 levels against 30). Order matters at every
level, not just the first.

## Whole tree, real data

| | Julia | Rust, 1 thread | Rust, 10 threads |
|---|---|---|---|
| wall time | 86 s | 0.37 s | 0.15 s |
| peak RSS | 5.8 GB | 58 MB | 58 MB |
| levels | 27 | 29 | 29 |
| radius at level 0 | 6.92 | 8.59 | 8.59 |

The trees differ more here, for two reasons that are both on the reference side.

**The radius depends on the file order.** Above `partition-size` (10,000 cells)
the reference splits the input into partitions by line order and takes the
*minimum* of the per-partition percentiles. This file is sorted by sample, so
each partition is more homogeneous than the whole, and the minimum lands at 6.92
against 8.59 for the full distribution. A smaller radius means more groups: 16,514
at level 0 against Rust's 10,800. Shuffle the file and the reference's radius
moves.

**One level goes missing.** The reference runs the partitioned framework until
`n` drops below `partition-size`, then switches to direct execution without
advancing the iteration counter. Both write `iteration_00000001.groups.jsonl.gz`,
so the direct run overwrites the last partitioned level. In this run the
16,514 to 1,949 merge disappears: the 458 groups at iteration 2 have all 16,514
iteration-1 groups as direct children, and the 1,949 representatives never
appear in `vertices.csv`. The tree stays connected (one root, edges = nodes - 1)
because compiled groups carry all their descendants, so nothing errors. It hits
every run larger than `partition-size`, which is every atlas-scale run.

## Large real data

Julia wasn't run at 500,000 cells; at its default `partition-size` that is 50
partitions per level.

Rust, whole tree:

| threads | seconds |
|---|---|
| 10 | 1.80 |
| 1 | 5.84 |

## Where the reference loses its time

Not because it's Julia. The data structures cost it, and they'd cost as much in
Rust.

- Every vector lives in a `Dict{String,Vector{Float32}}`, so every distance
  pays two string hashes and a pointer chase. It's also why sweep order follows
  the cell ids.
- Each partition materialises its full `n x n` distance matrix plus a vector of
  all `n(n-1)/2` id pairs, which is where the gigabytes go and why partitions
  are capped at 10,000 cells.
- Representatives go back out as CSV and get re-parsed at every level.

milk-rs keeps one flat row-major matrix with `u32` indices, never materialises
pairwise distances, samples the radius, and runs each sweep as a GEMM prefilter
against blocks of representatives followed by an exact rescoring. The result
equals the plain greedy sweep at any block size and thread count; the tests
check that against a naive implementation.

## What the paper says and the code doesn't do

- **"Maintains the order of input data throughout."** Only at partition
  granularity: within a partition, order is the hash order of the ids. milk-rs
  does keep input order.
- **"Approximated from a randomly sampled subset of objects."** The code
  enumerates every pair in sampled partition files. milk-rs samples pairs.
- **"Without explicitly computing the full pairwise distance matrix."** True
  across partitions, false inside each one. milk-rs never computes one.
