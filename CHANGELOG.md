# Changelog

## 0.1.0

First release.

**Features**

- `milk_tree`: the full recursion, from a dense row-major matrix to a tree.
  Recomputes the radius at every level, stops at `sample_size` objects.
- `MilkTree::parent_array` and `MilkTree::vertices`: the tree as one parent id
  per node, plus representative cell, leaf count, iteration and radius per node.
  Single-child nodes are kept.
- `milk_level`: one level on its own.
- Euclidean, cosine and correlation distances, generic over `f32` and `f64`.
- `accelerate` feature, on by default: the sweep's GEMM goes through Apple
  Accelerate on macOS. A no-op elsewhere.

**Deliberate divergences from the Julia reference**

- Every level sweeps in input order, not in the hash order of the cell ids.
- The radius is a percentile of 1e6 sampled pairs by default, not of all pairs.
- No partitioning and merging.
- `spread` and `specificity` aren't computed.
