//! Timing harness on an in-process Gaussian mixture.
//!
//! `cargo run --release --example bench -- <n> [dim] [k] [percentile] [verbose]`

use milk_rs::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: f64| args.get(i).map_or(d, |s| s.parse().unwrap());
    let n = arg(1, 10_000.0) as usize;
    let dim = arg(2, 50.0) as usize;
    let k = arg(3, 12.0) as usize;
    let percentile = arg(4, 1.0);
    let verbose = arg(5, 0.0) > 0.0;

    let mut rng = StdRng::seed_from_u64(123);
    let noise = Normal::new(0.0f32, 1.0).unwrap();
    let centres: Vec<f32> = (0..k * dim).map(|_| 4.0 * noise.sample(&mut rng)).collect();
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = rng.random_range(0..k);
        data.extend((0..dim).map(|j| centres[c * dim + j] + noise.sample(&mut rng)));
    }

    let start = Instant::now();
    let tree = milk_tree(
        &data,
        n,
        dim,
        MilkDist::Euclidean,
        percentile,
        1,
        None,
        verbose,
    )
    .unwrap();
    println!(
        "n {n} dim {dim} threads {}: {} levels, G0 {}, {:.3}s",
        rayon::current_num_threads(),
        tree.levels.len(),
        tree.levels[0].reps.len(),
        start.elapsed().as_secs_f64()
    );
}
