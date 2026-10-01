//! Kill-gate spike: one MILK level on a MILK-format CSV (id column, no header).
//!
//! `cargo run --release --example spike -- <csv> [n_shuffles]`
//!
//! Prints the radius, group counts and stage timings for exact and sampled
//! radii, then group counts over shuffled input orders to compare against the
//! reference, whose sweep order is a hash of the cell ids.

use milk_rs::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: spike <csv> [n_shuffles]");
    let n_shuffles: usize = args.get(2).map_or(5, |s| s.parse().expect("n_shuffles"));

    let text = std::fs::read_to_string(path).expect("read csv");
    let mut data: Vec<f32> = Vec::new();
    let mut n = 0;
    for line in text.lines().filter(|l| !l.is_empty()) {
        data.extend(
            line.split(',')
                .skip(1)
                .map(|v| v.parse::<f32>().expect("float")),
        );
        n += 1;
    }
    let dim = data.len() / n;
    println!(
        "{n} x {dim}, {} rayon threads",
        rayon::current_num_threads()
    );

    for metric in ["euclidean", "cosine", "correlation"] {
        let dist = parse_milk_dist(metric).unwrap();
        for p in [1.0, 0.1] {
            for (label, params) in [
                ("exact", LevelParams::new(None, 42)),
                ("sampled", LevelParams::default()),
            ] {
                print!("{metric} p={p} {label}: ");
                let start = Instant::now();
                let res = milk_level(&data, n, dim, dist, p, Some(params), true).unwrap();
                println!(
                    "    tau {:.6} | G {} -> {} | total {:.3}s",
                    res.threshold,
                    res.n_groups_pass1,
                    res.reps.len(),
                    start.elapsed().as_secs_f64()
                );
            }

            let mut g = Vec::new();
            for seed in 0..n_shuffles as u64 {
                let mut perm: Vec<usize> = (0..n).collect();
                perm.shuffle(&mut StdRng::seed_from_u64(seed));
                let shuffled: Vec<f32> = perm
                    .iter()
                    .flat_map(|&i| data[i * dim..(i + 1) * dim].iter().copied())
                    .collect();
                let res = milk_level(
                    &shuffled,
                    n,
                    dim,
                    dist,
                    p,
                    Some(LevelParams::new(None, 42)),
                    false,
                )
                .unwrap();
                g.push((res.n_groups_pass1, res.reps.len()));
            }
            println!("    shuffled orders (G pass 1, G pass 2): {g:?}");
        }
    }
}
