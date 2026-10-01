//! Full MILK run on a MILK-format CSV (id column, no header). Writes one row per
//! node: `node,parent,representative,group_size,iteration,threshold`.
//!
//! `cargo run --release --example tree -- <csv> <out.csv> [metric] [percentile]`

use milk_rs::prelude::*;
use std::io::Write;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, out) = (&args[1], &args[2]);
    let metric = parse_milk_dist(args.get(3).map_or("euclidean", |s| s.as_str())).unwrap();
    let percentile: f64 = args.get(4).map_or(1.0, |s| s.parse().expect("percentile"));

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

    let start = Instant::now();
    let tree = milk_tree(&data, n, dim, metric, percentile, 1, None, false).unwrap();
    println!(
        "{n} x {dim}: {} levels, {} nodes, {:.3}s",
        tree.levels.len(),
        tree.n_nodes(),
        start.elapsed().as_secs_f64()
    );

    let parent = tree.parent_array();
    let v = tree.vertices();
    let mut w = std::io::BufWriter::new(std::fs::File::create(out).expect("create"));
    writeln!(
        w,
        "node,parent,representative,group_size,iteration,threshold"
    )
    .unwrap();
    for (i, &par) in parent.iter().enumerate() {
        let p = if par == NO_NODE { -1 } else { par as i64 };
        writeln!(
            w,
            "{i},{p},{},{},{},{}",
            v.representative[i], v.group_size[i], v.iteration[i], v.threshold[i]
        )
        .unwrap();
    }
}
