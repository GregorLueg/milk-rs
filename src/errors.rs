//! Errors for the milk-rs crate.

use thiserror::Error;

/// Errors that can be returned by milk-rs
#[derive(Debug, Error)]
pub enum MilkErrors {
    // -- input validation --
    /// No objects were supplied
    #[error("The input contains no objects.")]
    EmptyInput,

    /// The flat data buffer does not match the stated shape
    #[error("Data has {len} values, expected n ({n}) x dim ({dim}).")]
    ShapeMismatch {
        /// Length of the supplied buffer
        len: usize,
        /// Stated number of objects
        n: usize,
        /// Stated dimensionality
        dim: usize,
    },

    /// A parameter is outside its valid range
    #[error("Invalid parameter '{param}': {reason}")]
    InvalidParam {
        /// Name of the parameter
        param: &'static str,
        /// Why the value was rejected
        reason: String,
    },

    // -- recursion --
    /// A level produced as many groups as it had objects
    #[error(
        "Level {level} merged none of its {n} objects; the radius is below every pairwise distance."
    )]
    NoProgress {
        /// Zero-based level index
        level: usize,
        /// Objects at that level
        n: usize,
    },

    // -- metrics --
    /// The requested distance metric is not implemented
    #[error("Distance metric '{0}' is not supported. Use euclidean, cosine or correlation.")]
    DistanceNotSupported(String),
}
