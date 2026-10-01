//! Float trait shared across the crate.

use ann_search_rs::prelude::SimdDistance;
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;
use std::iter::Sum;

/// Core float trait for MILK
pub trait MilkFloat:
    Float
    + FromPrimitive
    + ToPrimitive
    + Send
    + Sync
    + Debug
    + Sum
    + Default
    + SimdDistance
    + ComplexField
    + 'static
{
}

impl<T> MilkFloat for T where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + Debug
        + Sum
        + Default
        + SimdDistance
        + ComplexField
        + 'static
{
}
