//! Learnable virtual attention sink.
//!
//! Ferriox places one learnable scalar `z'` per head as a virtual "null"
//! key with value zero.  The sink participates in the same online softmax
//! as real entries and provides a stable baseline for the exponentiated
//! sum.  This follows the observation that initial tokens often act as
//! attention sinks (StreamingLLM, FSA), adapted to Ferriox's sparse
//! regime.
//!
//! In Q-outer execution the sink is initialised as `(m, ell, O_tilde) =
//! (z', 1, 0)`, which is equivalent to processing it as a virtual first
//! entry with value zero (§5.1 of CODESIGN.md).
//!
//! Note: FSA's "attention sink separate allocation" refers to the first
//! real KV block of a high-degree head, not Ferriox's virtual sink.  The
//! two are not interchangeable (see §5.4).

/// Number of sink values per head (exactly one).
pub const SINK_VALUES_PER_HEAD: u32 = 1;

/// Initialise per-head sink logits to zero.
///
/// Returns a `[H_core]` tensor of zeros.  At training time these are
/// learnable parameters updated by the sink gradient, which is
/// query-local and reduced across queries with a fixed-order head
/// reduction (§9.8).
pub fn init_sink_logits(num_heads: u32) -> Vec<f32> {
    vec![0.0_f32; num_heads as usize]
}

/// Compute the sink gradient from per-query contributions.
///
/// Each query produces a scalar sink gradient; this function sums them
/// in a fixed order for deterministic backward.
pub fn reduce_sink_grads(per_query_grads: &[f32], num_heads: u32) -> Vec<f32> {
    let num_queries = per_query_grads.len() / num_heads as usize;
    let mut grads = vec![0.0_f32; num_heads as usize];
    for q in 0..num_queries {
        for h in 0..num_heads as usize {
            grads[h] += per_query_grads[q * num_heads as usize + h];
        }
    }
    grads
}
