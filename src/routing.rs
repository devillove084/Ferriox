//! Block-routing logic and reverse view construction.
//!
//! Implements the routing contract from §2.5 of CODESIGN.md:
//! entry scoring, block max-pool, deterministic tie-break, and
//! construction of the CSR → CSC reverse view for KV-outer and backward.

use crate::config::{BlockSelection, ReverseBlockIndex};
use crate::error::FerrioxError;

/// Construct the reverse (CSC) block view from a forward selection.
///
/// Given the `[B, S, κ]` block IDs, build a per-block query list:
///   - `block_offsets[r]`  → start index in `query_handles`
///   - `block_degrees[r]`  → number of queries selecting block `r`
///   - `query_handles`     → packed `(query_id, slot)` pairs
///   - `block_count`       → number of blocks with ≥1 selector
///
/// The output is used by KV-outer forward and CSC backward passes.
/// Order is fixed for determinism: `(block_id, query_id, slot)`.
pub fn build_reverse_view(
    selection: &BlockSelection,
    num_routing_blocks: u32,
) -> Result<ReverseBlockIndex, FerrioxError> {
    let total_queries = selection.batch * selection.seq_len;
    let kappa = selection.kappa() as usize;
    let num_blocks = num_routing_blocks as usize;

    // First pass: count degree per block.
    let mut degrees = vec![0u32; num_blocks];
    for q in 0..total_queries as usize {
        for slot in 0..kappa {
            let bid = selection.block_ids[q * kappa + slot] as usize;
            if bid != u32::MAX as usize {
                degrees[bid] += 1;
            }
        }
    }

    // Prefix sum → offsets.
    let mut offsets = vec![0u32; num_blocks + 1];
    for r in 0..num_blocks {
        offsets[r + 1] = offsets[r] + degrees[r];
    }
    let total_edges = offsets[num_blocks] as usize;

    // Second pass: fill handles in stable (block, query, slot) order.
    let mut handles = vec![0u32; total_edges];
    let mut cursors = offsets[..num_blocks].to_vec();
    for q in 0..total_queries as usize {
        for slot in 0..kappa {
            let bid = selection.block_ids[q * kappa + slot] as usize;
            if bid != u32::MAX as usize {
                let pos = cursors[bid] as usize;
                handles[pos] = ((q as u32) << 16) | (slot as u32);
                cursors[bid] += 1;
            }
        }
    }

    Ok(ReverseBlockIndex {
        block_offsets: offsets,
        query_handles: handles,
        block_degrees: degrees,
        block_count: num_blocks as u32,
    })
}

/// Deterministic block entry-score comparator.
///
/// Ferriox uses the StreamIndex tie-break: smaller entry index wins a tie.
/// Legal entries only; masked/sentinel entries are excluded before comparison.
/// A NaN score is converted to `-infinity` and remains in the legal domain.
///
/// **Open contract:** the V4 paper has not published its exact selector order
/// or tie-break.  In FP4, ReLU-masked, and high-zero-score regimes, ties are
/// not theoretical; a different tie-break produces a different selected set
/// (§6.7).
#[derive(Debug, Clone, Copy)]
pub struct EntryComparator;

impl EntryComparator {
    /// Compare two `(score, global_entry_index)` pairs.
    /// Returns `true` if `a` outranks `b`.
    #[inline]
    pub fn gt(a_score: f32, a_idx: u32, b_score: f32, b_idx: u32) -> bool {
        a_score > b_score || (a_score == b_score && a_idx < b_idx)
    }
}

/// Convert a compressed-entry index to its routing-block index.
#[inline]
pub fn entry_to_block(entry_idx: u32, b_k: u32) -> u32 {
    entry_idx / b_k
}

/// Convert a routing-block index to the start entry index.
#[inline]
pub fn block_to_entry_start(block_idx: u32, b_k: u32) -> u32 {
    block_idx * b_k
}
