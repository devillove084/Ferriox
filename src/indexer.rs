//! Lightning Indexer — Stage A of the Ferriox pipeline.
//!
//! The indexer computes per-query block scores (head-reduced, ReLU-gated)
//! and selects the top-κ routing blocks.  It must operate within a bounded
//! memory window to avoid materialising the full `[S, H_I, T]` tensor.
//!
//! See §4 of CODESIGN.md for the complete specification.

use cuda_core::{CudaContext, CudaStream};

use crate::config::{BlockSelection, CsaConfig};
use crate::error::FerrioxError;
use crate::tensor::Tensor;

// ── Public API ──────────────────────────────────────────────────────────────

/// Select the top-κ routing blocks for each query.
///
/// This is the entry point for Stage A.  It reads indexer query/key tensors,
/// scores every legal (query, compressed-entry) pair, max-pools into routing
/// blocks, and selects the top `κ` block IDs.
///
/// # Arguments
/// * `q_idx` — indexer queries, shape `[B, S, H_I, c_I]`.
/// * `k_comp` — compressed indexer keys, shape `[R, H_I, c_I]` where
///   `R` is the number of compressed entries in the sequence.
/// * `w_idx` — indexer head weights, shape `[B, S, H_I]`.
///
/// Returns a `BlockSelection` with shape `[B, S, κ]`.
pub fn select_blocks(
    _ctx: &CudaContext,
    _stream: &CudaStream,
    _q_idx: &Tensor<f32>,
    _k_comp: &Tensor<f32>,
    _w_idx: &Tensor<f32>,
    _config: &CsaConfig,
) -> Result<BlockSelection, FerrioxError> {
    // TODO: Implement the streaming block-indexer kernel (§4.2–§4.4).
    //
    // Algorithm sketch:
    //   1. For each query chunk (bounded to fit HBM):
    //      a. Load chunk of q_idx, w_idx into registers/shared memory.
    //      b. Stream through K^IComp chunks.
    //      c. For each (query, entry): compute head-reduced ReLU score.
    //      d. Max-pool entry scores into routing-block scores.
    //      e. Maintain top-κ block IDs and scores in registers.
    //   2. Write final [B, S, κ] block IDs.
    //
    // Variants:
    //   A. Fused score + local selection (write block scores, then select).
    //   B. Persistent query microtile (keep query in rf/shared).
    //   C. Hierarchical partition-merge (tree reduction for large S).
    //
    // Edge conditions (§4.5):
    //   - T_legal(t) = floor((t+1)/m): first m-1 queries see 0 legal entries.
    //   - T_legal(t) = 0 → all-`-1` (u32::MAX) block-ID row.
    //   - Packed sequence boundaries reset compression and window state.
    //   - Entry ties use global compressed-entry ID; block ties use global block ID.

    let _ = (_ctx, _stream, _q_idx, _k_comp, _w_idx, _config);
    todo!("Stage A indexer not yet implemented — see docs/CODESIGN.md §4")
}

/// Return the legal number of compressed entries for query position `t`.
///
/// This implements the zero-based convention: `T_legal(t) = floor((t + 1) / m)`.
/// For `t < m-1`, returns 0 (empty legal domain).
///
/// Note: the V4 paper writes `s < floor(t/m)` with one-based `t`; the two
/// forms are believed equivalent after adjusting the position base, but
/// equivalence should be confirmed against the published V4 kernel (§2.4,
/// §6.7).
#[inline]
pub fn legal_domain_size(t: u32, compression_stride: u32) -> u32 {
    (t + 1) / compression_stride
}

/// Return the effective V4 `k` budget at position `t`.
#[inline]
pub fn effective_k(t: u32, compression_stride: u32, k_v4: u32) -> u32 {
    k_v4.min(legal_domain_size(t, compression_stride))
}
