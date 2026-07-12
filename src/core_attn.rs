//! Stage B: Fixed-Block Sparse CoreAttn with online softmax.
//!
//! Given the selected routing blocks (Stage A), Stage B expands each block
//! on-chip, appends the tagged sliding-window entries and a learnable virtual
//! sink, and computes exact sparse attention with the online-softmax algorithm.
//!
//! See §5 of CODESIGN.md.

use cuda_core::{CudaContext, CudaStream};

use crate::config::{BlockSelection, CoreAttnOutput, CsaConfig};
use crate::error::FerrioxError;
use crate::tensor::Tensor;

// ── Q-Outer Backend (§5.3) ─────────────────────────────────────────────────

/// Q-outer fixed-block sparse CoreAttn forward.
///
/// Each query iterates over its selected compressed blocks, loads one
/// `b_K × c` block at a time, computes `QK^T` and `PV` using dense MMA
/// tiles, and maintains a running online-softmax state.
///
/// This is the default backend for the million-token profile because it
/// avoids the enormous 976.6 GiB KV-outer output buffer.
pub fn q_outer_core_attn(
    _ctx: &CudaContext,
    _stream: &CudaStream,
    _q_core: &Tensor<f32>,
    _c_comp: &Tensor<f32>,
    _selection: &BlockSelection,
    _sink_value: f32,
    _config: &CsaConfig,
) -> Result<CoreAttnOutput, FerrioxError> {
    // TODO: Implement Q-outer sparse CoreAttn (§5.2–§5.3).
    //
    // Algorithm (Algorithm 2 in CODESIGN.md):
    //   for each (batch, head_group):
    //       for each query t:
    //           initialise (m, ell, O_tilde) = (z', 1, 0)  // sink-start
    //           for each slot r in selected_blocks[t]:
    //               block_start = block_ids[t, r] * b_K
    //               load C_block[block_start .. block_start + b_K, :]
    //               compute QK^T and PV with C_block
    //               update (m, ell, O_tilde) via online softmax
    //           append tagged window entries
    //           finalise O = O_tilde / ell
    //           store O and LSE = m + ln(ell)
    //
    // Key invariants:
    //   - Q-outer uses one recurrence directly (§5.1, Algorithm 1).
    //   - Sink (value zero) is equivalent to initialising (z', 1, 0).
    //   - Block content is causal-partial masked before softmax.
    let _ = (
        _ctx,
        _stream,
        _q_core,
        _c_comp,
        _selection,
        _sink_value,
        _config,
    );
    todo!("Q-outer CoreAttn not yet implemented — see docs/CODESIGN.md §5.2")
}

// ── KV-Outer Backend (§5.4) ─────────────────────────────────────────────────

/// KV-outer gather-and-densify CoreAttn forward.
///
/// Each selected KV block gathers all queries that chose it, packs query/head
/// rows into full MMA shapes, and reuses the block across those rows.  Partial
/// online-softmax states are emitted per query/head and later combined.
///
/// This backend is attractive when block degree is high but is not the default
/// because the two-phase output buffer is enormous (∼977 GiB at 1M tokens).
///
/// **Base partial requirement:** each query owns a base partial initialised
/// from the tagged sliding-window entries and exactly one virtual sink (§5.1)
/// before any routing-block partial is combined.  Without this, the sink is
/// either duplicated or omitted.
pub fn kv_outer_core_attn(
    _ctx: &CudaContext,
    _stream: &CudaStream,
    _q_core: &Tensor<f32>,
    _c_comp: &Tensor<f32>,
    _reverse: &crate::config::ReverseBlockIndex,
    _sink_value: f32,
    _config: &CsaConfig,
) -> Result<CoreAttnOutput, FerrioxError> {
    // TODO: Implement KV-outer sparse CoreAttn (§5.4).
    //
    // Algorithm:
    //   for each (routing_block, query_chunk) task:
    //       load C_block once
    //       gather query/head rows selecting the block
    //       compute dense QK/PV physical tiles
    //       emit one partial online-softmax state per query/head
    //
    //   for each query:
    //       base_partial = sink + tagged_window
    //       for each routing_block partial owned by this query:
    //           base_partial = online_softmax_merge(base_partial, block_partial)
    //
    // Edge cases:
    //   - Sink allocated exactly once per query (not once per block partial).
    //   - Window entries must enter the softmax; they are not part of any
    //     routing block.
    //   - Slab/head-group buffering required for 1M-token sequences.
    let _ = (
        _ctx,
        _stream,
        _q_core,
        _c_comp,
        _reverse,
        _sink_value,
        _config,
    );
    todo!("KV-outer CoreAttn not yet implemented — see docs/CODESIGN.md §5.4")
}
