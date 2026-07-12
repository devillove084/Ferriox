//! Forward-pass orchestration.
//!
//! Host-side entry points for the full two-stage CSA forward pipeline:
//! Stage A (indexer) → Stage B (CoreAttn).

use cuda_core::{CudaContext, CudaStream};

use crate::config::{BlockSelection, CoreAttnOutput, CsaConfig};
use crate::error::FerrioxError;
use crate::tensor::Tensor;
use crate::{core_attn, routing};

/// Run Stage A: select the top-κ routing blocks for every query.
///
/// See [`crate::indexer::select_blocks`] for the full specification.
pub use crate::indexer::select_blocks;

/// Run Stage B: compute sparse CoreAttn over the selected blocks.
///
/// Uses the Q-outer backend by default.  For KV-outer, construct a
/// `ReverseBlockIndex` first with [`crate::routing::build_reverse_view`].
pub fn core_attn_fwd(
    ctx: &CudaContext,
    stream: &CudaStream,
    q_core: &Tensor<f32>,
    c_comp: &Tensor<f32>,
    selection: &BlockSelection,
    config: &CsaConfig,
) -> Result<CoreAttnOutput, FerrioxError> {
    let sink_value = 0.0_f32; // learnable virtual sink, zero-initialised
    core_attn::q_outer_core_attn(ctx, stream, q_core, c_comp, selection, sink_value, config)
}

/// Build the reverse block index for KV-outer execution and backward.
///
/// Wraps [`routing::build_reverse_view`].
pub fn build_reverse_index(
    selection: &BlockSelection,
    num_routing_blocks: u32,
) -> Result<crate::config::ReverseBlockIndex, FerrioxError> {
    routing::build_reverse_view(selection, num_routing_blocks)
}
