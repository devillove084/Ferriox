//! Ferriox — Block-Routed Memory-Bounded Compressed Sparse Attention in Pure Rust.
//!
//! # Architecture
//!
//! Ferriox implements DeepSeek-V4-style Compressed Sparse Attention (CSA) with a
//! novel block-routing abstraction. The design separates two block scales:
//!
//! 1. **V4 compression stride** `m = 4` — one compressed entry per four new
//!    source positions.
//! 2. **Ferriox routing stride** `b_K = 32` — one routing block per 32
//!    consecutive compressed entries (128 source positions).
//!
//! The primary profile selects `κ = 16` routing blocks per query, yielding
//! at most `κ·b_K = 512` compressed entries — the same maximum attention budget
//! as V4, but with block-regular selection and execution.
//!
//! # Stages
//!
//! - **Stage A** (Lightning Indexer): streaming block-level max-pool scoring
//!   and partition-merge top-κ selection.  Memory-bounded: never materialises
//!   the full `[S, H_I, T]` score tensor.
//! - **Stage B** (CoreAttn): fixed-block sparse attention over the selected
//!   compressed entries plus a tagged sliding window and a learnable virtual
//!   sink.  Supports Q-outer (default) and KV-outer (optional) backends.
//! - **Backward**: dual-pass atomic-free design with CSR query-owner `dQ`
//!   and CSC compressed-KV-owner `dC` passes.
//!
//! # Quick start
//!
//! ```no_run
//! use ferriox::{CsaConfig, select_blocks, core_attn_fwd};
//! use cuda_core::CudaContext;
//!
//! let ctx = CudaContext::new(0).unwrap();
//! let stream = ctx.default_stream();
//!
//! let config = CsaConfig {
//!     model_dim: 7168,
//!     query_latent_dim: 1536,
//!     core_dim: 512,
//!     index_dim: 128,
//!     core_heads: 64,
//!     index_heads: 128,
//!     compression: 4,
//!     routing_block_entries: 32,
//!     selected_blocks: 16,
//!     window: 512,
//!     output_groups: 1,
//!     group_intermediate_dim: 0,
//!     sm_scale: 1.0 / (512.0_f32).sqrt(),
//! };
//!
//! // let selection = select_blocks(&ctx, &stream, &q_idx, &k_comp, &w_idx, &config)?;
//! // let output = core_attn_fwd(&ctx, &stream, &q_core, &c_comp, &selection, &config)?;
//! ```

pub mod config;
pub mod error;
pub mod tensor;

pub mod backward;
pub mod compression;
pub mod core_attn;
pub mod device;
pub mod forward;
pub mod indexer;
pub mod routing;
pub mod scheduler;
pub mod sink;
pub mod window;
pub mod workspace;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use config::{
    BackwardBackend, BlockSelection, CoreAttnOutput, CsaConfig, ExecutionPolicy, ReverseBlockIndex,
};
pub use error::FerrioxError;
pub use forward::{build_reverse_index, core_attn_fwd, select_blocks};
pub use tensor::Tensor;

/// Convenience re-export: select blocks then run CoreAttn in one call.
///
/// This is the primary public entry point for inference-style forward passes.
pub fn csa_forward(
    _ctx: &cuda_core::CudaContext,
    _stream: &cuda_core::CudaStream,
    _q_idx: &Tensor<f32>,
    _k_comp: &Tensor<f32>,
    _w_idx: &Tensor<f32>,
    _q_core: &Tensor<f32>,
    _c_comp: &Tensor<f32>,
    _config: &CsaConfig,
) -> Result<CoreAttnOutput, FerrioxError> {
    todo!("CSA forward not yet implemented — see docs/CODESIGN.md")
}
