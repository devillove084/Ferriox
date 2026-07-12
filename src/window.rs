//! Sliding-window attention.
//!
//! The tagged sliding window provides `w` tokens of dense attention in
//! addition to the selected compressed entries.  Window and compressed
//! entries share one forward softmax and therefore share the same `LSE`
//! and `D` (see §9.8 of CODESIGN.md).
//!
//! Window entries are never deduplicated against compressed entries: if
//! a source token appears in both paths, it contributes twice to the
//! attention output (once through compression, once directly).

use cuda_core::{CudaContext, CudaStream};

use crate::config::CsaConfig;
use crate::error::FerrioxError;
use crate::tensor::Tensor;

/// Extract the tagged sliding-window KV for a given query position.
///
/// Returns keys and values for tokens in `[t - w, t)` (causal window).
/// At sequence boundaries or packed-sample crossings, the window is
/// truncated.
pub fn window_kv(
    _ctx: &CudaContext,
    _stream: &CudaStream,
    _kv_full: &Tensor<f32>,
    _query_pos: u32,
    _config: &CsaConfig,
) -> Result<(Tensor<f32>, Tensor<f32>), FerrioxError> {
    let _ = (_ctx, _stream, _kv_full, _query_pos, _config);
    todo!("Sliding-window KV extraction not yet implemented")
}

/// Compute gradients for window KV entries.
///
/// This runs as a separate banded key-owner kernel (§9.8).
pub fn window_backward(
    _ctx: &CudaContext,
    _stream: &CudaStream,
    _q: &Tensor<f32>,
    _k_window: &Tensor<f32>,
    _v_window: &Tensor<f32>,
    _dout: &Tensor<f32>,
    _lse: &Tensor<f32>,
    _config: &CsaConfig,
) -> Result<(Tensor<f32>, Tensor<f32>), FerrioxError> {
    let _ = (_ctx, _stream, _q, _k_window, _v_window, _dout, _lse, _config);
    todo!("Sliding-window backward not yet implemented")
}
