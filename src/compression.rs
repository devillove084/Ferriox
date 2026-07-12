//! KV Compression — channel-wise weighted softmax over `2m` source tokens.
//!
//! See §2.3 of CODESIGN.md.  The compressor maps `2m` consecutive source
//! tokens into one compressed entry via a learned weighted softmax.  Each
//! channel `i` computes:
//!
//! ```text
//! C^Comp_i = sum_{j=0}^{2m-1} s_{i,j} * x_{i,j}
//! ```
//!
//! where `s` is the per-channel softmax over the `2m` legal positions.
//! The overlapped arrangement (`C^a` and `C^b`) ensures every source token
//! contributes to exactly two compressed entries (except sequence boundaries).

use cuda_core::{CudaContext, CudaStream};

use crate::config::CsaConfig;
use crate::error::FerrioxError;
use crate::tensor::Tensor;

/// Forward compression: produce `C^Comp` from source KV.
///
/// Operates offline (pre-processing) or intra-layer.  The output has shape
/// `[T_b, c]` where `T_b = floor(L/m)` is the number of complete compressed
/// entries.
pub fn compress_kv(
    _ctx: &CudaContext,
    _stream: &CudaStream,
    _kv_source: &Tensor<f32>,
    _config: &CsaConfig,
) -> Result<Tensor<f32>, FerrioxError> {
    // TODO: Implement per-channel weighted softmax compression.
    //
    // Algorithm:
    //   for each channel i:
    //       for each compressed entry t:
    //           logits = linear(kv_slice[t*m : t*m + 2m, i])
    //           s = softmax(logits) over the 2m positions
    //           C^Comp[t, i] = sum_j s_j * kv_slice[t*m + j, i]
    //
    // Edge conditions:
    //   - Partial tails (< 2m) mask extra positions.
    //   - Packed sequence boundaries reset the window.
    //   - Overlapped C^a/C^b paths handled by separate compression calls.
    let _ = (_ctx, _stream, _kv_source, _config);
    todo!("KV compression not yet implemented — see docs/CODESIGN.md §2.3")
}

/// Backward compression adjoint.
///
/// Given upstream `dC`, propagate gradients to the source KV tokens:
///
/// ```text
/// dx_{i,j}     += s_{i,j} * u_i
/// dlogit_{i,j}  = s_{i,j} * u_i * (x_{i,j} - C_i)
/// ```
pub fn compress_kv_backward(
    _ctx: &CudaContext,
    _stream: &CudaStream,
    _kv_source: &Tensor<f32>,
    _c_comp: &Tensor<f32>,
    _dc: &Tensor<f32>,
    _config: &CsaConfig,
) -> Result<Tensor<f32>, FerrioxError> {
    let _ = (_ctx, _stream, _kv_source, _c_comp, _dc, _config);
    todo!("KV compression backward not yet implemented")
}
