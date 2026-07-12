//! Backend A: KV-Major with Atomic `dQ` (§9.4).
//!
//! FlashMoBA-style performance baseline.  Each CSC task owns one routing
//! block, gathers queries that selected it, densifies them on-chip, and
//! writes its `dC`.  Per-query `dQ` contributions are accumulated via
//! global FP32 atomics.
//!
//! This backend computes each interaction exactly once (no recomputation)
//! but is nondeterministic due to floating-point atomic arrival order.
//! It also requires query/head slabs because the full 122 GiB `dQaccum`
//! does not fit in L2.

use cuda_core::{CudaContext, CudaStream};

use crate::config::{CsaConfig, ReverseBlockIndex};
use crate::error::FerrioxError;
use crate::tensor::Tensor;

/// Execute the KV-major atomic backward pass.
///
/// # Arguments
/// * `q_core` — core queries, shape `[B, S, H_core, c]`.
/// * `c_comp` — compressed KV, shape `[T_b, b_K, c]`.
/// * `o` — forward output (needed for recompute if not saved).
/// * `dout` — gradient w.r.t. output.
/// * `lse` — log-sum-exp from forward.
/// * `reverse` — CSC view of the selection.
///
/// Returns `(dQ, dC)`.
pub fn backward_kv_major(
    _ctx: &CudaContext,
    _stream: &CudaStream,
    _q_core: &Tensor<f32>,
    _c_comp: &Tensor<f32>,
    _o: &Tensor<f32>,
    _dout: &Tensor<f32>,
    _lse: &Tensor<f32>,
    _reverse: &ReverseBlockIndex,
    _config: &CsaConfig,
) -> Result<(Tensor<f32>, Tensor<f32>), FerrioxError> {
    // TODO: Implement KV-major atomic backward.
    //
    // Algorithm:
    //   for each routing_block owner:
    //       load C_block[b_K, c] once
    //       for each (query, head) selecting this block:
    //           load Q[t, a, :], dO[t, a, :]
    //           recompute p, dz
    //           dC += Q^T @ dz (value) + dO^T @ p (key)
    //           atomically add beta * dz @ C_block^T to dQaccum[t, a, :]
    //   post-kernel: cast dQaccum (FP32) to dQ (BF16/FP32)
    //
    // Upper-bound atomics: E_b * H * c ≈ 524.288 billion FP32 adds.
    // Address traffic: ~3.815 TiB before cache effects.
    // Requires query/head slabs because dQaccum (122 GiB) > GPU L2.
    let _ = (
        _ctx, _stream, _q_core, _c_comp, _o, _dout, _lse, _reverse, _config,
    );
    todo!("KV-major atomic backward not yet implemented — see docs/CODESIGN.md §9.4")
}
