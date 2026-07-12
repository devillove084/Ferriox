//! Backend C: CSR/CSC Dual-Pass Atomic-Free Backward (§9.6).
//!
//! The deterministic research backend performs two ownership passes:
//!
//! 1. **Query pass**: each `(query, head_group)` owner traverses its CSR
//!    selected blocks, recomputes `p, dz`, accumulates `dQ`, processes
//!    window entries and sink, then writes `dQ` once in canonical order.
//! 2. **Compressed-KV pass**: each routing-block owner traverses its stable
//!    CSC query list, gathers query/head rows, recomputes `p, dz`, and
//!    accumulates shared `dC`.  Hot blocks are split across CTAs with
//!    numbered partials and fixed-order reduction.
//!
//! Slab-crossing `dC` contributions (§9.6) must be handled by one of:
//!   (1) full q,g residency
//!   (2) serial slab-ordered dC update
//!   (3) numbered slab partials + fixed-order reduction
//!   (4) atomics for slab-crossing dC

use cuda_core::{CudaContext, CudaStream};

use crate::config::{BlockSelection, CsaConfig, ExecutionPolicy, ReverseBlockIndex};
use crate::error::FerrioxError;
use crate::tensor::Tensor;

/// Execute the dual-pass backward.
///
/// # Pass 1 — Query owner → `dQ`
///
/// ```text
/// for each (query, head_group):
///     load D[t, h, :], LSE[t, h]
///     for each selected block slot r:
///         recompute S = Q @ C_block^T
///         compute p = softmax(S), dz = p * dO
///         accumulate dQ += dz @ C_block  (6c FLOPs per interaction)
///     process tagged window entries
///     process sink contribution
///     write dQ once in canonical source order
/// ```
///
/// # Pass 2 — Compressed-KV owner → `dC` (within slab scope)
///
/// ```text
/// for each routing_block owner:
///     load C_block[b_K, c]
///     for each query selecting this block:
///         load Q[t, a, :], dO[t, a, :]
///         recompute p, dz
///         dC += Q^T @ dz  (value) + dO^T @ p  (key)  (8c FLOPs per interaction)
///     write ordinary block once  (within slab)
///     split hot blocks into numbered chunks
/// reduce hot partials in fixed chunk order
/// reduce slab-dC partials in fixed slab order
/// ```
pub fn backward_dual_pass(
    _ctx: &CudaContext,
    _stream: &CudaStream,
    _q_core: &Tensor<f32>,
    _c_comp: &Tensor<f32>,
    _o: &Tensor<f32>,
    _dout: &Tensor<f32>,
    _lse: &Tensor<f32>,
    _selection: &BlockSelection,
    _reverse: &ReverseBlockIndex,
    _config: &CsaConfig,
    _policy: &ExecutionPolicy,
) -> Result<(Tensor<f32>, Tensor<f32>), FerrioxError> {
    // TODO: Implement dual-pass backward.
    //
    // Key design decisions:
    //   - Query pass writes dQ once (no atomics per query).
    //   - KV pass writes dC once per block within a slab, then reduces
    //     slab partials in fixed order.
    //   - Hot blocks (> 1 CTA) use numbered 64 KiB partials with fixed-order
    //     reduction.
    //   - Determinism: fixed CSR order for dQ, stable CSC order for dC,
    //     slab-ID order for slab reduction.
    //
    // Total arithmetic: ~14c N^cap ≈ 234.88 TFLOPs at reference shape
    // (1.40× fused arithmetic).  Slab-dC reduction adds overhead that
    // depends on the chosen mechanism.
    let _ = (
        _ctx, _stream, _q_core, _c_comp, _o, _dout, _lse, _selection, _reverse, _config, _policy,
    );
    todo!("Dual-pass backward not yet implemented — see docs/CODESIGN.md §9.6")
}
