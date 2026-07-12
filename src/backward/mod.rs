//! Backward module — gradient computation for CSA.
//!
//! Ferriox supports three backward backends (§9.4–§9.6):
//!
//! - **A: KV-Major Atomic** (FlashMoBA-style baseline)
//! - **B: FSA-Style Partial `dQ`** (experimental slab-local)
//! - **C: CSR/CSC Dual-Pass** (default deterministic)

pub mod dual_pass;
pub mod kv_major;

use cuda_core::{CudaContext, CudaStream};

use crate::config::{BackwardBackend, CsaConfig, ExecutionPolicy};
use crate::error::FerrioxError;
use crate::tensor::Tensor;

/// Run the backward pass using the configured backend.
///
/// Computes `dQ`, `dC` (compressed KV gradients), and optionally sink
/// and window gradients from the saved forward-pass state.
///
/// See §9 of CODESIGN.md for the complete specification.
pub fn csa_backward(
    ctx: &CudaContext,
    stream: &CudaStream,
    q_core: &Tensor<f32>,
    c_comp: &Tensor<f32>,
    o: &Tensor<f32>,
    dout: &Tensor<f32>,
    lse: &Tensor<f32>,
    selection: &crate::config::BlockSelection,
    reverse: &crate::config::ReverseBlockIndex,
    config: &CsaConfig,
    policy: &ExecutionPolicy,
) -> Result<(Tensor<f32>, Tensor<f32>), FerrioxError> {
    match policy.backward {
        BackwardBackend::KvMajorAtomicDq => {
            kv_major::backward_kv_major(ctx, stream, q_core, c_comp, o, dout, lse, reverse, config)
        }
        BackwardBackend::BufferedPartialDq => {
            // FSA-style: slab-local partial dQ, then reduction.
            todo!("FSA-style partial dQ backward not yet implemented")
        }
        BackwardBackend::DualPassAtomicFree => {
            dual_pass::backward_dual_pass(
                ctx, stream, q_core, c_comp, o, dout, lse, selection, reverse, config, policy,
            )
        }
    }
}
