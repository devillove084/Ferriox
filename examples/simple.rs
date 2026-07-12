//! CSA end-to-end example.
//!
//! Build with: `cargo build --example simple`
//!
//! Note: this example is currently a skeleton.  It will run (and fail with
//! `todo!()`) until Stage A and Stage B kernels are implemented.

use cuda_core::{CudaContext, DeviceBuffer};
use ferriox::{CsaConfig, Tensor, build_reverse_index, core_attn_fwd, select_blocks};

fn main() {
    let ctx = CudaContext::new(0).expect("no CUDA device found");
    let stream = ctx.default_stream();

    // V4-Flash dimensions
    let config = CsaConfig::v4_flash();

    // Toy problem: tiny batch/seq for smoke testing
    let batch = 1u32;
    let seq_len = 16u32;
    let h_i = config.index_heads;
    let c_i = config.index_dim;
    let h_c = config.core_heads;
    let c = config.core_dim;
    let m = config.compression;
    let t_b = seq_len / m; // compressed entries

    // Allocate indexer tensors
    let n_idx = (batch * seq_len * h_i * c_i) as usize;
    let n_key = (t_b * h_i * c_i) as usize;
    let n_weights = (batch * seq_len * h_i) as usize;

    let q_idx = DeviceBuffer::<f32>::zeroed(&stream, n_idx).unwrap();
    let k_comp = DeviceBuffer::<f32>::zeroed(&stream, n_key).unwrap();
    let w_idx = DeviceBuffer::<f32>::zeroed(&stream, n_weights).unwrap();

    let q_idx_t = Tensor::new(q_idx, vec![batch, seq_len, h_i, c_i]);
    let k_comp_t = Tensor::new(k_comp, vec![t_b, h_i, c_i]);
    let w_idx_t = Tensor::new(w_idx, vec![batch, seq_len, h_i]);

    // --- Stage A: select routing blocks ---
    println!(
        "Stage A: selecting top-{} routing blocks...",
        config.selected_blocks
    );
    let selection = select_blocks(&ctx, &stream, &q_idx_t, &k_comp_t, &w_idx_t, &config)
        .expect("Stage A failed");

    // --- Stage B: sparse CoreAttn ---
    let n_core = (batch * seq_len * h_c * c) as usize;
    let q_core = DeviceBuffer::<f32>::zeroed(&stream, n_core).unwrap();
    let c_comp =
        DeviceBuffer::<f32>::zeroed(&stream, (t_b * config.routing_block_entries * c) as usize)
            .unwrap();

    let q_core_t = Tensor::new(q_core, vec![batch, seq_len, h_c, c]);
    let c_comp_t = Tensor::new(c_comp, vec![t_b, config.routing_block_entries, c]);

    println!("Stage B: running sparse CoreAttn...");
    let output = core_attn_fwd(&ctx, &stream, &q_core_t, &c_comp_t, &selection, &config)
        .expect("Stage B failed");

    // Build reverse index for backward / KV-outer
    let reverse = build_reverse_index(&selection, t_b).expect("reverse index failed");
    println!(
        "Reverse index: {} blocks, {} total edges",
        reverse.block_count,
        reverse.query_handles.len()
    );

    println!(
        "CSA forward complete: output shape {:?}, LSE shape {:?}",
        output.core_output.len(),
        output.lse.len(),
    );
}
