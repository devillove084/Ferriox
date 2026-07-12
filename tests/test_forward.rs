//! Integration tests for the CSA forward pass.
//!
//! Compare against reference implementations to catch regressions.
//! All tests are ignored until kernels are implemented.
//!
//! Run with: `cargo test`

use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer};
use ferriox::{CsaConfig, Tensor, build_reverse_index, select_blocks};

fn create_ctx() -> (Arc<CudaContext>, Arc<cuda_core::CudaStream>) {
    let ctx = CudaContext::new(0).expect("no CUDA device");
    let stream = ctx.default_stream();
    (ctx, stream)
}

/// Verify that the legal domain function matches the documented contract.
#[test]
fn legal_domain_contract() {
    use ferriox::indexer::legal_domain_size;
    let m = 4;
    assert_eq!(legal_domain_size(0, m), 0);
    assert_eq!(legal_domain_size(1, m), 0);
    assert_eq!(legal_domain_size(2, m), 0);
    assert_eq!(legal_domain_size(3, m), 1);
    assert_eq!(legal_domain_size(4, m), 1);
    assert_eq!(legal_domain_size(7, m), 2);
}

/// Smoke-test: zeroed inputs should not panic.
#[test]
#[ignore = "kernel not yet implemented"]
fn stage_a_zero_inputs_does_not_panic() {
    let (_ctx, _stream) = create_ctx();
    let config = CsaConfig::v4_flash();

    let batch = 1u32;
    let seq_len = 16u32;
    let t_b = seq_len / config.compression;

    let n_idx = (batch * seq_len * config.index_heads * config.index_dim) as usize;
    let q_idx = DeviceBuffer::<f32>::zeroed(&_stream, n_idx).unwrap();
    let k_comp = DeviceBuffer::<f32>::zeroed(
        &_stream,
        (t_b * config.index_heads * config.index_dim) as usize,
    )
    .unwrap();
    let w_idx =
        DeviceBuffer::<f32>::zeroed(&_stream, (batch * seq_len * config.index_heads) as usize)
            .unwrap();

    let q_t = Tensor::new(
        q_idx,
        vec![batch, seq_len, config.index_heads, config.index_dim],
    );
    let k_t = Tensor::new(k_comp, vec![t_b, config.index_heads, config.index_dim]);
    let w_t = Tensor::new(w_idx, vec![batch, seq_len, config.index_heads]);

    let selection = select_blocks(&_ctx, &_stream, &q_t, &k_t, &w_t, &config)
        .expect("Stage A should not panic on zero inputs");
    assert_eq!(selection.kappa(), config.selected_blocks);
}

/// Verify the reverse view builder produces correct edge counts.
#[test]
fn reverse_view_edge_count() {
    let config = CsaConfig::v4_flash();
    let batch = 2u32;
    let seq_len = 8u32;
    let kappa = config.selected_blocks as usize;

    let total_queries = (batch * seq_len) as usize;
    let mut block_ids = vec![u32::MAX; total_queries * kappa];
    for q in 0..total_queries {
        block_ids[q * kappa] = 0;
        block_ids[q * kappa + 1] = 1;
    }

    let selection = ferriox::BlockSelection::new(block_ids, batch, seq_len, kappa as u32);

    let reverse = build_reverse_index(&selection, 2).expect("reverse view build should succeed");
    assert_eq!(reverse.block_degrees[0], total_queries as u32);
    assert_eq!(reverse.block_degrees[1], total_queries as u32);
    assert_eq!(reverse.query_handles.len(), (total_queries * 2) as usize);
}

/// Verify the entry comparator tie-break rule.
#[test]
fn entry_comparator_tie_break() {
    use ferriox::routing::EntryComparator;
    assert!(EntryComparator::gt(1.0, 5, 0.5, 0));
    assert!(EntryComparator::gt(1.0, 3, 1.0, 7));
    assert!(!EntryComparator::gt(1.0, 7, 1.0, 3));
}

/// Verify the workspace allocator.
#[test]
fn workspace_alloc_alignment() {
    use ferriox::workspace::Workspace;
    let mut ws = Workspace::new(1024);
    let off = ws.alloc(100, 16).unwrap();
    assert_eq!(off % 16, 0);
    let off2 = ws.alloc(200, 64).unwrap();
    assert_eq!(off2 % 64, 0);
    assert!(ws.high_water() <= 1024);
    ws.reset();
    let off3 = ws.alloc(500, 1).unwrap();
    assert_eq!(off3, 0);
}

/// Verify the sink module produces correct shapes.
#[test]
fn sink_init_and_reduce() {
    use ferriox::sink::{init_sink_logits, reduce_sink_grads};
    let h = 8u32;
    let logits = init_sink_logits(h);
    assert_eq!(logits.len(), h as usize);
    assert!(logits.iter().all(|&x| x == 0.0));

    let per_query: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let grads = reduce_sink_grads(&per_query, h);
    assert_eq!(grads.len(), h as usize);
    assert_eq!(grads[0], 48.0);
}
