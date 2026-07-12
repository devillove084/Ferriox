//! Stage A: Lightning Indexer GPU kernels.

use cuda_device::cuda_module;

#[cuda_module]
pub mod indexer_kernels {

    use cuda_device::{DisjointSlice, kernel, thread};

    #[kernel]
    pub fn indexer_kernel(
        _q_idx: &[f32],
        _k_comp: &[f32],
        _w_idx: &[f32],
        _out_ids: DisjointSlice<u32>,
        _t_legal: &[u32],
        _batch: u32,
        _seq_len: u32,
        _index_heads: u32,
        _index_dim: u32,
        _num_compressed: u32,
        _b_k: u32,
        _kappa: u32,
    ) {
        let _idx = thread::index_1d();
        todo!("Indexer kernel not yet implemented")
    }
}
