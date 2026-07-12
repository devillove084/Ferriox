//! Backward GPU kernels.

use cuda_device::cuda_module;

#[cuda_module]
pub mod backward_kernels {

    use cuda_device::{DisjointSlice, kernel, thread};

    #[kernel]
    pub fn dq_pass_kernel(
        _q: &[f32],
        _c_comp: &[f32],
        _dout: &[f32],
        _lse: &[f32],
        _block_ids: &[u32],
        _dq: DisjointSlice<f32>,
        _batch: u32,
        _seq_len: u32,
        _core_heads: u32,
        _core_dim: u32,
        _b_k: u32,
        _kappa: u32,
        _sm_scale: f32,
    ) {
        let _idx = thread::index_1d();
        todo!("CSR dQ kernel not yet implemented")
    }

    #[kernel]
    pub fn dc_pass_kernel(
        _q: &[f32],
        _c_comp: &[f32],
        _dout: &[f32],
        _lse: &[f32],
        _query_handles: &[u32],
        _block_offsets: &[u32],
        _dc: DisjointSlice<f32>,
        _batch: u32,
        _seq_len: u32,
        _core_heads: u32,
        _core_dim: u32,
        _b_k: u32,
        _chunk_id: u32,
        _sm_scale: f32,
    ) {
        let _idx = thread::index_1d();
        todo!("CSC dC kernel not yet implemented")
    }
}
