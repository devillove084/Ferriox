//! Stage B: Fixed-Block Sparse CoreAttn GPU kernels.

use cuda_device::cuda_module;

#[cuda_module]
pub mod core_attn_kernels {

    use cuda_device::{DisjointSlice, kernel, thread};

    #[kernel]
    pub fn core_attn_fwd_kernel(
        _q_core: &[f32],
        _c_comp: &[f32],
        _block_ids: &[u32],
        _out: DisjointSlice<f32>,
        _lse: DisjointSlice<f32>,
        _sink_logits: &[f32],
        _batch: u32,
        _seq_len: u32,
        _core_heads: u32,
        _core_dim: u32,
        _b_k: u32,
        _kappa: u32,
        _sm_scale: f32,
    ) {
        let _idx = thread::index_1d();
        todo!("CoreAttn forward kernel not yet implemented")
    }
}
