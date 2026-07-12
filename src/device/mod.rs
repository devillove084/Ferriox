//! Device kernel modules.
//!
//! GPU kernels compiled to PTX via cuda-oxide.  Each `#[cuda_module]` is
//! automatically compiled and embedded in the binary as an `.oxart` section.
//! Load them on the host with `ModuleName::load(&ctx)`.

pub mod indexer;
pub mod core_attn;
pub mod backward;
