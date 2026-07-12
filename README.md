# Ferriox

**Block-Routed Memory-Bounded Compressed Sparse Attention in Pure Rust**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Ferriox is a pure-Rust co-design of DeepSeek-V4-style Compressed Sparse
Attention (CSA), block routing, exact block-sparse CoreAttn, and a
dependency-aware native backward runtime for NVIDIA GPUs through
[cuda-oxide](https://github.com/NVlabs/cuda-oxide).

## Overview

DeepSeek-V4's compressed sparse attention selects the top-*k* compressed
KV entries per query, but a naive implementation materialises a head-wise
score tensor that reaches 256 GiB at 65K tokens. Ferriox solves this by
introducing a **routing block** abstraction: consecutive compressed entries
are grouped into blocks (`b_K = 32`), and selection operates on block-level
max scores (`κ = 16` blocks, totalling `κ·b_K = 512` entries per query).
This keeps V4's maximum attention budget while making selection and
execution block-regular.

The design separates two block scales that must not be conflated:

- **V4 compression stride** `m = 4`: one compressed entry per four new source positions
- **Ferriox routing stride** `b_K = 32`: one routing block per 32 compressed entries (128 source positions)

### Architecture

```
                   ┌─────────────────────────┐
                   │  Hidden States h_t       │
                   └──────┬──────────┬───────┘
                          │          │
              ┌───────────┘          └───────────┐
              ▼                                  ▼
     ┌─────────────────┐              ┌─────────────────┐
     │ Stage A          │              │ KV Compression  │
     │ Lightning Indexer│              │ C^Comp (offline)│
     │ Streaming block  │              └────────┬────────┘
     │ scoring + top-κ  │                       │
     └────────┬────────┘                       │
              │ block IDs [B,S,κ]               │
              └──────────┬──────────────────────┘
                         ▼
              ┌─────────────────────┐
              │ Stage B              │
              │ Fixed-Block Sparse   │
              │ CoreAttn             │
              │ (Q-outer / KV-outer) │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Output + LSE         │
              └─────────────────────┘
```

### Key Properties

| Property | Value |
|---|---|
| Compression stride | `m = 4` |
| Routing block size | `b_K = 32` compressed entries |
| Selected blocks | `κ = 16` |
| Max attention entries | `κ·b_K = 512` |
| Legal domain | `T_legal(t) = ⌊(t+1)/m⌋`, zero-based |
| Stage A arithmetic | O(S·T·H_I·c_I) — quadratic, but memory-bounded |
| Stage B arithmetic | O(S·κ·b_K·H·c) — fixed-capacity |
| Backward FLOPs | ~1.40× fused (dual-pass) |
| Backward determinism | yes (with stable CSR/CSC + slab reduction order) |

### Profiles

| Profile | `b_K` | `κ` | Description |
|---|---|---|---|
| Ferriox block routing | 32 | 16 | Primary: block-regular selection and execution |
| V4-compatible reference | 1 | 512 | Per-entry selector (Ferriox deterministic reference; full checkpoint parity requires boundary and tie-break closure) |

## Getting Started

### Prerequisites

- Rust nightly (see `rust-toolchain.toml`)
- NVIDIA GPU with CUDA support
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide) toolchain

### Build

```bash
cargo build --release
```

### Example

```bash
cargo run --example simple
```

```rust
use ferriox::{CsaConfig, CsaForward, CsaBackward};
use cuda_core::CudaContext;

let ctx = CudaContext::new(0)?;
let stream = ctx.default_stream();

let config = CsaConfig {
    model_dim: 7168,
    query_latent_dim: 1536,
    core_dim: 512,
    index_dim: 128,
    core_heads: 64,
    index_heads: 128,
    compression: 4,
    routing_block_entries: 32,
    selected_blocks: 16,
    window: 512,
    output_groups: 1,
    group_intermediate_dim: 0,
    sm_scale: 1.0 / (512.0_f32).sqrt(),
};

// Stage A — select routing blocks
let selection = ferriox::select_blocks(
    &ctx, &stream, &q, &k_comp, &w_idx, &config
)?;

// Stage B — sparse CoreAttn
let output = ferriox::core_attn_fwd(
    &ctx, &stream, &q_core, &c_comp, &selection, &config
)?;
```

## Project Status

**Phase I** (in progress): Block-sparse semantic contract and atomic baseline
- ✅ Crate structure and core types
- ✅ CSA configuration and block selection contract
- 🚧 FP32 reference indexer
- 🚧 Q-outer block CoreAttn
- 🚧 Reverse index and atomic backward baseline

**Phase II** (planned): Stable CSR/CSC and atomic-free dual-pass backward

**Phase III** (planned): Torch-independent Rust runtime and distributed training

See [docs/CODESIGN.md](docs/CODESIGN.md) for the complete technical design document.

## License

Apache-2.0
