//! CSA configuration and block-selection types.
//!
//! Defines the full configuration space for Ferriox's Compressed Sparse
//! Attention, matching §10 of the design document.  The `CsaConfig` struct
//! is the single source of truth for shape, profile, and precision choices.

/// CSA configuration — the single source of truth for all shape, profile,
/// and precision choices (§10 of CODESIGN.md).
#[derive(Debug, Clone)]
pub struct CsaConfig {
    /// Main model dimension (d in V4 notation).
    pub model_dim: u32,
    /// Query latent dimension after `W^{DQ}` projection.
    pub query_latent_dim: u32,
    /// Core-attention head dimension `c` (V4 core head dim).
    pub core_dim: u32,
    /// Indexer head dimension `c_I`.
    pub index_dim: u32,
    /// Number of core-attention heads `H`.
    pub core_heads: u32,
    /// Number of indexer heads `H_I`.
    pub index_heads: u32,
    /// Compression stride `m`: one compressed entry per `m` new tokens.
    pub compression: u32,
    /// Routing block size `b_K`: number of consecutive compressed entries
    /// in one routing block.
    pub routing_block_entries: u32,
    /// Number of selected routing blocks `κ` per query.
    pub selected_blocks: u32,
    /// Sliding-window size `w` in tokens.
    pub window: u32,
    /// Number of output-attention groups for grouped-query attention
    /// (1 = standard MHA).
    pub output_groups: u32,
    /// Intermediate dimension for gated grouped output (0 = disabled).
    pub group_intermediate_dim: u32,
    /// Softmax scale: typically `1 / sqrt(core_dim)`.
    pub sm_scale: f32,
}

impl CsaConfig {
    /// V4-Flash-compatible dimensions.
    pub fn v4_flash() -> Self {
        Self {
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
        }
    }

    /// V4-compatible per-entry profile (b_K = 1, κ = 512).
    ///
    /// Note: this reproduces Ferriox's deterministic per-entry reference.
    /// Full checkpoint-path parity with the published V4 kernel requires
    /// closure of the legal-domain and tie-break contracts (§6.7).
    pub fn v4_compat() -> Self {
        Self {
            routing_block_entries: 1,
            selected_blocks: 512,
            ..Self::v4_flash()
        }
    }
}

/// Compact block-selection result: `selected_blocks` per query.
///
/// Shape: `[batch, seq_len, selected_blocks]`, row-major.
/// Invalid/empty slots are filled with sentinel value `u32::MAX`.
#[derive(Debug, Clone)]
pub struct BlockSelection {
    /// Flat buffer of `u32` block IDs, shape `[B, S, κ]`.
    pub block_ids: Vec<u32>,
    /// Row-major strides: `[S*κ, κ, 1]`.
    pub strides: [u32; 3],
    /// Batch size.
    pub batch: u32,
    /// Sequence length.
    pub seq_len: u32,
}

impl BlockSelection {
    /// Build a selection from pre-computed block IDs.
    ///
    /// The `block_ids` buffer is `[B, S, κ]` row-major.  Invalid slots use
    /// `u32::MAX` as sentinel.
    pub fn new(block_ids: Vec<u32>, batch: u32, seq_len: u32, kappa: u32) -> Self {
        assert_eq!(
            block_ids.len(),
            (batch * seq_len * kappa) as usize,
            "block_ids length mismatch"
        );
        Self {
            strides: [seq_len * kappa, kappa, 1],
            block_ids,
            batch,
            seq_len,
        }
    }

    /// Return the number of selected blocks per query (κ).
    pub fn kappa(&self) -> u32 {
        self.strides[1]
    }
}

/// Reverse view: for each routing block, which queries selected it.
///
/// CSC-equivalent structure for KV-outer execution and backward.
#[derive(Debug, Clone)]
pub struct ReverseBlockIndex {
    /// Exclusive scan: `block_offsets[r]` is the start index in `query_handles`
    /// for routing block `r`.  Length `R + 1`.
    pub block_offsets: Vec<u32>,
    /// Packed `(query_id, slot_index)` handles.  Length = total edges.
    pub query_handles: Vec<u32>,
    /// Number of queries that selected each block.  Length `R`.
    pub block_degrees: Vec<u32>,
    /// Total number of selected blocks (routing blocks with ≥1 selector).
    pub block_count: u32,
}

/// Forward-pass output: core attention result and optional LSE.
#[derive(Debug, Clone)]
pub struct CoreAttnOutput {
    /// Core-attention output, shape `[B, S, H_core, c]`.
    pub core_output: Vec<f32>,
    /// Log-sum-exp per query-head, shape `[B, S, H_core]`.
    /// Required for backward.
    pub lse: Vec<f32>,
}

/// Backward backend selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardBackend {
    /// KV-major with atomic `dQ` (FlashMoBA-style baseline).
    KvMajorAtomicDq,
    /// FSA-style buffered partial `dQ` and reduction.
    BufferedPartialDq,
    /// CSR/CSC dual-pass atomic-free (default deterministic).
    DualPassAtomicFree,
}

/// Runtime execution policy.
#[derive(Debug, Clone)]
pub struct ExecutionPolicy {
    /// Which backward backend to use.
    pub backward: BackwardBackend,
    /// Require bitwise-deterministic outputs.
    pub deterministic: bool,
    /// Maximum workspace to allocate (bytes); 0 = use heuristics.
    pub workspace_limit_bytes: u64,
    /// Number of query positions per slab (0 = auto).
    pub query_slab: u32,
    /// Number of heads processed together (0 = auto).
    pub head_group: u32,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            backward: BackwardBackend::DualPassAtomicFree,
            deterministic: true,
            workspace_limit_bytes: 0,
            query_slab: 0,
            head_group: 0,
        }
    }
}
