//! Persistent task scheduler for backward passes.
//!
//! The dual-pass backward decomposes into independently schedulable tasks:
//! query-owner CSR tasks, ordinary KV-owner CSC tasks, hot KV chunks,
//! slab-dC reductions, and hot-partial reductions.  A global atomic counter
//! assigns work items; stream/event dependencies enforce input readiness.
//!
//! See §9.6 and §II.5 of CODESIGN.md.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::config::ExecutionPolicy;

/// A unit of backward work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    /// CSR query-owner pass: compute `dQ` for a query / head-group slice.
    QueryDQ,
    /// CSC KV-owner pass: compute `dC` for an ordinary routing block.
    OrdinaryDC,
    /// Hot-block chunk: compute a numbered chunk of a hot routing block's `dC`.
    HotDCChunk,
    /// Reduce hot-block chunk partials into final `dC`.
    HotDCReduce,
    /// Reduce slab-level `dC` partials into global `dC`.
    SlabDCReduce,
}

/// A persistent task queue with atomic counter.
pub struct TaskQueue {
    counter: AtomicU32,
    total: u32,
    policy: ExecutionPolicy,
}

impl TaskQueue {
    /// Create a new task queue with `total` work items.
    pub fn new(total: u32, policy: ExecutionPolicy) -> Self {
        Self {
            counter: AtomicU32::new(0),
            total,
            policy,
        }
    }

    /// Claim the next task index.  Returns `None` when all tasks are claimed.
    pub fn next(&self) -> Option<u32> {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed);
        if idx < self.total { Some(idx) } else { None }
    }

    /// Reset the counter for a new pass.
    pub fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }

    /// Return the total number of tasks.
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Return a reference to the execution policy.
    pub fn policy(&self) -> &ExecutionPolicy {
        &self.policy
    }
}

/// Compute the number of slab-dC reduction tasks.
///
/// Given `num_slabs` and `num_routing_blocks`, each block may need its
/// slab partials reduced into the global `dC`.  The exact count depends
/// on the chosen slab-crossing mechanism (§9.6).
pub fn slab_dc_reduction_count(num_slabs: u32, num_routing_blocks: u32) -> u32 {
    // Conservative upper bound: every block chosen in every slab.
    // Real count depends on actual selection distribution.
    num_slabs * num_routing_blocks
}
