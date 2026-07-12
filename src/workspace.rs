//! Workspace / arena allocator for fixed-size intermediate buffers.
//!
//! Ferriox operations need scratch space for block scores, partial
//! softmax states, hot-block `dC` partials, and slab-crossing reduction
//! buffers.  This module provides a bounded arena that enforces the
//! configured `workspace_limit_bytes`.
//!
//! See §7.6–§7.7 of CODESIGN.md for capacity planning.

use crate::error::FerrioxError;

/// A pre-allocated HBM workspace arena.
pub struct Workspace {
    /// Raw allocation (on device).
    _buf: Vec<u8>,
    /// Total capacity in bytes.
    capacity: u64,
    /// Current allocation cursor.
    cursor: u64,
    /// High-water mark for debugging / profiling.
    high_water: u64,
}

impl Workspace {
    /// Allocate a new workspace of the given capacity (bytes).
    ///
    /// In a real implementation this would allocate device memory via
    /// cuda-oxide.  The stub uses host `Vec<u8>` for prototyping.
    pub fn new(capacity: u64) -> Self {
        Self {
            _buf: vec![0u8; capacity as usize],
            capacity,
            cursor: 0,
            high_water: 0,
        }
    }

    /// Reserve a contiguous slice of `size` bytes, aligned to `align`.
    ///
    /// Returns the byte offset into the workspace.  Panics if the
    /// allocation would exceed capacity.
    pub fn alloc(&mut self, size: u64, align: u64) -> Result<u64, FerrioxError> {
        let offset = (self.cursor + align - 1) & !(align - 1);
        let end = offset + size;
        if end > self.capacity {
            return Err(FerrioxError::InvalidParam(format!(
                "workspace exhausted: requested {size} bytes at offset {offset}, capacity {}",
                self.capacity
            )));
        }
        self.cursor = end;
        self.high_water = self.high_water.max(end);
        Ok(offset)
    }

    /// Reset the allocation cursor (but retain the allocation).
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Return the current high-water mark in bytes.
    pub fn high_water(&self) -> u64 {
        self.high_water
    }

    /// Return the total capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}
