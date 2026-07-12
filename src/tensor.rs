use cuda_core::DeviceBuffer;

/// A handle to a device-side tensor with known shape and strides.
///
/// Wraps a [`DeviceBuffer<T>`] with metadata so that kernels can index into the
/// flat buffer correctly.
///
/// `Tensor` does not implement `Clone` — use shared references (`&Tensor<T>`)
/// to pass the same buffer to multiple consumers.
pub struct Tensor<T> {
    pub buf: DeviceBuffer<T>,
    /// Row-major shape, e.g. `[batch, heads, seq_len, head_dim]`.
    pub shape: Vec<u32>,
    /// Row-major strides in elements (not bytes).
    pub strides: Vec<u32>,
}

impl<T: Copy> Tensor<T> {
    pub fn new(buf: DeviceBuffer<T>, shape: Vec<u32>) -> Self {
        let strides = compute_strides(&shape);
        Self {
            buf,
            shape,
            strides,
        }
    }

    pub fn num_elems(&self) -> u32 {
        self.shape.iter().product()
    }
}

impl<T> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish_non_exhaustive()
    }
}

fn compute_strides(shape: &[u32]) -> Vec<u32> {
    let ndim = shape.len();
    let mut strides = vec![1u32; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
