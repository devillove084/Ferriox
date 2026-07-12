use thiserror::Error;

#[derive(Error, Debug)]
pub enum FerrioxError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    #[error("unsupported configuration: {0}")]
    Unsupported(String),

    #[error("kernel launch failed: {0}")]
    KernelLaunch(String),
}

impl FerrioxError {
    /// Create a CUDA error from anything that can display.
    pub fn cuda(msg: impl std::fmt::Display) -> Self {
        FerrioxError::Cuda(msg.to_string())
    }
}
