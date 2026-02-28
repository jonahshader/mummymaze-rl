use thiserror::Error;

#[derive(Error, Debug)]
pub enum MummyMazeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Markov solver failed to converge after {0} iterations")]
    ConvergenceFailure(usize),
}

pub type Result<T> = std::result::Result<T, MummyMazeError>;
