//! Error type for `oersted`  
//!
//! WIP: perhaps this should be multiple errors?

#[derive(Debug)]
pub enum OerstedError {
    FileOpenError,
    FloatParseError,
    ColNumberChange,
}
