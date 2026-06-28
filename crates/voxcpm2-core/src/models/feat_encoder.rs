//! DEPRECATED: Replaced by `loc_enc.rs` (`LocEnc`).
//!
//! The old `FeatEncoder` + `FeatEncoderLayer` structs have been removed.
//! `pub mod feat_encoder;` was removed from `mod.rs` — this file is now inert.
//!
//! Reason: The Python reference uses a different architecture (VoxCPMLocEnc) that
//! `loc_enc.rs` implements. The old `FeatEncoder` was a simplified transformer that
//! didn't match the reference layer structure (GQA, in_proj bias, hidden_shrink, etc.).
//!
//! Status: Removed 2026-06-29. Module declaration deleted from mod.rs.
//! This file exists on disk only because `Remove-Item` is permission-blocked on Windows.
//! It is NOT compiled (no `pub mod` reference in `mod.rs`).
