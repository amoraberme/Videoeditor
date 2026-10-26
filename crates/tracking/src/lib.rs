//! Motion tracking and stabilization for OpenCut.
//!
//! Everything here works on plain luma frames, so any decoder (or a test)
//! can feed it, and it runs the same on desktop, in the browser, or headless.
//!
//! - [`Tracker`] follows a region through a clip, one frame at a time, with
//!   one of three engines: [`TrackerKind::Features`] (corner points and
//!   optical flow, precise), [`TrackerKind::Object`] (template correlation,
//!   robust to blur and fast motion) or [`TrackerKind::Frame`] (the dominant
//!   motion of the whole frame, i.e. the camera).
//! - [`Track`] stores the result as segments that can be split, deleted,
//!   connected across gaps and corrected by hand.
//! - [`MotionPath`] turns a track into the motion a layer follows, filtered
//!   ([`MotionFilter`]) and reduced to the components that should carry over
//!   ([`MotionComponents`]), or into the inverse that locks the video onto
//!   the target.
//! - [`stabilize`] smooths the camera path measured by
//!   [`CameraMotionEstimator`] and frames the result.

mod estimate;
mod features;
mod flow;
mod frame;
mod geometry;
mod motion;
mod object;
mod stabilize;
mod track;
mod tracker;

pub use estimate::{MotionModel, homography_from_quads};
pub use frame::Frame;
pub use geometry::{Decomposition, MotionComponents, Point, Quad, Transform};
pub use motion::{MotionFilter, MotionPath, MotionSettings};
pub use stabilize::{
    AnalysisQuality, CameraMotion, CameraMotionEstimator, EdgeFill, Framing, PairMotion,
    Stabilization, StabilizationMode, StabilizerSettings, stabilize, warp,
};
pub use track::{Continuity, EditScope, Segment, Track, TrackSample};
pub use tracker::{RegionShape, Tracker, TrackerKind, TrackerSettings};

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrackingError {
    /// The pixel buffer does not match the stated frame size.
    InvalidFrame,
    /// A frame's size differs from the frame tracking started on.
    FrameSizeMismatch,
    /// The region is degenerate or lies outside the frame.
    InvalidRegion,
    /// The region has too little texture to track.
    LowDetail,
}

impl fmt::Display for TrackingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidFrame => "the pixel buffer does not match the frame size",
            Self::FrameSizeMismatch => "the frame size changed during tracking",
            Self::InvalidRegion => "the region is degenerate or outside the frame",
            Self::LowDetail => "low detail, try a different area or adjust settings",
        })
    }
}

impl std::error::Error for TrackingError {}

#[cfg(test)]
pub(crate) mod testing;
