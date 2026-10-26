//! Recorded tracking results and the edits that repair them: split, delete,
//! clear, connect across a gap, and moving the tracked point by hand.

use crate::estimate::{MotionModel, fit};
use crate::geometry::{Point, Quad, Transform};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Continuity {
    /// The target was found in this frame.
    Tracked,
    /// The target was lost; the sample holds its last known position.
    Searching,
    /// Filled in when two segments were connected.
    Interpolated,
}

/// Where the target is in one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackSample {
    /// Motion from the reference frame to this one, as the track's motion
    /// model records it.
    pub transform: Transform,
    /// The tracking region in this frame.
    pub quad: Quad,
    /// 0 to 1.
    pub confidence: f32,
    pub continuity: Continuity,
}

impl TrackSample {
    pub fn reference(quad: Quad) -> Self {
        Self {
            transform: Transform::IDENTITY,
            quad,
            confidence: 1.0,
            continuity: Continuity::Tracked,
        }
    }
}

/// Consecutive frames of tracking.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    start: u32,
    samples: Vec<TrackSample>,
}

impl Segment {
    pub fn start(&self) -> u32 {
        self.start
    }

    /// The last frame, inclusive.
    pub fn end(&self) -> u32 {
        self.start + self.samples.len() as u32 - 1
    }

    pub fn contains(&self, frame: u32) -> bool {
        (self.start..=self.end()).contains(&frame)
    }

    pub fn samples(&self) -> &[TrackSample] {
        &self.samples
    }

    pub fn get(&self, frame: u32) -> Option<&TrackSample> {
        frame
            .checked_sub(self.start)
            .and_then(|i| self.samples.get(i as usize))
    }
}

/// How a hand correction at the playhead spreads through its segment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EditScope {
    /// Only the sample at the playhead.
    #[default]
    Current,
    /// The playhead and every later sample in the segment.
    Forward,
    /// Ramps the correction in from the start of the segment, then applies
    /// all of it from the playhead on, to undo slow drift.
    DriftCorrection,
}

/// Tracking for one target: the region it started from and the segments of
/// frames it was followed through.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    reference: Quad,
    motion_model: MotionModel,
    segments: Vec<Segment>,
}

impl Track {
    pub fn new(reference: Quad, motion_model: MotionModel) -> Self {
        Self {
            reference,
            motion_model,
            segments: Vec::new(),
        }
    }

    pub fn reference(&self) -> Quad {
        self.reference
    }

    pub fn motion_model(&self) -> MotionModel {
        self.motion_model
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn sample(&self, frame: u32) -> Option<&TrackSample> {
        self.segment_index(frame)
            .and_then(|i| self.segments[i].get(frame))
    }

    fn segment_index(&self, frame: u32) -> Option<usize> {
        self.segments
            .iter()
            .position(|segment| segment.contains(frame))
    }

    /// Stores the result for `frame`, replacing what was there, extending
    /// the neighbouring segment, or joining two segments it closes the gap
    /// between. Frames can arrive in either direction.
    pub fn record(&mut self, frame: u32, sample: TrackSample) {
        if let Some(i) = self.segment_index(frame) {
            let segment = &mut self.segments[i];
            segment.samples[(frame - segment.start) as usize] = sample;
            return;
        }

        let before = self
            .segments
            .iter()
            .position(|s| s.end().checked_add(1) == Some(frame));
        let after = self
            .segments
            .iter()
            .position(|s| frame.checked_add(1) == Some(s.start));
        match (before, after) {
            (Some(b), Some(a)) => {
                let tail = self.segments.remove(a);
                let b = if a < b { b - 1 } else { b };
                self.segments[b].samples.push(sample);
                self.segments[b].samples.extend(tail.samples);
            }
            (Some(b), None) => self.segments[b].samples.push(sample),
            (None, Some(a)) => {
                let segment = &mut self.segments[a];
                segment.samples.insert(0, sample);
                segment.start = frame;
            }
            (None, None) => {
                let at = self.segments.partition_point(|s| s.start < frame);
                self.segments.insert(
                    at,
                    Segment {
                        start: frame,
                        samples: vec![sample],
                    },
                );
            }
        }
    }

    /// Splits the segment containing `frame` so a new one starts there.
    pub fn split(&mut self, frame: u32) -> bool {
        let Some(i) = self.segment_index(frame) else {
            return false;
        };
        let segment = &mut self.segments[i];
        if frame == segment.start {
            return false;
        }
        let tail = segment.samples.split_off((frame - segment.start) as usize);
        self.segments.insert(
            i + 1,
            Segment {
                start: frame,
                samples: tail,
            },
        );
        true
    }

    /// Removes the segment containing `frame`.
    pub fn delete_segment(&mut self, frame: u32) -> bool {
        match self.segment_index(frame) {
            Some(i) => {
                self.segments.remove(i);
                true
            }
            None => false,
        }
    }

    pub fn clear(&mut self) {
        self.segments.clear();
    }

    /// Joins the segment at or before `frame` with the next one, filling the
    /// gap with a cubic Hermite curve that leaves each end at the speed it
    /// was moving.
    pub fn connect(&mut self, frame: u32) -> bool {
        let Some(i) = self
            .segments
            .iter()
            .rposition(|segment| segment.start <= frame)
        else {
            return false;
        };
        if i + 1 >= self.segments.len() {
            return false;
        }

        let next = self.segments.remove(i + 1);
        let previous = &self.segments[i];
        let gap = next.start - previous.end();
        let (from, to) = (previous.samples.as_slice(), next.samples.as_slice());
        let (last, first) = (from[from.len() - 1], to[0]);
        // The motion is interpolated as the path of the reference corners.
        let motion_corners = |sample: &TrackSample| self.reference.map(&sample.transform);
        let speed = |a: &TrackSample, b: &TrackSample| {
            (
                difference(&b.quad, &a.quad),
                difference(&motion_corners(b), &motion_corners(a)),
            )
        };
        let zero = [Point::default(); 4];
        let (start_quad_speed, start_motion_speed) = from
            .len()
            .checked_sub(2)
            .map_or((zero, zero), |k| speed(&from[k], &last));
        let (end_quad_speed, end_motion_speed) = to
            .get(1)
            .map_or((zero, zero), |second| speed(&first, second));
        let (start_quad, start_motion) = (last.quad, motion_corners(&last));
        let (end_quad, end_motion) = (first.quad, motion_corners(&first));

        let mut filled = Vec::with_capacity(gap.saturating_sub(1) as usize);
        for step in 1..gap {
            let t = step as f64 / gap as f64;
            let quad = hermite(
                &start_quad,
                &start_quad_speed,
                &end_quad,
                &end_quad_speed,
                t,
                gap as f64,
            );
            let motion = hermite(
                &start_motion,
                &start_motion_speed,
                &end_motion,
                &end_motion_speed,
                t,
                gap as f64,
            );
            filled.push(TrackSample {
                transform: self.transform_onto(&motion),
                quad,
                confidence: 0.0,
                continuity: Continuity::Interpolated,
            });
        }

        let segment = &mut self.segments[i];
        segment.samples.extend(filled);
        segment.samples.extend(next.samples);
        true
    }

    /// Moves the tracked region at `frame` by `delta`, spreading the change
    /// through the segment as `scope` says.
    pub fn edit(&mut self, frame: u32, delta: Point, scope: EditScope) -> bool {
        let Some(i) = self.segment_index(frame) else {
            return false;
        };
        let segment = &mut self.segments[i];
        let (start, end) = (segment.start, segment.end());
        let weight = |f: u32| -> f64 {
            match scope {
                EditScope::Current => f64::from(u8::from(f == frame)),
                EditScope::Forward => f64::from(u8::from(f >= frame)),
                EditScope::DriftCorrection if f >= frame => 1.0,
                EditScope::DriftCorrection => (f - start) as f64 / (frame - start) as f64,
            }
        };
        for f in start..=end {
            let w = weight(f);
            if w == 0.0 {
                continue;
            }
            let shift = delta * w;
            let sample = &mut segment.samples[(f - start) as usize];
            sample.quad = sample.quad.translated(shift);
            sample.transform = sample
                .transform
                .then(&Transform::translation(shift.x, shift.y));
        }
        true
    }

    /// The track's motion model fitted from the reference region onto
    /// `corners`.
    fn transform_onto(&self, corners: &Quad) -> Transform {
        fit(self.motion_model, &self.reference.corners, &corners.corners)
            .unwrap_or(Transform::IDENTITY)
    }
}

fn difference(a: &Quad, b: &Quad) -> [Point; 4] {
    std::array::from_fn(|i| a.corners[i] - b.corners[i])
}

/// Cubic Hermite interpolation of each corner; speeds are per frame and the
/// gap is `span` frames long.
fn hermite(
    start: &Quad,
    start_speed: &[Point; 4],
    end: &Quad,
    end_speed: &[Point; 4],
    t: f64,
    span: f64,
) -> Quad {
    let (t2, t3) = (t * t, t * t * t);
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    Quad::new(std::array::from_fn(|i| {
        start.corners[i] * h00
            + start_speed[i] * (h10 * span)
            + end.corners[i] * h01
            + end_speed[i] * (h11 * span)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_points_close;

    fn reference() -> Quad {
        Quad::from_rect(0.0, 0.0, 10.0, 10.0)
    }

    fn at(x: f64) -> TrackSample {
        let transform = Transform::translation(x, 0.0);
        TrackSample {
            transform,
            quad: reference().map(&transform),
            confidence: 1.0,
            continuity: Continuity::Tracked,
        }
    }

    fn track(frames: impl IntoIterator<Item = u32>) -> Track {
        let mut track = Track::new(reference(), MotionModel::Translation);
        for frame in frames {
            track.record(frame, at(frame as f64 * 2.0));
        }
        track
    }

    fn spans(track: &Track) -> Vec<(u32, u32)> {
        track
            .segments()
            .iter()
            .map(|s| (s.start(), s.end()))
            .collect()
    }

    #[test]
    fn records_in_any_order_and_merges() {
        let mut track = track([5, 6, 7, 2, 3]);
        assert_eq!(spans(&track), vec![(2, 3), (5, 7)]);
        track.record(4, at(8.0));
        assert_eq!(spans(&track), vec![(2, 7)]);
        track.record(1, at(2.0));
        assert_eq!(spans(&track), vec![(1, 7)]);
    }

    #[test]
    fn splits_deletes_and_clears() {
        let mut track = track(0..10);
        assert!(track.split(4));
        assert!(!track.split(4));
        assert_eq!(spans(&track), vec![(0, 3), (4, 9)]);
        assert!(track.delete_segment(6));
        assert_eq!(spans(&track), vec![(0, 3)]);
        assert!(track.sample(6).is_none());
        track.clear();
        assert!(track.is_empty());
    }

    #[test]
    fn connects_across_a_gap_smoothly() {
        let mut track = track((0..5).chain(10..15));
        assert!(track.connect(2));
        assert_eq!(spans(&track), vec![(0, 14)]);
        // Both sides move 2 px per frame, so the curve keeps that pace.
        for frame in 5..10 {
            let sample = track.sample(frame).unwrap();
            assert_eq!(sample.continuity, Continuity::Interpolated);
            assert_points_close(
                sample.quad.corners[0],
                Point::new(frame as f64 * 2.0, 0.0),
                1e-9,
            );
            assert_points_close(
                sample.transform.apply(Point::default()),
                Point::new(frame as f64 * 2.0, 0.0),
                1e-9,
            );
        }
        assert!(!track.connect(2));
    }

    #[test]
    fn edits_spread_by_scope() {
        let shift = Point::new(0.0, 4.0);
        let y = |track: &Track, frame| track.sample(frame).unwrap().quad.corners[0].y;

        let mut current = track(0..10);
        current.edit(5, shift, EditScope::Current);
        assert_eq!(
            (y(&current, 4), y(&current, 5), y(&current, 6)),
            (0.0, 4.0, 0.0)
        );

        let mut forward = track(0..10);
        forward.edit(5, shift, EditScope::Forward);
        assert_eq!(
            (y(&forward, 4), y(&forward, 5), y(&forward, 9)),
            (0.0, 4.0, 4.0)
        );

        let mut drift = track(0..10);
        drift.edit(4, shift, EditScope::DriftCorrection);
        assert_eq!(
            (y(&drift, 0), y(&drift, 2), y(&drift, 4), y(&drift, 9)),
            (0.0, 2.0, 4.0, 4.0)
        );
        assert_eq!(
            drift.sample(2).unwrap().transform.apply(Point::default()).y,
            2.0
        );
    }
}
