//! Turning a track into the motion something follows: filtered, reduced to
//! the components that should carry over, and applied to a layer, a corner
//! pin, or (inverted) the video itself.

use crate::estimate::homography_from_quads;
use crate::geometry::{MotionComponents, Point, Quad, Transform};
use crate::track::Track;

/// How tracked motion is processed before it is applied.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum MotionFilter {
    /// The motion as tracked.
    #[default]
    None,
    /// Holds the target as still as possible: removes micro jitter and locks
    /// tightly to it.
    Lock,
    /// Softens the path without delay, removing small shakes.
    Smooth,
    /// Keeps only the tiny vibrations, so applying the inverse steadies the
    /// frame around the target without following it.
    Stabilize,
    /// Follows the target softly, with a gentle delay.
    Follow,
}

impl MotionFilter {
    pub const ALL: [Self; 5] = [
        Self::None,
        Self::Lock,
        Self::Smooth,
        Self::Stabilize,
        Self::Follow,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Lock => "Lock",
            Self::Smooth => "Smooth",
            Self::Stabilize => "Stabilize",
            Self::Follow => "Follow",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::None => "Uses the motion exactly as tracked.",
            Self::Lock => "Keeps the target as stable as possible, removing micro jitter.",
            Self::Smooth => "Softens the path without lag, removing small shakes.",
            Self::Stabilize => "Removes only the micro shake around the target, not its motion.",
            Self::Follow => "Follows the target softly, with a gentle delay.",
        }
    }

    /// Filters one coordinate's path. `rest` is where it sits with no motion.
    fn apply(self, values: &[f64], rest: f64, strength: f64) -> Vec<f64> {
        let s = strength.clamp(0.0, 1.0);
        match self {
            Self::None => values.to_vec(),
            Self::Smooth => gaussian(values, 1.0 + 11.0 * s),
            Self::Lock => gaussian(
                &median(values, 1 + (3.0 * s).round() as usize),
                0.5 + 2.5 * s,
            ),
            Self::Stabilize => {
                let smooth = gaussian(values, 2.0 + 18.0 * s);
                values
                    .iter()
                    .zip(smooth)
                    .map(|(value, smooth)| rest + value - smooth)
                    .collect()
            }
            Self::Follow => follow(values, 0.4 + 0.5 * s),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionSettings {
    pub filter: MotionFilter,
    /// How strongly the filter acts, 0 to 1.
    pub strength: f64,
    /// Which parts of the motion carry over; the rest are frozen.
    pub components: MotionComponents,
}

impl Default for MotionSettings {
    fn default() -> Self {
        Self {
            filter: MotionFilter::None,
            strength: 0.5,
            components: MotionComponents::ALL,
        }
    }
}

/// Per-frame motion derived from a [`Track`]. Frames outside the track hold
/// the nearest tracked motion before them (or the first one).
#[derive(Clone, Debug, PartialEq)]
pub struct MotionPath {
    reference: Quad,
    segments: Vec<(u32, Vec<Transform>)>,
}

impl MotionPath {
    pub fn new(track: &Track, settings: &MotionSettings) -> Self {
        let reference = track.reference();
        let anchor = reference.center();
        let segments = track
            .segments()
            .iter()
            .map(|segment| {
                let samples = segment.samples();
                let corners: Vec<Quad> = samples
                    .iter()
                    .map(|s| reference.map(&s.transform))
                    .collect();
                let filtered = filter_corners(&corners, &reference, settings);
                let transforms = samples
                    .iter()
                    .zip(&filtered)
                    .map(|(sample, quad)| {
                        let motion = match settings.filter {
                            MotionFilter::None => sample.transform,
                            _ => {
                                homography_from_quads(&reference, quad).unwrap_or(sample.transform)
                            }
                        };
                        motion.restricted(anchor, settings.components)
                    })
                    .collect();
                (segment.start(), transforms)
            })
            .collect();

        Self {
            reference,
            segments,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Motion from the reference frame to `frame`.
    pub fn transform_at(&self, frame: u32) -> Transform {
        let held = self
            .segments
            .iter()
            .rev()
            .find(|(start, _)| *start <= frame)
            .or(self.segments.first());
        let Some((start, transforms)) = held else {
            return Transform::IDENTITY;
        };
        let index = (frame.saturating_sub(*start) as usize).min(transforms.len() - 1);
        transforms[index]
    }

    /// The tracked region carried to `frame`.
    pub fn region_at(&self, frame: u32) -> Quad {
        self.reference.map(&self.transform_at(frame))
    }

    /// A layer placed with `layer` (its own coordinates to the canvas) in the
    /// reference frame, moved along with the target.
    pub fn layer_transform(&self, frame: u32, layer: &Transform) -> Transform {
        layer.then(&self.transform_at(frame))
    }

    /// Maps a `width` x `height` layer onto the tracked region (corner pin).
    pub fn corner_pin(&self, frame: u32, width: f64, height: f64) -> Option<Transform> {
        homography_from_quads(
            &Quad::from_rect(0.0, 0.0, width, height),
            &self.region_at(frame),
        )
    }

    /// The transform to apply to the video at `frame` so the target stays
    /// where it was in the reference frame. With [`MotionFilter::Lock`] this
    /// locks the shot onto the target; with [`MotionFilter::Stabilize`] it
    /// removes only the shake around it.
    pub fn reframe(&self, frame: u32) -> Transform {
        self.transform_at(frame)
            .inverse()
            .unwrap_or(Transform::IDENTITY)
    }
}

fn filter_corners(corners: &[Quad], reference: &Quad, settings: &MotionSettings) -> Vec<Quad> {
    if settings.filter == MotionFilter::None {
        return corners.to_vec();
    }
    let mut out = corners.to_vec();
    for corner in 0..4 {
        for axis in 0..2 {
            let get = |p: Point| if axis == 0 { p.x } else { p.y };
            let values: Vec<f64> = corners.iter().map(|q| get(q.corners[corner])).collect();
            let rest = get(reference.corners[corner]);
            let filtered = settings.filter.apply(&values, rest, settings.strength);
            for (quad, value) in out.iter_mut().zip(filtered) {
                let point = &mut quad.corners[corner];
                if axis == 0 {
                    point.x = value;
                } else {
                    point.y = value;
                }
            }
        }
    }
    out
}

/// `values[index]`, extended past either end by point reflection so straight
/// paths stay straight through the filters' edges.
fn padded(values: &[f64], index: isize) -> f64 {
    let n = values.len() as isize;
    if index < 0 {
        2.0 * values[0] - values[(-index).min(n - 1) as usize]
    } else if index >= n {
        2.0 * values[(n - 1) as usize] - values[(2 * (n - 1) - index).max(0) as usize]
    } else {
        values[index as usize]
    }
}

/// Zero-lag Gaussian smoothing over `sigma` frames.
pub(crate) fn gaussian(values: &[f64], sigma: f64) -> Vec<f64> {
    if sigma < 0.3 || values.len() < 2 {
        return values.to_vec();
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let weights: Vec<f64> = (-radius..=radius)
        .map(|k| (-(k * k) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    (0..values.len() as isize)
        .map(|i| {
            (-radius..=radius)
                .zip(&weights)
                .map(|(k, w)| w * padded(values, i + k))
                .sum::<f64>()
                / total
        })
        .collect()
}

fn median(values: &[f64], radius: usize) -> Vec<f64> {
    let r = radius as isize;
    (0..values.len() as isize)
        .map(|i| {
            let mut window: Vec<f64> = (-r..=r).map(|k| padded(values, i + k)).collect();
            window.sort_by(f64::total_cmp);
            window[window.len() / 2]
        })
        .collect()
}

/// Two cascaded exponential followers: a soft, slightly delayed chase.
fn follow(values: &[f64], retention: f64) -> Vec<f64> {
    let Some(&first) = values.first() else {
        return Vec::new();
    };
    let (mut fast, mut slow) = (first, first);
    values
        .iter()
        .map(|&value| {
            fast = retention * fast + (1.0 - retention) * value;
            slow = retention * slow + (1.0 - retention) * fast;
            slow
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::MotionModel;
    use crate::testing::assert_points_close;
    use crate::track::{Continuity, TrackSample};

    fn jitter(t: usize) -> f64 {
        // Deterministic, zero-mean ±1.5 px shake.
        [1.5, -1.0, 0.5, -1.5, 1.0, -0.5, 0.0][t % 7]
    }

    /// A target moving right at 3 px per frame with shake.
    fn shaky_track() -> Track {
        let reference = Quad::from_rect(100.0, 100.0, 20.0, 20.0);
        let mut track = Track::new(reference, MotionModel::Translation);
        for t in 0..60u32 {
            let transform =
                Transform::translation(3.0 * t as f64 + jitter(t as usize), jitter(t as usize + 3));
            track.record(
                t,
                TrackSample {
                    transform,
                    quad: reference.map(&transform),
                    confidence: 1.0,
                    continuity: Continuity::Tracked,
                },
            );
        }
        track
    }

    fn settings(filter: MotionFilter, strength: f64) -> MotionSettings {
        MotionSettings {
            filter,
            strength,
            ..MotionSettings::default()
        }
    }

    fn offset(path: &MotionPath, frame: u32) -> Point {
        path.transform_at(frame).apply(Point::default())
    }

    /// Mean distance from the jitter-free path over the middle frames.
    fn deviation(path: &MotionPath, ideal: impl Fn(u32) -> Point) -> f64 {
        (10..50)
            .map(|t| offset(path, t).distance(ideal(t)))
            .sum::<f64>()
            / 40.0
    }

    #[test]
    fn smooth_removes_shake_without_lag() {
        let track = shaky_track();
        let raw = MotionPath::new(&track, &settings(MotionFilter::None, 0.0));
        let smooth = MotionPath::new(&track, &settings(MotionFilter::Smooth, 0.6));
        let ideal = |t: u32| Point::new(3.0 * t as f64, 0.0);
        assert!(deviation(&smooth, ideal) < deviation(&raw, ideal) / 3.0);
        assert!(deviation(&smooth, ideal) < 0.3);
    }

    #[test]
    fn follow_lags_behind() {
        let track = shaky_track();
        let follow = MotionPath::new(&track, &settings(MotionFilter::Follow, 0.8));
        let lag = (20..50)
            .map(|t| 3.0 * t as f64 - offset(&follow, t).x)
            .sum::<f64>()
            / 30.0;
        assert!(lag > 3.0, "lag {lag}");
    }

    #[test]
    fn lock_holds_tighter_than_raw() {
        let track = shaky_track();
        let raw = MotionPath::new(&track, &settings(MotionFilter::None, 0.0));
        let lock = MotionPath::new(&track, &settings(MotionFilter::Lock, 1.0));
        let ideal = |t: u32| Point::new(3.0 * t as f64, 0.0);
        assert!(deviation(&lock, ideal) < deviation(&raw, ideal) / 2.0);
    }

    #[test]
    fn stabilize_keeps_only_the_shake() {
        let track = shaky_track();
        let stabilize = MotionPath::new(&track, &settings(MotionFilter::Stabilize, 0.6));
        // The intended pan is gone; what is left is close to the shake.
        let mean_x = (10..50).map(|t| offset(&stabilize, t).x).sum::<f64>() / 40.0;
        assert!(mean_x.abs() < 0.5, "{mean_x}");
        let residual = (10..50)
            .map(|t| (offset(&stabilize, t).x - jitter(t as usize)).abs())
            .sum::<f64>()
            / 40.0;
        assert!(residual < 0.6, "{residual}");
        // Reframing with it cancels the shake at the target.
        let frame = 20;
        let target = track.sample(frame).unwrap().quad.center();
        let steadied = stabilize.reframe(frame).apply(target);
        assert!((steadied.x - (110.0 + 3.0 * frame as f64)).abs() < 0.8);
    }

    #[test]
    fn frozen_components_and_holding_outside_the_track() {
        let track = shaky_track();
        let path = MotionPath::new(
            &track,
            &MotionSettings {
                components: MotionComponents {
                    vertical: false,
                    ..MotionComponents::ALL
                },
                ..MotionSettings::default()
            },
        );
        assert!(offset(&path, 7).y.abs() < 1e-9);
        assert!((offset(&path, 7).x - (21.0 + jitter(7))).abs() < 1e-9);
        assert_eq!(path.transform_at(500), path.transform_at(59));
        let pinned = path.corner_pin(0, 40.0, 40.0).unwrap();
        assert_points_close(
            pinned.apply(Point::new(40.0, 40.0)),
            path.region_at(0).corners[2],
            1e-6,
        );
    }
}
