use crate::TrackingError;
use crate::estimate::{MotionModel, Rng, RobustFit, fit, fit_robust, homography_from_quads};
use crate::features::{FeatureParams, detect};
use crate::flow::{FlowParams, coarse_shift, refine_against_reference, track_points};
use crate::frame::{Frame, Level, Plane, Pyramid, analysis_factor};
use crate::geometry::{MotionComponents, Point, Quad, Transform};
use crate::object::ObjectTracker;
use crate::track::{Continuity, TrackSample};

const PYRAMID_LEVELS: usize = 4;
const RANSAC_THRESHOLD: f64 = 2.0;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum TrackerKind {
    /// Corner points followed with optical flow. Precise on clear, steady
    /// footage; less reliable with fast motion or blur.
    #[default]
    Features,
    /// Template correlation. Robust to fast motion and low-quality footage,
    /// prioritizing stability over precision, and re-finds a lost object.
    Object,
    /// The dominant motion of the whole frame, i.e. the camera, including
    /// affine and perspective changes.
    Frame,
}

impl TrackerKind {
    pub const ALL: [Self; 3] = [Self::Features, Self::Object, Self::Frame];

    pub fn label(self) -> &'static str {
        match self {
            Self::Features => "Features",
            Self::Object => "Object",
            Self::Frame => "Frame",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Features => {
                "Precise on clear, steady videos. Less reliable with fast motion or blur."
            }
            Self::Object => {
                "Reliable with fast motion or lower-quality footage, prioritizing stability."
            }
            Self::Frame => "Tracks dominant motion across the full frame, including perspective.",
        }
    }
}

/// The selector drawn around the target, and how it follows the motion.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum RegionShape {
    /// Follows size, without rotation or perspective.
    Circle,
    /// Follows size and rotation.
    #[default]
    Rectangle,
    /// Also follows perspective, with corner pin.
    Quad,
}

impl RegionShape {
    pub const ALL: [Self; 3] = [Self::Circle, Self::Rectangle, Self::Quad];

    pub fn label(self) -> &'static str {
        match self {
            Self::Circle => "Circle",
            Self::Rectangle => "Rectangle",
            Self::Quad => "Quad",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerSettings {
    pub kind: TrackerKind,
    /// The motion the track records.
    pub motion_model: MotionModel,
    pub region_shape: RegionShape,
    /// Resize the region as the tracked surface changes size.
    pub follow_size: bool,
    /// Rotate a rectangle or quad region with the tracked surface.
    pub follow_rotation: bool,
    /// Let all four corners of a quad region follow perspective.
    pub corner_pin: bool,
    /// Longest side, in pixels, of the frames analysis runs on. Higher
    /// tracks finer detail but takes longer.
    pub analysis_size: usize,
    /// How many points the feature and frame trackers keep alive.
    pub max_points: usize,
}

impl Default for TrackerSettings {
    fn default() -> Self {
        Self {
            kind: TrackerKind::default(),
            motion_model: MotionModel::default(),
            region_shape: RegionShape::default(),
            follow_size: true,
            follow_rotation: true,
            corner_pin: false,
            analysis_size: 960,
            max_points: 150,
        }
    }
}

impl TrackerSettings {
    pub fn corner_pin_active(&self) -> bool {
        self.corner_pin && self.region_shape == RegionShape::Quad
    }

    /// The parts of the measured motion the region itself follows. Corner
    /// pin implies size and rotation.
    pub fn region_components(&self) -> MotionComponents {
        let pin = self.corner_pin_active();
        MotionComponents {
            horizontal: true,
            vertical: true,
            scale: self.follow_size || pin,
            rotation: (self.follow_rotation && self.region_shape != RegionShape::Circle) || pin,
            affine: pin,
            perspective: pin,
        }
    }

    /// The richest model either the track or the region needs measured.
    fn measured_model(&self) -> MotionModel {
        let region = self.region_components();
        let region_model = if region.perspective {
            MotionModel::Perspective
        } else if region.scale || region.rotation {
            MotionModel::Similarity
        } else {
            MotionModel::Translation
        };
        self.motion_model.max(region_model)
    }
}

/// The components of a motion that `model` can express.
pub(crate) fn model_components(model: MotionModel) -> MotionComponents {
    match model {
        MotionModel::Translation => MotionComponents::POSITION,
        MotionModel::Similarity => MotionComponents {
            scale: true,
            rotation: true,
            ..MotionComponents::POSITION
        },
        MotionModel::Affine => MotionComponents {
            perspective: false,
            ..MotionComponents::ALL
        },
        MotionModel::Perspective => MotionComponents::ALL,
    }
}

/// Follows a region through a clip, one frame at a time, in either
/// direction: feed frames in reverse to track backward.
pub struct Tracker {
    settings: TrackerSettings,
    width: usize,
    height: usize,
    to_analysis: Transform,
    reference: Quad,
    engine: Engine,
    last: TrackSample,
}

enum Engine {
    Points(Box<PointTracker>),
    Object(ObjectTracker),
}

impl Tracker {
    /// Starts tracking `region` (frame pixels, pixel centers on integers) as
    /// it appears in `frame`.
    pub fn new(
        settings: TrackerSettings,
        frame: &Frame,
        region: Quad,
    ) -> Result<Self, TrackingError> {
        let (width, height) = (frame.width(), frame.height());
        let (min, max) = region.bounds();
        let inside =
            min.x >= -0.5 && min.y >= -0.5 && max.x <= width as f64 && max.y <= height as f64;
        if !inside || !region.is_convex() || region.area().abs() < 16.0 {
            return Err(TrackingError::InvalidRegion);
        }

        let factor = analysis_factor(width, height, settings.analysis_size);
        let offset = (factor as f64 - 1.0) / 2.0;
        let to_analysis =
            Transform::translation(-offset, -offset).then(&Transform::scaling(1.0 / factor as f64));
        let pyramid = Pyramid::new(Plane::from_frame(frame, factor), PYRAMID_LEVELS);
        let analysis_region = region.map(&to_analysis);

        let engine = match settings.kind {
            TrackerKind::Features => Engine::Points(Box::new(PointTracker::region(
                pyramid,
                analysis_region,
                settings.region_shape,
                settings.measured_model(),
                settings.max_points,
            )?)),
            TrackerKind::Frame => Engine::Points(Box::new(PointTracker::full_frame(
                pyramid,
                settings.measured_model(),
                settings.max_points * 2,
            )?)),
            TrackerKind::Object => {
                let (min, max) = analysis_region.bounds();
                let follow_scale =
                    settings.follow_size || settings.motion_model != MotionModel::Translation;
                Engine::Object(
                    ObjectTracker::new(
                        &pyramid,
                        analysis_region.center(),
                        (max.x - min.x, max.y - min.y),
                        follow_scale,
                    )
                    .ok_or(TrackingError::LowDetail)?,
                )
            }
        };

        Ok(Self {
            settings,
            width,
            height,
            to_analysis,
            reference: region,
            engine,
            last: TrackSample::reference(region),
        })
    }

    pub fn settings(&self) -> &TrackerSettings {
        &self.settings
    }

    /// The region as it was given, in the frame tracking started on.
    pub fn reference(&self) -> Quad {
        self.reference
    }

    /// The most recent result.
    pub fn last(&self) -> &TrackSample {
        &self.last
    }

    /// Follows the region into the next frame. When the target cannot be
    /// found the result holds its last position and is marked
    /// [`Continuity::Searching`]; the object tracker keeps looking.
    pub fn track(&mut self, frame: &Frame) -> Result<TrackSample, TrackingError> {
        if frame.width() != self.width || frame.height() != self.height {
            return Err(TrackingError::FrameSizeMismatch);
        }
        let factor = analysis_factor(self.width, self.height, self.settings.analysis_size);
        let pyramid = Pyramid::new(Plane::from_frame(frame, factor), PYRAMID_LEVELS);

        let (analysis_motion, tracked, confidence) = match &mut self.engine {
            Engine::Points(points) => {
                let step = points.step(pyramid);
                (step.motion, step.tracked, step.confidence)
            }
            Engine::Object(object) => {
                let step = object.step(&pyramid);
                (step.motion, step.tracked, step.confidence)
            }
        };

        let from_analysis = self.to_analysis.inverse().unwrap_or(Transform::IDENTITY);
        let measured = self.to_analysis.then(&analysis_motion).then(&from_analysis);
        let anchor = self.reference.center();
        let sample = TrackSample {
            transform: measured.restricted(anchor, model_components(self.settings.motion_model)),
            quad: self
                .reference
                .map(&measured.restricted(anchor, self.settings.region_components())),
            confidence,
            continuity: if tracked {
                Continuity::Tracked
            } else {
                Continuity::Searching
            },
        };
        self.last = sample;
        Ok(sample)
    }
}

struct PointStep {
    motion: Transform,
    tracked: bool,
    confidence: f32,
}

struct Attempt {
    /// Reference positions (feature tracking) or previous positions (frame
    /// tracking) of the points that were followed.
    from: Vec<Point>,
    to: Vec<Point>,
    fit: Option<(RobustFit, Transform)>,
}

/// Shared engine of the feature tracker (points pinned to the reference
/// frame) and the frame tracker (points chained frame to frame).
struct PointTracker {
    full_frame: bool,
    model: MotionModel,
    /// The frame tracking started on, which feature tracking re-matches
    /// against.
    reference_level: Option<Level>,
    prev: Pyramid,
    points: Vec<Point>,
    /// Where each point was in the reference frame (feature tracking).
    references: Vec<Point>,
    /// Reference frame to the current frame.
    motion: Transform,
    /// Previous frame to the current frame, used to predict the next one.
    last_step: Transform,
    region: Quad,
    shape: RegionShape,
    target: usize,
    rng: Rng,
}

impl PointTracker {
    fn region(
        pyramid: Pyramid,
        region: Quad,
        shape: RegionShape,
        model: MotionModel,
        target: usize,
    ) -> Result<Self, TrackingError> {
        let mut tracker = Self {
            full_frame: false,
            model,
            reference_level: Some(pyramid.base().clone()),
            prev: pyramid,
            points: Vec::new(),
            references: Vec::new(),
            motion: Transform::IDENTITY,
            last_step: Transform::IDENTITY,
            region,
            shape,
            target: target.max(8),
            rng: Rng::new(0x5EED),
        };
        tracker.replenish(true);
        if tracker.points.len() < tracker.min_inliers() {
            return Err(TrackingError::LowDetail);
        }
        Ok(tracker)
    }

    fn full_frame(
        pyramid: Pyramid,
        model: MotionModel,
        target: usize,
    ) -> Result<Self, TrackingError> {
        let (w, h) = (
            pyramid.base().image.width as f64,
            pyramid.base().image.height as f64,
        );
        let mut tracker = Self {
            full_frame: true,
            model,
            reference_level: None,
            prev: pyramid,
            points: Vec::new(),
            references: Vec::new(),
            motion: Transform::IDENTITY,
            last_step: Transform::IDENTITY,
            region: Quad::from_rect(0.0, 0.0, w - 1.0, h - 1.0),
            shape: RegionShape::Rectangle,
            target: target.max(16),
            rng: Rng::new(0x5EED),
        };
        tracker.replenish(true);
        if tracker.points.len() < tracker.min_inliers() {
            return Err(TrackingError::LowDetail);
        }
        Ok(tracker)
    }

    /// Inliers needed before a fit is trusted.
    fn min_inliers(&self) -> usize {
        self.model.minimal_points() * 2 + 2
    }

    fn step(&mut self, next: Pyramid) -> PointStep {
        // A region keeps moving the way it was going. The camera's shake does
        // not, so frame tracking starts from the whole frame's shift instead.
        let guide = if self.full_frame {
            let shift = coarse_shift(&self.prev, &next);
            Transform::translation(shift.x, shift.y)
        } else {
            self.last_step
        };
        let guesses: Vec<Point> = self.points.iter().map(|&p| guide.apply(p)).collect();
        let mut attempt = self.attempt(&next, Some(&guesses));
        // When the guess leads nowhere, try again from where the points were.
        if attempt.fit.is_none() && guide != Transform::IDENTITY {
            let unpredicted = self.attempt(&next, None);
            if unpredicted.fit.is_some() || unpredicted.to.len() > attempt.to.len() {
                attempt = unpredicted;
            }
        }

        let result = match attempt.fit {
            Some((fit, motion)) => {
                let keep = |values: &[Point]| -> Vec<Point> {
                    values
                        .iter()
                        .zip(&fit.inliers)
                        .filter(|(_, inlier)| **inlier)
                        .map(|(p, _)| *p)
                        .collect()
                };
                let (from, mut to) = (keep(&attempt.from), keep(&attempt.to));
                let mut motion = motion;
                if let Some(reference) = &self.reference_level
                    && let Some((refined, refined_motion)) =
                        self.refine(reference, next.base(), &from, &to, &motion)
                {
                    to = refined;
                    motion = refined_motion;
                }

                let confidence = (fit.inlier_count as f32 / self.target as f32).min(1.0)
                    * (fit.inlier_count as f32 / attempt.from.len() as f32);
                self.points = to;
                if !self.full_frame {
                    self.references = from;
                }
                self.last_step = self
                    .motion
                    .inverse()
                    .map_or(Transform::IDENTITY, |inverse| inverse.then(&motion));
                self.motion = motion;
                PointStep {
                    motion,
                    tracked: true,
                    confidence,
                }
            }
            None => {
                // Keep following the survivors and hold the last motion; they
                // may agree again once an occlusion clears.
                self.points = attempt.to;
                if !self.full_frame {
                    self.references = attempt.from;
                }
                self.last_step = Transform::IDENTITY;
                PointStep {
                    motion: self.motion,
                    tracked: false,
                    confidence: 0.0,
                }
            }
        };
        self.prev = next;

        // Frame tracking can restart from whatever is visible now; a lost
        // region cannot, since where it went is unknown.
        if (result.tracked || self.full_frame) && self.points.len() < self.target * 3 / 5 {
            self.replenish(false);
        }
        result
    }

    /// Follows the points into `next` and fits the model to the survivors.
    fn attempt(&mut self, next: &Pyramid, guesses: Option<&[Point]>) -> Attempt {
        let tracked = track_points(
            &self.prev,
            next,
            &self.points,
            guesses,
            &FlowParams::default(),
        );
        let mut from = Vec::with_capacity(tracked.len());
        let mut to = Vec::with_capacity(tracked.len());
        for (i, result) in tracked.into_iter().enumerate() {
            if let Some(point) = result {
                from.push(if self.full_frame {
                    self.points[i]
                } else {
                    self.references[i]
                });
                to.push(point);
            }
        }

        let fit = (from.len() >= self.min_inliers())
            .then(|| fit_robust(self.model, &from, &to, RANSAC_THRESHOLD, &mut self.rng))
            .flatten()
            .filter(|fit| fit.inlier_count >= self.min_inliers())
            .and_then(|fit| {
                let motion = if self.full_frame {
                    self.motion.then(&fit.transform)
                } else {
                    fit.transform
                };
                self.plausible(&motion).then_some((fit, motion))
            });
        Attempt { from, to, fit }
    }

    /// Re-matches each point against its look in the reference frame and
    /// refits the motion, or `None` if too few points re-match.
    fn refine(
        &self,
        reference: &Level,
        next: &Level,
        references: &[Point],
        points: &[Point],
        motion: &Transform,
    ) -> Option<(Vec<Point>, Transform)> {
        let to_reference = motion.inverse()?;
        let params = FlowParams::default();
        let refined: Vec<Point> = references
            .iter()
            .zip(points)
            .map(|(&reference_point, &point)| {
                refine_against_reference(
                    reference,
                    &to_reference,
                    next,
                    motion.apply(reference_point),
                    point,
                    &params,
                )
                .unwrap_or(point)
            })
            .collect();
        let refit = fit(self.model, references, &refined)?;
        let consistent = references
            .iter()
            .zip(&refined)
            .filter(|(r, p)| refit.apply(**r).distance(**p) <= RANSAC_THRESHOLD)
            .count();
        (consistent >= self.min_inliers() && self.plausible(&refit)).then_some((refined, refit))
    }

    /// Detects new points inside the region as it is now, so tracking keeps
    /// enough of them as the old ones are lost.
    fn replenish(&mut self, initial: bool) {
        let level = self.prev.base();
        let current = if self.full_frame {
            self.region
        } else {
            self.region.map(&self.motion)
        };
        let (min, max) = current.bounds();
        let area = current.area().abs().max(1.0);
        let min_distance = ((area / self.target as f64).sqrt() * 0.7).clamp(3.0, 24.0);
        let to_unit = (self.shape == RegionShape::Circle)
            .then(|| homography_from_quads(&current, &Quad::from_rect(0.0, 0.0, 1.0, 1.0)))
            .flatten();
        let mask = |p: Point| match &to_unit {
            Some(to_unit) => {
                let u = to_unit.apply(p);
                (u.x - 0.5).powi(2) + (u.y - 0.5).powi(2) <= 0.25
            }
            None => self.full_frame || current.contains(p),
        };
        let params = FeatureParams {
            max_count: self.target.saturating_sub(self.points.len()),
            quality: 0.02,
            min_distance,
        };
        let found = detect(level, min, max, mask, &self.points, &params);

        if self.full_frame {
            // Frame tracking chains frame to frame and keeps no references.
        } else if initial {
            self.references.extend(&found);
        } else {
            let Some(inverse) = self.motion.inverse() else {
                return;
            };
            self.references
                .extend(found.iter().map(|&p| inverse.apply(p)));
        }
        self.points.extend(found);
    }

    /// Rejects motions no rigid surface can make: the region flipping,
    /// collapsing, exploding or crossing the horizon.
    fn plausible(&self, motion: &Transform) -> bool {
        let mapped = self.region.map(motion);
        let ratio = mapped.area() / self.region.area();
        self.region
            .corners
            .iter()
            .all(|&corner| motion.depth(corner) > 0.0)
            && mapped.is_convex()
            && (0.04..=25.0).contains(&ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Texture, assert_points_close, render_object};

    const W: usize = 320;
    const H: usize = 240;

    fn settings(kind: TrackerKind, motion_model: MotionModel) -> TrackerSettings {
        TrackerSettings {
            kind,
            motion_model,
            ..TrackerSettings::default()
        }
    }

    /// Texture motion at frame `t`: a drift with rotation and growth about
    /// the middle of the frame.
    fn similarity_motion(t: usize) -> Transform {
        let t = t as f64;
        Transform::similarity(1.0 + 0.008 * t, 0.012 * t, 0.0, 0.0)
            .about(Point::new(160.0, 120.0))
            .then(&Transform::translation(2.5 * t, -1.5 * t))
    }

    fn assert_quads_close(actual: &Quad, expected: &Quad, tolerance: f64) {
        for (a, e) in actual.corners.iter().zip(&expected.corners) {
            assert_points_close(*a, *e, tolerance);
        }
    }

    #[test]
    fn features_follow_a_similarity_motion() {
        let texture = Texture::new(11);
        let region = Quad::from_rect(110.0, 80.0, 100.0, 80.0);
        let mut tracker = Tracker::new(
            settings(TrackerKind::Features, MotionModel::Similarity),
            &texture.render(W, H, &similarity_motion(0)),
            region,
        )
        .unwrap();

        for t in 1..=12 {
            let sample = tracker
                .track(&texture.render(W, H, &similarity_motion(t)))
                .unwrap();
            assert_eq!(sample.continuity, Continuity::Tracked, "frame {t}");
            assert_quads_close(
                &region.map(&sample.transform),
                &region.map(&similarity_motion(t)),
                0.6,
            );
            assert_quads_close(&sample.quad, &region.map(&similarity_motion(t)), 0.6);
        }
    }

    #[test]
    fn features_measure_perspective() {
        let texture = Texture::new(5);
        let region = Quad::from_rect(100.0, 70.0, 120.0, 100.0);
        let motion = |t: usize| {
            let t = t as f64;
            Transform::from_matrix([
                [1.0, 0.0, 1.5 * t],
                [0.0, 1.0, 0.5 * t],
                [0.00035 * t, 0.0002 * t, 1.0],
            ])
            .about(Point::new(160.0, 120.0))
        };
        let mut tracker = Tracker::new(
            TrackerSettings {
                region_shape: RegionShape::Quad,
                corner_pin: true,
                ..settings(TrackerKind::Features, MotionModel::Perspective)
            },
            &texture.render(W, H, &motion(0)),
            region,
        )
        .unwrap();

        let mut last = None;
        for t in 1..=8 {
            last = Some(tracker.track(&texture.render(W, H, &motion(t))).unwrap());
        }
        let sample = last.unwrap();
        assert_quads_close(&sample.quad, &region.map(&motion(8)), 0.8);
    }

    #[test]
    fn region_shape_limits_what_the_region_follows() {
        let texture = Texture::new(11);
        let region = Quad::from_rect(110.0, 80.0, 100.0, 80.0);
        let mut tracker = Tracker::new(
            TrackerSettings {
                region_shape: RegionShape::Circle,
                follow_size: false,
                ..settings(TrackerKind::Features, MotionModel::Similarity)
            },
            &texture.render(W, H, &similarity_motion(0)),
            region,
        )
        .unwrap();
        let mut sample = None;
        for t in 1..=6 {
            sample = Some(
                tracker
                    .track(&texture.render(W, H, &similarity_motion(t)))
                    .unwrap(),
            );
        }
        let sample = sample.unwrap();

        // The region only moves; the recorded motion still has it all.
        let center = similarity_motion(6).apply(region.center());
        assert_quads_close(
            &sample.quad,
            &region.translated(center - region.center()),
            0.6,
        );
        assert_quads_close(
            &region.map(&sample.transform),
            &region.map(&similarity_motion(6)),
            0.6,
        );
    }

    #[test]
    fn object_tracker_follows_and_recovers() {
        let texture = Texture::new(21);
        let rect = (60.0, 50.0, 60.0, 50.0);
        let region = Quad::from_rect(rect.0, rect.1, rect.2, rect.3);
        let path = |t: usize| Transform::translation(6.0 * t as f64, 3.0 * t as f64);
        let render = |t: usize| render_object(W, H, &texture, rect, &path(t), 90);
        let mut tracker = Tracker::new(
            settings(TrackerKind::Object, MotionModel::Translation),
            &render(0),
            region,
        )
        .unwrap();

        for t in 1..=10 {
            let sample = tracker.track(&render(t)).unwrap();
            assert_eq!(sample.continuity, Continuity::Tracked, "frame {t}");
            assert_points_close(sample.quad.center(), path(t).apply(region.center()), 1.0);
        }

        // The object disappears, then comes back somewhere else.
        let blank = Frame::from_luma(W, H, vec![90; W * H]).unwrap();
        assert_eq!(
            tracker.track(&blank).unwrap().continuity,
            Continuity::Searching
        );
        assert_eq!(
            tracker.track(&blank).unwrap().continuity,
            Continuity::Searching
        );
        let back = Transform::translation(90.0, 70.0);
        let sample = tracker
            .track(&render_object(W, H, &texture, rect, &back, 90))
            .unwrap();
        assert_eq!(sample.continuity, Continuity::Tracked);
        assert_points_close(sample.quad.center(), back.apply(region.center()), 1.0);
    }

    #[test]
    fn object_tracker_follows_size() {
        let texture = Texture::new(8);
        let rect = (120.0, 90.0, 80.0, 60.0);
        let region = Quad::from_rect(rect.0, rect.1, rect.2, rect.3);
        let path = |t: usize| Transform::scaling(1.0 + 0.03 * t as f64).about(region.center());
        let render = |t: usize| render_object(W, H, &texture, rect, &path(t), 60);
        let mut tracker = Tracker::new(
            settings(TrackerKind::Object, MotionModel::Similarity),
            &render(0),
            region,
        )
        .unwrap();
        let mut sample = None;
        for t in 1..=8 {
            sample = Some(tracker.track(&render(t)).unwrap());
        }
        let sample = sample.unwrap();
        let expected = region.map(&path(8));
        let width = |q: &Quad| q.corners[1].x - q.corners[0].x;
        assert!(
            (width(&sample.quad) - width(&expected)).abs() < 4.0,
            "{:?}",
            sample.quad
        );
    }

    #[test]
    fn frame_tracker_measures_camera_motion() {
        let texture = Texture::new(17);
        let region = Quad::from_rect(0.0, 0.0, W as f64 - 1.0, H as f64 - 1.0);
        let shake = |t: usize| {
            let t = t as f64;
            Transform::similarity(1.0, 0.01 * (t * 1.3).sin(), 0.0, 0.0)
                .about(Point::new(160.0, 120.0))
                .then(&Transform::translation(
                    4.0 * (t * 0.9).sin(),
                    3.0 * (t * 1.7).cos(),
                ))
        };
        let mut tracker = Tracker::new(
            settings(TrackerKind::Frame, MotionModel::Similarity),
            &texture.render(W, H, &shake(0)),
            region,
        )
        .unwrap();

        for t in 1..=10 {
            let sample = tracker.track(&texture.render(W, H, &shake(t))).unwrap();
            let expected = shake(0).inverse().unwrap().then(&shake(t));
            assert_quads_close(&region.map(&sample.transform), &region.map(&expected), 1.0);
        }
    }

    #[test]
    fn flat_regions_report_low_detail() {
        let flat = Frame::from_luma(W, H, vec![128; W * H]).unwrap();
        let region = Quad::from_rect(50.0, 50.0, 80.0, 80.0);
        for kind in TrackerKind::ALL {
            assert_eq!(
                Tracker::new(settings(kind, MotionModel::Similarity), &flat, region).err(),
                Some(TrackingError::LowDetail),
                "{kind:?}"
            );
        }
        assert_eq!(
            Tracker::new(
                TrackerSettings::default(),
                &flat,
                Quad::from_rect(300.0, 200.0, 80.0, 80.0)
            )
            .err(),
            Some(TrackingError::InvalidRegion)
        );
    }
}
