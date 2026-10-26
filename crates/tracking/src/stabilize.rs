//! Video stabilization: measure the camera's motion frame to frame, smooth
//! the path it took, and frame the corrected video.

use crate::TrackingError;
use crate::estimate::{MotionModel, homography_from_quads};
use crate::frame::Frame;
use crate::geometry::{MotionComponents, Point, Quad, Transform};
use crate::motion::gaussian;
use crate::track::Continuity;
use crate::tracker::{Tracker, TrackerKind, TrackerSettings, model_components};

/// How camera motion is estimated and corrected.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum StabilizationMode {
    /// Position, rotation and uniform zoom; for straightforward movement.
    Simple,
    /// Also stretch and skew; works for most handheld footage.
    #[default]
    Standard,
    /// A whole-clip camera path with perspective. Smoother, but may crop or
    /// distort more.
    Cinematic,
}

impl StabilizationMode {
    pub const ALL: [Self; 3] = [Self::Simple, Self::Standard, Self::Cinematic];

    pub fn label(self) -> &'static str {
        match self {
            Self::Simple => "Simple",
            Self::Standard => "Standard",
            Self::Cinematic => "Cinematic",
        }
    }

    fn components(self) -> MotionComponents {
        model_components(match self {
            Self::Simple => MotionModel::Similarity,
            Self::Standard => MotionModel::Affine,
            Self::Cinematic => MotionModel::Perspective,
        })
    }
}

/// How the empty edges stabilization uncovers are handled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum Framing {
    /// One fixed zoom that keeps edges out of frame everywhere. Steady, but
    /// crops the most.
    #[default]
    Crop,
    /// Zoom that varies over time, cropping less; the framing may breathe.
    AutoScale,
    /// No zoom; uncovered edges are filled per [`EdgeFill`].
    Extend,
}

impl Framing {
    pub const ALL: [Self; 3] = [Self::Crop, Self::AutoScale, Self::Extend];

    pub fn label(self) -> &'static str {
        match self {
            Self::Crop => "Cropping",
            Self::AutoScale => "Scaling",
            Self::Extend => "Extending",
        }
    }
}

/// What uncovered pixels show.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum EdgeFill {
    #[default]
    Black,
    /// Repeat the nearest edge pixel.
    Clamp,
    /// Reflect the image at its edge.
    Mirror,
}

impl EdgeFill {
    pub const ALL: [Self; 3] = [Self::Black, Self::Clamp, Self::Mirror];

    pub fn label(self) -> &'static str {
        match self {
            Self::Black => "Black",
            Self::Clamp => "Clamp",
            Self::Mirror => "Mirror",
        }
    }
}

/// How many points analysis follows. Higher is slower but more reliable.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum AnalysisQuality {
    Fast,
    #[default]
    Balanced,
    High,
}

impl AnalysisQuality {
    pub const ALL: [Self; 3] = [Self::Fast, Self::Balanced, Self::High];

    pub fn label(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Balanced => "Balanced",
            Self::High => "High",
        }
    }

    fn points(self) -> usize {
        match self {
            Self::Fast => 60,
            Self::Balanced => 120,
            Self::High => 240,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StabilizerSettings {
    pub mode: StabilizationMode,
    pub framing: Framing,
    pub edge_fill: EdgeFill,
    /// How much of the measured shake is corrected, 0 to 1.
    pub amount: f64,
    pub quality: AnalysisQuality,
    /// Longest side, in pixels, of the frames analysis runs on.
    pub analysis_size: usize,
}

impl Default for StabilizerSettings {
    fn default() -> Self {
        Self {
            mode: StabilizationMode::default(),
            framing: Framing::default(),
            edge_fill: EdgeFill::default(),
            amount: 0.6,
            quality: AnalysisQuality::default(),
            analysis_size: 640,
        }
    }
}

/// The camera's motion from one frame to the next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairMotion {
    /// Maps frame `i` coordinates to frame `i + 1`.
    pub transform: Transform,
    /// 0 to 1; 0 marks a weak pair that could not be measured.
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraMotion {
    pub width: usize,
    pub height: usize,
    /// `pairs[i]` takes frame `i` to frame `i + 1`.
    pub pairs: Vec<PairMotion>,
}

impl CameraMotion {
    pub fn frame_count(&self) -> usize {
        self.pairs.len() + 1
    }
}

/// Measures camera motion as frames are decoded.
pub struct CameraMotionEstimator {
    settings: TrackerSettings,
    tracker: Option<Tracker>,
    size: Option<(usize, usize)>,
    frames: usize,
    previous: Transform,
    pairs: Vec<PairMotion>,
}

impl CameraMotionEstimator {
    pub fn new(settings: &StabilizerSettings) -> Self {
        Self {
            settings: TrackerSettings {
                kind: TrackerKind::Frame,
                motion_model: MotionModel::Perspective,
                analysis_size: settings.analysis_size,
                max_points: settings.quality.points(),
                ..TrackerSettings::default()
            },
            tracker: None,
            size: None,
            frames: 0,
            previous: Transform::IDENTITY,
            pairs: Vec::new(),
        }
    }

    pub fn push(&mut self, frame: &Frame) -> Result<(), TrackingError> {
        let size = (frame.width(), frame.height());
        if *self.size.get_or_insert(size) != size {
            return Err(TrackingError::FrameSizeMismatch);
        }
        let index = self.frames;
        self.frames += 1;
        let pair = match &mut self.tracker {
            Some(tracker) => {
                let sample = tracker.track(frame)?;
                let step = self
                    .previous
                    .inverse()
                    .map_or(Transform::IDENTITY, |inverse| {
                        inverse.then(&sample.transform)
                    });
                self.previous = sample.transform;
                match sample.continuity {
                    Continuity::Tracked => PairMotion {
                        transform: step,
                        confidence: sample.confidence,
                    },
                    _ => PairMotion {
                        transform: Transform::IDENTITY,
                        confidence: 0.0,
                    },
                }
            }
            None => {
                // Start (or restart after featureless frames) on this one.
                let region = Quad::from_rect(0.0, 0.0, size.0 as f64 - 1.0, size.1 as f64 - 1.0);
                self.tracker = Tracker::new(self.settings, frame, region).ok();
                self.previous = Transform::IDENTITY;
                PairMotion {
                    transform: Transform::IDENTITY,
                    confidence: 0.0,
                }
            }
        };
        if index > 0 {
            self.pairs.push(pair);
        }
        Ok(())
    }

    pub fn finish(self) -> CameraMotion {
        let (width, height) = self.size.unwrap_or((0, 0));
        CameraMotion {
            width,
            height,
            pairs: self.pairs,
        }
    }
}

/// Per-frame corrections for a clip.
#[derive(Clone, Debug, PartialEq)]
pub struct Stabilization {
    /// Source frame to output frame, for each frame.
    pub transforms: Vec<Transform>,
    /// The zoom included in each transform (1 means none).
    pub zooms: Vec<f64>,
    pub edge_fill: EdgeFill,
    /// Frames whose motion could not be measured and were left as they were.
    pub weak_frames: usize,
}

impl Stabilization {
    pub fn apply(&self, index: usize, frame: &Frame, black: u8) -> Frame {
        let transform = self
            .transforms
            .get(index)
            .copied()
            .unwrap_or(Transform::IDENTITY);
        warp(frame, &transform, self.edge_fill, black)
    }
}

const MAX_ZOOM: f64 = 3.0;

/// Smooths the measured camera path and works out each frame's correction.
pub fn stabilize(motion: &CameraMotion, settings: &StabilizerSettings) -> Stabilization {
    let (w, h) = (motion.width as f64, motion.height as f64);
    let center = Point::new((w - 1.0) / 2.0, (h - 1.0) / 2.0);
    let frame = Quad::from_rect(0.0, 0.0, w - 1.0, h - 1.0);
    let components = settings.mode.components();
    let amount = settings.amount.clamp(0.0, 1.0);

    let mut weak_frames = 0;
    let mut path = Vec::with_capacity(motion.frame_count());
    path.push(Transform::IDENTITY);
    for pair in &motion.pairs {
        let step = if pair.confidence <= 0.0 {
            weak_frames += 1;
            Transform::IDENTITY
        } else {
            pair.transform.restricted(center, components)
        };
        let next = path[path.len() - 1].then(&step);
        path.push(next);
    }

    let corners: Vec<Quad> = path.iter().map(|camera| frame.map(camera)).collect();
    let smoothed = smooth_path(&corners, settings.mode, amount);
    let corrections: Vec<Transform> = corners
        .iter()
        .zip(&smoothed)
        .map(|(actual, smooth)| {
            homography_from_quads(actual, smooth)
                .map(|correction| correction.restricted(center, components))
                .unwrap_or(Transform::IDENTITY)
        })
        .collect();

    let needed: Vec<f64> = corrections
        .iter()
        .map(|c| required_zoom(c, w, h, center))
        .collect();
    let zooms = match settings.framing {
        Framing::Extend => vec![1.0; needed.len()],
        Framing::Crop => vec![needed.iter().copied().fold(1.0, f64::max); needed.len()],
        Framing::AutoScale => breathing_zoom(&needed, 15),
    };
    let transforms = corrections
        .iter()
        .zip(&zooms)
        .map(|(correction, &zoom)| correction.then(&Transform::scaling(zoom).about(center)))
        .collect();

    Stabilization {
        transforms,
        zooms,
        edge_fill: settings.edge_fill,
        weak_frames,
    }
}

fn smooth_path(corners: &[Quad], mode: StabilizationMode, amount: f64) -> Vec<Quad> {
    let n = corners.len();
    let mut out = corners.to_vec();
    for corner in 0..4 {
        for axis in 0..2 {
            let get = |p: Point| if axis == 0 { p.x } else { p.y };
            let values: Vec<f64> = corners.iter().map(|q| get(q.corners[corner])).collect();
            let target = match mode {
                StabilizationMode::Simple => gaussian(&values, 25.0 * amount),
                StabilizationMode::Standard => gaussian(&values, 35.0 * amount),
                StabilizationMode::Cinematic => {
                    // Keep the clip's overall drift (a steady pan) and smooth
                    // everything else over a long window.
                    let trend = linear_trend(&values);
                    let residual: Vec<f64> =
                        values.iter().zip(&trend).map(|(v, t)| v - t).collect();
                    let smooth = gaussian(&residual, (n as f64 / 3.0).max(60.0));
                    trend.iter().zip(smooth).map(|(t, s)| t + s).collect()
                }
            };
            for (i, quad) in out.iter_mut().enumerate() {
                let value = match mode {
                    StabilizationMode::Cinematic => values[i] + amount * (target[i] - values[i]),
                    _ => target[i],
                };
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

fn linear_trend(values: &[f64]) -> Vec<f64> {
    let n = values.len() as f64;
    let mean_t = (n - 1.0) / 2.0;
    let mean_v = values.iter().sum::<f64>() / n;
    let (mut covariance, mut variance) = (0.0, 0.0);
    for (t, v) in values.iter().enumerate() {
        let dt = t as f64 - mean_t;
        covariance += dt * (v - mean_v);
        variance += dt * dt;
    }
    let slope = if variance > 0.0 {
        covariance / variance
    } else {
        0.0
    };
    (0..values.len())
        .map(|t| mean_v + slope * (t as f64 - mean_t))
        .collect()
}

/// The smallest zoom about the center after which `correction` leaves no
/// uncovered pixels in a `w` x `h` frame.
fn required_zoom(correction: &Transform, w: f64, h: f64, center: Point) -> f64 {
    let covered = |zoom: f64| {
        let output = correction.then(&Transform::scaling(zoom).about(center));
        let Some(inverse) = output.inverse() else {
            return false;
        };
        let (right, bottom) = (w - 1.0, h - 1.0);
        let edge = [
            Point::new(0.0, 0.0),
            Point::new(right / 2.0, 0.0),
            Point::new(right, 0.0),
            Point::new(right, bottom / 2.0),
            Point::new(right, bottom),
            Point::new(right / 2.0, bottom),
            Point::new(0.0, bottom),
            Point::new(0.0, bottom / 2.0),
        ];
        edge.iter().all(|&p| {
            let source = inverse.apply(p);
            inverse.depth(p) > 0.0
                && source.x >= -1e-6
                && source.y >= -1e-6
                && source.x <= right + 1e-6
                && source.y <= bottom + 1e-6
        })
    };
    if covered(1.0) {
        return 1.0;
    }
    if !covered(MAX_ZOOM) {
        return MAX_ZOOM;
    }
    let (mut low, mut high) = (1.0, MAX_ZOOM);
    for _ in 0..30 {
        let mid = 0.5 * (low + high);
        if covered(mid) {
            high = mid;
        } else {
            low = mid;
        }
    }
    high
}

/// A zoom that rises ahead of the frames that need it and eases off after,
/// never dropping below what any frame needs.
fn breathing_zoom(needed: &[f64], radius: usize) -> Vec<f64> {
    let peaks: Vec<f64> = (0..needed.len())
        .map(|i| {
            needed[i.saturating_sub(radius)..(i + radius + 1).min(needed.len())]
                .iter()
                .copied()
                .fold(1.0, f64::max)
        })
        .collect();
    gaussian(&peaks, radius as f64 / 2.0)
        .into_iter()
        .zip(needed)
        .map(|(smooth, &need)| smooth.max(need))
        .collect()
}

/// Resamples `frame` through `transform` (source to output). Pixels with no
/// source are filled per `edge_fill`, with `black` as the black level.
pub fn warp(frame: &Frame, transform: &Transform, edge_fill: EdgeFill, black: u8) -> Frame {
    let (width, height) = (frame.width(), frame.height());
    let Some(inverse) = transform.inverse() else {
        return frame.clone();
    };
    let data = frame.data();
    let (max_x, max_y) = ((width - 1) as f64, (height - 1) as f64);
    let fold = |value: f64, max: f64| -> f64 {
        match edge_fill {
            EdgeFill::Mirror if max > 0.0 => {
                let period = 2.0 * max;
                let wrapped = value.abs() % period;
                if wrapped > max {
                    period - wrapped
                } else {
                    wrapped
                }
            }
            _ => value.clamp(0.0, max),
        }
    };

    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let p = Point::new(x as f64, y as f64);
            let source = inverse.apply(p);
            let outside = inverse.depth(p) <= 0.0
                || source.x < -0.5
                || source.y < -0.5
                || source.x > max_x + 0.5
                || source.y > max_y + 0.5;
            if outside && edge_fill == EdgeFill::Black {
                out.push(black);
                continue;
            }
            let (sx, sy) = (fold(source.x, max_x), fold(source.y, max_y));
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
            let (fx, fy) = (sx - x0 as f64, sy - y0 as f64);
            let at = |x: usize, y: usize| data[y * width + x] as f64;
            let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
            let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
            out.push((top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8);
        }
    }
    Frame::from_luma(width, height, out).expect("same size as the source")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Texture;

    const W: usize = 320;
    const H: usize = 240;

    /// A slow pan with handheld shake: texture coordinates to frame `t`.
    fn camera(t: usize) -> Transform {
        let t = t as f64;
        let shake_x = 5.0 * (t * 1.9).sin() + 2.0 * (t * 4.1).cos();
        let shake_y = 4.0 * (t * 2.3).cos() + 1.5 * (t * 3.7).sin();
        Transform::similarity(1.0, 0.006 * (t * 2.7).sin(), 0.0, 0.0)
            .about(Point::new(160.0, 120.0))
            .then(&Transform::translation(-t + shake_x, shake_y))
    }

    fn measured(frames: usize) -> CameraMotion {
        let texture = Texture::new(31);
        let mut estimator = CameraMotionEstimator::new(&StabilizerSettings::default());
        for t in 0..frames {
            estimator.push(&texture.render(W, H, &camera(t))).unwrap();
        }
        estimator.finish()
    }

    /// Mean frame-to-frame acceleration of a texture point in the output.
    fn jerkiness(position: impl Fn(usize) -> Point, frames: usize) -> f64 {
        (1..frames - 1)
            .map(|t| (position(t + 1) - position(t) - (position(t) - position(t - 1))).length())
            .sum::<f64>()
            / (frames - 2) as f64
    }

    #[test]
    fn measures_pairs_and_steadies_the_shot() {
        let frames = 40;
        let motion = measured(frames);
        assert_eq!(motion.frame_count(), frames);
        for (i, pair) in motion.pairs.iter().enumerate() {
            let truth = camera(i).inverse().unwrap().then(&camera(i + 1));
            let probe = Point::new(100.0, 80.0);
            assert!(
                pair.transform.apply(probe).distance(truth.apply(probe)) < 0.5,
                "pair {i}"
            );
        }

        let settings = StabilizerSettings {
            amount: 1.0,
            ..StabilizerSettings::default()
        };
        let result = stabilize(&motion, &settings);
        assert_eq!(result.weak_frames, 0);
        let world = Point::new(160.0, 120.0);
        let raw = jerkiness(|t| camera(t).apply(world), frames);
        let steady = jerkiness(
            |t| result.transforms[t].apply(camera(t).apply(world)),
            frames,
        );
        assert!(steady < raw / 4.0, "raw {raw}, stabilized {steady}");
    }

    #[test]
    fn framing_controls_zoom() {
        let motion = measured(24);
        let settings = |framing| StabilizerSettings {
            framing,
            amount: 1.0,
            ..StabilizerSettings::default()
        };

        let crop = stabilize(&motion, &settings(Framing::Crop));
        assert!(crop.zooms[0] > 1.01);
        assert!(crop.zooms.iter().all(|&z| z == crop.zooms[0]));
        for transform in &crop.transforms {
            assert!(
                required_zoom(transform, W as f64, H as f64, Point::new(159.5, 119.5))
                    <= 1.0 + 1e-6
            );
        }

        let scale = stabilize(&motion, &settings(Framing::AutoScale));
        assert!(scale.zooms.iter().all(|&z| z <= crop.zooms[0] + 1e-9));
        for transform in &scale.transforms {
            assert!(
                required_zoom(transform, W as f64, H as f64, Point::new(159.5, 119.5))
                    <= 1.0 + 1e-6
            );
        }

        let extend = stabilize(&motion, &settings(Framing::Extend));
        assert!(extend.zooms.iter().all(|&z| z == 1.0));

        let none = stabilize(
            &motion,
            &StabilizerSettings {
                amount: 0.0,
                ..settings(Framing::Crop)
            },
        );
        for transform in &none.transforms {
            assert!(
                transform
                    .apply(Point::new(10.0, 10.0))
                    .distance(Point::new(10.0, 10.0))
                    < 1e-6
            );
        }
    }

    #[test]
    fn edge_fills() {
        let frame = Frame::from_luma(4, 1, vec![10, 20, 30, 40]).unwrap();
        let shift = Transform::translation(2.0, 0.0);
        assert_eq!(
            warp(&frame, &shift, EdgeFill::Black, 0).data(),
            &[0, 0, 10, 20]
        );
        assert_eq!(
            warp(&frame, &shift, EdgeFill::Clamp, 0).data(),
            &[10, 10, 10, 20]
        );
        assert_eq!(
            warp(&frame, &shift, EdgeFill::Mirror, 0).data(),
            &[30, 20, 10, 20]
        );
    }
}
