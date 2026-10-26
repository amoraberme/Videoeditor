//! Object tracking by normalized cross-correlation against an adapting
//! template. Less precise than following corners, but it tolerates blur,
//! fast motion and low-quality footage, and finds the object again after it
//! is briefly lost.

use crate::frame::Pyramid;
use crate::geometry::{Point, Transform};

/// Longest side of the template, in samples.
const TEMPLATE_SIDE: f64 = 32.0;
const SEARCH_FACTOR: f64 = 0.75;
const MAX_SEARCH_FACTOR: f64 = 3.0;
const TRACK_THRESHOLD: f64 = 0.5;
const RECOVER_THRESHOLD: f64 = 0.65;
const UPDATE_THRESHOLD: f64 = 0.8;

#[derive(Clone)]
struct Patch {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl Patch {
    /// Samples the box of `size` centered on `center` (analysis pixels) into
    /// a `width` x `height` grid, from the pyramid level closest to that
    /// density so it does not alias.
    fn extract(
        pyramid: &Pyramid,
        center: Point,
        size: (f64, f64),
        width: usize,
        height: usize,
    ) -> Self {
        let density = width as f64 / size.0;
        let level = if density >= 1.0 {
            0
        } else {
            ((1.0 / density).log2().floor() as usize).min(pyramid.levels.len() - 1)
        };
        let image = &pyramid.levels[level].image;
        let level_scale = 1.0 / (1u32 << level) as f64;
        let left = center.x - size.0 / 2.0;
        let top = center.y - size.1 / 2.0;
        let (step_x, step_y) = (size.0 / width as f64, size.1 / height as f64);

        let mut data = Vec::with_capacity(width * height);
        for j in 0..height {
            for i in 0..width {
                let x = left + (i as f64 + 0.5) * step_x;
                let y = top + (j as f64 + 0.5) * step_y;
                data.push(image.sample(x * level_scale, y * level_scale));
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Zero-mean, unit-norm copy for correlation, or `None` if flat.
    fn normalized(&self) -> Option<Vec<f32>> {
        let n = self.data.len() as f32;
        let mean = self.data.iter().sum::<f32>() / n;
        let norm = self
            .data
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>()
            .sqrt();
        (norm > 1.0 * n.sqrt()).then(|| self.data.iter().map(|v| (v - mean) / norm).collect())
    }

    fn blend(&mut self, others: &[(&Patch, f32)]) {
        let keep = 1.0 - others.iter().map(|(_, weight)| weight).sum::<f32>();
        for (i, value) in self.data.iter_mut().enumerate() {
            *value = *value * keep
                + others
                    .iter()
                    .map(|(patch, weight)| patch.data[i] * weight)
                    .sum::<f32>();
        }
    }
}

pub(crate) struct ObjectTracker {
    original: Patch,
    template: Patch,
    matcher: Vec<f32>,
    reference_center: Point,
    reference_size: (f64, f64),
    center: Point,
    velocity: Point,
    scale: f64,
    follow_scale: bool,
    searching: bool,
    search_factor: f64,
}

pub(crate) struct ObjectStep {
    pub(crate) motion: Transform,
    pub(crate) tracked: bool,
    pub(crate) confidence: f32,
}

impl ObjectTracker {
    /// `None` when the box is too small or too flat to match against.
    pub(crate) fn new(
        pyramid: &Pyramid,
        center: Point,
        size: (f64, f64),
        follow_scale: bool,
    ) -> Option<Self> {
        if size.0 < 6.0 || size.1 < 6.0 {
            return None;
        }
        let density = (TEMPLATE_SIDE / size.0.max(size.1)).min(1.0);
        let width = (size.0 * density).round().max(6.0) as usize;
        let height = (size.1 * density).round().max(6.0) as usize;
        let template = Patch::extract(pyramid, center, size, width, height);
        let matcher = template.normalized()?;

        Some(Self {
            original: template.clone(),
            template,
            matcher,
            reference_center: center,
            reference_size: size,
            center,
            velocity: Point::default(),
            scale: 1.0,
            follow_scale,
            searching: false,
            search_factor: SEARCH_FACTOR,
        })
    }

    pub(crate) fn step(&mut self, pyramid: &Pyramid) -> ObjectStep {
        let predicted = self.center + self.velocity;
        let scales: &[f64] = if self.follow_scale && !self.searching {
            &[1.0, 0.96, 1.04]
        } else {
            &[1.0]
        };

        let mut best: Option<(f64, Point, f64)> = None;
        for &relative in scales {
            let scale = (self.scale * relative).clamp(0.1, 10.0);
            if let Some((score, center)) = self.search(pyramid, predicted, scale) {
                // A slight preference for the current scale keeps size steady.
                let biased = if relative == 1.0 {
                    score
                } else {
                    score - 0.005
                };
                if best.is_none_or(|(best_score, _, _)| biased > best_score) {
                    best = Some((biased, center, scale));
                }
            }
        }

        let threshold = if self.searching {
            RECOVER_THRESHOLD
        } else {
            TRACK_THRESHOLD
        };
        match best {
            Some((score, center, scale)) if score >= threshold => {
                self.velocity = if self.searching {
                    Point::default()
                } else {
                    center - self.center
                };
                self.center = center;
                self.scale = scale;
                self.searching = false;
                self.search_factor = SEARCH_FACTOR;
                if score >= UPDATE_THRESHOLD {
                    self.update_template(pyramid);
                }
                ObjectStep {
                    motion: self.motion(),
                    tracked: true,
                    confidence: score as f32,
                }
            }
            _ => {
                self.searching = true;
                self.velocity = Point::default();
                self.search_factor = (self.search_factor * 1.5).min(MAX_SEARCH_FACTOR);
                ObjectStep {
                    motion: self.motion(),
                    tracked: false,
                    confidence: 0.0,
                }
            }
        }
    }

    fn motion(&self) -> Transform {
        let shift = self.center - self.reference_center;
        Transform::scaling(self.scale)
            .about(self.reference_center)
            .then(&Transform::translation(shift.x, shift.y))
    }

    fn update_template(&mut self, pyramid: &Pyramid) {
        let size = (
            self.reference_size.0 * self.scale,
            self.reference_size.1 * self.scale,
        );
        let current = Patch::extract(
            pyramid,
            self.center,
            size,
            self.template.width,
            self.template.height,
        );
        // Adapt to appearance changes while staying anchored to the first look.
        self.template
            .blend(&[(&current, 0.1), (&self.original, 0.1)]);
        if let Some(matcher) = self.template.normalized() {
            self.matcher = matcher;
        }
    }

    /// Best correlation of the template around `around` at `scale`, and the
    /// box center it was found at.
    fn search(&self, pyramid: &Pyramid, around: Point, scale: f64) -> Option<(f64, Point)> {
        let (tw, th) = (self.template.width, self.template.height);
        let size = (self.reference_size.0 * scale, self.reference_size.1 * scale);
        let density = tw as f64 / size.0;
        let radius = (self.search_factor * size.0.max(size.1) * density).ceil() as usize;
        let (sw, sh) = (tw + 2 * radius, th + 2 * radius);
        let window = Patch::extract(
            pyramid,
            around,
            (sw as f64 / density, sh as f64 / th as f64 * size.1),
            sw,
            sh,
        );

        let integral = Integral::new(&window);
        let n = (tw * th) as f64;
        let score_at = |px: usize, py: usize| -> f64 {
            let (sum, sum_sq) = integral.window(px, py, tw, th);
            let variance = sum_sq - sum * sum / n;
            if variance <= 1e-6 * n {
                return -1.0;
            }
            let mut dot = 0.0f64;
            for y in 0..th {
                let row = &window.data[(py + y) * sw + px..(py + y) * sw + px + tw];
                let template = &self.matcher[y * tw..(y + 1) * tw];
                dot += row
                    .iter()
                    .zip(template)
                    .map(|(a, b)| (a * b) as f64)
                    .sum::<f64>();
            }
            dot / variance.sqrt()
        };

        let span = 2 * radius;
        let mut best = (f64::MIN, 0usize, 0usize);
        for py in (0..=span).step_by(2) {
            for px in (0..=span).step_by(2) {
                let score = score_at(px, py);
                if score > best.0 {
                    best = (score, px, py);
                }
            }
        }
        let (cx, cy) = (best.1, best.2);
        for py in cy.saturating_sub(2)..=(cy + 2).min(span) {
            for px in cx.saturating_sub(2)..=(cx + 2).min(span) {
                let score = score_at(px, py);
                if score > best.0 {
                    best = (score, px, py);
                }
            }
        }

        let (score, px, py) = best;
        let peak = |minus: f64, center: f64, plus: f64| {
            let curvature = minus - 2.0 * center + plus;
            if curvature < -1e-9 {
                (0.5 * (minus - plus) / curvature).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        };
        let dx = if px > 0 && px < span {
            peak(score_at(px - 1, py), score, score_at(px + 1, py))
        } else {
            0.0
        };
        let dy = if py > 0 && py < span {
            peak(score_at(px, py - 1), score, score_at(px, py + 1))
        } else {
            0.0
        };

        let offset = Point::new(
            (px as f64 + dx - radius as f64) / density,
            (py as f64 + dy - radius as f64) / (th as f64 / size.1),
        );
        score.is_finite().then_some((score, around + offset))
    }
}

/// Summed-area tables of a patch and its square.
struct Integral {
    stride: usize,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
}

impl Integral {
    fn new(patch: &Patch) -> Self {
        let stride = patch.width + 1;
        let mut sum = vec![0.0; stride * (patch.height + 1)];
        let mut sum_sq = vec![0.0; stride * (patch.height + 1)];
        for y in 0..patch.height {
            let (mut row, mut row_sq) = (0.0, 0.0);
            for x in 0..patch.width {
                let value = patch.data[y * patch.width + x] as f64;
                row += value;
                row_sq += value * value;
                sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + row;
                sum_sq[(y + 1) * stride + x + 1] = sum_sq[y * stride + x + 1] + row_sq;
            }
        }
        Self {
            stride,
            sum,
            sum_sq,
        }
    }

    fn window(&self, x: usize, y: usize, width: usize, height: usize) -> (f64, f64) {
        let at = |table: &[f64], x: usize, y: usize| table[y * self.stride + x];
        let area = |table: &[f64]| {
            at(table, x + width, y + height) - at(table, x, y + height) - at(table, x + width, y)
                + at(table, x, y)
        };
        (area(&self.sum), area(&self.sum_sq))
    }
}
