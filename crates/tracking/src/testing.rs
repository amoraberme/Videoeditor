//! Synthetic footage for tests: procedural textures rendered through known
//! motion, so trackers can be checked against the truth.

use crate::{Frame, Point, Transform};

pub(crate) fn checkerboard(width: usize, height: usize, square: usize) -> Frame {
    let data = (0..width * height)
        .map(|i| {
            let (x, y) = (i % width / square, i / width / square);
            if (x + y) % 2 == 0 { 30 } else { 220 }
        })
        .collect();
    Frame::from_luma(width, height, data).unwrap()
}

/// A continuous texture: a sum of plane waves in many directions plus a few
/// soft blobs, which gives corners everywhere at sub-pixel precision.
/// Wavelengths are spread evenly on a log scale, like natural images, so
/// coarse pyramid levels keep detail to match.
pub(crate) struct Texture {
    waves: Vec<(f64, f64, f64, f64)>,
    blobs: Vec<(f64, f64, f64, f64)>,
}

impl Texture {
    pub(crate) fn new(seed: u64) -> Self {
        let mut rng = crate::estimate::Rng::new(seed);
        let mut unit = move || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let waves = (0..14)
            .map(|_| {
                let angle = unit() * std::f64::consts::TAU;
                let wavelength = 7.0 * (120.0f64 / 7.0).powf(unit());
                let k = std::f64::consts::TAU / wavelength;
                (
                    k * angle.cos(),
                    k * angle.sin(),
                    unit() * std::f64::consts::TAU,
                    0.5 + unit(),
                )
            })
            .collect();
        let blobs = (0..40)
            .map(|_| {
                (
                    unit() * 800.0 - 100.0,
                    unit() * 600.0 - 100.0,
                    4.0 + unit() * 10.0,
                    unit() * 2.0 - 1.0,
                )
            })
            .collect();
        Self { waves, blobs }
    }

    pub(crate) fn value(&self, p: Point) -> f64 {
        let waves: f64 = self
            .waves
            .iter()
            .map(|&(kx, ky, phase, amplitude)| amplitude * (kx * p.x + ky * p.y + phase).sin())
            .sum();
        let blobs: f64 = self
            .blobs
            .iter()
            .map(|&(x, y, radius, sign)| {
                let d2 = (p.x - x).powi(2) + (p.y - y).powi(2);
                sign * 2.5 * (-d2 / (2.0 * radius * radius)).exp()
            })
            .sum();
        (128.0 + 9.0 * waves + 30.0 * blobs).clamp(0.0, 255.0)
    }

    /// The frame a camera sees when the texture has moved by `motion`
    /// (texture coordinates to frame coordinates).
    pub(crate) fn render(&self, width: usize, height: usize, motion: &Transform) -> Frame {
        let inverse = motion.inverse().unwrap();
        let data = (0..width * height)
            .map(|i| {
                let p = Point::new((i % width) as f64, (i / width) as f64);
                self.value(inverse.apply(p)).round() as u8
            })
            .collect();
        Frame::from_luma(width, height, data).unwrap()
    }
}

/// Paints `patch` (a texture, moved by `motion`) over a flat background
/// inside the rectangle the motion carries `rect` to.
pub(crate) fn render_object(
    width: usize,
    height: usize,
    texture: &Texture,
    rect: (f64, f64, f64, f64),
    motion: &Transform,
    background: u8,
) -> Frame {
    let inverse = motion.inverse().unwrap();
    let (x, y, w, h) = rect;
    let data = (0..width * height)
        .map(|i| {
            let p = inverse.apply(Point::new((i % width) as f64, (i / width) as f64));
            if p.x >= x && p.x < x + w && p.y >= y && p.y < y + h {
                texture.value(p).round() as u8
            } else {
                background
            }
        })
        .collect();
    Frame::from_luma(width, height, data).unwrap()
}

pub(crate) fn assert_points_close(actual: Point, expected: Point, tolerance: f64) {
    assert!(
        actual.distance(expected) <= tolerance,
        "{actual:?} is {} from {expected:?} (tolerance {tolerance})",
        actual.distance(expected)
    );
}
