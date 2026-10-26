use crate::TrackingError;

/// An 8-bit luma image, the input every tracker works on. Decoders hand
/// over their Y plane; RGBA sources convert with [`Frame::from_rgba`].
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl Frame {
    pub fn from_luma(width: usize, height: usize, data: Vec<u8>) -> Result<Self, TrackingError> {
        if width == 0 || height == 0 || data.len() != width * height {
            return Err(TrackingError::InvalidFrame);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Copies a plane whose rows are `stride` bytes apart, as decoders lay
    /// them out.
    pub fn from_luma_strided(
        width: usize,
        height: usize,
        stride: usize,
        data: &[u8],
    ) -> Result<Self, TrackingError> {
        if width == 0 || height == 0 || stride < width || data.len() < stride * (height - 1) + width
        {
            return Err(TrackingError::InvalidFrame);
        }
        let mut pixels = Vec::with_capacity(width * height);
        for row in data.chunks(stride).take(height) {
            pixels.extend_from_slice(&row[..width]);
        }
        Self::from_luma(width, height, pixels)
    }

    pub fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Result<Self, TrackingError> {
        if rgba.len() != width * height * 4 {
            return Err(TrackingError::InvalidFrame);
        }
        // BT.709 luma weights in 8.8 fixed point.
        let data = rgba
            .chunks_exact(4)
            .map(|px| ((54 * px[0] as u32 + 183 * px[1] as u32 + 19 * px[2] as u32) >> 8) as u8)
            .collect();
        Self::from_luma(width, height, data)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

/// A float image the analysis runs on.
#[derive(Clone, Debug)]
pub(crate) struct Plane {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) data: Vec<f32>,
}

impl Plane {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    /// Box-averages the frame down by an integer `factor`, which keeps fine
    /// texture from aliasing at analysis resolution.
    pub(crate) fn from_frame(frame: &Frame, factor: usize) -> Self {
        let factor = factor.max(1);
        let width = (frame.width / factor).max(1);
        let height = (frame.height / factor).max(1);
        let mut plane = Self::new(width, height);
        let norm = 1.0 / (factor * factor) as f32;

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0u32;
                for sy in y * factor..((y + 1) * factor).min(frame.height) {
                    let row = &frame.data[sy * frame.width..(sy + 1) * frame.width];
                    for &value in &row[x * factor..((x + 1) * factor).min(frame.width)] {
                        sum += value as u32;
                    }
                }
                plane.data[y * width + x] = sum as f32 * norm;
            }
        }
        plane
    }

    #[inline]
    pub(crate) fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline]
    fn at_clamped(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.at(x, y)
    }

    /// Bilinear sample at pixel coordinates, clamped to the edge.
    #[inline]
    pub(crate) fn sample(&self, x: f64, y: f64) -> f32 {
        let max_x = (self.width - 1) as f64;
        let max_y = (self.height - 1) as f64;
        let x = x.clamp(0.0, max_x);
        let y = y.clamp(0.0, max_y);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = (x - x0 as f64) as f32;
        let fy = (y - y0 as f64) as f32;

        let top = self.at(x0, y0) * (1.0 - fx) + self.at(x1, y0) * fx;
        let bottom = self.at(x0, y1) * (1.0 - fx) + self.at(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    pub(crate) fn contains(&self, x: f64, y: f64, margin: f64) -> bool {
        x >= margin
            && y >= margin
            && x <= self.width as f64 - 1.0 - margin
            && y <= self.height as f64 - 1.0 - margin
    }

    /// Gaussian blur with the 5-tap binomial kernel, then every second pixel.
    pub(crate) fn pyramid_down(&self) -> Self {
        const KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
        let width = self.width.div_ceil(2);
        let height = self.height.div_ceil(2);

        let mut rows = Self::new(width, self.height);
        for y in 0..self.height {
            for x in 0..width {
                let cx = (x * 2) as isize;
                rows.data[y * width + x] = KERNEL
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * self.at_clamped(cx + k as isize - 2, y as isize))
                    .sum();
            }
        }

        let mut out = Self::new(width, height);
        for y in 0..height {
            let cy = (y * 2) as isize;
            for x in 0..width {
                out.data[y * width + x] = KERNEL
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * rows.at_clamped(x as isize, cy + k as isize - 2))
                    .sum();
            }
        }
        out
    }

    /// Scharr derivatives, normalized to intensity per pixel.
    pub(crate) fn gradients(&self) -> (Self, Self) {
        let mut gx = Self::new(self.width, self.height);
        let mut gy = Self::new(self.width, self.height);
        for y in 0..self.height as isize {
            for x in 0..self.width as isize {
                let p = |dx: isize, dy: isize| self.at_clamped(x + dx, y + dy);
                let i = y as usize * self.width + x as usize;
                gx.data[i] = (3.0 * (p(1, -1) - p(-1, -1))
                    + 10.0 * (p(1, 0) - p(-1, 0))
                    + 3.0 * (p(1, 1) - p(-1, 1)))
                    / 32.0;
                gy.data[i] = (3.0 * (p(-1, 1) - p(-1, -1))
                    + 10.0 * (p(0, 1) - p(0, -1))
                    + 3.0 * (p(1, 1) - p(1, -1)))
                    / 32.0;
            }
        }
        (gx, gy)
    }
}

#[derive(Clone)]
pub(crate) struct Level {
    pub(crate) image: Plane,
    pub(crate) gx: Plane,
    pub(crate) gy: Plane,
}

/// Image pyramid with gradients, finest level first.
pub(crate) struct Pyramid {
    pub(crate) levels: Vec<Level>,
}

impl Pyramid {
    pub(crate) fn new(base: Plane, max_levels: usize) -> Self {
        let mut levels = Vec::with_capacity(max_levels);
        let mut image = base;
        loop {
            let (gx, gy) = image.gradients();
            let next = (levels.len() + 1 < max_levels && image.width.min(image.height) >= 40)
                .then(|| image.pyramid_down());
            levels.push(Level { image, gx, gy });
            match next {
                Some(next) => image = next,
                None => break,
            }
        }
        Self { levels }
    }

    pub(crate) fn base(&self) -> &Level {
        &self.levels[0]
    }
}

/// Integer factor that brings the longest side of a `width` x `height`
/// frame down to at most `max_side`.
pub(crate) fn analysis_factor(width: usize, height: usize, max_side: usize) -> usize {
    width.max(height).div_ceil(max_side.max(32)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_buffers() {
        assert_eq!(
            Frame::from_luma(4, 4, vec![0; 15]),
            Err(TrackingError::InvalidFrame)
        );
        assert!(Frame::from_luma(4, 4, vec![0; 16]).is_ok());
    }

    #[test]
    fn copies_strided_rows() {
        let data = [1, 2, 0, 3, 4, 0];
        let frame = Frame::from_luma_strided(2, 2, 3, &data).unwrap();
        assert_eq!(frame.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn samples_bilinearly() {
        let frame = Frame::from_luma(2, 1, vec![0, 100]).unwrap();
        let plane = Plane::from_frame(&frame, 1);
        assert!((plane.sample(0.25, 0.0) - 25.0).abs() < 1e-4);
        assert!((plane.sample(5.0, 3.0) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn downsamples_by_box_average() {
        let frame = Frame::from_luma(4, 2, vec![0, 4, 8, 8, 4, 8, 8, 8]).unwrap();
        let plane = Plane::from_frame(&frame, 2);
        assert_eq!((plane.width, plane.height), (2, 1));
        assert_eq!(plane.data, vec![4.0, 8.0]);
    }

    #[test]
    fn gradient_of_a_ramp_is_its_slope() {
        let data = (0..64).map(|i| (i % 8) as u8 * 10).collect();
        let plane = Plane::from_frame(&Frame::from_luma(8, 8, data).unwrap(), 1);
        let (gx, gy) = plane.gradients();
        assert!((gx.at(4, 4) - 10.0).abs() < 1e-4);
        assert!(gy.at(4, 4).abs() < 1e-4);
    }
}
