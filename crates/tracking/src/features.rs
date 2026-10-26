//! Corner detection (Shi-Tomasi "good features to track").

use crate::frame::Level;
use crate::geometry::Point;

pub(crate) struct FeatureParams {
    pub(crate) max_count: usize,
    /// Fraction of the strongest corner's score a corner must reach.
    pub(crate) quality: f32,
    pub(crate) min_distance: f64,
}

/// Corners inside `[min, max)` that pass `mask`, strongest first, at least
/// `min_distance` from each other and from `existing`.
pub(crate) fn detect(
    level: &Level,
    min: Point,
    max: Point,
    mask: impl Fn(Point) -> bool,
    existing: &[Point],
    params: &FeatureParams,
) -> Vec<Point> {
    const BLOCK: isize = 1;
    const BORDER: isize = 4;
    let width = level.image.width as isize;
    let height = level.image.height as isize;
    let x0 = (min.x.floor() as isize).max(BORDER);
    let y0 = (min.y.floor() as isize).max(BORDER);
    let x1 = (max.x.ceil() as isize).min(width - BORDER);
    let y1 = (max.y.ceil() as isize).min(height - BORDER);
    if x1 - x0 < 3 || y1 - y0 < 3 || params.max_count == 0 {
        return Vec::new();
    }

    // Integral images of the structure tensor over the region plus the block
    // radius, so each pixel's window sums cost four lookups.
    let (ix0, iy0) = (x0 - BLOCK, y0 - BLOCK);
    let iw = (x1 - x0 + 2 * BLOCK) as usize;
    let ih = (y1 - y0 + 2 * BLOCK) as usize;
    let stride = iw + 1;
    let mut sums = vec![[0.0f64; 3]; stride * (ih + 1)];
    for y in 0..ih {
        let mut row = [0.0f64; 3];
        for x in 0..iw {
            let (px, py) = ((ix0 + x as isize) as usize, (iy0 + y as isize) as usize);
            let gx = level.gx.at(px, py) as f64;
            let gy = level.gy.at(px, py) as f64;
            row[0] += gx * gx;
            row[1] += gx * gy;
            row[2] += gy * gy;
            let above = sums[y * stride + x + 1];
            sums[(y + 1) * stride + x + 1] =
                [above[0] + row[0], above[1] + row[1], above[2] + row[2]];
        }
    }
    let block_sum = |x: usize, y: usize, k: usize| {
        let side = (2 * BLOCK + 1) as usize;
        sums[(y + side) * stride + x + side][k]
            - sums[y * stride + x + side][k]
            - sums[(y + side) * stride + x][k]
            + sums[y * stride + x][k]
    };

    let rw = (x1 - x0) as usize;
    let rh = (y1 - y0) as usize;
    let mut score = vec![0.0f32; rw * rh];
    let mut best = 0.0f32;
    for y in 0..rh {
        for x in 0..rw {
            let point = Point::new((x0 + x as isize) as f64, (y0 + y as isize) as f64);
            if !mask(point) {
                continue;
            }
            let (a, b, c) = (block_sum(x, y, 0), block_sum(x, y, 1), block_sum(x, y, 2));
            let half_trace = 0.5 * (a + c);
            let min_eigen = half_trace - ((0.5 * (a - c)).powi(2) + b * b).sqrt();
            let value = (min_eigen / ((2 * BLOCK + 1) * (2 * BLOCK + 1)) as f64) as f32;
            score[y * rw + x] = value;
            best = best.max(value);
        }
    }
    // Flat or noise-only regions: nothing worth tracking.
    if best < 1.0 {
        return Vec::new();
    }

    let threshold = best * params.quality;
    let mut candidates = Vec::new();
    for y in 1..rh.saturating_sub(1) {
        for x in 1..rw.saturating_sub(1) {
            let value = score[y * rw + x];
            if value < threshold {
                continue;
            }
            let is_peak = (-1isize..=1).all(|dy| {
                (-1isize..=1).all(|dx| {
                    let neighbor =
                        score[(y as isize + dy) as usize * rw + (x as isize + dx) as usize];
                    (dx == 0 && dy == 0)
                        || neighbor < value
                        || (neighbor == value && (dy, dx) > (0, 0))
                })
            });
            if is_peak {
                candidates.push((
                    value,
                    Point::new((x0 + x as isize) as f64, (y0 + y as isize) as f64),
                ));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut grid = SpacingGrid::new(params.min_distance);
    for &point in existing {
        grid.insert(point);
    }
    let mut selected = Vec::new();
    for (_, point) in candidates {
        if selected.len() >= params.max_count {
            break;
        }
        if grid.is_free(point) {
            grid.insert(point);
            selected.push(point);
        }
    }
    selected
}

/// Spatial hash answering "is anything within `min_distance` of here?".
struct SpacingGrid {
    cell: f64,
    min_distance: f64,
    cells: std::collections::HashMap<(i64, i64), Vec<Point>>,
}

impl SpacingGrid {
    fn new(min_distance: f64) -> Self {
        let min_distance = min_distance.max(1.0);
        Self {
            cell: min_distance,
            min_distance,
            cells: std::collections::HashMap::new(),
        }
    }

    fn key(&self, point: Point) -> (i64, i64) {
        (
            (point.x / self.cell).floor() as i64,
            (point.y / self.cell).floor() as i64,
        )
    }

    fn insert(&mut self, point: Point) {
        let key = self.key(point);
        self.cells.entry(key).or_default().push(point);
    }

    fn is_free(&self, point: Point) -> bool {
        let (cx, cy) = self.key(point);
        (cy - 1..=cy + 1).all(|y| {
            (cx - 1..=cx + 1).all(|x| {
                self.cells.get(&(x, y)).is_none_or(|points| {
                    points
                        .iter()
                        .all(|other| other.distance(point) >= self.min_distance)
                })
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Plane, Pyramid};
    use crate::testing::checkerboard;

    #[test]
    fn finds_checkerboard_corners_and_nothing_on_flat_ground() {
        let frame = checkerboard(96, 96, 16);
        let pyramid = Pyramid::new(Plane::from_frame(&frame, 1), 1);
        let params = FeatureParams {
            max_count: 100,
            quality: 0.1,
            min_distance: 6.0,
        };
        let corners = detect(
            pyramid.base(),
            Point::new(0.0, 0.0),
            Point::new(96.0, 96.0),
            |_| true,
            &[],
            &params,
        );
        assert!(corners.len() >= 16, "found {}", corners.len());
        for corner in &corners {
            let off_x = (corner.x - (corner.x / 16.0).round() * 16.0).abs();
            let off_y = (corner.y - (corner.y / 16.0).round() * 16.0).abs();
            assert!(off_x <= 1.5 && off_y <= 1.5, "{corner:?}");
        }
        for (i, a) in corners.iter().enumerate() {
            for b in &corners[i + 1..] {
                assert!(a.distance(*b) >= 6.0);
            }
        }

        let flat = crate::Frame::from_luma(64, 64, vec![128; 64 * 64]).unwrap();
        let pyramid = Pyramid::new(Plane::from_frame(&flat, 1), 1);
        let corners = detect(
            pyramid.base(),
            Point::new(0.0, 0.0),
            Point::new(64.0, 64.0),
            |_| true,
            &[],
            &params,
        );
        assert!(corners.is_empty());
    }
}
