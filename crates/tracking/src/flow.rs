//! Sparse optical flow: pyramidal Lucas-Kanade with a forward-backward check.

use crate::frame::{Level, Pyramid};
use crate::geometry::{Point, Transform};

pub(crate) struct FlowParams {
    /// Half the side of the square window each point is matched with.
    pub(crate) radius: isize,
    pub(crate) iterations: usize,
    /// Points whose round trip back to the first frame misses by more than
    /// this many pixels are dropped as unreliable.
    pub(crate) max_round_trip_error: f64,
    /// Mean absolute intensity difference above which a match is treated as
    /// occluded.
    pub(crate) max_residual: f32,
}

impl Default for FlowParams {
    fn default() -> Self {
        Self {
            radius: 7,
            iterations: 20,
            max_round_trip_error: 1.0,
            max_residual: 24.0,
        }
    }
}

/// Where each of `points` in `prev` moved to in `next`, or `None` where it
/// could not be followed reliably. `guesses` seed the search, e.g. with the
/// motion predicted from earlier frames.
pub(crate) fn track_points(
    prev: &Pyramid,
    next: &Pyramid,
    points: &[Point],
    guesses: Option<&[Point]>,
    params: &FlowParams,
) -> Vec<Option<Point>> {
    points
        .iter()
        .enumerate()
        .map(|(i, &point)| {
            let guess = guesses.map_or(point, |guesses| guesses[i]);
            let forward = track_point(prev, next, point, guess, params)?;
            let back = track_point(next, prev, forward, point, params)?;
            (back.distance(point) <= params.max_round_trip_error).then_some(forward)
        })
        .collect()
}

/// The whole-frame shift that best aligns `next` with `prev`, found by
/// exhaustive search on the coarsest pyramid level. Fine repeating detail
/// is blurred away there, so this settles which repetition points belong to
/// before optical flow refines them.
pub(crate) fn coarse_shift(prev: &Pyramid, next: &Pyramid) -> Point {
    let level = prev.levels.len().min(next.levels.len()) - 1;
    let (a, b) = (&prev.levels[level].image, &next.levels[level].image);
    let range = (a.width.min(a.height) / 4) as isize;
    let (width, height) = (a.width as isize, a.height as isize);
    if range < 1 || width <= 2 * range || height <= 2 * range {
        return Point::default();
    }

    let cost = |dx: isize, dy: isize| -> f64 {
        let mut sum = 0.0f64;
        for y in range..height - range {
            for x in range..width - range {
                let pa = a.at(x as usize, y as usize);
                let pb = b.at((x + dx) as usize, (y + dy) as usize);
                sum += (pa - pb).abs() as f64;
            }
        }
        sum
    };
    let mut best = (f64::MAX, 0, 0);
    for dy in -range..=range {
        for dx in -range..=range {
            let value = cost(dx, dy);
            if value < best.0 {
                best = (value, dx, dy);
            }
        }
    }

    let (value, dx, dy) = best;
    let vertex = |minus: f64, plus: f64| {
        let curvature = minus - 2.0 * value + plus;
        if curvature > 1e-9 {
            (0.5 * (minus - plus) / curvature).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    };
    let fx = if dx.abs() < range {
        vertex(cost(dx - 1, dy), cost(dx + 1, dy))
    } else {
        0.0
    };
    let fy = if dy.abs() < range {
        vertex(cost(dx, dy - 1), cost(dx, dy + 1))
    } else {
        0.0
    };
    Point::new(dx as f64 + fx, dy as f64 + fy).scale((1u32 << level) as f64)
}

/// Re-locates a point in `next` by matching how it looked in the reference
/// frame, warped by the current motion estimate (`to_reference` takes
/// `next` coordinates back to the reference). Matching against the original
/// look instead of the previous frame keeps long tracks from drifting.
/// `around` is where the motion puts the point; `start` where to begin.
pub(crate) fn refine_against_reference(
    reference: &Level,
    to_reference: &Transform,
    next: &Level,
    around: Point,
    start: Point,
    params: &FlowParams,
) -> Option<Point> {
    let r = params.radius;
    let side = (2 * r + 3) as usize;
    // The warped reference window, one pixel wider on each side for its
    // gradients.
    let mut window = Vec::with_capacity(side * side);
    for dy in -r - 1..=r + 1 {
        for dx in -r - 1..=r + 1 {
            let source = to_reference.apply(around + Point::new(dx as f64, dy as f64));
            window.push(reference.image.sample(source.x, source.y));
        }
    }

    let inner = (2 * r + 1) as usize;
    let count = inner * inner;
    let mut template = Vec::with_capacity(count);
    let mut gradients = Vec::with_capacity(count);
    let (mut gxx, mut gxy, mut gyy) = (0.0f64, 0.0f64, 0.0f64);
    for y in 1..side - 1 {
        for x in 1..side - 1 {
            let at = |x: usize, y: usize| window[y * side + x];
            let ix = 0.5 * (at(x + 1, y) - at(x - 1, y)) as f64;
            let iy = 0.5 * (at(x, y + 1) - at(x, y - 1)) as f64;
            template.push(at(x, y));
            gradients.push((ix, iy));
            gxx += ix * ix;
            gxy += ix * iy;
            gyy += iy * iy;
        }
    }
    let determinant = gxx * gyy - gxy * gxy;
    let min_eigen = 0.5 * (gxx + gyy) - ((0.5 * (gxx - gyy)).powi(2) + gxy * gxy).sqrt();
    if min_eigen / (count as f64) < 0.05 || determinant.abs() < 1e-9 {
        return None;
    }

    let mut position = start;
    let sample_window = |position: Point, i: usize| {
        let (dx, dy) = ((i % inner) as isize - r, (i / inner) as isize - r);
        next.image
            .sample(position.x + dx as f64, position.y + dy as f64)
    };
    for _ in 0..params.iterations {
        let (mut bx, mut by) = (0.0f64, 0.0f64);
        for (i, (&value, &(ix, iy))) in template.iter().zip(&gradients).enumerate() {
            let difference = (value - sample_window(position, i)) as f64;
            bx += difference * ix;
            by += difference * iy;
        }
        let step = Point::new(
            (gyy * bx - gxy * by) / determinant,
            (gxx * by - gxy * bx) / determinant,
        );
        position = position + step;
        if !position.x.is_finite() || !position.y.is_finite() {
            return None;
        }
        if step.length() < 0.01 {
            break;
        }
    }

    let residual = template
        .iter()
        .enumerate()
        .map(|(i, &value)| (value - sample_window(position, i)).abs())
        .sum::<f32>()
        / count as f32;
    let settled = position.distance(start) <= params.max_round_trip_error * 2.0;
    (settled && residual <= params.max_residual && next.image.contains(position.x, position.y, 0.0))
        .then_some(position)
}

fn track_point(
    prev: &Pyramid,
    next: &Pyramid,
    point: Point,
    guess: Point,
    params: &FlowParams,
) -> Option<Point> {
    let depth = prev.levels.len().min(next.levels.len());
    let top = depth - 1;
    let mut flow = (guess - point).scale(1.0 / (1u32 << top) as f64);

    for level in (0..depth).rev() {
        let scale = 1.0 / (1u32 << level) as f64;
        let at = point.scale(scale);
        let finest = level == 0;
        // A coarse level too blurred to match just passes its guess down;
        // only the full-resolution match can reject the point.
        let refined = match refine(
            &prev.levels[level],
            &next.levels[level],
            at,
            flow,
            params,
            finest,
        ) {
            Some(refined) => refined,
            None if !finest => flow,
            None => return None,
        };
        flow = if finest { refined } else { refined.scale(2.0) };
    }

    let end = point + flow;
    next.base().image.contains(end.x, end.y, 0.0).then_some(end)
}

/// One level of iterative Lucas-Kanade around `at` in `prev`, starting from
/// `flow`.
fn refine(
    prev: &Level,
    next: &Level,
    at: Point,
    mut flow: Point,
    params: &FlowParams,
    finest: bool,
) -> Option<Point> {
    let r = params.radius;
    let side = (2 * r + 1) as usize;
    let count = side * side;

    let mut template = Vec::with_capacity(count);
    let mut gradients = Vec::with_capacity(count);
    let (mut gxx, mut gxy, mut gyy) = (0.0f64, 0.0f64, 0.0f64);
    for dy in -r..=r {
        for dx in -r..=r {
            let (x, y) = (at.x + dx as f64, at.y + dy as f64);
            let ix = prev.gx.sample(x, y) as f64;
            let iy = prev.gy.sample(x, y) as f64;
            template.push(prev.image.sample(x, y));
            gradients.push((ix, iy));
            gxx += ix * ix;
            gxy += ix * iy;
            gyy += iy * iy;
        }
    }

    let determinant = gxx * gyy - gxy * gxy;
    let min_eigen = 0.5 * (gxx + gyy) - ((0.5 * (gxx - gyy)).powi(2) + gxy * gxy).sqrt();
    if min_eigen / (count as f64) < 0.05 || determinant.abs() < 1e-9 {
        return None;
    }

    for _ in 0..params.iterations {
        let (mut bx, mut by) = (0.0f64, 0.0f64);
        let mut i = 0;
        for dy in -r..=r {
            for dx in -r..=r {
                let x = at.x + flow.x + dx as f64;
                let y = at.y + flow.y + dy as f64;
                let difference = (template[i] - next.image.sample(x, y)) as f64;
                bx += difference * gradients[i].0;
                by += difference * gradients[i].1;
                i += 1;
            }
        }
        let step = Point::new(
            (gyy * bx - gxy * by) / determinant,
            (gxx * by - gxy * bx) / determinant,
        );
        flow = flow + step;
        if !flow.x.is_finite() || !flow.y.is_finite() {
            return None;
        }
        if step.length() < 0.01 {
            break;
        }
    }

    if finest {
        let mut residual = 0.0f32;
        let mut i = 0;
        for dy in -r..=r {
            for dx in -r..=r {
                let x = at.x + flow.x + dx as f64;
                let y = at.y + flow.y + dy as f64;
                residual += (template[i] - next.image.sample(x, y)).abs();
                i += 1;
            }
        }
        if residual / count as f32 > params.max_residual {
            return None;
        }
    }
    Some(flow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transform;
    use crate::frame::Plane;
    use crate::testing::{Texture, assert_points_close};

    fn pyramid(frame: &crate::Frame) -> Pyramid {
        Pyramid::new(Plane::from_frame(frame, 1), 4)
    }

    #[test]
    fn follows_subpixel_and_large_shifts() {
        let texture = Texture::new(3);
        let first = texture.render(240, 180, &Transform::IDENTITY);
        let points = [
            Point::new(80.0, 60.0),
            Point::new(150.0, 90.0),
            Point::new(120.0, 120.0),
        ];

        for shift in [
            Point::new(0.4, -0.3),
            Point::new(3.3, -2.1),
            Point::new(-21.0, 14.5),
        ] {
            let second = texture.render(240, 180, &Transform::translation(shift.x, shift.y));
            let tracked = track_points(
                &pyramid(&first),
                &pyramid(&second),
                &points,
                None,
                &FlowParams::default(),
            );
            for (point, result) in points.iter().zip(tracked) {
                assert_points_close(result.expect("lost a point"), *point + shift, 0.1);
            }
        }
    }

    #[test]
    fn drops_points_that_do_not_match() {
        let texture = Texture::new(3);
        let other = Texture::new(99);
        let first = texture.render(160, 120, &Transform::IDENTITY);
        let second = other.render(160, 120, &Transform::IDENTITY);
        let points: Vec<Point> = (0..6)
            .map(|i| Point::new(30.0 + i as f64 * 18.0, 60.0))
            .collect();

        let tracked = track_points(
            &pyramid(&first),
            &pyramid(&second),
            &points,
            None,
            &FlowParams::default(),
        );
        let kept = tracked.iter().filter(|point| point.is_some()).count();
        assert!(
            kept <= 1,
            "kept {kept} of {} unrelated points",
            points.len()
        );
    }

    #[test]
    fn finds_large_whole_frame_shifts() {
        let texture = Texture::new(4);
        let first = texture.render(320, 240, &Transform::IDENTITY);
        for shift in [Point::new(-37.0, 22.0), Point::new(45.5, -8.0)] {
            let second = texture.render(320, 240, &Transform::translation(shift.x, shift.y));
            let found = coarse_shift(&pyramid(&first), &pyramid(&second));
            // Coarse: within a couple of full-resolution pixels is plenty
            // for optical flow to take over.
            assert_points_close(found, shift, 2.5);
        }
    }
}
