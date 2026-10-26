//! Fitting motion models to point correspondences, plainly and robustly.

use crate::geometry::{Point, Quad, Transform};

/// How much of a region's motion a track records, from least to most.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MotionModel {
    /// Horizontal and vertical position only.
    Translation,
    /// Position, uniform scale and rotation.
    #[default]
    Similarity,
    /// Also stretch and skew; parallel lines stay parallel.
    Affine,
    /// The full planar transformation, for flat surfaces turning toward or
    /// away from the camera.
    Perspective,
}

impl MotionModel {
    pub const ALL: [Self; 4] = [
        Self::Translation,
        Self::Similarity,
        Self::Affine,
        Self::Perspective,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Translation => "Translation",
            Self::Similarity => "Similarity",
            Self::Affine => "Affine",
            Self::Perspective => "Perspective",
        }
    }

    /// Correspondences that determine the model exactly.
    pub(crate) fn minimal_points(self) -> usize {
        match self {
            Self::Translation => 1,
            Self::Similarity => 2,
            Self::Affine => 3,
            Self::Perspective => 4,
        }
    }
}

/// The homography taking each corner of `from` onto the matching corner of
/// `to`, e.g. a layer's rectangle onto a tracked surface (corner pin).
pub fn homography_from_quads(from: &Quad, to: &Quad) -> Option<Transform> {
    fit(MotionModel::Perspective, &from.corners, &to.corners)
}

/// Least-squares fit of `model` taking `from[i]` to `to[i]`.
pub(crate) fn fit(model: MotionModel, from: &[Point], to: &[Point]) -> Option<Transform> {
    debug_assert_eq!(from.len(), to.len());
    if from.len() < model.minimal_points() {
        return None;
    }
    let transform = match model {
        MotionModel::Translation => fit_translation(from, to),
        MotionModel::Similarity => fit_similarity(from, to),
        MotionModel::Affine => fit_affine(from, to),
        MotionModel::Perspective => fit_homography(from, to),
    }?;
    transform.is_finite().then_some(transform)
}

fn mean(points: &[Point]) -> Point {
    let sum = points.iter().fold(Point::default(), |sum, &p| sum + p);
    sum * (1.0 / points.len() as f64)
}

fn fit_translation(from: &[Point], to: &[Point]) -> Option<Transform> {
    let delta = mean(to) - mean(from);
    Some(Transform::translation(delta.x, delta.y))
}

fn fit_similarity(from: &[Point], to: &[Point]) -> Option<Transform> {
    let (from_mean, to_mean) = (mean(from), mean(to));
    let (mut dot, mut cross, mut norm) = (0.0, 0.0, 0.0);
    for (&p, &q) in from.iter().zip(to) {
        let (p, q) = (p - from_mean, q - to_mean);
        dot += p.x * q.x + p.y * q.y;
        cross += p.x * q.y - p.y * q.x;
        norm += p.x * p.x + p.y * p.y;
    }
    if norm < 1e-9 {
        return None;
    }
    let (a, b) = (dot / norm, cross / norm);
    let linear = Transform::from_matrix([[a, -b, 0.0], [b, a, 0.0], [0.0, 0.0, 1.0]]);
    let offset = to_mean - linear.apply(from_mean);
    Some(linear.then(&Transform::translation(offset.x, offset.y)))
}

fn fit_affine(from: &[Point], to: &[Point]) -> Option<Transform> {
    let (from_mean, to_mean) = (mean(from), mean(to));
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    let (mut xu, mut yu, mut xv, mut yv) = (0.0, 0.0, 0.0, 0.0);
    for (&p, &q) in from.iter().zip(to) {
        let (p, q) = (p - from_mean, q - to_mean);
        xx += p.x * p.x;
        xy += p.x * p.y;
        yy += p.y * p.y;
        xu += p.x * q.x;
        yu += p.y * q.x;
        xv += p.x * q.y;
        yv += p.y * q.y;
    }
    let determinant = xx * yy - xy * xy;
    // Collinear points leave the perpendicular direction undetermined.
    if determinant.abs() < 1e-9 * (xx + yy).powi(2).max(1e-12) {
        return None;
    }
    let solve = |bx: f64, by: f64| {
        (
            (yy * bx - xy * by) / determinant,
            (xx * by - xy * bx) / determinant,
        )
    };
    let (a, b) = solve(xu, yu);
    let (c, d) = solve(xv, yv);
    let linear = Transform::from_matrix([[a, b, 0.0], [c, d, 0.0], [0.0, 0.0, 1.0]]);
    let offset = to_mean - linear.apply(from_mean);
    Some(linear.then(&Transform::translation(offset.x, offset.y)))
}

/// Similarity that moves `points` to have their centroid at the origin and
/// mean distance √2 from it (Hartley normalization).
fn normalizer(points: &[Point]) -> Option<Transform> {
    let center = mean(points);
    let spread = points.iter().map(|&p| p.distance(center)).sum::<f64>() / points.len() as f64;
    if spread < 1e-9 {
        return None;
    }
    let scale = std::f64::consts::SQRT_2 / spread;
    Some(Transform::translation(-center.x, -center.y).then(&Transform::scaling(scale)))
}

/// Direct linear transform on normalized points with `h33 = 1`, which is
/// safe once the source centroid sits at the origin.
fn fit_homography(from: &[Point], to: &[Point]) -> Option<Transform> {
    let from_norm = normalizer(from)?;
    let to_norm = normalizer(to)?;

    let mut normal = [[0.0; 8]; 8];
    let mut rhs = [0.0; 8];
    for (&p, &q) in from.iter().zip(to) {
        let p = from_norm.apply(p);
        let q = to_norm.apply(q);
        let rows = [
            ([p.x, p.y, 1.0, 0.0, 0.0, 0.0, -q.x * p.x, -q.x * p.y], q.x),
            ([0.0, 0.0, 0.0, p.x, p.y, 1.0, -q.y * p.x, -q.y * p.y], q.y),
        ];
        for (row, target) in rows {
            for i in 0..8 {
                rhs[i] += row[i] * target;
                for j in 0..8 {
                    normal[i][j] += row[i] * row[j];
                }
            }
        }
    }

    let h = solve(normal, rhs)?;
    let normalized =
        Transform::from_matrix([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]);
    Some(
        from_norm
            .then(&normalized)
            .then(&to_norm.inverse()?)
            .normalized(),
    )
}

/// Gaussian elimination with partial pivoting.
pub(crate) fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for column in 0..N {
        let pivot =
            (column..N).max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))?;
        if a[pivot][column].abs() < 1e-12 {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        let pivot_row = a[column];
        for row in column + 1..N {
            let factor = a[row][column] / pivot_row[column];
            for (value, pivot) in a[row].iter_mut().zip(pivot_row).skip(column) {
                *value -= factor * pivot;
            }
            b[row] -= factor * b[column];
        }
    }

    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let tail: f64 = (row + 1..N).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x.iter().all(|value| value.is_finite()).then_some(x)
}

pub(crate) struct RobustFit {
    pub(crate) transform: Transform,
    pub(crate) inliers: Vec<bool>,
    pub(crate) inlier_count: usize,
}

/// Small deterministic generator, so the same footage always tracks the same.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(crate) fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

fn degenerate(model: MotionModel, points: &[Point]) -> bool {
    let close = |a: Point, b: Point| a.distance(b) < 1.0;
    let collinear = |a: Point, b: Point, c: Point| {
        ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() < 1.0
    };
    match model {
        MotionModel::Translation => false,
        MotionModel::Similarity => close(points[0], points[1]),
        MotionModel::Affine => collinear(points[0], points[1], points[2]),
        MotionModel::Perspective => (0..4).any(|skip| {
            let rest: Vec<Point> = (0..4).filter(|&i| i != skip).map(|i| points[i]).collect();
            collinear(rest[0], rest[1], rest[2])
        }),
    }
}

fn count_inliers(
    transform: &Transform,
    from: &[Point],
    to: &[Point],
    threshold: f64,
) -> (Vec<bool>, usize) {
    let limit = threshold * threshold;
    let inliers: Vec<bool> = from
        .iter()
        .zip(to)
        .map(|(&p, &q)| {
            let error = transform.apply(p) - q;
            transform.depth(p) > 0.0 && error.x * error.x + error.y * error.y <= limit
        })
        .collect();
    let count = inliers.iter().filter(|&&inlier| inlier).count();
    (inliers, count)
}

/// RANSAC: the model most correspondences agree with to within `threshold`
/// pixels, refit on all of them.
pub(crate) fn fit_robust(
    model: MotionModel,
    from: &[Point],
    to: &[Point],
    threshold: f64,
    rng: &mut Rng,
) -> Option<RobustFit> {
    let n = from.len();
    let m = model.minimal_points();
    if n < m {
        return None;
    }

    const CONFIDENCE: f64 = 0.995;
    const MAX_ITERATIONS: usize = 500;

    let mut best: Option<(Transform, usize)> = None;
    let mut needed = MAX_ITERATIONS;
    let mut iteration = 0;
    let mut sample = Vec::with_capacity(m);
    while iteration < needed.min(MAX_ITERATIONS) {
        iteration += 1;
        sample.clear();
        while sample.len() < m {
            let index = rng.below(n);
            if !sample.contains(&index) {
                sample.push(index);
            }
        }
        let sample_from: Vec<Point> = sample.iter().map(|&i| from[i]).collect();
        let sample_to: Vec<Point> = sample.iter().map(|&i| to[i]).collect();
        if degenerate(model, &sample_from) {
            continue;
        }
        let Some(candidate) = fit(model, &sample_from, &sample_to) else {
            continue;
        };
        let (_, count) = count_inliers(&candidate, from, to, threshold);
        if best.is_none_or(|(_, best_count)| count > best_count) {
            best = Some((candidate, count));
            // Samples needed to draw an all-inlier one with CONFIDENCE, at
            // the inlier ratio seen so far.
            let clean = (count as f64 / n as f64).powi(m as i32);
            needed = if clean >= 1.0 - f64::EPSILON {
                0
            } else if clean <= f64::EPSILON {
                MAX_ITERATIONS
            } else {
                ((1.0 - CONFIDENCE).ln() / (1.0 - clean).ln()).ceil() as usize
            };
        }
        if n == m {
            break;
        }
    }

    let (candidate, _) = best?;
    let (inliers, count) = count_inliers(&candidate, from, to, threshold);
    let inlier_from: Vec<Point> = (0..n).filter(|&i| inliers[i]).map(|i| from[i]).collect();
    let inlier_to: Vec<Point> = (0..n).filter(|&i| inliers[i]).map(|i| to[i]).collect();

    if let Some(refit) = fit(model, &inlier_from, &inlier_to) {
        let (refit_inliers, refit_count) = count_inliers(&refit, from, to, threshold);
        if refit_count >= count {
            return Some(RobustFit {
                transform: refit,
                inliers: refit_inliers,
                inlier_count: refit_count,
            });
        }
    }
    Some(RobustFit {
        transform: candidate,
        inliers,
        inlier_count: count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Vec<Point> {
        (0..6)
            .flat_map(|y| {
                (0..6).map(move |x| Point::new(20.0 + x as f64 * 17.0, 15.0 + y as f64 * 13.0))
            })
            .collect()
    }

    fn truth(model: MotionModel) -> Transform {
        match model {
            MotionModel::Translation => Transform::translation(4.5, -2.25),
            MotionModel::Similarity => Transform::similarity(1.1, 0.2, 7.0, -3.0),
            MotionModel::Affine => {
                Transform::from_matrix([[1.1, 0.15, 5.0], [-0.05, 0.92, 2.0], [0.0, 0.0, 1.0]])
            }
            MotionModel::Perspective => Transform::from_matrix([
                [1.02, 0.04, 6.0],
                [-0.03, 0.98, -4.0],
                [0.0006, -0.0004, 1.0],
            ]),
        }
    }

    fn max_error(a: &Transform, b: &Transform, points: &[Point]) -> f64 {
        points
            .iter()
            .map(|&p| a.apply(p).distance(b.apply(p)))
            .fold(0.0, f64::max)
    }

    #[test]
    fn fits_every_model_exactly() {
        let from = grid();
        for model in MotionModel::ALL {
            let expected = truth(model);
            let to: Vec<Point> = from.iter().map(|&p| expected.apply(p)).collect();
            let fitted = fit(model, &from, &to).unwrap();
            assert!(max_error(&fitted, &expected, &from) < 1e-6, "{model:?}");
        }
    }

    #[test]
    fn maps_quads_corner_to_corner() {
        let from = Quad::from_rect(0.0, 0.0, 100.0, 50.0);
        let to = Quad::new([
            Point::new(10.0, 12.0),
            Point::new(95.0, 4.0),
            Point::new(120.0, 70.0),
            Point::new(2.0, 60.0),
        ]);
        let homography = homography_from_quads(&from, &to).unwrap();
        for (a, b) in from.corners.iter().zip(&to.corners) {
            assert!(homography.apply(*a).distance(*b) < 1e-6);
        }
    }

    #[test]
    fn rejects_outliers() {
        let from = grid();
        let mut rng = Rng::new(7);
        for model in MotionModel::ALL {
            let expected = truth(model);
            let mut to: Vec<Point> = from.iter().map(|&p| expected.apply(p)).collect();
            for (i, point) in to.iter_mut().enumerate() {
                if i % 3 == 0 {
                    *point = *point + Point::new(25.0 + i as f64, -30.0);
                }
            }
            let robust = fit_robust(model, &from, &to, 1.0, &mut rng).unwrap();
            assert_eq!(
                robust.inlier_count,
                from.len() - from.len().div_ceil(3),
                "{model:?}"
            );
            assert!(
                max_error(&robust.transform, &expected, &from) < 1e-6,
                "{model:?}"
            );
        }
    }

    #[test]
    fn solves_linear_systems() {
        let x = solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0]).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12 && (x[1] - 1.4).abs() < 1e-12);
        assert!(solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]).is_none());
    }
}
