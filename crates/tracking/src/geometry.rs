use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn length(self) -> f64 {
        self.x.hypot(self.y)
    }

    pub fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }

    pub(crate) fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor)
    }
}

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl Mul<f64> for Point {
    type Output = Self;

    fn mul(self, factor: f64) -> Self {
        self.scale(factor)
    }
}

/// A projective transform of the plane, as a row-major 3x3 matrix acting on
/// column vectors. Every motion model (translation through homography) is
/// one of these, so they compose and invert the same way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub m: [[f64; 3]; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    pub const fn from_matrix(m: [[f64; 3]; 3]) -> Self {
        Self { m }
    }

    pub const fn translation(x: f64, y: f64) -> Self {
        Self {
            m: [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]],
        }
    }

    pub const fn scaling(factor: f64) -> Self {
        Self {
            m: [[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Uniform scale and counter-clockwise rotation (radians, y down) about
    /// the origin, then a translation.
    pub fn similarity(scale: f64, rotation: f64, x: f64, y: f64) -> Self {
        let (sin, cos) = rotation.sin_cos();
        Self {
            m: [
                [scale * cos, -scale * sin, x],
                [scale * sin, scale * cos, y],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// `self` applied about `center` instead of the origin.
    pub fn about(&self, center: Point) -> Self {
        Self::translation(-center.x, -center.y)
            .then(self)
            .then(&Self::translation(center.x, center.y))
    }

    pub fn apply(&self, point: Point) -> Point {
        let m = &self.m;
        let x = m[0][0] * point.x + m[0][1] * point.y + m[0][2];
        let y = m[1][0] * point.x + m[1][1] * point.y + m[1][2];
        let w = m[2][0] * point.x + m[2][1] * point.y + m[2][2];
        let w = if w.abs() < 1e-12 {
            1e-12_f64.copysign(w)
        } else {
            w
        };
        Point::new(x / w, y / w)
    }

    /// The homogeneous `w` a point maps to. Non-positive means the point
    /// crossed the horizon of a perspective transform.
    pub(crate) fn depth(&self, point: Point) -> f64 {
        self.m[2][0] * point.x + self.m[2][1] * point.y + self.m[2][2]
    }

    /// `self` followed by `next`.
    pub fn then(&self, next: &Self) -> Self {
        next.multiply(self)
    }

    fn multiply(&self, other: &Self) -> Self {
        let mut m = [[0.0; 3]; 3];
        for (row, out) in m.iter_mut().enumerate() {
            for (column, value) in out.iter_mut().enumerate() {
                *value = (0..3).map(|k| self.m[row][k] * other.m[k][column]).sum();
            }
        }
        Self { m }
    }

    pub fn determinant(&self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    pub fn inverse(&self) -> Option<Self> {
        let determinant = self.determinant();
        if !determinant.is_finite() || determinant.abs() < 1e-12 {
            return None;
        }

        let m = &self.m;
        let cofactor = |r0: usize, r1: usize, c0: usize, c1: usize| {
            m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
        };
        let inverse = [
            [
                cofactor(1, 2, 1, 2),
                -cofactor(0, 2, 1, 2),
                cofactor(0, 1, 1, 2),
            ],
            [
                -cofactor(1, 2, 0, 2),
                cofactor(0, 2, 0, 2),
                -cofactor(0, 1, 0, 2),
            ],
            [
                cofactor(1, 2, 0, 1),
                -cofactor(0, 2, 0, 1),
                cofactor(0, 1, 0, 1),
            ],
        ];

        Some(
            Self {
                m: inverse.map(|row| row.map(|value| value / determinant)),
            }
            .normalized(),
        )
    }

    /// Scales the matrix so its bottom-right entry is 1, when it can.
    pub fn normalized(&self) -> Self {
        let w = self.m[2][2];
        if w.abs() < 1e-12 {
            return *self;
        }
        Self {
            m: self.m.map(|row| row.map(|value| value / w)),
        }
    }

    pub fn is_finite(&self) -> bool {
        self.m.iter().flatten().all(|value| value.is_finite())
    }

    /// Splits the transform into translation of `anchor`, then uniform scale,
    /// rotation, residual stretch/skew and perspective about it, so each part
    /// can be kept or frozen independently.
    pub fn decompose(&self, anchor: Point) -> Option<Decomposition> {
        let target = self.apply(anchor);
        let local = Self::translation(anchor.x, anchor.y)
            .then(self)
            .then(&Self::translation(-target.x, -target.y));
        if local.m[2][2].abs() < 1e-12 {
            return None;
        }
        let local = local.normalized();

        let [[a, b, _], [c, d, _], [px, py, _]] = local.m;
        let scale = (a * d - b * c).abs().sqrt();
        if !scale.is_finite() || scale < 1e-9 {
            return None;
        }
        let rotation = (c - b).atan2(a + d);
        let (sin, cos) = rotation.sin_cos();
        // K = R(-rotation) * A / scale, which is the identity for a similarity.
        let stretch = [
            [(cos * a + sin * c) / scale, (cos * b + sin * d) / scale],
            [(-sin * a + cos * c) / scale, (-sin * b + cos * d) / scale],
        ];

        Some(Decomposition {
            translation: target - anchor,
            scale,
            rotation,
            stretch,
            perspective: [px, py],
        })
    }

    /// Keeps only the selected components of the motion, measured about
    /// `anchor`. A transform that cannot be decomposed keeps just the
    /// translation of the anchor.
    pub fn restricted(&self, anchor: Point, components: MotionComponents) -> Self {
        if components == MotionComponents::ALL {
            return *self;
        }
        match self.decompose(anchor) {
            Some(decomposition) => decomposition.restricted(components).compose(anchor),
            None => {
                let delta = self.apply(anchor) - anchor;
                Self::translation(
                    if components.horizontal { delta.x } else { 0.0 },
                    if components.vertical { delta.y } else { 0.0 },
                )
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decomposition {
    pub translation: Point,
    pub scale: f64,
    pub rotation: f64,
    /// Stretch and skew left over once scale and rotation are removed; its
    /// determinant is ±1.
    pub stretch: [[f64; 2]; 2],
    pub perspective: [f64; 2],
}

impl Decomposition {
    pub fn restricted(mut self, components: MotionComponents) -> Self {
        if !components.horizontal {
            self.translation.x = 0.0;
        }
        if !components.vertical {
            self.translation.y = 0.0;
        }
        if !components.scale {
            self.scale = 1.0;
        }
        if !components.rotation {
            self.rotation = 0.0;
        }
        if !components.affine {
            self.stretch = [[1.0, 0.0], [0.0, 1.0]];
        }
        if !components.perspective {
            self.perspective = [0.0, 0.0];
        }
        self
    }

    pub fn compose(&self, anchor: Point) -> Transform {
        let (sin, cos) = self.rotation.sin_cos();
        let k = self.stretch;
        let s = self.scale;
        let local = Transform::from_matrix([
            [
                s * (cos * k[0][0] - sin * k[1][0]),
                s * (cos * k[0][1] - sin * k[1][1]),
                0.0,
            ],
            [
                s * (sin * k[0][0] + cos * k[1][0]),
                s * (sin * k[0][1] + cos * k[1][1]),
                0.0,
            ],
            [self.perspective[0], self.perspective[1], 1.0],
        ]);
        let target = anchor + self.translation;

        Transform::translation(-anchor.x, -anchor.y)
            .then(&local)
            .then(&Transform::translation(target.x, target.y))
    }
}

/// Which parts of a motion carry over. Turning one off freezes it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotionComponents {
    pub horizontal: bool,
    pub vertical: bool,
    pub scale: bool,
    pub rotation: bool,
    /// Stretch and skew.
    pub affine: bool,
    pub perspective: bool,
}

impl MotionComponents {
    pub const ALL: Self = Self {
        horizontal: true,
        vertical: true,
        scale: true,
        rotation: true,
        affine: true,
        perspective: true,
    };

    pub const POSITION: Self = Self {
        horizontal: true,
        vertical: true,
        scale: false,
        rotation: false,
        affine: false,
        perspective: false,
    };
}

impl Default for MotionComponents {
    fn default() -> Self {
        Self::ALL
    }
}

/// Four corners, clockwise from the top left in a y-down frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub corners: [Point; 4],
}

impl Quad {
    pub const fn new(corners: [Point; 4]) -> Self {
        Self { corners }
    }

    pub fn from_rect(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self::new([
            Point::new(x, y),
            Point::new(x + width, y),
            Point::new(x + width, y + height),
            Point::new(x, y + height),
        ])
    }

    pub fn center(&self) -> Point {
        let sum = self
            .corners
            .iter()
            .fold(Point::default(), |sum, &corner| sum + corner);
        sum * 0.25
    }

    /// `(min, max)` corners of the axis-aligned bounding box.
    pub fn bounds(&self) -> (Point, Point) {
        let mut min = self.corners[0];
        let mut max = self.corners[0];
        for corner in &self.corners[1..] {
            min.x = min.x.min(corner.x);
            min.y = min.y.min(corner.y);
            max.x = max.x.max(corner.x);
            max.y = max.y.max(corner.y);
        }
        (min, max)
    }

    pub fn map(&self, transform: &Transform) -> Self {
        Self::new(self.corners.map(|corner| transform.apply(corner)))
    }

    pub fn translated(&self, delta: Point) -> Self {
        Self::new(self.corners.map(|corner| corner + delta))
    }

    pub fn area(&self) -> f64 {
        let c = &self.corners;
        0.5 * (0..4)
            .map(|i| {
                let (a, b) = (c[i], c[(i + 1) % 4]);
                a.x * b.y - b.x * a.y
            })
            .sum::<f64>()
    }

    /// Whether the corners form a convex quadrilateral in a consistent
    /// winding, the shape a plane seen through a camera can have.
    pub fn is_convex(&self) -> bool {
        let c = &self.corners;
        let mut sign = 0.0;
        for i in 0..4 {
            let (a, b, d) = (c[i], c[(i + 1) % 4], c[(i + 2) % 4]);
            let cross = (b.x - a.x) * (d.y - b.y) - (b.y - a.y) * (d.x - b.x);
            if cross.abs() < 1e-9 {
                return false;
            }
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
        true
    }

    /// Point-in-quad for convex quads of either winding.
    pub fn contains(&self, point: Point) -> bool {
        let c = &self.corners;
        let mut sign = 0.0;
        for i in 0..4 {
            let (a, b) = (c[i], c[(i + 1) % 4]);
            let cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
            if cross == 0.0 {
                continue;
            }
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Point, b: Point, tolerance: f64) {
        assert!(a.distance(b) <= tolerance, "{a:?} != {b:?}");
    }

    fn sample_homography() -> Transform {
        Transform::from_matrix([
            [1.05, 0.08, 12.0],
            [-0.06, 0.97, -7.0],
            [0.0004, -0.0003, 1.0],
        ])
    }

    #[test]
    fn composes_in_application_order() {
        let shift = Transform::translation(10.0, 0.0);
        let double = Transform::scaling(2.0);
        let point = Point::new(1.0, 1.0);

        assert_close(
            shift.then(&double).apply(point),
            Point::new(22.0, 2.0),
            1e-12,
        );
        assert_close(
            double.then(&shift).apply(point),
            Point::new(12.0, 2.0),
            1e-12,
        );
    }

    #[test]
    fn inverse_round_trips() {
        let transform = sample_homography();
        let inverse = transform.inverse().unwrap();
        let point = Point::new(37.0, -12.5);

        assert_close(inverse.apply(transform.apply(point)), point, 1e-9);
    }

    #[test]
    fn decomposition_round_trips() {
        let anchor = Point::new(120.0, 80.0);
        let transform = sample_homography();
        let rebuilt = transform.decompose(anchor).unwrap().compose(anchor);

        for point in [Point::new(0.0, 0.0), anchor, Point::new(300.0, 200.0)] {
            assert_close(rebuilt.apply(point), transform.apply(point), 1e-9);
        }
    }

    #[test]
    fn decomposes_a_similarity_exactly() {
        let anchor = Point::new(50.0, 40.0);
        let transform = Transform::similarity(1.2, 0.3, 0.0, 0.0)
            .about(anchor)
            .then(&Transform::translation(5.0, -3.0));
        let parts = transform.decompose(anchor).unwrap();

        assert!((parts.scale - 1.2).abs() < 1e-9);
        assert!((parts.rotation - 0.3).abs() < 1e-9);
        assert_close(parts.translation, Point::new(5.0, -3.0), 1e-9);
        assert!((parts.stretch[0][0] - 1.0).abs() < 1e-9);
        assert!(parts.stretch[1][0].abs() < 1e-9);
    }

    #[test]
    fn freezing_components_keeps_the_anchor_path() {
        let anchor = Point::new(50.0, 40.0);
        let transform = Transform::similarity(1.5, 0.4, 0.0, 0.0)
            .about(anchor)
            .then(&Transform::translation(8.0, 6.0));

        let position = transform.restricted(anchor, MotionComponents::POSITION);
        assert_close(position.apply(anchor), anchor + Point::new(8.0, 6.0), 1e-9);
        assert_close(
            position.apply(anchor + Point::new(10.0, 0.0)),
            anchor + Point::new(18.0, 6.0),
            1e-9,
        );

        let horizontal_only = transform.restricted(
            anchor,
            MotionComponents {
                vertical: false,
                ..MotionComponents::POSITION
            },
        );
        assert_close(
            horizontal_only.apply(anchor),
            anchor + Point::new(8.0, 0.0),
            1e-9,
        );
    }

    #[test]
    fn quad_convexity_and_containment() {
        let quad = Quad::from_rect(0.0, 0.0, 10.0, 10.0);
        assert!(quad.is_convex());
        assert!(quad.contains(Point::new(5.0, 5.0)));
        assert!(!quad.contains(Point::new(11.0, 5.0)));

        let bow_tie = Quad::new([
            Point::new(0.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(10.0, 0.0),
            Point::new(0.0, 10.0),
        ]);
        assert!(!bow_tie.is_convex());
    }
}
