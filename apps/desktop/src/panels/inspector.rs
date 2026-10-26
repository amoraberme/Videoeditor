use gpui::{
    AnyElement, ClickEvent, Context, FontWeight, SharedString, Window, div, prelude::*, px,
};
use tracking::{
    AnalysisQuality, EdgeFill, Framing, MotionFilter, MotionModel, MotionSettings, RegionShape,
    StabilizationMode, StabilizerSettings, TrackerKind, TrackerSettings,
};

use crate::components::{Button, ButtonSize, ButtonVariant, Label, Separator};
use crate::theme::ActiveTheme;

/// Longest sides offered for tracking analysis, in pixels.
const TRACKING_RESOLUTIONS: [usize; 4] = [640, 960, 1280, 1920];

/// Settings for tracking and stabilizing the selected clip. They are the
/// same values the `tracking` crate runs with, so applying them to a clip
/// is passing them along.
pub(crate) struct Inspector {
    tracker: TrackerSettings,
    motion: MotionSettings,
    stabilizer: StabilizerSettings,
}

impl Inspector {
    pub(crate) fn new() -> Self {
        Self {
            tracker: TrackerSettings::default(),
            motion: MotionSettings::default(),
            stabilizer: StabilizerSettings::default(),
        }
    }

    fn tracking_section(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let tracker = self.tracker;
        let pin = tracker.corner_pin_active();
        let is_circle = tracker.region_shape == RegionShape::Circle;

        section(
            "Motion Tracking",
            [
                field(
                    "Tracker type",
                    div()
                        .flex()
                        .flex_col()
                        .gap_1p5()
                        .child(choices(
                            "tracker-kind",
                            &TrackerKind::ALL,
                            tracker.kind,
                            TrackerKind::label,
                            |this, kind| this.tracker.kind = kind,
                            cx,
                        ))
                        .child(Label::new(tracker.kind.description()).muted()),
                ),
                field(
                    "Motion model",
                    choices(
                        "motion-model",
                        &MotionModel::ALL,
                        tracker.motion_model,
                        MotionModel::label,
                        |this, model| this.tracker.motion_model = model,
                        cx,
                    ),
                ),
                field(
                    "Selector shape",
                    choices(
                        "region-shape",
                        &RegionShape::ALL,
                        tracker.region_shape,
                        RegionShape::label,
                        |this, shape| this.tracker.region_shape = shape,
                        cx,
                    ),
                ),
                field(
                    "Region follows",
                    div()
                        .flex()
                        .flex_wrap()
                        .gap_1()
                        // Corner pin applies size and rotation itself; the
                        // underlying choices come back when it is turned off.
                        .child(toggle(
                            "follow-size",
                            "Size",
                            tracker.follow_size || pin,
                            pin,
                            |this| this.tracker.follow_size ^= true,
                            cx,
                        ))
                        .child(toggle(
                            "follow-rotation",
                            "Rotation",
                            (tracker.follow_rotation && !is_circle) || pin,
                            pin || is_circle,
                            |this| this.tracker.follow_rotation ^= true,
                            cx,
                        ))
                        .child(toggle(
                            "corner-pin",
                            "Corner Pin",
                            pin,
                            tracker.region_shape != RegionShape::Quad,
                            |this| this.tracker.corner_pin ^= true,
                            cx,
                        )),
                ),
                field(
                    "Tracking resolution",
                    choices(
                        "tracking-resolution",
                        &TRACKING_RESOLUTIONS,
                        tracker.analysis_size,
                        resolution_label,
                        |this, size| this.tracker.analysis_size = size,
                        cx,
                    ),
                ),
            ],
        )
    }

    fn motion_section(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let motion = self.motion;
        let components = motion.components;

        section(
            "Adjust Motion",
            [
                field(
                    "Filter",
                    div()
                        .flex()
                        .flex_col()
                        .gap_1p5()
                        .child(choices(
                            "motion-filter",
                            &MotionFilter::ALL,
                            motion.filter,
                            MotionFilter::label,
                            |this, filter| this.motion.filter = filter,
                            cx,
                        ))
                        .child(Label::new(motion.filter.description()).muted()),
                ),
                field(
                    "Strength",
                    stepper(
                        "motion-strength",
                        motion.strength,
                        motion.filter == MotionFilter::None,
                        |this, delta| {
                            this.motion.strength = step_fraction(this.motion.strength, delta)
                        },
                        cx,
                    ),
                ),
                field(
                    "Motion",
                    div()
                        .flex()
                        .flex_wrap()
                        .gap_1()
                        .child(toggle(
                            "keep-horizontal",
                            "Horizontal",
                            components.horizontal,
                            false,
                            |this| this.motion.components.horizontal ^= true,
                            cx,
                        ))
                        .child(toggle(
                            "keep-vertical",
                            "Vertical",
                            components.vertical,
                            false,
                            |this| this.motion.components.vertical ^= true,
                            cx,
                        ))
                        .child(toggle(
                            "keep-scale",
                            "Scale",
                            components.scale,
                            false,
                            |this| this.motion.components.scale ^= true,
                            cx,
                        ))
                        .child(toggle(
                            "keep-rotation",
                            "Rotation",
                            components.rotation,
                            false,
                            |this| this.motion.components.rotation ^= true,
                            cx,
                        ))
                        .child(toggle(
                            "keep-affine",
                            "Affine",
                            components.affine,
                            false,
                            |this| this.motion.components.affine ^= true,
                            cx,
                        ))
                        .child(toggle(
                            "keep-perspective",
                            "Perspective",
                            components.perspective,
                            false,
                            |this| this.motion.components.perspective ^= true,
                            cx,
                        )),
                ),
            ],
        )
    }

    fn stabilizer_section(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let stabilizer = self.stabilizer;
        let mut fields = vec![
            field(
                "Mode",
                choices(
                    "stabilizer-mode",
                    &StabilizationMode::ALL,
                    stabilizer.mode,
                    StabilizationMode::label,
                    |this, mode| this.stabilizer.mode = mode,
                    cx,
                ),
            ),
            field(
                "Framing",
                choices(
                    "stabilizer-framing",
                    &Framing::ALL,
                    stabilizer.framing,
                    Framing::label,
                    |this, framing| this.stabilizer.framing = framing,
                    cx,
                ),
            ),
        ];
        // Only extended framing uncovers edges that need filling.
        if stabilizer.framing == Framing::Extend {
            fields.push(field(
                "Edge fill",
                choices(
                    "stabilizer-edge-fill",
                    &EdgeFill::ALL,
                    stabilizer.edge_fill,
                    EdgeFill::label,
                    |this, fill| this.stabilizer.edge_fill = fill,
                    cx,
                ),
            ));
        }
        fields.push(field(
            "Amount",
            stepper(
                "stabilizer-amount",
                stabilizer.amount,
                false,
                |this, delta| this.stabilizer.amount = step_fraction(this.stabilizer.amount, delta),
                cx,
            ),
        ));
        fields.push(field(
            "Quality",
            choices(
                "stabilizer-quality",
                &AnalysisQuality::ALL,
                stabilizer.quality,
                AnalysisQuality::label,
                |this, quality| this.stabilizer.quality = quality,
                cx,
            ),
        ));

        section("Stabilization", fields)
    }
}

impl Render for Inspector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = window.theme().colors;

        div()
            .id("inspector")
            .flex()
            .flex_col()
            .w_1_4()
            .h_full()
            .overflow_y_scroll()
            .border_l_1()
            .border_color(colors.border)
            .bg(colors.card)
            .text_color(colors.card_foreground)
            .child(self.tracking_section(cx))
            .child(Separator::horizontal())
            .child(self.motion_section(cx))
            .child(Separator::horizontal())
            .child(self.stabilizer_section(cx))
    }
}

fn section(title: &'static str, fields: impl IntoIterator<Item = AnyElement>) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .flex_none()
        .gap_3()
        .p_3()
        .child(
            div()
                .text_sm()
                .font_weight(FontWeight::SEMIBOLD)
                .child(title),
        )
        .children(fields)
}

fn field(label: &'static str, control: impl IntoElement) -> AnyElement {
    div()
        .flex()
        .flex_col()
        .gap_1p5()
        .child(Label::new(label).muted())
        .child(control)
        .into_any_element()
}

/// One button per option, the selected one filled.
fn choices<T: Copy + PartialEq + 'static>(
    id: &'static str,
    options: &[T],
    selected: T,
    label: fn(T) -> &'static str,
    select: fn(&mut Inspector, T),
    cx: &mut Context<Inspector>,
) -> impl IntoElement {
    div()
        .flex()
        .flex_wrap()
        .gap_1()
        .children(options.iter().enumerate().map(|(index, &option)| {
            Button::new((id, index), label(option))
                .size(ButtonSize::Small)
                .variant(if option == selected {
                    ButtonVariant::Default
                } else {
                    ButtonVariant::Outline
                })
                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                    select(this, option);
                    cx.notify();
                }))
        }))
}

fn toggle(
    id: &'static str,
    label: &'static str,
    on: bool,
    locked: bool,
    flip: fn(&mut Inspector),
    cx: &mut Context<Inspector>,
) -> Button {
    Button::new(id, label)
        .size(ButtonSize::Small)
        .variant(if on {
            ButtonVariant::Default
        } else {
            ButtonVariant::Outline
        })
        .disabled(locked)
        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
            flip(this);
            cx.notify();
        }))
}

/// A 0 to 1 value shown as a percentage, stepped in tenths.
fn stepper(
    id: &'static str,
    value: f64,
    disabled: bool,
    change: fn(&mut Inspector, f64),
    cx: &mut Context<Inspector>,
) -> impl IntoElement {
    let button = |index: usize, label: &'static str, delta: f64, at_limit: bool| {
        Button::new((id, index), label)
            .size(ButtonSize::IconSmall)
            .variant(ButtonVariant::Outline)
            .disabled(disabled || at_limit)
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                change(this, delta);
                cx.notify();
            }))
    };

    div()
        .flex()
        .items_center()
        .gap_2()
        .child(button(0, "−", -0.1, value <= 0.0))
        .child(
            div()
                .w(px(40.0))
                .flex()
                .justify_center()
                .text_xs()
                .when(disabled, |this| this.opacity(0.5))
                .child(SharedString::from(format!("{:.0}%", value * 100.0))),
        )
        .child(button(1, "+", 0.1, value >= 1.0))
}

fn step_fraction(value: f64, delta: f64) -> f64 {
    ((value + delta) * 10.0).round().clamp(0.0, 10.0) / 10.0
}

fn resolution_label(size: usize) -> &'static str {
    match size {
        640 => "640 px",
        960 => "960 px",
        1280 => "1280 px",
        _ => "1920 px",
    }
}

#[cfg(test)]
mod tests {
    use super::step_fraction;

    #[test]
    fn steps_stay_on_tenths_within_range() {
        assert_eq!(step_fraction(0.5, 0.1), 0.6);
        assert_eq!(step_fraction(0.3, -0.1), 0.2);
        assert_eq!(step_fraction(0.95, 0.1), 1.0);
        assert_eq!(step_fraction(0.0, -0.1), 0.0);
    }
}
