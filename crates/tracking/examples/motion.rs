//! Tracking and stabilization from the command line, on YUV4MPEG2 video
//! (any video converts with `ffmpeg -i in.mp4 -pix_fmt yuv420p out.y4m`).
//!
//!   motion track <in.y4m> <x> <y> <w> <h> [--tracker features|object|frame]
//!                [--model translation|similarity|affine|perspective] [--overlay <out.y4m>]
//!   motion lock <in.y4m> <out.y4m> <x> <y> <w> <h> [--tracker ...] [--strength 0.5]
//!   motion stabilize <in.y4m> <out.y4m> [--mode simple|standard|cinematic]
//!                [--framing crop|scale|extend] [--fill black|clamp|mirror] [--amount 0.6]
//!
//! `track` prints one CSV row per frame with the region's corners. `lock`
//! reframes the video so the target stays where it started. `stabilize`
//! smooths the camera's path.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::process::ExitCode;

use tracking::{
    CameraMotionEstimator, Continuity, EdgeFill, Frame, Framing, MotionFilter, MotionModel,
    MotionPath, MotionSettings, Point, Quad, StabilizationMode, StabilizerSettings, Track, Tracker,
    TrackerKind, TrackerSettings, Transform, stabilize, warp,
};

type Result<T> = std::result::Result<T, String>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("track") => track(&args[1..]),
        Some("lock") => lock(&args[1..]),
        Some("stabilize") => stabilize_video(&args[1..]),
        _ => Err("usage: motion <track|lock|stabilize> ... (see the file header)".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("motion: {message}");
            ExitCode::FAILURE
        }
    }
}

fn track(args: &[String]) -> Result<()> {
    let (positional, options) = split_args(args);
    let [input, x, y, w, h] = positional[..] else {
        return Err("track <in.y4m> <x> <y> <w> <h> [options]".into());
    };
    let region = parse_region(x, y, w, h)?;
    let settings = tracker_settings(&options)?;
    let mut reader = Y4mReader::open(input)?;
    let mut overlay = option(&options, "overlay")
        .map(|path| Y4mWriter::create(path, &reader.header))
        .transpose()?;

    let first = reader.next_frame()?.ok_or("the video has no frames")?;
    let mut tracker = Tracker::new(settings, &first.luma, region).map_err(|e| e.to_string())?;
    println!("frame,continuity,confidence,x0,y0,x1,y1,x2,y2,x3,y3");
    print_row(0, tracker.last());
    if let Some(writer) = &mut overlay {
        writer.write(&first.with_quad(&region))?;
    }

    let mut index = 1;
    while let Some(frame) = reader.next_frame()? {
        let sample = tracker.track(&frame.luma).map_err(|e| e.to_string())?;
        print_row(index, &sample);
        if let Some(writer) = &mut overlay {
            writer.write(&frame.with_quad(&sample.quad))?;
        }
        index += 1;
    }
    Ok(())
}

fn lock(args: &[String]) -> Result<()> {
    let (positional, options) = split_args(args);
    let [input, output, x, y, w, h] = positional[..] else {
        return Err("lock <in.y4m> <out.y4m> <x> <y> <w> <h> [options]".into());
    };
    let region = parse_region(x, y, w, h)?;
    let settings = tracker_settings(&options)?;
    let strength = parse_number(option(&options, "strength").unwrap_or("0.5"))?;

    // Track the whole clip first, then reframe every frame by the filtered
    // inverse motion so the target holds still.
    let frames = Y4mReader::open(input)?.read_all()?;
    let first = frames.first().ok_or("the video has no frames")?;
    let mut tracker = Tracker::new(settings, &first.luma, region).map_err(|e| e.to_string())?;
    let mut track = Track::new(region, settings.motion_model);
    track.record(0, *tracker.last());
    for (index, frame) in frames.iter().enumerate().skip(1) {
        let sample = tracker.track(&frame.luma).map_err(|e| e.to_string())?;
        track.record(index as u32, sample);
    }

    let path = MotionPath::new(
        &track,
        &MotionSettings {
            filter: MotionFilter::Lock,
            strength,
            ..MotionSettings::default()
        },
    );
    let mut writer = Y4mWriter::create(output, &Y4mReader::open(input)?.header)?;
    for (index, frame) in frames.iter().enumerate() {
        writer.write(&frame.warped(&path.reframe(index as u32), EdgeFill::Black))?;
    }
    let lost = track
        .segments()
        .iter()
        .flat_map(|s| s.samples())
        .filter(|s| s.continuity != Continuity::Tracked)
        .count();
    eprintln!(
        "locked {} frames ({lost} without a fresh match)",
        frames.len()
    );
    Ok(())
}

fn stabilize_video(args: &[String]) -> Result<()> {
    let (positional, options) = split_args(args);
    let [input, output] = positional[..] else {
        return Err("stabilize <in.y4m> <out.y4m> [options]".into());
    };
    let settings = StabilizerSettings {
        mode: match option(&options, "mode").unwrap_or("standard") {
            "simple" => StabilizationMode::Simple,
            "standard" => StabilizationMode::Standard,
            "cinematic" => StabilizationMode::Cinematic,
            other => return Err(format!("unknown mode {other}")),
        },
        framing: match option(&options, "framing").unwrap_or("crop") {
            "crop" => Framing::Crop,
            "scale" => Framing::AutoScale,
            "extend" => Framing::Extend,
            other => return Err(format!("unknown framing {other}")),
        },
        edge_fill: match option(&options, "fill").unwrap_or("black") {
            "black" => EdgeFill::Black,
            "clamp" => EdgeFill::Clamp,
            "mirror" => EdgeFill::Mirror,
            other => return Err(format!("unknown fill {other}")),
        },
        amount: parse_number(option(&options, "amount").unwrap_or("0.6"))?,
        ..StabilizerSettings::default()
    };

    let frames = Y4mReader::open(input)?.read_all()?;
    let mut estimator = CameraMotionEstimator::new(&settings);
    for frame in &frames {
        estimator.push(&frame.luma).map_err(|e| e.to_string())?;
    }
    let result = stabilize(&estimator.finish(), &settings);

    let mut writer = Y4mWriter::create(output, &Y4mReader::open(input)?.header)?;
    for (frame, transform) in frames.iter().zip(&result.transforms) {
        writer.write(&frame.warped(transform, result.edge_fill))?;
    }
    let max_zoom = result.zooms.iter().copied().fold(1.0, f64::max);
    eprintln!(
        "stabilized {} frames, {} weak, zoom up to {max_zoom:.3}",
        frames.len(),
        result.weak_frames
    );
    Ok(())
}

fn print_row(index: usize, sample: &tracking::TrackSample) {
    let corners: Vec<String> = sample
        .quad
        .corners
        .iter()
        .map(|p| format!("{:.2},{:.2}", p.x, p.y))
        .collect();
    println!(
        "{index},{:?},{:.3},{}",
        sample.continuity,
        sample.confidence,
        corners.join(",")
    );
}

fn tracker_settings(options: &[(String, String)]) -> Result<TrackerSettings> {
    let kind = match option(options, "tracker").unwrap_or("features") {
        "features" => TrackerKind::Features,
        "object" => TrackerKind::Object,
        "frame" => TrackerKind::Frame,
        other => return Err(format!("unknown tracker {other}")),
    };
    let motion_model = match option(options, "model").unwrap_or("similarity") {
        "translation" => MotionModel::Translation,
        "similarity" => MotionModel::Similarity,
        "affine" => MotionModel::Affine,
        "perspective" => MotionModel::Perspective,
        other => return Err(format!("unknown model {other}")),
    };
    Ok(TrackerSettings {
        kind,
        motion_model,
        ..TrackerSettings::default()
    })
}

fn split_args(args: &[String]) -> (Vec<&str>, Vec<(String, String)>) {
    let mut positional = Vec::new();
    let mut options = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.strip_prefix("--") {
            Some(name) => {
                options.push((name.to_string(), iter.next().cloned().unwrap_or_default()))
            }
            None => positional.push(arg.as_str()),
        }
    }
    (positional, options)
}

fn option<'a>(options: &'a [(String, String)], name: &str) -> Option<&'a str> {
    options
        .iter()
        .rev()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

fn parse_number(text: &str) -> Result<f64> {
    text.parse().map_err(|_| format!("not a number: {text}"))
}

fn parse_region(x: &str, y: &str, w: &str, h: &str) -> Result<Quad> {
    Ok(Quad::from_rect(
        parse_number(x)?,
        parse_number(y)?,
        parse_number(w)?,
        parse_number(h)?,
    ))
}

/// One decoded frame: luma for analysis, chroma carried along for output.
struct VideoFrame {
    luma: Frame,
    chroma: Vec<Frame>,
    /// Chroma subsampling factor (2 for 4:2:0, 1 for 4:4:4).
    subsampling: usize,
}

impl VideoFrame {
    fn warped(&self, transform: &Transform, fill: EdgeFill) -> Self {
        let s = self.subsampling as f64;
        // The same motion in chroma-plane coordinates.
        let chroma_transform = Transform::scaling(s)
            .then(transform)
            .then(&Transform::scaling(1.0 / s));
        Self {
            luma: warp(&self.luma, transform, fill, 16),
            chroma: self
                .chroma
                .iter()
                .map(|plane| warp(plane, &chroma_transform, fill, 128))
                .collect(),
            subsampling: self.subsampling,
        }
    }

    /// A copy with the outline of `quad` drawn into the luma plane.
    fn with_quad(&self, quad: &Quad) -> Self {
        let (width, height) = (self.luma.width(), self.luma.height());
        let mut luma = self.luma.data().to_vec();
        for i in 0..4 {
            let (a, b) = (quad.corners[i], quad.corners[(i + 1) % 4]);
            let steps = (a.distance(b).ceil() as usize).max(1);
            for step in 0..=steps {
                let p: Point = a + (b - a) * (step as f64 / steps as f64);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1)] {
                    let (x, y) = (p.x.round() as isize + dx, p.y.round() as isize + dy);
                    if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                        luma[y as usize * width + x as usize] = 235;
                    }
                }
            }
        }
        Self {
            luma: Frame::from_luma(width, height, luma).expect("same size"),
            chroma: self.chroma.clone(),
            subsampling: self.subsampling,
        }
    }
}

struct Y4mReader {
    header: String,
    width: usize,
    height: usize,
    subsampling: Option<usize>,
    input: BufReader<File>,
}

impl Y4mReader {
    fn open(path: &str) -> Result<Self> {
        let mut input = BufReader::new(File::open(path).map_err(|e| format!("{path}: {e}"))?);
        let mut header = String::new();
        input.read_line(&mut header).map_err(|e| e.to_string())?;
        if !header.starts_with("YUV4MPEG2 ") {
            return Err(format!("{path} is not a YUV4MPEG2 file"));
        }
        let header = header.trim_end().to_string();
        let field = |tag: char| {
            header
                .split(' ')
                .find_map(|token| token.strip_prefix(tag))
                .map(str::to_string)
        };
        let width: usize = field('W')
            .and_then(|v| v.parse().ok())
            .ok_or("missing width")?;
        let height: usize = field('H')
            .and_then(|v| v.parse().ok())
            .ok_or("missing height")?;
        let colorspace = field('C').unwrap_or_else(|| "420jpeg".into());
        let subsampling = if colorspace.starts_with("420") {
            Some(2)
        } else if colorspace.starts_with("444") && !colorspace.contains("alpha") {
            Some(1)
        } else if colorspace.starts_with("mono") {
            None
        } else {
            return Err(format!(
                "unsupported colorspace C{colorspace}; use -pix_fmt yuv420p"
            ));
        };
        Ok(Self {
            header,
            width,
            height,
            subsampling,
            input,
        })
    }

    fn next_frame(&mut self) -> Result<Option<VideoFrame>> {
        let mut marker = String::new();
        if self
            .input
            .read_line(&mut marker)
            .map_err(|e| e.to_string())?
            == 0
        {
            return Ok(None);
        }
        if !marker.starts_with("FRAME") {
            return Err("corrupt frame marker".into());
        }
        let mut read_plane = |width: usize, height: usize| -> Result<Frame> {
            let mut data = vec![0; width * height];
            self.input
                .read_exact(&mut data)
                .map_err(|e| e.to_string())?;
            Frame::from_luma(width, height, data).map_err(|e| e.to_string())
        };
        let luma = read_plane(self.width, self.height)?;
        let (chroma, subsampling) = match self.subsampling {
            Some(s) => {
                let (w, h) = (self.width.div_ceil(s), self.height.div_ceil(s));
                (vec![read_plane(w, h)?, read_plane(w, h)?], s)
            }
            None => (Vec::new(), 1),
        };
        Ok(Some(VideoFrame {
            luma,
            chroma,
            subsampling,
        }))
    }

    fn read_all(mut self) -> Result<Vec<VideoFrame>> {
        let mut frames = Vec::new();
        while let Some(frame) = self.next_frame()? {
            frames.push(frame);
        }
        Ok(frames)
    }
}

struct Y4mWriter {
    output: BufWriter<File>,
}

impl Y4mWriter {
    fn create(path: &str, header: &str) -> Result<Self> {
        let mut output = BufWriter::new(File::create(path).map_err(|e| format!("{path}: {e}"))?);
        writeln!(output, "{header}").map_err(|e| e.to_string())?;
        Ok(Self { output })
    }

    fn write(&mut self, frame: &VideoFrame) -> Result<()> {
        self.output
            .write_all(b"FRAME\n")
            .map_err(|e| e.to_string())?;
        for plane in std::iter::once(&frame.luma).chain(&frame.chroma) {
            self.output
                .write_all(plane.data())
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
