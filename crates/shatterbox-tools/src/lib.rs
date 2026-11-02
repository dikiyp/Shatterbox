//! Offline helpers for the autotest harness: PNG diffing.
//!
//! The harness writes one PNG per screenshot; this crate compares two of them
//! so an A/B run (same seed, same scenario, render change) can be judged by
//! numbers instead of by eye.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;

/// One decoded 8-bit RGBA image.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    #[must_use]
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }

    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let index = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[index],
            self.pixels[index + 1],
            self.pixels[index + 2],
            self.pixels[index + 3],
        ]
    }
}

/// Numbers describing how different two images are.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct DiffReport {
    pub width: u32,
    pub height: u32,
    /// Mean absolute difference of the RGB channels, 0..255.
    pub mean_abs: f32,
    /// Fraction of pixels whose RGB difference exceeds the threshold, 0..1.
    pub changed_fraction: f32,
    /// Largest single-channel difference found.
    pub max_abs: u8,
}

/// Reads a PNG into tightly packed RGBA8.
pub fn decode_png(path: &Path) -> Result<RgbaImage> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .with_context(|| format!("failed to read PNG header of {}", path.display()))?;
    let mut buffer = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buffer)
        .with_context(|| format!("failed to decode {}", path.display()))?;
    buffer.truncate(info.buffer_size());
    let pixels = match info.color_type {
        png::ColorType::Rgba => buffer,
        png::ColorType::Rgb => buffer
            .chunks_exact(3)
            .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
            .collect(),
        png::ColorType::Grayscale => buffer
            .iter()
            .flat_map(|value| [*value, *value, *value, 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => buffer
            .chunks_exact(2)
            .flat_map(|ga| [ga[0], ga[0], ga[0], ga[1]])
            .collect(),
        png::ColorType::Indexed => bail!("indexed PNG is not supported: {}", path.display()),
    };
    Ok(RgbaImage::new(info.width, info.height, pixels))
}

/// Encodes tightly packed RGBA8 to a PNG file.
pub fn write_png(path: &Path, image: &RgbaImage) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), image.width, image.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .with_context(|| format!("failed to write PNG header for {}", path.display()))?;
    writer
        .write_image_data(&image.pixels)
        .with_context(|| format!("failed to write PNG pixels for {}", path.display()))?;
    Ok(())
}

/// Compares two images of identical size. Returns the report plus a diff image
/// where every differing pixel is amplified to full brightness.
pub fn diff(a: &RgbaImage, b: &RgbaImage, threshold: u8) -> Result<(DiffReport, RgbaImage)> {
    if a.width != b.width || a.height != b.height {
        bail!(
            "size mismatch: {}x{} vs {}x{}",
            a.width,
            a.height,
            b.width,
            b.height
        );
    }
    let pixel_count = (a.width as u64) * (a.height as u64);
    let mut sum = 0u64;
    let mut changed = 0u64;
    let mut max_abs = 0u8;
    let mut diff_pixels = Vec::with_capacity(a.pixels.len());
    for (left, right) in a.pixels.chunks_exact(4).zip(b.pixels.chunks_exact(4)) {
        let delta = [
            left[0].abs_diff(right[0]),
            left[1].abs_diff(right[1]),
            left[2].abs_diff(right[2]),
        ];
        let channel_max = delta[0].max(delta[1]).max(delta[2]);
        max_abs = max_abs.max(channel_max);
        sum += u64::from(delta[0]) + u64::from(delta[1]) + u64::from(delta[2]);
        if channel_max > threshold {
            changed += 1;
        }
        diff_pixels.extend_from_slice(&[
            channel_max.saturating_mul(8),
            channel_max.saturating_mul(4),
            channel_max.saturating_mul(2),
            255,
        ]);
    }
    let channels = pixel_count.max(1) * 3;
    Ok((
        DiffReport {
            width: a.width,
            height: a.height,
            mean_abs: sum as f32 / channels as f32,
            changed_fraction: changed as f32 / pixel_count.max(1) as f32,
            max_abs,
        },
        RgbaImage::new(a.width, a.height, diff_pixels),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, color: [u8; 4]) -> RgbaImage {
        RgbaImage::new(width, height, color.repeat((width * height) as usize))
    }

    #[test]
    fn identical_images_report_zero_difference() {
        let image = solid(4, 4, [10, 20, 30, 255]);
        let (report, _) = diff(&image, &image, 0).expect("diff");
        assert_eq!(report.mean_abs, 0.0);
        assert_eq!(report.changed_fraction, 0.0);
        assert_eq!(report.max_abs, 0);
    }

    #[test]
    fn one_dark_pixel_counts_as_changed() {
        let a = solid(2, 1, [0, 0, 0, 255]);
        let mut b = a.clone();
        b.pixels[4] = 40;
        let (report, diff_image) = diff(&a, &b, 2).expect("diff");
        assert!((report.mean_abs - 40.0 / 6.0).abs() < 1e-4);
        assert!((report.changed_fraction - 0.5).abs() < 1e-6);
        assert_eq!(report.max_abs, 40);
        assert_eq!(diff_image.pixel(1, 0)[0], 255);
        assert_eq!(diff_image.pixel(0, 0)[0], 0);
    }

    #[test]
    fn size_mismatch_is_an_error() {
        let a = solid(2, 2, [1, 2, 3, 255]);
        let b = solid(2, 3, [1, 2, 3, 255]);
        assert!(diff(&a, &b, 0).is_err());
    }

    #[test]
    fn png_round_trip_preserves_pixels() {
        let dir = std::env::temp_dir().join("shatterbox-tools-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("round-trip.png");
        let image = solid(3, 2, [7, 8, 9, 255]);
        write_png(&path, &image).expect("write");
        let decoded = decode_png(&path).expect("decode");
        assert_eq!(decoded, image);
        let _ = std::fs::remove_file(path);
    }
}
