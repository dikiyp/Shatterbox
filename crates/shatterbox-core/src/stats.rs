//! Frame timing displayed by the debug overlay.

/// Smoothed frame statistics. Times are in milliseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub frame_time_ms: f32,
    pub smoothed_frame_time_ms: f32,
    pub fps: f32,
    pub frame_index: u64,
}

impl FrameStats {
    /// Records a frame delta in seconds.
    pub fn update(&mut self, dt_seconds: f32) {
        if !dt_seconds.is_finite() || dt_seconds <= 0.0 {
            return;
        }
        let frame_time_ms = dt_seconds * 1000.0;
        self.frame_time_ms = frame_time_ms;
        if self.frame_index == 0 {
            self.smoothed_frame_time_ms = frame_time_ms;
        } else {
            // ~10-frame half-life at 144 fps, but stable on slower displays.
            let alpha = 1.0 - (-dt_seconds / 0.12).exp();
            self.smoothed_frame_time_ms += (frame_time_ms - self.smoothed_frame_time_ms) * alpha;
        }
        self.fps = if self.smoothed_frame_time_ms > f32::EPSILON {
            1000.0 / self.smoothed_frame_time_ms
        } else {
            0.0
        };
        self.frame_index = self.frame_index.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_144_fps() {
        let mut stats = FrameStats::default();
        for _ in 0..120 {
            stats.update(1.0 / 144.0);
        }
        assert!((stats.fps - 144.0).abs() < 2.0, "fps = {}", stats.fps);
        assert!((stats.frame_time_ms - 1000.0 / 144.0).abs() < 0.1);
    }

    #[test]
    fn invalid_deltas_are_ignored() {
        let mut stats = FrameStats::default();
        stats.update(-1.0);
        stats.update(f32::NAN);
        assert_eq!(stats.frame_index, 0);
    }
}
