//! `config/sky.ron` — user-tweakable sun and fog defaults.
//!
//! The F1 overlay writes this file when a sky slider is released, so the next
//! launch starts where the last session left off. Scenario runs ignore it on
//! purpose: their sky comes from the `SHATTERBOX_*` env vars and the renderer
//! defaults, so a screenshot never depends on a local config file.

use serde::{Deserialize, Serialize};

const PATH: &str = "config/sky.ron";

/// Sun power multiplier (linear HDR) and the two fog parameters, all in the
/// same ranges the renderer clamps to: sun 0..=5, fog density 0..=0.15, fog
/// height falloff 0.5..=60 m.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SkyConfig {
    pub sun_intensity: f32,
    pub fog_density: f32,
    pub fog_height: f32,
}

impl Default for SkyConfig {
    /// Matches the renderer defaults (`SHATTERBOX_SUN_INTENSITY=1.0`,
    /// `SHATTERBOX_FOG=0.012`, `SHATTERBOX_FOG_HEIGHT=5.0`), so a fresh
    /// checkout without the file looks identical.
    fn default() -> Self {
        Self {
            sun_intensity: 1.0,
            fog_density: 0.012,
            fog_height: 5.0,
        }
    }
}

impl SkyConfig {
    /// Reads `config/sky.ron`, falling back to the defaults when the file is
    /// missing (fresh checkout) or malformed (hand edit).
    pub fn load() -> Self {
        let text = match std::fs::read_to_string(PATH) {
            Ok(text) => text.trim_start_matches('\u{feff}').to_owned(),
            Err(error) => {
                tracing::debug!("no {PATH} ({error}); using the default sky");
                return Self::default();
            }
        };
        match ron::from_str(&text) {
            Ok(config) => config,
            Err(error) => {
                tracing::warn!("cannot parse {PATH} ({error}); using the default sky");
                Self::default()
            }
        }
    }

    /// Writes `config/sky.ron`. Best effort: a read-only checkout logs the
    /// failure and keeps running with the in-memory values.
    pub fn save(&self) {
        let text = ron::to_string(self).expect("sky config serializes");
        let write = || -> std::io::Result<()> {
            std::fs::create_dir_all(std::path::Path::new(PATH).parent().expect("non-root"))?;
            std::fs::write(PATH, text.as_bytes())
        };
        match write() {
            Ok(()) => tracing::info!(path = PATH, "sky config saved"),
            Err(error) => tracing::warn!("cannot write {PATH} ({error}); sky stays session-only"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_ron() {
        let text = ron::to_string(&SkyConfig::default()).expect("serialize");
        let parsed: SkyConfig = ron::from_str(&text).expect("parse");
        assert_eq!(parsed, SkyConfig::default());
    }

    #[test]
    fn parses_the_shipped_file_layout() {
        let text = "(\n    sun_intensity: 2.0,\n    fog_density: 0.05,\n    fog_height: 12.0,\n)";
        let parsed: SkyConfig = ron::from_str(text).expect("parse");
        assert_eq!(
            parsed,
            SkyConfig {
                sun_intensity: 2.0,
                fog_density: 0.05,
                fog_height: 12.0
            }
        );
    }
}
