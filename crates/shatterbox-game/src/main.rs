//! Shatterbox binary entry point.

mod game;
mod player;
mod scenario;
mod sky_config;
mod smoke;

use anyhow::Result;
use shatterbox_platform::{PlatformConfig, run};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,shatterbox=debug"));
    // The scenario report counts WARN/ERROR records, so the counter layer has
    // to see every event the fmt layer prints.
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(&scenario::LOG_COUNTERS)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let scenario_options = scenario::ScenarioOptions::parse(&args)?;

    let config = PlatformConfig {
        title: String::from("Shatterbox — Phase 3"),
        width: std::env::var("SHATTERBOX_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1600),
        height: std::env::var("SHATTERBOX_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(900),
        fullscreen: std::env::var("SHATTERBOX_FULLSCREEN")
            .ok()
            .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true")),
    };

    match scenario_options {
        Some(options) => {
            tracing::info!("scenario run: {}", options.describe());
            let game = game::Game::with_scenario(options)?;
            if let Err(error) = run(config, game) {
                tracing::error!("scenario loop failed: {error:#}");
            }
            // The report is written when the scenario finishes; the exit code
            // has to come from it, not from the windowing loop.
            std::process::exit(scenario::exit_code());
        }
        None => run(config, game::Game::new()),
    }
}
