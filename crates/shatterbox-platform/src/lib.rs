//! Window lifecycle, winit integration and platform input mapping.

mod app;
mod input;

pub use app::{App, AppEvent, PlatformConfig, run};
pub use input::Input;
