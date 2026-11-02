//! Frame presentation and debug UI rendering.
//!
//! The renderer owns the egui integration and the swapchain clear pass. Game
//! code supplies only the UI closure; it never touches `wgpu` types directly.

mod renderer;

pub use renderer::{
    AreaLight, CapturedFrame, DdgiQuality, DebrisView, FrameOutcome, MAX_DEBRIS_VOXELS,
    MAX_RENDER_DEBRIS, Particle, PointLight, RenderEffects, RenderStats, Renderer, ShadowMode,
};
