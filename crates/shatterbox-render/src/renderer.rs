//! Phase 2 renderer: voxel G-buffer, deferred PBR lighting, soft stochastic
//! sun shadows, bloom and temporal upscale.
//!
//! Frame graph:
//! 1. `gbuffer.wgsl` — primary DDA into albedo/AO, normal/roughness,
//!    material/visibility and depth at internal resolution.
//! 2. `lighting.wgsl` — deferred PBR sun + sky + point lights + fog into HDR.
//! 3. `temporal.wgsl` — depth-reprojected TAAU to output resolution.
//! 4. `bloom.wgsl` — quarter-res prefilter and separable blur.
//! 5. `post.wgsl` — bloom, ACES, grade, vignette, gamma to the swapchain.
//! 6. egui overlay at native resolution.
//!
//! Game code passes a `FreeCamera` and a `VoxelWorld`; it never sees wgpu
//! types.

use std::borrow::Cow;
use std::fs;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use egui_wgpu::{Renderer as EguiRenderer, RendererOptions, ScreenDescriptor};
use glam::{IVec3, Mat4, Vec2, Vec3};
use shatterbox_core::FreeCamera;
use shatterbox_gpu::GpuContext;
use shatterbox_voxel::{
    BRICKS_X, BRICKS_Y, BRICKS_Z, GPU_WORDS_PER_BRICK, MASK_WORDS_PER_BRICK, VoxelWorld, WORLD_MAX,
    WORLD_MIN,
};
use winit::event::WindowEvent;
use winit::window::Window;

const GBUFFER_ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const GBUFFER_NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const GBUFFER_MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const WORKGROUP: u32 = 8;
const TIMESTAMP_READ_INTERVAL: u64 = 30;
/// Two timestamps per measured pass. Pairs are independent: `total_ms` spans
/// from the first G-buffer timestamp to the post-pass end.
const TIMESTAMP_COUNT: u32 = 36;

// Pass pairs, in submission order. The numeric order does not have to match,
// but keeping it sequential makes the readback indices easy to audit.
const TS_GBUFFER: u32 = 0;
const TS_LIGHTING: u32 = 1;
const TS_UPSCALE: u32 = 2;
const TS_BLOOM_PREFILTER: u32 = 3;
const TS_POST: u32 = 4;
const TS_SHADOW: u32 = 5;
const TS_LIGHT_SHADOW: u32 = 6;
const TS_FOG: u32 = 7;
const TS_GI: u32 = 8;
const TS_BLOOM_BLUR_H: u32 = 9;
const TS_BLOOM_BLUR_V: u32 = 10;
/// FXAA at internal resolution, between lighting and the upscale.
const TS_FXAA: u32 = 11;
/// Explosion smoke/dust billboards, drawn into the HDR target after lighting.
const TS_PARTICLES: u32 = 12;
/// Per-tile debris candidate masks, built before the G-buffer.
const TS_DEBRIS_CULL: u32 = 13;
/// Depth-aware upsample of the half-res smoke buffer into the HDR image.
const TS_PARTICLES_COMPOSITE: u32 = 14;
/// DDGI albedo injection (G-buffer -> 1 m colour volume).
const TS_DDGI_INJECT: u32 = 15;
/// DDGI probe ray trace.
const TS_DDGI_UPDATE: u32 = 16;
/// DDGI cosine convolution + hysteresis.
const TS_DDGI_BLEND: u32 = 17;
/// Largest internal framebuffer the tile pass allocates masks for: 4K/8 px.
const DEBRIS_TILES_MAX: usize = (3840 / 8) * (2160 / 8);
const MAX_LIGHTS: usize = 12;
/// Hard cap on uploaded smoke/dust billboards. The game retires its own list at
/// 560 puffs; the extra headroom absorbs one frame of overlap between the two
/// caps without silently dropping the newest blast.
const MAX_PARTICLES: usize = 640;
const BLOOM_DOWNSCALE: u32 = 4;
/// Voxel irradiance volume used by the cone trace. The arena is 256x64x256 m,
/// so one cell is 8 m x 8 m x 8 m: coarse, but the cone trace only needs
/// low-frequency diffuse light.
// Irradiance probe grid over the arena. 64x16x64 over 256x64x256 m gives a
// 4 m cell: fine enough that a 12 m house owns a block of probes instead of
// interpolating two outside cells across its walls. The grid is rebuilt only
// when the world changes, so the extra probes are not a per-frame cost.
const PROBE_DIMS: (u32, u32, u32) = (64, 16, 64);

// --- DDGI ------------------------------------------------------------------
// Octahedral irradiance probes on an 8 m grid, updated incrementally and
// sampled by `gi.wgsl` for coloured diffuse bounce. The coarse `probe.wgsl`
// volume stays untouched: VCT and fog still read it, and it is what the
// colourless path falls back to when the DDGI bit is off.
//
// Grid: 64 x 16 x 64 probes over the 256 x 64 x 256 m arena = 4 m spacing.
// The test room is only 16 m across, so 8 m left two probes per axis inside it
// and the trilinear blend read mostly buried probes; 4 m gives 4x4x2 interior
// probes and the painted-wall bounce becomes readable.
// Atlas: each probe owns an 8x8 tile; columns are probe x, rows are
// (z * GRID_Y + y), so the atlas is 512 x 8192 texels.
pub const DDGI_PROBE_SPACING_M: f32 = 4.0;
const DDGI_GRID: (u32, u32, u32) = (64, 16, 64);
const DDGI_TILE: u32 = 8;
const DDGI_ATLAS_SIZE: (u32, u32) = (
    DDGI_GRID.0 * DDGI_TILE,
    DDGI_GRID.1 * DDGI_GRID.2 * DDGI_TILE,
);
/// 1 m albedo volume injected from the G-buffer. rgb = albedo,
/// a = 1 + emissive (a <= 0.5 = unknown). 256*64*256 rgba16f = 32 MB.
const DDGI_ALBEDO_DIMS: (u32, u32, u32) = (256, 64, 256);
/// Fraction of the previous probe value kept per update. Each z-slice is
/// refreshed once every 64 frames, so 0.85 still lets a changed room converge
/// in a couple of seconds while a single stray frame cannot flash a texel.
const DDGI_HYSTERESIS: f32 = 0.85;
/// Hysteresis used near a fresh crater for the priority window. 0.45 keeps
/// ~91% after three refreshes (~1.5 s at two slices per frame), which is the
/// gi_blast acceptance criterion.
const DDGI_PRIORITY_HYSTERESIS: f32 = 0.45;

/// Dynamic smoke density volume: 4 m cells over the 256x64x256 m arena. The
/// game advects these cells on the CPU (16k cells is cheap) and uploads R8.
pub const SMOKE_DIMS: (u32, u32, u32) = (64, 16, 64);

/// Capacity of the incremental world upload. The test arena uses ~5.7k bricks;
/// destruction frees ids and refilling holes reuses them, so the headroom covers
/// an entire match without ever moving the storage buffers.
const MAX_GPU_BRICKS: usize = 16_384;
/// Dynamic debris visible in one frame. Every piece stays in the physics world
/// until the simulation budget retires it; only the newest handful are traced by
/// the G-buffer. The game uses this to pick the pieces it uploads.
/// Debris instances drawn per frame. Raised from 64 after the per-frame cost
/// was measured: the instance buffer is tiny, so the old cap only cut rubble
/// visibility (a blast routinely produces 100-200 pieces).
pub const MAX_RENDER_DEBRIS: usize = 512;
/// One `u16` voxel is uploaded as one `u32` word so the shader can index it
/// without unpacking. 512k voxels is ~85 debris pieces at the 8–12 voxel radius
/// used by the gun, and only 2 MB of VRAM.
pub const MAX_DEBRIS_VOXELS: usize = 524_288;

/// Radius of the magenta `SHATTERBOX_DEBUG_HIT` marker painted on whatever
/// surface the CPU aim ray hit. Visible in both the swapchain and the harness
/// screenshots, so a crater can be compared against the real aim point.
const DEBUG_HIT_RADIUS_M: f32 = 0.12;

/// Side of the internal-depth window read back for `assert_hit_matches_render`,
/// in texels. Wide enough that the ±`DEPTH_TAP_RADIUS` taps sit inside the
/// crater a single shot opens, so a lateral shift of the image is measurable.
const DEPTH_WINDOW: u32 = 41;
const DEPTH_TAP_RADIUS: i32 = 16;
/// Centre tap first, then the axis neighbours and the diagonals. Offsets are in
/// internal texels.
const DEPTH_TAPS: [[i32; 2]; 9] = [
    [0, 0],
    [-DEPTH_TAP_RADIUS, 0],
    [DEPTH_TAP_RADIUS, 0],
    [0, -DEPTH_TAP_RADIUS],
    [0, DEPTH_TAP_RADIUS],
    [-DEPTH_TAP_RADIUS, -DEPTH_TAP_RADIUS],
    [DEPTH_TAP_RADIUS, -DEPTH_TAP_RADIUS],
    [-DEPTH_TAP_RADIUS, DEPTH_TAP_RADIUS],
    [DEPTH_TAP_RADIUS, DEPTH_TAP_RADIUS],
];

/// A persisted debris piece as seen by the renderer. The transform maps the
/// piece's local metre grid into world space; `voxels` is a dense grid indexed
/// by `x + dims.x * (y + dims.y * z)`.
pub struct DebrisView<'a> {
    pub model: Mat4,
    pub inverse_model: Mat4,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub dims: IVec3,
    pub voxel_offset: u32,
    pub voxels: &'a [u16],
}

/// Shader slots: 0 gbuffer, 1 lighting, 2 temporal, 3 bloom, 4 post, 5 gi,
/// 6 probe, 7 fog, 8 shadow, 9 FSR upscale, 10 local-light shadows,
/// 11 FXAA, 12 explosion particles, 13 per-tile debris candidate lists,
/// 14 DDGI trace, 15 DDGI albedo injection, 16 DDGI blend.
const SHADER_SLOTS: usize = 17;

const SHADER_NAMES: [&str; SHADER_SLOTS] = [
    "gbuffer.wgsl",
    "lighting.wgsl",
    "temporal.wgsl",
    "bloom.wgsl",
    "post.wgsl",
    "gi.wgsl",
    "probe.wgsl",
    "fog.wgsl",
    "shadow.wgsl",
    "fsr.wgsl",
    "light_shadow.wgsl",
    "fxaa.wgsl",
    "particles.wgsl",
    "debris_cull.wgsl",
    "ddgi_update.wgsl",
    "ddgi_inject.wgsl",
    "ddgi_blend.wgsl",
];

const EMBEDDED_SHADERS: [&str; SHADER_SLOTS] = [
    include_str!("../assets/shaders/gbuffer.wgsl"),
    include_str!("../assets/shaders/lighting.wgsl"),
    include_str!("../assets/shaders/temporal.wgsl"),
    include_str!("../assets/shaders/bloom.wgsl"),
    include_str!("../assets/shaders/post.wgsl"),
    include_str!("../assets/shaders/gi.wgsl"),
    include_str!("../assets/shaders/probe.wgsl"),
    include_str!("../assets/shaders/fog.wgsl"),
    include_str!("../assets/shaders/shadow.wgsl"),
    include_str!("../assets/shaders/fsr.wgsl"),
    include_str!("../assets/shaders/light_shadow.wgsl"),
    include_str!("../assets/shaders/fxaa.wgsl"),
    include_str!("../assets/shaders/particles.wgsl"),
    include_str!("../assets/shaders/debris_cull.wgsl"),
    include_str!("../assets/shaders/ddgi_update.wgsl"),
    include_str!("../assets/shaders/ddgi_inject.wgsl"),
    include_str!("../assets/shaders/ddgi_blend.wgsl"),
];

/// Which expensive renderer features are enabled. Exposed to the game so the
/// overlay can toggle each one at runtime without rebuilding pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderEffects {
    /// Voxel DDA shadows: one ray per half-res texel towards the sun and one
    /// per half-res texel per local light (point/capsule/rect).
    pub raytraced_shadows: bool,
    /// Screen-space ambient occlusion, computed at half internal resolution.
    pub ssao: bool,
    /// Screen-space specular reflections traced in `gi.wgsl`.
    pub ssr: bool,
    /// Simplified voxel cone tracing through the probe volume.
    pub vct: bool,
    /// Temporal accumulation with sub-pixel jitter (works at any scale).
    pub taa: bool,
    /// Exponential height fog with voxel-traced sun shafts and light glow.
    pub volumetric_fog: bool,
    /// Light anti-aliasing of voxel silhouettes *before* the upscale (bit 128).
    ///
    /// The internal image is rendered without MSAA and the upscaler only
    /// resamples it, so a 1 internal-pixel staircase becomes a 1.5 output-pixel
    /// staircase. This smooths just the detected luma gradient pixels in
    /// `fsr.wgsl` and leaves the rest untouched. Off by default: it softens the
    /// blocky look that the art direction may want, so it is an explicit flag.
    pub light_aa: bool,
    /// FXAA 3.11 luma-adaptive edge blur at *internal* resolution, before the
    /// upscaler (bit 256). Separate from `light_aa`: that one softens the
    /// silhouette inside the FSR shader at output resolution, this one filters
    /// the whole internal image spatially, with no history and no jitter, so it
    /// can be A/B'd against a raw frame (`SHATTERBOX_FXAA=0`).
    pub fxaa: bool,
    /// Rejects debris pieces against a world bounding sphere before loading
    /// their matrices in `cast_debris` (bit 512). Pure optimisation: the test
    /// only drops pieces the full path would have missed, so the image stays
    /// bit-identical, which is why it is on by default. `SHATTERBOX_DEBRIS_CULL=0`
    /// (or clearing the F1 checkbox) restores the old per-pixel loop for A/B.
    pub debris_cull: bool,
    /// Builds a 512-bit candidate mask per 8x8 pixel tile in
    /// `debris_cull.wgsl` and makes `cast_debris` walk only its own tile's bits
    /// (bit 1024). Without it every pixel transforms and slab-tests every live
    /// piece, which was 18 ms of a 27 ms G-buffer with ~250 pieces. The masks are
    /// conservative (a piece overlapping the tile is always kept), so the image
    /// does not change; `SHATTERBOX_DEBRIS_TILES=0` restores the old path.
    pub debris_tiles: bool,
    /// DDGI probe-atlas update (bit 2048), **off by default**.
    ///
    /// This only *builds* the octahedral irradiance/depth-moment atlas in
    /// `ddgi_update.wgsl`; nothing in `lighting.wgsl` samples it yet, so the
    /// image is unchanged whether it is on or off. It exists so the DDGI work
    /// can grow incrementally behind its own switch instead of being bolted
    /// onto the live VCT path.
    pub ddgi: bool,
    /// Dynamic smoke field (bit 8192), **off by default**.
    ///
    /// The game advects a 64x16x64 density volume around world obstacles and
    /// uploads it every frame; `fog.wgsl` adds it to the existing 16-step march.
    /// Off means the volume stays empty and no simulation runs, so the checkbox
    /// is a pure test switch and costs nothing when cleared.
    pub dynamic_smoke: bool,
}

impl Default for RenderEffects {
    fn default() -> Self {
        Self {
            raytraced_shadows: true,
            ssao: true,
            ssr: true,
            vct: true,
            taa: true,
            volumetric_fog: true,
            light_aa: false,
            fxaa: false,
            debris_cull: true,
            debris_tiles: true,
            ddgi: false,
            dynamic_smoke: false,
        }
    }
}

impl RenderEffects {
    /// Inverse of [`bitmask`](Self::bitmask); used by the scenario harness so a
    /// `.ron` file can name a set of effects with one integer.
    #[must_use]
    pub fn from_bitmask(mask: u32) -> Self {
        Self {
            raytraced_shadows: mask & 1 != 0,
            ssao: mask & 2 != 0,
            vct: mask & 8 != 0,
            taa: mask & 16 != 0,
            volumetric_fog: mask & 32 != 0,
            ssr: mask & 64 != 0,
            light_aa: mask & 128 != 0,
            fxaa: mask & 256 != 0,
            debris_cull: mask & 512 != 0,
            debris_tiles: mask & 1024 != 0,
            ddgi: mask & 2048 != 0,
            dynamic_smoke: mask & 8192 != 0,
        }
    }

    /// Bit layout shared with `params3.x` in every WGSL `Frame` struct:
    /// 1 shadows, 2 SSAO, 8 VCT, 16 TAA, 32 fog, 64 SSR, 128 light AA,
    /// 256 FXAA, 512 debris sphere culling, 1024 per-tile debris masks,
    /// 2048 DDGI atlas update, 8192 dynamic smoke field. Bit 4 (the old
    /// screen-space diffuse bounce) is retired and always clear.
    #[must_use]
    pub fn bitmask(self) -> u32 {
        let mut mask = 0;
        if self.raytraced_shadows {
            mask |= 1;
        }
        if self.ssao {
            mask |= 2;
        }
        if self.ssr {
            mask |= 64;
        }
        if self.vct {
            mask |= 8;
        }
        if self.taa {
            mask |= 16;
        }
        if self.volumetric_fog {
            mask |= 32;
        }
        if self.light_aa {
            mask |= 128;
        }
        if self.fxaa {
            mask |= 256;
        }
        if self.debris_cull {
            mask |= 512;
        }
        if self.debris_tiles {
            mask |= 1024;
        }
        if self.ddgi {
            mask |= 2048;
        }
        if self.dynamic_smoke {
            mask |= 8192;
        }
        mask
    }
}

/// DDGI quality preset. The bit 2048 stays the internal enable flag; this is
/// the user-facing control (`SHATTERBOX_GI`, F1 selector).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DdgiQuality {
    Off,
    /// 4x4 octahedral rays per probe.
    Low,
    /// 8x8 rays per probe, one z-slice per frame.
    Medium,
    /// 8x8 rays with two jittered samples each, two slices per frame.
    High,
}

impl DdgiQuality {
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => Some(Self::Off),
            "low" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    /// `SHATTERBOX_GI`, defaulting to **off** until part 2 is accepted: the
    /// measured budget keeps medium under the 1.5 ms gate, but the addendum
    /// (item 8) enables DDGI by default only after the user's acceptance. F1 or
    /// the environment can still select medium/high for A/B.
    #[must_use]
    pub fn from_env() -> Self {
        std::env::var("SHATTERBOX_GI")
            .ok()
            .and_then(|raw| Self::parse(&raw))
            .unwrap_or(Self::Off)
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// Rays are laid out in a `tile x tile` subset of the 8x8 atlas tile.
    #[must_use]
    pub fn tile(self) -> u32 {
        match self {
            Self::Off | Self::Medium | Self::High => 8,
            Self::Low => 4,
        }
    }

    /// Jittered samples per texel: high traces two rays per direction.
    #[must_use]
    pub fn samples(self) -> u32 {
        match self {
            Self::High => 2,
            _ => 1,
        }
    }

    /// Probe slices refreshed per frame while no edit priority is active.
    #[must_use]
    pub fn slices(self) -> u32 {
        match self {
            Self::High => 2,
            _ => 1,
        }
    }
}

/// Which upscaler produces the output-resolution image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Upscaler {
    /// Depth-reprojected temporal accumulation with bilinear upscale.
    Temporal,
    /// FSR 1 style spatial upscale (EASU/RCAS port). Sharper at low render
    /// scales; keeps the same history blend for anti-aliasing.
    #[default]
    Fsr,
}

impl Upscaler {
    /// Reads `SHATTERBOX_UPSCALER` (`fsr` or `taa`). Defaults to FSR.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var("SHATTERBOX_UPSCALER")
            .unwrap_or_else(|_| "fsr".to_owned())
            .to_ascii_lowercase()
            .as_str()
        {
            "taa" | "temporal" | "taau" => Self::Temporal,
            _ => Self::Fsr,
        }
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Temporal => "TAAU",
            Self::Fsr => "FSR1",
        }
    }
}

/// How the sun shadow buffer is traced and filtered.
///
/// The shadow pass used to run at half internal resolution and the lighting
/// pass sampled it bilinearly, which smeared a 2x2 texel block across every
/// shadow edge: at scale 0.7 that is ~1.4 internal pixels of stair-stepping
/// plus the per-texel ray tilt reading as grain. Every variant below keeps the
/// soft, dithered penumbra; none of them hardens the shadow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowMode {
    /// Trace at half internal resolution, bilinear sample. Original behaviour,
    /// kept as the A/B baseline (`SHATTERBOX_SHADOW_MODE=legacy`).
    HalfRes,
    /// Trace at half resolution but upsample depth/normal-aware
    /// (`upsample_half`), so the 2x2 block only blends where the surface is
    /// continuous. Measured: it changes only 0.13% of pixels, because on a flat
    /// surface depth/normal agree across the 2x2 block and the weights collapse
    /// to plain bilinear, so the grain stays at the half-res texel size.
    Bilateral,
    /// Trace one ray per internal pixel and then run a 3x3 depth/normal
    /// bilateral filter over the result. Grain is one internal pixel, but the
    /// dithered penumbra is averaged inside each surface, so the random
    /// 1-pixel speckles that read as "pepper" beside a long shadow disappear.
    /// Still a soft penumbra: no hard 0/1 edges. Default: measured 10.5 % ->
    /// 3.9 % speckle pixels against a plain full-res trace at the same cost.
    #[default]
    FullResFiltered,
    /// Trace one ray per internal pixel, no filtering at all. Kept as the A/B
    /// baseline for the filtered variant above (grain is one internal pixel,
    /// but the 1-pixel dither speckle stays visible).
    FullRes,
}

impl ShadowMode {
    /// Reads `SHATTERBOX_SHADOW_MODE` (`legacy`/`half`, `full`, `bilateral`,
    /// `filtered`/`denoise`).
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var("SHATTERBOX_SHADOW_MODE")
            .unwrap_or_else(|_| String::new())
            .to_ascii_lowercase()
            .as_str()
        {
            "legacy" | "half" | "halfres" => Self::HalfRes,
            "bilateral" | "depth" => Self::Bilateral,
            "filtered" | "denoise" => Self::FullResFiltered,
            "full" | "fullres" => Self::FullRes,
            _ => Self::FullResFiltered,
        }
    }

    /// Cycles legacy -> full -> bilateral -> filtered -> legacy for the F1
    /// overlay.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::HalfRes => Self::FullRes,
            Self::FullRes => Self::Bilateral,
            Self::Bilateral => Self::FullResFiltered,
            Self::FullResFiltered => Self::HalfRes,
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::HalfRes => "half-res bilinear",
            Self::FullRes => "full-res",
            Self::Bilateral => "half-res bilateral",
            Self::FullResFiltered => "full-res bilateral",
        }
    }

    /// Value written to `params3.z` so `lighting.wgsl` picks the upsample path.
    #[must_use]
    pub fn flag(self) -> f32 {
        match self {
            Self::HalfRes => 0.0,
            Self::FullRes => 1.0,
            Self::Bilateral => 2.0,
            Self::FullResFiltered => 3.0,
        }
    }

    /// Whether the shadow buffer must be allocated at full internal resolution.
    #[must_use]
    pub fn is_full_res(self) -> bool {
        matches!(self, Self::FullRes | Self::FullResFiltered)
    }
}

/// A shadowless point light for the deferred lighting pass.
#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

/// A shadowless *area* light for the deferred lighting pass (Phase 4).
///
/// The lighting pass clamps the shading point onto the emitter and treats the
/// clamped representative point as a point light, which is the standard cheap
/// approximation (Karis 2013). `half_extent` selects the shape:
/// * `> 0.0` — capsule: segment of length `2 * half_extent` centred on
///   `position` and oriented along `axis`;
/// * `< 0.0` — rectangle: `axis` is the emitting normal (one-sided) and
///   `|half_extent|` is the half size along both in-plane axes;
/// * `== 0.0` — degenerate, treated as a point light.
#[derive(Clone, Copy, Debug)]
pub struct AreaLight {
    pub position: [f32; 3],
    pub axis: [f32; 3],
    pub half_extent: f32,
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

impl AreaLight {
    /// Neon tube of `length` metres with a `radius`-metre influence sphere.
    pub fn capsule(
        position: [f32; 3],
        axis: [f32; 3],
        length: f32,
        radius: f32,
        color: [f32; 3],
        intensity: f32,
    ) -> Self {
        Self {
            position,
            axis,
            half_extent: (length * 0.5).max(0.0),
            radius,
            color,
            intensity,
        }
    }

    /// One-sided glowing panel of `size * size` metres.
    pub fn rect(
        position: [f32; 3],
        normal: [f32; 3],
        size: f32,
        radius: f32,
        color: [f32; 3],
        intensity: f32,
    ) -> Self {
        Self {
            position,
            axis: normal,
            half_extent: -(size * 0.5).max(0.0),
            radius,
            color,
            intensity,
        }
    }
}

/// Result of one presentation attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameOutcome {
    /// The frame was encoded, submitted and presented.
    Presented,
    /// The surface was not ready (occluded, timed out or lost); the caller
    /// should simply try again next iteration.
    Skipped,
}

/// One offscreen frame read back from the GPU, ready to be written as PNG.
///
/// `rgba` is tightly packed (no row padding), 8 bits per channel, display
/// colorspace of the swapchain format. `depth_distance` is the centre-pixel
/// distance along the view ray, used by `assert_hit_matches_render`.
#[derive(Clone, Debug, Default)]
pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Internal (pre-upscale) render size, needed to map tap pixels to rays.
    pub internal_size: (u32, u32),
    pub depth_distance: Option<f32>,
    /// Depth samples on a sparse cross around the centre. A centre-only sample
    /// cannot see a lateral shift: when the whole image is displaced one brick
    /// sideways the centre depth still matches, while the taps next to it do
    /// not (`SHATTERBOX_DEBUG_HIT` proved that shift was real).
    pub depth_taps: Vec<DepthTap>,
}

/// One depth sample of the internal G-buffer, in metres along the view ray.
/// `distance: None` means the ray escaped the world (shader stores `-1`).
#[derive(Clone, Copy, Debug)]
pub struct DepthTap {
    pub pixel: [u32; 2],
    pub distance: Option<f32>,
}

/// Snapshot of the renderer's previous frame.
#[derive(Clone, Copy, Debug)]
pub struct RenderStats {
    pub gbuffer_ms: f32,
    pub lighting_ms: f32,
    pub temporal_ms: f32,
    /// Half-resolution sun shadows.
    pub shadow_ms: f32,
    /// Half-resolution point/capsule/rect shadows.
    pub light_shadow_ms: f32,
    /// Half-resolution SSAO + SSGI + VCT + SSR.
    pub gi_ms: f32,
    /// Half-resolution volumetric fog.
    pub fog_ms: f32,
    /// Bloom prefilter + both blur passes.
    pub bloom_ms: f32,
    /// FXAA at internal resolution (zero when the effect is off).
    pub fxaa_ms: f32,
    /// Smoke/dust billboard pass (zero when nothing is burning).
    pub particles_ms: f32,
    /// Per-tile debris candidate mask build (zero when no piece is alive).
    pub debris_cull_ms: f32,
    /// DDGI albedo injection (G-buffer -> 1 m colour volume).
    pub ddgi_inject_ms: f32,
    /// DDGI probe ray trace.
    pub ddgi_update_ms: f32,
    /// DDGI cosine convolution + hysteresis.
    pub ddgi_blend_ms: f32,
    pub total_ms: f32,
    pub scale: f32,
    pub internal_size: (u32, u32),
    pub output_size: (u32, u32),
    pub target_fps: f32,
    pub timestamps_supported: bool,
    pub timestamps_valid: bool,
    pub hot_reload_enabled: bool,
    pub governor_enabled: bool,
    pub frame_index: u64,
    pub light_count: u32,
    pub debris_count: u32,
    /// Renderer cap on traced debris pieces (the simulation budget is larger).
    pub debris_limit: u32,
    pub time_of_day: f32,
    pub sun_temperature: f32,
    /// Active sun shadow trace mode (see [`ShadowMode`]).
    pub shadow_mode: &'static str,
    pub upscaler: &'static str,
}

impl Default for RenderStats {
    fn default() -> Self {
        Self {
            gbuffer_ms: 0.0,
            lighting_ms: 0.0,
            temporal_ms: 0.0,
            shadow_ms: 0.0,
            light_shadow_ms: 0.0,
            gi_ms: 0.0,
            fog_ms: 0.0,
            bloom_ms: 0.0,
            fxaa_ms: 0.0,
            particles_ms: 0.0,
            debris_cull_ms: 0.0,
            ddgi_inject_ms: 0.0,
            ddgi_update_ms: 0.0,
            ddgi_blend_ms: 0.0,
            total_ms: 0.0,
            scale: 1.0,
            internal_size: (1, 1),
            output_size: (1, 1),
            target_fps: 80.0,
            timestamps_supported: false,
            timestamps_valid: false,
            hot_reload_enabled: false,
            governor_enabled: true,
            frame_index: 0,
            light_count: 0,
            debris_count: 0,
            debris_limit: MAX_RENDER_DEBRIS as u32,
            time_of_day: 10.5,
            sun_temperature: 5800.0,
            shadow_mode: "full-res",
            upscaler: "FSR1",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    inv_view_proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    prev_camera_pos: [f32; 4],
    sun_dir: [f32; 4],
    /// rgb = sun/moon radiance, w = normalized daylight factor.
    sun_color: [f32; 4],
    world_min: [f32; 4],
    world_max: [f32; 4],
    params: [f32; 4],
    params2: [f32; 4],
    /// x = bitmask of enabled effects (1 shadows, 2 ssao, 4 ssgi, 8 vct,
    /// 16 taa, 32 fog, 64 ssr), y = fog density.
    params3: [f32; 4],
    internal_size: [f32; 4],
    light_pos_radius: [[f32; 4]; MAX_LIGHTS],
    light_color: [[f32; 4]; MAX_LIGHTS],
    /// xyz = emitter axis, w = half extent (negative = rectangle half size).
    /// Kept immediately after `light_color` so every WGSL `Frame` that models
    /// area lights sees the same offset as this struct (see the layout test).
    light_axis: [[f32; 4]; MAX_LIGHTS],
    /// World-space AABB around every active debris piece. The G-buffer rejects
    /// the whole debris set with a single slab test when the camera ray misses
    /// it, instead of running 64 inverse-matrix transforms.
    debris_min: [f32; 4],
    debris_max: [f32; 4],
    /// xyz = world point of the CPU aim raycast, w = marker radius in metres.
    /// `w <= 0` disables the debug marker (see `SHATTERBOX_DEBUG_HIT`).
    debug_hit: [f32; 4],
    /// x = 1 enables the debris reactive mask in the temporal upscaler,
    /// yzw reserved.
    params4: [f32; 4],
}

/// GPU auto-exposure accumulation state. Mirrors `ExposureState` in
/// `bloom.wgsl`. The first two words are atomics; the CPU only overwrites the
/// settings tail (offset 12), so `adapted` survives between frames.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExposureStateGpu {
    sum: u32,
    count: u32,
    adapted: f32,
    manual: f32,
    auto: f32,
    speed: f32,
    bias: f32,
    dt: f32,
    target: f32,
    _pad: [f32; 3],
}

/// Read-only copy consumed by `post.wgsl` (fragment bindings cannot contain
/// atomics). Written by the `cs_exposure` resolve pass every frame.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExposureValueGpu {
    adapted: f32,
    auto: f32,
    _pad: [f32; 2],
}

impl FrameUniform {
    #[allow(clippy::too_many_arguments)]
    fn new(
        camera: &FreeCamera,
        aspect: f32,
        prev_view_proj: Mat4,
        prev_camera_pos: Vec3,
        time: f32,
        frame_index: u64,
        jitter: [f32; 2],
        reset_history: bool,
        internal_size: (u32, u32),
        exposure: f32,
        debris_count: u32,
        lights: &[PointLight],
        area_lights: &[AreaLight],
        debris_bounds: Option<(Vec3, Vec3)>,
        debug_hit: Option<Vec3>,
        time_of_day: f32,
        shadow_distance: f32,
        sun_temperature: f32,
        sun_intensity: f32,
        effects: RenderEffects,
        fog_density: f32,
        fog_height: f32,
        shadow_mode: ShadowMode,
        upscale_clamp: bool,
    ) -> Self {
        let view_projection = camera.view_projection(aspect);
        // Sun path: sunrise at 06:00 on +X, noon overhead, sunset at 18:00 on
        // -X. At night the same vector is flipped to a dim blue moon so the
        // arena still casts soft shadows and the point lights have a base.
        let hours = time_of_day.rem_euclid(24.0);
        let theta = (hours - 6.0) / 12.0 * std::f32::consts::PI;
        let elevation = theta.sin();
        let mut sun = Vec3::new(theta.cos(), elevation * 0.8, 0.55).normalize();
        let daylight = smoothstep(-0.06, 0.24, elevation);
        let low_sun = smoothstep(0.0, 0.38, elevation);
        // The slider sets the photosphere temperature; the atmosphere reddens
        // it near the horizon, which is why a 5800 K sun looks amber at 07:00
        // and white at noon without a separate art-directed tint.
        let apparent_temperature = sun_temperature * (0.74 + 0.26 * low_sun);
        let mut radiance = blackbody_rgb(apparent_temperature) * 4.4 * daylight;
        if daylight < 0.02 {
            sun = -sun;
            radiance = Vec3::new(0.42, 0.56, 1.0) * 0.16;
        }
        // The slider scales the physical radiance, still in linear HDR, so 0 is
        // a black sun, 1 the calibrated look and 5 a deliberately blown-out one.
        radiance *= sun_intensity.max(0.0);
        // The moon still needs a positive y to light the arena from above.
        if sun.y < 0.05 {
            sun.y = 0.05;
            sun = sun.normalize();
        }
        let mut light_pos_radius = [[0.0; 4]; MAX_LIGHTS];
        let mut light_color = [[0.0; 4]; MAX_LIGHTS];
        let mut light_axis = [[0.0; 4]; MAX_LIGHTS];
        let mut light_count = 0usize;
        // Point lights claim the first slots so a scene that already fits keeps
        // the exact same uniform it had before area lights existed.
        for light in lights.iter().take(MAX_LIGHTS) {
            light_pos_radius[light_count] = [
                light.position[0],
                light.position[1],
                light.position[2],
                light.radius,
            ];
            light_color[light_count] = [
                light.color[0] * light.intensity,
                light.color[1] * light.intensity,
                light.color[2] * light.intensity,
                0.0,
            ];
            light_count += 1;
        }
        for light in area_lights.iter().take(MAX_LIGHTS - light_count) {
            light_pos_radius[light_count] = [
                light.position[0],
                light.position[1],
                light.position[2],
                light.radius,
            ];
            light_color[light_count] = [
                light.color[0] * light.intensity,
                light.color[1] * light.intensity,
                light.color[2] * light.intensity,
                0.0,
            ];
            light_axis[light_count] = [
                light.axis[0],
                light.axis[1],
                light.axis[2],
                light.half_extent,
            ];
            light_count += 1;
        }
        Self {
            inv_view_proj: view_projection.inverse().to_cols_array_2d(),
            view_proj: view_projection.to_cols_array_2d(),
            prev_view_proj: prev_view_proj.to_cols_array_2d(),
            camera_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
            prev_camera_pos: [prev_camera_pos.x, prev_camera_pos.y, prev_camera_pos.z, 1.0],
            sun_dir: [sun.x, sun.y, sun.z, shadow_distance.max(2.0)],
            sun_color: [radiance.x, radiance.y, radiance.z, daylight],
            world_min: [WORLD_MIN.x, WORLD_MIN.y, WORLD_MIN.z, 0.0],
            world_max: [WORLD_MAX.x, WORLD_MAX.y, WORLD_MAX.z, 0.0],
            params: [time, frame_index as f32, jitter[0], jitter[1]],
            params2: [
                debris_count.min(MAX_RENDER_DEBRIS as u32) as f32,
                if reset_history { 1.0 } else { 0.0 },
                light_count as f32,
                exposure,
            ],
            params3: [
                effects.bitmask() as f32,
                fog_density.max(0.0),
                shadow_mode.flag(),
                if upscale_clamp { 1.0 } else { 0.0 },
            ],
            internal_size: [
                internal_size.0 as f32,
                internal_size.1 as f32,
                1.0 / internal_size.0.max(1) as f32,
                1.0 / internal_size.1.max(1) as f32,
            ],
            light_pos_radius,
            light_color,
            light_axis,
            debris_min: debris_bounds.map_or([0.0; 4], |(min, _)| [min.x, min.y, min.z, 0.0]),
            debris_max: debris_bounds.map_or([0.0; 4], |(_, max)| [max.x, max.y, max.z, 0.0]),
            debug_hit: debug_hit.map_or([0.0; 4], |point| {
                [point.x, point.y, point.z, DEBUG_HIT_RADIUS_M]
            }),
            params4: [0.0, fog_height.max(0.5), 0.0, 0.0],
        }
    }
}

/// One explosion smoke/dust puff, drawn as a camera-facing soft disc.
///
/// Matches `Instance` in `particles.wgsl`: position + radius + premultiplied
/// albedo/opacity. The simulation lives in `shatterbox-game`; the renderer only
/// uploads the current set.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub radius: f32,
    /// rgb = albedo tint, a = opacity (0..1).
    pub color: [f32; 4],
    /// x = optical density, y = noise seed, z = puff kind (0 smoke, 1 dust,
    /// 2 trail, 3 grain), w = remaining heat 0..1. The shader uses these for
    /// the Beer-Lambert self-shadow, torn edges and fire-lit early colour.
    pub params: [f32; 4],
}

/// Matches `DebrisInstance` in `gbuffer.wgsl`. Keep the 16-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuDebrisInstance {
    model: [[f32; 4]; 4],
    inverse_model: [[f32; 4]; 4],
    aabb_min: [f32; 4],
    aabb_max: [f32; 4],
    data: [u32; 4],
}

struct WorldMirror {
    brick_slots: Vec<u32>,
    free_ids: Vec<u32>,
    next_id: u32,
    group_dims: [u32; 3],
    region_dims: [u32; 3],
    group_occupancy: Vec<u32>,
    region_occupancy: Vec<u32>,
}

impl WorldMirror {
    fn new(data: &shatterbox_voxel::GpuWorldData) -> Self {
        let mut mirror = Self {
            brick_slots: data.brick_indices.clone(),
            free_ids: Vec::new(),
            next_id: data.brick_count + 1,
            group_dims: data.group_dims,
            region_dims: data.region_dims,
            group_occupancy: vec![0; data.group_occupancy.len()],
            region_occupancy: vec![0; data.region_occupancy.len()],
        };
        mirror.rebuild_occupancy();
        mirror
    }

    fn allocate(&mut self) -> Option<u32> {
        if let Some(id) = self.free_ids.pop() {
            return Some(id);
        }
        if self.next_id as usize >= MAX_GPU_BRICKS {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        Some(id)
    }

    fn rebuild_occupancy(&mut self) {
        self.group_occupancy.fill(0);
        self.region_occupancy.fill(0);
        for (slot, id) in self.brick_slots.iter().enumerate() {
            if *id == 0 {
                continue;
            }
            let (gx, gy, gz) = slot_group(slot);
            let group = gx + self.group_dims[0] as usize * (gy + self.group_dims[1] as usize * gz);
            self.group_occupancy[group] = 1;
            let (rx, ry, rz) = slot_region(slot);
            let region =
                rx + self.region_dims[0] as usize * (ry + self.region_dims[1] as usize * rz);
            self.region_occupancy[region] = 1;
        }
    }

    /// Recomputes occupancy only for groups/regions touched by edited bricks
    /// and returns the texels whose value actually changed. A shot used to
    /// rebuild and upload both full 3D textures; that was the largest CPU-side
    /// cost per edit.
    fn update_occupancy(&mut self, touched_slots: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut groups = std::collections::HashSet::new();
        let mut regions = std::collections::HashSet::new();
        for &slot in touched_slots {
            let (gx, gy, gz) = slot_group(slot);
            groups
                .insert(gx + self.group_dims[0] as usize * (gy + self.group_dims[1] as usize * gz));
            let (rx, ry, rz) = slot_region(slot);
            regions.insert(
                rx + self.region_dims[0] as usize * (ry + self.region_dims[1] as usize * rz),
            );
        }

        let mut changed_groups = Vec::new();
        for group in groups {
            let gx = group % self.group_dims[0] as usize;
            let gy = (group / self.group_dims[0] as usize) % self.group_dims[1] as usize;
            let gz = group / (self.group_dims[0] * self.group_dims[1]) as usize;
            let value = (gx * 4..(gx * 4 + 4).min(BRICKS_X)).any(|x| {
                (gy * 4..(gy * 4 + 4).min(BRICKS_Y)).any(|y| {
                    (gz * 4..(gz * 4 + 4).min(BRICKS_Z))
                        .any(|z| self.brick_slots[x + BRICKS_X * (y + BRICKS_Y * z)] != 0)
                })
            }) as u32;
            if self.group_occupancy[group] != value {
                self.group_occupancy[group] = value;
                changed_groups.push(group);
            }
        }

        let mut changed_regions = Vec::new();
        for region in regions {
            let rx = region % self.region_dims[0] as usize;
            let ry = (region / self.region_dims[0] as usize) % self.region_dims[1] as usize;
            let rz = region / (self.region_dims[0] * self.region_dims[1]) as usize;
            let value = (rx * 16..(rx * 16 + 16).min(BRICKS_X)).any(|x| {
                (ry * 16..(ry * 16 + 16).min(BRICKS_Y)).any(|y| {
                    (rz * 16..(rz * 16 + 16).min(BRICKS_Z))
                        .any(|z| self.brick_slots[x + BRICKS_X * (y + BRICKS_Y * z)] != 0)
                })
            }) as u32;
            if self.region_occupancy[region] != value {
                self.region_occupancy[region] = value;
                changed_regions.push(region);
            }
        }
        (changed_groups, changed_regions)
    }
}

struct Pipelines {
    gbuffer: wgpu::ComputePipeline,
    shadow: wgpu::ComputePipeline,
    /// Half-resolution local point/capsule/rect light visibility mask.
    light_shadow: wgpu::ComputePipeline,
    lighting: wgpu::ComputePipeline,
    gi: wgpu::ComputePipeline,
    probe: wgpu::ComputePipeline,
    fog: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    /// FSR 1 style upscale (EASU/RCAS port), shares the temporal bind layout.
    fsr: wgpu::ComputePipeline,
    /// FXAA 3.11 at internal resolution, before the upscaler.
    fxaa: wgpu::ComputePipeline,
    bloom_prefilter: wgpu::ComputePipeline,
    bloom_blur_h: wgpu::ComputePipeline,
    bloom_blur_v: wgpu::ComputePipeline,
    /// Resolves the log-luminance accumulated by the bloom prefilter into the
    /// adapted exposure. One thread, once per frame.
    exposure: wgpu::ComputePipeline,
    /// Explosion smoke/dust billboards, drawn into the HDR target.
    particles: wgpu::RenderPipeline,
    /// Depth-aware fullscreen upsample of the half-res smoke buffer.
    particles_composite: wgpu::RenderPipeline,
    /// Builds the per-tile debris candidate masks before the G-buffer.
    debris_cull: wgpu::ComputePipeline,
    /// Splats G-buffer albedo into the 1 m DDGI colour volume.
    ddgi_inject: wgpu::ComputePipeline,
    /// Traces one ray per octahedral texel of the refreshed probe slice.
    ddgi_trace: wgpu::ComputePipeline,
    /// Convolves the ray cloud into irradiance and applies the hysteresis.
    ddgi: wgpu::ComputePipeline,
    post: wgpu::RenderPipeline,
}

struct Layouts {
    gbuffer: wgpu::BindGroupLayout,
    shadow: wgpu::BindGroupLayout,
    light_shadow: wgpu::BindGroupLayout,
    lighting: wgpu::BindGroupLayout,
    gi: wgpu::BindGroupLayout,
    probe: wgpu::BindGroupLayout,
    fog: wgpu::BindGroupLayout,
    temporal: wgpu::BindGroupLayout,
    bloom: wgpu::BindGroupLayout,
    post: wgpu::BindGroupLayout,
    fxaa: wgpu::BindGroupLayout,
    particles: wgpu::BindGroupLayout,
    /// Smoke composite: frame + depth + the half-res smoke buffer and sampler.
    particles_composite: wgpu::BindGroupLayout,
    debris_cull: wgpu::BindGroupLayout,
    /// DDGI albedo injection: depth/normal/albedo/material -> 1 m colour volume.
    ddgi_inject: wgpu::BindGroupLayout,
    /// DDGI trace: frame + bricks + colour volume + previous atlas -> ray cloud.
    ddgi_trace: wgpu::BindGroupLayout,
    /// DDGI blend: params + ray cloud + ping-pong irradiance/moments.
    ddgi: wgpu::BindGroupLayout,
    gbuffer_layout: wgpu::PipelineLayout,
    shadow_layout: wgpu::PipelineLayout,
    light_shadow_layout: wgpu::PipelineLayout,
    lighting_layout: wgpu::PipelineLayout,
    gi_layout: wgpu::PipelineLayout,
    probe_layout: wgpu::PipelineLayout,
    fog_layout: wgpu::PipelineLayout,
    temporal_layout: wgpu::PipelineLayout,
    bloom_layout: wgpu::PipelineLayout,
    post_layout: wgpu::PipelineLayout,
    fxaa_layout: wgpu::PipelineLayout,
    particles_layout: wgpu::PipelineLayout,
    particles_composite_layout: wgpu::PipelineLayout,
    debris_cull_layout: wgpu::PipelineLayout,
    ddgi_inject_layout: wgpu::PipelineLayout,
    ddgi_trace_layout: wgpu::PipelineLayout,
    ddgi_layout: wgpu::PipelineLayout,
}

/// Render targets. The textures are only kept alive here; every pass binds
/// their views, so most fields are deliberately never read.
#[allow(dead_code)]
struct Targets {
    albedo: wgpu::Texture,
    albedo_view: wgpu::TextureView,
    normal: wgpu::Texture,
    normal_view: wgpu::TextureView,
    material: wgpu::Texture,
    material_view: wgpu::TextureView,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    hdr: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    /// FXAA output: the same internal-resolution HDR image after the edge blur.
    hdr_aa: wgpu::Texture,
    hdr_aa_view: wgpu::TextureView,
    bloom_a: wgpu::Texture,
    bloom_a_view: wgpu::TextureView,
    bloom_b: wgpu::Texture,
    bloom_b_view: wgpu::TextureView,
    /// Half-resolution SSAO + SSGI + VCT result: rgb = indirect, a = AO.
    gi: wgpu::Texture,
    gi_view: wgpu::TextureView,
    /// Half-resolution volumetric fog: rgb = inscatter, a = transmittance.
    fog: wgpu::Texture,
    fog_view: wgpu::TextureView,
    /// Half-resolution voxel sun visibility (rgb = visibility, a = 1).
    shadow: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    /// Half-resolution local-light visibility bitset (bit i = light i visible).
    light_shadow: wgpu::Texture,
    light_shadow_view: wgpu::TextureView,
    /// Half-resolution screen-space reflection for glossy surfaces.
    reflection: wgpu::Texture,
    reflection_view: wgpu::TextureView,
    /// Dynamic smoke density (R8, 4 m cells), sampled by `fog.wgsl`.
    smoke: wgpu::Texture,
    smoke_view: wgpu::TextureView,
    /// Low-resolution voxel irradiance volume for the cone trace.
    probe: wgpu::Texture,
    probe_view: wgpu::TextureView,
    /// 1 m albedo/emissive volume injected from the G-buffer for the DDGI rays.
    ddgi_albedo: wgpu::Texture,
    ddgi_albedo_view: wgpu::TextureView,
    /// Raw ray radiance of the refreshed probe slice (`ddgi_update.wgsl`).
    ddgi_trace: wgpu::Texture,
    ddgi_trace_view: wgpu::TextureView,
    /// DDGI octahedral atlases (see `ddgi_blend.wgsl`).
    /// Index 0/1 are the ping-pong targets: the blend reads the other one.
    ddgi_irradiance: [wgpu::Texture; 2],
    ddgi_irradiance_views: [wgpu::TextureView; 2],
    ddgi_moments: [wgpu::Texture; 2],
    ddgi_moments_views: [wgpu::TextureView; 2],
    history_color: [wgpu::Texture; 2],
    history_color_views: [wgpu::TextureView; 2],
    history_depth: [wgpu::Texture; 2],
    history_depth_views: [wgpu::TextureView; 2],
    internal_size: (u32, u32),
    output_size: (u32, u32),
    bloom_size: (u32, u32),
    gi_size: (u32, u32),
    probe_size: (u32, u32, u32),
    /// Size of the sun shadow target: `gi_size` in half-res modes, `internal_size`
    /// in the full-res mode. The shadow dispatch reads this, otherwise the
    /// full-res texture is only traced in the top-left quarter and the rest of
    /// the screen reads uninitialised visibility (a flat ambient-only frame).
    shadow_size: (u32, u32),
    /// The sun shadow texture was allocated at full internal resolution.
    shadow_full: bool,
}

struct BindGroups {
    gbuffer: wgpu::BindGroup,
    shadow: wgpu::BindGroup,
    light_shadow: wgpu::BindGroup,
    lighting: wgpu::BindGroup,
    /// `[history][ddgi]`: `history` selects which HDR target SSGI reads, `ddgi`
    /// selects the atlas written last frame (the blend ping-pongs slices, so
    /// reading a fixed atlas left every other z slice black).
    gi: [[wgpu::BindGroup; 2]; 2],
    probe: wgpu::BindGroup,
    fog: wgpu::BindGroup,
    bloom_prefilter: wgpu::BindGroup,
    bloom_blur_h: wgpu::BindGroup,
    bloom_blur_v: wgpu::BindGroup,
    temporal: [wgpu::BindGroup; 2],
    /// Same as `temporal`, but reading the FXAA result instead of the raw HDR.
    temporal_aa: [wgpu::BindGroup; 2],
    fxaa: wgpu::BindGroup,
    particles: wgpu::BindGroup,
    /// Composite-only bind group (frame + depth + half-res smoke + sampler).
    particles_composite: wgpu::BindGroup,
    /// Per-tile debris candidate masks (see `debris_cull.wgsl`).
    debris_cull: wgpu::BindGroup,
    /// DDGI albedo injection (G-buffer -> 1 m colour volume).
    ddgi_inject: wgpu::BindGroup,
    /// DDGI trace bind groups: `[i]` reads atlas `1 - i` and writes the cloud.
    ddgi_trace: [wgpu::BindGroup; 2],
    /// Ping-pong DDGI blend bind groups: `[i]` reads atlas `1 - i`.
    ddgi: [wgpu::BindGroup; 2],
    post: [wgpu::BindGroup; 2],
}

/// One cached allocation set, keyed by the internal and output sizes.
struct TargetCache {
    internal: (u32, u32),
    output: (u32, u32),
    targets: Targets,
    bind_groups: BindGroups,
}

struct ShaderWatcher {
    directory: Option<PathBuf>,
    last_check: Instant,
    current: [String; SHADER_SLOTS],
    enabled: bool,
}

impl ShaderWatcher {
    fn poll(&mut self) -> Option<[String; SHADER_SLOTS]> {
        if !self.enabled || self.last_check.elapsed() < Duration::from_millis(500) {
            return None;
        }
        self.last_check = Instant::now();

        let directory = self.directory.as_ref()?;
        let mut changed = false;
        let mut next = self.current.clone();
        for (slot, name) in SHADER_NAMES.iter().enumerate() {
            let Ok(source) = fs::read_to_string(directory.join(name)) else {
                continue;
            };
            if source != next[slot] {
                next[slot] = source;
                changed = true;
            }
        }
        if !changed {
            return None;
        }
        self.current = next.clone();
        Some(next)
    }
}

/// Owns the egui integration, voxel world GPU resources and frame presentation.
pub struct Renderer {
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: EguiRenderer,
    clear_color: wgpu::Color,

    format: wgpu::TextureFormat,
    uniform_buffer: wgpu::Buffer,
    /// Atomic log-luminance accumulator + adaptation settings (see
    /// `ExposureStateGpu` and `cs_exposure` in `bloom.wgsl`).
    exposure_state: wgpu::Buffer,
    /// Plain adapted exposure read by `post.wgsl`.
    exposure_value: wgpu::Buffer,
    _world_texture: wgpu::Texture,
    world_view: wgpu::TextureView,
    _group_texture: wgpu::Texture,
    group_view: wgpu::TextureView,
    _region_texture: wgpu::Texture,
    region_view: wgpu::TextureView,
    brick_buffer: wgpu::Buffer,
    mask_buffer: wgpu::Buffer,
    palette_buffer: wgpu::Buffer,
    debris_buffer: wgpu::Buffer,
    debris_voxel_buffer: wgpu::Buffer,
    /// 16 bytes per piece: world sphere centre + radius, the cheap reject in
    /// `cast_debris` (effect bit 512).
    debris_sphere_buffer: wgpu::Buffer,
    /// 16 bytes per piece: conservative screen rectangle `(x0, y0, x1, y1)` in
    /// internal pixels, used by `debris_cull.wgsl` to bucket pieces per tile.
    debris_rect_buffer: wgpu::Buffer,
    /// `16 * tiles` u32 words: a 512-bit piece mask per 8x8 pixel tile.
    debris_tile_buffer: wgpu::Buffer,
    debris_count: u32,
    debris_voxel_len: u32,
    /// World-space AABB of the uploaded debris pieces, if any are alive.
    debris_bounds: Option<(Vec3, Vec3)>,
    /// Reused offscreen capture target (texture, view, size). See
    /// `render_offscreen`.
    offscreen_target: Option<(wgpu::Texture, wgpu::TextureView, (u32, u32))>,
    /// Skips debris in the G-buffer while a depth-probe frame is captured.
    debug_hide_debris: bool,
    /// CPU aim point painted by the debug marker (`SHATTERBOX_DEBUG_HIT`).
    debug_hit: Option<Vec3>,
    debris_instances_scratch: Vec<GpuDebrisInstance>,
    debris_voxels_scratch: Vec<u32>,
    debris_spheres_scratch: Vec<[f32; 4]>,
    /// Recycled instance buffer for `upload_debris` (filled, then filtered into
    /// `debris_instances_scratch`).
    debris_pending_scratch: Vec<GpuDebrisInstance>,
    /// Per-piece screen rectangles for the tile pass, recomputed each frame.
    debris_rects_scratch: Vec<[f32; 4]>,
    world_mirror: WorldMirror,
    sampler: wgpu::Sampler,

    layouts: Layouts,
    pipelines: Pipelines,
    targets: Targets,
    bind_groups: BindGroups,
    /// One recently used (internal, output) target set. The resolution governor
    /// flips between two neighbouring scales; keeping the previous allocation
    /// alive turns that into a pointer swap instead of a few hundred megabytes
    /// of texture creation in the middle of a frame.
    target_cache: Option<TargetCache>,
    history_parity: usize,
    history_reset: bool,

    query_set: Option<wgpu::QuerySet>,
    resolve_buffer: Option<wgpu::Buffer>,
    readback_buffer: Option<wgpu::Buffer>,
    timestamp_period: f32,
    timestamp_ready: Arc<AtomicBool>,
    timestamp_pending: bool,
    gbuffer_ms: f32,
    lighting_ms: f32,
    temporal_ms: f32,
    shadow_ms: f32,
    light_shadow_ms: f32,
    gi_ms: f32,
    fog_ms: f32,
    bloom_ms: f32,
    fxaa_ms: f32,
    particles_ms: f32,
    /// Tile-mask build pass for the debris culling (zero when no piece is alive).
    debris_cull_ms: f32,
    ddgi_inject_ms: f32,
    ddgi_update_ms: f32,
    ddgi_blend_ms: f32,
    total_ms: f32,
    timestamps_valid: bool,

    shader_watcher: ShaderWatcher,
    scale: f32,
    governor_enabled: bool,
    governor_elapsed: f32,
    governor_frame_ms: f32,
    /// Consecutive plausible frames above ~1.35x budget. The average can sit
    /// just under target while every fourth frame spikes; the governor steps
    /// down on this counter as well so 1% low does not collapse during
    /// destruction bursts.
    governor_spikes: u32,
    target_fps: f32,
    /// Upscale ceiling. At an 80 fps target the governor stops at 0.85
    /// (900p internal) instead of climbing to native 1440p, which would trade
    /// the 1% low for pixels the user explicitly does not need.
    max_scale: f32,
    exposure: f32,
    /// Post-stack diagnostic bitmask (`SHATTERBOX_POST_DEBUG`, scenario `Post`).
    /// Bits disable one stage at a time so the lighting_set A/B can name the
    /// stage that produces the crushed/neon image instead of guessing.
    post_debug: u32,
    /// Look variant selected by `SHATTERBOX_LOOK=a|b|c` (0 = a).
    look: u8,
    /// Auto exposure measured from the linear HDR frame. `exposure` remains the
    /// manual value and is used when this is off.
    auto_exposure: bool,
    exposure_speed: f32,
    exposure_bias: f32,
    /// Frame delta uploaded to the adaptation curve; kept here because
    /// `write_frame_uniform` does not receive `dt`.
    exposure_dt: f32,
    time: f32,
    time_of_day: f32,
    sun_temperature: f32,
    /// Multiplier on the sun radiance, in linear HDR. 1.0 is the physical value
    /// baked into the sun model; 0 kills the sun, 5 is a hard high-key look.
    /// The F1 slider and `config/sky.ron` drive it.
    sun_intensity: f32,
    fog_density: f32,
    /// Metres of height falloff: the fog density drops by `e` every
    /// `fog_height` metres above 1.5 m. 5 keeps the haze on the ground.
    fog_height: f32,
    shadow_distance: f32,
    effects: RenderEffects,
    /// Sun shadow trace resolution / upsample filter (A/B switch for the
    /// shadow-grain work; `SHATTERBOX_SHADOW_MODE` or the F1 overlay).
    shadow_mode: ShadowMode,
    /// Clamp the FSR/Catmull-Rom upscale to the fetched tap range. Stops the
    /// bright fringe / dark seam that the kernel invents on silhouettes
    /// (`SHATTERBOX_UPSCALE_CLAMP=0` restores the old, overshooting filter).
    upscale_clamp: bool,
    /// Debris reactive mask: the G-buffer flags pixels covered by a rigid body
    /// and the temporal upscaler drops their history, so a falling piece does
    /// not drag a ghost trail (`SHATTERBOX_DEBRIS_REACTIVE=0` restores the old
    /// behaviour).
    debris_reactive: bool,
    /// Sub-pixel Halton jitter feeding TAAU (`SHATTERBOX_TAA_JITTER=0` for A/B).
    taa_jitter: bool,
    /// Spatial upscaler used instead of the TAAU path.
    upscaler: Upscaler,
    /// The irradiance volume must be rebuilt when the world, sun or light rig
    /// changes; it does not depend on the camera, so it is not a per-frame cost.
    probe_dirty: bool,
    /// DDGI update cursor: one z-slice of the probe grid per frame while the
    /// effect bit is set, so a full cycle of 32 slices costs 32 frames.
    ddgi_slice: u32,
    /// The ping-pong atlases were just allocated or the effect was toggled on;
    /// the first cycle ignores the hysteresis instead of blending from zero.
    ddgi_reset: bool,
    /// User-facing ray budget preset (`SHATTERBOX_GI`, F1 selector).
    ddgi_quality: DdgiQuality,
    /// Frames left in the edit-priority window. Probes near a fresh crater
    /// refresh every frame with a lower hysteresis so the room re-lights in
    /// well under a second instead of waiting for the 64-frame slice cycle.
    ddgi_priority_frames: u32,
    ddgi_priority_center: Vec3,
    ddgi_priority_radius: f32,
    /// `{ slice, hysteresis, spacing, reset, tile, samples, slices, priority,
    /// priority_center, priority_radius, priority_hysteresis }` for the DDGI
    /// passes. 64 bytes.
    ddgi_state: wgpu::Buffer,
    frame_index: u64,
    prev_view_proj: Mat4,
    prev_camera_pos: Vec3,
    lights: Vec<PointLight>,
    area_lights: Vec<AreaLight>,
    /// Smoke/dust billboards for the current frame. Owned by the game and
    /// pushed through `set_particles`; the renderer only uploads them.
    particles: Vec<Particle>,
    particle_buffer: wgpu::Buffer,
    particle_count: u32,
    stats: RenderStats,
}

impl Renderer {
    /// Creates the Phase 2 renderer and uploads the sparse brick world.
    pub fn new(gpu: &GpuContext, window: &Window, world: &VoxelWorld) -> Result<Self> {
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());

        let max_texture_side = gpu.device.limits().max_texture_dimension_2d as usize;
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(max_texture_side),
        );
        let egui_renderer =
            EguiRenderer::new(&gpu.device, gpu.format(), RendererOptions::default());

        let world_data = world.to_gpu();
        let dims = world_data.brick_dims;
        let world_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shatterbox brick index"),
            size: wgpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &world_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&world_data.brick_indices),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims[0] * 4),
                rows_per_image: Some(dims[1]),
            },
            wgpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
        );
        let world_view = world_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let upload_occupancy = |label: &str, dims: [u32; 3], data: &[u32]| {
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: dims[0],
                    height: dims[1],
                    depth_or_array_layers: dims[2],
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dims[0] * 4),
                    rows_per_image: Some(dims[1]),
                },
                wgpu::Extent3d {
                    width: dims[0],
                    height: dims[1],
                    depth_or_array_layers: dims[2],
                },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (group_texture, group_view) = upload_occupancy(
            "Shatterbox coarse group occupancy",
            world_data.group_dims,
            &world_data.group_occupancy,
        );
        let (region_texture, region_view) = upload_occupancy(
            "Shatterbox region occupancy",
            world_data.region_dims,
            &world_data.region_occupancy,
        );

        let brick_bytes = (MAX_GPU_BRICKS * GPU_WORDS_PER_BRICK * size_of::<u32>()) as u64;
        let brick_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox brick storage"),
            size: brick_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&brick_buffer, 0, bytemuck::cast_slice(&world_data.bricks));

        let mask_bytes = (MAX_GPU_BRICKS * MASK_WORDS_PER_BRICK * size_of::<u32>()) as u64;
        let mask_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox brick masks"),
            size: mask_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &mask_buffer,
            0,
            bytemuck::cast_slice(&world_data.brick_masks),
        );

        let debris_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox debris instances"),
            size: (MAX_RENDER_DEBRIS * size_of::<GpuDebrisInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debris_voxel_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox debris voxels"),
            size: (MAX_DEBRIS_VOXELS * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let debris_sphere_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox debris spheres"),
            size: (MAX_RENDER_DEBRIS * size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let debris_rect_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox debris screen rects"),
            size: (MAX_RENDER_DEBRIS * size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debris_tile_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox debris tile masks"),
            size: (DEBRIS_TILES_MAX * 16 * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let palette_data = world.palette().gpu_data();
        let palette_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox palette"),
            size: (palette_data.len() * size_of::<[f32; 8]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&palette_buffer, 0, bytemuck::cast_slice(&palette_data));

        let uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox frame uniform"),
            size: size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ddgi_state = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox ddgi state"),
            // slice/hysteresis/spacing/reset + tile/samples/slices/priority +
            // priority centre/radius/hysteresis (see `DdgiParams` in WGSL).
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exposure_state = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox exposure state"),
            size: size_of::<ExposureStateGpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exposure_value = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox exposure value"),
            size: size_of::<ExposureValueGpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layouts = create_layouts(&gpu.device);
        let shader_watcher = ShaderWatcher::new();
        let pipelines = match build_pipelines(
            &gpu.device,
            &layouts,
            gpu.format(),
            &shader_watcher.current,
        ) {
            Ok(pipelines) => pipelines,
            Err(error) => {
                tracing::warn!(%error, "shader files failed validation; using embedded sources");
                build_pipelines(
                    &gpu.device,
                    &layouts,
                    gpu.format(),
                    &std::array::from_fn(|slot| EMBEDDED_SHADERS[slot].to_owned()),
                )
                .map_err(|error| anyhow!("failed to create Phase 2 pipelines: {error}"))?
            }
        };

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shatterbox temporal sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (output_width, output_height) = gpu.size();
        // Quality presets trade shadow reach and starting resolution. The
        // governor still owns the final scale, so `performance` is simply a
        // better starting point for 1440p+ output.
        let quality = std::env::var("SHATTERBOX_QUALITY").unwrap_or_default();
        // Shadow reach is now cheap enough (half-resolution pass) to cover the
        // whole visible arena; the numbers are metres along the sun.
        let (preset_scale, shadow_distance) = match quality.to_ascii_lowercase().as_str() {
            "performance" | "low" => (0.5, 56.0),
            "quality" | "high" => (0.85, 160.0),
            _ => (0.65, 112.0),
        };
        let scale_override = std::env::var("SHATTERBOX_RENDER_SCALE")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .map(|value| value.clamp(0.35, 1.0));
        let scale = scale_override.unwrap_or(preset_scale);
        let governor_enabled = scale_override.is_none();
        let exposure = std::env::var("SHATTERBOX_EXPOSURE")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(0.85)
            .clamp(0.2, 8.0);
        // Auto exposure defaults on: the linear HDR scene has a ~30x dynamic
        // range between a sunlit wall and a lamp-lit interior, and a single
        // manual value cannot keep both readable. `SHATTERBOX_AUTO_EXPOSURE=0`
        // restores the manual slider (also exposed in F1).
        let auto_exposure = !matches!(
            std::env::var("SHATTERBOX_AUTO_EXPOSURE")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "off"
        );
        let exposure_speed = std::env::var("SHATTERBOX_EXPOSURE_SPEED")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(1.8)
            .clamp(0.1, 16.0);
        let exposure_bias = std::env::var("SHATTERBOX_EXPOSURE_BIAS")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(0.0)
            .clamp(-4.0, 4.0);
        // 80 fps is the acceptance target (12.5 ms). The monitor refresh is no
        // longer the default: chasing 144 fps sacrificed the 1% low.
        let target_fps = std::env::var("SHATTERBOX_TARGET_FPS")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(80.0)
            .clamp(30.0, 144.0);
        let max_scale = if target_fps <= 90.0 { 0.85 } else { 1.0 };
        // Diagnostic switches for the lighting_set capture. The default is 0:
        // the production look always runs the full stack.
        let post_debug = std::env::var("SHATTERBOX_POST_DEBUG")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        let look = match std::env::var("SHATTERBOX_LOOK")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "b" | "2" => 1,
            "c" | "3" => 2,
            _ => 0,
        };
        let sun_temperature = std::env::var("SHATTERBOX_SUN_TEMPERATURE")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(5800.0)
            .clamp(2000.0, 12000.0);
        let fog_density = std::env::var("SHATTERBOX_FOG")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(0.012)
            .clamp(0.0, 0.15);
        // `config/sky.ron` overrides these two from the game side; the env vars
        // stay the A/B path for benchmark runs.
        let sun_intensity = std::env::var("SHATTERBOX_SUN_INTENSITY")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(1.0)
            .clamp(0.0, 5.0);
        let fog_height = std::env::var("SHATTERBOX_FOG_HEIGHT")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(5.0)
            .clamp(0.5, 60.0);
        let shadow_mode = ShadowMode::from_env();
        // `SHATTERBOX_UPSCALE_CLAMP=0` restores the overshooting Catmull-Rom
        // filter that drew bright fringes on silhouettes.
        let upscale_clamp = !matches!(
            std::env::var("SHATTERBOX_UPSCALE_CLAMP")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "off"
        );
        // Debris reactive mask is on by default; `SHATTERBOX_DEBRIS_REACTIVE=0`
        // restores the old upscaler that kept history on falling pieces.
        let debris_reactive = !matches!(
            std::env::var("SHATTERBOX_DEBRIS_REACTIVE")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "off"
        );
        gpu.queue.write_buffer(
            &exposure_state,
            0,
            bytemuck::bytes_of(&ExposureStateGpu {
                sum: 0,
                count: 0,
                adapted: exposure,
                manual: exposure,
                auto: u8::from(auto_exposure) as f32,
                speed: exposure_speed,
                bias: exposure_bias,
                dt: 1.0 / 60.0,
                target: 0.12,
                _pad: [0.0; 3],
            }),
        );
        gpu.queue.write_buffer(
            &exposure_value,
            0,
            bytemuck::bytes_of(&ExposureValueGpu {
                adapted: exposure,
                auto: u8::from(auto_exposure) as f32,
                _pad: [0.0; 2],
            }),
        );
        let targets = Targets::new(
            &gpu.device,
            output_width,
            output_height,
            scale,
            shadow_mode.is_full_res(),
        );
        let bind_groups = BindGroups::new(
            &gpu.device,
            &layouts,
            &uniform_buffer,
            &exposure_state,
            &exposure_value,
            &ddgi_state,
            &world_view,
            &group_view,
            &region_view,
            &brick_buffer,
            &mask_buffer,
            &palette_buffer,
            &debris_buffer,
            &debris_voxel_buffer,
            &debris_sphere_buffer,
            &debris_rect_buffer,
            &debris_tile_buffer,
            &targets,
            &sampler,
        );

        let query_set = gpu.timestamp_supported.then(|| {
            gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Shatterbox frame timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: TIMESTAMP_COUNT,
            })
        });
        let resolve_buffer = query_set.as_ref().map(|_| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shatterbox timestamp resolve"),
                size: u64::from(TIMESTAMP_COUNT) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let readback_buffer = query_set.as_ref().map(|_| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shatterbox timestamp readback"),
                size: u64::from(TIMESTAMP_COUNT) * 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // Particle instances are re-uploaded every frame; `MAX_PARTICLES` of
        // 32 bytes is 16 KiB, so the buffer is allocated once and never grown.
        let particle_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox particle instances"),
            size: (MAX_PARTICLES * size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let world_mirror = WorldMirror::new(&world_data);
        // `SHATTERBOX_EFFECTS` accepts "on"/"off" or a bitmask
        // (1 shadows, 2 ssao, 8 vct, 16 taa, 32 fog, 64 ssr). Handy for
        // A/B benchmarks without touching the overlay.
        let mut effects = std::env::var("SHATTERBOX_EFFECTS").map_or_else(
            |_| RenderEffects::default(),
            |raw| match raw.trim().to_ascii_lowercase().as_str() {
                "" | "on" | "all" => RenderEffects::default(),
                "off" | "none" => RenderEffects::from_bitmask(0),
                other => RenderEffects::from_bitmask(other.parse::<u32>().unwrap_or(u32::MAX)),
            },
        );
        // DDGI is user-facing through its quality preset. `SHATTERBOX_GI`
        // (off/low/medium/high) enables the internal bit 2048 itself; an unset
        // variable starts at off until part 2 is accepted. The measured medium
        // preset stays under the 1.5 ms gate.
        let ddgi_quality = DdgiQuality::from_env();
        effects.ddgi = ddgi_quality != DdgiQuality::Off;
        // `SHATTERBOX_LIGHT_AA=1` (or the F1 checkbox) enables the pre-upscale
        // silhouette smoothing. Off by default so old captures stay comparable.
        if std::env::var("SHATTERBOX_LIGHT_AA")
            .is_ok_and(|raw| !matches!(raw.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
        {
            effects.light_aa = true;
        }
        // `SHATTERBOX_FXAA=1` turns on the internal-resolution edge blur. Off by
        // default so the old no-AA captures stay comparable.
        if std::env::var("SHATTERBOX_FXAA")
            .is_ok_and(|raw| !matches!(raw.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
        {
            effects.fxaa = true;
        }
        // `SHATTERBOX_DEBRIS_CULL=0` restores the pre-part-1 debris loop in the
        // G-buffer (one 112-byte instance record per piece per pixel). On by
        // default because the sphere test cannot change the image - it only
        // skips pieces the ray misses.
        if std::env::var("SHATTERBOX_DEBRIS_CULL")
            .is_ok_and(|raw| matches!(raw.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
        {
            effects.debris_cull = false;
        }
        // `SHATTERBOX_DEBRIS_TILES=0` falls back to the per-group sphere culling.
        if std::env::var("SHATTERBOX_DEBRIS_TILES")
            .is_ok_and(|raw| matches!(raw.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
        {
            effects.debris_tiles = false;
        }
        // Which upscaler runs after lighting: our FSR 1 style EASU/RCAS port or
        // the TAAU path. `SHATTERBOX_UPSCALER=taa` restores the old behaviour.
        let upscaler = Upscaler::from_env();

        let stats = RenderStats {
            scale,
            internal_size: targets.internal_size,
            output_size: targets.output_size,
            target_fps,
            timestamps_supported: gpu.timestamp_supported,
            hot_reload_enabled: shader_watcher.enabled,
            governor_enabled,
            shadow_mode: shadow_mode.label(),
            ..RenderStats::default()
        };

        tracing::info!(
            bricks = world_data.brick_count,
            mask_words = world_data.brick_masks.len(),
            scale,
            target_fps,
            exposure,
            shadow_mode = shadow_mode.label(),
            timestamp_period = gpu.timestamp_period,
            "Phase 2 renderer ready"
        );

        Ok(Self {
            egui_ctx,
            egui_state,
            egui_renderer,
            clear_color: wgpu::Color {
                r: 0.02,
                g: 0.03,
                b: 0.05,
                a: 1.0,
            },
            format: gpu.format(),
            uniform_buffer,
            exposure_state,
            exposure_value,
            _world_texture: world_texture,
            world_view,
            _group_texture: group_texture,
            group_view,
            _region_texture: region_texture,
            region_view,
            brick_buffer,
            mask_buffer,
            palette_buffer,
            debris_buffer,
            debris_voxel_buffer,
            debris_sphere_buffer,
            debris_rect_buffer,
            debris_tile_buffer,
            debris_count: 0,
            debris_voxel_len: 0,
            debris_bounds: None,
            offscreen_target: None,
            debug_hide_debris: false,
            debug_hit: None,
            debris_instances_scratch: Vec::with_capacity(MAX_RENDER_DEBRIS),
            debris_voxels_scratch: Vec::new(),
            debris_spheres_scratch: Vec::with_capacity(MAX_RENDER_DEBRIS),
            debris_pending_scratch: Vec::with_capacity(MAX_RENDER_DEBRIS),
            debris_rects_scratch: Vec::with_capacity(MAX_RENDER_DEBRIS),
            world_mirror,
            sampler,
            layouts,
            pipelines,
            targets,
            bind_groups,
            target_cache: None,
            history_parity: 0,
            history_reset: true,
            query_set,
            resolve_buffer,
            readback_buffer,
            timestamp_period: gpu.timestamp_period,
            timestamp_ready: Arc::new(AtomicBool::new(false)),
            timestamp_pending: false,
            gbuffer_ms: 0.0,
            lighting_ms: 0.0,
            temporal_ms: 0.0,
            shadow_ms: 0.0,
            light_shadow_ms: 0.0,
            gi_ms: 0.0,
            fog_ms: 0.0,
            bloom_ms: 0.0,
            fxaa_ms: 0.0,
            particles_ms: 0.0,
            debris_cull_ms: 0.0,
            ddgi_inject_ms: 0.0,
            ddgi_update_ms: 0.0,
            ddgi_blend_ms: 0.0,
            total_ms: 0.0,
            timestamps_valid: false,
            shader_watcher,
            scale,
            governor_enabled,
            governor_elapsed: 0.0,
            governor_frame_ms: 0.0,
            governor_spikes: 0,
            target_fps,
            max_scale,
            exposure,
            post_debug,
            look,
            auto_exposure,
            exposure_speed,
            exposure_bias,
            exposure_dt: 1.0 / 60.0,
            time: 0.0,
            time_of_day: 10.5,
            sun_temperature,
            sun_intensity,
            fog_density,
            fog_height,
            shadow_distance,
            effects,
            shadow_mode,
            upscale_clamp,
            debris_reactive,
            // Jitter is OPT-IN: the user sees the sub-pixel crawl and it reads as
            // broken AA. `SHATTERBOX_TAA_JITTER=1` re-enables it for A/B of the
            // TAAU accumulation path.
            taa_jitter: matches!(
                std::env::var("SHATTERBOX_TAA_JITTER")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "on"
            ),
            upscaler,
            probe_dirty: true,
            ddgi_slice: 0,
            ddgi_reset: true,
            ddgi_quality,
            ddgi_priority_frames: 0,
            ddgi_priority_center: Vec3::ZERO,
            ddgi_priority_radius: 0.0,
            ddgi_state,
            frame_index: 0,
            prev_view_proj: Mat4::IDENTITY,
            prev_camera_pos: Vec3::ZERO,
            lights: Vec::new(),
            area_lights: Vec::new(),
            particles: Vec::new(),
            particle_buffer,
            particle_count: 0,
            stats,
        })
    }

    pub fn set_clear_color(&mut self, color: wgpu::Color) {
        self.clear_color = color;
    }

    pub fn context(&self) -> &egui::Context {
        &self.egui_ctx
    }

    pub fn stats(&self) -> RenderStats {
        self.stats
    }

    pub fn toggle_governor(&mut self) {
        self.governor_enabled = !self.governor_enabled;
    }

    /// Cycles through fixed scales and disables the automatic governor.
    /// Sets the internal render scale. The upscaler history is reset so it does
    /// not blend frames of two different resolutions.
    pub fn set_scale(&mut self, scale: f32) {
        let scale = scale.clamp(0.35, 1.0);
        if (scale - self.scale).abs() > 1.0e-6 {
            self.scale = scale;
            self.history_reset = true;
        }
    }

    /// Paints the debug aim marker at `point` (world space) on the next frame.
    /// `None` hides it. Driven by `SHATTERBOX_DEBUG_HIT` from the game.
    pub fn set_debug_hit(&mut self, point: Option<Vec3>) {
        self.debug_hit = point;
    }

    /// Replaces the smoke/dust billboards drawn this frame. The game rebuilds
    /// the list from its simulation every frame, so `&[]` clears them.
    pub fn set_particles(&mut self, particles: &[Particle]) {
        self.particles.clear();
        self.particles.extend_from_slice(particles);
    }

    /// Selects the sun shadow trace/upsample mode. Changing it re-allocates the
    /// shadow target on the next `ensure_targets`, so the first frame after a
    /// switch pays one target rebuild.
    pub fn set_shadow_mode(&mut self, mode: ShadowMode) {
        if mode != self.shadow_mode {
            self.shadow_mode = mode;
            // The shadow buffer changes resolution, so the temporal history and
            // the probe volume (which reads lit surfaces) must not be reused.
            self.history_reset = true;
            self.probe_dirty = true;
        }
    }

    #[must_use]
    pub fn shadow_mode(&self) -> ShadowMode {
        self.shadow_mode
    }

    /// Clamps (or stops clamping) the spatial upscale to the fetched tap range.
    /// Clamping removes the bright fringe and the dark seam the Catmull-Rom
    /// kernel draws on silhouettes; `SHATTERBOX_UPSCALE_CLAMP=0` is the
    /// pre-fix baseline used for A/B.
    pub fn set_upscale_clamp(&mut self, clamp: bool) {
        if clamp != self.upscale_clamp {
            self.upscale_clamp = clamp;
            self.history_reset = true;
        }
    }

    #[must_use]
    pub fn upscale_clamp(&self) -> bool {
        self.upscale_clamp
    }

    /// Turns the debris reactive mask on/off. With it off the temporal path
    /// keeps history on pixels covered by a falling piece, which is the
    /// pre-fix behaviour the overlay toggle compares against.
    pub fn set_debris_reactive(&mut self, reactive: bool) {
        if reactive != self.debris_reactive {
            self.debris_reactive = reactive;
            self.history_reset = true;
        }
    }

    #[must_use]
    pub fn debris_reactive(&self) -> bool {
        self.debris_reactive
    }

    /// Hides the dynamic debris for the next frame. Used by the autotest depth
    /// probe, which compares the rendered voxel world against the CPU raycast;
    /// held pieces must not act as occluders in that comparison.
    pub fn set_hide_debris(&mut self, hidden: bool) {
        self.debug_hide_debris = hidden;
    }

    pub fn cycle_scale(&mut self) {
        self.governor_enabled = false;
        self.scale = if self.scale < 0.6 {
            0.7
        } else if self.scale < 0.75 {
            0.8
        } else if self.scale < 0.9 {
            1.0
        } else {
            0.5
        };
    }

    /// Replaces the point light list (at most `MAX_LIGHTS` are uploaded).
    pub fn set_lights(&mut self, lights: &[PointLight]) {
        self.lights.clear();
        self.lights.extend(lights.iter().take(MAX_LIGHTS).copied());
        self.probe_dirty = true;
    }

    /// Replaces the area light list. Point lights keep the first slots, so the
    /// area lights only fill what is left of `MAX_LIGHTS`.
    pub fn set_area_lights(&mut self, lights: &[AreaLight]) {
        self.area_lights.clear();
        self.area_lights
            .extend(lights.iter().take(MAX_LIGHTS).copied());
        self.probe_dirty = true;
    }

    #[must_use]
    pub fn light_count(&self) -> usize {
        (self.lights.len() + self.area_lights.len()).min(MAX_LIGHTS)
    }

    /// Moves the sun along a 24 h arc. `hours` wraps, so 26.0 == 2.0.
    pub fn set_time_of_day(&mut self, hours: f32) {
        let hours = hours.rem_euclid(24.0);
        if (hours - self.time_of_day).abs() > 1e-4 {
            self.probe_dirty = true;
        }
        self.time_of_day = hours;
    }

    #[must_use]
    pub fn time_of_day(&self) -> f32 {
        self.time_of_day
    }

    /// Photosphere temperature of the sun in kelvin. It is the only colour
    /// control: the shader reddens it near the horizon and uses it for the
    /// moon tint at night.
    pub fn set_sun_temperature(&mut self, kelvin: f32) {
        let kelvin = kelvin.clamp(2000.0, 12000.0);
        if (kelvin - self.sun_temperature).abs() > 1e-3 {
            self.probe_dirty = true;
        }
        self.sun_temperature = kelvin;
    }

    #[must_use]
    pub fn sun_temperature(&self) -> f32 {
        self.sun_temperature
    }

    /// Optical density of the height fog at ground level, in 1/m.
    pub fn set_fog_density(&mut self, density: f32) {
        self.fog_density = density.clamp(0.0, 0.15);
    }

    #[must_use]
    pub fn fog_density(&self) -> f32 {
        self.fog_density
    }

    /// Multiplier on the sun/moon radiance, in linear HDR. Everything that reads
    /// the sun colour (direct light, fog in-scatter, probe visibility) follows,
    /// so this is an exposure-like control for the sky only.
    pub fn set_sun_intensity(&mut self, intensity: f32) {
        let intensity = intensity.clamp(0.0, 5.0);
        if (intensity - self.sun_intensity).abs() > 1e-3 {
            self.probe_dirty = true;
        }
        self.sun_intensity = intensity;
    }

    #[must_use]
    pub fn sun_intensity(&self) -> f32 {
        self.sun_intensity
    }

    /// Height falloff of the ground fog, in metres: the density drops by `e`
    /// every `metres` above the 1.5 m base. Smaller values keep the haze in the
    /// streets, larger ones fill the upper storeys.
    pub fn set_fog_height(&mut self, metres: f32) {
        self.fog_height = metres.clamp(0.5, 60.0);
    }

    #[must_use]
    pub fn fog_height(&self) -> f32 {
        self.fog_height
    }

    /// Current on/off state of the optional effects (shadows, SSAO, SSGI, VCT, TAA).
    #[must_use]
    pub fn effects(&self) -> RenderEffects {
        self.effects
    }

    /// Which upscaler turns the internal image into the output image.
    #[must_use]
    pub fn upscaler(&self) -> Upscaler {
        self.upscaler
    }

    /// Switches between the FSR 1 style upscale and the TAAU path. Both share
    /// the same bind group, so this only changes which pipeline is dispatched.
    pub fn set_upscaler(&mut self, upscaler: Upscaler) {
        if upscaler != self.upscaler {
            self.upscaler = upscaler;
            self.history_reset = true;
        }
    }

    /// Flips FSR <-> TAAU and returns the new selection (bound to a hotkey).
    pub fn cycle_upscaler(&mut self) -> Upscaler {
        let next = match self.upscaler {
            Upscaler::Fsr => Upscaler::Temporal,
            Upscaler::Temporal => Upscaler::Fsr,
        };
        self.set_upscaler(next);
        next
    }

    /// Replaces the effect set. The uniforms are re-uploaded next frame; no
    /// pipeline or bind-group rebuild happens, so toggling is free.
    pub fn set_effects(&mut self, effects: RenderEffects) {
        let jitter_changed = effects.taa != self.effects.taa;
        if effects.vct && !self.effects.vct {
            self.probe_dirty = true;
        }
        if effects.ddgi != self.effects.ddgi {
            if !effects.ddgi {
                self.ddgi_quality = DdgiQuality::Off;
            } else if self.ddgi_quality == DdgiQuality::Off {
                self.ddgi_quality = DdgiQuality::Medium;
            }
            self.ddgi_reset = true;
        }
        self.effects = effects;
        if jitter_changed {
            // The temporal history was accumulated with a different jitter
            // pattern (or with none at all); restart it to avoid a smear.
            self.history_reset = true;
        }
    }

    /// Current DDGI ray budget preset.
    #[must_use]
    pub fn ddgi_quality(&self) -> DdgiQuality {
        self.ddgi_quality
    }

    /// Changes the DDGI preset. `Off` clears the internal bit 2048; the other
    /// levels enable it and restart the atlas history.
    pub fn set_ddgi_quality(&mut self, quality: DdgiQuality) {
        if quality == self.ddgi_quality {
            return;
        }
        self.ddgi_quality = quality;
        self.effects.ddgi = quality != DdgiQuality::Off;
        self.ddgi_reset = true;
        self.ddgi_slice = 0;
    }

    /// Flips every effect off or on at once. Returns the new state.
    pub fn set_all_effects(&mut self, enabled: bool) -> RenderEffects {
        let effects = RenderEffects {
            raytraced_shadows: enabled,
            ssao: enabled,
            ssr: enabled,
            vct: enabled,
            taa: enabled,
            volumetric_fog: enabled,
            light_aa: enabled,
            fxaa: enabled,
            debris_cull: enabled,
            debris_tiles: enabled,
            ddgi: enabled,
            dynamic_smoke: enabled,
        };
        self.set_effects(effects);
        effects
    }

    /// Uploads the CPU-advected smoke density volume (`SMOKE_DIMS`, R8, 0..255).
    /// Called every frame while the dynamic-smoke checkbox is on; when the
    /// effect is off the texture keeps its last contents but `fog.wgsl` skips
    /// the lookup entirely, so no simulation work is wasted.
    pub fn set_smoke_density(&mut self, queue: &wgpu::Queue, density: &[u8]) {
        let expected = SMOKE_DIMS.0 as usize * SMOKE_DIMS.1 as usize * SMOKE_DIMS.2 as usize;
        if density.len() != expected {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.targets.smoke,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            density,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SMOKE_DIMS.0),
                rows_per_image: Some(SMOKE_DIMS.1),
            },
            wgpu::Extent3d {
                width: SMOKE_DIMS.0,
                height: SMOKE_DIMS.1,
                depth_or_array_layers: SMOKE_DIMS.2,
            },
        );
    }

    /// Sets the pre-tonemap exposure multiplier.
    pub fn set_exposure(&mut self, exposure: f32) {
        self.exposure = exposure.clamp(0.2, 8.0);
    }

    #[must_use]
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    /// Auto exposure meters the linear HDR frame and adapts over time; the
    /// manual slider is only used when this is off.
    pub fn set_auto_exposure(&mut self, enabled: bool) {
        self.auto_exposure = enabled;
    }

    #[must_use]
    pub fn auto_exposure(&self) -> bool {
        self.auto_exposure
    }

    /// Adaptation speed in `1/s`. Higher reacts faster but pumps on camera cuts.
    pub fn set_exposure_speed(&mut self, speed: f32) {
        self.exposure_speed = speed.clamp(0.1, 16.0);
    }

    #[must_use]
    pub fn exposure_speed(&self) -> f32 {
        self.exposure_speed
    }

    /// EV bias applied to the metered exposure (positive = brighter).
    pub fn set_exposure_bias(&mut self, bias: f32) {
        self.exposure_bias = bias.clamp(-4.0, 4.0);
    }

    #[must_use]
    pub fn exposure_bias(&self) -> f32 {
        self.exposure_bias
    }

    /// Post-stack diagnostic bitmask (see `post.wgsl`). Zero is the shipping
    /// look; each bit disables one stage for the lighting_set capture.
    pub fn set_post_debug(&mut self, bits: u32) {
        self.post_debug = bits;
    }

    #[must_use]
    pub fn post_debug(&self) -> u32 {
        self.post_debug
    }

    /// Look variant: 0 = a (neutral), 1 = b (warm/cool cinematic), 2 = c
    /// (high-contrast stylised). `SHATTERBOX_LOOK` and the lighting_set
    /// scenario use the same mapping.
    pub fn set_look(&mut self, look: u8) {
        self.look = look.min(2);
    }

    #[must_use]
    pub fn look(&self) -> u8 {
        self.look
    }

    /// Uploads dynamic debris transforms and (when `voxels_changed`) their voxel
    /// grids. Transforms change every physics tick, voxel data only when a new
    /// piece is created or a filled-in hole reuses a piece slot.
    pub fn upload_debris(
        &mut self,
        gpu: &GpuContext,
        views: &[DebrisView<'_>],
        voxels_changed: bool,
    ) {
        let mut pending = std::mem::take(&mut self.debris_instances_scratch);
        pending.clear();
        for view in views.iter().take(MAX_RENDER_DEBRIS) {
            let voxel_count = view.voxels.len();
            if voxel_count == 0 || view.voxel_offset as usize + voxel_count > MAX_DEBRIS_VOXELS {
                continue;
            }
            pending.push(GpuDebrisInstance {
                model: view.model.to_cols_array_2d(),
                inverse_model: view.inverse_model.to_cols_array_2d(),
                aabb_min: [view.aabb_min.x, view.aabb_min.y, view.aabb_min.z, 1.0],
                aabb_max: [view.aabb_max.x, view.aabb_max.y, view.aabb_max.z, 1.0],
                data: [
                    view.voxel_offset,
                    view.dims.x.max(1) as u32,
                    view.dims.y.max(1) as u32,
                    view.dims.z.max(1) as u32,
                ],
            });
            if voxels_changed {
                self.debris_voxels_scratch.clear();
                self.debris_voxels_scratch
                    .extend(view.voxels.iter().map(|voxel| u32::from(*voxel)));
                gpu.queue.write_buffer(
                    &self.debris_voxel_buffer,
                    u64::from(view.voxel_offset) * size_of::<u32>() as u64,
                    bytemuck::cast_slice(&self.debris_voxels_scratch),
                );
            }
            self.debris_voxel_len = self
                .debris_voxel_len
                .max(view.voxel_offset + voxel_count as u32);
        }
        // One world-space AABB around every piece: `cast_debris` in the G-buffer
        // rejects the whole set with a single slab test when the camera ray
        // misses it, instead of 64 inverse-matrix transforms per pixel.
        let mut bounds: Option<(Vec3, Vec3)> = None;
        let mut spheres = std::mem::take(&mut self.debris_spheres_scratch);
        spheres.clear();
        let mut instances = Vec::with_capacity(pending.len());
        for inst in &pending {
            let model = Mat4::from_cols_array_2d(&inst.model);
            let min = Vec3::new(inst.aabb_min[0], inst.aabb_min[1], inst.aabb_min[2]);
            let max = Vec3::new(inst.aabb_max[0], inst.aabb_max[1], inst.aabb_max[2]);
            let mut piece_lo = Vec3::splat(f32::MAX);
            let mut piece_hi = Vec3::splat(f32::MIN);
            for corner in 0..8 {
                let point = Vec3::new(
                    if corner & 1 == 0 { min.x } else { max.x },
                    if corner & 2 == 0 { min.y } else { max.y },
                    if corner & 4 == 0 { min.z } else { max.z },
                );
                let world = model.transform_point3(point);
                piece_lo = piece_lo.min(world);
                piece_hi = piece_hi.max(world);
                bounds = Some(match bounds {
                    None => (world, world),
                    Some((lo, hi)) => (lo.min(world), hi.max(world)),
                });
            }
            // World bounding sphere of the rotated piece: `cast_debris` rejects
            // with it before touching the instance record, and the tile pass
            // projects it into the piece's screen rectangle.
            let centre = (piece_lo + piece_hi) * 0.5;
            let radius = ((piece_hi - piece_lo) * 0.5).length();
            instances.push(*inst);
            spheres.push([centre.x, centre.y, centre.z, radius]);
        }
        self.debris_instances_scratch = instances;
        self.debris_pending_scratch = pending;
        self.debris_spheres_scratch = spheres;
        self.debris_count = self.debris_instances_scratch.len() as u32;
        self.debris_bounds = bounds;
        if self.debris_count > 0 {
            gpu.queue.write_buffer(
                &self.debris_buffer,
                0,
                bytemuck::cast_slice(&self.debris_instances_scratch),
            );
            gpu.queue.write_buffer(
                &self.debris_sphere_buffer,
                0,
                bytemuck::cast_slice(&self.debris_spheres_scratch),
            );
        }
    }

    /// Applies destroyed/filled bricks to the GPU mirrors. Only the changed
    /// brick-index texels, words and masks are written, never the whole world.
    pub fn apply_world_edits(
        &mut self,
        gpu: &GpuContext,
        world: &VoxelWorld,
        dirty_bricks: &[IVec3],
    ) {
        // Fresh destruction: the probes around the crater are the only ones the
        // eye notices changing, so arm the priority window. The centre is the
        // mean of the dirty brick coordinates; the radius covers a couple of
        // probes in every direction.
        if !dirty_bricks.is_empty() {
            let mut sum = IVec3::ZERO;
            for brick in dirty_bricks {
                sum += *brick;
            }
            let centre = sum.as_vec3() / dirty_bricks.len() as f32;
            self.ddgi_priority_center = (centre + Vec3::splat(0.5)) * 16.0 * 0.125;
            self.ddgi_priority_radius = 12.0;
            self.ddgi_priority_frames = 120;
        }
        let mut touched_slots = Vec::with_capacity(dirty_bricks.len());
        for &brick in dirty_bricks {
            let (Ok(bx), Ok(by), Ok(bz)) = (
                usize::try_from(brick.x),
                usize::try_from(brick.y),
                usize::try_from(brick.z),
            ) else {
                continue;
            };
            if bx >= BRICKS_X || by >= BRICKS_Y || bz >= BRICKS_Z {
                continue;
            }
            let slot = bx + BRICKS_X * (by + BRICKS_Y * bz);
            touched_slots.push(slot);
            let old_id = self.world_mirror.brick_slots[slot];
            let source = world.brick(bx, by, bz).filter(|brick| !brick.is_empty());

            let Some(source) = source else {
                if old_id != 0 {
                    self.world_mirror.brick_slots[slot] = 0;
                    self.world_mirror.free_ids.push(old_id);
                    let zero = [0u32];
                    gpu.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &self._world_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: bx as u32,
                                y: by as u32,
                                z: bz as u32,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        bytemuck::cast_slice(&zero),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: Some(1),
                        },
                        wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                continue;
            };

            let id = if old_id != 0 {
                old_id
            } else {
                let Some(id) = self.world_mirror.allocate() else {
                    tracing::warn!("debris world upload capacity exhausted");
                    continue;
                };
                self.world_mirror.brick_slots[slot] = id;
                id
            };

            let mut words = Vec::with_capacity(GPU_WORDS_PER_BRICK);
            source.append_gpu_words(&mut words);
            gpu.queue.write_buffer(
                &self.brick_buffer,
                brick_slot_offset(id),
                bytemuck::cast_slice(&words),
            );
            let mut masks = Vec::with_capacity(MASK_WORDS_PER_BRICK);
            source.append_gpu_mask_words(&mut masks);
            gpu.queue.write_buffer(
                &self.mask_buffer,
                mask_slot_offset(id),
                bytemuck::cast_slice(&masks),
            );

            let value = [id];
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self._world_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: bx as u32,
                        y: by as u32,
                        z: bz as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&value),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        let (changed_groups, changed_regions) = self.world_mirror.update_occupancy(&touched_slots);
        upload_occupancy_texels(
            gpu,
            &self._group_texture,
            self.world_mirror.group_dims,
            &self.world_mirror.group_occupancy,
            &changed_groups,
        );
        upload_occupancy_texels(
            gpu,
            &self._region_texture,
            self.world_mirror.region_dims,
            &self.world_mirror.region_occupancy,
            &changed_regions,
        );
        // Destruction changed the geometry the probe volume sees.
        self.probe_dirty = true;
    }

    pub fn handle_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit::EventResponse {
        self.egui_state.on_window_event(window, event)
    }

    /// Runs the full frame graph, composites UI and presents.
    pub fn render<F>(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        camera: &FreeCamera,
        dt: f32,
        build_ui: F,
    ) -> Result<FrameOutcome>
    where
        F: FnMut(&egui::Context),
    {
        self.frame_index += 1;
        self.time += dt.max(0.0);
        self.exposure_dt = dt.clamp(0.0, 0.25);
        self.poll_timestamps(gpu);
        self.update_governor(dt);
        self.maybe_hot_reload(gpu);

        let (output_width, output_height) = gpu.size();
        let output_width = output_width.max(1);
        let output_height = output_height.max(1);

        let (frame, suboptimal) = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(FrameOutcome::Skipped);
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                tracing::warn!("surface is outdated or lost; reconfiguring");
                gpu.reconfigure();
                return Ok(FrameOutcome::Skipped);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                bail!("surface texture acquisition failed validation");
            }
        };

        self.ensure_targets(gpu, output_width, output_height);
        let aspect = output_width as f32 / output_height as f32;
        self.write_frame_uniform(gpu, camera, aspect);

        let raw_input = self.egui_state.take_egui_input(window);
        let mut full_output = self.egui_ctx.run_pass(raw_input, build_ui);
        self.egui_state
            .handle_platform_output(window, full_output.platform_output);
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [output_width, output_height],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, image_deltas) in &full_output.textures_delta.set {
            for image_delta in image_deltas {
                self.egui_renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
            }
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Shatterbox frame encoder"),
            });
        let ui_commands = self.egui_renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // Rebuild the coarse irradiance volume first: the cone trace and the
        // fog sun shafts read it in the same frame. It only depends on the
        // world, the sun and the light rig, so it is rebuilt when one of them
        // changes instead of every frame.
        self.encode_frame_graph(&mut encoder, &gpu.queue);

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let render_timestamps = self.render_timestamp_writes(TS_POST * 2, TS_POST * 2 + 1);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Shatterbox post + UI pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.clear_color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: render_timestamps,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            pass.set_pipeline(&self.pipelines.post);
            pass.set_bind_group(0, &self.bind_groups.post[self.history_parity], &[]);
            pass.draw(0..3, 0..1);
            self.egui_renderer
                .render(&mut pass, &paint_jobs, &screen_descriptor);
        }

        let read_timestamps = self.query_set.is_some()
            && !self.timestamp_pending
            && self.frame_index.is_multiple_of(TIMESTAMP_READ_INTERVAL);
        if read_timestamps {
            let query_set = self.query_set.as_ref().expect("query set checked");
            let resolve = self.resolve_buffer.as_ref().expect("resolve buffer");
            let readback = self.readback_buffer.as_ref().expect("readback buffer");
            let byte_size = u64::from(TIMESTAMP_COUNT) * 8;
            encoder.resolve_query_set(query_set, 0..TIMESTAMP_COUNT, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, byte_size);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        full_output.textures_delta.clear();

        gpu.queue.submit(
            ui_commands
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        gpu.queue.present(frame);

        if read_timestamps {
            self.begin_timestamp_map(gpu);
        }

        self.prev_view_proj = camera.view_projection(aspect);
        self.prev_camera_pos = camera.position;
        self.history_parity ^= 1;
        self.history_reset = false;

        self.stats = RenderStats {
            gbuffer_ms: self.gbuffer_ms,
            lighting_ms: self.lighting_ms,
            temporal_ms: self.temporal_ms,
            shadow_ms: self.shadow_ms,
            light_shadow_ms: self.light_shadow_ms,
            gi_ms: self.gi_ms,
            fog_ms: self.fog_ms,
            bloom_ms: self.bloom_ms,
            fxaa_ms: self.fxaa_ms,
            particles_ms: self.particles_ms,
            debris_cull_ms: self.debris_cull_ms,
            ddgi_inject_ms: self.ddgi_inject_ms,
            ddgi_update_ms: self.ddgi_update_ms,
            ddgi_blend_ms: self.ddgi_blend_ms,
            total_ms: self.total_ms,
            scale: self.scale,
            internal_size: self.targets.internal_size,
            output_size: (output_width, output_height),
            target_fps: self.target_fps,
            timestamps_supported: self.query_set.is_some(),
            timestamps_valid: self.timestamps_valid,
            hot_reload_enabled: self.shader_watcher.enabled,
            governor_enabled: self.governor_enabled,
            frame_index: self.frame_index,
            light_count: self.light_count() as u32,
            debris_count: self.debris_count,
            debris_limit: MAX_RENDER_DEBRIS as u32,
            time_of_day: self.time_of_day,
            sun_temperature: self.sun_temperature,
            shadow_mode: self.shadow_mode.label(),
            upscaler: self.upscaler.name(),
        };

        if suboptimal {
            gpu.reconfigure();
        }

        Ok(FrameOutcome::Presented)
    }

    /// Writes this frame's `FrameUniform`. Shared by the present path and the
    /// offscreen capture path so both see identical camera/effect state.
    fn write_frame_uniform(&mut self, gpu: &GpuContext, camera: &FreeCamera, aspect: f32) {
        // Sub-pixel jitter for TAAU: a Halton(2,3) sequence over the 8-frame
        // cycle, in internal pixels. `SHATTERBOX_TAA_JITTER=0` restores the old
        // jitter-free path for A/B (the temporal pass then just upsamples).
        let jitter = if self.taa_jitter && self.effects.taa {
            let index = (self.frame_index % 8) as u32 + 1;
            [halton(index, 2) - 0.5, halton(index, 3) - 0.5]
        } else {
            [0.0, 0.0]
        };
        let mut uniform = FrameUniform::new(
            camera,
            aspect,
            self.prev_view_proj,
            self.prev_camera_pos,
            self.time,
            self.frame_index,
            jitter,
            self.history_reset,
            self.targets.internal_size,
            self.exposure,
            // `assert_hit_matches_render` probes the *voxel world* with the CPU
            // raycast, so the capture frames hide the dynamic debris: a piece
            // hanging 0.3 m in front of a fresh crater would otherwise report a
            // hit where the world is already empty.
            if self.debug_hide_debris {
                0
            } else {
                self.debris_count
            },
            &self.lights,
            &self.area_lights,
            self.debris_bounds,
            self.debug_hit,
            self.time_of_day,
            self.shadow_distance,
            self.sun_temperature,
            self.sun_intensity,
            self.effects,
            self.fog_density,
            self.fog_height,
            self.shadow_mode,
            self.upscale_clamp,
        );
        // Debris reactive mask: the temporal upscaler must not keep history on
        // a pixel covered by a falling rigid body, or the piece leaves a trail.
        uniform.params4[0] = if self.debris_reactive { 1.0 } else { 0.0 };
        // Lighting/post diagnostics. `z` is the stage-disable bitmask and `w`
        // selects the look; both are read directly from the shared uniform so
        // scenario steps can switch them without rebuilding pipelines.
        uniform.params4[2] = self.post_debug as f32;
        uniform.params4[3] = f32::from(self.look);
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        // Everything after `adapted` is CPU-owned. Writing the tail through a
        // typed view keeps offset 8 (the adapted exposure) untouched while the
        // exposure pass keeps accumulating across frames.
        let exposure_tail: [f32; 9] = [
            self.exposure,
            f32::from(self.auto_exposure),
            self.exposure_speed,
            self.exposure_bias,
            self.exposure_dt,
            0.12,
            0.0,
            0.0,
            0.0,
        ];
        gpu.queue.write_buffer(
            &self.exposure_state,
            size_of::<u32>() as u64 * 3,
            bytemuck::cast_slice(&exposure_tail),
        );
        self.upload_particles(gpu);
        self.upload_debris_tiles(gpu, camera, aspect);
    }

    /// Projects every uploaded debris piece into a conservative pixel rectangle
    /// and hands it to `debris_cull.wgsl`, which buckets the pieces per 8x8 tile.
    /// A piece that straddles the near plane gets the whole screen, so a mask can
    /// only ever be too generous — the culling never removes a visible piece.
    fn upload_debris_tiles(&mut self, gpu: &GpuContext, camera: &FreeCamera, aspect: f32) {
        let count = (self.debris_count as usize).min(MAX_RENDER_DEBRIS);
        if count == 0 {
            return;
        }
        let view_proj = camera.view_projection(aspect);
        let (width, height) = self.targets.internal_size;
        let size = Vec2::new(width.max(1) as f32, height.max(1) as f32);
        self.debris_rects_scratch.clear();
        for index in 0..count.min(self.debris_spheres_scratch.len()) {
            let sphere = self.debris_spheres_scratch[index];
            let centre = Vec3::new(sphere[0], sphere[1], sphere[2]);
            let radius = sphere[3].max(1e-3);
            let mut lo = Vec2::splat(f32::MAX);
            let mut hi = Vec2::splat(f32::MIN);
            let mut front = 0;
            for corner in 0..8 {
                let point = centre
                    + Vec3::new(
                        if corner & 1 == 0 { -radius } else { radius },
                        if corner & 2 == 0 { -radius } else { radius },
                        if corner & 4 == 0 { -radius } else { radius },
                    );
                let clip = view_proj * point.extend(1.0);
                if clip.w <= 1e-4 {
                    continue;
                }
                front += 1;
                let ndc = clip.truncate() / clip.w;
                lo = lo.min(Vec2::new(ndc.x, ndc.y));
                hi = hi.max(Vec2::new(ndc.x, ndc.y));
            }
            // Behind the camera: no ray from the eye can hit it, so the piece can
            // be dropped from every tile. This is the common case right after a
            // blast (the camera sees the outside of the house), and treating it as
            // "straddles the near plane" is what made the first tile run *slower*
            // than no culling at all: every such piece landed in every tile.
            let rect = if front == 0 {
                [2.0e6, 2.0e6, 2.0e6, 2.0e6]
            } else {
                // A piece that crosses the near plane projects to unbounded NDC, so
                // clamp the rectangle to a few screens: everything visible is still
                // inside it, and one such piece costs a full-screen mask instead of
                // a mask with a 1e5 x screen bogus rectangle (which is what the
                // first version produced and why it was slower than no culling).
                let lo = lo.max(Vec2::splat(-4.0)).min(Vec2::splat(4.0));
                let hi = hi.max(Vec2::splat(-4.0)).min(Vec2::splat(4.0));
                // One pixel of slack on every side: the mask must be conservative.
                [
                    (lo.x * 0.5 + 0.5) * size.x - 1.0,
                    (0.5 - hi.y * 0.5) * size.y - 1.0,
                    (hi.x * 0.5 + 0.5) * size.x + 1.0,
                    (0.5 - lo.y * 0.5) * size.y + 1.0,
                ]
            };
            self.debris_rects_scratch.push(rect);
        }
        // Debug aid: `SHATTERBOX_DEBUG_TILES=1` prints how many pieces survive the
        // screen-rect test per frame and how much screen they cover. A tile list
        // only helps while the pieces are small on screen.
        if debug_tiles() {
            let (mut full, mut off, mut area) = (0, 0, 0.0f64);
            let screen = (size.x as f64) * (size.y as f64);
            for rect in &self.debris_rects_scratch {
                let w = (rect[2] - rect[0]).max(0.0) as f64;
                let h = (rect[3] - rect[1]).max(0.0) as f64;
                if rect[0] > 1.0e5 {
                    off += 1;
                } else if w * h >= screen {
                    full += 1;
                }
                area += w * h;
            }
            let total = (size.x as f64) * (size.y as f64);
            // Mirror the shader's overlap test so the log shows the real candidate
            // count per tile, not just the piece count.
            let tiles_x = size.x.ceil() as i64 / 8 + 1;
            let tiles_y = size.y.ceil() as i64 / 8 + 1;
            let mut per_tile = vec![0u32; (tiles_x * tiles_y) as usize];
            for rect in &self.debris_rects_scratch {
                if rect[0] > 1.0e5 {
                    continue;
                }
                let x0 = (rect[0] / 8.0).floor().max(0.0) as i64;
                let y0 = (rect[1] / 8.0).floor().max(0.0) as i64;
                let x1 = ((rect[2] / 8.0).ceil() as i64).min(tiles_x - 1);
                let y1 = ((rect[3] / 8.0).ceil() as i64).min(tiles_y - 1);
                for y in y0..=y1 {
                    let row = y * tiles_x;
                    for x in x0..=x1 {
                        per_tile[(row + x) as usize] += 1;
                    }
                }
            }
            let mut sorted = per_tile.clone();
            sorted.sort_unstable();
            let pick = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
            tracing::info!(
                pieces = count,
                full_screen = full,
                offscreen = off,
                mean_screen_frac = area / total.max(1.0) / count.max(1) as f64,
                cand_p50 = pick(0.5),
                cand_p99 = pick(0.99),
                cand_max = sorted[sorted.len() - 1],
                "debris tile rects"
            );
        }
        if !self.debris_rects_scratch.is_empty() {
            gpu.queue.write_buffer(
                &self.debris_rect_buffer,
                0,
                bytemuck::cast_slice(&self.debris_rects_scratch),
            );
        }
    }

    /// Pushes the billboards collected by `set_particles` into the instance
    /// buffer. Zero puffs still updates `particle_count`, which is what gates
    /// the render pass.
    fn upload_particles(&mut self, gpu: &GpuContext) {
        let count = self.particles.len().min(MAX_PARTICLES);
        self.particle_count = count as u32;
        if count == 0 {
            return;
        }
        gpu.queue.write_buffer(
            &self.particle_buffer,
            0,
            bytemuck::cast_slice(&self.particles[..count]),
        );
    }

    /// Dispatches the whole compute graph into `encoder`, up to and including
    /// the bloom blur. The caller adds the composite pass, so the same graph
    /// feeds the swapchain and the offscreen screenshot texture.
    fn encode_frame_graph(&mut self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        // Any world or lighting change invalidates the probe history: retrace
        // the cycle without hysteresis instead of smearing the old room.
        if self.probe_dirty {
            self.ddgi_reset = true;
        }
        if (self.effects.vct || self.effects.volumetric_fog) && self.probe_dirty {
            self.dispatch_volume(
                encoder,
                &self.pipelines.probe,
                &self.bind_groups.probe,
                self.targets.probe_size,
            );
            self.probe_dirty = false;
        }
        // Per-tile debris candidate masks. The G-buffer reads them to skip every
        // piece that cannot cover its tile, instead of transforming all of them
        // per pixel (see `debris_cull.wgsl`).
        if self.effects.debris_tiles && self.debris_count > 0 {
            let (width, height) = self.targets.internal_size;
            self.dispatch(
                encoder,
                Some(TS_DEBRIS_CULL),
                &self.pipelines.debris_cull,
                &self.bind_groups.debris_cull,
                (width.div_ceil(8), height.div_ceil(8)),
            );
        }
        self.dispatch(
            encoder,
            Some(TS_GBUFFER),
            &self.pipelines.gbuffer,
            &self.bind_groups.gbuffer,
            self.targets.internal_size,
        );
        // Sun shadows read the G-buffer depth/normal written above and are
        // sampled by the lighting pass below. The dispatch size follows the
        // shadow target, not the half-res GI target it usually shares.
        if self.effects.raytraced_shadows {
            self.dispatch(
                encoder,
                Some(TS_SHADOW),
                &self.pipelines.shadow,
                &self.bind_groups.shadow,
                self.targets.shadow_size,
            );
            // Local lights share the same gate: with shadows off nothing reads
            // the mask, so tracing it would be pure waste. With no active
            // lights the pass has nothing to trace either (an empty arena rig
            // used to pay for a full half-res dispatch every frame).
            if self.light_count() > 0 {
                self.dispatch(
                    encoder,
                    Some(TS_LIGHT_SHADOW),
                    &self.pipelines.light_shadow,
                    &self.bind_groups.light_shadow,
                    self.targets.gi_size,
                );
            }
        }
        // DDGI: splat this frame's albedo, trace the refreshed probe slices and
        // convolve them into irradiance. Freshly destroyed geometry arms a
        // priority window: probes near the crater are refreshed every frame with
        // a lower hysteresis, so a room re-lights in well under a second instead
        // of waiting for the 64-frame slice cycle.
        if self.effects.ddgi {
            let priority = self.ddgi_priority_frames > 0;
            let slices = if priority {
                self.ddgi_quality.slices().max(2)
            } else {
                self.ddgi_quality.slices()
            };
            let priority_center =
                self.ddgi_priority_center / DDGI_PROBE_SPACING_M - Vec3::splat(0.5);
            let state = [
                self.ddgi_slice,
                DDGI_HYSTERESIS.to_bits(),
                DDGI_PROBE_SPACING_M.to_bits(),
                u32::from(self.ddgi_reset),
                self.ddgi_quality.tile(),
                self.ddgi_quality.samples(),
                slices,
                u32::from(priority),
                priority_center.x.to_bits(),
                priority_center.y.to_bits(),
                priority_center.z.to_bits(),
                self.ddgi_priority_radius.to_bits(),
                DDGI_PRIORITY_HYSTERESIS.to_bits(),
                0,
                0,
                0,
            ];
            queue.write_buffer(&self.ddgi_state, 0, bytemuck::cast_slice(&state));
            self.dispatch(
                encoder,
                Some(TS_DDGI_INJECT),
                &self.pipelines.ddgi_inject,
                &self.bind_groups.ddgi_inject,
                self.targets.gi_size,
            );
            let extent = (DDGI_GRID.0 * DDGI_TILE, DDGI_GRID.1 * DDGI_TILE);
            for _ in 0..slices {
                let parity = (self.ddgi_slice % 2) as usize;
                self.dispatch(
                    encoder,
                    Some(TS_DDGI_UPDATE),
                    &self.pipelines.ddgi_trace,
                    &self.bind_groups.ddgi_trace[parity],
                    extent,
                );
                self.dispatch(
                    encoder,
                    Some(TS_DDGI_BLEND),
                    &self.pipelines.ddgi,
                    &self.bind_groups.ddgi[parity],
                    extent,
                );
                self.ddgi_slice = (self.ddgi_slice + 1) % DDGI_GRID.2;
                if self.ddgi_slice == 0 {
                    // One full cycle has been written without history: from now
                    // on the blend keeps `DDGI_HYSTERESIS` of the previous texel.
                    self.ddgi_reset = false;
                }
            }
            if self.ddgi_priority_frames > 0 {
                self.ddgi_priority_frames -= 1;
            }
        }
        if self.effects.volumetric_fog {
            self.dispatch(
                encoder,
                Some(TS_FOG),
                &self.pipelines.fog,
                &self.bind_groups.fog,
                self.targets.gi_size,
            );
        }
        // Atlas written by the last blend dispatch above (the cursor already
        // points at the next slice).
        let ddgi_atlas = if self.effects.ddgi {
            (((self.ddgi_slice + DDGI_GRID.2 - 1) % DDGI_GRID.2) % 2) as usize
        } else {
            0
        };
        // SSAO, SSR and the cone trace share one half-resolution pass. When all
        // three are off, the lighting pass ignores the textures entirely, so
        // the dispatch is skipped.
        if self.effects.ssao || self.effects.ssr || self.effects.vct || self.effects.ddgi {
            self.dispatch(
                encoder,
                Some(TS_GI),
                &self.pipelines.gi,
                &self.bind_groups.gi[self.history_parity][ddgi_atlas],
                self.targets.gi_size,
            );
        }
        self.dispatch(
            encoder,
            Some(TS_LIGHTING),
            &self.pipelines.lighting,
            &self.bind_groups.lighting,
            self.targets.internal_size,
        );
        // Smoke and dust are camera-facing billboards composited over the lit
        // HDR image: emitted after lighting (they are not traced geometry, so
        // they must not occlude anything in the G-buffer) and before
        // FXAA/upscale (so the edge blur, the fog and the temporal history all
        // see them as part of the frame). Zero puffs records no pass at all.
        if self.particle_count > 0 {
            // Billboards draw into the half-res GI target, which lighting has
            // already consumed: the cloud costs a quarter of the pixels, and the
            // depth-aware composite below hides the half-res edge.
            let timestamps = self.render_timestamp_writes(TS_PARTICLES * 2, TS_PARTICLES * 2 + 1);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shatterbox particle pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.gi_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timestamps,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines.particles);
            pass.set_bind_group(0, &self.bind_groups.particles, &[]);
            pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
            pass.draw(0..6, 0..self.particle_count);
            drop(pass);

            let timestamps = self.render_timestamp_writes(
                TS_PARTICLES_COMPOSITE * 2,
                TS_PARTICLES_COMPOSITE * 2 + 1,
            );
            let mut composite = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shatterbox smoke composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timestamps,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            composite.set_pipeline(&self.pipelines.particles_composite);
            composite.set_bind_group(0, &self.bind_groups.particles_composite, &[]);
            composite.draw(0..3, 0..1);
        }
        // FXAA filters the internal image in place of the raw HDR, so the
        // upscaler below has to read a different binding.
        let upscale_source = if self.effects.fxaa {
            self.dispatch(
                encoder,
                Some(TS_FXAA),
                &self.pipelines.fxaa,
                &self.bind_groups.fxaa,
                self.targets.internal_size,
            );
            &self.bind_groups.temporal_aa[self.history_parity]
        } else {
            &self.bind_groups.temporal[self.history_parity]
        };
        let upscale_pipeline = match self.upscaler {
            Upscaler::Fsr => &self.pipelines.fsr,
            Upscaler::Temporal => &self.pipelines.temporal,
        };
        self.dispatch(
            encoder,
            Some(TS_UPSCALE),
            upscale_pipeline,
            upscale_source,
            self.targets.output_size,
        );
        self.dispatch(
            encoder,
            Some(TS_BLOOM_PREFILTER),
            &self.pipelines.bloom_prefilter,
            &self.bind_groups.bloom_prefilter,
            self.targets.bloom_size,
        );
        // The prefilter accumulated one log-luminance sample per quarter-res
        // texel. Resolve it before post reads the adapted value.
        self.dispatch(
            encoder,
            None,
            &self.pipelines.exposure,
            &self.bind_groups.bloom_prefilter,
            (1, 1),
        );
        self.dispatch(
            encoder,
            Some(TS_BLOOM_BLUR_H),
            &self.pipelines.bloom_blur_h,
            &self.bind_groups.bloom_blur_h,
            self.targets.bloom_size,
        );
        self.dispatch(
            encoder,
            Some(TS_BLOOM_BLUR_V),
            &self.pipelines.bloom_blur_v,
            &self.bind_groups.bloom_blur_v,
            self.targets.bloom_size,
        );
    }

    /// Renders one frame into an offscreen texture of `output_size` and returns
    /// the pixels. Used by the autotest harness: the swapchain is not touched,
    /// so the screenshot is exactly the frames the graph produced, at a fixed
    /// resolution independent of the window.
    ///
    /// No egui pass is recorded, so the PNG never contains the debug overlay.
    pub fn render_offscreen(
        &mut self,
        gpu: &mut GpuContext,
        camera: &FreeCamera,
        dt: f32,
        output_size: (u32, u32),
        capture: bool,
    ) -> Result<Option<CapturedFrame>> {
        self.frame_index += 1;
        self.time += dt.max(0.0);
        self.exposure_dt = dt.clamp(0.0, 0.25);
        self.poll_timestamps(gpu);
        self.maybe_hot_reload(gpu);

        let width = output_size.0.max(1);
        let height = output_size.1.max(1);
        self.ensure_targets(gpu, width, height);
        let aspect = width as f32 / height as f32;
        self.write_frame_uniform(gpu, camera, aspect);

        let format = gpu.format();
        // The capture target is reused across frames. Allocating a full-size
        // texture every frame made the harness report ~100 ms frame spikes that
        // the interactive path (rendering into the swapchain) never shows.
        if self
            .offscreen_target
            .as_ref()
            .is_none_or(|(_, _, size)| *size != (width, height))
        {
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Shatterbox offscreen capture target"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.offscreen_target = Some((texture, view, (width, height)));
        }
        let (target, target_view) = self
            .offscreen_target
            .as_ref()
            .map(|(texture, view, _)| (texture.clone(), view.clone()))
            .expect("capture target created above");

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Shatterbox offscreen encoder"),
            });
        self.encode_frame_graph(&mut encoder, &gpu.queue);
        let render_timestamps = self.render_timestamp_writes(TS_POST * 2, TS_POST * 2 + 1);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Shatterbox offscreen composite pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.clear_color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: render_timestamps,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            pass.set_pipeline(&self.pipelines.post);
            pass.set_bind_group(0, &self.bind_groups.post[self.history_parity], &[]);
            pass.draw(0..3, 0..1);
        }

        let read_timestamps = self.query_set.is_some()
            && !self.timestamp_pending
            && self.frame_index.is_multiple_of(TIMESTAMP_READ_INTERVAL);
        if read_timestamps {
            let query_set = self.query_set.as_ref().expect("query set checked");
            let resolve = self.resolve_buffer.as_ref().expect("resolve buffer");
            let readback = self.readback_buffer.as_ref().expect("readback buffer");
            let byte_size = u64::from(TIMESTAMP_COUNT) * 8;
            encoder.resolve_query_set(query_set, 0..TIMESTAMP_COUNT, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, byte_size);
        }

        // Textures must be copied on a 256-byte row pitch, so the readback
        // buffer is padded and the padding is stripped when the PNG is built.
        let bpp = 4u32;
        let unpadded_bytes_per_row = width * bpp;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(256) * 256;
        let color_buffer = capture.then(|| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shatterbox screenshot readback"),
                size: u64::from(padded_bytes_per_row) * u64::from(height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        if let Some(buffer) = &color_buffer {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        // A small window of the depth buffer, for `assert_hit_matches_render`.
        // Depth holds the linear distance along the view ray, so the world point
        // is `camera + ray * depth`. A window (not a single centre texel) is
        // needed because a one-brick lateral shift leaves the centre distance
        // identical on a flat wall and only shows up next to a hole edge.
        let depth_buffer = capture.then(|| {
            let internal = self.targets.internal_size;
            let window = DEPTH_WINDOW.min(internal.0.min(internal.1)).max(1);
            let row = (window * 4).div_ceil(256) * 256;
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shatterbox depth window readback"),
                size: u64::from(row) * u64::from(window),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.targets.depth,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: (internal.0 - window) / 2,
                        y: (internal.1 - window) / 2,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(window),
                    },
                },
                wgpu::Extent3d {
                    width: window,
                    height: window,
                    depth_or_array_layers: 1,
                },
            );
            (buffer, window, row)
        });

        gpu.queue.submit(std::iter::once(encoder.finish()));
        if read_timestamps {
            self.begin_timestamp_map(gpu);
        }

        self.prev_view_proj = camera.view_projection(aspect);
        self.prev_camera_pos = camera.position;
        self.history_parity ^= 1;
        self.history_reset = false;
        self.stats = RenderStats {
            gbuffer_ms: self.gbuffer_ms,
            lighting_ms: self.lighting_ms,
            temporal_ms: self.temporal_ms,
            shadow_ms: self.shadow_ms,
            light_shadow_ms: self.light_shadow_ms,
            gi_ms: self.gi_ms,
            fog_ms: self.fog_ms,
            bloom_ms: self.bloom_ms,
            fxaa_ms: self.fxaa_ms,
            particles_ms: self.particles_ms,
            debris_cull_ms: self.debris_cull_ms,
            ddgi_inject_ms: self.ddgi_inject_ms,
            ddgi_update_ms: self.ddgi_update_ms,
            ddgi_blend_ms: self.ddgi_blend_ms,
            total_ms: self.total_ms,
            scale: self.scale,
            internal_size: self.targets.internal_size,
            output_size: (width, height),
            target_fps: self.target_fps,
            timestamps_supported: self.query_set.is_some(),
            timestamps_valid: self.timestamps_valid,
            hot_reload_enabled: self.shader_watcher.enabled,
            governor_enabled: self.governor_enabled,
            frame_index: self.frame_index,
            light_count: self.light_count() as u32,
            debris_count: self.debris_count,
            debris_limit: MAX_RENDER_DEBRIS as u32,
            time_of_day: self.time_of_day,
            sun_temperature: self.sun_temperature,
            shadow_mode: self.shadow_mode.label(),
            upscaler: self.upscaler.name(),
        };

        let (Some(color_buffer), Some((depth_buffer, depth_window, depth_row))) =
            (color_buffer, depth_buffer)
        else {
            return Ok(None);
        };
        let color_slice = color_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        color_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let depth_slice = depth_buffer.slice(..);
        let (depth_sender, depth_receiver) = std::sync::mpsc::channel();
        depth_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = depth_sender.send(result);
        });
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .context("screenshot readback channel closed")?
            .context("screenshot readback failed")?;
        let depth_result = depth_receiver.recv().ok().and_then(|result| result.ok());

        let mut rgba = Vec::with_capacity((width * height * bpp) as usize);
        {
            let mapped = color_slice
                .get_mapped_range()
                .context("screenshot colour buffer map failed")?;
            let swap_red_blue = matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            );
            for row in 0..height as usize {
                let start = row * padded_bytes_per_row as usize;
                let line = &mapped[start..start + unpadded_bytes_per_row as usize];
                if swap_red_blue {
                    for pixel in line.chunks_exact(4) {
                        rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
                    }
                } else {
                    rgba.extend_from_slice(line);
                }
            }
        }
        color_buffer.unmap();
        let internal_size = self.targets.internal_size;
        let mut depth_distance = None;
        let mut depth_taps = Vec::with_capacity(DEPTH_TAPS.len());
        if depth_result.is_some()
            && let Ok(mapped) = depth_slice.get_mapped_range()
        {
            for (index, offset) in DEPTH_TAPS.iter().enumerate() {
                let x = (depth_window / 2) as i32 + offset[0];
                let y = (depth_window / 2) as i32 + offset[1];
                if x < 0 || y < 0 || x >= depth_window as i32 || y >= depth_window as i32 {
                    continue;
                }
                let byte = y as usize * depth_row as usize + x as usize * 4;
                let Some(bytes) = mapped.get(byte..byte + 4) else {
                    continue;
                };
                let raw = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                // The G-buffer writes `-1` for a ray that left the world.
                let distance = (raw >= 0.0).then_some(raw);
                if index == 0 {
                    depth_distance = distance;
                }
                depth_taps.push(DepthTap {
                    pixel: [
                        internal_size.0 / 2 - depth_window / 2 + x as u32,
                        internal_size.1 / 2 - depth_window / 2 + y as u32,
                    ],
                    distance,
                });
            }
        }
        depth_buffer.unmap();

        Ok(Some(CapturedFrame {
            width,
            height,
            rgba,
            internal_size,
            depth_distance,
            depth_taps,
        }))
    }

    fn timestamp_writes(
        &self,
        begin: u32,
        end: u32,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.query_set
            .as_ref()
            .map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(begin),
                end_of_pass_write_index: Some(end),
            })
    }

    fn render_timestamp_writes(
        &self,
        begin: u32,
        end: u32,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.query_set
            .as_ref()
            .map(|query_set| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(begin),
                end_of_pass_write_index: Some(end),
            })
    }

    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pair: Option<u32>,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        size: (u32, u32),
    ) {
        let timestamps = pair.and_then(|pair| self.timestamp_writes(pair * 2, pair * 2 + 1));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Shatterbox compute"),
            timestamp_writes: timestamps,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(WORKGROUP), size.1.div_ceil(WORKGROUP), 1);
    }

    /// Dispatches a compute pipeline over a 3D voxel grid (the irradiance probe
    /// volume). Its shader uses a 4x4x4 workgroup: 8x8x8 would exceed the
    /// 256-invocation limit most adapters advertise.
    fn dispatch_volume(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        size: (u32, u32, u32),
    ) {
        const VOLUME_WORKGROUP: u32 = 4;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Shatterbox irradiance volume"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(
            size.0.div_ceil(VOLUME_WORKGROUP),
            size.1.div_ceil(VOLUME_WORKGROUP),
            size.2.div_ceil(VOLUME_WORKGROUP),
        );
    }

    fn ensure_targets(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        let internal = internal_dimensions(width, height, self.scale);
        let output = (width.max(1), height.max(1));
        let shadow_full = self.shadow_mode.is_full_res();
        if internal == self.targets.internal_size
            && output == self.targets.output_size
            && shadow_full == self.targets.shadow_full
        {
            return;
        }
        let cached = self
            .target_cache
            .take()
            .filter(|cache| cache.internal == internal && cache.output == output);
        let (targets, bind_groups) = match cached {
            // The cache is keyed on resolution only; a shadow-mode switch keeps
            // the same sizes but needs a differently sized shadow texture.
            Some(cache) if cache.targets.shadow_full == shadow_full => {
                (cache.targets, cache.bind_groups)
            }
            _ => {
                let targets = Targets::new(&gpu.device, width, height, self.scale, shadow_full);
                let bind_groups = BindGroups::new(
                    &gpu.device,
                    &self.layouts,
                    &self.uniform_buffer,
                    &self.exposure_state,
                    &self.exposure_value,
                    &self.ddgi_state,
                    &self.world_view,
                    &self.group_view,
                    &self.region_view,
                    &self.brick_buffer,
                    &self.mask_buffer,
                    &self.palette_buffer,
                    &self.debris_buffer,
                    &self.debris_voxel_buffer,
                    &self.debris_sphere_buffer,
                    &self.debris_rect_buffer,
                    &self.debris_tile_buffer,
                    &targets,
                    &self.sampler,
                );
                (targets, bind_groups)
            }
        };
        let previous_targets = std::mem::replace(&mut self.targets, targets);
        let previous_bind_groups = std::mem::replace(&mut self.bind_groups, bind_groups);
        self.target_cache = Some(TargetCache {
            internal: previous_targets.internal_size,
            output: previous_targets.output_size,
            targets: previous_targets,
            bind_groups: previous_bind_groups,
        });
        // New history textures start empty; dropping the temporal accumulation
        // would smear a stale frame over the new resolution anyway.
        self.history_reset = true;
        // `Targets` also owns the irradiance volume, which is reallocated empty.
        // Without this the probe pass stays skipped until the world or the light
        // rig changes, leaving VCT and the fog light shafts black.
        self.probe_dirty = true;
        // The DDGI atlases were reallocated empty (or reused with stale
        // content): restart the slice cursor and run one cycle without
        // hysteresis so the new storage does not blend up from black.
        self.ddgi_slice = 0;
        self.ddgi_reset = true;
    }

    fn update_governor(&mut self, dt: f32) {
        self.governor_elapsed += dt.max(0.0);
        let frame_ms = dt * 1000.0;
        let target_ms = 1000.0 / self.target_fps.max(1.0);
        // Shader reloads, window resizes and compositor stalls produce frames
        // that say nothing about steady-state cost; folding them into the
        // average is what used to drop the internal resolution after a hitch.
        let plausible = frame_ms > 0.0 && frame_ms < target_ms * 2.5;
        if plausible {
            let alpha = 1.0 - (-dt / 0.25).exp();
            self.governor_frame_ms = if self.governor_frame_ms <= 0.0 {
                frame_ms
            } else {
                self.governor_frame_ms + (frame_ms - self.governor_frame_ms) * alpha
            };
            // The average can sit just under budget while every destruction
            // burst frame spikes; count those separately for the 1% low.
            if frame_ms > target_ms * 1.35 {
                self.governor_spikes = self.governor_spikes.saturating_add(1);
            } else {
                self.governor_spikes = self.governor_spikes.saturating_sub(1);
            }
        }
        if self.governor_elapsed < 1.5 {
            return;
        }
        self.governor_elapsed = 0.0;
        if !self.governor_enabled || self.governor_frame_ms <= 0.0 {
            return;
        }

        let target_ms = 1000.0 / self.target_fps.max(1.0);
        // Wide dead band plus a 1.5 s dwell. Stepping up needs real headroom
        // (10%), stepping down needs a real overshoot (12%): the governor used
        // to hunt between two scales every second, and each switch re-allocated
        // the internal targets, which read as a visible hitch.
        if (self.governor_frame_ms > target_ms * 1.12 || self.governor_spikes >= 3)
            && self.scale > 0.35
        {
            self.scale = (self.scale - 0.05).max(0.35);
            self.governor_spikes = 0;
        } else if self.governor_frame_ms < target_ms * 0.90 && self.scale < self.max_scale {
            // At an 80 fps target the ceiling is 0.85; climbing to native 1440p
            // would spend the whole budget on pixels and collapse the 1% low.
            self.scale = (self.scale + 0.05).min(self.max_scale);
            self.governor_spikes = 0;
        } else {
            self.governor_spikes = 0;
        }
    }

    fn maybe_hot_reload(&mut self, gpu: &GpuContext) {
        let Some(sources) = self.shader_watcher.poll() else {
            return;
        };
        match build_pipelines(&gpu.device, &self.layouts, self.format, &sources) {
            Ok(pipelines) => self.pipelines = pipelines,
            Err(error) => {
                tracing::error!(%error, "shader reload rejected; keeping the previous pipelines");
                return;
            }
        }
        tracing::info!("Phase 2 shaders reloaded");
    }

    fn begin_timestamp_map(&mut self, gpu: &GpuContext) {
        let Some(buffer) = &self.readback_buffer else {
            return;
        };
        self.timestamp_ready.store(false, Ordering::SeqCst);
        let ready = Arc::clone(&self.timestamp_ready);
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |_| {
            ready.store(true, Ordering::SeqCst);
        });
        self.timestamp_pending = true;
        // Progress the map without blocking the CPU on GPU completion.
        let _ = gpu.device.poll(wgpu::PollType::Poll);
    }

    fn poll_timestamps(&mut self, gpu: &GpuContext) {
        if !self.timestamp_pending {
            return;
        }
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        if !self.timestamp_ready.load(Ordering::SeqCst) {
            return;
        }
        let Some(buffer) = &self.readback_buffer else {
            self.timestamp_pending = false;
            return;
        };
        let mapped = match buffer.slice(..).get_mapped_range() {
            Ok(mapped) => mapped,
            Err(error) => {
                tracing::warn!(?error, "failed to map timestamp readback buffer");
                buffer.unmap();
                self.timestamp_pending = false;
                return;
            }
        };
        let values: &[u64] = bytemuck::cast_slice(&mapped);
        if values.len() >= TIMESTAMP_COUNT as usize && values[0] != 0 {
            let period = self.timestamp_period;
            let ms = |begin: usize, end: usize| {
                values[end].wrapping_sub(values[begin]) as f32 * period / 1_000_000.0
            };
            let pair = |pass: u32| ms(pass as usize * 2, pass as usize * 2 + 1);
            self.gbuffer_ms = pair(TS_GBUFFER);
            self.lighting_ms = pair(TS_LIGHTING);
            self.temporal_ms = pair(TS_UPSCALE);
            self.shadow_ms = pair(TS_SHADOW);
            self.light_shadow_ms = pair(TS_LIGHT_SHADOW);
            self.gi_ms = pair(TS_GI);
            self.fog_ms = pair(TS_FOG);
            self.fxaa_ms = pair(TS_FXAA);
            self.particles_ms = pair(TS_PARTICLES) + pair(TS_PARTICLES_COMPOSITE);
            self.debris_cull_ms = pair(TS_DEBRIS_CULL);
            self.ddgi_inject_ms = pair(TS_DDGI_INJECT);
            self.ddgi_update_ms = pair(TS_DDGI_UPDATE);
            self.ddgi_blend_ms = pair(TS_DDGI_BLEND);
            // Bloom spans prefilter + two blur passes; the intermediate pairs
            // are written too, so take the first begin and the last end.
            self.bloom_ms = ms(
                TS_BLOOM_PREFILTER as usize * 2,
                TS_BLOOM_BLUR_V as usize * 2 + 1,
            );
            self.total_ms = ms(0, TS_POST as usize * 2 + 1);
            self.timestamps_valid = true;
        }
        drop(mapped);
        buffer.unmap();
        self.timestamp_pending = false;
    }
}

impl Targets {
    fn new(device: &wgpu::Device, width: u32, height: u32, scale: f32, shadow_full: bool) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let internal = internal_dimensions(width, height, scale);
        let bloom_size = (
            (width / BLOOM_DOWNSCALE).max(1),
            (height / BLOOM_DOWNSCALE).max(1),
        );
        let make = |label: &str, size: (u32, u32), format: wgpu::TextureFormat| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    // The half-res HDR targets double as render attachments:
                    // `gi_view` receives the smoke billboards after lighting.
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };

        let (albedo, albedo_view) =
            make("Shatterbox gbuffer albedo", internal, GBUFFER_ALBEDO_FORMAT);
        let (normal, normal_view) =
            make("Shatterbox gbuffer normal", internal, GBUFFER_NORMAL_FORMAT);
        let (material, material_view) = make(
            "Shatterbox gbuffer material",
            internal,
            GBUFFER_MATERIAL_FORMAT,
        );
        let (depth, depth_view) = {
            // `COPY_SRC` lets the autotest harness read the centre depth texel
            // for `assert_hit_matches_render`; it costs nothing at runtime.
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Shatterbox gbuffer depth"),
                size: wgpu::Extent3d {
                    width: internal.0,
                    height: internal.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        // The HDR target is written by the lighting compute pass *and* by the
        // smoke/dust billboards, so it needs both a storage and a render
        // attachment usage. `make` only allocates the storage/texture pair.
        let (hdr, hdr_view) = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Shatterbox hdr"),
                size: wgpu::Extent3d {
                    width: internal.0,
                    height: internal.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (hdr_aa, hdr_aa_view) = make("Shatterbox hdr fxaa", internal, HDR_FORMAT);
        let (bloom_a, bloom_a_view) = make("Shatterbox bloom a", bloom_size, HDR_FORMAT);
        let (bloom_b, bloom_b_view) = make("Shatterbox bloom b", bloom_size, HDR_FORMAT);
        // Half-resolution GI: AO is a high-frequency signal, indirect light is
        // not, so both live comfortably at half internal resolution.
        let gi_size = ((internal.0 / 2).max(1), (internal.1 / 2).max(1));
        let (gi, gi_view) = make("Shatterbox gi", gi_size, HDR_FORMAT);
        let (fog, fog_view) = make("Shatterbox fog", gi_size, HDR_FORMAT);
        let shadow_size = if shadow_full { internal } else { gi_size };
        let (shadow, shadow_view) = make(
            "Shatterbox shadow",
            shadow_size,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        // One bit per light, so a single u32 texel covers all 12 slots.
        let (light_shadow, light_shadow_view) = make(
            "Shatterbox light shadow",
            gi_size,
            wgpu::TextureFormat::R32Uint,
        );
        let (reflection, reflection_view) = make("Shatterbox reflection", gi_size, HDR_FORMAT);
        let smoke = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shatterbox smoke density"),
            size: wgpu::Extent3d {
                width: SMOKE_DIMS.0,
                height: SMOKE_DIMS.1,
                depth_or_array_layers: SMOKE_DIMS.2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let smoke_view = smoke.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let probe = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shatterbox irradiance probe volume"),
            size: wgpu::Extent3d {
                width: PROBE_DIMS.0,
                height: PROBE_DIMS.1,
                depth_or_array_layers: PROBE_DIMS.2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let probe_view = probe.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        // DDGI albedo volume + ray cloud. The volume is 1 m across the arena so a
        // 2 m brick holds eight cells: enough to keep a painted wall's colour
        // separate from the wall behind it.
        let ddgi_albedo = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shatterbox ddgi albedo volume"),
            size: wgpu::Extent3d {
                width: DDGI_ALBEDO_DIMS.0,
                height: DDGI_ALBEDO_DIMS.1,
                depth_or_array_layers: DDGI_ALBEDO_DIMS.2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ddgi_albedo_view = ddgi_albedo.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let (ddgi_trace, ddgi_trace_view) =
            make("Shatterbox ddgi ray cloud", DDGI_ATLAS_SIZE, HDR_FORMAT);
        // DDGI atlases are resolution independent: they only depend on the probe
        // grid, so reallocating them with the screen targets is harmless (the
        // update cursor is reset and the first cycle runs without hysteresis).
        let (ddgi_irradiance_0, ddgi_irradiance_view_0) =
            make("Shatterbox ddgi irradiance 0", DDGI_ATLAS_SIZE, HDR_FORMAT);
        let (ddgi_irradiance_1, ddgi_irradiance_view_1) =
            make("Shatterbox ddgi irradiance 1", DDGI_ATLAS_SIZE, HDR_FORMAT);
        let (ddgi_moments_0, ddgi_moments_view_0) =
            make("Shatterbox ddgi moments 0", DDGI_ATLAS_SIZE, HDR_FORMAT);
        let (ddgi_moments_1, ddgi_moments_view_1) =
            make("Shatterbox ddgi moments 1", DDGI_ATLAS_SIZE, HDR_FORMAT);
        let (history_color_0, history_color_view_0) =
            make("Shatterbox history color 0", (width, height), HDR_FORMAT);
        let (history_color_1, history_color_view_1) =
            make("Shatterbox history color 1", (width, height), HDR_FORMAT);
        let (history_depth_0, history_depth_view_0) =
            make("Shatterbox history depth 0", (width, height), DEPTH_FORMAT);
        let (history_depth_1, history_depth_view_1) =
            make("Shatterbox history depth 1", (width, height), DEPTH_FORMAT);

        Self {
            albedo,
            albedo_view,
            normal,
            normal_view,
            material,
            material_view,
            depth,
            depth_view,
            hdr,
            hdr_view,
            hdr_aa,
            hdr_aa_view,
            bloom_a,
            bloom_a_view,
            bloom_b,
            bloom_b_view,
            gi,
            gi_view,
            fog,
            fog_view,
            shadow,
            shadow_view,
            light_shadow,
            light_shadow_view,
            reflection,
            reflection_view,
            smoke,
            smoke_view,
            probe,
            probe_view,
            ddgi_albedo,
            ddgi_albedo_view,
            ddgi_trace,
            ddgi_trace_view,
            ddgi_irradiance: [ddgi_irradiance_0, ddgi_irradiance_1],
            ddgi_irradiance_views: [ddgi_irradiance_view_0, ddgi_irradiance_view_1],
            ddgi_moments: [ddgi_moments_0, ddgi_moments_1],
            ddgi_moments_views: [ddgi_moments_view_0, ddgi_moments_view_1],
            history_color: [history_color_0, history_color_1],
            history_color_views: [history_color_view_0, history_color_view_1],
            history_depth: [history_depth_0, history_depth_1],
            history_depth_views: [history_depth_view_0, history_depth_view_1],
            internal_size: internal,
            output_size: (width, height),
            bloom_size,
            gi_size,
            probe_size: PROBE_DIMS,
            shadow_size,
            shadow_full,
        }
    }
}

impl BindGroups {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        layouts: &Layouts,
        uniform_buffer: &wgpu::Buffer,
        exposure_state: &wgpu::Buffer,
        exposure_value: &wgpu::Buffer,
        ddgi_state: &wgpu::Buffer,
        world_view: &wgpu::TextureView,
        group_view: &wgpu::TextureView,
        region_view: &wgpu::TextureView,
        brick_buffer: &wgpu::Buffer,
        mask_buffer: &wgpu::Buffer,
        palette_buffer: &wgpu::Buffer,
        debris_buffer: &wgpu::Buffer,
        debris_voxel_buffer: &wgpu::Buffer,
        debris_sphere_buffer: &wgpu::Buffer,
        debris_rect_buffer: &wgpu::Buffer,
        debris_tile_buffer: &wgpu::Buffer,
        targets: &Targets,
        sampler: &wgpu::Sampler,
    ) -> Self {
        let gbuffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox gbuffer bind group"),
            layout: &layouts.gbuffer,
            entries: &[
                entry(0, uniform_buffer.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(world_view)),
                entry(2, brick_buffer.as_entire_binding()),
                entry(3, mask_buffer.as_entire_binding()),
                entry(4, palette_buffer.as_entire_binding()),
                entry(5, wgpu::BindingResource::TextureView(group_view)),
                entry(6, wgpu::BindingResource::TextureView(&targets.albedo_view)),
                entry(7, wgpu::BindingResource::TextureView(&targets.normal_view)),
                entry(
                    8,
                    wgpu::BindingResource::TextureView(&targets.material_view),
                ),
                entry(9, wgpu::BindingResource::TextureView(&targets.depth_view)),
                entry(10, wgpu::BindingResource::TextureView(region_view)),
                entry(11, debris_buffer.as_entire_binding()),
                entry(12, debris_voxel_buffer.as_entire_binding()),
                entry(13, debris_sphere_buffer.as_entire_binding()),
                entry(16, debris_tile_buffer.as_entire_binding()),
            ],
        });
        let lighting = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox lighting bind group"),
            layout: &layouts.lighting,
            entries: &[
                entry(0, uniform_buffer.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(&targets.albedo_view)),
                entry(2, wgpu::BindingResource::TextureView(&targets.normal_view)),
                entry(
                    3,
                    wgpu::BindingResource::TextureView(&targets.material_view),
                ),
                entry(4, wgpu::BindingResource::TextureView(&targets.depth_view)),
                entry(5, wgpu::BindingResource::TextureView(&targets.hdr_view)),
                entry(6, wgpu::BindingResource::TextureView(&targets.gi_view)),
                entry(7, wgpu::BindingResource::TextureView(&targets.fog_view)),
                entry(8, wgpu::BindingResource::TextureView(&targets.shadow_view)),
                entry(9, wgpu::BindingResource::Sampler(sampler)),
                entry(
                    10,
                    wgpu::BindingResource::TextureView(&targets.light_shadow_view),
                ),
                entry(
                    11,
                    wgpu::BindingResource::TextureView(&targets.reflection_view),
                ),
            ],
        });
        let shadow = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox shadow bind group"),
            layout: &layouts.shadow,
            entries: &[
                entry(0, uniform_buffer.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(world_view)),
                entry(2, mask_buffer.as_entire_binding()),
                entry(3, wgpu::BindingResource::TextureView(group_view)),
                entry(4, wgpu::BindingResource::TextureView(region_view)),
                entry(5, wgpu::BindingResource::TextureView(&targets.depth_view)),
                entry(6, wgpu::BindingResource::TextureView(&targets.normal_view)),
                entry(7, wgpu::BindingResource::TextureView(&targets.shadow_view)),
            ],
        });
        let light_shadow = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox light shadow bind group"),
            layout: &layouts.light_shadow,
            entries: &[
                entry(0, uniform_buffer.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(world_view)),
                entry(2, mask_buffer.as_entire_binding()),
                entry(3, wgpu::BindingResource::TextureView(group_view)),
                entry(4, wgpu::BindingResource::TextureView(region_view)),
                entry(5, wgpu::BindingResource::TextureView(&targets.depth_view)),
                entry(6, wgpu::BindingResource::TextureView(&targets.normal_view)),
                entry(
                    7,
                    wgpu::BindingResource::TextureView(&targets.light_shadow_view),
                ),
            ],
        });
        let make_gi = |write: usize, ddgi_atlas: usize| {
            let read = 1 - write;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox gi bind group"),
                layout: &layouts.gi,
                entries: &[
                    entry(0, uniform_buffer.as_entire_binding()),
                    entry(1, wgpu::BindingResource::TextureView(&targets.depth_view)),
                    entry(2, wgpu::BindingResource::TextureView(&targets.normal_view)),
                    // Previous frame's HDR result: one-bounce SSGI without a
                    // feedback loop in the lighting pass.
                    entry(
                        3,
                        wgpu::BindingResource::TextureView(&targets.history_color_views[read]),
                    ),
                    entry(4, wgpu::BindingResource::TextureView(&targets.probe_view)),
                    entry(5, wgpu::BindingResource::Sampler(sampler)),
                    entry(6, wgpu::BindingResource::TextureView(&targets.gi_view)),
                    entry(
                        7,
                        wgpu::BindingResource::TextureView(&targets.reflection_view),
                    ),
                    // The atlas written by this frame's blend. `gi.wgsl` runs
                    // after DDGI in the same encoder, so this is the freshest
                    // irradiance; the old fixed atlas 0 missed every odd z
                    // slice.
                    entry(
                        8,
                        wgpu::BindingResource::TextureView(
                            &targets.ddgi_irradiance_views[ddgi_atlas],
                        ),
                    ),
                    entry(
                        9,
                        wgpu::BindingResource::TextureView(&targets.ddgi_moments_views[ddgi_atlas]),
                    ),
                    entry(10, ddgi_state.as_entire_binding()),
                ],
            })
        };
        let probe = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox probe bind group"),
            layout: &layouts.probe,
            entries: &[
                entry(0, uniform_buffer.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(world_view)),
                entry(2, wgpu::BindingResource::TextureView(&targets.probe_view)),
            ],
        });
        let fog = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox fog bind group"),
            layout: &layouts.fog,
            entries: &[
                entry(0, uniform_buffer.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(&targets.depth_view)),
                entry(2, wgpu::BindingResource::TextureView(&targets.probe_view)),
                entry(3, wgpu::BindingResource::TextureView(&targets.fog_view)),
                entry(4, wgpu::BindingResource::TextureView(&targets.shadow_view)),
                entry(5, wgpu::BindingResource::TextureView(&targets.smoke_view)),
                entry(6, wgpu::BindingResource::Sampler(sampler)),
            ],
        });
        let bloom_prefilter = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox bloom prefilter bind group"),
            layout: &layouts.bloom,
            entries: &[
                entry(0, wgpu::BindingResource::TextureView(&targets.hdr_view)),
                entry(1, wgpu::BindingResource::TextureView(&targets.bloom_a_view)),
                entry(2, exposure_state.as_entire_binding()),
                entry(3, exposure_value.as_entire_binding()),
            ],
        });
        let bloom_blur_h = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox bloom blur h bind group"),
            layout: &layouts.bloom,
            entries: &[
                entry(0, wgpu::BindingResource::TextureView(&targets.bloom_a_view)),
                entry(1, wgpu::BindingResource::TextureView(&targets.bloom_b_view)),
                entry(2, exposure_state.as_entire_binding()),
                entry(3, exposure_value.as_entire_binding()),
            ],
        });
        let bloom_blur_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox bloom blur v bind group"),
            layout: &layouts.bloom,
            entries: &[
                entry(0, wgpu::BindingResource::TextureView(&targets.bloom_b_view)),
                entry(1, wgpu::BindingResource::TextureView(&targets.bloom_a_view)),
                entry(2, exposure_state.as_entire_binding()),
                entry(3, exposure_value.as_entire_binding()),
            ],
        });

        let make_temporal = |write: usize, source: &wgpu::TextureView| {
            let read = 1 - write;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox temporal bind group"),
                layout: &layouts.temporal,
                entries: &[
                    entry(0, uniform_buffer.as_entire_binding()),
                    entry(1, wgpu::BindingResource::TextureView(source)),
                    entry(2, wgpu::BindingResource::TextureView(&targets.depth_view)),
                    entry(
                        3,
                        wgpu::BindingResource::TextureView(&targets.history_color_views[read]),
                    ),
                    entry(
                        4,
                        wgpu::BindingResource::TextureView(&targets.history_depth_views[read]),
                    ),
                    entry(5, wgpu::BindingResource::Sampler(sampler)),
                    entry(
                        6,
                        wgpu::BindingResource::TextureView(&targets.history_color_views[write]),
                    ),
                    entry(
                        7,
                        wgpu::BindingResource::TextureView(&targets.history_depth_views[write]),
                    ),
                    entry(
                        8,
                        wgpu::BindingResource::TextureView(&targets.material_view),
                    ),
                ],
            })
        };
        // DDGI albedo injection: the G-buffer of this frame into the 1 m volume.
        let ddgi_inject = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox ddgi inject bind group"),
            layout: &layouts.ddgi_inject,
            entries: &[
                entry(0, uniform_buffer.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(&targets.depth_view)),
                entry(2, wgpu::BindingResource::TextureView(&targets.normal_view)),
                entry(3, wgpu::BindingResource::TextureView(&targets.albedo_view)),
                entry(
                    4,
                    wgpu::BindingResource::TextureView(&targets.material_view),
                ),
                entry(
                    5,
                    wgpu::BindingResource::TextureView(&targets.ddgi_albedo_view),
                ),
            ],
        });
        // DDGI trace: `[i]` reads atlas `1 - i` (previous bounce) and writes the
        // fresh ray cloud for the refreshed slice.
        let make_ddgi_trace = |write: usize| {
            let read = 1 - write;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox ddgi trace bind group"),
                layout: &layouts.ddgi_trace,
                entries: &[
                    entry(0, uniform_buffer.as_entire_binding()),
                    entry(1, wgpu::BindingResource::TextureView(world_view)),
                    entry(2, mask_buffer.as_entire_binding()),
                    entry(3, wgpu::BindingResource::TextureView(group_view)),
                    entry(4, wgpu::BindingResource::TextureView(region_view)),
                    entry(
                        5,
                        wgpu::BindingResource::TextureView(&targets.ddgi_albedo_view),
                    ),
                    entry(6, wgpu::BindingResource::Sampler(sampler)),
                    entry(
                        7,
                        wgpu::BindingResource::TextureView(&targets.ddgi_irradiance_views[read]),
                    ),
                    entry(
                        8,
                        wgpu::BindingResource::TextureView(&targets.ddgi_trace_view),
                    ),
                    entry(9, ddgi_state.as_entire_binding()),
                ],
            })
        };
        // DDGI blend ping-pong: group `n` writes atlas `n` and reads atlas `1 - n`.
        let make_ddgi = |write: usize| {
            let read = 1 - write;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox ddgi blend bind group"),
                layout: &layouts.ddgi,
                entries: &[
                    entry(0, ddgi_state.as_entire_binding()),
                    entry(
                        1,
                        wgpu::BindingResource::TextureView(&targets.ddgi_trace_view),
                    ),
                    entry(
                        2,
                        wgpu::BindingResource::TextureView(&targets.ddgi_irradiance_views[read]),
                    ),
                    entry(
                        3,
                        wgpu::BindingResource::TextureView(&targets.ddgi_moments_views[read]),
                    ),
                    entry(
                        4,
                        wgpu::BindingResource::TextureView(&targets.ddgi_irradiance_views[write]),
                    ),
                    entry(
                        5,
                        wgpu::BindingResource::TextureView(&targets.ddgi_moments_views[write]),
                    ),
                ],
            })
        };
        let make_post = |index: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox post bind group"),
                layout: &layouts.post,
                entries: &[
                    entry(0, uniform_buffer.as_entire_binding()),
                    entry(
                        1,
                        wgpu::BindingResource::TextureView(&targets.history_color_views[index]),
                    ),
                    entry(2, wgpu::BindingResource::TextureView(&targets.bloom_a_view)),
                    entry(3, exposure_value.as_entire_binding()),
                ],
            })
        };

        Self {
            gbuffer,
            shadow,
            light_shadow,
            lighting,
            gi: [
                [make_gi(0, 0), make_gi(0, 1)],
                [make_gi(1, 0), make_gi(1, 1)],
            ],
            probe,
            fog,
            ddgi_inject,
            ddgi_trace: [make_ddgi_trace(0), make_ddgi_trace(1)],
            ddgi: [make_ddgi(0), make_ddgi(1)],
            bloom_prefilter,
            bloom_blur_h,
            bloom_blur_v,
            temporal: [
                make_temporal(0, &targets.hdr_view),
                make_temporal(1, &targets.hdr_view),
            ],
            temporal_aa: [
                make_temporal(0, &targets.hdr_aa_view),
                make_temporal(1, &targets.hdr_aa_view),
            ],
            fxaa: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox fxaa bind group"),
                layout: &layouts.fxaa,
                entries: &[
                    entry(0, wgpu::BindingResource::TextureView(&targets.hdr_view)),
                    entry(1, wgpu::BindingResource::TextureView(&targets.hdr_aa_view)),
                ],
            }),
            particles: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox particles bind group"),
                layout: &layouts.particles,
                entries: &[
                    entry(0, uniform_buffer.as_entire_binding()),
                    entry(1, wgpu::BindingResource::TextureView(&targets.depth_view)),
                    entry(2, wgpu::BindingResource::TextureView(&targets.fog_view)),
                ],
            }),
            // The composite pass cannot share the billboard bind group: the
            // billboard pass writes `gi_view`, and one texture cannot be a
            // colour target and a sampled resource in the same pass scope.
            particles_composite: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox smoke composite bind group"),
                layout: &layouts.particles_composite,
                entries: &[
                    entry(0, uniform_buffer.as_entire_binding()),
                    entry(1, wgpu::BindingResource::TextureView(&targets.depth_view)),
                    entry(3, wgpu::BindingResource::TextureView(&targets.gi_view)),
                    entry(4, wgpu::BindingResource::Sampler(sampler)),
                ],
            }),
            debris_cull: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox debris cull bind group"),
                layout: &layouts.debris_cull,
                entries: &[
                    entry(0, uniform_buffer.as_entire_binding()),
                    entry(1, debris_rect_buffer.as_entire_binding()),
                    entry(2, debris_tile_buffer.as_entire_binding()),
                ],
            }),
            post: [make_post(0), make_post(1)],
        }
    }
}

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

/// Byte offset of brick `id` inside the packed brick buffer.
///
/// Brick ids are one-based (`0` means "empty" in the index texture), while the
/// buffer starts at word 0, so the shader reads `(id - 1) * BRICK_WORDS`. A
/// write at `id * BRICK_WORDS` lands one brick too far and the edited brick is
/// never updated: the hole then shows up on the neighbouring brick, which is
/// exactly the "crater is shifted ~2 m from the crosshair" bug.
fn brick_slot_offset(id: u32) -> u64 {
    u64::from(id.saturating_sub(1)) * GPU_WORDS_PER_BRICK as u64 * size_of::<u32>() as u64
}

fn mask_slot_offset(id: u32) -> u64 {
    u64::from(id.saturating_sub(1)) * MASK_WORDS_PER_BRICK as u64 * size_of::<u32>() as u64
}

fn slot_group(slot: usize) -> (usize, usize, usize) {
    let x = slot % BRICKS_X;
    let y = (slot / BRICKS_X) % BRICKS_Y;
    let z = slot / (BRICKS_X * BRICKS_Y);
    (x / 4, y / 4, z / 4)
}

fn slot_region(slot: usize) -> (usize, usize, usize) {
    let x = slot % BRICKS_X;
    let y = (slot / BRICKS_X) % BRICKS_Y;
    let z = slot / (BRICKS_X * BRICKS_Y);
    (x / 16, y / 16, z / 16)
}

/// Uploads only the bounding box around the changed occupancy texels. A shot
/// changes a handful of groups/regions near the impact, so this is a few KB
/// instead of the whole volume.
fn upload_occupancy_texels(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    dims: [u32; 3],
    data: &[u32],
    changed: &[usize],
) {
    if changed.is_empty() {
        return;
    }
    let (dx, dy) = (dims[0] as usize, dims[1] as usize);
    let coords: Vec<(usize, usize, usize)> = changed
        .iter()
        .map(|&index| (index % dx, (index / dx) % dy, index / (dx * dy)))
        .collect();
    let min_x = coords.iter().map(|c| c.0).min().unwrap_or(0);
    let max_x = coords.iter().map(|c| c.0).max().unwrap_or(0);
    let min_y = coords.iter().map(|c| c.1).min().unwrap_or(0);
    let max_y = coords.iter().map(|c| c.1).max().unwrap_or(0);
    let min_z = coords.iter().map(|c| c.2).min().unwrap_or(0);
    let max_z = coords.iter().map(|c| c.2).max().unwrap_or(0);
    let width = (max_x - min_x + 1) as u32;
    let height = (max_y - min_y + 1) as u32;
    let depth = (max_z - min_z + 1) as u32;
    let mut bytes = Vec::with_capacity((width * height * depth) as usize);
    for z in min_z..=max_z {
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                bytes.push(data[x + dx * (y + dy * z)]);
            }
        }
    }
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: min_x as u32,
                y: min_y as u32,
                z: min_z as u32,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&bytes),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
    );
}

fn create_layouts(device: &wgpu::Device) -> Layouts {
    // Every shader binds the same `FrameUniform`, so declaring the full size
    // makes wgpu reject a truncated/empty uniform up front instead of letting
    // the shader read garbage past the end of the buffer.
    let frame_size = NonZeroU64::new(size_of::<FrameUniform>() as u64)
        .expect("FrameUniform is never zero-sized");
    let uniform = |visibility| wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(frame_size),
        },
        count: None,
    };

    let storage_texture = |binding: u32, format: wgpu::TextureFormat| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    };
    let sampled = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    // A runtime-sized array's minimum binding size is the stride of one
    // element, not always 4: `array<GpuMaterial>` is 32 bytes and
    // `array<DebrisInstance>` is 176. Declaring a smaller value makes wgpu
    // reject the pipeline, which is exactly what happened to `gbuffer`
    // binding 4 in Э0.
    let read_buffer = |binding: u32, element_size: u64| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: Some(
                NonZeroU64::new(element_size).expect("storage elements are never zero-sized"),
            ),
        },
        count: None,
    };
    let storage_buffer =
        |binding: u32, visibility: wgpu::ShaderStages, read_only: bool, size: u64| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        NonZeroU64::new(size).expect("storage elements are never zero-sized"),
                    ),
                },
                count: None,
            }
        };
    let uint_3d = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    };
    // Uint textures have no filtering, so the shader does its own 2x2 gather.
    let uint_2d = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let sampled_3d = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    };
    let storage_texture_3d =
        |binding: u32, format: wgpu::TextureFormat| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        };
    let filtering_sampler = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };

    let gbuffer = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox gbuffer layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            uint_3d(1),
            read_buffer(2, size_of::<u32>() as u64),
            read_buffer(3, size_of::<u32>() as u64),
            read_buffer(4, size_of::<[f32; 8]>() as u64),
            uint_3d(5),
            storage_texture(6, GBUFFER_ALBEDO_FORMAT),
            storage_texture(7, GBUFFER_NORMAL_FORMAT),
            storage_texture(8, GBUFFER_MATERIAL_FORMAT),
            storage_texture(9, DEPTH_FORMAT),
            uint_3d(10),
            read_buffer(11, size_of::<GpuDebrisInstance>() as u64),
            read_buffer(12, size_of::<u32>() as u64),
            read_buffer(13, size_of::<[f32; 4]>() as u64),
            read_buffer(16, size_of::<u32>() as u64),
        ],
    });

    let lighting = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox lighting layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            sampled(1, false),
            sampled(2, false),
            sampled(3, false),
            sampled(4, false),
            storage_texture(5, HDR_FORMAT),
            sampled(6, false),
            sampled(7, false),
            // Half-resolution sun visibility, sampled bilinearly.
            sampled(8, true),
            filtering_sampler(9),
            // Half-resolution local-light bitmask, gathered manually.
            uint_2d(10),
            // Half-resolution screen-space reflection (rgb = reflected radiance).
            sampled(11, false),
        ],
    });

    // Half-resolution voxel sun shadows.
    let shadow = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox shadow layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            uint_3d(1),
            read_buffer(2, size_of::<u32>() as u64),
            uint_3d(3),
            uint_3d(4),
            sampled(5, false),
            sampled(6, false),
            storage_texture(7, wgpu::TextureFormat::Rgba8Unorm),
        ],
    });

    // Half-resolution local point/capsule/rect light shadows. Same inputs as
    // the sun pass, but the output is a u32 bitmask of visible lights.
    let light_shadow = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox light shadow layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            uint_3d(1),
            read_buffer(2, size_of::<u32>() as u64),
            uint_3d(3),
            uint_3d(4),
            sampled(5, false),
            sampled(6, false),
            storage_texture(7, wgpu::TextureFormat::R32Uint),
        ],
    });

    // Half-resolution SSAO + SSGI + cone trace + DDGI sampling.
    let gi = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox gi layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            sampled(1, false),
            sampled(2, false),
            sampled(3, true),
            sampled_3d(4, true),
            filtering_sampler(5),
            storage_texture(6, HDR_FORMAT),
            // Screen-space reflection written alongside the GI buffer
            // (rgb = radiance, a = screen-space hit confidence).
            storage_texture(7, HDR_FORMAT),
            // DDGI octahedral atlas: irradiance + depth moments.
            sampled(8, false),
            sampled(9, false),
            // `{ slice, hysteresis, spacing, reset }` shared with the DDGI passes.
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            },
        ],
    });

    // Voxel irradiance volume build.
    let probe = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox probe layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            uint_3d(1),
            storage_texture_3d(2, HDR_FORMAT),
        ],
    });

    // DDGI albedo injection: G-buffer -> 1 m colour volume.
    let ddgi_inject = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox ddgi inject layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            sampled(1, false),
            sampled(2, false),
            sampled(3, false),
            sampled(4, false),
            storage_texture_3d(5, HDR_FORMAT),
        ],
    });

    // DDGI trace: frame + voxel masks + colour volume + previous atlas.
    let ddgi_trace = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox ddgi trace layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            uint_3d(1),
            read_buffer(2, size_of::<u32>() as u64),
            uint_3d(3),
            uint_3d(4),
            sampled_3d(5, true),
            filtering_sampler(6),
            sampled(7, false),
            storage_texture(8, HDR_FORMAT),
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            },
        ],
    });

    // DDGI blend: ray cloud + ping-pong irradiance/moments + hysteresis params.
    // 0 params, 1 ray cloud, 2/3 previous atlas, 4/5 the new one.
    let ddgi = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox ddgi blend layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            },
            sampled(1, false),
            sampled(2, false),
            sampled(3, false),
            storage_texture(4, HDR_FORMAT),
            storage_texture(5, HDR_FORMAT),
        ],
    });

    // Half-resolution volumetric fog.
    let fog = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox fog layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            sampled(1, false),
            sampled_3d(2, false),
            storage_texture(3, HDR_FORMAT),
            // Voxel sun shadow for the screen-space god-ray march.
            sampled(4, true),
            // Dynamic smoke density volume (R8).
            sampled_3d(5, true),
            filtering_sampler(6),
        ],
    });

    let temporal = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox temporal layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            sampled(1, true),
            sampled(2, false),
            sampled(3, true),
            sampled(4, false),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            storage_texture(6, HDR_FORMAT),
            storage_texture(7, DEPTH_FORMAT),
            // Reactive mask lives in the fourth material channel; the upscaler
            // drops history where it is set so moving debris cannot smear.
            sampled(8, false),
        ],
    });

    let bloom = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox bloom layout"),
        entries: &[
            sampled(0, false),
            storage_texture(1, HDR_FORMAT),
            // Auto-exposure accumulation and resolve. The prefilter is also the
            // luminance meter, so the state lives on the bloom bind group.
            storage_buffer(
                2,
                wgpu::ShaderStages::COMPUTE,
                false,
                size_of::<ExposureStateGpu>() as u64,
            ),
            storage_buffer(
                3,
                wgpu::ShaderStages::COMPUTE,
                false,
                size_of::<ExposureValueGpu>() as u64,
            ),
        ],
    });

    let post = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox post layout"),
        entries: &[
            uniform(wgpu::ShaderStages::FRAGMENT),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage_buffer(
                3,
                wgpu::ShaderStages::FRAGMENT,
                true,
                size_of::<ExposureValueGpu>() as u64,
            ),
        ],
    });

    // FXAA reads the internal HDR image and writes the filtered copy; the
    // upscaler then samples whichever of the two the effect flag selects.
    let fxaa = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox fxaa layout"),
        entries: &[sampled(0, false), storage_texture(1, HDR_FORMAT)],
    });

    // Explosion smoke/dust: a render pass over camera-facing billboards. It
    // reads the frame uniform, the G-buffer depth (soft-particle fade) and the
    // half-res fog buffer (aerials sit *in* the fog), and writes into `hdr`.
    let particles = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox particles layout"),
        entries: &[
            uniform(wgpu::ShaderStages::VERTEX_FRAGMENT),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Half-res smoke buffer written by the billboard pass and read by
            // the depth-aware composite.
        ],
    });

    // The composite pass needs the cloud buffer and a sampler, but the billboard
    // pass must NOT bind the texture it renders into. Two layouts keep the
    // usages legal: the scene pass only sees frame/depth/fog, the composite only
    // frame/depth/smoke/sampler.
    let particles_composite = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox smoke composite layout"),
        entries: &[
            uniform(wgpu::ShaderStages::VERTEX_FRAGMENT),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // Per-tile debris masks: reads the frame uniform (count + internal size), the
    // screen rectangles and writes the mask words.
    let debris_cull = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Shatterbox debris cull layout"),
        entries: &[
            uniform(wgpu::ShaderStages::COMPUTE),
            read_buffer(1, size_of::<[f32; 4]>() as u64),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size_of::<u32>() as u64),
                },
                count: None,
            },
        ],
    });

    let compute_pipeline_layout = |label: &str, layout: &wgpu::BindGroupLayout| {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        })
    };

    let gbuffer_layout = compute_pipeline_layout("Shatterbox gbuffer pipeline layout", &gbuffer);
    let shadow_layout = compute_pipeline_layout("Shatterbox shadow pipeline layout", &shadow);
    let light_shadow_layout =
        compute_pipeline_layout("Shatterbox light shadow pipeline layout", &light_shadow);
    let lighting_layout = compute_pipeline_layout("Shatterbox lighting pipeline layout", &lighting);
    let gi_layout = compute_pipeline_layout("Shatterbox gi pipeline layout", &gi);
    let probe_layout = compute_pipeline_layout("Shatterbox probe pipeline layout", &probe);
    let fog_layout = compute_pipeline_layout("Shatterbox fog pipeline layout", &fog);
    let temporal_layout = compute_pipeline_layout("Shatterbox temporal pipeline layout", &temporal);
    let bloom_layout = compute_pipeline_layout("Shatterbox bloom pipeline layout", &bloom);
    let fxaa_layout = compute_pipeline_layout("Shatterbox fxaa pipeline layout", &fxaa);
    let particles_layout =
        compute_pipeline_layout("Shatterbox particles pipeline layout", &particles);
    let particles_composite_layout = compute_pipeline_layout(
        "Shatterbox smoke composite pipeline layout",
        &particles_composite,
    );
    let debris_cull_layout =
        compute_pipeline_layout("Shatterbox debris cull pipeline layout", &debris_cull);
    let ddgi_inject_layout =
        compute_pipeline_layout("Shatterbox ddgi inject pipeline layout", &ddgi_inject);
    let ddgi_trace_layout =
        compute_pipeline_layout("Shatterbox ddgi trace pipeline layout", &ddgi_trace);
    let ddgi_layout = compute_pipeline_layout("Shatterbox ddgi blend pipeline layout", &ddgi);
    let post_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Shatterbox post pipeline layout"),
        bind_group_layouts: &[Some(&post)],
        immediate_size: 0,
    });

    Layouts {
        gbuffer,
        shadow,
        light_shadow,
        lighting,
        gi,
        probe,
        fog,
        temporal,
        bloom,
        post,
        fxaa,
        particles,
        particles_composite,
        debris_cull,
        ddgi_inject,
        ddgi_trace,
        ddgi,
        gbuffer_layout,
        shadow_layout,
        light_shadow_layout,
        lighting_layout,
        gi_layout,
        probe_layout,
        fog_layout,
        temporal_layout,
        bloom_layout,
        fxaa_layout,
        particles_layout,
        particles_composite_layout,
        debris_cull_layout,
        ddgi_inject_layout,
        ddgi_trace_layout,
        ddgi_layout,
        post_layout,
    }
}

fn build_pipelines(
    device: &wgpu::Device,
    layouts: &Layouts,
    format: wgpu::TextureFormat,
    sources: &[String; SHADER_SLOTS],
) -> Result<Pipelines, String> {
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let module = |label: &str, source: &str| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        })
    };
    let gbuffer_module = module("Shatterbox gbuffer shader", &sources[0]);
    let lighting_module = module("Shatterbox lighting shader", &sources[1]);
    let temporal_module = module("Shatterbox temporal shader", &sources[2]);
    let bloom_module = module("Shatterbox bloom shader", &sources[3]);
    let post_module = module("Shatterbox post shader", &sources[4]);
    let gi_module = module("Shatterbox gi shader", &sources[5]);
    let probe_module = module("Shatterbox probe shader", &sources[6]);
    let fog_module = module("Shatterbox fog shader", &sources[7]);
    let shadow_module = module("Shatterbox shadow shader", &sources[8]);
    let light_shadow_module = module("Shatterbox light shadow shader", &sources[10]);
    let ddgi_module = module("Shatterbox ddgi trace shader", &sources[14]);
    let ddgi_inject_module = module("Shatterbox ddgi inject shader", &sources[15]);
    let ddgi_blend_module = module("Shatterbox ddgi blend shader", &sources[16]);

    let compute = |label: &str,
                   layout: &wgpu::PipelineLayout,
                   module: &wgpu::ShaderModule,
                   entry_point: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let gbuffer = compute(
        "Shatterbox gbuffer pipeline",
        &layouts.gbuffer_layout,
        &gbuffer_module,
        "cs_main",
    );
    let lighting = compute(
        "Shatterbox lighting pipeline",
        &layouts.lighting_layout,
        &lighting_module,
        "cs_main",
    );
    let shadow = compute(
        "Shatterbox shadow pipeline",
        &layouts.shadow_layout,
        &shadow_module,
        "cs_main",
    );
    let light_shadow = compute(
        "Shatterbox light shadow pipeline",
        &layouts.light_shadow_layout,
        &light_shadow_module,
        "cs_main",
    );
    let temporal = compute(
        "Shatterbox temporal pipeline",
        &layouts.temporal_layout,
        &temporal_module,
        "cs_main",
    );
    let fsr_module = module("Shatterbox fsr shader", &sources[9]);
    let fsr = compute(
        "Shatterbox fsr pipeline",
        &layouts.temporal_layout,
        &fsr_module,
        "cs_main",
    );
    let gi = compute(
        "Shatterbox gi pipeline",
        &layouts.gi_layout,
        &gi_module,
        "cs_main",
    );
    let probe = compute(
        "Shatterbox probe pipeline",
        &layouts.probe_layout,
        &probe_module,
        "cs_main",
    );
    let ddgi = compute(
        "Shatterbox ddgi blend pipeline",
        &layouts.ddgi_layout,
        &ddgi_blend_module,
        "cs_main",
    );
    let ddgi_inject = compute(
        "Shatterbox ddgi inject pipeline",
        &layouts.ddgi_inject_layout,
        &ddgi_inject_module,
        "cs_main",
    );
    let ddgi_trace = compute(
        "Shatterbox ddgi trace pipeline",
        &layouts.ddgi_trace_layout,
        &ddgi_module,
        "cs_main",
    );
    let fog = compute(
        "Shatterbox fog pipeline",
        &layouts.fog_layout,
        &fog_module,
        "cs_main",
    );
    let bloom_prefilter = compute(
        "Shatterbox bloom prefilter",
        &layouts.bloom_layout,
        &bloom_module,
        "cs_prefilter",
    );
    let bloom_blur_h = compute(
        "Shatterbox bloom blur h",
        &layouts.bloom_layout,
        &bloom_module,
        "cs_blur_h",
    );
    let bloom_blur_v = compute(
        "Shatterbox bloom blur v",
        &layouts.bloom_layout,
        &bloom_module,
        "cs_blur_v",
    );
    let exposure = compute(
        "Shatterbox exposure resolve",
        &layouts.bloom_layout,
        &bloom_module,
        "cs_exposure",
    );

    let fxaa_module = module("Shatterbox fxaa shader", &sources[11]);
    let fxaa = compute(
        "Shatterbox fxaa pipeline",
        &layouts.fxaa_layout,
        &fxaa_module,
        "cs_main",
    );

    // Smoke/dust billboards: instanced quads into the HDR target. The blend is
    // premultiplied-alpha `One / OneMinusSrcAlpha`, matching the `color*alpha`
    // the fragment shader writes, so a puff darkens nothing behind it and the
    // FXAA/upscale pass sees a normal HDR image.
    let particles_module = module("Shatterbox particles shader", &sources[12]);
    let debris_cull_module = module("Shatterbox debris cull shader", &sources[13]);
    let particles = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Shatterbox particles pipeline"),
        layout: Some(&layouts.particles_layout),
        vertex: wgpu::VertexState {
            module: &particles_module,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<Particle>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 32,
                        shader_location: 3,
                    },
                ],
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &particles_module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    // Depth-aware fullscreen upsample of the half-res smoke buffer into the
    // internal HDR image. Same bind group and same premultiplied blend as the
    // billboard pass, so this is simply "draw the cloud texture over the scene".
    let particles_composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Shatterbox smoke composite pipeline"),
        layout: Some(&layouts.particles_composite_layout),
        vertex: wgpu::VertexState {
            module: &particles_module,
            entry_point: Some("vs_composite"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &particles_module,
            entry_point: Some("fs_composite"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    let debris_cull = compute(
        "Shatterbox debris cull pipeline",
        &layouts.debris_cull_layout,
        &debris_cull_module,
        "cs_main",
    );

    let post = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Shatterbox post pipeline"),
        layout: Some(&layouts.post_layout),
        vertex: wgpu::VertexState {
            module: &post_module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &post_module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    if let Some(error) = pollster::block_on(error_scope.pop()) {
        tracing::error!(%error, "shader pipeline validation failed");
        return Err(error.to_string());
    }
    Ok(Pipelines {
        gbuffer,
        shadow,
        light_shadow,
        lighting,
        gi,
        probe,
        ddgi_inject,
        ddgi_trace,
        ddgi,
        fog,
        temporal,
        fsr,
        bloom_prefilter,
        bloom_blur_h,
        bloom_blur_v,
        exposure,
        fxaa,
        particles,
        particles_composite,
        debris_cull,
        post,
    })
}

impl ShaderWatcher {
    fn new() -> Self {
        let enabled = std::env::var_os("SHATTERBOX_NO_HOT_RELOAD").is_none();
        let directory = shader_directory();
        let current = directory
            .as_ref()
            .map(|dir| read_shader_sources(dir))
            .unwrap_or_else(|| {
                tracing::info!("using embedded shader sources");
                std::array::from_fn(|slot| EMBEDDED_SHADERS[slot].to_owned())
            });
        Self {
            directory,
            last_check: Instant::now(),
            current,
            enabled,
        }
    }
}

fn read_shader_sources(directory: &Path) -> [String; SHADER_SLOTS] {
    std::array::from_fn(|slot| {
        let path = directory.join(SHADER_NAMES[slot]);
        fs::read_to_string(&path).unwrap_or_else(|error| {
            tracing::warn!(%error, path = %path.display(), "failed to read shader; using embedded");
            EMBEDDED_SHADERS[slot].to_owned()
        })
    })
}

fn shader_directory() -> Option<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(dir) = std::env::var_os("SHATTERBOX_SHADER_DIR") {
        dirs.push(PathBuf::from(dir));
    }
    dirs.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/shaders"));
    dirs.push(PathBuf::from("assets/shaders"));
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dirs.push(parent.join("shaders"));
    }
    for dir in dirs {
        if SHADER_NAMES.iter().all(|name| dir.join(name).is_file()) {
            tracing::info!(directory = %dir.display(), "shaders loaded from disk");
            return Some(dir);
        }
    }
    None
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Approximate blackbody chromaticity (Tanner Helland's fit, 1000–40000 K),
/// normalised so the brightest channel is 1. Multiplying by a scalar radiance
/// then reproduces a plausible sun/moon tint at any slider temperature.
fn blackbody_rgb(kelvin: f32) -> Vec3 {
    let t = (kelvin.clamp(1000.0, 40000.0)) / 100.0;
    let r = if t <= 66.0 {
        255.0
    } else {
        329.698_73 * (t - 60.0).powf(-0.133_204_76)
    };
    let g = if t <= 66.0 {
        99.470_8 * t.ln() - 161.119_57
    } else {
        288.122_17 * (t - 60.0).powf(-0.075_514_85)
    };
    let b = if t >= 66.0 {
        255.0
    } else if t <= 19.0 {
        0.0
    } else {
        138.517_73 * (t - 10.0).ln() - 305.044_8
    };
    let color = Vec3::new(r, g, b).clamp(Vec3::ZERO, Vec3::splat(255.0)) / 255.0;
    // Keep the overall light level independent of the tint: a red 2000 K sun
    // should not also be a dimmer sun.
    let peak = color.max_element().max(1e-3);
    color / peak
}

fn internal_dimensions(width: u32, height: u32, scale: f32) -> (u32, u32) {
    let scale = scale.clamp(0.35, 1.0);
    (
        ((width as f32 * scale).round() as u32).max(1),
        ((height as f32 * scale).round() as u32).max(1),
    )
}

/// Low-discrepancy sample point. Kept for the future TAA/TAAU work; the
/// renderer no longer jitters the projection, so nothing calls it in-graph.
#[allow(dead_code)]
fn halton(mut index: u32, base: u32) -> f32 {
    let mut fraction = 1.0;
    let mut result = 0.0;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// `SHATTERBOX_DEBUG_TILES=1` makes `upload_debris_tiles` log how many pieces
/// survive the screen-rect test and how much of the screen they cover, once per
/// second. Diagnostic only: it never changes what is rendered.
fn debug_tiles() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static TICKS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let enabled = *ENABLED
        .get_or_init(|| matches!(std::env::var("SHATTERBOX_DEBUG_TILES").as_deref(), Ok("1")));
    if !enabled {
        return false;
    }
    TICKS.fetch_add(1, Ordering::Relaxed).is_multiple_of(60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shatterbox_voxel::VOXEL_SIZE_M;

    #[test]
    fn ddgi_quality_presets_are_stable() {
        assert_eq!(DdgiQuality::parse("off"), Some(DdgiQuality::Off));
        assert_eq!(DdgiQuality::parse("LOW"), Some(DdgiQuality::Low));
        assert_eq!(DdgiQuality::parse("medium"), Some(DdgiQuality::Medium));
        assert_eq!(DdgiQuality::parse("high"), Some(DdgiQuality::High));
        assert_eq!(DdgiQuality::parse("ultra"), None);
        assert_eq!(DdgiQuality::Low.tile(), 4);
        assert_eq!(DdgiQuality::Medium.tile(), 8);
        assert_eq!(DdgiQuality::Low.samples(), 1);
        assert_eq!(DdgiQuality::High.samples(), 2);
        assert_eq!(DdgiQuality::High.slices(), 2);
        assert_eq!(DdgiQuality::Medium.slices(), 1);
    }

    #[test]
    fn frame_uniform_is_16_byte_aligned() {
        assert_eq!(size_of::<FrameUniform>(), 992);
        assert_eq!(size_of::<FrameUniform>() % 16, 0);
        assert_eq!(VOXEL_SIZE_M, 0.125);
    }

    /// The GPU brick ids are one-based while the storage buffer starts at zero;
    /// an off-by-one write leaves the edited brick stale and the crater appears
    /// one brick (2 m) to the side. This pins the two conventions together.
    #[test]
    fn brick_slot_offsets_are_one_based() {
        assert_eq!(brick_slot_offset(0), 0);
        assert_eq!(brick_slot_offset(1), 0);
        assert_eq!(
            brick_slot_offset(2),
            GPU_WORDS_PER_BRICK as u64 * size_of::<u32>() as u64
        );
        assert_eq!(mask_slot_offset(0), 0);
        assert_eq!(mask_slot_offset(1), 0);
        assert_eq!(
            mask_slot_offset(2),
            MASK_WORDS_PER_BRICK as u64 * size_of::<u32>() as u64
        );
    }

    /// Size of one WGSL `Frame` field in bytes. Every field in the shared header
    /// is either a 4x4 matrix or a vec4, so the offsets are plain sums.
    fn wgsl_field_size(ty: &str) -> usize {
        let ty = ty.trim().trim_end_matches(',');
        if ty == "mat4x4<f32>" {
            return 64;
        }
        if ty == "vec4<f32>" {
            return 16;
        }
        if let Some(rest) = ty.strip_prefix("array<vec4<f32>,") {
            let count: usize = rest
                .trim()
                .trim_end_matches('>')
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("unsupported array length in {ty}"));
            return count * 16;
        }
        panic!("unsupported Frame field type {ty}");
    }

    /// Parses `struct Frame { ... }` and returns `(field, byte offset, size)`.
    fn wgsl_frame_fields(source: &str) -> Vec<(String, usize, usize)> {
        const HEADER: &str = "struct Frame {";
        let start = source
            .find(HEADER)
            .unwrap_or_else(|| panic!("no {HEADER} in shader"))
            + HEADER.len();
        let body = &source[start..];
        let end = body.find("\n}").expect("unterminated Frame struct");
        let mut offset = 0usize;
        let mut fields = Vec::new();
        for line in body[..end].lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            let (name, ty) = line
                .split_once(':')
                .unwrap_or_else(|| panic!("bad Frame field {line}"));
            let size = wgsl_field_size(ty);
            offset = offset.next_multiple_of(16);
            fields.push((name.trim().to_owned(), offset, size));
            offset += size;
        }
        fields
    }

    fn rust_frame_offset(name: &str) -> usize {
        match name {
            "inv_view_proj" => std::mem::offset_of!(FrameUniform, inv_view_proj),
            "view_proj" => std::mem::offset_of!(FrameUniform, view_proj),
            "prev_view_proj" => std::mem::offset_of!(FrameUniform, prev_view_proj),
            "camera_pos" => std::mem::offset_of!(FrameUniform, camera_pos),
            "prev_camera_pos" => std::mem::offset_of!(FrameUniform, prev_camera_pos),
            "sun_dir" => std::mem::offset_of!(FrameUniform, sun_dir),
            "sun_color" => std::mem::offset_of!(FrameUniform, sun_color),
            "world_min" => std::mem::offset_of!(FrameUniform, world_min),
            "world_max" => std::mem::offset_of!(FrameUniform, world_max),
            "params" => std::mem::offset_of!(FrameUniform, params),
            "params2" => std::mem::offset_of!(FrameUniform, params2),
            "params3" => std::mem::offset_of!(FrameUniform, params3),
            "internal_size" => std::mem::offset_of!(FrameUniform, internal_size),
            "light_pos_radius" => std::mem::offset_of!(FrameUniform, light_pos_radius),
            "light_color" => std::mem::offset_of!(FrameUniform, light_color),
            "light_axis" => std::mem::offset_of!(FrameUniform, light_axis),
            "debris_min" => std::mem::offset_of!(FrameUniform, debris_min),
            "debris_max" => std::mem::offset_of!(FrameUniform, debris_max),
            "debug_hit" => std::mem::offset_of!(FrameUniform, debug_hit),
            "params4" => std::mem::offset_of!(FrameUniform, params4),
            other => panic!("WGSL field `{other}` is missing from FrameUniform"),
        }
    }

    /// The `light_axis` bug this guards against: it sat after the debris AABB in
    /// Rust but right after `light_color` in WGSL, so capsule/rect lights were
    /// read as point lights. Every shader that binds the uniform must declare
    /// the same fields, in the same order, at the same offsets. A shader may
    /// stop early (only `light_shadow` uses everything up to `light_axis`), but
    /// it may not skip or reorder a field.
    #[test]
    fn frame_uniform_matches_wgsl_layout() {
        const RUST_FIELDS: [&str; 20] = [
            "inv_view_proj",
            "view_proj",
            "prev_view_proj",
            "camera_pos",
            "prev_camera_pos",
            "sun_dir",
            "sun_color",
            "world_min",
            "world_max",
            "params",
            "params2",
            "params3",
            "internal_size",
            "light_pos_radius",
            "light_color",
            "light_axis",
            "debris_min",
            "debris_max",
            "debug_hit",
            "params4",
        ];
        // `bloom.wgsl` and `fxaa.wgsl` have no Frame; the other eleven do.
        let mut checked = 0;
        for (slot, name) in SHADER_NAMES.iter().enumerate() {
            let source = EMBEDDED_SHADERS[slot];
            if !source.contains("struct Frame {") {
                continue;
            }
            checked += 1;
            let fields = wgsl_frame_fields(source);
            assert!(!fields.is_empty(), "{name}: empty Frame");
            let mut shader_size = 0;
            for (index, (field, offset, size)) in fields.iter().enumerate() {
                assert_eq!(
                    field, RUST_FIELDS[index],
                    "{name}: field {index} is `{field}`, Rust has `{}`",
                    RUST_FIELDS[index]
                );
                assert_eq!(
                    *offset,
                    rust_frame_offset(field),
                    "{name}: `{field}` is at byte {offset}, Rust has it at {}",
                    rust_frame_offset(field)
                );
                shader_size = offset + size;
            }
            assert!(
                shader_size <= size_of::<FrameUniform>(),
                "{name}: Frame is {shader_size} bytes, uniform is {}",
                size_of::<FrameUniform>()
            );
        }
        // `bloom.wgsl`, `ddgi_blend.wgsl` and `fxaa.wgsl` are the slots with no
        // `Frame` struct (they only see their own local buffers).
        assert_eq!(checked, SHADER_SLOTS - 3, "unexpected Frame shader count");
    }

    #[test]
    fn center_pixel_matches_camera_forward() {
        // The crosshair sits at NDC (0, 0) and the fire ray uses
        // `camera.forward()`. Both must be the same direction, otherwise a
        // crater appears offset from the reticle.
        let camera = FreeCamera::new(Vec3::new(128.0, 6.0, 180.0), 0.37, -0.21);
        let uniform = FrameUniform::new(
            &camera,
            16.0 / 9.0,
            Mat4::IDENTITY,
            camera.position,
            0.0,
            1,
            [0.0, 0.0],
            false,
            (1600, 900),
            1.0,
            0,
            &[],
            &[],
            None,
            None,
            10.5,
            14.0,
            5800.0,
            1.0,
            RenderEffects::default(),
            0.012,
            5.0,
            ShadowMode::HalfRes,
            true,
        );
        let inverse = Mat4::from_cols_array_2d(&uniform.inv_view_proj);
        let near = inverse.project_point3(Vec3::new(0.0, 0.0, 0.0));
        let far = inverse.project_point3(Vec3::new(0.0, 0.0, 1.0));
        let ray = (far - near).normalize();
        assert!(
            (ray - camera.forward()).length() < 1.0e-3,
            "center ray {ray:?} != forward {:?}",
            camera.forward()
        );
    }

    #[test]
    fn internal_dimensions_respect_scale_bounds() {
        assert_eq!(internal_dimensions(1600, 900, 0.5), (800, 450));
        assert_eq!(internal_dimensions(1600, 900, 0.1), (560, 315));
        assert_eq!(internal_dimensions(1600, 900, 2.0), (1600, 900));
    }

    #[test]
    fn halton_is_in_unit_range() {
        for index in 1..64 {
            let x = halton(index, 2);
            let y = halton(index, 3);
            assert!((0.0..1.0).contains(&x), "x = {x}");
            assert!((0.0..1.0).contains(&y), "y = {y}");
        }
    }

    /// Layout elements referenced as `array<T>` in WGSL. A runtime-sized
    /// array's `min_binding_size` must be at least one element, so these sizes
    /// are also what `create_layouts` declares.
    #[test]
    fn storage_element_sizes_match_wgsl() {
        assert_eq!(size_of::<[f32; 8]>(), 32, "GpuMaterial in WGSL is 32 bytes");
        assert_eq!(
            size_of::<GpuDebrisInstance>(),
            176,
            "DebrisInstance in WGSL is 176 bytes"
        );
    }

    /// Builds every real pipeline on a headless device. This catches binding
    /// layout drift (e.g. a `min_binding_size` smaller than the WGSL array
    /// element) without launching the game. Skipped when no adapter exists.
    #[test]
    fn pipelines_build_on_a_headless_device() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })) {
                Ok(adapter) => adapter,
                Err(error) => {
                    eprintln!("skipping pipeline smoke test: no adapter ({error})");
                    return;
                }
            };
        let (device, _queue) =
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("Shatterbox pipeline smoke device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })) {
                Ok(pair) => pair,
                Err(error) => {
                    eprintln!("skipping pipeline smoke test: no device ({error})");
                    return;
                }
            };

        let layouts = create_layouts(&device);
        let sources = std::array::from_fn(|slot| EMBEDDED_SHADERS[slot].to_owned());
        if let Err(error) = build_pipelines(
            &device,
            &layouts,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            &sources,
        ) {
            panic!("pipeline validation failed: {error}");
        }
    }

    #[test]
    fn embedded_shaders_parse() {
        for (slot, name) in SHADER_NAMES.iter().enumerate() {
            let source = EMBEDDED_SHADERS[slot];
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name} failed to parse: {error:?}"));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|error| panic!("{name} failed validation: {error:?}"));
        }
    }
}
