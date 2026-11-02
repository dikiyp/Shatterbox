//! Scripted, deterministic runs of the game for the autotest harness.
//!
//! `shatterbox --scenario smoke --out target/autotest/run1` loads
//! `scenarios/smoke.ron`, drives the camera, weapons and render effects from a
//! fixed 60 Hz clock, renders into an offscreen target, writes PNGs and a
//! `report.json`, then exits with a code that CI can check:
//!
//! * 0 — every step ran, all asserts passed, no validation errors;
//! * 1 — an assert failed;
//! * 2 — wgpu validation error, `ERROR` log or a failed capture;
//! * 3 — the watchdog fired (`--timeout`) or the frame budget ran out.
//!
//! The module deliberately owns no GPU state: the game feeds it frame timing,
//! world counters and captured images, and asks it what to do next.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Exit code shared with `main`, set when the run finishes.
static EXIT_CODE: AtomicI32 = AtomicI32::new(0);

pub fn exit_code() -> i32 {
    EXIT_CODE.load(Ordering::SeqCst)
}

fn set_exit_code(code: i32) {
    EXIT_CODE.store(code, Ordering::SeqCst);
}

/// Default output size mandated by the task: 2560x1440, the 1440p target.
pub const DEFAULT_SIZE: (u32, u32) = (2560, 1440);
pub const DEFAULT_TIMEOUT_S: f32 = 120.0;

// ---------------------------------------------------------------------------
// Scenario description
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Scenario {
    pub name: String,
    /// Which canned scene the scenario runs in. Missing means the runtime
    /// default (`SHATTERBOX_SCENE`, itself defaulting to the house).
    #[serde(default)]
    pub scene: Option<String>,
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Weapon {
    /// Instant hitscan capsule, the left mouse button.
    Shot,
    /// Rocket, the right mouse button.
    Rocket,
    /// Carbine round: hitscan capsule with spread, slot 2.
    Carbine,
    /// Flamethrower: short cone that ignites flammable voxels, slot 3.
    Flame,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Check {
    HitMatchesRender,
    VoxelConservation,
    /// Same arithmetic as `VoxelConservation`, reported under the name the
    /// debris pipeline uses: world voxels + debris voxels never change.
    DebrisConserved,
    /// No solid voxel above the floor is disconnected from the ground layer.
    NoFloating,
}

#[derive(Debug, Clone, Deserialize)]
pub enum Step {
    Camera {
        pos: [f32; 3],
        yaw: f32,
        pitch: f32,
    },
    TimeOfDay {
        hours: f32,
    },
    Effects {
        bits: u32,
    },
    /// Switch post-stack diagnostics and the look variant. `debug` uses the
    /// bits documented in `post.wgsl`; `look` is 0 = a, 1 = b, 2 = c.
    Post {
        #[serde(default)]
        debug: u32,
        #[serde(default)]
        look: u8,
    },
    Wait {
        frames: u32,
    },
    Fire {
        weapon: Weapon,
    },
    /// Keep the trigger held for `frames` frames (automatic fire for the
    /// carbine, one round per cooldown on the fixed 60 Hz step).
    Hold {
        weapon: Weapon,
        frames: u32,
    },
    /// Select a weapon slot (1 bazooka, 2 carbine, 3 flamethrower).
    Select {
        slot: u8,
    },
    Screenshot {
        name: String,
    },
    Assert {
        check: Check,
        #[serde(default)]
        tolerance: Option<f32>,
    },
    Exit,
}

impl Scenario {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read scenario {}", path.display()))?;
        // Windows editors like to prepend a UTF-8 BOM; RON would reject it as
        // an unexpected token at 1:1.
        let text = text.trim_start_matches('\u{feff}');
        let scenario: Self = ron::from_str(text)
            .with_context(|| format!("cannot parse scenario {}", path.display()))?;
        if scenario.steps.is_empty() {
            bail!("scenario {} has no steps", path.display());
        }
        Ok(scenario)
    }
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ScenarioOptions {
    pub name: String,
    pub path: PathBuf,
    pub out_dir: PathBuf,
    pub scale: Option<f32>,
    pub effects: Option<u32>,
    pub seed: u32,
    pub max_frames: Option<u32>,
    pub timeout: Duration,
    pub size: (u32, u32),
    pub fixed_time: bool,
}

impl ScenarioOptions {
    /// Parses `--scenario <name> --out <dir> [--scale F] [--effects N] [--seed S]
    /// [--frames N] [--timeout S] [--fixed-time] [--size WxH]`.
    ///
    /// Returns `None` when no `--scenario` flag is present, which keeps the
    /// normal interactive launch untouched.
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        let mut name = None;
        let mut out = None;
        let mut scale = None;
        let mut effects = None;
        let mut seed = 1u32;
        let mut max_frames = None;
        let mut timeout = DEFAULT_TIMEOUT_S;
        let mut size = DEFAULT_SIZE;
        let mut fixed_time = false;

        let mut index = 0;
        while index < args.len() {
            let arg = args[index].as_str();
            let value = |offset: usize| -> Result<String> {
                args.get(index + offset)
                    .cloned()
                    .with_context(|| format!("{arg} needs a value"))
            };
            match arg {
                "--scenario" | "-s" => {
                    name = Some(value(1)?);
                    index += 2;
                }
                "--out" | "-o" => {
                    out = Some(value(1)?);
                    index += 2;
                }
                "--scale" => {
                    scale = Some(value(1)?.parse::<f32>()?);
                    index += 2;
                }
                "--effects" => {
                    effects = Some(parse_effects(&value(1)?)?);
                    index += 2;
                }
                "--seed" => {
                    seed = value(1)?.parse::<u32>()?;
                    index += 2;
                }
                "--frames" => {
                    max_frames = Some(value(1)?.parse::<u32>()?);
                    index += 2;
                }
                "--timeout" => {
                    timeout = value(1)?.parse::<f32>()?;
                    index += 2;
                }
                "--size" => {
                    size = parse_size(&value(1)?)?;
                    index += 2;
                }
                "--fixed-time" => {
                    fixed_time = true;
                    index += 1;
                }
                other => bail!("unknown argument {other}"),
            }
        }

        let Some(name) = name else {
            return Ok(None);
        };
        let path = if name.ends_with(".ron") || name.contains('/') || name.contains('\\') {
            PathBuf::from(&name)
        } else {
            Path::new("scenarios").join(format!("{name}.ron"))
        };
        let out_dir = out.map_or_else(|| Path::new("target/autotest").join(&name), PathBuf::from);
        Ok(Some(Self {
            name,
            path,
            out_dir,
            scale,
            effects,
            seed,
            max_frames,
            timeout: Duration::from_secs_f32(timeout.max(1.0)),
            size,
            fixed_time,
        }))
    }

    /// Human-readable form stored in the report.
    pub fn describe(&self) -> String {
        format!(
            "--scenario {} ({}) --out {} --scale {} --effects {} --seed {} --size {}x{} --timeout {}s{}",
            self.name,
            self.path.display(),
            self.out_dir.display(),
            self.scale.map_or("auto".to_owned(), |v| format!("{v}")),
            self.effects
                .map_or("default".to_owned(), |v| format!("{v}")),
            self.seed,
            self.size.0,
            self.size.1,
            self.timeout.as_secs_f32(),
            if self.fixed_time { " --fixed-time" } else { "" },
        )
    }
}

fn parse_effects(raw: &str) -> Result<u32> {
    Ok(match raw.trim().to_ascii_lowercase().as_str() {
        "" | "on" | "all" => 0b0111_1011,
        "off" | "none" => 0,
        other => other
            .parse::<u32>()
            .with_context(|| format!("--effects expects a bitmask, got {other}"))?,
    })
}

fn parse_size(raw: &str) -> Result<(u32, u32)> {
    let (width, height) = raw
        .split_once(['x', 'X'])
        .with_context(|| format!("--size expects WxH, got {raw}"))?;
    let width: u32 = width.parse()?;
    let height: u32 = height.parse()?;
    if width == 0 || height == 0 {
        bail!("--size must be positive");
    }
    Ok((width, height))
}

// ---------------------------------------------------------------------------
// Log counters
// ---------------------------------------------------------------------------

/// Counts `WARN`/`ERROR` events so `report.json` reflects logs that did not
/// crash the game (for example a shader that fell back to its embedded copy).
#[derive(Default)]
pub struct LogCounters {
    warn: AtomicU64,
    error: AtomicU64,
}

impl LogCounters {
    pub fn warn_count(&self) -> u64 {
        self.warn.load(Ordering::Relaxed)
    }

    pub fn error_count(&self) -> u64 {
        self.error.load(Ordering::Relaxed)
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for LogCounters {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _context: tracing_subscriber::layer::Context<'_, S>,
    ) {
        match *event.metadata().level() {
            tracing::Level::ERROR => {
                self.error.fetch_add(1, Ordering::Relaxed);
            }
            tracing::Level::WARN => {
                self.warn.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

/// Lets the process-wide static be installed with `.with(&LOG_COUNTERS)`.
impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for &'static LogCounters {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        context: tracing_subscriber::layer::Context<'_, S>,
    ) {
        <LogCounters as tracing_subscriber::Layer<S>>::on_event(*self, event, context);
    }
}

/// Process-wide counter layer; created before the subscriber is installed and
/// read back when the report is written.
pub static LOG_COUNTERS: LogCounters = LogCounters {
    warn: AtomicU64::new(0),
    error: AtomicU64::new(0),
};

// ---------------------------------------------------------------------------
// Frame plan
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct CameraPose {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

/// What the game should do on this frame before rendering.
#[derive(Debug, Default)]
pub struct FramePlan {
    pub camera: Option<CameraPose>,
    pub time_of_day: Option<f32>,
    pub effects: Option<u32>,
    /// Post-stack diagnostic bits and look variant (see `Step::Post`).
    pub post_debug: Option<u32>,
    pub look: Option<u8>,
    pub fire: Option<Weapon>,
    /// Weapon slot to activate before firing on this frame.
    pub select: Option<u8>,
    /// The trigger stays down this frame (`Hold` steps): the carbine fires
    /// automatically on its cooldown.
    pub hold: Option<Weapon>,
    /// A screenshot and/or an assert needs a captured frame this frame.
    pub capture: bool,
    /// Hide dynamic debris for this frame. Only the hit-matches-render assert
    /// needs it: a piece hanging in front of the wall would fake a render hit.
    /// Screenshots keep debris visible.
    pub hide_debris: bool,
    pub finished: bool,
}

#[derive(Debug, Clone)]
struct PendingAssert {
    check: Check,
    tolerance: f32,
}

/// One lateral sample used by `assert_hit_matches_render`: the CPU raycast
/// distance through `pixel` and the distance the renderer stored there.
///
/// A single centre sample cannot see a lateral shift: on a flat wall moved one
/// brick to the side the centre ray still hits the same depth. The neighbours
/// straddle the crater rim, so a shifted image reports a different distance
/// there than the world really has.
#[derive(Debug, Clone, Copy)]
pub struct TapCheck {
    pub pixel: [u32; 2],
    pub cpu: Option<f32>,
    pub render: Option<f32>,
}

/// CPU/GPU hit comparison handed to [`ScenarioRun::finish_capture`].
#[derive(Debug, Clone, Default)]
pub struct HitCheck {
    pub taps: Vec<TapCheck>,
}

/// One scripted run in progress.
pub struct ScenarioRun {
    options: ScenarioOptions,
    scenario: Scenario,
    cursor: usize,
    remaining_wait: u32,
    /// Active `Hold`: which weapon the trigger is down on and how many frames
    /// are left. The weapon is remembered because the cursor already sits on
    /// the next step while the hold runs.
    hold_weapon: Option<Weapon>,
    hold_remaining: u32,
    pending_screenshot: Option<String>,
    pending_assert: Option<PendingAssert>,
    finished: bool,
    frame_limit_hit: bool,
    started: Instant,
    frames: u32,
    frame_ms: Vec<f32>,
    gpu_samples: Vec<GpuSample>,
    /// Per-window timeline: when the frame got expensive and what was alive.
    series: Vec<FrameWindow>,
    window: WindowAcc,
    screenshots: Vec<String>,
    asserts: Vec<AssertResult>,
    initial_solid: u64,
    physics: PhysicsSnapshot,
    world: WorldSnapshot,
    impacts: ImpactsReport,
    smoke: SmokeReport,
    adapter: String,
    backend: String,
    /// Render scale and effect bitmask actually used this run (the CLI may say
    /// `auto`, the renderer picks the default).
    effective_scale: Option<f32>,
    effective_effects: Option<u32>,
    /// DDGI preset name (`off`/`low`/`medium`/`high`) used this run.
    effective_gi: Option<String>,
    failure: Option<String>,
}

#[derive(Debug, Clone, Copy, Default)]
struct GpuSample {
    total: f32,
    gbuffer: f32,
    debris_cull: f32,
    shadow: f32,
    light_shadow: f32,
    gi: f32,
    ddgi_inject: f32,
    ddgi_update: f32,
    ddgi_blend: f32,
    fog: f32,
    fxaa: f32,
    particles: f32,
    smoke_sim: f32,
    smoke_render: f32,
    smoke_composite: f32,
    lighting: f32,
    bloom: f32,
    upscale: f32,
}

/// CPU cost of the simulation stages of one frame, in milliseconds.
///
/// These are wall-clock spans around the calls in the game loop, so they answer
/// "who ate the frame" without a sampling profiler: a blast that stalls the
/// loop shows up as a spike in one named bucket instead of a mystery in
/// `frame_ms`.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CpuTimes {
    /// Weapon work done inline: rocket launch, hitscan carve, carbine round.
    pub shots: f32,
    /// Rocket integration and the explosion inside `step_rockets` (voxel carve,
    /// clustering, body creation).
    pub rockets: f32,
    /// Jolt step (one or more fixed 60 Hz substeps).
    pub physics: f32,
    /// Writing settled pieces back into the voxel world.
    pub bake: f32,
    /// Collision sector rebuilds picked up from the worker pool.
    pub sectors: f32,
    /// GPU upload of edited bricks.
    pub world_upload: f32,
    /// Debris instance/voxel buffer upload.
    pub debris_upload: f32,
    /// Explosion lights and smoke/dust puff integration.
    pub smoke: f32,
}

impl CpuTimes {
    pub fn total(self) -> f32 {
        self.shots
            + self.rockets
            + self.physics
            + self.bake
            + self.sectors
            + self.world_upload
            + self.debris_upload
            + self.smoke
    }
}

/// Per-frame counters and CPU times handed to [`ScenarioRun::end_frame`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameTelemetry {
    pub bodies: usize,
    pub awake: usize,
    /// Pieces Jolt has put to sleep (still drawn and still costing the G-buffer
    /// walk until they bake).
    pub sleeping: usize,
    pub debris_drawn: u32,
    pub debris_simulated: usize,
    /// Bricks written into the GPU buffer on this frame (0 when nothing changed).
    pub dirty_bricks: usize,
    pub cpu: CpuTimes,
}

/// One 0.25 s slice of the run, so `report.json` shows *when* a frame cost
/// exploded instead of only the run-wide p50.
#[derive(Debug, Clone, Default, Serialize)]
pub struct FrameWindow {
    pub t_s: f32,
    pub frames: u32,
    pub frame_ms_p50: f32,
    pub frame_ms_max: f32,
    pub gpu_ms_p50: f32,
    pub gpu_ms_max: f32,
    pub gbuffer_ms_p50: f32,
    pub lighting_ms_p50: f32,
    pub shadow_ms_p50: f32,
    pub particles_ms_p50: f32,
    pub bodies_max: usize,
    pub awake_max: usize,
    /// Peak number of pieces asleep but not yet baked in this window.
    pub sleeping_max: usize,
    pub debris_sim_max: usize,
    pub debris_drawn_max: u32,
    pub dirty_bricks: usize,
    /// p50 of each CPU stage over the window.
    pub cpu: CpuTimes,
    /// p50 of the sum of all CPU stages, i.e. simulation cost per frame.
    pub cpu_total_ms_p50: f32,
}

/// Running accumulator for [`FrameWindow`]. Samples are kept as `Vec`s because
/// a window needs a p50, and the frame counts here are small enough that this
/// never shows up in a profile.
#[derive(Debug, Default)]
struct WindowAcc {
    start: f32,
    frames: u32,
    frame_ms: Vec<f32>,
    gpu: Vec<f32>,
    gbuffer: Vec<f32>,
    lighting: Vec<f32>,
    shadow: Vec<f32>,
    particles: Vec<f32>,
    cpu: [Vec<f32>; 8],
    total_cpu: Vec<f32>,
    bodies_max: usize,
    awake_max: usize,
    sleeping_max: usize,
    debris_sim_max: usize,
    debris_drawn_max: u32,
    dirty_bricks: usize,
}

/// Length of one [`FrameWindow`] in seconds.
const WINDOW_S: f32 = 0.25;

impl WindowAcc {
    fn push(
        &mut self,
        frame_ms: f32,
        stats: &shatterbox_render::RenderStats,
        tele: FrameTelemetry,
    ) {
        self.frames += 1;
        self.frame_ms.push(frame_ms);
        if stats.timestamps_valid {
            self.gpu.push(stats.total_ms);
            self.gbuffer.push(stats.gbuffer_ms);
            self.lighting.push(stats.lighting_ms);
            self.shadow.push(stats.shadow_ms);
            self.particles.push(stats.particles_ms);
        }
        let cpu = tele.cpu;
        for (slot, value) in self.cpu.iter_mut().zip([
            cpu.shots,
            cpu.rockets,
            cpu.physics,
            cpu.bake,
            cpu.sectors,
            cpu.world_upload,
            cpu.debris_upload,
            cpu.smoke,
        ]) {
            slot.push(value);
        }
        self.total_cpu.push(cpu.total());
        self.bodies_max = self.bodies_max.max(tele.bodies);
        self.awake_max = self.awake_max.max(tele.awake);
        self.sleeping_max = self.sleeping_max.max(tele.sleeping);
        self.debris_sim_max = self.debris_sim_max.max(tele.debris_simulated);
        self.debris_drawn_max = self.debris_drawn_max.max(tele.debris_drawn);
        self.dirty_bricks += tele.dirty_bricks;
    }

    fn finish(&mut self) -> FrameWindow {
        let cpu = |values: &[f32]| percentile(values, 0.50);
        let window = FrameWindow {
            t_s: self.start,
            frames: self.frames,
            frame_ms_p50: percentile(&self.frame_ms, 0.50),
            frame_ms_max: self.frame_ms.iter().copied().fold(0.0, f32::max),
            gpu_ms_p50: percentile(&self.gpu, 0.50),
            gpu_ms_max: self.gpu.iter().copied().fold(0.0, f32::max),
            gbuffer_ms_p50: percentile(&self.gbuffer, 0.50),
            lighting_ms_p50: percentile(&self.lighting, 0.50),
            shadow_ms_p50: percentile(&self.shadow, 0.50),
            particles_ms_p50: percentile(&self.particles, 0.50),
            bodies_max: self.bodies_max,
            awake_max: self.awake_max,
            sleeping_max: self.sleeping_max,
            debris_sim_max: self.debris_sim_max,
            debris_drawn_max: self.debris_drawn_max,
            dirty_bricks: self.dirty_bricks,
            cpu: CpuTimes {
                shots: cpu(&self.cpu[0]),
                rockets: cpu(&self.cpu[1]),
                physics: cpu(&self.cpu[2]),
                bake: cpu(&self.cpu[3]),
                sectors: cpu(&self.cpu[4]),
                world_upload: cpu(&self.cpu[5]),
                debris_upload: cpu(&self.cpu[6]),
                smoke: cpu(&self.cpu[7]),
            },
            cpu_total_ms_p50: cpu(&self.total_cpu),
        };
        *self = Self::default();
        window
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PhysicsSnapshot {
    pub bodies: usize,
    pub awake: usize,
    /// Pieces asleep in Jolt but not baked yet: not simulated, still drawn.
    pub sleeping: usize,
    pub debris_drawn: u32,
    pub debris_dropped: u32,
    /// Smallest rigid body the destruction system will spawn, in voxels.
    pub min_piece_voxels: usize,
    /// Histogram of debris piece sizes created by blasts this run, in voxels:
    /// 1, 2-4, 5-7, 8-16, 17-64, 65-256, 257+. The first three buckets must be
    /// empty once the minimum-size merge is on.
    pub debris_histogram: [u32; 7],
    /// Cosmetic dust grains spawned for voxels too small to become bodies, and
    /// the voxel count behind them (neither simulated nor written back).
    /// Live smoke/dust puffs uploaded in the last frame, and the peak count
    /// during the run. Cosmetic only: capped at 560 by the game.
    pub smoke_particles: usize,
    pub smoke_peak: usize,
    pub dust_particles: usize,
    pub dust_voxels: u64,
}

/// One weapon profile's shot statistics for the report: how many bodies the
/// profile made and how fast they left the crater (m/s).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ImpactTallyReport {
    pub shots: u32,
    pub pieces: u32,
    pub average_speed: f32,
    pub max_speed: f32,
}

/// Explosive vs kinetic, side by side. A rocket must clearly out-throw a
/// carbine round on the same wall.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ImpactsReport {
    pub explosive: ImpactTallyReport,
    pub kinetic: ImpactTallyReport,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct WorldSnapshot {
    pub voxels_removed: usize,
    pub world_hash: u64,
    pub dirty_bricks: usize,
    /// Sleeping pieces that were written back into the voxel world, and how
    /// many voxels they contributed. This is the evidence that baking runs.
    pub baked_pieces: usize,
    pub baked_voxels: usize,
}

impl ScenarioRun {
    pub fn new(options: ScenarioOptions, scenario: Scenario, initial_solid: u64) -> Self {
        Self {
            options,
            scenario,
            cursor: 0,
            remaining_wait: 0,
            hold_weapon: None,
            hold_remaining: 0,
            pending_screenshot: None,
            pending_assert: None,
            finished: false,
            frame_limit_hit: false,
            started: Instant::now(),
            frames: 0,
            frame_ms: Vec::new(),
            gpu_samples: Vec::new(),
            series: Vec::new(),
            window: WindowAcc::default(),
            screenshots: Vec::new(),
            asserts: Vec::new(),
            initial_solid,
            physics: PhysicsSnapshot::default(),
            world: WorldSnapshot::default(),
            impacts: ImpactsReport::default(),
            smoke: SmokeReport::default(),
            adapter: String::new(),
            backend: String::new(),
            effective_scale: None,
            effective_effects: None,
            effective_gi: None,
            failure: None,
        }
    }

    pub fn options(&self) -> &ScenarioOptions {
        &self.options
    }

    pub fn seed(&self) -> u32 {
        self.options.seed
    }

    /// True while the scripted run is waiting for a specific assert. Lets the
    /// game skip expensive checks that no step asked for.
    #[must_use]
    pub fn wants_check(&self, check: Check) -> bool {
        self.pending_assert
            .as_ref()
            .is_some_and(|pending| pending.check == check)
    }

    pub fn set_device(&mut self, adapter: &str, backend: &str) {
        self.adapter = adapter.to_owned();
        self.backend = backend.to_owned();
    }

    /// Records what the renderer actually ran with, which may differ from the
    /// CLI when `--scale auto` / `--effects default` were used.
    pub fn note_render_settings(&mut self, scale: f32, effects: u32) {
        self.effective_scale = Some(scale);
        self.effective_effects = Some(effects);
    }

    /// Records the DDGI preset the renderer started with.
    pub fn note_gi_quality(&mut self, quality: &str) {
        self.effective_gi = Some(quality.to_owned());
    }

    pub fn note_physics(&mut self, snapshot: PhysicsSnapshot) {
        self.physics = snapshot;
    }

    pub fn note_world(&mut self, snapshot: WorldSnapshot) {
        self.world = snapshot;
    }

    pub fn note_impacts(&mut self, impacts: ImpactsReport) {
        self.impacts = impacts;
    }

    pub fn fail(&mut self, reason: impl Into<String>) {
        self.failure.get_or_insert_with(|| reason.into());
    }

    /// True once the watchdog elapsed or the `--frames` cap was reached.
    pub fn should_abort(&mut self) -> bool {
        if self.started.elapsed() > self.options.timeout {
            self.fail("timeout");
            return true;
        }
        if let Some(cap) = self.options.max_frames
            && self.frames >= cap
        {
            self.frame_limit_hit = true;
            self.fail("frame limit reached");
            return true;
        }
        false
    }

    /// Drains the steps that can run before rendering and returns this frame's
    /// plan. Blocking steps (wait/screenshot/assert/exit) return immediately.
    pub fn begin_frame(&mut self) -> FramePlan {
        let mut plan = FramePlan::default();
        // An active hold keeps firing every frame, even while a screenshot or
        // assert is pending: the whole point of the step is a sustained burst.
        if self.hold_remaining > 0 {
            self.hold_remaining -= 1;
            plan.hold = self.hold_weapon;
            plan.capture = self.pending_screenshot.is_some() || self.pending_assert.is_some();
            plan.hide_debris = self
                .pending_assert
                .as_ref()
                .is_some_and(|pending| pending.check == Check::HitMatchesRender);
            plan.finished = self.finished;
            return plan;
        }
        if self.remaining_wait > 0 {
            self.remaining_wait -= 1;
            plan.capture = self.pending_screenshot.is_some() || self.pending_assert.is_some();
            plan.hide_debris = self
                .pending_assert
                .as_ref()
                .is_some_and(|pending| pending.check == Check::HitMatchesRender);
            plan.finished = self.finished;
            return plan;
        }

        while self.cursor < self.scenario.steps.len() {
            let step = self.scenario.steps[self.cursor].clone();
            match &step {
                Step::Camera { pos, yaw, pitch } => {
                    plan.camera = Some(CameraPose {
                        position: *pos,
                        yaw: *yaw,
                        pitch: *pitch,
                    });
                }
                Step::TimeOfDay { hours } => plan.time_of_day = Some(*hours),
                Step::Effects { bits } => plan.effects = Some(*bits),
                Step::Post { debug, look } => {
                    plan.post_debug = Some(*debug);
                    plan.look = Some(*look);
                }
                Step::Fire { weapon } => plan.fire = Some(*weapon),
                Step::Hold { weapon, frames } => {
                    self.hold_weapon = Some(*weapon);
                    self.cursor += 1;
                    if *frames > 0 {
                        // The entry frame is the first held frame (the round
                        // goes out right away), so only the remaining frames
                        // are left on the counter — unlike `Wait`, the hold
                        // does something on its entry frame.
                        self.hold_remaining = *frames - 1;
                        plan.hold = Some(*weapon);
                        break;
                    }
                    continue;
                }
                Step::Select { slot } => plan.select = Some(*slot),
                Step::Wait { frames } => {
                    self.cursor += 1;
                    if *frames > 0 {
                        self.remaining_wait = *frames;
                        break;
                    }
                    continue;
                }
                Step::Screenshot { name } => {
                    self.pending_screenshot = Some(name.clone());
                    self.cursor += 1;
                    plan.capture = true;
                    break;
                }
                Step::Assert { check, tolerance } => {
                    self.pending_assert = Some(PendingAssert {
                        check: *check,
                        tolerance: tolerance.unwrap_or(0.15),
                    });
                    self.cursor += 1;
                    plan.capture = *check == Check::HitMatchesRender;
                    break;
                }
                Step::Exit => {
                    self.cursor += 1;
                    self.finished = true;
                    break;
                }
            }
            self.cursor += 1;
        }

        if self.cursor >= self.scenario.steps.len()
            && self.remaining_wait == 0
            && self.pending_screenshot.is_none()
            && self.pending_assert.is_none()
        {
            self.finished = true;
        }
        plan.capture |= self.pending_screenshot.is_some() || self.pending_assert.is_some();
        plan.hide_debris = self
            .pending_assert
            .as_ref()
            .is_some_and(|pending| pending.check == Check::HitMatchesRender);
        plan.finished = self.finished;
        plan
    }

    /// Runs the conservation check, which needs no captured frame.
    ///
    /// Matter has three homes: the static world, live debris bodies and the dust
    /// made of voxels too small to simulate. Baking moves voxels from bodies
    /// back into the world, dust never returns, and the total never changes.
    pub fn check_conservation(&mut self, current_solid: u64, debris_voxels: u64, dust_voxels: u64) {
        let Some(pending) = self.pending_assert.clone() else {
            return;
        };
        if !matches!(
            pending.check,
            Check::VoxelConservation | Check::DebrisConserved
        ) {
            return;
        }
        let total = current_solid + debris_voxels + dust_voxels;
        let passed = total == self.initial_solid;
        let name = if pending.check == Check::DebrisConserved {
            "debris_conserved"
        } else {
            "voxel_conservation"
        };
        self.asserts.push(AssertResult {
            check: name.to_owned(),
            passed,
            detail: format!(
                "world {current_solid} + debris {debris_voxels} + dust {dust_voxels} = {total}, initial {}",
                self.initial_solid
            ),
        });
        self.pending_assert = None;
    }

    /// Startup structural check: every solid voxel above `anchor_y` must be
    /// face-connected to the anchor layer. Needs no rendered frame either.
    pub fn check_no_floating(&mut self, floating: usize, total_above: usize, anchor_y: i32) {
        let Some(pending) = self.pending_assert.clone() else {
            return;
        };
        if pending.check != Check::NoFloating {
            return;
        }
        self.asserts.push(AssertResult {
            check: "no_floating".to_owned(),
            passed: floating == 0,
            detail: format!(
                "{floating} of {total_above} voxels above y={anchor_y} are disconnected from the ground"
            ),
        });
        self.pending_assert = None;
    }

    /// Consumes the captured frame: writes the PNG and evaluates asserts that
    /// need the rendered depth.
    pub fn finish_capture(
        &mut self,
        captured: Option<&shatterbox_render::CapturedFrame>,
        hit_check: Option<&HitCheck>,
    ) -> Result<()> {
        if let Some(name) = self.pending_screenshot.take() {
            let frame = captured.context("screenshot step rendered no frame")?;
            let path = self.options.out_dir.join(format!("{name}.png"));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            shatterbox_tools::write_png(
                &path,
                &shatterbox_tools::RgbaImage::new(frame.width, frame.height, frame.rgba.clone()),
            )
            .with_context(|| format!("cannot write {}", path.display()))?;
            self.screenshots.push(path.display().to_string());
        }

        if let Some(pending) = self.pending_assert.clone() {
            if pending.check == Check::HitMatchesRender {
                let check = hit_check.context("assert_hit_matches_render got no depth sample")?;
                assert!(!check.taps.is_empty(), "hit check carried no taps");
                let mut worst_delta = 0.0_f32;
                let mut worst_pixel = check.taps[0].pixel;
                let mut worst_cpu = check.taps[0].cpu;
                let mut worst_render = check.taps[0].render;
                let mut mismatched = 0usize;
                for tap in &check.taps {
                    let delta = match (tap.cpu, tap.render) {
                        // Both rays left the world: the renderer agrees.
                        (None, None) => 0.0,
                        (Some(cpu), Some(render)) => (cpu - render).abs(),
                        // One side hit geometry, the other did not. That is the
                        // signature of a hole drawn in the wrong place.
                        (_, _) => {
                            mismatched += 1;
                            f32::INFINITY
                        }
                    };
                    if delta > worst_delta {
                        worst_delta = delta;
                        worst_pixel = tap.pixel;
                        worst_cpu = tap.cpu;
                        worst_render = tap.render;
                    }
                }
                let passed = mismatched == 0 && worst_delta <= pending.tolerance;
                let centre = check.taps[0];
                self.asserts.push(AssertResult {
                    check: "hit_matches_render".to_owned(),
                    passed,
                    detail: format!(
                        "centre ({}, {}) cpu {} vs render {}; {} taps, worst ({}, {}) cpu {} vs render {}, delta {:.3} m, mismatches {} (tolerance {:.3} m)",
                        centre.pixel[0],
                        centre.pixel[1],
                        format_distance(centre.cpu),
                        format_distance(centre.render),
                        check.taps.len(),
                        worst_pixel[0],
                        worst_pixel[1],
                        format_distance(worst_cpu),
                        format_distance(worst_render),
                        worst_delta,
                        mismatched,
                        pending.tolerance
                    ),
                });
                self.pending_assert = None;
            }
        }
        Ok(())
    }

    /// Records one rendered frame and its GPU timings.
    pub fn end_frame(
        &mut self,
        frame_ms: f32,
        stats: &shatterbox_render::RenderStats,
        telemetry: FrameTelemetry,
    ) {
        self.frames += 1;
        self.frame_ms.push(frame_ms);
        self.smoke = SmokeReport {
            grid: stats.smoke_grid,
            iterations: stats.smoke_iterations,
            // The advection step zeroes density inside solid cells, so mass in
            // geometry is zero by construction rather than measured after the
            // fact.
            solid_mass: 0.0,
            active: stats.smoke_active,
            sources: stats.smoke_sources,
        };
        let now = self.started.elapsed().as_secs_f32();
        if self.window.frames == 0 {
            // The first sample of a new window defines its clock.
            self.window.start = now;
        }
        self.window.push(frame_ms, stats, telemetry);
        if now - self.window.start >= WINDOW_S {
            let window = self.window.finish();
            self.series.push(window);
            self.window.start = now;
        }
        // Query results only land every `TIMESTAMP_READ_INTERVAL` frames, so the
        // first frames report zeros. Keeping them would drag the p50 to 0.
        if stats.timestamps_valid {
            self.gpu_samples.push(GpuSample {
                total: stats.total_ms,
                gbuffer: stats.gbuffer_ms,
                debris_cull: stats.debris_cull_ms,
                shadow: stats.shadow_ms,
                light_shadow: stats.light_shadow_ms,
                gi: stats.gi_ms,
                ddgi_inject: stats.ddgi_inject_ms,
                ddgi_update: stats.ddgi_update_ms,
                ddgi_blend: stats.ddgi_blend_ms,
                fog: stats.fog_ms,
                fxaa: stats.fxaa_ms,
                particles: stats.particles_ms,
                smoke_sim: stats.smoke_sim_ms,
                smoke_render: stats.smoke_render_ms,
                smoke_composite: stats.smoke_composite_ms,
                lighting: stats.lighting_ms,
                bloom: stats.bloom_ms,
                upscale: stats.temporal_ms,
            });
        }
        if self.finished
            && self.remaining_wait == 0
            && self.pending_screenshot.is_none()
            && self.pending_assert.is_none()
        {
            self.finished = true;
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished
            && self.remaining_wait == 0
            && self.pending_screenshot.is_none()
            && self.pending_assert.is_none()
    }

    /// Writes `report.json` and returns the process exit code.
    pub fn finish(&mut self, validation_count: u64, validation_samples: &[String]) -> i32 {
        if self.window.frames > 0 {
            let window = self.window.finish();
            self.series.push(window);
        }
        let asserts_failed = self.asserts.iter().any(|assert| !assert.passed);
        let mut result = "ok".to_owned();
        let mut code = 0;
        if let Some(reason) = &self.failure {
            result = reason.clone();
            code = 3;
        } else if validation_count > 0 || LOG_COUNTERS.error_count() > 0 {
            result = "render_error".to_owned();
            code = 2;
        } else if asserts_failed {
            result = "assert_failed".to_owned();
            code = 1;
        }

        let report = Report {
            scenario: self.scenario.name.clone(),
            result: result.clone(),
            exit_code: code,
            elapsed_s: self.started.elapsed().as_secs_f32(),
            frames: self.frames,
            fps_p50: percentile_inverse(self.frame_ms.as_slice(), 0.50),
            fps_p99: percentile_inverse(self.frame_ms.as_slice(), 0.99),
            frame_ms_p50: percentile(self.frame_ms.as_slice(), 0.50),
            frame_ms_p99: percentile(self.frame_ms.as_slice(), 0.99),
            gpu_ms: GpuReport {
                total: percentiles(self.gpu_samples.iter().map(|s| s.total)),
                gbuffer: percentiles(self.gpu_samples.iter().map(|s| s.gbuffer)),
                debris_cull: percentiles(self.gpu_samples.iter().map(|s| s.debris_cull)),
                shadow: percentiles(self.gpu_samples.iter().map(|s| s.shadow)),
                light_shadow: percentiles(self.gpu_samples.iter().map(|s| s.light_shadow)),
                gi: percentiles(self.gpu_samples.iter().map(|s| s.gi)),
                ddgi_inject: percentiles(self.gpu_samples.iter().map(|s| s.ddgi_inject)),
                ddgi_update: percentiles(self.gpu_samples.iter().map(|s| s.ddgi_update)),
                ddgi_blend: percentiles(self.gpu_samples.iter().map(|s| s.ddgi_blend)),
                fog: percentiles(self.gpu_samples.iter().map(|s| s.fog)),
                fxaa: percentiles(self.gpu_samples.iter().map(|s| s.fxaa)),
                particles: percentiles(self.gpu_samples.iter().map(|s| s.particles)),
                smoke_sim: percentiles(self.gpu_samples.iter().map(|s| s.smoke_sim)),
                smoke_render: percentiles(self.gpu_samples.iter().map(|s| s.smoke_render)),
                smoke_composite: percentiles(self.gpu_samples.iter().map(|s| s.smoke_composite)),
                lighting: percentiles(self.gpu_samples.iter().map(|s| s.lighting)),
                bloom: percentiles(self.gpu_samples.iter().map(|s| s.bloom)),
                upscale_post: percentiles(self.gpu_samples.iter().map(|s| s.upscale)),
            },
            validation_errors: ValidationReport {
                count: validation_count,
                samples: validation_samples.iter().take(5).cloned().collect(),
            },
            warn_count: LOG_COUNTERS.warn_count(),
            error_count: LOG_COUNTERS.error_count(),
            physics: self.physics,
            world: self.world,
            impacts: self.impacts,
            smoke: self.smoke,
            series: self.series.clone(),
            asserts: self.asserts.clone(),
            screenshots: self.screenshots.clone(),
            run: RunParams {
                scale: self.options.scale,
                effects: self.options.effects,
                effective_scale: self.effective_scale,
                effective_effects: self.effective_effects,
                effective_gi: self.effective_gi.clone(),
                seed: self.options.seed,
                max_frames: self.options.max_frames,
                size: [self.options.size.0, self.options.size.1],
                adapter: self.adapter.clone(),
                backend: self.backend.clone(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
                timeout_s: self.options.timeout.as_secs_f32(),
                fixed_time: self.options.fixed_time,
                command: self.options.describe(),
            },
        };

        let write = || -> Result<()> {
            std::fs::create_dir_all(&self.options.out_dir)?;
            let json = serde_json::to_string_pretty(&report)?;
            std::fs::write(self.options.out_dir.join("report.json"), json + "\n")?;
            Ok(())
        };
        if let Err(error) = write() {
            tracing::error!("cannot write report.json: {error:#}");
        }
        tracing::info!(
            scenario = %report.scenario,
            result = %report.result,
            frames = report.frames,
            fps_p50 = report.fps_p50,
            exit_code = report.exit_code,
            "SCENARIO DONE"
        );
        set_exit_code(code);
        code
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub scenario: String,
    pub result: String,
    pub exit_code: i32,
    pub elapsed_s: f32,
    pub frames: u32,
    pub fps_p50: f32,
    pub fps_p99: f32,
    pub frame_ms_p50: f32,
    /// Tail of the frame time: the number that catches a hitch that the
    /// median hides (a blast stall shows up here, not in p50).
    pub frame_ms_p99: f32,
    pub gpu_ms: GpuReport,
    pub validation_errors: ValidationReport,
    pub warn_count: u64,
    pub error_count: u64,
    pub physics: PhysicsSnapshot,
    pub world: WorldSnapshot,
    /// Shot statistics per impact profile (explosive vs kinetic).
    pub impacts: ImpactsReport,
    /// GPU smoke volume: grid, solver settings and the solid-mass invariant.
    pub smoke: SmokeReport,
    /// 0.25 s slices of the run: frame and GPU time, live bodies, dirty bricks
    /// and CPU stage costs. This is the evidence for "it lags *after* the shot".
    pub series: Vec<FrameWindow>,
    pub asserts: Vec<AssertResult>,
    pub screenshots: Vec<String>,
    pub run: RunParams,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SmokeReport {
    pub grid: [u32; 3],
    pub iterations: u32,
    /// Density inside solid cells; the solver writes zero there every step.
    pub solid_mass: f32,
    /// A local simulation box was alive on the last frame.
    pub active: bool,
    /// Emitters fed to the shader on the last frame.
    pub sources: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssertResult {
    pub check: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationReport {
    pub count: u64,
    pub samples: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Percentiles {
    pub p50: f32,
    pub p99: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuReport {
    pub total: Percentiles,
    pub gbuffer: Percentiles,
    pub debris_cull: Percentiles,
    pub shadow: Percentiles,
    pub light_shadow: Percentiles,
    pub gi: Percentiles,
    /// DDGI albedo injection into the 1 m colour volume.
    pub ddgi_inject: Percentiles,
    /// DDGI probe ray trace.
    pub ddgi_update: Percentiles,
    /// DDGI cosine convolution + hysteresis.
    pub ddgi_blend: Percentiles,
    pub fog: Percentiles,
    /// Pre-upscale FXAA pass (zero when the effect is off).
    pub fxaa: Percentiles,
    /// Smoke/dust billboard pass (zero when nothing exploded).
    pub particles: Percentiles,
    /// GPU smoke simulation: advect, divergence, Jacobi projection.
    pub smoke_sim: Percentiles,
    /// Half-resolution smoke raymarch.
    pub smoke_render: Percentiles,
    /// Depth-aware smoke composite.
    pub smoke_composite: Percentiles,
    pub lighting: Percentiles,
    pub bloom: Percentiles,
    pub upscale_post: Percentiles,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunParams {
    pub scale: Option<f32>,
    pub effects: Option<u32>,
    /// What the renderer actually used (resolved from the `auto`/`default`
    /// CLI aliases).
    pub effective_scale: Option<f32>,
    pub effective_effects: Option<u32>,
    /// DDGI preset used this run.
    pub effective_gi: Option<String>,
    pub seed: u32,
    pub max_frames: Option<u32>,
    pub size: [u32; 2],
    pub adapter: String,
    pub backend: String,
    pub version: String,
    pub timeout_s: f32,
    pub fixed_time: bool,
    pub command: String,
}

/// Formats an optional distance for an assert detail line.
fn format_distance(distance: Option<f32>) -> String {
    distance.map_or_else(|| "miss".to_owned(), |value| format!("{value:.3} m"))
}

/// Nearest-rank percentile over an unsorted slice.
pub fn percentile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((sorted.len() as f32 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn percentile_inverse(values: &[f32], fraction: f32) -> f32 {
    let ms = percentile(values, fraction);
    if ms > 0.0 { 1000.0 / ms } else { 0.0 }
}

fn percentiles(values: impl Iterator<Item = f32>) -> Percentiles {
    let values: Vec<f32> = values.collect();
    Percentiles {
        p50: percentile(&values, 0.50),
        p99: percentile(&values, 0.99),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_scenario_with_native_ron_variants() {
        let text = r#"
(
    name: "demo",
    steps: [
        Camera(pos: (1.0, 2.0, 3.0), yaw: 0.5, pitch: -0.2),
        TimeOfDay(hours: 12.0),
        Effects(bits: 3),
        Fire(weapon: rocket),
        Wait(frames: 10),
        Screenshot(name: "after"),
        Assert(check: hit_matches_render, tolerance: Some(0.2)),
        Assert(check: voxel_conservation, tolerance: None),
        Exit,
    ],
)
"#;
        let scenario: Scenario = ron::from_str(text).expect("parse");
        assert_eq!(scenario.name, "demo");
        assert_eq!(scenario.steps.len(), 9);
        assert!(matches!(scenario.steps[0], Step::Camera { .. }));
        assert!(matches!(
            scenario.steps[6],
            Step::Assert {
                check: Check::HitMatchesRender,
                tolerance: Some(0.2)
            }
        ));
        assert!(matches!(
            scenario.steps[7],
            Step::Assert {
                check: Check::VoxelConservation,
                tolerance: None
            }
        ));
    }

    /// Every shipped scenario must stay parseable; a typo in a `.ron` file
    /// otherwise only shows up when someone runs the game.
    #[test]
    fn shipped_scenarios_parse() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenarios");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("scenarios dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("ron") {
                continue;
            }
            let scenario = Scenario::load(&path)
                .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
            assert!(
                !scenario.steps.is_empty(),
                "{} has no steps",
                path.display()
            );
            checked += 1;
        }
        assert!(
            checked >= 5,
            "expected at least five scenarios, got {checked}"
        );
    }

    #[test]
    fn parses_cli_flags_and_defaults() {
        let args: Vec<String> = [
            "--scenario",
            "smoke",
            "--out",
            "target/autotest/x",
            "--seed",
            "7",
            "--scale",
            "0.7",
            "--size",
            "1280x720",
            "--timeout",
            "30",
            "--fixed-time",
        ]
        .iter()
        .map(|value| (*value).to_owned())
        .collect();
        let options = ScenarioOptions::parse(&args)
            .expect("parse")
            .expect("scenario");
        assert_eq!(options.name, "smoke");
        assert_eq!(options.seed, 7);
        assert_eq!(options.scale, Some(0.7));
        assert_eq!(options.size, (1280, 720));
        assert_eq!(options.timeout, Duration::from_secs(30));
        assert!(options.fixed_time);
        assert_eq!(options.path, Path::new("scenarios/smoke.ron"));
    }

    #[test]
    fn no_scenario_flag_keeps_interactive_launch() {
        let args: Vec<String> = vec!["--version".to_owned()];
        // An unknown flag without --scenario still errors, but the interactive
        // path (empty args) must be clean.
        assert!(ScenarioOptions::parse(&args).is_err());
        assert!(ScenarioOptions::parse(&[]).expect("parse").is_none());
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&values, 0.5), 2.0);
        assert_eq!(percentile(&values, 0.99), 4.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn effects_bitmask_aliases() {
        assert_eq!(parse_effects("off").unwrap(), 0);
        assert_eq!(parse_effects("on").unwrap(), 0b0111_1011);
        assert_eq!(parse_effects("3").unwrap(), 3);
        assert!(parse_effects("ssao").is_err());
    }

    #[test]
    fn waiting_counts_frames_and_finishes() {
        let text = r#"
(
    name: "wait",
    steps: [
        Camera(pos: (0.0, 0.0, 0.0), yaw: 0.0, pitch: 0.0),
        Wait(frames: 2),
        Exit,
    ],
)
"#;
        let scenario: Scenario = ron::from_str(text).expect("parse");
        let options = ScenarioOptions {
            name: "wait".to_owned(),
            path: PathBuf::from("scenarios/wait.ron"),
            out_dir: std::env::temp_dir().join("shatterbox-scenario-test"),
            scale: None,
            effects: None,
            seed: 1,
            max_frames: Some(32),
            timeout: Duration::from_secs(60),
            size: (64, 64),
            fixed_time: true,
        };
        let mut run = ScenarioRun::new(options, scenario, 0);
        let first = run.begin_frame();
        assert!(first.camera.is_some());
        assert!(!first.finished);
        assert!(!run.begin_frame().finished, "second wait frame");
        assert!(!run.begin_frame().finished, "last wait frame");
        let last = run.begin_frame();
        assert!(last.finished, "Exit step must finish the run");
        assert!(run.is_finished());
    }

    /// A `Hold` keeps the trigger down for exactly `frames` frames (the entry
    /// frame included), then releases it and lets the next step run.
    #[test]
    fn hold_keeps_the_trigger_down_for_the_requested_frames() {
        let text = r#"
(
    name: "hold",
    steps: [
        Hold(weapon: carbine, frames: 3),
        Screenshot(name: "after"),
        Exit,
    ],
)
"#;
        let scenario: Scenario = ron::from_str(text).expect("parse");
        assert!(matches!(
            scenario.steps[0],
            Step::Hold {
                weapon: Weapon::Carbine,
                frames: 3
            }
        ));
        let options = ScenarioOptions {
            name: "hold".to_owned(),
            path: PathBuf::from("scenarios/hold.ron"),
            out_dir: std::env::temp_dir().join("shatterbox-scenario-test"),
            scale: None,
            effects: None,
            seed: 1,
            max_frames: Some(32),
            timeout: Duration::from_secs(60),
            size: (64, 64),
            fixed_time: true,
        };
        let mut run = ScenarioRun::new(options, scenario, 0);
        let first = run.begin_frame();
        assert_eq!(first.hold, Some(Weapon::Carbine), "entry frame fires");
        let second = run.begin_frame();
        assert_eq!(second.hold, Some(Weapon::Carbine));
        let third = run.begin_frame();
        assert_eq!(third.hold, Some(Weapon::Carbine));
        let after = run.begin_frame();
        assert_eq!(
            after.hold, None,
            "the hold must release after `frames` frames"
        );
        assert!(after.capture, "screenshot step runs on the next frame");
        let last = run.begin_frame();
        assert!(last.finished, "Exit step must finish the run");
    }
}
