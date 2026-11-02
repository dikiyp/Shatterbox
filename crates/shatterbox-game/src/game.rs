//! Phase 3 game shell: voxel arena, walk-mode player, destruction and overlay.

use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Mat4, Vec3};
use shatterbox_core::{Action, FrameStats, FreeCamera};
use shatterbox_destruction::sectors::SectorCollision;
use shatterbox_destruction::{DestructionSystem, ImpactProfile};
use shatterbox_gpu::GpuContext;
use shatterbox_physics::JoltPhysics;
use shatterbox_platform::{App, AppEvent, Input};
use shatterbox_render::{
    AreaLight, DdgiQuality, DebrisView, FrameOutcome, MAX_DEBRIS_VOXELS, MAX_RENDER_DEBRIS,
    Particle, PointLight, RenderEffects, RenderStats, Renderer, SMOKE_GRID, SMOKE_SPACING,
    ShadowMode, SmokeSource,
};
use shatterbox_voxel::scene::{
    GROUND_ANCHOR_VOXEL, SceneKind, fx_lab_lamp_positions, house_lamp_positions,
    test_room_lamp_positions,
};
use shatterbox_voxel::{Voxel, VoxelWorld, materials};
use winit::event::{DeviceEvent, WindowEvent};

/// Active local smoke box: fixed centre, lifetime after the last emitter and
/// the time spent waiting for destruction edits to settle before rebuilding
/// the obstacle mask.
struct SmokeBox {
    center: Vec3,
    ttl: f32,
    dirty_age: f32,
}

/// Zero density for the coarse 4 m fog field while the GPU volume is active, so
/// the same smoke is not counted twice.
static ZERO_DENSITY: [u8; 64 * 16 * 64] = [0; 64 * 16 * 64];
use winit::window::Window;

use crate::player::Player;
use crate::scenario::{HitCheck, ScenarioOptions, ScenarioRun, TapCheck, Weapon};
use crate::smoke::SmokeField;

const FIXED_DT: f32 = 1.0 / 60.0;
const FIRE_RADIUS_VOXELS: i32 = 10;
/// Rocket launcher: muzzle speed, gravity, blast radius and yield. The blast is
/// bigger than the hand tool because the projectile has to travel first.
const ROCKET_SPEED_MPS: f32 = 42.0;
const ROCKET_GRAVITY_MPS2: f32 = -9.0;
// 20 voxels = 2.5 m. The old 1.75 m crater read as a dent, not an explosion;
// the destruction clamp is now 32 so the 1.4x blast roll (28 voxels) is not
// truncated. The radius grows the removed volume ~2.9x; charge only grows 1.5x
// so the rubble is not launched across the whole arena.
const ROCKET_RADIUS_VOXELS: i32 = 18;
const ROCKET_LIFETIME_S: f32 = 6.0;
const MAX_ROCKETS: usize = 8;
/// Mass of the tool's shaped charge, in kilograms of TNT equivalent. The
/// deposited kinetic energy follows from the detonation energy of TNT, so
/// fragment velocities stay a physical consequence instead of a tuned number.
// 0.45 kg TNT equivalent: enough throw for a 2.5 m crater to read as a blast,
// while the 20-voxel radius still leaves the surrounding structure standing.
const SHOT_CHARGE_KG: f32 = 0.40;
/// Detonation energy of TNT (4.184 MJ/kg).
const TNT_ENERGY_J_PER_KG: f32 = 4.184e6;
const SHOT_ENERGY_JOULES: f32 = SHOT_CHARGE_KG * TNT_ENERGY_J_PER_KG;

/// World statistics captured once at startup; scanning bricks every frame would
/// waste the CPU budget for a diagnostics panel.
#[derive(Clone, Copy)]
struct VoxelStats {
    regions: usize,
    bricks: usize,
    solid: u64,
    hash: u64,
    build_ms: f32,
}

/// Owns all Phase 2 runtime state.
pub struct Game {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    world: VoxelWorld,
    /// Which canned scene `world` was built from; drives spawn and the light rig.
    scene: SceneKind,
    voxel_stats: VoxelStats,
    input: Input,
    camera: FreeCamera,
    player: Player,
    stats: FrameStats,
    last_frame: Option<Instant>,
    /// Screenshot readback cost of the previous frame, in milliseconds. A
    /// scenario `Screenshot` step does a blocking capture *after* the frame's own
    /// work, so its cost lands in the next frame's delta; subtracting it keeps
    /// the "slow frame" log reporting gameplay stalls only.
    last_capture_ms: f32,
    /// F1 overlay: bake pieces that stop moving back into the world file. Off
    /// reproduces the pre-fix behaviour where only Jolt-slept pieces were baked.
    debris_settle: bool,
    /// Draw dust billboards for voxels too small to become bodies
    /// (`SHATTERBOX_DUST_PARTICLES=0` disables it for A/B).
    dust_grains: bool,
    /// Coarse dynamic smoke field uploaded to `fog.wgsl` (cosmetic).
    smoke_field: SmokeField,
    /// Local GPU smoke box (part 3). `None` means no active source.
    smoke_box: Option<SmokeBox>,
    /// Emitters collected for the GPU step: payload plus remaining lifetime.
    smoke_emitters: Vec<(SmokeSource, f32)>,
    /// `SHATTERBOX_SMOKE_SIM=0` restores the coarse CPU field and billboards.
    smoke_sim: bool,
    /// CPU cost of the last obstacle rebuild, ms (reported via `cpu.smoke`).
    smoke_mask_ms: f32,
    /// Voxels currently on fire (flamethrower / spreading burn).
    burning: Vec<Burning>,
    show_overlay: bool,
    /// `SHATTERBOX_DEBUG_HIT`: paint the CPU aim intersection as a small marker
    /// so a mismatch between the reticle and the carved hole is attributable.
    debug_hit: bool,
    benchmark: Option<Benchmark>,
    destruction: Option<DestructionSystem<JoltPhysics>>,
    sectors: SectorCollision,
    physics_accumulator: f32,
    physics_alpha: f32,
    pending_edits: Vec<IVec3>,
    /// Cumulative count of dirty bricks handed to the renderer, reported by the
    /// autotest harness.
    flushed_dirty_bricks: usize,
    /// Dirty bricks flushed on the current frame only, for the frame timeline.
    frame_dirty_bricks: usize,
    /// Sleeping pieces baked back into the world since startup (autotest).
    baked_pieces: usize,
    baked_voxels: usize,
    debris_voxels_dirty: bool,
    /// Cumulative histogram of created debris piece sizes (voxels) for the
    /// autotest report: 1, 2-4, 5-7 (the sizes the minimum-size merge should
    /// erase), 8-16, 17-64, 65-256, 257+.
    debris_histogram: [u32; 7],
    /// Dust grains spawned as billboards for voxels too small to become bodies.
    dust_particles: usize,
    /// Peak number of live smoke/dust puffs seen this run, for `report.json`.
    smoke_peak: usize,
    /// Shot statistics per impact profile, so the report can show that a rocket
    /// throws much harder than a carbine round.
    impact_explosive: ImpactTally,
    impact_kinetic: ImpactTally,
    /// Pieces alive in the physics system but not uploaded to the renderer
    /// (64-instance cap or debris voxel budget). Shown in the overlay.
    debris_dropped: usize,
    /// Hash of the uploaded debris index order; a change forces a voxel re-upload.
    debris_layout: u64,
    shot_sequence: u32,
    shots: u32,
    removed_total: usize,
    filled_total: usize,
    last_shot: Option<ShotSummary>,
    /// Selected weapon and its animation state.
    weapon: WeaponKind,
    /// Carbine cooldown left, in fixed 60 Hz ticks (see
    /// [`CARBINE_COOLDOWN_TICKS`]). Integer on purpose: float seconds drift.
    carbine_cooldown_ticks: i32,
    /// Fixed-step weapon simulation: ticks since start (carries the shot
    /// seed), frame-time accumulator and the trigger state. Runs on the same
    /// 60 Hz grid as physics, so the fire rate never depends on fps.
    sim_tick: u64,
    weapon_accumulator: f32,
    /// LMB is held: the carbine keeps firing on its cooldown.
    trigger_held: bool,
    /// A click that has not fired yet. Sticky until the cooldown allows the
    /// shot, so a fast single click always produces exactly one round.
    trigger_pending: bool,
    weapon_recoil: f32,
    muzzle_flash: f32,
    weapon_time: f32,
    /// Carbine tracers still visible this frame.
    tracers: Vec<Tracer>,
    /// Rotational camera shake from the last blast, seconds remaining.
    weapon_shake: f32,
    /// Impact spark from the last carbine hit (position + age).
    spark: Option<(Vec3, f32)>,
    /// Rockets in flight. Stepped at the fixed simulation rate and detonated at
    /// the first world hit, so the blast lands where the projectile stopped.
    rockets: Vec<Rocket>,
    /// Short-lived blast flashes (see [`Explosion`]).
    explosions: Vec<Explosion>,
    /// Smoke and dust billboards thrown by the blasts. Simulated here and
    /// uploaded to the renderer every frame; the spawn is seeded from the blast
    /// command so a replay produces the same cloud.
    smoke: Vec<SmokePuff>,
    particle_scratch: Vec<Particle>,
    /// Steady-state lighting, kept so a flash can be composed on top of it
    /// without rebuilding the light rig every frame.
    lights: Vec<PointLight>,
    area_lights: Vec<AreaLight>,
    light_scratch: Vec<PointLight>,
    /// `Some` in autotest mode: the game runs a `.ron` script instead of the
    /// interactive loop, renders offscreen and writes `report.json`.
    scenario: Option<ScenarioRun>,
}

/// A blast flash. The explosion is sold by a bright, quickly decaying point
/// light, a rotational camera shake, the crater the blast carved and the smoke
/// and dust thrown out of it (see [`SmokePuff`]).
struct Explosion {
    position: Vec3,
    age: f32,
}

/// One smoke or dust puff: a soft billboard with a simple ballistic-drag path.
///
/// `tint` is rgb albedo plus the *peak* opacity; the uploaded opacity is that
/// value times the fade-in/fade-out envelope, so the simulation owns the colour
/// and the render list stays a pure snapshot.
/// Which emitter a puff came from. The shader reads this as a float so it can
/// pick density, torn-edge scale and how strongly the blast flash lights it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PuffKind {
    /// Dense fireball smoke: dark and hot early, grey and pale late.
    Smoke,
    /// Low, fast, settling dust from the shock wave.
    Dust,
    /// Thin rocket-exhaust trail: short-lived and almost transparent.
    Trail,
    /// Single-voxel dust grain for chips too small to become bodies.
    Grain,
}

impl PuffKind {
    fn code(self) -> f32 {
        match self {
            Self::Smoke => 0.0,
            Self::Dust => 1.0,
            Self::Trail => 2.0,
            Self::Grain => 3.0,
        }
    }
}

#[derive(Clone, Copy)]
struct SmokePuff {
    position: Vec3,
    velocity: Vec3,
    start_radius: f32,
    end_radius: f32,
    radius: f32,
    age: f32,
    life: f32,
    tint: [f32; 4],
    /// Early colour while the puff is still fire-lit; the colour ramps from
    /// `hot` to `tint` over the first half of the lifetime.
    hot: [f32; 3],
    /// Extinction multiplier for the Beer-Lambert sun term in `particles.wgsl`.
    density: f32,
    /// Per-puff noise offset so neighbouring puffs tear differently.
    seed: f32,
    kind: PuffKind,
    /// Fraction of the horizontal velocity lost per second.
    drag: f32,
    /// Upward acceleration, m/s² (hot smoke rises, dust settles).
    rise: f32,
}

/// How long the fireball light lives. Long enough to be seen, short enough that
/// a rocket volley does not stack a dozen lights into one white blob.
const EXPLOSION_FLASH_SECONDS: f32 = 0.38;
/// Live debris cap. Above this the farthest pieces are baked into the world
/// instead of being deleted, and every live piece fits the render instance
/// budget, so the normal `dropped` count is zero.
const MAX_LIVE_DEBRIS_PIECES: usize = MAX_RENDER_DEBRIS;
const EXPLOSION_LIGHT_INTENSITY: f32 = 60.0;
const EXPLOSION_LIGHT_RADIUS: f32 = 24.0;
/// Smoke/dust puffs per explosion at `scale = 1`, plus their lifetime range.
/// Both sets are spawned from the blast seed, so the cloud is deterministic.
/// Smoke/dust puffs per explosion at `scale = 1`. The cloud reads as a volume
/// only when many overlapping puffs fill it; 90 + 40 keeps a single blast under
/// ~150 puffs, well inside the 200–600 live budget shared by several blasts.
const SMOKE_PUFFS: usize = 90;
const DUST_PUFFS: usize = 40;
const SMOKE_LIFE_S: (f32, f32) = (1.9, 2.9);
const DUST_LIFE_S: (f32, f32) = (0.9, 1.6);
/// Carbine: 10 rounds per second, a 2-voxel capsule and a fraction of the
/// rocket's energy, so it chips cover instead of opening a wall.
///
/// The interval is integer ticks on the fixed 60 Hz step, not float seconds:
/// `0.1 - 6/60` in f32 leaves a rounding remainder that stretches the period
/// to 7 ticks (~8.6 rounds/s). Ticks keep it exactly 6.
const CARBINE_COOLDOWN_TICKS: i32 = 6;
const CARBINE_RADIUS_VOXELS: i32 = 2;
const CARBINE_ENERGY_JOULES: f32 = SHOT_ENERGY_JOULES * 0.02;
const CARBINE_SPREAD_RAD: f32 = 0.006;
const TRACER_SECONDS: f32 = 0.06;
const MUZZLE_FLASH_SECONDS: f32 = 0.06;
const WEAPON_SHAKE_SECONDS: f32 = 0.45;
const WEAPON_RECOIL_SECONDS: f32 = 0.18;
/// Slots in the renderer's debris instance cap kept for the viewmodel, rockets
/// and tracers. Debris never consumes the whole limit.
const RESERVED_SYNTH_SLOTS: usize = 12;
/// A single mouse sample larger than this is a cursor-grab warp, not a flick.
const MAX_LOOK_DELTA_PX: f32 = 260.0;

/// One in-flight rocket.
#[derive(Clone, Copy)]
struct Rocket {
    position: Vec3,
    velocity: Vec3,
    age: f32,
    seed: u32,
    scale: f32,
}

/// Active weapon. The bazooka and the carbine are the two destruction tools;
/// the flamethrower is the third slot and ignites flammable voxels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeaponKind {
    Bazooka,
    Carbine,
    Flamethrower,
}

impl WeaponKind {
    const ALL: [Self; 3] = [Self::Bazooka, Self::Carbine, Self::Flamethrower];

    const fn name(self) -> &'static str {
        match self {
            Self::Bazooka => "Bazooka",
            Self::Carbine => "Carbine",
            Self::Flamethrower => "Flamethrower",
        }
    }

    /// Next slot in the given wheel direction, wrapping.
    const fn cycle(self, direction: i32) -> Self {
        let index = match self {
            Self::Bazooka => 0,
            Self::Carbine => 1,
            Self::Flamethrower => 2,
        };
        let next = if direction >= 0 {
            (index + 1) % 3
        } else {
            (index + 2) % 3
        };
        Self::ALL[next]
    }

    /// Number-key slot → weapon. Must match the order the weapon bar draws
    /// (`ALL`), otherwise key 1 would select the second slot.
    const fn from_slot(slot: usize) -> Self {
        Self::ALL[match slot {
            1 => 0,
            2 => 1,
            3 => 2,
            _ => 0,
        }]
    }
}

/// One burning voxel: seconds of fuel left and the countdown to the next
/// neighbour spread.
struct Burning {
    cell: IVec3,
    fuel: f32,
    spread: f32,
}

/// Maps a scripted scenario weapon onto the interactive weapon bar, so `Hold`
/// and `Fire` route through the same weapon sim as the player's input. `Shot`
/// has no slot — it is the harness's instant carve and never reaches this
/// mapper's result.
fn weapon_kind_for(weapon: Weapon) -> WeaponKind {
    match weapon {
        Weapon::Rocket => WeaponKind::Bazooka,
        Weapon::Carbine => WeaponKind::Carbine,
        Weapon::Flame => WeaponKind::Flamethrower,
        Weapon::Shot => WeaponKind::Bazooka,
    }
}

/// Debug switch for the part 1 measurements: `SHATTERBOX_HIDE_DEBRIS=1` draws
/// the whole run without the debris instances, so the G-buffer cost of the pile
/// can be separated from the cost of the crater it left in the brick map.
fn hide_debris_forced() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("SHATTERBOX_HIDE_DEBRIS")
            .is_ok_and(|raw| !matches!(raw.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
    })
}

/// `SHATTERBOX_DEBRIS_SETTLE=0` keeps pieces simulating forever (the pre-fix
/// behaviour, where only pieces Jolt itself put to sleep were baked), so the
/// settle timer can be A/B'd without a rebuild.
fn settle_debris_default() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        // Baking rewrites a resting piece into floor-quantised voxel cells,
        // which visibly changes its geometry the moment it turns static.
        // Default: keep rubble dynamic and let the over-budget path retire only
        // far pieces; `SHATTERBOX_DEBRIS_SETTLE=1` restores the old bake.
        std::env::var("SHATTERBOX_DEBRIS_SETTLE")
            .is_ok_and(|raw| matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "on"))
    })
}

/// `SHATTERBOX_DUST_PARTICLES=0` hides the dust billboards, the visual part of
/// the minimum-piece rule, so the old look can be A/B'd.
fn dust_grains_default() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        // Default OFF: these billboards are single free-floating voxels that
        // read as pixels hovering in the air, with nothing attaching them to a
        // chunk. The dust is already covered by the smoke clouds. Set
        // `SHATTERBOX_DUST_PARTICLES=1` to bring the old grains back for A/B.
        std::env::var("SHATTERBOX_DUST_PARTICLES")
            .is_ok_and(|raw| matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "on"))
    })
}

/// Deterministic 0..1 value from a seed, used for blast size and carbine spread.
fn hash_unit(seed: u32) -> f32 {
    let mut h = seed ^ 0x9e37_79b9;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h as f32 / u32::MAX as f32
}

/// Per-round seed for the carbine: folds the fixed-step simulation tick with
/// the shot sequence, so a scripted burst replays to the same pattern (same
/// tick and sequence numbers) while two bursts still differ. The tick is what
/// makes the seed frame-rate independent: on any fps the same hold produces
/// the same rounds, because the weapon sim always steps at 60 Hz.
fn carbine_seed(tick: u64, seq: u32) -> u32 {
    let mut h = (tick as u32).wrapping_mul(0x9e37_79b9) ^ seq.rotate_left(11) ^ 0x5bd1_e995;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

/// Blast size multiplier for a command seed: 0.7..1.4x of the base crater.
///
/// Deterministic on purpose - the same command log replays to the same crater,
/// while different shots vary so destruction does not look stamped out.
fn blast_scale(variation: bool, seed: u32) -> f32 {
    if variation {
        0.7 + 0.7 * hash_unit(seed)
    } else {
        1.0
    }
}

/// Crater radius in voxels and the base energy handed to the destruction system.
/// The `scale` also becomes the shot's `power`, so a stronger roll both opens a
/// bigger hole and scatters the chips further.
fn blast_geometry(scale: f32) -> (i32, f32) {
    (
        ((ROCKET_RADIUS_VOXELS as f32) * scale).round() as i32,
        SHOT_ENERGY_JOULES,
    )
}

/// `SHATTERBOX_MIN_PIECE=1` restores the pre-part-2 behaviour where every crumb
/// became its own Jolt body, so the new minimum size can be A/B'd without a
/// rebuild.
fn min_piece_default() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("SHATTERBOX_MIN_PIECE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(shatterbox_destruction::MIN_PIECE_VOXELS)
    })
}

/// Per-profile shot statistics for `report.json`: how many bodies each weapon
/// made and how fast they left the crater.
#[derive(Clone, Copy, Default)]
struct ImpactTally {
    shots: u32,
    pieces: u32,
    speed_sum: f32,
    max_speed: f32,
}

impl ImpactTally {
    fn record(&mut self, result: &shatterbox_destruction::FireResult) {
        self.shots += 1;
        self.pieces += result.pieces as u32;
        self.speed_sum += result.speed_sum;
        self.max_speed = self.max_speed.max(result.speed_max);
    }

    fn average_speed(self) -> f32 {
        if self.pieces == 0 {
            0.0
        } else {
            self.speed_sum / self.pieces as f32
        }
    }
}

/// Builds one deterministic puff from a blast seed.
///
/// `salt` separates the smoke stream from the dust stream, `index` the individual
/// puffs, and each property draws from its own hashed slot (`r(k)`), so changing
/// one range cannot reshuffle the rest of the cloud.
#[allow(clippy::too_many_arguments)]
fn make_puff(
    seed: u32,
    salt: u32,
    index: u32,
    point: Vec3,
    blast_radius: f32,
    life_range: (f32, f32),
    speed: f32,
    tint: [f32; 4],
    drag: f32,
    rise: f32,
    growth: f32,
) -> SmokePuff {
    let base = seed ^ salt ^ index.wrapping_mul(0x9e37_79b9);
    let r = |k: u32| hash_unit(base.wrapping_add(k.wrapping_mul(0x85eb_ca6b)));
    // Uniform point on the unit sphere, then a directional bias: rising smoke
    // becomes a column, ground dust stays close to horizontal.
    let z = r(1) * 2.0 - 1.0;
    let angle = r(2) * std::f32::consts::TAU;
    let ring = (1.0 - z * z).max(0.0).sqrt();
    let sphere = Vec3::new(ring * angle.cos(), z, ring * angle.sin());
    let direction = if rise >= 0.0 {
        (sphere * 0.7 + Vec3::Y * 0.7).normalize()
    } else {
        (sphere + Vec3::Y * 0.15).normalize()
    };
    let position = point + direction * (blast_radius * 0.4 * r(3));
    let start_radius = blast_radius * (0.16 + 0.16 * r(4));
    let end_radius = start_radius + growth * blast_radius * (0.7 + 0.9 * r(5));
    let life = life_range.0 + (life_range.1 - life_range.0) * r(6);
    let velocity = direction * (speed * (0.45 + 0.95 * r(7))) + Vec3::Y * (0.35 + 1.15 * r(8));
    let shade = 0.82 + 0.36 * r(9);
    let alpha = (tint[3] * (0.72 + 0.55 * r(10))).min(1.0);
    SmokePuff {
        position,
        velocity,
        start_radius,
        end_radius,
        radius: start_radius,
        age: 0.0,
        life,
        tint: [tint[0] * shade, tint[1] * shade, tint[2] * shade, alpha],
        // Defaults: cold, dense, noise offset from the same deterministic
        // stream. Emitters override `hot`/`density`/`kind`.
        hot: [
            tint[0] * shade * 0.35,
            tint[1] * shade * 0.30,
            tint[2] * shade * 0.28,
        ],
        density: 1.0,
        seed: r(11) * 97.0,
        kind: PuffKind::Smoke,
        drag,
        rise,
    }
}

/// One carbine tracer shot: a thin emissive segment plus a short spark light.
#[derive(Clone, Copy)]
struct Tracer {
    from: Vec3,
    to: Vec3,
    age: f32,
}

/// Small voxel model uploaded through the debris path. Weapons and projectiles
/// ride the existing instanced debug grid instead of adding another pipeline.
struct SynthModel {
    dims: IVec3,
    voxels: Vec<u16>,
}

impl SynthModel {
    fn new(dims: IVec3) -> Self {
        let len = (dims.x * dims.y * dims.z).max(0) as usize;
        Self {
            dims,
            voxels: vec![Voxel::AIR.raw(); len],
        }
    }

    fn set(&mut self, x: i32, y: i32, z: i32, material: u16, tint: u8) {
        if x < 0 || y < 0 || z < 0 || x >= self.dims.x || y >= self.dims.y || z >= self.dims.z {
            return;
        }
        let index = (x + self.dims.x * (y + self.dims.y * z)) as usize;
        self.voxels[index] = Voxel::new(material, tint).raw();
    }

    fn fill(&mut self, min: IVec3, max: IVec3, material: u16, tint: u8) {
        for z in min.z..max.z {
            for y in min.y..max.y {
                for x in min.x..max.x {
                    self.set(x, y, z, material, tint);
                }
            }
        }
    }

    fn view<'a>(&'a self, model: Mat4, voxel_offset: u32) -> DebrisView<'a> {
        let half = self.dims.as_vec3() * (0.5 * shatterbox_voxel::VOXEL_SIZE_M);
        DebrisView {
            model,
            inverse_model: model.inverse(),
            aabb_min: -half,
            aabb_max: half,
            dims: self.dims,
            voxel_offset,
            voxels: &self.voxels,
        }
    }
}

/// Chunky first-person bazooka: metal tube, rubber grip, wooden shoulder rest.
fn bazooka_model() -> SynthModel {
    let mut model = SynthModel::new(IVec3::new(5, 6, 13));
    model.fill(
        IVec3::new(1, 2, 0),
        IVec3::new(4, 5, 13),
        materials::CONCRETE,
        4,
    );
    model.fill(
        IVec3::new(2, 1, 3),
        IVec3::new(3, 3, 6),
        materials::RUBBER,
        8,
    );
    model.fill(
        IVec3::new(1, 1, 7),
        IVec3::new(4, 3, 9),
        materials::WOOD,
        12,
    );
    model.fill(
        IVec3::new(0, 2, 11),
        IVec3::new(5, 5, 13),
        materials::METAL,
        14,
    );
    model
}

/// Carbine: slim barrel, receiver, magazine and stock.
fn carbine_model() -> SynthModel {
    let mut model = SynthModel::new(IVec3::new(4, 5, 11));
    model.fill(
        IVec3::new(1, 2, 2),
        IVec3::new(3, 4, 11),
        materials::CONCRETE,
        2,
    );
    model.fill(
        IVec3::new(0, 1, 0),
        IVec3::new(4, 4, 5),
        materials::CONCRETE,
        6,
    );
    model.fill(
        IVec3::new(1, 0, 2),
        IVec3::new(2, 2, 5),
        materials::RUBBER,
        10,
    );
    model.fill(
        IVec3::new(1, 0, 1),
        IVec3::new(2, 2, 3),
        materials::METAL,
        16,
    );
    model
}

/// Rocket body with an emissive tail so it reads as a projectile in flight.
fn rocket_model() -> SynthModel {
    let mut model = SynthModel::new(IVec3::new(3, 3, 7));
    model.fill(
        IVec3::new(0, 0, 0),
        IVec3::new(3, 3, 5),
        materials::CONCRETE,
        40,
    );
    model.fill(
        IVec3::new(1, 1, 5),
        IVec3::new(2, 2, 7),
        materials::EMISSIVE,
        52,
    );
    model
}

/// Muzzle flash: a small emissive cross in front of the barrel.
fn muzzle_model() -> SynthModel {
    let mut model = SynthModel::new(IVec3::new(3, 3, 2));
    model.fill(
        IVec3::new(1, 0, 0),
        IVec3::new(2, 3, 2),
        materials::EMISSIVE,
        60,
    );
    model.fill(
        IVec3::new(0, 1, 0),
        IVec3::new(3, 2, 2),
        materials::EMISSIVE,
        60,
    );
    model
}

/// Tracer: two emissive voxels stretched along the shot.
fn tracer_model() -> SynthModel {
    let mut model = SynthModel::new(IVec3::new(1, 1, 2));
    model.fill(
        IVec3::new(0, 0, 0),
        IVec3::new(1, 1, 2),
        materials::EMISSIVE,
        45,
    );
    model
}

/// Optional headless-ish benchmark: `SHATTERBOX_BENCHMARK=<seconds>` runs the
/// scene for that long after a warm-up and logs fps/GPU timings before exiting.
struct Benchmark {
    warmup_seconds: f32,
    duration_seconds: f32,
    elapsed: f32,
    frame_times_ms: Vec<f32>,
    gpu_times_ms: Vec<f32>,
    gbuffer_times_ms: Vec<f32>,
    lighting_times_ms: Vec<f32>,
    temporal_times_ms: Vec<f32>,
    shadow_times_ms: Vec<f32>,
    light_shadow_times_ms: Vec<f32>,
    gi_times_ms: Vec<f32>,
    fog_times_ms: Vec<f32>,
    bloom_times_ms: Vec<f32>,
    skipped_frames: u64,
}

/// Neutral diagnostic rig for the studio scene: the screens and the colonnade
/// are lit so shadow shape and penumbra are readable, with two dim coloured
/// fills so albedo bounce can be judged without painting the arena neon.
fn studio_lights() -> Vec<PointLight> {
    vec![
        PointLight {
            position: [104.0, 6.0, 160.0],
            radius: 30.0,
            color: [1.0, 0.97, 0.93],
            intensity: 6.0,
        },
        PointLight {
            position: [152.0, 6.0, 160.0],
            radius: 30.0,
            color: [1.0, 0.97, 0.93],
            intensity: 6.0,
        },
        PointLight {
            position: [104.0, 6.0, 136.0],
            radius: 30.0,
            color: [1.0, 0.97, 0.93],
            intensity: 6.0,
        },
        PointLight {
            position: [152.0, 6.0, 136.0],
            radius: 30.0,
            color: [1.0, 0.97, 0.93],
            intensity: 6.0,
        },
        PointLight {
            position: [128.0, 7.0, 172.0],
            radius: 26.0,
            color: [1.0, 0.96, 0.90],
            intensity: 5.0,
        },
        PointLight {
            position: [128.0, 5.0, 182.0],
            radius: 20.0,
            color: [1.0, 0.99, 0.97],
            intensity: 3.0,
        },
        PointLight {
            position: [100.0, 3.2, 122.0],
            radius: 15.0,
            color: [0.30, 0.62, 1.0],
            intensity: 4.0,
        },
        PointLight {
            position: [156.0, 3.2, 122.0],
            radius: 15.0,
            color: [1.0, 0.55, 0.22],
            intensity: 4.0,
        },
    ]
}

/// Five warm ceiling lamps for the house. Five of the twelve point-light slots
/// are used, so seven stay free for muzzle flashes and blast lights.
fn house_lights() -> Vec<PointLight> {
    house_lamp_positions()
        .into_iter()
        .map(|position| PointLight {
            position: position.to_array(),
            // A 9 m radius let every lamp reach every wall, so five lamps
            // summed into a uniform cream room with no pools. 6.5 m keeps a
            // warm core under each fixture and lets the corners fall into
            // cool ambient instead of being repainted by direct light.
            radius: 6.5,
            color: [1.0, 0.88, 0.72],
            intensity: 1.8,
        })
        .collect()
}

/// Two warm ceiling lamps plus a wall-wash pair next to the painted walls.
/// The wash lights exist for the DDGI test: they give the red and blue walls a
/// strong radiance without sunlight, so their coloured bounce is visible in the
/// night shot (a white room lit only by white sky averages to white).
fn test_room_lights() -> Vec<PointLight> {
    let mut lights: Vec<PointLight> = test_room_lamp_positions()
        .into_iter()
        .map(|position| PointLight {
            position: position.to_array(),
            radius: 14.0,
            color: [1.0, 0.90, 0.76],
            intensity: 9.0,
        })
        .collect();
    for x in [129.5, 142.5] {
        lights.push(PointLight {
            position: [x, 4.5, 136.0],
            radius: 9.0,
            color: [1.0, 0.94, 0.86],
            intensity: 16.0,
        });
    }
    lights
}

/// FX lab: one lamp per interior zone, warm. Four of twelve slots are used, so
/// muzzle flashes and fire lights have room. The blast room and the shaft are
/// deliberately unlit: daylight must enter them through the opening the
/// scenario makes.
fn fx_lab_lights() -> Vec<PointLight> {
    // A/B hook for the colour-bounce test: with the warm lamps off, the enclosed
    // room is lit only by daylight through its window plus indirect light, so a
    // red/blue tint on the white floor can only come from DDGI bounce.
    // `SHATTERBOX_FXLAB_LAMPS=0` disables the rig; the default keeps it.
    if std::env::var("SHATTERBOX_FXLAB_LAMPS").is_ok_and(|value| value == "0") {
        tracing::info!("fx_lab lamp rig disabled by SHATTERBOX_FXLAB_LAMPS=0");
        return Vec::new();
    }
    // (intensity, radius): enclosed room, corridor, lit room, hall. The lab is
    // a test scene, so the fixtures are deliberately strong enough to read the
    // smoke and columns at noon without daylight help.
    const RIG: [(f32, f32); 4] = [(20.0, 9.0), (15.0, 8.0), (25.0, 10.0), (40.0, 16.0)];
    fx_lab_lamp_positions()
        .into_iter()
        .zip(RIG)
        .map(|(position, (intensity, radius))| PointLight {
            position: position.to_array(),
            radius,
            color: [1.0, 0.88, 0.72],
            intensity,
        })
        .collect()
}

impl Game {
    pub fn new() -> Self {
        Self::with_scene(SceneKind::from_env())
    }

    /// Builds the world for a specific canned scene. Scenarios pin their scene
    /// in the `.ron` file so an old studio run does not silently move to the
    /// house after the default changed.
    pub fn with_scene(scene: SceneKind) -> Self {
        let build_start = Instant::now();
        let world = scene.build();
        let build_ms = build_start.elapsed().as_secs_f32() * 1000.0;
        let voxel_stats = VoxelStats {
            regions: world.non_empty_regions(),
            bricks: world.brick_count(),
            solid: world.solid_voxels(),
            hash: world.deterministic_hash(),
            build_ms,
        };
        tracing::info!(
            scene = scene.name(),
            regions = voxel_stats.regions,
            bricks = voxel_stats.bricks,
            solid = voxel_stats.solid,
            hash = format_args!("{:#018x}", voxel_stats.hash),
            build_ms,
            "voxel scene built"
        );
        let (spawn, _, _) = scene.spawn();

        Self {
            window: None,
            gpu: None,
            renderer: None,
            world,
            scene,
            voxel_stats,
            input: Input::new(),
            camera: FreeCamera::default(),
            player: Player::new(spawn),
            stats: FrameStats::default(),
            last_frame: None,
            last_capture_ms: 0.0,
            debris_settle: settle_debris_default(),
            dust_grains: dust_grains_default(),
            smoke_field: SmokeField::new(),
            smoke_box: None,
            smoke_emitters: Vec::new(),
            smoke_sim: !matches!(
                std::env::var("SHATTERBOX_SMOKE_SIM")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "off"
            ),
            smoke_mask_ms: 0.0,
            burning: Vec::new(),
            show_overlay: true,
            benchmark: std::env::var("SHATTERBOX_BENCHMARK")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .map(|duration| Benchmark {
                    warmup_seconds: 3.0,
                    duration_seconds: duration.clamp(1.0, 120.0),
                    elapsed: 0.0,
                    frame_times_ms: Vec::new(),
                    gpu_times_ms: Vec::new(),
                    gbuffer_times_ms: Vec::new(),
                    lighting_times_ms: Vec::new(),
                    temporal_times_ms: Vec::new(),
                    shadow_times_ms: Vec::new(),
                    light_shadow_times_ms: Vec::new(),
                    gi_times_ms: Vec::new(),
                    fog_times_ms: Vec::new(),
                    bloom_times_ms: Vec::new(),
                    skipped_frames: 0,
                }),
            destruction: None,
            sectors: SectorCollision::new(),
            physics_accumulator: 0.0,
            physics_alpha: 0.0,
            pending_edits: Vec::new(),
            flushed_dirty_bricks: 0,
            frame_dirty_bricks: 0,
            baked_pieces: 0,
            baked_voxels: 0,
            debris_voxels_dirty: false,
            debris_histogram: [0; 7],
            dust_particles: 0,
            smoke_peak: 0,
            impact_explosive: ImpactTally::default(),
            impact_kinetic: ImpactTally::default(),
            debris_dropped: 0,
            debris_layout: 0,
            shot_sequence: 1,
            shots: 0,
            removed_total: 0,
            filled_total: 0,
            last_shot: None,
            weapon: WeaponKind::Bazooka,
            carbine_cooldown_ticks: 0,
            sim_tick: 0,
            weapon_accumulator: 0.0,
            trigger_held: false,
            trigger_pending: false,
            weapon_recoil: 0.0,
            muzzle_flash: 0.0,
            weapon_time: 0.0,
            tracers: Vec::new(),
            weapon_shake: 0.0,
            spark: None,
            rockets: Vec::new(),
            explosions: Vec::new(),
            smoke: Vec::new(),
            particle_scratch: Vec::new(),
            lights: Vec::new(),
            area_lights: Vec::new(),
            light_scratch: Vec::new(),
            debug_hit: std::env::var_os("SHATTERBOX_DEBUG_HIT").is_some(),
            scenario: None,
        }
    }

    /// Builds the game in autotest mode. The scenario file is loaded before the
    /// window exists so a typo fails fast instead of after GPU init.
    pub fn with_scenario(options: ScenarioOptions) -> anyhow::Result<Self> {
        let scenario = crate::scenario::Scenario::load(&options.path)?;
        let scene = match scenario.scene.as_deref() {
            Some(name) => SceneKind::parse(name).ok_or_else(|| {
                anyhow::anyhow!("unknown scene {name:?} in {}", options.path.display())
            })?,
            None => SceneKind::from_env(),
        };
        let mut game = Self::with_scene(scene);
        let initial_solid = game.world.solid_voxels();
        game.scenario = Some(ScenarioRun::new(options, scenario, initial_solid));
        Ok(game)
    }

    /// One frame of the scripted autotest loop. Simulation still advances by a
    /// fixed 60 Hz step; the wall clock is only measured so `report.json` can
    /// show real pacing.
    #[allow(clippy::too_many_lines)]
    fn redraw_scenario(&mut self, _window: &Window) -> AppEvent {
        let now = Instant::now();
        let render_dt = self
            .last_frame
            .map_or(FIXED_DT, |last| now.duration_since(last).as_secs_f32())
            .clamp(1.0e-4, 0.1);
        self.last_frame = Some(now);
        self.stats.update(render_dt);

        let Some(run) = self.scenario.as_mut() else {
            return AppEvent::Exit;
        };
        if run.should_abort() {
            return self.finish_scenario();
        }
        let plan = run.begin_frame();

        if let Some(pose) = plan.camera {
            self.camera.position = Vec3::from_array(pose.position);
            self.camera.yaw = pose.yaw;
            self.camera.pitch = pose.pitch;
            // Keep the player under the scripted camera so debris baking never
            // pastes a piece into the lens.
            self.player.position = self.camera.position - Vec3::Y * self.player.eye_height;
            self.player.velocity = Vec3::ZERO;
        }
        if let Some(hours) = plan.time_of_day
            && let Some(renderer) = &mut self.renderer
        {
            renderer.set_time_of_day(hours);
        }
        if let Some(bits) = plan.effects
            && let Some(renderer) = &mut self.renderer
        {
            renderer.set_effects(RenderEffects::from_bitmask(bits));
        }
        if let Some(bits) = plan.post_debug
            && let Some(renderer) = &mut self.renderer
        {
            renderer.set_post_debug(bits);
        }
        if let Some(look) = plan.look
            && let Some(renderer) = &mut self.renderer
        {
            renderer.set_look(look);
        }
        if let Some(slot) = plan.select {
            self.weapon = match slot {
                1 => WeaponKind::Bazooka,
                2 => WeaponKind::Carbine,
                3 => WeaponKind::Flamethrower,
                _ => self.weapon,
            };
        }
        let mut cpu = crate::scenario::CpuTimes::default();
        // Scripted shots go through the same fixed-step weapon sim as the
        // interactive loop: `Fire` latches a click, `Hold` keeps the trigger
        // down for automatic fire. The harness `Shot` stays a direct carve: it
        // is not a weapon slot but the instant-rocket primitive that old
        // scenarios compare against.
        if let Some(weapon) = plan.hold {
            self.weapon = weapon_kind_for(weapon);
            self.trigger_held = true;
        }
        let direct_carve = plan.fire == Some(Weapon::Shot);
        if !direct_carve && let Some(weapon) = plan.fire {
            self.weapon = weapon_kind_for(weapon);
            self.trigger_pending = true;
        }
        let start = Instant::now();
        if direct_carve {
            self.carve_shot();
        } else {
            self.step_weapons();
            self.step_burning(FIXED_DT);
        }
        self.trigger_held = false;
        cpu.shots = start.elapsed().as_secs_f32() * 1000.0;

        self.frame_dirty_bricks = 0;
        let start = Instant::now();
        self.step_physics(FIXED_DT);
        cpu.physics = start.elapsed().as_secs_f32() * 1000.0;
        self.advance_weapon_animations(FIXED_DT);
        let start = Instant::now();
        self.bake_debris();
        cpu.bake = start.elapsed().as_secs_f32() * 1000.0;
        let start = Instant::now();
        self.step_rockets(FIXED_DT);
        cpu.rockets = start.elapsed().as_secs_f32() * 1000.0;
        let start = Instant::now();
        self.update_explosions(FIXED_DT);
        self.update_smoke(FIXED_DT);
        cpu.smoke = start.elapsed().as_secs_f32() * 1000.0;
        let start = Instant::now();
        self.flush_world_edits();
        cpu.world_upload = start.elapsed().as_secs_f32() * 1000.0;
        let start = Instant::now();
        self.update_sectors();
        cpu.sectors = start.elapsed().as_secs_f32() * 1000.0;
        let start = Instant::now();
        self.upload_debris();
        cpu.debris_upload = start.elapsed().as_secs_f32() * 1000.0;
        // CPU reference for `SHATTERBOX_DEBUG_HIT`, taken from the same camera
        // the frame is rendered with.
        let cpu_ray = self
            .world
            .raycast(self.camera.position, self.camera.forward(), 256.0);
        let debug_hit = self.debug_hit.then(|| {
            cpu_ray.map(|hit| self.camera.position + self.camera.forward() * hit.distance)
        });
        let debug_hit = debug_hit.flatten();

        let size = self
            .scenario
            .as_ref()
            .map_or((2560, 1440), |run| run.options().size);
        let render_start = Instant::now();
        let (captured, stats) = {
            let (Some(gpu), Some(renderer)) = (&mut self.gpu, &mut self.renderer) else {
                return AppEvent::Exit;
            };
            renderer.set_debug_hit(debug_hit);
            // The depth probe must see the voxel world only (see
            // `Renderer::set_hide_debris`). `SHATTERBOX_HIDE_DEBRIS=1` forces
            // the same switch for a whole run: it separates the G-buffer cost of
            // the pile from the cost of the crater it left in the brick map.
            renderer.set_hide_debris(plan.hide_debris || hide_debris_forced());
            match renderer.render_offscreen(gpu, &self.camera, FIXED_DT, size, plan.capture) {
                Ok(frame) => (frame, renderer.stats()),
                Err(error) => {
                    let message = format!("offscreen render failed: {error:#}");
                    tracing::error!("{message}");
                    if let Some(run) = self.scenario.as_mut() {
                        run.fail(message);
                    }
                    return self.finish_scenario();
                }
            }
        };

        let render_ms = render_start.elapsed().as_secs_f32() * 1000.0;

        // Lateral samples for `assert_hit_matches_render`. The centre sample
        // alone cannot see a sideways shift on a flat wall, so every tap is
        // re-raycast on the CPU through the exact pixel the renderer sampled.
        let hit_check = captured.as_ref().map(|frame| {
            let aspect = frame.width.max(1) as f32 / frame.height.max(1) as f32;
            let inverse = self.camera.view_projection(aspect).inverse();
            let (width, height) = frame.internal_size;
            let taps = frame
                .depth_taps
                .iter()
                .map(|tap| {
                    let ndc = Vec3::new(
                        (tap.pixel[0] as f32 + 0.5) / width.max(1) as f32 * 2.0 - 1.0,
                        1.0 - (tap.pixel[1] as f32 + 0.5) / height.max(1) as f32 * 2.0,
                        0.0,
                    );
                    let near = inverse.project_point3(ndc);
                    let far = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
                    let direction = (far - near).normalize_or_zero();
                    let cpu = self
                        .world
                        .raycast(self.camera.position, direction, 256.0)
                        .map(|hit| hit.distance);
                    TapCheck {
                        pixel: tap.pixel,
                        cpu,
                        render: tap.distance,
                    }
                })
                .collect();
            HitCheck { taps }
        });

        let solid = self.world.solid_voxels();
        let (debris_voxels, pieces) =
            self.destruction
                .as_ref()
                .map_or((0u64, 0usize), |destruction| {
                    let pieces = destruction.pieces();
                    let debris: u64 = pieces
                        .iter()
                        .map(|piece| u64::from(piece.voxel_count))
                        .sum();
                    (debris, pieces.len())
                });
        let dust_voxels = self
            .destruction
            .as_ref()
            .map_or(0u64, DestructionSystem::dust_voxels);
        let min_piece_voxels = self
            .destruction
            .as_ref()
            .map_or(0, DestructionSystem::min_piece);
        let body_count = self
            .destruction
            .as_ref()
            .map_or(0, DestructionSystem::total_bodies);
        let awake_bodies = self
            .destruction
            .as_ref()
            .map_or(0, DestructionSystem::active_bodies);
        let sleeping_bodies = self
            .destruction
            .as_ref()
            .map_or(0, DestructionSystem::sleeping_bodies);
        let world_hash = self.world.deterministic_hash();
        let dirty_bricks = self.flushed_dirty_bricks;
        let voxels_removed = self.removed_total;
        let debris_drawn = stats.debris_count;
        let debris_dropped = pieces.saturating_sub(debris_drawn as usize) as u32;
        let impacts = self.impacts_report();

        if let Some(run) = self.scenario.as_mut() {
            // Anything slow enough to be felt (33 ms = 30 fps) is logged with the
            // capture flag: the harness screenshot does a blocking readback, so a
            // stall on a capture frame is not a gameplay stall. The readback runs
            // after this frame's work, so it also has to be taken out of the next
            // frame's delta (`last_capture_ms`).
            let capture_overhead_ms = self.last_capture_ms;
            let gameplay_ms = (render_dt * 1000.0 - capture_overhead_ms).max(0.0);
            self.last_capture_ms = 0.0;
            if gameplay_ms > 33.0 {
                tracing::warn!(
                    target: "shatterbox::perf",
                    frame_ms = gameplay_ms,
                    // Work done inside this call, and the part of it spent in
                    // `render_offscreen`. When `frame_ms` is large but `work_ms`
                    // is not, the stall is between frames (swapchain, driver or
                    // the harness loop), not in our frame code.
                    work_ms = now.elapsed().as_secs_f32() * 1000.0,
                    render_ms,
                    capture_overhead_ms,
                    capture = plan.capture,
                    cpu = ?cpu,
                    "slow frame"
                );
            }
            let capture_start = Instant::now();
            if let Err(error) = run.finish_capture(captured.as_ref(), hit_check.as_ref()) {
                tracing::error!("scenario capture failed: {error:#}");
                run.fail(format!("capture failed: {error:#}"));
            }
            if plan.capture {
                self.last_capture_ms = capture_start.elapsed().as_secs_f32() * 1000.0;
            }
            run.check_conservation(solid, debris_voxels, dust_voxels);
            if run.wants_check(crate::scenario::Check::NoFloating) {
                let (floating, above) = self.world.count_unanchored_above(GROUND_ANCHOR_VOXEL);
                run.check_no_floating(floating, above, GROUND_ANCHOR_VOXEL);
            }
            run.note_physics(crate::scenario::PhysicsSnapshot {
                bodies: body_count,
                awake: awake_bodies,
                sleeping: sleeping_bodies,
                debris_drawn,
                debris_dropped,
                min_piece_voxels,
                debris_histogram: self.debris_histogram,
                dust_particles: self.dust_particles,
                smoke_particles: self.smoke.len(),
                smoke_peak: self.smoke_peak,
                dust_voxels,
            });
            run.note_impacts(impacts);
            run.note_world(crate::scenario::WorldSnapshot {
                voxels_removed,
                world_hash,
                dirty_bricks,
                baked_pieces: self.baked_pieces,
                baked_voxels: self.baked_voxels,
            });
            run.end_frame(
                render_dt * 1000.0,
                &stats,
                crate::scenario::FrameTelemetry {
                    bodies: body_count,
                    awake: awake_bodies,
                    sleeping: sleeping_bodies,
                    debris_drawn,
                    debris_simulated: pieces,
                    dirty_bricks: self.frame_dirty_bricks,
                    cpu,
                },
            );
        }

        if self.scenario.as_ref().is_some_and(ScenarioRun::is_finished) {
            return self.finish_scenario();
        }
        AppEvent::Continue
    }

    /// Writes the report and stops the loop.
    fn finish_scenario(&mut self) -> AppEvent {
        let count = shatterbox_gpu::diagnostics::validation_error_count();
        let samples = shatterbox_gpu::diagnostics::validation_error_samples(5);
        if let Some(run) = self.scenario.as_mut() {
            run.finish(count, &samples);
        }
        self.scenario = None;
        AppEvent::Exit
    }

    fn redraw(&mut self, window: &Window) -> AppEvent {
        if self.scenario.is_some() {
            return self.redraw_scenario(window);
        }
        let now = Instant::now();
        let dt = self
            .last_frame
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32())
            .min(0.1);
        self.last_frame = Some(now);
        self.stats.update(dt);
        self.update_controls(dt);
        self.step_physics(dt);
        self.bake_debris();
        self.step_rockets(dt);
        self.update_explosions(dt);
        self.update_smoke(dt);
        self.flush_world_edits();
        self.update_sectors();
        self.upload_debris();

        let aim_direction = self.camera.forward();
        let aim = self
            .world
            .raycast(self.camera.position, aim_direction, 256.0);
        // `SHATTERBOX_DEBUG_HIT` paints the exact CPU intersection point in the
        // lighting pass. If the crater ever sits beside the reticle again, this
        // separates "the ray is wrong" from "the image is shifted".
        let debug_hit = self
            .debug_hit
            .then(|| aim.map(|hit| self.camera.position + aim_direction * hit.distance))
            .flatten();
        let ray = aim.map(|hit| RaySummary {
            distance: hit.distance,
            material: self.world.palette().material(hit.voxel.material()).name,
            position: hit.position.to_array(),
            normal: hit.normal.to_array(),
        });
        // Project the *exact* intersection point with the same view-projection
        // the renderer uses. Snapping this to the voxel centre instead made the
        // dot sit beside the reticle at close range even though the ray was
        // dead centre, which read as a broken crosshair.
        let aim_marker = {
            let (width, height) = self.gpu.as_ref().map_or((1, 1), GpuContext::size);
            let aspect = width.max(1) as f32 / height.max(1) as f32;
            let view_proj = self.camera.view_projection(aspect);
            aim.and_then(|hit| {
                let point = self.camera.position + aim_direction * hit.distance;
                let clip = view_proj * point.extend(1.0);
                (clip.w > 1e-4).then(|| {
                    let ndc = clip.truncate() / clip.w;
                    [ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5]
                })
            })
        };
        let render_stats = self
            .renderer
            .as_ref()
            .map_or_else(RenderStats::default, Renderer::stats);
        let (pieces, awake_bodies) = self.destruction.as_ref().map_or((0, 0), |destruction| {
            (destruction.pieces().len(), destruction.active_bodies())
        });

        let render_camera = self.shaken_camera();
        let (Some(gpu), Some(renderer)) = (&mut self.gpu, &mut self.renderer) else {
            return AppEvent::Continue;
        };
        renderer.set_debug_hit(debug_hit);

        let mut actions = UiActions {
            effects: renderer.effects(),
            ddgi_quality: renderer.ddgi_quality(),
            time_of_day: renderer.time_of_day(),
            sun_temperature: renderer.sun_temperature(),
            sun_intensity: renderer.sun_intensity(),
            fog_density: renderer.fog_density(),
            fog_height: renderer.fog_height(),
            fog_smooth: renderer.fog_smooth(),
            exposure: renderer.exposure(),
            auto_exposure: renderer.auto_exposure(),
            exposure_bias: renderer.exposure_bias(),
            look: renderer.look(),
            sky_save: false,
            shadow_mode: renderer.shadow_mode(),
            upscale_clamp: renderer.upscale_clamp(),
            debris_reactive: renderer.debris_reactive(),
            settle_debris: self.debris_settle,
            dust_grains: self.dust_grains,
            weapon: None,
        };
        let overlay = OverlayData {
            fps: self.stats.fps,
            frame_time_ms: self.stats.frame_time_ms,
            smoothed_frame_time_ms: self.stats.smoothed_frame_time_ms,
            frame_index: self.stats.frame_index,
            adapter: gpu.adapter_name().to_owned(),
            backend: gpu.backend_name(),
            resolution: gpu.size(),
            render: render_stats,
            voxels: self.voxel_stats,
            ray,
            aim_marker,
            cursor_captured: self.input.cursor_captured(),
            camera_position: self.camera.position.to_array(),
            camera_yaw_degrees: self.camera.yaw.to_degrees(),
            camera_pitch_degrees: self.camera.pitch.to_degrees(),
            show_overlay: self.show_overlay,
            pieces,
            debris_dropped: self.debris_dropped,
            rockets: self.rockets.len(),
            awake_bodies,
            shots: self.shots,
            removed_total: self.removed_total,
            filled_total: self.filled_total,
            last_shot: self.last_shot,
            weapon: self.weapon.name(),
        };

        let result = renderer.render(gpu, window, &render_camera, dt, |ctx| {
            draw_overlay(ctx, &overlay, &mut actions);
        });
        renderer.set_effects(actions.effects);
        // Keep the preset consistent with the internal enable bit: "All off"
        // turns the preset off, "All on" restores medium.
        let quality = if actions.effects.ddgi {
            match actions.ddgi_quality {
                DdgiQuality::Off => DdgiQuality::Medium,
                other => other,
            }
        } else {
            DdgiQuality::Off
        };
        renderer.set_ddgi_quality(quality);
        renderer.set_time_of_day(actions.time_of_day);
        renderer.set_sun_temperature(actions.sun_temperature);
        renderer.set_sun_intensity(actions.sun_intensity);
        renderer.set_fog_density(actions.fog_density);
        renderer.set_fog_height(actions.fog_height);
        renderer.set_fog_smooth(actions.fog_smooth);
        renderer.set_exposure(actions.exposure);
        renderer.set_auto_exposure(actions.auto_exposure);
        renderer.set_exposure_bias(actions.exposure_bias);
        renderer.set_look(actions.look);
        if actions.sky_save {
            crate::sky_config::SkyConfig {
                sun_intensity: actions.sun_intensity,
                fog_density: actions.fog_density,
                fog_height: actions.fog_height,
            }
            .save();
        }
        renderer.set_shadow_mode(actions.shadow_mode);
        renderer.set_upscale_clamp(actions.upscale_clamp);
        renderer.set_debris_reactive(actions.debris_reactive);
        self.debris_settle = actions.settle_debris;
        self.dust_grains = actions.dust_grains;
        if let Some(weapon) = actions.weapon {
            self.weapon = weapon;
        }
        self.input.end_frame();

        if self.benchmark.is_some() {
            let render_stats = self
                .renderer
                .as_ref()
                .map_or_else(RenderStats::default, Renderer::stats);
            let presented = matches!(result, Ok(FrameOutcome::Presented));
            if let Some(benchmark) = &mut self.benchmark {
                benchmark.elapsed += dt;
                if !presented {
                    benchmark.skipped_frames += 1;
                }
                if presented && benchmark.elapsed > benchmark.warmup_seconds && dt > 0.0 {
                    benchmark.frame_times_ms.push(dt * 1000.0);
                    if render_stats.timestamps_valid {
                        benchmark.gpu_times_ms.push(render_stats.total_ms);
                        benchmark.gbuffer_times_ms.push(render_stats.gbuffer_ms);
                        benchmark.lighting_times_ms.push(render_stats.lighting_ms);
                        benchmark.temporal_times_ms.push(render_stats.temporal_ms);
                        benchmark.shadow_times_ms.push(render_stats.shadow_ms);
                        benchmark
                            .light_shadow_times_ms
                            .push(render_stats.light_shadow_ms);
                        benchmark.gi_times_ms.push(render_stats.gi_ms);
                        benchmark.fog_times_ms.push(render_stats.fog_ms);
                        benchmark.bloom_times_ms.push(render_stats.bloom_ms);
                    }
                }
                if benchmark.elapsed > benchmark.warmup_seconds + benchmark.duration_seconds {
                    benchmark.report(
                        render_stats.output_size,
                        render_stats.internal_size,
                        render_stats.scale,
                    );
                    return AppEvent::Exit;
                }
            }
        }

        match result {
            Ok(_) => AppEvent::Continue,
            Err(error) => {
                tracing::error!("{error:#}");
                AppEvent::Exit
            }
        }
    }

    fn step_physics(&mut self, dt: f32) {
        self.physics_accumulator = (self.physics_accumulator + dt.min(0.1)).min(0.25);
        let impacts = {
            let Some(destruction) = &mut self.destruction else {
                return;
            };
            destruction.set_settle(self.debris_settle);
            let mut steps = 0;
            while self.physics_accumulator >= FIXED_DT && steps < 4 {
                destruction.step(FIXED_DT);
                self.physics_accumulator -= FIXED_DT;
                steps += 1;
            }
            if steps == 4 {
                // Do not let a long frame build an unbounded catch-up queue.
                self.physics_accumulator = 0.0;
            }
            destruction.take_impacts()
        };
        self.physics_alpha = (self.physics_accumulator / FIXED_DT).clamp(0.0, 1.0);
        self.apply_debris_impacts(impacts);
    }

    /// Heavy debris striking the world carves a small crater where it landed.
    /// Rate-limited to two craters per frame so a collapsing pile cannot chain
    /// into a spike, and seeded from the contact point so replays match.
    fn apply_debris_impacts(&mut self, impacts: Vec<(Vec3, f32)>) {
        if impacts.is_empty() {
            return;
        }
        let Some(destruction) = &mut self.destruction else {
            return;
        };
        let mut applied = 0;
        for (point, speed) in impacts {
            if applied >= 2 {
                break;
            }
            let energy = (speed * speed * 8.0).clamp(2_000.0, 40_000.0);
            let seed = ((point.x * 13.0 + point.y * 7.0 + point.z * 11.0) as i32 as u32)
                ^ ((speed * 100.0) as u32);
            let result = destruction.explode(
                &mut self.world,
                point,
                Vec3::Y,
                4,
                seed,
                ImpactProfile::explosive(energy, 1.0),
            );
            if result.hit {
                applied += 1;
            }
        }
    }

    /// Bakes settled debris back into the voxel world. Sleeping pieces stop
    /// costing a body and, more importantly, become solid again: a second shot
    /// can break the rubble on the floor instead of passing through it.
    fn bake_debris(&mut self) {
        let Some(destruction) = &mut self.destruction else {
            return;
        };
        let bake = destruction.bake_sleeping(&mut self.world, self.player.eye(), 4);
        if bake.pieces == 0 {
            return;
        }
        self.baked_pieces += bake.pieces;
        self.baked_voxels += bake.voxels;
        self.debris_voxels_dirty = true;
        self.pending_edits.extend(bake.dirty_bricks);
    }

    fn flush_world_edits(&mut self) {
        if self.pending_edits.is_empty() {
            return;
        }
        let edits = std::mem::take(&mut self.pending_edits);
        self.frame_dirty_bricks += edits.len();
        self.flushed_dirty_bricks += edits.len();
        self.sectors.mark_dirty_bricks(&edits);
        let (Some(gpu), Some(renderer)) = (&self.gpu, &mut self.renderer) else {
            return;
        };
        renderer.apply_world_edits(gpu, &self.world, &edits);
    }

    /// Creates sectors around the player and debris, applies finished greedy
    /// merges and drops sectors nothing is near. This is a no-op for the
    /// physics backend when no rebuild finished, so it can run every frame.
    fn update_sectors(&mut self) {
        let Some(destruction) = &mut self.destruction else {
            return;
        };
        let mut bodies = Vec::with_capacity(destruction.pieces().len());
        bodies.extend(destruction.pieces().iter().map(|piece| piece.position));
        let player = self.player.eye();
        self.sectors
            .update(&self.world, player, &bodies, destruction.physics_mut());
    }

    fn upload_debris(&mut self) {
        let (Some(gpu), Some(renderer), Some(destruction)) =
            (&self.gpu, &mut self.renderer, &self.destruction)
        else {
            return;
        };
        let pieces = destruction.pieces();
        // The renderer has 64 debris slots. Rank them by what is actually
        // visible instead of taking the newest: distance to the camera minus a
        // bonus for large pieces. Sleeping pieces are baked back into the world
        // by the destruction system, so a piece losing its slot costs nothing.
        let camera = self.camera.position;
        let mut ranked: Vec<(f32, usize)> = pieces
            .iter()
            .enumerate()
            .map(|(index, piece)| {
                let size = (piece.local_max - piece.local_min).length();
                (piece.position.distance(camera) - size * 3.0, index)
            })
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        ranked.truncate(MAX_RENDER_DEBRIS.saturating_sub(RESERVED_SYNTH_SLOTS));
        ranked.sort_unstable_by_key(|&(_, index)| index);

        let mut views = Vec::with_capacity(MAX_RENDER_DEBRIS);
        let mut offset = 0u32;
        for &(_, index) in &ranked {
            let piece = &pieces[index];
            let (position, rotation) =
                destruction.interpolated_transform(piece, self.physics_alpha);
            let len = piece.grid_len() as u32;
            if offset as usize + len as usize > MAX_DEBRIS_VOXELS {
                break;
            }
            let model = Mat4::from_rotation_translation(rotation, position);
            views.push(DebrisView {
                model,
                inverse_model: model.inverse(),
                aabb_min: piece.local_min,
                aabb_max: piece.local_max,
                dims: piece.dims,
                voxel_offset: offset,
                voxels: &piece.voxels,
            });
            offset += len;
        }
        // Overlay counters: how many pieces the renderer actually traced, how
        // many the physics system simulates, and how many lost a slot (either
        // the 64-instance cap or the debris voxel budget). Without the split a
        // piece that stops being drawn looks like a renderer bug.
        let simulated = pieces.len();
        let drawn = views.len();

        // Weapon viewmodel, in-flight rockets and carbine tracers ride the same
        // instanced voxel grid, packed after the debris. They are rebuilt every
        // frame because their model matrices move; the voxel payload is tiny.
        let mut synth: Vec<SynthModel> = Vec::new();
        let mut synth_models: Vec<Mat4> = Vec::new();
        let forward = self.camera.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward);
        let bob = (self.weapon_time * 7.0).sin() * 0.008;
        let recoil = self.weapon_recoil;
        let gun_position = self.camera.position + forward * (0.75 - 0.08 * recoil) + right * 0.30
            - up * (0.22 - bob);
        // The voxel model is authored at real size; shrink it to a viewmodel so
        // a 1.6 m tube does not fill half the screen.
        let gun_scale = 0.22;
        let gun_rotation = Mat4::from_cols(
            right.extend(0.0) * gun_scale,
            up.extend(0.0) * gun_scale,
            forward.extend(0.0) * gun_scale,
            gun_position.extend(1.0),
        );
        synth.push(match self.weapon {
            WeaponKind::Bazooka => bazooka_model(),
            WeaponKind::Carbine | WeaponKind::Flamethrower => carbine_model(),
        });
        synth_models.push(gun_rotation);
        if self.muzzle_flash > 0.0 {
            let flicker = 0.8 + 0.4 * hash_unit(self.shot_sequence ^ 0x77aa_11cc);
            let muzzle_position =
                self.camera.position + forward * (0.98 - 0.08 * recoil) + right * 0.30 - up * 0.22;
            synth.push(muzzle_model());
            synth_models.push(
                Mat4::from_translation(muzzle_position)
                    * Mat4::from_scale(Vec3::splat(0.22 * flicker)),
            );
        }
        let rocket_template = (!self.rockets.is_empty()).then(rocket_model);
        for rocket in &self.rockets {
            let template = rocket_template.as_ref().expect("checked above");
            let direction = rocket.velocity.normalize_or_zero();
            let rotation = glam::Quat::from_rotation_arc(Vec3::Z, direction);
            synth.push(SynthModel {
                dims: template.dims,
                voxels: template.voxels.clone(),
            });
            synth_models.push(Mat4::from_rotation_translation(rotation, rocket.position));
        }
        let tracer_template = (!self.tracers.is_empty()).then(tracer_model);
        for tracer in &self.tracers {
            let template = tracer_template.as_ref().expect("checked above");
            let delta = tracer.to - tracer.from;
            let length = delta.length();
            if length <= 1.0e-4 {
                continue;
            }
            let rotation = glam::Quat::from_rotation_arc(Vec3::Z, delta / length);
            let center = tracer.from + delta * 0.5;
            let model = Mat4::from_scale_rotation_translation(
                Vec3::new(1.0, 1.0, length / (2.0 * shatterbox_voxel::VOXEL_SIZE_M)),
                rotation,
                center,
            );
            synth.push(SynthModel {
                dims: template.dims,
                voxels: template.voxels.clone(),
            });
            synth_models.push(model);
        }
        for (model, template) in synth_models.iter().zip(synth.iter()) {
            let len = template.voxels.len() as u32;
            if offset as usize + len as usize > MAX_DEBRIS_VOXELS {
                break;
            }
            views.push(template.view(*model, offset));
            offset += len;
        }
        let has_synthetic = !synth.is_empty();
        // The voxel buffer is packed in the order of `ranked`, so when the set
        // of uploaded pieces changes while the *voxels* did not (a piece wins or
        // loses a slot as the camera moves), every offset shifts and the old
        // contents would be read as another piece's voxels - pieces then show
        // flipped normals and foreign materials. Force a rewrite whenever the
        // layout changes, not only when the destruction system says so.
        let mut layout = 0x9e37_79b9_7f4a_7c15_u64;
        for &(_, index) in &ranked {
            layout ^= index as u64;
            layout = layout.wrapping_mul(0x0100_0000_01b3);
        }
        let layout_changed = layout != self.debris_layout;
        self.debris_layout = layout;
        let voxels_changed =
            std::mem::take(&mut self.debris_voxels_dirty) || layout_changed || has_synthetic;
        renderer.upload_debris(gpu, &views, voxels_changed);
        self.debris_dropped = simulated.saturating_sub(drawn);
    }

    /// Fires a rocket from the eye along the aim ray. Destruction happens on
    /// impact (see [`step_rockets`]) rather than instantly at a ray hit, so a
    /// corner, a railing or a prop in the way catches the blast first.
    fn fire_shot(&mut self) {
        if self.rockets.len() >= MAX_ROCKETS {
            return;
        }
        let direction = self.camera.forward();
        let seed = self.shot_sequence;
        self.shot_sequence = self.shot_sequence.wrapping_add(1);
        // Blast size is 0.7..1.4x of the base and deterministic in the command
        // seed. `SHATTERBOX_BLAST_VARIATION=0` pins it to 1.0 for A/B.
        let variation = std::env::var("SHATTERBOX_BLAST_VARIATION").ok().as_deref() != Some("0");
        let scale = blast_scale(variation, seed);
        tracing::debug!(
            target: "shatterbox::weapon",
            seed,
            scale = format_args!("{scale:.3}"),
            "rocket launched"
        );
        self.rockets.push(Rocket {
            position: self.camera.position + direction * 0.6,
            velocity: direction * ROCKET_SPEED_MPS,
            age: 0.0,
            seed,
            scale,
        });
        self.weapon_recoil = 1.0;
        self.muzzle_flash = MUZZLE_FLASH_SECONDS;
        self.shots += 1;
        self.last_shot = Some(ShotSummary {
            hit: false,
            removed: 0,
            pieces: 0,
        });
    }

    /// Integrates every rocket and detonates the ones that hit geometry.
    fn step_rockets(&mut self, dt: f32) {
        if self.rockets.is_empty() {
            return;
        }
        let step = dt.clamp(0.0, 0.05);
        let mut detonations: Vec<(Vec3, Vec3, u32, f32)> = Vec::new();
        let mut trails: Vec<(Vec3, u32)> = Vec::new();
        for rocket in &mut self.rockets {
            rocket.age += step;
            rocket.velocity.y += ROCKET_GRAVITY_MPS2 * step;
            let travel = rocket.velocity * step;
            let distance = travel.length();
            if distance <= 1.0e-5 {
                continue;
            }
            let direction = travel / distance;
            // Exhaust trail: thin, short-lived puffs left behind the projectile.
            // Every other tick is enough at 60 Hz; the cloud still reads as a
            // continuous ribbon and costs half the particles.
            if self.stats.frame_index.is_multiple_of(2) {
                trails.push((rocket.position, rocket.seed));
            }
            if let Some(hit) = self.world.raycast(rocket.position, direction, distance) {
                // DDA reports a zero face normal if the ray starts inside a
                // solid (for example a rocket launched from a tight doorway).
                // The incoming ray direction is then the only reliable outward
                // face; without it the blast's surface normal collapses to +Y
                // and the wall push cancels.
                let hit_normal = if hit.normal == glam::IVec3::ZERO {
                    -direction
                } else {
                    hit.normal.as_vec3()
                };
                // Explode slightly outside the surface so the sphere is not
                // half-buried: a rocket hitting a wall must open the wall.
                let point = (hit.position.as_vec3() + Vec3::splat(0.5)) * 0.125 + hit_normal * 0.07;
                tracing::debug!(
                    target: "shatterbox::aim",
                    start = ?rocket.position.to_array(),
                    direction = ?direction.to_array(),
                    travel = distance,
                    voxel = ?hit.position.to_array(),
                    normal = ?hit_normal.to_array(),
                    point = ?point.to_array(),
                    "rocket impact"
                );
                detonations.push((point, hit_normal, rocket.seed, rocket.scale));
                rocket.age = ROCKET_LIFETIME_S;
            } else {
                rocket.position += travel;
            }
        }
        for (point, seed) in trails {
            self.spawn_trail(point, seed);
        }
        for (point, normal, seed, scale) in detonations {
            self.detonate(point, normal, seed, scale);
        }
        self.rockets.retain(|rocket| rocket.age < ROCKET_LIFETIME_S);
    }

    /// Ages the blast flashes and re-uploads the light rig while any is alive.
    /// The flash decays with a squared falloff so the first 100 ms are bright and
    /// the tail is short, which is what makes a hit read as an explosion instead
    /// of a light switch.
    fn update_explosions(&mut self, dt: f32) {
        let spark_alive = self.spark.is_some_and(|(_, age)| age < 0.12);
        if self.explosions.is_empty() && self.rockets.is_empty() && !spark_alive {
            return;
        }
        for explosion in &mut self.explosions {
            explosion.age += dt;
        }
        self.explosions
            .retain(|explosion| explosion.age < EXPLOSION_FLASH_SECONDS);
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        if self.explosions.is_empty() && self.rockets.is_empty() && !spark_alive {
            renderer.set_lights(&self.lights);
            return;
        }
        let mut scratch = std::mem::take(&mut self.light_scratch);
        scratch.clear();
        scratch.extend_from_slice(&self.lights);
        for explosion in self.explosions.iter().rev().take(2) {
            let t = 1.0 - (explosion.age / EXPLOSION_FLASH_SECONDS).clamp(0.0, 1.0);
            let intensity = EXPLOSION_LIGHT_INTENSITY * t * t;
            scratch.push(PointLight {
                position: explosion.position.to_array(),
                radius: EXPLOSION_LIGHT_RADIUS,
                color: [1.0, 0.62, 0.28],
                intensity,
            });
        }
        // Projectile tail lights and the carbine impact spark come after the
        // flashes, so a blast always wins the last free slots.
        for rocket in self.rockets.iter().rev().take(2) {
            scratch.push(PointLight {
                position: rocket.position.to_array(),
                radius: 5.0,
                color: [1.0, 0.55, 0.2],
                intensity: 3.0,
            });
        }
        if let Some((position, age)) = self.spark
            && age < 0.12
        {
            let t = 1.0 - (age / 0.12).clamp(0.0, 1.0);
            scratch.push(PointLight {
                position: position.to_array(),
                radius: 3.0,
                color: [1.0, 0.78, 0.42],
                intensity: 9.0 * t,
            });
        }
        // The uniform holds 12 lights; the arena rig already uses most of them,
        // so the flash is appended last and simply dropped if the list overflows.
        renderer.set_lights(&scratch);
        self.light_scratch = scratch;
    }

    /// Spawns the smoke and dust billboards for a blast. Every random draw comes
    /// from `seed`, so replaying the same command log produces the same cloud.
    fn spawn_smoke(&mut self, point: Vec3, seed: u32, scale: f32) {
        let radius =
            (ROCKET_RADIUS_VOXELS as f32 * scale * shatterbox_voxel::VOXEL_SIZE_M).max(0.6);
        let smoke_count = ((SMOKE_PUFFS as f32) * scale).round() as usize;
        let dust_count = ((DUST_PUFFS as f32) * scale).round() as usize;
        // Dark, hot smoke rises inside the crater volume; pale dust is thrown
        // sideways along the ground at roughly twice the speed and settles.
        for index in 0..smoke_count {
            let mut puff = make_puff(
                seed,
                0x51ed_270b,
                index as u32,
                point,
                radius,
                SMOKE_LIFE_S,
                4.6,
                [0.34, 0.33, 0.32, 0.55],
                1.7,
                1.1,
                0.95,
            );
            puff.kind = PuffKind::Smoke;
            // The fireball lights the first moments; the colour then ramps to
            // the grey `tint` and finally toward pale dust as the envelope fades.
            puff.hot = [0.30, 0.13, 0.05];
            puff.density = 1.0;
            self.smoke.push(puff);
        }
        for index in 0..dust_count {
            let mut puff = make_puff(
                seed,
                0x2c1b_3e5f,
                index as u32,
                point,
                radius,
                DUST_LIFE_S,
                6.5,
                [0.42, 0.37, 0.29, 0.34],
                3.2,
                -0.5,
                1.3,
            );
            puff.kind = PuffKind::Dust;
            // Dust is lit by the same flash but scatters it much less.
            puff.hot = [0.30, 0.25, 0.18];
            puff.density = 0.55;
            self.smoke.push(puff);
        }
    }

    /// One short, thin puff behind a flying rocket. Deliberately much weaker
    /// than the blast cloud: the trail marks the path without hiding the scene.
    fn spawn_trail(&mut self, point: Vec3, seed: u32) {
        let index = self.smoke.len() as u32;
        let mut puff = make_puff(
            seed,
            0x7f4a_1c2d,
            index,
            point,
            0.55,
            (0.35, 0.7),
            0.6,
            [0.55, 0.55, 0.56, 0.16],
            4.0,
            0.9,
            1.4,
        );
        puff.kind = PuffKind::Trail;
        // Still fire-lit at the nozzle; cools within a few frames.
        puff.hot = [0.85, 0.45, 0.18];
        puff.density = 0.25;
        self.smoke.push(puff);
    }

    /// Spawns one short billboard per dust grain: voxels that were too small to
    /// become rigid bodies still have to be seen leaving the crater, but they
    /// must not cost a Jolt body. Their world positions come from the
    /// destruction system, so the dust is where the matter actually was.
    fn spawn_dust_grains(&mut self, points: &[Vec3], seed: u32) {
        if !self.dust_grains || points.is_empty() {
            return;
        }
        for (index, point) in points.iter().enumerate() {
            let index = index as u32;
            let grain = 0.030 + 0.025 * hash_unit(seed ^ index.wrapping_mul(0x9e37_79b9));
            let mut puff = make_puff(
                seed,
                0x7ad1_c3a5,
                index,
                *point,
                0.12,
                (0.35, 0.75),
                1.6,
                [0.42, 0.36, 0.28, 0.45],
                5.5,
                -1.6,
                0.35,
            );
            // Grains are much smaller than a smoke puff, so the blast-sized
            // radii the helper derived are replaced.
            puff.start_radius = grain;
            puff.end_radius = grain * (1.6 + hash_unit(seed ^ index));
            puff.radius = grain;
            puff.kind = PuffKind::Grain;
            puff.density = 0.4;
            puff.hot = [0.42, 0.36, 0.28];
            self.smoke.push(puff);
        }
        self.dust_particles += points.len();
    }

    /// Advances the smoke/dust simulation and pushes the current billboard list
    /// to the renderer. Puffs grow fast at first and then settle, and fade in
    /// over the first few ticks so a spawn does not pop.
    fn update_smoke(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        self.smoke_field.step(dt, &self.world);
        // Light arena wind, m/s²: weak enough not to read as a fan.
        const WIND: Vec3 = Vec3::new(0.55, 0.0, 0.30);
        let world = &self.world;
        for puff in &mut self.smoke {
            puff.age += dt;
            let drag = (1.0 - puff.drag * dt).clamp(0.0, 1.0);
            puff.velocity.x *= drag;
            puff.velocity.z *= drag;
            let t = (puff.age / puff.life).clamp(0.0, 1.0);
            let heat = 1.0 - t;
            // Hot gas keeps rising; as it cools, buoyancy drops and the puff
            // settles into the ambient drift instead of shooting upward.
            puff.velocity.y = puff.velocity.y * drag + puff.rise * dt * (0.25 + 1.5 * heat);
            puff.velocity += WIND * (0.35 + 0.65 * heat) * dt;
            // Cheap divergence-free-ish swirl: three out-of-phase sines keyed by
            // the puff seed. Enough to break the perfect radial expansion that
            // made every cloud look stamped out.
            let swirl = Vec3::new(
                (puff.position.y * 1.7 + puff.seed + puff.age * 1.9).sin(),
                0.6 * (puff.position.x * 1.3 + puff.seed * 0.7 + puff.age * 1.4).sin(),
                (puff.position.y * 1.1 + puff.seed * 1.3 + puff.age * 1.6).cos(),
            );
            puff.velocity += swirl * (0.55 * heat) * dt;

            let next = puff.position + puff.velocity * dt;
            let solid = |p: Vec3| {
                world
                    .voxel(
                        (p.x / shatterbox_voxel::VOXEL_SIZE_M).floor() as i32,
                        (p.y / shatterbox_voxel::VOXEL_SIZE_M).floor() as i32,
                        (p.z / shatterbox_voxel::VOXEL_SIZE_M).floor() as i32,
                    )
                    .is_solid()
            };
            // Smoke cannot enter a wall. Slide along it instead of stopping
            // dead: try each axis on its own and kill only the blocked
            // component, so a puff hugging a facade runs along it.
            let mut moved = puff.position;
            if !solid(next) {
                moved = next;
            } else {
                let mut candidate = puff.position;
                candidate.x = next.x;
                if solid(candidate) {
                    puff.velocity.x *= -0.2;
                } else {
                    moved.x = next.x;
                }
                let mut candidate = puff.position;
                candidate.z = next.z;
                if solid(candidate) {
                    puff.velocity.z *= -0.2;
                } else {
                    moved.z = next.z;
                }
                let mut candidate = puff.position;
                candidate.y = next.y;
                if solid(candidate) {
                    puff.velocity.y = -puff.velocity.y.abs() * 0.1;
                } else {
                    moved.y = next.y;
                }
            }
            puff.position = moved;

            // `1 - (1 - t)²` reaches 75 % of the final radius at half the
            // lifetime: the fireball expands, then the cloud slows down.
            let grow = 1.0 - (1.0 - t) * (1.0 - t);
            puff.radius = puff.start_radius + (puff.end_radius - puff.start_radius) * grow;
        }
        self.smoke.retain(|puff| puff.age < puff.life);
        // 200–600 is the agreed FX budget. Past the cap the oldest clouds go
        // first: young puffs carry the blast silhouette, old ones are already
        // thin and half transparent.
        const MAX_PUFFS: usize = 560;
        if self.smoke.len() > MAX_PUFFS {
            self.smoke
                .sort_unstable_by(|a, b| (a.age / a.life).total_cmp(&(b.age / b.life)));
            self.smoke.truncate(MAX_PUFFS);
        }
        self.smoke_peak = self.smoke_peak.max(self.smoke.len());

        let mut scratch = std::mem::take(&mut self.particle_scratch);
        scratch.clear();
        for puff in &self.smoke {
            let t = (puff.age / puff.life).clamp(0.0, 1.0);
            let fade_in = (t / 0.06).min(1.0);
            // Squared fade-out: the cloud thins gradually instead of switching
            // off at full opacity, which reads as a deleted sprite.
            let fade_out = 1.0 - ((t - 0.4) / 0.6).clamp(0.0, 1.0);
            let opacity = puff.tint[3]
                * fade_in
                * fade_out
                * fade_out
                // The local volume owns the visible density; puffs stay only as a
                // hot fire accent. 0.45 still drew opaque sprite balls that cut
                // through walls, which read as the "smoke ignores voxels" bug.
                * if self.smoke_sim { 0.06 } else { 1.0 };
            if opacity <= 0.004 || puff.radius <= 1.0e-3 {
                continue;
            }
            // Cool from the fire-lit colour to the puff's grey, then let dense
            // smoke drift toward pale dust over its last third.
            let cool = ((t - 0.03) / 0.45).clamp(0.0, 1.0);
            let cool = cool * cool * (3.0 - 2.0 * cool);
            let lift = if puff.kind == PuffKind::Smoke {
                ((t - 0.6) / 0.4).clamp(0.0, 1.0) * 0.10
            } else {
                0.0
            };
            let mix = |hot: f32, cold: f32| hot + (cold - hot) * cool + lift;
            scratch.push(Particle {
                position: puff.position.to_array(),
                radius: puff.radius,
                color: [
                    mix(puff.hot[0], puff.tint[0]),
                    mix(puff.hot[1], puff.tint[1]),
                    mix(puff.hot[2], puff.tint[2]),
                    opacity,
                ],
                // x = extinction, y = noise offset, z = emitter kind,
                // w = remaining heat (blast-flash response).
                params: [puff.density, puff.seed, puff.kind.code(), 1.0 - t],
            });
        }
        // Back-to-front by camera distance. A few hundred large clouds sorted on
        // the CPU is cheaper and far steadier than per-pixel order-independent
        // transparency, and it stops puffs swapping order (flicker) on camera
        // movement.
        let camera = self.camera.position;
        scratch.sort_unstable_by(|a, b| {
            let da = Vec3::from_array(a.position).distance_squared(camera);
            let db = Vec3::from_array(b.position).distance_squared(camera);
            db.total_cmp(&da)
        });
        if let (Some(gpu), Some(renderer)) = (&self.gpu, &mut self.renderer) {
            renderer.set_particles(&scratch);
            renderer.set_smoke_density(&gpu.queue, self.smoke_field.bytes());
        }
        self.update_gpu_smoke(dt);
        self.particle_scratch = scratch;
    }

    /// Maintains the local GPU smoke box and feeds it the emitters collected by
    /// explosions and burning voxels. When `SHATTERBOX_SMOKE_SIM=0` (or the
    /// dynamic-smoke bit is off) the coarse CPU field above is uploaded instead.
    fn update_gpu_smoke(&mut self, dt: f32) {
        let (Some(gpu), Some(renderer)) = (self.gpu.as_ref(), self.renderer.as_mut()) else {
            return;
        };
        if !self.smoke_sim || !renderer.effects().dynamic_smoke {
            if renderer.smoke_active() {
                renderer.clear_smoke_box();
            }
            self.smoke_box = None;
            self.smoke_emitters.clear();
            renderer.set_smoke_density(&gpu.queue, self.smoke_field.bytes());
            return;
        }
        for (_, ttl) in &mut self.smoke_emitters {
            *ttl -= dt;
        }
        self.smoke_emitters.retain(|(_, ttl)| *ttl > 0.0);
        let sources: Vec<SmokeSource> = self
            .smoke_emitters
            .iter()
            .take(4)
            .map(|(source, _)| *source)
            .collect();
        let half = Vec3::from_array(shatterbox_render::SMOKE_HALF_EXTENTS);
        let build = |world: &VoxelWorld, center: Vec3| {
            crate::smoke_box::build_mask(world, center - half, SMOKE_SPACING, SMOKE_GRID)
        };

        if self.smoke_box.is_none() {
            if let Some(source) = sources.first().copied() {
                let center = source.position + Vec3::new(0.0, 4.0, 0.0);
                let started = std::time::Instant::now();
                let mask = build(&self.world, center);
                self.smoke_mask_ms = started.elapsed().as_secs_f32() * 1000.0;
                renderer.set_smoke_box(&gpu.queue, center, &mask);
                self.smoke_box = Some(SmokeBox {
                    center,
                    ttl: 6.0,
                    dirty_age: 0.0,
                });
            }
        }

        // Follow the active source when a fire crawls across the courtyard.
        let mut recenter = None;
        if let (Some(source), Some(bx)) = (sources.first(), self.smoke_box.as_ref()) {
            let desired = source.position + Vec3::new(0.0, 4.0, 0.0);
            if (desired - bx.center).length() > 6.0 {
                recenter = Some(desired);
            }
        }
        if let Some(center) = recenter {
            let started = std::time::Instant::now();
            let mask = build(&self.world, center);
            self.smoke_mask_ms = started.elapsed().as_secs_f32() * 1000.0;
            renderer.set_smoke_box(&gpu.queue, center, &mask);
            self.smoke_box = Some(SmokeBox {
                center,
                ttl: 6.0,
                dirty_age: 0.0,
            });
        }

        if let Some(bx) = &mut self.smoke_box {
            if sources.is_empty() {
                bx.ttl -= dt;
            } else {
                bx.ttl = 6.0;
            }
            if self.frame_dirty_bricks > 0 {
                bx.dirty_age += dt;
            }
            if bx.dirty_age >= 0.05 {
                bx.dirty_age = 0.0;
                let started = std::time::Instant::now();
                let mask = build(&self.world, bx.center);
                self.smoke_mask_ms = started.elapsed().as_secs_f32() * 1000.0;
                renderer.update_smoke_obstacles(&gpu.queue, &mask);
            }
        }

        if self.smoke_box.as_ref().is_some_and(|bx| bx.ttl <= 0.0) {
            renderer.clear_smoke_box();
            self.smoke_box = None;
            self.smoke_emitters.clear();
        }
        renderer.set_smoke_sources(&sources);
        // The GPU volume owns the smoke while it is alive; zero the coarse fog
        // field so the same plume is not integrated twice.
        renderer.set_smoke_density(&gpu.queue, &ZERO_DENSITY);
    }

    /// Stores an emitter for the GPU smoke box; it keeps emitting for `ttl`
    /// seconds. Only the newest eight are kept, four reach the shader.
    fn push_smoke_emitter(&mut self, source: SmokeSource, ttl: f32) {
        if self.smoke_emitters.len() >= 8 {
            self.smoke_emitters.remove(0);
        }
        self.smoke_emitters.push((source, ttl.max(0.05)));
    }

    /// Buckets the sizes of newly created debris pieces for `report.json`.
    fn record_piece_sizes(&mut self, sizes: &[u32]) {
        for &size in sizes {
            let bucket = match size {
                0 | 1 => 0,
                2..=4 => 1,
                5..=7 => 2,
                8..=16 => 3,
                17..=64 => 4,
                65..=256 => 5,
                _ => 6,
            };
            self.debris_histogram[bucket] += 1;
        }
    }

    /// Shot statistics per impact profile, in the shape `report.json` wants.
    fn impacts_report(&self) -> crate::scenario::ImpactsReport {
        let convert = |tally: ImpactTally| crate::scenario::ImpactTallyReport {
            shots: tally.shots,
            pieces: tally.pieces,
            average_speed: tally.average_speed(),
            max_speed: tally.max_speed,
        };
        crate::scenario::ImpactsReport {
            explosive: convert(self.impact_explosive),
            kinetic: convert(self.impact_kinetic),
        }
    }

    fn detonate(&mut self, point: Vec3, normal: Vec3, seed: u32, scale: f32) {
        let (radius, energy) = blast_geometry(scale);
        tracing::debug!(
            target: "shatterbox::weapon",
            point = ?point.to_array().map(|v| (v * 1000.0).round() / 1000.0),
            normal = ?normal.to_array().map(|v| (v * 1000.0).round() / 1000.0),
            seed,
            scale = format_args!("{scale:.3}"),
            radius,
            "detonate"
        );
        self.explosions.push(Explosion {
            position: point,
            age: 0.0,
        });
        self.spawn_smoke(point, seed, scale);
        self.smoke_field
            .add_source(point, radius.max(2) as f32 * 0.125 * 1.8, 0.6);
        self.push_smoke_emitter(
            SmokeSource {
                position: point,
                radius: 2.6,
                // Dense enough to read as a volumetric cloud inside a room,
                // not a faint haze. The raymarch extinction turns this into
                // ~80 % opacity over the first metre.
                density: 9.0,
                // Smoke, not a fireball: the orange glow belongs to the first
                // ~0.2 s, the body is soot that cools to grey.
                temperature: 0.8,
                updraft: 1.8,
            },
            // Keep feeding the plume for 2.5 s. A 0.9 s source left a puff that
            // the flow carried out of the box before it could fill the room.
            2.5,
        );
        self.weapon_shake = WEAPON_SHAKE_SECONDS * scale;
        let Some(destruction) = &mut self.destruction else {
            return;
        };
        let started = std::time::Instant::now();
        let result = destruction.explode(
            &mut self.world,
            point,
            normal,
            radius,
            seed,
            ImpactProfile::explosive(energy, scale),
        );
        let explode_ms = started.elapsed().as_secs_f32() * 1000.0;
        // Keep the live debris under the Jolt budget without deleting matter:
        // the farthest pieces are baked back into the world, nearest-first
        // rendering keeps the close-up rubble simulated.
        let bake = destruction.retire_into_world(
            &mut self.world,
            self.camera.position,
            MAX_LIVE_DEBRIS_PIECES,
        );
        let retire_ms = started.elapsed().as_secs_f32() * 1000.0 - explode_ms;
        // A blast that costs a whole frame is a stall the player feels, so log it
        // with the split that explains it: carve+cluster+body creation vs baking
        // the far pieces back into the world.
        if explode_ms + retire_ms > 4.0 {
            tracing::info!(
                target: "shatterbox::weapon",
                explode_ms,
                retire_ms,
                pieces = result.pieces,
                removed = result.removed,
                baked = bake.pieces,
                "slow detonation"
            );
        }
        if bake.pieces > 0 {
            self.debris_voxels_dirty = true;
            self.baked_pieces += bake.pieces;
            self.baked_voxels += bake.voxels;
            self.pending_edits.extend(bake.dirty_bricks.iter().copied());
        }
        self.record_piece_sizes(&result.piece_sizes);
        self.impact_explosive.record(&result);
        self.spawn_dust_grains(&result.dust_points, seed);
        if result.removed > 0 {
            self.debris_voxels_dirty = true;
        }
        self.removed_total += result.removed;
        tracing::debug!(
            target: "shatterbox::aim",
            removed = result.removed,
            pieces = result.pieces,
            dust = result.dust_voxels,
            dirty_bricks = result.dirty_bricks.len(),
            "explode result"
        );
        self.pending_edits
            .extend(result.dirty_bricks.iter().copied());
        self.last_shot = Some(ShotSummary {
            hit: true,
            removed: result.removed,
            pieces: result.pieces,
        });
    }

    /// Flamethrower: three deterministic rays in a narrow cone. A hit on a
    /// material with flammability > 0 starts a burn at that voxel; the burn
    /// spreads to its six neighbours and finally removes the charred voxel.
    /// The cone is deliberately short (8 m) and the cooldown 6 ticks, so the
    /// weapon paints fire instead of instantly deleting a wall.
    fn flame_shot(&mut self) {
        let origin = self.camera.position;
        let forward = self.camera.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward);
        let seed = carbine_seed(self.sim_tick, self.shot_sequence);
        self.shot_sequence = self.shot_sequence.wrapping_add(1);
        self.muzzle_flash = MUZZLE_FLASH_SECONDS;
        self.weapon_recoil = 0.12;
        self.carbine_cooldown_ticks = 6;
        let mut hits = 0u32;
        let mut ignited = 0u32;
        for i in 0..3u32 {
            let sx = (hash_unit(seed ^ (0x9e37_79b9 + i)) - 0.5) * 0.10;
            let sy = (hash_unit(seed ^ (0x85eb_ca6b + i)) - 0.5) * 0.10;
            let direction = (forward + right * sx + up * sy).normalize_or_zero();
            let Some(hit) = self.world.raycast(origin, direction, 8.0) else {
                continue;
            };
            hits += 1;
            let cell = hit.position;
            let material = self.world.voxel(cell.x, cell.y, cell.z).material();
            let point = (cell.as_vec3() + Vec3::splat(0.5)) * shatterbox_voxel::VOXEL_SIZE_M;
            self.smoke_field.add_source(point, 1.8, 0.05);
            if self.world.palette().material(material).flammability > 0.05 {
                self.ignite(cell, 2.2);
                ignited += 1;
            }
        }
        let _ = (hits, ignited);
    }

    /// Adds a voxel to the burn list. The list is capped so a flamethrower
    /// sweep cannot turn the whole room into a per-tick solver workload.
    fn ignite(&mut self, cell: IVec3, seconds: f32) {
        const MAX_BURNING: usize = 192;
        if self.burning.len() >= MAX_BURNING {
            return;
        }
        if self.burning.iter().any(|burn| burn.cell == cell) {
            return;
        }
        self.burning.push(Burning {
            cell,
            fuel: seconds,
            spread: 0.45,
        });
    }

    /// Ticks the burn list: spread to flammable neighbours, emit smoke, and
    /// remove a voxel once its fuel is gone (the only world edit it performs).
    fn step_burning(&mut self, dt: f32) {
        const SPREAD_INTERVAL: f32 = 0.45;
        let mut spread = Vec::new();
        let mut finished = Vec::new();
        let mut flame_points = Vec::new();
        for index in 0..self.burning.len() {
            let cell = self.burning[index].cell;
            self.burning[index].fuel -= dt;
            self.burning[index].spread -= dt;
            if self.burning[index].spread <= 0.0 {
                self.burning[index].spread = SPREAD_INTERVAL;
                for offset in [
                    IVec3::new(1, 0, 0),
                    IVec3::new(-1, 0, 0),
                    IVec3::new(0, 1, 0),
                    IVec3::new(0, -1, 0),
                    IVec3::new(0, 0, 1),
                    IVec3::new(0, 0, -1),
                ] {
                    let neighbour = cell + offset;
                    let material = self
                        .world
                        .voxel(neighbour.x, neighbour.y, neighbour.z)
                        .material();
                    if self.world.palette().material(material).flammability > 0.25
                        && !self.burning.iter().any(|other| other.cell == neighbour)
                    {
                        spread.push(neighbour);
                    }
                }
            }
            if self.burning[index].fuel <= 0.0 {
                finished.push(cell);
            }
            if (index as u64 + self.sim_tick).is_multiple_of(6) {
                flame_points.push(cell);
            }
        }
        for cell in spread {
            self.ignite(cell, 1.6);
        }
        for cell in finished {
            if self.world.voxel(cell.x, cell.y, cell.z).is_solid() {
                self.world.set_voxel(cell.x, cell.y, cell.z, Voxel::AIR);
                self.pending_edits.push(VoxelWorld::voxel_to_brick(cell));
                self.removed_total += 1;
            }
            let point = (cell.as_vec3() + Vec3::splat(0.5)) * shatterbox_voxel::VOXEL_SIZE_M;
            self.smoke_field.add_source(point, 2.2, 0.12);
        }
        self.burning.retain(|burn| burn.fuel > 0.0);
        for cell in flame_points {
            let point = (cell.as_vec3() + Vec3::splat(0.5)) * shatterbox_voxel::VOXEL_SIZE_M;
            self.push_smoke_emitter(
                SmokeSource {
                    position: point,
                    radius: 3.0,
                    density: 10.0,
                    // Hot enough for the yellow-white core of the blackbody
                    // ramp; the flame now follows the simulated flow instead of
                    // being an orange sprite glued to the voxel.
                    temperature: 1.1,
                    updraft: 3.2,
                },
                0.6,
            );
            let index = self.smoke.len() as u32;
            let seed = self.sim_tick as u32 ^ cell.x as u32 ^ cell.z as u32;
            let mut puff = make_puff(
                seed,
                0x51ed_270b,
                index,
                point,
                0.28,
                (0.22, 0.5),
                0.7,
                [1.0, 0.42, 0.10, 0.85],
                3.0,
                1.9,
                1.2,
            );
            puff.kind = PuffKind::Smoke;
            puff.hot = [1.0, 0.58, 0.16];
            puff.density = 0.55;
            self.smoke.push(puff);
            self.smoke_field.add_source(point, 1.4, 0.015);
        }
    }

    /// Carbine: deterministic spread, a 2-voxel capsule and a tracer. The ray is
    /// cast once for the tracer endpoint and then `fire` carves along the same
    /// direction, so the hole and the drawn tracer agree.
    fn carbine_shot(&mut self) {
        let origin = self.camera.position;
        let forward = self.camera.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward);
        let seed = carbine_seed(self.sim_tick, self.shot_sequence);
        self.shot_sequence = self.shot_sequence.wrapping_add(1);
        let spread_x = (hash_unit(seed ^ 0x51ed_270b) - 0.5) * 2.0 * CARBINE_SPREAD_RAD;
        let spread_y = (hash_unit(seed ^ 0xa511_e9b3) - 0.5) * 2.0 * CARBINE_SPREAD_RAD;
        let direction = (forward + right * spread_x + up * spread_y).normalize_or_zero();
        let impact = self.world.raycast(origin, direction, 256.0);
        let to = impact.as_ref().map_or(origin + direction * 256.0, |hit| {
            (hit.position.as_vec3() + Vec3::splat(0.5)) * shatterbox_voxel::VOXEL_SIZE_M
        });
        self.tracers.push(Tracer {
            from: origin + forward * 0.55 + right * 0.18 - up * 0.12,
            to,
            age: 0.0,
        });
        self.spark = impact
            .as_ref()
            .map(|hit| ((hit.position.as_vec3() + Vec3::splat(0.5)) * 0.125, 0.0));
        self.muzzle_flash = MUZZLE_FLASH_SECONDS;
        self.weapon_recoil = 0.35;
        self.carbine_cooldown_ticks = CARBINE_COOLDOWN_TICKS;
        let Some(destruction) = &mut self.destruction else {
            return;
        };
        let result = destruction.fire(
            &mut self.world,
            origin,
            direction,
            CARBINE_RADIUS_VOXELS,
            seed,
            ImpactProfile::kinetic(CARBINE_ENERGY_JOULES),
        );
        // Per-round destruction cost is bounded by design: a 2-voxel capsule,
        // 2% of the rocket's energy, chips merged to the minimum piece size and
        // a dust cap. At 10 rounds per second the *live bodies* still pile up
        // over a long burst, so apply the same budget a rocket blast uses: the
        // farthest pieces bake into the world instead of accumulating.
        let bake = destruction.retire_into_world(&mut self.world, origin, MAX_LIVE_DEBRIS_PIECES);
        if bake.pieces > 0 {
            self.debris_voxels_dirty = true;
            self.baked_pieces += bake.pieces;
            self.baked_voxels += bake.voxels;
            self.pending_edits.extend(bake.dirty_bricks.iter().copied());
        }
        self.record_piece_sizes(&result.piece_sizes);
        self.impact_kinetic.record(&result);
        self.spawn_dust_grains(&result.dust_points, seed);
        if result.removed > 0 {
            self.debris_voxels_dirty = true;
        }
        self.removed_total += result.removed;
        self.pending_edits
            .extend(result.dirty_bricks.iter().copied());
        self.shots += 1;
        self.last_shot = Some(ShotSummary {
            hit: result.hit,
            removed: result.removed,
            pieces: result.pieces,
        });
    }

    /// Instant hit-scan carve, used by the `shot` weapon in scenarios. The
    /// interactive game only shoots rockets, but the harness needs a cheap
    /// deterministic way to open a wall without waiting for a projectile.
    fn carve_shot(&mut self) {
        let origin = self.camera.position;
        let direction = self.camera.forward();
        let seed = self.shot_sequence;
        self.shot_sequence = self.shot_sequence.wrapping_add(1);
        let Some(destruction) = &mut self.destruction else {
            return;
        };
        let result = destruction.fire(
            &mut self.world,
            origin,
            direction,
            FIRE_RADIUS_VOXELS,
            seed,
            // The harness `shot` is an instant rocket-sized carve, kept on the
            // explosive profile so old scenarios still compare.
            ImpactProfile::explosive(SHOT_ENERGY_JOULES, 1.0),
        );
        self.record_piece_sizes(&result.piece_sizes);
        self.impact_explosive.record(&result);
        self.spawn_dust_grains(&result.dust_points, seed);
        if result.removed > 0 {
            self.debris_voxels_dirty = true;
        }
        // Post-carve ground truth for the crosshair diagnostics: if the CPU ray
        // now escapes through the hole, the world data really was cut at the
        // crosshair and any visual offset lives in the upload/render path.
        let after = self
            .world
            .raycast(origin, direction, 256.0)
            .map(|hit| hit.distance);
        tracing::debug!(
            target: "shatterbox::aim",
            before = ?result.hit,
            removed = result.removed,
            after_distance = ?after,
            "carve_shot"
        );
        self.shots += 1;
        self.removed_total += result.removed;
        self.pending_edits
            .extend(result.dirty_bricks.iter().copied());
        self.last_shot = Some(ShotSummary {
            hit: result.hit,
            removed: result.removed,
            pieces: result.pieces,
        });
    }

    /// Camera used for the presented frame. Blast shake is rotational only and
    /// is applied to a copy: the aim camera (and therefore the next shot) never
    /// moves, so the crater stays under the crosshair.
    fn shaken_camera(&self) -> FreeCamera {
        let mut camera = self.camera.clone();
        if self.weapon_shake > 0.0 {
            let t = (self.weapon_shake / WEAPON_SHAKE_SECONDS).clamp(0.0, 1.0);
            let amplitude = t * t * 0.035;
            camera.yaw += (self.weapon_time * 37.0).sin() * amplitude;
            camera.pitch += (self.weapon_time * 29.0).cos() * amplitude * 0.8;
        }
        camera
    }

    /// Advances cosmetic weapon timers. Runs in both the interactive path and
    /// the scenario path, so harness screenshots show recoil, tracers and shake.
    /// The carbine cooldown is NOT touched here: it belongs to the fixed-step
    /// weapon sim (`step_weapons`), so it always ticks at 60 Hz.
    fn advance_weapon_animations(&mut self, dt: f32) {
        self.weapon_time += dt;
        self.weapon_recoil = (self.weapon_recoil - dt / WEAPON_RECOIL_SECONDS).max(0.0);
        self.muzzle_flash = (self.muzzle_flash - dt).max(0.0);
        self.weapon_shake = (self.weapon_shake - dt).max(0.0);
        for tracer in &mut self.tracers {
            tracer.age += dt;
        }
        self.tracers.retain(|tracer| tracer.age < TRACER_SECONDS);
        if let Some((_, age)) = &mut self.spark {
            *age += dt;
            if *age > 0.12 {
                self.spark = None;
            }
        }
    }

    /// One fixed 60 Hz step of the weapon simulation. The trigger state is fed
    /// by `update_controls` (interactive) or the scenario plan; both run this on
    /// the fixed grid so the fire rate and the per-round seed are independent
    /// of the frame rate. The carbine is an automatic: while the trigger is
    /// held it fires on its cooldown. The bazooka and fill are semi-automatic:
    /// one click, one shot — but a click that lands during the cooldown stays
    /// pending until the shot goes out, so a click never gets swallowed.
    fn step_weapons(&mut self) {
        self.sim_tick += 1;
        if self.carbine_cooldown_ticks > 0 {
            self.carbine_cooldown_ticks -= 1;
        }
        let wants_shot = self.trigger_pending
            || (self.trigger_held
                && matches!(self.weapon, WeaponKind::Carbine | WeaponKind::Flamethrower));
        if !wants_shot || self.carbine_cooldown_ticks > 0 {
            return;
        }
        match self.weapon {
            WeaponKind::Bazooka => self.fire_shot(),
            WeaponKind::Carbine => self.carbine_shot(),
            WeaponKind::Flamethrower => self.flame_shot(),
        }
        self.trigger_pending = false;
    }

    fn update_controls(&mut self, dt: f32) {
        self.advance_weapon_animations(dt);
        for (slot, action) in [
            (1usize, Action::Weapon1),
            (2, Action::Weapon2),
            (3, Action::Weapon3),
        ] {
            if self.input.state().pressed(action) {
                self.weapon = WeaponKind::from_slot(slot);
            }
        }
        if self.input.state().pressed(Action::ToggleFullscreen)
            && let Some(window) = &self.window
        {
            window.set_fullscreen(if window.fullscreen().is_some() {
                None
            } else {
                Some(winit::window::Fullscreen::Borderless(None))
            });
        }
        if self.input.state().pressed(Action::ToggleOverlay) {
            self.show_overlay = !self.show_overlay;
        }
        if self.input.state().pressed(Action::ResetCamera) {
            let (spawn, yaw, pitch) = self.scene.spawn();
            self.player.spawn_on_floor(&self.world, spawn);
            self.camera.position = self.player.eye();
            self.camera.yaw = yaw;
            self.camera.pitch = pitch;
        }
        if let Some(renderer) = &mut self.renderer {
            if self.input.state().pressed(Action::ToggleGovernor) {
                renderer.toggle_governor();
            }
            if self.input.state().pressed(Action::CycleRenderScale) {
                renderer.cycle_scale();
            }
            if self.input.state().pressed(Action::CycleUpscaler) {
                let upscaler = renderer.cycle_upscaler();
                tracing::info!("upscaler switched to {}", upscaler.name());
            }
            let step = if self.input.state().down(Action::Sprint) {
                2.0
            } else {
                0.5
            };
            if self.input.state().pressed(Action::TimeBackward) {
                renderer.set_time_of_day(renderer.time_of_day() - step);
            }
            if self.input.state().pressed(Action::TimeForward) {
                renderer.set_time_of_day(renderer.time_of_day() + step);
            }
        }

        // Apply look and movement before the shot rays are traced. Otherwise a
        // click on a frame with mouse motion fires from the previous yaw/pitch
        // while the renderer shows the new one, which reads as an off-centre
        // crater. The shot and the render then both use this exact camera.
        let mouse_delta = self.input.take_mouse_delta();
        if self.input.cursor_captured() {
            // A cursor grab (or the OS warp that follows it) reports one huge
            // motion sample. Applying it spins the view by a random angle, which
            // is exactly the "aim jumps sideways when I click" report. Real
            // flicks at 60+ fps are far below this threshold, so drop the outlier
            // instead of dragging the whole frame with it.
            if mouse_delta.length() <= MAX_LOOK_DELTA_PX {
                self.camera.look(mouse_delta);
            } else {
                tracing::debug!(
                    delta = format_args!("{:.1}", mouse_delta.length()),
                    "dropped cursor warp mouse delta"
                );
            }
        }

        let scroll = self.input.take_scroll();
        if scroll.abs() > f32::EPSILON {
            self.weapon = self.weapon.cycle(if scroll > 0.0 { 1 } else { -1 });
        }
        self.player
            .update(&self.world, dt, self.input.state(), self.camera.yaw);
        self.camera.position = self.player.eye();

        // Weapon simulation runs on the fixed 60 Hz grid like physics, so the
        // fire rate never tracks the frame rate. A click is latched into
        // `trigger_pending` and spent by the next step that can fire; holding
        // the button keeps the carbine firing on its cooldown.
        if self.input.state().pressed(Action::Fire) {
            self.trigger_pending = true;
        }
        self.trigger_held = self.input.state().down(Action::Fire);
        if self.input.state().pressed(Action::AltFire) {
            self.flame_shot();
        }
        self.weapon_accumulator = (self.weapon_accumulator + dt).min(0.25);
        let mut steps = 0;
        while self.weapon_accumulator >= FIXED_DT && steps < 4 {
            self.step_weapons();
            self.step_burning(FIXED_DT);
            self.weapon_accumulator -= FIXED_DT;
            steps += 1;
        }
        if steps == 4 {
            // More than ~66 ms of accumulated sim time: drop the backlog
            // instead of firing a burst of stale rounds after a hitch.
            self.weapon_accumulator = 0.0;
        }
    }
}

impl Benchmark {
    fn report(&self, output: (u32, u32), internal: (u32, u32), scale: f32) {
        if self.frame_times_ms.is_empty() {
            tracing::info!("benchmark: no frames captured");
            return;
        }
        let count = self.frame_times_ms.len();
        let sum: f32 = self.frame_times_ms.iter().sum();
        let average_ms = sum / count as f32;
        let mut sorted = self.frame_times_ms.clone();
        sorted.sort_by(f32::total_cmp);
        let min_ms = sorted[0];
        let p50_ms = sorted[count / 2];
        let p99_ms = sorted[(count * 99 / 100).min(count - 1)];
        let worst_fps = 1000.0 / p99_ms;
        let gpu_average = mean(&self.gpu_times_ms);
        let gbuffer_average = mean(&self.gbuffer_times_ms);
        let lighting_average = mean(&self.lighting_times_ms);
        let temporal_average = mean(&self.temporal_times_ms);
        let shadow_average = mean(&self.shadow_times_ms);
        let light_shadow_average = mean(&self.light_shadow_times_ms);
        let gi_average = mean(&self.gi_times_ms);
        let fog_average = mean(&self.fog_times_ms);
        let bloom_average = mean(&self.bloom_times_ms);

        tracing::info!(
            frames = count,
            output = format_args!("{}x{}", output.0, output.1),
            internal = format_args!("{}x{}", internal.0, internal.1),
            scale = format_args!("{:.2}", scale),
            fps_mean = format_args!("{:.1}", 1000.0 / average_ms),
            fps_p50 = format_args!("{:.1}", 1000.0 / p50_ms),
            fps_p99 = format_args!("{:.1}", worst_fps),
            frametime_mean_ms = format_args!("{:.2}", average_ms),
            frametime_best_ms = format_args!("{:.2}", min_ms),
            frametime_p99_ms = format_args!("{:.2}", p99_ms),
            gpu_mean_ms = format_args!("{:.2}", gpu_average),
            gbuffer_ms = format_args!("{:.2}", gbuffer_average),
            lighting_ms = format_args!("{:.2}", lighting_average),
            temporal_ms = format_args!("{:.2}", temporal_average),
            shadow_ms = format_args!("{:.2}", shadow_average),
            light_shadow_ms = format_args!("{:.2}", light_shadow_average),
            gi_ms = format_args!("{:.2}", gi_average),
            fog_ms = format_args!("{:.2}", fog_average),
            bloom_ms = format_args!("{:.2}", bloom_average),
            skipped = self.skipped_frames,
            "BENCHMARK"
        );
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

impl App for Game {
    fn resumed(&mut self, window: Arc<Window>) -> AppEvent {
        if self.gpu.is_some() {
            return AppEvent::Continue;
        }

        let gpu = match pollster::block_on(GpuContext::new(window.clone())) {
            Ok(gpu) => gpu,
            Err(error) => {
                tracing::error!("failed to initialize GPU: {error:#}");
                return AppEvent::Exit;
            }
        };
        tracing::info!(
            adapter = gpu.adapter_name(),
            backend = gpu.backend_name(),
            timestamp_queries = gpu.timestamp_supported,
            "GPU ready"
        );

        let physics = match JoltPhysics::new() {
            Ok(physics) => physics,
            Err(error) => {
                tracing::error!("failed to initialize Jolt: {error:#}");
                return AppEvent::Exit;
            }
        };
        // Static collision comes from the voxel sectors, not hand-placed boxes:
        // a shot that opens a hole now changes the actual colliders.

        let renderer = match Renderer::new(&gpu, &window, &self.world) {
            Ok(mut renderer) => {
                // Scene-owned light rig. The house gets five warm ceiling lamps
                // (leaving seven of the twelve point slots free for blasts); the
                // studio keeps its neutral diagnostic rig.
                self.lights = match self.scene {
                    SceneKind::Studio => studio_lights(),
                    SceneKind::House => house_lights(),
                    SceneKind::TestRoom => test_room_lights(),
                    SceneKind::FxLab => fx_lab_lights(),
                };
                renderer.set_lights(&self.lights);
                self.area_lights = match self.scene {
                    SceneKind::Studio => vec![
                        // Long capsule over the screens and a one-sided panel.
                        AreaLight::capsule(
                            [128.0, 6.5, 148.0],
                            [1.0, 0.0, 0.0],
                            60.0,
                            40.0,
                            [1.0, 0.98, 0.95],
                            2.5,
                        ),
                        AreaLight::rect(
                            [154.0, 4.0, 136.0],
                            [0.0, 0.0, -1.0],
                            3.0,
                            14.0,
                            [0.80, 0.90, 1.0],
                            3.0,
                        ),
                    ],
                    SceneKind::House => Vec::new(),
                    SceneKind::TestRoom => Vec::new(),
                    SceneKind::FxLab => Vec::new(),
                };
                renderer.set_area_lights(&self.area_lights);
                let hours = std::env::var("SHATTERBOX_TIME")
                    .ok()
                    .and_then(|value| value.parse::<f32>().ok())
                    .unwrap_or(10.5);
                renderer.set_time_of_day(hours);
                renderer
            }
            Err(error) => {
                tracing::error!("failed to initialize renderer: {error:#}");
                return AppEvent::Exit;
            }
        };

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        let mut destruction = DestructionSystem::new(physics);
        destruction.set_min_piece(min_piece_default());
        self.destruction = Some(destruction);
        let (spawn, yaw, pitch) = self.scene.spawn();
        self.player.spawn_on_floor(&self.world, spawn);
        self.camera.position = self.player.eye();
        self.camera.yaw = yaw;
        self.camera.pitch = pitch;

        if let Some(run) = self.scenario.as_mut() {
            // Scripted mode owns the camera and input: never grab the cursor,
            // and apply the run overrides before the first frame.
            let scale = run.options().scale;
            let effects = run.options().effects;
            let seed = run.seed();
            let adapter = self
                .gpu
                .as_ref()
                .map_or_else(String::new, |gpu| gpu.adapter_name().to_owned());
            let backend = self
                .gpu
                .as_ref()
                .map_or_else(String::new, |gpu| gpu.backend_name().to_owned());
            run.set_device(&adapter, &backend);
            if let Some(renderer) = &mut self.renderer {
                if let Some(scale) = scale {
                    renderer.set_scale(scale);
                }
                if let Some(bits) = effects {
                    renderer.set_effects(RenderEffects::from_bitmask(bits));
                }
                let applied_scale = renderer.stats().scale;
                let applied_effects = renderer.effects().bitmask();
                run.note_render_settings(applied_scale, applied_effects);
                run.note_gi_quality(renderer.ddgi_quality().name());
            }
            self.shot_sequence = seed.max(1);
        } else {
            // Interactive session: the sky defaults come from config/sky.ron,
            // except where the matching SHATTERBOX_* env var is set — env stays
            // the A/B path and overrides the saved config. Scenario runs skip
            // this on purpose: their screenshots must not depend on a local
            // config file.
            let sky = crate::sky_config::SkyConfig::load();
            if let Some(renderer) = &mut self.renderer {
                if std::env::var_os("SHATTERBOX_SUN_INTENSITY").is_none() {
                    renderer.set_sun_intensity(sky.sun_intensity);
                }
                if std::env::var_os("SHATTERBOX_FOG").is_none() {
                    renderer.set_fog_density(sky.fog_density);
                }
                if std::env::var_os("SHATTERBOX_FOG_HEIGHT").is_none() {
                    renderer.set_fog_height(sky.fog_height);
                }
            }
            self.input.capture(&window);
        }
        self.window = Some(window);
        self.last_frame = Some(Instant::now());
        AppEvent::Continue
    }

    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> AppEvent {
        match event {
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => return self.redraw(window),
            _ => {}
        }

        let mut consumed = false;
        if let Some(renderer) = &mut self.renderer {
            consumed = renderer.handle_event(window, event).consumed;
        }
        // egui eagerly consumes keyboard input while the overlay is visible.
        // Gameplay keys (time of day, scale, fullscreen) must still reach the
        // input layer, so only pointer events are gated by `consumed`.
        let is_keyboard = matches!(event, WindowEvent::KeyboardInput { .. });
        if consumed && !is_keyboard {
            self.input.end_frame();
            return AppEvent::Continue;
        }

        self.input.handle_window_event(window, event);

        if self.input.state().pressed(Action::Cancel) {
            self.input.release_cursor(window);
        }

        AppEvent::Continue
    }

    fn device_event(&mut self, event: &DeviceEvent) {
        self.input.handle_device_event(event);
    }
}

/// Immutable snapshot handed to the UI closure so the renderer can borrow the
/// GPU context mutably during presentation.
struct OverlayData {
    fps: f32,
    frame_time_ms: f32,
    smoothed_frame_time_ms: f32,
    frame_index: u64,
    adapter: String,
    backend: String,
    resolution: (u32, u32),
    render: RenderStats,
    voxels: VoxelStats,
    ray: Option<RaySummary>,
    /// Normalised screen position of the predicted crater centre.
    aim_marker: Option<[f32; 2]>,
    cursor_captured: bool,
    camera_position: [f32; 3],
    camera_yaw_degrees: f32,
    camera_pitch_degrees: f32,
    show_overlay: bool,
    pieces: usize,
    /// Simulated pieces that were not uploaded this frame.
    debris_dropped: usize,
    rockets: usize,
    awake_bodies: usize,
    shots: u32,
    removed_total: usize,
    filled_total: usize,
    last_shot: Option<ShotSummary>,
    weapon: &'static str,
}

#[derive(Clone, Copy)]
struct ShotSummary {
    hit: bool,
    removed: usize,
    pieces: usize,
}

#[derive(Clone, Copy)]
struct RaySummary {
    distance: f32,
    material: &'static str,
    position: [i32; 3],
    normal: [i32; 3],
}

#[allow(deprecated)]
/// Mutable UI state collected from the overlay and applied after the frame.
/// egui runs inside `Renderer::render`, so the widgets cannot call the renderer
/// directly; they write here and `redraw` pushes the values afterwards.
struct UiActions {
    effects: RenderEffects,
    /// DDGI ray budget selected in F1 (`SHATTERBOX_GI` sets the startup value).
    ddgi_quality: DdgiQuality,
    time_of_day: f32,
    sun_temperature: f32,
    /// Sun power multiplier (linear HDR, 0..=5) and fog parameters, persisted
    /// to `config/sky.ron` when a slider is released.
    sun_intensity: f32,
    fog_density: f32,
    fog_height: f32,
    /// Trilinear probe sampling in fog; A/B switch for the 8 m square bubble.
    fog_smooth: bool,
    /// Linear HDR exposure, auto-exposure toggle, EV bias and post look.
    exposure: f32,
    auto_exposure: bool,
    exposure_bias: f32,
    look: u8,
    /// Sky sliders changed: write `config/sky.ron` after the frame.
    sky_save: bool,
    shadow_mode: ShadowMode,
    upscale_clamp: bool,
    /// Debris reactive mask toggle (F1 overlay / `SHATTERBOX_DEBRIS_REACTIVE`).
    debris_reactive: bool,
    /// Bake pieces that stopped moving (F1 overlay / `SHATTERBOX_DEBRIS_SETTLE`).
    settle_debris: bool,
    /// Draw dust billboards for voxels too small to become bodies
    /// (F1 overlay / `SHATTERBOX_DUST_PARTICLES`).
    dust_grains: bool,
    /// Weapon chosen in the bottom selector, applied after the frame.
    weapon: Option<WeaponKind>,
}

fn draw_overlay(ctx: &egui::Context, data: &OverlayData, actions: &mut UiActions) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::unique("shatterbox_crosshair"),
    ));
    let center = ctx.content_rect().center();
    let color = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210);
    let stroke = egui::Stroke::new(1.4, color);
    for (from, to) in [
        (egui::vec2(-11.0, 0.0), egui::vec2(-3.0, 0.0)),
        (egui::vec2(3.0, 0.0), egui::vec2(11.0, 0.0)),
        (egui::vec2(0.0, -11.0), egui::vec2(0.0, -3.0)),
        (egui::vec2(0.0, 3.0), egui::vec2(0.0, 11.0)),
    ] {
        painter.line_segment([center + from, center + to], stroke);
    }
    painter.circle_filled(center, 1.3, color);

    // Where the shot will actually carve. It should sit under the reticle; any
    // visible offset means the aim ray and the rendered centre ray disagree.
    if let Some([u, v]) = data.aim_marker {
        let rect = ctx.content_rect();
        let point = egui::pos2(
            rect.left() + u * rect.width(),
            rect.top() + v * rect.height(),
        );
        if rect.shrink(2.0).contains(point) {
            let marker = egui::Color32::from_rgba_unmultiplied(120, 235, 255, 150);
            painter.circle_stroke(point, 2.6, egui::Stroke::new(1.0, marker));
        }
    }

    // Weapon strip sits at the bottom centre so it never covers the reticle or
    // the top-left debug panel. Clicking a slot selects it; 1/2/3 and the wheel
    // do the same from the input path.
    egui::Area::new(egui::Id::unique("shatterbox_weapons"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -14.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_rgba_unmultiplied(12, 14, 18, 170))
                .inner_margin(egui::Margin::symmetric(8, 5))
                .corner_radius(6.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (slot, kind) in (1..=3).zip(WeaponKind::ALL) {
                            let active = data.weapon == kind.name();
                            let label = format!("{slot}  {}", kind.name());
                            if ui.selectable_label(active, label).clicked() {
                                actions.weapon = Some(kind);
                            }
                        }
                    });
                });
        });

    if !data.show_overlay {
        return;
    }

    // The panel has grown past the height of a 900p window (every effect has a
    // checkbox now), so it scrolls instead of running off the bottom edge.
    let overlay_height = (ctx.content_rect().height() - 24.0).max(240.0);
    egui::Window::new("Shatterbox — Phase 3")
        .resizable(false)
        .collapsible(false)
        .vscroll(true)
        // `max_height` alone is ignored by a non-resizable window in egui 0.37:
        // the area keeps the small rect it remembered from the earlier, shorter
        // panel and clips the rest. Pinning min == max makes the scroll area
        // take the whole screen height instead.
        .fixed_size(egui::vec2(560.0, overlay_height))
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .show(ctx, |ui| {
            ui.heading(format!("{:.0} FPS", data.fps));
            ui.label(format!(
                "{:.2} ms frame (smoothed {:.2} ms)",
                data.frame_time_ms, data.smoothed_frame_time_ms
            ));
            if data.render.timestamps_valid {
                ui.label(format!("GPU {:.2} ms total", data.render.total_ms));
                ui.label(format!(
                    "gbuffer {:.2} · light {:.2} · taa {:.2} ms",
                    data.render.gbuffer_ms, data.render.lighting_ms, data.render.temporal_ms
                ));
                ui.label(format!(
                    "shadow {:.2} · lshadow {:.2} · gi {:.2} · fog {:.2} · bloom {:.2} · particles {:.2} · fxaa {:.2} ms",
                    data.render.shadow_ms,
                    data.render.light_shadow_ms,
                    data.render.gi_ms,
                    data.render.fog_ms,
                    data.render.bloom_ms,
                    data.render.particles_ms,
                    data.render.fxaa_ms
                ));
            } else if data.render.timestamps_supported {
                ui.label("GPU timings warming up…");
            } else {
                ui.label("GPU timestamps unsupported");
            }
            ui.separator();
            egui::Grid::new("shatterbox_stats")
                .num_columns(2)
                .spacing([14.0, 3.0])
                .show(ui, |ui| {
                    ui.label("GPU");
                    ui.monospace(&data.adapter);
                    ui.end_row();

                    ui.label("Backend");
                    ui.monospace(&data.backend);
                    ui.end_row();

                    ui.label("Output");
                    ui.monospace(format!("{}×{}", data.resolution.0, data.resolution.1));
                    ui.end_row();

                    ui.label("Internal");
                    ui.monospace(format!(
                        "{}×{} @ {:.0}%",
                        data.render.internal_size.0,
                        data.render.internal_size.1,
                        data.render.scale * 100.0
                    ));
                    ui.end_row();

                    ui.label("Upscaler");
                    ui.monospace(format!("{}  [U]", data.render.upscaler));
                    ui.end_row();

                    ui.label("Governor");
                    ui.monospace(format!(
                        "{} · target {:.0} fps",
                        if data.render.governor_enabled {
                            "auto"
                        } else {
                            "manual"
                        },
                        data.render.target_fps
                    ));
                    ui.end_row();

                    ui.label("Lights");
                    ui.monospace(format!("{} active", data.render.light_count));
                    ui.end_row();

                    ui.label("Time");
                    ui.monospace(format!(
                        "{:02}:{:02}",
                        data.render.time_of_day as u32,
                        ((data.render.time_of_day.fract() * 60.0) as u32).min(59)
                    ));
                    ui.end_row();

                    ui.label("Sky");
                    ui.monospace(format!(
                        "sun x{:.2} · fog {:.3} · fall {:.0} m",
                        actions.sun_intensity, actions.fog_density, actions.fog_height
                    ));
                    ui.end_row();

                    ui.label("Exposure");
                    ui.monospace(if actions.auto_exposure {
                        format!("auto · bias {:+.2} EV", actions.exposure_bias)
                    } else {
                        format!("manual x{:.2}", actions.exposure)
                    });
                    ui.end_row();

                    ui.label("Look");
                    ui.monospace(match actions.look {
                        1 => "B · cinematic",
                        2 => "C · punchy",
                        _ => "A · neutral",
                    });
                    ui.end_row();

                    ui.label("Weapon");
                    ui.monospace(format!(
                        "{}  (1/2/3 or wheel)",
                        data.weapon
                    ));
                    ui.end_row();

                    ui.label("Debris");
                    ui.monospace(format!(
                        "{} drawn · {} simulated ({} awake) · {} dropped  (cap {})",
                        data.render.debris_count,
                        data.pieces,
                        data.awake_bodies,
                        data.debris_dropped,
                        data.render.debris_limit
                    ));
                    ui.end_row();

                    ui.label("Shots");
                    ui.monospace(format!(
                        "{} fired · {} in flight · {} voxels · {} rebuilt",
                        data.shots, data.rockets, data.removed_total, data.filled_total
                    ));
                    ui.end_row();

                    ui.label("Shaders");
                    ui.monospace(if data.render.hot_reload_enabled {
                        "hot reload"
                    } else {
                        "embedded"
                    });
                    ui.end_row();

                    ui.label("World");
                    ui.monospace(format!(
                        "{} regions · {} bricks",
                        data.voxels.regions, data.voxels.bricks
                    ));
                    ui.end_row();

                    ui.label("Voxels");
                    ui.monospace(format!(
                        "{} solid · {:.0} ms build",
                        data.voxels.solid, data.voxels.build_ms
                    ));
                    ui.end_row();

                    ui.label("Hash");
                    ui.monospace(format!("{:#018x}", data.voxels.hash));
                    ui.end_row();

                    ui.label("Frames");
                    ui.monospace(data.frame_index.to_string());
                    ui.end_row();

                    ui.label("Mouse");
                    ui.monospace(if data.cursor_captured {
                        "captured"
                    } else {
                        "free"
                    });
                    ui.end_row();
                });
            ui.separator();
            ui.heading("Environment");
            ui.add(
                egui::Slider::new(&mut actions.time_of_day, 0.0..=24.0)
                    .text("time of day")
                    .custom_formatter(|value, _| {
                        let hours = value.rem_euclid(24.0);
                        format!(
                            "{:02}:{:02}",
                            hours as u32,
                            ((hours.fract() * 60.0) as u32).min(59)
                        )
                    }),
            );
            ui.add(
                egui::Slider::new(&mut actions.sun_temperature, 2000.0..=12000.0)
                    .text("sun temperature")
                    .step_by(50.0)
                    .suffix(" K"),
            );
            // Live sky look: all three apply immediately (no restart) and are
            // persisted to config/sky.ron when the slider is released.
            let sun_power = ui.add(
                egui::Slider::new(&mut actions.sun_intensity, 0.0..=5.0)
                    .text("sun power")
                    .step_by(0.05)
                    .suffix("x"),
            );
            let fog = ui.add(
                egui::Slider::new(&mut actions.fog_density, 0.0..=0.15)
                    .text("fog density")
                    .step_by(0.001),
            );
            let fog_falloff = ui.add(
                egui::Slider::new(&mut actions.fog_height, 0.5..=60.0)
                    .text("fog height falloff")
                    .step_by(0.5)
                    .suffix(" m"),
            );
            ui.checkbox(
                &mut actions.fog_smooth,
                "Smooth fog probes  (trilinear; off = old 8 m squares)",
            );
            ui.checkbox(
                &mut actions.auto_exposure,
                "Auto exposure  (log-luminance meter, adapted)",
            );
            ui.add_enabled(
                !actions.auto_exposure,
                egui::Slider::new(&mut actions.exposure, 0.2..=8.0)
                    .text("manual exposure")
                    .step_by(0.01)
                    .suffix("x"),
            );
            ui.add(
                egui::Slider::new(&mut actions.exposure_bias, -2.0..=2.0)
                    .text("exposure bias")
                    .step_by(0.05)
                    .suffix(" EV"),
            );
            ui.horizontal(|ui| {
                ui.label("look");
                egui::ComboBox::from_id_salt("shatterbox_look")
                    .selected_text(match actions.look {
                        1 => "B · cinematic",
                        2 => "C · punchy",
                        _ => "A · neutral",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut actions.look, 0, "A · neutral");
                        ui.selectable_value(&mut actions.look, 1, "B · cinematic");
                        ui.selectable_value(&mut actions.look, 2, "C · punchy");
                    });
            });
            // Save on release (and on keyboard steps, where the pointer is
            // up), not on every drag pixel: the file is the default for the
            // next launch, not a per-frame write.
            if sun_power.drag_stopped()
                || fog.drag_stopped()
                || fog_falloff.drag_stopped()
                || ((sun_power.changed() || fog.changed() || fog_falloff.changed())
                    && !ui.input(|i| i.pointer.primary_down()))
            {
                actions.sky_save = true;
            }
            ui.separator();
            ui.heading("Effects");
            ui.checkbox(
                &mut actions.effects.raytraced_shadows,
                "Voxel RT shadows  (1 ray/pixel towards the sun)",
            );
            // Shadow trace resolution / filter. All three keep the dithered soft
            // penumbra; only the filtering of the traced buffer changes. Left in
            // the overlay so the A/B stays reproducible after a restart.
            ui.horizontal(|ui| {
                ui.label("shadow filter");
                for mode in [
                    ShadowMode::HalfRes,
                    ShadowMode::FullRes,
                    ShadowMode::Bilateral,
                    ShadowMode::FullResFiltered,
                ] {
                    if ui
                        .selectable_label(actions.shadow_mode == mode, mode.label())
                        .clicked()
                    {
                        actions.shadow_mode = mode;
                    }
                }
            });
            // Anti-overshoot clamp around the spatial upscale. Off restores the
            // old filter, which drew bright teeth on silhouettes and a thin dark
            // seam underneath them.
            ui.checkbox(
                &mut actions.upscale_clamp,
                "Clamp upscale to tap range  (no bright fringe / dark seam)",
            );
            // Reactive mask for falling debris: with it off the temporal path
            // keeps history on the piece and it leaves a ghost trail.
            ui.checkbox(
                &mut actions.debris_reactive,
                "Debris reactive mask  (no ghost trail behind falling pieces)",
            );
            ui.checkbox(
                &mut actions.effects.ssao,
                "SSAO  (half-res screen-space ambient occlusion)",
            );
            ui.checkbox(&mut actions.effects.ssr, "SSR  (screen-space reflections)");
            ui.checkbox(
                &mut actions.effects.vct,
                "Voxel cone tracing  (coarse 4 m irradiance volume)",
            );
            ui.checkbox(
                &mut actions.effects.taa,
                "TAA  (sub-pixel jitter + temporal AA)",
            );
            ui.checkbox(
                &mut actions.effects.volumetric_fog,
                "Volumetric fog  (height fog + sun shafts)",
            );
            ui.checkbox(
                &mut actions.effects.light_aa,
                "Light AA before upscale  (soften voxel silhouettes)",
            );
            ui.checkbox(
                &mut actions.effects.fxaa,
                "FXAA before upscale  (internal-res edge blur, no history)",
            );
            ui.checkbox(
                &mut actions.effects.debris_cull,
                "Debris sphere culling  (cheap reject before the instance record)",
            );
            ui.checkbox(
                &mut actions.effects.debris_tiles,
                "Debris tile lists  (per 8x8 tile candidate masks in the G-buffer)",
            );
            ui.horizontal(|ui| {
                ui.label("DDGI quality");
                egui::ComboBox::from_id_salt("shatterbox_ddgi_quality")
                    .selected_text(actions.ddgi_quality.name())
                    .show_ui(ui, |ui| {
                        for quality in [
                            DdgiQuality::Off,
                            DdgiQuality::Low,
                            DdgiQuality::Medium,
                            DdgiQuality::High,
                        ] {
                            if ui
                                .selectable_label(actions.ddgi_quality == quality, quality.name())
                                .clicked()
                            {
                                actions.ddgi_quality = quality;
                                actions.effects.ddgi = quality != DdgiQuality::Off;
                            }
                        }
                    });
                ui.label("coloured diffuse bounce (octahedral probes)");
            });
            ui.checkbox(
                &mut actions.effects.dynamic_smoke,
                "Dynamic smoke  (advects around obstacles, fed by fire/explosions)",
            );
            ui.checkbox(
                &mut actions.settle_debris,
                "Bake settled debris  (freeze slow pieces into the world)",
            );
            ui.checkbox(
                &mut actions.dust_grains,
                "Dust grains  (billboards for chips too small to simulate)",
            );
            ui.horizontal(|ui| {
                if ui.button("All on").clicked() {
                    actions.effects = RenderEffects {
                        raytraced_shadows: true,
                        ssao: true,
                        ssr: true,
                        vct: true,
                        taa: true,
                        volumetric_fog: true,
                        light_aa: true,
                        fxaa: true,
                        debris_cull: true,
                        debris_tiles: true,
                        ddgi: true,
                        dynamic_smoke: true,
                    };
                }
                if ui.button("All off").clicked() {
                    actions.effects = RenderEffects {
                        raytraced_shadows: false,
                        ssao: false,
                        ssr: false,
                        vct: false,
                        taa: false,
                        volumetric_fog: false,
                        light_aa: false,
                        fxaa: false,
                        debris_cull: false,
                        debris_tiles: false,
                        ddgi: false,
                        dynamic_smoke: false,
                    };
                }
            });
            ui.separator();
            match data.ray {
                Some(ray) => {
                    ui.monospace(format!("ray  {} @ {:.1} m", ray.material, ray.distance));
                    ui.monospace(format!(
                        "vox  {}, {}, {}   n {}, {}, {}",
                        ray.position[0],
                        ray.position[1],
                        ray.position[2],
                        ray.normal[0],
                        ray.normal[1],
                        ray.normal[2]
                    ));
                }
                None => {
                    ui.monospace("ray  miss");
                }
            }
            match data.last_shot {
                Some(shot) if shot.hit => ui.monospace(format!(
                    "shot {} voxels → {} debris pieces",
                    shot.removed, shot.pieces
                )),
                Some(_) => ui.monospace("shot miss"),
                None => ui.monospace("shot  —"),
            };
            ui.monospace(format!(
                "pos  {:.1}, {:.1}, {:.1}",
                data.camera_position[0], data.camera_position[1], data.camera_position[2]
            ));
            ui.monospace(format!(
                "yaw  {:.1}°   pitch {:.1}°",
                data.camera_yaw_degrees, data.camera_pitch_degrees
            ));
            ui.separator();
            ui.label("LMB shatter · RMB fill · mouse look (Esc releases)");
            ui.label("WASD walk · Space jump · Shift sprint · wheel sensitivity");
            ui.label("[ / ] time of day (Shift = 2 h)");
            ui.label("R respawn · F1 overlay · F2 governor · F3 scale · F11 fullscreen");
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blast_size_is_deterministic_in_the_seed() {
        for seed in 0..64u32 {
            let a = blast_geometry(blast_scale(true, seed));
            let b = blast_geometry(blast_scale(true, seed));
            assert_eq!(a.0, b.0, "radius must not depend on anything but the seed");
            assert_eq!(a.1, b.1, "energy must not depend on anything but the seed");
        }
    }

    #[test]
    fn blast_size_stays_inside_the_configured_range() {
        // 0.7..1.4x of the 18-voxel base crater (2.25 m), so the radius has to
        // land in 13..25 voxels and the energy stays at the base value (the
        // destruction clamp is 32).
        let base = ROCKET_RADIUS_VOXELS as f32;
        for seed in 0..512u32 {
            let scale = blast_scale(true, seed);
            assert!((0.7..=1.4).contains(&scale), "scale {scale} out of range");
            let (radius, energy) = blast_geometry(scale);
            assert!(
                (13..=25).contains(&radius),
                "radius {radius} outside 13..=25 voxels"
            );
            assert!((radius as f32) <= base * 1.4 + 0.5);
            assert_eq!(energy, SHOT_ENERGY_JOULES);
        }
    }

    #[test]
    fn different_seeds_produce_different_craters() {
        let radii: Vec<i32> = (0..32u32)
            .map(|seed| blast_geometry(blast_scale(true, seed)).0)
            .collect();
        let first = radii[0];
        assert!(
            radii.iter().any(|&r| r != first),
            "32 seeds collapsed to one crater size"
        );
        // And the spread has to be visible, not a single voxel of jitter.
        let min = *radii.iter().min().unwrap();
        let max = *radii.iter().max().unwrap();
        assert!(max - min >= 4, "crater spread only {min}..{max} voxels");
    }

    #[test]
    fn pinned_variation_keeps_the_base_crater() {
        let scale = blast_scale(false, 12345);
        assert_eq!(scale, 1.0);
        assert_eq!(blast_geometry(scale).0, ROCKET_RADIUS_VOXELS);
        assert_eq!(blast_geometry(scale).1, SHOT_ENERGY_JOULES);
    }

    /// The carbine seed folds the fixed tick and the shot sequence: same pair
    /// replays, any difference in either moves the round.
    #[test]
    fn carbine_seed_is_deterministic_in_tick_and_sequence() {
        assert_eq!(carbine_seed(7, 3), carbine_seed(7, 3));
        assert_ne!(
            carbine_seed(7, 3),
            carbine_seed(8, 3),
            "the tick must change the seed"
        );
        assert_ne!(
            carbine_seed(7, 3),
            carbine_seed(7, 4),
            "the sequence must change the seed"
        );
        let seeds: Vec<u32> = (0..64u64).map(|tick| carbine_seed(tick, 1)).collect();
        let distinct = seeds.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 60, "64 ticks collapsed to {distinct} seeds");
    }

    #[test]
    fn smoke_cloud_is_deterministic_in_the_seed() {
        let cloud = |seed: u32| {
            let puffs: Vec<SmokePuff> = (0..SMOKE_PUFFS as u32)
                .map(|i| {
                    make_puff(
                        seed,
                        0x51ed_270b,
                        i,
                        Vec3::new(1.0, 2.0, 3.0),
                        1.75,
                        SMOKE_LIFE_S,
                        4.6,
                        [0.1, 0.1, 0.1, 0.5],
                        1.7,
                        1.1,
                        0.95,
                    )
                })
                .collect();
            puffs
                .iter()
                .map(|p| {
                    (
                        p.position.to_array().map(f32::to_bits),
                        p.start_radius.to_bits(),
                        p.end_radius.to_bits(),
                        p.velocity.to_array().map(f32::to_bits),
                        p.life.to_bits(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(cloud(7), cloud(7), "same seed must rebuild the same cloud");
        assert_ne!(cloud(7), cloud(8), "different seed must move the cloud");
    }

    /// The number keys must select what the weapon bar draws: slot 1 is the
    /// first entry of `ALL`, not an off-by-one neighbour.
    #[test]
    fn weapon_slots_match_the_weapon_bar_order() {
        for (index, kind) in WeaponKind::ALL.into_iter().enumerate() {
            assert_eq!(
                WeaponKind::from_slot(index + 1),
                kind,
                "slot {} must map to {}",
                index + 1,
                kind.name()
            );
            assert_eq!(kind.cycle(1), WeaponKind::ALL[(index + 1) % 3]);
            assert_eq!(kind.cycle(-1), WeaponKind::ALL[(index + 2) % 3]);
        }
        assert_eq!(WeaponKind::from_slot(1), WeaponKind::Bazooka);
        assert_eq!(WeaponKind::from_slot(2), WeaponKind::Carbine);
        assert_eq!(WeaponKind::from_slot(3), WeaponKind::Flamethrower);
    }
}
