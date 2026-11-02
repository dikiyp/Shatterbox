//! Deterministic voxel destruction: remove a sphere, split the removed voxels
//! into irregular pieces, and hand every piece to `PhysicsBackend` as a
//! compound collider. Pieces are persisted forever; Jolt puts them to sleep.

pub mod sectors;

use glam::{IVec3, Mat3, Quat, Vec3};
use shatterbox_physics::{BodyDesc, BodyHandle, BodyState, CompoundBox, PhysicsBackend};
use shatterbox_voxel::BRICK_DIM;
use shatterbox_voxel::RayHit;
use shatterbox_voxel::Voxel;
use shatterbox_voxel::edit::{
    EditReport, RemovedVoxel, VoxelCommand, apply_command, apply_voxel_edits,
};
use shatterbox_voxel::world::{VOXEL_SIZE_M, VOXELS_X, VOXELS_Y, VOXELS_Z, VoxelWorld};
use std::collections::{HashMap, HashSet, VecDeque};

/// No rigid body may be smaller than this many voxels. A 1-4 voxel body costs a
/// Jolt body, a render instance and a bake, but reads as a speck or jitters
/// inside the pile; below the minimum the piece is merged into a neighbour
/// (still one body) or, when nothing is close enough, turned into cosmetic dust.
pub const MIN_PIECE_VOXELS: usize = 20;
/// Distance within which a piece below the minimum is merged into a neighbour
/// even when the two do not share a face (0.6 m is about five voxels).
const MERGE_RADIUS_M: f32 = 1.4;
/// Upper bound on the dust grain positions handed back to the game per shot.
const MAX_DUST_GRAINS: usize = 96;
/// A kinetic round pushes the chips it freed out of the hole at roughly this
/// speed (m/s): enough to fall clear, far from a shock wave throw.
const KINETIC_PUSH_SPEED: f32 = 3.4;
/// Kinematic chips tumble at most this fast (rad/s).
const KINETIC_SPIN: f32 = 1.1;

/// How a weapon deposits its energy into the world.
///
/// The impulse solver is shared by every weapon; the profile only decides how
/// the energy reaches the fragments. An explosive pushes every fragment inside
/// the blast radius radially, with a speed that falls off as `1/r^2` and with
/// the fragment mass. A kinetic round has no shock wave: the chips that lost
/// their support are pushed out of the hole along the bullet direction at a
/// small uniform speed and then only gravity and collisions act on them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImpactProfile {
    /// Rocket or explosive charge. `power` is the shot's seed-driven multiplier
    /// (0.7..1.4 in the game): it scales the deposited energy, so a stronger
    /// shot scatters further.
    Explosive { energy_joules: f32, power: f32 },
    /// Bullet or pellet. `energy_joules` only decides how finely the hole is
    /// chipped; the motion is a small push out of the hole, not a blast.
    Kinetic { energy_joules: f32 },
}

impl ImpactProfile {
    #[must_use]
    pub fn explosive(energy_joules: f32, power: f32) -> Self {
        Self::Explosive {
            energy_joules,
            power,
        }
    }

    #[must_use]
    pub fn kinetic(energy_joules: f32) -> Self {
        Self::Kinetic { energy_joules }
    }

    /// Energy handed to the impulse solver. Kinetic rounds deposit none: their
    /// fragments are pushed out of the hole instead of being thrown by a blast.
    #[must_use]
    pub fn blast_energy(self) -> f32 {
        match self {
            Self::Explosive {
                energy_joules,
                power,
            } => energy_joules.max(0.0) * power.clamp(0.05, 4.0),
            Self::Kinetic { .. } => 0.0,
        }
    }

    /// Energy that decides the fragment size. A bullet still shatters the voxels
    /// it cuts, so its nominal energy takes part here even though it does not
    /// drive the impulse solve.
    #[must_use]
    pub fn fragmentation_energy(self) -> f32 {
        match self {
            Self::Explosive { .. } => self.blast_energy(),
            Self::Kinetic { energy_joules } => energy_joules.max(0.0),
        }
    }
}

/// Jolt rejects a body whose solved velocities are non-finite or above its
/// `limits::MAX_LINEAR_VELOCITY` / `MAX_ANGULAR_VELOCITY`; these stay well
/// inside them so a huge close-range impulse still produces a visible piece.
const MAX_LINEAR_SPEED: f32 = 200.0;
const MAX_ANGULAR_SPEED: f32 = 30.0;

fn clamp_velocity(velocity: Vec3, limit: f32) -> Vec3 {
    if velocity.is_finite() {
        velocity.clamp_length_max(limit)
    } else {
        Vec3::ZERO
    }
}

/// Centre of the destruction sphere: exactly the first solid voxel under the
/// crosshair. Offsetting it along the face normal (or worse, along `abs(normal)`
/// as the old code did) slid the crater sideways on negative faces, so the
/// visible hole no longer matched the reticle.
fn crater_center(hit: &RayHit) -> IVec3 {
    hit.position
}

/// A piece of the static world that is now a rigid body. `voxels` is a dense
/// local grid (`dims.x * dims.y * dims.z`), with air in the gaps left by the
/// crater shape. Grid coordinate zero is `local_min`.
pub struct DebrisPiece {
    pub handle: BodyHandle,
    /// Mass derived from the voxel volume and the palette density, in kg.
    pub mass_kg: f32,
    pub dims: IVec3,
    pub voxels: Vec<u16>,
    pub local_min: Vec3,
    pub local_max: Vec3,
    pub voxel_count: u32,
    /// Previous and current fixed-step transforms. Rendering interpolates
    /// between them with the accumulator fraction, so debris does not judder
    /// when the render rate differs from the 60 Hz simulation.
    pub prev_position: Vec3,
    pub prev_rotation: Quat,
    pub position: Vec3,
    pub rotation: Quat,
    /// Seconds this piece has been moving slower than the settle thresholds.
    /// Jolt keeps resting debris in a pile awake (it never reports `sleeping`
    /// for them in practice), so the game decides "this piece is done" from the
    /// observed motion instead, and bakes it back into the world.
    pub rest_time: f32,
    /// Speed of the previous fixed step, m/s. A large drop means the piece hit
    /// something and deposits its energy there (see `step`).
    last_speed: f32,
}

impl DebrisPiece {
    /// Number of voxels in the local grid, used by the renderer's voxel buffer.
    pub fn grid_len(&self) -> usize {
        self.voxels.len()
    }
}

/// Result of one shot, consumed by the game for HUD/debug and by the renderer
/// to know which brick indices/masks became dirty.
pub struct FireResult {
    pub hit: bool,
    pub removed: usize,
    pub pieces: usize,
    /// Voxel count of every spawned piece, in spawn order. The autotest report
    /// turns this into a size histogram.
    pub piece_sizes: Vec<u32>,
    pub dirty_bricks: Vec<IVec3>,
    /// Voxels that left the world but were too small to become a body and were
    /// turned into cosmetic dust. They are not simulated and never written back,
    /// so the conservation check counts them as a third term.
    pub dust_voxels: usize,
    /// World-space grain positions for the game's dust billboards (capped).
    pub dust_points: Vec<Vec3>,
    /// Sum and maximum of the initial linear speed of the spawned bodies, m/s.
    /// The report averages these per impact profile: a rocket must obviously
    /// out-throw a carbine round.
    pub speed_sum: f32,
    pub speed_max: f32,
}

impl FireResult {
    fn missed(hit: bool, dirty_bricks: Vec<IVec3>) -> Self {
        Self {
            hit,
            removed: 0,
            pieces: 0,
            piece_sizes: Vec::new(),
            dirty_bricks,
            dust_voxels: 0,
            dust_points: Vec::new(),
            speed_sum: 0.0,
            speed_max: 0.0,
        }
    }
}

/// What [`DestructionSystem::spawn_pieces`] produced for one shot.
struct ShotSpawn {
    sizes: Vec<u32>,
    dust_voxels: usize,
    dust_points: Vec<Vec3>,
    speed_sum: f32,
    speed_max: f32,
}

/// Pieces that a bake wrote back into the world this frame.
#[derive(Clone, Debug, Default)]
pub struct BakeReport {
    pub pieces: usize,
    pub voxels: usize,
    pub dirty_bricks: Vec<IVec3>,
}

pub struct DestructionSystem<B: PhysicsBackend> {
    physics: B,
    pieces: Vec<DebrisPiece>,
    sequence: u32,
    /// When set, pieces that stop moving are counted as settled even if Jolt
    /// still reports them awake, so `bake_sleeping` can freeze them into the
    /// world. `SHATTERBOX_DEBRIS_SETTLE=0` starts with this off for A/B runs.
    settle: bool,
    /// Bodies below this many voxels are merged into a neighbour or turned into
    /// dust (see [`MIN_PIECE_VOXELS`]). Exposed so the game can A/B the old
    /// "every crumb is a body" behaviour with `SHATTERBOX_MIN_PIECE=1`.
    min_piece: usize,
    /// Voxels turned into cosmetic dust over the whole run. Matter conservation
    /// is `world + debris bodies + dust`.
    dust_voxels: u64,
    /// Heavy pieces that just lost most of their speed this step: `(point,
    /// impact speed)`. The game drains this to damage the world under them.
    impacts: Vec<(Vec3, f32)>,
}

impl<B: PhysicsBackend> DestructionSystem<B> {
    pub fn new(physics: B) -> Self {
        Self {
            physics,
            pieces: Vec::new(),
            sequence: 0,
            settle: true,
            min_piece: MIN_PIECE_VOXELS,
            dust_voxels: 0,
            impacts: Vec::new(),
        }
    }

    /// Smallest allowed rigid body, in voxels. `1` restores the pre-profile
    /// behaviour where every crumb became its own body.
    pub fn set_min_piece(&mut self, min_piece: usize) {
        self.min_piece = min_piece.clamp(1, 64);
    }

    #[must_use]
    pub fn min_piece(&self) -> usize {
        self.min_piece
    }

    /// Voxels turned into dust (removed from the world, never simulated, never
    /// written back).
    #[must_use]
    pub fn dust_voxels(&self) -> u64 {
        self.dust_voxels
    }

    /// Turns the game-side settle timer on or off (F1 overlay / env var).
    pub fn set_settle(&mut self, settle: bool) {
        if self.settle == settle {
            return;
        }
        self.settle = settle;
        if !settle {
            for piece in &mut self.pieces {
                piece.rest_time = 0.0;
            }
        }
    }

    /// A piece slower than this for [`Self::SETTLE_SECONDS`] is treated as
    /// resting. Both limits are generous compared to a falling chip: 12 cm/s is
    /// below a slow crawl, and Jolt's own sleep thresholds are stricter, which
    /// is why they never fire inside a pile.
    pub const SETTLE_SPEED: f32 = 0.12;
    /// Radians per second around any axis, same reasoning as [`Self::SETTLE_SPEED`].
    pub const SETTLE_SPIN: f32 = 0.6;
    /// How long a piece has to stay this still before it is baked.
    /// Deliberately longer than the old 0.6 s: a pile still visibly rocking at
    /// 0.6 s was baked into the world on the next frame, and the floor-quantised
    /// pose read as a one-frame twitch. 1.2 s lets the contact solver finish.
    pub const SETTLE_SECONDS: f32 = 1.2;

    pub fn physics(&self) -> &B {
        &self.physics
    }

    pub fn physics_mut(&mut self) -> &mut B {
        &mut self.physics
    }

    pub fn pieces(&self) -> &[DebrisPiece] {
        &self.pieces
    }

    pub fn piece_state(&self, piece: &DebrisPiece) -> Option<BodyState> {
        self.physics.state(piece.handle)
    }

    /// Render transform between the last two fixed steps. `alpha` is the
    /// physics accumulator divided by the fixed step (0..1).
    pub fn interpolated_transform(&self, piece: &DebrisPiece, alpha: f32) -> (Vec3, Quat) {
        let alpha = alpha.clamp(0.0, 1.0);
        (
            piece.prev_position.lerp(piece.position, alpha),
            piece.prev_rotation.slerp(piece.rotation, alpha),
        )
    }

    pub fn active_bodies(&self) -> usize {
        self.physics.awake_count()
    }

    pub fn total_bodies(&self) -> usize {
        self.physics.body_count()
    }

    /// Pieces Jolt has put to sleep but that have not been baked yet. This is
    /// the number that matters for the "is the blast still costing anything?"
    /// question: a sleeping piece is not simulated, only drawn.
    pub fn sleeping_bodies(&self) -> usize {
        self.pieces
            .iter()
            .filter(|piece| {
                self.physics
                    .state(piece.handle)
                    .is_some_and(|state| state.sleeping)
            })
            .count()
    }

    /// Live debris budget. Jolt's body pool is finite, and a long demolition
    /// session used to end with `the world is full` warnings and pieces that
    /// simply failed to spawn. Retiring the oldest bodies keeps the newest
    /// crater simulated and bounds both physics and render cost.
    pub const LIVE_PIECE_BUDGET: usize = 320;

    /// Removes awake pieces until `budget` remains. Sleeping pieces do not
    /// count: Jolt already stopped simulating them and they are about to be
    /// baked back into the voxel world, so the cap only bounds the live
    /// simulations the player is watching.
    pub fn enforce_piece_budget(&mut self, budget: usize) -> usize {
        let budget = budget.min(Self::LIVE_PIECE_BUDGET);
        let is_awake = |physics: &B, piece: &DebrisPiece| {
            physics
                .state(piece.handle)
                .is_some_and(|state| state.awake && !state.sleeping)
        };
        let mut awake = self
            .pieces
            .iter()
            .filter(|piece| is_awake(&self.physics, piece))
            .count();
        let mut retired = 0;
        let mut index = 0;
        while awake > budget && index < self.pieces.len() {
            if is_awake(&self.physics, &self.pieces[index]) {
                let piece = self.pieces.remove(index);
                self.physics.remove(piece.handle);
                retired += 1;
                awake -= 1;
            } else {
                index += 1;
            }
        }
        retired
    }

    /// Bakes settled pieces back into the voxel world.
    ///
    /// A piece Jolt put to sleep is no longer simulated, but it is also not part
    /// of the world: it holds a body slot and a second shot cannot hit it. Baking
    /// writes its voxels back through a deterministic edit, removes the body and
    /// returns the dirty bricks for the renderer and the collision sectors.
    ///
    /// A piece is skipped while any of its voxels would land on an existing
    /// solid voxel, so no matter is created or destroyed.
    pub fn bake_sleeping(
        &mut self,
        world: &mut VoxelWorld,
        player: Vec3,
        budget: usize,
    ) -> BakeReport {
        let mut report = BakeReport::default();
        let mut index = 0;
        while index < self.pieces.len() && report.pieces < budget {
            let piece = &self.pieces[index];
            let Some(state) = self.physics.state(piece.handle) else {
                index += 1;
                continue;
            };
            if state.awake && !state.sleeping {
                // Jolt keeps a resting pile awake; the settle timer is what
                // actually decides "this chip is done".
                if piece.rest_time < Self::SETTLE_SECONDS {
                    index += 1;
                    continue;
                }
            }
            // Same limits as the settle timer: below them the piece is not going
            // anywhere visible, so freezing it is not a visual pop.
            if state.linear_velocity.length_squared() > Self::SETTLE_SPEED * Self::SETTLE_SPEED
                || state.angular_velocity.length_squared() > Self::SETTLE_SPIN * Self::SETTLE_SPIN
            {
                index += 1;
                continue;
            }
            // Never bury the player: a piece that would wrap around the capsule
            // keeps simulating (and never collides with it visually because it
            // is already resting).
            let radius = piece.local_min.abs().max(piece.local_max.abs()).length();
            // Keep watched rubble dynamic. Baking is a quantised pose swap; at
            // +1.2 m it happened inside the player's normal viewing distance and
            // showed up as a twitch a second after the hit. Only bake pieces far
            // enough away that the snap is sub-pixel, or when the live budget is
            // exceeded (`retire_into_world` still handles that case).
            if state.position.distance(player) < radius + 9.0 {
                index += 1;
                continue;
            }

            let Some(edits) = self.bake_edits(index, world) else {
                index += 1;
                continue;
            };
            let expected = edits.len();
            // Baking must be all-or-nothing. `apply_voxel_edits` writes voxels
            // it can and skips the rest, so a partial bake used to add geometry
            // to the world while the body lived on, breaking conservation and
            // duplicating voxels.
            let applied = apply_voxel_edits(world, &edits);
            if applied.added as usize != expected {
                // Geometry changed under us (another bake in the same frame):
                // leave the body, it will find room on a later frame or stay
                // collision-only.
                index += 1;
                continue;
            }
            let piece = self.pieces.remove(index);
            self.physics.remove(piece.handle);
            report.pieces += 1;
            report.voxels += applied.added as usize;
            report.dirty_bricks.extend(applied.dirty_bricks);
        }
        report
            .dirty_bricks
            .sort_unstable_by_key(|brick| (brick.x, brick.y, brick.z));
        report.dirty_bricks.dedup();
        report
    }

    /// Builds the world edits that would bake one piece at its current
    /// transform, or `None` when no room can be found for every voxel.
    ///
    /// Refusing a piece whose voxels overlap anything left the whole pile
    /// simulated (and drawn) for the rest of the match: rubble rests *on* the
    /// surface it landed on, so `floor()` maps its bottom layer onto the top row
    /// of the floor, which is always solid, and rotation plus `floor()` aliases
    /// some voxels into one cell. A voxel that cannot occupy its intended cell
    /// therefore settles into the nearest free one instead. Every voxel still
    /// owns exactly one air cell, so `world + debris` stays exact — matter is
    /// moved by at most a couple of voxels, never created or deleted.
    fn bake_edits(&self, index: usize, world: &VoxelWorld) -> Option<Vec<(IVec3, Voxel)>> {
        let piece = self.pieces.get(index)?;
        let state = self.physics.state(piece.handle)?;
        let (dims, voxels, local_min) = (piece.dims, &piece.voxels, piece.local_min);
        let mut wanted: Vec<(IVec3, Voxel)> = Vec::with_capacity(piece.voxel_count as usize);
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let raw = voxels[(x + dims.x * (y + dims.y * z)) as usize];
                    if raw == 0 {
                        continue;
                    }
                    let local = local_min
                        + Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) * VOXEL_SIZE_M;
                    let world_position = state.position + state.rotation * local;
                    let cell = (world_position / VOXEL_SIZE_M).floor().as_ivec3();
                    wanted.push((cell, Voxel::from_raw(raw)));
                }
            }
        }
        wanted.sort_unstable_by_key(|(position, _)| (position.x, position.y, position.z));
        let mut taken: HashSet<IVec3> = HashSet::with_capacity(wanted.len());
        let mut edits: Vec<(IVec3, Voxel)> = Vec::with_capacity(wanted.len());
        for (cell, voxel) in wanted {
            let mut placed = None;
            for offset in settle_offsets() {
                let candidate = cell + *offset;
                // `VoxelWorld::set_voxel` ignores out-of-bounds writes without
                // telling the caller, so a piece resting half a voxel below the
                // floor would silently lose that layer. Keep every baked voxel
                // inside the world instead; if no offset fits, the piece keeps
                // simulating.
                if !in_bounds(candidate)
                    || taken.contains(&candidate)
                    || !world.voxel(candidate.x, candidate.y, candidate.z).is_air()
                {
                    continue;
                }
                placed = Some(candidate);
                break;
            }
            let cell = placed?;
            taken.insert(cell);
            edits.push((cell, voxel));
        }
        edits.sort_unstable_by_key(|(position, _)| (position.x, position.y, position.z));
        Some(edits)
    }

    /// Retires the farthest pieces into the voxel world when the live-body
    /// budget is exceeded. Matter is never deleted: a piece either bakes (and
    /// becomes part of the world) or keeps its body for the next attempt.
    /// Farthest-first keeps the visible near field simulated for longer.
    pub fn retire_into_world(
        &mut self,
        world: &mut VoxelWorld,
        player: Vec3,
        budget: usize,
    ) -> BakeReport {
        let mut report = BakeReport::default();
        let mut order: Vec<(f32, BodyHandle)> = self
            .pieces
            .iter()
            .map(|piece| (piece.position.distance_squared(player), piece.handle))
            .collect();
        order.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, handle) in order {
            if self.pieces.len() <= budget {
                break;
            }
            let Some(index) = self.pieces.iter().position(|piece| piece.handle == handle) else {
                continue;
            };
            let Some(edits) = self.bake_edits(index, world) else {
                continue;
            };
            let expected = edits.len();
            let applied = apply_voxel_edits(world, &edits);
            if applied.added as usize != expected {
                continue;
            }
            let piece = self.pieces.remove(index);
            self.physics.remove(piece.handle);
            report.pieces += 1;
            report.voxels += applied.added as usize;
            report.dirty_bricks.extend(applied.dirty_bricks);
        }
        report
            .dirty_bricks
            .sort_unstable_by_key(|brick| (brick.x, brick.y, brick.z));
        report.dirty_bricks.dedup();
        report
    }

    pub fn step(&mut self, dt: f32) {
        for piece in &mut self.pieces {
            piece.prev_position = piece.position;
            piece.prev_rotation = piece.rotation;
        }
        self.physics.step(dt);
        let mut hits: Vec<(Vec3, f32)> = Vec::new();
        for piece in &mut self.pieces {
            if let Some(state) = self.physics.state(piece.handle) {
                piece.position = state.position;
                piece.rotation = state.rotation;
            }
            let speed = if dt > 0.0 {
                (piece.position - piece.prev_position).length() / dt
            } else {
                0.0
            };
            // A heavy chunk that suddenly loses most of its speed has hit
            // something: report the contact so the caller can damage the world
            // under it. The queue is capped so a collapsing pile cannot turn one
            // frame into a barrage of craters.
            if dt > 0.0
                && piece.last_speed > 8.0
                && speed < piece.last_speed * 0.35
                && piece.mass_kg > 20.0
                && hits.len() < 8
            {
                hits.push((piece.position, piece.last_speed));
            }
            piece.last_speed = speed;
            if dt <= 0.0 || !self.settle {
                piece.rest_time = 0.0;
                continue;
            }
            // Motion of this fixed step, in m/s and rad/s. A piece resting in a
            // pile still jitters slightly, so the limits are far above zero.
            let spin = piece.prev_rotation.angle_between(piece.rotation) / dt;
            if speed <= Self::SETTLE_SPEED && spin <= Self::SETTLE_SPIN {
                piece.rest_time += dt;
            } else {
                piece.rest_time = 0.0;
            }
        }
        self.impacts.extend(hits);
    }

    /// Drains debris contact events gathered by [`Self::step`]. Each entry is
    /// the contact point and the speed the piece lost there.
    pub fn take_impacts(&mut self) -> Vec<(Vec3, f32)> {
        std::mem::take(&mut self.impacts)
    }

    /// Carves a sphere at the first world hit and converts the removed voxels
    /// into debris. `seed` makes the crater rim and the break-up deterministic.
    ///
    /// `profile` decides how the shot moves the fragments: a rocket deposits
    /// energy into a radial shock wave, a carbine round only pushes the chips it
    /// freed out of the hole. Fragment velocity, spin and the split between them
    /// are solved from mass, inertia tensor and the blast geometry (see
    /// [`DestructionSystem::spawn_clusters`]).
    pub fn fire(
        &mut self,
        world: &mut VoxelWorld,
        origin: Vec3,
        direction: Vec3,
        radius_voxels: i32,
        seed: u32,
        profile: ImpactProfile,
    ) -> FireResult {
        let Some(hit) = world.raycast(origin, direction, 256.0) else {
            return FireResult::missed(false, Vec::new());
        };

        // Carve *into* the surface. `hit.position` is the first solid voxel, so
        // the capsule starts exactly there: its entry cap cuts a disc of
        // `radius` centred on the crosshair whatever the impact angle, and the
        // sweep drills forward instead of spilling the opening sideways.
        let radius = radius_voxels.clamp(1, 14);
        let center = crater_center(&hit);
        // Push the brush ~0.9 radii deep so a perpendicular hit drills a short
        // tunnel and a glancing hit still opens a round hole at the crosshair.
        let sweep = (radius as f32 * 0.9).round() as i32;
        let extent = (direction.normalize_or_zero() * sweep as f32)
            .round()
            .as_ivec3();
        let command = VoxelCommand::remove_capsule(
            u64::from(self.sequence),
            self.sequence,
            center,
            extent,
            radius,
            1,
            seed,
        );
        let mut report = apply_command(world, &command);
        self.sequence = self.sequence.wrapping_add(1);

        if report.removed.is_empty() {
            return FireResult::missed(true, report.dirty_bricks);
        }

        // Ground truth for the crosshair-alignment diagnostics: the exact voxel
        // the ray hit, the brush centre and the centroid of what actually left
        // the world. If the rendered hole ever drifts away from the crosshair,
        // this line tells whether the carve or the upload moved.
        {
            let mut min = IVec3::splat(i32::MAX);
            let mut max = IVec3::splat(i32::MIN);
            let mut sum = glam::DVec3::ZERO;
            for removed in &report.removed {
                min = min.min(removed.position);
                max = max.max(removed.position);
                sum += removed.position.as_dvec3();
            }
            let centroid = sum / report.removed.len() as f64;
            tracing::debug!(
                target: "shatterbox::aim",
                hit_voxel = ?hit.position,
                hit_distance = hit.distance,
                normal = ?hit.normal,
                center = ?center,
                radius,
                extent = ?extent,
                removed = report.removed.len(),
                min = ?min,
                max = ?max,
                centroid = ?centroid,
                "carve"
            );
        }

        let impact = (hit.position.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE_M;
        let spawn = self.spawn_pieces(
            world,
            &mut report,
            impact,
            hit.normal.as_vec3(),
            direction.normalize_or_zero(),
            radius as f32 * VOXEL_SIZE_M,
            profile,
            seed,
        );
        FireResult {
            hit: true,
            removed: report.removed.len(),
            pieces: spawn.sizes.len(),
            piece_sizes: spawn.sizes,
            dirty_bricks: report.dirty_bricks,
            dust_voxels: spawn.dust_voxels,
            dust_points: spawn.dust_points,
            speed_sum: spawn.speed_sum,
            speed_max: spawn.speed_max,
        }
    }

    /// Detonates a rocket at an arbitrary world point: carves a sphere (plus a
    /// short scatter lobe inside the solid) and spawns debris.
    ///
    /// Unlike [`fire`] this does not need a ray hit, so a rocket that lands on
    /// the floor, a wall or the side of a prop explodes where it actually
    /// stopped instead of where the crosshair happens to look.
    pub fn explode(
        &mut self,
        world: &mut VoxelWorld,
        point: Vec3,
        surface_normal: Vec3,
        radius_voxels: i32,
        seed: u32,
        profile: ImpactProfile,
    ) -> FireResult {
        let radius = radius_voxels.clamp(4, 20);
        let center = (point / VOXEL_SIZE_M).floor().as_ivec3();
        let command = VoxelCommand::remove(
            u64::from(self.sequence),
            self.sequence,
            center,
            radius,
            seed,
        );
        self.sequence = self.sequence.wrapping_add(1);
        let mut report = apply_command(world, &command);
        if report.removed.is_empty() {
            return FireResult::missed(true, report.dirty_bricks);
        }
        // Real blasts are not circles. Add two or three deterministic lobes
        // around the main sphere so the opening reads as a chaotic splat
        // instead of a drilled disc. The command seed drives every offset, so a
        // replay carves exactly the same hole.
        let sample = |k: u32| hash_unit(seed, k.wrapping_mul(0x9e37_79b9));
        let lobe_count = 2 + (sample(1) * 2.0) as i32;
        for lobe in 0..lobe_count {
            let angle = sample(2 + lobe as u32) * std::f32::consts::TAU;
            // Lobes reach well outside the main sphere: with offsets capped at
            // the radius the union of spheres stayed visually circular.
            let reach = radius as f32 * (0.55 + 0.75 * sample(8 + lobe as u32));
            let lift = (sample(14 + lobe as u32) - 0.35) * reach * 0.8;
            let offset = IVec3::new(
                (angle.cos() * reach) as i32,
                lift as i32,
                (angle.sin() * reach) as i32,
            );
            let lobe_radius = (radius as f32 * (0.30 + 0.35 * sample(20 + lobe as u32)))
                .round()
                .max(3.0) as i32;
            let command = VoxelCommand::remove(
                u64::from(self.sequence),
                self.sequence,
                center + offset,
                lobe_radius,
                seed.rotate_left(11 + lobe as u32),
            );
            self.sequence = self.sequence.wrapping_add(1);
            let mut extra = apply_command(world, &command);
            report.added += extra.added;
            report.removed.append(&mut extra.removed);
            report.dirty_bricks.append(&mut extra.dirty_bricks);
        }
        // Rim bites: a handful of smaller off-centre removals chew the crater
        // edge, so the silhouette is a ragged splat instead of a union of
        // circles. Each bite is its own deterministic command, so the command
        // log still replays to the same world hash.
        let bite_count = 3 + (sample(30) * 3.0) as i32;
        for bite in 0..bite_count {
            let angle = sample(40 + bite as u32) * std::f32::consts::TAU;
            let reach = radius as f32 * (0.75 + 0.55 * sample(50 + bite as u32));
            let bite_radius = (radius as f32 * (0.18 + 0.16 * sample(60 + bite as u32)))
                .round()
                .max(2.0) as i32;
            let offset = IVec3::new(
                (angle.cos() * reach) as i32,
                ((sample(70 + bite as u32) - 0.4) * reach * 0.9) as i32,
                (angle.sin() * reach) as i32,
            );
            let command = VoxelCommand::remove(
                u64::from(self.sequence),
                self.sequence,
                center + offset,
                bite_radius,
                seed.rotate_left(19 + bite as u32),
            );
            self.sequence = self.sequence.wrapping_add(1);
            let mut extra = apply_command(world, &command);
            report.added += extra.added;
            report.removed.append(&mut extra.removed);
            report.dirty_bricks.append(&mut extra.dirty_bricks);
        }
        report
            .dirty_bricks
            .sort_unstable_by_key(|brick| (brick.x, brick.y, brick.z));
        report.dirty_bricks.dedup();
        // The impact face decides which half-space the blast may push into.
        // Deriving it from the crater centre gave a zero vector whenever the
        // rocket hit the centre of a voxel, which then fell back to +Y and
        // cancelled every wall-normal impulse: the classic "bazooka only
        // drilled a hole" look. Use the caller's real surface normal, and only
        // fall back to the radial crater direction if it is degenerate.
        let radial = (center.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE_M - point;
        let normal = if surface_normal.length_squared() > 1.0e-6 {
            surface_normal.normalize()
        } else if radial.length_squared() > 1.0e-6 {
            radial.normalize()
        } else {
            Vec3::Y
        };
        let spawn = self.spawn_pieces(
            world,
            &mut report,
            point,
            normal,
            normal,
            radius as f32 * VOXEL_SIZE_M,
            profile,
            seed,
        );
        FireResult {
            hit: true,
            removed: report.removed.len(),
            pieces: spawn.sizes.len(),
            piece_sizes: spawn.sizes,
            dirty_bricks: report.dirty_bricks,
            dust_voxels: spawn.dust_voxels,
            dust_points: spawn.dust_points,
            speed_sum: spawn.speed_sum,
            speed_max: spawn.speed_max,
        }
    }

    /// Right-click test tool: fills an air pocket with a material sphere.
    pub fn fill(
        &mut self,
        world: &mut VoxelWorld,
        origin: Vec3,
        direction: Vec3,
        radius_voxels: i32,
        material: u16,
        seed: u32,
    ) -> Edits {
        let Some(hit) = world.raycast(origin, direction, 256.0) else {
            return Edits::default();
        };
        // The ray hits the crater floor; the air pocket to refill is in front of
        // that face, so the fill sphere sits one voxel outside the surface.
        let center = hit.position + hit.normal;
        let command = VoxelCommand::add(
            u64::from(self.sequence),
            self.sequence,
            center,
            radius_voxels.clamp(3, 14),
            material,
            seed,
        );
        self.sequence = self.sequence.wrapping_add(1);
        let report = apply_command(world, &command);
        Edits {
            changed: report.added as usize,
            dirty_bricks: report.dirty_bricks,
        }
    }

    /// Turns one crater into debris:
    ///
    /// 1. A 6-connected flood fill finds the solid components the crater cut
    ///    free from the ground or the search boundary; each becomes one body
    ///    shaped by its own geometry (a roof slab stays a slab).
    /// 2. The voxels inside the crater are shattered into chips whose size falls
    ///    with the deposited energy, so a rocket turns concrete into rubble while
    ///    a weak hit leaves fewer, bigger chunks.
    /// 3. Chips below [`MIN_PIECE_VOXELS`] are merged into a neighbour that they
    ///    touch (or one within [`MERGE_RADIUS_M`]) so the pile keeps a natural
    ///    size spread instead of hundreds of one-voxel bodies. A leftover with
    ///    no neighbour becomes cosmetic dust.
    /// 4. Every remaining cluster is split into its 6-connected components and
    ///    only the largest one stays a body (see [`keep_connected_clusters`]):
    ///    merged crumbs and multi-seed growth can leave islands that would fly
    ///    as lonely voxels and make the chunk look perforated.
    ///
    /// Detached voxels leave the world through [`apply_voxel_edits`], so every
    /// chunk that becomes a body is recorded and none is silently deleted.
    #[allow(clippy::too_many_arguments)]
    fn spawn_pieces(
        &mut self,
        world: &mut VoxelWorld,
        report: &mut EditReport,
        impact: Vec3,
        surface_normal: Vec3,
        aim: Vec3,
        blast_radius: f32,
        profile: ImpactProfile,
        seed: u32,
    ) -> ShotSpawn {
        let core = split_core(
            &report.removed,
            impact,
            blast_radius,
            profile.fragmentation_energy(),
            seed,
        );
        let (detached, extra_dirty) = detach_unanchored(world, &report.removed);
        report.dirty_bricks.extend(extra_dirty);
        report
            .dirty_bricks
            .sort_unstable_by_key(|brick| (brick.x, brick.y, brick.z));
        report.dirty_bricks.dedup();

        let mut clusters: Vec<(Vec<VoxelSample>, f32)> =
            Vec::with_capacity(core.len() + detached.len());
        clusters.extend(core.into_iter().map(|piece| (piece, 1.0)));
        clusters.extend(
            detached
                .into_iter()
                .map(|piece| (piece, DETACHED_BLAST_SHARE)),
        );
        let (clusters, consolidated_dust) = consolidate_clusters(clusters, self.min_piece);
        // A merged cluster is big enough but not necessarily one piece of
        // matter: `split_core` grows several seeds at once and crumbs are
        // grouped blindly. Keep the largest connected component per body and
        // return the islands as dust, so nothing flies as a lonely cube.
        let (clusters, island_dust) = keep_connected_clusters(clusters, self.min_piece);
        // About a third of the small clusters vanish into dust instead of
        // becoming rigid bodies. A real blast pulverises part of the wall, and
        // those chips were exactly the "visual litter" that flew across the
        // map with nothing holding them together. The voxels still count as
        // dust, so voxel conservation is unchanged.
        let mut kept: Vec<(Vec<VoxelSample>, f32)> = Vec::with_capacity(clusters.len());
        let mut vanished = consolidated_dust;
        vanished.extend(island_dust);
        for (cluster, share) in clusters {
            let roll = hash_unit(seed ^ 0x5bd1_e995, cluster.len() as u32);
            // Pulverised material that reads as visual litter: a little under
            // half of everything up to ~120 voxels simply vanishes as dust, so
            // a blast leaves intact chunks and a cloud, not a carpet of chips.
            // The first two clusters always survive, so even a small crater
            // still produces visible debris.
            if cluster.len() <= 120 && kept.len() >= 2 && roll < 0.45 {
                vanished.extend(cluster);
            } else {
                kept.push((cluster, share));
            }
        }
        let clusters = kept;
        let dust = vanished;
        self.dust_voxels += dust.len() as u64;
        let dust_points = dust_grains(&dust);
        let (sizes, speed_sum, speed_max) = self.spawn_clusters(
            world,
            &clusters,
            impact,
            surface_normal,
            aim,
            blast_radius,
            profile,
            seed,
        );
        ShotSpawn {
            sizes,
            dust_voxels: dust.len(),
            dust_points,
            speed_sum,
            speed_max,
        }
    }

    /// Splits the removed voxels into clusters and spawns one rigid body per
    /// cluster. Nothing about the motion is authored by hand:
    ///
    /// - mass = voxel volume x palette density (kg);
    /// - centre of mass and the inertia tensor come from the same voxel set;
    /// - the shot energy is deposited over the crater volume, so the fluence
    ///   inside it is close to uniform: each fragment absorbs energy in
    ///   proportion to the area it presents to the blast (and falls off as
    ///   `1/r^2` outside the crater);
    /// - the impulse acts at the fragment's centre of pressure (the centroid of
    ///   the voxels facing the blast, weighted by how squarely they face it),
    ///   so an asymmetric fragment gets a real lever arm and tumbles while a
    ///   symmetric one barely spins;
    /// - the component pointing into the remaining solid is cancelled by it,
    ///   so debris slides out of the crater instead of being driven into the
    ///   floor.
    ///
    /// For an impulse `J` along `n` applied at lever `r`:
    /// `dv = J n / m`, `domega = I^-1 (r x J n)`. Solving
    /// `0.5 J^2 (1/m + (r x n)^T I^-1 (r x n)) = E` makes the fragment's total
    /// kinetic energy equal to the energy it absorbed, so translation and
    /// rotation are solved, never tuned.
    ///
    /// Kinetic rounds skip the solve: their fragments get a small, uniform push
    /// out of the hole (see [`ImpactProfile`]) and only gravity moves them after.
    #[allow(clippy::too_many_arguments)]
    fn spawn_clusters(
        &mut self,
        world: &mut VoxelWorld,
        clusters: &[(Vec<VoxelSample>, f32)],
        impact: Vec3,
        surface_normal: Vec3,
        aim: Vec3,
        blast_radius: f32,
        profile: ImpactProfile,
        seed: u32,
    ) -> (Vec<u32>, f32, f32) {
        let palette = world.palette();
        let voxel_volume = VOXEL_SIZE_M * VOXEL_SIZE_M * VOXEL_SIZE_M;

        // First pass: mass properties and blast geometry, so the energy can be
        // normalised over the whole crater instead of per fragment.
        let mut builds = Vec::with_capacity(clusters.len());
        let mut total_weight = 0.0f32;
        for (members, share) in clusters {
            if members.is_empty() {
                builds.push(None);
                continue;
            }
            let mut min = IVec3::splat(i32::MAX);
            let mut max = IVec3::splat(i32::MIN);
            let mut mass = 0.0f32;
            let mut weighted_center = Vec3::ZERO;
            for member in members {
                let position = member.position;
                min = min.min(position);
                max = max.max(position);
                // Palette density is g/cm^3, which is numerically t/m^3.
                let voxel_mass =
                    voxel_volume * palette.material(member.voxel.material()).density * 1000.0;
                mass += voxel_mass;
                let center = position.as_vec3() + Vec3::splat(0.5);
                weighted_center += center * voxel_mass;
            }
            let mass = mass.max(1e-3);
            let com = weighted_center / mass;
            let body_center = com * VOXEL_SIZE_M;
            let count = members.len() as u32;
            let distance = (body_center - impact).length();
            let radial = if distance > 1e-4 {
                (body_center - impact) / distance
            } else {
                surface_normal
            };

            let dims = max - min + IVec3::ONE;
            let mut voxels = vec![0u16; (dims.x * dims.y * dims.z) as usize];
            let mut inertia = Mat3::ZERO;
            // Centre of pressure: the centroid of the voxels facing the blast,
            // weighted by how squarely they face it. That is where the
            // resultant of the distributed blast pressure acts, so it is the
            // torque arm. A symmetric fragment puts it on the radial line and
            // gets no spin; an irregular one gets a real lever.
            let mut pressure_sum = 0.0f32;
            let mut pressure_center = Vec3::ZERO;
            for member in members {
                let local = member.position - min;
                let index = local.x + dims.x * (local.y + dims.y * local.z);
                voxels[index as usize] = member.voxel.raw();
                let voxel_mass =
                    voxel_volume * palette.material(member.voxel.material()).density * 1000.0;
                let offset = ((member.position.as_vec3() + Vec3::splat(0.5)) - com) * VOXEL_SIZE_M;
                // Point-mass inertia: sum m * (|r|^2 * identity - r * r^T).
                let outer =
                    Mat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z);
                inertia += (Mat3::IDENTITY * offset.length_squared() - outer) * voxel_mass;
                let facing = (-offset.dot(radial)).max(0.0);
                if facing > 0.0 {
                    pressure_sum += facing;
                    pressure_center += offset * facing;
                }
            }
            // A single voxel, or a flat wall panel, contributes a
            // near-singular tensor: one principal moment is almost zero. Jolt
            // rejects bodies whose inertia is not positive definite and well
            // conditioned, which used to return those pieces to the world and
            // make them freeze mid-air. Lift every axis by 1% of the trace so
            // the condition number stays bounded for thin clusters too.
            let trace = inertia.x_axis.x + inertia.y_axis.y + inertia.z_axis.z;
            inertia += Mat3::IDENTITY * (trace * 0.01).max(mass * 1e-4).max(1e-7);
            let lever = if pressure_sum > 0.0 {
                pressure_center / pressure_sum
            } else {
                Vec3::ZERO
            };

            // Area of the fragment's box projected onto the blast direction.
            let half = (max.as_vec3() + Vec3::ONE - min.as_vec3()) * (VOXEL_SIZE_M * 0.5);
            let area = 4.0
                * (radial.x.abs() * half.y * half.z
                    + radial.y.abs() * half.x * half.z
                    + radial.z.abs() * half.x * half.y);
            // Spawn the body just clear of the old static collider. Sector
            // collisions rebuild asynchronously, so a piece authored inside
            // the intact wall would be trapped by the stale boxes and lose its
            // outward velocity before the new sector lands: it then just sags
            // down the facade. Offset by the half-extent along the impact
            // normal, plus a small skin, so even the deepest cells start free.
            let skin = half.dot(surface_normal.abs()) + 0.04 + blast_radius * 0.03;
            let spawn_center = body_center + surface_normal * skin;
            // Energy is deposited across the whole crater, so inside it the
            // fluence is close to uniform; towards the rim it falls off as 1/r^2.
            // Pieces beyond the blast radius are outside the shock wave
            // altogether: they just fall, without an impulse, so the wave has a
            // hard edge instead of an endless 1/r^2 tail.
            let inside_wave = distance <= blast_radius;
            let reach = distance.max(blast_radius * 0.5).max(VOXEL_SIZE_M);
            // A detached structural component (a roof, a wall panel) only gets a
            // nudge; the crater chips get the full blast fluence. The old hard
            // zero at the radius left the whole detached wall frozen in place,
            // which is what made a bazooka hit look like it only carved a hole.
            // A short 1/r^2 tail keeps the wave coherent while still moving the
            // panels just outside the crater.
            let weight = if inside_wave {
                area / (reach * reach) * *share
            } else {
                let tail = (blast_radius / distance.max(blast_radius)).clamp(0.0, 1.0);
                // Detached panels often sit outside the nominal crater even
                // though the blast unseated them. A linear tail keeps real
                // wall chunks moving instead of cancelling their energy.
                area / (reach * reach) * *share * tail * 0.60
            };
            total_weight += weight;

            // A surface blast must push the wall panel out of its own face.
            // A raw radial vector points back into the wall for fragments
            // behind the impact plane, and merely adding a normal component
            // still left them moving inward. Strip the inward normal part,
            // keep the sideways scatter, then add a strong outward push for
            // every fragment.
            let along = radial.dot(surface_normal);
            let tangential = radial - surface_normal * along;
            let direction =
                (tangential + surface_normal * (0.90 + along.max(0.0))).normalize_or_zero();
            // Explosions must throw rubble sideways as well as away from the
            // camera. A radial-only impulse on a flat wall mostly cancels into
            // the remaining solid, so add a deterministic tangent fan and a
            // little lift. The hash is keyed by the shot seed and the piece
            // origin, so replays still produce the same debris field.
            let lateral_axis = if surface_normal.length_squared() > 1e-6 {
                surface_normal.cross(Vec3::Y)
            } else {
                Vec3::Y.cross(radial)
            };
            let lateral_axis = if lateral_axis.length_squared() > 1e-4 {
                lateral_axis.normalize()
            } else {
                Vec3::X
            };
            let local_seed = hash_u32(
                seed,
                (min.x as u32).wrapping_mul(0x9e37_79b9)
                    ^ (min.y as u32).wrapping_mul(0x85eb_ca6b)
                    ^ (min.z as u32).wrapping_mul(0xc2b2_ae35),
            );
            let fan = hash_unit(seed ^ 0x85eb_ca6b, local_seed) * 2.0 - 1.0;
            let lift = 0.20 + 0.35 * hash_unit(seed.rotate_left(13), local_seed);
            let direction = (direction + lateral_axis * fan + Vec3::Y * lift).normalize_or_zero();

            builds.push(Some(PieceBuild {
                dims,
                voxels,
                min,
                center_voxels: com,
                local_min: (min.as_vec3() - com) * VOXEL_SIZE_M,
                local_max: (max.as_vec3() + Vec3::ONE - com) * VOXEL_SIZE_M,
                body_center: spawn_center,
                mass,
                inertia,
                inertia_inverse: inertia.inverse(),
                lever,
                direction,
                weight,
                voxel_count: count,
                local_seed,
            }));
        }

        // Second pass: solve the impulse and spawn.
        let blast_energy = profile.blast_energy();
        let mut sizes = Vec::with_capacity(builds.iter().flatten().count());
        let mut speed_sum = 0.0f32;
        let mut speed_max = 0.0f32;
        for build in builds.into_iter().flatten() {
            let (linear_velocity, angular_velocity) = match profile {
                ImpactProfile::Explosive { .. } => {
                    let energy = if total_weight > 0.0 {
                        blast_energy * build.weight / total_weight
                    } else {
                        0.0
                    };
                    let mass_inverse = 1.0 / build.mass;
                    let torque_axis = build.lever.cross(build.direction);
                    let rotational = torque_axis.dot(build.inertia_inverse * torque_axis);
                    let impulse = (2.0 * energy / (mass_inverse + rotational).max(1e-9))
                        .max(0.0)
                        .sqrt();
                    // A close-range shot concentrates the whole TNT budget into
                    // one light piece, which can push the solved spin past
                    // Jolt's `MAX_ANGULAR_VELOCITY`; Jolt then rejects the body
                    // entirely and the debris silently disappears. Clamp the
                    // solved values instead.
                    (
                        clamp_velocity(
                            build.direction * (impulse * mass_inverse),
                            MAX_LINEAR_SPEED,
                        ),
                        clamp_velocity(
                            build.inertia_inverse * (torque_axis * impulse),
                            MAX_ANGULAR_SPEED,
                        ),
                    )
                }
                // A kinetic hit has no shock wave: every chip it freed is pushed
                // out of the hole along the bullet direction (plus the surface
                // normal), with a small deterministic spread and a slight tumble.
                ImpactProfile::Kinetic { .. } => {
                    let outward = (build.body_center - impact).normalize_or_zero();
                    let normal = if surface_normal.length_squared() > 1e-6 {
                        surface_normal
                    } else {
                        Vec3::Y
                    };
                    let push = (normal * 0.55 + aim * 0.6 + outward * 0.25).normalize_or_zero();
                    let spin_axis =
                        (normal.cross(aim) + outward.cross(Vec3::Y)).normalize_or_zero();
                    let grain = hash_unit(seed, build.local_seed);
                    let speed = KINETIC_PUSH_SPEED * (0.7 + 0.6 * grain);
                    let spin = KINETIC_SPIN * (0.4 + 0.8 * grain);
                    (
                        clamp_velocity(push * speed, MAX_LINEAR_SPEED),
                        clamp_velocity(spin_axis * spin, MAX_ANGULAR_SPEED),
                    )
                }
            };

            let boxes = compound_boxes(build.dims, &build.voxels, build.min, build.center_voxels);
            let mut desc = BodyDesc::new(build.body_center, boxes, build.mass);
            desc.rotation = Quat::IDENTITY;
            // The tensor is symmetric, so columns and rows are the same.
            desc.inertia = Some(build.inertia.to_cols_array_2d());
            desc.linear_velocity = linear_velocity;
            desc.angular_velocity = angular_velocity;

            if let Some(handle) = self.physics.spawn(&desc) {
                speed_sum += desc.linear_velocity.length();
                speed_max = speed_max.max(desc.linear_velocity.length());
                self.pieces.push(DebrisPiece {
                    handle,
                    mass_kg: build.mass,
                    dims: build.dims,
                    voxels: build.voxels,
                    local_min: build.local_min,
                    local_max: build.local_max,
                    voxel_count: build.voxel_count,
                    prev_position: build.body_center,
                    prev_rotation: Quat::IDENTITY,
                    position: build.body_center,
                    rotation: Quat::IDENTITY,
                    rest_time: 0.0,
                    last_speed: 0.0,
                });
                sizes.push(build.voxel_count);
            } else {
                // Jolt can still reject a body (degenerate mass/inertia). The
                // voxels were already removed from the world, so putting them
                // back is the only way to keep matter conserved; otherwise a
                // rejected chip silently disappears from the simulation.
                tracing::warn!(
                    voxels = build.voxel_count,
                    "physics rejected a debris body; returning its voxels to the world"
                );
                for z in 0..build.dims.z {
                    for y in 0..build.dims.y {
                        for x in 0..build.dims.x {
                            let raw =
                                build.voxels[(x + build.dims.x * (y + build.dims.y * z)) as usize];
                            if raw != 0 {
                                world.set_voxel(
                                    build.min.x + x,
                                    build.min.y + y,
                                    build.min.z + z,
                                    Voxel::from_raw(raw),
                                );
                            }
                        }
                    }
                }
            }
        }
        (sizes, speed_sum, speed_max)
    }
}

/// Per-cluster geometry and mass properties collected before the impulse solve.
struct PieceBuild {
    dims: IVec3,
    voxels: Vec<u16>,
    min: IVec3,
    center_voxels: Vec3,
    local_min: Vec3,
    local_max: Vec3,
    body_center: Vec3,
    mass: f32,
    inertia: Mat3,
    inertia_inverse: Mat3,
    lever: Vec3,
    direction: Vec3,
    weight: f32,
    voxel_count: u32,
    /// Stable per-piece random slot, so the kinetic spread is deterministic and
    /// differs between fragments of the same shot.
    local_seed: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Edits {
    pub changed: usize,
    pub dirty_bricks: Vec<IVec3>,
}

/// One voxel that left the static world and now belongs to a body.
#[derive(Clone, Copy)]
struct VoxelSample {
    position: IVec3,
    voxel: Voxel,
}

/// A detached structural component keeps only a small share of the blast
/// energy: it was released by the crater, not thrown by it, so it should drop
/// under gravity with a nudge instead of being launched like a chip.
const DETACHED_BLAST_SHARE: f32 = 0.12;

/// Half-extent of the box searched for components cut loose by a crater.
///
/// 3 m instead of 6 m: the wider box reached whole load-bearing slabs that had
/// nothing to do with the shot, and shooter feedback was "I hit a wall and the
/// floor above fell off".
const DETACH_MARGIN: i32 = 16;
/// Upper bound on voxels visited per shot, so a pathological solid volume can
/// never turn one trigger pull into a multi-millisecond flood fill.
const DETACH_VISIT_BUDGET: usize = 80_000;
/// A component larger than this is treated as load-bearing and stays static.
///
/// 600 voxels is roughly 1.2 m3: a chunk of masonry, not a structural element.
/// The old 16 000 allowed a whole storey to be handed to Jolt as one body.
const DETACH_MAX_COMPONENT: usize = 300;
/// Total voxels one shot may turn into detached bodies.
const DETACH_MAX_VOXELS: usize = 1_800;
/// Total detached bodies one shot may create.
const DETACH_MAX_COMPONENTS: usize = 12;

const NEIGHBOURS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// 6-connected flood fill for the components a crater freed.
///
/// Seeds are the solid voxels touching the removed set. A component is anchored
/// when it reaches the floor (`y <= 0`) or the edge of the search box, i.e. it
/// still hangs on the rest of the world. Everything else is detached, removed
/// from the world and returned as voxel samples, one body per component.
///
/// The traversal only reads; the world is mutated afterwards in sorted order,
/// so the outcome does not depend on hash-map iteration order.
fn detach_unanchored(
    world: &mut VoxelWorld,
    removed: &[RemovedVoxel],
) -> (Vec<Vec<VoxelSample>>, Vec<IVec3>) {
    if removed.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    for voxel in removed {
        min = min.min(voxel.position);
        max = max.max(voxel.position);
    }
    let lo = min - IVec3::splat(DETACH_MARGIN);
    let hi = max + IVec3::splat(DETACH_MARGIN);
    let inside = |point: IVec3| point.cmpge(lo).all() && point.cmple(hi).all();

    let mut seeds: Vec<IVec3> = Vec::new();
    for voxel in removed {
        for direction in NEIGHBOURS {
            let point = voxel.position + direction;
            if inside(point) && world.voxel(point.x, point.y, point.z).is_solid() {
                seeds.push(point);
            }
        }
    }
    seeds.sort_unstable_by_key(|point| (point.x, point.y, point.z));
    seeds.dedup();
    // Thin, mostly free-standing voxels first: those are the likely detached
    // pieces, so they are classified before a big solid mass can eat the budget.
    let mut cost: Vec<(u8, IVec3)> = seeds
        .iter()
        .map(|&seed| {
            let neighbours = NEIGHBOURS
                .iter()
                .filter(|&&direction| {
                    let point = seed + direction;
                    inside(point) && world.voxel(point.x, point.y, point.z).is_solid()
                })
                .count();
            (neighbours as u8, seed)
        })
        .collect();
    cost.sort_unstable_by_key(|(neighbours, seed)| (*neighbours, seed.x, seed.y, seed.z));

    let mut seen: std::collections::HashSet<IVec3> = std::collections::HashSet::new();
    let mut budget = DETACH_VISIT_BUDGET;
    let mut candidates: Vec<Vec<IVec3>> = Vec::new();
    for (_, seed) in cost {
        if budget == 0 || candidates.len() >= DETACH_MAX_COMPONENTS {
            break;
        }
        if seen.contains(&seed) || !world.voxel(seed.x, seed.y, seed.z).is_solid() {
            continue;
        }
        let mut stack = vec![seed];
        seen.insert(seed);
        let mut component = Vec::new();
        let mut anchored = false;
        while let Some(point) = stack.pop() {
            component.push(point);
            if point.y <= 0
                || point.x <= lo.x
                || point.x >= hi.x
                || point.y >= hi.y
                || point.z <= lo.z
                || point.z >= hi.z
                || component.len() > DETACH_MAX_COMPONENT
            {
                anchored = true;
            }
            if anchored {
                break;
            }
            for direction in NEIGHBOURS {
                let next = point + direction;
                if inside(next)
                    && !seen.contains(&next)
                    && world.voxel(next.x, next.y, next.z).is_solid()
                {
                    seen.insert(next);
                    stack.push(next);
                }
            }
            budget = budget.saturating_sub(1);
            if budget == 0 {
                break;
            }
        }
        if !anchored {
            candidates.push(component);
        }
    }

    let mut edits: Vec<(IVec3, Voxel)> = Vec::new();
    let mut pieces: Vec<Vec<VoxelSample>> = Vec::new();
    let mut detached_voxels = 0usize;
    for component in candidates {
        if detached_voxels + component.len() > DETACH_MAX_VOXELS {
            break;
        }
        detached_voxels += component.len();
        let mut samples = Vec::with_capacity(component.len());
        for position in component {
            let voxel = world.voxel(position.x, position.y, position.z);
            if !voxel.is_solid() {
                continue;
            }
            samples.push(VoxelSample { position, voxel });
            edits.push((position, Voxel::AIR));
        }
        if !samples.is_empty() {
            pieces.push(samples);
        }
    }
    if edits.is_empty() {
        return (Vec::new(), Vec::new());
    }
    edits.sort_unstable_by_key(|(position, _)| (position.x, position.y, position.z));
    let report = apply_voxel_edits(world, &edits);
    debug_assert_eq!(report.removed.len(), detached_voxels);
    (pieces, report.dirty_bricks)
}

/// Shatters the crater voxels into chips.
///
/// Pieces grow from deterministic seeds along 6-connectivity, so every body is
/// one connected piece of real crater geometry rather than a slice of a
/// jittered grid. The target size is a function of distance from the impact
/// (the core shatters into crumbs, the rim keeps bigger sections), of the
/// deposited energy (a stronger charge makes everything finer) and of a hash
/// (a wide power-law spread: many tiny chips, fewer medium chunks, rare big
/// ones). Boundary voxels are then eroded by 10-30 % and re-emitted as small
/// crumb pieces, which keeps the mass exact and makes the fracture edges rough.
fn split_core(
    removed: &[RemovedVoxel],
    impact: Vec3,
    blast_radius: f32,
    energy_joules: f32,
    seed: u32,
) -> Vec<Vec<VoxelSample>> {
    use std::collections::{HashMap, HashSet, VecDeque};

    if removed.is_empty() {
        return Vec::new();
    }
    let mut index_of: HashMap<IVec3, u32> = HashMap::with_capacity(removed.len());
    for (index, voxel) in removed.iter().enumerate() {
        index_of.insert(voxel.position, index as u32);
    }
    let mut assigned = vec![false; removed.len()];
    let impact_voxel = impact / VOXEL_SIZE_M;
    let radius_voxels = (blast_radius / VOXEL_SIZE_M).max(1.0);
    let energy_scale = (energy_joules.max(0.0) / 40_000.0).clamp(0.0, 1.0);
    let core_shrink = 1.0 - 0.35 * energy_scale;

    // Grow from the rim inward. The outer seeds get the large targets, so a
    // blast leaves a few big wall sections at the edge and chips in the core,
    // instead of cutting everything to the same average size. Ties break on a
    // seeded hash, which keeps the command deterministic.
    let mut order: Vec<u32> = (0..removed.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        let da = (removed[a as usize].position.as_vec3() - impact_voxel).length_squared();
        let db = (removed[b as usize].position.as_vec3() - impact_voxel).length_squared();
        db.partial_cmp(&da)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| hash_u32(seed ^ 0x51ed_270b, a).cmp(&hash_u32(seed ^ 0x51ed_270b, b)))
    });

    const NEIGHBOURS: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    let mut pieces: Vec<Vec<VoxelSample>> = Vec::new();
    let mut crumbs: Vec<VoxelSample> = Vec::new();
    for &start in &order {
        if assigned[start as usize] {
            continue;
        }
        let start_voxel = removed[start as usize].position;
        let radial =
            ((start_voxel.as_vec3() - impact_voxel).length() / radius_voxels).clamp(0.0, 1.5);
        let random = hash_u32(seed ^ 0xa511_e9b3, start) as f32 / u32::MAX as f32;
        // Quadratic-to-cubic bias: most pieces sit near the low end, a few are
        // much larger, and the whole scale grows with distance from the core.
        let base = (2.5 + 150.0 * radial.powf(1.7)) * core_shrink;
        let mut target = (base * (0.35 + 1.9 * random * random * random)).round() as usize;
        target = target.clamp(1, 320);
        // A sprinkle of single-voxel chips in the hot core.
        if radial < 0.75 && hash_u32(seed ^ 0x9e37_79b1, start) % 100 < 8 {
            target = 1;
        }
        if pieces.len() >= 96 {
            // Hard cap on body count per blast; the rest becomes one large pile
            // instead of a thousand Jolt bodies.
            target = 96;
        }

        let mut queue = VecDeque::new();
        let mut members = Vec::new();
        assigned[start as usize] = true;
        queue.push_back(start);
        while let Some(index) = queue.pop_front() {
            members.push(VoxelSample {
                position: removed[index as usize].position,
                voxel: removed[index as usize].voxel,
            });
            if members.len() >= target {
                // Hand the discovered-but-uncollected neighbours back to the
                // pool so the next seed can use them.
                for pending in queue.drain(..) {
                    assigned[pending as usize] = false;
                }
                break;
            }
            let position = removed[index as usize].position;
            for (dx, dy, dz) in NEIGHBOURS {
                let neighbour = position + IVec3::new(dx, dy, dz);
                if let Some(&next) = index_of.get(&neighbour)
                    && !assigned[next as usize]
                {
                    assigned[next as usize] = true;
                    queue.push_back(next);
                }
            }
        }

        // Erode 10-30 % of the boundary voxels into crumbs. Only pieces with
        // enough interior are eroded; a 1-voxel chip has no boundary to lose.
        if members.len() >= 8 {
            let occupied: HashSet<IVec3> = members.iter().map(|member| member.position).collect();
            let mut boundary: Vec<usize> = Vec::new();
            for (index, member) in members.iter().enumerate() {
                let mut solid_neighbours = 0u32;
                for (dx, dy, dz) in NEIGHBOURS {
                    if occupied.contains(&(member.position + IVec3::new(dx, dy, dz))) {
                        solid_neighbours += 1;
                    }
                }
                if solid_neighbours < 6 {
                    boundary.push(index);
                }
            }
            boundary.sort_unstable_by_key(|&index| {
                hash_u32(seed ^ 0xb529_7a4d, index as u32 ^ (members.len() as u32))
            });
            let fraction =
                0.10 + 0.20 * (hash_u32(seed ^ 0x68e3_1da4, start) as f32 / u32::MAX as f32);
            let erode = ((boundary.len() as f32) * fraction).round() as usize;
            let mut eroded = vec![false; members.len()];
            for &index in boundary.iter().take(erode) {
                eroded[index] = true;
            }
            let mut kept = Vec::with_capacity(members.len() - erode);
            for (index, member) in members.drain(..).enumerate() {
                if eroded[index] {
                    crumbs.push(member);
                } else {
                    kept.push(member);
                }
            }
            members = kept;
        }
        if !members.is_empty() {
            pieces.push(members);
        }
    }

    // Group crumbs into small 1-4 voxel bodies so erosion does not create a
    // thousand singleton colliders. Sorting keeps groups spatially local.
    crumbs.sort_unstable_by_key(|voxel| (voxel.position.x, voxel.position.y, voxel.position.z));
    for chunk in crumbs.chunks(4) {
        pieces.push(chunk.to_vec());
    }
    pieces
}

/// True when a voxel coordinate is inside the world. Baked debris must stay in
/// bounds: `VoxelWorld::set_voxel` silently ignores writes outside them.
fn in_bounds(cell: IVec3) -> bool {
    cell.x >= 0
        && cell.y >= 0
        && cell.z >= 0
        && (cell.x as usize) < VOXELS_X
        && (cell.y as usize) < VOXELS_Y
        && (cell.z as usize) < VOXELS_Z
}

/// Cells a resting piece tries in order when its own cell is already occupied
/// (see [`DestructionSystem::bake_edits`]). Radius 2 voxels covers the usual
/// "rubble lying on the floor tile it broke off" case without smearing a whole
/// pile outwards; the order is distance first, then a fixed axis order, so a
/// bake is deterministic.
fn settle_offsets() -> &'static [IVec3] {
    static OFFSETS: std::sync::OnceLock<Vec<IVec3>> = std::sync::OnceLock::new();
    OFFSETS.get_or_init(|| {
        let mut offsets = Vec::new();
        for z in -2..=2 {
            for y in -2..=2 {
                for x in -2..=2 {
                    if x == 0 && y == 0 && z == 0 {
                        continue;
                    }
                    offsets.push(IVec3::new(x, y, z));
                }
            }
        }
        offsets
            .sort_unstable_by_key(|offset| (offset.length_squared(), offset.x, offset.y, offset.z));
        offsets
    })
}

fn hash_u32(seed: u32, salt: u32) -> u32 {
    let mut value = seed ^ salt.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

/// Deterministic 0..1 value, used for the kinetic spread.
fn hash_unit(seed: u32, salt: u32) -> f32 {
    hash_u32(seed, salt) as f32 / u32::MAX as f32
}

/// Smallest voxel of a cluster, used as a deterministic tie-breaker.
fn min_position(samples: &[VoxelSample]) -> (i32, i32, i32) {
    samples
        .iter()
        .fold((i32::MAX, i32::MAX, i32::MAX), |acc, sample| {
            let p = sample.position;
            (acc.0.min(p.x), acc.1.min(p.y), acc.2.min(p.z))
        })
}

/// Average voxel centre of a cluster, in voxel units.
fn centroid(samples: &[VoxelSample]) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for sample in samples {
        sum += sample.position.as_vec3() + Vec3::splat(0.5);
    }
    sum / samples.len().max(1) as f32
}

/// Sort key for the merge order: size first, then the lowest voxel position.
type ClusterKey = (usize, (i32, i32, i32));

/// Merges clusters below `min_piece` into neighbours, so no rigid body is a
/// single voxel, and returns the voxels that had no neighbour as dust.
///
/// Preference order: a touching cluster that is already big enough, then the
/// largest touching cluster, then the nearest cluster within
/// [`MERGE_RADIUS_M`]. Pieces that already meet the minimum are never touched,
/// so the size spread of the rubble survives while the crumb count collapses.
fn consolidate_clusters(
    mut clusters: Vec<(Vec<VoxelSample>, f32)>,
    min_piece: usize,
) -> (Vec<(Vec<VoxelSample>, f32)>, Vec<VoxelSample>) {
    let mut dust = Vec::new();
    if clusters.len() < 2 || min_piece <= 1 {
        return (clusters, dust);
    }
    let total: usize = clusters.iter().map(|(samples, _)| samples.len()).sum();
    let mut owner: HashMap<IVec3, u32> = HashMap::with_capacity(total);
    for (id, (samples, _)) in clusters.iter().enumerate() {
        for sample in samples {
            owner.insert(sample.position, id as u32);
        }
    }

    loop {
        // Smallest cluster first: crumbs disappear before slightly larger chips
        // are touched, and the order is fully deterministic.
        let mut pick: Option<(usize, ClusterKey)> = None;
        for (id, (samples, _)) in clusters.iter().enumerate() {
            if samples.is_empty() || samples.len() >= min_piece {
                continue;
            }
            let key = (samples.len(), min_position(samples));
            if pick.is_none_or(|(_, best)| key < best) {
                pick = Some((id, key));
            }
        }
        let Some((id, _)) = pick else {
            break;
        };

        // 1) A cluster this one actually touches. Prefer one that is already a
        //    legal body, then the largest, then the lowest id.
        let mut touch: Option<(bool, usize, usize)> = None;
        for sample in &clusters[id].0 {
            for offset in NEIGHBOURS {
                let Some(&other) = owner.get(&(sample.position + offset)) else {
                    continue;
                };
                let other = other as usize;
                if other == id || clusters[other].0.is_empty() {
                    continue;
                }
                let size = clusters[other].0.len();
                let rank = (size >= min_piece, size);
                if touch.is_none_or(|(big, best_size, best_id)| {
                    rank > (big, best_size) || (rank == (big, best_size) && other < best_id)
                }) {
                    touch = Some((rank.0, rank.1, other));
                }
            }
        }

        // 2) Otherwise the nearest cluster within the merge radius, which keeps
        //    a chip that landed a few centimetres away in the same body.
        let nearest = if touch.is_some() {
            None
        } else {
            let center = centroid(&clusters[id].0);
            let mut nearest: Option<(f32, usize)> = None;
            for (other, (samples, _)) in clusters.iter().enumerate() {
                if other == id || samples.is_empty() {
                    continue;
                }
                let distance = centroid(samples).distance_squared(center);
                if distance > MERGE_RADIUS_M * MERGE_RADIUS_M {
                    continue;
                }
                if nearest.is_none_or(|(best, best_id)| {
                    distance < best || (distance == best && other < best_id)
                }) {
                    nearest = Some((distance, other));
                }
            }
            nearest.map(|(_, other)| other)
        };

        let target = touch.map(|(_, _, other)| other).or(nearest);
        let Some(target) = target else {
            // Nothing to merge into: the voxels become cosmetic dust and leave
            // the simulation for good. `DestructionSystem::dust_voxels` keeps
            // them in the conservation identity.
            let samples = std::mem::take(&mut clusters[id].0);
            dust.extend(samples);
            continue;
        };

        let samples = std::mem::take(&mut clusters[id].0);
        let source_share = clusters[id].1;
        let source_len = samples.len();
        let destination = &mut clusters[target];
        let destination_len = destination.0.len();
        destination.0.extend(samples);
        destination.1 = if destination_len + source_len > 0 {
            (destination.1 * destination_len as f32 + source_share * source_len as f32)
                / (destination_len + source_len) as f32
        } else {
            destination.1
        };
        for sample in &destination.0[destination_len..] {
            owner.insert(sample.position, target as u32);
        }
    }

    clusters.retain(|(samples, _)| !samples.is_empty());
    (clusters, dust)
}

/// Keeps only the largest 6-connected component of every cluster; the islands
/// become dust.
///
/// [`consolidate_clusters`] only guarantees a minimum *count*, never that the
/// voxels form one lump. Two things break connectivity: `split_core` grows one
/// body from several seeds at once (the fronts can leave a gap), and the eroded
/// crumbs are grouped four at a time regardless of where they sit. A body with
/// islands renders as a chunk with floating specks around it and a hole in the
/// middle, which is the artefact this pass removes.
///
/// A cluster of one voxel is always dust, and so is a largest component below
/// `min_piece`: neither can be a rigid body worth simulating.
fn keep_connected_clusters(
    clusters: Vec<(Vec<VoxelSample>, f32)>,
    min_piece: usize,
) -> (Vec<(Vec<VoxelSample>, f32)>, Vec<VoxelSample>) {
    let mut dust = Vec::new();
    let mut kept = Vec::with_capacity(clusters.len());
    let mut index: HashMap<IVec3, u32> = HashMap::new();
    let mut visited: Vec<bool> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut member: Vec<u32> = Vec::new();
    let mut best: Vec<u32> = Vec::new();
    let minimum = min_piece.max(2);

    for (samples, share) in clusters {
        if samples.len() < minimum {
            dust.extend(samples);
            continue;
        }

        index.clear();
        index.reserve(samples.len());
        for (slot, sample) in samples.iter().enumerate() {
            index.insert(sample.position, slot as u32);
        }
        visited.clear();
        visited.resize(samples.len(), false);
        best.clear();

        // Components are discovered in sample order and ties keep the first
        // one, so the pick is deterministic for a given voxel order.
        for start in 0..samples.len() as u32 {
            if visited[start as usize] {
                continue;
            }
            member.clear();
            visited[start as usize] = true;
            queue.clear();
            queue.push_back(start);
            while let Some(current) = queue.pop_front() {
                member.push(current);
                for offset in NEIGHBOURS {
                    let position = samples[current as usize].position + offset;
                    let Some(&next) = index.get(&position) else {
                        continue;
                    };
                    if visited[next as usize] {
                        continue;
                    }
                    visited[next as usize] = true;
                    queue.push_back(next);
                }
            }
            if member.len() > best.len() {
                best.clear();
                best.extend_from_slice(&member);
            }
        }

        if best.len() < minimum {
            dust.extend(samples);
            continue;
        }
        if best.len() == samples.len() {
            kept.push((samples, share));
            continue;
        }

        let mut keep = vec![false; samples.len()];
        for &slot in &best {
            keep[slot as usize] = true;
        }
        let mut body = Vec::with_capacity(best.len());
        for (slot, sample) in samples.into_iter().enumerate() {
            if keep[slot] {
                body.push(sample);
            } else {
                dust.push(sample);
            }
        }
        kept.push((body, share));
    }

    (kept, dust)
}

/// Evenly spaced world positions for the dust billboards, capped so one blast
/// cannot flood the particle pass.
fn dust_grains(dust: &[VoxelSample]) -> Vec<Vec3> {
    if dust.is_empty() {
        return Vec::new();
    }
    let count = dust.len().min(MAX_DUST_GRAINS);
    (0..count)
        .map(|index| {
            let sample = &dust[index * dust.len() / count];
            (sample.position.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE_M
        })
        .collect()
}

fn compound_boxes(dims: IVec3, voxels: &[u16], min: IVec3, centroid: Vec3) -> Vec<CompoundBox> {
    // Occupancy-aware greedy merge. The old fixed 4-voxel chunks emitted a box
    // for every chunk in the AABB, so a sparse shard carried invisible colliders
    // across its holes and floated on corners that were never solid. This walks
    // the actual local grid, grows each box along X, then Z, then Y, and marks
    // the covered cells. A solid slab becomes one box instead of hundreds.
    let (dx, dy, dz) = (dims.x as usize, dims.y as usize, dims.z as usize);
    if dx == 0 || dy == 0 || dz == 0 || voxels.len() < dx * dy * dz {
        return Vec::new();
    }
    let solid = |x: usize, y: usize, z: usize| voxels[x + dx * (y + dy * z)] != 0;
    let cell = |x: usize, y: usize, z: usize| x + dx * (y + dy * z);
    let mut used = vec![false; dx * dy * dz];
    let mut boxes = Vec::new();

    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                if !solid(x, y, z) || used[cell(x, y, z)] {
                    continue;
                }
                let mut w = 1;
                while x + w < dx && solid(x + w, y, z) && !used[cell(x + w, y, z)] {
                    w += 1;
                }
                let mut d = 1;
                'depth: while z + d < dz {
                    for xx in x..x + w {
                        if !solid(xx, y, z + d) || used[cell(xx, y, z + d)] {
                            break 'depth;
                        }
                    }
                    d += 1;
                }
                let mut h = 1;
                'height: while y + h < dy {
                    for zz in z..z + d {
                        for xx in x..x + w {
                            if !solid(xx, y + h, zz) || used[cell(xx, y + h, zz)] {
                                break 'height;
                            }
                        }
                    }
                    h += 1;
                }
                for yy in y..y + h {
                    for zz in z..z + d {
                        for xx in x..x + w {
                            used[cell(xx, yy, zz)] = true;
                        }
                    }
                }
                let size = IVec3::new(w as i32, h as i32, d as i32);
                let local_center = Vec3::new(
                    x as f32 + w as f32 * 0.5,
                    y as f32 + h as f32 * 0.5,
                    z as f32 + d as f32 * 0.5,
                );
                let center = (min.as_vec3() + local_center - centroid) * VOXEL_SIZE_M;
                let half = size.as_vec3() * (VOXEL_SIZE_M * 0.5);
                boxes.push(CompoundBox::new(center, half));
            }
        }
    }
    boxes
}

/// Conservative sanity: no voxel is lost without a piece. Kept as a public
/// helper for tests and diagnostics.
pub fn brick_of_voxel(position: IVec3) -> IVec3 {
    IVec3::new(
        position.x.div_euclid(BRICK_DIM as i32),
        position.y.div_euclid(BRICK_DIM as i32),
        position.z.div_euclid(BRICK_DIM as i32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use shatterbox_voxel::Voxel;
    use shatterbox_voxel::palette::materials;
    use shatterbox_voxel::scene::build_test_arena;

    /// Records bodies without running a solver so the destruction bookkeeping
    /// can be checked independently of Jolt.
    #[derive(Default)]
    struct FakePhysics {
        bodies: Vec<BodyHandle>,
        descs: Vec<BodyDesc>,
        next: u64,
    }

    impl PhysicsBackend for FakePhysics {
        fn add_static_box(&mut self, _center: Vec3, _half_extents: Vec3) {}

        fn replace_static_sector(&mut self, _key: IVec3, _boxes: &[CompoundBox]) {}

        fn remove_static_sector(&mut self, _key: IVec3) {}

        fn spawn(&mut self, desc: &BodyDesc) -> Option<BodyHandle> {
            self.next += 1;
            self.bodies.push(self.next);
            self.descs.push(desc.clone());
            Some(self.next)
        }

        fn remove(&mut self, handle: BodyHandle) {
            self.bodies.retain(|body| *body != handle);
        }

        fn step(&mut self, _dt: f32) {}

        fn state(&self, handle: BodyHandle) -> Option<BodyState> {
            let desc = self.descs.get((handle - 1) as usize)?;
            Some(BodyState {
                position: desc.position,
                rotation: desc.rotation,
                linear_velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
                awake: false,
                sleeping: true,
            })
        }

        fn body_count(&self) -> usize {
            self.bodies.len()
        }

        fn awake_count(&self) -> usize {
            self.bodies.len()
        }

        fn set_linear_velocity(&mut self, _handle: BodyHandle, _velocity: Vec3) {}
    }

    const ORIGIN: Vec3 = Vec3::new(128.0, 3.0, 120.0);
    const DOWN: Vec3 = Vec3::new(0.0, -1.0, 0.0);

    #[test]
    fn fire_turns_every_removed_voxel_into_debris() {
        let mut world = build_test_arena();
        let mut destruction = DestructionSystem::new(FakePhysics::default());
        let before = world.solid_voxels();
        let result = destruction.fire(
            &mut world,
            ORIGIN,
            DOWN,
            6,
            7,
            ImpactProfile::explosive(12_000.0, 1.0),
        );

        assert!(result.hit);
        assert!(result.removed > 0);
        assert!(result.pieces >= 2);
        assert_eq!(result.pieces, destruction.pieces().len());
        assert_eq!(destruction.total_bodies(), result.pieces);
        assert!(!result.dirty_bricks.is_empty());

        let mut in_pieces = 0u32;
        for piece in destruction.pieces() {
            assert_eq!(
                piece.grid_len(),
                (piece.dims.x * piece.dims.y * piece.dims.z) as usize
            );
            assert!(piece.local_min.x < piece.local_max.x);
            assert!(piece.local_min.y < piece.local_max.y);
            assert!(piece.local_min.z < piece.local_max.z);
            let solid = piece
                .voxels
                .iter()
                .filter(|raw| shatterbox_voxel::Voxel::from_raw(**raw).is_solid())
                .count();
            assert_eq!(solid as u32, piece.voxel_count);
            in_pieces += piece.voxel_count;
        }
        // Conservation: every voxel that left the world is inside a body grid.
        // This now also covers flood-filled detached components, not just the
        // crater brush.
        assert_eq!(
            in_pieces as u64 + result.dust_voxels as u64,
            before - world.solid_voxels()
        );
    }

    #[test]
    fn fill_restores_the_crater_surface() {
        let mut world = build_test_arena();
        // A 1 m slab cannot stop a 0.15 kg charge, so the default floor is
        // punched straight through and there is no surface left to fill. Build
        // a thick block under the test ray first; the fill logic is what is
        // under test here, not how deep the slab happens to be.
        for x in 1016..1032 {
            for z in 952..968 {
                for y in 0..16 {
                    world.set_voxel(x, y, z, Voxel::new(materials::CONCRETE, 0));
                }
            }
        }
        let mut destruction = DestructionSystem::new(FakePhysics::default());
        let fired = destruction.fire(
            &mut world,
            ORIGIN,
            DOWN,
            6,
            11,
            ImpactProfile::explosive(12_000.0, 1.0),
        );
        assert!(fired.removed > 0);
        assert!(world.raycast(ORIGIN, DOWN, 256.0).is_some());

        let edits = destruction.fill(&mut world, ORIGIN, DOWN, 6, materials::CONCRETE, 11);
        assert!(edits.changed > 0);
        assert!(!edits.dirty_bricks.is_empty());
        assert!(world.raycast(ORIGIN, DOWN, 256.0).is_some());
    }

    #[test]
    fn crater_centre_follows_the_hit_voxel_on_every_face() {
        // Regression: the old code used `position + max(normal, 0) - min(normal, 0)`,
        // i.e. `abs(normal)`, which shifted the crater one voxel towards +X/+Y/+Z
        // no matter which face was hit. The hole drifted off the reticle.
        let position = IVec3::new(40, 12, 40);
        for normal in [
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 0, 1),
            IVec3::new(0, 0, -1),
        ] {
            let hit = RayHit {
                voxel: Voxel::new(materials::CONCRETE, 0),
                position,
                normal,
                distance: 4.0,
            };
            assert_eq!(crater_center(&hit), position);
        }
    }

    #[test]
    fn fire_is_deterministic_for_the_same_seed() {
        let mut first = build_test_arena();
        let mut second = build_test_arena();
        let mut a = DestructionSystem::new(FakePhysics::default());
        let mut b = DestructionSystem::new(FakePhysics::default());
        let ra = a.fire(
            &mut first,
            ORIGIN,
            DOWN,
            8,
            42,
            ImpactProfile::explosive(12_000.0, 1.0),
        );
        let rb = b.fire(
            &mut second,
            ORIGIN,
            DOWN,
            8,
            42,
            ImpactProfile::explosive(12_000.0, 1.0),
        );

        assert_eq!(ra.removed, rb.removed);
        assert_eq!(ra.pieces, rb.pieces);
        assert_eq!(ra.dirty_bricks, rb.dirty_bricks);
        let pa = a.pieces();
        let pb = b.pieces();
        assert_eq!(pa.len(), pb.len());
        for (left, right) in pa.iter().zip(pb) {
            assert_eq!(left.dims, right.dims);
            assert_eq!(left.voxels, right.voxels);
            assert_eq!(left.local_min, right.local_min);
            assert_eq!(left.local_max, right.local_max);
        }
    }

    /// Fragment mass must be real kilograms: voxel volume x palette density.
    #[test]
    fn fragment_mass_comes_from_voxel_volume_and_density() {
        let mut world = build_test_arena();
        let mut destruction = DestructionSystem::new(FakePhysics::default());
        destruction.fire(
            &mut world,
            ORIGIN,
            DOWN,
            8,
            42,
            ImpactProfile::explosive(12_000.0, 1.0),
        );

        assert!(!destruction.pieces().is_empty());
        for piece in destruction.pieces() {
            let per_voxel = piece.mass_kg / piece.voxel_count as f32;
            // Palette densities span 0.7..7.8 t/m^3, i.e. 1.4..15.2 kg per
            // 12.5 cm voxel. Anything near 0.002 kg means the unit scaling is
            // broken again.
            assert!(
                (1.0..16.5).contains(&per_voxel),
                "implausible voxel mass: {per_voxel} kg"
            );
        }
    }

    /// Velocity and spin are solved from the shot energy, so quadrupling the
    /// energy must double both for the same fragment. Fragmentation itself now
    /// depends on energy (stronger blasts make smaller chips), so the clusters
    /// are built once and handed to both systems.
    #[test]
    fn fragment_motion_scales_with_shot_energy() {
        let mut world = VoxelWorld::default();
        // Asymmetric L so the centre of pressure is off the radial line and the
        // solve has a real torque arm.
        let mut samples = Vec::new();
        for x in 0..4 {
            for z in 0..4 {
                samples.push(VoxelSample {
                    position: IVec3::new(x, 0, z),
                    voxel: Voxel::new(materials::CONCRETE, 0),
                });
            }
        }
        for y in 1..3 {
            samples.push(VoxelSample {
                position: IVec3::new(0, y, 0),
                voxel: Voxel::new(materials::CONCRETE, 0),
            });
        }
        let clusters = vec![(samples, 1.0)];
        let impact = Vec3::new(0.1, 0.0, 0.1);
        let mut low = DestructionSystem::new(FakePhysics::default());
        let mut high = DestructionSystem::new(FakePhysics::default());
        low.spawn_clusters(
            &mut world,
            &clusters,
            impact,
            Vec3::Y,
            Vec3::Y,
            0.5,
            ImpactProfile::explosive(8_000.0, 1.0),
            7,
        );
        high.spawn_clusters(
            &mut world,
            &clusters,
            impact,
            Vec3::Y,
            Vec3::Y,
            0.5,
            ImpactProfile::explosive(32_000.0, 1.0),
            7,
        );
        assert_eq!(low.pieces().len(), 1);
        assert_eq!(high.pieces().len(), 1);

        let soft = &low.physics().descs[0];
        let hard = &high.physics().descs[0];
        let speed = soft.linear_velocity.length();
        assert!(speed > 0.0, "fragments must not stay frozen");
        if speed < MAX_LINEAR_SPEED * 0.99 {
            let ratio = hard.linear_velocity.length() / speed;
            assert!(
                (ratio - 2.0).abs() < 0.02,
                "linear speed ratio {ratio} is not sqrt(4)"
            );
        }
        let spin = soft.angular_velocity.length();
        assert!(spin > 1e-6, "the asymmetric fragment must spin");
        if spin < MAX_ANGULAR_SPEED * 0.99 {
            let ratio = hard.angular_velocity.length() / spin;
            assert!(
                (ratio - 2.0).abs() < 0.02,
                "angular speed ratio {ratio} is not sqrt(4)"
            );
        }
    }

    /// The solved velocities must survive the real Jolt body: fragments leave
    /// the crater and settle on the floor instead of falling through or
    /// exploding numerically.
    #[test]
    fn jolt_accepts_physics_driven_debris() {
        let mut world = build_test_arena();
        let mut destruction =
            DestructionSystem::new(shatterbox_physics::JoltPhysics::new().expect("jolt world"));
        // Deep slab: its top is the visual floor (y = 1.0), but it reaches far
        // down so crater debris spawned inside it is always pushed up and out,
        // never through the nearest (bottom) face.
        destruction
            .physics_mut()
            .add_static_box(Vec3::new(128.0, -1.0, 120.0), Vec3::new(32.0, 2.0, 32.0));
        let result = destruction.fire(
            &mut world,
            ORIGIN,
            DOWN,
            8,
            42,
            ImpactProfile::explosive(24_000.0, 1.0),
        );
        assert!(result.pieces >= 2);

        let start: Vec<Vec3> = destruction
            .pieces()
            .iter()
            .map(|piece| destruction.piece_state(piece).expect("state").position)
            .collect();
        for _ in 0..240 {
            destruction.step(1.0 / 60.0);
        }

        let mut moved = 0;
        for (piece, origin) in destruction.pieces().iter().zip(&start) {
            let state = destruction.piece_state(piece).expect("state");
            assert!(state.position.is_finite(), "non-finite body position");
            assert!(state.rotation.is_finite(), "non-finite body rotation");
            // Chips carved out of the very bottom of the crater spawn inside the
            // test's proxy slab (the real game rebuilds the sector from the
            // world, so that cell is air). Only bodies that started above the
            // slab top can be checked for tunnelling.
            if origin.y > 1.0 {
                assert!(
                    state.position.y > -1.0,
                    "fragment fell through the floor: {:?} (voxels {}, dims {:?}, mass {})",
                    state.position,
                    piece.voxel_count,
                    piece.dims,
                    piece.mass_kg
                );
            }
            if state.position.distance(*origin) > 0.02 {
                moved += 1;
            }
        }
        assert!(moved > 0, "physics never applied the solved velocity");
    }

    #[test]
    fn compound_boxes_merge_solid_cells_and_skip_air() {
        let dims = IVec3::splat(4);
        let mut voxels = vec![1u16; (4 * 4 * 4) as usize];
        let boxes = compound_boxes(dims, &voxels, IVec3::ZERO, Vec3::splat(2.0));
        assert_eq!(boxes.len(), 1, "a solid 4^3 block must be one box");
        assert!((boxes[0].half_extents - Vec3::splat(0.25)).length() < 1e-5);

        voxels.fill(0);
        voxels[0] = 1;
        let boxes = compound_boxes(dims, &voxels, IVec3::ZERO, Vec3::ZERO);
        assert_eq!(boxes.len(), 1, "one voxel is one box");
        assert!((boxes[0].half_extents - Vec3::splat(VOXEL_SIZE_M * 0.5)).length() < 1e-5);
        assert!(
            (boxes[0].center - Vec3::splat(VOXEL_SIZE_M * 0.5)).length() < 1e-5,
            "air cells must not move the collider: {:?}",
            boxes[0].center
        );
    }

    #[test]
    fn spawned_debris_carries_the_voxel_inertia_tensor() {
        let mut world = build_test_arena();
        let mut destruction = DestructionSystem::new(FakePhysics::default());
        let result = destruction.fire(
            &mut world,
            ORIGIN,
            DOWN,
            6,
            3,
            ImpactProfile::explosive(12_000.0, 1.0),
        );
        assert!(result.pieces > 0);
        let physics = destruction.physics();
        for desc in &physics.descs {
            let tensor = desc.inertia.expect("voxel inertia must reach the backend");
            assert_eq!(tensor[0][1], tensor[1][0], "tensor must be symmetric");
            assert_eq!(tensor[0][2], tensor[2][0]);
            assert_eq!(tensor[1][2], tensor[2][1]);
            assert!(tensor[0][0] > 0.0 && tensor[1][1] > 0.0 && tensor[2][2] > 0.0);
            assert!(tensor.iter().flatten().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn step_records_previous_and_current_transforms() {
        let mut world = build_test_arena();
        let mut destruction =
            DestructionSystem::new(shatterbox_physics::JoltPhysics::new().expect("jolt world"));
        destruction
            .physics_mut()
            .add_static_box(Vec3::new(128.0, -1.0, 120.0), Vec3::new(32.0, 2.0, 32.0));
        destruction.fire(
            &mut world,
            ORIGIN,
            DOWN,
            8,
            11,
            ImpactProfile::explosive(24_000.0, 1.0),
        );
        for _ in 0..5 {
            let before: Vec<(Vec3, Quat)> = destruction
                .pieces()
                .iter()
                .map(|piece| (piece.position, piece.rotation))
                .collect();
            destruction.step(1.0 / 60.0);
            for (piece, (position, rotation)) in destruction.pieces().iter().zip(before) {
                assert_eq!(piece.prev_position, position);
                assert_eq!(piece.prev_rotation, rotation);
            }
        }
        for piece in destruction.pieces() {
            let (mid, _) = destruction.interpolated_transform(piece, 0.5);
            assert!(
                mid.distance(piece.prev_position)
                    <= piece.position.distance(piece.prev_position) + 1e-5,
                "interpolated point must lie between the two steps"
            );
        }
    }

    /// A room whose only support is blown out must drop the slab above it. The
    /// flood fill has to find the now-free component, take it out of the static
    /// world and hand it to the solver as one body.
    #[test]
    fn undermining_the_support_drops_the_slab_above_it() {
        let mut world = VoxelWorld::default();
        let concrete = Voxel::new(materials::CONCRETE, 0);
        // Floor, one pillar, one roof slab. The pillar is the only path from
        // the slab to the ground.
        for x in 0..24 {
            for z in 0..24 {
                world.set_voxel(x, 0, z, concrete);
            }
        }
        for y in 1..=8 {
            world.set_voxel(11, y, 11, concrete);
        }
        for x in 8..=14 {
            for z in 8..=14 {
                world.set_voxel(x, 9, z, concrete);
            }
        }
        let before = world.solid_voxels();
        assert_eq!(before, 24 * 24 + 8 + 49);

        let mut destruction = DestructionSystem::new(FakePhysics::default());
        let point = Vec3::new(11.5 * VOXEL_SIZE_M, 2.0 * VOXEL_SIZE_M, 11.5 * VOXEL_SIZE_M);
        let result = destruction.explode(
            &mut world,
            point,
            Vec3::Y,
            4,
            5,
            ImpactProfile::explosive(4_000.0, 1.0),
        );
        assert!(result.removed > 0);

        // The slab lost its only support, so it must no longer be static.
        assert_eq!(world.voxel(11, 9, 11), Voxel::AIR, "roof must detach");
        let slab = destruction
            .pieces()
            .iter()
            .find(|piece| piece.voxel_count >= 49)
            .expect("the roof must be one large body");
        assert!(
            slab.voxel_count >= 49,
            "the slab keeps its own geometry, got {} voxels",
            slab.voxel_count
        );

        let in_pieces: u32 = destruction
            .pieces()
            .iter()
            .map(|piece| piece.voxel_count)
            .sum();
        assert_eq!(
            in_pieces as u64 + result.dust_voxels as u64,
            before - world.solid_voxels()
        );
    }

    #[test]
    fn three_shots_at_the_same_spot_make_different_rubble() {
        let mut shapes = Vec::new();
        for seed in [11u32, 12, 13] {
            let mut world = build_test_arena();
            let mut destruction = DestructionSystem::new(FakePhysics::default());
            let result = destruction.fire(
                &mut world,
                ORIGIN,
                DOWN,
                6,
                seed,
                ImpactProfile::explosive(12_000.0, 1.0),
            );
            assert!(result.removed > 0);
            let mut sizes: Vec<u32> = destruction
                .pieces()
                .iter()
                .map(|piece| piece.voxel_count)
                .collect();
            sizes.sort_unstable();
            shapes.push(sizes);
        }
        assert_ne!(
            shapes[0], shapes[1],
            "the seed must change the fragment sizes"
        );
        assert_ne!(shapes[1], shapes[2]);
    }

    #[test]
    fn blast_energy_controls_chip_size() {
        let mut removed = Vec::new();
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    removed.push(RemovedVoxel {
                        position: IVec3::new(x, y, z),
                        voxel: Voxel::new(materials::CONCRETE, 0),
                    });
                }
            }
        }
        let impact = Vec3::splat(5.0) * VOXEL_SIZE_M;
        let radius = 14.0 * VOXEL_SIZE_M;
        let soft = split_core(&removed, impact, radius, 4_000.0, 3);
        let hard = split_core(&removed, impact, radius, 40_000.0, 3);
        assert!(
            hard.len() > soft.len(),
            "more energy must shatter into more chips"
        );
        let average = |pieces: &[Vec<VoxelSample>]| removed.len() as f32 / pieces.len() as f32;
        assert!(average(&hard) < average(&soft));

        // Conservation: every voxel lands in exactly one chip.
        let total: usize = hard.iter().map(Vec::len).sum();
        assert_eq!(total, removed.len());
    }

    #[test]
    fn sleeping_pieces_bake_back_into_the_world() {
        let mut world = build_test_arena();
        let mut destruction = DestructionSystem::new(FakePhysics::default());
        let before = world.solid_voxels();
        let result = destruction.fire(
            &mut world,
            ORIGIN,
            DOWN,
            6,
            21,
            ImpactProfile::explosive(12_000.0, 1.0),
        );
        assert!(result.pieces > 0);
        let in_pieces: u32 = destruction
            .pieces()
            .iter()
            .map(|piece| piece.voxel_count)
            .sum();
        assert_eq!(
            in_pieces as u64 + result.dust_voxels as u64,
            before - world.solid_voxels()
        );

        // FakePhysics reports every body as asleep, so this call must bake all
        // of them. The player stands far away, so nothing blocks the bake.
        let bake = destruction.bake_sleeping(&mut world, Vec3::splat(-50.0), 64);
        assert_eq!(bake.pieces, result.pieces);
        assert!(bake.voxels > 0);
        assert!(!bake.dirty_bricks.is_empty());
        assert!(destruction.pieces().is_empty());
        assert_eq!(
            world.solid_voxels() + destruction.dust_voxels(),
            before,
            "baking must not lose voxels (pulverised dust stays dust)"
        );
    }

    /// The render/bodies budget must retire pieces by baking, never by deleting
    /// them, and moving the camera far away must not change the body count.
    #[test]
    fn retiring_pieces_conserves_voxels_and_ignores_camera_distance() {
        let mut world = build_test_arena();
        let mut destruction = DestructionSystem::new(FakePhysics::default());
        let before = world.solid_voxels();
        let result = destruction.explode(
            &mut world,
            Vec3::new(128.0, 0.5, 168.0),
            Vec3::Y,
            14,
            9,
            ImpactProfile::explosive(12_000.0, 1.0),
        );
        assert!(result.pieces > 4);
        let in_pieces: u64 = destruction
            .pieces()
            .iter()
            .map(|piece| u64::from(piece.voxel_count))
            .sum();
        assert_eq!(
            in_pieces + result.dust_voxels as u64,
            before - world.solid_voxels()
        );
        let spawned = result.pieces;

        // Camera leaves to 60 m: with a budget above the piece count nothing
        // may be retired or hidden.
        let far = Vec3::new(128.0, 2.0, 228.0);
        let bake = destruction.retire_into_world(&mut world, far, spawned);
        assert_eq!(bake.pieces, 0);
        assert_eq!(destruction.pieces().len(), spawned);

        // Back at the wall with a smaller budget: the farthest pieces bake into
        // the world, and the world + bodies total is unchanged.
        let bake = destruction.retire_into_world(&mut world, Vec3::splat(0.0), 2);
        assert!(bake.pieces > 0, "the budget must retire something");
        assert_eq!(destruction.pieces().len(), 2);
        let remaining: u64 = destruction
            .pieces()
            .iter()
            .map(|piece| u64::from(piece.voxel_count))
            .sum();
        assert_eq!(
            world.solid_voxels() + remaining + destruction.dust_voxels(),
            before
        );
    }

    /// Baking the same destruction twice must produce the same world hash.
    #[test]
    fn baking_is_deterministic() {
        let mut hashes = Vec::new();
        for _ in 0..2 {
            let mut world = build_test_arena();
            let mut destruction = DestructionSystem::new(FakePhysics::default());
            destruction.explode(
                &mut world,
                Vec3::new(128.0, 0.5, 168.0),
                Vec3::Y,
                14,
                9,
                ImpactProfile::explosive(12_000.0, 1.0),
            );
            destruction.bake_sleeping(&mut world, Vec3::splat(-50.0), 1024);
            hashes.push(world.deterministic_hash());
        }
        assert_eq!(hashes[0], hashes[1]);
    }

    /// Part 2: no rigid body may be a speck. Every piece that reaches Jolt has
    /// to be big enough to collide, render and bake like a real chunk.
    #[test]
    fn no_rigid_body_is_smaller_than_the_minimum() {
        let mut world = build_test_arena();
        let mut destruction = DestructionSystem::new(FakePhysics::default());
        let result = destruction.fire(
            &mut world,
            ORIGIN,
            DOWN,
            8,
            42,
            ImpactProfile::explosive(12_000.0, 1.0),
        );
        assert!(result.pieces > 0);
        assert_eq!(destruction.min_piece(), MIN_PIECE_VOXELS);
        for piece in destruction.pieces() {
            assert!(
                piece.voxel_count as usize >= MIN_PIECE_VOXELS,
                "body of {} voxels survived the merge",
                piece.voxel_count
            );
        }
    }

    /// The minimum size has to halve the body count without deleting matter:
    /// crumbs merge into a neighbour, the rest becomes dust, and the total is
    /// still `world + bodies + dust`.
    #[test]
    fn min_piece_halves_the_body_count_and_keeps_the_matter() {
        let mut old_world = build_test_arena();
        let before = old_world.solid_voxels();
        let mut old = DestructionSystem::new(FakePhysics::default());
        old.set_min_piece(1);
        let old_result = old.fire(
            &mut old_world,
            ORIGIN,
            DOWN,
            8,
            42,
            ImpactProfile::explosive(12_000.0, 1.0),
        );

        let mut new_world = build_test_arena();
        let mut new = DestructionSystem::new(FakePhysics::default());
        let new_result = new.fire(
            &mut new_world,
            ORIGIN,
            DOWN,
            8,
            42,
            ImpactProfile::explosive(12_000.0, 1.0),
        );

        assert_eq!(old_result.removed, new_result.removed);
        assert!(
            new_result.pieces * 2 <= old_result.pieces,
            "{} bodies must collapse to at most {}",
            old_result.pieces,
            old_result.pieces / 2
        );
        let in_bodies: u64 = new
            .pieces()
            .iter()
            .map(|piece| u64::from(piece.voxel_count))
            .sum();
        assert_eq!(
            in_bodies + new.dust_voxels(),
            before - new_world.solid_voxels()
        );
    }

    /// The same wall, the same seed: a rocket must throw the chips much faster
    /// than a carbine round, because only the rocket carries a shock wave.
    #[test]
    fn rocket_outthrows_a_carbine_round() {
        let mut rocket_world = build_test_arena();
        let mut rocket = DestructionSystem::new(FakePhysics::default());
        let rocket_hit = rocket.fire(
            &mut rocket_world,
            ORIGIN,
            DOWN,
            6,
            7,
            // The game's rocket: 0.15 kg of TNT-equivalent.
            ImpactProfile::explosive(628_000.0, 1.0),
        );

        let mut carbine_world = build_test_arena();
        let mut carbine = DestructionSystem::new(FakePhysics::default());
        let carbine_hit = carbine.fire(
            &mut carbine_world,
            ORIGIN,
            DOWN,
            2,
            7,
            ImpactProfile::kinetic(12_552.0),
        );

        assert!(rocket_hit.pieces > 0 && carbine_hit.pieces > 0);
        assert!(
            rocket_hit.speed_max > carbine_hit.speed_max * 2.0,
            "rocket {:.2} m/s vs carbine {:.2} m/s",
            rocket_hit.speed_max,
            carbine_hit.speed_max
        );
        assert!(
            carbine_hit.speed_max <= KINETIC_PUSH_SPEED * 1.3,
            "a bullet must not throw like a blast: {:.2} m/s",
            carbine_hit.speed_max
        );
    }

    /// Blast speed must fall with distance from the centre and be exactly zero
    /// outside the wave radius: those pieces just drop.
    #[test]
    fn blast_speed_falls_with_distance_and_stops_at_the_radius() {
        let mut world = VoxelWorld::default();
        let cube = |offset: IVec3| {
            let mut samples = Vec::new();
            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        samples.push(VoxelSample {
                            position: offset + IVec3::new(x, y, z),
                            voxel: Voxel::new(materials::CONCRETE, 0),
                        });
                    }
                }
            }
            samples
        };
        // Radius 2 m = 16 voxels: the far cube sits at 1.6 m, the far one at
        // 3 m, i.e. outside the wave.
        let near = cube(IVec3::ZERO);
        let mid = cube(IVec3::new(12, 0, 0));
        let far = cube(IVec3::new(24, 0, 0));
        let clusters = vec![(near, 1.0), (mid, 1.0), (far, 1.0)];
        let mut destruction = DestructionSystem::new(FakePhysics::default());
        destruction.spawn_clusters(
            &mut world,
            &clusters,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            16.0 * VOXEL_SIZE_M,
            ImpactProfile::explosive(20_000.0, 1.0),
            5,
        );
        let descs = &destruction.physics().descs;
        let near_speed = descs[0].linear_velocity.length();
        let mid_speed = descs[1].linear_velocity.length();
        let far_speed = descs[2].linear_velocity.length();
        assert!(near_speed > 0.0, "the centre chip must fly");
        assert!(
            near_speed > mid_speed * 1.5,
            "speed must fall with distance: {near_speed:.2} vs {mid_speed:.2}"
        );
        // The wave now has a short 1/r^2 tail instead of a hard edge: a detached
        // wall panel just outside the crater should be nudged, not frozen.
        assert!(far_speed > 0.0, "the soft tail must nudge the outer panel");
        assert!(
            far_speed < near_speed * 0.5,
            "the tail must stay a nudge: {far_speed:.2} vs {near_speed:.2}"
        );
    }

    /// A crumb with nothing to merge into must become dust, not a one-voxel
    /// body, and it must not vanish from the matter count either.
    #[test]
    fn isolated_crumbs_become_dust() {
        let cube = |offset: IVec3, size: i32| {
            let mut samples = Vec::new();
            for x in 0..size {
                for y in 0..size {
                    for z in 0..size {
                        samples.push(VoxelSample {
                            position: offset + IVec3::new(x, y, z),
                            voxel: Voxel::new(materials::CONCRETE, 0),
                        });
                    }
                }
            }
            samples
        };
        let clusters = vec![
            // 27 voxels: comfortably above `MIN_PIECE_VOXELS`, so it survives.
            (cube(IVec3::ZERO, 3), 1.0),
            (cube(IVec3::new(400, 0, 0), 1), 1.0),
        ];
        let (merged, dust) = consolidate_clusters(clusters, MIN_PIECE_VOXELS);
        assert_eq!(merged.len(), 1, "a body above the minimum must survive");
        assert_eq!(merged[0].0.len(), 27);
        assert_eq!(dust.len(), 1, "the lone voxel becomes dust");
        assert_eq!(dust_grains(&dust).len(), 1);
    }

    /// A body must be one lump: an island inside a cluster (the shape
    /// `split_core` leaves behind when several growth fronts meet) has to leave
    /// as dust instead of flying as a lonely cube next to a holey chunk.
    #[test]
    fn clusters_keep_only_their_largest_component() {
        let cube = |offset: IVec3, size: i32| {
            let mut samples = Vec::new();
            for x in 0..size {
                for y in 0..size {
                    for z in 0..size {
                        samples.push(VoxelSample {
                            position: offset + IVec3::new(x, y, z),
                            voxel: Voxel::new(materials::CONCRETE, 0),
                        });
                    }
                }
            }
            samples
        };
        // 27-voxel cube, a detached 2-voxel chip inside the merge radius and a
        // lone voxel; only the cube may stay a body.
        let mut samples = cube(IVec3::ZERO, 3);
        samples.extend(cube(IVec3::new(6, 0, 0), 1));
        samples.push(VoxelSample {
            position: IVec3::new(6, 4, 0),
            voxel: Voxel::new(materials::CONCRETE, 0),
        });
        let clusters = vec![(samples, 1.0), (cube(IVec3::new(400, 0, 0), 1), 1.0)];
        let before: usize = clusters.iter().map(|(s, _)| s.len()).sum();

        let (kept, dust) = keep_connected_clusters(clusters, MIN_PIECE_VOXELS);
        assert_eq!(kept.len(), 1, "only the connected cube is a body");
        assert_eq!(kept[0].0.len(), 27);
        assert_eq!(dust.len(), before - 27, "islands and singletons are dust");
        assert_eq!(
            kept.iter().map(|(s, _)| s.len()).sum::<usize>() + dust.len(),
            before,
            "connectivity must not lose matter"
        );
    }

    /// Kinetic spread is seed-driven: same wall, same weapon, different seeds
    /// must not produce identical chip speeds.
    #[test]
    fn kinetic_spread_differs_per_seed() {
        let mut speeds = Vec::new();
        for seed in [11u32, 12, 13] {
            let mut world = build_test_arena();
            let mut destruction = DestructionSystem::new(FakePhysics::default());
            let result = destruction.fire(
                &mut world,
                ORIGIN,
                DOWN,
                3,
                seed,
                ImpactProfile::kinetic(12_552.0),
            );
            speeds.push(result.speed_max);
        }
        assert_ne!(speeds[0], speeds[1]);
        assert_ne!(speeds[1], speeds[2]);
    }
}
