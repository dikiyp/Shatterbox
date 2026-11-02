//! Deterministic voxel edit commands shared by tools, destruction and tests.
//!
//! A command never contains floating point data: the brush centre, radius and
//! jitter are integers in voxel space. Applying the same command log to the
//! same world therefore produces the same voxels in the same order on every
//! machine.

use glam::{I64Vec3, IVec3};

use crate::brick::BRICK_DIM;
use crate::voxel::Voxel;
use crate::world::VoxelWorld;

/// Operation carried by a [`VoxelCommand`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditOp {
    Remove,
    Add,
}

/// Integer sphere brush. `jitter` expands or shrinks the radius per voxel by
/// up to this many voxels, which gives the crater a chipped, irregular rim
/// instead of a perfect ball.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SphereBrush {
    pub center: IVec3,
    pub radius: i32,
    pub jitter: u8,
}

impl SphereBrush {
    pub const fn new(center: IVec3, radius: i32, jitter: u8) -> Self {
        Self {
            center,
            radius,
            jitter,
        }
    }
}

/// The canonical edit record. `tick`/`seq` order the log, `region` is the
/// 32 m region the edit belongs to (for interest management and save deltas).
///
/// `extent` sweeps the sphere along a segment: the voxel is affected when its
/// centre is within `brush.radius` of the segment `center .. center + extent`.
/// A zero extent keeps the classic spherical brush. Sweeping matters for
/// weapon hits: a sphere centred on the first solid voxel cuts a hole shaped
/// like a sphere *surface*, so a glancing shot spills sideways and the visible
/// opening no longer sits under the crosshair. A capsule always cuts a circle
/// of exactly `radius` around the entry point, whatever the impact angle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelCommand {
    pub tick: u64,
    pub seq: u32,
    pub region: IVec3,
    pub op: EditOp,
    pub brush: SphereBrush,
    pub extent: IVec3,
    pub material: u16,
    pub seed: u32,
}

impl VoxelCommand {
    pub fn remove(tick: u64, seq: u32, center: IVec3, radius: i32, seed: u32) -> Self {
        Self::build(
            tick,
            seq,
            center,
            IVec3::ZERO,
            EditOp::Remove,
            radius,
            2,
            0,
            seed,
        )
    }

    /// Capsule-shaped removal: `extent` is the in-world sweep of the brush in
    /// voxels, `jitter` the rim irregularity (0 gives a perfectly round tunnel).
    pub fn remove_capsule(
        tick: u64,
        seq: u32,
        center: IVec3,
        extent: IVec3,
        radius: i32,
        jitter: u8,
        seed: u32,
    ) -> Self {
        Self::build(
            tick,
            seq,
            center,
            extent,
            EditOp::Remove,
            radius,
            jitter,
            0,
            seed,
        )
    }

    pub fn add(tick: u64, seq: u32, center: IVec3, radius: i32, material: u16, seed: u32) -> Self {
        Self::build(
            tick,
            seq,
            center,
            IVec3::ZERO,
            EditOp::Add,
            radius,
            1,
            material,
            seed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        tick: u64,
        seq: u32,
        center: IVec3,
        extent: IVec3,
        op: EditOp,
        radius: i32,
        jitter: u8,
        material: u16,
        seed: u32,
    ) -> Self {
        Self {
            tick,
            seq,
            region: IVec3::new(
                center.x.div_euclid(256),
                center.y.div_euclid(256),
                center.z.div_euclid(256),
            ),
            op,
            brush: SphereBrush::new(center, radius, jitter),
            extent,
            material,
            seed,
        }
    }
}

/// A voxel that left the static world. The destruction system turns these into
/// dynamic debris, so nothing is ever deleted without a physical record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemovedVoxel {
    pub position: IVec3,
    pub voxel: Voxel,
}

/// Result of one command. `dirty_bricks` is sorted, which lets the renderer
/// upload in a stable order and makes the hash independent of hash-map state.
#[derive(Clone, Debug, Default)]
pub struct EditReport {
    pub removed: Vec<RemovedVoxel>,
    pub added: u32,
    pub dirty_bricks: Vec<IVec3>,
}

impl EditReport {
    pub fn is_empty(&self) -> bool {
        self.removed.is_empty() && self.added == 0
    }
}

/// Applies one command. Traversal is fixed (`x`, then `y`, then `z`) and the
/// rim jitter is a pure hash of `(seed, axial slice)`.
///
/// The jitter is keyed by the axial slice (the closest point on the brush
/// segment) instead of by voxel, so the entry opening stays a perfect circle
/// and only the tunnel walls get chipped. Keying it per voxel made the hole rim
/// ragged towards one side, which read as "the shot landed left of the
/// crosshair".
pub fn apply_command(world: &mut VoxelWorld, command: &VoxelCommand) -> EditReport {
    let center = command.brush.center;
    let end = center + command.extent;
    let radius = command.brush.radius.max(1);
    let jitter = command.brush.jitter as i32;
    let mut report = EditReport::default();
    let mut dirty = Vec::new();

    let reach = radius + jitter;
    let min = center.min(end) - IVec3::splat(reach);
    let max = center.max(end) + IVec3::splat(reach);

    for x in min.x..=max.x {
        for y in min.y..=max.y {
            for z in min.z..=max.z {
                let point = IVec3::new(x, y, z);
                let axis = closest_on_segment(center, command.extent, point);
                // A swept capsule keeps a clean circular entry (jitter keyed by
                // the axial slice); a plain sphere blast gets a per-voxel rim so
                // the crater is a ragged splat, not a scaled ball.
                let limit = if command.extent == IVec3::ZERO {
                    sphere_radius(radius, jitter, command.seed, point)
                } else {
                    axial_radius(radius, jitter, command.seed, axis)
                };
                let delta = point - axis;
                if i64::from(delta.length_squared()) > i64::from(limit) * i64::from(limit) {
                    continue;
                }

                match command.op {
                    EditOp::Remove => {
                        let voxel = world.voxel(x, y, z);
                        if voxel.is_air() {
                            continue;
                        }
                        world.set_voxel(x, y, z, Voxel::AIR);
                        report.removed.push(RemovedVoxel {
                            position: IVec3::new(x, y, z),
                            voxel,
                        });
                        dirty.push(brick_of(x, y, z));
                    }
                    EditOp::Add => {
                        if !world.voxel(x, y, z).is_air() {
                            continue;
                        }
                        let tint = (hash_voxel(command.seed, x, y, z) & 0x3f) as u8;
                        world.set_voxel(x, y, z, Voxel::new(command.material, tint));
                        report.added += 1;
                        dirty.push(brick_of(x, y, z));
                    }
                }
            }
        }
    }

    dirty.sort_unstable_by_key(|brick| (brick.x, brick.y, brick.z));
    dirty.dedup();
    report.dirty_bricks = dirty;
    report
}

/// Applies an explicit list of voxel writes. Used by the destruction system for
/// edits that are not a sphere: detaching a flood-filled component and baking a
/// settled piece back into the world.
///
/// The caller keeps `edits` sorted by position, which makes the result
/// independent of any hash-map iteration order. Existing solid voxels are never
/// overwritten, so baking can only fill air and detaching can only clear the
/// voxels it actually captured.
pub fn apply_voxel_edits(world: &mut VoxelWorld, edits: &[(IVec3, Voxel)]) -> EditReport {
    let mut report = EditReport::default();
    let mut dirty = Vec::new();
    for &(position, voxel) in edits {
        let current = world.voxel(position.x, position.y, position.z);
        if voxel.is_air() {
            if current.is_air() {
                continue;
            }
            world.set_voxel(position.x, position.y, position.z, Voxel::AIR);
            report.removed.push(RemovedVoxel {
                position,
                voxel: current,
            });
        } else {
            if !current.is_air() {
                continue;
            }
            world.set_voxel(position.x, position.y, position.z, voxel);
            report.added += 1;
        }
        dirty.push(brick_of(position.x, position.y, position.z));
    }
    dirty.sort_unstable_by_key(|brick| (brick.x, brick.y, brick.z));
    dirty.dedup();
    report.dirty_bricks = dirty;
    report
}

/// The point of the brush segment closest to `point`, in integer voxel space.
/// Fixed-point (1/1024 of the segment) keeps the result identical on every
/// machine and avoids any float rounding in the deterministic edit path.
fn closest_on_segment(start: IVec3, extent: IVec3, point: IVec3) -> IVec3 {
    let segment = extent.as_i64vec3();
    let length_sq = segment.length_squared();
    if length_sq == 0 {
        return start;
    }
    let relative = (point - start).as_i64vec3();
    let projection = relative.dot(segment).clamp(0, length_sq);
    let along = (segment * projection).div_euclid(I64Vec3::splat(length_sq));
    start + along.as_ivec3()
}

/// Radius of a plain sphere brush at `point`: the rim is chipped per voxel, so
/// a blast hole is irregular in every direction instead of one uniform ball.
fn sphere_radius(radius: i32, jitter: i32, seed: u32, point: IVec3) -> i32 {
    if jitter == 0 {
        return radius;
    }
    let span = (jitter * 2 + 1).max(1) as u32;
    let offset = (hash_voxel(seed, point.x, point.y, point.z) % span) as i32 - jitter;
    (radius + offset).max(1)
}

/// Radius of the slice at `axis`. Constant along the entry cap (so the opening
/// is a clean circle) and chipped along the tunnel.
fn axial_radius(radius: i32, jitter: i32, seed: u32, axis: IVec3) -> i32 {
    if jitter == 0 {
        return radius;
    }
    let span = (jitter * 2 + 1).max(1) as u32;
    let offset = (hash_voxel(seed, axis.x, axis.y, axis.z) % span) as i32 - jitter;
    (radius + offset).max(1)
}

fn hash_voxel(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    let mut hash = seed ^ 0x9e37_79b9;
    for value in [x as u32, y as u32, z as u32] {
        hash ^= value.wrapping_mul(0x85eb_ca6b);
        hash = hash.rotate_left(13).wrapping_mul(0xc2b2_ae35);
        hash ^= hash >> 16;
    }
    hash
}

fn brick_of(x: i32, y: i32, z: i32) -> IVec3 {
    IVec3::new(
        x.div_euclid(BRICK_DIM as i32),
        y.div_euclid(BRICK_DIM as i32),
        z.div_euclid(BRICK_DIM as i32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::{Palette, materials};
    use crate::world::VoxelWorld;

    #[test]
    fn remove_is_deterministic_and_reports_voxels() {
        let mut first = VoxelWorld::new(Palette::default());
        let mut second = VoxelWorld::new(Palette::default());
        for world in [&mut first, &mut second] {
            for x in 30..34 {
                for y in 10..14 {
                    for z in 30..34 {
                        world.set_voxel(x, y, z, Voxel::new(materials::BRICK, 7));
                    }
                }
            }
        }
        let command = VoxelCommand::remove(1, 0, IVec3::new(32, 12, 32), 3, 91);
        let a = apply_command(&mut first, &command);
        let b = apply_command(&mut second, &command);
        assert_eq!(a.removed, b.removed);
        assert_eq!(a.dirty_bricks, b.dirty_bricks);
        assert!(!a.removed.is_empty());
        assert_eq!(first.deterministic_hash(), second.deterministic_hash());
        assert!(a.removed.iter().all(|removed| {
            first
                .voxel(removed.position.x, removed.position.y, removed.position.z)
                .is_air()
        }));
    }

    #[test]
    fn add_restores_air_and_uses_material() {
        let mut world = VoxelWorld::new(Palette::default());
        let command = VoxelCommand::add(2, 1, IVec3::new(20, 20, 20), 2, materials::CONCRETE, 5);
        let report = apply_command(&mut world, &command);
        assert!(report.added > 0);
        assert_eq!(world.voxel(20, 20, 20).material(), materials::CONCRETE);
    }

    #[test]
    fn capsule_entry_hole_is_round_and_centred() {
        // A wall at z = 40, one voxel thick, shot at a shallow angle: the
        // capsule must still open a disc of `radius` centred on the entry
        // voxel. The old sphere brush spilled the opening sideways.
        let mut world = VoxelWorld::new(Palette::default());
        for x in 0..64 {
            for y in 0..64 {
                world.set_voxel(x, y, 40, Voxel::new(materials::CONCRETE, 0));
            }
        }
        let entry = IVec3::new(32, 32, 40);
        let command = VoxelCommand::remove_capsule(1, 0, entry, IVec3::new(6, 0, 6), 4, 1, 17);
        let report = apply_command(&mut world, &command);
        assert!(!report.removed.is_empty());

        let mut min = IVec3::splat(i32::MAX);
        let mut max = IVec3::splat(i32::MIN);
        let mut removed = 0;
        for x in 0..64 {
            for y in 0..64 {
                if world.voxel(x, y, 40).is_air() {
                    removed += 1;
                    min = min.min(IVec3::new(x, y, 40));
                    max = max.max(IVec3::new(x, y, 40));
                }
            }
        }
        assert!(removed > 0);
        // The hole is a disc of radius 4 (plus the 1 voxel rim jitter): across
        // the sweep it stays symmetric around the entry column, and along the
        // sweep it drills forward instead of spilling sideways.
        assert!(max.y - entry.y <= 5 && entry.y - min.y <= 5, "{min}..{max}");
        assert!(entry.x - min.x <= 5, "{min}..{max}");
        assert!(max.x >= entry.x + 6, "{min}..{max}");
    }
}
