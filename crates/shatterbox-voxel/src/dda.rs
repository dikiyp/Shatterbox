//! CPU reference DDA through regions, bricks and voxels.

use glam::{IVec3, Vec3};

use crate::voxel::Voxel;
use crate::world::{VOXEL_SIZE_M, VoxelWorld, WORLD_MAX, WORLD_MIN};

/// First solid voxel hit by a ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub voxel: Voxel,
    /// Voxel coordinate of the hit.
    pub position: IVec3,
    /// Face normal. Zero when the ray started inside a solid voxel.
    pub normal: IVec3,
    pub distance: f32,
}

/// Amanatides–Woo traversal with explicit empty-space handling. The implementation
/// is deliberately straightforward: it is the CPU reference for shader tests.
pub fn raycast(
    world: &VoxelWorld,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
) -> Option<RayHit> {
    if direction.length_squared() < 1e-12 || max_distance <= 0.0 || !origin.is_finite() {
        return None;
    }
    let direction = direction.normalize();
    let (t_enter, t_exit) = intersect_world(origin, direction);
    if t_exit < 0.0 || t_enter > max_distance || t_enter > t_exit {
        return None;
    }

    let t_start = t_enter.max(0.0);
    // Nudge the sample off the exact face to avoid landing on a boundary.
    let start = origin + direction * (t_start + 1e-4);
    let mut cell = IVec3::new(
        (start.x / VOXEL_SIZE_M).floor() as i32,
        (start.y / VOXEL_SIZE_M).floor() as i32,
        (start.z / VOXEL_SIZE_M).floor() as i32,
    );
    let max_cell = IVec3::new(
        (WORLD_MAX.x / VOXEL_SIZE_M) as i32 - 1,
        (WORLD_MAX.y / VOXEL_SIZE_M) as i32 - 1,
        (WORLD_MAX.z / VOXEL_SIZE_M) as i32 - 1,
    );
    cell = cell.clamp(IVec3::ZERO, max_cell);

    let mut step = IVec3::new(
        if direction.x >= 0.0 { 1 } else { -1 },
        if direction.y >= 0.0 { 1 } else { -1 },
        if direction.z >= 0.0 { 1 } else { -1 },
    );
    if direction.x == 0.0 {
        step.x = 0;
    }
    if direction.y == 0.0 {
        step.y = 0;
    }
    if direction.z == 0.0 {
        step.z = 0;
    }

    let mut t_max = Vec3::splat(f32::INFINITY);
    let mut t_delta = Vec3::splat(f32::INFINITY);
    for axis in 0..3 {
        if direction[axis].abs() > 1e-8 {
            let boundary = if step[axis] > 0 {
                (cell[axis] + 1) as f32 * VOXEL_SIZE_M
            } else {
                cell[axis] as f32 * VOXEL_SIZE_M
            };
            t_max[axis] = (boundary - origin[axis]) / direction[axis];
            t_delta[axis] = VOXEL_SIZE_M / direction[axis].abs();
        }
    }

    let mut distance = t_start;
    let mut normal = IVec3::ZERO;
    let mut previous = cell;

    // 2048 voxels along the longest arena axis plus slack for diagonal rays.
    for _ in 0..8192 {
        if cell.x < 0
            || cell.y < 0
            || cell.z < 0
            || cell.x > max_cell.x
            || cell.y > max_cell.y
            || cell.z > max_cell.z
        {
            return None;
        }

        let voxel = world.voxel(cell.x, cell.y, cell.z);
        if voxel.is_solid() {
            if distance == t_start && previous == cell {
                normal = IVec3::ZERO;
            }
            return Some(RayHit {
                voxel,
                position: cell,
                normal,
                distance: distance.max(0.0),
            });
        }

        previous = cell;
        let axis = if t_max.x <= t_max.y && t_max.x <= t_max.z {
            0
        } else if t_max.y <= t_max.z {
            1
        } else {
            2
        };
        distance = t_max[axis];
        if distance > max_distance {
            return None;
        }
        cell[axis] += step[axis];
        t_max[axis] += t_delta[axis];
        normal = match axis {
            0 => IVec3::new(-step.x, 0, 0),
            1 => IVec3::new(0, -step.y, 0),
            _ => IVec3::new(0, 0, -step.z),
        };
    }

    None
}

fn intersect_world(origin: Vec3, direction: Vec3) -> (f32, f32) {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    for axis in 0..3 {
        let d = direction[axis];
        let o = origin[axis];
        if d.abs() < 1e-8 {
            if o < WORLD_MIN[axis] || o > WORLD_MAX[axis] {
                return (1.0, -1.0);
            }
        } else {
            let inv = 1.0 / d;
            let mut t0 = (WORLD_MIN[axis] - o) * inv;
            let mut t1 = (WORLD_MAX[axis] - o) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
        }
    }
    (t_min, t_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::materials;
    use crate::world::VoxelWorld;

    fn floor_world() -> VoxelWorld {
        let mut world = VoxelWorld::default();
        for z in 0..64 {
            for x in 0..64 {
                for y in 0..8 {
                    world.set_voxel(x, y, z, Voxel::new(materials::CONCRETE, 0));
                }
            }
        }
        world
    }

    #[test]
    fn hits_floor_from_above() {
        let world = floor_world();
        let hit = world
            .raycast(Vec3::new(4.0, 3.0, 4.0), Vec3::NEG_Y, 20.0)
            .expect("floor hit");
        assert_eq!(hit.normal, IVec3::Y);
        assert_eq!(hit.position.y, 7);
        assert!((hit.distance - 2.0).abs() < 0.01);
    }

    #[test]
    fn misses_sideways() {
        let world = floor_world();
        assert!(
            world
                .raycast(Vec3::new(2.0, 3.0, 2.0), Vec3::X, 20.0)
                .is_none()
        );
    }

    #[test]
    fn handles_origin_outside_and_negative() {
        let world = floor_world();
        let hit = world
            .raycast(Vec3::new(-5.0, 0.5, 4.0), Vec3::X, 20.0)
            .expect("enter world and hit floor");
        assert_eq!(hit.position.x, 0);
        // The ray starts exactly at the world boundary, so the initial cell has
        // no face normal; shading falls back to `-direction`.
        assert_eq!(hit.normal, IVec3::ZERO);
    }

    #[test]
    fn respects_max_distance() {
        let world = floor_world();
        assert!(
            world
                .raycast(Vec3::new(4.0, 3.0, 4.0), Vec3::NEG_Y, 1.0)
                .is_none()
        );
    }

    #[test]
    fn zero_direction_is_none() {
        let world = floor_world();
        assert!(world.raycast(Vec3::ZERO, Vec3::ZERO, 10.0).is_none());
    }
}
