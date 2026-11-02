//! Walk-mode player controller: gravity, voxel collision and step-up.
//!
//! Collision is resolved against the static voxel world only. Debris bodies are
//! ignored for movement so the player cannot get pushed through a wall by a
//! sleeping piece; Jolt still owns their simulation.

use glam::{Vec3, Vec3Swizzles};
use shatterbox_core::{Action, InputState};
use shatterbox_voxel::{VOXEL_SIZE_M, VoxelWorld};

const GRAVITY: f32 = -24.0;
const WALK_SPEED: f32 = 4.8;
const SPRINT_SPEED: f32 = 7.8;
const JUMP_SPEED: f32 = 7.2;
const TERMINAL_VELOCITY: f32 = -45.0;
/// Maximum ledge the controller steps onto without jumping (m).
const STEP_HEIGHT: f32 = 0.55;
/// Longest axis move before a collision sample (m); keeps fast falls from
/// tunnelling through the 12.5 cm floor slab.
const MAX_SUBSTEP: f32 = 0.12;

/// First-person walker. `position` is the feet centre, the camera sits at
/// `position + eye_height`.
#[derive(Clone, Debug)]
pub struct Player {
    pub position: Vec3,
    pub velocity: Vec3,
    pub grounded: bool,
    pub eye_height: f32,
    pub half_extents: Vec3,
}

impl Default for Player {
    fn default() -> Self {
        Self::new(Vec3::new(128.0, 2.0, 180.0))
    }
}

impl Player {
    #[must_use]
    pub fn new(feet: Vec3) -> Self {
        Self {
            position: feet,
            velocity: Vec3::ZERO,
            grounded: false,
            eye_height: 1.62,
            half_extents: Vec3::new(0.3, 0.9, 0.3),
        }
    }

    #[must_use]
    pub fn eye(&self) -> Vec3 {
        self.position + Vec3::Y * self.eye_height
    }

    /// Drops the player onto the first solid surface below `hint`.
    pub fn spawn_on_floor(&mut self, world: &VoxelWorld, hint: Vec3) {
        let mut feet = hint;
        if let Some(hit) = world.raycast(hint, Vec3::NEG_Y, 64.0) {
            feet = hint - Vec3::Y * hit.distance;
        }
        // Two voxels of clearance above the surface.
        self.position = feet + Vec3::Y * (VOXEL_SIZE_M * 2.0);
        self.velocity = Vec3::ZERO;
        self.grounded = false;
    }

    /// One fixed-ish update. `yaw` comes from the camera so WASD follows the
    /// view; pitch is deliberately ignored for movement.
    pub fn update(&mut self, world: &VoxelWorld, dt: f32, input: &InputState, yaw: f32) {
        if !input.gameplay_enabled() || dt <= 0.0 {
            return;
        }
        let dt = dt.min(0.05);
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let forward = Vec3::new(sin_yaw, 0.0, -cos_yaw);
        let right = Vec3::new(cos_yaw, 0.0, sin_yaw);

        let mut wish = Vec3::ZERO;
        if input.down(Action::MoveForward) {
            wish += forward;
        }
        if input.down(Action::MoveBackward) {
            wish -= forward;
        }
        if input.down(Action::MoveRight) {
            wish += right;
        }
        if input.down(Action::MoveLeft) {
            wish -= right;
        }
        let speed = if input.down(Action::Sprint) {
            SPRINT_SPEED
        } else {
            WALK_SPEED
        };
        self.velocity.x = wish.x * speed;
        self.velocity.z = wish.z * speed;

        if self.grounded && input.pressed(Action::MoveUp) {
            self.velocity.y = JUMP_SPEED;
            self.grounded = false;
        }
        self.velocity.y = (self.velocity.y + GRAVITY * dt).max(TERMINAL_VELOCITY);

        self.grounded = false;
        self.move_vertical(world, self.velocity.y * dt);
        let horizontal = self.velocity.xz() * dt;
        self.move_horizontal(world, horizontal);

        // A short downward probe makes flat ground "grounded" even if gravity
        // happened to be cancelled by a sub-step.
        if !self.grounded && self.probe_ground(world, 0.06) {
            self.grounded = true;
        }
        if self.position.y < 0.0 {
            self.position.y = 0.0;
            self.velocity.y = 0.0;
            self.grounded = true;
        }
    }

    fn move_vertical(&mut self, world: &VoxelWorld, dy: f32) {
        if dy == 0.0 {
            return;
        }
        let steps = ((dy.abs() / MAX_SUBSTEP).ceil() as i32).max(1);
        let step = dy / steps as f32;
        for _ in 0..steps {
            let candidate = self.position + Vec3::Y * step;
            if self.overlaps(world, candidate) {
                self.velocity.y = 0.0;
                if step < 0.0 {
                    self.grounded = true;
                }
                return;
            }
            self.position = candidate;
        }
    }

    fn move_horizontal(&mut self, world: &VoxelWorld, delta: glam::Vec2) {
        if delta.length_squared() == 0.0 {
            return;
        }
        let steps = ((delta.length() / MAX_SUBSTEP).ceil() as i32).max(1);
        let step = delta / steps as f32;
        for _ in 0..steps {
            self.slide_axis(world, Vec3::X, step.x);
            self.slide_axis(world, Vec3::Z, step.y);
        }
    }

    fn slide_axis(&mut self, world: &VoxelWorld, axis: Vec3, amount: f32) {
        if amount == 0.0 {
            return;
        }
        let candidate = self.position + axis * amount;
        if !self.overlaps(world, candidate) {
            self.position = candidate;
            return;
        }
        // Step-up: lift, slide, and keep the higher position. Gravity settles
        // the player back onto the stair on the next update.
        let lifted = self.position + Vec3::Y * STEP_HEIGHT;
        let stepped = lifted + axis * amount;
        if !self.overlaps(world, lifted) && !self.overlaps(world, stepped) {
            self.position = stepped;
            self.grounded = true;
        }
    }

    fn probe_ground(&self, world: &VoxelWorld, distance: f32) -> bool {
        self.overlaps(world, self.position - Vec3::Y * distance)
    }

    /// Axis-aligned box test against solid voxels. Reads at most a 6×16×6
    /// neighbourhood, so it is cheap enough to call several times per step.
    #[must_use]
    pub fn overlaps(&self, world: &VoxelWorld, feet: Vec3) -> bool {
        let center = feet + Vec3::Y * self.half_extents.y;
        let min = center - self.half_extents;
        let max = center + self.half_extents;
        let x0 = (min.x / VOXEL_SIZE_M).floor() as i32;
        let y0 = (min.y / VOXEL_SIZE_M).floor() as i32;
        let z0 = (min.z / VOXEL_SIZE_M).floor() as i32;
        let x1 = (max.x / VOXEL_SIZE_M).floor() as i32;
        let y1 = (max.y / VOXEL_SIZE_M).floor() as i32;
        let z1 = (max.z / VOXEL_SIZE_M).floor() as i32;
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    if world.voxel(x, y, z).is_solid() {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shatterbox_voxel::Voxel;
    use shatterbox_voxel::materials;

    /// Flat 8×8 m slab with its top surface at y = 1.0 m.
    fn floor_world() -> VoxelWorld {
        let mut world = VoxelWorld::new(Default::default());
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
    fn player_falls_and_rests_on_the_floor() {
        let world = floor_world();
        let mut player = Player::new(Vec3::new(4.0, 5.0, 4.0));
        let input = InputState::default();
        for _ in 0..240 {
            player.update(&world, 1.0 / 60.0, &input, 0.0);
        }
        assert!(player.grounded, "player never landed");
        assert!(
            (player.position.y - 1.0).abs() < 0.05,
            "feet at {}, expected 1.0",
            player.position.y
        );
        assert!((player.eye().y - 2.62).abs() < 0.05);
    }

    #[test]
    fn player_walks_and_stops_at_a_wall() {
        let mut world = floor_world();
        for z in 0..64 {
            for y in 8..24 {
                world.set_voxel(16, y, z, Voxel::new(materials::BRICK, 0));
            }
        }
        let mut player = Player::new(Vec3::new(0.5, 1.0, 1.0));
        player.grounded = true;
        let mut input = InputState::default();
        input.set_action(Action::MoveForward, true);
        // yaw = +PI/2 makes forward +X.
        for _ in 0..240 {
            player.update(&world, 1.0 / 60.0, &input, std::f32::consts::FRAC_PI_2);
        }
        assert!(
            player.position.x < 16.0 * VOXEL_SIZE_M - 0.2,
            "player walked through the wall: x = {}",
            player.position.x
        );
        assert!(player.position.x > 1.0);
    }

    #[test]
    fn player_steps_onto_a_curb() {
        let mut world = floor_world();
        for z in 0..64 {
            for x in 16..32 {
                for y in 8..12 {
                    world.set_voxel(x, y, z, Voxel::new(materials::CONCRETE, 0));
                }
            }
        }
        let mut player = Player::new(Vec3::new(1.0, 1.0, 2.0));
        player.grounded = true;
        let mut input = InputState::default();
        input.set_action(Action::MoveForward, true);
        for _ in 0..40 {
            player.update(&world, 1.0 / 60.0, &input, std::f32::consts::FRAC_PI_2);
        }
        assert!(
            player.position.y >= 1.4,
            "player failed to step up: y = {}",
            player.position.y
        );
    }
}
