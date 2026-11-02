//! Free-flying debug camera used by the Phase 0 shell.

use glam::{Mat4, Vec2, Vec3};

use crate::input::{Action, InputState};

/// A yaw/pitch camera with WASD + vertical movement.
///
/// The camera uses a right-handed coordinate system. At yaw zero it looks
/// down `-Z`; positive yaw turns towards `+X`.
#[derive(Clone, Debug)]
pub struct FreeCamera {
    pub position: Vec3,
    /// Rotation around `+Y`, in radians.
    pub yaw: f32,
    /// Rotation around the local right axis, in radians. Clamped to ±89°.
    pub pitch: f32,
    pub fov_y_radians: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub move_speed: f32,
    pub look_sensitivity: f32,
}

impl Default for FreeCamera {
    fn default() -> Self {
        Self {
            position: Vec3::new(128.0, 10.0, 180.0),
            yaw: 0.0,
            pitch: -0.12,
            fov_y_radians: 70.0_f32.to_radians(),
            z_near: 0.05,
            z_far: 512.0,
            move_speed: 12.0,
            look_sensitivity: 0.0022,
        }
    }
}

impl FreeCamera {
    #[must_use]
    pub fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        Self {
            position,
            yaw,
            pitch,
            ..Self::default()
        }
    }

    /// Resets the camera to the test-arena spawn point.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    #[must_use]
    pub fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
    }

    #[must_use]
    pub fn right(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_yaw, 0.0, sin_yaw)
    }

    /// Applies a mouse delta in physical pixels, in screen convention: `+x` is
    /// right and `+y` is down, so dragging the mouse right turns the view
    /// right and dragging it down looks down.
    pub fn look(&mut self, delta: Vec2) {
        self.yaw += delta.x * self.look_sensitivity;
        self.pitch = (self.pitch - delta.y * self.look_sensitivity)
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    /// Moves the camera according to the current input snapshot.
    pub fn update(&mut self, dt: f32, input: &InputState) {
        if !input.gameplay_enabled() || dt <= 0.0 {
            return;
        }

        let mut direction = Vec3::ZERO;
        if input.down(Action::MoveForward) {
            direction += self.forward();
        }
        if input.down(Action::MoveBackward) {
            direction -= self.forward();
        }
        if input.down(Action::MoveRight) {
            direction += self.right();
        }
        if input.down(Action::MoveLeft) {
            direction -= self.right();
        }
        if input.down(Action::MoveUp) {
            direction += Vec3::Y;
        }
        if input.down(Action::MoveDown) {
            direction -= Vec3::Y;
        }

        if direction.length_squared() > 0.0 {
            let sprint = if input.down(Action::Sprint) { 4.0 } else { 1.0 };
            self.position += direction.normalize() * self.move_speed * sprint * dt;
        }
    }

    /// Adjusts movement speed with the mouse wheel, clamped to a sane range.
    pub fn adjust_speed(&mut self, scroll: f32) {
        if scroll.abs() < f32::EPSILON {
            return;
        }
        self.move_speed = (self.move_speed * (1.0 + scroll * 0.08)).clamp(1.0, 120.0);
    }

    #[must_use]
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    #[must_use]
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(
            self.fov_y_radians,
            aspect.max(0.01),
            self.z_near,
            self.z_far,
        )
    }

    #[must_use]
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_forward_is_negative_z() {
        let camera = FreeCamera::new(Vec3::ZERO, 0.0, 0.0);
        assert!((camera.forward() - Vec3::NEG_Z).length() < 1.0e-5);
        assert!((camera.right() - Vec3::X).length() < 1.0e-5);
    }

    #[test]
    fn pitch_is_clamped() {
        let mut camera = FreeCamera::default();
        camera.look(Vec2::new(0.0, -100_000.0));
        assert!(camera.pitch <= 89.0_f32.to_radians());
        camera.look(Vec2::new(0.0, 100_000.0));
        assert!(camera.pitch >= -89.0_f32.to_radians());
    }

    #[test]
    fn mouse_motion_is_not_inverted() {
        // Dragging right must turn the view towards screen right, i.e. +X.
        let mut camera = FreeCamera::new(Vec3::ZERO, 0.0, 0.0);
        camera.look(Vec2::new(100.0, 0.0));
        assert!(camera.forward().x > 0.0, "right drag turned left");
        // Dragging down must look down.
        let mut camera = FreeCamera::new(Vec3::ZERO, 0.0, 0.0);
        camera.look(Vec2::new(0.0, 100.0));
        assert!(camera.forward().y < 0.0, "down drag looked up");
    }

    #[test]
    fn projection_has_expected_depth_range() {
        let camera = FreeCamera::default();
        let projection = camera.projection_matrix(16.0 / 9.0);
        assert!(projection.is_finite());
    }
}
