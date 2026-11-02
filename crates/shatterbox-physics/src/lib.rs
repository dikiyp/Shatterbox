//! Physics facade. The game only sees [`PhysicsBackend`]; Jolt lives entirely
//! behind [`JoltPhysics`] so no `unsafe` or third-party type leaks upwards.

use std::collections::HashMap;

use glam::{IVec3, Quat, Vec3};
use oxijolt::prelude::*;
use thiserror::Error;

/// Stable handle owned by the game/destruction layer, independent of Jolt's
/// 8-bit sequence numbers.
pub type BodyHandle = u64;

/// Jolt refuses a body whose velocities are non-finite or above its internal
/// `limits::MAX_*_VELOCITY`, and the whole piece is dropped. Everything that
/// reaches the backend is clamped here so a bad impulse can never delete debris.
const MAX_LINEAR_VELOCITY: f32 = 200.0;
const MAX_ANGULAR_VELOCITY: f32 = 30.0;

fn sanitize(velocity: Vec3, limit: f32) -> Vec3 {
    if velocity.is_finite() {
        velocity.clamp_length_max(limit)
    } else {
        Vec3::ZERO
    }
}

/// One box in a compound collider, in the body's local space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompoundBox {
    pub center: Vec3,
    pub half_extents: Vec3,
}

impl CompoundBox {
    pub const fn new(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            center,
            half_extents,
        }
    }
}

/// Everything needed to create a debris body. The shape is a compound of boxes
/// because a voxel chunk is mostly empty space and a convex hull around all of
/// it would collide with air.
#[derive(Clone, Debug)]
pub struct BodyDesc {
    pub position: Vec3,
    pub rotation: Quat,
    pub boxes: Vec<CompoundBox>,
    pub mass: f32,
    /// Full inertia tensor around the centre of mass, in body-local axes.
    /// `None` lets Jolt derive it from the compound shape.
    pub inertia: Option<[[f32; 3]; 3]>,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub friction: f32,
    pub restitution: f32,
}

impl BodyDesc {
    pub fn new(position: Vec3, boxes: Vec<CompoundBox>, mass: f32) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
            boxes,
            mass: mass.max(0.1),
            inertia: None,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            friction: 0.72,
            restitution: 0.03,
        }
    }
}

/// Sampled body state for rendering and diagnostics.
#[derive(Clone, Copy, Debug)]
pub struct BodyState {
    pub position: Vec3,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub awake: bool,
    pub sleeping: bool,
}

/// Small interface the game and destruction system rely on. A CPU solver or
/// Rapier adapter can implement the same surface without touching gameplay.
pub trait PhysicsBackend {
    fn add_static_box(&mut self, center: Vec3, half_extents: Vec3);
    /// Replaces all static geometry belonging to one 8 m collision sector.
    /// `boxes` are already in world space; an empty slice removes the sector.
    fn replace_static_sector(&mut self, key: IVec3, boxes: &[CompoundBox]);
    fn remove_static_sector(&mut self, key: IVec3);
    fn spawn(&mut self, desc: &BodyDesc) -> Option<BodyHandle>;
    fn remove(&mut self, handle: BodyHandle);
    fn step(&mut self, dt: f32);
    fn state(&self, handle: BodyHandle) -> Option<BodyState>;
    fn body_count(&self) -> usize;
    fn awake_count(&self) -> usize;
    fn set_linear_velocity(&mut self, handle: BodyHandle, velocity: Vec3);
}

#[derive(Debug, Error)]
pub enum PhysicsError {
    #[error("failed to create Jolt world: {0}")]
    World(String),
    #[error("failed to create collider shape: {0}")]
    Shape(String),
    #[error("failed to create body: {0}")]
    Body(String),
}

/// Jolt 5.6 adapter. All oxijolt calls stay in this file.
pub struct JoltPhysics {
    world: PhysicsWorld,
    handles: HashMap<BodyHandle, BodyId>,
    static_sectors: HashMap<IVec3, BodyId>,
    next_handle: BodyHandle,
}

impl JoltPhysics {
    pub fn new() -> Result<Self, PhysicsError> {
        // Jolt `MAX_BODIES` is a hard pool: once it is exhausted every spawn
        // fails with "the world is full" and debris silently disappears. The
        // destruction system keeps its own live-piece budget well below this,
        // so the pool is only a backstop for pathological scenes.
        let settings = WorldSettings::default()
            .max_bodies(16384)
            .velocity_steps(10);
        let world =
            PhysicsWorld::new(settings).map_err(|error| PhysicsError::World(error.to_string()))?;
        Ok(Self {
            world,
            handles: HashMap::new(),
            static_sectors: HashMap::new(),
            next_handle: 1,
        })
    }
}

impl JoltPhysics {
    fn compound_shape(boxes: &[CompoundBox]) -> Option<Shape> {
        let mut shapes = Vec::with_capacity(boxes.len());
        for part in boxes {
            let shape = match Shape::new_box(part.half_extents.max(Vec3::splat(0.02)).into()) {
                Ok(shape) => shape,
                Err(error) => {
                    tracing::warn!(%error, "sector box rejected");
                    return None;
                }
            };
            shapes.push(shape);
        }
        let children: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| CompoundChild {
                shape,
                position: boxes[index].center.into(),
                rotation: Quat::IDENTITY.into(),
                user_data: index as u32,
            })
            .collect();
        match Shape::new_compound(&children) {
            Ok(shape) => Some(shape),
            Err(error) => {
                tracing::warn!(%error, "sector compound rejected");
                None
            }
        }
    }
}

impl PhysicsBackend for JoltPhysics {
    fn replace_static_sector(&mut self, key: IVec3, boxes: &[CompoundBox]) {
        if let Some(old) = self.static_sectors.remove(&key) {
            let _ = self.world.remove_body(old);
        }
        if boxes.is_empty() {
            return;
        }
        let Some(shape) = Self::compound_shape(boxes) else {
            return;
        };
        // Boxes are world-space, so the sector body sits at the origin.
        let settings = BodySettings::new_static();
        match self.world.create_body(&shape, &settings) {
            Ok(body) => {
                self.static_sectors.insert(key, body);
            }
            Err(error) => tracing::error!(%error, "static sector body rejected"),
        }
    }

    fn remove_static_sector(&mut self, key: IVec3) {
        if let Some(body) = self.static_sectors.remove(&key) {
            let _ = self.world.remove_body(body);
        }
    }

    fn add_static_box(&mut self, center: Vec3, half_extents: Vec3) {
        let shape = match Shape::new_box(half_extents.into()) {
            Ok(shape) => shape,
            Err(error) => {
                tracing::error!(%error, "static box shape rejected");
                return;
            }
        };
        let settings = BodySettings::new_static().position(center.into());
        if let Err(error) = self.world.create_body(&shape, &settings) {
            tracing::error!(%error, "static box body rejected");
        }
    }

    fn spawn(&mut self, desc: &BodyDesc) -> Option<BodyHandle> {
        if desc.boxes.is_empty() {
            return None;
        }

        let mut shapes = Vec::with_capacity(desc.boxes.len());
        for part in &desc.boxes {
            let shape = match Shape::new_box(part.half_extents.max(Vec3::splat(0.02)).into()) {
                Ok(shape) => shape,
                Err(error) => {
                    tracing::warn!(%error, "debris box rejected");
                    return None;
                }
            };
            shapes.push(shape);
        }
        let children: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| CompoundChild {
                shape,
                position: desc.boxes[index].center.into(),
                rotation: Quat::IDENTITY.into(),
                user_data: index as u32,
            })
            .collect();

        let shape = match Shape::new_compound(&children) {
            Ok(shape) => shape,
            Err(error) => {
                tracing::warn!(%error, "debris compound rejected");
                return None;
            }
        };
        let settings = BodySettings::new_dynamic()
            .position(desc.position.into())
            .rotation(desc.rotation.into())
            .linear_velocity(sanitize(desc.linear_velocity, MAX_LINEAR_VELOCITY).into())
            .angular_velocity(sanitize(desc.angular_velocity, MAX_ANGULAR_VELOCITY).into())
            .friction(desc.friction)
            .restitution(desc.restitution)
            .linear_damping(0.035)
            .angular_damping(0.08);
        // The destruction system computes the tensor from the actual voxels.
        // Letting Jolt derive it from the compound would use the box hull and
        // make a thin shard spin like a slab.
        let settings = match desc.inertia {
            Some(tensor) => settings.mass_and_inertia(desc.mass, Inertia::Tensor(tensor)),
            None => settings.mass(desc.mass),
        };
        let settings = settings
            .allow_sleeping(true)
            .motion_quality(MotionQuality::LinearCast);
        let body = match self.world.create_body(&shape, &settings) {
            Ok(body) => body,
            Err(error) => {
                tracing::warn!(%error, "debris body rejected");
                return None;
            }
        };
        let handle = self.next_handle;
        self.next_handle += 1;
        self.handles.insert(handle, body);
        Some(handle)
    }

    fn remove(&mut self, handle: BodyHandle) {
        if let Some(body) = self.handles.remove(&handle) {
            let _ = self.world.remove_body(body);
        }
    }

    fn step(&mut self, dt: f32) {
        if let Err(error) = self.world.step(dt.clamp(0.0, 1.0 / 20.0)) {
            tracing::warn!(%error, "Jolt step failed");
        }
    }

    fn state(&self, handle: BodyHandle) -> Option<BodyState> {
        let id = *self.handles.get(&handle)?;
        let body = self.world.body(id).ok()?;
        Some(BodyState {
            position: body.position().into(),
            rotation: body.rotation().into(),
            linear_velocity: body.linear_velocity().into(),
            angular_velocity: body.angular_velocity().into(),
            awake: body.is_active(),
            sleeping: body.is_sleeping(),
        })
    }

    fn body_count(&self) -> usize {
        self.handles.len()
    }

    fn awake_count(&self) -> usize {
        self.handles
            .values()
            .filter(|id| self.world.body(**id).is_ok_and(|body| body.is_active()))
            .count()
    }

    fn set_linear_velocity(&mut self, handle: BodyHandle, velocity: Vec3) {
        if let Some(id) = self.handles.get(&handle).copied()
            && let Ok(mut body) = self.world.body_mut(id)
        {
            let _ = body.set_linear_velocity(velocity.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_body_falls_rests_and_sleeps() {
        let mut physics = JoltPhysics::new().expect("jolt world");
        physics.add_static_box(Vec3::new(0.0, -0.5, 0.0), Vec3::new(8.0, 0.5, 8.0));
        let desc = BodyDesc::new(
            Vec3::new(0.0, 3.0, 0.0),
            vec![CompoundBox::new(Vec3::ZERO, Vec3::splat(0.25))],
            2.0,
        );
        let handle = physics.spawn(&desc).expect("spawn debris");
        assert_eq!(physics.body_count(), 1);

        let mut was_awake = false;
        for _ in 0..600 {
            physics.step(1.0 / 60.0);
            let state = physics.state(handle).expect("state");
            was_awake |= state.awake;
        }

        let state = physics.state(handle).expect("state");
        assert!(was_awake, "body never woke up");
        assert!(
            (state.position.y - 0.25).abs() < 0.08,
            "body did not settle on the floor: y={}",
            state.position.y
        );
        assert!(state.sleeping, "settled body should sleep");

        physics.remove(handle);
        assert_eq!(physics.body_count(), 0);
    }

    #[test]
    fn static_sector_supports_and_replaces_a_body() {
        let mut physics = JoltPhysics::new().expect("jolt world");
        let floor = [CompoundBox::new(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(8.0, 0.5, 8.0),
        )];
        physics.replace_static_sector(IVec3::ZERO, &floor);
        let desc = BodyDesc::new(
            Vec3::new(0.0, 3.0, 0.0),
            vec![CompoundBox::new(Vec3::ZERO, Vec3::splat(0.25))],
            2.0,
        );
        let handle = physics.spawn(&desc).expect("spawn");
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
        }
        let state = physics.state(handle).expect("state");
        assert!(
            (state.position.y - 0.25).abs() < 0.1,
            "body fell through the sector floor: y={}",
            state.position.y
        );

        // Replacing the sector with an empty slice must drop the collider.
        physics.replace_static_sector(IVec3::ZERO, &[]);
        assert_eq!(physics.body_count(), 1);
    }
}
