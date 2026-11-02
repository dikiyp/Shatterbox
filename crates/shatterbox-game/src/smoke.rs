//! Coarse dynamic smoke field.
//!
//! 64 x 16 x 64 cells of 4 m over the 256 x 64 x 256 m arena. Semi-Lagrangian
//! advection with a constant wind and a buoyancy term, exponential decay, and
//! an obstacle mask from the voxel world: a cell whose centre is solid is
//! cleared, and a backward trace that starts inside geometry contributes
//! nothing. That is what makes the plume bend around a wall instead of passing
//! through it. The field is cosmetic: it never feeds back into destruction or
//! gameplay, so it is allowed to be approximate and frame-rate dependent.

use glam::Vec3;
use shatterbox_voxel::{VOXEL_SIZE_M, VoxelWorld};

pub const DIM_X: usize = 64;
pub const DIM_Y: usize = 16;
pub const DIM_Z: usize = 64;
pub const CELL_M: f32 = 4.0;

/// Seconds between simulation steps. The field moves slowly; 20 Hz halves the
/// CPU cost of the 65k-cell loop without a visible difference.
const STEP_S: f32 = 0.05;
const WIND: Vec3 = Vec3::new(1.1, 0.0, 0.55);
const BUOYANCY: f32 = 0.55;
const DECAY: f32 = 0.94;
const MAX_DENSITY: f32 = 1.4;

pub struct SmokeField {
    density: Vec<f32>,
    next: Vec<f32>,
    bytes: Vec<u8>,
    accumulator: f32,
}

impl Default for SmokeField {
    fn default() -> Self {
        Self::new()
    }
}

impl SmokeField {
    #[must_use]
    pub fn new() -> Self {
        let cells = DIM_X * DIM_Y * DIM_Z;
        Self {
            density: vec![0.0; cells],
            next: vec![0.0; cells],
            bytes: vec![0; cells],
            accumulator: 0.0,
        }
    }

    pub fn add_source(&mut self, position: Vec3, radius: f32, amount: f32) {
        let radius_cells = (radius / CELL_M).max(1.0);
        let center = position / CELL_M;
        let lo = [
            (center.x - radius_cells).floor().max(0.0) as usize,
            (center.y - radius_cells).floor().max(0.0) as usize,
            (center.z - radius_cells).floor().max(0.0) as usize,
        ];
        let hi = [
            ((center.x + radius_cells).ceil() as usize).min(DIM_X - 1),
            ((center.y + radius_cells).ceil() as usize).min(DIM_Y - 1),
            ((center.z + radius_cells).ceil() as usize).min(DIM_Z - 1),
        ];
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let cell = Vec3::new(x as f32, y as f32, z as f32) + Vec3::splat(0.5);
                    let falloff = 1.0 - (cell.distance(center) / radius_cells).clamp(0.0, 1.0);
                    if falloff <= 0.0 {
                        continue;
                    }
                    let index = Self::index(x, y, z);
                    self.density[index] =
                        (self.density[index] + amount * falloff * falloff).min(MAX_DENSITY);
                }
            }
        }
    }

    /// Advances the field by `dt` seconds of wall time (sub-stepped at 20 Hz).
    pub fn step(&mut self, dt: f32, world: &VoxelWorld) {
        self.accumulator += dt.clamp(0.0, 0.1);
        while self.accumulator >= STEP_S {
            self.accumulator -= STEP_S;
            self.substep(world);
        }
    }

    fn substep(&mut self, world: &VoxelWorld) {
        let wind = WIND * (STEP_S / CELL_M);
        let buoyancy = BUOYANCY * (STEP_S / CELL_M);
        for z in 0..DIM_Z {
            for y in 0..DIM_Y {
                for x in 0..DIM_X {
                    let index = Self::index(x, y, z);
                    let solid = Self::solid_at(world, x, y, z);
                    if solid {
                        self.next[index] = 0.0;
                        continue;
                    }
                    // Back-trace the wind and buoyancy. A plume rises, so the
                    // source of a cell is below it.
                    let src = Vec3::new(x as f32, y as f32, z as f32)
                        - wind
                        - Vec3::new(0.0, buoyancy, 0.0);
                    let sampled = Self::sample(&self.density, src, world);
                    self.next[index] = (sampled * DECAY).min(MAX_DENSITY);
                }
            }
        }
        std::mem::swap(&mut self.density, &mut self.next);
    }

    /// Trilinear sample of the previous field; solid sources are skipped so a
    /// plank does not "emit" the smoke that sat inside it.
    fn sample(field: &[f32], p: Vec3, world: &VoxelWorld) -> f32 {
        let x0 = p.x.floor();
        let y0 = p.y.floor();
        let z0 = p.z.floor();
        let fx = p.x - x0;
        let fy = p.y - y0;
        let fz = p.z - z0;
        let mut sum = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let xi = x0 as i32 + dx;
                    let yi = y0 as i32 + dy;
                    let zi = z0 as i32 + dz;
                    if xi < 0
                        || yi < 0
                        || zi < 0
                        || xi >= DIM_X as i32
                        || yi >= DIM_Y as i32
                        || zi >= DIM_Z as i32
                    {
                        continue;
                    }
                    let wx = if dx == 0 { 1.0 - fx } else { fx };
                    let wy = if dy == 0 { 1.0 - fy } else { fy };
                    let wz = if dz == 0 { 1.0 - fz } else { fz };
                    let weight = wx * wy * wz;
                    if weight <= 0.0 {
                        continue;
                    }
                    let value = field[Self::index(xi as usize, yi as usize, zi as usize)];
                    if value > 0.0 && Self::solid_at(world, xi as usize, yi as usize, zi as usize) {
                        continue;
                    }
                    sum += value * weight;
                }
            }
        }
        sum
    }

    fn solid_at(world: &VoxelWorld, x: usize, y: usize, z: usize) -> bool {
        let world_pos = Vec3::new(
            (x as f32 + 0.5) * CELL_M,
            (y as f32 + 0.5) * CELL_M,
            (z as f32 + 0.5) * CELL_M,
        );
        let voxel = (world_pos / VOXEL_SIZE_M).floor();
        world
            .voxel(voxel.x as i32, voxel.y as i32, voxel.z as i32)
            .is_solid()
    }

    fn index(x: usize, y: usize, z: usize) -> usize {
        x + DIM_X * (y + DIM_Y * z)
    }

    /// R8 payload for `Renderer::set_smoke_density`.
    pub fn bytes(&mut self) -> &[u8] {
        for (byte, value) in self.bytes.iter_mut().zip(self.density.iter()) {
            *byte = (value.clamp(0.0, 1.0) * 255.0) as u8;
        }
        &self.bytes
    }
}
