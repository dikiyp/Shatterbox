//! Arena-scale voxel world: 8 × 2 × 8 regions of 32 m.

use glam::{IVec3, Vec3};

use crate::brick::{BRICK_DIM, Brick};
use crate::dda::{RayHit, raycast};
use crate::gpu::GpuWorldData;
use crate::palette::Palette;
use crate::region::{REGION_BRICKS, REGION_VOXELS, Region};
use crate::voxel::Voxel;

pub const REGIONS_X: usize = 8;
pub const REGIONS_Y: usize = 2;
pub const REGIONS_Z: usize = 8;
pub const REGION_COUNT: usize = REGIONS_X * REGIONS_Y * REGIONS_Z;

pub const BRICKS_X: usize = REGIONS_X * REGION_BRICKS;
pub const BRICKS_Y: usize = REGIONS_Y * REGION_BRICKS;
pub const BRICKS_Z: usize = REGIONS_Z * REGION_BRICKS;

pub const VOXELS_X: usize = REGIONS_X * REGION_VOXELS;
pub const VOXELS_Y: usize = REGIONS_Y * REGION_VOXELS;
pub const VOXELS_Z: usize = REGIONS_Z * REGION_VOXELS;

/// Edge length of one voxel in metres.
pub const VOXEL_SIZE_M: f32 = 0.125;
pub const REGION_SIZE_M: f32 = 32.0;
pub const WORLD_MIN: Vec3 = Vec3::ZERO;
pub const WORLD_MAX: Vec3 = Vec3::new(256.0, 64.0, 256.0);

/// Sparse arena world. Regions are allocated lazily, which keeps an empty world
/// nearly free and makes the test scene fast to build.
pub struct VoxelWorld {
    regions: Vec<Option<Region>>,
    palette: Palette,
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self::new(Palette::default())
    }
}

impl VoxelWorld {
    pub fn new(palette: Palette) -> Self {
        Self {
            regions: (0..REGION_COUNT).map(|_| None).collect(),
            palette,
        }
    }

    pub fn palette(&self) -> &Palette {
        &self.palette
    }

    pub const fn region_index(x: usize, y: usize, z: usize) -> usize {
        x + REGIONS_X * (y + REGIONS_Y * z)
    }

    pub fn region(&self, x: usize, y: usize, z: usize) -> Option<&Region> {
        if x >= REGIONS_X || y >= REGIONS_Y || z >= REGIONS_Z {
            return None;
        }
        self.regions[Self::region_index(x, y, z)].as_ref()
    }

    /// Reads a voxel in world voxel coordinates. Out-of-bounds is air.
    pub fn voxel(&self, x: i32, y: i32, z: i32) -> Voxel {
        if x < 0 || y < 0 || z < 0 {
            return Voxel::AIR;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
        if x >= VOXELS_X || y >= VOXELS_Y || z >= VOXELS_Z {
            return Voxel::AIR;
        }
        let region = (x / REGION_VOXELS, y / REGION_VOXELS, z / REGION_VOXELS);
        match self.region(region.0, region.1, region.2) {
            Some(region) => region.get(x % REGION_VOXELS, y % REGION_VOXELS, z % REGION_VOXELS),
            None => Voxel::AIR,
        }
    }

    /// Writes a voxel in world voxel coordinates. Out-of-bounds writes are
    /// ignored; scene builders can pass unclipped boxes safely.
    pub fn set_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel) {
        if x < 0 || y < 0 || z < 0 {
            return;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
        if x >= VOXELS_X || y >= VOXELS_Y || z >= VOXELS_Z {
            return;
        }
        let index = Self::region_index(x / REGION_VOXELS, y / REGION_VOXELS, z / REGION_VOXELS);
        let region = self.regions[index].get_or_insert_with(Region::default);
        region.set(
            x % REGION_VOXELS,
            y % REGION_VOXELS,
            z % REGION_VOXELS,
            voxel,
        );
    }

    /// Removes a voxel and returns its previous value. Used by destruction so
    /// the exact material/tint can be preserved in the debris body.
    pub fn take_voxel(&mut self, x: i32, y: i32, z: i32) -> Voxel {
        let previous = self.voxel(x, y, z);
        if !previous.is_air() {
            self.set_voxel(x, y, z, Voxel::AIR);
        }
        previous
    }

    /// Writes at a position in metres.
    pub fn set_voxel_m(&mut self, position: Vec3, voxel: Voxel) {
        let p = (position * (1.0 / VOXEL_SIZE_M)).floor();
        self.set_voxel(p.x as i32, p.y as i32, p.z as i32, voxel);
    }

    pub fn non_empty_regions(&self) -> usize {
        self.regions
            .iter()
            .flatten()
            .filter(|region| !region.is_empty())
            .count()
    }

    pub fn brick_count(&self) -> usize {
        self.regions
            .iter()
            .flatten()
            .map(Region::non_empty_bricks)
            .sum::<u32>() as usize
    }

    pub fn solid_voxels(&self) -> u64 {
        self.regions
            .iter()
            .flatten()
            .flat_map(Region::iter_non_empty)
            .map(|(_, brick)| brick.solid_count() as u64)
            .sum()
    }

    /// Counts solid voxels above `anchor_y` that are not 6-connected to a solid
    /// voxel in the `anchor_y` layer. Returns `(floating, total_above_anchor)`.
    ///
    /// The scan walks non-empty bricks instead of the 2048 x 512 x 2048 grid, so
    /// it is cheap enough for a startup `assert_no_floating`. Only solids at or
    /// above the anchor layer participate: the plate below it is the ground and
    /// is never reported as floating.
    pub fn count_unanchored_above(&self, anchor_y: i32) -> (usize, usize) {
        use std::collections::HashSet;

        let mut seeds: Vec<u32> = Vec::new();
        let mut total = 0usize;
        for rz in 0..REGIONS_Z {
            for ry in 0..REGIONS_Y {
                for rx in 0..REGIONS_X {
                    let Some(region) = self.region(rx, ry, rz) else {
                        continue;
                    };
                    for (slot, brick) in region.iter_non_empty() {
                        let bx = (slot % REGION_BRICKS) * BRICK_DIM;
                        let by = (slot / REGION_BRICKS % REGION_BRICKS) * BRICK_DIM;
                        let bz = (slot / (REGION_BRICKS * REGION_BRICKS)) * BRICK_DIM;
                        let base_x = (rx * REGION_VOXELS + bx) as i32;
                        let base_y = (ry * REGION_VOXELS + by) as i32;
                        let base_z = (rz * REGION_VOXELS + bz) as i32;
                        for lz in 0..BRICK_DIM {
                            for ly in 0..BRICK_DIM {
                                for lx in 0..BRICK_DIM {
                                    if !brick.get(lx, ly, lz).is_solid() {
                                        continue;
                                    }
                                    let y = base_y + ly as i32;
                                    if y < anchor_y {
                                        continue;
                                    }
                                    total += 1;
                                    if y == anchor_y {
                                        seeds.push(pack_voxel(
                                            base_x + lx as i32,
                                            y,
                                            base_z + lz as i32,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut visited: HashSet<u32> = HashSet::with_capacity(total.min(1 << 20));
        let mut queue = seeds;
        for &seed in &queue {
            visited.insert(seed);
        }
        let mut head = 0usize;
        while head < queue.len() {
            let packed = queue[head];
            head += 1;
            let x = (packed & 0x7ff) as i32;
            let y = ((packed >> 11) & 0x1ff) as i32;
            let z = ((packed >> 20) & 0x7ff) as i32;
            for (dx, dy, dz) in [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ] {
                let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                if ny < anchor_y || !self.voxel(nx, ny, nz).is_solid() {
                    continue;
                }
                let next = pack_voxel(nx, ny, nz);
                if visited.insert(next) {
                    queue.push(next);
                }
            }
        }

        (total.saturating_sub(visited.len()), total)
    }

    /// FNV-1a over the sparse structure in fixed slot order. Used by determinism
    /// tests and shown in the diagnostics overlay.
    pub fn deterministic_hash(&self) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut hash = OFFSET;
        let mut mix = |value: u32| {
            hash ^= value as u64;
            hash = hash.wrapping_mul(PRIME);
        };

        for region in &self.regions {
            match region {
                None => mix(0),
                Some(region) => {
                    mix(1);
                    for (slot, brick) in region.iter_non_empty() {
                        mix(slot as u32);
                        for word in brick.raw_voxels().chunks_exact(2) {
                            mix(word[0] as u32 | ((word[1] as u32) << 16));
                        }
                    }
                }
            }
        }
        hash
    }

    pub fn to_gpu(&self) -> GpuWorldData {
        GpuWorldData::from_world(self)
    }

    /// CPU reference raycast, used to validate the GPU DDA.
    pub fn raycast(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<RayHit> {
        raycast(self, origin, direction, max_distance)
    }

    /// Returns the brick containing a brick-grid coordinate.
    pub fn brick(&self, x: usize, y: usize, z: usize) -> Option<&Brick> {
        if x >= BRICKS_X || y >= BRICKS_Y || z >= BRICKS_Z {
            return None;
        }
        self.region(x / REGION_BRICKS, y / REGION_BRICKS, z / REGION_BRICKS)
            .and_then(|region| {
                region.brick_at(x % REGION_BRICKS, y % REGION_BRICKS, z % REGION_BRICKS)
            })
    }

    pub fn voxel_to_brick(voxel: IVec3) -> IVec3 {
        IVec3::new(
            voxel.x.div_euclid(BRICK_DIM as i32),
            voxel.y.div_euclid(BRICK_DIM as i32),
            voxel.z.div_euclid(BRICK_DIM as i32),
        )
    }
}

/// Packs non-negative voxel coordinates into 31 bits: 11 x, 9 y, 11 z.
fn pack_voxel(x: i32, y: i32, z: i32) -> u32 {
    (x as u32 & 0x7ff) | ((y as u32 & 0x1ff) << 11) | ((z as u32 & 0x7ff) << 20)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::materials;

    #[test]
    fn floating_voxel_is_reported_and_grounded_is_not() {
        let mut world = VoxelWorld::default();
        // Grounded pillar: bottom voxel sits exactly on the anchor layer.
        world.set_voxel(10, 8, 10, Voxel::new(materials::CONCRETE, 0));
        world.set_voxel(10, 9, 10, Voxel::new(materials::CONCRETE, 0));
        // Floating island two metres above the floor.
        world.set_voxel(20, 24, 20, Voxel::new(materials::BRICK, 0));

        let (floating, total) = world.count_unanchored_above(8);
        assert_eq!(floating, 1, "one disconnected voxel expected");
        assert_eq!(total, 3, "two grounded + one floating voxel");
    }

    #[test]
    fn diagonal_contact_does_not_anchor_a_voxel() {
        let mut world = VoxelWorld::default();
        world.set_voxel(10, 8, 10, Voxel::new(materials::CONCRETE, 0));
        // Touches the anchor only through an edge, not a face.
        world.set_voxel(11, 9, 10, Voxel::new(materials::CONCRETE, 0));
        assert_eq!(world.count_unanchored_above(8).0, 1);
    }
}
