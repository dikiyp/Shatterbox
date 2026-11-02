//! A 32 m region: 16³ bricks addressed by a sparse brick index.

use crate::brick::{BRICK_DIM, Brick};
use crate::voxel::Voxel;

pub const REGION_BRICKS: usize = 16;
pub const REGION_VOXELS: usize = REGION_BRICKS * BRICK_DIM;
pub const REGION_BRICK_SLOTS: usize = REGION_BRICKS * REGION_BRICKS * REGION_BRICKS;

/// Sparse storage for one region. `brick_index` maps a 16³ brick slot to a
/// one-based pool index; zero means the brick has never been allocated.
pub struct Region {
    brick_index: Box<[u32; REGION_BRICK_SLOTS]>,
    bricks: Vec<Brick>,
    non_empty_bricks: u32,
}

impl Default for Region {
    fn default() -> Self {
        Self {
            brick_index: Box::new([0; REGION_BRICK_SLOTS]),
            bricks: Vec::new(),
            non_empty_bricks: 0,
        }
    }
}

impl Region {
    pub const fn brick_slot(x: usize, y: usize, z: usize) -> usize {
        x + REGION_BRICKS * (y + REGION_BRICKS * z)
    }

    /// Reads a voxel in region-local coordinates (`0..256`).
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        let slot = Self::brick_slot(x / BRICK_DIM, y / BRICK_DIM, z / BRICK_DIM);
        let entry = self.brick_index[slot];
        if entry == 0 {
            Voxel::AIR
        } else {
            self.bricks[(entry - 1) as usize].get(x % BRICK_DIM, y % BRICK_DIM, z % BRICK_DIM)
        }
    }

    /// Writes a voxel in region-local coordinates.
    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        let slot = Self::brick_slot(x / BRICK_DIM, y / BRICK_DIM, z / BRICK_DIM);
        let local = (x % BRICK_DIM, y % BRICK_DIM, z % BRICK_DIM);
        let entry = self.brick_index[slot];

        if entry == 0 {
            if voxel.is_air() {
                return;
            }
            let mut brick = Brick::default();
            brick.set(local.0, local.1, local.2, voxel);
            self.bricks.push(brick);
            self.brick_index[slot] = self.bricks.len() as u32;
            self.non_empty_bricks += 1;
            return;
        }

        let index = (entry - 1) as usize;
        let was_empty = self.bricks[index].is_empty();
        self.bricks[index].set(local.0, local.1, local.2, voxel);
        let is_empty = self.bricks[index].is_empty();
        if was_empty && !is_empty {
            self.non_empty_bricks += 1;
        } else if !was_empty && is_empty {
            self.non_empty_bricks -= 1;
        }
    }

    pub fn brick_at(&self, x: usize, y: usize, z: usize) -> Option<&Brick> {
        let entry = self.brick_index[Self::brick_slot(x, y, z)];
        if entry == 0 {
            None
        } else {
            Some(&self.bricks[(entry - 1) as usize])
        }
    }

    pub fn is_empty(&self) -> bool {
        self.non_empty_bricks == 0
    }

    pub fn non_empty_bricks(&self) -> u32 {
        self.non_empty_bricks
    }

    /// Bricks that currently contain at least one solid voxel, in slot order.
    pub fn iter_non_empty(&self) -> impl Iterator<Item = (usize, &Brick)> {
        self.brick_index
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| {
                if *entry == 0 {
                    None
                } else {
                    let brick = &self.bricks[(*entry - 1) as usize];
                    (!brick.is_empty()).then_some((slot, brick))
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::materials;

    #[test]
    fn sparse_allocation_and_roundtrip() {
        let mut region = Region::default();
        assert!(region.is_empty());
        region.set(255, 255, 255, Voxel::new(materials::BRICK, 4));
        assert!(!region.is_empty());
        assert_eq!(region.non_empty_bricks(), 1);
        assert_eq!(region.get(255, 255, 255).material(), materials::BRICK);
        assert_eq!(region.brick_at(15, 15, 15).unwrap().solid_count(), 1);
        assert!(region.brick_at(0, 0, 0).is_none());
    }

    #[test]
    fn clearing_last_voxel_marks_region_empty() {
        let mut region = Region::default();
        region.set(1, 2, 3, Voxel::new(materials::WOOD, 0));
        region.set(1, 2, 3, Voxel::AIR);
        assert!(region.is_empty());
        assert_eq!(region.iter_non_empty().count(), 0);
    }
}
