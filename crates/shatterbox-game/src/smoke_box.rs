//! Obstacle mask for the local GPU smoke box.
//!
//! The simulation reads one `u32` per cell; a cell is solid when its centre is
//! inside a solid voxel, and the mask is dilated by one cell so the velocity
//! stays zero in the layer touching a wall (no-slip). Thin 12.5 cm walls are
//! therefore never crossed: a cell either contains the wall or is marked as the
//! dilated boundary.

use glam::Vec3;
use shatterbox_voxel::{VOXEL_SIZE_M, VoxelWorld};

/// Builds the mask for `grid` cells of `spacing` starting at `origin`.
#[must_use]
pub fn build_mask(
    world: &VoxelWorld,
    origin: Vec3,
    spacing: [f32; 3],
    grid: (u32, u32, u32),
) -> Vec<u32> {
    let cells = (grid.0 * grid.1 * grid.2) as usize;
    let mut solid = vec![0u32; cells];
    let spacing = Vec3::from_array(spacing);
    for z in 0..grid.2 {
        for y in 0..grid.1 {
            for x in 0..grid.0 {
                let centre =
                    origin + (Vec3::new(x as f32, y as f32, z as f32) + Vec3::splat(0.5)) * spacing;
                let voxel = (centre / VOXEL_SIZE_M).floor();
                let index = (x + grid.0 * (y + grid.1 * z)) as usize;
                if world
                    .voxel(voxel.x as i32, voxel.y as i32, voxel.z as i32)
                    .is_solid()
                {
                    solid[index] = 1;
                }
            }
        }
    }
    dilate(&mut solid, grid);
    solid
}

/// Marks every empty cell that shares a face with a solid one.
pub fn dilate(mask: &mut [u32], grid: (u32, u32, u32)) {
    let source = mask.to_vec();
    let index = |x: u32, y: u32, z: u32| (x + grid.0 * (y + grid.1 * z)) as usize;
    for z in 0..grid.2 {
        for y in 0..grid.1 {
            for x in 0..grid.0 {
                let i = index(x, y, z);
                if source[i] != 0 {
                    continue;
                }
                let mut near = false;
                if x > 0 {
                    near |= source[index(x - 1, y, z)] != 0;
                }
                if x + 1 < grid.0 {
                    near |= source[index(x + 1, y, z)] != 0;
                }
                if y > 0 {
                    near |= source[index(x, y - 1, z)] != 0;
                }
                if y + 1 < grid.1 {
                    near |= source[index(x, y + 1, z)] != 0;
                }
                if z > 0 {
                    near |= source[index(x, y, z - 1)] != 0;
                }
                if z + 1 < grid.2 {
                    near |= source[index(x, y, z + 1)] != 0;
                }
                if near {
                    mask[i] = 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilation_spreads_one_cell_along_faces() {
        let grid = (5, 3, 5);
        let mut mask = vec![0u32; (grid.0 * grid.1 * grid.2) as usize];
        let centre = (2 + grid.0 * (1 + grid.1 * 2)) as usize;
        mask[centre] = 1;
        dilate(&mut mask, grid);
        let at = |x: u32, y: u32, z: u32| mask[(x + grid.0 * (y + grid.1 * z)) as usize];
        assert_eq!(at(2, 1, 2), 1);
        assert_eq!(at(1, 1, 2), 1);
        assert_eq!(at(2, 0, 2), 1);
        assert_eq!(at(2, 1, 3), 1);
        // The diagonal corner is not a face neighbour.
        assert_eq!(at(1, 0, 2), 0);
        assert_eq!(at(3, 1, 3), 0);
    }
}
