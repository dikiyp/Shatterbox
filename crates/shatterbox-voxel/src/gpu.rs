//! GPU upload layout for the sparse brick world.

use crate::brick::{BRICK_DIM, GPU_WORDS_PER_BRICK, MASK_WORDS_PER_BRICK};
use crate::region::REGION_BRICKS;
use crate::world::{BRICKS_X, BRICKS_Y, BRICKS_Z, VoxelWorld};

/// Data consumed by `gbuffer.wgsl` (brick indices, voxels and occupancy at
/// 2 m, 4 m, 8 m and 32 m granularity).
#[derive(Clone, Debug)]
pub struct GpuWorldData {
    pub brick_dims: [u32; 3],
    /// 0 = empty, otherwise one-based index into `bricks`.
    pub brick_indices: Vec<u32>,
    /// Packed u16 voxels, two per word.
    pub bricks: Vec<u32>,
    /// Occupancy masks, [`MASK_WORDS_PER_BRICK`] words per brick.
    pub brick_masks: Vec<u32>,
    pub brick_count: u32,
    /// Coarse occupancy, one word per 4x4x4 brick group (8 m).
    pub group_dims: [u32; 3],
    pub group_occupancy: Vec<u32>,
    /// Region occupancy (32 m), used to skip whole regions when empty.
    pub region_dims: [u32; 3],
    pub region_occupancy: Vec<u32>,
}

/// Number of bricks per axis in one coarse occupancy group.
pub const GROUP_BRICKS: u32 = 4;

impl GpuWorldData {
    pub fn from_world(world: &VoxelWorld) -> Self {
        let brick_dims = [BRICKS_X as u32, BRICKS_Y as u32, BRICKS_Z as u32];
        let mut brick_indices = vec![0u32; BRICKS_X * BRICKS_Y * BRICKS_Z];
        let mut bricks = Vec::new();
        let mut brick_masks = Vec::new();

        for bz in 0..BRICKS_Z {
            for by in 0..BRICKS_Y {
                for bx in 0..BRICKS_X {
                    let Some(brick) = world.brick(bx, by, bz) else {
                        continue;
                    };
                    if brick.is_empty() {
                        continue;
                    }
                    let id = (bricks.len() / GPU_WORDS_PER_BRICK) as u32 + 1;
                    let index = bx + BRICKS_X * (by + BRICKS_Y * bz);
                    brick_indices[index] = id;
                    brick.append_gpu_words(&mut bricks);
                    brick.append_gpu_mask_words(&mut brick_masks);
                }
            }
        }

        let brick_count = (bricks.len() / GPU_WORDS_PER_BRICK) as u32;
        if bricks.is_empty() {
            // Storage buffers cannot be empty in some backends. A zero brick is
            // never referenced by a non-zero brick index.
            bricks.resize(GPU_WORDS_PER_BRICK, 0);
            brick_masks.resize(MASK_WORDS_PER_BRICK, 0);
        }

        let group_dims = [
            brick_dims[0].div_ceil(GROUP_BRICKS),
            brick_dims[1].div_ceil(GROUP_BRICKS),
            brick_dims[2].div_ceil(GROUP_BRICKS),
        ];
        let mut group_occupancy =
            vec![0u32; (group_dims[0] * group_dims[1] * group_dims[2]) as usize];
        for gz in 0..group_dims[2] {
            for gy in 0..group_dims[1] {
                for gx in 0..group_dims[0] {
                    let mut occupied = false;
                    'group: for dz in 0..GROUP_BRICKS {
                        for dy in 0..GROUP_BRICKS {
                            for dx in 0..GROUP_BRICKS {
                                let bx = gx * GROUP_BRICKS + dx;
                                let by = gy * GROUP_BRICKS + dy;
                                let bz = gz * GROUP_BRICKS + dz;
                                if bx >= brick_dims[0] || by >= brick_dims[1] || bz >= brick_dims[2]
                                {
                                    continue;
                                }
                                let index =
                                    (bx + brick_dims[0] * (by + brick_dims[1] * bz)) as usize;
                                if brick_indices[index] != 0 {
                                    occupied = true;
                                    break 'group;
                                }
                            }
                        }
                    }
                    let index = gx + group_dims[0] * (gy + group_dims[1] * gz);
                    group_occupancy[index as usize] = u32::from(occupied);
                }
            }
        }

        let region_dims = [
            brick_dims[0].div_ceil(REGION_BRICKS as u32),
            brick_dims[1].div_ceil(REGION_BRICKS as u32),
            brick_dims[2].div_ceil(REGION_BRICKS as u32),
        ];
        let mut region_occupancy =
            vec![0u32; (region_dims[0] * region_dims[1] * region_dims[2]) as usize];
        for bz in 0..BRICKS_Z {
            for by in 0..BRICKS_Y {
                for bx in 0..BRICKS_X {
                    if brick_indices[bx + BRICKS_X * (by + BRICKS_Y * bz)] == 0 {
                        continue;
                    }
                    let rx = (bx / REGION_BRICKS) as u32;
                    let ry = (by / REGION_BRICKS) as u32;
                    let rz = (bz / REGION_BRICKS) as u32;
                    region_occupancy[(rx + region_dims[0] * (ry + region_dims[1] * rz)) as usize] =
                        1;
                }
            }
        }

        Self {
            brick_dims,
            brick_indices,
            bricks,
            brick_masks,
            brick_count,
            group_dims,
            group_occupancy,
            region_dims,
            region_occupancy,
        }
    }

    pub fn index(&self, x: usize, y: usize, z: usize) -> u32 {
        self.brick_indices[x + self.brick_dims[0] as usize * (y + self.brick_dims[1] as usize * z)]
    }

    pub fn chunks_per_axis() -> usize {
        REGION_BRICKS * BRICK_DIM / BRICK_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::materials;
    use crate::voxel::Voxel;

    #[test]
    fn empty_world_has_one_dummy_brick() {
        let data = GpuWorldData::from_world(&VoxelWorld::default());
        assert_eq!(data.brick_count, 0);
        assert_eq!(data.bricks.len(), GPU_WORDS_PER_BRICK);
        assert_eq!(data.brick_masks.len(), MASK_WORDS_PER_BRICK);
        assert!(data.brick_indices.iter().all(|value| *value == 0));
        assert!(data.group_occupancy.iter().all(|value| *value == 0));
        assert!(data.region_occupancy.iter().all(|value| *value == 0));
    }

    /// Test-only replica of `cast_ray_simple` from `gbuffer.wgsl`. It walks the
    /// same uploaded arrays with the same arithmetic so the GPU traversal can be
    /// compared against the reference CPU DDA without a GPU in the loop.
    fn cast_ray_replica(
        data: &GpuWorldData,
        origin: glam::Vec3,
        direction: glam::Vec3,
        max_distance: f32,
    ) -> Option<([i32; 3], f32)> {
        let voxel = crate::world::VOXEL_SIZE_M;
        let max_cell = [2047, 511, 2047];
        let mut distance = 1e-4f32;
        let start = origin + direction * distance;
        let mut cell = [
            ((start.x / voxel).floor() as i32).clamp(0, max_cell[0]),
            ((start.y / voxel).floor() as i32).clamp(0, max_cell[1]),
            ((start.z / voxel).floor() as i32).clamp(0, max_cell[2]),
        ];
        let step = [
            if direction.x >= 0.0 { 1 } else { -1 },
            if direction.y >= 0.0 { 1 } else { -1 },
            if direction.z >= 0.0 { 1 } else { -1 },
        ];
        let t_delta = [
            if direction.x.abs() > 1e-8 {
                voxel / direction.x.abs()
            } else {
                1e30
            },
            if direction.y.abs() > 1e-8 {
                voxel / direction.y.abs()
            } else {
                1e30
            },
            if direction.z.abs() > 1e-8 {
                voxel / direction.z.abs()
            } else {
                1e30
            },
        ];
        let mut t_max = [0.0f32; 3];
        for axis in 0..3 {
            let dir = match axis {
                0 => direction.x,
                1 => direction.y,
                _ => direction.z,
            };
            t_max[axis] = if dir.abs() < 1e-9 {
                1e30
            } else {
                let face = (cell[axis] + if step[axis] > 0 { 1 } else { 0 }) as f32 * voxel;
                ((face - origin[axis]) / dir).max(distance)
            };
        }

        for _ in 0..4096 {
            if distance > max_distance {
                break;
            }
            if cell.iter().any(|value| *value < 0) || (0..3).any(|axis| cell[axis] > max_cell[axis])
            {
                break;
            }
            let coord = [cell[0] / 16, cell[1] / 16, cell[2] / 16];
            let dims = data.brick_dims;
            let id = if coord.iter().any(|value| *value < 0)
                || (0..3).any(|axis| coord[axis] >= dims[axis] as i32)
            {
                0
            } else {
                data.index(coord[0] as usize, coord[1] as usize, coord[2] as usize)
            };
            if id != 0 {
                let local = [cell[0] & 15, cell[1] & 15, cell[2] & 15];
                let bit = local[0] + 16 * (local[1] + 16 * local[2]);
                let word = data.brick_masks
                    [(id as usize - 1) * MASK_WORDS_PER_BRICK + (bit / 32) as usize];
                if (word >> (bit % 32)) & 1 != 0 {
                    return Some((cell, distance));
                }
            }
            let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
                0
            } else if t_max[1] <= t_max[2] {
                1
            } else {
                2
            };
            distance = t_max[axis];
            cell[axis] += step[axis];
            t_max[axis] += t_delta[axis];
        }
        None
    }

    /// Test-only replica of the hierarchical `cast_ray` from `gbuffer.wgsl`,
    /// including the coarse region/group/brick jumps and the mask2/4/8 skips.
    /// Diffing it against `cast_ray_replica` and the CPU reference isolates a
    /// divergence to the coarse-skip arithmetic instead of the GPU plumbing.
    /// The second return value is the number of loop iterations, which is what
    /// `MAX_STEPS` in the shader bounds.
    fn cast_ray_hier_replica(
        data: &GpuWorldData,
        origin: glam::Vec3,
        direction: glam::Vec3,
        max_distance: f32,
    ) -> (Option<([i32; 3], f32)>, u32) {
        const MASK8_OFFSET: usize = 128;
        const MASK4_OFFSET: usize = 144;
        const MASK2_OFFSET: usize = 146;
        const BRICK_DIM: i32 = 16;
        const GROUP_VOXELS: i32 = 64;
        const REGION_VOXELS: i32 = 256;

        let voxel = crate::world::VOXEL_SIZE_M;
        let max_cell = [2047, 511, 2047];
        let ro = [origin.x, origin.y, origin.z];
        let rd = [direction.x, direction.y, direction.z];

        fn block_boundary(t_max: f32, t_delta: f32, cell: i32, step: i32, size: i32) -> f32 {
            let local = cell - (cell / size) * size;
            // Mirrors gbuffer.wgsl: backward rays leave the block after `local`
            // cells, forward rays after `size - 1 - local`.
            let remaining = if step < 0 { local } else { size - 1 - local };
            t_max + t_delta * remaining as f32
        }
        fn min_boundary3(a: f32, b: f32, c: f32) -> (f32, usize) {
            if a <= b && a <= c {
                (a, 0)
            } else if b <= c {
                (b, 1)
            } else {
                (c, 2)
            }
        }
        fn block_occupied(
            masks: &[u32],
            mask_base: usize,
            offset: usize,
            local: [i32; 3],
            size: i32,
        ) -> bool {
            let block = [local[0] / size, local[1] / size, local[2] / size];
            let dim = 16 / size;
            let bit = (block[0] + dim * (block[1] + dim * block[2])) as u32;
            let word = masks[mask_base + offset + (bit / 32) as usize];
            (word >> (bit % 32)) & 1 != 0
        }
        let occupancy_at = |occ: &[u32], dims: [u32; 3], cell: [i32; 3], size: i32| -> u32 {
            let coord = [cell[0] / size, cell[1] / size, cell[2] / size];
            if (0..3).any(|axis| coord[axis] < 0 || coord[axis] >= dims[axis] as i32) {
                return 0;
            }
            occ[(coord[0] + dims[0] as i32 * (coord[1] + dims[1] as i32 * coord[2])) as usize]
        };

        let mut distance = 1e-5f32;
        let mut cell = [0i32; 3];
        for axis in 0..3 {
            let p = ro[axis] + rd[axis] * distance;
            cell[axis] = ((p / voxel).floor() as i32).clamp(0, max_cell[axis]);
        }
        let step = [
            if rd[0] >= 0.0 { 1 } else { -1 },
            if rd[1] >= 0.0 { 1 } else { -1 },
            if rd[2] >= 0.0 { 1 } else { -1 },
        ];
        let t_delta = [
            if rd[0].abs() > 1e-8 {
                voxel / rd[0].abs()
            } else {
                1e30
            },
            if rd[1].abs() > 1e-8 {
                voxel / rd[1].abs()
            } else {
                1e30
            },
            if rd[2].abs() > 1e-8 {
                voxel / rd[2].abs()
            } else {
                1e30
            },
        ];
        let mut t_max = [0.0f32; 3];
        for axis in 0..3 {
            t_max[axis] = if rd[axis].abs() < 1e-9 {
                1e30
            } else {
                let face = (cell[axis] + if step[axis] > 0 { 1 } else { 0 }) as f32 * voxel;
                ((face - ro[axis]) / rd[axis]).max(distance)
            };
        }

        let mut brick_coord = [-1i32; 3];
        let mut id = 0u32;
        let mut mask_base = 0usize;
        let mut region_coord = [-1i32; 3];
        let mut region_empty = false;
        let mut group_coord = [-1i32; 3];
        let mut group_empty = false;
        let mut steps = 0u32;

        for _ in 0..4096 {
            steps += 1;
            if distance > max_distance {
                break;
            }
            if (0..3).any(|axis| cell[axis] < 0 || cell[axis] > max_cell[axis]) {
                break;
            }

            // Coarse empty-space skip, largest level first: 32 m region, then
            // 8 m group. Coordinates are cached and only re-read when the ray
            // enters another block of that level.
            let region = [
                cell[0] / REGION_VOXELS,
                cell[1] / REGION_VOXELS,
                cell[2] / REGION_VOXELS,
            ];
            if region != region_coord {
                region_coord = region;
                region_empty = occupancy_at(
                    &data.region_occupancy,
                    data.region_dims,
                    cell,
                    REGION_VOXELS,
                ) == 0;
            }
            let mut coarse = 0;
            if region_empty {
                coarse = REGION_VOXELS;
            } else {
                let group = [
                    cell[0] / GROUP_VOXELS,
                    cell[1] / GROUP_VOXELS,
                    cell[2] / GROUP_VOXELS,
                ];
                if group != group_coord {
                    group_coord = group;
                    group_empty =
                        occupancy_at(&data.group_occupancy, data.group_dims, cell, GROUP_VOXELS)
                            == 0;
                }
                if group_empty {
                    coarse = GROUP_VOXELS;
                }
            }
            if coarse != 0 {
                let boundary = min_boundary3(
                    block_boundary(t_max[0], t_delta[0], cell[0], step[0], coarse),
                    block_boundary(t_max[1], t_delta[1], cell[1], step[1], coarse),
                    block_boundary(t_max[2], t_delta[2], cell[2], step[2], coarse),
                );
                if boundary.0 > max_distance || boundary.0 >= 1e29 {
                    break;
                }
                let previous = cell;
                distance = boundary.0;
                let sample = glam::Vec3::new(
                    ro[0] + rd[0] * (distance + 1e-5),
                    ro[1] + rd[1] * (distance + 1e-5),
                    ro[2] + rd[2] * (distance + 1e-5),
                );
                let mut next = [
                    (sample.x / voxel).floor() as i32,
                    (sample.y / voxel).floor() as i32,
                    (sample.z / voxel).floor() as i32,
                ];
                if next[boundary.1] == cell[boundary.1] {
                    next[boundary.1] += step[boundary.1];
                }
                cell = next;
                for axis in 0..3 {
                    // Crossings form a progression in +t, so a backward cell
                    // advance still means later time: use |cell - previous|.
                    let advance = (cell[axis] - previous[axis]) * step[axis];
                    t_max[axis] = (t_max[axis] + advance as f32 * t_delta[axis]).max(distance);
                }
                continue;
            }

            let coord = [
                cell[0] / BRICK_DIM,
                cell[1] / BRICK_DIM,
                cell[2] / BRICK_DIM,
            ];
            if coord != brick_coord {
                brick_coord = coord;
                id = if (0..3)
                    .any(|axis| coord[axis] < 0 || coord[axis] >= data.brick_dims[axis] as i32)
                {
                    0
                } else {
                    data.index(coord[0] as usize, coord[1] as usize, coord[2] as usize)
                };
                mask_base = id.saturating_sub(1) as usize * MASK_WORDS_PER_BRICK;
            }

            if id == 0 {
                let boundary = min_boundary3(
                    block_boundary(t_max[0], t_delta[0], cell[0], step[0], BRICK_DIM),
                    block_boundary(t_max[1], t_delta[1], cell[1], step[1], BRICK_DIM),
                    block_boundary(t_max[2], t_delta[2], cell[2], step[2], BRICK_DIM),
                );
                if boundary.0 > max_distance || boundary.0 >= 1e29 {
                    break;
                }
                let previous = cell;
                distance = boundary.0;
                let sample = glam::Vec3::new(
                    ro[0] + rd[0] * (distance + 1e-5),
                    ro[1] + rd[1] * (distance + 1e-5),
                    ro[2] + rd[2] * (distance + 1e-5),
                );
                let mut next = [
                    (sample.x / voxel).floor() as i32,
                    (sample.y / voxel).floor() as i32,
                    (sample.z / voxel).floor() as i32,
                ];
                if next[boundary.1] == cell[boundary.1] {
                    next[boundary.1] += step[boundary.1];
                }
                cell = next;
                for axis in 0..3 {
                    let advance = (cell[axis] - previous[axis]) * step[axis];
                    t_max[axis] = (t_max[axis] + advance as f32 * t_delta[axis]).max(distance);
                }
                continue;
            }

            let local = [cell[0] & 15, cell[1] & 15, cell[2] & 15];
            let mut skip = 0;
            if !block_occupied(&data.brick_masks, mask_base, MASK2_OFFSET, local, 8) {
                skip = 8;
            } else if !block_occupied(&data.brick_masks, mask_base, MASK4_OFFSET, local, 4) {
                skip = 4;
            } else if !block_occupied(&data.brick_masks, mask_base, MASK8_OFFSET, local, 2) {
                skip = 2;
            }
            if skip != 0 {
                let boundary = min_boundary3(
                    block_boundary(t_max[0], t_delta[0], cell[0], step[0], skip),
                    block_boundary(t_max[1], t_delta[1], cell[1], step[1], skip),
                    block_boundary(t_max[2], t_delta[2], cell[2], step[2], skip),
                );
                if boundary.0 > max_distance || boundary.0 >= 1e29 {
                    break;
                }
                let previous = cell;
                distance = boundary.0;
                let sample = glam::Vec3::new(
                    ro[0] + rd[0] * (distance + 1e-5),
                    ro[1] + rd[1] * (distance + 1e-5),
                    ro[2] + rd[2] * (distance + 1e-5),
                );
                let mut next = [
                    (sample.x / voxel).floor() as i32,
                    (sample.y / voxel).floor() as i32,
                    (sample.z / voxel).floor() as i32,
                ];
                if next[boundary.1] == cell[boundary.1] {
                    next[boundary.1] += step[boundary.1];
                }
                cell = next;
                for axis in 0..3 {
                    let advance = (cell[axis] - previous[axis]) * step[axis];
                    t_max[axis] = (t_max[axis] + advance as f32 * t_delta[axis]).max(distance);
                }
                continue;
            }

            let bit = (local[0] + BRICK_DIM * (local[1] + BRICK_DIM * local[2])) as u32;
            let word = data.brick_masks[mask_base + (bit / 32) as usize];
            if (word >> (bit % 32)) & 1 != 0 {
                return (Some((cell, distance)), steps);
            }

            let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
                0
            } else if t_max[1] <= t_max[2] {
                1
            } else {
                2
            };
            distance = t_max[axis];
            cell[axis] += step[axis];
            t_max[axis] += t_delta[axis];
        }
        (None, steps)
    }

    /// Camera-like rays travel in the negative direction on at least one axis,
    /// which is where the hierarchical traversal used to diverge: the coarse
    /// occupancy bit for a partially carved block was cleared as soon as one
    /// voxel was removed, so the skip jumped through solid geometry and
    /// reported a hit far inside the surface. The pixel then shaded from a
    /// point *below* the floor, so its shadow ray started inside geometry and
    /// the floor turned into per-voxel shadow speckle.
    #[test]
    fn descending_traversal_matches_reference_in_cell_and_distance() {
        use crate::scene::build_test_arena;
        use glam::Vec3;

        let world = build_test_arena();
        let data = GpuWorldData::from_world(&world);
        let voxel = crate::world::VOXEL_SIZE_M;
        let mut mismatches = Vec::new();
        let mut exact_cells = 0;
        let mut tested = 0;
        // Off-grid directions: exact axis ratios would make the ray pass
        // through voxel corners, where either touching voxel is a valid answer
        // and comparing cells becomes meaningless.
        for origin_y in [3.0f32, 10.0, 24.0] {
            for dx in [-0.71f32, 0.0, 0.68] {
                for dz in [-0.913f32, -0.451, 0.437, 0.887] {
                    for dy in [-0.061f32, -0.203, -0.497, -1.19] {
                        let direction = Vec3::new(dx, dy, dz).normalize();
                        let origin = Vec3::new(128.0, origin_y, 120.0);
                        let reference = world
                            .raycast(origin, direction, 192.0)
                            .map(|hit| (hit.position, hit.distance));
                        let simple = cast_ray_replica(&data, origin, direction, 192.0);
                        let (hier, _steps) = cast_ray_hier_replica(&data, origin, direction, 192.0);
                        tested += 1;
                        let ok = match (reference, simple, hier) {
                            (Some((position, rt)), Some((_, st)), Some((hc, ht))) => {
                                let expected = position.to_array();
                                if expected == hc {
                                    exact_cells += 1;
                                }
                                // Distances must match; the reported cell must be
                                // solid and must actually contain the hit point.
                                // Rays grazing a cell boundary may legitimately
                                // stop in either of the two touching voxels.
                                let point = origin + direction * ht;
                                let t_ok = (st - ht).abs() < 1e-2 && (rt - ht).abs() < 2e-2;
                                let solid = world.voxel(hc[0], hc[1], hc[2]).is_solid();
                                let contains = (0..3).all(|axis| {
                                    let lo = hc[axis] as f32 * voxel;
                                    point[axis] >= lo - 1e-3 && point[axis] <= lo + voxel + 1e-3
                                });
                                if !(t_ok && solid && contains) {
                                    mismatches.push((direction, expected, hc, ht, rt, st));
                                }
                                t_ok && solid && contains
                            }
                            (None, None, None) => true,
                            _ => {
                                // Encode which paths hit: [reference, simple, hier].
                                mismatches.push((
                                    direction,
                                    [
                                        i32::from(reference.is_some()),
                                        i32::from(simple.is_some()),
                                        i32::from(hier.is_some()),
                                    ],
                                    hier.map_or([0; 3], |h| h.0),
                                    hier.map_or(-1.0, |h| h.1),
                                    reference.map_or(-1.0, |r| r.1),
                                    simple.map_or(-1.0, |s| s.1),
                                ));
                                false
                            }
                        };
                        assert!(ok || !mismatches.is_empty());
                    }
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "{} of {} rays disagree, first 4: {:#?}",
            mismatches.len(),
            tested,
            &mismatches[..mismatches.len().min(4)]
        );
        // Guard against a traversal that merely hits *something* solid nearby:
        // most rays must still land in exactly the voxel the reference picks.
        assert!(
            exact_cells * 2 > tested,
            "only {exact_cells} of {tested} hits landed in the reference voxel"
        );
    }

    #[test]
    fn hierarchical_traversal_matches_simple() {
        use crate::scene::build_test_arena;
        use glam::Vec3;

        let world = build_test_arena();
        let data = GpuWorldData::from_world(&world);
        let sun = Vec3::new(0.52, 0.46, 0.30).normalize();
        let mut mismatches = Vec::new();
        for x in (70..190).step_by(5) {
            for z in (70..190).step_by(5) {
                // Shader shadow origin: floor top face + 0.2 m along the sun.
                let origin = Vec3::new(x as f32, 1.0, z as f32) + sun * 0.2;
                let simple = cast_ray_replica(&data, origin, sun, 64.0);
                let (hier, _steps) = cast_ray_hier_replica(&data, origin, sun, 64.0);
                let reference = world
                    .raycast(origin, sun, 64.0)
                    .map(|hit| (hit.position, hit.distance));
                if simple.is_some() != hier.is_some() || reference.is_some() != hier.is_some() {
                    mismatches.push((x, z, simple, hier, reference));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "{} mismatches, first 5: {:#?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(5)]
        );
    }

    #[test]
    fn gpu_traversal_matches_cpu_reference() {
        use crate::scene::build_test_arena;
        use glam::Vec3;

        let world = build_test_arena();
        let data = GpuWorldData::from_world(&world);
        let sun = Vec3::new(0.52, 0.46, 0.30).normalize();

        // Sweep the whole floor: reference and replica must always agree.
        let mut clear = 0;
        for x in (70..190).step_by(7) {
            for z in (70..190).step_by(7) {
                let origin = Vec3::new(x as f32, 1.19, z as f32);
                let reference = world.raycast(origin, sun, 48.0);
                let replica = cast_ray_replica(&data, origin, sun, 48.0);
                match (reference, replica) {
                    (None, None) => clear += 1,
                    (Some(hit), Some((cell, distance))) => {
                        assert_eq!(
                            cell,
                            [hit.position.x, hit.position.y, hit.position.z],
                            "floor sweep origin {origin:?}"
                        );
                        assert!(
                            (hit.distance - distance).abs() < 0.4,
                            "floor sweep origin {origin:?}: {:.3} vs {:.3}",
                            hit.distance,
                            distance
                        );
                    }
                    (reference, replica) => panic!(
                        "floor sweep origin {origin:?}: reference={:?} replica={:?}",
                        reference.map(|hit| (hit.position, hit.distance)),
                        replica
                    ),
                }
            }
        }
        // Most of the open floor away from the crates must stay unshadowed.
        assert!(clear > 200, "only {clear} clear floor samples");
        // A known-open probe: the replica must agree there too.
        let probe = Vec3::new(128.0, 1.19, 180.0);
        assert!(world.raycast(probe, sun, 48.0).is_none());
        assert!(cast_ray_replica(&data, probe, sun, 48.0).is_none());

        // Random rays from camera-like positions must agree on hit position.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state as f32 / u32::MAX as f32
        };
        for _ in 0..512 {
            let origin = Vec3::new(
                64.0 + next() * 128.0,
                2.0 + next() * 12.0,
                64.0 + next() * 128.0,
            );
            let target = Vec3::new(64.0 + next() * 128.0, next() * 8.0, 64.0 + next() * 128.0);
            let direction = (target - origin).normalize();
            let reference = world.raycast(origin, direction, 192.0);
            let replica = cast_ray_replica(&data, origin, direction, 192.0);
            match (reference, replica) {
                (None, None) => {}
                (Some(hit), Some((cell, distance))) => {
                    assert!(
                        (hit.distance - distance).abs() < 0.4,
                        "origin {origin:?} dir {direction:?}: distance {:.3} vs {:.3}",
                        hit.distance,
                        distance
                    );
                    let expected = [hit.position.x, hit.position.y, hit.position.z];
                    assert_eq!(cell, expected, "origin {origin:?} dir {direction:?}");
                }
                (reference, replica) => panic!(
                    "origin {origin:?} dir {direction:?}: reference={:?} replica={:?}",
                    reference.map(|hit| (hit.position, hit.distance)),
                    replica
                ),
            }
        }
    }

    #[test]
    fn bricks_are_indexed_one_based() {
        let mut world = VoxelWorld::default();
        world.set_voxel(0, 0, 0, Voxel::new(materials::STONE, 0));
        world.set_voxel(40, 0, 0, Voxel::new(materials::STONE, 0));
        let data = GpuWorldData::from_world(&world);
        assert_eq!(data.brick_count, 2);
        assert_eq!(data.index(0, 0, 0), 1);
        assert_eq!(data.index(2, 0, 0), 2);
        assert_eq!(data.bricks.len(), GPU_WORDS_PER_BRICK * 2);
        assert_eq!(data.brick_masks.len(), MASK_WORDS_PER_BRICK * 2);
        // The first voxel of brick 0 is at bit 0 of mask16.
        assert_eq!(data.brick_masks[0] & 1, 1);
        // The second brick starts at word MASK_WORDS_PER_BRICK; its voxel is
        // at x = 8 inside the brick (40 = 2 * 16 + 8).
        let second = MASK_WORDS_PER_BRICK;
        assert_eq!(data.brick_masks[second] & (1 << 8), 1 << 8);
        assert_eq!(data.group_dims, [32, 8, 32]);
        assert_eq!(data.group_occupancy.len(), 32 * 8 * 32);
        assert_eq!(data.group_occupancy[0], 1);
        assert_eq!(data.group_occupancy[1], 0);
        assert_eq!(data.region_dims, [8, 2, 8]);
        assert_eq!(data.region_occupancy.len(), 8 * 2 * 8);
        assert_eq!(data.region_occupancy[0], 1);
    }

    /// `MAX_STEPS` in the shader bounds the traversal. A grazing primary ray
    /// hugs a surface and advances voxel by voxel, which is the worst case: the
    /// budget must still be large enough to reach the 192 m far plane.
    #[test]
    fn grazing_rays_fit_the_shader_step_budget() {
        use crate::scene::build_test_arena;
        use glam::Vec3;

        let world = build_test_arena();
        let data = GpuWorldData::from_world(&world);
        let mut max_steps = 0;
        let mut worst = (Vec3::ZERO, Vec3::ZERO);
        for yaw_step in 0..96 {
            // Off-grid yaw and pitch: an exactly axis-aligned ray that starts on
            // a voxel corner plane can oscillate between two cells in the DDA
            // (a pre-existing trait of the traversal), which says nothing about
            // the step budget of real camera rays.
            let yaw = (yaw_step as f32 + 0.37) * std::f32::consts::TAU / 96.0;
            for pitch in [-0.113f32, -0.041, -0.0091, 0.0063, 0.037, 0.11] {
                let direction = Vec3::new(
                    yaw.cos() * pitch.cos(),
                    pitch.sin(),
                    yaw.sin() * pitch.cos(),
                );
                for origin in [
                    Vec3::new(128.3137, 1.5931, 128.271),
                    Vec3::new(70.317, 2.213, 190.113),
                    Vec3::new(190.121, 9.037, 70.319),
                    Vec3::new(128.077, 24.041, 127.931),
                ] {
                    let (_, steps) = cast_ray_hier_replica(&data, origin, direction, 192.0);
                    if steps > max_steps {
                        max_steps = steps;
                        worst = (origin, direction);
                    }
                }
            }
        }
        eprintln!("max steps {max_steps} at {worst:?}");
        assert!(
            max_steps <= 512,
            "grazing ray {worst:?} needs {max_steps} steps"
        );
    }
}
