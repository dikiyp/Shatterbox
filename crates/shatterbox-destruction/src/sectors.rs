//! Lazy 8 m static collision sectors.
//!
//! The world is a voxel brick map, but Jolt only sees boxes. Rebuilding a
//! collider for the whole arena on every shot would stall the frame, so the
//! world is split into 8 m sectors (64³ voxels). A sector is snapshotted,
//! greedily merged into boxes on a worker thread, and applied when the result
//! comes back. Sectors are created around the player and active debris and
//! dropped when nothing is near them.

use std::collections::HashSet;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use glam::{IVec3, Vec3};
use shatterbox_physics::{CompoundBox, PhysicsBackend};
use shatterbox_voxel::BRICK_DIM;
use shatterbox_voxel::world::{VOXEL_SIZE_M, VoxelWorld};

/// Sector edge in metres.
pub const SECTOR_SIZE_M: f32 = 8.0;
/// Sector edge in voxels (12.5 cm).
pub const SECTOR_VOXELS: i32 = 64;
/// How many sectors around the player are kept resident (Chebyshev radius 1 =
/// 24 m across). Debris loads only the sector it occupies.
const PLAYER_RADIUS: i32 = 1;
/// Debris further than this from the player does not create sectors.
const ACTIVE_DISTANCE_M: f32 = 32.0;
/// Rebuilds in flight. More than this just queues work that is already stale.
const MAX_IN_FLIGHT: usize = 4;

struct Job {
    key: IVec3,
    bits: Vec<u64>,
}

/// Owns the worker threads and the sector state.
pub struct SectorCollision {
    /// Sectors whose boxes are applied to the physics backend.
    resident: HashSet<IVec3>,
    /// A rebuild is queued or running.
    pending: HashSet<IVec3>,
    /// Edited after the snapshot was taken; rebuild again when it lands.
    dirty: HashSet<IVec3>,
    /// Sectors that were ever edited. The arena has at most 8x2x8 sectors, so
    /// keeping these resident is bounded and means a distant crater still has
    /// colliders even when no body is near it.
    edited: HashSet<IVec3>,
    jobs: Option<Sender<Job>>,
    results: Receiver<(IVec3, Vec<CompoundBox>)>,
    workers: Vec<JoinHandle<()>>,
}

impl Default for SectorCollision {
    fn default() -> Self {
        Self::new()
    }
}

impl SectorCollision {
    pub fn new() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let (result_tx, result_rx) = mpsc::channel();
        let workers = (0..2)
            .map(|index| {
                let jobs = Arc::clone(&job_rx);
                let results = result_tx.clone();
                thread::Builder::new()
                    .name(format!("shatterbox-sector-{index}"))
                    .spawn(move || {
                        loop {
                            let job = match jobs.lock() {
                                Ok(jobs) => match jobs.recv() {
                                    Ok(job) => job,
                                    Err(_) => break,
                                },
                                Err(_) => break,
                            };
                            let boxes = greedy_boxes(job.key, &job.bits);
                            if results.send((job.key, boxes)).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("spawn sector worker")
            })
            .collect();
        Self {
            resident: HashSet::new(),
            pending: HashSet::new(),
            dirty: HashSet::new(),
            edited: HashSet::new(),
            jobs: Some(job_tx),
            results: result_rx,
            workers,
        }
    }

    pub fn resident_sectors(&self) -> usize {
        self.resident.len()
    }

    pub fn pending_sectors(&self) -> usize {
        self.pending.len()
    }

    /// Marks the sectors touched by edited bricks. Resident sectors are rebuilt
    /// immediately; others are marked and will be rebuilt if they become
    /// resident later.
    pub fn mark_dirty_bricks(&mut self, bricks: &[IVec3]) {
        for &brick in bricks {
            let min = brick * BRICK_DIM as i32;
            let max = min + BRICK_DIM as i32 - 1;
            let lo = min.div_euclid(IVec3::splat(SECTOR_VOXELS));
            let hi = max.div_euclid(IVec3::splat(SECTOR_VOXELS));
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let key = IVec3::new(x, y, z);
                        self.edited.insert(key);
                        if self.resident.contains(&key) {
                            self.dirty.insert(key);
                        }
                    }
                }
            }
        }
    }

    /// Applies finished rebuilds and starts new ones. Call once per frame.
    pub fn update<B: PhysicsBackend>(
        &mut self,
        world: &VoxelWorld,
        player: Vec3,
        bodies: &[Vec3],
        physics: &mut B,
    ) {
        while let Ok((key, boxes)) = self.results.try_recv() {
            self.pending.remove(&key);
            if !self.resident.contains(&key) {
                continue;
            }
            physics.replace_static_sector(key, &boxes);
            if self.dirty.remove(&key) {
                self.schedule(world, key);
            }
        }

        let mut desired: Vec<IVec3> = self.desired_sectors(player, bodies).into_iter().collect();
        // Stable order keeps the first sectors that load identical between runs.
        desired.sort_by_key(|key| (key.z, key.y, key.x));
        let stale: Vec<IVec3> = self
            .resident
            .iter()
            .copied()
            .filter(|key| !desired.contains(key))
            .collect();
        for key in stale {
            self.resident.remove(&key);
            self.dirty.remove(&key);
            physics.remove_static_sector(key);
        }

        // New sectors are snapshotted at most two per frame: copying 64³
        // voxels is the only main-thread part of the pipeline.
        let mut started = 0usize;
        for key in desired {
            if self.resident.contains(&key) {
                continue;
            }
            self.resident.insert(key);
            if started < 2 {
                self.schedule(world, key);
                started += 1;
            } else {
                self.dirty.insert(key);
            }
        }
        // Resident sectors edited while no job was running (or whose previous
        // job was dropped) get another chance once capacity frees up.
        if self.pending.len() < MAX_IN_FLIGHT {
            let mut ready: Vec<IVec3> = self
                .dirty
                .iter()
                .copied()
                .filter(|key| self.resident.contains(key))
                .collect();
            ready.sort_by_key(|key| (key.z, key.y, key.x));
            ready.truncate(MAX_IN_FLIGHT - self.pending.len());
            for key in ready {
                self.dirty.remove(&key);
                self.schedule(world, key);
            }
        }
    }

    fn desired_sectors(&self, player: Vec3, bodies: &[Vec3]) -> HashSet<IVec3> {
        let mut desired = HashSet::new();
        let player_key = sector_of_world(player);
        for z in -PLAYER_RADIUS..=PLAYER_RADIUS {
            for y in -PLAYER_RADIUS..=PLAYER_RADIUS {
                for x in -PLAYER_RADIUS..=PLAYER_RADIUS {
                    desired.insert(player_key + IVec3::new(x, y, z));
                }
            }
        }
        for &body in bodies {
            if body.distance_squared(player) <= ACTIVE_DISTANCE_M * ACTIVE_DISTANCE_M {
                desired.insert(sector_of_world(body));
            }
        }
        desired.extend(self.edited.iter().copied());
        desired
    }

    fn schedule(&mut self, world: &VoxelWorld, key: IVec3) {
        if self.pending.len() >= MAX_IN_FLIGHT || self.pending.contains(&key) {
            self.dirty.insert(key);
            return;
        }
        let Some(bits) = snapshot(world, key) else {
            return;
        };
        let Some(jobs) = &self.jobs else {
            return;
        };
        if jobs.send(Job { key, bits }).is_ok() {
            self.pending.insert(key);
        }
    }
}

impl Drop for SectorCollision {
    fn drop(&mut self) {
        // Closing the job channel makes both workers exit after their current
        // job; joining keeps the game from tearing down while a thread still
        // touches its own snapshot (nothing is borrowed from the game there).
        self.jobs = None;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// World position to sector key, using floor so negative coordinates also map.
pub fn sector_of_world(position: Vec3) -> IVec3 {
    IVec3::new(
        (position.x / SECTOR_SIZE_M).floor() as i32,
        (position.y / SECTOR_SIZE_M).floor() as i32,
        (position.z / SECTOR_SIZE_M).floor() as i32,
    )
}

fn snapshot(world: &VoxelWorld, key: IVec3) -> Option<Vec<u64>> {
    let origin = key * SECTOR_VOXELS;
    let mut bits = vec![0u64; (SECTOR_VOXELS * SECTOR_VOXELS * SECTOR_VOXELS / 64) as usize];
    let mut any = false;
    for z in 0..SECTOR_VOXELS {
        for y in 0..SECTOR_VOXELS {
            for x in 0..SECTOR_VOXELS {
                if world
                    .voxel(origin.x + x, origin.y + y, origin.z + z)
                    .is_solid()
                {
                    let index = (x + SECTOR_VOXELS * (y + SECTOR_VOXELS * z)) as usize;
                    bits[index / 64] |= 1u64 << (index % 64);
                    any = true;
                }
            }
        }
    }
    any.then_some(bits)
}

/// Greedy 3D merge over one sector. `key` places the boxes in world space.
fn greedy_boxes(key: IVec3, bits: &[u64]) -> Vec<CompoundBox> {
    const N: usize = SECTOR_VOXELS as usize;
    if bits.len() < N * N * N / 64 {
        return Vec::new();
    }
    let solid = |x: usize, y: usize, z: usize| {
        let index = x + N * (y + N * z);
        bits[index / 64] & (1u64 << (index % 64)) != 0
    };
    let cell = |x: usize, y: usize, z: usize| x + N * (y + N * z);
    let mut used = vec![false; N * N * N];
    let mut boxes = Vec::new();
    let origin = key * SECTOR_VOXELS;

    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                if !solid(x, y, z) || used[cell(x, y, z)] {
                    continue;
                }
                let mut w = 1;
                while x + w < N && solid(x + w, y, z) && !used[cell(x + w, y, z)] {
                    w += 1;
                }
                let mut d = 1;
                'depth: while z + d < N {
                    for xx in x..x + w {
                        if !solid(xx, y, z + d) || used[cell(xx, y, z + d)] {
                            break 'depth;
                        }
                    }
                    d += 1;
                }
                let mut h = 1;
                'height: while y + h < N {
                    for zz in z..z + d {
                        for xx in x..x + w {
                            if !solid(xx, y + h, zz) || used[cell(xx, y + h, zz)] {
                                break 'height;
                            }
                        }
                    }
                    h += 1;
                }
                for yy in y..y + h {
                    for zz in z..z + d {
                        for xx in x..x + w {
                            used[cell(xx, yy, zz)] = true;
                        }
                    }
                }
                let local = Vec3::new(
                    x as f32 + w as f32 * 0.5,
                    y as f32 + h as f32 * 0.5,
                    z as f32 + d as f32 * 0.5,
                );
                let center = (origin.as_vec3() + local) * VOXEL_SIZE_M;
                let half = Vec3::new(w as f32, h as f32, d as f32) * (VOXEL_SIZE_M * 0.5);
                boxes.push(CompoundBox::new(center, half));
            }
        }
    }
    boxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_merges_a_full_sector_into_one_box() {
        let mut bits = vec![0u64; 64 * 64 * 64 / 64];
        bits.fill(u64::MAX);
        let boxes = greedy_boxes(IVec3::ZERO, &bits);
        assert_eq!(boxes.len(), 1);
        assert!((boxes[0].half_extents - Vec3::splat(4.0)).length() < 1e-5);
        assert!((boxes[0].center - Vec3::splat(4.0)).length() < 1e-5);
    }

    #[test]
    fn greedy_box_is_in_sector_world_space() {
        let mut bits = vec![0u64; 64 * 64 * 64 / 64];
        // One voxel at (0,0,0) of sector (1,0,0).
        bits[0] = 1;
        let boxes = greedy_boxes(IVec3::new(1, 0, 0), &bits);
        assert_eq!(boxes.len(), 1);
        assert!((boxes[0].center - Vec3::new(8.0625, 0.0625, 0.0625)).length() < 1e-5);
        assert!((boxes[0].half_extents - Vec3::splat(VOXEL_SIZE_M * 0.5)).length() < 1e-5);
    }

    #[test]
    fn sparse_voxels_do_not_get_a_box_over_the_air() {
        let mut bits = vec![0u64; 64 * 64 * 64 / 64];
        bits[0] = 1 | (1 << 2);
        let boxes = greedy_boxes(IVec3::ZERO, &bits);
        assert_eq!(boxes.len(), 2);
    }

    #[test]
    fn bricks_map_to_their_sector() {
        // Brick 4 is 8 m, so it starts exactly on the sector boundary.
        assert_eq!(sector_of_world(Vec3::new(7.99, 0.0, 0.0)), IVec3::ZERO);
        assert_eq!(sector_of_world(Vec3::new(8.0, 0.0, 0.0)), IVec3::X);
    }
}
