//! Voxel storage, palette, sparse brick world and CPU DDA reference.

pub mod brick;
pub mod dda;
pub mod edit;
pub mod gpu;
pub mod palette;
pub mod region;
pub mod scene;
pub mod voxel;
pub mod world;

pub use brick::{
    BRICK_BYTES, BRICK_DIM, BRICK_VOXELS, Brick, GPU_WORDS_PER_BRICK, MASK_WORDS_PER_BRICK,
};
pub use dda::{RayHit, raycast};
pub use edit::{EditOp, EditReport, RemovedVoxel, SphereBrush, VoxelCommand, apply_command};
pub use gpu::GpuWorldData;
pub use palette::{Material, Palette, materials};
pub use region::{REGION_BRICKS, REGION_VOXELS, Region};
pub use voxel::Voxel;
pub use world::{
    BRICKS_X, BRICKS_Y, BRICKS_Z, REGION_COUNT, REGION_SIZE_M, REGIONS_X, REGIONS_Y, REGIONS_Z,
    VOXEL_SIZE_M, VOXELS_X, VOXELS_Y, VOXELS_Z, VoxelWorld, WORLD_MAX, WORLD_MIN,
};
