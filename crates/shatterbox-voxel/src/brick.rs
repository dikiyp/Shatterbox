//! 16³ brick storage with occupancy masks at 16³, 8³, 4³ and 2³.

use crate::voxel::Voxel;

pub const BRICK_DIM: usize = 16;
pub const BRICK_VOXELS: usize = BRICK_DIM * BRICK_DIM * BRICK_DIM;
pub const BRICK_BYTES: usize = BRICK_VOXELS * size_of::<u16>();
pub const GPU_WORDS_PER_BRICK: usize = BRICK_VOXELS / 2;

/// GPU mask layout for one brick, in `u32` words:
/// `0..128` mask16, `128..144` mask8, `144..146` mask4, `146` mask2.
pub const MASK_WORDS_PER_BRICK: usize = 128 + 16 + 2 + 1;

/// One 2 m brick. `voxels` is the authoritative data; masks are an acceleration
/// structure for the CPU and GPU traversal.
#[derive(Clone)]
pub struct Brick {
    voxels: Box<[u16; BRICK_VOXELS]>,
    mask16: [u64; 64],
    mask8: [u64; 8],
    mask4: u64,
    mask2: u8,
    solid_count: u32,
}

impl Default for Brick {
    fn default() -> Self {
        Self {
            voxels: Box::new([0; BRICK_VOXELS]),
            mask16: [0; 64],
            mask8: [0; 8],
            mask4: 0,
            mask2: 0,
            solid_count: 0,
        }
    }
}

impl Brick {
    /// Linear voxel index with X as the fastest axis.
    pub const fn voxel_index(x: usize, y: usize, z: usize) -> usize {
        x + BRICK_DIM * (y + BRICK_DIM * z)
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        Voxel::from_raw(self.voxels[Self::voxel_index(x, y, z)])
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        let index = Self::voxel_index(x, y, z);
        let was_solid = (self.voxels[index] & Voxel::MATERIAL_MASK) != 0;
        let is_solid = voxel.is_solid();
        self.voxels[index] = voxel.raw();

        if was_solid == is_solid {
            return;
        }

        if is_solid {
            self.set_mask_bit(x, y, z, true);
            self.solid_count += 1;
        } else {
            self.set_mask_bit(x, y, z, false);
            self.solid_count -= 1;
        }
    }

    /// Fills the whole brick with one value. Much faster than 4096 `set` calls
    /// for large floors and walls.
    pub fn fill(&mut self, voxel: Voxel) {
        if voxel.is_air() {
            self.voxels.fill(0);
            self.mask16 = [0; 64];
            self.mask8 = [0; 8];
            self.mask4 = 0;
            self.mask2 = 0;
            self.solid_count = 0;
            return;
        }

        self.voxels.fill(voxel.raw());
        self.mask16 = [u64::MAX; 64];
        self.mask8 = [u64::MAX; 8];
        self.mask4 = u64::MAX;
        self.mask2 = u8::MAX;
        self.solid_count = BRICK_VOXELS as u32;
    }

    pub fn is_empty(&self) -> bool {
        self.solid_count == 0
    }

    pub fn solid_count(&self) -> u32 {
        self.solid_count
    }

    pub fn mask16(&self) -> &[u64; 64] {
        &self.mask16
    }

    pub fn mask8(&self) -> &[u64; 8] {
        &self.mask8
    }

    pub fn mask4(&self) -> u64 {
        self.mask4
    }

    pub fn mask2(&self) -> u8 {
        self.mask2
    }

    pub fn raw_voxels(&self) -> &[u16; BRICK_VOXELS] {
        &self.voxels
    }

    /// Packs two u16 voxels per u32, low half first, for the GPU storage buffer.
    pub fn append_gpu_words(&self, out: &mut Vec<u32>) {
        out.reserve(GPU_WORDS_PER_BRICK);
        for pair in self.voxels.chunks_exact(2) {
            out.push(pair[0] as u32 | ((pair[1] as u32) << 16));
        }
    }

    /// Appends the occupancy masks in the layout documented by
    /// [`MASK_WORDS_PER_BRICK`]. Each `u64` is split into low/high `u32`.
    pub fn append_gpu_mask_words(&self, out: &mut Vec<u32>) {
        out.reserve(MASK_WORDS_PER_BRICK);
        for word in &self.mask16 {
            out.push(*word as u32);
            out.push((*word >> 32) as u32);
        }
        for word in &self.mask8 {
            out.push(*word as u32);
            out.push((*word >> 32) as u32);
        }
        out.push(self.mask4 as u32);
        out.push((self.mask4 >> 32) as u32);
        out.push(u32::from(self.mask2));
    }

    /// Updates the occupancy pyramid for one voxel. Setting a bit is monotonic,
    /// but clearing one must not clear the whole coarse block: the coarse masks
    /// are ORs over the level below, and the DDA skips any block whose bit is
    /// zero. Clearing blindly used to hide the remaining solid voxels in an
    /// 8/4/2-voxel block (e.g. carving tile seams into a solid floor), which made
    /// rays skip through geometry. Each coarse bit is therefore recomputed from
    /// its eight children when a voxel is removed.
    fn set_mask_bit(&mut self, x: usize, y: usize, z: usize, solid: bool) {
        let index16 = x + BRICK_DIM * (y + BRICK_DIM * z);
        set_bit(&mut self.mask16, index16, solid);

        let index8 = (x / 2) + 8 * ((y / 2) + 8 * (z / 2));
        if solid {
            set_bit(&mut self.mask8, index8, true);
        } else {
            let (bx, by, bz) = (x / 2 * 2, y / 2 * 2, z / 2 * 2);
            let mut occupied = false;
            for dz in 0..2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bit = (bx + dx) + BRICK_DIM * ((by + dy) + BRICK_DIM * (bz + dz));
                        occupied |= get_bit(&self.mask16, bit);
                    }
                }
            }
            set_bit(&mut self.mask8, index8, occupied);
        }

        let index4 = (x / 4) + 4 * ((y / 4) + 4 * (z / 4));
        if solid {
            self.mask4 |= 1u64 << index4;
        } else {
            let (bx, by, bz) = (x / 4 * 2, y / 4 * 2, z / 4 * 2);
            let mut occupied = false;
            for dz in 0..2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bit = (bx + dx) + 8 * ((by + dy) + 8 * (bz + dz));
                        occupied |= get_bit(&self.mask8, bit);
                    }
                }
            }
            if occupied {
                self.mask4 |= 1u64 << index4;
            } else {
                self.mask4 &= !(1u64 << index4);
            }
        }

        let index2 = (x / 8) + 2 * ((y / 8) + 2 * (z / 8));
        if solid {
            self.mask2 |= 1u8 << index2;
        } else {
            let (bx, by, bz) = (x / 8 * 2, y / 8 * 2, z / 8 * 2);
            let mut occupied = false;
            for dz in 0..2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bit = (bx + dx) + 4 * ((by + dy) + 4 * (bz + dz));
                        occupied |= (self.mask4 >> bit) & 1 != 0;
                    }
                }
            }
            if occupied {
                self.mask2 |= 1u8 << index2;
            } else {
                self.mask2 &= !(1u8 << index2);
            }
        }
    }
}

fn get_bit(words: &[u64], bit: usize) -> bool {
    (words[bit / 64] >> (bit % 64)) & 1 != 0
}

fn set_bit(words: &mut [u64], bit: usize, value: bool) {
    let word = &mut words[bit / 64];
    let mask = 1u64 << (bit % 64);
    if value {
        *word |= mask;
    } else {
        *word &= !mask;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::materials;

    #[test]
    fn set_updates_all_masks() {
        let mut brick = Brick::default();
        assert!(brick.is_empty());
        brick.set(3, 5, 7, Voxel::new(materials::STONE, 1));
        assert_eq!(brick.solid_count(), 1);
        assert_eq!(brick.get(3, 5, 7).material(), materials::STONE);
        let index16 = 3 + BRICK_DIM * (5 + BRICK_DIM * 7);
        let index8 = (3 / 2) + 8 * ((5 / 2) + 8 * (7 / 2));
        assert_eq!(
            brick.mask16()[index16 / 64] & (1 << (index16 % 64)),
            1 << (index16 % 64)
        );
        assert_eq!(
            brick.mask8()[index8 / 64] & (1 << (index8 % 64)),
            1 << (index8 % 64)
        );
        assert_eq!(brick.mask4() & (1 << 20), 1 << 20);
        assert_eq!(brick.mask2() & 1, 1);

        brick.set(3, 5, 7, Voxel::AIR);
        assert!(brick.is_empty());
        assert_eq!(brick.mask16()[0], 0);
        assert_eq!(brick.mask8()[0], 0);
        assert_eq!(brick.mask4(), 0);
        assert_eq!(brick.mask2(), 0);
    }

    #[test]
    fn fill_sets_every_voxel_and_mask() {
        let mut brick = Brick::default();
        brick.fill(Voxel::new(materials::CONCRETE, 0));
        assert_eq!(brick.solid_count(), BRICK_VOXELS as u32);
        assert_eq!(brick.mask16(), &[u64::MAX; 64]);
        assert_eq!(brick.mask8(), &[u64::MAX; 8]);
        assert_eq!(brick.mask4(), u64::MAX);
        assert_eq!(brick.mask2(), u8::MAX);
        assert!(brick.get(15, 15, 15).is_solid());
    }

    #[test]
    fn removing_voxels_keeps_coarse_occupancy_until_block_is_empty() {
        let mut brick = Brick::default();
        brick.fill(Voxel::new(materials::CONCRETE, 0));
        let (x, y, z) = (9, 7, 1);
        let bit2 = (x / 8) + 2 * ((y / 8) + 2 * (z / 8));
        let bit4 = (x / 4) + 4 * ((y / 4) + 4 * (z / 4));
        let bit8 = (x / 2) + 8 * ((y / 2) + 8 * (z / 2));

        brick.set(x, y, z, Voxel::AIR);
        assert_eq!(brick.mask2() & (1 << bit2), 1 << bit2);
        assert_eq!(brick.mask4() & (1 << bit4), 1 << bit4);
        assert_eq!(
            brick.mask8()[bit8 / 64] & (1 << (bit8 % 64)),
            1 << (bit8 % 64)
        );

        // Emptying the whole 2-voxel block clears only the mask8 bit.
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    brick.set(x / 2 * 2 + dx, y / 2 * 2 + dy, z / 2 * 2 + dz, Voxel::AIR);
                }
            }
        }
        assert_eq!(brick.mask8()[bit8 / 64] & (1 << (bit8 % 64)), 0);
        assert_eq!(brick.mask4() & (1 << bit4), 1 << bit4);
        assert_eq!(brick.mask2() & (1 << bit2), 1 << bit2);

        brick.fill(Voxel::AIR);
        assert_eq!(brick.mask2(), 0);
        assert_eq!(brick.mask4(), 0);
        assert_eq!(brick.mask8(), &[0u64; 8]);
        assert_eq!(brick.mask16(), &[0u64; 64]);
    }

    #[test]
    fn gpu_words_pack_little_endian_pairs() {
        let mut brick = Brick::default();
        brick.set(0, 0, 0, Voxel::from_raw(0x1234));
        brick.set(1, 0, 0, Voxel::from_raw(0xabcd));
        let mut words = Vec::new();
        brick.append_gpu_words(&mut words);
        assert_eq!(words.len(), GPU_WORDS_PER_BRICK);
        assert_eq!(words[0], 0xabcd1234);
    }

    #[test]
    fn size_matches_design() {
        assert_eq!(BRICK_BYTES, 8192);
        assert_eq!(BRICK_VOXELS, 4096);
    }
}
