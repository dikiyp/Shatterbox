//! Compact voxel representation: 10 bits of material plus 6 bits of tint.

/// A single voxel. Material zero is air, so a tint on air is discarded.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Voxel(u16);

impl Voxel {
    /// Empty voxel / air.
    pub const AIR: Self = Self(0);
    /// Number of bits reserved for the palette index.
    pub const MATERIAL_BITS: u16 = 10;
    /// Mask selecting the palette index.
    pub const MATERIAL_MASK: u16 = (1 << Self::MATERIAL_BITS) - 1;
    /// Number of bits reserved for the colour variant.
    pub const TINT_BITS: u16 = 6;
    /// Mask selecting the colour variant.
    pub const TINT_MASK: u16 = (1 << Self::TINT_BITS) - 1;

    /// Packs a palette index and a tint variant. Air always maps to `AIR`.
    pub const fn new(material: u16, tint: u8) -> Self {
        if material == 0 {
            Self::AIR
        } else {
            Self((material & Self::MATERIAL_MASK) | (((tint as u16) & Self::TINT_MASK) << 10))
        }
    }

    /// Reconstructs a voxel from its raw bit representation.
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    /// Raw bit representation used by the storage and GPU upload path.
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Palette index.
    pub const fn material(self) -> u16 {
        self.0 & Self::MATERIAL_MASK
    }

    /// Colour variant in `0..=63`.
    pub const fn tint(self) -> u8 {
        ((self.0 >> 10) & Self::TINT_MASK) as u8
    }

    /// Whether the voxel is empty.
    pub const fn is_air(self) -> bool {
        self.material() == 0
    }

    /// Whether the voxel contains a solid material.
    pub const fn is_solid(self) -> bool {
        !self.is_air()
    }
}

impl std::fmt::Debug for Voxel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Voxel")
            .field("material", &self.material())
            .field("tint", &self.tint())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_material_and_tint() {
        let voxel = Voxel::new(7, 42);
        assert_eq!(voxel.material(), 7);
        assert_eq!(voxel.tint(), 42);
        assert_eq!(voxel.raw(), 7 | (42 << 10));
        assert!(voxel.is_solid());
    }

    #[test]
    fn air_ignores_tint() {
        let voxel = Voxel::new(0, 63);
        assert_eq!(voxel, Voxel::AIR);
        assert!(voxel.is_air());
        assert_eq!(voxel.raw(), 0);
    }

    #[test]
    fn masks_large_values() {
        let voxel = Voxel::new(u16::MAX, 0xff);
        assert_eq!(voxel.material(), Voxel::MATERIAL_MASK);
        assert_eq!(voxel.tint(), Voxel::TINT_MASK as u8);
    }
}
