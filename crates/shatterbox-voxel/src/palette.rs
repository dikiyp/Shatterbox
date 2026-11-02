//! Material palette. Voxels store only an index into this table.

/// Stable material indices used by the test arena and future content.
pub mod materials {
    pub const AIR: u16 = 0;
    pub const CONCRETE: u16 = 1;
    pub const STONE: u16 = 2;
    pub const BRICK: u16 = 3;
    pub const WOOD: u16 = 4;
    pub const METAL: u16 = 5;
    pub const GLASS: u16 = 6;
    pub const SAND: u16 = 7;
    pub const GRASS: u16 = 8;
    pub const DIRT: u16 = 9;
    pub const EMISSIVE: u16 = 10;
    pub const HAZARD: u16 = 11;
    pub const RUBBER: u16 = 12;
    /// Damp, polished concrete: low roughness keeps soft sky/sun highlights.
    pub const WET_CONCRETE: u16 = 13;
    /// Coloured neon strips for sign/corridor lighting tests.
    pub const NEON_BLUE: u16 = 14;
    pub const NEON_MAGENTA: u16 = 15;
    pub const NEON_GREEN: u16 = 16;
    pub const NEON_WHITE: u16 = 17;
    /// Recessed brick joint. Softer than brick in the palette data so a future
    /// hardness-aware carve can follow the beds, and visibly lighter/duller.
    pub const MORTAR: u16 = 18;
    /// Fired brick for the house scene. A separate index from `BRICK` because
    /// `lighting.wgsl` paints a 1 m procedural brick pattern on `BRICK`; the
    /// house has real per-voxel brick relief and must not get that pattern on
    /// top of it.
    pub const HOUSE_BRICK: u16 = 19;
    /// Saturated paint for the light-check room. `tint` only scales brightness,
    /// so the red/blue bounce has to come from the albedo itself.
    pub const RED_PAINT: u16 = 20;
    pub const BLUE_PAINT: u16 = 21;
    /// Polished metal mirror for the SSR/mirror test room. Metalness 1 makes the
    /// albedo the reflection tint, roughness 0.03 keeps the mirrored image sharp.
    pub const MIRROR: u16 = 22;
    /// Clean architectural glass: little roughness, low opacity, near-neutral
    /// tint. Shaded as a reflective pane (see `lighting.wgsl`), not as a solid.
    pub const GLASS_CLEAR: u16 = 23;
}

/// A palette entry. Rendering parameters are intentionally kept small and
/// style-agnostic; stylised looks are expressed with flags on top of PBR.
#[derive(Clone, Debug)]
pub struct Material {
    pub name: &'static str,
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub metalness: f32,
    pub emissive: f32,
    pub opacity: f32,
    pub density: f32,
    pub hardness: f32,
    pub flammability: f32,
    pub sound: &'static str,
}

impl Material {
    pub const fn new(name: &'static str, albedo: [f32; 3]) -> Self {
        Self {
            name,
            albedo,
            roughness: 0.7,
            metalness: 0.0,
            emissive: 0.0,
            opacity: 1.0,
            density: 1.0,
            hardness: 1.0,
            flammability: 0.0,
            sound: "default",
        }
    }

    pub const fn with_roughness(mut self, value: f32) -> Self {
        self.roughness = value;
        self
    }

    pub const fn with_metalness(mut self, value: f32) -> Self {
        self.metalness = value;
        self
    }

    pub const fn with_emissive(mut self, value: f32) -> Self {
        self.emissive = value;
        self
    }

    pub const fn with_opacity(mut self, value: f32) -> Self {
        self.opacity = value;
        self
    }

    pub const fn with_density(mut self, value: f32) -> Self {
        self.density = value;
        self
    }

    pub const fn with_hardness(mut self, value: f32) -> Self {
        self.hardness = value;
        self
    }

    pub const fn with_flammability(mut self, value: f32) -> Self {
        self.flammability = value;
        self
    }

    pub const fn with_sound(mut self, value: &'static str) -> Self {
        self.sound = value;
        self
    }

    /// Packed layout consumed by the visibility shader:
    /// `[albedo.rgb, roughness, metalness, emissive, opacity, reserved]`.
    pub fn gpu_data(&self) -> [f32; 8] {
        [
            self.albedo[0],
            self.albedo[1],
            self.albedo[2],
            self.roughness,
            self.metalness,
            self.emissive,
            self.opacity,
            0.0,
        ]
    }
}

/// Material table. Index zero is always air.
#[derive(Clone, Debug)]
pub struct Palette {
    materials: Vec<Material>,
}

impl Default for Palette {
    fn default() -> Self {
        let materials = vec![
            Material::new("air", [0.0, 0.0, 0.0]).with_opacity(0.0),
            Material::new("concrete", [0.31, 0.31, 0.32])
                .with_roughness(0.88)
                .with_density(2.3)
                .with_hardness(0.72)
                .with_sound("concrete"),
            Material::new("stone", [0.27, 0.28, 0.30])
                .with_roughness(0.82)
                .with_density(2.6)
                .with_hardness(0.85)
                .with_sound("stone"),
            Material::new("brick", [0.40, 0.23, 0.18])
                .with_roughness(0.8)
                .with_density(1.9)
                .with_hardness(0.55)
                .with_sound("brick"),
            Material::new("wood", [0.46, 0.30, 0.16])
                .with_roughness(0.65)
                .with_density(0.7)
                .with_hardness(0.32)
                .with_flammability(0.85)
                .with_sound("wood"),
            Material::new("metal", [0.33, 0.35, 0.38])
                .with_roughness(0.32)
                .with_metalness(0.9)
                .with_density(7.8)
                .with_hardness(0.95)
                .with_sound("metal"),
            Material::new("glass", [0.35, 0.58, 0.62])
                .with_roughness(0.08)
                .with_metalness(0.05)
                .with_opacity(0.35)
                .with_density(2.5)
                .with_hardness(0.15)
                .with_sound("glass"),
            Material::new("sand", [0.72, 0.63, 0.45])
                .with_roughness(0.95)
                .with_density(1.6)
                .with_hardness(0.12)
                .with_sound("sand"),
            Material::new("grass", [0.22, 0.42, 0.16])
                .with_roughness(0.9)
                .with_density(0.9)
                .with_hardness(0.18)
                .with_flammability(0.6)
                .with_sound("grass"),
            Material::new("dirt", [0.30, 0.22, 0.14])
                .with_roughness(0.94)
                .with_density(1.5)
                .with_hardness(0.2)
                .with_sound("dirt"),
            Material::new("emissive", [0.95, 0.55, 0.18])
                .with_roughness(0.4)
                .with_emissive(5.0)
                .with_sound("energy"),
            Material::new("hazard", [0.85, 0.58, 0.05])
                .with_roughness(0.55)
                .with_metalness(0.2)
                .with_density(2.0)
                .with_hardness(0.5)
                .with_sound("metal"),
            Material::new("rubber", [0.08, 0.09, 0.10])
                .with_roughness(0.95)
                .with_density(1.1)
                .with_hardness(0.25)
                .with_sound("rubber"),
            Material::new("wet_concrete", [0.235, 0.245, 0.255])
                .with_roughness(0.25)
                .with_metalness(0.04)
                .with_density(2.3)
                .with_hardness(0.72)
                .with_sound("concrete"),
            Material::new("neon_blue", [0.16, 0.52, 1.0])
                .with_roughness(0.35)
                .with_emissive(6.0)
                .with_sound("energy"),
            Material::new("neon_magenta", [1.0, 0.14, 0.62])
                .with_roughness(0.35)
                .with_emissive(6.0)
                .with_sound("energy"),
            Material::new("neon_green", [0.18, 1.0, 0.42])
                .with_roughness(0.35)
                .with_emissive(6.0)
                .with_sound("energy"),
            Material::new("neon_white", [0.92, 0.96, 1.0])
                .with_roughness(0.3)
                .with_emissive(6.0)
                .with_sound("energy"),
            // House scene entries. Appended after the neon set so every
            // pre-existing index stays stable for saves and scenarios.
            Material::new("mortar", [0.33, 0.32, 0.30])
                .with_roughness(0.96)
                .with_density(1.7)
                .with_hardness(0.28)
                .with_sound("sand"),
            Material::new("house_brick", [0.47, 0.24, 0.17])
                .with_roughness(0.86)
                .with_density(1.9)
                .with_hardness(0.55)
                .with_sound("brick"),
            // Test-room paints. Deep, saturated albedos: an 0.85+ roughness
            // keeps the bounce diffuse and readable through SSGI instead of
            // turning the wall into a mirror. Non-emissive on purpose, the
            // point is to show light *reflecting* off them.
            Material::new("red_paint", [0.70, 0.055, 0.045])
                .with_roughness(0.78)
                .with_density(1.8)
                .with_hardness(0.35)
                .with_sound("concrete"),
            Material::new("blue_paint", [0.045, 0.13, 0.62])
                .with_roughness(0.78)
                .with_density(1.8)
                .with_hardness(0.35)
                .with_sound("concrete"),
            // Mirror and clear glass for the reflection room. The surface shader
            // reads these indices directly (`lighting.wgsl`), so the palette
            // values are the only tuning knob: a mirror tint is the albedo, a
            // glass tint is the low albedo the transmitted ambient picks up.
            Material::new("mirror", [0.92, 0.93, 0.96])
                .with_roughness(0.03)
                .with_metalness(1.0)
                .with_density(2.5)
                .with_hardness(0.6)
                .with_sound("metal"),
            Material::new("glass_clear", [0.86, 0.93, 0.96])
                .with_roughness(0.04)
                .with_metalness(0.0)
                .with_opacity(0.2)
                .with_density(2.5)
                .with_hardness(0.15)
                .with_sound("glass"),
        ];
        Self { materials }
    }
}

impl Palette {
    pub fn new(materials: Vec<Material>) -> Self {
        assert!(
            materials
                .first()
                .is_some_and(|material| material.name == "air"),
            "palette index zero must be air"
        );
        Self { materials }
    }

    /// Returns the requested material or air for an out-of-range index.
    pub fn material(&self, index: u16) -> &Material {
        self.materials
            .get(index as usize)
            .unwrap_or(&self.materials[0])
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Packed GPU palette. One `[f32; 8]` per material.
    pub fn gpu_data(&self) -> Vec<[f32; 8]> {
        self.materials.iter().map(Material::gpu_data).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_zero_is_air() {
        let palette = Palette::default();
        assert_eq!(palette.material(0).name, "air");
        assert_eq!(palette.material(u16::MAX).name, "air");
    }

    #[test]
    fn gpu_layout_has_eight_floats() {
        let palette = Palette::default();
        let data = palette.gpu_data();
        assert_eq!(data.len(), palette.len());
        assert_eq!(data[materials::EMISSIVE as usize][5], 5.0);
    }
}
