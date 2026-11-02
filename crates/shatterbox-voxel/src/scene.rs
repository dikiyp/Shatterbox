//! Canned test scenes.
//!
//! `SceneKind::House` (default) builds a quarter-area concrete plate with a
//! two-storey house: concrete frame, real 2x1x1 brick infill with relief and
//! recessed joints, openings, a stair and five ceiling lamps. `SceneKind::Studio`
//! keeps the sparse white room used by the lighting and destruction regression
//! scenarios: flat white floor, white perimeter, three screens, columns,
//! coloured primitives, glass and a thin wall a single rocket can blow apart.

use glam::{IVec3, Vec3};

use crate::palette::{Material, Palette, materials};
use crate::voxel::Voxel;
use crate::world::{VOXEL_SIZE_M, VoxelWorld};

/// First air layer above the floor. The floor surface is at 1.0 m.
pub const FLOOR_TOP_VOXEL: i32 = 8;
/// Top solid voxel of the plate. Structural connectivity is measured from this
/// layer, so a wall that does not actually reach the plate is reported floating
/// even though it is connected to itself.
pub const GROUND_ANCHOR_VOXEL: i32 = FLOOR_TOP_VOXEL - 1;
/// Metres to the floor surface, i.e. the feet height of a grounded player.
const FLOOR_TOP_M: f32 = FLOOR_TOP_VOXEL as f32 * VOXEL_SIZE_M;
/// Playable extent in voxels: 64 m .. 192 m.
const PLAY_MIN: i32 = 512;
const PLAY_MAX: i32 = 1536;

/// Which canned scene the game loads. The studio is kept for the lighting and
/// destruction regression scenarios; the house is the default from Phase A on;
/// the test room is the small light/reflection box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    Studio,
    House,
    TestRoom,
}

impl SceneKind {
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "studio" => Some(Self::Studio),
            "house" => Some(Self::House),
            "testroom" | "test_room" => Some(Self::TestRoom),
            _ => None,
        }
    }

    /// `SHATTERBOX_SCENE=studio|house|testroom`; the house is the default.
    #[must_use]
    pub fn from_env() -> Self {
        std::env::var("SHATTERBOX_SCENE")
            .ok()
            .and_then(|value| Self::parse(&value))
            .unwrap_or(Self::House)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Studio => "studio",
            Self::House => "house",
            Self::TestRoom => "testroom",
        }
    }

    #[must_use]
    pub fn build(self) -> VoxelWorld {
        match self {
            Self::Studio => build_test_arena(),
            Self::House => build_house(),
            Self::TestRoom => build_test_room(),
        }
    }

    /// Feet spawn hint, yaw and pitch. `spawn_on_floor` snaps the hint down to
    /// the plate, so the height only has to be above the floor.
    #[must_use]
    pub const fn spawn(self) -> (Vec3, f32, f32) {
        match self {
            Self::Studio => (Vec3::new(128.0, 30.0, 180.0), 0.0, -0.12),
            // In front of the south door, looking at the facade.
            Self::House => (Vec3::new(126.0, 30.0, 138.0), 0.0, -0.05),
            // Inside the room, in the south-east corner, looking north along
            // the red (west) and blue (east) walls.
            Self::TestRoom => (Vec3::new(139.0, 20.0, ROOM_MAX_Z_M - 2.5), 0.0, -0.02),
        }
    }
}

/// The white studio surface. See [`studio_palette`] for why this slot is used.
const WHITE: u16 = materials::NEON_WHITE;
/// Maximum tint: the shaders map tint 63 to a 1.14x albedo boost, which keeps
/// the white surfaces at the top of the range even under the gentle macro
/// variation applied by the lighting pass.
const WHITE_TINT: u8 = 63;
/// Fixed glass tint; per-voxel variation reads as dirt on transparent panes.
const GLASS_TINT: u8 = 24;

/// Builds the white studio inside the 256 m world. Fully deterministic.
pub fn build_test_arena() -> VoxelWorld {
    let mut world = VoxelWorld::new(studio_palette());

    // One flat three-voxel white plate (0.375 m); the top face is at 1.0 m.
    fill_box_vox(
        &mut world,
        IVec3::new(PLAY_MIN, FLOOR_TOP_VOXEL - 3, PLAY_MIN),
        IVec3::new(PLAY_MAX, FLOOR_TOP_VOXEL, PLAY_MAX),
        WHITE,
        WHITE_TINT,
    );

    build_perimeter(&mut world);
    build_screens(&mut world);
    build_columns(&mut world);
    build_coloured_cubes(&mut world);
    build_glass(&mut world);
    build_destruction_target(&mut world);

    world
}

/// `palette.rs` has no plain white material: `concrete`, `stone` and
/// `wet_concrete` all render 4 m slab joints and panel stains in
/// `material_detail`, while `neon_white` is the only entry the shaders treat as
/// detail-free (indices 14..=17 return a constant detail of 1.0) - but the stock
/// entry emits 6.0 and therefore lights the scene instead of receiving light.
/// Reusing that slot as a non-emissive white gives an evenly lit studio surface
/// without editing `palette.rs` or the shaders. Every other palette entry keeps
/// its original value and index.
fn studio_palette() -> Palette {
    let source = Palette::default();
    let mut entries: Vec<Material> = (0..source.len())
        .map(|index| source.material(index as u16).clone())
        .collect();
    entries[WHITE as usize] = Material::new("studio_white", [0.93, 0.93, 0.93])
        .with_roughness(0.9)
        .with_density(2.3)
        .with_hardness(0.7)
        .with_sound("concrete");
    Palette::new(entries)
}

/// 1 m thick, 4 m tall white walls matching the physics colliders at
/// x/z 64.5 and 191.5.
fn build_perimeter(world: &mut VoxelWorld) {
    let top = FLOOR_TOP_M + 4.0;
    for (min, max) in [
        (
            Vec3::new(64.0, FLOOR_TOP_M, 64.0),
            Vec3::new(65.0, top, 192.0),
        ),
        (
            Vec3::new(191.0, FLOOR_TOP_M, 64.0),
            Vec3::new(192.0, top, 192.0),
        ),
        (
            Vec3::new(65.0, FLOOR_TOP_M, 64.0),
            Vec3::new(191.0, top, 65.0),
        ),
        (
            Vec3::new(65.0, FLOOR_TOP_M, 191.0),
            Vec3::new(191.0, top, 192.0),
        ),
    ] {
        fill_box_m(world, min, max, WHITE, WHITE_TINT);
    }
}

/// Three parallel white screens 12 m apart, 0.5 m thick, 4-5 m tall. Only the
/// one facing the spawn has a 1.5 x 2.5 m doorway, so a low sun still draws a
/// light shaft across the floor while the other screens shade the props cleanly.
fn build_screens(world: &mut VoxelWorld) {
    // Near screen, closest to the spawn: the doorway is cut here.
    screen_wall(world, 100.0, 156.0, 156.0, 156.5, 4.5, &[(126.0, 127.5)]);
    screen_wall(world, 100.0, 156.0, 144.0, 144.5, 4.0, &[]);
    screen_wall(world, 100.0, 156.0, 132.0, 132.5, 5.0, &[]);
}

/// One 0.5 m thick white wall with door openings cut from the floor to 2.5 m.
fn screen_wall(
    world: &mut VoxelWorld,
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    height: f32,
    doors: &[(f32, f32)],
) {
    fill_box_m(
        world,
        Vec3::new(x0, FLOOR_TOP_M, z0),
        Vec3::new(x1, FLOOR_TOP_M + height, z1),
        WHITE,
        WHITE_TINT,
    );
    for (door_x0, door_x1) in doors {
        fill_box_m(
            world,
            Vec3::new(*door_x0, FLOOR_TOP_M, z0 - 0.25),
            Vec3::new(*door_x1, FLOOR_TOP_M + 2.5, z1 + 0.25),
            materials::AIR,
            0,
        );
    }
}

/// Four white 0.5 x 4 x 0.5 m columns, 14 m clear of the screen ends and 24 m
/// apart along z. They give the sun a second, thinner shadow caster.
fn build_columns(world: &mut VoxelWorld) {
    for (x, z) in [(86.0, 156.0), (86.0, 132.0), (170.0, 156.0), (170.0, 132.0)] {
        fill_box_m(
            world,
            Vec3::new(x, FLOOR_TOP_M, z),
            Vec3::new(x + 0.5, FLOOR_TOP_M + 4.0, z + 0.5),
            WHITE,
            WHITE_TINT,
        );
    }
}

/// Five coloured primitives on a 10.5 m arc opening towards the spawn: 8 m of
/// air between neighbours, and roughly 5 m to the near screen. They only exist
/// to put coloured bounce and a few roughness/metalness contrasts into frame.
fn build_coloured_cubes(world: &mut VoxelWorld) {
    // (x, z, size, material). Semicircle around (128, 172), the apex points
    // away from the spawn so the row never blocks the view of the screens.
    let cubes = [
        (138.5, 172.0, 1.5, materials::BRICK),
        (135.4, 164.6, 1.0, materials::WOOD),
        (128.0, 161.5, 0.5, materials::METAL),
        (120.6, 164.6, 1.25, materials::STONE),
        (117.5, 172.0, 0.75, materials::EMISSIVE),
    ];
    for (x, z, size, material) in cubes {
        fill_box_m(
            world,
            Vec3::new(x, FLOOR_TOP_M, z),
            Vec3::new(x + size, FLOOR_TOP_M + size, z + size),
            material,
            WHITE_TINT,
        );
    }
}

/// Two free-standing 2 x 2.5 m glass panes (0.25 m thick) and one sealed
/// 2 x 2 x 2 m glass block with a two-voxel (0.25 m) wall.
fn build_glass(world: &mut VoxelWorld) {
    for (x, z) in [(98.0, 172.0), (108.0, 178.0)] {
        fill_box_m(
            world,
            Vec3::new(x, FLOOR_TOP_M, z),
            Vec3::new(x + 2.0, FLOOR_TOP_M + 2.5, z + 0.25),
            materials::GLASS,
            GLASS_TINT,
        );
    }

    glass_block(
        world,
        Vec3::new(152.0, FLOOR_TOP_M, 176.0),
        Vec3::new(154.0, FLOOR_TOP_M + 2.0, 178.0),
    );
}

/// A glass cube with 0.25 m walls: fill the box, then carve the interior.
fn glass_block(world: &mut VoxelWorld, min: Vec3, max: Vec3) {
    fill_box_m(world, min, max, materials::GLASS, GLASS_TINT);
    let inner_min = vox_min(min) + IVec3::splat(2);
    let inner_max = vox_max(max) - IVec3::splat(2);
    fill_box_vox(world, inner_min, inner_max, materials::AIR, 0);
}

/// Free-standing 4 x 3 m wall, 0.25 m thick, plus a 1.5 m cube 4 m beside it: a
/// single rocket leaves a clean crater and a readable debris field.
fn build_destruction_target(world: &mut VoxelWorld) {
    fill_box_m(
        world,
        Vec3::new(146.0, FLOOR_TOP_M, 168.0),
        Vec3::new(150.0, FLOOR_TOP_M + 3.0, 168.25),
        WHITE,
        WHITE_TINT,
    );
    fill_box_m(
        world,
        Vec3::new(152.0, FLOOR_TOP_M, 163.0),
        Vec3::new(153.5, FLOOR_TOP_M + 1.5, 164.5),
        materials::STONE,
        WHITE_TINT,
    );
}

// ---------------------------------------------------------------------------
// House scene
// ---------------------------------------------------------------------------

/// Ground plate: half the studio's side, a quarter of its area. 64 x 64 m,
/// 512 x 512 x 3 voxels, top face at 1.0 m like the studio floor.
const HOUSE_PLATE_MIN: i32 = 768; // 96 m
const HOUSE_PLATE_MAX: i32 = 1280; // 160 m
/// Footprint: 96 x 80 voxels = 12 x 10 m, centred on the plate.
const HOUSE_MIN_X: i32 = 976; // 122 m
const HOUSE_MAX_X: i32 = 1072; // 134 m
const HOUSE_MIN_Z: i32 = 984; // 123 m
const HOUSE_MAX_Z: i32 = 1064; // 133 m
/// Inter-floor slab and roof slab, two voxels (0.25 m) thick. Upper faces at
/// 4.25 m and 7.5 m; the walls between them are 22 voxels (2.75 m).
const HOUSE_SLAB1: i32 = 32;
const HOUSE_TOP: i32 = 58;
const HOUSE_ROOF_TOP: i32 = 60;
/// Stairwell opening in the inter-floor slab.
const STAIR_MIN_X: i32 = 980;
const STAIR_MAX_X: i32 = 1032;
const STAIR_MIN_Z: i32 = 1020;
const STAIR_MAX_Z: i32 = 1032;

const HOUSE_FRAME: u16 = materials::CONCRETE;
const HOUSE_BRICK: u16 = materials::HOUSE_BRICK;
const HOUSE_MORTAR: u16 = materials::MORTAR;
/// Ceiling lamps: (voxel x, voxel y, voxel z) of the lower corner of a 2x1x2
/// emissive block. Three on the ground floor, two upstairs.
const HOUSE_LAMPS: [(i32, i32, i32); 5] = [
    (1000, 31, 1044),
    (1048, 31, 1044),
    (1010, 31, 1000),
    (1000, 57, 1044),
    (1048, 57, 1000),
];

/// Builds the concrete plate, the two-storey house and its five ceiling lamps.
/// Everything is deterministic from the scene seed.
pub fn build_house() -> VoxelWorld {
    build_house_with_seed(0x5a17_3b1d)
}

/// Same as [`build_house`], with an explicit seed for tests.
pub fn build_house_with_seed(seed: u32) -> VoxelWorld {
    let mut world = VoxelWorld::new(Palette::default());

    // Quarter-area plate. The house stands on it, so every lower column and
    // wall voxel has a face-sharing neighbour in the plate.
    fill_box_vox(
        &mut world,
        IVec3::new(HOUSE_PLATE_MIN, FLOOR_TOP_VOXEL - 3, HOUSE_PLATE_MIN),
        IVec3::new(HOUSE_PLATE_MAX, FLOOR_TOP_VOXEL, HOUSE_PLATE_MAX),
        HOUSE_FRAME,
        26,
    );

    // Brick infill first; the concrete frame then overwrites the cells it owns.
    for (wall_id, face) in WallFace::ALL.into_iter().enumerate() {
        build_brick_wall(&mut world, seed, face, wall_id as u32, 8, 30);
        build_brick_wall(&mut world, seed, face, wall_id as u32, 34, 56);
    }

    build_frame(&mut world);
    build_slabs(&mut world);
    build_stairs(&mut world);
    build_lamps(&mut world);
    carve_openings(&mut world);

    world
}

/// Columns, ring beams and the perimeter brackets that carry the slabs.
fn build_frame(world: &mut VoxelWorld) {
    // 2x2 voxel columns at the corners and the mid-span of each long wall.
    for x in [HOUSE_MIN_X, 1022, HOUSE_MAX_X - 2] {
        for z in [HOUSE_MIN_Z, 1022, HOUSE_MAX_Z - 2] {
            fill_box_vox(
                world,
                IVec3::new(x, FLOOR_TOP_VOXEL, z),
                IVec3::new(x + 2, HOUSE_ROOF_TOP, z + 2),
                HOUSE_FRAME,
                22,
            );
        }
    }
    // Ring beams 2 voxels deep under both slabs.
    for y in [HOUSE_SLAB1 - 2, HOUSE_TOP - 2] {
        for (x0, x1, z0, z1) in [
            (HOUSE_MIN_X, HOUSE_MAX_X, HOUSE_MAX_Z - 2, HOUSE_MAX_Z),
            (HOUSE_MIN_X, HOUSE_MAX_X, HOUSE_MIN_Z, HOUSE_MIN_Z + 2),
            (
                HOUSE_MIN_X,
                HOUSE_MIN_X + 2,
                HOUSE_MIN_Z + 2,
                HOUSE_MAX_Z - 2,
            ),
            (
                HOUSE_MAX_X - 2,
                HOUSE_MAX_X,
                HOUSE_MIN_Z + 2,
                HOUSE_MAX_Z - 2,
            ),
        ] {
            fill_box_vox(
                world,
                IVec3::new(x0, y, z0),
                IVec3::new(x1, y + 2, z1),
                HOUSE_FRAME,
                24,
            );
        }
    }
}

/// Inter-floor slab with the stairwell cut out, and the roof slab.
fn build_slabs(world: &mut VoxelWorld) {
    fill_box_vox(
        world,
        IVec3::new(HOUSE_MIN_X, HOUSE_SLAB1, HOUSE_MIN_Z),
        IVec3::new(HOUSE_MAX_X, HOUSE_SLAB1 + 2, HOUSE_MAX_Z),
        HOUSE_FRAME,
        28,
    );
    fill_box_vox(
        world,
        IVec3::new(STAIR_MIN_X, HOUSE_SLAB1, STAIR_MIN_Z),
        IVec3::new(STAIR_MAX_X, HOUSE_SLAB1 + 2, STAIR_MAX_Z),
        materials::AIR,
        0,
    );
    fill_box_vox(
        world,
        IVec3::new(HOUSE_MIN_X, HOUSE_TOP, HOUSE_MIN_Z),
        IVec3::new(HOUSE_MAX_X, HOUSE_ROOF_TOP, HOUSE_MAX_Z),
        HOUSE_FRAME,
        28,
    );
}

/// Thirteen 0.25 m risers with 0.5 m treads, solid down to the plate. The top
/// step is flush with the upper slab, so the stair is walkable and cannot float.
fn build_stairs(world: &mut VoxelWorld) {
    for step in 0..13 {
        let x0 = STAIR_MIN_X + step * 4;
        let top = 10 + step * 2;
        fill_box_vox(
            world,
            IVec3::new(x0, FLOOR_TOP_VOXEL, STAIR_MIN_Z),
            IVec3::new(x0 + 4, top, STAIR_MAX_Z),
            HOUSE_FRAME,
            30,
        );
    }
}

/// Emissive lamp blocks; their point lights are created by the game layer.
fn build_lamps(world: &mut VoxelWorld) {
    for (x, y, z) in HOUSE_LAMPS {
        fill_box_vox(
            world,
            IVec3::new(x, y, z),
            IVec3::new(x + 2, y + 1, z + 2),
            materials::NEON_WHITE,
            63,
        );
    }
}

/// World-space point-light positions of the ceiling lamps, 0.2 m below the
/// emissive block so the emitter itself is not inside the light origin.
#[must_use]
pub fn house_lamp_positions() -> [Vec3; 5] {
    HOUSE_LAMPS.map(|(x, y, z)| {
        Vec3::new(
            (x + 1) as f32 * VOXEL_SIZE_M,
            y as f32 * VOXEL_SIZE_M - 0.2,
            (z + 1) as f32 * VOXEL_SIZE_M,
        )
    })
}

// ---------------------------------------------------------------------------
// Test room scene
// ---------------------------------------------------------------------------

/// Interior footprint: 16 x 16 m, 6 m tall, centred on its own 64 m plate.
const ROOM_MIN_X: i32 = 1024; // 128 m
const ROOM_MAX_X: i32 = 1152; // 144 m
const ROOM_MIN_Z: i32 = 1024; // 128 m
const ROOM_MAX_Z: i32 = 1152; // 144 m
const ROOM_MIN_X_M: f32 = ROOM_MIN_X as f32 * VOXEL_SIZE_M;
const ROOM_MAX_X_M: f32 = ROOM_MAX_X as f32 * VOXEL_SIZE_M;
const ROOM_MAX_Z_M: f32 = ROOM_MAX_Z as f32 * VOXEL_SIZE_M;
/// Underside and top of the ceiling slab: 7.0 m and 7.25 m above the floor.
const ROOM_CEILING: i32 = 56;
const ROOM_CEILING_TOP: i32 = 58;
/// Recessed window opening: 3 m wide, sill at 2 m, head at 5 m.
const ROOM_WINDOW_Y0: i32 = 16;
const ROOM_WINDOW_Y1: i32 = 40;
const ROOM_WINDOW_INSET: i32 = 24; // 3 m from the wall corner
const ROOM_WINDOW_WIDTH: i32 = 24; // 3 m
const ROOM_WALL_DEPTH: i32 = 2; // 0.25 m
/// Emissive ceiling lamps: lower corner of a 2x1x2 voxel block.
const ROOM_LAMPS: [(i32, i32, i32); 2] = [(1062, 54, 1086), (1110, 54, 1086)];
/// Neutral tint for the painted walls: `tint` only scales brightness, the hue
/// comes from the material itself.
const PAINT_TINT: u8 = 31;

/// White floor/ceiling/side walls, one saturated red wall and one saturated
/// blue wall, windows cut through every wall, and glass panes (in two windows
/// plus one free-standing) for the SSR path.
pub fn build_test_room() -> VoxelWorld {
    let mut world = VoxelWorld::new(studio_palette());

    // Same 64 m white plate as the house, so the room stands on real ground.
    fill_box_vox(
        &mut world,
        IVec3::new(HOUSE_PLATE_MIN, FLOOR_TOP_VOXEL - 3, HOUSE_PLATE_MIN),
        IVec3::new(HOUSE_PLATE_MAX, FLOOR_TOP_VOXEL, HOUSE_PLATE_MAX),
        WHITE,
        WHITE_TINT,
    );

    let floor = FLOOR_TOP_VOXEL;
    // The red wall is west, the blue wall is east, so a camera looking north
    // has both in frame and sees the bounce on the white floor and ceiling.
    fill_box_vox(
        &mut world,
        IVec3::new(ROOM_MIN_X, floor, ROOM_MIN_Z),
        IVec3::new(ROOM_MIN_X + ROOM_WALL_DEPTH, ROOM_CEILING, ROOM_MAX_Z),
        materials::RED_PAINT,
        PAINT_TINT,
    );
    fill_box_vox(
        &mut world,
        IVec3::new(ROOM_MAX_X - ROOM_WALL_DEPTH, floor, ROOM_MIN_Z),
        IVec3::new(ROOM_MAX_X, ROOM_CEILING, ROOM_MAX_Z),
        materials::BLUE_PAINT,
        PAINT_TINT,
    );
    fill_box_vox(
        &mut world,
        IVec3::new(ROOM_MIN_X + ROOM_WALL_DEPTH, floor, ROOM_MIN_Z),
        IVec3::new(
            ROOM_MAX_X - ROOM_WALL_DEPTH,
            ROOM_CEILING,
            ROOM_MIN_Z + ROOM_WALL_DEPTH,
        ),
        WHITE,
        WHITE_TINT,
    );
    fill_box_vox(
        &mut world,
        IVec3::new(
            ROOM_MIN_X + ROOM_WALL_DEPTH,
            floor,
            ROOM_MAX_Z - ROOM_WALL_DEPTH,
        ),
        IVec3::new(ROOM_MAX_X - ROOM_WALL_DEPTH, ROOM_CEILING, ROOM_MAX_Z),
        WHITE,
        WHITE_TINT,
    );
    // White slab ceiling over the whole footprint. It has to overlap the wall
    // tops: an inset slab only touches them diagonally and counts as floating.
    fill_box_vox(
        &mut world,
        IVec3::new(ROOM_MIN_X, ROOM_CEILING, ROOM_MIN_Z),
        IVec3::new(ROOM_MAX_X, ROOM_CEILING_TOP, ROOM_MAX_Z),
        WHITE,
        WHITE_TINT,
    );

    // Two openings per wall: one in the west half, one in the east half. The
    // red and blue windows carry glass; the white walls stay open so daylight
    // falls in as a shaft and the room is readable at noon.
    let opening = |along_start: i32| (along_start, along_start + ROOM_WINDOW_WIDTH);
    let windows_x = [
        opening(ROOM_MIN_X + ROOM_WINDOW_INSET),
        opening(ROOM_MIN_X + ROOM_WINDOW_INSET * 3),
    ];
    let windows_z = [
        opening(ROOM_MIN_Z + ROOM_WINDOW_INSET),
        opening(ROOM_MIN_Z + ROOM_WINDOW_INSET * 3),
    ];
    for (x0, x1) in windows_x {
        // North and south walls.
        fill_box_vox(
            &mut world,
            IVec3::new(x0, ROOM_WINDOW_Y0, ROOM_MIN_Z),
            IVec3::new(x1, ROOM_WINDOW_Y1, ROOM_MIN_Z + ROOM_WALL_DEPTH),
            materials::AIR,
            0,
        );
        fill_box_vox(
            &mut world,
            IVec3::new(x0, ROOM_WINDOW_Y0, ROOM_MAX_Z - ROOM_WALL_DEPTH),
            IVec3::new(x1, ROOM_WINDOW_Y1, ROOM_MAX_Z),
            materials::AIR,
            0,
        );
    }
    for (z0, z1) in windows_z {
        // West (red) and east (blue) walls: open, so direct sun paints the
        // ceiling and the opposite wall with the wall's own colour.
        for (x0, x1) in [
            (ROOM_MIN_X, ROOM_MIN_X + ROOM_WALL_DEPTH),
            (ROOM_MAX_X - ROOM_WALL_DEPTH, ROOM_MAX_X),
        ] {
            fill_box_vox(
                &mut world,
                IVec3::new(x0, ROOM_WINDOW_Y0, z0),
                IVec3::new(x1, ROOM_WINDOW_Y1, z1),
                materials::AIR,
                0,
            );
        }
    }

    // Free-standing glass pane, 2 x 2.5 m and 0.25 m thick, in the west third
    // of the room. `GLASS_CLEAR` is the reflective pane material: low roughness,
    // almost no diffuse, strong fresnel reflection plus the transmission tint
    // added in `lighting.wgsl`.
    fill_box_vox(
        &mut world,
        IVec3::new(1044, floor, 1104),
        IVec3::new(1060, floor + 20, 1106),
        materials::GLASS_CLEAR,
        GLASS_TINT,
    );
    // Wall mirror on the north wall, between the two windows. Metal mirror
    // (`MIRROR`, metalness 1, roughness 0.03) gives the SSR path a surface whose
    // whole job is to show the room: the red wall on the left, the blue wall on
    // the right, the windows behind the camera.
    fill_box_vox(
        &mut world,
        IVec3::new(1076, floor + 4, ROOM_MAX_Z - ROOM_WALL_DEPTH - 2),
        IVec3::new(1092, floor + 28, ROOM_MAX_Z - ROOM_WALL_DEPTH),
        materials::MIRROR,
        63,
    );

    // Flammable props for the flamethrower + dynamic smoke test: a stack of
    // wooden crates and a concrete pillar. The pillar exists so a plume from a
    // crate has something to flow around; both rest on the floor, so the
    // no-floating check still passes.
    fill_box_vox(
        &mut world,
        IVec3::new(1096, floor, 1072),
        IVec3::new(1112, floor + 12, 1088),
        materials::WOOD,
        40,
    );
    fill_box_vox(
        &mut world,
        IVec3::new(1096, floor + 12, 1072),
        IVec3::new(1108, floor + 20, 1084),
        materials::WOOD,
        38,
    );
    fill_box_vox(
        &mut world,
        IVec3::new(1116, floor, 1056),
        IVec3::new(1128, floor + 32, 1068),
        materials::CONCRETE,
        WHITE_TINT,
    );

    for (x, y, z) in ROOM_LAMPS {
        fill_box_vox(
            &mut world,
            IVec3::new(x, y, z),
            IVec3::new(x + 2, y + 2, z + 2),
            // `studio_palette` keeps index 17 non-emissive for the white walls,
            // so the fixture uses the palette's warm emissive entry instead.
            materials::EMISSIVE,
            63,
        );
    }

    world
}

/// World-space point-light positions of the two ceiling lamps, 0.2 m below the
/// emissive block so the emitter is not inside the light origin.
#[must_use]
pub fn test_room_lamp_positions() -> [Vec3; 2] {
    ROOM_LAMPS.map(|(x, y, z)| {
        Vec3::new(
            (x + 1) as f32 * VOXEL_SIZE_M,
            y as f32 * VOXEL_SIZE_M - 0.2,
            (z + 1) as f32 * VOXEL_SIZE_M,
        )
    })
}

/// Interior extent of the test room in metres, for scenario framing.
#[must_use]
pub fn test_room_bounds() -> (Vec3, Vec3) {
    (
        Vec3::new(ROOM_MIN_X_M, FLOOR_TOP_M, ROOM_MIN_Z as f32 * VOXEL_SIZE_M),
        Vec3::new(
            ROOM_MAX_X_M,
            ROOM_CEILING as f32 * VOXEL_SIZE_M,
            ROOM_MAX_Z_M,
        ),
    )
}

#[derive(Clone, Copy)]
enum WallFace {
    South,
    North,
    West,
    East,
}

impl WallFace {
    const ALL: [Self; 4] = [Self::South, Self::North, Self::West, Self::East];

    fn span(self) -> std::ops::Range<i32> {
        match self {
            Self::South | Self::North => HOUSE_MIN_X..HOUSE_MAX_X,
            Self::West | Self::East => HOUSE_MIN_Z..HOUSE_MAX_Z,
        }
    }

    /// `depth` counts inward from the outdoor face: 0 is the face voxel, 1 the
    /// backing voxel, -1 an optional protrusion.
    fn position(self, s: i32, depth: i32, y: i32) -> IVec3 {
        match self {
            Self::South => IVec3::new(s, y, HOUSE_MAX_Z - 1 - depth),
            Self::North => IVec3::new(s, y, HOUSE_MIN_Z + depth),
            Self::West => IVec3::new(HOUSE_MIN_X + depth, y, s),
            Self::East => IVec3::new(HOUSE_MAX_X - 1 - depth, y, s),
        }
    }
}

/// One wall layer: a 2-voxel backing course plus a relief face. Brick cells are
/// 2 voxels long and 1 voxel high in a running bond; a 1-voxel mortar joint
/// separates them, and every fourth course is a recessed bed joint. Depth is
/// varied per brick by +-1 voxel, which is the smallest relief the 12.5 cm grid
/// can express.
fn build_brick_wall(
    world: &mut VoxelWorld,
    seed: u32,
    face: WallFace,
    wall_id: u32,
    y0: i32,
    y1: i32,
) {
    let span = face.span();
    for y in y0..y1 {
        let course = y - FLOOR_TOP_VOXEL;
        let shift = course & 1;
        let bed_joint = course.rem_euclid(4) == 3;
        for s in span.clone() {
            let along = s - span.start + shift;
            let joint = along.rem_euclid(3) == 2;
            if joint || bed_joint {
                // Mortar is one voxel RECESSED behind the brick faces. The old
                // version filled both depth 0 and 1, so joints sat flush and
                // the only relief was a sparse random stud (the "polka dot"
                // facade). A uniform recess gives a real 3D brick lay.
                let p = face.position(s, 1, y);
                world.set_voxel(p.x, p.y, p.z, Voxel::new(HOUSE_MORTAR, 34));
                continue;
            }
            let brick_id = along.div_euclid(3) as u32;
            let h = hash_u32(
                seed,
                wall_id.wrapping_mul(131).wrapping_add(brick_id),
                y,
                0x51ed,
            );
            let tint = (12 + (h & 0x1f)) as u8;
            let brick = Voxel::new(HOUSE_BRICK, tint);
            let p = face.position(s, 1, y);
            world.set_voxel(p.x, p.y, p.z, brick);

            // Every brick is one voxel proud of the recessed mortar, so the
            // facade reads as courses of real bricks instead of isolated studs.
            let p = face.position(s, 0, y);
            world.set_voxel(p.x, p.y, p.z, brick);
            // Rare chipped corner: take the proud face voxel off a brick. The
            // backing course stays, so the wall never opens into the room.
            let chip = hash_u32(seed ^ 0x9e37_79b9, wall_id, s, y as u32) % 100;
            if chip < 1 {
                let p = face.position(s, 0, y);
                world.set_voxel(p.x, p.y, p.z, Voxel::AIR);
            }
        }
    }
}

/// Doors and windows are through-openings carved last, so the frame and the
/// brick relief cannot close them again.
fn carve_openings(world: &mut VoxelWorld) {
    for (x0, x1, y0, y1, z0, z1) in [
        // South facade: front door and a ground-floor window.
        (1004, 1012, 8, 25, HOUSE_MAX_Z - 3, HOUSE_MAX_Z + 1),
        (1032, 1044, 16, 26, HOUSE_MAX_Z - 3, HOUSE_MAX_Z + 1),
        // South, upper floor.
        (988, 1000, 40, 50, HOUSE_MAX_Z - 3, HOUSE_MAX_Z + 1),
        (1036, 1048, 40, 50, HOUSE_MAX_Z - 3, HOUSE_MAX_Z + 1),
        // North facade: back door and a ground-floor window.
        (1040, 1048, 8, 25, HOUSE_MIN_Z - 1, HOUSE_MIN_Z + 3),
        (996, 1008, 16, 26, HOUSE_MIN_Z - 1, HOUSE_MIN_Z + 3),
        (988, 1000, 40, 50, HOUSE_MIN_Z - 1, HOUSE_MIN_Z + 3),
        (1036, 1048, 40, 50, HOUSE_MIN_Z - 1, HOUSE_MIN_Z + 3),
        // West and east walls, one window per floor.
        (HOUSE_MIN_X - 1, HOUSE_MIN_X + 3, 16, 26, 1000, 1012),
        (HOUSE_MIN_X - 1, HOUSE_MIN_X + 3, 40, 50, 1030, 1042),
        (HOUSE_MAX_X - 3, HOUSE_MAX_X + 1, 16, 26, 1000, 1012),
        (HOUSE_MAX_X - 3, HOUSE_MAX_X + 1, 40, 50, 1030, 1042),
    ] {
        fill_box_vox(
            world,
            IVec3::new(x0, y0, z0),
            IVec3::new(x1, y1, z1),
            materials::AIR,
            0,
        );
    }
}

/// Deterministic integer hash. Stable across platforms and runs.
fn hash_u32(seed: u32, a: u32, b: i32, c: u32) -> u32 {
    let mut h = seed ^ a.wrapping_mul(0x9e37_79b1) ^ (b as u32).wrapping_mul(0x85eb_ca6b) ^ c;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

fn vox_min(position: Vec3) -> IVec3 {
    (position / VOXEL_SIZE_M).floor().as_ivec3()
}

fn vox_max(position: Vec3) -> IVec3 {
    (position / VOXEL_SIZE_M).ceil().as_ivec3()
}

fn fill_box_vox(world: &mut VoxelWorld, min: IVec3, max: IVec3, material: u16, tint: u8) {
    for z in min.z..max.z {
        for y in min.y..max.y {
            for x in min.x..max.x {
                world.set_voxel(x, y, z, Voxel::new(material, tint));
            }
        }
    }
}

fn fill_box_m(world: &mut VoxelWorld, min: Vec3, max: Vec3, material: u16, tint: u8) {
    fill_box_vox(world, vox_min(min), vox_max(max), material, tint);
}

/// Camera spawn for the test arena, facing the screens to the north.
pub fn spawn_position() -> Vec3 {
    Vec3::new(128.0, 10.0, 180.0)
}

pub fn spawn_yaw() -> f32 {
    0.0
}

pub fn spawn_pitch() -> f32 {
    -0.12
}

/// Voxel coordinate of a metre-space point, for debug output.
pub fn voxel_coord(position: Vec3) -> IVec3 {
    vox_min(position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BRICK_DIM;
    use crate::region::{REGION_BRICKS, REGION_VOXELS};
    use crate::world::{REGIONS_X, REGIONS_Y, REGIONS_Z};

    /// Visits every solid voxel through the sparse brick structure. Cheap
    /// enough for tests and avoids scanning the empty 256 m world.
    fn visit_solid(world: &VoxelWorld, mut visit: impl FnMut(IVec3, Voxel)) {
        for rz in 0..REGIONS_Z {
            for ry in 0..REGIONS_Y {
                for rx in 0..REGIONS_X {
                    let Some(region) = world.region(rx, ry, rz) else {
                        continue;
                    };
                    for (slot, brick) in region.iter_non_empty() {
                        let bx = (slot % REGION_BRICKS) * BRICK_DIM;
                        let by = (slot / REGION_BRICKS % REGION_BRICKS) * BRICK_DIM;
                        let bz = (slot / (REGION_BRICKS * REGION_BRICKS)) * BRICK_DIM;
                        for lz in 0..BRICK_DIM {
                            for ly in 0..BRICK_DIM {
                                for lx in 0..BRICK_DIM {
                                    let voxel = brick.get(lx, ly, lz);
                                    if !voxel.is_air() {
                                        let position = IVec3::new(
                                            rx as i32 * REGION_VOXELS as i32
                                                + bx as i32
                                                + lx as i32,
                                            ry as i32 * REGION_VOXELS as i32
                                                + by as i32
                                                + ly as i32,
                                            rz as i32 * REGION_VOXELS as i32
                                                + bz as i32
                                                + lz as i32,
                                        );
                                        visit(position, voxel);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scene_is_deterministic() {
        let first = build_test_arena();
        let second = build_test_arena();
        assert_eq!(first.deterministic_hash(), second.deterministic_hash());
        // The 128 x 128 m floor alone touches 64 x 64 bricks (bricks are 2 m),
        // so 4096 is the hard lower bound for any full-arena floor; a 3000
        // brick budget is physically impossible. The walls and props add
        // roughly 700 more.
        assert!(
            first.brick_count() < 5_200,
            "brick budget blown: {}",
            first.brick_count()
        );
        assert!(
            first.solid_voxels() < 5_000_000,
            "solid budget blown: {}",
            first.solid_voxels()
        );
    }

    #[test]
    fn spawn_stands_on_the_floor() {
        let world = build_test_arena();
        let floor = world.voxel(1024, 7, 1440);
        assert!(floor.is_solid(), "no floor under the spawn");
        assert_eq!(
            floor.material(),
            WHITE,
            "spawn floor is not the white studio material"
        );
        assert!(
            world.voxel(1024, FLOOR_TOP_VOXEL, 1440).is_air(),
            "spawn floor is not flush"
        );
        let hit = world
            .raycast(Vec3::new(128.0, 2.0, 180.0), Vec3::NEG_Y, 4.0)
            .expect("spawn ray must hit the floor");
        assert!(
            (hit.distance - 1.0).abs() < 0.2,
            "spawn floor at {:.3} m below the feet, expected 1.0",
            hit.distance
        );
    }

    #[test]
    fn spawn_point_is_clear_for_the_player() {
        let world = build_test_arena();
        let feet = voxel_coord(Vec3::new(128.0, 2.0, 180.0));
        // Floor to 2.5 m above the spawn point, and 0.5 m to each side.
        let reach = 4; // 0.5 m
        for dx in -reach..=reach {
            for dz in -reach..=reach {
                for y in feet.y..(feet.y + 16) {
                    let voxel = world.voxel(feet.x + dx, y, feet.z + dz);
                    assert!(
                        voxel.is_air(),
                        "spawn bubble blocked at ({dx}, {dz}), y = {y}: {voxel:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn white_walls_cast_shadows() {
        let world = build_test_arena();
        let mut high_white = 0u32;
        visit_solid(&world, |position, voxel| {
            if voxel.material() == WHITE && position.y > 24 {
                high_white += 1;
            }
        });
        assert!(
            high_white > 20_000,
            "not enough white surface above 3 m: {high_white}"
        );
        // A point inside the near screen still has its white studio material.
        let sample = world.voxel(880, 32, 1250); // 110 m, 4 m, 156.25 m
        assert_eq!(sample.material(), WHITE);
        // And the near screen blocks a horizontal ray at chest height.
        let hit = world
            .raycast(Vec3::new(118.0, 2.0, 150.0), Vec3::Z, 8.0)
            .expect("screen should occlude the ray");
        assert!(hit.distance < 8.0);
    }

    #[test]
    fn coloured_cubes_and_glass_exist() {
        let world = build_test_arena();
        let mut seen = [false; 18];
        visit_solid(&world, |_, voxel| {
            if let Some(flag) = seen.get_mut(voxel.material() as usize) {
                *flag = true;
            }
        });
        let distinct = seen.iter().filter(|flag| **flag).count();
        assert!(
            distinct >= 3,
            "expected at least three materials, found {distinct}"
        );
        assert!(seen[WHITE as usize], "no white studio surface in the scene");
        assert!(
            seen[materials::GLASS as usize],
            "no glass in the scene (material 6)"
        );
        for material in [
            materials::BRICK,
            materials::WOOD,
            materials::METAL,
            materials::STONE,
            materials::EMISSIVE,
        ] {
            assert!(
                seen[material as usize],
                "material {} is missing from the scene",
                world.palette().material(material).name
            );
        }
    }

    #[test]
    fn arena_stays_within_bounds() {
        let world = build_test_arena();
        let mut solid = 0u64;
        visit_solid(&world, |position, _| {
            assert!(
                (PLAY_MIN..PLAY_MAX).contains(&position.x)
                    && (PLAY_MIN..PLAY_MAX).contains(&position.z),
                "voxel {position:?} is outside the 64..192 m play area"
            );
            solid += 1;
        });
        assert!(solid > 1_000_000, "scene appears to be empty: {solid}");
    }

    #[test]
    fn house_scene_is_deterministic_and_within_budget() {
        let first = build_house();
        let second = build_house();
        assert_eq!(first.deterministic_hash(), second.deterministic_hash());
        // The 64 m plate alone is 32 x 32 bricks, and the house adds a few
        // hundred; anything above 2000 means the plate grew back to full size.
        assert!(
            first.brick_count() < 2_000,
            "house brick budget blown: {}",
            first.brick_count()
        );
        assert!(first.solid_voxels() < 2_000_000);
    }

    #[test]
    fn house_plate_is_a_quarter_of_the_studio_floor() {
        let world = build_house();
        assert!(world.voxel(768, 7, 768).is_solid());
        assert!(world.voxel(1279, 7, 1279).is_solid());
        assert!(!world.voxel(767, 7, 768).is_solid());
        assert!(!world.voxel(1280, 7, 768).is_solid());
        assert!(!world.voxel(768, FLOOR_TOP_VOXEL, 768).is_solid());
        // 512 x 512 voxels = 64 x 64 m = 4096 m2, a quarter of the studio's
        // 128 x 128 m = 16384 m2.
        assert_eq!(HOUSE_PLATE_MAX - HOUSE_PLATE_MIN, 512);
    }

    #[test]
    fn house_structure_is_grounded() {
        let world = build_house();
        let (floating, total) = world.count_unanchored_above(GROUND_ANCHOR_VOXEL);
        assert_eq!(
            floating, 0,
            "{floating} of {total} voxels above the plate are disconnected"
        );
        assert!(total > 50_000, "house appears to be empty: {total}");
    }

    #[test]
    fn house_has_brick_relief_mortar_and_a_concrete_frame() {
        let world = build_house();
        let mut bricks = 0u32;
        let mut mortar = 0u32;
        let mut concrete = 0u32;
        let mut tints = [false; 64];
        visit_solid(&world, |position, voxel| match voxel.material() {
            HOUSE_BRICK => {
                bricks += 1;
                if position.y < HOUSE_SLAB1 {
                    tints[(voxel.tint() & 0x3f) as usize] = true;
                }
            }
            HOUSE_MORTAR => mortar += 1,
            HOUSE_FRAME => concrete += 1,
            _ => {}
        });
        assert!(bricks > 5_000, "not enough brick: {bricks}");
        assert!(mortar > 500, "no recessed mortar joints: {mortar}");
        assert!(concrete > 5_000, "no concrete frame: {concrete}");
        let variants = tints.iter().filter(|seen| **seen).count();
        assert!(
            variants >= 16,
            "brick colour variation is too narrow: {variants} variants"
        );
        assert_eq!(
            world.palette().len(),
            24,
            "house palette entries missing (20 base + red/blue paint + mirror/clear glass)"
        );
    }

    #[test]
    fn house_doors_and_windows_are_openings() {
        let world = build_house();
        // Front door (south), back door (north).
        assert!(world.voxel(1008, 12, HOUSE_MAX_Z - 1).is_air());
        assert!(world.voxel(1044, 12, HOUSE_MIN_Z).is_air());
        // Lintel above the front door and wall above the window sill. Check the
        // backing layer at z = HOUSE_MAX_Z - 2: the face voxel can be a recessed
        // joint or a chipped brick, the backing course is always solid.
        assert!(world.voxel(1008, 25, HOUSE_MAX_Z - 1).is_solid());
        assert!(world.voxel(1038, 20, HOUSE_MAX_Z - 1).is_air());
        assert!(world.voxel(1038, 26, HOUSE_MAX_Z - 2).is_solid());
        // The stairwell is cut through the inter-floor slab.
        assert!(world.voxel(1000, HOUSE_SLAB1, 1026).is_air());
        assert!(world.voxel(1000, HOUSE_SLAB1, 1044).is_solid());
    }

    #[test]
    fn house_stair_reaches_the_upper_floor() {
        let world = build_house();
        // The bottom step's top face is at y = 10 (the fill excludes the max),
        // and the top step's top face is flush with the upper slab.
        assert!(world.voxel(STAIR_MIN_X + 1, 9, STAIR_MIN_Z + 1).is_solid());
        assert!(world.voxel(STAIR_MIN_X + 1, 10, STAIR_MIN_Z + 1).is_air());
        assert!(world.voxel(STAIR_MAX_X - 1, 33, STAIR_MIN_Z + 1).is_solid());
        assert!(
            world
                .voxel(STAIR_MAX_X - 1, HOUSE_SLAB1, STAIR_MIN_Z + 1)
                .is_solid()
        );
    }

    #[test]
    fn test_room_has_painted_walls_windows_and_glass() {
        let world = build_test_room();
        assert_eq!(world.palette().len(), 24);
        assert_eq!(
            world.palette().material(materials::RED_PAINT).name,
            "red_paint"
        );
        assert_eq!(
            world.palette().material(materials::BLUE_PAINT).name,
            "blue_paint"
        );
        assert_eq!(world.palette().material(materials::MIRROR).name, "mirror");
        assert_eq!(
            world.palette().material(materials::GLASS_CLEAR).name,
            "glass_clear"
        );
        // West wall is red, east wall is blue at mid-height, away from windows.
        assert_eq!(
            world.voxel(ROOM_MIN_X, 30, 1080).material(),
            materials::RED_PAINT
        );
        assert_eq!(
            world.voxel(ROOM_MAX_X - 1, 30, 1080).material(),
            materials::BLUE_PAINT
        );
        // Open window in the north wall and in the west wall.
        assert!(world.voxel(1060, 28, ROOM_MIN_Z).is_air());
        assert!(world.voxel(ROOM_MIN_X, 28, 1060).is_air());
        // Free-standing glass pane near the south wall.
        assert_eq!(
            world.voxel(1050, 20, 1104).material(),
            materials::GLASS_CLEAR
        );
        // Wall mirror between the north windows.
        assert_eq!(
            world
                .voxel(1084, 20, ROOM_MAX_Z - ROOM_WALL_DEPTH - 1)
                .material(),
            materials::MIRROR
        );
        // The ceiling fixture must be emissive: `studio_palette` makes index 17
        // non-emissive for the walls, so a white fixture would read as a dark box.
        let (lamp_x, lamp_y, lamp_z) = ROOM_LAMPS[0];
        assert_eq!(
            world.voxel(lamp_x, lamp_y, lamp_z).material(),
            materials::EMISSIVE
        );
        // White plate below and white ceiling above.
        assert!(world.voxel(1088, FLOOR_TOP_VOXEL - 1, 1088).is_solid());
        assert!(world.voxel(1088, ROOM_CEILING, 1088).is_solid());
        let (floating, total) = world.count_unanchored_above(GROUND_ANCHOR_VOXEL);
        assert_eq!(
            floating, 0,
            "{floating} of {total} room voxels are disconnected"
        );
        assert!(total > 10_000, "test room appears to be empty: {total}");
    }

    #[test]
    fn test_room_scene_kind_round_trips() {
        assert_eq!(SceneKind::parse("testroom"), Some(SceneKind::TestRoom));
        assert_eq!(SceneKind::parse("Test_Room"), Some(SceneKind::TestRoom));
        assert_eq!(SceneKind::TestRoom.name(), "testroom");
        let (spawn, _, _) = SceneKind::TestRoom.spawn();
        assert!(
            spawn.x > 128.0 && spawn.x < 144.0 && spawn.z > 128.0 && spawn.z < 144.0,
            "spawn {spawn} is outside the room"
        );
    }
}
