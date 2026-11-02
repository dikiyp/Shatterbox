// Phase 2 G-buffer pass: primary voxel DDA with hierarchical skipping.
//
// Outputs albedo + baked voxel AO, world normal + roughness, material
// parameters and linear depth. Lighting is deferred into `lighting.wgsl`.
//
// The traversal is monotonic by construction: every coarse skip advances the
// cell index along the axis that produced the smallest boundary time, so a
// ray that hugs a brick/group boundary cannot stall or step backwards. That
// was the source of the one-frame "transparent stripe" artefacts in Phase 1.
//
// Performance notes (RTX 3050, 1600x900 internal):
// - No divisions inside the loop: block boundaries are derived from the
//   per-voxel `t_max`/`t_delta` already maintained by the DDA.
// - The brick index and the occupancy words are cached and only re-fetched
//   when the ray enters another brick, so the common case is one 3D-texture
//   read per 2 m of travel instead of one per voxel step. Empty space is
//   skipped at 32 m / 8 m / 4 m / 2 m granularity before the brick is touched.
// - Voxel AO only reads occupancy bits of the hit brick (cache-friendly)
//   instead of re-sampling the brick index for eight neighbours.

struct Frame {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    prev_camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    // rgb = sun/moon radiance, w = normalized daylight factor 0..1
    sun_color: vec4<f32>,
    world_min: vec4<f32>,
    world_max: vec4<f32>,
    // x = time, y = frame index, z = jitter x, w = jitter y (internal pixels)
    params: vec4<f32>,
    params2: vec4<f32>,
    // x = effect bitmask (1 shadows, 2 ssao, 4 ssgi, 8 vct, 16 taa, 32 fog,
    // 64 ssr).
    params3: vec4<f32>,
    internal_size: vec4<f32>,
    light_pos_radius: array<vec4<f32>, 12>,
    light_color: array<vec4<f32>, 12>,
    // xyz = emitter axis, w = half extent (negative = rectangle half size).
    light_axis: array<vec4<f32>, 12>,
    // World-space AABB of every active debris piece (see `cast_debris`).
    debris_min: vec4<f32>,
    debris_max: vec4<f32>,
};

struct GpuMaterial {
    albedo_rough: vec4<f32>,
    params: vec4<f32>,
};

struct Hit {
    valid: bool,
    voxel: vec3<i32>,
    normal: vec3<f32>,
    raw: u32,
    id: u32,
    distance: f32,
};

// One rigid voxel piece. `data` packs the first index in `debris_voxels` and
// the local grid dimensions. The model is rigid (rotation + translation), so a
// world ray can be transformed into the piece's local space and traced there.
struct DebrisInstance {
    model: mat4x4<f32>,
    inverse_model: mat4x4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    data: vec4<u32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var brick_index_tex: texture_3d<u32>;
@group(0) @binding(2) var<storage, read> bricks: array<u32>;
@group(0) @binding(3) var<storage, read> masks: array<u32>;
@group(0) @binding(4) var<storage, read> palette: array<GpuMaterial>;
@group(0) @binding(5) var group_occupancy_tex: texture_3d<u32>;
@group(0) @binding(10) var region_occupancy_tex: texture_3d<u32>;
@group(0) @binding(11) var<storage, read> debris: array<DebrisInstance>;
@group(0) @binding(12) var<storage, read> debris_voxels: array<u32>;
// Compact stand-in for `debris`: xyzw = world sphere centre + radius. Reading
// the 112-byte instance record (two mat4s) for a piece the ray never touches
// dominated the G-buffer once ~250 pieces were alive, so `cast_debris` first
// rejects with this 16-byte entry and only loads the matrices on a hit
// candidate. Effect bit 512 switches back to the old straight loop.
@group(0) @binding(13) var<storage, read> debris_spheres: array<vec4<f32>>;
// 512-bit candidate mask per 8x8 pixel tile, 16 u32 words per tile, written by
// `debris_cull.wgsl`. Effect bit 1024.
@group(0) @binding(16) var<storage, read> debris_tiles: array<u32>;
@group(0) @binding(6) var out_albedo: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var out_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var out_material: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(9) var out_depth: texture_storage_2d<r32float, write>;

const VOXEL_SIZE: f32 = 0.125;
const BRICK_DIM: i32 = 16;
const BRICK_WORDS: u32 = 2048u;
const GROUP_BRICKS: i32 = 4;
/// A 192 m ray crosses at most ~200 cells with the hierarchical skips; the
/// guard is `gpu.rs::grazing_rays_fit_the_shader_step_budget`. Halving the old
/// 1024 bound costs nothing measurable and caps pathological grazes.
const MAX_STEPS: i32 = 512;
const MASK_WORDS: u32 = 147u;
const MASK8_OFFSET: u32 = 128u;
const MASK4_OFFSET: u32 = 144u;
const MASK2_OFFSET: u32 = 146u;
/// One coarse occupancy group covers 4 bricks (8 m) per axis.
const GROUP_VOXELS: i32 = 64;
/// One region is 32 m per axis, i.e. 16 bricks.
const REGION_VOXELS: i32 = 256;
const MATERIAL_CONCRETE: u32 = 1u;
const MATERIAL_STONE: u32 = 2u;
const MATERIAL_BRICK: u32 = 3u;
const MATERIAL_WOOD: u32 = 4u;
const MATERIAL_METAL: u32 = 5u;
const MATERIAL_GLASS: u32 = 6u;
const MATERIAL_SAND: u32 = 7u;
const MATERIAL_GRASS: u32 = 8u;
const MATERIAL_EMISSIVE: u32 = 10u;
const MATERIAL_HAZARD: u32 = 11u;
const MATERIAL_WET_CONCRETE: u32 = 13u;
const MATERIAL_MIRROR: u32 = 22u;
const MATERIAL_GLASS_CLEAR: u32 = 23u;

fn slab_axis(o: f32, d: f32, mn: f32, mx: f32) -> vec2<f32> {
    if (abs(d) < 1e-8) {
        if (o < mn || o > mx) {
            return vec2<f32>(1.0, -1.0);
        }
        return vec2<f32>(-1e30, 1e30);
    }
    let inv = 1.0 / d;
    let t0 = (mn - o) * inv;
    let t1 = (mx - o) * inv;
    return vec2<f32>(min(t0, t1), max(t0, t1));
}

fn intersect_world(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    let x = slab_axis(ro.x, rd.x, frame.world_min.x, frame.world_max.x);
    let y = slab_axis(ro.y, rd.y, frame.world_min.y, frame.world_max.y);
    let z = slab_axis(ro.z, rd.z, frame.world_min.z, frame.world_max.z);
    return vec2<f32>(max(max(x.x, y.x), z.x), min(min(x.y, y.y), z.y));
}

// Absolute ray parameter of the boundary of the `size`-voxel cell that
// contains `cell`, in the direction of travel. Derived from the next-voxel
// boundary `t_max` so the loop needs no divisions.
fn block_boundary(t_max: f32, t_delta: f32, cell: i32, step: i32, size: i32) -> f32 {
    let local = cell - (cell / size) * size;
    // Forward rays leave the block after the remaining cells; backward rays
    // leave it after `local` cells. Using the forward form for both skips solid
    // voxels behind the ray and reports a hit distance far inside the surface.
    let remaining = select(size - 1 - local, local, step < 0);
    return t_max + t_delta * f32(remaining);
}

fn raw_voxel_in_brick(cell: vec3<i32>, id: u32) -> u32 {
    let brick_origin = (cell / BRICK_DIM) * BRICK_DIM;
    let local = cell - brick_origin;
    let linear = local.x + BRICK_DIM * local.y + BRICK_DIM * BRICK_DIM * local.z;
    let word_index = (id - 1u) * BRICK_WORDS + u32(linear / 2);
    let word = bricks[word_index];
    let shift = u32(linear & 1) * 16u;
    return (word >> shift) & 0xffffu;
}

// Normal of the face the ray crossed to enter the current cell. The DDA knows
// the crossing axis exactly, so this stays exact for grazing rays where
// reconstructing the face from the hit point or from the entry parameters
// silently picks the wrong axis. A wrong normal is not cosmetic: it flips NВ·L
// and the AO basis per voxel, which reads as black speckle on a flat floor.
fn step_normal(step: vec3<i32>, axis: i32) -> vec3<f32> {
    if (axis == 0) {
        return vec3<f32>(-f32(step.x), 0.0, 0.0);
    }
    if (axis == 1) {
        return vec3<f32>(0.0, -f32(step.y), 0.0);
    }
    return vec3<f32>(0.0, 0.0, -f32(step.z));
}

fn init_t_max(cell: vec3<i32>, step: vec3<i32>, ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    var t = vec3<f32>(1e30);
    if (abs(rd.x) > 1e-9) {
        let face = select(f32(cell.x), f32(cell.x + 1), step.x > 0);
        t.x = (face * VOXEL_SIZE - ro.x) / rd.x;
    }
    if (abs(rd.y) > 1e-9) {
        let face = select(f32(cell.y), f32(cell.y + 1), step.y > 0);
        t.y = (face * VOXEL_SIZE - ro.y) / rd.y;
    }
    if (abs(rd.z) > 1e-9) {
        let face = select(f32(cell.z), f32(cell.z + 1), step.z > 0);
        t.z = (face * VOXEL_SIZE - ro.z) / rd.z;
    }
    return t;
}

fn min_boundary3(a: f32, b: f32, c: f32) -> vec2<f32> {
    if (a <= b && a <= c) {
        return vec2<f32>(a, 0.0);
    }
    if (b <= c) {
        return vec2<f32>(b, 1.0);
    }
    return vec2<f32>(c, 2.0);
}

// Moves the cell to the one containing the point just past `t`, forcing the
// crossing axis to advance at least one cell. This is what keeps grazing rays
// monotonic.
fn jump_cell(
    cell: vec3<i32>,
    step: vec3<i32>,
    ro: vec3<f32>,
    rd: vec3<f32>,
    t: f32,
    axis: i32,
) -> vec3<i32> {
    let sample = ro + rd * (t + 1e-5);
    var next = vec3<i32>(floor(sample / VOXEL_SIZE));
    if (axis == 0 && next.x == cell.x) {
        next.x += step.x;
    } else if (axis == 1 && next.y == cell.y) {
        next.y += step.y;
    } else if (axis == 2 && next.z == cell.z) {
        next.z += step.z;
    }
    return next;
}

// Tests one occupancy bit of a 16Ві brick. `size` is the block edge in voxels
// (8, 4 or 2) and `offset` the matching mask base inside `MASK_WORDS`.
fn block_occupied(mask_base: u32, offset: u32, local: vec3<i32>, size: i32) -> bool {
    let block = local / vec3<i32>(size);
    let dim = 16 / size;
    let bit = u32(block.x + dim * (block.y + dim * block.z));
    let word = masks[mask_base + offset + (bit >> 5u)];
    return ((word >> (bit & 31u)) & 1u) != 0u;
}

// Traverses the brick map. `want_hit == false` stops at the first solid voxel
// and is used for shadow rays, where only occlusion matters. `ignore_cell` is
// the surface voxel a shadow ray starts in, which must not shadow itself.
fn cast_ray(
    ro: vec3<f32>,
    rd: vec3<f32>,
    max_distance: f32,
    want_hit: bool,
    ignore_cell: vec3<i32>,
    has_ignore: bool,
) -> Hit {
    var result = Hit(false, vec3<i32>(0), vec3<f32>(0.0), 0u, 0u, 0.0);
    if (dot(rd, rd) < 1e-12) {
        return result;
    }
    let max_cell = vec3<i32>(frame.world_max.xyz / VOXEL_SIZE) - vec3<i32>(1);
    // A ray that leaves the world can never hit geometry, so the DDA gets a hard
    // exit distance and breaks there instead of grinding MAX_STEPS into the sky.
    let bounds = intersect_world(ro, rd);
    if (bounds.y <= 0.0 || bounds.x > bounds.y) {
        return result;
    }
    let t_exit = min(max_distance, bounds.y);
    // The camera normally sits inside the arena, so the slab entry is only
    // needed when it has been flown outside the world.
    let inside_world = all(ro >= frame.world_min.xyz) && all(ro < frame.world_max.xyz);
    var distance = 1e-5;
    if (!inside_world) {
        distance = max(bounds.x, 0.0) + 1e-5;
    }
    var cell = clamp(
        vec3<i32>(floor((ro + rd * distance) / VOXEL_SIZE)),
        vec3<i32>(0),
        max_cell,
    );
    let step = vec3<i32>(
        select(-1, 1, rd.x >= 0.0),
        select(-1, 1, rd.y >= 0.0),
        select(-1, 1, rd.z >= 0.0),
    );
    let t_delta = vec3<f32>(
        select(1e30, VOXEL_SIZE / abs(rd.x), abs(rd.x) > 1e-8),
        select(1e30, VOXEL_SIZE / abs(rd.y), abs(rd.y) > 1e-8),
        select(1e30, VOXEL_SIZE / abs(rd.z), abs(rd.z) > 1e-8),
    );
    var t_max = max(init_t_max(cell, step, ro, rd), vec3<f32>(distance));

    // Cached lookups: brick index per 2 m brick, mask16/8 words per 16/8 voxels,
    // and the coarse region/group/4 m occupancy bytes.
    var brick_coord = vec3<i32>(-1);
    var id = 0u;
    var mask_base = 0u;
    var mask_word_index = 0xffffffffu;
    var mask_word_value = 0u;
    var region_coord = vec3<i32>(-1);
    var region_empty = false;
    var group_coord = vec3<i32>(-1);
    var group_empty = false;
    // Axis crossed by the last step, i.e. the face the current cell was
    // entered through. Coarse jumps report their own axis.
    var entry_axis = 1;

    for (var i = 0; i < MAX_STEPS; i = i + 1) {
        if (distance >= t_exit) {
            break;
        }
        if (any(cell < vec3<i32>(0)) || any(cell > max_cell)) {
            break;
        }

        // Coarse empty-space skip before touching the brick map: air is the
        // common case for long primary rays, and a 32 m region jump costs one
        // texture read instead of sixteen 2 m brick reads.
        let region = cell / REGION_VOXELS;
        if (any(region != region_coord)) {
            region_coord = region;
            region_empty = textureLoad(region_occupancy_tex, region, 0).r == 0u;
        }
        var coarse = 0;
        if (region_empty) {
            coarse = REGION_VOXELS;
        } else {
            let group = cell / GROUP_VOXELS;
            if (any(group != group_coord)) {
                group_coord = group;
                group_empty = textureLoad(group_occupancy_tex, group, 0).r == 0u;
            }
            if (group_empty) {
                coarse = GROUP_VOXELS;
            }
        }
        if (coarse != 0) {
            let boundary = min_boundary3(
                block_boundary(t_max.x, t_delta.x, cell.x, step.x, coarse),
                block_boundary(t_max.y, t_delta.y, cell.y, step.y, coarse),
                block_boundary(t_max.z, t_delta.z, cell.z, step.z, coarse),
            );
            if (boundary.x >= t_exit || boundary.x >= 1e29) {
                break;
            }
            let previous = cell;
            distance = boundary.x;
            cell = jump_cell(cell, step, ro, rd, distance, i32(boundary.y));
            entry_axis = i32(boundary.y);
            t_max = max(t_max + vec3<f32>((cell - previous) * step) * t_delta, vec3<f32>(distance));
            continue;
        }

        let coord = cell / BRICK_DIM;
        if (any(coord != brick_coord)) {
            brick_coord = coord;
            id = textureLoad(brick_index_tex, coord, 0).r;
            mask_base = (id - 1u) * MASK_WORDS;
            mask_word_index = 0xffffffffu;
        }

        if (id == 0u) {
            // Empty 2 m brick inside a non-empty group: step to the brick
            // boundary instead of per-voxel.
            let boundary = min_boundary3(
                block_boundary(t_max.x, t_delta.x, cell.x, step.x, BRICK_DIM),
                block_boundary(t_max.y, t_delta.y, cell.y, step.y, BRICK_DIM),
                block_boundary(t_max.z, t_delta.z, cell.z, step.z, BRICK_DIM),
            );
            if (boundary.x >= t_exit || boundary.x >= 1e29) {
                break;
            }
            let previous = cell;
            distance = boundary.x;
            cell = jump_cell(cell, step, ro, rd, distance, i32(boundary.y));
            entry_axis = i32(boundary.y);
            // Crossings are an arithmetic progression in the positive t
            // direction, so a backward cell advance still means later time:
            // advance by |cell - previous| crossings, not by the signed delta.
            t_max = max(t_max + vec3<f32>((cell - previous) * step) * t_delta, vec3<f32>(distance));
            continue;
        }

        // Non-empty brick: hierarchical empty-space skip inside the 16Ві brick.
        // `mask2` holds one bit per 8Ві voxel block (1 m), `mask4` per 4Ві
        // (0.5 m), `mask8` per 2Ві (0.25 m). Coarse levels are tested first, so
        // an open brick costs one bit test per metre and a dense one falls
        // straight through to the per-voxel mask16 check.
        let local = cell & vec3<i32>(15);
        var skip = 0;
        if (!block_occupied(mask_base, MASK2_OFFSET, local, 8)) {
            skip = 8;
        } else if (!block_occupied(mask_base, MASK4_OFFSET, local, 4)) {
            skip = 4;
        } else if (!block_occupied(mask_base, MASK8_OFFSET, local, 2)) {
            skip = 2;
        }
        if (skip != 0) {
            let boundary = min_boundary3(
                block_boundary(t_max.x, t_delta.x, cell.x, step.x, skip),
                block_boundary(t_max.y, t_delta.y, cell.y, step.y, skip),
                block_boundary(t_max.z, t_delta.z, cell.z, step.z, skip),
            );
            if (boundary.x >= t_exit || boundary.x >= 1e29) {
                break;
            }
            let previous = cell;
            distance = boundary.x;
            cell = jump_cell(cell, step, ro, rd, distance, i32(boundary.y));
            entry_axis = i32(boundary.y);
            // Crossings are an arithmetic progression in the positive t
            // direction, so a backward cell advance still means later time:
            // advance by |cell - previous| crossings, not by the signed delta.
            t_max = max(t_max + vec3<f32>((cell - previous) * step) * t_delta, vec3<f32>(distance));
            continue;
        }

        let bit = u32(local.x + BRICK_DIM * (local.y + BRICK_DIM * local.z));
        let word_index = bit >> 5u;
        if (word_index != mask_word_index) {
            mask_word_index = word_index;
            mask_word_value = masks[mask_base + word_index];
        }
        // Shadow rays start on the very surface they shade, so the surface
        // voxel itself must not count as an occluder. Ignoring only this cell
        // (and only right at the start) keeps thin-geometry shadows intact.
        let ignore = has_ignore && distance < 0.2 && all(cell == ignore_cell);
        if (!ignore && ((mask_word_value >> (bit & 31u)) & 1u) != 0u) {
            result.valid = true;
            result.distance = distance;
            if (want_hit) {
                result.voxel = cell;
                result.normal = step_normal(step, entry_axis);
                result.raw = raw_voxel_in_brick(cell, id);
                result.id = id;
            }
            return result;
        }

        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            distance = t_max.x;
            cell.x = cell.x + step.x;
            t_max.x = t_max.x + t_delta.x;
            entry_axis = 0;
        } else if (t_max.y <= t_max.z) {
            distance = t_max.y;
            cell.y = cell.y + step.y;
            t_max.y = t_max.y + t_delta.y;
            entry_axis = 1;
        } else {
            distance = t_max.z;
            cell.z = cell.z + step.z;
            t_max.z = t_max.z + t_delta.z;
            entry_axis = 2;
        }
    }

    return result;
}

// --- Stylised pixel detail -------------------------------------------------
// Each voxel is 12.5 cm, so per-voxel patterns read as chunky "pixel art"
// texels from a distance while staying perfectly voxel-aligned up close.

fn hash_u32(value: u32) -> u32 {
    var x = value;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash01(a: i32, b: i32, seed: u32) -> f32 {
    var h = hash_u32(bitcast<u32>(a) ^ 0x9e3779b9u);
    h = hash_u32(h ^ bitcast<u32>(b));
    h = hash_u32(h ^ (seed * 0x85ebca6bu));
    return f32(h & 0xffffu) / 65535.0;
}

fn face_uv(cell: vec3<i32>, normal: vec3<f32>) -> vec2<i32> {
    if (abs(normal.y) > 0.5) {
        return vec2<i32>(cell.x, cell.z);
    }
    if (abs(normal.x) > 0.5) {
        return vec2<i32>(cell.z, cell.y);
    }
    return vec2<i32>(cell.x, cell.y);
}

fn material_detail(material: u32, cell: vec3<i32>, normal: vec3<f32>) -> f32 {
    let uv = face_uv(cell, normal);
    let coarse = hash01(uv.x / 3, uv.y / 3, material);

    // Neon tubes must stay uniform: any per-voxel brightness variation reads as
    // dirt on a light source. Palette indices 14-17 are the coloured neon set.
    if (material >= 14u && material <= 17u) {
        return 1.0;
    }

    if (material == MATERIAL_BRICK) {
        let row = uv.y >> 2;
        let shifted = uv.x + (row & 1) * 4;
        if ((uv.y & 3) == 0 || (shifted % 8) == 0) {
            return 0.76;
        }
        return 0.90 + 0.16 * coarse;
    }
    if (material == MATERIAL_CONCRETE || material == MATERIAL_STONE) {
        var value = 0.90 + 0.14 * coarse;
        // 4 m slab joints: they give the floor a readable large scale instead of
        // a uniform field of 12.5 cm tiles.
        if ((uv.x & 31) == 0 || (uv.y & 31) == 0) {
            value = value * 0.87;
        }
        // Panel brightness drifts per 4 m slab so the grid is not the only cue.
        value = value * (0.94 + 0.12 * hash01(uv.x >> 5, uv.y >> 5, material + 31u));
        if (hash01(uv.x, uv.y, material + 5u) < 0.08) {
            value = value * 0.86;
        }
        return value;
    }
    if (material == MATERIAL_WET_CONCRETE) {
        var wet = 0.88 + 0.10 * coarse;
        if ((uv.x & 31) == 0 || (uv.y & 31) == 0) {
            wet = wet * 0.82;
        }
        wet = wet * (0.92 + 0.10 * hash01(uv.x >> 5, uv.y >> 5, 83u));
        return wet;
    }
    if (material == MATERIAL_WOOD) {
        let plank = f32((uv.y >> 2) & 1);
        let grain = f32((uv.x >> 1u) & 3i) / 3.0;
        return 0.85 + 0.07 * plank + 0.08 * grain + 0.05 * coarse;
    }
    if (material == MATERIAL_METAL) {
        return 0.94 + 0.08 * hash01(uv.x / 2, uv.y / 4, 17u);
    }
    if (material == MATERIAL_SAND) {
        return 0.89 + 0.15 * hash01(uv.x, uv.y, 23u);
    }
    if (material == MATERIAL_GRASS) {
        return 0.85 + 0.18 * hash01(uv.x / 2, uv.y / 2, 29u);
    }
    if (material == MATERIAL_HAZARD) {
        let stripe = (uv.x + uv.y) & 7;
        return select(0.32, 1.0, stripe < 4);
    }
    if (material == MATERIAL_EMISSIVE) {
        return 1.0;
    }
    if (material == MATERIAL_GLASS) {
        return 0.97 + 0.04 * coarse;
    }
    return 0.92 + 0.12 * hash01(uv.x, uv.y, 41u);
}

// 1 m block variation plus a 2-3 m "staining" field. Both are coarse enough
// never to alias, so they survive after the per-voxel detail has faded and stop
// large surfaces from turning into flat colour.
fn macro_variation(cell: vec3<i32>, material: u32) -> f32 {
    let block = cell / 8;
    let stain = hash01(cell.x / 24 + cell.z / 16, cell.y / 12, material + 57u);
    return 0.88 + 0.10 * hash01(block.x + block.y * 37, block.z, material + 101u) + 0.07 * stain;
}

// Cheap voxel crease AO. Four cardinal neighbours in the face plane are read
// from occupancy masks. A neighbour that crosses a 2 m brick boundary is looked
// up in that brick instead of counting as air: treating it as unoccluded left a
// one-voxel bright seam every 2 m, most visible on flat floors and walls.
fn voxel_ao(cell: vec3<i32>, normal: vec3<f32>, id: u32) -> f32 {
    let n = vec3<i32>(normal);
    var tangent = vec3<i32>(1, 0, 0);
    var bitangent = vec3<i32>(0, 0, 1);
    if (abs(normal.x) > 0.5) {
        tangent = vec3<i32>(0, 1, 0);
        bitangent = vec3<i32>(0, 0, 1);
    } else if (abs(normal.z) > 0.5) {
        tangent = vec3<i32>(1, 0, 0);
        bitangent = vec3<i32>(0, 1, 0);
    }
    let base = cell + n;
    let brick_coord = cell / BRICK_DIM;
    let brick_origin = brick_coord * BRICK_DIM;
    let mask_base = (id - 1u) * MASK_WORDS;
    var occluded = 0.0;
    // Four cardinal neighbours instead of all eight: the diagonal samples add
    // little contact darkening but double the occupancy reads per pixel.
    let cardinals = array<vec2<i32>, 4>(
        vec2<i32>(-1, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, -1),
        vec2<i32>(0, 1),
    );
    for (var k = 0; k < 4; k = k + 1) {
        let neighbour = base + tangent * cardinals[k].x + bitangent * cardinals[k].y;
        let neighbour_coord = neighbour / BRICK_DIM;
        var neighbour_id = id;
        var neighbour_origin = brick_origin;
        var neighbour_mask = mask_base;
        if (any(neighbour_coord != brick_coord)) {
            // Out-of-world coordinates read 0, which is an empty slot, so the
            // same branch also handles the arena border.
            neighbour_id = textureLoad(brick_index_tex, neighbour_coord, 0).r;
            if (neighbour_id == 0u) {
                continue;
            }
            neighbour_origin = neighbour_coord * BRICK_DIM;
            neighbour_mask = (neighbour_id - 1u) * MASK_WORDS;
        }
        let local = neighbour - neighbour_origin;
        if (any(local < vec3<i32>(0)) || any(local >= vec3<i32>(BRICK_DIM))) {
            continue;
        }
        let bit = u32(local.x + BRICK_DIM * (local.y + BRICK_DIM * local.z));
        let word = masks[neighbour_mask + (bit >> 5u)];
        if (((word >> (bit & 31u)) & 1u) != 0u) {
            occluded = occluded + 1.0;
        }
    }
    return clamp(1.0 - occluded * 0.11, 0.0, 1.0);
}

// Starts a shadow ray just outside the voxel that contains `point` along the
// sun direction. A plain `point + normal * epsilon` offset can still begin
// inside the surface at grazing angles, which shows up as salt-and-pepper
// self-shadowing; exiting the voxel is exact for any angle.
fn shadow_origin(point: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let cell = floor(point / VOXEL_SIZE);
    let lo = cell * VOXEL_SIZE;
    let hi = lo + vec3<f32>(VOXEL_SIZE);
    let face = select(lo, hi, sun > vec3<f32>(0.0));
    let safe = select(vec3<f32>(1e-6), sun, abs(sun) > vec3<f32>(1e-6));
    let t = (face - point) / safe;
    let exit = max(min(min(t.x, t.y), t.z), 0.0);
    return point + sun * (exit + 1e-4);
}

/// Tilts the sun direction by a sub-solar-disc angle. `dither` comes from the
/// pixel position and the frame index, so every pixel samples a different point
/// of the disc on every frame and the temporal pass averages them into a clean
/// penumbra. A world-space hash would freeze the pattern into the image and
/// read as wavy bands up close and speckle at a distance.
fn sun_soft_direction(sun: vec3<f32>, dither: vec2<f32>, radius: f32) -> vec3<f32> {
    // Any axis not parallel to the sun works as the basis seed.
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(sun.y) > 0.9);
    let tangent = normalize(cross(up, sun));
    let bitangent = cross(sun, tangent);
    let angle = 6.2831853 * dither.x;
    let offset = (tangent * cos(angle) + bitangent * sin(angle)) * (sqrt(dither.y) * radius);
    return normalize(sun + offset);
}

fn component_max3(v: vec3<f32>) -> f32 {
    return max(max(v.x, v.y), v.z);
}

fn component_min3(v: vec3<f32>) -> f32 {
    return min(min(v.x, v.y), v.z);
}

struct DebrisSpan {
    enter: f32,
    exit: f32,
    // Axis of the AABB face the ray crossed (0 = x, 1 = y, 2 = z).
    axis: i32,
};

fn debris_slab(
    ro: vec3<f32>,
    rd: vec3<f32>,
    lo: vec3<f32>,
    hi: vec3<f32>,
) -> DebrisSpan {
    let safe = select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
    let inv = vec3<f32>(1.0) / safe;
    let t0 = (lo - ro) * inv;
    let t1 = (hi - ro) * inv;
    let near = min(t0, t1);
    let far = max(t0, t1);
    var span: DebrisSpan;
    span.enter = component_max3(near);
    span.exit = component_min3(far);
    // The axis that produced the entry distance is the face the ray crossed.
    // Using the dominant ray axis instead (`debris_normal`) makes the normal of
    // the very first voxel depend on the view direction whenever the entry
    // point sits on the box face, which reads as light bending across a piece.
    span.axis = 2;
    if (near.x >= near.y && near.x >= near.z) {
        span.axis = 0;
    } else if (near.y >= near.z) {
        span.axis = 1;
    }
    return span;
}

fn debris_normal(cell_dir: vec3<f32>) -> vec3<f32> {
    let a = abs(cell_dir);
    if (a.x >= a.y && a.x >= a.z) {
        return vec3<f32>(-sign(cell_dir.x), 0.0, 0.0);
    }
    if (a.y >= a.z) {
        return vec3<f32>(0.0, -sign(cell_dir.y), 0.0);
    }
    return vec3<f32>(0.0, 0.0, -sign(cell_dir.z));
}

// Traces one piece in its own local space and keeps the nearest hit against the
// world. `incoming` is passed by value: the compiler inlines this, so the copy
// lives in a register and the early-out on `best.distance` still works.
fn trace_piece(i: u32, ro: vec3<f32>, rd: vec3<f32>, incoming: Hit) -> Hit {
    var best = incoming;
    let inst = debris[i];
    let local_ro = (inst.inverse_model * vec4<f32>(ro, 1.0)).xyz;
    let local_rd = (inst.inverse_model * vec4<f32>(rd, 0.0)).xyz;
    let span = debris_slab(local_ro, local_rd, inst.aabb_min.xyz, inst.aabb_max.xyz);
    let t_enter = max(span.enter, 0.0);
    if (span.exit <= t_enter || t_enter >= best.distance) {
        return best;
    }

    let dims = vec3<i32>(inst.data.yzw);
    let offset = inst.data.x;
    // The piece grid starts at its local AABB minimum, not at local zero:
    // `local_min` is the min voxel corner minus the centroid, so it is
    // usually negative. Shift the ray into grid space before walking it,
    // otherwise every cell index is off by the piece origin (or negative)
    // and the piece tests as empty -- which looked like debris vanishing.
    let grid_origin = inst.aabb_min.xyz;
    let grid_ro = local_ro - grid_origin;
    let start = grid_ro + local_rd * t_enter;
    var cell = clamp(
        vec3<i32>(floor(start / VOXEL_SIZE)),
        vec3<i32>(0),
        max(dims - vec3<i32>(1), vec3<i32>(0)),
    );
    let step = vec3<i32>(sign(local_rd));
    let safe_rd = select(local_rd, vec3<f32>(1e-6), abs(local_rd) < vec3<f32>(1e-6));
    let inv_rd = vec3<f32>(1.0) / safe_rd;
    let next_boundary =
        (vec3<f32>(cell) + max(vec3<f32>(step), vec3<f32>(0.0))) * VOXEL_SIZE;
    var t_max = (next_boundary - grid_ro) * inv_rd;
    t_max = select(t_max, vec3<f32>(1e30), step == vec3<i32>(0));
    // When the entry point lies on the AABB face the clamped first cell can put
    // the exit of an axis behind `t_enter`; without this clamp time runs
    // backwards for one step and the walk tests a voxel that is not the first
    // hit. The static DDA does the same (`max(init_t_max, vec3(distance))`).
    t_max = max(t_max, vec3<f32>(t_enter));
    var t_delta = abs(vec3<f32>(VOXEL_SIZE) * inv_rd);
    var t = t_enter;
    // Entering through the box face means the face normal is known exactly. A
    // ray that starts inside the box has no entry face: fall back to the
    // dominant axis of travel, which is what the first cell would show anyway.
    var local_normal = debris_normal(local_rd);
    if (t_enter > 0.0) {
        let face = -sign(safe_rd);
        if (span.axis == 0) {
            local_normal = vec3<f32>(face.x, 0.0, 0.0);
        } else if (span.axis == 1) {
            local_normal = vec3<f32>(0.0, face.y, 0.0);
        } else {
            local_normal = vec3<f32>(0.0, 0.0, face.z);
        }
    }
    let max_iterations = dims.x + dims.y + dims.z + 4;

    for (var iteration = 0; iteration < max_iterations; iteration = iteration + 1) {
        if (t >= best.distance || t > span.exit) {
            break;
        }
        if (all(cell >= vec3<i32>(0)) && all(cell < dims)) {
            let index =
                offset + u32(cell.x) +
                u32(dims.x) * (u32(cell.y) + u32(dims.y) * u32(cell.z));
            let raw = debris_voxels[index];
            if (raw != 0u) {
                let world_normal = normalize((inst.model * vec4<f32>(local_normal, 0.0)).xyz);
                best.valid = true;
                best.voxel = cell;
                best.normal = world_normal;
                best.raw = raw;
                best.id = 0x80000000u | i;
                best.distance = t;
                break;
            }
        }

        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            t = t_max.x;
            cell.x = cell.x + step.x;
            t_max.x = t_max.x + t_delta.x;
            local_normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
        } else if (t_max.y <= t_max.z) {
            t = t_max.y;
            cell.y = cell.y + step.y;
            t_max.y = t_max.y + t_delta.y;
            local_normal = vec3<f32>(0.0, -f32(step.y), 0.0);
        } else {
            t = t_max.z;
            cell.z = cell.z + step.z;
            t_max.z = t_max.z + t_delta.z;
            local_normal = vec3<f32>(0.0, 0.0, -f32(step.z));
        }
    }
    return best;
}

// Traces every active debris grid and keeps the nearest hit against the static
// world. The local grids are small (a 12-voxel-radius crater produces at most
// 25^3 cells), so a direct voxel walk is cheaper than rebuilding any
// acceleration structure per physics tick.
//
// Effect bit 512 adds a per-piece bounding-sphere reject before the 112-byte
// instance record is touched. Effect bit 1024 replaces the whole loop with the
// per-tile candidate masks from `debris_cull.wgsl`, which is what actually
// scales: after a house comes down most pixels miss most pieces, and walking a
// handful of candidate bits beats a sphere test per live piece. Every reject
// here is conservative: it only drops pieces the walk would have missed.
fn cast_debris(ro: vec3<f32>, rd: vec3<f32>, start_hit: Hit, pixel: vec2<u32>) -> Hit {
    var best = start_hit;
    let count = u32(max(frame.params2.x, 0.0));
    // One slab test against the union of every piece: when the camera ray misses
    // that box (or nothing was destroyed) every instance transform and local DDA
    // walk is skipped entirely, which is the common case.
    let debris_inv_rd = 1.0 / select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
    let box_t0 = (frame.debris_min.xyz - ro) * debris_inv_rd;
    let box_t1 = (frame.debris_max.xyz - ro) * debris_inv_rd;
    let box_near = max(max(min(box_t0.x, box_t1.x), min(box_t0.y, box_t1.y)), min(box_t0.z, box_t1.z));
    let box_far = min(min(max(box_t0.x, box_t1.x), max(box_t0.y, box_t1.y)), max(box_t0.z, box_t1.z));
    if (count == 0u || box_far <= max(box_near, 0.0) || box_near >= best.distance) {
        return best;
    }
    let bits = u32(frame.params3.x);
    // Effect bit 1024: per-tile candidate lists built by `debris_cull.wgsl`.
    // Each tile holds a 512-bit mask of the pieces whose projected rectangle
    // overlaps it, so a pixel walks a handful of pieces instead of every live
    // one. Conservative by construction, so the image is unchanged.
    if (bits & 1024u) != 0u {
        let size = vec2<u32>(max(frame.internal_size.xy, vec2<f32>(1.0)));
        let tiles_x = (size.x + 7u) / 8u;
        let tile = pixel / 8u;
        let base = (tile.y * tiles_x + tile.x) * 16u;
        for (var word = 0u; word < 16u; word = word + 1u) {
            var mask = debris_tiles[base + word];
            loop {
                if (mask == 0u) {
                    break;
                }
                let bit = firstTrailingBit(mask);
                mask = mask & (mask - 1u);
                best = trace_piece(word * 32u + bit, ro, rd, best);
            }
        }
        return best;
    }
    // Fallback without the tile pass: one sphere test per live piece. Cheaper
    // than touching the instance record, but still O(pieces) per pixel.
    if ((bits & 512u) == 0u) {
        for (var i = 0u; i < count; i = i + 1u) {
            best = trace_piece(i, ro, rd, best);
        }
        return best;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        let sphere = debris_spheres[i];
        let to_centre = sphere.xyz - ro;
        let along = dot(to_centre, rd);
        let radius = sphere.w;
        if (along + radius <= 0.0 || along - radius >= best.distance) {
            continue;
        }
        if (dot(to_centre, to_centre) - along * along > radius * radius) {
            continue;
        }
        best = trace_piece(i, ro, rd, best);
    }
    return best;
}

fn debris_ao(index: u32, cell: vec3<i32>, normal: vec3<f32>) -> f32 {
    let inst = debris[index];
    let dims = vec3<i32>(inst.data.yzw);
    let offset = inst.data.x;
    let local_normal = normalize((inst.inverse_model * vec4<f32>(normal, 0.0)).xyz);
    let n = vec3<i32>(round(local_normal));
    let centre = cell + n;
    var occluded = 0.0;
    let probes = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),
        vec3<i32>(-1, 0, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(0, -1, 0),
        vec3<i32>(0, 0, 1),
        vec3<i32>(0, 0, -1),
    );
    for (var i = 0; i < 6; i = i + 1) {
        let p = centre + probes[i];
        if (!all(p >= vec3<i32>(0)) || !all(p < dims)) {
            continue;
        }
        let index2 =
            offset + u32(p.x) + u32(dims.x) * (u32(p.y) + u32(dims.y) * u32(p.z));
        if (debris_voxels[index2] != 0u) {
            occluded = occluded + select(0.55, 1.0, dot(vec3<f32>(probes[i]), vec3<f32>(n)) > 0.5);
        }
    }
    return clamp(1.0 - occluded * 0.11, 0.0, 1.0);
}

// +/-4 % brightness/tint variation keyed by world position. Two scales keep it
// from reading as noise: 25 cm grain breaks up a face, 2 m patches keep large
// concrete fields from feeling like painted plastic.
fn micro_jitter(world: vec3<f32>, material: u32) -> f32 {
    let fine = floor(world / 0.25);
    let coarse = floor(world / 2.0);
    let a = hash01(i32(fine.x) + i32(fine.z) * 31, i32(fine.y), material + 211u);
    let b = hash01(i32(coarse.x), i32(coarse.z) + i32(coarse.y) * 17, material + 307u);
    return 0.96 + 0.055 * a + 0.025 * b;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_albedo);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let pixel = vec2<f32>(gid.xy) + vec2<f32>(0.5);
    let uv = pixel / vec2<f32>(dims);
    var ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // Sub-pixel jitter for the temporal upscale (Halton 2,3 on the CPU).
    ndc = ndc + frame.params.zw * 2.0 / vec2<f32>(dims);

    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near = near_h.xyz / near_h.w;
    let far = far_h.xyz / far_h.w;
    let ro = frame.camera_pos.xyz;
    let rd = normalize(far - near);

    // 192 m covers the whole arena from any corner while staying inside the
    // fog range (at 192 m the transmittance is already below 10 %), so geometry
    // no longer pops out of existence on a 128 m sphere.
    var hit = cast_ray(ro, rd, 192.0, true, vec3<i32>(0), false);
    // Phase 4 glass: a pane is shaded as its own surface with a very low
    // roughness, so `lighting.wgsl` can add a screen-space reflection to it.
    // (The earlier version skipped the pane and transmitted the surface behind
    // it, which made windows look like tinted holes and killed reflections.)
    const GLASS_MATERIAL: u32 = 6u;
    let glass_tint = vec3<f32>(1.0);
    if (!hit.valid) {
        hit.distance = 1e30;
    }
    hit = cast_debris(ro, rd, hit, gid.xy);
    if (!hit.valid) {
        textureStore(out_albedo, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(out_normal, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(out_material, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(out_depth, vec2<i32>(gid.xy), vec4<f32>(-1.0, 0.0, 0.0, 0.0));
        return;
    }

    let material_index = hit.raw & 0x3ffu;
    let material = palette[material_index];
    let tint = f32((hit.raw >> 10u) & 0x3fu) / 63.0;
    // Stylised per-voxel detail (`material_detail`, `macro_variation`,
    // `micro_jitter`) moved to `lighting.wgsl`: the primary pass now writes the
    // plain palette colour, which keeps this kernel's register pressure low
    // enough for the DDA to stay the fastest thing in the frame.
    let tint_scale = mix(0.86, 1.14, tint);
    let is_debris = (hit.id & 0x80000000u) != 0u;
    let debris_index = hit.id & 0x7fffffffu;
    var ao = 1.0;
    // Contact AO only: it darkens the crease where two voxels meet, which is
    // what makes a voxel scene readable. It is weighted by an analytic falloff
    // instead of a distance ramp, so there is no visible shell/sphere around
    // the camera where the effect switches off вЂ” the old `smoothstep(7,12)`
    // drew exactly that shell on large floors. Everything further out is left
    // to the screen-space AO in `gi.wgsl`, which covers the whole screen.
    let contact = 1.0 / (1.0 + hit.distance * hit.distance * 0.30);
    if (contact > 0.02) {
        let sample_ao = select(
            voxel_ao(hit.voxel, hit.normal, hit.id),
            debris_ao(debris_index, hit.voxel, hit.normal),
            is_debris,
        );
        ao = mix(1.0, sample_ao, contact);
    }
    let jitter = 1.0;

    // Reactive mask for the temporal upscaler. Debris is the only geometry that
    // moves fast enough to leave a visible history trail, so it is flagged
    // here and `fsr.wgsl` drops the history weight for those pixels. The
    // fourth material channel used to be a constant 1.0, so nothing else reads
    // it and the lighting pass keeps ignoring it.
    let reactive = select(0.0, 1.0, is_debris);

    let albedo = material.albedo_rough.xyz * tint_scale * jitter * glass_tint;
    textureStore(
        out_albedo,
        vec2<i32>(gid.xy),
        vec4<f32>(clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(ao, 0.0, 1.0)),
    );
    textureStore(
        out_normal,
        vec2<i32>(gid.xy),
        vec4<f32>(
            hit.normal,
            clamp(material.albedo_rough.w * select(1.0, 0.10, material_index == GLASS_MATERIAL || material_index == MATERIAL_GLASS_CLEAR), 0.03, 1.0),
        ),
    );
    textureStore(
        out_material,
        vec2<i32>(gid.xy),
        vec4<f32>(
            clamp(material.params.x, 0.0, 1.0),
            clamp(material.params.y / 8.0, 0.0, 1.0),
            f32(material_index) / 255.0,
            reactive,
        ),
    );
    textureStore(out_depth, vec2<i32>(gid.xy), vec4<f32>(hit.distance, 0.0, 0.0, 0.0));
}
