// DDGI trace pass: one ray per octahedral texel of every probe in the slice.
//
// Layout
// ------
// A probe owns an 8x8 tile. Columns are the probe x index; rows are
// `z * GRID_Y + y`. One z-slice of the grid is refreshed per frame, so the
// whole 64 x 16 x 64 grid is covered in 64 frames (see `encode_frame_graph`).
//
// This pass writes the *raw ray radiance* plus the hit distance into
// `out_trace`; `ddgi_blend.wgsl` turns that into irradiance with a cosine
// convolution and applies the temporal hysteresis. Splitting the two passes
// keeps the trace kernel cheap (one ray per thread, no 64-tap loop) and lets
// the blend replace the whole tile without re-tracing it.
//
// Radiance model
// --------------
// A hit reads the surface albedo injected from the G-buffer (1 m voxel volume,
// see `ddgi_inject.wgsl`), traces the sun through the real voxel mask hierarchy
// and sums all twelve local lights plus the previous probe atlas for a cheap
// second bounce. The mask DDA is what makes the indirect light correct indoors:
// a 2 m brick marks a whole window as solid, so a brick-level shadow ray would
// report every interior surface as unlit and the room's bounce would be dim.
//
// Misses return the analytic sky, so a probe near a window still sees daylight.

struct Frame {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    prev_camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    world_min: vec4<f32>,
    world_max: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    params3: vec4<f32>,
    internal_size: vec4<f32>,
    light_pos_radius: array<vec4<f32>, 12>,
    light_color: array<vec4<f32>, 12>,
};

struct DdgiParams {
    // z index of the probe slice refreshed by this dispatch.
    slice: u32,
    // Fraction of the previous texel kept (0..1).
    hysteresis: f32,
    // Probe spacing in metres (same value as DDGI_PROBE_SPACING_M in Rust).
    spacing: f32,
    // 1 during the first cycle after (re)allocation.
    reset: u32,
    // Rays are laid out in a `tile x tile` subset of the 8x8 atlas tile.
    tile: u32,
    // Jittered samples per texel (2 on the high preset).
    samples: u32,
    // Probe slices refreshed this frame.
    slices: u32,
    // 1 while the post-destruction priority window is active.
    priority: u32,
    // Probes within this sphere use `priority_hysteresis`.
    priority_center: vec3<f32>,
    priority_radius: f32,
    priority_hysteresis: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> frame: Frame;
// Brick occupancy: one texel per 2 m brick (R32Uint brick index, 0 = empty).
@group(0) @binding(1) var brick_index_tex: texture_3d<u32>;
// Per-brick mask words (16^3 bits + the 8/4/2 m occupancy levels).
@group(0) @binding(2) var<storage, read> masks: array<u32>;
@group(0) @binding(3) var group_occupancy_tex: texture_3d<u32>;
@group(0) @binding(4) var region_occupancy_tex: texture_3d<u32>;
// 1 m albedo volume injected from the G-buffer. rgb = albedo,
// a = 1 + emissive (a <= 0.5 means "unknown, use neutral grey").
@group(0) @binding(5) var albedo_volume: texture_3d<f32>;
@group(0) @binding(6) var volume_sampler: sampler;
@group(0) @binding(7) var prev_irradiance: texture_2d<f32>;
// rgb = ray radiance, a = hit distance / MAX_DISTANCE.
@group(0) @binding(8) var out_trace: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var<uniform> params: DdgiParams;

// Must match DDGI_GRID / DDGI_TILE in `renderer.rs` and `ddgi_blend.wgsl`.
const GRID_X: u32 = 64u;
const GRID_Y: u32 = 16u;
const GRID_Z: u32 = 64u;
const TILE: u32 = 8u;
const PI: f32 = 3.14159265;
// 40 x 0.65 m = 26 m of reach, a little more than MAX_DISTANCE so a ray that
// leaves the probe's neighbourhood still ends on the sky instead of a stale
// brick.
const TRACE_STEPS: i32 = 40;
const TRACE_STEP_M: f32 = 0.65;
const MAX_DISTANCE: f32 = 24.0;
/// Sun rays through the voxel grid. The arena is 256 m across, but the sun
/// direction reaches the world bound far sooner; the DDA early-outs on empty
/// regions, so this cap is only a guard.
const SUN_MAX_DISTANCE: f32 = 96.0;

// Voxel DDA constants, identical to `shadow.wgsl` / `gbuffer.wgsl`.
const VOXEL_SIZE: f32 = 0.125;
const BRICK_DIM: i32 = 16;
const MASK_WORDS: u32 = 147u;
const MASK8_OFFSET: u32 = 128u;
const MASK4_OFFSET: u32 = 144u;
const MASK2_OFFSET: u32 = 146u;
const GROUP_VOXELS: i32 = 64;
const REGION_VOXELS: i32 = 256;
const DDA_STEPS: i32 = 1024;

fn world_to_cell(p: vec3<f32>) -> vec3<i32> {
    let dims = vec3<f32>(textureDimensions(brick_index_tex));
    let size = max(frame.world_max.xyz - frame.world_min.xyz, vec3<f32>(1.0));
    let uv = (p - frame.world_min.xyz) / size;
    return vec3<i32>(clamp(uv * dims, vec3<f32>(0.0), dims - vec3<f32>(1.0)));
}

/// Coarse (2 m brick) occupancy. Used for the radiance ray march and as the
/// conservative "is this brick empty" fast path.
fn occupancy(p: vec3<f32>) -> u32 {
    return textureLoad(brick_index_tex, world_to_cell(p), 0).r;
}

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

fn block_boundary(t_max: f32, t_delta: f32, cell: i32, step: i32, size: i32) -> f32 {
    let local = cell - (cell / size) * size;
    let remaining = select(size - 1 - local, local, step < 0);
    return t_max + t_delta * f32(remaining);
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

fn block_occupied(mask_base: u32, offset: u32, local: vec3<i32>, size: i32) -> bool {
    let block = local / vec3<i32>(size);
    let dim = 16 / size;
    let bit = u32(block.x + dim * (block.y + dim * block.z));
    let word = masks[mask_base + offset + (bit >> 5u)];
    return ((word >> (bit & 31u)) & 1u) != 0u;
}

/// True when anything solid blocks the ray before `max_distance`. Straight port
/// of the `shadow.wgsl` traversal (brick index -> 2/4/8 m occupancy -> 12.5 cm
/// voxel mask), so a sun ray can pass through a window opening.
fn occluded(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> bool {
    let max_cell = vec3<i32>(frame.world_max.xyz / VOXEL_SIZE) - vec3<i32>(1);
    let inside_world = all(ro >= frame.world_min.xyz) && all(ro < frame.world_max.xyz);
    var distance = 1e-5;
    if (!inside_world) {
        let bounds = intersect_world(ro, rd);
        if (bounds.y <= 0.0 || bounds.x > bounds.y) {
            return false;
        }
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

    var brick_coord = vec3<i32>(-1);
    var id = 0u;
    var mask_base = 0u;
    var mask_word_index = 0xffffffffu;
    var mask_word_value = 0u;

    for (var i = 0; i < DDA_STEPS; i = i + 1) {
        if (distance > max_distance) {
            break;
        }
        if (any(cell < vec3<i32>(0)) || any(cell > max_cell)) {
            break;
        }

        let coord = cell / BRICK_DIM;
        if (any(coord != brick_coord)) {
            brick_coord = coord;
            id = textureLoad(brick_index_tex, coord, 0).r;
            mask_base = (id - 1u) * MASK_WORDS;
            mask_word_index = 0xffffffffu;
        }

        if (id == 0u) {
            var boundary = min_boundary3(
                block_boundary(t_max.x, t_delta.x, cell.x, step.x, BRICK_DIM),
                block_boundary(t_max.y, t_delta.y, cell.y, step.y, BRICK_DIM),
                block_boundary(t_max.z, t_delta.z, cell.z, step.z, BRICK_DIM),
            );
            let region = cell / REGION_VOXELS;
            if (textureLoad(region_occupancy_tex, region, 0).r == 0u) {
                boundary = min_boundary3(
                    block_boundary(t_max.x, t_delta.x, cell.x, step.x, REGION_VOXELS),
                    block_boundary(t_max.y, t_delta.y, cell.y, step.y, REGION_VOXELS),
                    block_boundary(t_max.z, t_delta.z, cell.z, step.z, REGION_VOXELS),
                );
            } else {
                let group = cell / GROUP_VOXELS;
                if (textureLoad(group_occupancy_tex, group, 0).r == 0u) {
                    boundary = min_boundary3(
                        block_boundary(t_max.x, t_delta.x, cell.x, step.x, GROUP_VOXELS),
                        block_boundary(t_max.y, t_delta.y, cell.y, step.y, GROUP_VOXELS),
                        block_boundary(t_max.z, t_delta.z, cell.z, step.z, GROUP_VOXELS),
                    );
                }
            }
            if (boundary.x > max_distance || boundary.x >= 1e29) {
                break;
            }
            let previous = cell;
            distance = boundary.x;
            cell = jump_cell(cell, step, ro, rd, distance, i32(boundary.y));
            t_max = max(t_max + vec3<f32>((cell - previous) * step) * t_delta, vec3<f32>(distance));
            continue;
        }

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
            if (boundary.x > max_distance || boundary.x >= 1e29) {
                break;
            }
            let previous = cell;
            distance = boundary.x;
            cell = jump_cell(cell, step, ro, rd, distance, i32(boundary.y));
            t_max = max(t_max + vec3<f32>((cell - previous) * step) * t_delta, vec3<f32>(distance));
            continue;
        }

        let bit = u32(local.x + BRICK_DIM * (local.y + BRICK_DIM * local.z));
        let word_index = bit >> 5u;
        if (word_index != mask_word_index) {
            mask_word_index = word_index;
            mask_word_value = masks[mask_base + word_index];
        }
        if (((mask_word_value >> (bit & 31u)) & 1u) != 0u) {
            return true;
        }

        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            distance = t_max.x;
            cell.x = cell.x + step.x;
            t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y <= t_max.z) {
            distance = t_max.y;
            cell.y = cell.y + step.y;
            t_max.y = t_max.y + t_delta.y;
        } else {
            distance = t_max.z;
            cell.z = cell.z + step.z;
            t_max.z = t_max.z + t_delta.z;
        }
    }
    return false;
}

fn solid_normal(p: vec3<f32>) -> vec3<f32> {
    let d = 0.9;
    let gx = f32(occupancy(p + vec3<f32>(d, 0.0, 0.0))) - f32(occupancy(p - vec3<f32>(d, 0.0, 0.0)));
    let gy = f32(occupancy(p + vec3<f32>(0.0, d, 0.0))) - f32(occupancy(p - vec3<f32>(0.0, d, 0.0)));
    let gz = f32(occupancy(p + vec3<f32>(0.0, 0.0, d))) - f32(occupancy(p - vec3<f32>(0.0, 0.0, d)));
    let gradient = vec3<f32>(gx, gy, gz);
    let length_sq = dot(gradient, gradient);
    if (length_sq < 1e-6) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return -gradient / sqrt(length_sq);
}

fn octahedral_direction(uv: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
    if (d.z < 0.0) {
        d = vec3<f32>((1.0 - abs(d.yx)) * sign(d.xy), d.z);
    }
    return normalize(d);
}

/// Analytic sky, close enough to `lighting.wgsl` for a probe ray. The bright
/// sun halo matters: without it a probe looking through a window sees a flat
/// blue instead of the warm shaft the room is actually lit by.
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let day = clamp(frame.sun_color.w, 0.0, 1.0);
    let up = clamp(dir.y, 0.0, 1.0);
    let zenith = mix(vec3<f32>(0.004, 0.008, 0.022), vec3<f32>(0.018, 0.065, 0.30), day);
    let horizon = mix(vec3<f32>(0.020, 0.032, 0.060), vec3<f32>(0.34, 0.52, 0.78), day);
    var color = mix(horizon, zenith, pow(up, 0.55));
    let sun_amount = max(dot(dir, normalize(frame.sun_dir.xyz)), 0.0);
    color = color + frame.sun_color.rgb * pow(sun_amount, 8.0) * 0.35 * day;
    return color;
}

fn world_to_volume_uv(p: vec3<f32>) -> vec3<f32> {
    let size = max(frame.world_max.xyz - frame.world_min.xyz, vec3<f32>(1.0));
    return clamp((p - frame.world_min.xyz) / size, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Sun visibility through the real voxel grid.
fn sun_visibility(p: vec3<f32>) -> f32 {
    let sun = normalize(frame.sun_dir.xyz);
    return select(0.0, 1.0, !occluded(p, sun, SUN_MAX_DISTANCE));
}

/// Sum of the twelve local lights at a hit point. No shadow ray: a probe only
/// needs the low-frequency bounce, and a lamp hidden behind a wall is already
/// dimmed by the inverse-square term at these distances.
fn local_light_radiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let count = i32(frame.params2.z);
    for (var i = 0; i < 12; i = i + 1) {
        if (i >= count) {
            break;
        }
        let light = frame.light_pos_radius[i];
        let radius = max(light.w, 0.1);
        let to_light = light.xyz - p;
        let distance_sq = dot(to_light, to_light);
        if (distance_sq > radius * radius || distance_sq < 1e-4) {
            continue;
        }
        let distance = sqrt(distance_sq);
        let l = to_light / distance;
        let ndl = max(dot(n, l), 0.0);
        if (ndl <= 0.0) {
            continue;
        }
        let normalized = clamp(distance / radius, 0.0, 1.0);
        let cutoff = 1.0 - normalized * normalized;
        let falloff = cutoff * cutoff / max(distance_sq, 0.4);
        // Local lights used to ignore occlusion, so a lamp behind a one-voxel
        // wall lit the other side through DDGI. One short mask ray per light
        // fixes that; the far end is cut by the light radius already.
        if (occluded(p + n * 0.2, l, distance - 0.2)) {
            continue;
        }
        sum = sum + frame.light_color[i].rgb * falloff * ndl;
    }
    return sum;
}

fn probe_atlas_texel(probe: vec3<i32>, dir: vec3<f32>) -> vec2<i32> {
    let p = dir.xy / max(abs(dir.x) + abs(dir.y) + abs(dir.z), 1e-4);
    let oct = select(p, vec2<f32>((1.0 - abs(p.yx)) * sign(p.xy)), dir.z < 0.0);
    let uv = clamp(oct * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    let texel = clamp(uv * f32(TILE), vec2<f32>(0.0), vec2<f32>(f32(TILE) - 1.0));
    return vec2<i32>(
        probe.x * i32(TILE) + i32(texel.x),
        probe.z * i32(GRID_Y * TILE) + probe.y * i32(TILE) + i32(texel.y),
    );
}

/// Trilinear probe lookup of the previous atlas. Bilinear inside a tile is
/// skipped here: one tap per probe is enough for a second bounce, and this runs
/// once per ray hit rather than once per pixel.
fn prev_irradiance_at(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let dims = vec3<i32>(i32(GRID_X), i32(GRID_Y), i32(GRID_Z));
    let base = (p - frame.world_min.xyz) / max(params.spacing, 0.5) - vec3<f32>(0.5);
    let cell = floor(base);
    let frac = clamp(base - cell, vec3<f32>(0.0), vec3<f32>(1.0));
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var corner = 0; corner < 8; corner = corner + 1) {
        let offset = vec3<f32>(f32(corner & 1), f32((corner >> 1) & 1), f32((corner >> 2) & 1));
        let index = clamp(vec3<i32>(cell) + vec3<i32>(offset), vec3<i32>(0), dims - vec3<i32>(1));
        let w = mix(1.0 - frac, frac, offset).x
            * mix(1.0 - frac, frac, offset).y
            * mix(1.0 - frac, frac, offset).z;
        sum = sum + textureLoad(prev_irradiance, probe_atlas_texel(index, n), 0).rgb * w;
        weight_sum = weight_sum + w;
    }
    return sum / max(weight_sum, 1e-4);
}

/// Reads the injected albedo near a brick hit. Occupancy is a 2 m brick, so the
/// ray stops up to two metres *before* the real surface: walk forward along the
/// ray until a cell the camera actually saw is found.
fn sample_albedo(p: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    let dims = textureDimensions(albedo_volume);
    for (var i = 0; i < 5; i = i + 1) {
        let uv = world_to_volume_uv(p + dir * (0.2 + f32(i) * 0.5));
        let cell = clamp(
            vec3<i32>(floor(uv * vec3<f32>(dims))),
            vec3<i32>(0),
            vec3<i32>(dims) - vec3<i32>(1),
        );
        let sample = textureLoad(albedo_volume, cell, 0);
        if (sample.a > 0.5) {
            return sample;
        }
    }
    return vec4<f32>(0.42, 0.42, 0.40, 1.0);
}

/// Returns `(radiance.rgb, hit_distance)` for one direction.
fn trace_radiance(origin: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    if (occupancy(origin) != 0u) {
        // Buried probe: it sees neither sky nor sun.
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let sun = normalize(frame.sun_dir.xyz);
    var t = 0.4;
    for (var i = 0; i < TRACE_STEPS; i = i + 1) {
        let p = origin + dir * t;
        if (occupancy(p) != 0u) {
            let n = solid_normal(p);
            let albedo_sample = sample_albedo(p, dir);
            let albedo = albedo_sample.rgb;
            let emissive = max(albedo_sample.a - 1.0, 0.0);
            let visibility = sun_visibility(p + n * 0.2);
            let direct = frame.sun_color.rgb * max(dot(n, sun), 0.0) * visibility;
            let lights = local_light_radiance(p + n * 0.3, n);
            // Second bounce was a cheap feedback loop, but in a closed white
            // room it amplified the atlas until the wall read as a flat
            // saturated colour. One bounce is plenty at this probe density.
            let indirect = vec3<f32>(0.0);
            let ambient = sky_radiance(n) * 0.35;
            let radiance = albedo * (direct / PI + lights / PI + indirect + ambient)
                + albedo * emissive * 1.5;
            return vec4<f32>(clamp(radiance, vec3<f32>(0.0), vec3<f32>(32.0)), t);
        }
        t = t + TRACE_STEP_M;
    }
    return vec4<f32>(sky_radiance(dir), MAX_DISTANCE);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_extent = vec2<u32>(GRID_X * TILE, GRID_Y * TILE);
    if (gid.x >= tile_extent.x || gid.y >= tile_extent.y) {
        return;
    }
    let tile = clamp(params.tile, 1u, TILE);
    let samples = max(params.samples, 1u);
    let z = min(params.slice, GRID_Z - 1u);
    // Probe x from the column of tiles, probe y from the row of tiles.
    let probe = vec3<u32>(gid.x / TILE, gid.y / TILE, z);
    let texel = vec2<u32>(gid.x % TILE, gid.y % TILE);
    if (texel.x >= tile || texel.y >= tile) {
        return;
    }
    let atlas = vec2<i32>(i32(gid.x), i32(z * tile_extent.y + gid.y));

    let origin = frame.world_min.xyz + (vec3<f32>(probe) + vec3<f32>(0.5)) * params.spacing;
    var radiance = vec3<f32>(0.0);
    var distance = 0.0;
    for (var s = 0u; s < samples; s = s + 1u) {
        // High preset: two jittered rays per texel, averaged. The sub-texel
        // offset is ±0.25 texel, small enough that the cosine convolution in
        // `ddgi_blend.wgsl` still sees the same direction.
        let offset = (f32(s) + 0.5) / f32(samples) - 0.5;
        let uv = (vec2<f32>(texel) + vec2<f32>(0.5 + offset)) / f32(tile) * 2.0
            - vec2<f32>(1.0);
        let dir = octahedral_direction(uv);
        let sample = trace_radiance(origin, dir);
        radiance = radiance + sample.rgb;
        distance = distance + sample.w;
    }
    let inverse = 1.0 / f32(samples);
    textureStore(
        out_trace,
        atlas,
        vec4<f32>(radiance * inverse, distance * inverse / MAX_DISTANCE),
    );
}
