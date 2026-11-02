// Half-resolution local-light shadow mask.
//
// The deferred pass used to add all 12 point/capsule/rect lights without any
// visibility test, so neon and lamps happily lit the far side of a wall. This
// pass traces exactly one DDA ray per *half-resolution* texel towards each
// light that can actually reach the surface, and stores the result as a bitmask
// (bit i = light i visible). At half internal resolution with an early-out on
// the light radius the cost is roughly a twelfth of tracing per full-res pixel
// per light, and `lighting.wgsl` turns the mask into a soft penumbra by
// averaging a 2x2 block of texels.
//
// The DDA is the same kernel as `shadow.wgsl`; keeping it identical means the
// hierarchy masks and the surface-offset rule behave the same way for sun and
// local lights.

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
    // xyz = emitter axis, w = half extent. w > 0 is a capsule of length 2w,
    // w < 0 is a one-sided rectangle with half size -w and normal xyz, w == 0
    // is a plain point light.
    light_axis: array<vec4<f32>, 12>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var brick_index_tex: texture_3d<u32>;
@group(0) @binding(2) var<storage, read> masks: array<u32>;
@group(0) @binding(3) var group_occupancy_tex: texture_3d<u32>;
@group(0) @binding(4) var region_occupancy_tex: texture_3d<u32>;
@group(0) @binding(5) var depth_tex: texture_2d<f32>;
@group(0) @binding(6) var normal_tex: texture_2d<f32>;
// Bit i set = light i contributes to this texel.
@group(0) @binding(7) var out_light: texture_storage_2d<r32uint, write>;

const VOXEL_SIZE: f32 = 0.125;
const BRICK_DIM: i32 = 16;
const GROUP_BRICKS: i32 = 4;
const MAX_STEPS: i32 = 1024;
const MASK_WORDS: u32 = 147u;
const MASK8_OFFSET: u32 = 128u;
const MASK4_OFFSET: u32 = 144u;
const MASK2_OFFSET: u32 = 146u;
const GROUP_VOXELS: i32 = 64;
const REGION_VOXELS: i32 = 256;
const MAX_LIGHTS: i32 = 12;

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

/// True when anything solid blocks the ray before `max_distance`. The ray starts
/// just outside the surface and runs away from it, so no cell is skipped.
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

    for (var i = 0; i < MAX_STEPS; i = i + 1) {
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

fn world_from_uv(uv: vec2<f32>, distance: f32, jitter: vec2<f32>) -> vec3<f32> {
    let internal = frame.internal_size.xy;
    var ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    ndc = ndc + jitter * 2.0 / internal;
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let rd = normalize(far_h.xyz / far_h.w - near_h.xyz / near_h.w);
    return frame.camera_pos.xyz + rd * distance;
}

// Representative point of an area emitter (Karis 2013), identical to
// `emitter_point` in `lighting.wgsl`. `w` is 0 when the shading point sits
// behind a one-sided rectangle, so the caller can skip the light entirely.
fn emitter_point(base: vec3<f32>, axis: vec3<f32>, extent: f32, world: vec3<f32>) -> vec4<f32> {
    if (extent > 0.0) {
        let dir = normalize(axis + vec3<f32>(0.0, 1e-6, 0.0));
        let along = clamp(dot(world - base, dir), -extent, extent);
        return vec4<f32>(base + dir * along, 1.0);
    }
    if (extent < 0.0) {
        let n = normalize(axis + vec3<f32>(0.0, 0.0, 1e-6));
        let half_size = -extent;
        let offset = world - base;
        let facing = dot(offset, n);
        if (facing <= 0.0) {
            return vec4<f32>(base, 0.0);
        }
        let planar = clamp(offset - n * facing, vec3<f32>(-half_size), vec3<f32>(half_size));
        return vec4<f32>(base + planar, 1.0);
    }
    return vec4<f32>(base, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_light);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let full_dims = vec2<i32>(textureDimensions(depth_tex));

    // Every texel covers a 2x2 block of the G-buffer; the closest of the four
    // depths wins so a silhouette against a far background is shaded with the
    // near surface instead of letting light leak through the edge.
    let base = pixel * 2;
    var depth = -1.0;
    var depth_pixel = base;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let p = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0), full_dims - vec2<i32>(1));
            let d = textureLoad(depth_tex, p, 0).r;
            if (d >= 0.0 && (depth < 0.0 || d < depth)) {
                depth = d;
                depth_pixel = p;
            }
        }
    }

    if (depth < 0.0) {
        // Sky: nothing to occlude, every light is "visible" but the lighting
        // pass returns the sky colour for these pixels anyway.
        textureStore(out_light, pixel, vec4<u32>(0xffffffffu));
        return;
    }

    let normal = textureLoad(normal_tex, depth_pixel, 0).xyz;
    let uv = (vec2<f32>(depth_pixel) + vec2<f32>(0.5)) / vec2<f32>(full_dims);
    let point = world_from_uv(uv, depth, frame.params.zw);
    // Step out of the surface voxel. The trace starts in air and every solid
    // cell on the way is a real occluder.
    let origin = point + normal * 0.02;

    var mask = 0xffffffffu;
    let light_count = min(i32(frame.params2.z), MAX_LIGHTS);
    for (var i = 0; i < MAX_LIGHTS; i = i + 1) {
        if (i >= light_count) {
            break;
        }
        let axis = frame.light_axis[i];
        let emitter = emitter_point(frame.light_pos_radius[i].xyz, axis.xyz, axis.w, point);
        if (emitter.w < 0.5) {
            continue;
        }
        let radius = max(frame.light_pos_radius[i].w, 0.1);
        let to_light = emitter.xyz - point;
        let distance_sq = dot(to_light, to_light);
        // Out of range: the deferred pass skips this light too.
        if (distance_sq > radius * radius) {
            continue;
        }
        let distance = sqrt(distance_sq);
        let l = to_light / max(distance, 1e-4);
        // Back-facing: the surface cannot see the emitter, and leaving the bit
        // set let the specular lobe shine through the wall. Clear it.
        if (dot(normal, l) <= 0.0) {
            mask = mask & ~(1u << u32(i));
            continue;
        }
        // Stop just short of the emitter so its own housing does not block it.
        let reach = distance - 0.06;
        if (reach <= 0.0) {
            continue;
        }
        if (occluded(origin, l, reach)) {
            mask = mask & ~(1u << u32(i));
        }
    }

    textureStore(out_light, pixel, vec4<u32>(mask));
}
