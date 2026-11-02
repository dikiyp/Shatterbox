// Phase 1 voxel visibility: primary DDA into an internal colour target.
//
// The shader traverses the arena at 12.5 cm voxel granularity. Empty bricks
// are skipped in O(1) through the 3D brick-index texture. Hard sun shadows use
// the same traversal function.

struct Camera {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    world_min: vec4<f32>,
    world_max: vec4<f32>,
    // x = time, y = frame index, zw = reserved
    params: vec4<f32>,
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
    distance: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var brick_index_tex: texture_3d<u32>;
@group(0) @binding(2) var<storage, read> bricks: array<u32>;
@group(0) @binding(3) var<storage, read> palette: array<GpuMaterial>;
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var group_occupancy_tex: texture_3d<u32>;

const VOXEL_SIZE: f32 = 0.125;
const BRICK_DIM: i32 = 16;
const BRICK_WORDS: u32 = 2048u;
const GROUP_BRICKS: i32 = 4;
const MAX_STEPS: i32 = 4096;

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
    let x = slab_axis(ro.x, rd.x, camera.world_min.x, camera.world_max.x);
    let y = slab_axis(ro.y, rd.y, camera.world_min.y, camera.world_max.y);
    let z = slab_axis(ro.z, rd.z, camera.world_min.z, camera.world_max.z);
    return vec2<f32>(max(max(x.x, y.x), z.x), min(min(x.y, y.y), z.y));
}

fn next_boundary(cell: i32, step: i32, ro: f32, rd: f32) -> f32 {
    if (abs(rd) < 1e-8) {
        return 1e30;
    }
    let face = select(f32(cell), f32(cell + 1), step > 0);
    return (face * VOXEL_SIZE - ro) / rd;
}

fn next_brick_boundary(cell: i32, step: i32, ro: f32, rd: f32) -> f32 {
    if (abs(rd) < 1e-8) {
        return 1e30;
    }
    let brick = cell / BRICK_DIM;
    let face = select(f32(brick), f32(brick + 1), step > 0);
    return (face * VOXEL_SIZE * f32(BRICK_DIM) - ro) / rd;
}

fn next_group_boundary(cell: i32, step: i32, ro: f32, rd: f32) -> f32 {
    if (abs(rd) < 1e-8) {
        return 1e30;
    }
    let group = cell / (BRICK_DIM * GROUP_BRICKS);
    let face = select(f32(group), f32(group + 1), step > 0);
    return (face * VOXEL_SIZE * f32(BRICK_DIM * GROUP_BRICKS) - ro) / rd;
}

fn brick_id(cell: vec3<i32>) -> u32 {
    let coord = cell / BRICK_DIM;
    return textureLoad(brick_index_tex, coord, 0).r;
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

fn face_normal(cell: vec3<i32>, point: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    let local = point / VOXEL_SIZE - vec3<f32>(cell);
    let dx = select(1.0 - local.x, local.x, rd.x >= 0.0);
    let dy = select(1.0 - local.y, local.y, rd.y >= 0.0);
    let dz = select(1.0 - local.z, local.z, rd.z >= 0.0);
    if (dx <= dy && dx <= dz) {
        return vec3<f32>(select(1.0, -1.0, rd.x >= 0.0), 0.0, 0.0);
    }
    if (dy <= dz) {
        return vec3<f32>(0.0, select(1.0, -1.0, rd.y >= 0.0), 0.0);
    }
    return vec3<f32>(0.0, 0.0, select(1.0, -1.0, rd.z >= 0.0));
}

fn init_t_max(cell: vec3<i32>, step: vec3<i32>, ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        next_boundary(cell.x, step.x, ro.x, rd.x),
        next_boundary(cell.y, step.y, ro.y, rd.y),
        next_boundary(cell.z, step.z, ro.z, rd.z),
    );
}

fn cast_ray(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> Hit {
    var result = Hit(false, vec3<i32>(0), vec3<f32>(0.0), 0u, 0.0);
    if (dot(rd, rd) < 1e-12) {
        return result;
    }
    let bounds = intersect_world(ro, rd);
    if (bounds.y < 0.0 || bounds.x > bounds.y) {
        return result;
    }

    let t_start = max(bounds.x, 0.0);
    let sample = ro + rd * (t_start + 0.001);
    var cell = vec3<i32>(floor(sample / VOXEL_SIZE));
    let max_cell = vec3<i32>(camera.world_max.xyz / VOXEL_SIZE) - vec3<i32>(1);
    if (any(cell < vec3<i32>(0)) || any(cell > max_cell)) {
        return result;
    }

    var step = vec3<i32>(
        select(-1, 1, rd.x >= 0.0),
        select(-1, 1, rd.y >= 0.0),
        select(-1, 1, rd.z >= 0.0),
    );
    var t_delta = vec3<f32>(
        select(1e30, VOXEL_SIZE / abs(rd.x), abs(rd.x) > 1e-8),
        select(1e30, VOXEL_SIZE / abs(rd.y), abs(rd.y) > 1e-8),
        select(1e30, VOXEL_SIZE / abs(rd.z), abs(rd.z) > 1e-8),
    );
    var t_max = init_t_max(cell, step, ro, rd);

    var distance = t_start;

    for (var i = 0; i < MAX_STEPS; i = i + 1) {
        if (distance > max_distance) {
            break;
        }
        if (any(cell < vec3<i32>(0)) || any(cell > max_cell)) {
            break;
        }

        let id = brick_id(cell);
        if (id == 0u) {
            // Empty brick: jump straight to the next 2 m brick boundary. The
            // brick index covers the whole world, so no region-level test is
            // needed for this arena.
            let t_brick = min(
                min(
                    next_brick_boundary(cell.x, step.x, ro.x, rd.x),
                    next_brick_boundary(cell.y, step.y, ro.y, rd.y),
                ),
                next_brick_boundary(cell.z, step.z, ro.z, rd.z),
            );
            var t_skip = t_brick;
            // If the whole 8 m group is empty, skip four bricks at once.
            let group = cell / (BRICK_DIM * GROUP_BRICKS);
            if (textureLoad(group_occupancy_tex, group, 0).r == 0u) {
                t_skip = min(
                    min(
                        next_group_boundary(cell.x, step.x, ro.x, rd.x),
                        next_group_boundary(cell.y, step.y, ro.y, rd.y),
                    ),
                    next_group_boundary(cell.z, step.z, ro.z, rd.z),
                );
            }
            if (t_skip > max_distance || t_skip >= 1e29) {
                break;
            }
            distance = t_skip + 0.001;
            let sample = ro + rd * distance;
            cell = vec3<i32>(floor(sample / VOXEL_SIZE));
            t_max = init_t_max(cell, step, ro, rd);
            continue;
        }

        let raw = raw_voxel_in_brick(cell, id);
        if ((raw & 0x3ffu) != 0u) {
            result.valid = true;
            result.voxel = cell;
            result.normal = face_normal(cell, ro + rd * distance, rd);
            result.raw = raw;
            result.distance = distance;
            return result;
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

    return result;
}

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
    let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3<f32>(0.62, 0.73, 0.86), vec3<f32>(0.07, 0.14, 0.30), up);
}

fn shade(hit: Hit, ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    let material = palette[hit.raw & 0x3ffu];
    let tint = f32((hit.raw >> 10u) & 0x3fu) / 63.0;
    var albedo = material.albedo_rough.xyz * mix(0.82, 1.18, tint);

    let roughness = clamp(material.albedo_rough.w, 0.04, 1.0);
    let metalness = clamp(material.params.x, 0.0, 1.0);
    let emissive = material.params.y;

    var normal = normalize(-rd);
    if (length(hit.normal) > 0.5) {
        normal = normalize(hit.normal);
    }

    let sun = normalize(camera.sun_dir.xyz);
    let ndl = max(dot(normal, sun), 0.0);
    var visibility = 1.0;
    if (ndl > 0.001) {
        let shadow_origin = ro + rd * hit.distance + normal * 0.015;
        let shadow = cast_ray(shadow_origin, sun, 32.0);
        visibility = select(1.0, 0.0, shadow.valid);
    }

    let ambient = mix(vec3<f32>(0.06, 0.08, 0.11), vec3<f32>(0.24, 0.27, 0.32), normal.y * 0.5 + 0.5);
    let sun_color = vec3<f32>(1.0, 0.95, 0.86);
    var color = albedo * (ambient + sun_color * ndl * visibility);
    color = color + albedo * emissive;

    let view_dir = -rd;
    let half_dir = normalize(sun + view_dir);
    let specular = pow(max(dot(normal, half_dir), 0.0), mix(64.0, 8.0, roughness));
    color = color + vec3<f32>(specular * metalness * visibility);

    return color;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near_h = camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near = near_h.xyz / near_h.w;
    let far = far_h.xyz / far_h.w;
    let ro = camera.camera_pos.xyz;
    let rd = normalize(far - near);

    var color = sky_color(rd);
    let hit = cast_ray(ro, rd, 512.0);
    if (hit.valid) {
        color = shade(hit, ro, rd);
    }

    // Simple filmic curve, then gamma. The internal target is UNORM, so the
    // composite pass only needs a blit.
    let mapped = color / (color + vec3<f32>(1.0));
    let gamma = pow(max(mapped, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(gamma, 1.0));
}
